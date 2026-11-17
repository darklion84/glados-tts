"""Переключатель голоса: один Wyoming-голос для HA, а отвечает тот синтезатор, который положен говорящему.

HA задаёт голос на весь конвейер и посреди запроса его не меняет. Поэтому конвейер смотрит на этот
посредник, а он пересылает текст в один из настоящих синтезаторов (options: voices) — в тот, что
назначен узнанному человеку (speaker_voices), иначе в default_voice. Кто говорит, сообщает
посредник распознавания в том же процессе: `Router.speaker_heard` вызывается до того, как текст
запроса уйдёт в HA.

HA хранит озвученные фразы по тексту и не знает, что голос у нас зависит от человека: «Таймер
запущен», сказанное одним голосом, он отдал бы тем же голосом и другому. Поэтому, когда следующий
ответ нужен другим голосом, чем лежит в хранилище, посредник просит HA его очистить (tts.clear_cache).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.request
from dataclasses import dataclass

from wyoming.audio import AudioChunk, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeStopped

_LOGGER = logging.getLogger("tts_router")

# Сколько решение «кто спросил» ждёт свой ответ. Дольше — и чужое объявление (таймер, автоматизация)
# прозвучало бы голосом последнего спросившего.
TURN_SECONDS = 45.0
CONNECT_TIMEOUT = 1.5
VOICE_NAME = "auto"


class HomeAssistant:
    """Вызовы HA через Supervisor (нужен homeassistant_api: true)."""

    def __init__(self, token: str, base: str = "http://supervisor/core/api") -> None:
        self.token = token
        self.base = base

    def _post(self, path: str, body: dict) -> None:
        req = urllib.request.Request(
            self.base + path, data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=3).close()

    async def post(self, path: str, body: dict) -> bool:
        try:
            await asyncio.get_running_loop().run_in_executor(None, self._post, path, body)
            return True
        except OSError as err:
            _LOGGER.warning("HA %s: %s", path, err)
            return False


@dataclass
class Voice:
    name: str
    host: str
    port: int
    streaming: bool | None = None  # принимает ли текст потоком; None — ещё не спрашивали
    languages: tuple[str, ...] = ()


class Router:
    def __init__(self, voices: list[Voice], default: str, speaker_voices: dict[str, str],
                 ha: HomeAssistant | None) -> None:
        self.voices = {v.name: v for v in voices}
        if default not in self.voices:
            raise ValueError(f"default_voice «{default}» нет в voices")
        self.default = default
        self.speaker_voices = {s: v for s, v in speaker_voices.items() if v in self.voices}
        for speaker, voice in speaker_voices.items():
            if voice not in self.voices:
                _LOGGER.warning("speaker_voices: у «%s» голос «%s», которого нет в voices", speaker, voice)
        self.ha = ha
        self._turn: tuple[str, float] | None = None
        # Какими голосами озвучено то, что лежит в хранилище HA. При старте неизвестно — «?».
        self._cached: set[str] = {"?"}

    def voice_for(self, speaker: str | None) -> str:
        return self.speaker_voices.get(speaker or "", self.default)

    async def speaker_heard(self, speaker: str | None) -> str:
        """Запомнить голос для ответа на этот запрос; вернуть его имя."""
        voice = self.voice_for(speaker)
        self._turn = (voice, time.monotonic() + TURN_SECONDS)
        if self._cached - {voice} and self.ha is not None:
            if await self.ha.post("/services/tts/clear_cache", {}):
                _LOGGER.info("хранилище озвучки HA очищено: следующий ответ голосом %s", voice)
                self._cached = set()
        return voice

    def take(self) -> str:
        """Голос для начинающегося синтеза. Решение по запросу действует на один ответ."""
        turn, self._turn = self._turn, None
        return turn[0] if turn and time.monotonic() < turn[1] else self.default

    async def describe(self, voice: Voice) -> None:
        """Узнать у синтезатора, принимает ли он текст потоком, и его языки."""
        async with AsyncTcpClient(voice.host, voice.port) as client:
            await client.write_event(Describe().event())
            while True:
                event = await asyncio.wait_for(client.read_event(), 10)
                if event is None:
                    raise ConnectionError("синтезатор закрыл соединение")
                if Info.is_type(event.type):
                    programs = Info.from_event(event).tts
                    voice.streaming = any(p.supports_synthesize_streaming for p in programs)
                    voice.languages = tuple(sorted({lang for p in programs for v in p.voices for lang in v.languages}))
                    return

    async def connect(self, name: str) -> tuple[Voice, AsyncTcpClient]:
        """Соединение с синтезатором; если нужный недоступен — с синтезатором по умолчанию."""
        for candidate in dict.fromkeys((name, self.default)):
            voice = self.voices[candidate]
            try:
                if voice.streaming is None:
                    await asyncio.wait_for(self.describe(voice), CONNECT_TIMEOUT + 2)
                client = AsyncTcpClient(voice.host, voice.port)
                await asyncio.wait_for(client.connect(), CONNECT_TIMEOUT)
            except (OSError, asyncio.TimeoutError) as err:
                _LOGGER.error("голос %s (%s:%s) недоступен: %s", voice.name, voice.host, voice.port, err)
                continue
            self._cached.add(voice.name)
            return voice, client
        raise ConnectionError("ни один синтезатор не отвечает")

    def info(self) -> Info:
        attribution = Attribution(name="stt_filter", url="")
        return Info(tts=[TtsProgram(
            name="voice-router", attribution=attribution, installed=True, version="1.1.2",
            description="Голос ответа по говорящему: " + ", ".join(self.voices),
            supports_synthesize_streaming=True,
            voices=[TtsVoice(name=VOICE_NAME, attribution=attribution, installed=True, version="1.1.2",
                             description="По говорящему", languages=list(self.voices[self.default].languages))],
        )])


class RouterHandler(AsyncEventHandler):
    """Одно соединение HA. Синтезатор выбирается на первом событии синтеза."""

    def __init__(self, router: Router, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.router = router
        self.info_event = router.info().event()
        self.voice: Voice | None = None
        self.upstream: AsyncTcpClient | None = None
        self.relay: asyncio.Task | None = None
        self.client_streams = False

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info_event)
            return True
        starts = SynthesizeStart.is_type(event.type)
        whole = Synthesize.is_type(event.type)
        if self.upstream is None:
            if not (starts or whole):
                return True
            try:
                self.voice, self.upstream = await self.router.connect(self.router.take())
            except ConnectionError as err:
                await self.write_event(Error(text=str(err), code="tts-unavailable").event())
                return False
            self.client_streams = starts
            _LOGGER.info("озвучка: голос %s%s", self.voice.name, ", текст потоком" if starts else "")
            self.relay = asyncio.create_task(self._relay(self.upstream, self.voice, time.monotonic()))
        assert self.voice is not None
        # Имя голоса у HA наше («auto»), синтезатору оно незнакомо: пусть берёт свой голос по умолчанию.
        if whole:
            # Потоковый клиент повторяет весь текст одним Synthesize для старых серверов; потоковый
            # синтезатор его пропустит сам, а непотоковому это единственное, что нужно.
            await self.upstream.write_event(Synthesize(text=Synthesize.from_event(event).text).event())
        elif self.voice.streaming:
            if starts:
                await self.upstream.write_event(SynthesizeStart().event())
            elif SynthesizeChunk.is_type(event.type) or SynthesizeStop.is_type(event.type):
                await self.upstream.write_event(event)
        return True

    async def _relay(self, upstream: AsyncTcpClient, voice: Voice, started: float) -> None:
        """Звук от синтезатора — в HA, до конца ответа."""
        first = True
        try:
            while (event := await upstream.read_event()) is not None:
                if first and AudioChunk.is_type(event.type):
                    first = False
                    _LOGGER.info("озвучка: первый звук от %s через %.2f с", voice.name, time.monotonic() - started)
                await self.write_event(event)
                if AudioStop.is_type(event.type) and self.client_streams and not voice.streaming:
                    # Непотоковый синтезатор не скажет, что закончил; потоковый клиент этого ждёт.
                    await self.write_event(SynthesizeStopped().event())
                    break
                if SynthesizeStopped.is_type(event.type) or (AudioStop.is_type(event.type) and not self.client_streams):
                    break
            else:
                _LOGGER.warning("синтезатор %s закрыл соединение посреди ответа", voice.name)
                self.writer.close()
        except OSError as err:
            _LOGGER.warning("связь с %s оборвалась: %s", voice.name, err)
            self.writer.close()
        finally:
            await self._close(upstream)

    async def _close(self, upstream: AsyncTcpClient | None) -> None:
        if upstream is not None and upstream is self.upstream:
            self.upstream = None
        if upstream is not None:
            try:
                await upstream.disconnect()
            except Exception:  # noqa: BLE001 — синтезатор уже мог закрыть соединение
                pass

    async def disconnect(self) -> None:
        if self.relay is not None and not self.relay.done():
            self.relay.cancel()
        await self._close(self.upstream)

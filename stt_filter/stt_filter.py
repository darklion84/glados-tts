"""Wyoming-посредник распознавания речи: пропускает звук к Parakeet и чистит его ответ.

parakeet-tdt-0.6b-v3 сам определяет язык (задать «только русский» нельзя) и на шуме в комнате
выдаёт короткие английские «пустышки»: «Yeah.», «Um stop.». HA принимает их за запрос, и модель
переспрашивает вместо того, чтобы колонка закрыла диалог. Посредник превращает такие ответы в
пустой текст (HA: stt-no-text-recognized, Pulsar играет сигнал отмены), а «stop» среди
слов-паразитов — в «стоп» (его ловит проверка текста в Pulsar).

Решение по каждому ответу пишется в лог вместе с длительностью и уровнем звука — по ним правила
подстраиваются; сам звук не сохраняется.

Пока Parakeet распознаёт, по той же записи определяется, кто говорит (speaker.py). Результат до
отправки текста в HA получают переключатель голоса (tts_router.py) и сенсор sensor.voice_speaker.
"""

from __future__ import annotations

import array
import asyncio
import logging
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

if TYPE_CHECKING:
    from speaker import Voiceprints
    from tts_router import HomeAssistant, Router

_LOGGER = logging.getLogger("stt_filter")

_CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)
_WORD = re.compile(r"[a-zа-яё']+")
# Слова-паразиты, которые не мешают считать фразу командой «стоп».
FILLERS = frozenset({
    "um", "uh", "uhm", "hmm", "hm", "mm", "mhm", "ah", "oh", "eh", "er", "erm", "so", "well", "okay", "ok",
    "э", "ээ", "эээ", "эм", "мм", "ммм", "а", "ну", "ой", "так",
})
# Стоп-слова Pulsar («стоп», «хватит») и то, как их пишет Parakeet латиницей.
STOP_WORDS = frozenset({"stop", "stopp", "стоп", "хватит"})
# Дольше этого звук для отпечатка голоса не копится (16 кГц, 16 бит, моно).
MAX_VOICE_BYTES = 30 * 16000 * 2
SPEAKER_SENSOR = "sensor.voice_speaker"


@dataclass
class Settings:
    upstream_host: str
    upstream_port: int
    max_latin_words: int
    timeout: float = 30.0


def clean(text: str, max_latin_words: int) -> tuple[str, str]:
    """(новый текст, причина).

    Стоп-слово среди одних слов-паразитов (на любом языке) → ровно «стоп»: проверка текста в
    Pulsar принимает только стоп-слова без других слов. Ответ без кириллицы не длиннее
    max_latin_words слов → пусто. Остальное, в том числе любые русские ответы, — как есть:
    короткие «да», «нет» нужны продолжению диалога.
    """
    words = _WORD.findall(text.lower())
    if words and any(w in STOP_WORDS for w in words) and all(w in STOP_WORDS | FILLERS for w in words):
        return "стоп", "stop"
    if not words or _CYRILLIC.search(text):
        return text, "pass"
    if len(words) <= max_latin_words:
        return "", "latin-short"
    return text, "pass-latin-long"


def _dbfs(rms: float) -> float:
    return 20 * math.log10(rms / 32768) if rms > 0 else -120.0


class FilterHandler(AsyncEventHandler):
    def __init__(self, info: Info, args: Settings, voiceprints: Voiceprints | None, router: Router | None,
                 ha: HomeAssistant | None, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.info_event = info.event()
        self.args = args
        self.voiceprints = voiceprints
        self.router = router
        self.ha = ha
        self.upstream: AsyncTcpClient | None = None
        self.samples = 0
        self.rate = 16000
        self.mono16 = True
        self.sum_sq = 0.0
        self.peak = 0
        self.pcm = bytearray()

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info_event)
            return True
        if Transcribe.is_type(event.type):
            await self._close_upstream()
            self.upstream = AsyncTcpClient(self.args.upstream_host, self.args.upstream_port)
            await self.upstream.connect()
            self.samples, self.sum_sq, self.peak = 0, 0.0, 0
            self.pcm = bytearray()
            await self.upstream.write_event(event)
            return True
        if self.upstream is None:
            return True
        if AudioStart.is_type(event.type):
            start = AudioStart.from_event(event)
            self.rate, self.mono16 = start.rate, (start.width == 2 and start.channels == 1)
        elif AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            if chunk.width == 2 and chunk.audio:
                pcm = array.array("h", chunk.audio[: len(chunk.audio) // 2 * 2])  # s16le, хост little-endian
                self.samples += len(pcm)
                self.sum_sq += sum(x * x for x in pcm)
                self.peak = max(self.peak, max((abs(x) for x in pcm), default=0))
                if len(self.pcm) < MAX_VOICE_BYTES:
                    self.pcm += chunk.audio
        await self.upstream.write_event(event)
        if AudioStop.is_type(event.type):
            await self._answer()
            return False
        return True

    async def _answer(self) -> None:
        assert self.upstream is not None
        loop = asyncio.get_running_loop()
        # Отпечаток голоса считается, пока Parakeet распознаёт: ответа это не задерживает.
        embedding_job = None
        if self.voiceprints is not None and self.mono16:
            embedding_job = loop.run_in_executor(None, self.voiceprints.embed, bytes(self.pcm), self.rate)
        try:
            while True:
                reply = await asyncio.wait_for(self.upstream.read_event(), self.args.timeout)
                if reply is None:
                    raise ConnectionError("upstream closed")
                if Transcript.is_type(reply.type):
                    break
            original = Transcript.from_event(reply)
            text, why = clean(original.text, self.args.max_latin_words)
            dur = self.samples / self.rate if self.rate else 0.0
            rms = math.sqrt(self.sum_sq / self.samples) if self.samples else 0.0
            speaker, scores = None, {}
            if embedding_job is not None:
                embedding = await embedding_job
                speaker, scores = self.voiceprints.identify(embedding)
                self.voiceprints.record(text, dur, _dbfs(rms), embedding, speaker, scores)
            voice = await self._announce_speaker(speaker, scores) if text and text != "стоп" else None
            _LOGGER.info("%s «%s» → «%s» | %.2f s rms %.0f dBFS peak %.0f dBFS | говорит %s%s%s",
                         why, original.text, text, dur, _dbfs(rms), _dbfs(self.peak), speaker or "неизвестно",
                         " (" + ", ".join(f"{k} {v:.2f}" for k, v in sorted(scores.items(), key=lambda kv: -kv[1])) + ")"
                         if scores else "", f" → голос {voice}" if voice else "")
            await self.write_event(Transcript(text=text, language=original.language).event())
        finally:
            await self._close_upstream()

    async def _announce_speaker(self, speaker: str | None, scores: dict[str, float]) -> str | None:
        """Сообщить, кто говорит, до того как HA получит текст: ответ готовится уже с этим знанием."""
        jobs = []
        if self.router is not None:
            jobs.append(self.router.speaker_heard(speaker))
        if self.ha is not None:
            jobs.append(self.ha.post(f"/states/{SPEAKER_SENSOR}", {
                "state": speaker or "unknown",
                "attributes": {"friendly_name": "Кто говорил с колонкой",
                               "scores": {k: round(v, 2) for k, v in scores.items()}},
            }))
        results = await asyncio.gather(*jobs)
        return results[0] if self.router is not None else None

    async def _close_upstream(self) -> None:
        if self.upstream is not None:
            try:
                await self.upstream.disconnect()
            except Exception:  # noqa: BLE001 — upstream уже мог закрыть соединение
                pass
            self.upstream = None

    async def disconnect(self) -> None:
        await self._close_upstream()


async def upstream_info(args: Settings) -> Info:
    async with AsyncTcpClient(args.upstream_host, args.upstream_port) as c:
        await c.write_event(Describe().event())
        while True:
            e = await asyncio.wait_for(c.read_event(), 10)
            if e is not None and Info.is_type(e.type):
                info = Info.from_event(e)
                for asr in info.asr:
                    asr.name = f"stt-filter ({asr.name})"
                    asr.description = f"Фильтр английских «пустышек» → {asr.description}"
                return info

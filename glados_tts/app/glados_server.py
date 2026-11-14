"""Wyoming TTS server for TeraTTS glados2-g2p-vits.

Text path per sentence:
  mitrokun normalizers (numbers, units, time, Latin, ё; patched by units_ru/latin_ru/stress_fix_ru)
  -> silero-stress for words whose stress the dictionary cannot decide (homographs, unknown words)
  -> glados_g2p (TeraTTS phonemes) -> onnxruntime.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import threading
import time
from functools import partial

import cpu_threads
import latin_ru
import stress_fix_ru
import units_ru

# Patch onnxruntime/torch thread pools and the normalizers BEFORE anything instantiates them.
cpu_threads.install()
units_ru.install()
stress_fix_ru.install()
latin_ru.install()

import numpy as np  # noqa: E402
from num2words import num2words  # noqa: E402
import onnxruntime  # noqa: E402
from silero_stress import load_accentor  # noqa: E402
from wyoming.audio import AudioChunk, AudioStart, AudioStop  # noqa: E402
from wyoming.event import Event  # noqa: E402
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice  # noqa: E402
from wyoming.server import AsyncEventHandler, AsyncServer  # noqa: E402
from wyoming.tts import Synthesize  # noqa: E402
from wyoming_piper.english_normalizer import EnglishNormalizer  # noqa: E402
from wyoming_piper.russian_normalizer import RussianNormalizer  # noqa: E402

import glados_g2p as g2p  # noqa: E402

_LOG = logging.getLogger("glados")
VOICE = "ru_RU-glados-medium"
RATE, WIDTH, CHANNELS = 22050, 2, 1
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
_LATIN = re.compile(r"[A-Za-z]")


class Engine:
    def __init__(self, voice_dir: str, length_scale: float) -> None:
        cfg = json.load(open(os.path.join(voice_dir, "config.json"), encoding="utf-8"))
        self.g2p = g2p.G2P(os.path.join(voice_dir, "dictionary.sqlite"), cfg["phoneme_id_map"])
        self.session = onnxruntime.InferenceSession(
            os.path.join(voice_dir, "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self.length_scale = length_scale
        self.eng = EnglishNormalizer()
        self.rus = RussianNormalizer(use_yo=True)
        self.accentor = load_accentor()
        self.lock = threading.Lock()

    def prepare(self, sentence: str) -> str:
        text = sentence
        if _LATIN.search(text):
            text = self.eng.normalize(text)
        text = self.rus.normalize(text)
        text = spell_numbers(text)  # mitrokun leaves plain digits to espeak; this model has none
        text = g2p.normalize_marks(text)
        tokens = g2p.words(text)
        if any(self.g2p.needs_stress(w) for w in tokens):
            text = self._apply_accentor(text, tokens)
        return text

    def _apply_accentor(self, text: str, tokens: list[str]) -> str:
        try:
            stressed = g2p.normalize_marks(self.accentor(text))
        except Exception as err:  # keep speaking without context stress
            _LOG.warning("silero-stress failed: %s", err)
            return text
        out_words = g2p.words(stressed)
        if len(out_words) != len(tokens) or any(
            g2p.fold(a) != g2p.fold(b) for a, b in zip(tokens, out_words)
        ):
            _LOG.debug("silero output does not align, skipped: %r -> %r", text, stressed)
            return text
        it = iter(out_words)

        def repl(m: re.Match) -> str:
            mine, theirs = m.group(0), next(it)
            return theirs if self.g2p.needs_stress(mine) else mine

        return g2p._WORD_RE.sub(repl, text)

    def synthesize(self, sentence: str) -> bytes:
        t0 = time.perf_counter()
        text = self.prepare(sentence)
        phonemes, trace = self.g2p.phonemes(text)
        if not phonemes:
            return b""
        ids = self.g2p.ids(phonemes)
        with self.lock:
            audio = self.session.run(None, {
                "input": np.array([ids], dtype=np.int64),
                "input_lengths": np.array([len(ids)], dtype=np.int64),
                "scales": np.array([0.667, self.length_scale, 0.8], dtype=np.float32),
            })[0].reshape(-1)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        _LOG.debug("text: %s | words: %s", text, " ".join(trace))
        _LOG.info("%.2fs audio in %.2fs: %s", len(pcm) / (RATE * WIDTH), time.perf_counter() - t0, sentence[:80])
        return pcm


_TIME = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_DECIMAL = re.compile(r"\b(\d+)[.,](\d+)\b")
_INT = re.compile(r"\d+")


def _words(value) -> str:
    try:
        return num2words(value, lang="ru")
    except Exception:  # absurdly large numbers etc.: read digit by digit
        return " ".join(num2words(int(d), lang="ru") for d in str(value) if d.isdigit())


def _minutes(mm: str) -> str:
    if mm == "00":
        return "ноль ноль"
    return ("ноль " if mm[0] == "0" else "") + _words(int(mm))


def spell_numbers(text: str) -> str:
    """Digits left after normalization -> words: 6:45, 21,5 / 21.5, integers."""
    text = _TIME.sub(lambda m: f"{_words(int(m.group(1)))} {_minutes(m.group(2))}", text)
    text = _DECIMAL.sub(lambda m: _words(float(f"{m.group(1)}.{m.group(2)}")), text)
    return _INT.sub(lambda m: _words(int(m.group(0))), text)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]


class Handler(AsyncEventHandler):
    def __init__(self, info: Info, engine: Engine, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.info_event = info.event()
        self.engine = engine

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info_event)
            return True
        if Synthesize.is_type(event.type):
            text = Synthesize.from_event(event).text or ""
            loop = asyncio.get_running_loop()
            await self.write_event(AudioStart(rate=RATE, width=WIDTH, channels=CHANNELS).event())
            for sentence in split_sentences(text):
                pcm = await loop.run_in_executor(None, self.engine.synthesize, sentence)
                for i in range(0, len(pcm), 8192):
                    await self.write_event(
                        AudioChunk(audio=pcm[i:i + 8192], rate=RATE, width=WIDTH, channels=CHANNELS).event()
                    )
            await self.write_event(AudioStop().event())
        return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="tcp://0.0.0.0:10200")
    ap.add_argument("--voice-dir", default="/voice-glados")
    ap.add_argument("--length-scale", type=float, default=1.2)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)s:%(name)s:%(message)s")

    engine = Engine(args.voice_dir, args.length_scale)
    engine.synthesize("Тест.")  # warm-up: model, dictionary, normalizers, silero
    attribution = Attribution(name="TeraTTS", url="https://huggingface.co/TeraTTS/glados2-g2p-vits")
    info = Info(tts=[TtsProgram(
        name="glados-tts", attribution=attribution, installed=True,
        description="TeraTTS GLaDOS (VITS) with mitrokun normalization and silero-stress",
        version="1.0.0",
        voices=[TtsVoice(name=VOICE, attribution=attribution, installed=True,
                         description="GLaDOS (ru)", version="1.0.0", languages=["ru"])],
    )])
    _LOG.info("Ready (length_scale=%s)", args.length_scale)
    server = AsyncServer.from_uri(args.uri)
    asyncio.run(server.run(partial(Handler, info, engine)))


if __name__ == "__main__":
    main()

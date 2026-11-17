"""Кто говорит: отпечаток голоса по записи запроса и сравнение с записанными образцами.

Модель (CAM++ из 3D-Speaker, через sherpa-onnx) превращает запись в вектор — «отпечаток». Обучать её
под человека не нужно: образец человека — это несколько отпечатков его обычных фраз. Они лежат в
speakers.json (его пишет voiceprints.py с компьютера), людей может быть сколько угодно. Запрос относится к
тому, на чей усреднённый образец он похож больше всего, — если сходство выше порога и заметно выше,
чем у следующего; иначе говорящий неизвестен.

Каждый запрос дописывается в utterances.jsonl вместе с отпечатком: из этих строк набираются образцы
и по ним подбирается порог. Сам звук не сохраняется.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
from datetime import datetime

import numpy as np
import sherpa_onnx

_LOGGER = logging.getLogger("speaker")

# Короче этого отпечаток не считается: на секунде звука модель путает людей.
MIN_SECONDS = 1.0
LOG_ROTATE_BYTES = 20 * 1024 * 1024


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) or 1.0)


def encode(v: np.ndarray) -> str:
    return base64.b64encode(v.astype("<f4").tobytes()).decode("ascii")


def decode(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype="<f4")


def choose(scores: dict[str, float], threshold: float, margin: float) -> str | None:
    """Самый похожий, если он выше порога и опережает следующего на margin; иначе None."""
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else -1.0
    return best if top >= threshold and top - second >= margin else None


class Voiceprints:
    def __init__(self, model: str, store_dir: str, threshold: float, margin: float) -> None:
        self.model_name = os.path.splitext(os.path.basename(model))[0]
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=model, num_threads=1)
        )
        self.lock = threading.Lock()
        self.threshold = threshold
        self.margin = margin
        self.speakers_path = os.path.join(store_dir, "speakers.json")
        self.log_path = os.path.join(store_dir, "utterances.jsonl")
        self.centroids: dict[str, np.ndarray] = {}
        self._mtime: float | None = None
        os.makedirs(store_dir, exist_ok=True)
        self._reload()

    def _reload(self) -> None:
        """Перечитать speakers.json, если файл изменился (образцы добавляют без перезапуска аддона)."""
        try:
            mtime = os.stat(self.speakers_path).st_mtime
        except FileNotFoundError:
            mtime = 0.0
        if mtime == self._mtime:
            return
        self._mtime = mtime
        centroids: dict[str, np.ndarray] = {}
        if mtime:
            try:
                with open(self.speakers_path, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("model") != self.model_name:
                    _LOGGER.warning("speakers.json снят другой моделью (%s), образцы не используются", data.get("model"))
                else:
                    for name, spk in data.get("speakers", {}).items():
                        vectors = [_unit(decode(e)) for e in spk.get("embeddings", [])]
                        if vectors:
                            centroids[name] = _unit(np.mean(vectors, axis=0))
            except (OSError, ValueError) as err:
                _LOGGER.error("speakers.json не читается: %s", err)
        self.centroids = centroids
        _LOGGER.info("образцы голосов: %s", ", ".join(sorted(centroids)) or "нет")

    def embed(self, pcm: bytes, rate: int) -> np.ndarray | None:
        """Отпечаток записи (s16le, моно). Вызывается из рабочего потока."""
        samples = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768
        if rate <= 0 or len(samples) < MIN_SECONDS * rate:
            return None
        with self.lock:
            stream = self.extractor.create_stream()
            stream.accept_waveform(rate, samples)
            stream.input_finished()
            if not self.extractor.is_ready(stream):
                return None
            return _unit(np.array(self.extractor.compute(stream), dtype=np.float32))

    def identify(self, embedding: np.ndarray | None) -> tuple[str | None, dict[str, float]]:
        """(кто говорит или None, сходство с каждым образцом)."""
        self._reload()
        if embedding is None or not self.centroids:
            return None, {}
        scores = {name: float(c @ embedding) for name, c in self.centroids.items()}
        speaker = choose(scores, self.threshold, self.margin)
        return speaker, scores

    def record(self, text: str, seconds: float, rms_dbfs: float, embedding: np.ndarray | None,
               speaker: str | None, scores: dict[str, float]) -> None:
        """Дописать запрос в utterances.jsonl."""
        if embedding is None:
            return
        row = {
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "text": text,
            "seconds": round(seconds, 2),
            "rms_dbfs": round(rms_dbfs),
            "speaker": speaker,
            "scores": {k: round(v, 3) for k, v in scores.items()},
            "model": self.model_name,
            "embedding": encode(embedding),
        }
        try:
            if os.path.exists(self.log_path) and os.path.getsize(self.log_path) > LOG_ROTATE_BYTES:
                os.replace(self.log_path, self.log_path + ".1")
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as err:
            _LOGGER.error("utterances.jsonl не пишется: %s", err)

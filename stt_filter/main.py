"""Запуск аддона: посредник распознавания (10300) и переключатель голоса (10200) в одном процессе.

Они связаны: кто говорит, определяется на распознавании, а нужно это при озвучке ответа.
Настройки — из options.json аддона.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from functools import partial

from wyoming.server import AsyncServer

from speaker import Voiceprints
from stt_filter import FilterHandler, Settings, upstream_info
from tts_router import HomeAssistant, Router, RouterHandler, Voice

_LOGGER = logging.getLogger("main")


async def wait_for(what: str, job):
    """Повторять, пока соседний сервис не поднимется: аддон может стартовать раньше него."""
    while True:
        try:
            return await job()
        except Exception as err:  # noqa: BLE001 — любая причина значит «ещё не готов»
            _LOGGER.warning("%s недоступен (%s), повтор через 10 с", what, err)
            await asyncio.sleep(10)


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--options", default="/data/options.json")
    p.add_argument("--store", default="/config", help="папка для speakers.json и utterances.jsonl")
    p.add_argument("--model", default="/models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx")
    p.add_argument("--stt-uri", default="tcp://0.0.0.0:10300")
    p.add_argument("--tts-uri", default="tcp://0.0.0.0:10200")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    with open(args.options, encoding="utf-8") as f:
        opts = json.load(f)

    settings = Settings(opts["upstream_host"], int(opts["upstream_port"]), int(opts["max_latin_words"]))
    token = os.environ.get("SUPERVISOR_TOKEN")
    ha = HomeAssistant(token) if token else None
    voiceprints = Voiceprints(args.model, args.store, float(opts["speaker_threshold"]), float(opts["speaker_margin"]))
    router = Router(
        [Voice(v["name"], v["host"], int(v["port"])) for v in opts["voices"]],
        opts["default_voice"],
        {m["speaker"]: m["voice"] for m in opts.get("speaker_voices", [])},
        ha,
    )

    info = await wait_for(f"распознавание {settings.upstream_host}:{settings.upstream_port}",
                          partial(upstream_info, settings))
    default = router.voices[router.default]
    await wait_for(f"голос {default.name} {default.host}:{default.port}", partial(router.describe, default))
    _LOGGER.info("распознавание: %s; max_latin_words=%s; порог голоса %.2f, отрыв %.2f",
                 [a.name for a in info.asr], settings.max_latin_words, voiceprints.threshold, voiceprints.margin)
    _LOGGER.info("голоса: %s; по умолчанию %s; по людям %s",
                 ", ".join(router.voices), router.default, router.speaker_voices or "—")
    await asyncio.gather(
        AsyncServer.from_uri(args.stt_uri).run(partial(FilterHandler, info, settings, voiceprints, router, ha)),
        AsyncServer.from_uri(args.tts_uri).run(partial(RouterHandler, router)),
    )


if __name__ == "__main__":
    asyncio.run(main())

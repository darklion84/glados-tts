"""Unit tests for the stt_filter add-on: text cleaning, speaker choice, voice routing.

Need the add-on's own packages (no model, no network beyond localhost):
    pip install wyoming numpy sherpa-onnx
Without them the module is skipped, so `python -m unittest discover -s tests` still passes.
"""

import asyncio
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stt_filter"))
try:
    import tts_router
    from speaker import choose
    from stt_filter import clean
    from tts_router import HomeAssistant, Router, Voice
except ImportError as err:
    raise unittest.SkipTest(f"stt_filter dependencies are not installed: {err}")


class CleanTest(unittest.TestCase):
    """Parakeet answers with short English fillers on room noise; Russian answers must pass untouched."""

    CASES = [
        ("Yeah.", "", "latin-short"),
        ("Thank you.", "Thank you.", "pass-latin-long"),
        ("Um stop.", "стоп", "stop"),
        ("Stop.", "стоп", "stop"),
        ("Эм, стоп", "стоп", "stop"),
        ("Ну хватит", "стоп", "stop"),
        ("Stop the music", "Stop the music", "pass-latin-long"),
        ("Стоп музыку", "Стоп музыку", "pass"),
        ("Хватит ли бензина?", "Хватит ли бензина?", "pass"),
        ("Да.", "Да.", "pass"),
        ("Знаешь что?", "Знаешь что?", "pass"),
        ("", "", "pass"),
        ("...", "...", "pass"),
    ]

    def test_cases(self):
        for text, want, why in self.CASES:
            with self.subTest(text=text):
                self.assertEqual(clean(text, 1), (want, why))

    def test_latin_limit_is_configurable(self):
        self.assertEqual(clean("I do.", 2), ("", "latin-short"))
        self.assertEqual(clean("I do.", 1), ("I do.", "pass-latin-long"))


class ChooseTest(unittest.TestCase):
    def test_nobody_enrolled(self):
        self.assertIsNone(choose({}, 0.5, 0.05))

    def test_threshold(self):
        self.assertEqual(choose({"a": 0.62}, 0.5, 0.05), "a")
        self.assertIsNone(choose({"a": 0.49}, 0.5, 0.05))

    def test_margin_between_two_people(self):
        self.assertEqual(choose({"a": 0.62, "b": 0.40}, 0.5, 0.05), "a")
        self.assertIsNone(choose({"a": 0.62, "b": 0.60}, 0.5, 0.05))

    def test_more_similar_person_wins_below_the_others_threshold(self):
        # A second household member scoring 0.48 against the owner is not the owner once she is enrolled.
        self.assertEqual(choose({"owner": 0.48, "wife": 0.66}, 0.57, 0.05), "wife")


class _FakeHA(BaseHTTPRequestHandler):
    calls: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.calls.append((self.path, self.headers["Authorization"], body))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"[]")

    def log_message(self, *args):
        pass


class RouterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _FakeHA)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.ha = HomeAssistant("tok", f"http://127.0.0.1:{cls.server.server_port}/core/api")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        _FakeHA.calls.clear()
        voices = [Voice("alisa", "h", 1), Voice("glados", "h", 2)]
        self.router = Router(voices, "alisa", {"owner": "glados", "ghost": "missing"}, self.ha)

    def test_voice_for(self):
        self.assertEqual(self.router.voice_for("owner"), "glados")
        self.assertEqual(self.router.voice_for(None), "alisa")
        self.assertEqual(self.router.voice_for("ghost"), "alisa")  # mapped to a voice that does not exist

    def test_decision_serves_one_answer(self):
        self.assertEqual(self.router.take(), "alisa")  # an announcement with no request before it
        self.assertEqual(asyncio.run(self.router.speaker_heard("owner")), "glados")
        self.assertEqual(self.router.take(), "glados")
        self.assertEqual(self.router.take(), "alisa")

    def test_cache_is_cleared_only_when_the_voice_changes(self):
        asyncio.run(self.router.speaker_heard("owner"))  # unknown cache content at start: cleared
        self.assertEqual([c[:2] for c in _FakeHA.calls], [("/core/api/services/tts/clear_cache", "Bearer tok")])
        self.router._cached = {"glados"}
        asyncio.run(self.router.speaker_heard("owner"))
        self.assertEqual(len(_FakeHA.calls), 1)
        asyncio.run(self.router.speaker_heard(None))
        self.assertEqual(len(_FakeHA.calls), 2)
        self.assertEqual(self.router._cached, set())

    def test_decision_expires(self):
        asyncio.run(self.router.speaker_heard("owner"))
        real = tts_router.time.monotonic
        tts_router.time.monotonic = lambda: real() + tts_router.TURN_SECONDS + 1
        try:
            self.assertEqual(self.router.take(), "alisa")
        finally:
            tts_router.time.monotonic = real

    def test_home_assistant_down(self):
        router = Router([Voice("alisa", "h", 1)], "alisa", {}, HomeAssistant("tok", "http://127.0.0.1:9/core/api"))
        self.assertEqual(asyncio.run(router.speaker_heard(None)), "alisa")
        self.assertEqual(router._cached, {"?"})  # still suspect: retried on the next request

    def test_default_voice_must_exist(self):
        with self.assertRaises(ValueError):
            Router([Voice("alisa", "h", 1)], "missing", {}, None)


if __name__ == "__main__":
    unittest.main()

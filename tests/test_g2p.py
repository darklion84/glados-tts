"""Unit tests for glados_g2p (stdlib only): python -m unittest discover -s tests"""

import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "glados_tts", "app"))
import glados_g2p as g  # noqa: E402

ID_MAP = {p: [i] for i, p in enumerate(
    ["_", "^", "$", " ", "!", "'", "(", ")", ",", "-", ".", ":", ";", "?",
     "a0", "a1", "b", "d", "dj", "e0", "e1", "j", "k", "l", "m", "mj", "n", "o0", "o1",
     "s", "sj", "t", "u0", "u1", "v", "z", "zh"])}


class Rules(unittest.TestCase):
    def test_stress_and_softness(self):
        self.assertEqual(g.rules("д+ома"), ["d", "o1", "m", "a0"])
        self.assertEqual(g.rules("дом+а"), ["d", "o0", "m", "a1"])
        self.assertEqual(g.rules("вс+ё"), ["v", "sj", "o1"])

    def test_iotation_after_vowel_and_soft_sign(self):
        self.assertEqual(g.rules("+яма"), ["j", "a1", "m", "a0"])
        self.assertEqual(g.rules("съ+ел"), ["s", "j", "e1", "l"])

    def test_mark_index(self):
        self.assertEqual(g.mark_index("зам+ок"), 1)
        self.assertEqual(g.mark_index("з+амок"), 0)
        self.assertIsNone(g.mark_index("замок"))

    def test_normalize_marks(self):
        self.assertEqual(g.normalize_marks("Ве́ка — Всё…"), "в+ека - всё.")


class Dictionary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
        db = sqlite3.connect(cls.tmp.name)
        db.execute("CREATE TABLE d (word TEXT, rank INTEGER, ph TEXT)")
        db.executemany("INSERT INTO d VALUES (?,?,?)", [
            ("замок", 0, "z a0 m o1 k"), ("замок", 1, "z a1 m o0 k"),
            ("уже", 0, "u0 zh e1"), ("уже", 1, "u1 zh e0"),
            ("все", 0, "v sj e1"), ("все", 1, "v sj o1"),
            ("дом", 0, "d o1 m"),
        ])
        db.commit(); db.close()
        cls.g2p = g.G2P(cls.tmp.name, ID_MAP)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.tmp.name)

    def test_stress_mark_picks_matching_variant(self):
        self.assertEqual(self.g2p.word_phonemes("з+амок")[0], "z a1 m o0 k".split())
        self.assertEqual(self.g2p.word_phonemes("зам+ок")[0], "z a0 m o1 k".split())

    def test_unmarked_homograph_follows_spelling_not_last_entry(self):
        self.assertEqual(self.g2p.word_phonemes("все")[0], "v sj e1".split())
        self.assertEqual(self.g2p.word_phonemes("уже")[0], "u0 zh e1".split())

    def test_needs_stress(self):
        self.assertTrue(self.g2p.needs_stress("замок"))
        self.assertFalse(self.g2p.needs_stress("з+амок"))
        self.assertFalse(self.g2p.needs_stress("дом"))       # one vowel
        self.assertTrue(self.g2p.needs_stress("незнакомо"))  # unknown word

    def test_ids_layout(self):
        ids = self.g2p.ids(["d", "o1"])
        m = ID_MAP
        inner = m["^"] + m["_"] + m["d"] + m["_"] + m["o1"] + m["_"] + m["$"]
        self.assertEqual(ids[1::2], inner)
        self.assertEqual(set(ids[0::2]), {0})


if __name__ == "__main__":
    unittest.main()

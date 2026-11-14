"""Text -> phoneme ids for TeraTTS g2p-vits models (glados2-g2p-vits).

Mirrors TeraTTS `TokenizerG2P` + `TTS._get_seq` (github.com/Tera2Space/TeraTTS):
the same phoneme set, word/punctuation split, Piper-style ids and the extra interspersed 0.
Differences, on purpose:
- homographs: TeraTTS takes the LAST dictionary entry for a word («уже» -> у́же, «все» -> всё);
  here the entry whose stressed vowel matches the stress mark (`+` before the vowel, from the
  user, the normalizer or silero-stress) wins; without a mark, the entry closest to the spelling;
- «ё» is always stressed (TeraTTS rules leave a bare «ё» unstressed);
- the dictionary is an SQLite file, so it is not held in RAM.
"""

from __future__ import annotations

import difflib
import re
import sqlite3

VOWELS = "аяоёуюэеиы"
_VOWEL_SET = frozenset(VOWELS)
_SOFT_NEXT = frozenset("яёюиье")
_SYLLABLE_START = frozenset("#ъьаяоёуюэеиы-")
_DROP = frozenset(["#", "+", "-", "ь", "ъ"])
_SOFT_HARD = {"б": "b", "в": "v", "г": "g", "д": "d", "з": "z", "к": "k", "л": "l", "м": "m",
              "н": "n", "п": "p", "р": "r", "с": "s", "т": "t", "ф": "f", "х": "h"}
_OTHER_CONS = {"ж": "zh", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch", "й": "j"}
_VOWEL_PH = {"а": "a", "я": "a", "у": "u", "ю": "u", "о": "o", "ё": "o", "э": "e", "е": "e",
             "и": "i", "ы": "y"}

PUNCT = "!'(),-.:;?"
_TOKEN_RE = re.compile(r"[а-яё+]+|[!'(),\-.:;?]")
_WORD_RE = re.compile(r"[а-яё+]+")
_ACUTE_RE = re.compile(r"([аеёиоуыэюя])́", re.I)


def rules(word: str) -> list[str]:
    """TeraTTS g2p.convert(): letters (+ before the stressed vowel) -> phonemes."""
    phones: list[list] = []
    stress = 0
    for ch in "#" + word + "#":
        if ch == "+":
            stress = 1
        else:
            phones.append([ch, stress])
            stress = 0
    for i in range(len(phones) - 1):
        c = phones[i][0]
        if c in _SOFT_HARD:
            phones[i] = [_SOFT_HARD[c] + ("j" if phones[i + 1][0] in _SOFT_NEXT else ""), 0]
        if c in _OTHER_CONS:
            phones[i] = [_OTHER_CONS[c], 0]
    out: list[str] = []
    prev = ""
    for ch, st in phones:
        if prev in _SYLLABLE_START and ch in "яюеё":
            out.append("j")
        out.append(_VOWEL_PH[ch] + str(st) if ch in _VOWEL_PH else ch)
        prev = ch
    return [p for p in out if p not in _DROP]


def _vowel_count(word: str) -> int:
    return sum(ch in _VOWEL_SET for ch in word)


def mark_index(word: str) -> int | None:
    """Index of the vowel after the first '+', counted among vowels; None if no mark."""
    pos = word.find("+")
    return None if pos < 0 else _vowel_count(word[:pos])


def ph_stress_index(phs: list[str]) -> int | None:
    vowels = [p for p in phs if p[-1:] in ("0", "1")]
    for i, p in enumerate(vowels):
        if p.endswith("1"):
            return i
    return None


def _similarity(a: list[str], b: list[str], ignore_stress: bool) -> float:
    if ignore_stress:
        a = [p.rstrip("01") for p in a]
        b = [p.rstrip("01") for p in b]
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def normalize_marks(text: str) -> str:
    """Lowercase; vowel+U+0301 -> '+vowel'; drop stray accents; '—'/'–' -> '-', '…' -> '.'."""
    text = _ACUTE_RE.sub(r"+\1", text).replace("́", "")
    return text.replace("—", "-").replace("–", "-").replace("…", ".").lower()


def fold(word: str) -> str:
    return word.replace("+", "").replace("ё", "е")


class G2P:
    def __init__(self, dict_path: str, phoneme_id_map: dict[str, list[int]]) -> None:
        self._db = sqlite3.connect(f"file:{dict_path}?mode=ro", uri=True, check_same_thread=False)
        self.id_map = phoneme_id_map

    def lookup(self, plain: str) -> list[list[str]]:
        rows = self._db.execute("SELECT ph FROM d WHERE word=? ORDER BY rank", (plain,)).fetchall()
        return [r[0].split() for r in rows]

    def needs_stress(self, word: str) -> bool:
        """True if the word's stress is ambiguous without context (homograph or unknown word)."""
        if "+" in word or "ё" in word or _vowel_count(word) < 2:
            return False  # one-vowel words: function words read unstressed, dictionary order decides
        return len(self.lookup(word)) != 1

    def word_phonemes(self, word: str) -> tuple[list[str], str]:
        """Phonemes for one lowercase word (may carry '+'); second value says where they came from."""
        plain = word.replace("+", "")
        if "ё" in plain and "+" not in word:
            word = word.replace("ё", "+ё", 1)
        variants = self.lookup(plain)
        k = mark_index(word)
        if k is not None:
            ref = rules(word)
            matching = [v for v in variants if ph_stress_index(v) == k]
            if matching:
                return max(matching, key=lambda v: _similarity(v, ref, False)), "dict+stress"
            return ref, "rules+stress"
        if variants:
            if len(variants) == 1:
                return variants[0], "dict"
            ref = rules(plain)
            return max(variants, key=lambda v: _similarity(v, ref, True)), "dict-ambiguous"
        if _vowel_count(plain) == 1:
            i = next(i for i, ch in enumerate(plain) if ch in _VOWEL_SET)
            return rules(plain[:i] + "+" + plain[i:]), "rules"
        return rules(plain), "rules-unstressed"

    def phonemes(self, text: str) -> tuple[list[str], list[str]]:
        """TeraTTS tokenizer layout: punctuation as-is, ' ' before every word but the first."""
        out: list[str] = []
        trace: list[str] = []
        for tok in _TOKEN_RE.findall(text):
            if tok in PUNCT:
                out.append(tok)
                continue
            if not tok.replace("+", ""):
                continue
            if out:
                out.append(" ")
            phs, src = self.word_phonemes(tok)
            out.extend(phs)
            trace.append(f"{tok}[{src}]")
        return out, trace

    def ids(self, phonemes: list[str]) -> list[int]:
        m = self.id_map
        seq = list(m["^"]) + list(m["_"])
        for p in phonemes:
            if p in m:
                seq += m[p] + m["_"]
        seq += m["$"]
        inter = [0] * (len(seq) * 2 + 1)
        inter[1::2] = seq
        return inter


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text)

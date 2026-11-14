"""Grammar rules that override silero-stress on homographs it gets wrong in assistant speech.

silero-stress (the fork's homograph resolver) puts «дома́» in «Все уже дома», «Дома у нас тепло»,
«сети́» in «Нет сети», «двери́» in «Закрой двери», «среду́» in «в среду», «стои́т» in «Доллар
стоит девяносто рублей», and «села́» everywhere — it never outputs the verb «се́ла» («Батарейка
села»). Found with eval/homograph_bench.py (AGENTS.md journal 2026-11-05). Each rule fires only on
a context that grammar decides:

- дома → до́ма right after «уже»/«ещё» («уже дома», «ещё дома?»), after «мы/вы/они все»,
  before «у нас/меня/…» («дома у нас тепло») and before «всё» spelled with ё in the input
  («дома всё в порядке»);
- сети → се́ти after a genitive-only word: нет, из, от, без, до, для, у, около, возле, вне;
- двери → две́ри unless «на/в/во» is one of the two preceding words (двери́ is only the locative
  «на двери», «на входной двери»);
- среду → сре́ду right after «в/во» (Wednesday; the environment takes an adjective: «в окружающую
  среду»);
- стоит → сто́ит when followed by a number and a currency word («стоит девяносто два рубля»);
- села → се́ла (verb) unless right after a preposition that governs the genitive («из села»).
  The genitive of «село» after a noun («жители села») becomes се́ла too: silero's constant
  «села́» was right there only by accident, and the verb is far more common in assistant speech.

Words only pair when separated by whitespace (no punctuation between). Everything else keeps
silero's answer. `install()` wraps silero_stress.load_accentor, so the fork's server
(docker/entrypoint.sh) and train/textprep.py get the same corrected accentor; the wrapper keeps
the original as `.accentor`.
"""

from __future__ import annotations

import re

_VOWELS = "аеёиоуыэюя"
_TOKEN = re.compile(r"[\w+]+|[^\w+]+")
_GEN_ONLY = {"нет", "из", "от", "без", "до", "для", "у", "около", "возле", "вне"}
_GEN_PREP = _GEN_ONLY - {"нет"} | {"вдоль", "близ", "вблизи", "напротив", "посреди", "среди",
                                    "вокруг", "после", "кроме", "мимо", "позади"}
_LOC_PREP = {"на", "в", "во"}
_U_PRONOUNS = {"нас", "меня", "тебя", "вас", "них", "него", "нее", "неё"}
_NUMBER = re.compile(
    r"\d+|ноль|од(ин|на|но|ну)|дв(а|е|ести)|три|четыре|пять|шесть|семь|восемь|девять|десять"
    r"|\w+надцать|двадцать|тридцать|сорок|пятьдесят|шестьдесят|семьдесят|восемьдесят|девяносто"
    r"|сто|триста|четыреста|\w+сот|тысяч\w*|миллион\w*|миллиард\w*|полтора|полторы")
_CURRENCY = re.compile(r"(руб|доллар|евро|копе|юан|бакс|тенге|фунт|франк|йен|лир|гривн)\w*")


def _key(word: str) -> str:
    return word.replace("+", "").lower().replace("ё", "е")


def _stress(word: str, vowel: int) -> str:
    """`word` (letters as given, '+' marks dropped) with '+' before vowel number `vowel`."""
    out, n = [], -1
    for ch in word.replace("+", ""):
        if ch.lower() in _VOWELS:
            n += 1
            if n == vowel:
                out.append("+")
        out.append(ch)
    return "".join(out)


def _rule(words: list[str], gaps: list[str], i: int) -> int | None:
    """Stressed vowel index forced for words[i] (input letters), or None to keep silero's.

    gaps[i] is the separator before words[i]; adjacency = whitespace-only separator. A word the
    user stressed with '+' is never touched."""
    if "+" in words[i]:
        return None
    k = _key(words[i])

    def prev(j: int = 1) -> str | None:
        if i - j < 0 or any(g.strip() for g in gaps[i - j + 1:i + 1]):
            return None
        return _key(words[i - j])

    def nxt(j: int = 1) -> str | None:
        if i + j >= len(words) or any(g.strip() for g in gaps[i + 1:i + j + 1]):
            return None
        return words[i + j].lower()

    if k == "дома":
        if prev() in {"уже", "еще"}:
            return 0
        if prev() == "все" and prev(2) in {"мы", "вы", "они"}:
            return 0
        if nxt() == "у" and (nxt(2) or "") in _U_PRONOUNS:
            return 0
        if nxt() == "всё":
            return 0
    elif k == "сети":
        if prev() in _GEN_ONLY:
            return 0
    elif k == "двери":
        if prev() not in _LOC_PREP and prev(2) not in _LOC_PREP:
            return 0
    elif k == "среду":
        if prev() in {"в", "во"}:
            return 0
    elif k == "стоит":
        j = 1
        while (w := nxt(j)) is not None and _NUMBER.fullmatch(w):
            j += 1
        if j > 1 and w is not None and _CURRENCY.fullmatch(w):
            return 0
    elif k == "села":
        if prev() not in _GEN_PREP:
            return 0
    return None


def fix(text: str, stressed: str) -> str:
    """Apply the rules to the accentor output `stressed` for input `text`."""
    tin, tout = _TOKEN.findall(text), _TOKEN.findall(stressed)
    if len(tin) != len(tout):
        return stressed
    words, gaps, pos, gap = [], [], [], ""
    for j, t in enumerate(tin):
        if t[0].isalnum() or t[0] in "_+":
            if _key(t) != _key(tout[j]):
                return stressed
            words.append(t)
            gaps.append(gap)
            pos.append(j)
            gap = ""
        else:
            gap = t
    for wi, j in enumerate(pos):
        v = _rule(words, gaps, wi)
        if v is not None:
            tout[j] = _stress(tout[j], v)
    return "".join(tout)


class FixedAccentor:
    def __init__(self, accentor) -> None:
        self.accentor = accentor

    def __call__(self, text: str, *args, **kwargs) -> str:
        return fix(text, self.accentor(text, *args, **kwargs))

    def __getattr__(self, name: str):
        return getattr(self.accentor, name)


# Words outside the fork's homographs.txt never reach silero, so their context rules mark the
# input text with the fork's own '+' before normalization (the fork turns '+' into U+0301).
# espeak (mitrokun) reads «хрена» as хре́на; «ни/до хрена» is хрена́ (user report 2026-11-05).
_PRE_MARKS = [
    (re.compile(r"(?i)\b(ни|до)(\s+)хрена\b"), r"\1\2хрен+а"),
]


def pre_mark(text: str) -> str:
    for pattern, repl in _PRE_MARKS:
        text = pattern.sub(repl, text)
    return text


def install() -> None:
    import silero_stress
    from wyoming_piper.russian_normalizer import RussianNormalizer

    original = silero_stress.load_accentor
    if getattr(original, "_stress_fix_ru", False):
        return

    def load_accentor(*args, **kwargs):
        return FixedAccentor(original(*args, **kwargs))

    load_accentor._stress_fix_ru = True  # type: ignore[attr-defined]
    silero_stress.load_accentor = load_accentor

    normalize = RussianNormalizer.normalize

    def normalize_marked(self, text: str) -> str:
        return normalize(self, pre_mark(text)) if text else text

    RussianNormalizer.normalize = normalize_marked

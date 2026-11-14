"""Spell out measurement units the mitrokun RussianNormalizer leaves as symbols.

espeak reads "21 °C" as "двадцать один си" and "5 м/с" as "пять эм эс". Only masculine units
are handled: the number stays as digits (espeak / the fork's float handling read them), and only
the unit word is inflected, so number–noun gender agreement is never needed.

`install()` wraps RussianNormalizer.normalize; both the server (docker/entrypoint.sh) and
dataset preparation (train/textprep.py) call it so training text matches server text.
"""

from __future__ import annotations

import re

# unit regex → (1, 2–4, 5+) forms; fractions take the 2–4 form ("21,5 градуса").
_UNITS: list[tuple[str, tuple[str, str, str]]] = [
    (r"°\s*[CС]\b|°", ("градус", "градуса", "градусов")),
    (r"м/с\b", ("метр в секунду", "метра в секунду", "метров в секунду")),
    (r"км/ч\b", ("километр в час", "километра в час", "километров в час")),
    (r"мм\.?\s*рт\.?\s*ст\.?", ("миллиметр ртутного столба", "миллиметра ртутного столба",
                                "миллиметров ртутного столба")),
    (r"гПа\b", ("гектопаскаль", "гектопаскаля", "гектопаскалей")),
    (r"кВт[·⋅*]?ч\b", ("киловатт-час", "киловатт-часа", "киловатт-часов")),
    (r"кВт\b", ("киловатт", "киловатта", "киловатт")),
    (r"Вт\b", ("ватт", "ватта", "ватт")),
    (r"мм\b", ("миллиметр", "миллиметра", "миллиметров")),
]

_NUM = r"(?P<sign>[-−+]?)(?P<num>\d+(?:[.,]\d+)?)"
_PATTERNS = [(re.compile(_NUM + r"\s*(?:" + rx + r")"), forms) for rx, forms in _UNITS]


def _form(num: str, forms: tuple[str, str, str]) -> str:
    if "," in num or "." in num:
        return forms[1]
    n = int(num)
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


def expand_units(text: str) -> str:
    for pattern, forms in _PATTERNS:
        def repl(m: re.Match, forms=forms) -> str:
            sign = {"-": "минус ", "−": "минус ", "+": "плюс "}.get(m.group("sign"), "")
            return f"{sign}{m.group('num')} {_form(m.group('num'), forms)}"

        text = pattern.sub(repl, text)
    return text


def install() -> None:
    """Expand units before both fork normalizers. The handler runs EnglishNormalizer first when the
    text has Latin letters, and the "C" in "°C" is Latin, so it must be expanded there too."""
    from wyoming_piper.english_normalizer import EnglishNormalizer
    from wyoming_piper.russian_normalizer import RussianNormalizer

    for cls in (EnglishNormalizer, RussianNormalizer):
        original = cls.normalize
        if getattr(original, "_units_ru", False):
            continue

        def normalize(self, text: str, _original=original) -> str:
            return _original(self, expand_units(text)) if text else text

        normalize._units_ru = True  # type: ignore[attr-defined]
        cls.normalize = normalize

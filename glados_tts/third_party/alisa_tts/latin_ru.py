"""Russian spelling for Latin words: brand/artist dictionary first, letter rules for the rest.

The fork transliterates many English words (YouTube → ютуб, Wi-Fi → вай фай), but unknown ones
(Spotify, Netflix, OpenAI, Ozon) reach espeak in Latin, and the ru voice then switches to English
phonemes (spˈɒtɪfˌaɪ, ˈəʊpən ˈeɪ ˈaɪ). The model never hears those phonemes in Russian training
data, so they come out garbled; and some of the fork's own transliterations of names are odd
(Metallica → «мэтэликэ», Billie Eilish → «били еилиш»).

`install()` wraps both fork normalizers: before EnglishNormalizer the dictionary (`BRANDS`, also
multi-word names) replaces known names; before RussianNormalizer any Latin word still left gets the
dictionary, abbreviation letter names (USB → ю эс би) or letter rules (Ozon → озон). Used by
docker/entrypoint.sh and train/textprep.py alike.
"""

from __future__ import annotations

import re

BRANDS = {
    "spotify": "спотифай", "netflix": "нетфликс", "zigbee": "зигби", "openai": "опен эй ай",
    "chatgpt": "чат джи пи ти", "gpt": "джи пи ти", "deepseek": "дипсик", "xbox": "иксбокс",
    "tiktok": "тикток", "macbook": "макбук", "airpods": "эйрподс", "nvidia": "энвидиа",
    "coldplay": "колдплей", "rammstein": "рамштайн", "eminem": "эминем", "github": "гитхаб",
    "proxmox": "проксмокс", "portainer": "портейнер", "ozon": "озон", "wildberries": "вайлдберриз",
    "avito": "авито", "sber": "сбер", "alexa": "алекса", "siri": "сири", "ipad": "айпад",
    "kindle": "киндл", "uber": "убер", "ikea": "икеа", "amd": "эй эм ди", "intel": "интел",
    "queen": "куин", "metallica": "металлика", "beatles": "битлз", "zoom": "зум", "skype": "скайп",
    "excel": "эксель", "docker": "докер", "twitch": "твич", "discord": "дискорд", "steam": "стим",
    "tesla": "тесла", "samsung": "самсунг", "xiaomi": "сяоми", "philips": "филипс", "hue": "хью",
    "android": "андроид", "linux": "линукс", "ubuntu": "убунту", "windows": "виндоус",
    "instagram": "инстаграм", "facebook": "фейсбук", "twitter": "твиттер", "lego": "лего",
    "adidas": "адидас", "nike": "найк", "bmw": "бэ эм вэ", "toyota": "тойота", "lexus": "лексус",
    "audi": "ауди", "disney": "дисней", "marvel": "марвел", "hbo": "эйч би о", "mqtt": "эм ку ти ти",
    "esphome": "эс пи хоум", "wyoming": "вайоминг", "piper": "пайпер", "silero": "силеро",
    "the beatles": "битлз", "depeche mode": "депеш мод", "billie eilish": "билли айлиш",
    "imagine dragons": "имэджин драгонс", "pink floyd": "пинк флойд", "daft punk": "дафт панк",
    "arctic monkeys": "арктик манкиз", "lana del rey": "лана дель рей", "ed sheeran": "эд ширан",
    "linkin park": "линкин парк", "bohemian rhapsody": "богемиан рапсоди",
}
_KNOWN = re.compile(
    r"(?<![A-Za-z])(" + "|".join(re.escape(k) for k in sorted(BRANDS, key=len, reverse=True))
    + r")(?![A-Za-z])", re.IGNORECASE)


def known_to_cyrillic(text: str) -> str:
    return _KNOWN.sub(lambda m: BRANDS[m.group(1).lower()], text)


# Letter rules for the fallback, longest first; vowels read as written («Ozon» → «озон»).
_RULES = [
    ("sch", "ш"), ("tch", "ч"), ("sh", "ш"), ("ch", "ч"), ("zh", "ж"), ("th", "т"), ("ph", "ф"),
    ("kh", "х"), ("ck", "к"), ("qu", "кв"), ("ee", "и"), ("oo", "у"), ("ya", "я"), ("yu", "ю"),
    ("yo", "ё"), ("ye", "е"), ("ts", "ц"),
    ("a", "а"), ("b", "б"), ("c", "к"), ("d", "д"), ("e", "е"), ("f", "ф"), ("g", "г"), ("h", "х"),
    ("i", "и"), ("j", "дж"), ("k", "к"), ("l", "л"), ("m", "м"), ("n", "н"), ("o", "о"), ("p", "п"),
    ("q", "к"), ("r", "р"), ("s", "с"), ("t", "т"), ("u", "у"), ("v", "в"), ("w", "в"), ("x", "кс"),
    ("y", "и"), ("z", "з"),
]
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z']*")


def _letters(word: str) -> str:
    w, out, i = word.lower().replace("'", ""), [], 0
    while i < len(w):
        for src, dst in _RULES:
            if w.startswith(src, i):
                out.append(dst)
                i += len(src)
                break
        else:
            i += 1
    return "".join(out)


_LETTER_NAMES = dict(zip(
    "abcdefghijklmnopqrstuvwxyz",
    "эй би си ди и эф джи эйч ай джей кей эл эм эн оу пи кью ар эс ти ю ви дабл-ю экс уай зед".split(),
))


def to_cyrillic(text: str) -> str:
    def repl(m: re.Match) -> str:
        word = m.group()
        if len(word) <= 4 and word.isupper():
            return " ".join(_LETTER_NAMES[c] for c in word.lower())  # USB → ю эс би
        return _letters(word)

    return _LATIN_WORD.sub(repl, known_to_cyrillic(text))


def install() -> None:
    from wyoming_piper.english_normalizer import EnglishNormalizer
    from wyoming_piper.russian_normalizer import RussianNormalizer

    for cls, convert in ((EnglishNormalizer, known_to_cyrillic), (RussianNormalizer, to_cyrillic)):
        original = cls.normalize
        if getattr(original, "_latin_ru", False):
            continue

        def normalize(self, text: str, _original=original, _convert=convert) -> str:
            return _original(self, _convert(text)) if text else text

        normalize._latin_ru = True  # type: ignore[attr-defined]
        cls.normalize = normalize

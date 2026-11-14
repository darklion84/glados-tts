# alisa-tts text modules

These four modules come from the **alisa-tts** project (a Russian Piper voice for Home Assistant)
by a third-party author. They are included **unmodified**, with the author's permission to use and
modify them on the condition that derived work is published openly — this repository is that
publication. They are not covered by this repository's MIT license; if you reuse them elsewhere,
keep this notice and publish your result openly as well.

| Module | What it does here |
|---|---|
| `cpu_threads.py` | Sizes onnxruntime / torch thread pools to `TTS_THREADS` (or the CPU quota) and disables onnxruntime spin-waiting |
| `units_ru.py` | Spells out units the mitrokun normalizer leaves as symbols: °C, м/с, км/ч, %… with number agreement |
| `latin_ru.py` | Russian spelling for Latin words: brand/artist dictionary (Spotify → «спотифай»), abbreviations, letter rules |
| `stress_fix_ru.py` | Grammar rules over silero-stress for frequent homographs in assistant speech («уже до́ма», «нет се́ти», «в сре́ду»…) |

Comments inside the modules mention files of the original project (`docker/entrypoint.sh`,
`train/textprep.py`, `docs/plan.md`); those files are not part of this repository.

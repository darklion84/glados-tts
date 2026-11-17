# Third-party components

The MIT license in [LICENSE](LICENSE) covers the code of this repository except
`glados_tts/third_party/` and the components below (GLaDOS TTS first, then the STT Filter add-on), which keep their own terms.

| Component | Where | Terms |
|---|---|---|
| **GLaDOS voice model and pronunciation dictionary** — [TeraTTS/glados2-g2p-vits](https://huggingface.co/TeraTTS/glados2-g2p-vits) | downloaded at build time, **not included** | No license stated by the authors. GLaDOS is a character owned by Valve. Personal, non-commercial use only. |
| **TeraTTS g2p rules** — [Tera2Space/TeraTTS](https://github.com/Tera2Space/TeraTTS) | re-implemented in `glados_tts/app/glados_g2p.py` (`rules()`), input layout of `TokenizerG2P` / `TTS._get_seq` | MIT, © Tera2Space |
| **Russian/English text normalizers** — [mitrokun/espeak-ng-data](https://github.com/mitrokun/espeak-ng-data) (`wyoming_piper_accent/2.0+`) | downloaded at build time from a pinned commit, **not included** | No license stated in the upstream repository |
| **alisa-tts text modules** (`cpu_threads`, `units_ru`, `latin_ru`, `stress_fix_ru`) | `glados_tts/third_party/alisa_tts/` | Provided by their author with permission to use and modify, on the condition that derived work is published openly. See the README in that folder. |
| [silero-stress](https://github.com/snakers4/silero-stress) | pip | MIT |
| [Wyoming](https://github.com/OHF-Voice/wyoming) | pip | MIT |
| [num2words](https://github.com/savoirfairelinux/num2words) | pip | LGPL-2.1 |
| [onnxruntime](https://github.com/microsoft/onnxruntime), [PyTorch](https://github.com/pytorch/pytorch) | pip | MIT / BSD-style (see the PyTorch repository) |
| **Speaker embedding model** (STT Filter) — CAM++ `speech_campplus_sv_zh_en_16k-common_advanced` from [3D-Speaker](https://github.com/modelscope/3D-Speaker), ONNX export from the [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models) | downloaded at build time, **not included** | Apache-2.0 |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (STT Filter) | pip | Apache-2.0 |
| [NumPy](https://github.com/numpy/numpy) | pip | BSD-3-Clause |

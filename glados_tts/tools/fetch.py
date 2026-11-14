"""Build-time downloads (nothing below is redistributed in this repository).

1. TeraTTS/glados2-g2p-vits from Hugging Face at a pinned revision, verified by SHA-256.
2. mitrokun's text normalizers for wyoming-piper (github.com/mitrokun/espeak-ng-data) at a pinned
   commit, installed as the `wyoming_piper` package (only the modules the normalizers need).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import tarfile
import urllib.request
from pathlib import Path

HF_REPO = "TeraTTS/glados2-g2p-vits"
HF_FILES = {
    "model.onnx": "27696246a6dbaa15aa7a02f3972b78b17f2c14e82a754fd59c63ee0639c87086",
    "config.json": "a8f1434cd629d56d53a856776fb40324d405b3c41785556708997ce40a2f9de4",
    "dictionary.txt": "0da7b639bf4ff0bf15fd3a31b3d082b1681204cab4b752f00d54a7409290b165",
}
MITROKUN_REPO = "mitrokun/espeak-ng-data"
MITROKUN_DIR = "wyoming_piper_accent/2.0+/"
MITROKUN_FILES = {
    "__init__.py", "russian_normalizer.py", "english_normalizer.py", "sentence_boundary.py",
    "vse_yoficator.py", "yo.dat", "yo.txt", "user.txt",
}


def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "glados-tts-build"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def fetch_model(revision: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, sha in HF_FILES.items():
        data = download(f"https://huggingface.co/{HF_REPO}/resolve/{revision}/exported/{name}")
        got = hashlib.sha256(data).hexdigest()
        if got != sha:
            raise SystemExit(f"{name}: sha256 {got} != expected {sha}")
        (dest / name).write_bytes(data)
        print(f"{name}: {len(data)} bytes, sha256 ok")


def fetch_normalizers(commit: str, site_packages: Path) -> None:
    data = download(f"https://codeload.github.com/{MITROKUN_REPO}/tar.gz/{commit}")
    pkg = site_packages / "wyoming_piper"
    pkg.mkdir(parents=True, exist_ok=True)
    found = set()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            _, _, rel = member.name.partition("/")  # strip "<repo>-<commit>/"
            if not rel.startswith(MITROKUN_DIR) or not member.isfile():
                continue
            name = rel[len(MITROKUN_DIR):]
            if name in MITROKUN_FILES:
                (pkg / name).write_bytes(tar.extractfile(member).read())
                found.add(name)
    missing = MITROKUN_FILES - found
    if missing:
        raise SystemExit(f"mitrokun {commit}: missing {sorted(missing)}")
    print(f"wyoming_piper normalizers from {MITROKUN_REPO}@{commit[:12]}: {len(found)} files")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-revision", required=True)
    ap.add_argument("--mitrokun-commit", required=True)
    ap.add_argument("--voice-dir", type=Path, required=True)
    ap.add_argument("--site-packages", type=Path, required=True)
    args = ap.parse_args()
    fetch_model(args.hf_revision, args.voice_dir)
    fetch_normalizers(args.mitrokun_commit, args.site_packages)


if __name__ == "__main__":
    main()

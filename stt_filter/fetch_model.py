"""Сборка образа: скачать модель отпечатков голоса из релиза sherpa-onnx и сверить SHA-256."""
import hashlib
import os
import sys
import urllib.parse
import urllib.request

name, sha256, out_dir = sys.argv[1:4]
url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/" + urllib.parse.quote(name)
data = urllib.request.urlopen(url, timeout=300).read()
if hashlib.sha256(data).hexdigest() != sha256:
    sys.exit(f"{name}: не та контрольная сумма")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, name), "wb") as f:
    f.write(data)

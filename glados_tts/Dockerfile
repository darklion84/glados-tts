# GLaDOS TTS: Wyoming server for Home Assistant.
# Builds as a plain Docker image and as a Home Assistant add-on (amd64).
# Model, dictionary and mitrokun normalizers are downloaded at build time (pinned, verified),
# they are not part of this repository.
FROM python:3.12-slim

ARG TORCH_VERSION=2.15.0
ARG HF_REVISION=d4a7c43fc4c2ebe77c8bacc58d4c3bcfc52ff519
ARG MITROKUN_COMMIT=ee2e19b3cd16cde72cab0e67b7c131adf74e05d5

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    PYTHONPATH=/opt/glados/lib

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}" \
 && pip install --no-cache-dir -r /tmp/requirements.txt

COPY tools/ /opt/glados/tools/
RUN SP="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
 && python /opt/glados/tools/fetch.py --hf-revision "${HF_REVISION}" --mitrokun-commit "${MITROKUN_COMMIT}" \
      --voice-dir /voice-glados --site-packages "${SP}" \
 && python /opt/glados/tools/build_dict.py /voice-glados/dictionary.txt /voice-glados/config.json /voice-glados/dictionary.sqlite \
 && rm /voice-glados/dictionary.txt

COPY third_party/alisa_tts/ /opt/glados/lib/
COPY app/ /opt/glados/app/
COPY run.sh /run.sh

EXPOSE 10200
ENTRYPOINT ["/bin/sh", "/run.sh"]

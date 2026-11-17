# sherpa-onnx ставится готовым колесом только под glibc, поэтому Debian, а не Alpine.
FROM python:3.12-slim
ARG MODEL=3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx
ARG MODEL_SHA256=aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir wyoming==1.11.0 sherpa-onnx==2.0.1 numpy==2.5.4
# Модель отпечатков голоса (3D-Speaker CAM++, 28 МБ) — из релиза sherpa-onnx, с проверкой суммы.
COPY fetch_model.py /tmp/fetch_model.py
RUN python /tmp/fetch_model.py "${MODEL}" "${MODEL_SHA256}" /models && rm /tmp/fetch_model.py
COPY main.py stt_filter.py speaker.py tts_router.py /app/
COPY run.sh /run.sh
ENTRYPOINT ["/bin/sh", "/run.sh"]

# Chatterbox TTS serverless image for RunPod.
#
# Pre-bakes Turbo EN + Multilingual weights so cold starts skip the
# ~20-min pip install Boris hit during the field test. All pip activity
# happens in a single RUN so uninstalled packages (gradio, pre-commit,
# whisper, etc.) don't leak into earlier layers.

FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Single RUN so uninstalls + cache purges actually shrink the layer.
# - Drop base image's torch 2.8 triplet before installing torch 2.6 (saves ~4 GB).
# - Pin torchvision 0.21 so transformers' lazy image-model imports don't trip
#   `operator torchvision::nms does not exist`.
# - Install chatterbox-tts + its minimum-viable runtime deps.
# - Uninstall the web-demo / dev-tool transitive deps (gradio, pre-commit,
#   whisper, fastapi-cli, gradio-client, etc.) — we only render sentences.
# - Purge pip + HF caches at the end so nothing compressed sticks around.
RUN pip uninstall -y torch torchvision torchaudio 2>&1 | tail -3 \
 && pip install --upgrade pip \
 && pip install 'setuptools<81' \
 && pip install --index-url https://download.pytorch.org/whl/cu124 \
      torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 \
 && pip install \
      chatterbox-tts==0.1.7 \
      transformers==5.2.0 \
      soundfile==0.13.1 \
      numpy==1.26.4 \
      runpod==1.7.7 \
 && pip uninstall -y \
      gradio gradio-client \
      pre-commit \
      openai-whisper \
      fastapi-cli fastapi-cloud-cli safehttpx \
      uvicorn uvloop httptools watchfiles \
      rich-toolkit typer typer-slim \
      inquirerpy pfzy shellingham \
      virtualenv nodeenv identify cfgv \
      sentry-sdk semantic-version \
      paramiko bcrypt pynacl \
      boto3 botocore s3transfer jmespath \
      aiohttp aiohttp-retry aiosignal aiofiles aiohappyeyeballs aiodns \
      orjson python-multipart pydantic-settings pydantic-extra-types \
      fastar ffmpy groovy invoke tqdm-loggable rignore \
      2>&1 | tail -5 \
 && rm -rf /root/.cache/pip /tmp/*

# Pre-download both model variants (runs on CPU — we only care about weight
# files landing in HF_HOME, which ships with the image).
RUN python -c "from chatterbox.tts_turbo import ChatterboxTurboTTS; ChatterboxTurboTTS.from_pretrained(device='cpu')" \
 && python -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS; ChatterboxMultilingualTTS.from_pretrained(device='cpu')"

WORKDIR /app
COPY handler.py /app/handler.py

ENTRYPOINT ["python", "-u", "/app/handler.py"]

# Chatterbox TTS serverless image for RunPod.
#
# Bakes both model variants (Turbo EN + Multilingual) into the image so cold
# starts don't re-download ~3.5 GB of weights. Built on RunPod's PyTorch 2.8
# + CUDA 12.8 base; chatterbox-tts 0.1.7 is the fleet-canonical version, so
# we re-pin torch==2.6.0/torchaudio==2.6.0 to match what m1/m2/m4 run.

FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Uninstall the base image's torch 2.8 + pip junk drawer FIRST so we don't
# carry both triplets in a single layer (saves ~4 GB of compressed pull).
RUN pip uninstall -y torch torchvision torchaudio || true
RUN pip cache purge || true

# perth (a chatterbox transitive dep) imports pkg_resources, removed in
# setuptools >=81. Pin it first so the downstream install doesn't pull a
# breaking newer wheel.
RUN pip install --upgrade pip \
 && pip install 'setuptools<81'

# Re-pin torch to the fleet version (2.6.0) and torchvision to the matching
# 0.21.0 so transformers' image-model lazy imports don't trip
# `torchvision::nms does not exist`.
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
      torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0

RUN pip install \
      chatterbox-tts==0.1.7 \
      transformers==5.2.0 \
      soundfile==0.13.1 \
      numpy==1.26.4 \
      runpod==1.7.7

# Drop chatterbox's transitive dev/web-demo deps we never touch at runtime.
# gradio alone is ~200 MB, pre-commit ~50 MB, whisper ~80 MB. Lose them and
# the compressed layer gets meaningfully smaller.
RUN pip uninstall -y \
      gradio gradio-client \
      pre-commit \
      openai-whisper \
      fastapi fastapi-cli fastapi-cloud-cli safehttpx \
      uvicorn uvloop httptools watchfiles websockets \
      rich rich-toolkit typer typer-slim \
      inquirerpy pfzy prompt_toolkit shellingham \
      virtualenv nodeenv identify cfgv \
      sentry-sdk semantic-version \
      paramiko bcrypt pynacl \
      boto3 botocore s3transfer jmespath \
      aiohttp aiohttp-retry aiosignal aiofiles aiohappyeyeballs aiodns \
      orjson starlette itsdangerous \
      python-multipart python-dotenv pydantic-settings pydantic-extra-types \
      fastar ffmpy groovy invoke tqdm-loggable rignore \
      2>&1 | tail -3

RUN pip cache purge || true

# Pre-download both model variants to /root/.cache/huggingface. This bakes
# ~3.5 GB of weights into the image layer so cold starts are fast. Run
# against CPU — we only care about downloading, not loading onto a GPU.
RUN python -c "from chatterbox.tts_turbo import ChatterboxTurboTTS; ChatterboxTurboTTS.from_pretrained(device='cpu')" \
 && python -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS; ChatterboxMultilingualTTS.from_pretrained(device='cpu')"

WORKDIR /app
COPY handler.py /app/handler.py

# runpod.serverless.start listens on stdin/stdout for local tests; in
# serverless mode it auto-detects RUNPOD_ENDPOINT_ID / worker env vars.
ENTRYPOINT ["python", "-u", "/app/handler.py"]

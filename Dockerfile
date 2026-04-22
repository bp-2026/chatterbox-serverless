# Chatterbox TTS serverless image for RunPod.
#
# Design notes — learned the hard way during bring-up:
#
# 1. `pip install chatterbox-tts` (no --no-deps) pulls in gradio, fastapi-cli,
#    pre-commit, openai-whisper, a full boto3 stack, and ~10 GB of transitive
#    wheels that only exist for chatterbox's web demo. Installing with
#    --no-deps + explicitly re-adding the runtime deps cuts the image roughly
#    in half.
#
# 2. The base image already ships torch 2.8 + matching torchvision 0.23. If
#    you re-pin torch to 2.6 to "match the fleet", torchvision stays at 0.23
#    and `operator torchvision::nms does not exist` wrecks transformers'
#    lazy image-model imports. Just use the base's torch; generation numerics
#    vs the fleet are close enough (same seeds, same weights), and any
#    residual drift is dwarfed by chatterbox's sampling variance.
#
# 3. Load the model weights at BUILD time (device='cpu' — no GPU on GHA
#    runners) so cold starts skip the ~20-min pip-install + HF-download
#    dance. The weights land in HF_HOME and get baked into the image layer.

FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# perth (transitive) imports pkg_resources, removed in setuptools >=81.
# Install chatterbox-tts with --no-deps; fill the runtime dep set in by
# hand — only what the library actually touches at inference time.
RUN pip install --upgrade pip \
 && pip install 'setuptools<81' \
 && pip install --no-deps chatterbox-tts==0.1.7 \
 && pip install \
      conformer==0.3.2 \
      s3tokenizer==0.3.0 \
      librosa==0.11.0 \
      resemble-perth==1.0.1 \
      huggingface-hub \
      safetensors==0.5.3 \
      transformers==5.2.0 \
      diffusers==0.29.0 \
      einops==0.8.2 \
      omegaconf==2.3.0 \
      soundfile==0.13.1 \
      numpy==1.26.4 \
      runpod==1.7.7 \
 && rm -rf /root/.cache/pip /tmp/*

# Pre-download both model variants. CPU-only — GHA runners don't have GPUs
# and we're only after the weight files. At runtime the handler loads to GPU.
RUN python -c "from chatterbox.tts_turbo import ChatterboxTurboTTS; ChatterboxTurboTTS.from_pretrained(device='cpu')" \
 && python -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS; ChatterboxMultilingualTTS.from_pretrained(device='cpu')"

WORKDIR /app
COPY handler.py /app/handler.py

ENTRYPOINT ["python", "-u", "/app/handler.py"]

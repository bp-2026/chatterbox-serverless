#!/usr/bin/env python3
"""RunPod serverless handler for Chatterbox TTS.

One request = one sentence. Both model variants (Turbo EN + Multilingual)
are loaded onto GPU at module import time so they stay resident across
requests. Voice-prompt conditionals for the Turbo path are cached by SHA256
so back-to-back requests for the same voice skip the ~0.5s prepare step.

Input JSON (the "input" field of a RunPod job):
    {
      "text":             "<sentence to render>",
      "language":         "en-gb" | "fr-fr" | any "en-*" or other ISO lang-region,
      "voice_prompt_b64": "<base64-encoded reference WAV>",
      "seed":             <optional int>,
      "options": {
        "exaggeration":  <optional float>,
        "cfg_weight":    <optional float>,
        "temperature":   <optional float>,
        "is_heading":    <optional bool, uses heading defaults>
      }
    }

Output JSON:
    {
      "wav_b64":         "<base64-encoded 24 kHz mono PCM16 WAV>",
      "sr":              24000,
      "render_time_s":   8.4,
      "engine":          "turbo" | "multilingual"
    }

Defaults mirror pipelines/audio/tools/chatterbox_sentence_render.py so
cloud-rendered sentences are drop-in compatible with locally-rendered ones.
"""
from __future__ import annotations

import base64
import hashlib
import io
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio

ENGLISH_DEFAULTS = {
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "temperature": 0.7,
    "heading_exaggeration": 0.6,
    "heading_cfg_weight": 0.3,
}
MULTILINGUAL_DEFAULTS = {
    "exaggeration": 0.35,
    "cfg_weight": 0.3,
    "temperature": 0.45,
    "heading_exaggeration": 0.4,
    "heading_cfg_weight": 0.25,
}

TARGET_SR = 24000
TAIL_FADE_MS = 35

_VOICE_CACHE_DIR = Path(tempfile.gettempdir()) / "chatterbox-voices"
_VOICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = _resolve_device()
print(f"[handler] device={DEVICE} torch={torch.__version__} cuda_available={torch.cuda.is_available()}")

# Lazy-loaded on first use; both are small-ish compared to typical serverless
# cold starts and we want them warm before the first request arrives, so we
# eagerly load at import time.
print("[handler] loading ChatterboxTurboTTS…")
from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: E402
TURBO_EN = ChatterboxTurboTTS.from_pretrained(device=DEVICE)
print(f"[handler] turbo sr={TURBO_EN.sr}")

print("[handler] loading ChatterboxMultilingualTTS…")
from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: E402
MULTI = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
print(f"[handler] multilingual sr={MULTI.sr}")

# Turbo uses prepare_conditionals(voice_prompt) — we track which voice it's
# currently conditioned on so we only re-prepare when it changes.
_turbo_prepared_sha: str | None = None


def _b64_to_voice_file(voice_b64: str) -> tuple[Path, str]:
    """Decode a base64 voice prompt to a temp file keyed by its SHA256.
    Returns (path, sha256). Reuses the same path on cache hit."""
    raw = base64.b64decode(voice_b64)
    sha = hashlib.sha256(raw).hexdigest()
    path = _VOICE_CACHE_DIR / f"{sha}.wav"
    if not path.exists():
        path.write_bytes(raw)
    return path, sha


def _apply_tail_fade(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if TAIL_FADE_MS <= 0:
        return audio
    fade_samples = min(audio.shape[-1], int(sample_rate * TAIL_FADE_MS / 1000))
    if fade_samples <= 1:
        return audio
    fade = torch.linspace(1.0, 0.0, fade_samples, device=audio.device, dtype=audio.dtype)
    audio = audio.clone()
    audio[..., -fade_samples:] *= fade
    return audio


def _tensor_to_wav_bytes(audio: torch.Tensor, sample_rate: int) -> bytes:
    """Mono 24 kHz PCM16 WAV — matches save_as_wav() in the local renderer."""
    if audio.dim() == 2:
        audio = audio.mean(dim=0, keepdim=True) if audio.shape[0] > 1 else audio
    else:
        audio = audio.unsqueeze(0)
    if sample_rate != TARGET_SR:
        audio = torchaudio.functional.resample(audio, sample_rate, TARGET_SR)
    audio = _apply_tail_fade(audio, TARGET_SR)
    arr = audio.squeeze(0).detach().cpu().numpy().astype(np.float32)
    arr = np.clip(arr, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, arr, TARGET_SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _pick_defaults(lang_iso: str, is_heading: bool) -> dict:
    table = ENGLISH_DEFAULTS if lang_iso == "en" else MULTILINGUAL_DEFAULTS
    if is_heading:
        return {
            "exaggeration": table["heading_exaggeration"],
            "cfg_weight": table["heading_cfg_weight"],
            "temperature": table["temperature"],
        }
    return {
        "exaggeration": table["exaggeration"],
        "cfg_weight": table["cfg_weight"],
        "temperature": table["temperature"],
    }


def _render(
    text: str,
    language: str,
    voice_path: Path,
    voice_sha: str,
    options: dict,
) -> tuple[bytes, str, int]:
    global _turbo_prepared_sha

    lang_iso = (language or "en").split("-")[0].lower()
    is_heading = bool(options.get("is_heading", False))
    defaults = _pick_defaults(lang_iso, is_heading)
    exag = float(options.get("exaggeration", defaults["exaggeration"]))
    cfg = float(options.get("cfg_weight", defaults["cfg_weight"]))
    temp = float(options.get("temperature", defaults["temperature"]))

    if lang_iso != "en":
        audio = MULTI.generate(
            text=text,
            language_id=lang_iso,
            audio_prompt_path=str(voice_path),
            exaggeration=exag,
            cfg_weight=cfg,
            temperature=temp,
        )
        engine = "multilingual"
        src_sr = MULTI.sr
    else:
        if _turbo_prepared_sha != voice_sha:
            TURBO_EN.prepare_conditionals(str(voice_path))
            _turbo_prepared_sha = voice_sha
        audio = TURBO_EN.generate(
            text=text,
            temperature=temp,
        )
        engine = "turbo"
        src_sr = TURBO_EN.sr

    wav_bytes = _tensor_to_wav_bytes(audio, src_sr)
    return wav_bytes, engine, src_sr


def _coerce_input(event: dict[str, Any]) -> dict[str, Any]:
    """RunPod wraps the job body as {"input": {...}}; accept either shape."""
    if isinstance(event, dict) and "input" in event and isinstance(event["input"], dict):
        return event["input"]
    return event


def handler(event: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    payload = _coerce_input(event)

    text = payload.get("text")
    if not text or not isinstance(text, str):
        return {"error": "missing 'text' (string)"}

    voice_b64 = payload.get("voice_prompt_b64")
    if not voice_b64 or not isinstance(voice_b64, str):
        return {"error": "missing 'voice_prompt_b64' (base64 WAV)"}

    language = payload.get("language") or "en"
    options = payload.get("options") or {}
    seed = payload.get("seed")

    if seed is not None:
        try:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        except Exception as exc:
            return {"error": f"bad seed: {exc}"}

    try:
        voice_path, voice_sha = _b64_to_voice_file(voice_b64)
    except Exception as exc:
        return {"error": f"voice decode failed: {exc}"}

    try:
        wav_bytes, engine, _src_sr = _render(text, language, voice_path, voice_sha, options)
    except Exception as exc:
        return {"error": f"render failed: {type(exc).__name__}: {exc}"}

    return {
        "wav_b64": base64.b64encode(wav_bytes).decode("ascii"),
        "sr": TARGET_SR,
        "render_time_s": round(time.perf_counter() - t0, 3),
        "engine": engine,
    }


if __name__ == "__main__":
    # Dev mode: read a JSON event from stdin, write response to stdout.
    # RunPod sets RUNPOD_POD_ID / RUNPOD_ENDPOINT_ID in serverless; in that
    # case we hand control to runpod.serverless.start().
    if os.environ.get("RUNPOD_ENDPOINT_ID") or os.environ.get("RUNPOD_POD_ID") or os.environ.get("RUNPOD_SERVERLESS") == "1":
        import runpod
        runpod.serverless.start({"handler": handler})
    else:
        import json
        import sys
        raw = sys.stdin.read()
        event = json.loads(raw) if raw.strip() else {}
        result = handler(event)
        sys.stdout.write(json.dumps(result))

#!/usr/bin/env python3
"""RunPod serverless handler for Chatterbox TTS — with aggressive logging.

One request = one sentence. Both model variants (Turbo EN + Multilingual)
load on demand, cached at module scope so subsequent requests reuse them.
Voice-prompt conditionals for the Turbo path are cached by SHA256.

Heavy step-by-step logging is intentional — when a worker misbehaves,
RunPod's per-worker console logs are the only signal we have, and silent
failures in model loading or SDK init are exactly what we need to catch.

Input JSON (the "input" field of a RunPod job):
    {
      "text":             "<sentence to render>",
      "language":         "en-gb" | "fr-fr" | ...,
      "voice_prompt_b64": "<base64-encoded reference WAV>",
      "seed":             <optional int>,
      "options":          { ... }   # exaggeration/cfg_weight/temperature/is_heading
    }

Special diagnostic requests:
    {"diag": "ping"}   — returns immediately without touching any model
    {"diag": "info"}   — returns python/torch/GPU info + model load status
"""
from __future__ import annotations

import base64
import hashlib
import io
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any


def _log(msg: str) -> None:
    """Timestamped stderr log that flushes immediately — RunPod captures
    both stdout and stderr and the flush guards against buffered disappearance."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True, file=sys.stderr)


_BOOT_T0 = time.time()
_log(f"handler.py booting — pid={os.getpid()} python={sys.version_info[:3]}")
_log(f"env: RUNPOD_ENDPOINT_ID={os.environ.get('RUNPOD_ENDPOINT_ID','-')} "
     f"RUNPOD_POD_ID={os.environ.get('RUNPOD_POD_ID','-')} "
     f"RUNPOD_SERVERLESS={os.environ.get('RUNPOD_SERVERLESS','-')}")

try:
    import numpy as np
    _log(f"imported numpy {np.__version__}")
except Exception as e:
    _log(f"FATAL numpy import: {e!r}")
    raise

try:
    import soundfile as sf
    _log(f"imported soundfile {sf.__version__}")
except Exception as e:
    _log(f"FATAL soundfile import: {e!r}")
    raise

try:
    import torch
    import torchaudio
    _log(f"imported torch {torch.__version__} torchaudio {torchaudio.__version__}")
    _log(f"cuda: available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            _log(f"  gpu[{i}]={torch.cuda.get_device_name(i)} "
                 f"mem_total={torch.cuda.get_device_properties(i).total_memory/1e9:.1f}GB")
except Exception as e:
    _log(f"FATAL torch import: {e!r}")
    raise


ENGLISH_DEFAULTS = {
    "exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.7,
    "heading_exaggeration": 0.6, "heading_cfg_weight": 0.3,
}
MULTILINGUAL_DEFAULTS = {
    "exaggeration": 0.35, "cfg_weight": 0.3, "temperature": 0.45,
    "heading_exaggeration": 0.4, "heading_cfg_weight": 0.25,
}

TARGET_SR = 24000
TAIL_FADE_MS = 35

_VOICE_CACHE_DIR = Path(tempfile.gettempdir()) / "chatterbox-voices"
_VOICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_log(f"resolved DEVICE={DEVICE}")

# Models are loaded lazily on first request — keeps cold-start below RunPod's
# internal init timeout. First en-gb request pays ~30-60s for Turbo; first
# non-en pays ~60-90s for Multilingual.
_TURBO_EN = None
_MULTI = None
_turbo_prepared_sha: str | None = None


def _ensure_turbo():
    global _TURBO_EN
    if _TURBO_EN is None:
        t0 = time.time()
        _log("loading ChatterboxTurboTTS from_pretrained…")
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        _TURBO_EN = ChatterboxTurboTTS.from_pretrained(device=DEVICE)
        _log(f"ChatterboxTurboTTS ready in {time.time()-t0:.1f}s  sr={_TURBO_EN.sr}")
    return _TURBO_EN


def _ensure_multi():
    global _MULTI
    if _MULTI is None:
        t0 = time.time()
        _log("loading ChatterboxMultilingualTTS from_pretrained…")
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        _MULTI = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
        _log(f"ChatterboxMultilingualTTS ready in {time.time()-t0:.1f}s  sr={_MULTI.sr}")
    return _MULTI


def _b64_to_voice_file(voice_b64: str) -> tuple[Path, str]:
    raw = base64.b64decode(voice_b64)
    sha = hashlib.sha256(raw).hexdigest()
    path = _VOICE_CACHE_DIR / f"{sha}.wav"
    if not path.exists():
        path.write_bytes(raw)
    return path, sha


def _apply_tail_fade(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    fade_samples = min(audio.shape[-1], int(sample_rate * TAIL_FADE_MS / 1000))
    if fade_samples <= 1:
        return audio
    fade = torch.linspace(1.0, 0.0, fade_samples, device=audio.device, dtype=audio.dtype)
    audio = audio.clone()
    audio[..., -fade_samples:] *= fade
    return audio


def _tensor_to_wav_bytes(audio: torch.Tensor, sample_rate: int) -> bytes:
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


def _render(text, language, voice_path, voice_sha, options):
    global _turbo_prepared_sha

    lang_iso = (language or "en").split("-")[0].lower()
    is_heading = bool(options.get("is_heading", False))
    defaults = _pick_defaults(lang_iso, is_heading)
    exag = float(options.get("exaggeration", defaults["exaggeration"]))
    cfg = float(options.get("cfg_weight", defaults["cfg_weight"]))
    temp = float(options.get("temperature", defaults["temperature"]))

    if lang_iso != "en":
        multi = _ensure_multi()
        _log(f"render: multilingual lang={lang_iso} len={len(text)}")
        audio = multi.generate(
            text=text, language_id=lang_iso,
            audio_prompt_path=str(voice_path),
            exaggeration=exag, cfg_weight=cfg, temperature=temp,
        )
        return audio, "multilingual", multi.sr

    turbo = _ensure_turbo()
    if _turbo_prepared_sha != voice_sha:
        _log(f"turbo prepare_conditionals voice_sha={voice_sha[:12]}…")
        turbo.prepare_conditionals(str(voice_path))
        _turbo_prepared_sha = voice_sha
    _log(f"render: turbo len={len(text)}")
    audio = turbo.generate(text=text, temperature=temp)
    return audio, "turbo", turbo.sr


def _coerce_input(event):
    if isinstance(event, dict) and "input" in event and isinstance(event["input"], dict):
        return event["input"]
    return event if isinstance(event, dict) else {}


def handler(event):
    t0 = time.perf_counter()
    payload = _coerce_input(event)

    diag = payload.get("diag")
    if diag == "ping":
        return {"pong": True, "uptime_s": round(time.time() - _BOOT_T0, 2)}
    if diag == "info":
        return {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "turbo_loaded": _TURBO_EN is not None,
            "multi_loaded": _MULTI is not None,
            "uptime_s": round(time.time() - _BOOT_T0, 2),
        }

    text = payload.get("text")
    if not text or not isinstance(text, str):
        return {"error": "missing 'text' (string)"}
    voice_b64 = payload.get("voice_prompt_b64")
    if not voice_b64 or not isinstance(voice_b64, str):
        return {"error": "missing 'voice_prompt_b64'"}

    language = payload.get("language") or "en"
    options = payload.get("options") or {}
    seed = payload.get("seed")
    if seed is not None:
        try:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        except Exception as e:
            return {"error": f"bad seed: {e}"}

    try:
        voice_path, voice_sha = _b64_to_voice_file(voice_b64)
    except Exception as e:
        _log(f"voice decode error: {e!r}")
        return {"error": f"voice decode: {e}"}

    try:
        audio, engine, src_sr = _render(text, language, voice_path, voice_sha, options)
        wav_bytes = _tensor_to_wav_bytes(audio, src_sr)
    except Exception as e:
        tb = traceback.format_exc()
        _log(f"render error: {e!r}\n{tb}")
        return {"error": f"render: {type(e).__name__}: {e}", "traceback": tb[-800:]}

    elapsed = round(time.perf_counter() - t0, 3)
    _log(f"render OK engine={engine} bytes={len(wav_bytes)} wall={elapsed}s")
    return {
        "wav_b64": base64.b64encode(wav_bytes).decode("ascii"),
        "sr": TARGET_SR,
        "render_time_s": elapsed,
        "engine": engine,
    }


if __name__ == "__main__":
    _log(f"module import phase complete in {time.time()-_BOOT_T0:.1f}s")
    if (os.environ.get("RUNPOD_ENDPOINT_ID")
            or os.environ.get("RUNPOD_POD_ID")
            or os.environ.get("RUNPOD_SERVERLESS") == "1"):
        _log("entering runpod.serverless.start")
        import runpod
        _log(f"runpod lib path={runpod.__file__}")
        runpod.serverless.start({"handler": handler})
    else:
        _log("no RunPod env vars set — reading JSON event from stdin")
        import json
        raw = sys.stdin.read()
        event = json.loads(raw) if raw.strip() else {}
        result = handler(event)
        sys.stdout.write(json.dumps(result))

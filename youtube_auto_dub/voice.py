"""Multi-engine speech synthesis: edge-tts default, Qwen3-TTS optional."""

import asyncio
import json
import logging
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Optional

import edge_tts
import soundfile as sf

from youtube_auto_dub.models import (
    EDGE_TTS_RETRIES,
    EDGE_TTS_RETRY_DELAY,
    EDGE_TTS_TIMEOUT,
    LANG_ALIAS,
    LANG_MAP_PATH,
    QWEN_CLONE_MAX_DURATION,
    QWEN_CLONE_MIN_DURATION,
    QWEN_CLONE_MIN_SPAN,
    QWEN_CLONE_MIN_WORDS,
    QWEN_COMPUTE_DTYPE,
    QWEN_DEFAULT_DEVICE,
    QWEN_MODEL_NAME,
    SR_TTS,
    THEME_PROMPTS,
    VOICE_MIN_FILE_SIZE,
    VOICE_PERSONAS,
)
from youtube_auto_dub.ui import console

log = logging.getLogger(__name__)


# ── Edge TTS voice lookup ───────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_lang_map() -> dict:
    with open(LANG_MAP_PATH, encoding="utf-8") as f:
        return json.load(f)


def pick_voice(lang: str, gender: str = "male") -> str:
    data = _load_lang_map()
    if lang not in data:
        raise ValueError(f"Language {lang} not in map")
    voices = data[lang]["voices"].get(gender, [])
    if not voices:
        alt = "female" if gender == "male" else "male"
        voices = data[lang]["voices"].get(alt, [])
        if not voices:
            raise ValueError(f"No voice for {lang}")
        console.warning(f"No {gender} voice, falling back to {alt}")
    return voices[0]


# ── Edge TTS ────────────────────────────────────────────────────────────

async def speak_edge(text: str, voice: str, dest: Path, retries: int = None, timeout: int = None):
    if retries is None:
        retries = EDGE_TTS_RETRIES
    if timeout is None:
        timeout = EDGE_TTS_TIMEOUT
    last = None
    for attempt in range(retries + 1):
        try:
            c = edge_tts.Communicate(text, voice)
            await asyncio.wait_for(c.save(str(dest)), timeout=timeout)
            if dest.exists() and dest.stat().st_size >= VOICE_MIN_FILE_SIZE:
                return
            raise RuntimeError("Empty output")
        except asyncio.TimeoutError:
            last = TimeoutError(f"Timeout after {timeout}s (attempt {attempt + 1})")
            console.warning(f"Edge TTS timeout, retry {attempt + 1}")
        except Exception as e:
            last = e
            console.warning(f"Edge TTS error (attempt {attempt + 1}): {e}")
        dest.unlink(missing_ok=True)
        if attempt < retries:
            await asyncio.sleep(EDGE_TTS_RETRY_DELAY)
    raise last or RuntimeError("Edge TTS failed")


# ── Qwen3-TTS ───────────────────────────────────────────────────────────

_QWEN_CACHE: dict = {}


def _qwen_model(device: str):
    """One model per device: loading it per segment multiplies load time by N."""
    if device not in _QWEN_CACHE:
        try:
            from chatterbox import Chatterbox
        except ImportError:
            raise ImportError("Install chatterbox-tts for Qwen3-TTS")
        _QWEN_CACHE[device] = Chatterbox.from_pretrained(
            QWEN_MODEL_NAME,
            device_map=device,
            dtype=QWEN_COMPUTE_DTYPE,
        )
    return _QWEN_CACHE[device]


async def speak_qwen(
    text: str,
    dest: Path,
    voice_sample: Optional[Path] = None,
    ref_text: Optional[str] = None,
    language: str = "en",
    device: str = None,
    retries: int = None,
):
    if device is None:
        device = QWEN_DEFAULT_DEVICE
    if retries is None:
        retries = EDGE_TTS_RETRIES

    for attempt in range(retries + 1):
        try:
            model = _qwen_model(device)
            audio = model.synthesize(
                text,
                voice_sample=str(voice_sample) if voice_sample else None,
                ref_text=ref_text,
                language=language,
            )
            sf.write(str(dest), audio, SR_TTS)
            if dest.exists() and dest.stat().st_size >= VOICE_MIN_FILE_SIZE:
                return
            raise RuntimeError("Empty output")
        except ImportError:
            raise
        except Exception as e:
            log.warning("Qwen attempt %d: %s", attempt + 1, e)
            _QWEN_CACHE.pop(device, None)
            dest.unlink(missing_ok=True)
            if attempt < retries:
                await asyncio.sleep(EDGE_TTS_RETRY_DELAY)
    raise RuntimeError("Qwen3-TTS failed")


# ── Voice theme system ──────────────────────────────────────────────────


def resolve_persona(
    name: str,
    lang_code: str,
    device: str = None,
) -> tuple[str, str]:
    """Return (voice_sample_path, ref_text) for a persona."""
    if device is None:
        device = QWEN_DEFAULT_DEVICE
    if name not in VOICE_PERSONAS:
        raise ValueError(f"Unknown persona: {name}. Options: {list(VOICE_PERSONAS)}")
    gender, eng_instruct = VOICE_PERSONAS[name]
    lang = LANG_ALIAS.get(lang_code, "English")
    ref = THEME_PROMPTS.get(lang, THEME_PROMPTS["English"])

    cache = Path(tempfile.gettempdir()) / "yad-themes" / name
    cache.mkdir(parents=True, exist_ok=True)
    wav = cache / f"{lang_code}.wav"
    if wav.exists() and wav.stat().st_size > 0:
        return str(wav), ref

    log.info("Generating persona %s for %s", name, lang)
    model = _qwen_model(device)
    audio = model.design_voice(text=ref, language=lang, instruct=eng_instruct)
    sf.write(wav, audio, SR_TTS)
    return str(wav), ref


def auto_clone_voice(audio_path: Path, srt_entries: list[dict], dest_dir: Path) -> Optional[Path]:
    """Extract best 20-60s speech segment from source for voice cloning."""
    if not srt_entries:
        return None
    total = srt_entries[-1]["end"] - srt_entries[0]["start"]
    if total < QWEN_CLONE_MIN_SPAN:
        return None

    best_i = best_j = best_w = 0
    for i in range(len(srt_entries)):
        wc = 0
        for j in range(i, len(srt_entries)):
            span = srt_entries[j]["end"] - srt_entries[i]["start"]
            wc += len(srt_entries[j]["text"].split())
            if span > QWEN_CLONE_MAX_DURATION:
                break
            if span >= QWEN_CLONE_MIN_DURATION and wc > best_w:
                best_w, best_i, best_j = wc, i, j

    if best_w < QWEN_CLONE_MIN_WORDS:
        return None

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "clone.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path),
         "-ss", str(srt_entries[best_i]["start"]),
         "-to", str(srt_entries[best_j]["end"]),
         "-ar", str(SR_TTS), "-ac", "1", "-c:a", "pcm_s16le", str(dest)],
        check=True, capture_output=True,
    )
    console.step(f"Auto-cloned {best_w} words ({best_i}-{best_j})")
    return dest

# youtube-auto-dub

[![CI](https://github.com/heyncth/youtube-auto-dub/actions/workflows/ci.yml/badge.svg)](https://github.com/heyncth/youtube-auto-dub/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Translate, subtitle, and dub any YouTube video automatically. Takes a URL and a target language — handles download, transcription, translation, TTS, and final render locally.

---

## How it works

```
YouTube URL → Download → Transcribe (Whisper) → Chunk → Translate (Google) → TTS → Mix → Render
```

1. **Download** — video + audio via `yt-dlp`
2. **Transcribe** — Whisper ASR with VAD and temperature fallback
3. **Chunk** — groups into natural speech segments (≤10s, ≤84 chars, split at silences)
4. **Translate** — Google Translate (RPC → scrape fallback), 25 segments per request
5. **TTS** — Edge TTS or Qwen3-TTS with voice cloning & personas
6. **Mix** — tempo-aligned TTS overlaid with optional background music
7. **Render** — burns subtitles and/or replaces audio via FFmpeg

---

## Getting started

### Prerequisites

- Python 3.10+
- FFmpeg in PATH
- (Optional) CUDA GPU for faster Whisper

```bash
# macOS
brew install ffmpeg
# Ubuntu/Debian
sudo apt install ffmpeg
# Windows — https://ffmpeg.org/download.html
```

### Install

```bash
git clone https://github.com/heyncth/youtube-auto-dub.git
cd youtube-auto-dub
pip install .
```

For GPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Usage

```bash
# Dub + subtitles in Vietnamese (default)
youtube-auto-dub "https://youtube.com/watch?v=VIDEO_ID"

# Or via python -m
python -m youtube_auto_dub "https://youtube.com/watch?v=VIDEO_ID"

# Just subtitles, Spanish
youtube-auto-dub "https://youtube.com/watch?v=VIDEO_ID" -m sub -l es

# Dubbing only, French, female voice
youtube-auto-dub "https://youtube.com/watch?v=VIDEO_ID" -m dub -l fr -g female

# Different languages for subs and dub
youtube-auto-dub "https://youtube.com/watch?v=VIDEO_ID" -m both -s en -d vi

# Age-restricted — pull cookies from browser
youtube-auto-dub "https://youtube.com/watch?v=VIDEO_ID" -b chrome
```

### All options

| Flag | Short | Description |
|------|-------|-------------|
| `url` | | YouTube URL (required) |
| `--mode` | `-m` | `sub`, `dub`, or `both` (default: `both`) |
| `--lang` | `-l` | Target language for both sub and dub |
| `--sub-lang` | `-s` | Override subtitle language |
| `--dub-lang` | `-d` | Override dubbing language |
| `--gender` | `-g` | `male` or `female` voice (default: `female`) |
| `--model` | | Whisper model: `tiny`, `base`, `small`, `medium`, `large` (default: `base`) |
| `--browser` | `-b` | Cookie source: `chrome`, `edge`, `firefox` |
| `--tts-engine` | `-e` | `edge` or `qwen` (default: `edge`) |
| `--voice` | | Qwen3-TTS persona: `narrator-m`, `young-f`, etc. |
| `--voice-clone` | | Auto-clone voice from source audio |
| `--no-tempo` | | Disable tempo alignment |
| `--no-vad` | | Disable voice-activity detection |
| `--bg-music` | | Mix original background audio into dub |
| `--output-dir` | `-o` | Output directory (default: `./output`) |
| `--version` | | Show version |

---

## Configuration

All defaults are centralized in `youtube_auto_dub/models.py` and can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `YAD_CACHE_DIR` | `./.cache` | Download cache + pipeline caches |
| `YAD_TEMP_DIR` | `./temp` | Intermediates (cleaned by pattern each run) |
| `YAD_OUTPUT_DIR` | `./output` | Final videos |
| `YAD_SAMPLE_RATE` | `24000` | TTS / export sample rate |
| `YAD_AMBIENT_GAIN` | `0.15` | Background music volume |

Voices are mapped in `youtube_auto_dub/language_map.json`. Edit to add languages or change voices:

```json
{
  "es": {
    "name": "Spanish",
    "voices": {
      "female": ["es-ES-ElviraNeural"],
      "male": ["es-ES-JorgeNeural"]
    }
  }
}
```

Common codes: `es` `fr` `de` `it` `pt` `ja` `ko` `zh` `ar` `hi` `ru` `vi` `th`

---

## Caching and reruns

Everything lives in `YAD_CACHE_DIR/<video_id>/`:

| File | Contents |
|------|----------|
| `<video_id>.mp4`, `<video_id>.wav` | Source media (re-download skipped when present) |
| `segments_<model>.json` | Transcription, keyed by Whisper model |
| `trans_<model>_<src>_<tgt>.json` | Translation, keyed by model and language pair |

A rerun with the same flags skips download, transcription, and translation — it goes straight to TTS and render. Change the Whisper model or target language and only that step re-runs. Delete the video's folder to force a full run.

Files that are re-created each run (`tts_*.wav`, `dub_*.wav`, `subtitles.srt`, ...) are cleaned from `YAD_TEMP_DIR` at startup by pattern; other files in that directory are left alone.

Final videos land in `output/Output_<mode>_<langs>_<video_id>.mp4`.

---

## Errors

- Any failure exits with code `1`, so scripts and cron jobs can detect it.
- Translation failures abort with the underlying RPC/scrape error. The tool never renders a video with untranslated text labeled as success.

---

## Project structure

```
youtube_auto_dub/
├── __init__.py       # Package version
├── __main__.py       # python -m support
├── cli.py            # argparse with global-standard flags
├── core.py           # Pipeline orchestrator (7-step run)
├── models.py         # Dataclasses + all centralized constants
├── audio.py          # Chunking, mixing, tempo align, loudness, render
├── speech.py         # Whisper ASR with VAD & metadata prompt
├── subs.py           # SRT parsing, resplit, refinement
├── voice.py          # Edge TTS + Qwen3-TTS synthesis
├── googlev4.py       # Google Translate (RPC + scrape)
├── youtube.py        # yt-dlp wrapper with metadata
├── ui.py             # Rich console helpers
└── language_map.json # Language → voice mappings
tests/
├── test_models.py    # Dataclass tests
├── test_googlev4.py  # Translation logic tests (no network)
├── test_audio.py     # Grouping + tempo budget tests
└── test_voice.py     # Voice lookup tests
.github/workflows/
└── ci.yml            # Ruff lint + pytest
```

---

## Known issues

**FFmpeg not found** — add FFmpeg to PATH.

**CUDA OOM** — use `--model tiny` or `--model base`.

**YouTube auth errors** — close browser fully before `--browser chrome`.

---

## Contributing

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check .
```

Issues and PRs welcome. Include both male and female voices when adding a language to `language_map.json`.

---

## License

MIT. See [LICENSE](LICENSE).

*Built by Nguyen Cong Thuan Huy ([@mangodxd](https://github.com/mangodxd))*

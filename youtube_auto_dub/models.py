"""Data models, path constants, and centralized configuration."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Environment-backed paths ──────────────────────────────────────────────

_BASE = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(os.environ.get("YAD_CACHE_DIR", _BASE / ".cache"))
OUTPUT_DIR = Path(os.environ.get("YAD_OUTPUT_DIR", _BASE / "output"))
TEMP_DIR = Path(os.environ.get("YAD_TEMP_DIR", _BASE / "temp"))
LANG_MAP_PATH = _BASE / "youtube_auto_dub" / "language_map.json"

for d in [CACHE_DIR, OUTPUT_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Sample rates ──────────────────────────────────────────────────────────

SR_TTS = int(os.environ.get("YAD_SAMPLE_RATE", "24000"))
SR_WHISPER = 16000


# ── Whisper defaults ──────────────────────────────────────────────────────

WHISPER_DEFAULT_MODEL = "base"
WHISPER_BEAM = 5
WHISPER_BATCH = 16
WHISPER_TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
WHISPER_COMPRESSION_RATIO_THRESHOLD = 2.4
WHISPER_LOG_PROB_THRESHOLD = -1.0
WHISPER_NO_SPEECH_THRESHOLD = 0.6


def pick_whisper_compute_type(device: str) -> str:
    if device == "cuda":
        return "float16"
    return "int8"


# ── VAD defaults ──────────────────────────────────────────────────────────

VAD_THRESHOLD = 0.35
VAD_MIN_SILENCE_MS = 500
VAD_SPEECH_PAD_MS = 200
VAD_GUARD_SECONDS = 1.0


# ── Segment grouping / subtitle defaults ──────────────────────────────────

SEGMENT_GAP_THRESHOLD = 0.8
SEGMENT_MAX_DURATION = 10.0
SUBTITLE_MAX_CHARS = 84
SUBTITLE_MAX_DUR = 5.0


# ── Audio processing defaults ─────────────────────────────────────────────

AUDIO_CROSSFADE_IN_MS = 15
AUDIO_CROSSFADE_OUT_MS = 50
AUDIO_SILENCE_FLOOR_DB = -40.0
AUDIO_SILENCE_FRAME_MS = 20
AUDIO_TRIM_BACKOFF = 0.2
AUDIO_DUB_GAIN_DB = -12
AUDIO_DEFAULT_AMBIENT_GAIN = float(os.environ.get("YAD_AMBIENT_GAIN", "0.15"))
AUDIO_FINAL_FADE_MS = 300
AUDIO_FINAL_PAD_MS = 100
AUDIO_LOUDNESS_CEIL = -30.0
AUDIO_LOUDNESS_TARGET = "TP=-1.5:LRA=11"
AUDIO_HPSS_KERNEL = 31
AUDIO_HPSS_MARGIN = 4.0


# ── Tempo stretch defaults ────────────────────────────────────────────────

TEMPO_MAX_SPEED = 1.5
TEMPO_GAP_MS = 50
TEMPO_TAIL_SECONDS = 2.0
TEMPO_OVERBUDGET_RATIO = 1.05
TEMPO_UNDERBUDGET_RATIO = 0.95
TEMPO_SLOWDOWN_DIVISOR = 0.92
TEMPO_SLOWDOWN_FLOOR = 0.82


# ── Voice / TTS defaults ──────────────────────────────────────────────────

VOICE_MIN_FILE_SIZE = 256
DEFAULT_TTS_ENGINE = "edge"
DEFAULT_GENDER = "female"
EDGE_TTS_RETRIES = 2
EDGE_TTS_TIMEOUT = 60
EDGE_TTS_RETRY_DELAY = 2
TTS_CONCURRENCY = 4
QWEN_MODEL_NAME = "chatterbox-tts/qwen3-tts"
QWEN_COMPUTE_DTYPE = "bfloat16"
QWEN_DEFAULT_DEVICE = "cuda:0"
QWEN_CLONE_MIN_DURATION = 20.0
QWEN_CLONE_MAX_DURATION = 60.0
QWEN_CLONE_MIN_SPAN = 20.0
QWEN_CLONE_MIN_WORDS = 20


# ── Translation defaults ──────────────────────────────────────────────────

TRANSLATE_TIMEOUT = 30
TRANSLATE_API_URL = "https://translate.google.com/_/TranslateWebserverUi/data/batchexecute"
TRANSLATE_SCRAPE_URL = "https://translate.google.com/m"
TRANSLATE_TOKEN_URL = "https://translate.google.com/"
TRANSLATE_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
)


# ── YouTube / yt-dlp defaults ─────────────────────────────────────────────

YT_FORMAT = (
    "bestvideo[ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/best[ext=mp4]/best"
)
YT_MIN_FILE_SIZE = 1024 * 100
YT_AUDIO_EXPORT_SR = int(os.environ.get("YAD_SAMPLE_RATE", "24000"))


# ── FFmpeg defaults ───────────────────────────────────────────────────────

FFMPEG_VIDEO_CODEC = "libx264"
FFMPEG_AUDIO_CODEC = "aac"
FFMPEG_MUX_VIDEO_CODEC = "copy"


# ── Voice persona / theme data (was hardcoded in voice.py) ────────────────

THEME_PROMPTS: Dict[str, str] = {
    "English": (
        "Welcome everyone, today we are going to explore something truly "
        "fascinating. Throughout history, philosophers and scientists have "
        "questioned the very fabric of existence, pushing boundaries through "
        "observation, experimentation, and sheer curiosity."
    ),
    "Chinese": (
        "大家好，今天我们将一起探索一些非常精彩的内容。"
        "从古老的丝绸之路到现代的量子物理学，每一个发现都充满了惊喜和挑战。"
    ),
    "Japanese": (
        "皆さんこんにちは、今日はとても魅力的なテーマを一緒に探っていきましょう。"
        "歴史を通じて、哲学者や科学者たちは存在の本質を問い続けてきました。"
    ),
    "Korean": (
        "여러분 안녕하세요, 오늘은 정말 흥미로운 주제를 함께 살펴보겠습니다. "
        "역사를 통해 철학자와 과학자들은 존재의 본질에 대해 끊임없이 질문해 왔습니다."
    ),
}

VOICE_PERSONAS: Dict[str, Tuple[str, str]] = {
    "narrator-m": (
        "male",
        "baritone, warm and steady, professional narrator with clear enunciation",
    ),
    "narrator-f": (
        "female",
        "mezzo-soprano, smooth and engaging, professional narrator",
    ),
    "young-m": (
        "male",
        "tenor, energetic and casual, natural youthful conversational tone",
    ),
    "young-f": (
        "female",
        "soprano, bright and cheerful, friendly youthful style",
    ),
    "deep-m": (
        "male",
        "bass-baritone, deep and commanding, authoritative presence",
    ),
    "deep-f": (
        "female",
        "contralto, deep and resonant, commanding presence",
    ),
}

LANG_ALIAS: Dict[str, str] = {
    "en": "English",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
    "it": "Italian",
}


# ── Model classes ─────────────────────────────────────────────────────────


@dataclass
class SubtitleSegment:
    start: float
    end: float
    source_text: str
    translated_text_sub: Optional[str] = None
    translated_text_dub: Optional[str] = None
    tts_audio_path: Optional[Path] = None
    index: int = 0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class VideoMetadata:
    title: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    upload_date: Optional[str] = None
    duration: float = 0.0
    channel: str = ""
    view_count: int = 0
    like_count: int = 0


@dataclass
class ProjectContext:
    video_id: str
    video_path: Path
    audio_path: Path
    segments: List[SubtitleSegment] = field(default_factory=list)
    subtitle_path: Optional[Path] = None
    dub_audio_path: Optional[Path] = None
    metadata: Optional[VideoMetadata] = None

    @property
    def project_dir(self) -> Path:
        p = CACHE_DIR / self.video_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save_cache(self, key: str, data: Any) -> None:
        path = self.project_dir / f"{key}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_cache(self, key: str) -> Any:
        path = self.project_dir / f"{key}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

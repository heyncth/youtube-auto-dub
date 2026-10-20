"""yt-dlp wrapper for downloading YouTube videos with metadata."""

import subprocess
from typing import Optional

import yt_dlp

from youtube_auto_dub.models import CACHE_DIR, YT_AUDIO_EXPORT_SR, YT_FORMAT, YT_MIN_FILE_SIZE, ProjectContext, VideoMetadata
from youtube_auto_dub.ui import console


def _extract_metadata(info: dict) -> VideoMetadata:
    tags = info.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    return VideoMetadata(
        title=info.get("title", ""),
        description=info.get("description", ""),
        tags=tags,
        upload_date=info.get("upload_date"),
        duration=info.get("duration", 0.0),
        channel=info.get("channel") or info.get("uploader", ""),
        view_count=info.get("view_count", 0),
        like_count=info.get("like_count", 0),
    )


def download_project(url: str, browser: Optional[str] = None) -> ProjectContext:
    opts = {
        "format": YT_FORMAT,
        "outtmpl": str(CACHE_DIR / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }
    if browser:
        opts["cookiesfrombrowser"] = (browser.lower(),)

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info["id"]
        video_path = CACHE_DIR / f"{video_id}.mp4"
        audio_path = CACHE_DIR / f"{video_id}.wav"
        metadata = _extract_metadata(info)

    if not audio_path.exists() or audio_path.stat().st_size < YT_MIN_FILE_SIZE:
        console.step("Extracting audio format...")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(YT_AUDIO_EXPORT_SR),
                "-ac",
                "1",
                str(audio_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    console.step(f"Downloaded source ({video_id})")
    return ProjectContext(
        video_id=video_id,
        video_path=video_path,
        audio_path=audio_path,
        metadata=metadata,
    )

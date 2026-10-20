"""Tests for data models and path constants."""

from pathlib import Path

from youtube_auto_dub.models import (
    CACHE_DIR,
    LANG_MAP_PATH,
    OUTPUT_DIR,
    TEMP_DIR,
    ProjectContext,
    SubtitleSegment,
)


class TestSubtitleSegment:
    def test_basic_creation(self):
        seg = SubtitleSegment(start=1.0, end=5.0, source_text="hello world")
        assert seg.start == 1.0
        assert seg.end == 5.0
        assert seg.source_text == "hello world"
        assert seg.duration == 4.0
        assert seg.translated_text_sub is None
        assert seg.translated_text_dub is None
        assert seg.tts_audio_path is None

    def test_duration_property(self):
        seg = SubtitleSegment(start=10.5, end=15.25, source_text="test")
        assert seg.duration == 4.75

    def test_optional_fields(self):
        seg = SubtitleSegment(
            start=0.0,
            end=2.0,
            source_text="hi",
            translated_text_sub="hola",
            translated_text_dub="hola",
            tts_audio_path=Path("/tmp/tts.mp3"),
        )
        assert seg.translated_text_sub == "hola"
        assert seg.translated_text_dub == "hola"
        assert seg.tts_audio_path == Path("/tmp/tts.mp3")


class TestProjectContext:
    def test_basic_creation(self):
        ctx = ProjectContext(
            video_id="abc123",
            video_path=Path("/tmp/vid.mp4"),
            audio_path=Path("/tmp/aud.wav"),
        )
        assert ctx.video_id == "abc123"
        assert ctx.video_path == Path("/tmp/vid.mp4")
        assert ctx.audio_path == Path("/tmp/aud.wav")
        assert ctx.segments == []
        assert ctx.subtitle_path is None
        assert ctx.dub_audio_path is None


class TestPathConstants:
    def test_directories_exist(self):
        assert CACHE_DIR.exists()
        assert OUTPUT_DIR.exists()
        assert TEMP_DIR.exists()

    def test_lang_map_path(self):
        assert LANG_MAP_PATH.exists()
        assert LANG_MAP_PATH.name == "language_map.json"

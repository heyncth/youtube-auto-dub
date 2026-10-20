"""Tests for grouping rules and tempo alignment budget math."""

from pathlib import Path

import numpy as np

from youtube_auto_dub.audio import build_align_info, group_segments
from youtube_auto_dub.models import SUBTITLE_MAX_CHARS, SubtitleSegment


class TestGroupSegments:
    def test_merge_respects_char_limit(self):
        raw = [
            {"start": 0.0, "end": 4.0, "text": "A" * 80},
            {"start": 4.1, "end": 8.0, "text": "B" * 80},
        ]
        segs = group_segments(raw)
        assert len(segs) == 2
        assert all(len(s.source_text) <= SUBTITLE_MAX_CHARS for s in segs)

    def test_merges_short_adjacent(self):
        raw = [
            {"start": 0.0, "end": 2.0, "text": "hello"},
            {"start": 2.3, "end": 4.0, "text": "world"},
        ]
        segs = group_segments(raw)
        assert len(segs) == 1
        assert segs[0].source_text == "hello world"

    def test_empty(self):
        assert group_segments([]) == []


class TestAlignBudget:
    def test_target_dur_is_source_slot_not_tts_length(self):
        seg = SubtitleSegment(start=0.0, end=2.0, source_text="x",
                              tts_audio_path=Path("a.wav"))
        info = build_align_info([seg])
        # If this equals the TTS length, the speed-up branch can never run
        assert info[0]["target_dur"] == seg.duration == 2.0

    def test_overlong_clip_gets_sped_up(self, monkeypatch, tmp_path):
        import youtube_auto_dub.audio as audio
        stretched = []

        monkeypatch.setattr(
            audio, "_load_raw",
            lambda path, s: np.zeros(int(2.0 * s), dtype=np.float32),
        )

        def fake_stretch(path, factor, dst, s):
            stretched.append(factor)
            return np.zeros(int(2.0 * s), dtype=np.float32)

        monkeypatch.setattr(audio, "_stretch", fake_stretch)
        monkeypatch.setattr(audio.sf, "write", lambda *a, **k: None)

        info = [
            {"start": 0.0, "target_dur": 1.0, "wav_path": Path("a.wav")},
            {"start": 1.05, "target_dur": 1.0, "wav_path": Path("b.wav")},
        ]
        audio.align_segments(info, source_duration=3.0, output=tmp_path / "out.wav")
        assert stretched, "2s clip in a 1s slot must be stretched"
        assert stretched[0] > 1.0

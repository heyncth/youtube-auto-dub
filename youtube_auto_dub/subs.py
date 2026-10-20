"""Subtitle parsing, splitting, and SRT generation — single unified layer."""

import re
from typing import List

from youtube_auto_dub.models import SUBTITLE_MAX_CHARS, SUBTITLE_MAX_DUR


def _ts_to_sec(ts: str) -> float:
    h, m, r = ts.split(":")
    s, ms = r.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_srt(text: str) -> List[dict]:
    entries = []
    for block in re.split(r"\n\n+", text.strip()):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        m = re.match(r"([\d:,]+)\s*-->\s*([\d:,]+)", lines[1])
        if not m:
            continue
        entries.append({
            "id": lines[0].strip(),
            "start": _ts_to_sec(m.group(1)),
            "end": _ts_to_sec(m.group(2)),
            "text": " ".join(lines[2:]),
        })
    return entries


def read_srt(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return parse_srt(f.read())


# ── Smart segment splitting ────────────────────────────────────────────

_DEFAULT_MAX_CHARS = SUBTITLE_MAX_CHARS
_DEFAULT_MAX_DUR = SUBTITLE_MAX_DUR


def _split_words(words: List[dict], max_chars: int, max_dur: float) -> List[dict]:
    chunks = []
    buf, btext, bstart = [], "", None
    best_idx, best_gap = None, -1.0

    def emit(up_to=None):
        nonlocal buf, btext, bstart, best_idx, best_gap
        if up_to is None:
            up_to = len(buf)
        if up_to > 0:
            chunk_text = " ".join(w.get("word", "") for w in buf[:up_to]).strip()
            chunks.append({
                "start": bstart,
                "end": buf[up_to - 1].get("end", bstart),
                "text": chunk_text,
            })
        buf = buf[up_to:]
        btext = " ".join(w.get("word", "") for w in buf).strip() if buf else ""
        bstart = next((w.get("start") for w in buf if w.get("start") is not None), None)
        best_idx, best_gap = None, -1.0

    for w in words:
        wt = w.get("word", "")
        ws, we = w.get("start"), w.get("end")
        if ws is None or we is None:
            buf.append(w)
            btext = (btext + " " + wt).strip() if btext else wt
            continue
        candidate = (btext + " " + wt).strip() if btext else wt
        cand_dur = we - (bstart if bstart is not None else ws)
        if btext and (len(candidate) > max_chars or cand_dur > max_dur):
            if best_idx is not None and best_idx > 0:
                emit(best_idx + 1)
                buf.append(w)
                btext = (btext + " " + wt).strip() if btext else wt
                if bstart is None:
                    bstart = ws
            else:
                emit()
                buf, btext, bstart = [w], wt, ws
        else:
            if buf:
                prev_end = buf[-1].get("end")
                if prev_end is not None and ws is not None:
                    gap = ws - prev_end
                    if gap > best_gap:
                        best_gap, best_idx = gap, len(buf) - 1
            buf.append(w)
            btext, bstart = candidate, bstart if bstart is not None else ws

    if btext.strip():
        emit()
    return chunks


def _split_ratio(text: str, start: float, end: float, max_chars: int) -> List[dict]:
    words = text.split()
    chunks = []
    buf = ""
    for w in words:
        c = (buf + " " + w).strip() if buf else w
        if buf and len(c) > max_chars:
            chunks.append(buf.strip())
            buf = w
        else:
            buf = c
    if buf.strip():
        chunks.append(buf.strip())

    total = len(text) or 1
    dur = end - start
    result, t = [], start
    for ch in chunks:
        ce = min(t + dur * len(ch) / total, end)
        result.append({"start": t, "end": ce, "text": ch})
        t = ce
    if result:
        result[-1]["end"] = end
    return result


def refine_segments(segments: List[dict]) -> List[dict]:
    """Split long segments into subtitle-friendly chunks."""
    out = []
    for seg in segments:
        text = seg.get("text", "").strip()
        s, e = seg["start"], seg["end"]
        if len(text) <= _DEFAULT_MAX_CHARS and (e - s) <= _DEFAULT_MAX_DUR:
            out.append({"start": s, "end": e, "text": text})
            continue
        words = seg.get("words", [])
        if words and all("start" in w and "end" in w for w in words):
            out.extend(_split_words(words, _DEFAULT_MAX_CHARS, _DEFAULT_MAX_DUR))
        else:
            out.extend(_split_ratio(text, s, e, _DEFAULT_MAX_CHARS))
    return out

"""Core pipeline — orchestrates download → transcribe → translate → speak → assemble → render."""

import asyncio
import logging
from pathlib import Path

import torch
from rich.table import Table

from youtube_auto_dub.audio import (
    align_segments,
    build_align_info,
    finalize_audio,
    group_segments,
    overlay_dub,
    render_video,
    write_srt,
)
from youtube_auto_dub.googlev4 import GoogleTranslator
from youtube_auto_dub.models import (
    AUDIO_DEFAULT_AMBIENT_GAIN,
    DEFAULT_TTS_ENGINE,
    TEMP_DIR,
    TTS_CONCURRENCY,
    WHISPER_DEFAULT_MODEL,
    SubtitleSegment,
)
from youtube_auto_dub.speech import build_hint, transcribe
from youtube_auto_dub.subs import read_srt
from youtube_auto_dub.ui import console
from youtube_auto_dub.voice import (
    auto_clone_voice,
    pick_voice,
    resolve_persona,
    speak_edge,
    speak_qwen,
)
from youtube_auto_dub.youtube import download_project

log = logging.getLogger(__name__)


async def run(args) -> None:
    base_lang = args.lang or "en"
    sub_lang = args.lang_sub or base_lang
    dub_lang = args.lang_dub or base_lang
    out_root = Path(args.output_dir) if getattr(args, "output_dir", None) else Path("output")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = args.whisper_model or WHISPER_DEFAULT_MODEL

    tts_engine = getattr(args, "tts_engine", DEFAULT_TTS_ENGINE)
    use_tempo = not getattr(args, "no_tempo", False)
    keep_bg = getattr(args, "preserve_bg", False)
    do_clone = getattr(args, "auto_clone", False)
    persona = getattr(args, "voice_theme", None)

    # ── UI ────────────────────────────────────────────────────────────
    console.header("YouTube Auto Dub")
    console.header("Configuration", center=False)
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_row("URL", f"[#e5e7eb]{args.url}[/#e5e7eb]")
    t.add_row("Mode", f"[#e5e7eb]{args.mode.upper()}[/#e5e7eb]")
    if args.mode in ("sub", "both"):
        t.add_row("Subs", f"[#e5e7eb]{sub_lang.upper()}[/#e5e7eb]")
    if args.mode in ("dub", "both"):
        t.add_row("Dub", f"[#e5e7eb]{dub_lang.upper()}[/#e5e7eb]")
        t.add_row("Gender", f"[#e5e7eb]{args.gender.title()}[/#e5e7eb]")
    t.add_row("ASR", f"[#e5e7eb]{model_name.upper()} ({device.upper()})[/#e5e7eb]")
    t.add_row("TTS", f"[#e5e7eb]{tts_engine.upper()}[/#e5e7eb]")
    if persona:
        t.add_row("Persona", f"[#e5e7eb]{persona}[/#e5e7eb]")
    if do_clone:
        t.add_row("Clone", "[#e5e7eb]ON[/#e5e7eb]")
    if keep_bg:
        t.add_row("Ambient", "[#e5e7eb]ON[/#e5e7eb]")
    console.print(t)
    console.print()

    with console.status("Processing..."):
        # ── 1. Download ──────────────────────────────────────────────
        console.info("Downloading media")
        project = download_project(args.url, args.browser)

        # ── 2. Transcribe ────────────────────────────────────────────
        console.info(f"Transcribing ({model_name})")

        seg_cache_key = f"segments_{model_name}"
        cached = project.load_cache(seg_cache_key)
        if cached:
            console.step("Using cached transcription")
            project.segments = [
                SubtitleSegment(start=s["start"], end=s["end"],
                                source_text=s["source_text"], index=i)
                for i, s in enumerate(cached)
            ]
            lang_detected = cached[0].get("lang")
        else:
            hint = build_hint(project.metadata)
            if hint:
                console.step("Prompting with video metadata")

            raw, lang_detected = transcribe(
                project.audio_path,
                model_name=model_name,
                device=device,
                hint=hint,
                use_vad=not getattr(args, "no_vad", False),
            )
            console.success(f"Detected: {lang_detected}")

            console.info("Grouping segments")
            project.segments = group_segments(raw)

            cache_data = [
                {"index": i, "start": s.start, "end": s.end,
                 "source_text": s.source_text, "lang": lang_detected}
                for i, s in enumerate(project.segments)
            ]
            project.save_cache(seg_cache_key, cache_data)

        texts = [seg.source_text for seg in project.segments]

        # ── 3. Translate ─────────────────────────────────────────────
        console.info("Translating")

        async def translate_cached(target: str) -> list:
            """Reuse cached output: reruns must not re-translate the video."""
            key = f"trans_{model_name}_{lang_detected}_{target}"
            hit = project.load_cache(key)
            if isinstance(hit, list) and len(hit) == len(texts):
                console.step(f"Using cached translation -> {target.upper()}")
                return hit
            console.step(f"Translating {len(texts)} segs -> {target.upper()}")
            xl = GoogleTranslator()
            try:
                out = await xl.translate_batch(
                    texts, source=lang_detected or "auto", target=target
                )
            finally:
                await xl.close()
            project.save_cache(key, out)
            return out

        sub_out = dub_out = texts
        if args.mode in ("sub", "both"):
            if lang_detected and lang_detected == sub_lang:
                console.step(f"Source == target ({sub_lang}), skipping")
                sub_out = texts
            else:
                sub_out = await translate_cached(sub_lang)
            for i, seg in enumerate(project.segments):
                seg.translated_text_sub = sub_out[i].strip() or seg.source_text

        if args.mode in ("dub", "both"):
            if args.mode == "both" and dub_lang == sub_lang:
                console.step("Reusing subtitle translation for dubbing")
                dub_out = sub_out
            elif lang_detected and lang_detected == dub_lang:
                console.step(f"Source == target ({dub_lang}), skipping")
                dub_out = texts
            else:
                dub_out = await translate_cached(dub_lang)
            for i, seg in enumerate(project.segments):
                seg.translated_text_dub = dub_out[i].strip() or seg.source_text

        console.success("Translation done")

        # ── 4. Speech synthesis ───────────────────────────────────────
        if args.mode in ("dub", "both"):
            console.info(f"Synthesizing ({tts_engine})")

            # Resolve voice source
            sample = ref_txt = None

            if tts_engine == "qwen" and do_clone:
                srt_path = TEMP_DIR / "ref.srt"
                write_srt(project.segments, srt_path)
                entries = read_srt(str(srt_path))
                sample = auto_clone_voice(project.audio_path, entries,
                                          project.project_dir / "clone")

            if tts_engine == "qwen" and persona and not sample:
                sample, ref_txt = resolve_persona(
                    persona, dub_lang, device=f"{device}:0",
                )
                console.step(f"Persona: {persona}")

            # Generate TTS per segment
            voice = pick_voice(dub_lang, args.gender) if tts_engine != "qwen" else None
            sem = asyncio.Semaphore(TTS_CONCURRENCY)

            async def run_edge(text: str, dest: Path) -> None:
                # Bounded: hundreds of parallel requests get rate-limited
                async with sem:
                    await speak_edge(text, voice, dest)

            tasks = []
            for i, seg in enumerate(project.segments):
                seg.tts_audio_path = TEMP_DIR / f"tts_{i}.wav"
                if tts_engine == "qwen":
                    tasks.append(speak_qwen(
                        seg.translated_text_dub, seg.tts_audio_path,
                        voice_sample=Path(sample) if sample else None,
                        ref_text=ref_txt, language=dub_lang,
                        device=f"{device}:0",
                    ))
                else:
                    tasks.append(run_edge(seg.translated_text_dub, seg.tts_audio_path))

            await asyncio.gather(*tasks)

            # ── 5. Assemble & finalise ────────────────────────────────
            project.dub_audio_path = TEMP_DIR / "dub_final.wav"

            if use_tempo:
                src_dur = float(project.metadata.duration) if project.metadata else 0.0
                raw_mix = TEMP_DIR / "dub_raw.wav"
                align_segments(build_align_info(project.segments), src_dur, raw_mix)
                finalize_audio(
                    raw_mix, project.audio_path,
                    project.dub_audio_path,
                    match_loudness=True,
                    mix_ambient=keep_bg,
                    ambient_gain=AUDIO_DEFAULT_AMBIENT_GAIN if keep_bg else 0.0,
                )
            else:
                overlay_dub(project.audio_path, project.segments,
                            project.dub_audio_path)

            console.success("Dubbing complete")

        # ── 6. Subtitles ──────────────────────────────────────────────
        if args.mode in ("sub", "both"):
            console.info("Writing subtitles")
            project.subtitle_path = TEMP_DIR / "subtitles.srt"
            write_srt(project.segments, project.subtitle_path)
            console.success("Subtitles saved")

        # ── 7. Render ─────────────────────────────────────────────────
        console.info("Rendering video")
        info = f"L-{base_lang}"
        if args.lang_sub:
            info += f"_S-{sub_lang}"
        if args.lang_dub:
            info += f"_D-{dub_lang}"
        out = out_root / f"Output_{args.mode}_{info}_{project.video_id}.mp4"

        render_video(
            video_path=project.video_path,
            subtitle_path=project.subtitle_path if args.mode in ("sub", "both") else None,
            dub_audio_path=project.dub_audio_path if args.mode in ("dub", "both") else None,
            output_path=out,
        )
        console.success("Video rendered")

    console.print()
    console.print(f"[bold #38bdf8]Output: {out.resolve()}[/bold #38bdf8]")

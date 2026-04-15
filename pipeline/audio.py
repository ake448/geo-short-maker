"""
audio.py — Voiceover generation, Whisper alignment, beat duration sync.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import VOICES_DIR


def generate_voiceover(script: Dict, run_dir: Path, voice_path: Optional[Path] = None) -> Optional[Path]:
    """Generate voiceover audio from beat narrations using Qwen Voice Cloning.
    If voice_path is provided, clones that specific voice, otherwise uses the default.
    """
    import generate_attenborough_audio as tts_module

    out_path = run_dir / "voiceover.mp3"
    if out_path.exists():
        print("  [CACHED] voiceover.mp3 already exists")
        return out_path

    beats = script.get("beats", [])
    full_script = " ".join(b.get("narration", "") for b in beats).strip()
    if not full_script:
        print("  [SKIP] No narration text in script")
        return None

    word_count = len(full_script.split())
    print(f"  Script: {word_count} words, {len(full_script)} chars")
    
    # 1. Register or retrieve Voice ID
    try:
        vp = None
        if voice_path:
            # Try absolute/relative path as provided
            if voice_path.exists():
                vp = str(voice_path.resolve())
            else:
                # Fallback: check if the filename exists in the global voices/ folder
                fallback = VOICES_DIR / voice_path.name
                if fallback.exists():
                    print(f"  [INFO] Voice file not found locally, using: {fallback}")
                    vp = str(fallback.resolve())

        print(f"  Fetching cloned voice ID (source: {vp or 'default'})...")
        voice_id = tts_module.create_cloned_voice(file_path=vp)
        print(f"  [OK] Voice ID active.")
    except Exception as e:
        print(f"  [FAIL] Could not register/retrieve voice ID: {e}")
        return None

    # 2. Generate Long Form Audio
    print(f"  Generating and crossfading audio...")
    try:
        tts_module.generate_long_form_audio(
            text=full_script,
            voice_id=voice_id,
            output_path=str(out_path)
        )
        if out_path.exists():
            mb = out_path.stat().st_size / 1048576
            print(f"  [OK] Voiceover saved: {out_path.name} ({mb:.1f} MB)")
            return out_path
    except Exception as e:
        print(f"  [FAIL] Voice generation failed: {e}")

    return None


def run_whisper_alignment(audio_path: Path) -> Optional[List[Dict]]:
    """Run Whisper on voiceover to get word-level timestamps."""
    json_cache = audio_path.parent / "whisper_segments.json"
    if json_cache.exists():
        print("  [CACHED] whisper_segments.json")
        return json.loads(json_cache.read_text(encoding="utf-8"))

    try:
        import whisper
    except ImportError:
        print("  [SKIP] whisper not installed")
        return None

    print("  Loading Whisper (base.en)...")
    model = whisper.load_model("base.en")
    result = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language="en",
        verbose=False,
        beam_size=1,
        best_of=1,
        condition_on_previous_text=True,
        temperature=0.0,
    )

    segments = [
        {
            "start": float(s["start"]),
            "end":   float(s["end"]),
            "text":  s["text"].strip(),
            "words": [
                {"word": w["word"], "start": float(w["start"]), "end": float(w["end"])}
                for w in s.get("words", [])
            ],
        }
        for s in result.get("segments", [])
    ]
    total_dur = segments[-1]["end"] if segments else 0
    print(f"  Whisper: {len(segments)} segments, {total_dur:.1f}s")

    json_cache.write_text(json.dumps(segments, indent=2), encoding="utf-8")
    return segments


def _update_beat_durations_from_whisper(
    script: Dict[str, Any], whisper_segs: List[Dict]
) -> Dict[str, Any]:
    """Update beat durations using real Whisper-measured spoken timings."""
    beats = script.get("beats", [])
    if not beats or not whisper_segs:
        return script

    BUFFER_SEC = 0.5
    MIN_DUR = 3.0
    MAX_DUR = 8.5   # cap beat clip to 8.5s — matches script target, prevents runaway long beats
    LOOKAHEAD = 6        # was 4 — wider window catches longer narrations
    MIN_SCORE = 0.12     # slightly more forgiving than 0.15

    def _word_overlap(a: str, b: str) -> float:
        """Jaccard similarity on word sets."""
        wa = set(a.lower().split())
        wb = set(b.lower().split())
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)

    seg_idx = 0
    last_known_end = 0.0  # tracks the last successfully resolved audio_end

    for bi, beat in enumerate(beats):
        narration = beat.get("narration", "").lower().strip()

        # --- No narration: interpolate a short gap and move on ---
        if not narration:
            beat["audio_start"] = round(last_known_end, 3)
            beat["audio_end"] = round(last_known_end + 4.0, 3)
            beat["duration_sec"] = 4.0
            last_known_end = beat["audio_end"]
            continue

        # --- Exhausted segments: fill remaining beats from last known end ---
        if seg_idx >= len(whisper_segs):
            beat["audio_start"] = round(last_known_end, 3)
            beat["audio_end"] = round(last_known_end + float(beat.get("duration_sec", 7.0)), 3)
            last_known_end = beat["audio_end"]
            continue

        # --- Try to match within lookahead window ---
        best_end_idx = seg_idx
        best_score = 0.0
        accumulated_text = ""

        for si in range(seg_idx, min(seg_idx + LOOKAHEAD, len(whisper_segs))):
            accumulated_text += " " + whisper_segs[si].get("text", "")
            score = _word_overlap(narration, accumulated_text)
            if score > best_score:
                best_score = score
                best_end_idx = si

        audio_start = float(whisper_segs[seg_idx]["start"])
        audio_end = float(whisper_segs[best_end_idx]["end"])

        if best_score >= MIN_SCORE:
            # Good match — use Whisper timestamps
            real_dur = max(MIN_DUR, min(MAX_DUR, audio_end - audio_start + BUFFER_SEC))
            beat["audio_start"] = round(audio_start, 3)
            beat["audio_end"] = round(audio_end, 3)
            beat["duration_sec"] = round(real_dur, 2)
            beat["_whisper_match"] = round(best_score, 3)  # debug field
            last_known_end = audio_end
            seg_idx = best_end_idx + 1  # KEY: always advance past matched segments

        else:
            # Poor match — don't stall. Assign timestamps from last known position,
            # advance seg_idx by 1 so the cascade doesn't propagate.
            fallback_dur = float(beat.get("duration_sec", 7.0))
            beat["audio_start"] = round(last_known_end, 3)
            beat["audio_end"] = round(last_known_end + fallback_dur, 3)
            beat["duration_sec"] = round(fallback_dur, 2)
            beat["_whisper_match"] = round(best_score, 3)  # shows 0.0x so you can spot failures
            last_known_end = beat["audio_end"]
            seg_idx += 1  # KEY: still advance, never stall

        old_dur = beat.get("duration_sec", 0)
        print(f"    Beat {beat.get('beat_id', bi+1)}: "
              f"score={best_score:.2f} | "
              f"{old_dur:.1f}s -> {beat['duration_sec']:.1f}s "
              f"[{beat['audio_start']:.1f}s - {beat['audio_end']:.1f}s]")

    script["total_duration_sec"] = round(
        sum(float(b.get("duration_sec", 7)) for b in beats), 1
    )
    script["_overlong_audio"] = bool(script["total_duration_sec"] > 60.0)
    if script["_overlong_audio"]:
        print(f"    [WARN] Spoken runtime is over target: {script['total_duration_sec']:.1f}s")
    print(f"    Total duration updated: {script['total_duration_sec']:.1f}s")
    return script

"""
captions.py — ASS caption generation and burning.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List

from .config import ROOT, OUT_W, OUT_H, FONTS_DIR, CAPTION_FONT_NAME


def _format_ass_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def generate_ass_captions_from_whisper(whisper_segments: List[Dict]) -> str:
    """Generate two-line karaoke ASS captions with smooth word highlighting."""
    words: List[Dict[str, Any]] = []
    for seg in whisper_segments or []:
        for w in seg.get("words", []) or []:
            text = str(w.get("word", "")).strip()
            if not text:
                continue
            try:
                start = float(w.get("start", 0.0))
                end = float(w.get("end", start + 0.2))
            except Exception:
                continue
            if end <= start:
                end = start + 0.2
            words.append({"word": text, "start": start, "end": end})

    if not words:
        return ""

    ass = (
        "[Script Info]\n"
        "Title: Geography Short Captions\n"
        "ScriptType: v4.00+\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: TV.709\n"
        f"PlayResX: {OUT_W}\n"
        f"PlayResY: {OUT_H}\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        # Urban Atlas / GeoGlobeX caption style:
        #   - Pure white primary, yellow secondary (active word highlight)
        #   - Black outline 3px — hard "cutout" shadow (0px blur = crisp edge)
        #   - No background box (transparent BackColour)
        #   - Alignment 2 = bottom-center; MarginV 480 = 75% down on 1920px frame
        #   - Font size 62, Bold
        # ASS colours: &HAABBGGRR  (alpha, blue, green, red — little-endian)
        # White:  &H00FFFFFF  Yellow: &H0000D7FF  Black: &H00000000  Transparent: &H00000000
        f"Style: Default,{CAPTION_FONT_NAME},62,&H00FFFFFF,&H0000D7FF,"
        "&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,2,30,30,480,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text\n"
    )

    def _clean_word(raw: str) -> str:
        return (
            str(raw)
            .replace("\\N", " ")
            .replace("\n", " ")
            .replace("\r", " ")
            .replace("{", "")
            .replace("}", "")
            .strip()
        )

    def _chunk_words(in_words: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        chunks: List[List[Dict[str, Any]]] = []
        cur: List[Dict[str, Any]] = []
        max_words = 8
        max_chars = 40
        for item in in_words:
            wtxt = _clean_word(item["word"])
            if not wtxt:
                continue
            candidate = cur + [{"word": wtxt, "start": item["start"], "end": item["end"]}]
            char_count = len(" ".join(x["word"] for x in candidate))
            if cur and (len(candidate) > max_words or char_count > max_chars):
                chunks.append(cur)
                cur = [{"word": wtxt, "start": item["start"], "end": item["end"]}]
            else:
                cur = candidate
        if cur:
            chunks.append(cur)
        return chunks

    for chunk in _chunk_words(words):
        tokens = [_clean_word(w["word"]).upper() for w in chunk]
        if not any(tokens):
            continue
        full_text_str = " ".join(tokens)
        split_idx = len(tokens)
        
        if len(full_text_str) > 25 and len(tokens) > 1:
            best_delta = 1_000_000
            for k in range(1, len(tokens)):
                left_len = len(" ".join(tokens[:k]))
                right_len = len(" ".join(tokens[k:]))
                delta = abs(left_len - right_len)
                if delta < best_delta:
                    best_delta = delta
                    split_idx = k

        for i in range(len(chunk)):
            start = float(chunk[i]["start"])
            if i < len(chunk) - 1:
                end = float(chunk[i+1]["start"])
            else:
                end = max(start + 0.1, float(chunk[i]["end"]))
                
            start_t = _format_ass_time(start)
            end_t = _format_ass_time(end)

            out_str = ""
            for j, t in enumerate(tokens):
                prefix = ""
                if j > 0 and j != split_idx:
                    prefix = " "
                elif j == split_idx:
                    prefix = "\\N"
                
                if j < i:
                    # Already-spoken word: white, full opacity
                    out_str += prefix + t
                elif j == i:
                    # Current word: yellow highlight — CapCut "Pop" style
                    out_str += prefix + "{\\c&H0000D7FF&}" + t + "{\\c&H00FFFFFF&}"
                else:
                    # Upcoming word: grey-out (hidden until its turn)
                    out_str += prefix + "{\\alpha&HCC&}" + t + "{\\alpha&H00&}"

            anim = "{\\an2}"
            ass += f"Dialogue: 0,{start_t},{end_t},Default,,0,0,0,,{anim}{out_str}\n"

    return ass


def burn_ass_captions(video_path: Path, ass_path: Path, out_path: Path) -> bool:
    """Burn ASS subtitles with explicit fontsdir to avoid drawtext escaping issues."""
    base_dir = ROOT.parent
    try:
        ass_rel = ass_path.relative_to(base_dir).as_posix()
        fonts_rel = FONTS_DIR.relative_to(base_dir).as_posix()
    except Exception:
        ass_rel = ass_path.as_posix()
        fonts_rel = FONTS_DIR.as_posix()

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path.resolve()),
        "-vf", f"subtitles={ass_rel}:fontsdir={fonts_rel}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-an",
        str(out_path.resolve()),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=str(base_dir))
        if r.returncode != 0:
            err = (r.stderr or "").splitlines()[-6:]
            print(f"  [FAIL] ASS burn failed: {' | '.join(err)}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("  [FAIL] ASS burn timed out")
        return False

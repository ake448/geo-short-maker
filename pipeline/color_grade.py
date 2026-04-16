"""
color_grade.py — Consistent cinematic color grading for all clips.

Applies a unified look across YouTube-sourced footage and generated maps
so the final Short feels cohesive. Uses ffmpeg's eq + colorchannelmixer
for a cool-shadow / warm-highlight documentary style.
"""
from __future__ import annotations

from pathlib import Path

from .ffmpeg_utils import run_ffmpeg
from .config import FPS


# ── Grade profiles ────────────────────────────────────────────────────────────
# Each profile is an ffmpeg filter string.

GRADES = {
    # Default cinematic: slight contrast boost, desaturated shadows,
    # faint cool-blue shift in shadows, warm in highlights.
    "cinematic": (
        "eq=contrast=1.08:brightness=0.02:saturation=1.12,"
        "colorchannelmixer="
        "rr=1.02:rg=0.0:rb=0.0:"      # reds: slight boost
        "gr=0.0:gg=1.0:gb=0.02:"       # greens: tiny blue bleed
        "br=0.0:bg=0.02:bb=1.06,"      # blues: cool lift in shadows
        "curves=m='0/0 0.25/0.22 0.5/0.5 0.75/0.80 1/1'"  # gentle S-curve
    ),

    # Moodier night / darker topics
    "dark": (
        "eq=contrast=1.12:brightness=-0.02:saturation=1.05,"
        "colorchannelmixer="
        "rr=1.0:rg=0.0:rb=0.0:"
        "gr=0.0:gg=0.98:gb=0.03:"
        "br=0.0:bg=0.03:bb=1.10,"
        "curves=m='0/0 0.25/0.18 0.5/0.48 0.75/0.78 1/1'"
    ),

    # No grading — passthrough
    "none": "",
}

DEFAULT_GRADE = "cinematic"


def grade_clip(input_path: Path, output_path: Path,
               profile: str = DEFAULT_GRADE) -> bool:
    """Apply color grading to a single clip.

    Returns True if grading succeeded and output exists.
    Falls through gracefully on failure (returns False, original clip untouched).
    """
    grade_filter = GRADES.get(profile, GRADES[DEFAULT_GRADE])
    if not grade_filter:
        return False

    ok = run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", grade_filter,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "20",
        "-r", str(FPS), "-an",
        str(output_path)
    ], timeout=120)

    return bool(ok and output_path.exists())

"""
hook_card.py — Animated hook text overlay for the first 1-2 seconds of a Short.

Renders bold, centered text that fades in quickly with a slight scale-up,
creating the punchy "scroll-stopping" hook that top Shorts channels use.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

from .config import OUT_W, OUT_H, FPS
from .ffmpeg_utils import run_ffmpeg


# ── Font ──────────────────────────────────────────────────────────────────────
def _hook_font(size: int) -> ImageFont.FreeTypeFont:
    from .overlays import _font
    return _font(size)


# ── Settings ──────────────────────────────────────────────────────────────────
HOOK_DURATION = 2.0       # seconds the hook text is visible
FADE_IN = 0.35            # seconds to fade in
HOLD = 1.3                # seconds at full opacity
FADE_OUT = 0.35           # seconds to fade out
TEXT_COLOR = (255, 255, 255)
SHADOW_COLOR = (0, 0, 0)
SHADOW_OFFSET = 4
MAX_FONT_SIZE = 82
MIN_FONT_SIZE = 54
LINE_SPACING = 16
TEXT_MARGIN = 80           # horizontal margin from edges
# Vertical position: upper third of the frame (below safe zone for platform UI)
TEXT_Y_CENTER = int(OUT_H * 0.20)


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Word-wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = font.getbbox(test)
        if bbox[2] - bbox[0] > max_width and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)
    return lines


def _find_font_size(text: str, max_width: int, max_lines: int = 3) -> tuple[ImageFont.FreeTypeFont, list[str]]:
    """Find largest font size that fits text in max_width within max_lines."""
    for size in range(MAX_FONT_SIZE, MIN_FONT_SIZE - 1, -2):
        font = _hook_font(size)
        lines = _wrap_text(text, font, max_width)
        if len(lines) <= max_lines:
            return font, lines
    font = _hook_font(MIN_FONT_SIZE)
    return font, _wrap_text(text, font, max_width)[:max_lines]


def _render_text_frame(text_lines: list[str], font: ImageFont.FreeTypeFont,
                       alpha: float) -> Image.Image:
    """Render text lines centered on a transparent RGBA frame."""
    frame = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(frame)

    # Calculate total text block height
    line_heights = []
    line_widths = []
    for line in text_lines:
        bbox = font.getbbox(line)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    total_h = sum(line_heights) + LINE_SPACING * (len(text_lines) - 1)
    y = TEXT_Y_CENTER - total_h // 2

    a = int(alpha * 255)
    shadow_a = int(alpha * 180)

    for i, line in enumerate(text_lines):
        x = (OUT_W - line_widths[i]) // 2
        # Shadow (multiple offsets for thickness)
        for dx, dy in [(SHADOW_OFFSET, SHADOW_OFFSET), (-1, -1), (2, 0), (0, 2)]:
            draw.text((x + dx, y + dy), line, font=font,
                      fill=SHADOW_COLOR + (shadow_a,))
        # Main text
        draw.text((x, y), line, font=font, fill=TEXT_COLOR + (a,))
        y += line_heights[i] + LINE_SPACING

    return frame


def burn_hook_text(video_path: Path, output_path: Path, hook_text: str) -> bool:
    """Burn an animated hook text overlay onto the first HOOK_DURATION seconds of a clip.

    Renders text frames as a PNG sequence, then composites via ffmpeg overlay
    with a fade-in/hold/fade-out envelope.
    """
    if not hook_text or not hook_text.strip():
        return False

    text = hook_text.strip().upper()
    max_width = OUT_W - 2 * TEXT_MARGIN
    font, lines = _find_font_size(text, max_width)

    if not lines:
        return False

    total_frames = int(HOOK_DURATION * FPS)
    fade_in_frames = int(FADE_IN * FPS)
    fade_out_frames = int(FADE_OUT * FPS)
    hold_start = fade_in_frames
    hold_end = total_frames - fade_out_frames

    # Render PNG sequence to temp dir
    tmp_dir = video_path.parent / "_hook_frames"
    tmp_dir.mkdir(exist_ok=True)

    for f_idx in range(total_frames):
        if f_idx < hold_start:
            alpha = f_idx / max(fade_in_frames, 1)
        elif f_idx >= hold_end:
            alpha = (total_frames - 1 - f_idx) / max(fade_out_frames, 1)
        else:
            alpha = 1.0
        alpha = max(0.0, min(1.0, alpha))

        frame = _render_text_frame(lines, font, alpha)
        frame.save(str(tmp_dir / f"hook_{f_idx:04d}.png"), "PNG")

    # Overlay the hook PNG sequence onto the video
    ok = run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-framerate", str(FPS),
        "-i", str(tmp_dir / "hook_%04d.png"),
        "-filter_complex",
        f"[1:v]format=rgba[hook];[0:v][hook]overlay=0:0:eof_action=pass:enable='lte(t,{HOOK_DURATION})'",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "20",
        "-r", str(FPS), "-an",
        str(output_path)
    ], timeout=120)

    # Cleanup
    import shutil
    shutil.rmtree(str(tmp_dir), ignore_errors=True)

    if ok and output_path.exists():
        print(f"  [hook] Burned hook text onto {output_path.name}")
        return True
    print(f"  [hook] FAILED for {video_path.name}")
    return False

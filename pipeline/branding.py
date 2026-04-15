"""
branding.py — Yellow border overlay + Urban Vectors watermark for all shorts.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter
import numpy as np

from .config import OUT_W, OUT_H
from .ffmpeg_utils import run_ffmpeg

# Paths
_LOGO_SRC = Path(__file__).resolve().parent.parent / "Urban Vectors logo on purple gradient.png"
_CACHE_DIR = Path(__file__).resolve().parent.parent / ".branding_cache"

# Branding constants
BORDER_COLOR = (255, 215, 0)  # Gold yellow #FFD700
BORDER_WIDTH = 10             # pixels — thick enough to read on mobile
WATERMARK_OPACITY = 0.65      # 65% opacity per user latest request
WATERMARK_HEIGHT = 100        # Down to 100px per final user request
WATERMARK_MARGIN = 20         # 20px margins to keep it closely in the corner


def _ensure_cache_dir():
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _extract_logo_transparent() -> Path:
    """Extract logo from purple gradient background -> white logo on transparent."""
    out = _CACHE_DIR / "logo_transparent.png"
    if out.exists():
        return out
    _ensure_cache_dir()

    img = Image.open(str(_LOGO_SRC)).convert("RGBA")
    arr = np.array(img, dtype=np.float32)

    # The logo is dark charcoal on light purple gradient.
    # Background ~155 lum, logo text/arrow ~100-120 lum, shadow ~130 lum.
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    # Sample background from all 4 corners (large patches for accuracy)
    corners = [lum[:20, :20], lum[:20, -20:], lum[-20:, :20], lum[-20:, -20:]]
    bg_lum = np.mean([c.mean() for c in corners])

    # The source logo is already a transparent PNG, but the RGB channels 
    # contain a purple background gradient. We keep the original high-quality
    # alpha mask and simply force the logo to be pure white.
    orig_alpha = arr[:, :, 3].astype(np.uint8)

    # REMOVE GHOST PIXELS: The source logo has alpha values of 1-2 at the 
    # very edges of the frame, which prevents getbbox() from cropping tightly.
    # We threshold anything below alpha 10 to zero to ensure a crisp box.
    orig_alpha[orig_alpha < 10] = 0

    # CALCULATE BBOX ON ALPHA ONLY: getbbox() uses all channels. 
    # Since we make RGB pure white (255) below, it would incorrectly 
    # consider the whole frame as bounds.
    mask_img = Image.fromarray(orig_alpha, "L")
    bbox = mask_img.getbbox()

    # Create white logo with original alpha
    result = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    result[:, :, 0] = 255  # R
    result[:, :, 1] = 255  # G
    result[:, :, 2] = 255  # B
    result[:, :, 3] = orig_alpha

    out_img = Image.fromarray(result, "RGBA")

    # Crop to content using the alpha bounding box
    if bbox:
        out_img = out_img.crop(bbox)

    out_img.save(str(out), "PNG")
    print(f"  [branding] Extracted logo -> {out}")
    return out


def _create_branding_overlay() -> Path:
    """Create a single RGBA overlay with yellow border + watermark logo."""
    out = _CACHE_DIR / "branding_overlay.png"
    if out.exists():
        return out
    _ensure_cache_dir()

    overlay = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw yellow border — render at 2x then downscale for smooth anti-aliased edges
    ss = 2  # supersampling factor
    border_layer = Image.new("RGBA", (OUT_W * ss, OUT_H * ss), (0, 0, 0, 0))
    bd = ImageDraw.Draw(border_layer)
    bw = BORDER_WIDTH * ss
    bc = BORDER_COLOR + (255,)
    # Outer rectangle (full frame)
    bd.rectangle([0, 0, OUT_W * ss - 1, OUT_H * ss - 1], outline=bc, width=bw)
    # Downscale with LANCZOS for smooth edges
    border_layer = border_layer.resize((OUT_W, OUT_H), Image.LANCZOS)
    overlay.paste(border_layer, (0, 0), border_layer)

    # Add watermark logo (bottom-right)
    logo_path = _extract_logo_transparent()
    logo = Image.open(str(logo_path)).convert("RGBA")

    # Scale to target height
    scale = WATERMARK_HEIGHT / logo.height
    new_w = int(logo.width * scale)
    logo = logo.resize((new_w, WATERMARK_HEIGHT), Image.LANCZOS)

    # Apply watermark opacity
    logo_arr = np.array(logo, dtype=np.float32)
    logo_arr[:, :, 3] *= WATERMARK_OPACITY
    logo = Image.fromarray(logo_arr.astype(np.uint8), "RGBA")

    # Position: bottom-right with margin (above bottom border)
    x = OUT_W - new_w - WATERMARK_MARGIN
    y = OUT_H - WATERMARK_HEIGHT - WATERMARK_MARGIN - BORDER_WIDTH
    overlay.paste(logo, (x, y), logo)

    overlay.save(str(out), "PNG")
    print(f"  [branding] Created overlay -> {out}")
    return out


def apply_branding(video_path: Path, output_path: Path) -> bool:
    """Apply branding overlay (yellow border + watermark) to a video via ffmpeg."""
    overlay_path = _create_branding_overlay()

    ok = run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(overlay_path),
        "-filter_complex", "[0:v][1:v]overlay=0:0:format=auto",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "copy",
        str(output_path)
    ], timeout=180)

    if ok and output_path.exists():
        print(f"  [branding] Applied to {output_path.name}")
        return True
    print(f"  [branding] FAILED for {video_path.name}")
    return False

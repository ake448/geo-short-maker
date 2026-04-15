"""
overlays.py — Map rendering, feature highlights, city labels, title overlay.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from .config import (
    _SSL_CTX, ROOT, TILE_SIZE, OUT_W, OUT_H, FPS, USER_AGENT, FONTS_DIR,
)
from .geodata import lat_lon_to_pixel, composite_to_frame, rings_to_pixels
from .ffmpeg_utils import run_ffmpeg


def _font(size):
    for fp in [str(ROOT.parent / "fonts" / "Montserrat-Bold.ttf"),
               "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"]:
        try:
            return ImageFont.truetype(fp, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _font_reg(size):
    for fp in [str(ROOT.parent / "fonts" / "Montserrat-Regular.ttf"),
               "C:/Windows/Fonts/arial.ttf"]:
        try:
            return ImageFont.truetype(fp, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def hex_rgba(h, a=255):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a)


def make_mask(size, pixel_rings):
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for ring in pixel_rings:
        pts = [(int(round(x)), int(round(y))) for x, y in ring]
        if len(pts) >= 3:
            draw.polygon(pts, fill=255)
    return mask


def draw_cities(img, pixel_rings, cities, geo, font_lg, font_sm,
                color=(255, 255, 255, 230)):
    """Draw city labels inside visible polygon area."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    mask = make_mask(img.size, pixel_rings)
    zoom, cols, rows = geo["zoom"], geo["cols"], geo["rows"]
    cw, ch = cols * TILE_SIZE, rows * TILE_SIZE

    for city in cities:
        cpx, cpy = lat_lon_to_pixel(city["lat"], city["lon"],
                                     geo["lat"], geo["lon"], zoom, cols, rows)
        fpx, fpy = composite_to_frame(cpx, cpy, cw, ch)
        ix, iy = int(round(fpx)), int(round(fpy))
        if ix < 0 or ix >= img.size[0] or iy < 0 or iy >= img.size[1]:
            continue
        try:
            if mask.getpixel((ix, iy)) < 128:
                continue
        except IndexError:
            continue

        font = font_lg if city.get("rank", 3) <= 2 else font_sm
        name = city["name"]
        bbox = draw.textbbox((0, 0), name, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = max(5, min(ix - tw // 2, img.size[0] - tw - 5))

        dot_r = 4 if city.get("rank", 3) <= 2 else 3
        draw.ellipse([ix-dot_r, iy-dot_r, ix+dot_r, iy+dot_r], fill=color)
        ty = iy + dot_r + 4
        draw.text((tx+2, ty+2), name, font=font, fill=(0, 0, 0, 160))
        draw.text((tx, ty), name, font=font, fill=color)

    base = img.convert("RGBA") if img.mode != "RGBA" else img
    return Image.alpha_composite(base, overlay)


def fetch_feature_geometry(query: str, geo: Dict) -> Optional[List]:
    """Fetch polygon/linestring geometry for a specific feature (river, desert, etc.)
    from Nominatim and convert to pixel coordinates for overlay drawing."""
    print(f"    Fetching feature geometry: {query}")
    time.sleep(1.2)
    params = urllib.parse.urlencode({
        "q": query, "format": "jsonv2", "limit": 3,
        "polygon_geojson": 1, "polygon_threshold": 0.001,
    })
    url = f"https://nominatim.openstreetmap.org/search?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"    [WARN] Feature lookup failed: {e}")
        return None

    if not data:
        print(f"    [WARN] No results for feature: {query}")
        return None

    for r in data:
        geojson = r.get("geojson")
        if not geojson:
            continue
        gtype = geojson.get("type", "")
        coords = geojson.get("coordinates", [])

        rings = []
        if gtype == "Polygon":
            for ring in coords:
                rings.append([(c[1], c[0]) for c in ring])
        elif gtype == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    rings.append([(c[1], c[0]) for c in ring])
        elif gtype == "LineString":
            rings.append([(c[1], c[0]) for c in coords])
        elif gtype == "MultiLineString":
            for line in coords:
                rings.append([(c[1], c[0]) for c in line])
        else:
            continue

        if not rings:
            continue

        zoom = geo.get("zoom", 8)
        cols = geo.get("cols", 10)
        rows = geo.get("rows", 14)
        pixel_rings = rings_to_pixels(rings, geo["lat"], geo["lon"], zoom, cols, rows)
        pixel_rings = [r for r in pixel_rings
                       if any(0 <= x <= OUT_W and 0 <= y <= OUT_H for x, y in r)]

        if pixel_rings:
            total_pts = sum(len(r) for r in pixel_rings)
            print(f"    [OK] Feature '{query}': {len(pixel_rings)} ring(s), {total_pts} pts")
            return pixel_rings

    print(f"    [WARN] No usable geometry for: {query}")
    return None


def draw_feature_highlight(img: Image.Image, feature_rings: List,
                           color_hex: str = "#44AAFF",
                           opacity: int = 80,
                           is_line: bool = False) -> Image.Image:
    """Draw a semi-transparent colored polygon/line overlay for a geographic feature."""
    base = img.convert("RGBA") if img.mode != "RGBA" else img.copy()
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)

    for ring in feature_rings:
        pts = [(int(round(x)), int(round(y))) for x, y in ring]
        if len(pts) < 2:
            continue

        if is_line or len(pts) < 4:
            draw.line(pts, fill=(r, g, b, 40), width=18)
            draw.line(pts, fill=(r, g, b, 80), width=10)
            draw.line(pts, fill=(r, g, b, 220), width=4)
        else:
            if len(pts) >= 3:
                draw.polygon(pts, fill=(r, g, b, opacity))
            draw.line(pts + [pts[0]], fill=(r, g, b, 200), width=3)

    return Image.alpha_composite(base, overlay)


def burn_title_overlay(video_path: Path, title_text: str, out_path: Path,
                       display_duration: float = 4.0) -> bool:
    """Burn an animated title card overlay onto the first N seconds of a video."""
    if not video_path.exists():
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)

    words = title_text.split()
    lines = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 > 28:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        lines.append(cur)

    font_size = 42
    line_h = font_size + 8
    num_lines = len(lines)
    box_h = num_lines * line_h + 30
    rest_y = 100
    start_y = -box_h - 20
    box_w = min(OUT_W - 80, max(len(l) for l in lines) * (font_size * 0.55) + 60)
    box_x = int((OUT_W - box_w) / 2)

    slide_in_dur = 0.4
    fade_out_start = display_duration - 0.5
    fade_out_dur = 0.5

    y_expr = f"{start_y}+({rest_y}-({start_y}))*min(t/{slide_in_dur}\\,1)"

    alpha_expr = (
        f"if(lt(t\\,0.3)\\, t/0.3\\, "
        f"if(gt(t\\,{fade_out_start})\\, "
        f"({display_duration}-t)/{fade_out_dur}\\, 1))"
    )

    vf_parts = []

    vf_parts.append(
        f"drawbox=x={box_x}:y='{y_expr}':w={int(box_w)}:h={box_h}"
        f":color=white@0.92:t=fill"
        f":enable='between(t,0,{display_duration})'"
    )

    for i, line in enumerate(lines):
        escaped = line.replace("'", "'\\\\\\'").replace(":", "\\\\:")
        text_y_offset = 15 + i * line_h
        text_y_expr = f"{start_y}+({rest_y}-({start_y}))*min(t/{slide_in_dur}\\,1)+{text_y_offset}"
        vf_parts.append(
            f"drawtext=text='{escaped}'"
            f":fontsize={font_size}:fontcolor=black:alpha='{alpha_expr}'"
            f":x=(w-text_w)/2:y='{text_y_expr}'"
            f":enable='between(t,0,{display_duration})'"
        )

    vf = ",".join(vf_parts)

    ok = run_ffmpeg([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-r", str(FPS), "-an",
        str(out_path)
    ], timeout=180)

    return ok

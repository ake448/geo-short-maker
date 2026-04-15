"""
broll.py — B-roll generators: satellite, map, dark cutout, earth pan, 3D wrappers.
"""
from __future__ import annotations

import io
import importlib.util
import math
import os
import random
import shutil
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from .config import (
    ROOT, TILE_SIZE, OUT_W, OUT_H, FPS, CACHE_DIR, USER_AGENT
)
from .ffmpeg_utils import run_ffmpeg
from .geodata import fetch_boundary
from .overlays import _font, _font_reg, hex_rgba, make_mask, draw_cities


_GEN_CINEMATICS_MODULE = None

CLR_CYAN       = (0, 229, 255)
CLR_AMBER      = (245, 158, 11)
CLR_RED        = (255, 68, 68)
CLR_GREEN      = (34, 197, 94)
CLR_PURPLE     = (168, 85, 247)
CLR_WHITE      = (255, 255, 255)
CLR_NAVY_BG    = (5, 8, 16)
CLR_OCEAN      = (13, 27, 45)
CLR_LAND_DARK  = (20, 30, 48)


def _ease_smooth(t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def _ease_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    return 1.0 - (1.0 - t) ** 3


def _frames_to_mp4(frames_dir: Path, out_path: Path, fps: int = FPS, duration: float = 5.0) -> bool:
    """Encode frames_dir/frame_%04d.jpg to a silent H.264 MP4."""
    return run_ffmpeg([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%04d.jpg"),
        "-vf", (
            "eq=contrast=1.18:brightness=-0.08:saturation=1.1,"
            "colorchannelmixer=rr=0.95:gg=0.97:bb=1.08:rb=0.03,"
            "unsharp=3:3:0.4:3:3:0,"
            "vignette=angle=0.22,format=yuv420p"
        ),
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-r", str(fps), "-t", str(duration), "-an",
        str(out_path)
    ], timeout=180)


def _scale_rgba_alpha(img: Image.Image, opacity: float) -> Image.Image:
    opacity = max(0.0, min(1.0, float(opacity)))
    layer = img.convert("RGBA").copy()
    a = np.array(layer.split()[3], dtype=np.float32)
    layer.putalpha(Image.fromarray(np.clip(a * opacity, 0, 255).astype(np.uint8)))
    return layer


def _grade_tile_dark(img: Image.Image) -> Image.Image:
    base = img.convert("RGBA").resize((OUT_W, OUT_H), Image.LANCZOS)
    base = ImageEnhance.Color(base).enhance(0.72)
    base = ImageEnhance.Contrast(base).enhance(1.22)
    base = ImageEnhance.Brightness(base).enhance(0.54)
    wash = Image.new("RGBA", base.size, (*CLR_NAVY_BG, 92))
    return Image.alpha_composite(base, wash)


def _fallback_dark_terrain() -> Image.Image:
    img = Image.new("RGBA", (OUT_W, OUT_H), (*CLR_NAVY_BG, 255))
    d = ImageDraw.Draw(img)
    # Subtle grid
    for y in range(0, OUT_H, 80):
        a = 30 if (y // 80) % 2 == 0 else 18
        d.line([(0, y), (OUT_W, y)], fill=(44, 64, 96, a), width=1)
    for x in range(0, OUT_W, 80):
        a = 26 if (x // 80) % 2 == 0 else 14
        d.line([(x, 0), (x, OUT_H)], fill=(44, 64, 96, a), width=1)
    # Sparse star field so glows have visual contrast to read against
    rng = random.Random(42)
    for _ in range(320):
        sx, sy = rng.randint(0, OUT_W - 1), rng.randint(0, OUT_H - 1)
        sa = rng.randint(60, 130)
        d.point((sx, sy), fill=(200, 220, 255, sa))
    return img


def _draw_glow_border(base: Image.Image, pixel_rings: list,
                      color_rgb: tuple = CLR_CYAN,
                      core_width: int = 2) -> Image.Image:
    """Composite a 3-layer neon border onto base."""
    r, g, b = color_rgb
    # Hot core is color-tinted pale — not pure white — so the neon hue reads through.
    cr = min(255, int(r * 0.35 + 255 * 0.65))
    cg = min(255, int(g * 0.35 + 255 * 0.65))
    cb = min(255, int(b * 0.35 + 255 * 0.65))
    core = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(core)
    for ring in pixel_rings or []:
        pts = [(int(round(x)), int(round(y))) for x, y in ring]
        if len(pts) >= 2:
            d.line(pts + [pts[0]], fill=(cr, cg, cb, 200), width=core_width)

    tight = core.copy().filter(ImageFilter.GaussianBlur(8))
    ta = np.array(tight.split()[3], dtype=np.float32)
    tight = Image.merge("RGBA", [
        Image.fromarray((ta * (r / 255.0)).clip(0, 255).astype(np.uint8)),
        Image.fromarray((ta * (g / 255.0)).clip(0, 255).astype(np.uint8)),
        Image.fromarray((ta * (b / 255.0)).clip(0, 255).astype(np.uint8)),
        Image.fromarray((ta * 0.85).clip(0, 255).astype(np.uint8)),
    ])

    small = core.resize((max(1, base.size[0] // 4), max(1, base.size[1] // 4)), Image.LANCZOS)
    wide_small = small.filter(ImageFilter.GaussianBlur(12))
    wide = wide_small.resize(base.size, Image.LANCZOS)
    wa = np.array(wide.split()[3], dtype=np.float32)
    wide = Image.merge("RGBA", [
        Image.fromarray((wa * (r / 255.0)).clip(0, 255).astype(np.uint8)),
        Image.fromarray((wa * (g / 255.0)).clip(0, 255).astype(np.uint8)),
        Image.fromarray((wa * (b / 255.0)).clip(0, 255).astype(np.uint8)),
        Image.fromarray((wa * 0.52).clip(0, 255).astype(np.uint8)),
    ])

    out = base.convert("RGBA")
    out = Image.alpha_composite(out, wide)
    out = Image.alpha_composite(out, tight)
    return Image.alpha_composite(out, core)


def _outline_layer(pixel_rings: list, size: Tuple[int, int] = (OUT_W, OUT_H),
                   color_rgb: tuple = CLR_CYAN, core_width: int = 2,
                   close: bool = True) -> Image.Image:
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    return _draw_glow_border(layer, pixel_rings, color_rgb=color_rgb, core_width=core_width)


def _partial_rings(pixel_rings: list, progress: float) -> list:
    progress = max(0.0, min(1.0, float(progress)))
    out = []
    for ring in pixel_rings or []:
        pts = [(float(x), float(y)) for x, y in ring]
        if len(pts) < 2:
            continue
        closed = pts + [pts[0]]
        segs = []
        total = 0.0
        for a, b in zip(closed[:-1], closed[1:]):
            dist = math.hypot(b[0] - a[0], b[1] - a[1])
            segs.append((a, b, dist))
            total += dist
        target = total * progress
        if target <= 0:
            continue
        drawn = []
        rem = target
        for a, b, dist in segs:
            if not drawn:
                drawn.append(a)
            if rem >= dist:
                drawn.append(b)
                rem -= dist
            else:
                k = rem / max(dist, 1e-6)
                drawn.append((a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k))
                break
        if len(drawn) >= 2:
            out.append(drawn)
    return out


def _fill_layer(pixel_rings: list, color_rgb: tuple, alpha: int,
                size: Tuple[int, int] = (OUT_W, OUT_H)) -> Image.Image:
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for ring in pixel_rings or []:
        pts = [(int(round(x)), int(round(y))) for x, y in ring]
        if len(pts) >= 3:
            d.polygon(pts, fill=(*color_rgb, alpha))
    return layer


def _zoom_frame(img: Image.Image, zoom: float, dx: float = 0.0, dy: float = 0.0) -> Image.Image:
    zoom = max(1.0, float(zoom))
    w, h = img.size
    nw, nh = int(w * zoom), int(h * zoom)
    big = img.resize((nw, nh), Image.LANCZOS)
    left = int((nw - w) / 2 + dx)
    top = int((nh - h) / 2 + dy)
    left = max(0, min(left, nw - w))
    top = max(0, min(top, nh - h))
    return big.crop((left, top, left + w, top + h))


def _rdp_simplify_ring(ring: list, tolerance: float = 0.002) -> list:
    """Ramer-Douglas-Peucker ring simplification. tolerance in degrees (~200m at equator)."""
    pts = np.asarray(ring, dtype=np.float64)
    n = len(pts)
    if n <= 200:
        return ring
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        s, e = stack.pop()
        if e - s <= 1:
            continue
        seg = pts[e] - pts[s]
        seg_len = np.linalg.norm(seg) + 1e-30
        seg_dir = seg / seg_len
        vecs = pts[s + 1:e] - pts[s]
        proj = np.dot(vecs, seg_dir)
        perp = vecs - proj[:, None] * seg_dir[None, :]
        dists = np.linalg.norm(perp, axis=1)
        mi = int(np.argmax(dists))
        if dists[mi] > tolerance:
            keep[s + 1 + mi] = True
            stack.append((s, s + 1 + mi))
            stack.append((s + 1 + mi, e))
    return [tuple(p) for p in pts[keep]]


def _simplify_rings(rings: list, tolerance: float = 0.002) -> list:
    """Simplify all rings, dropping tiny slivers, keeping enough detail for pixel rendering."""
    out = []
    for ring in rings or []:
        simplified = _rdp_simplify_ring(ring, tolerance)
        if len(simplified) >= 4:
            out.append(simplified)
    return out


def _rings_bbox(rings: list) -> Optional[Tuple[float, float, float, float]]:
    pts = [(float(lat), float(lon)) for ring in rings or [] for lat, lon in ring]
    if not pts:
        return None
    lats = [p[0] for p in pts]
    lons = [p[1] for p in pts]
    return min(lats), min(lons), max(lats), max(lons)


def _rings_center(rings: list, fallback: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
    bbox = _rings_bbox(rings)
    if not bbox:
        return fallback
    min_lat, min_lon, max_lat, max_lon = bbox
    return (min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0


def _mercator_m(lat: float, lon: float) -> Tuple[float, float]:
    lat = max(-85.05112878, min(85.05112878, float(lat)))
    r = 6378137.0
    x = r * math.radians(float(lon))
    y = r * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return x, y


def _rings_mercator_bounds(rings_list: List[list]) -> Optional[Tuple[float, float, float, float]]:
    pts = []
    for rings in rings_list or []:
        for ring in rings or []:
            for lat, lon in ring:
                pts.append(_mercator_m(lat, lon))
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _nice_km(value: float) -> int:
    value = max(1.0, float(value))
    exp = 10 ** int(math.floor(math.log10(value)))
    for m in (1, 2, 5, 10):
        nice = m * exp
        if nice >= value:
            return int(nice)
    return int(value)


def _fit_px_per_km(rings_list: List[list], box_w: int, box_h: int, fill: float = 0.72) -> float:
    bounds = _rings_mercator_bounds(rings_list)
    if not bounds:
        return 8.0
    min_x, min_y, max_x, max_y = bounds
    w_km = max(1.0, (max_x - min_x) / 1000.0)
    h_km = max(1.0, (max_y - min_y) / 1000.0)
    return max(0.05, min((box_w * fill) / w_km, (box_h * fill) / h_km))


def _project_rings_same_scale(rings: list, px_per_km: float, size: Tuple[int, int],
                              center_latlon: Optional[Tuple[float, float]] = None,
                              offset: Tuple[float, float] = (0, 0)) -> list:
    if not rings:
        return []
    if center_latlon is None:
        center_latlon = _rings_center(rings)
    cx_m, cy_m = _mercator_m(center_latlon[0], center_latlon[1])
    w, h = size
    ox, oy = offset
    out = []
    for ring in rings:
        pts = []
        for lat, lon in ring:
            x_m, y_m = _mercator_m(lat, lon)
            x = w / 2 + ox + ((x_m - cx_m) / 1000.0) * px_per_km
            y = h / 2 + oy - ((y_m - cy_m) / 1000.0) * px_per_km
            pts.append((x, y))
        out.append(pts)
    return out


def _project_rings_fit(rings: list, size: Tuple[int, int] = (OUT_W, OUT_H),
                       margin: float = 0.16, center_latlon: Optional[Tuple[float, float]] = None) -> list:
    if not rings:
        return []
    rings = _simplify_rings(rings)
    if not rings:
        return []
    w, h = size
    px_per_km = _fit_px_per_km([rings], int(w), int(h), fill=1.0 - margin)
    return _project_rings_same_scale(rings, px_per_km, size, center_latlon=center_latlon)


def _translate_rings(pixel_rings: list, dx: float, dy: float) -> list:
    return [[(x + dx, y + dy) for x, y in ring] for ring in pixel_rings or []]


def _project_points_same_scale(points: list, px_per_km: float, size: Tuple[int, int],
                               center_latlon: Tuple[float, float]) -> list:
    cx_m, cy_m = _mercator_m(center_latlon[0], center_latlon[1])
    w, h = size
    out = []
    for lat, lon in points or []:
        x_m, y_m = _mercator_m(lat, lon)
        out.append((w / 2 + ((x_m - cx_m) / 1000.0) * px_per_km,
                    h / 2 - ((y_m - cy_m) / 1000.0) * px_per_km))
    return out


def _draw_centered_text(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int],
                        text: str, font, fill=(*CLR_WHITE, 255), stroke_width: int = 0,
                        stroke_fill=(0, 0, 0, 200)):
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = box[0] + (box[2] - box[0] - tw) / 2
    y = box[1] + (box[3] - box[1] - th) / 2
    draw.text((x, y), text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)


def _draw_scaled_center_label(base: Image.Image, text: str, center: Tuple[int, int],
                              font, scale: float, fill=(*CLR_WHITE, 255)) -> Image.Image:
    if not text:
        return base
    tmp = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), text, font=font, stroke_width=3)
    tw, th = bbox[2] - bbox[0] + 16, bbox[3] - bbox[1] + 16
    label = Image.new("RGBA", (max(1, tw), max(1, th)), (0, 0, 0, 0))
    ld = ImageDraw.Draw(label)
    ld.text((8, 8), text, font=font, fill=fill, stroke_width=3, stroke_fill=(0, 0, 0, 220))
    scale = max(0.1, float(scale))
    label = label.resize((max(1, int(tw * scale)), max(1, int(th * scale))), Image.LANCZOS)
    tmp.alpha_composite(label, (int(center[0] - label.size[0] / 2), int(center[1] - label.size[1] / 2)))
    return Image.alpha_composite(base.convert("RGBA"), tmp)


def _draw_scale_bar(img: Image.Image, px_per_km: float, y: int = OUT_H - 125) -> Image.Image:
    px_per_km = max(0.01, float(px_per_km))
    km = _nice_km(180 / px_per_km)
    bar_w = int(km * px_per_km)
    bar_w = max(60, min(340, bar_w))
    x0 = (OUT_W - bar_w) // 2
    x1 = x0 + bar_w
    d = ImageDraw.Draw(img)
    d.line([(x0, y), (x1, y)], fill=(255, 255, 255, 230), width=4)
    d.line([(x0, y - 10), (x0, y + 10)], fill=(255, 255, 255, 230), width=3)
    d.line([(x1, y - 10), (x1, y + 10)], fill=(255, 255, 255, 230), width=3)
    text = f"{km} km"
    font = _font_reg(28)
    bbox = d.textbbox((0, 0), text, font=font)
    d.text(((OUT_W - (bbox[2] - bbox[0])) / 2, y + 14), text, font=font,
           fill=(255, 255, 255, 230), stroke_width=2, stroke_fill=(0, 0, 0, 200))
    return img


def _comparison_data(geo: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    beat = geo.get("_beat") if isinstance(geo.get("_beat"), dict) else {}
    comp = beat.get("comparison") if isinstance(beat.get("comparison"), dict) else {}
    name_b = str(comp.get("region_b_name") or comp.get("name_b") or "").strip()
    if not name_b:
        return None
    rings_a = geo.get("rings") or []
    if not rings_a:
        return None
    center_b, rings_b, bbox_b = fetch_boundary(name_b)
    if not rings_b:
        return None
    label_a = str(comp.get("label_a") or geo.get("_region") or "REGION A").strip().upper()
    label_b = str(comp.get("label_b") or name_b).strip().upper()
    center_a = (float(geo.get("lat") or _rings_center(rings_a)[0]),
                float(geo.get("lon") or _rings_center(rings_a)[1]))
    if not center_b:
        center_b = (float(comp.get("region_b_lat") or _rings_center(rings_b)[0]),
                    float(comp.get("region_b_lon") or _rings_center(rings_b)[1]))
    return {
        "rings_a": rings_a,
        "rings_b": rings_b,
        "bbox_a": geo.get("bbox") or _rings_bbox(rings_a),
        "bbox_b": bbox_b or _rings_bbox(rings_b),
        "center_a": center_a,
        "center_b": center_b,
        "label_a": label_a,
        "label_b": label_b,
        "region_b_name": name_b,
    }


def _zoom_for_rings(rings_list: List[list]) -> int:
    bbox = _rings_bbox([pt for rings in rings_list for pt in rings])
    if not bbox:
        return 8
    min_lat, min_lon, max_lat, max_lon = bbox
    span = max(abs(max_lat - min_lat), abs(max_lon - min_lon))
    if span > 35:
        return 3
    if span > 18:
        return 4
    if span > 8:
        return 5
    if span > 4:
        return 6
    if span > 1.5:
        return 7
    if span > 0.7:
        return 8
    return 9


def _geo_frame_path(geo: Dict[str, Any], key: str) -> Optional[Path]:
    raw = str(geo.get(key, "") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.name in {"", "."}:
        return None
    return path


def _load_generate_cinematics_module(force_reload: bool = False):
    global _GEN_CINEMATICS_MODULE
    if force_reload:
        _GEN_CINEMATICS_MODULE = None
    if _GEN_CINEMATICS_MODULE is not None:
        return _GEN_CINEMATICS_MODULE
    os.environ.setdefault("GA_RENDER_SCALE", "0.42")
    os.environ.setdefault("GA_RENDER_FPS", "36")
    os.environ.setdefault("GA_ATLAS_WORKERS", "4")
    os.environ.setdefault("GA_FRAME_WORKERS", "1")
    os.environ.setdefault("GA_FRAME_BATCH", "1")
    gc_path = ROOT / "generate_cinematics.py"
    if not gc_path.exists():
        _GEN_CINEMATICS_MODULE = False
        return None
    try:
        spec = importlib.util.spec_from_file_location("_geo_generate_cinematics", str(gc_path))
        if not spec or not spec.loader:
            _GEN_CINEMATICS_MODULE = False
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _GEN_CINEMATICS_MODULE = mod
        return mod
    except Exception:
        _GEN_CINEMATICS_MODULE = False
        return None


# ── Satellite/Map generators ─────────────────────────────────────────────────

def gen_satellite_zoom(geo, out_path, duration=6.0):
    """Zoom-in effect on satellite image using fast crop+scale approach."""
    src = _geo_frame_path(geo, "satellite_wide")
    if not src or not src.exists():
        src = _geo_frame_path(geo, "satellite_frame")
    if not src or not src.exists():
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scale = 3
    pw, ph = OUT_W * scale, OUT_H * scale
    prescale = out_path.parent / f"_pre_zoom_{out_path.stem}.jpg"
    if not run_ffmpeg([
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"scale={pw}:{ph}:force_original_aspect_ratio=increase,crop={pw}:{ph}",
        str(prescale)], timeout=60):
        return False
    total_n = int(duration * FPS)
    vf = (
        f"crop="
        f"w='trunc({pw}*(1-n*2/(3*{total_n})))':"
        f"h='trunc({ph}*(1-n*2/(3*{total_n})))':"
        f"x='trunc(({pw}-trunc({pw}*(1-n*2/(3*{total_n}))))/2)':"
        f"y='trunc(({ph}-trunc({ph}*(1-n*2/(3*{total_n}))))/2)',"
        f"scale={OUT_W}:{OUT_H},format=yuv420p"
    )
    ok = run_ffmpeg([
        "ffmpeg", "-y", "-loop", "1", "-framerate", str(FPS), "-i", str(prescale),
        "-vf", vf, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
        "-r", str(FPS), "-frames:v", str(total_n), str(out_path)], timeout=120)
    prescale.unlink(missing_ok=True)
    return ok


def gen_satellite_pan(geo, out_path, duration=5.0, direction="left_to_right"):
    """Pan across satellite imagery using fast crop approach."""
    src = _geo_frame_path(geo, "satellite_frame")
    if not src or not src.exists():
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if direction == "auto":
        direction = random.choice(["left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"])
    multiplier = random.uniform(1.8, 2.3)

    is_vertical = direction in ("top_to_bottom", "bottom_to_top")
    pw = OUT_W if is_vertical else int(OUT_W * multiplier)
    ph = int(OUT_H * multiplier) if is_vertical else OUT_H

    prescale = out_path.parent / f"_pre_pan_{out_path.stem}.jpg"
    if not run_ffmpeg([
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"scale={pw}:{ph}:force_original_aspect_ratio=increase,crop={pw}:{ph}",
        str(prescale)], timeout=60):
        return False

    total_n = int(float(duration) * FPS)
    if is_vertical:
        pan_px = ph - OUT_H
        y_expr = f"trunc({pan_px}*n/{total_n})" if direction == "top_to_bottom" else f"trunc({pan_px}*(1-n/{total_n}))"
        vf = f"crop={OUT_W}:{OUT_H}:x=0:y='{y_expr}',format=yuv420p"
    else:
        pan_px = pw - OUT_W
        x_expr = f"trunc({pan_px}*(1-n/{total_n}))" if direction == "right_to_left" else f"trunc({pan_px}*n/{total_n})"
        vf = f"crop={OUT_W}:{OUT_H}:x='{x_expr}':y=0,format=yuv420p"

    ok = run_ffmpeg([
        "ffmpeg", "-y", "-loop", "1", "-framerate", str(FPS), "-i", str(prescale),
        "-vf", vf, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
        "-r", str(FPS), "-frames:v", str(total_n), str(out_path)], timeout=120)
    prescale.unlink(missing_ok=True)
    return ok


def _fetch_terrain_tile(z: int, x: int, y: int, cache_dir: Path) -> Optional[Image.Image]:
    """Fetch a single Mapbox terrain tile."""
    MAPBOX_TOKEN = os.environ.get("MAPBOX_TOKEN", "")
    
    if MAPBOX_TOKEN:
        # Mapbox satellite-streets style — terrain texture + labels
        url = (f"https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v12"
               f"/tiles/512/{z}/{x}/{y}@2x?access_token={MAPBOX_TOKEN}")
    else:
        # OpenTopoMap fallback — free, no key needed, real terrain shading
        url = f"https://tile.opentopomap.org/{z}/{x}/{y}.png"
    
    cache_path = cache_dir / f"tile_{z}_{x}_{y}.png"
    if cache_path.exists():
        return Image.open(cache_path).convert("RGBA")
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data)).convert("RGBA")
        img.save(cache_path)
        return img
    except Exception:
        return None


def _lat_lon_to_tile(lat: float, lon: float, zoom: int):
    """Convert lat/lon to tile XY at given zoom."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(math.radians(lat)) +
             1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def _build_terrain_base(geo: Dict, zoom: int = 10) -> Optional[Image.Image]:
    """
    Stitch terrain tiles into a full OUT_W x OUT_H base image
    centered on the region's lat/lon.
    """
    # Use explicit lat/lon as the center — bbox ordering varies by source (Nominatim
    # returns [min_lat, max_lat, min_lon, max_lon]) so deriving center from bbox is error-prone.
    center_lat = float(geo.get("lat") or 0.0)
    center_lon = float(geo.get("lon") or 0.0)
    if center_lat == 0.0 and center_lon == 0.0:
        # Last resort: derive from rings
        rings = geo.get("rings", [])
        if rings:
            all_lats = [pt[0] for ring in rings for pt in ring]
            all_lons = [pt[1] for ring in rings for pt in ring]
            if all_lats and all_lons:
                center_lat = (min(all_lats) + max(all_lats)) / 2
                center_lon = (min(all_lons) + max(all_lons)) / 2
    if center_lat == 0.0 and center_lon == 0.0:
        return None
    
    cx, cy = _lat_lon_to_tile(center_lat, center_lon, zoom)
    
    # Fetch a 5x5 grid of tiles around the center
    GRID = 3
    tile_size = 512
    canvas_size = tile_size * (GRID * 2 + 1)
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (20, 20, 30, 255))
    
    cache_dir = CACHE_DIR / "terrain_tiles"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_and_paste(args):
        dx, dy = args
        tile = _fetch_terrain_tile(zoom, cx + dx, cy + dy, cache_dir)
        if tile:
            tile = tile.resize((tile_size, tile_size), Image.LANCZOS)
            px = (dx + GRID) * tile_size
            py = (dy + GRID) * tile_size
            return (px, py, tile)
        return None
    
    with ThreadPoolExecutor(max_workers=8) as pool:
        coords = [(dx, dy) for dx in range(-GRID, GRID+1)
                           for dy in range(-GRID, GRID+1)]
        for result in pool.map(fetch_and_paste, coords):
            if result:
                px, py, tile = result
                canvas.paste(tile, (px, py))
    
    # Crop to output size
    cx_px = canvas_size // 2
    cy_px = canvas_size // 2
    left = cx_px - OUT_W // 2
    top  = cy_px - OUT_H // 2
    return canvas.crop((left, top, left + OUT_W, top + OUT_H))


def gen_map_highlight(geo: Dict, out_path: Path, duration: float = 7.0) -> bool:
    """
    Cinematic map reveal:
      Phase 1 (0-2s):  Wide terrain view, slow Ken Burns zoom in
      Phase 2 (2-4s):  Region outline draws on with neon glow 
      Phase 3 (4-6s):  Fill pulses in, cities appear
      Phase 4 (6-7s):  Hold + slow pan
    """
    pixel_rings = geo.get("pixel_rings", [])
    if not pixel_rings:
        return False
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = out_path.parent / f"_mh_{out_path.stem}"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # --- Build terrain base ---
    terrain = _build_terrain_base(geo, zoom=9)
    if terrain is None:
        # Fallback to existing map_frame if tiles fail
        map_frame = _geo_frame_path(geo, "map_frame")
        if not map_frame or not map_frame.exists():
            return False
        terrain = Image.open(map_frame).convert("RGBA")
        terrain = terrain.resize((OUT_W, OUT_H), Image.LANCZOS)

    # Darken terrain slightly for cinematic look
    darkener = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 60))
    terrain = Image.alpha_composite(terrain, darkener)

    total_frames = int(duration * FPS)
    p1_end = int(2.0 * FPS)   # terrain zoom
    p2_end = int(4.0 * FPS)   # outline draw-on
    p3_end = int(6.0 * FPS)   # fill + cities
    # p4: hold + pan until total_frames

    NEON = (0, 200, 255)
    FILL = (80, 160, 255)

    # Pre-render the full outline layer (used progressively in phase 2)
    full_outline = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    fo_draw = ImageDraw.Draw(full_outline)
    for ring in pixel_rings:
        pts = [(int(round(x)), int(round(y))) for x, y in ring]
        fo_draw.line(pts + [pts[0]], fill=(*NEON, 220), width=3)
    full_glow = full_outline.filter(ImageFilter.GaussianBlur(radius=10))

    # Pre-render full fill layer
    fill_layer = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    fl_draw = ImageDraw.Draw(fill_layer)
    for ring in pixel_rings:
        pts = [(int(round(x)), int(round(y))) for x, y in ring]
        if len(pts) >= 3:
            fl_draw.polygon(pts, fill=(*FILL, 60))

    # Pre-render city layer
    cities_layer = terrain.copy()
    cities = geo.get("cities", [])
    if cities:
        cities_layer = draw_cities(
            Image.new("RGBA", (OUT_W, OUT_H), (0,0,0,0)),
            pixel_rings, cities, geo, _font(28), _font_reg(20),
            color=(220, 240, 255, 230)
        )

    for i in range(total_frames):
        t_global = i / max(total_frames - 1, 1)

        # --- Phase 1: Ken Burns zoom in ---
        if i <= p1_end:
            t = i / max(p1_end, 1)
            t_ease = t * t * (3 - 2 * t)
            # Zoom from 1.15x down to 1.0x (zooming IN)
            zoom_factor = 1.15 - 0.15 * t_ease
            new_w = int(OUT_W * zoom_factor)
            new_h = int(OUT_H * zoom_factor)
            left = (new_w - OUT_W) // 2
            top  = (new_h - OUT_H) // 2
            frame_base = terrain.resize((new_w, new_h), Image.LANCZOS)
            frame_base = frame_base.crop((left, top, left + OUT_W, top + OUT_H))
            frame = frame_base.copy()

        # --- Phase 2: Outline draws on progressively ---
        elif i <= p2_end:
            t = (i - p1_end) / max(p2_end - p1_end, 1)
            t_ease = t * t * (3 - 2 * t)
            frame = terrain.copy()
            # Composite glow at increasing alpha
            glow_alpha = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
            glow_alpha.paste(full_glow, mask=Image.fromarray(
                (np.array(full_glow.split()[3]) * t_ease).astype(np.uint8)
            ))
            outline_alpha = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
            outline_alpha.paste(full_outline, mask=Image.fromarray(
                (np.array(full_outline.split()[3]) * t_ease).astype(np.uint8)
            ))
            frame = Image.alpha_composite(frame, glow_alpha)
            frame = Image.alpha_composite(frame, outline_alpha)

        # --- Phase 3: Fill pulses in + cities fade in ---
        elif i <= p3_end:
            t = (i - p2_end) / max(p3_end - p2_end, 1)
            t_ease = t * t * (3 - 2 * t)
            # Subtle pulse on fill alpha
            pulse = 0.85 + 0.15 * math.sin(t * math.pi * 2)
            frame = terrain.copy()
            frame = Image.alpha_composite(frame, full_glow)
            frame = Image.alpha_composite(frame, full_outline)
            fill_alpha = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
            fill_alpha.paste(fill_layer, mask=Image.fromarray(
                (np.array(fill_layer.split()[3]) * t_ease * pulse).clip(0, 255).astype(np.uint8)
            ))
            frame = Image.alpha_composite(frame, fill_alpha)
            # Cities fade in
            if cities:
                city_mask = Image.fromarray(
                    (np.array(cities_layer.split()[3]) * t_ease).astype(np.uint8)
                )
                city_fade = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
                city_fade.paste(cities_layer, mask=city_mask)
                frame = Image.alpha_composite(frame, city_fade)

        # --- Phase 4: Slow pan right while holding full composite ---
        else:
            t = (i - p3_end) / max(total_frames - p3_end - 1, 1)
            pan_px = int(t * OUT_W * 0.04)  # subtle rightward drift
            frame = terrain.copy()
            # Shift terrain slightly
            frame = Image.fromarray(
                np.roll(np.array(frame), -pan_px, axis=1)
            )
            frame = Image.alpha_composite(frame, full_glow)
            frame = Image.alpha_composite(frame, full_outline)
            frame = Image.alpha_composite(frame, fill_layer)
            if cities:
                frame = Image.alpha_composite(frame, cities_layer)

        frame.convert("RGB").save(
            str(frames_dir / f"f_{i:04d}.jpg"), "JPEG", quality=88
        )

    ok = run_ffmpeg([
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", str(frames_dir / "f_%04d.jpg"),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-r", str(FPS), str(out_path)
    ], timeout=300)

    shutil.rmtree(str(frames_dir), ignore_errors=True)
    return ok


def gen_dark_cutout(geo, out_path, duration=5.0):
    """Cinematic dark cutout — neon-glow region outline over smooth animated dark grid."""
    map_frame = _geo_frame_path(geo, "map_frame")
    pixel_rings = geo.get("pixel_rings", [])
    if not map_frame or not map_frame.exists() or not pixel_rings:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scale = 2
    bw, bh = OUT_W * scale, OUT_H * scale
    bg = Image.new("RGBA", (bw, bh), (15, 18, 22, 255))
    bd = ImageDraw.Draw(bg)
    grid_spacing = 80
    for x in range(0, bw + grid_spacing, grid_spacing):
        bd.line([(x, 0), (x, bh)], fill=(32, 38, 42, 255), width=1)
    for y in range(0, bh + grid_spacing, grid_spacing):
        bd.line([(0, y), (bw, y)], fill=(32, 38, 42, 255), width=1)
    mp = Image.open(str(map_frame)).convert("RGBA")
    mask = make_mask((OUT_W, OUT_H), pixel_rings)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=2))
    cutout = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    cutout.paste(mp, (0, 0), mask)
    tint = Image.new("RGBA", (OUT_W, OUT_H), (10, 20, 40, 140))
    cutout = Image.alpha_composite(cutout, tint)
    bg_clear = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    bg_clear.paste(cutout, (0, 0), mask)
    cutout = bg_clear
    neon_color = (0, 220, 255)
    glow_layer = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)
    for ring in pixel_rings:
        pts = [(int(round(x)), int(round(y))) for x, y in ring]
        if len(pts) >= 3:
            glow_draw.line(pts + [pts[0]], fill=(*neon_color, 120), width=12)
    glow_blurred = glow_layer.filter(ImageFilter.GaussianBlur(radius=18))
    glow_med = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    gm_draw = ImageDraw.Draw(glow_med)
    for ring in pixel_rings:
        pts = [(int(round(x)), int(round(y))) for x, y in ring]
        if len(pts) >= 3:
            gm_draw.line(pts + [pts[0]], fill=(*neon_color, 160), width=6)
    glow_blurred = Image.alpha_composite(glow_blurred, glow_med.filter(ImageFilter.GaussianBlur(radius=6)))
    core_line = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    cl_draw = ImageDraw.Draw(core_line)
    for ring in pixel_rings:
        pts = [(int(round(x)), int(round(y))) for x, y in ring]
        if len(pts) >= 3:
            cl_draw.line(pts + [pts[0]], fill=(180, 240, 255, 240), width=2)
    fg = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    fg = Image.alpha_composite(fg, cutout)
    fg = Image.alpha_composite(fg, glow_blurred)
    fg = Image.alpha_composite(fg, core_line)
    cities = geo.get("cities", [])
    if cities:
        fg = draw_cities(fg, pixel_rings, cities, geo, _font(30), _font_reg(20), color=(220, 240, 255, 230))
    paste_x = (bw - OUT_W) // 2
    paste_y = (bh - OUT_H) // 2
    bg.paste(fg, (paste_x, paste_y), fg)
    comp_path = out_path.parent / f"_dc_comp_{out_path.stem}.png"
    bg.convert("RGB").save(str(comp_path), "PNG")
    frames = int(float(duration) * FPS)
    zp = (
        f"zoompan=z='min(1.0+on/{frames}*0.12,1.12)'"
        f":x='(iw-iw/zoom)/2 + on/{frames}*(iw*0.04)'"
        f":y='(ih-ih/zoom)/2 - on/{frames}*(ih*0.02)'"
        f":d={frames}:s={OUT_W}x{OUT_H}:fps={FPS}"
    )
    ok = run_ffmpeg([
        "ffmpeg", "-y", "-loop", "1", "-i", str(comp_path),
        "-vf", zp, "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        str(out_path)
    ], timeout=180)
    comp_path.unlink(missing_ok=True)
    return ok


def gen_map_wipe(geo, out_path, duration=3.0):
    """Left-to-right wipe of map overlay onto satellite."""
    sat = _geo_frame_path(geo, "satellite_frame")
    mp = _geo_frame_path(geo, "map_frame")
    pixel_rings = geo.get("pixel_rings", [])
    if not sat or not mp or not sat.exists() or not mp.exists():
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = out_path.parent / f"_wipe_{out_path.stem}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    sat_img = Image.open(str(sat)).convert("RGBA")
    map_img = Image.open(str(mp)).convert("RGBA")
    province_mask = make_mask((OUT_W, OUT_H), pixel_rings) if pixel_rings else Image.new("L", (OUT_W, OUT_H), 255)
    province_mask = province_mask.filter(ImageFilter.GaussianBlur(radius=2))
    alpha_map = map_img.copy()
    a = alpha_map.split()[3].point(lambda p: int(p * 0.55))
    alpha_map.putalpha(a)
    masked = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    masked.paste(alpha_map, (0, 0), province_mask)
    outline_layer = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    od = ImageDraw.Draw(outline_layer)
    for ring in pixel_rings:
        pts = [(int(round(x)), int(round(y))) for x, y in ring]
        od.line(pts + [pts[0]], fill=hex_rgba("#FFFFFF", 40), width=8)
    for ring in pixel_rings:
        pts = [(int(round(x)), int(round(y))) for x, y in ring]
        od.line(pts + [pts[0]], fill=hex_rgba("#FFFFFF", 160), width=2)
    full = Image.alpha_composite(sat_img, masked)
    full = Image.alpha_composite(full, outline_layer)
    cities = geo.get("cities", [])
    if cities:
        full = draw_cities(full, pixel_rings, cities, geo, _font(28), _font_reg(18), color=(220, 240, 255, 220))
    total_frames = int(duration * FPS)
    for i in range(total_frames):
        t = i / max(total_frames - 1, 1)
        t = t * t * (3 - 2 * t)
        wipe_x = int(t * OUT_W)
        wipe_mask = Image.new("L", (OUT_W, OUT_H), 0)
        if wipe_x > 0:
            wd = ImageDraw.Draw(wipe_mask)
            feather = 30
            wd.rectangle([0, 0, max(0, wipe_x - feather), OUT_H], fill=255)
            for f in range(feather):
                alpha = int(255 * (1 - f / feather))
                xp = wipe_x - f
                if 0 <= xp < OUT_W:
                    wd.line([(xp, 0), (xp, OUT_H)], fill=alpha, width=1)
        frame = Image.composite(full, sat_img, wipe_mask)
        frame.convert("RGB").save(str(frames_dir / f"f_{i:04d}.jpg"), "JPEG", quality=88)
    ok = run_ffmpeg([
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", str(frames_dir / "f_%04d.jpg"),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-r", str(FPS), str(out_path)], timeout=120)
    shutil.rmtree(str(frames_dir), ignore_errors=True)
    return ok


def _build_outline_mask(mask: np.ndarray) -> np.ndarray:
    """Create a slightly thickened outline mask from a binary region mask."""
    outline = np.zeros_like(mask)
    outline[:-1, :] |= (mask[:-1, :] ^ mask[1:, :])
    outline[:, :-1] |= (mask[:, :-1] ^ mask[:, 1:])

    thick_outline = outline.copy()
    thick_outline[1:, :] |= outline[:-1, :]
    thick_outline[:-1, :] |= outline[1:, :]
    thick_outline[:, 1:] |= outline[:, :-1]
    thick_outline[:, :-1] |= outline[:, 1:]
    return thick_outline


def _mean_lon_deg(lons: np.ndarray) -> float:
    lons_rad = np.radians(lons)
    s = float(np.mean(np.sin(lons_rad)))
    c = float(np.mean(np.cos(lons_rad)))
    return float(np.degrees(np.arctan2(s, c)))


def _boundary_view_profile(boundary_rings: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
    all_lats = np.concatenate([r[0] for r in boundary_rings])
    all_lons = np.concatenate([r[1] for r in boundary_rings])

    min_lat = float(np.min(all_lats))
    max_lat = float(np.max(all_lats))
    min_lon = float(np.min(all_lons))
    max_lon = float(np.max(all_lons))

    lat_span = max(0.05, max_lat - min_lat)
    mid_lat = (min_lat + max_lat) / 2.0
    lon_span = max(0.05, (max_lon - min_lon) * max(0.25, math.cos(math.radians(mid_lat))))
    span = max(lat_span, lon_span)

    target_px = 960.0
    ppd = target_px / span
    zoom = int(round(math.log2((ppd * 360.0) / TILE_SIZE)))
    zoom = max(4, min(10, zoom))

    tile_deg = 360.0 / (2 ** zoom)
    desired_deg = span * 2.6
    radius = int(math.ceil((desired_deg / tile_deg - 1.0) / 2.0))
    radius = max(4, min(10, radius))

    if span <= 0.5:
        altitude = 240.0
    elif span <= 1.0:
        altitude = 280.0
    elif span <= 2.5:
        altitude = 340.0
    elif span <= 6.0:
        altitude = 420.0
    elif span <= 12.0:
        altitude = 560.0
    elif span <= 22.0:
        altitude = 760.0
    elif span <= 40.0:
        altitude = 980.0
    else:
        altitude = 1280.0

    return {
        "min_lat": min_lat,
        "max_lat": max_lat,
        "min_lon": min_lon,
        "max_lon": max_lon,
        "mid_lat": mid_lat,
        "mid_lon": _mean_lon_deg(all_lons),
        "span_deg": span,
        "atlas_zoom": float(zoom),
        "atlas_radius": float(radius),
        "altitude_km": float(altitude),
    }


def gen_outline_reveal(geo, out_path, duration=4.0):
    """Use exact generate_cinematics.py blue-highlight renderer (atlas + mask + raycast)."""
    lat = float(geo.get("lat", 0.0))
    lon = float(geo.get("lon", 0.0))
    hard_fail = False

    rings = geo.get("rings") or []
    boundary_rings = []
    for ring in rings:
        if not ring:
            continue
        ring_lats = np.array([pt[0] for pt in ring], dtype=np.float64)
        ring_lons = np.array([pt[1] for pt in ring], dtype=np.float64)
        if ring_lats.size >= 3 and ring_lons.size >= 3:
            boundary_rings.append((ring_lats, ring_lons))

    view_profile = _boundary_view_profile(boundary_rings) if boundary_rings else None
    if view_profile:
        lat = float(view_profile["mid_lat"])
        lon = float(view_profile["mid_lon"])

    gc_profiles = [
        ("0.42", "36"),
        ("0.38", "34"),
        ("0.34", "32"),
    ]
    for attempt_idx, (scale, render_fps) in enumerate(gc_profiles, start=1):
        os.environ["GA_RENDER_SCALE"] = scale
        os.environ["GA_RENDER_FPS"] = render_fps
        gc = _load_generate_cinematics_module(force_reload=True)
        if not (gc and hasattr(gc, "_fetch_atlas") and hasattr(gc, "gen_blue_highlight")):
            hard_fail = True
            continue
        try:
            print(f" [outline-gc-attempt:{attempt_idx} scale={scale} fps={render_fps}]", end="", flush=True)

            if not boundary_rings:
                region_name = str(geo.get("_region") or "").strip()
                if region_name and hasattr(gc, "_get_boundary"):
                    fetched_rings, _, fetched_bbox = gc._get_boundary(region_name)
                    if fetched_rings:
                        boundary_rings = fetched_rings
                        geo["bbox"] = geo.get("bbox") or fetched_bbox

            if not boundary_rings:
                hard_fail = True
                continue

            profile_zoom = int(view_profile["atlas_zoom"]) if view_profile else 8
            profile_radius = int(view_profile["atlas_radius"]) if view_profile else 8
            atlas_zoom = int(os.environ.get("GEO_OUTLINE_GC_ATLAS_ZOOM", str(profile_zoom)))
            atlas_radius = max(4, int(os.environ.get("GEO_OUTLINE_GC_ATLAS_RADIUS", str(profile_radius))))
            atlas_arr, bounds = gc._fetch_atlas(lat, lon, zoom=atlas_zoom, radius=atlas_radius, workers=gc.ATLAS_WORKERS)
            bbox = geo.get("bbox")
            if not bbox:
                all_lats = np.concatenate([r[0] for r in boundary_rings])
                all_lons = np.concatenate([r[1] for r in boundary_rings])
                bbox = [
                    float(np.min(all_lats)),
                    float(np.min(all_lons)),
                    float(np.max(all_lats)),
                    float(np.max(all_lons)),
                ]

            atlas_mask = gc._build_atlas_mask(bounds, atlas_arr.shape, boundary_rings, bbox)
            atlas_outline_masks = gc._build_outline_masks(atlas_mask)

            safe_name = out_path.stem
            render_gc_duration = min(float(duration), float(os.environ.get("GEO_OUTLINE_GC_MAX_SEC", "5.5")))

            ok = gc.gen_blue_highlight(
                atlas_arr,
                bounds,
                atlas_mask,
                atlas_outline_masks,
                out_path.parent,
                lat,
                lon,
                safe_name,
                duration=render_gc_duration,
                altitude_km=float(view_profile["altitude_km"]) if view_profile else 360.0,
            )
            gc_out = out_path.parent / f"{safe_name}_blue_highlight.mp4"
            min_ok_bytes = 256 * 1024
            if ok and gc_out.exists() and gc_out.stat().st_size >= min_ok_bytes:
                final_gc_out = gc_out
                if float(duration) > render_gc_duration + 0.05:
                    extended = out_path.parent / f"{safe_name}_blue_highlight_ext.mp4"
                    hold_extra = max(0.0, float(duration) - render_gc_duration)
                    ext_ok = run_ffmpeg([
                        "ffmpeg", "-y", "-i", str(gc_out),
                        "-vf", f"tpad=stop_mode=clone:stop_duration={hold_extra:.2f},fps={FPS}",
                        "-t", f"{float(duration):.2f}",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                        "-pix_fmt", "yuv420p",
                        str(extended),
                    ], timeout=180)
                    if ext_ok and extended.exists() and extended.stat().st_size > gc_out.stat().st_size:
                        gc_out.unlink(missing_ok=True)
                        final_gc_out = extended
                else:
                    final_gc_out = gc_out

                if str(final_gc_out.resolve()) != str(out_path.resolve()):
                    shutil.move(str(final_gc_out), str(out_path))
                print(" [outline-gc-blue:OK]", end="", flush=True)
                return True

            if gc_out.exists() and gc_out.stat().st_size < min_ok_bytes:
                gc_out.unlink(missing_ok=True)
                hard_fail = True
                print(" [outline-gc-blue:tiny-output]", end="", flush=True)
        except Exception as e:
            hard_fail = True
            print(f" [outline-gc-fail:{e}]", end="", flush=True)

    allow_legacy = hard_fail and str(os.environ.get("GEO_OUTLINE_LEGACY_ON_HARD_FAIL", "1")).strip().lower() not in {"0", "false", "no"}
    if not allow_legacy:
        print(" [outline-legacy-skip]", end="", flush=True)
        return False

    sat = _geo_frame_path(geo, "satellite_frame")
    pixel_rings = geo.get("pixel_rings", [])
    if not sat or not sat.exists() or not pixel_rings:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sat_img = Image.open(str(sat)).convert("RGBA")

    region_mask = np.array(make_mask((OUT_W, OUT_H), pixel_rings), dtype=np.uint8) > 0
    outline_mask = _build_outline_mask(region_mask)

    outlined = sat_img.copy()

    ol = np.zeros((OUT_H, OUT_W, 4), dtype=np.uint8)
    ol[outline_mask] = np.array([68, 170, 255, 235], dtype=np.uint8)
    ol_img = Image.fromarray(ol, mode="RGBA")

    glow = np.zeros((OUT_H, OUT_W, 4), dtype=np.uint8)
    glow[outline_mask] = np.array([68, 170, 255, 140], dtype=np.uint8)
    glow_img = Image.fromarray(glow, mode="RGBA").filter(ImageFilter.GaussianBlur(radius=4))

    fill_layer = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    fd = ImageDraw.Draw(fill_layer)
    for ring in pixel_rings:
        pts = [(int(round(x)), int(round(y))) for x, y in ring]
        if len(pts) >= 3:
            fd.polygon(pts, fill=(68, 170, 255, 26))
    outlined = Image.alpha_composite(outlined, fill_layer)
    outlined = Image.alpha_composite(outlined, glow_img)
    outlined = Image.alpha_composite(outlined, ol_img)

    plain_tmp = out_path.parent / f"_plain_{out_path.stem}.jpg"
    outlined_tmp = out_path.parent / f"_outl_{out_path.stem}.jpg"
    sat_img.convert("RGB").save(str(plain_tmp), "JPEG", quality=95)
    outlined.convert("RGB").save(str(outlined_tmp), "JPEG", quality=95)
    blend_expr = f"A*(1-T/{duration})+B*(T/{duration})"
    ok = run_ffmpeg([
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(plain_tmp), "-loop", "1", "-i", str(outlined_tmp),
        "-filter_complex",
        f"[0]scale={OUT_W}:{OUT_H}[a];[1]scale={OUT_W}:{OUT_H}[b];"
        f"[a][b]blend=all_expr='{blend_expr}'",
        "-t", str(duration), "-r", str(FPS),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        str(out_path)], timeout=120)
    plain_tmp.unlink(missing_ok=True)
    outlined_tmp.unlink(missing_ok=True)
    if ok and out_path.exists():
        print(" [outline-fallback-blue:OK]", end="", flush=True)
    return ok


def gen_region_comparison(geo: Dict, out_path: Path, duration: float) -> bool:
    """Side-by-side same-scale comparison between primary region and beat.comparison region B."""
    try:
        comp = _comparison_data(geo)
        if not comp:
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frames_dir = out_path.parent / f"_region_comparison_{out_path.stem}"
        frames_dir.mkdir(parents=True, exist_ok=True)

        pane_w, pane_h = OUT_W // 2, OUT_H
        px_per_km = _fit_px_per_km([comp["rings_a"], comp["rings_b"]], pane_w, pane_h - 300, fill=0.80)
        rings_a = _project_rings_same_scale(comp["rings_a"], px_per_km, (pane_w, pane_h),
                                            center_latlon=comp["center_a"], offset=(0, -40))
        rings_b = _translate_rings(
            _project_rings_same_scale(comp["rings_b"], px_per_km, (pane_w, pane_h),
                                      center_latlon=comp["center_b"], offset=(0, -40)),
            pane_w, 0,
        )
        zoom = _zoom_for_rings([comp["rings_a"], comp["rings_b"]])
        base_a = _build_terrain_base({"bbox": _rings_bbox(comp["rings_a"]), "rings": comp["rings_a"]}, zoom=zoom)
        base_b = _build_terrain_base({"bbox": _rings_bbox(comp["rings_b"]), "rings": comp["rings_b"]}, zoom=zoom)
        base_a = _grade_tile_dark(base_a) if base_a else _fallback_dark_terrain()
        base_b = _grade_tile_dark(base_b) if base_b else _fallback_dark_terrain()
        pane_a = base_a.resize((pane_w, pane_h), Image.LANCZOS)
        pane_b = base_b.resize((pane_w, pane_h), Image.LANCZOS)
        fill_a = _fill_layer(rings_a, CLR_CYAN, 42)
        fill_b = _fill_layer(rings_b, CLR_AMBER, 42)
        total_frames = max(1, int(float(duration) * FPS))
        label_font = _font(48)

        for i in range(total_frames):
            sec = i / FPS
            fade = _ease_smooth(sec / 0.8)
            frame = Image.new("RGBA", (OUT_W, OUT_H), (*CLR_NAVY_BG, 255))
            bg = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
            bg.alpha_composite(pane_a, (0, 0))
            bg.alpha_composite(pane_b, (pane_w, 0))
            frame = Image.alpha_composite(frame, _scale_rgba_alpha(bg, fade))
            ImageDraw.Draw(frame).line([(pane_w, 0), (pane_w, OUT_H)], fill=(255, 255, 255, int(180 * fade)), width=2)

            left_p = _ease_smooth((sec - 0.8) / 1.2)
            right_p = _ease_smooth((sec - 1.6) / 1.2)
            if left_p > 0:
                frame = _draw_glow_border(frame, _partial_rings(rings_a, left_p), CLR_CYAN, core_width=2)
            if right_p > 0:
                frame = _draw_glow_border(frame, _partial_rings(rings_b, right_p), CLR_AMBER, core_width=2)
            fill_p = _ease_smooth((sec - 2.8) / 0.8)
            if fill_p > 0:
                pulse = 0.82 + 0.18 * math.sin(sec * math.pi * 2.0)
                frame = Image.alpha_composite(frame, _scale_rgba_alpha(fill_a, fill_p * pulse))
                frame = Image.alpha_composite(frame, _scale_rgba_alpha(fill_b, fill_p * pulse))
            label_p = _ease_smooth((sec - 3.6) / 0.6)
            if label_p > 0:
                scale = 0.8 + 0.2 * label_p
                frame = _draw_scaled_center_label(frame, comp["label_a"], (pane_w // 2, OUT_H - 205),
                                                  label_font, scale, fill=(255, 255, 255, int(255 * label_p)))
                frame = _draw_scaled_center_label(frame, comp["label_b"], (pane_w + pane_w // 2, OUT_H - 205),
                                                  label_font, scale, fill=(255, 255, 255, int(255 * label_p)))
            if fade > 0.6:
                frame = _draw_scale_bar(frame, px_per_km, y=OUT_H - 92)
            frame.convert("RGB").save(frames_dir / f"frame_{i:04d}.jpg", "JPEG", quality=85)

        ok = _frames_to_mp4(frames_dir, out_path, duration=duration)
        shutil.rmtree(str(frames_dir), ignore_errors=True)
        return ok
    except Exception as e:
        print(f" [region-comparison-fail:{e}]", end="", flush=True)
        return False




def gen_multi_region_reveal(geo: Dict, out_path: Path, duration: float) -> bool:
    """Sequentially light up multiple regions on a dark regional map."""
    try:
        beat = geo.get("_beat") if isinstance(geo.get("_beat"), dict) else {}
        multi = beat.get("multi_regions") if isinstance(beat.get("multi_regions"), list) else []
        regions = []
        for item in multi[:8]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            center, rings, _bbox = fetch_boundary(name)
            if rings:
                regions.append({"name": name.upper(), "rings": rings, "center": center})
        if not regions:
            rings = geo.get("rings") or []
            if not rings:
                return False
            regions.append({"name": str(geo.get("_region") or "REGION").upper(), "rings": rings,
                            "center": (float(geo.get("lat") or 0.0), float(geo.get("lon") or 0.0))})

        out_path.parent.mkdir(parents=True, exist_ok=True)
        frames_dir = out_path.parent / f"_multi_reveal_{out_path.stem}"
        frames_dir.mkdir(parents=True, exist_ok=True)
        all_rings = [r["rings"] for r in regions]
        bounds = _rings_mercator_bounds(all_rings)
        if bounds:
            min_x, min_y, max_x, max_y = bounds
            cx_m, cy_m = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
        else:
            cx_m, cy_m = _mercator_m(float(geo.get("lat") or 0.0), float(geo.get("lon") or 0.0))
        px_per_km = _fit_px_per_km(all_rings, OUT_W, OUT_H - 250, fill=0.78)
        projected = []
        for region in regions:
            pixel = []
            for ring in region["rings"]:
                pr = []
                for lat, lon in ring:
                    x_m, y_m = _mercator_m(lat, lon)
                    pr.append((OUT_W / 2 + ((x_m - cx_m) / 1000.0) * px_per_km,
                               OUT_H / 2 - 60 - ((y_m - cy_m) / 1000.0) * px_per_km))
                pixel.append(pr)
            projected.append(pixel)

        merged_rings = [ring for region in regions for ring in region["rings"]]
        terrain = _build_terrain_base({"rings": merged_rings, "bbox": _rings_bbox(merged_rings)},
                                      zoom=_zoom_for_rings(all_rings))
        terrain = _grade_tile_dark(terrain) if terrain else _fallback_dark_terrain()
        colors = [CLR_CYAN, CLR_GREEN, CLR_AMBER, CLR_PURPLE, CLR_RED]
        label_font = _font_reg(28)
        total_frames = max(1, int(float(duration) * FPS))
        all_done_time = 0.4 * max(len(regions) - 1, 0) + 1.0

        for i in range(total_frames):
            sec = i / FPS
            frame = terrain.copy()
            for idx, region in enumerate(regions):
                color = colors[idx % len(colors)]
                start = 0.35 + idx * 0.4
                draw_p = _ease_smooth((sec - start) / 0.65)
                label_p = _ease_smooth((sec - start - 0.85) / 0.25)
                if draw_p <= 0:
                    continue
                pixel = projected[idx]
                frame = Image.alpha_composite(frame, _scale_rgba_alpha(_fill_layer(pixel, color, 52), draw_p))
                frame = _draw_glow_border(frame, _partial_rings(pixel, draw_p), color, core_width=2)
                if label_p > 0:
                    pts = [(x, y) for ring in pixel for x, y in ring]
                    if pts:
                        lx = int(sum(p[0] for p in pts) / len(pts))
                        ly = int(sum(p[1] for p in pts) / len(pts))
                        d = ImageDraw.Draw(frame)
                        name = region["name"][:22]
                        d.text((lx + 2, ly + 2), name, font=label_font, fill=(0, 0, 0, int(180 * label_p)))
                        d.text((lx, ly), name, font=label_font, fill=(255, 255, 255, int(230 * label_p)))
            zoom = 1.06 - 0.06 * _ease_smooth(max(0.0, (sec - all_done_time) / max(0.8, duration - all_done_time)))
            frame = _zoom_frame(frame, zoom)
            frame.convert("RGB").save(frames_dir / f"frame_{i:04d}.jpg", "JPEG", quality=85)

        ok = _frames_to_mp4(frames_dir, out_path, duration=duration)
        shutil.rmtree(str(frames_dir), ignore_errors=True)
        return ok
    except Exception as e:
        print(f" [multi-region-fail:{e}]", end="", flush=True)
        return False


def gen_zoom_to_region(geo: Dict, out_path: Path, duration: float) -> bool:
    """Pure tile-based exponential zoom from low zoom to the target region."""
    try:
        lat = float(geo.get("lat") or 0.0)
        lon = float(geo.get("lon") or 0.0)
        if lat == 0.0 and lon == 0.0:
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frames_dir = out_path.parent / f"_zoom_to_region_{out_path.stem}"
        frames_dir.mkdir(parents=True, exist_ok=True)

        bbox = geo.get("bbox") or _rings_bbox(geo.get("rings") or [])
        span = max(abs(float(bbox[2]) - float(bbox[0])), abs(float(bbox[3]) - float(bbox[1]))) if bbox else 2.0
        z_start = 3
        z_end = 12 if span < 0.4 else (11 if span < 1.2 else (10 if span < 4 else 9))
        bases = {}

        def build(z):
            base = _build_terrain_base({"bbox": [lat - 0.2, lon - 0.2, lat + 0.2, lon + 0.2],
                                        "rings": geo.get("rings", [])}, zoom=z)
            return z, (_grade_tile_dark(base) if base else None)

        with ThreadPoolExecutor(max_workers=3) as pool:
            for z, base in pool.map(build, range(z_start, z_end + 1)):
                if base:
                    bases[z] = base
        if not bases:
            frame_path = _geo_frame_path(geo, "satellite_frame") or _geo_frame_path(geo, "map_frame")
            if not frame_path or not frame_path.exists():
                return False
            bases[z_end] = _grade_tile_dark(Image.open(frame_path).convert("RGBA"))

        total_frames = max(1, int(float(duration) * FPS))
        for i in range(total_frames):
            t = i / max(total_frames - 1, 1)
            zf = z_start + (z_end - z_start) * (t ** 1.8)
            zi = int(math.floor(zf))
            zi = min(bases.keys(), key=lambda k: abs(k - zi))
            frac = max(0.0, min(1.0, zf - math.floor(zf)))
            frame = _zoom_frame(bases[zi], 1.0 + frac * 0.55)
            if t > 0.25:
                frame = frame.filter(ImageFilter.GaussianBlur(radius=0.8 * min(1.0, (t - 0.25) / 0.45)))
            if zf < 5.0:
                limb = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
                ld = ImageDraw.Draw(limb)
                for k in range(18):
                    a = int((1.0 - k / 18.0) * 36 * (1.0 - (zf - z_start) / 2.0))
                    ld.rectangle([k * 9, k * 12, OUT_W - k * 9, OUT_H - k * 12],
                                 outline=(220, 245, 255, max(0, a)), width=8)
                frame = Image.alpha_composite(frame, limb)
            frame.convert("RGB").save(frames_dir / f"frame_{i:04d}.jpg", "JPEG", quality=85)

        ok = _frames_to_mp4(frames_dir, out_path, duration=duration)
        shutil.rmtree(str(frames_dir), ignore_errors=True)
        return ok
    except Exception as e:
        print(f" [zoom-to-region-fail:{e}]", end="", flush=True)
        return False


def gen_size_comparison_overlay(geo: Dict, out_path: Path, duration: float) -> bool:
    """Overlay region B over region A at identical pixels-per-km scale."""
    try:
        comp = _comparison_data(geo)
        if not comp:
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frames_dir = out_path.parent / f"_size_overlay_{out_path.stem}"
        frames_dir.mkdir(parents=True, exist_ok=True)

        px_per_km = _fit_px_per_km([comp["rings_a"], comp["rings_b"]], OUT_W, OUT_H - 420, fill=0.80)
        rings_a = _project_rings_same_scale(comp["rings_a"], px_per_km, (OUT_W, OUT_H),
                                            center_latlon=comp["center_a"], offset=(0, -50))
        rings_b_center = _project_rings_same_scale(comp["rings_b"], px_per_km, (OUT_W, OUT_H),
                                                   center_latlon=comp["center_b"], offset=(0, -50))
        fill_a = _fill_layer(rings_a, CLR_CYAN, 38)
        label_font = _font(42)
        top_font = _font(52)
        total_frames = max(1, int(float(duration) * FPS))

        for i in range(total_frames):
            sec = i / FPS
            frame = Image.new("RGBA", (OUT_W, OUT_H), (*CLR_NAVY_BG, 255))
            d = ImageDraw.Draw(frame)
            for y in range(0, OUT_H, 90):
                d.line([(0, y), (OUT_W, y)], fill=(34, 50, 78, 32), width=1)
            for x in range(0, OUT_W, 90):
                d.line([(x, 0), (x, OUT_H)], fill=(34, 50, 78, 32), width=1)
            _draw_centered_text(d, (0, 70, OUT_W, 150), "SAME SCALE", top_font,
                                fill=(255, 255, 255, 235), stroke_width=3)
            a_p = _ease_smooth(sec / 1.15)
            frame = Image.alpha_composite(frame, _scale_rgba_alpha(fill_a, a_p))
            if a_p > 0:
                frame = _draw_glow_border(frame, _partial_rings(rings_a, a_p), CLR_CYAN, core_width=2)

            b_p = _ease_out_cubic((sec - 1.45) / 0.6)
            if b_p > 0:
                rings_b = _translate_rings(rings_b_center, (1.0 - b_p) * (OUT_W + 240), 0)
                frame = Image.alpha_composite(frame, _fill_layer(rings_b, CLR_AMBER, 38))
                frame = _draw_glow_border(frame, rings_b, CLR_AMBER, core_width=2)
                if b_p > 0.98:
                    mask_a = np.array(_fill_layer(rings_a, CLR_WHITE, 255).split()[3], dtype=np.uint8)
                    mask_b = np.array(_fill_layer(rings_b, CLR_WHITE, 255).split()[3], dtype=np.uint8)
                    ov = Image.new("RGBA", (OUT_W, OUT_H), (255, 255, 255, 0))
                    ov.putalpha(Image.fromarray(((mask_a > 0) & (mask_b > 0)).astype(np.uint8) * 64))
                    frame = Image.alpha_composite(frame, ov)

            d = ImageDraw.Draw(frame)
            d.text((54, OUT_H - 175), comp["label_a"], font=label_font, fill=(*CLR_CYAN, 235),
                   stroke_width=3, stroke_fill=(0, 0, 0, 220))
            bb = d.textbbox((0, 0), comp["label_b"], font=label_font, stroke_width=3)
            d.text((OUT_W - (bb[2] - bb[0]) - 54, OUT_H - 175), comp["label_b"],
                   font=label_font, fill=(*CLR_AMBER, 235), stroke_width=3, stroke_fill=(0, 0, 0, 220))
            frame.convert("RGB").save(frames_dir / f"frame_{i:04d}.jpg", "JPEG", quality=85)

        ok = _frames_to_mp4(frames_dir, out_path, duration=duration)
        shutil.rmtree(str(frames_dir), ignore_errors=True)
        return ok
    except Exception as e:
        print(f" [size-overlay-fail:{e}]", end="", flush=True)
        return False


def _polyline_lengths(points: list) -> Tuple[list, float]:
    lengths, total = [], 0.0
    for a, b in zip(points[:-1], points[1:]):
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        lengths.append(dist)
        total += dist
    return lengths, total


def _sample_polyline(points: list, u: float) -> Tuple[float, float, float]:
    if len(points) < 2:
        return (points[0][0], points[0][1], 0.0) if points else (0.0, 0.0, 0.0)
    lengths, total = _polyline_lengths(points)
    target = (u % 1.0) * max(total, 1e-6)
    acc = 0.0
    for idx, dist in enumerate(lengths):
        if acc + dist >= target:
            k = (target - acc) / max(dist, 1e-6)
            a, b = points[idx], points[idx + 1]
            return (a[0] + (b[0] - a[0]) * k,
                    a[1] + (b[1] - a[1]) * k,
                    math.atan2(b[1] - a[1], b[0] - a[0]))
        acc += dist
    a, b = points[-2], points[-1]
    return b[0], b[1], math.atan2(b[1] - a[1], b[0] - a[0])


def gen_chokepoint_flow(geo: Dict, out_path: Path, duration: float) -> bool:
    """Animated snake/arrow flow lines through a chokepoint."""
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frames_dir = out_path.parent / f"_chokepoint_flow_{out_path.stem}"
        frames_dir.mkdir(parents=True, exist_ok=True)
        rings = geo.get("rings") or []
        pixel_rings = geo.get("pixel_rings") or (_project_rings_fit(rings) if rings else [])
        terrain = _build_terrain_base(geo, zoom=8) or _build_terrain_base(geo, zoom=7)
        terrain = _grade_tile_dark(terrain) if terrain else _fallback_dark_terrain()
        border = _outline_layer(pixel_rings, color_rgb=CLR_CYAN, core_width=2) if pixel_rings else Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))

        beat = geo.get("_beat") if isinstance(geo.get("_beat"), dict) else {}
        clean_path = []
        for p in beat.get("flow_path", []) if isinstance(beat.get("flow_path"), list) else []:
            try:
                clean_path.append((float(p[0]), float(p[1])))
            except Exception:
                pass
        if len(clean_path) < 2:
            bbox = geo.get("bbox") or _rings_bbox(rings) or [float(geo.get("lat", 0)) - 1, float(geo.get("lon", 0)) - 1,
                                                              float(geo.get("lat", 0)) + 1, float(geo.get("lon", 0)) + 1]
            min_lat, min_lon, max_lat, max_lon = bbox
            clean_path = [((min_lat + max_lat) / 2.0, min_lon), ((min_lat + max_lat) / 2.0, max_lon)]

        context_rings = rings[:] if rings else [[clean_path[0], clean_path[-1]]]
        px_per_km = _fit_px_per_km([context_rings], OUT_W, OUT_H - 250, fill=0.86)
        center = _rings_center(context_rings, fallback=(float(geo.get("lat") or 0.0), float(geo.get("lon") or 0.0)))
        path_px = _project_points_same_scale(clean_path, px_per_km, (OUT_W, OUT_H), center)
        if len(path_px) < 2:
            return False
        title = str(geo.get("_region") or beat.get("flow_name") or "CHOKEPOINT").strip().upper()
        flow_label = str(beat.get("flow_label") or "").strip().upper()
        title_font, stat_font = _font(46), _font_reg(34)
        total_frames = max(1, int(float(duration) * FPS))

        for i in range(total_frames):
            sec = i / FPS
            frame = Image.alpha_composite(terrain.copy(), _scale_rgba_alpha(border, 0.40))
            flow = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
            fd = ImageDraw.Draw(flow)
            for lane in range(5):
                lane_offset = (lane - 2) * 14
                for trail in range(16):
                    x, y, angle = _sample_polyline(path_px, sec * 1.5 + lane * 0.12 - trail * 0.012)
                    nx, ny = -math.sin(angle), math.cos(angle)
                    x += nx * lane_offset
                    y += ny * lane_offset
                    a = int(210 * (1 - trail / 16.0))
                    r = max(2, int(7 - trail * 0.25))
                    fd.ellipse([x - r, y - r, x + r, y + r], fill=(255, int(220 - trail * 5), int(90 + trail * 8), a))
                hx, hy, angle = _sample_polyline(path_px, sec * 1.5 + lane * 0.12)
                nx, ny = -math.sin(angle), math.cos(angle)
                hx += nx * lane_offset
                hy += ny * lane_offset
                fd.polygon([(hx + math.cos(angle) * 18, hy + math.sin(angle) * 18),
                            (hx - math.cos(angle) * 10 + nx * 9, hy - math.sin(angle) * 10 + ny * 9),
                            (hx - math.cos(angle) * 10 - nx * 9, hy - math.sin(angle) * 10 - ny * 9)],
                           fill=(255, 244, 180, 230))
            frame = Image.alpha_composite(frame, _scale_rgba_alpha(flow.filter(ImageFilter.GaussianBlur(5)), 0.8))
            frame = Image.alpha_composite(frame, flow)
            d = ImageDraw.Draw(frame)
            _draw_centered_text(d, (80, OUT_H // 2 - 58, OUT_W - 80, OUT_H // 2 + 26), title,
                                title_font, fill=(255, 255, 255, 230), stroke_width=4)
            if flow_label:
                bb = d.textbbox((0, 0), flow_label, font=stat_font)
                box_w, box_h = bb[2] - bb[0] + 54, bb[3] - bb[1] + 34
                x0, y0 = (OUT_W - box_w) // 2, OUT_H - 280
                d.rounded_rectangle([x0, y0, x0 + box_w, y0 + box_h], radius=8, fill=(0, 0, 0, 150), outline=(255, 255, 255, 70))
                d.text((x0 + 27, y0 + 16), flow_label, font=stat_font, fill=(255, 255, 255, 235))
            frame.convert("RGB").save(frames_dir / f"frame_{i:04d}.jpg", "JPEG", quality=85)

        ok = _frames_to_mp4(frames_dir, out_path, duration=duration)
        shutil.rmtree(str(frames_dir), ignore_errors=True)
        return ok
    except Exception as e:
        print(f" [chokepoint-flow-fail:{e}]", end="", flush=True)
        return False


def _haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    r = 6371.0
    dlat = math.radians(b_lat - a_lat)
    dlon = math.radians(b_lon - a_lon)
    aa = math.sin(dlat / 2) ** 2 + math.cos(math.radians(a_lat)) * math.cos(math.radians(b_lat)) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(aa)))




def _format_stat_value(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value - round(value)) < 0.05:
        return f"{value:.0f}"
    return f"{value:.1f}".rstrip("0").rstrip(".")


def gen_stat_counter_clip(geo: Dict, out_path: Path, duration: float) -> bool:
    """Animated full-clip statistic counter over a glowing map background."""
    try:
        beat = geo.get("_beat") if isinstance(geo.get("_beat"), dict) else {}
        if beat.get("stat_value") is None:
            return gen_map_highlight(geo, out_path, duration)
        final_value = float(beat.get("stat_value"))
        top_label = str(beat.get("stat_label") or "STAT").strip().upper()
        unit_label = str(beat.get("stat_unit") or "").strip().upper()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frames_dir = out_path.parent / f"_stat_counter_{out_path.stem}"
        frames_dir.mkdir(parents=True, exist_ok=True)

        terrain = _build_terrain_base(geo, zoom=9)
        terrain = _grade_tile_dark(terrain) if terrain else _fallback_dark_terrain()
        pixel_rings = geo.get("pixel_rings") or (_project_rings_fit(geo.get("rings") or []) if geo.get("rings") else [])
        if pixel_rings:
            terrain = Image.alpha_composite(terrain, _fill_layer(pixel_rings, CLR_CYAN, 36))
            terrain = _draw_glow_border(terrain, pixel_rings, CLR_CYAN, core_width=2)

        label_font, num_font, unit_font = _font(44), _font(120), _font(36)
        total_frames = max(1, int(float(duration) * FPS))
        count_end = max(1, int(total_frames * 0.86))
        for i in range(total_frames):
            count_t = min(1.0, i / max(count_end - 1, 1))
            number = _format_stat_value(final_value * (count_t ** 0.6) if i < count_end else final_value)
            frame = terrain.copy()
            d = ImageDraw.Draw(frame)
            panel_w, panel_h = 900, 390
            x0, y0 = (OUT_W - panel_w) // 2, OUT_H // 2 - panel_h // 2
            d.rounded_rectangle([x0, y0, x0 + panel_w, y0 + panel_h], radius=16, fill=(0, 0, 0, 166),
                                outline=(255, 255, 255, 42), width=2)
            _draw_centered_text(d, (x0 + 40, y0 + 42, x0 + panel_w - 40, y0 + 112), top_label,
                                label_font, fill=(255, 255, 255, 230), stroke_width=2)
            _draw_centered_text(d, (x0 + 40, y0 + 120, x0 + panel_w - 40, y0 + 275), number,
                                num_font, fill=(255, 255, 255, 255), stroke_width=4)
            if unit_label:
                _draw_centered_text(d, (x0 + 40, y0 + 292, x0 + panel_w - 40, y0 + 356), unit_label,
                                    unit_font, fill=(230, 242, 255, 230), stroke_width=2)
            if i in (count_end, count_end + 1):
                frame = Image.alpha_composite(frame, Image.new("RGBA", (OUT_W, OUT_H), (255, 255, 255, 115)))
            frame.convert("RGB").save(frames_dir / f"frame_{i:04d}.jpg", "JPEG", quality=85)

        ok = _frames_to_mp4(frames_dir, out_path, duration=duration)
        shutil.rmtree(str(frames_dir), ignore_errors=True)
        return ok
    except Exception as e:
        print(f" [stat-counter-fail:{e}]", end="", flush=True)
        return False


def gen_terrain_map(geo, out_path):
    """Map frame with outlined boundary."""
    map_frame = _geo_frame_path(geo, "map_frame")
    pixel_rings = geo.get("pixel_rings", [])
    if not map_frame or not map_frame.exists():
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(str(map_frame)).convert("RGBA")
    if pixel_rings:
        ol = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
        od = ImageDraw.Draw(ol)
        for ring in pixel_rings:
            pts = [(int(round(x)), int(round(y))) for x, y in ring]
            od.line(pts + [pts[0]], fill=hex_rgba("#FFFFFF", 180), width=3)
        img = Image.alpha_composite(img, ol)
    cities = geo.get("cities", [])
    if cities:
        img = draw_cities(img, pixel_rings or [[(0,0),(OUT_W,0),(OUT_W,OUT_H),(0,OUT_H)]], cities, geo, _font(28), _font_reg(20))
    img.convert("RGB").save(str(out_path), "JPEG", quality=95)
    return True

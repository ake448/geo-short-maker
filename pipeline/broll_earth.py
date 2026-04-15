"""
broll_earth.py — Google Earth-style raycaster and 3D B-roll wrappers + BROLL_GENERATORS dispatch.
"""
from __future__ import annotations

import importlib.util
import io
import logging
import math
import os
import random
import shutil
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageFilter

# Silence noisy Playwright/websockets shutdown logs
logging.getLogger("websockets.protocol").setLevel(logging.CRITICAL)
logging.getLogger("websockets.connection").setLevel(logging.CRITICAL)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

from .config import ROOT, TILE_SIZE, OUT_W, OUT_H, FPS, CACHE_DIR
from .ffmpeg_utils import run_ffmpeg
from .broll import (
    gen_satellite_pan, gen_map_highlight, gen_dark_cutout, gen_map_wipe,
    gen_outline_reveal, gen_terrain_map, gen_satellite_zoom,
    gen_region_comparison, gen_multi_region_reveal,
    gen_zoom_to_region, gen_size_comparison_overlay, gen_chokepoint_flow,
    gen_stat_counter_clip,
)
from .footage_stock import gen_wikipedia_image


# ── generate_cinematics.py lazy loader ───────────────────────────────────────

_CINEMATICS_MOD = None
_CINEMATICS_TRIED = False


def _get_cinematics_mod():
    """Lazily load generate_cinematics.py from geography root (ROOT)."""
    global _CINEMATICS_MOD, _CINEMATICS_TRIED
    if _CINEMATICS_TRIED:
        return _CINEMATICS_MOD
    _CINEMATICS_TRIED = True
    gc_path = ROOT / "generate_cinematics.py"
    if not gc_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("generate_cinematics", str(gc_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CINEMATICS_MOD = mod
    except Exception as e:
        print(f"[cinematics] import failed: {e}")
    return _CINEMATICS_MOD


def gen_cinematic_fallback(geo: Dict, out_path: Path, dur: float) -> bool:
    """
    Fallback for real_city / real_geography beats when YouTube sourcing fails.
    Calls generate_cinematics.py orbit-pan or night-pan cinematic renderers,
    producing a visually distinct clip specific to the beat's lat/lon.
    """
    mod = _get_cinematics_mod()
    if not mod:
        return False
    lat = float(geo.get("lat") or 0)
    lon = float(geo.get("lon") or 0)
    if lat == 0 and lon == 0:
        return False
    try:
        atlas_arr, bounds = mod._fetch_atlas(lat, lon)
        if atlas_arr is None:
            return False
    except Exception as e:
        print(f" [cinematic-atlas-err:{e}]", end="", flush=True)
        return False

    out_dir = out_path.parent
    safe_name = out_path.stem
    # Alternate between orbit and night pan by beat id (seeded for reproducibility)
    bid = 0
    stem = out_path.stem
    if stem.startswith("beat"):
        try:
            bid = int(stem[4:6])
        except (ValueError, IndexError):
            pass
    pick = bid % 2  # 0 = orbit, 1 = night pan

    try:
        if pick == 0:
            ok = mod.gen_orbit_pan(atlas_arr, bounds, out_dir, lat, lon, safe_name, duration=dur)
            produced_name = f"{safe_name}_orbit.mp4"
        else:
            ok = mod.gen_night_zoom(atlas_arr, bounds, out_dir, lat, lon, safe_name, duration=dur)
            produced_name = f"{safe_name}_night_pan.mp4"
    except Exception as e:
        print(f" [cinematic-render-err:{e}]", end="", flush=True)
        return False

    if ok:
        produced = out_dir / produced_name
        if produced.exists() and produced.stat().st_size > 10240:
            if produced.resolve() != out_path.resolve():
                shutil.move(str(produced), str(out_path))
            return True
    return False


# ── Google Earth-Style Raycaster ──────────────────────────────────────────────

def _normalize(v):
    if v.ndim == 1:
        return v / (np.linalg.norm(v) + 1e-30)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-30)

def _lat_lon_to_ecef(lat_deg, lon_deg, alt_km=0.0):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    r = 6371.0 + alt_km
    return np.array([r * np.cos(lat) * np.cos(lon), r * np.cos(lat) * np.sin(lon), r * np.sin(lat)])

def _ecef_to_lat_lon(points):
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    return lat, lon

def _sample_atlas_bilinear(atlas_arr, bounds, lats, lons):
    lat_min, lat_max, lon_min, lon_max = bounds
    h, w = atlas_arr.shape[:2]
    u = (lons - lon_min) / (lon_max - lon_min + 1e-10)
    v = (lat_max - lats) / (lat_max - lat_min + 1e-10)
    px = np.clip(u * (w - 1), 0, w - 1)
    py = np.clip(v * (h - 1), 0, h - 1)
    px0 = np.floor(px).astype(np.int32)
    py0 = np.floor(py).astype(np.int32)
    px1 = np.clip(px0 + 1, 0, w - 1)
    py1 = np.clip(py0 + 1, 0, h - 1)
    wx1, wy1 = px - px0, py - py0
    wx0, wy0 = 1.0 - wx1, 1.0 - wy1
    c00 = atlas_arr[py0, px0].astype(np.float32)
    c10 = atlas_arr[py0, px1].astype(np.float32)
    c01 = atlas_arr[py1, px0].astype(np.float32)
    c11 = atlas_arr[py1, px1].astype(np.float32)
    return (wx0[:,:,None]*wy0[:,:,None]*c00 + wx1[:,:,None]*wy0[:,:,None]*c10 +
            wx0[:,:,None]*wy1[:,:,None]*c01 + wx1[:,:,None]*wy1[:,:,None]*c11)

def _render_earth_shot(atlas_arr, bounds, target_lat, target_lon, out_w=1080, out_h=1920,
                       altitude_km=200.0, pitch_deg=-35.0, bearing_deg=0.0):
    EARTH_R = 6371.0
    ATMO_THICKNESS = 60.0
    target_pos = _lat_lon_to_ecef(target_lat, target_lon, 0.0)
    up = _normalize(target_pos)
    north_pole = np.array([0, 0, EARTH_R])
    east = _normalize(np.cross(north_pole, up))
    north = _normalize(np.cross(up, east))
    bearing_rad = np.radians(bearing_deg)
    view_dir_flat = north * math.cos(bearing_rad) + east * math.sin(bearing_rad)
    pitch_rad = np.radians(pitch_deg)
    dist_back = altitude_km / math.tan(-pitch_rad) if pitch_rad < 0 else 0
    cam_pos = target_pos - view_dir_flat * dist_back + up * altitude_km
    c_lat, c_lon = _ecef_to_lat_lon(cam_pos[None, :])
    cam_pos = _lat_lon_to_ecef(c_lat[0], c_lon[0], altitude_km)
    forward = _normalize(target_pos - cam_pos)
    cam_up_world = _normalize(cam_pos)
    right = _normalize(np.cross(forward, cam_up_world))
    cam_up = _normalize(np.cross(right, forward))
    fov_h = np.radians(55.0)
    fov_v = 2.0 * np.arctan((out_h / out_w) * np.tan(fov_h / 2.0))
    uu, vv = np.meshgrid(np.linspace(-1, 1, out_w, dtype=np.float32),
                          np.linspace(-1, 1, out_h, dtype=np.float32))
    half_w = np.tan(fov_h / 2.0)
    half_h = np.tan(fov_v / 2.0)
    ray_dirs = (forward[None,None,:] + (uu[:,:,None]*half_w)*right[None,None,:]
                - (vv[:,:,None]*half_h)*cam_up[None,None,:])
    ray_dirs = ray_dirs / (np.linalg.norm(ray_dirs, axis=-1, keepdims=True) + 1e-30)
    C = cam_pos.astype(np.float32)
    D = ray_dirs
    dot_CD = np.einsum("k,ijk->ij", C, D)
    dot_DD = np.einsum("ijk,ijk->ij", D, D)
    dot_CC = float(np.dot(C, C))
    disc = dot_CD**2 - dot_DD * (dot_CC - EARTH_R**2)
    hit_mask = disc >= 0
    sqrt_disc = np.sqrt(np.maximum(disc, 0))
    t_hit = np.full_like(disc, np.inf)
    t_hit[hit_mask] = (-dot_CD[hit_mask] - sqrt_disc[hit_mask]) / dot_DD[hit_mask]
    hit_points = C[None,None,:] + t_hit[:,:,None] * D
    hit_lat, hit_lon = _ecef_to_lat_lon(hit_points)
    colors = _sample_atlas_bilinear(atlas_arr, bounds, hit_lat, hit_lon)
    colors_f = colors.astype(np.float32)
    gray = np.mean(colors_f, axis=-1, keepdims=True)
    colors_f = colors_f + (colors_f - gray) * 0.45
    c_norm = np.clip(colors_f / 255.0, 0, 1)
    c_norm = c_norm * c_norm * (3.0 - 2.0 * c_norm)
    colors_f = np.clip(c_norm ** 0.85 * 255.0, 0, 255)
    output = np.zeros((out_h, out_w, 3), dtype=np.float32)
    output[hit_mask] = colors_f[hit_mask]
    closest_dist_sq = dot_CC - dot_CD**2 / (dot_DD + 1e-30)
    closest_dist = np.sqrt(np.maximum(closest_dist_sq, 0))
    atmo_outer = EARTH_R + ATMO_THICKNESS * 0.6
    limb_mask = (~hit_mask) & (closest_dist < atmo_outer) & (dot_CD < 0)
    atmo_raw = np.zeros_like(closest_dist)
    if np.any(limb_mask):
        norm = (closest_dist[limb_mask] - EARTH_R) / (ATMO_THICKNESS * 0.6)
        atmo_raw[limb_mask] = np.clip(1.0 - norm, 0, 1) ** 3.0
    limb_bright = np.array([230, 245, 255], dtype=np.float32)
    limb_deep = np.array([10, 25, 60], dtype=np.float32)
    atmo_color = atmo_raw[:,:,None] * (atmo_raw[:,:,None]*limb_bright[None,None,:] + (1-atmo_raw[:,:,None])*limb_deep[None,None,:])
    output[~hit_mask] = atmo_color[~hit_mask]
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))


_EARTH_SHOT_PROFILES = [
    {"altitude": 180, "pitch": -30, "bearing_start": -10, "bearing_sweep":  25},
    {"altitude": 250, "pitch": -40, "bearing_start":  10, "bearing_sweep": -20},
    {"altitude": 150, "pitch": -25, "bearing_start":   0, "bearing_sweep":  30},
    {"altitude": 300, "pitch": -50, "bearing_start": -15, "bearing_sweep":  15},
    {"altitude": 200, "pitch": -35, "bearing_start":   5, "bearing_sweep": -25},
]


def gen_google_earth_pan(geo, out_path, duration=6.0, altitude=200.0, pitch=-35.0, bearing_sweep=20.0, _profile_seed=None):
    """Pure Python Google Earth-style panoramic sweep."""
    if _profile_seed is not None:
        import random as _rnd
        p = _EARTH_SHOT_PROFILES[_rnd.Random(_profile_seed).randrange(len(_EARTH_SHOT_PROFILES))]
        altitude, pitch, bearing_sweep = p["altitude"], p["pitch"], p["bearing_sweep"]
        bearing_start = p["bearing_start"]
    else:
        bearing_start = 0.0
    lat, lon = geo.get("lat", 0), geo.get("lon", 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = out_path.parent / f"_ge_{out_path.stem}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    zoom = 9
    radius = max(6, int(os.getenv("GOOGLE_EARTH_TILE_RADIUS", "12")))
    tile_workers = max(2, int(os.getenv("GOOGLE_EARTH_TILE_WORKERS", "8")))
    frame_pause_ms = max(0, int(os.getenv("GOOGLE_EARTH_FRAME_PAUSE_MS", "0")))
    n = 2 ** zoom
    lat_rad = math.radians(lat)
    cx = int((lon + 180.0) / 360.0 * n) % n
    cy = max(0, min(n - 1, int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)))
    print(f" [google-earth: fetching {radius*2+1}x{radius*2+1} tiles, workers={tile_workers}...]", end="", flush=True)
    def _dl(tx, ty):
        cache = CACHE_DIR / f"{zoom}_{tx}_{ty}.jpg"
        if cache.exists():
            try: return (tx, ty), Image.open(cache).convert("RGB")
            except: pass
        url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ty}/{tx}"
        for attempt in range(3):
            try:
                r = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "broll/1.0"}), timeout=10)
                cache.parent.mkdir(parents=True, exist_ok=True)
                data = r.read()
                cache.write_bytes(data)
                return (tx, ty), Image.open(io.BytesIO(data)).convert("RGB")
            except Exception:
                if attempt == 2:
                    return (tx, ty), None
                time.sleep(0.4 * (attempt + 1))
    grid = radius * 2 + 1
    atlas = Image.new("RGB", (grid * 256, grid * 256), (8, 15, 30))
    with ThreadPoolExecutor(max_workers=tile_workers) as pool:
        jobs = []
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                if 0 <= cy + dy < n: jobs.append(pool.submit(_dl, (cx+dx)%n, cy+dy))
        for f in jobs:
            (tx, ty), img = f.result()
            if img:
                dx = ((tx - cx + n // 2) % n) - n // 2
                dy = ty - cy
                atlas.paste(img.resize((256, 256)), ((dx + radius) * 256, (dy + radius) * 256))
    tl_tx, tl_ty = (cx - radius) % n, max(0, cy - radius)
    br_tx, br_ty = (cx + radius) % n, min(n - 1, cy + radius)
    def bounds(tx, ty):
        return (math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ty / n)))),
                tx / n * 360.0 - 180.0,
                math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ty + 1) / n)))),
                (tx + 1) / n * 360.0 - 180.0)
    lat_max, lon_min, _, _ = bounds(tl_tx, tl_ty)
    _, _, lat_min, lon_max = bounds(br_tx, br_ty)
    if lon_min > lon_max: lon_max += 360.0
    atlas_arr = np.array(atlas)
    bnds = (lat_min, lat_max, lon_min, lon_max)
    print(f" [rendering frames...]", end="", flush=True)
    total_frames = int(duration * FPS)
    for i in range(total_frames):
        frame_file = frames_dir / f"frame_{i:04d}.jpg"
        if frame_file.exists() and frame_file.stat().st_size > 0:
            continue
        t = i / max(1, total_frames - 1)
        ease = t * t * (3 - 2 * t)
        bearing = bearing_start + bearing_sweep * ease
        img = _render_earth_shot(atlas_arr, bnds, lat, lon, out_w=OUT_W, out_h=OUT_H,
                                 altitude_km=altitude, pitch_deg=pitch, bearing_deg=bearing)
        img = img.filter(ImageFilter.SHARPEN)
        img.save(frame_file, quality=90)
        if frame_pause_ms:
            time.sleep(frame_pause_ms / 1000.0)
        if i and i % max(1, FPS * 2) == 0:
            print(f" [frame {i}/{total_frames}]", end="", flush=True)
    ok = run_ffmpeg([
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", str(frames_dir / "frame_%04d.jpg"),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-r", str(FPS), str(out_path)
    ], timeout=180)
    shutil.rmtree(str(frames_dir), ignore_errors=True)
    return ok


# ── 3D cinematic B-roll via Mapbox/Cesium ─────────────────────────────────────
try:
    from gen import gen_3d_shot, gen_comparison_map
    _HAS_3D_RENDERER = True
except ImportError:
    _HAS_3D_RENDERER = False
    print("[WARN] gen.py not importable - 3D rendering disabled")


def gen_3d_gtazoom(geo, out_path, dur):
    """GTA-style zoom from space to ground using Cesium and Google 3D Tiles."""
    lat, lon = geo.get("lat", 0), geo.get("lon", 0)
    render_mode = str(geo.get("_beat_render_mode", "auto") or "auto")
    frames_dir = out_path.parent / f"_gtazoom_{out_path.stem}"
    geography_dir = Path(__file__).resolve().parent.parent
    frames_dir.mkdir(parents=True, exist_ok=True)
    print(f" [cinematic-gtazoom:{lat:.2f}, {lon:.2f}]", end="", flush=True)
    cmd = ["node", "render_cesium.js", str(lon), str(lat), str(dur), str(frames_dir), render_mode]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(geography_dir))
        if r.returncode != 0:
            print(f" [gtazoom-fail:{r.stderr[-100:]}]", end="", flush=True)
            return False
    except Exception as e:
        print(f" [gtazoom-fail:{e}]", end="", flush=True)
        return False
    ok = run_ffmpeg([
        "ffmpeg", "-y", "-framerate", "30",
        "-i", str(frames_dir / "frame_%04d.jpg"),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-r", "30", str(out_path)], timeout=120)
    shutil.rmtree(str(frames_dir), ignore_errors=True)
    return ok


def _render_digital_twin_shot(geo, out_path, dur, animation):
    lat, lon = geo.get("lat", 0), geo.get("lon", 0)
    render_mode = str(geo.get("_beat_render_mode", "auto") or "auto")
    geography_dir = Path(__file__).resolve().parent.parent
    frames_dir = out_path.parent / f"_twin_{out_path.stem}"
    frames_dir.mkdir(parents=True, exist_ok=True)

    shot_map = {
        "orbit": "orbit",
        "flyover": "flyover",
        "zoom_in": "zoom",
        "close_oblique": "close_oblique",
        "city_flyover": "skyline_descend",
    }
    shot = shot_map.get(str(animation or "").strip().lower(), "orbit")
    print(f" [digital-twin:{shot} {lat:.2f}, {lon:.2f}]", end="", flush=True)

    cmd = [
        "node", "render_digital_twin.js",
        "--lat", str(lat),
        "--lon", str(lon),
        "--shot", shot,
        "--duration", str(dur),
        "--fps", "30",
        "--out-dir", str(frames_dir),
        "--render-mode", render_mode,
        "--wait-ms", "120",
        "--timeout-ms", "9000",
        "--quality", "92",
    ]

    try:
        timeout_s = max(180, int(float(dur) * 90))
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, cwd=str(geography_dir))
        if r.returncode != 0:
            tail = (r.stderr or r.stdout or "")[-180:].replace("\n", " ")
            print(f" [twin-fail:{tail}]", end="", flush=True)
            shutil.rmtree(str(frames_dir), ignore_errors=True)
            return False
    except Exception as e:
        print(f" [twin-fail:{e}]", end="", flush=True)
        shutil.rmtree(str(frames_dir), ignore_errors=True)
        return False

    ok = run_ffmpeg([
        "ffmpeg", "-y", "-framerate", "30",
        "-i", str(frames_dir / "frame_%04d.jpg"),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-r", "30", str(out_path)
    ], timeout=180)
    shutil.rmtree(str(frames_dir), ignore_errors=True)
    return ok


def _gen_3d(geo, out_path, dur, animation):
    """Wrapper to call gen_3d_shot with geo data, highlight, and annotations."""
    use_digital_twin = str(os.environ.get("GEO_USE_DIGITAL_TWIN_3D", "")).strip().lower() in ("1", "true", "yes")
    if geo.get("_beat_use_digital_twin") is True:
        use_digital_twin = True

    if use_digital_twin:
        return _render_digital_twin_shot(geo, out_path, dur, animation)

    if not _HAS_3D_RENDERER:
        return False
    lat, lon = geo.get("lat", 0), geo.get("lon", 0)
    render_mode = str(geo.get("_beat_render_mode", "auto") or "auto")
    run_dir = str(out_path.parent.parent)
    stem = out_path.stem
    bid = 0
    if stem.startswith("beat"):
        try: bid = int(stem[4:6])
        except (ValueError, IndexError): pass
    result = gen_3d_shot(
        beat_id=bid, lon=lon, lat=lat, animation=animation, run_dir=run_dir,
        duration=dur, highlight=geo.get("_beat_highlight"), annotations=geo.get("_beat_annotations", []),
        render_mode=render_mode,
    )
    if result and Path(result).exists():
        if str(Path(result).resolve()) != str(out_path.resolve()):
            shutil.move(result, str(out_path))
        return True
    return False


# B-roll type -> generator function mapping
BROLL_GENERATORS = {
    "satellite_pan":    lambda geo, out, dur: gen_satellite_pan(geo, out, dur, direction="auto"),
    "map_highlight":    lambda geo, out, dur: gen_map_highlight(geo, out) or True,
    "dark_cutout":      lambda geo, out, dur: gen_dark_cutout(geo, out, dur),
    "map_wipe":         lambda geo, out, dur: gen_map_wipe(geo, out, dur),
    "outline_reveal":   lambda geo, out, dur: gen_outline_reveal(geo, out, dur),
    "terrain_map":      lambda geo, out, dur: gen_terrain_map(geo, out) or True,
    "region_comparison": lambda geo, out, dur: gen_region_comparison(geo, out, dur),
    "multi_region_reveal": lambda geo, out, dur: gen_multi_region_reveal(geo, out, dur),
    "zoom_to_region":   lambda geo, out, dur: gen_zoom_to_region(geo, out, dur),
    "size_comparison_overlay": lambda geo, out, dur: gen_size_comparison_overlay(geo, out, dur),
    "chokepoint_flow":  lambda geo, out, dur: gen_chokepoint_flow(geo, out, dur),
    "stat_counter_clip": lambda geo, out, dur: gen_stat_counter_clip(geo, out, dur),
    "google_earth_pan": lambda geo, out, dur: gen_google_earth_pan(
        geo, out, dur,
        _profile_seed=int(abs(geo.get("lat", 0) * 1000 + geo.get("lon", 0) * 100 + dur * 7)) % 5,
    ),
    "3d_orbit":         lambda geo, out, dur: _gen_3d(geo, out, dur, "orbit"),
    "3d_flyover":       lambda geo, out, dur: _gen_3d(geo, out, dur, "flyover"),
    "3d_zoom":          lambda geo, out, dur: _gen_3d(geo, out, dur, "zoom_in"),
    "3d_close_oblique": lambda geo, out, dur: _gen_3d(geo, out, dur, "close_oblique"),
    # Cinematic fallback for real_city / real_geography beats
    "cinematic_orbit":  lambda geo, out, dur: gen_cinematic_fallback(geo, out, dur),
}

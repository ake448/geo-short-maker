"""
geodata.py — Stage 2: Tile math, boundary fetching, satellite/map tile downloads.
"""
from __future__ import annotations

import io
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .config import (
    _SSL_CTX, TILE_SIZE, OUT_W, OUT_H, USER_AGENT,
    ESRI_URL, OSM_URL, CACHE_DIR,
)


TILE_REQUIRED_BROLL = {
    "satellite_pan", "map_wipe", "map_highlight", "terrain_map", "dark_cutout",
    "outline_reveal", "satellite_zoom",
}

BOUNDARY_REQUIRED_BROLL = {
    "map_highlight", "dark_cutout", "outline_reveal",
}


def _is_connectivity_error(message: Optional[str]) -> bool:
    text = str(message or "").lower()
    if not text:
        return False
    needles = (
        "winerror 10061",
        "winerror 11001",
        "winerror 10060",
        "timed out",
        "name or service not known",
        "temporary failure in name resolution",
        "connection refused",
        "actively refused",
        "nodename nor servname provided",
    )
    return any(needle in text for needle in needles)


def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return max(0, min(x, n - 1)), max(0, min(y, n - 1))


def lat_lon_to_pixel(lat, lon, center_lat, center_lon, zoom, cols, rows):
    n = 2 ** zoom
    x_tile = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y_tile = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    cx, cy = lat_lon_to_tile(center_lat, center_lon, zoom)
    px = (x_tile - (cx - cols // 2)) * TILE_SIZE
    py = (y_tile - (cy - rows // 2)) * TILE_SIZE
    return px, py


def composite_to_frame(px, py, comp_w, comp_h):
    ratio = 9.0 / 16.0
    if comp_w / comp_h > ratio:
        new_w = int(comp_h * ratio)
        left = (comp_w - new_w) / 2.0
        return (px - left) * (OUT_W / new_w), py * (OUT_H / comp_h)
    else:
        new_h = int(comp_w / ratio)
        top = (comp_h - new_h) / 2.0
        return px * (OUT_W / comp_w), (py - top) * (OUT_H / new_h)


def download_tile(url, retries=3):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=20, context=_SSL_CTX) as resp:
                return resp.read(), None
        except Exception as exc:
            last_error = exc
            if _is_connectivity_error(exc):
                break
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
    return None, str(last_error) if last_error else "unknown error"


def download_composite(lat, lon, zoom, cols, rows, tile_url, label=""):
    cx, cy = lat_lon_to_tile(lat, lon, zoom)
    n = 2 ** zoom
    hc, hr = cols // 2, rows // 2
    comp = Image.new("RGB", (cols * TILE_SIZE, rows * TILE_SIZE), (20, 30, 50))
    total = cols * rows
    ok = 0
    attempted = 0
    initial_failures = 0
    last_error = None
    for dy in range(-hr, hr):
        for dx in range(-hc, hc):
            tx = (cx + dx) % n
            ty = cy + dy
            if ty < 0 or ty >= n:
                continue
            attempted += 1
            data, error = download_tile(tile_url.format(z=zoom, x=tx, y=ty))
            if error:
                last_error = error
                if ok == 0:
                    initial_failures += 1
            if data:
                try:
                    comp.paste(Image.open(io.BytesIO(data)),
                               ((dx + hc) * TILE_SIZE, (dy + hr) * TILE_SIZE))
                    ok += 1
                except Exception:
                    pass
            if ok == 0 and (initial_failures >= 12 or _is_connectivity_error(last_error)):
                print(f"\n    [WARN] {label or 'tiles'} download aborted early: "
                      f"0/{attempted} tiles succeeded. Last error: {last_error}")
                return None
            if ok % 20 == 0:
                print(f"\r    [{label}] z={zoom}: {ok}/{total} ({ok*100//max(total,1)}%)", end="", flush=True)
            time.sleep(0.05)
    print(f"\r    [{label}] z={zoom}: {ok}/{total} (100%)   ")
    return comp if ok > total * 0.3 else None


def crop_916(img, out_path=None):
    w, h = img.size
    ratio = 9.0 / 16.0
    if w / h > ratio:
        nw = int(h * ratio)
        left = (w - nw) // 2
        cropped = img.crop((left, 0, left + nw, h))
    else:
        nh = int(w / ratio)
        top = (h - nh) // 2
        cropped = img.crop((0, top, w, top + nh))
    resized = cropped.resize((OUT_W, OUT_H), Image.LANCZOS)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        resized.save(str(out_path), "JPEG", quality=95)
    return resized


def fetch_boundary(query):
    params = urllib.parse.urlencode({
        "q": query, "format": "jsonv2", "limit": 5,
        "polygon_geojson": 1, "polygon_threshold": 0.0,
    })
    url = f"https://nominatim.openstreetmap.org/search?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"    [WARN] Nominatim: {e}")
        return None, None, None
    if not data:
        return None, None, None

    top = data[0]
    lat = float(top["lat"])
    lon = float(top["lon"])
    bbox = top.get("boundingbox", [])

    best_rings, best_score = None, 0
    for r in data:
        geo = r.get("geojson")
        if not geo or geo.get("type") not in ("Polygon", "MultiPolygon"):
            continue
        rings = _extract_rings(geo)
        pts = sum(len(ring) for ring in rings)
        score = pts + (10000 if r.get("osm_type") == "relation" else 0)
        if score > best_score:
            best_rings, best_score = rings, score

    return (lat, lon), best_rings, bbox


def _extract_rings(geo):
    rings = []
    coords = geo.get("coordinates", [])
    if geo["type"] == "Polygon":
        for ring in coords:
            rings.append([(c[1], c[0]) for c in ring])
    elif geo["type"] == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                rings.append([(c[1], c[0]) for c in ring])
    return rings


def _clean_geo_query(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.split("|", 1)[0].strip()
    text = text.split("—", 1)[0].strip()
    if "(" in text:
        text = text.split("(", 1)[0].strip()
    return text


def _rings_bbox(rings: List[List[Tuple[float, float]]]) -> Optional[Tuple[float, float, float, float]]:
    if not rings:
        return None
    lats = [pt[0] for ring in rings for pt in ring]
    lons = [pt[1] for ring in rings for pt in ring]
    if not lats or not lons:
        return None
    return (min(lats), max(lats), min(lons), max(lons))


def _bbox_contains(lat: float, lon: float, bbox_like: Tuple[float, float, float, float]) -> bool:
    min_lat, max_lat, min_lon, max_lon = bbox_like
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def script_requires_tiles(script: Dict[str, Any]) -> bool:
    beats = script.get("beats", []) if isinstance(script, dict) else []
    for beat in beats if isinstance(beats, list) else []:
        btype = str((beat or {}).get("broll_type", "")).strip().lower()
        if btype in TILE_REQUIRED_BROLL:
            return True
    return False


def script_requires_boundary(script: Dict[str, Any]) -> bool:
    beats = script.get("beats", []) if isinstance(script, dict) else []
    for beat in beats if isinstance(beats, list) else []:
        btype = str((beat or {}).get("broll_type", "")).strip().lower()
        if btype in BOUNDARY_REQUIRED_BROLL:
            return True
    return False


def _bbox_from_cities(cities: List[Dict[str, Any]]) -> Optional[Tuple[float, float, float, float]]:
    pts: List[Tuple[float, float]] = []
    for city in cities or []:
        try:
            lat = float(city.get("lat"))
            lon = float(city.get("lon"))
            pts.append((lat, lon))
        except (TypeError, ValueError):
            continue
    if not pts:
        return None

    lats = [p[0] for p in pts]
    lons = [p[1] for p in pts]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    lat_span = max(0.08, max_lat - min_lat)
    mid_lat = (min_lat + max_lat) / 2.0
    lon_span = max(0.08, (max_lon - min_lon) * max(0.25, math.cos(math.radians(mid_lat))))
    lat_pad = max(0.04, lat_span * 0.55)
    lon_pad = max(0.04, lon_span * 0.55)
    return (
        min_lat - lat_pad,
        max_lat + lat_pad,
        min_lon - lon_pad,
        max_lon + lon_pad,
    )


def _pick_best_boundary_query(region: str, script: Dict[str, Any]) -> Tuple[str, Optional[Tuple[float, float]], Optional[List[List[Tuple[float, float]]]], Optional[List[str]]]:
    cities = script.get("cities", []) if isinstance(script, dict) else []
    primary_city = cities[0] if cities and isinstance(cities[0], dict) else {}
    city_lat = primary_city.get("lat")
    city_lon = primary_city.get("lon")
    try:
        city_lat = float(city_lat) if city_lat is not None else None
        city_lon = float(city_lon) if city_lon is not None else None
    except (TypeError, ValueError):
        city_lat, city_lon = None, None

    # Single words that are not place names (Gemini sometimes extracts these from prompt titles)
    _non_geo_words = {
        "nobody", "everybody", "everyone", "someone", "something", "nothing",
        "anyone", "anything", "nowhere", "everywhere", "somebody", "people",
        "person", "wants", "lives", "moved", "left", "abandoned", "forgotten",
        "lost", "border", "borders", "boundary", "line", "conflict", "war",
        "history", "mystery", "secret", "reason", "truth", "story", "problem",
        "crisis", "most", "least", "worst", "best", "biggest", "smallest",
        "poorest", "richest", "deadliest", "dangerous", "safest",
    }

    candidates: List[str] = []
    for raw in (
        script.get("geodata_query") if isinstance(script, dict) else "",
        script.get("location_focus") if isinstance(script, dict) else "",
        region,
        script.get("subject") if isinstance(script, dict) else "",
        script.get("region") if isinstance(script, dict) else "",
    ):
        cleaned = _clean_geo_query(raw)
        if not cleaned:
            continue
        # Skip single common English words that aren't place names
        if len(cleaned.split()) == 1 and cleaned.lower() in _non_geo_words:
            continue
        if cleaned.lower() not in {c.lower() for c in candidates}:
            candidates.append(cleaned)

    if not candidates:
        return region, None, None, None

    best = None
    for candidate in candidates[:4]:
        center, rings, bbox = fetch_boundary(candidate)
        if not center:
            continue
        score = 0.0
        if rings:
            score += 120.0
            pts = float(sum(len(r) for r in rings))
            score += min(60.0, pts / 80.0)

        bbox_like = _rings_bbox(rings) if rings else None
        if city_lat is not None and city_lon is not None and bbox_like:
            if _bbox_contains(city_lat, city_lon, bbox_like):
                score += 180.0
            c_lat, c_lon = float(center[0]), float(center[1])
            score -= min(120.0, abs(c_lat - city_lat) + abs(c_lon - city_lon))

        if best is None or score > best[0]:
            best = (score, candidate, center, rings, bbox)

    if best is None:
        return region, None, None, None

    _, query, center, rings, bbox = best
    return query, center, rings, bbox


def rings_to_pixels(rings, center_lat, center_lon, zoom, cols, rows):
    cw, ch = cols * TILE_SIZE, rows * TILE_SIZE
    out = []
    for ring in rings:
        pr = []
        for lat, lon in ring:
            cpx, cpy = lat_lon_to_pixel(lat, lon, center_lat, center_lon, zoom, cols, rows)
            fpx, fpy = composite_to_frame(cpx, cpy, cw, ch)
            pr.append((fpx, fpy))
        out.append(pr)
    return out


def choose_zoom_grid(bbox):
    """Pick zoom level and grid size so the region fills ~60-70% of the 9:16 frame."""
    if bbox and len(bbox) >= 4:
        lat_span = abs(float(bbox[1]) - float(bbox[0]))
        lon_span = abs(float(bbox[3]) - float(bbox[2]))
    else:
        lat_span = 5.0
        lon_span = 5.0

    lat_span = max(lat_span, 0.01)
    lon_span = max(lon_span, 0.01)

    FILL_FRAC = 0.45
    target_w_px = OUT_W / FILL_FRAC
    target_h_px = OUT_H / FILL_FRAC

    ppd_lon = target_w_px / lon_span
    ppd_lat = target_h_px / lat_span
    ppd_needed = max(ppd_lon, ppd_lat)

    z_float = math.log2(ppd_needed * 360.0 / TILE_SIZE)
    z = int(round(z_float))
    z = max(3, min(14, z))

    ppd = (2 ** z) * TILE_SIZE / 360.0
    region_h_px = lat_span * ppd
    needed_h = region_h_px / FILL_FRAC
    rows = max(8, int(math.ceil(needed_h / TILE_SIZE)) + 1)
    cols = max(6, int(math.ceil(rows * 9.0 / 16.0)) + 1)

    MAX_TILES = 500
    while cols * rows > MAX_TILES and z > 3:
        z -= 1
        ppd = (2 ** z) * TILE_SIZE / 360.0
        region_h_px = lat_span * ppd
        needed_h = region_h_px / FILL_FRAC
        rows = max(8, int(math.ceil(needed_h / TILE_SIZE)) + 1)
        cols = max(6, int(math.ceil(rows * 9.0 / 16.0)) + 1)

    return z, cols, rows


def gather_geo_data(region: str, run_dir: Path, script: Dict) -> Dict[str, Any]:
    """Stage 2: Fetch boundary, download satellite + map tiles."""
    print("\n[S2] Gathering geodata...")

    geo = {"region": region}
    cities = script.get("cities", [])
    need_tiles = script_requires_tiles(script)
    need_boundary = script_requires_boundary(script)

    print("  Resolving center/span...")
    time.sleep(1.2)
    boundary_query, center, rings, bbox = _pick_best_boundary_query(region, script)
    if boundary_query and boundary_query.strip().lower() != region.strip().lower():
        print(f"  Boundary query resolved to: {boundary_query}")

    if not rings and cities and need_boundary:
        # Fallback: if region boundary fails, try the primary city boundary.
        primary_city_name = cities[0].get("name", "")
        if primary_city_name:
            print(f"  [WARN] Region boundary failed, trying primary city: {primary_city_name}")
            c2, r2, b2 = fetch_boundary(primary_city_name)
            if r2:
                center, rings, bbox = c2, r2, b2

    city_bbox = _bbox_from_cities(cities)
    if city_bbox and (not bbox or len(bbox) < 4):
        bbox = [city_bbox[0], city_bbox[1], city_bbox[2], city_bbox[3]]

    if center:
        geo["lat"], geo["lon"] = center[0], center[1]
        print(f"  Center: {geo['lat']:.4f}, {geo['lon']:.4f}")
    else:
        primary_city = cities[0] if cities else {}
        try:
            geo["lat"] = float(primary_city.get("lat"))
            geo["lon"] = float(primary_city.get("lon"))
            print(f"  [WARN] Using primary city as center: {primary_city.get('name')} ({geo['lat']:.4f}, {geo['lon']:.4f})")
        except (TypeError, ValueError):
            print("  [FAIL] Could not geocode region. Falling back to 0,0")
            geo["lat"], geo["lon"] = 0.0, 0.0

    geo["bbox"] = [float(bbox[0]), float(bbox[2]), float(bbox[1]), float(bbox[3])] if bbox and len(bbox) >= 4 else []
    geo["rings"] = rings if (rings and need_boundary) else []

    if cities:
        geo["cities"] = cities
        print(f"  Cities: {len(cities)} from script")
        # If there is a primary city, use its coordinates as the viewer focal point
        # This prevents state-centroid issues (e.g. "Illinois" centering on rural cornfields)
        primary_city = cities[0]
        if primary_city.get("lat") and primary_city.get("lon"):
            geo["lat"] = float(primary_city["lat"])
            geo["lon"] = float(primary_city["lon"])
            print(f"  Focal point overridden to primary city: {primary_city.get('name')} "
                  f"({geo['lat']:.4f}, {geo['lon']:.4f})")
    else:
        geo["cities"] = []

    zoom, cols, rows = choose_zoom_grid(bbox)
    geo["zoom"] = zoom
    geo["cols"] = cols
    geo["rows"] = rows
    print(f"  Auto zoom: z={zoom}, grid={cols}x{rows}")

    if need_boundary and geo["rings"]:
        filtered_rings = []
        for ring in geo["rings"]:
            lats = [pt[0] for pt in ring]
            lons = [pt[1] for pt in ring]
            lat_span = max(lats) - min(lats)
            lon_span = max(lons) - min(lons)
            if lat_span > 0.02 or lon_span > 0.02:
                filtered_rings.append(ring)
        if not filtered_rings:
            filtered_rings = geo["rings"]

        total_pts = sum(len(r) for r in filtered_rings)
        print(f"  Boundary: {len(filtered_rings)} ring(s) (from {len(geo['rings'])} raw), {total_pts} pts")
        geo["rings"] = filtered_rings
        geo["pixel_rings"] = rings_to_pixels(filtered_rings, geo["lat"], geo["lon"], zoom, cols, rows)
        geo["pixel_rings"] = [r for r in geo["pixel_rings"]
                              if any(0 <= x <= OUT_W and 0 <= y <= OUT_H for x, y in r)]
    else:
        geo["rings"] = []
        geo["pixel_rings"] = []

    cities = script.get("cities", [])
    if cities:
        geo["cities"] = cities
        print(f"  Cities: {len(cities)} from script")
        # If there is a primary city, use its coordinates as the viewer focal point
        # This prevents state-centroid issues (e.g. "Illinois" centering on rural cornfields)
        primary_city = cities[0] if cities else None
        if primary_city and primary_city.get("lat") and primary_city.get("lon"):
            geo["lat"] = float(primary_city["lat"])
            geo["lon"] = float(primary_city["lon"])
            print(f"  Focal point overridden to primary city: {primary_city.get('name')} "
                  f"({geo['lat']:.4f}, {geo['lon']:.4f})")
    else:
        geo["cities"] = []

    if not need_tiles:
        geo["satellite_frame"] = ""
        geo["map_frame"] = ""
        geo["satellite_wide"] = ""
        geo_save = dict(geo)
        geo_save["boundary_rings_count"] = len(geo.get("rings", []))
        geo_save["boundary_pts_total"] = sum(len(r) for r in geo.get("rings", []))
        (run_dir / "s2_geodata.json").write_text(json.dumps(geo_save, indent=2), encoding="utf-8")
        print("  [SKIP] Tiles not needed for this beat mix")
        return geo

    tiles_dir = run_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    tile_failures: List[str] = []

    # Satellite
    sat_path = tiles_dir / f"satellite_z{zoom}.jpg"
    if not sat_path.exists():
        print(f"  Downloading satellite tiles z={zoom}...")
        comp = download_composite(geo["lat"], geo["lon"], zoom, cols, rows, ESRI_URL, "sat")
        if comp:
            crop_916(comp, sat_path)
            print(f"  -> {sat_path.name}")
        else:
            tile_failures.append(f"satellite z={zoom}")
    else:
        print(f"  [reuse] {sat_path.name}")
    geo["satellite_frame"] = str(sat_path)

    # Map
    map_path = tiles_dir / f"map_z{zoom}.jpg"
    if not map_path.exists():
        print(f"  Downloading OSM map tiles z={zoom}...")
        comp = download_composite(geo["lat"], geo["lon"], zoom, cols, rows, OSM_URL, "map")
        if comp:
            crop_916(comp, map_path)
            print(f"  -> {map_path.name}")
        else:
            tile_failures.append(f"osm map z={zoom}")
    else:
        print(f"  [reuse] {map_path.name}")
    geo["map_frame"] = str(map_path)

    # Wider satellite for zoom-in (z-2)
    zoom_wide = max(3, zoom - 2)
    wide_cols = max(6, cols - 2)
    wide_rows = max(8, int(wide_cols * 16 / 9) + 2)
    wide_path = tiles_dir / f"satellite_z{zoom_wide}_wide.jpg"
    if not wide_path.exists():
        print(f"  Downloading wide satellite z={zoom_wide}...")
        comp = download_composite(geo["lat"], geo["lon"], zoom_wide, wide_cols, wide_rows, ESRI_URL, "sat-wide")
        if comp:
            crop_916(comp, wide_path)
        else:
            tile_failures.append(f"wide satellite z={zoom_wide}")
    else:
        print(f"  [reuse] {wide_path.name}")
    geo["satellite_wide"] = str(wide_path)

    missing_outputs = [path for path in (sat_path, map_path, wide_path) if not path.exists()]
    if missing_outputs:
        failed = ", ".join(tile_failures) if tile_failures else "tile downloads"
        missing_names = ", ".join(path.name for path in missing_outputs)
        raise RuntimeError(
            f"Geodata tile fetch failed ({failed}); missing outputs: {missing_names}. "
            "Check network access to Nominatim/OSM/Esri tile services."
        )

    # Save geodata
    geo_save = dict(geo)
    geo_save["boundary_rings_count"] = len(geo.get("rings", []))
    geo_save["boundary_pts_total"] = sum(len(r) for r in geo.get("rings", []))
    (run_dir / "s2_geodata.json").write_text(json.dumps(geo_save, indent=2), encoding="utf-8")

    return geo

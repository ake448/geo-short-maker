#!/usr/bin/env python3
"""
geo_short_maker.py — End-to-end Geography Short pipeline.

Usage:
  python geo_short_maker.py --region "Nova Scotia, Canada"
  python geo_short_maker.py --region "Utah, United States" --style terrain

Flow:
  Stage 1: Gemini generates narration script + beat map (structured JSON)
  Stage 2: B-roll router maps each beat to a visual asset type
  Stage 3: Asset generator creates all visual clips
  Stage 4: Caption renderer burns styled text onto each clip
  Stage 5: Final assembly stitches clips into a single 9:16 Short

B-roll types available (auto-generated, no manual input):
  map_highlight  — Political map with region highlighted in color
  satellite_zoom — Zoom into ESRI satellite imagery
  satellite_pan  — Left/right pan across satellite terrain
  dark_cutout    — Region shape on dark grid with city labels
  map_wipe       — L-to-R map overlay reveal on satellite
  outline_reveal — Satellite with boundary polygon fading in
  terrain_map    — Terrain/relief map centered on region
  real_city        = real footage of city streets / downtown / skyline (PREFERRED)
    real_people    = real footage of people walking / daily life (PREFERRED)
    real_geography = real footage of nature/terrain/coast/mountains (PREFERRED)
    3d_orbit       = cinematic 3D satellite orbit around the region (Mapbox/Cesium)
    3d_flyover     = cinematic 3D flyover at an oblique angle
    3d_zoom        = zooming from space down to the region in 3D
    3d_curvature   = ultra-wide view showing Earth's curvature
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import shutil
import ssl
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


ROOT = Path(__file__).resolve().parent


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    try:
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass


_load_dotenv(ROOT / ".env")

# ─── SSL ──────────────────────────────────────────────────────────────────────
try:
    _SSL_CTX = ssl.create_default_context()
except Exception:
    _SSL_CTX = ssl._create_unverified_context()
try:
    urllib.request.urlopen(
        urllib.request.Request("https://server.arcgisonline.com", method="HEAD"),
        timeout=5, context=_SSL_CTX)
except Exception:
    _SSL_CTX = ssl._create_unverified_context()

# ─── Constants ────────────────────────────────────────────────────────────────
RUNS_DIR = ROOT / "runs"
CACHE_DIR = ROOT / "cache" / "tiles"
TILE_SIZE = 256
OUT_W, OUT_H = 1080, 1920
FPS = 30
USER_AGENT = "broll-api-geo-short/1.0"

ESRI_URL  = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
OSM_URL   = "https://cartodb-basemaps-a.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png"

# Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash").strip() or "gemini-2.0-flash"

# Background music
MUSIC_DIR = Path(os.environ.get("GEO_MUSIC_DIR", str(ROOT / "music"))).resolve()
MUSIC_VOL_DB = -18  # duck music well below voice
FONTS_DIR = Path(os.environ.get("GEO_FONTS_DIR", str(ROOT.parent / "fonts"))).resolve()
CAPTION_FONT_NAME = os.environ.get("CAPTION_FONT_NAME", "Montserrat Bold").strip() or "Montserrat Bold"

# Real footage sourcing (YouTube)
YTDLP_PATH = os.environ.get("YTDLP_PATH", "yt-dlp").strip() or "yt-dlp"
COOKIES_YT = os.environ.get("YOUTUBE_COOKIES_PATH", "").strip()
REAL_MIN_SRC_SEC = 12
REAL_MAX_SRC_SEC = 240

# Blur-bar background filter for fitting non-9:16 footage into 9:16 frame.
# Background: scale to fill + crop + heavy blur.  Foreground: scale to fit (preserve AR).
# Overlay sharp foreground centered on blurred background.
BLUR_BG_FILTER = (
    f"split[bg_in][fg_in];"
    f"[bg_in]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
    f"crop={OUT_W}:{OUT_H},boxblur=25:5[bg];"
    f"[fg_in]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2:color=black@0[fg];"
    f"[bg][fg]overlay=0:0,format=yuv420p"
)




# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: GEMINI SCRIPT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def call_gemini(prompt: str, temperature: float = 0.75) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY. Set it in geography/.env or environment.")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{urllib.parse.quote(GEMINI_MODEL)}:generateContent"
        f"?key={urllib.parse.quote(GEMINI_API_KEY)}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=90, context=_SSL_CTX) as res:
            raw = json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Gemini HTTP {err.code}: {detail[:500]}") from err

    candidates = raw.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini returned no candidates.")
    parts = candidates[0].get("content", {}).get("parts", [])
    combined = "\n".join(p.get("text", "") for p in parts if "text" in p).strip()
    if not combined:
        raise RuntimeError("Gemini response had no text.")
    return combined


def extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        first, last = text.find("{"), text.rfind("}")
        if first != -1 and last > first:
            return json.loads(text[first:last+1])
        first, last = text.find("["), text.rfind("]")
        if first != -1 and last > first:
            return json.loads(text[first:last+1])
        raise


SCRIPT_PROMPT = textwrap.dedent("""\
You are writing a YouTube Shorts script about the geography of a specific region.
The video style is like @urban_atlas — punchy, visual, education-meets-entertainment.

Region: {region}

Rules:
- Total runtime: 80-95 seconds (target about 90 seconds)
- 12-18 beats (each beat = one narration line + one visual clip)
- Beat pacing target: each beat should be 2.5-6 seconds
- Beat 1 MUST be a curiosity-driven hook (question or shocking statement)
- Each narration line: 5-15 words MAX
- Narration should sound like a confident narrator, not a textbook. It should pair well with an Attenborough-style voice and dramatic background music.
- Every fact must be real and verifiable.
- Return ONLY valid JSON. Escape all quotes properly. No markdown outside the JSON block.

REQUIRED BEAT STRUCTURE (follow this closely):
  Beat 1 — HOOK: Open with a dramatic map visual (e.g., 3d_orbit, satellite_pan, or dark_cutout).
            Ask a provocative question or state a shocking fact.
  Beats 2-4 — CONTEXT: Use map/satellite showing the region/highlight (e.g., 3d_zoom, map_highlight, outline_reveal).
  Beats 5-16 — DEEP DIVE: The MAJORITY of beats MUST use high-quality real footage (real_geography, real_city, real_people).
               These must be cinematic, aesthetically pleasing shots (e.g., drone flights over terrain, clean cityscapes,
               people walking to show culture). 
               IMPORTANT: Specify in the `visual_note` that the footage must be "CLEAN with NO text, NO logos, and NO overlays."
               Use 3d_* or map_* b-roll sparingly here for geographical context.
  Beat Last — CLOSING: Use a cinematic real_geography shot or a wide 3d_curvature view. Callback to the hook if possible.

Available broll_types:
  dark_cutout      = region shape on animated dark grid with city labels (openers/closers ONLY)
  satellite_pan    = slow pan across terrain (use sparingly)
  map_highlight    = political map with region highlighted in color
  map_wipe         = map overlay wiping L-to-R over satellite
  outline_reveal   = boundary outline appearing on satellite
  terrain_map      = terrain/relief map centered on region
  real_city        = real footage of city streets / downtown / skyline (HIGHLY PREFERRED)
  real_people      = real footage of the region functioning: cars driving, ports working, street life, daily activities (HIGHLY PREFERRED)
  real_geography   = real footage of nature/terrain/coast/mountains (HIGHLY PREFERRED)
  real_concept     = stock footage of a CONCEPT related to the topic, NOT a location
                     (e.g. thermometer in extreme heat, ice melting, crowded subway, volcanic lava)
                     Use when narration discusses WHY something happens or describes an effect.
                     visual_note MUST describe the CONCEPT, not a place.
  wikipedia_image  = an image overlay from a Wikipedia article.
                     visual_note MUST be the exact Wikipedia article title, e.g. "Dnieper River"
  google_earth_pan = sweeping cinematic curved view using satellite imagery (HIGHLY PREFERRED map type)
  3d_orbit         = cinematic 3D satellite orbit around the region (BEST for establishing shots)
  3d_flyover       = cinematic 3D flyover at an oblique angle
  3d_zoom          = zooming from space down to the region in 3D
  3d_curvature     = ultra-wide view showing Earth's curvature
  3d_close_oblique = dramatic close-up tilted view looking across terrain (like Google Earth)
  comparison_map   = animated comparison overlay: highlights a sub-region vs the whole country/region
                     with data on screen. USE when narration compares sizes, populations, or areas.
                     Requires a "comparison" object in the beat (see schema below).


VISUAL VARIETY RULES:
- The MAJORITY of beats MUST use real_* broll types.
- Ensure `visual_note` for real_* beats includes "clean, no text overlays, no logos, high quality cinematic".
- Do not use the exact same `broll_type` (except real_* variants) two beats in a row.

Return STRICT JSON:
{{
  "region": "string",
  "title": "string (catchy, 3-6 words)",
  "total_duration_sec": number,
  "beats": [
    {{
      "beat_id": 1,
      "script_type": "HOOK|ORIENTATION|PRIMARY_FACT|DETAIL|CUT",
      "narration": "string (the spoken line)",
      "broll_type": "string (from the list above)",
      "duration_sec": number (2.5-6.0 seconds),
      "visual_note": "string (SPECIFIC description of what viewer sees — include 'clean, no text overlays' for real footage)",
      "youtube_query": "string (optional specific YouTube search, e.g. 'Kherson city drone 4k'. Use only if a very exact clip is needed.)",
      "caption_text": "string (key phrase, ALL CAPS, 2-5 words — visual emphasis only)"
    }}
  ],
  "cities": [
    {{"name": "string", "lat": number, "lon": number, "rank": 1}}
  ],
  "youtube_metadata": {{
    "title": "string (optimized for YouTube Shorts SEO, max 50 chars)",
    "description": "string (Extremely minimal description: JUST 1-2 punchy sentences followed by 3-5 relevant hashtags. E.g., 'Why the Great Plains are so flat? #geography #geology #usa')",
    "tags": ["tag1", "tag2", "tag3"],
    "sources": ["Citation 1: Book/Article", "Citation 2: Website / Research Paper"]
  }}
}}

Include 8-16 cities with lat/lon for the region (rank 1=capital/biggest, 4=small).
""")


TOPIC_SCRIPT_PROMPT = textwrap.dedent("""\
You are writing a YouTube Shorts script about a SPECIFIC geography question or fact.
The video style is like @urban_atlas — punchy, visual, education-meets-entertainment.

User's prompt: {prompt}

First, extract:
  - The REGION this is about (e.g. "Quebec, Canada", "Arabian Peninsula", "Eastern Canada")
  - The SUBJECT being discussed (e.g. "St. Lawrence River", "Sahara Desert", "cities")
  - The HOOK QUESTION or HOOK FACT to open with

Rules:
- Total runtime: 80-95 seconds (target about 90 seconds)
- 12-18 beats (each beat = one narration line + one visual clip)
- Beat pacing target: each beat should be 2.5-6 seconds
- Beat 1 MUST be a curiosity-driven hook (question or shocking statement)
- Each narration line: 5-15 words MAX
- Narration should sound like a confident narrator, not a textbook. It should pair well with an Attenborough-style voice and dramatic background music.
- Every fact must be real and verifiable.
- Return ONLY valid JSON. Escape all quotes properly. No markdown outside the JSON block.

REQUIRED BEAT STRUCTURE:
  Beat 1 — HOOK: Open with a dramatic map visual (e.g., 3d_orbit, satellite_pan, or dark_cutout). Restate the user's question/fact as a hook.
  Beats 2-4 — CONTEXT: Use map/satellite showing the region/highlight (e.g., 3d_zoom, map_highlight, outline_reveal).
  Beats 5-16 — DEEP DIVE: The MAJORITY of beats MUST use high-quality real footage (real_geography, real_city, real_people).
               These must be cinematic, aesthetically pleasing shots (e.g., drone flights over terrain, clean cityscapes,
               people walking to show culture). 
               IMPORTANT: Specify in the `visual_note` that the footage must be "CLEAN with NO text, NO logos, and NO overlays."
               Use 3d_* or map_* b-roll sparingly here for geographical context.
               When discussing a SPECIFIC geographic feature (river, lake, etc.), include a "highlight" object in that beat to visually call it out.
  Beat Last+ — CLOSING: Use a cinematic real_geography shot or a wide 3d_curvature view. Callback to the hook with a satisfying answer.

Highlight objects (optional per beat, use when discussing specific features):
  "highlight": {{
    "type": "river|lake|desert|mountain|area|coastline",
    "query": "exact name for Nominatim search, e.g. 'St. Lawrence River'",
    "color": "hex color, e.g. '#44AAFF' for water, '#FF8833' for desert, '#44FF66' for land"
  }}
  Use highlights on satellite_pan, outline_reveal, or map_highlight beats.
  Do NOT use highlights on real_* or dark_cutout beats.

Available broll_types:
  dark_cutout      = region shape on animated dark grid with city labels (openers/closers ONLY)
  satellite_pan    = slow pan across terrain (use sparingly)
  map_highlight    = political map with region highlighted in color
  map_wipe         = map overlay wiping L-to-R over satellite
  outline_reveal   = boundary outline appearing on satellite
  terrain_map      = terrain/relief map centered on region
  real_city        = real footage of city streets / downtown / skyline (HIGHLY PREFERRED)
  real_people      = real footage of the region functioning: cars driving, ports working, street life, daily activities (HIGHLY PREFERRED)
  real_geography   = real footage of nature/terrain/coast/mountains (HIGHLY PREFERRED)
  real_concept     = stock footage of a CONCEPT related to the topic, NOT a location
                     (e.g. thermometer in extreme heat, ice melting, crowded subway, volcanic lava)
                     Use when narration discusses WHY something happens or describes an effect.
                     visual_note MUST describe the CONCEPT, not a place.
  wikipedia_image  = an image overlay from a Wikipedia article.
                     visual_note MUST be the exact Wikipedia article title, e.g. "Dnieper River"
  google_earth_pan = sweeping cinematic curved view using satellite imagery (HIGHLY PREFERRED map type)
  3d_orbit         = cinematic 3D satellite orbit around the region (BEST for establishing shots)
  3d_flyover       = cinematic 3D flyover at an oblique angle
  3d_zoom          = zooming from space down to the region in 3D
  3d_curvature     = ultra-wide view showing Earth's curvature
  3d_close_oblique = dramatic close-up tilted view looking across terrain (like Google Earth)
  comparison_map   = animated comparison overlay: highlights a sub-region vs the whole country
                     Requires a "comparison" object in the beat.

VISUAL VARIETY RULES:
- The MAJORITY of beats MUST use real_* broll types.
- Ensure `visual_note` for real_* beats includes "clean, no text overlays, no logos, high quality cinematic".
- Do not use the exact same `broll_type` (except real_* variants) two beats in a row.

Return STRICT JSON:
{{
  "region": "string (the geographic region this is about)",
  "subject": "string (the specific feature/topic being discussed)",
  "hook_question": "string (the opening question/statement)",
  "title": "string (catchy, 3-6 words)",
  "total_duration_sec": number,
  "beats": [
    {{
      "beat_id": 1,
      "script_type": "HOOK|ORIENTATION|PRIMARY_FACT|DETAIL|CUT",
      "narration": "string (the spoken line)",
      "broll_type": "string (from the list above)",
      "duration_sec": number (2.5-6.0 seconds),
      "visual_note": "string (SPECIFIC description of what viewer sees — include 'clean, no text overlays' for real footage)",
      "youtube_query": "string (optional specific YouTube search, e.g. 'Kherson city drone 4k')",
      "caption_text": "string (key phrase, ALL CAPS, 2-5 words)",
      "highlight": {{"type": "string", "query": "string", "color": "string"}} or null,
      "comparison": {{"base": "country/region name", "highlight": "sub-region name",
                     "stat": "value to display", "stat_label": "LABEL",
                     "color": "#hex"}} or null
    }}
  ],
  "cities": [
    {{"name": "string", "lat": number, "lon": number, "rank": 1}}
  ],
  "youtube_metadata": {{
    "title": "string (optimized for YouTube Shorts SEO, max 50 chars)",
    "description": "string (Extremely minimal description: JUST 1-2 punchy sentences followed by 3-5 relevant hashtags. E.g., 'Why the Great Plains are so flat? #geography #geology #usa')",
    "tags": ["tag1", "tag2", "tag3"],
    "sources": ["Citation 1: Book/Article", "Citation 2: Website / Research Paper"]
  }}
}}

""")


def generate_topic_script(prompt: str, run_dir: Path) -> Dict[str, Any]:
    """Stage 1 (prompt mode): Call Gemini to produce topic-driven script + beat map."""
    print("\n[S1] Generating topic script via Gemini...")
    print(f"  Prompt: {prompt}")
    gemini_prompt = TOPIC_SCRIPT_PROMPT.format(prompt=prompt)

    try:
        raw = call_gemini(gemini_prompt)
        script = extract_json(raw)
        script["_source"] = "gemini_topic"
        script["_user_prompt"] = prompt
        print(f"  [OK] Region: {script.get('region', '?')}")
        print(f"  [OK] Subject: {script.get('subject', '?')}")
        print(f"  [OK] {len(script.get('beats', []))} beats, "
              f"{script.get('total_duration_sec', '?')}s total")
    except Exception as e:
        print(f"  [WARN] Gemini topic script failed: {e}")
        print("  Falling back to region-based script...")
        # Try to extract a region from the prompt for fallback
        region_guess = prompt.strip().rstrip('?').split()[-1]
        script = fallback_script(region_guess)
        script["_user_prompt"] = prompt

    script = normalize_script_plan(script)

    path = run_dir / "s1_script.json"
    path.write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {path.name}")
    return script


def normalize_script_plan(script: Dict[str, Any]) -> Dict[str, Any]:
    beats = script.get("beats", [])
    if not isinstance(beats, list):
        script["beats"] = []
        return script

    total = len(beats)
    for i, beat in enumerate(beats):
        dur = float(beat.get("duration_sec", 4))
        beat["duration_sec"] = max(2.5, min(6.0, dur))

        if not beat.get("script_type"):
            if i == 0:
                beat["script_type"] = "HOOK"
            elif i == 1:
                beat["script_type"] = "ORIENTATION"
            elif i == total - 1:
                beat["script_type"] = "CUT"
            elif i == 2:
                beat["script_type"] = "PRIMARY_FACT"
            else:
                beat["script_type"] = "DETAIL"

    # Stretch total runtime toward ~90s while preserving beat proportions.
    target_total = 90.0
    current_total = sum(float(b.get("duration_sec", 10.0)) for b in beats) if beats else 0.0
    if beats and current_total > 0 and current_total < 82.0:
        scale = target_total / current_total
        for beat in beats:
            nd = float(beat.get("duration_sec", 10.0)) * scale
            beat["duration_sec"] = max(2.5, min(9.0, nd))

    # Removed explicit real footage enforcement to favor maps/3D
    # Prevent consecutive beats using the same satellite base image
    # (e.g. satellite_zoom + outline_reveal both show the same satellite underneath)
    satellite_based = {"satellite_pan", "outline_reveal", "map_wipe",
                       "3d_orbit", "3d_flyover", "3d_zoom", "3d_close_oblique"}
    real_cycle = ["real_geography", "real_city", "real_people"]
    for i in range(1, len(beats)):
        prev_type = str(beats[i-1].get("broll_type", ""))
        curr_type = str(beats[i].get("broll_type", ""))
        if prev_type in satellite_based and curr_type in satellite_based:
            # Swap current to a real type that differs from previous
            for rt in real_cycle:
                if rt != str(beats[max(0, i-1)].get("broll_type", "")):
                    beats[i]["broll_type"] = rt
                    if not beats[i].get("visual_note", "").startswith("Real"):
                        beats[i]["visual_note"] = f"Real footage of {beats[i].get('narration', 'the region')[:50]}"
                    break

    script["total_duration_sec"] = round(sum(float(b.get("duration_sec", 7)) for b in beats), 1)
    return script


def generate_script(region: str, run_dir: Path) -> Dict[str, Any]:
    """Stage 1: Call Gemini to produce the script + beat map."""
    print("\n[S1] Generating script via Gemini...")
    prompt = SCRIPT_PROMPT.format(region=region)

    try:
        raw = call_gemini(prompt)
        script = extract_json(raw)
        script["_source"] = "gemini"
        print(f"  [OK] {len(script.get('beats', []))} beats, "
              f"{script.get('total_duration_sec', '?')}s total")
    except Exception as e:
        print(f"  [WARN] Gemini failed: {e}")
        print("  Falling back to template script...")
        script = fallback_script(region)

    script = normalize_script_plan(script)

    path = run_dir / "s1_script.json"
    path.write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {path.name}")
    return script


def fallback_script(region: str) -> Dict[str, Any]:
    """Fallback: try direct Gemini 2.0 flash lookup for a fact if region script fails."""
    try:
        lookup_prompt = f"Longest river in {region}"
        fact = call_gemini(lookup_prompt, temperature=0.0)
        narration = fact.strip().replace("\n", " ")
        if narration:
            return {
                "region": region,
                "title": f"Fact: {lookup_prompt}",
                "total_duration_sec": 18,
                "_source": "fallback_lookup",
                "beats": [
                    {"beat_id": 1, "narration": narration,
                     "script_type": "FACT", "broll_type": "real_geography", "duration_sec": 18,
                     "visual_note": f"Clean, no text overlays, high quality visual for: {lookup_prompt}",
                     "caption_text": lookup_prompt.upper()},
                ],
                "cities": [],
            }
    except Exception:
        pass
    # If lookup fails, use the old hardcoded fallback
    return {
        "region": region,
        "title": f"{region} Geography Explained!",
        "total_duration_sec": 90,
        "_source": "fallback",
        "beats": [
            {"beat_id": 1, "narration": f"What makes {region} so different from anywhere else?",
             "script_type": "HOOK", "broll_type": "dark_cutout", "duration_sec": 12,
             "visual_note": f"Region shape of {region} on animated dark grid with city labels",
             "caption_text": "WHAT MAKES IT DIFFERENT"},
            {"beat_id": 2, "narration": f"The landscape here is like nowhere else on Earth.",
             "script_type": "PRIMARY_FACT", "broll_type": "real_geography", "duration_sec": 12,
             "visual_note": f"Clean, no text overlays, cinematic aerial drone footage of {region} terrain and landscape",
             "caption_text": "NOWHERE ELSE"},
            {"beat_id": 3, "narration": f"And these are the cities that call it home.",
             "script_type": "DETAIL", "broll_type": "real_city", "duration_sec": 12,
             "visual_note": f"Clean, no text overlays, cinematic street level view of the largest city in {region}",
             "caption_text": "THE CITIES"},
            {"beat_id": 4, "narration": f"The people here have a way of life all their own.",
             "script_type": "DETAIL", "broll_type": "real_people", "duration_sec": 12,
             "visual_note": f"Clean, no text overlays, cinematic people walking in markets or streets of {region}",
             "caption_text": "THEIR WAY OF LIFE"},
            {"beat_id": 5, "narration": f"Look at how the coastline frames the whole region.",
             "script_type": "DETAIL", "broll_type": "real_geography", "duration_sec": 12,
             "visual_note": f"Clean, no text overlays, cinematic coastal cliffs or shoreline drone footage in {region}",
             "caption_text": "THE COASTLINE"},
            {"beat_id": 6, "narration": f"Now watch how the borders line up with the terrain.",
             "script_type": "DETAIL", "broll_type": "outline_reveal", "duration_sec": 12,
             "visual_note": f"Boundary outline appearing over satellite of {region}",
             "caption_text": "THE BORDERS"},
            {"beat_id": 7, "narration": f"Now you see {region} differently.",
             "script_type": "CUT", "broll_type": "dark_cutout", "duration_sec": 18,
             "visual_note": f"Region shape on dark grid — closing callback",
             "caption_text": "SEE IT NOW"},
        ],
        "cities": [],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: GEO DATA (boundary, tiles, cities)
# ═══════════════════════════════════════════════════════════════════════════════

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
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=20, context=_SSL_CTX) as resp:
                return resp.read()
        except Exception:
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
    return None


def download_composite(lat, lon, zoom, cols, rows, tile_url, label=""):
    cx, cy = lat_lon_to_tile(lat, lon, zoom)
    n = 2 ** zoom
    hc, hr = cols // 2, rows // 2
    comp = Image.new("RGB", (cols * TILE_SIZE, rows * TILE_SIZE), (20, 30, 50))
    total = cols * rows
    ok = 0
    for dy in range(-hr, hr):
        for dx in range(-hc, hc):
            tx = (cx + dx) % n
            ty = cy + dy
            if ty < 0 or ty >= n:
                continue
            data = download_tile(tile_url.format(z=zoom, x=tx, y=ty))
            if data:
                try:
                    comp.paste(Image.open(io.BytesIO(data)),
                               ((dx + hc) * TILE_SIZE, (dy + hr) * TILE_SIZE))
                    ok += 1
                except Exception:
                    pass
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

    # Pick best bounding box from first result for center
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
    """Pick zoom level and grid size so the region fills ~60-70% of the 9:16 frame.

    Works for any region size — from NYC (0.2° span) to continents (60°+ span).
    The zoom is computed mathematically from the bounding box, not a hardcoded table.
    """
    if bbox and len(bbox) >= 4:
        lat_span = abs(float(bbox[1]) - float(bbox[0]))
        lon_span = abs(float(bbox[3]) - float(bbox[2]))
    else:
        lat_span = 5.0
        lon_span = 5.0

    # Ensure minimum span to avoid division issues on point-like results
    lat_span = max(lat_span, 0.01)
    lon_span = max(lon_span, 0.01)

    # Target: region should fill ~45% of the output frame (1080x1920)
    # leaving generous padding for visual breathing room and context.
    FILL_FRAC = 0.45
    target_w_px = OUT_W / FILL_FRAC   # target map width in pixels to fit region
    target_h_px = OUT_H / FILL_FRAC   # target map height in pixels to fit region

    # At zoom z, each tile is 256px and covers 360/2^z degrees of longitude.
    # So 1 degree of longitude = (2^z * 256) / 360 pixels.
    # For latitude it varies with mercator, but at typical latitudes the
    # approximation is close enough for zoom selection.

    # Required pixels per degree to fill the frame width/height:
    ppd_lon = target_w_px / lon_span
    ppd_lat = target_h_px / lat_span

    # Use the MORE constraining axis (the one that needs higher zoom)
    # to ensure the whole region fits. 9:16 is tall, so lat usually constrains.
    ppd_needed = max(ppd_lon, ppd_lat)

    # Solve: ppd_needed = (2^z * 256) / 360  =>  z = log2(ppd_needed * 360 / 256)
    z_float = math.log2(ppd_needed * 360.0 / TILE_SIZE)
    z = int(round(z_float))
    z = max(3, min(14, z))  # clamp to sane tile range

    # Grid: size the composite so the region fills ~65% of height after crop_916.
    # For portrait (9:16), height usually constrains. Wide regions overflow
    # horizontally (edges cropped off — fine for portrait video).
    ppd = (2 ** z) * TILE_SIZE / 360.0
    region_h_px = lat_span * ppd
    # Composite height so region fills FILL_FRAC of it:
    needed_h = region_h_px / FILL_FRAC
    rows = max(8, int(math.ceil(needed_h / TILE_SIZE)) + 1)
    # Width: match 9:16 crop ratio + small margin
    cols = max(6, int(math.ceil(rows * 9.0 / 16.0)) + 1)

    # Cap tiles to keep download time reasonable (~25s at 0.05s/tile)
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

    # --- Geocode + boundary ---
    print("  Fetching boundary polygon...")
    time.sleep(1.2)
    center, rings, bbox = fetch_boundary(region)
    if center:
        geo["lat"], geo["lon"] = center
        print(f"  Center: {center[0]:.4f}, {center[1]:.4f}")
    else:
        print("  [FAIL] Could not geocode region. Falling back to 0,0")
        geo["lat"], geo["lon"] = 0.0, 0.0

    # Compute zoom/grid
    zoom, cols, rows = choose_zoom_grid(bbox)
    geo["zoom"] = zoom
    geo["cols"] = cols
    geo["rows"] = rows
    print(f"  Auto zoom: z={zoom}, grid={cols}x{rows}")

    if rings:
        # Filter out tiny offshore island rings and maritime boundary artifacts.
        # Keep only rings whose bounding box is at least 0.02° wide OR tall
        # (roughly > 2 km), which removes scattered ocean-border slivers
        # while keeping all meaningful land boundaries.
        filtered_rings = []
        for ring in rings:
            lats = [pt[0] for pt in ring]
            lons = [pt[1] for pt in ring]
            lat_span = max(lats) - min(lats)
            lon_span = max(lons) - min(lons)
            if lat_span > 0.02 or lon_span > 0.02:
                filtered_rings.append(ring)
        # If aggressive filter removed everything, fall back to originals
        if not filtered_rings:
            filtered_rings = rings

        total_pts = sum(len(r) for r in filtered_rings)
        print(f"  Boundary: {len(filtered_rings)} ring(s) (from {len(rings)} raw), {total_pts} pts")
        geo["rings"] = filtered_rings
        geo["pixel_rings"] = rings_to_pixels(filtered_rings, geo["lat"], geo["lon"], zoom, cols, rows)
        # filter to visible
        geo["pixel_rings"] = [r for r in geo["pixel_rings"]
                              if any(0 <= x <= OUT_W and 0 <= y <= OUT_H for x, y in r)]
    else:
        geo["rings"] = []
        geo["pixel_rings"] = []

    # --- Cities from script or from Gemini ---
    cities = script.get("cities", [])
    if cities:
        geo["cities"] = cities
        print(f"  Cities: {len(cities)} from script")
    else:
        geo["cities"] = []

    # --- Tiles ---
    tiles_dir = run_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # Satellite
    sat_path = tiles_dir / f"satellite_z{zoom}.jpg"
    if not sat_path.exists():
        print(f"  Downloading satellite tiles z={zoom}...")
        comp = download_composite(geo["lat"], geo["lon"], zoom, cols, rows, ESRI_URL, "sat")
        if comp:
            crop_916(comp, sat_path)
            print(f"  -> {sat_path.name}")
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
        print(f"  [reuse] {wide_path.name}")
    geo["satellite_wide"] = str(wide_path)

    # Save geodata
    geo_save = {k: v for k, v in geo.items() if k not in ("rings", "pixel_rings")}
    geo_save["boundary_rings_count"] = len(geo.get("rings", []))
    geo_save["boundary_pts_total"] = sum(len(r) for r in geo.get("rings", []))
    (run_dir / "s2_geodata.json").write_text(json.dumps(geo_save, indent=2), encoding="utf-8")

    return geo


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3: B-ROLL ASSET GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE HIGHLIGHTING (rivers, deserts, sub-regions)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_feature_geometry(query: str, geo: Dict) -> Optional[List]:
    """Fetch polygon/linestring geometry for a specific feature (river, desert, etc.)
    from Nominatim and convert to pixel coordinates for overlay drawing."""
    print(f"    Fetching feature geometry: {query}")
    time.sleep(1.2)  # Nominatim rate limit
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

    # Find the best result with geometry
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

        # Convert to pixel coords using the same projection as the region
        zoom = geo.get("zoom", 8)
        cols = geo.get("cols", 10)
        rows = geo.get("rows", 14)
        pixel_rings = rings_to_pixels(rings, geo["lat"], geo["lon"], zoom, cols, rows)
        # Filter to visible rings
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
    """Draw a semi-transparent colored polygon/line overlay for a geographic feature.

    For rivers/coastlines: bright colored line with glow
    For areas/deserts: semi-transparent fill with outline
    """
    base = img.convert("RGBA") if img.mode != "RGBA" else img.copy()
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)

    for ring in feature_rings:
        pts = [(int(round(x)), int(round(y))) for x, y in ring]
        if len(pts) < 2:
            continue

        if is_line or len(pts) < 4:
            # Line features (rivers, coastlines): glow + bright line
            # Wide glow
            draw.line(pts, fill=(r, g, b, 40), width=18)
            # Medium glow
            draw.line(pts, fill=(r, g, b, 80), width=10)
            # Bright core
            draw.line(pts, fill=(r, g, b, 220), width=4)
        else:
            # Area features (deserts, sub-regions): fill + outline
            if len(pts) >= 3:
                draw.polygon(pts, fill=(r, g, b, opacity))
            # Bright outline
            draw.line(pts + [pts[0]], fill=(r, g, b, 200), width=3)

    return Image.alpha_composite(base, overlay)


def burn_title_overlay(video_path: Path, title_text: str, out_path: Path,
                       display_duration: float = 4.0) -> bool:
    """Burn an animated title card overlay onto the first N seconds of a video.

    Matches @urban_atlas style: white box at top with black bold text.
    Animation: slides down from off-screen + fades in, then fades out at end.
    """
    if not video_path.exists():
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Word-wrap the title to fit ~25 chars per line
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

    # Calculate box size
    font_size = 42
    line_h = font_size + 8
    num_lines = len(lines)
    box_h = num_lines * line_h + 30
    rest_y = 100  # final resting Y position
    start_y = -box_h - 20  # starts off-screen above
    box_w = min(OUT_W - 80, max(len(l) for l in lines) * (font_size * 0.55) + 60)
    box_x = int((OUT_W - box_w) / 2)

    # Animation timings
    slide_in_dur = 0.4  # seconds to slide in
    fade_out_start = display_duration - 0.5
    fade_out_dur = 0.5

    # Y position: slides from start_y to rest_y during slide_in_dur, stays, then fades
    # FFmpeg expression: y = start_y + (rest_y - start_y) * min(t/slide_in_dur, 1)
    y_expr = f"{start_y}+({rest_y}-({start_y}))*min(t/{slide_in_dur}\\,1)"

    # Alpha: fades in during first 0.3s, stays, fades out in last 0.5s
    # For drawbox: use alpha channel in color (static for now to avoid complex filters)
    # For drawtext: use alpha parameter with expression
    alpha_expr = (
        f"if(lt(t\\,0.3)\\, t/0.3\\, "
        f"if(gt(t\\,{fade_out_start})\\, "
        f"({display_duration}-t)/{fade_out_dur}\\, 1))"
    )

    vf_parts = []

    # White box with animated Y and alpha
    vf_parts.append(
        f"drawbox=x={box_x}:y='{y_expr}':w={int(box_w)}:h={box_h}"
        f":color=white@0.92:t=fill"
        f":enable='between(t,0,{display_duration})'"
    )

    # Text lines with same animated Y and alpha
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


# ═══════════════════════════════════════════════════════════════════════════════
# TTS + VOICEOVER
# ═══════════════════════════════════════════════════════════════════════════════

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
                fallback = Path(__file__).resolve().parent.parent / "voices" / voice_path.name
                if fallback.exists():
                    print(f"  [INFO] Voice file not found locally, using: {fallback}")
                    vp = str(fallback.resolve())
                else:
                    print(f"  [WARN] Voice file not found: {voice_path} (also checked {fallback})")
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
    try:
        import whisper
    except ImportError:
        print("  [SKIP] whisper not installed")
        return None

    json_cache = audio_path.parent / "whisper_segments.json"
    if json_cache.exists():
        print("  [CACHED] whisper_segments.json")
        return json.loads(json_cache.read_text(encoding="utf-8"))

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
    """Update beat durations using real Whisper-measured spoken timings.

    Maps each Whisper segment to the closest beat by text similarity,
    then sets beat["duration_sec"] to the actual spoken duration + buffer.
    Also stores beat["audio_start"] and beat["audio_end"] for assembly.
    """
    beats = script.get("beats", [])
    if not beats or not whisper_segs:
        return script

    # Combine Whisper segments into sentence-level chunks that map to beats.
    # Strategy: greedily assign Whisper segments to beats by matching narration text.
    beat_narrations = [b.get("narration", "").lower().strip() for b in beats]

    # Build a list of (segment_index, beat_index, similarity) pairs
    def _word_overlap(a: str, b: str) -> float:
        """Jaccard similarity on word sets."""
        wa = set(a.lower().split())
        wb = set(b.lower().split())
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)

    # Assign segments to beats greedily (in order)
    seg_idx = 0
    BUFFER_SEC = 0.5  # extra padding after spoken audio ends

    for bi, beat in enumerate(beats):
        narration = beat_narrations[bi]
        if not narration or seg_idx >= len(whisper_segs):
            continue

        # Accumulate consecutive Whisper segments until we match this beat's narration
        best_end_idx = seg_idx
        best_score = 0.0
        accumulated_text = ""

        for si in range(seg_idx, min(seg_idx + 4, len(whisper_segs))):
            accumulated_text += " " + whisper_segs[si].get("text", "")
            score = _word_overlap(narration, accumulated_text)
            if score > best_score:
                best_score = score
                best_end_idx = si

        if best_score < 0.15:
            # Very low match — skip, keep Gemini's guess
            continue

        # Compute real duration from matched segments
        audio_start = float(whisper_segs[seg_idx]["start"])
        audio_end = float(whisper_segs[best_end_idx]["end"])
        real_dur = audio_end - audio_start + BUFFER_SEC
        real_dur = max(3.0, min(20.0, real_dur))  # clamp to sane range

        old_dur = beat.get("duration_sec", 0)
        beat["duration_sec"] = round(real_dur, 2)
        beat["audio_start"] = round(audio_start, 3)
        beat["audio_end"] = round(audio_end, 3)

        print(f"    Beat {beat.get('beat_id', bi+1)}: "
              f"{old_dur:.1f}s (guessed) -> {real_dur:.1f}s (spoken)")

        seg_idx = best_end_idx + 1

    # Update total duration
    script["total_duration_sec"] = round(
        sum(float(b.get("duration_sec", 7)) for b in beats), 1
    )
    print(f"    Total duration updated: {script['total_duration_sec']:.1f}s")
    return script


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
        f"Style: Default,{CAPTION_FONT_NAME},58,&H00FFFFFF,&H00FFFFFF,"
        "&H00101010,&H8A000000,-1,0,0,0,100,100,0,0,1,4,1,1,30,30,180,1\n"
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
        # Split words into chunks that fit two lines (by word/char count)
        chunks: List[List[Dict[str, Any]]] = []
        cur: List[Dict[str, Any]] = []
        max_words = 18
        max_chars = 80
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

    def _to_two_lines(tokens: List[str]) -> str:
        if len(tokens) <= 1:
            return tokens[0] if tokens else ""
        target = max(1, len(tokens) // 2)
        split_idx = target
        best_delta = 1_000_000
        for i in range(1, len(tokens)):
            left = " ".join(tokens[:i])
            right = " ".join(tokens[i:])
            delta = abs(len(left) - len(right))
            if delta < best_delta:
                best_delta = delta
                split_idx = i
        line1 = " ".join(tokens[:split_idx]).strip()
        line2 = " ".join(tokens[split_idx:]).strip()
        return f"{line1}\\N{line2}" if line2 else line1

    for chunk in _chunk_words(words):
        tokens = [_clean_word(w["word"]).upper() for w in chunk]
        original_words = [_clean_word(w["word"]) for w in chunk]
        if not any(tokens):
            continue
        # Decide the full string structure upfront to keep layout stable
        full_text_str = " ".join(tokens)
        split_idx = len(tokens)  # default: no split
        
        if len(full_text_str) > 36 and len(tokens) > 1:
            best_delta = 1_000_000
            for k in range(1, len(tokens)):
                left_len = len(" ".join(tokens[:k]))
                right_len = len(" ".join(tokens[k:]))
                delta = abs(left_len - right_len)
                if delta < best_delta:
                    best_delta = delta
                    split_idx = k

        for i in range(len(chunk)):
            # Start when THIS word is spoken
            start = float(chunk[i]["start"])
            # End when the NEXT word starts
            if i < len(chunk) - 1:
                end = float(chunk[i+1]["start"])
            else:
                end = max(start + 0.1, float(chunk[i]["end"]))
                
            start_t = _format_ass_time(start)
            end_t = _format_ass_time(end)

            # Build the line: visible up to word i, fully transparent after word i.
            # This ensures the text block is always fully sized and centers correctly,
            # revealing words left-to-right without shifting.
            out_str = ""
            for j, t in enumerate(tokens):
                prefix = ""
                if j > 0 and j != split_idx:
                    prefix = " "
                elif j == split_idx:
                    prefix = "\\N"
                
                orig = original_words[j]
                # Highlight if it's a long word or starts with a capital letter (excluding simple sentence starters if possible, but keeping it simple)
                is_important = len(orig) > 5 or (orig and orig[0].isupper() and j > 0)
                
                if is_important:
                    # Apply color highlight
                    t_formatted = "{\\c&H00D7FF&}" + t + "{\\c&HFFFFFF&}"
                else:
                    t_formatted = t
                    
                if j < i:
                    out_str += prefix + t_formatted
                elif j == i:
                    out_str += prefix + "{\\alpha&HFF&\\t(0,150,\\alpha&H00&)}" + t_formatted
                else:
                    out_str += prefix + "{\\alpha&HFF&}" + t_formatted

            anim = "{\\an2}"  # Bottom-center aligned, which is standard for these shorts. 
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


def mix_voiceover(video_path: Path, voiceover_path: Path, run_dir: Path,
                  music_path: Optional[Path] = None) -> Optional[Path]:
    """Mix voiceover (and optional music) onto the silent video."""
    out_path = run_dir / "final_short_with_audio.mp4"

    if music_path and music_path.exists():
        # 3-input mix: video + voiceover + music
        print(f"  Mixing voiceover + music onto video...")
        ok = run_ffmpeg([
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(voiceover_path),
            "-i", str(music_path),
            "-filter_complex",
            f"[1:a]aresample=44100[vo];"
            f"[2:a]aloop=loop=-1:size=2e+09,atrim=duration=120,"
            f"volume={MUSIC_VOL_DB}dB,aresample=44100[mu];"
            f"[vo][mu]amix=inputs=2:duration=shortest:dropout_transition=2:normalize=0[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(out_path)
        ], timeout=120)
    else:
        # 2-input: video + voiceover only
        print(f"  Mixing voiceover onto video...")
        ok = run_ffmpeg([
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(voiceover_path),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(out_path)
        ], timeout=120)

    if ok and out_path.exists():
        mb = out_path.stat().st_size / 1048576
        print(f"  [OK] {out_path.name} ({mb:.1f} MB)")
        return out_path
    else:
        print("  [FAIL] Audio mix failed")
        return None


def run_ffmpeg(cmd, timeout=300):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            # Show last 500 chars of stderr (skip the banner)
            err = r.stderr.strip().split('\n')
            useful = [l for l in err if not l.startswith(('  configuration:', '  lib', '  built with'))]
            print(f"    [FFmpeg ERR] {'  '.join(useful[-5:])}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("    [FFmpeg TIMEOUT]")
        return False


def _extract_search_hint(beat: Dict[str, Any]) -> str:
    """Pull key subject from narration or visual_note to sharpen YouTube search.

    Uses a simple heuristic: grab the visual_note first (most specific),
    fall back to narration.  Strip generic filler words and keep it short
    so the YouTube search stays focused.
    """
    raw = beat.get("visual_note") or beat.get("narration") or ""
    # Remove very generic phrases that dilute the search (case-insensitive)
    for noise in (
        "real footage related to:", "real footage of",
        "footage of", "montage of", "showing", "clip of", "video of",
        "people", "person", "the region", "in the background",
    ):
        raw = re.sub(re.escape(noise), "", raw, flags=re.IGNORECASE)
    # Collapse whitespace and limit to ~60 chars so YT search stays relevant
    hint = " ".join(raw.split())[:60].strip().rstrip(".")
    return hint


def _yt_query_for_beat(region: str, beat: Dict[str, Any], script_subject: str = "") -> str:
    yt_query = beat.get("youtube_query")
    if yt_query and str(yt_query).strip():
        # User/Gemini specifically defined an exact search
        return str(yt_query).strip()

    btype = str(beat.get("broll_type", "")).strip().lower()
    hint = _extract_search_hint(beat)

    # If beat has a highlight subject, use it for a more specific search
    highlight = beat.get("highlight")
    if highlight and isinstance(highlight, dict):
        subject = highlight.get("query", "")
        if subject:
            htype = highlight.get("type", "")
            if htype in ("river", "lake", "coastline"):
                return f"{subject} aerial drone 4k"
            elif htype in ("desert", "mountain"):
                return f"{subject} landscape drone 4k"
            else:
                return f"{subject} {region} 4k"

    search_context = region
    if script_subject and script_subject.lower() not in region.lower():
        search_context = f"{script_subject} {region}"

    if btype == "real_city":
        if hint:
            return f"{hint} city 4k"
        return f"{search_context} city walk 4k"
    if btype == "real_people":
        if hint:
            return f"{hint} {search_context}"
        return f"{search_context} street life people walking"
    # real_geography — use narration content to get specific footage
    if hint:
        return f"{hint} {search_context} 4k"
    return f"{search_context} nature landscape drone"


def _yt_search_candidates(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    ytdlp_ok = False
    if YTDLP_PATH:
        if "/" in YTDLP_PATH or "\\" in YTDLP_PATH:
            ytdlp_ok = Path(YTDLP_PATH).exists()
        else:
            ytdlp_ok = shutil.which(YTDLP_PATH) is not None
    if not ytdlp_ok:
        return []

    cmd = [
        YTDLP_PATH,
        f"ytsearch{max_results}:{query}",
        "--flat-playlist",
        "--print", "%(id)s|||%(title)s|||%(duration)s|||%(view_count)s|||%(webpage_url)s",
        "--match-filter", f"duration > {REAL_MIN_SRC_SEC} & duration < {REAL_MAX_SRC_SEC} & !is_live",
        "--no-warnings", "--ignore-errors",
    ]
    if COOKIES_YT and Path(COOKIES_YT).exists():
        cmd.extend(["--cookies", COOKIES_YT])

    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60, check=False)
        text = (r.stdout or b"").decode("utf-8", errors="replace")
    except Exception:
        return []

    bad_terms = {
        "podcast", "interview", "news", "compilation", "slideshow", "top 10",
        "facts", "documentary", "history explained", "lecture", "reaction",
        "stock", "shutterstock", "getty", "istock", "dreamstime", "adobe stock",
        "footage farm", "videoblocks", "storyblocks", "pond5", "artgrid",
        "royalty free", "royalty-free", "stock footage", "preview", "watermark",
        "vlog", "lyrics", "text", "music video"
    }

    # Extract required core keywords from query for strict vetting
    core_kw = set(w.lower() for w in query.split() if len(w) > 3 and w.lower() not in {"drone", "aerial", "city", "street", "walking", "landscape"})

    out: List[Dict[str, Any]] = []
    for line in text.splitlines():
        parts = line.split("|||")
        if len(parts) < 5:
            continue
        vid, title, dur, views, url = parts[:5]
        title_l = title.lower()
        if any(term in title_l for term in bad_terms):
            continue

        # Vetting: Ensure the video relates to the topic. If there are core keywords in the query,
        # at least one of them must be present in the video title or we penalize heavily.
        if core_kw:
            if not any(kw in title_l for kw in core_kw):
                continue

        try:
            d = float(dur or 0)
        except Exception:
            d = 0
        try:
            v = float(views or 0)
        except Exception:
            v = 0

        score = 0.0
        if "4k" in title_l or "60fps" in title_l:
            score += 2.0
        if any(k in title_l for k in ("walk", "drone", "aerial", "street", "nature", "coast", "mountain")):
            score += 2.0
        if d >= 30:
            score += 1.0
        if v > 10000:
            score += 1.0

        out.append({"video_id": vid, "title": title, "duration": d, "views": v, "url": url, "score": score})

    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def _yt_download_and_trim(video: Dict[str, Any], out_path: Path, duration_sec: float) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_path = out_path.parent / f"_yt_{video['video_id']}_full.mp4"

    dl_cmd = [
        YTDLP_PATH,
        "-f", "bv*[vcodec^=avc1][height<=1080]+ba[ext=m4a]/b[ext=mp4]/best",
        "--recode-video", "mp4",
        "--merge-output-format", "mp4",
        "-o", str(full_path),
        "--retries", "2",
        "--limit-rate", "2M",
        "--sleep-interval", "1",
        "--max-sleep-interval", "3",
        "--no-warnings", "--ignore-errors",
        video.get("url") or f"https://www.youtube.com/watch?v={video['video_id']}",
    ]
    if COOKIES_YT and Path(COOKIES_YT).exists():
        dl_cmd[1:1] = ["--cookies", COOKIES_YT]

    try:
        # Reduced timeout from 600s to 120s to prevent long hangs
        r = subprocess.run(dl_cmd, capture_output=True, timeout=120)
        if r.returncode != 0 or not full_path.exists():
            return False
    except Exception:
        return False

    # Trim a centered snippet for better pacing
    target = max(5.0, min(8.0, float(duration_sec)))
    src_dur = max(0.0, float(video.get("duration") or 0.0))
    if src_dur > target + 6:
        start = max(2.0, (src_dur - target) / 2)
    else:
        start = 1.0

    ok = run_ffmpeg([
        "ffmpeg", "-y", "-ss", str(start), "-i", str(full_path),
        "-t", str(target),
        "-filter_complex", BLUR_BG_FILTER,
        "-r", str(FPS),
        "-c:v", "libx264", "-preset", "fast", "-crf", "21", "-an",
        str(out_path)
    ], timeout=180)

    full_path.unlink(missing_ok=True)
    return ok


def gen_real_youtube_clip(region: str, beat: Dict[str, Any], out_path: Path, duration: float, subject: str = "") -> bool:
    """Try real footage sourcing from YouTube only."""
    query = _yt_query_for_beat(region, beat, subject)

    # 1) YouTube via yt-dlp (highest hit quality when available)
    ytdlp_ok = False
    if YTDLP_PATH:
        if "/" in YTDLP_PATH or "\\" in YTDLP_PATH:
            ytdlp_ok = Path(YTDLP_PATH).exists()
        else:
            ytdlp_ok = shutil.which(YTDLP_PATH) is not None
    if ytdlp_ok:
        candidates = _yt_search_candidates(query)
        print(f" [yt:{len(candidates)}]", end="", flush=True)
        for i, cand in enumerate(candidates[:4]):
            print(f".{i+1}", end="", flush=True)
            if _yt_download_and_trim(cand, out_path, duration):
                return True

    return False


def gen_real_concept_clip(
    region: str, beat: Dict[str, Any], out_path: Path, duration: float
) -> bool:
    """Source conceptual footage from YouTube only using visual_note intent."""
    # Use visual_note directly as the search query — it describes a concept
    query = beat.get("visual_note", "").strip()
    if not query:
        query = beat.get("narration", "").strip()
    if not query:
        return False

    # Trim to 5 keywords max for better stock footage search results
    words = query.split()
    if len(words) > 5:
        query = " ".join(words[:5])

    beat_query = dict(beat)
    beat_query["youtube_query"] = query
    return gen_real_youtube_clip(region, beat_query, out_path, duration)


def gen_wikipedia_image(geo: Dict, beat: Dict[str, Any], out_path: Path, duration: float) -> bool:
    """Fetch the main image from a Wikipedia page based on the visual_note title."""
    article_title = beat.get("visual_note", "").strip()
    if not article_title:
        return False
        
    print(f" [wiki:'{article_title}']", end="", flush=True)

    # Convert spaces for URL encoding
    title_enc = urllib.parse.quote(article_title)
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=pageimages&titles={title_enc}&pithumbsize=1080&redirects=1&format=json"

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=10, context=_SSL_CTX) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return False

    pages = data.get("query", {}).get("pages", {})
    if not pages or "-1" in pages:
        return False

    # Get the URL from the first found page
    page = next(iter(pages.values()))
    img_url = page.get("thumbnail", {}).get("source", "")

    if not img_url:
        return False

    # Download the image directly to the output path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(img_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as resp:
            out_path.write_bytes(resp.read())
            return out_path.exists() and out_path.stat().st_size > 1024
    except Exception:
        out_path.unlink(missing_ok=True)
        return False


# --- Individual B-roll generators ---

def gen_satellite_zoom(geo, out_path, duration=6.0):
    """Zoom-in effect on satellite image using fast crop+scale approach."""
    src = Path(geo["satellite_wide"])
    if not src.exists():
        src = Path(geo["satellite_frame"])
    if not src.exists():
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-scale to 3x output for zoom headroom
    scale = 3
    pw, ph = OUT_W * scale, OUT_H * scale
    prescale = out_path.parent / f"_pre_zoom_{out_path.stem}.jpg"
    if not run_ffmpeg([
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"scale={pw}:{ph}:force_original_aspect_ratio=increase,crop={pw}:{ph}",
        str(prescale)], timeout=60):
        return False

    total_n = int(duration * FPS)
    # crop shrinks from full 3x down to 1x as frame n progresses (zoom-in)
    # factor = 1 - n*(2/3)/total_n  →  goes from 1.0 to 0.333
    vf = (
        f"crop="
        f"w='trunc({pw}*(1-n*2/(3*{total_n})))':"
        f"h='trunc({ph}*(1-n*2/(3*{total_n})))':"
        f"x='trunc(({pw}-trunc({pw}*(1-n*2/(3*{total_n}))))/2)':"
        f"y='trunc(({ph}-trunc({ph}*(1-n*2/(3*{total_n}))))/2)',"
        f"scale={OUT_W}:{OUT_H},format=yuv420p"
    )
    ok = run_ffmpeg([
        "ffmpeg", "-y",
        "-loop", "1", "-framerate", str(FPS), "-i", str(prescale),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
        "-r", str(FPS), "-frames:v", str(total_n),
        str(out_path)], timeout=120)
    prescale.unlink(missing_ok=True)
    return ok


def gen_satellite_pan(geo, out_path, duration=5.0, direction="left_to_right"):
    """Pan across satellite imagery using fast crop approach."""
    src = Path(geo["satellite_frame"])
    if not src.exists():
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Scale to 2x output width for pan headroom
    pw, ph = OUT_W * 2, OUT_H
    prescale = out_path.parent / f"_pre_pan_{out_path.stem}.jpg"
    if not run_ffmpeg([
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"scale={pw}:{ph}:force_original_aspect_ratio=increase,crop={pw}:{ph}",
        str(prescale)], timeout=60):
        return False

    total_n = int(float(duration) * FPS)
    pan_px = pw - OUT_W  # total pixels to travel
    if direction == "right_to_left":
        x_expr = f"trunc({pan_px}*(1-n/{total_n}))"
    else:
        x_expr = f"trunc({pan_px}*n/{total_n})"
    vf = f"crop={OUT_W}:{OUT_H}:x='{x_expr}':y=0,format=yuv420p"
    ok = run_ffmpeg([
        "ffmpeg", "-y",
        "-loop", "1", "-framerate", str(FPS), "-i", str(prescale),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
        "-r", str(FPS), "-frames:v", str(total_n),
        str(out_path)], timeout=120)
    prescale.unlink(missing_ok=True)
    return ok


def gen_map_highlight(geo, out_path):
    """Political map with region highlighted via polygon fill + outline."""
    map_frame = Path(geo["map_frame"])
    pixel_rings = geo.get("pixel_rings", [])
    if not map_frame.exists() or not pixel_rings:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(str(map_frame)).convert("RGBA")

    # Semi-transparent red/orange highlight fill
    highlight = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    hd = ImageDraw.Draw(highlight)
    for ring in pixel_rings:
        pts = [(int(round(x)), int(round(y))) for x, y in ring]
        if len(pts) >= 3:
            hd.polygon(pts, fill=(220, 60, 30, 100))  # red tint
    img = Image.alpha_composite(img, highlight)

    # Bright outline
    outline = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    od = ImageDraw.Draw(outline)
    for ring in pixel_rings:
        pts = [(int(round(x)), int(round(y))) for x, y in ring]
        od.line(pts + [pts[0]], fill=hex_rgba("#FF4422", 200), width=4)
    img = Image.alpha_composite(img, outline)

    # City labels
    cities = geo.get("cities", [])
    if cities:
        img = draw_cities(img, pixel_rings, cities, geo, _font(28), _font_reg(20))

    img.convert("RGB").save(str(out_path), "JPEG", quality=95)
    return True


def gen_dark_cutout(geo, out_path, duration=5.0):
    """Cinematic dark cutout — neon-glow region outline over smooth animated dark grid.

    Uses a single high-res composited PNG + FFmpeg zoompan for silky-smooth motion,
    instead of rendering individual frames in Python (which was slow and choppy).
    """
    map_frame = Path(geo["map_frame"])
    pixel_rings = geo.get("pixel_rings", [])
    if not map_frame.exists() or not pixel_rings:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Build the dark grid background at 2x resolution for zoom headroom ──
    scale = 2
    bw, bh = OUT_W * scale, OUT_H * scale
    bg = Image.new("RGBA", (bw, bh), (15, 18, 22, 255))  # near-black
    bd = ImageDraw.Draw(bg)

    grid_spacing = 80
    # Draw a clean grid with subtle lines
    for x in range(0, bw + grid_spacing, grid_spacing):
        bd.line([(x, 0), (x, bh)], fill=(32, 38, 42, 255), width=1)
    for y in range(0, bh + grid_spacing, grid_spacing):
        bd.line([(0, y), (bw, y)], fill=(32, 38, 42, 255), width=1)

    # ── 2. Region cutout with blue/dark tint ──
    mp = Image.open(str(map_frame)).convert("RGBA")
    mask = make_mask((OUT_W, OUT_H), pixel_rings)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=2))
    cutout = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    cutout.paste(mp, (0, 0), mask)

    # Dark tint so the region map doesn't feel bright
    tint = Image.new("RGBA", (OUT_W, OUT_H), (10, 20, 40, 140))
    cutout = Image.alpha_composite(cutout, tint)
    bg_clear = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    bg_clear.paste(cutout, (0, 0), mask)
    cutout = bg_clear

    # ── 3. NEON GLOW OUTLINE — multi-pass Gaussian blur for bloom effect ──
    glow_layer = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)
    neon_color = (0, 220, 255)  # bright cyan

    for ring in pixel_rings:
        pts = [(int(round(x)), int(round(y))) for x, y in ring]
        if len(pts) < 3:
            continue
        # Wide glow pass (thick, will be blurred)
        glow_draw.line(pts + [pts[0]], fill=(*neon_color, 120), width=12)

    # Blur the wide glow to create bloom
    glow_blurred = glow_layer.filter(ImageFilter.GaussianBlur(radius=18))

    # Medium glow
    glow_med = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    gm_draw = ImageDraw.Draw(glow_med)
    for ring in pixel_rings:
        pts = [(int(round(x)), int(round(y))) for x, y in ring]
        if len(pts) < 3:
            continue
        gm_draw.line(pts + [pts[0]], fill=(*neon_color, 160), width=6)
    glow_blurred = Image.alpha_composite(glow_blurred,
                                         glow_med.filter(ImageFilter.GaussianBlur(radius=6)))

    # Sharp bright core line
    core_line = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    cl_draw = ImageDraw.Draw(core_line)
    for ring in pixel_rings:
        pts = [(int(round(x)), int(round(y))) for x, y in ring]
        if len(pts) < 3:
            continue
        cl_draw.line(pts + [pts[0]], fill=(180, 240, 255, 240), width=2)

    # ── 4. Composite all foreground layers at 1x ──
    fg = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    fg = Image.alpha_composite(fg, cutout)
    fg = Image.alpha_composite(fg, glow_blurred)
    fg = Image.alpha_composite(fg, core_line)

    # City labels on top
    cities = geo.get("cities", [])
    if cities:
        fg = draw_cities(fg, pixel_rings, cities, geo,
                         _font(30), _font_reg(20),
                         color=(220, 240, 255, 230))

    # ── 5. Paste foreground centered onto the 2x background ──
    paste_x = (bw - OUT_W) // 2
    paste_y = (bh - OUT_H) // 2
    bg.paste(fg, (paste_x, paste_y), fg)

    # Save the single composited PNG
    comp_path = out_path.parent / f"_dc_comp_{out_path.stem}.png"
    bg.convert("RGB").save(str(comp_path), "PNG")

    # ── 6. Smooth cinematic motion via FFmpeg zoompan ──
    # Slow drift + subtle zoom gives a polished, alive feel
    frames = int(float(duration) * FPS)
    # Gentle drift leftward + slight zoom in
    zp = (
        f"zoompan=z='min(1.0+on/{frames}*0.12,1.12)'"
        f":x='(iw-iw/zoom)/2 + on/{frames}*(iw*0.04)'"
        f":y='(ih-ih/zoom)/2 - on/{frames}*(ih*0.02)'"
        f":d={frames}:s={OUT_W}x{OUT_H}:fps={FPS}"
    )

    ok = run_ffmpeg([
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(comp_path),
        "-vf", zp,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(out_path)
    ], timeout=180)

    comp_path.unlink(missing_ok=True)
    return ok


def gen_map_wipe(geo, out_path, duration=3.0):
    """Left-to-right wipe of map overlay onto satellite."""
    sat = Path(geo["satellite_frame"])
    mp = Path(geo["map_frame"])
    pixel_rings = geo.get("pixel_rings", [])
    if not sat.exists() or not mp.exists():
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = out_path.parent / f"_wipe_{out_path.stem}"
    frames_dir.mkdir(parents=True, exist_ok=True)

    sat_img = Image.open(str(sat)).convert("RGBA")
    map_img = Image.open(str(mp)).convert("RGBA")

    # Build final composite
    province_mask = make_mask((OUT_W, OUT_H), pixel_rings) if pixel_rings else Image.new("L", (OUT_W, OUT_H), 255)
    province_mask = province_mask.filter(ImageFilter.GaussianBlur(radius=2))

    alpha_map = map_img.copy()
    a = alpha_map.split()[3].point(lambda p: int(p * 0.55))
    alpha_map.putalpha(a)
    masked = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    masked.paste(alpha_map, (0, 0), province_mask)

    # Outline
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

    # City labels
    cities = geo.get("cities", [])
    if cities:
        full = draw_cities(full, pixel_rings, cities, geo, _font(28), _font_reg(18),
                           color=(220, 240, 255, 220))

    total_frames = int(duration * FPS)
    for i in range(total_frames):
        t = i / max(total_frames - 1, 1)
        t = t * t * (3 - 2 * t)  # smoothstep
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

    # Cleanup frames
    import shutil
    shutil.rmtree(str(frames_dir), ignore_errors=True)
    return ok


def gen_outline_reveal(geo, out_path, duration=4.0):
    """Cross-fade from plain satellite to outlined version."""
    sat = Path(geo["satellite_frame"])
    pixel_rings = geo.get("pixel_rings", [])
    if not sat.exists() or not pixel_rings:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Create outlined version
    sat_img = Image.open(str(sat)).convert("RGBA")
    outlined = sat_img.copy()

    # Outline + glow
    ol = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ol)
    for ring in pixel_rings:
        pts = [(int(round(x)), int(round(y))) for x, y in ring]
        od.line(pts + [pts[0]], fill=hex_rgba("#00FF88", 60), width=12)
    for ring in pixel_rings:
        pts = [(int(round(x)), int(round(y))) for x, y in ring]
        od.line(pts + [pts[0]], fill=hex_rgba("#00FF88", 200), width=3)

    # Semi-transparent fill
    fill = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    fd = ImageDraw.Draw(fill)
    for ring in pixel_rings:
        pts = [(int(round(x)), int(round(y))) for x, y in ring]
        if len(pts) >= 3:
            fd.polygon(pts, fill=(0, 255, 136, 30))

    outlined = Image.alpha_composite(outlined, fill)
    outlined = Image.alpha_composite(outlined, ol)

    # Save temp frames
    plain_tmp = out_path.parent / f"_plain_{out_path.stem}.jpg"
    outlined_tmp = out_path.parent / f"_outl_{out_path.stem}.jpg"
    sat_img.convert("RGB").save(str(plain_tmp), "JPEG", quality=95)
    outlined.convert("RGB").save(str(outlined_tmp), "JPEG", quality=95)

    blend_expr = f"A*(1-T/{duration})+B*(T/{duration})"
    ok = run_ffmpeg([
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(plain_tmp),
        "-loop", "1", "-i", str(outlined_tmp),
        "-filter_complex",
        f"[0]scale={OUT_W}:{OUT_H}[a];[1]scale={OUT_W}:{OUT_H}[b];"
        f"[a][b]blend=all_expr='{blend_expr}'",
        "-t", str(duration), "-r", str(FPS),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        str(out_path)], timeout=120)

    plain_tmp.unlink(missing_ok=True)
    outlined_tmp.unlink(missing_ok=True)
    return ok


def gen_terrain_map(geo, out_path):
    """Map frame with outlined boundary — simpler version of map_highlight."""
    map_frame = Path(geo["map_frame"])
    pixel_rings = geo.get("pixel_rings", [])
    if not map_frame.exists():
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
        img = draw_cities(img, pixel_rings or [[(0,0),(OUT_W,0),(OUT_W,OUT_H),(0,OUT_H)]],
                          cities, geo, _font(28), _font_reg(20))

    img.convert("RGB").save(str(out_path), "JPEG", quality=95)
    return True


# ── Google Earth-Style Raycaster ──────────────────────────────────────────────

def _normalize(v):
    if v.ndim == 1:
        return v / (np.linalg.norm(v) + 1e-30)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-30)

def _lat_lon_to_ecef(lat_deg, lon_deg, alt_km=0.0):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    r = 6371.0 + alt_km
    return np.array([
        r * np.cos(lat) * np.cos(lon),
        r * np.cos(lat) * np.sin(lon),
        r * np.sin(lat),
    ])

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

    return (
        wx0[:, :, None] * wy0[:, :, None] * c00 +
        wx1[:, :, None] * wy0[:, :, None] * c10 +
        wx0[:, :, None] * wy1[:, :, None] * c01 +
        wx1[:, :, None] * wy1[:, :, None] * c11
    )

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
    u = np.linspace(-1, 1, out_w, dtype=np.float32)
    v = np.linspace(-1, 1, out_h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    half_w = np.tan(fov_h / 2.0)
    half_h = np.tan(fov_v / 2.0)

    ray_dirs = (
        forward[None, None, :]
        + (uu[:, :, None] * half_w) * right[None, None, :]
        - (vv[:, :, None] * half_h) * cam_up[None, None, :]
    )
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

    hit_points = C[None, None, :] + t_hit[:, :, None] * D
    hit_lat, hit_lon = _ecef_to_lat_lon(hit_points)

    colors = _sample_atlas_bilinear(atlas_arr, bounds, hit_lat, hit_lon)
    colors_f = colors.astype(np.float32)
    gray = np.mean(colors_f, axis=-1, keepdims=True)
    
    # Vibrant grading
    colors_f = colors_f + (colors_f - gray) * 0.45
    c_norm = np.clip(colors_f / 255.0, 0, 1)
    c_norm = c_norm * c_norm * (3.0 - 2.0 * c_norm)
    colors_f = np.clip(c_norm ** 0.85 * 255.0, 0, 255)

    output = np.zeros((out_h, out_w, 3), dtype=np.float32)
    output[hit_mask] = colors_f[hit_mask]

    # Atmosphere
    closest_dist_sq = dot_CC - dot_CD**2 / (dot_DD + 1e-30)
    closest_dist = np.sqrt(np.maximum(closest_dist_sq, 0))
    atmo_outer = EARTH_R + ATMO_THICKNESS * 0.6
    limb_mask = (~hit_mask) & (closest_dist < atmo_outer) & (dot_CD < 0)
    
    atmo_raw = np.zeros_like(closest_dist)
    if np.any(limb_mask):
        norm = (closest_dist[limb_mask] - EARTH_R) / (ATMO_THICKNESS * 0.6)
        atmo_raw[limb_mask] = np.clip(1.0 - norm, 0, 1) ** 3.0

    limb_bright = np.array([230, 245, 255], dtype=np.float32)
    limb_deep   = np.array([ 10,  25,  60], dtype=np.float32)

    atmo_color = atmo_raw[:, :, None] * (atmo_raw[:, :, None] * limb_bright[None, None, :] + (1 - atmo_raw[:, :, None]) * limb_deep[None, None, :])
    output[~hit_mask] = atmo_color[~hit_mask]

    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))


def gen_google_earth_pan(geo, out_path, duration=6.0, altitude=200.0, pitch=-35.0, bearing_sweep=20.0):
    """Pure Python Google Earth-style panoramic sweep."""
    import numpy as np
    
    lat = geo.get("lat", 0)
    lon = geo.get("lon", 0)
    
    # The map script already downloaded a grid of tiles for us, but it might not be big enough
    # or high res enough. Let's rely on geo['satellite_map'] if it exists, but the original 
    # google_earth_generator explicitly fetched zoom 9 tiles. We will use the existing tiles 
    # if possible by relying on the composite function.
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = out_path.parent / f"_ge_{out_path.stem}"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Fast tile fetcher at zoom 9 (similar to what was in deleted file)
    zoom = 9
    radius = 15
    n = 2 ** zoom
    lat_rad = math.radians(lat)
    cx = int((lon + 180.0) / 360.0 * n) % n
    cy = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)
    cy = max(0, min(n - 1, cy))
    
    print(f" [google-earth: fetching {radius*2+1}x{radius*2+1} tiles...]", end="", flush=True)
    from concurrent.futures import ThreadPoolExecutor
    
    def _dl(tx, ty):
        cache = CACHE_DIR / f"{zoom}_{tx}_{ty}.jpg"
        if cache.exists():
            try: return (tx, ty), Image.open(cache).convert("RGB")
            except: pass
        url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ty}/{tx}"
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "broll/1.0"}), timeout=5)
            cache.parent.mkdir(parents=True, exist_ok=True)
            data = r.read()
            cache.write_bytes(data)
            return (tx, ty), Image.open(io.BytesIO(data)).convert("RGB")
        except: return (tx, ty), None

    grid = radius * 2 + 1
    atlas = Image.new("RGB", (grid * 256, grid * 256), (8, 15, 30))
    with ThreadPoolExecutor(max_workers=24) as pool:
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
        t = i / max(1, total_frames - 1)
        ease = t * t * (3 - 2 * t)
        bearing = 0.0 + bearing_sweep * ease
        
        img = _render_earth_shot(atlas_arr, bnds, lat, lon, out_w=OUT_W, out_h=OUT_H, 
                                 altitude_km=altitude, pitch_deg=pitch, bearing_deg=bearing)
        img = img.filter(ImageFilter.SHARPEN)
        img.save(frames_dir / f"frame_{i:04d}.jpg", quality=90)
        
    ok = run_ffmpeg([
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", str(frames_dir / "frame_%04d.jpg"),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-r", str(FPS), str(out_path)
    ], timeout=180)
    
    import shutil
    shutil.rmtree(str(frames_dir), ignore_errors=True)
    return ok



# ── 3D cinematic B-roll via Mapbox Static API ─────────────────────────────────
try:
    from gen import gen_3d_shot, gen_comparison_map
    _HAS_3D_RENDERER = True
except ImportError:
    _HAS_3D_RENDERER = False
    print("[WARN] gen.py not importable - 3D rendering disabled")


def gen_3d_gtazoom(geo, out_path, dur):
    """GTA-style zoom from space to ground using Cesium and Google 3D Tiles."""
    lat = geo.get("lat", 0)
    lon = geo.get("lon", 0)
    frames_dir = out_path.parent / f"_gtazoom_{out_path.stem}"
    geography_dir = Path(__file__).resolve().parent
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f" [cinematic-gtazoom:{lat:.2f}, {lon:.2f}]", end="", flush=True)
    
    # Run the Node.js capture script
    cmd = ["node", "render_cesium.js", str(lon), str(lat), str(dur), str(frames_dir)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(geography_dir))
        if r.returncode != 0:
            print(f" [gtazoom-fail:{r.stderr[-100:]}]", end="", flush=True)
            return False
    except Exception as e:
        print(f" [gtazoom-fail:{e}]", end="", flush=True)
        return False

    # Stitched frames to MP4
    ok = run_ffmpeg([
        "ffmpeg", "-y", "-framerate", "30",
        "-i", str(frames_dir / "frame_%04d.jpg"),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-r", "30", str(out_path)], timeout=120)

    # Cleanup
    import shutil
    shutil.rmtree(str(frames_dir), ignore_errors=True)
    return ok


def _gen_3d(geo, out_path, dur, animation):
    """Wrapper to call gen_3d_shot with geo data, highlight, and annotations."""
    if not _HAS_3D_RENDERER:
        return False
    lat = geo.get("lat", 0)
    lon = geo.get("lon", 0)
    run_dir = str(out_path.parent.parent)
    # Extract beat_id from filename like beat03_3d_orbit.mp4
    stem = out_path.stem
    bid = 0
    if stem.startswith("beat"):
        try:
            bid = int(stem[4:6])
        except (ValueError, IndexError):
            pass
    result = gen_3d_shot(
        beat_id=bid, lon=lon, lat=lat,
        animation=animation, run_dir=run_dir,
        duration=dur,
        highlight=geo.get("_beat_highlight"),
        annotations=geo.get("_beat_annotations", []),
    )
    if result and Path(result).exists():
        # Move to expected output path if different
        if str(Path(result).resolve()) != str(out_path.resolve()):
            import shutil
            shutil.move(result, str(out_path))
        return True
    return False


# B-roll type -> generator function mapping
BROLL_GENERATORS = {
    "satellite_pan":   lambda geo, out, dur: gen_satellite_pan(geo, out, dur),
    "map_highlight":   lambda geo, out, dur: gen_map_highlight(geo, out) or True,
    "dark_cutout":     lambda geo, out, dur: gen_dark_cutout(geo, out, dur),
    "map_wipe":        lambda geo, out, dur: gen_map_wipe(geo, out, dur),
    "outline_reveal":  lambda geo, out, dur: gen_outline_reveal(geo, out, dur),
    "terrain_map":     lambda geo, out, dur: gen_terrain_map(geo, out) or True,
    "wikipedia_image": lambda geo, out, dur: gen_wikipedia_image(geo, geo.get("_beat", {}), out, dur) or True,
    "google_earth_pan":lambda geo, out, dur: gen_google_earth_pan(geo, out, dur),
    "3d_orbit":        lambda geo, out, dur: _gen_3d(geo, out, dur, "orbit"),
    "3d_flyover":      lambda geo, out, dur: _gen_3d(geo, out, dur, "flyover"),
    "3d_zoom":         lambda geo, out, dur: _gen_3d(geo, out, dur, "zoom_in"),
    "3d_close_oblique": lambda geo, out, dur: _gen_3d(geo, out, dur, "close_oblique"),
    "3d_gtazoom":      lambda geo, out, dur: gen_3d_gtazoom(geo, out, dur),
    # real_concept and comparison_map are handled specially in generate_assets
}


def _apply_highlight_to_still(still_path: Path, beat: Dict, geo: Dict) -> bool:
    """If beat has a highlight object, fetch feature geometry and draw overlay."""
    highlight = beat.get("highlight")
    if not highlight or not isinstance(highlight, dict):
        return False

    query = highlight.get("query", "")
    color = highlight.get("color", "#44AAFF")
    htype = highlight.get("type", "area")
    if not query:
        return False

    feature_rings = fetch_feature_geometry(query, geo)
    if not feature_rings:
        return False

    is_line = htype in ("river", "coastline", "line")
    try:
        img = Image.open(str(still_path)).convert("RGBA")
        img = draw_feature_highlight(img, feature_rings, color_hex=color, is_line=is_line)
        img.convert("RGB").save(str(still_path), "JPEG", quality=95)
        print(f" [highlight:{query}]", end="", flush=True)
        return True
    except Exception as e:
        print(f" [highlight-fail:{e}]", end="", flush=True)
        return False


def _apply_highlight_to_video(video_path: Path, beat: Dict, geo: Dict) -> bool:
    """For video clips, overlay a highlight by rendering highlighted frames.
    Applies to satellite_zoom, satellite_pan, outline_reveal, etc."""
    highlight = beat.get("highlight")
    if not highlight or not isinstance(highlight, dict):
        return False

    query = highlight.get("query", "")
    color = highlight.get("color", "#44AAFF")
    htype = highlight.get("type", "area")
    if not query:
        return False

    feature_rings = fetch_feature_geometry(query, geo)
    if not feature_rings:
        return False

    is_line = htype in ("river", "coastline", "line")
    r_c, g_c, b_c = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

    # Build an FFmpeg overlay filter to draw the highlight.
    # For simplicity, we use drawbox/eq fallback or a PNG overlay approach.
    # The most reliable approach: create a transparent PNG overlay and composite.
    overlay_path = video_path.parent / f"_hl_{video_path.stem}.png"
    overlay_img = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay_img)

    for ring in feature_rings:
        pts = [(int(round(x)), int(round(y))) for x, y in ring]
        if len(pts) < 2:
            continue
        if is_line or len(pts) < 4:
            draw.line(pts, fill=(r_c, g_c, b_c, 40), width=18)
            draw.line(pts, fill=(r_c, g_c, b_c, 80), width=10)
            draw.line(pts, fill=(r_c, g_c, b_c, 220), width=4)
        else:
            if len(pts) >= 3:
                draw.polygon(pts, fill=(r_c, g_c, b_c, 80))
            draw.line(pts + [pts[0]], fill=(r_c, g_c, b_c, 200), width=3)

    overlay_img.save(str(overlay_path), "PNG")

    # Composite the PNG overlay onto the video
    highlighted_path = video_path.parent / f"_hl_out_{video_path.stem}.mp4"
    ok = run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(overlay_path),
        "-filter_complex", "[0:v][1:v]overlay=0:0:format=auto",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-r", str(FPS), "-an",
        str(highlighted_path)
    ], timeout=180)

    overlay_path.unlink(missing_ok=True)

    if ok and highlighted_path.exists():
        # Replace original with highlighted version
        import shutil
        shutil.move(str(highlighted_path), str(video_path))
        print(f" [highlight:{query}]", end="", flush=True)
        return True

    highlighted_path.unlink(missing_ok=True)
    return False


def generate_assets(script: Dict, geo: Dict, run_dir: Path, region: str,
                    allow_real_footage: bool = True) -> Dict[str, Path]:
    """Stage 3: Generate B-roll assets for each beat."""
    print("\n[S3] Generating B-roll assets...")
    clips_dir = run_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    assets = {}  # beat_id -> path

    for beat in script.get("beats", []):
        bid = beat["beat_id"]
        btype = beat["broll_type"]
        dur = beat.get("duration_sec", 5)
        label = f"beat{bid:02d}_{btype}"

        is_video_type = btype in (
            "satellite_pan", "map_wipe", "outline_reveal",
            "dark_cutout",
            "real_city", "real_people", "real_geography", "real_concept",
            "3d_orbit", "3d_flyover", "3d_zoom", "3d_close_oblique",
            "comparison_map",
        )
        # wikipedia_image comes back as a .jpg, outline_reveal returns .mp4, etc.
        ext = ".mp4" if is_video_type else ".jpg"
        out_path = clips_dir / f"{label}{ext}"

        # Skip if a valid (non-empty) file for this beat already exists
        existing = [p for p in clips_dir.glob(f"beat{bid:02d}_*")
                    if not p.name.startswith("_") and p.stat().st_size > 10240]
        if existing:
            assets[bid] = existing[0]
            size_kb = existing[0].stat().st_size / 1024
            print(f"  Beat {bid}: {btype} ({dur}s)... [CACHED] {size_kb:.0f}KB")
            continue

        print(f"  Beat {bid}: {btype} ({dur}s)...", end="", flush=True)

        # Stash beat-level highlight/annotation data into geo for 3D renderer
        geo["_beat"] = beat
        geo["_beat_highlight"] = beat.get("highlight")
        geo["_beat_annotations"] = beat.get("annotations", [])

        ok = False
        is_real = btype in ("real_city", "real_people", "real_geography", "real_concept")

        # Handle comparison_map specially
        if btype == "comparison_map" and _HAS_3D_RENDERER:
            comp_data = beat.get("comparison", {})
            if comp_data and comp_data.get("base") and comp_data.get("highlight"):
                result = gen_comparison_map(
                    beat_id=bid,
                    comparison=comp_data,
                    run_dir=str(run_dir),
                    duration=dur,
                    geo=geo,
                )
                if result and Path(result).exists():
                    out_path = Path(result)
                    ok = True
            if not ok:
                # Fallback to 3d_flyover if comparison fails
                ok = _gen_3d(geo, out_path, dur, "flyover")
        elif btype == "real_concept" and allow_real_footage:
            # Concept clips use YouTube-only intent search
            ok = gen_real_concept_clip(region, beat, out_path, dur)
        elif is_real and allow_real_footage:
            subject = script.get("subject", "")
            ok = gen_real_youtube_clip(region, beat, out_path, dur, subject)

        if is_real and not ok:
            fallbacks = {
                "real_geography": ["3d_flyover", "outline_reveal"],
                "real_city":      ["3d_orbit", "map_wipe"],
                "real_people":    ["3d_zoom", "dark_cutout"],
                "real_concept":   ["real_geography", "3d_flyover"],
            }
            for fb_type in fallbacks.get(btype, ["3d_orbit", "satellite_pan"]):
                fb_gen = BROLL_GENERATORS.get(fb_type)
                if fb_gen:
                    fb_ext = ".mp4" if fb_type in (
                        "satellite_pan", "map_wipe",
                        "outline_reveal", "dark_cutout",
                        "3d_orbit", "3d_flyover", "3d_zoom", "3d_curvature") else ".jpg"
                    fb_path = clips_dir / f"beat{bid:02d}_{fb_type}{fb_ext}"
                    ok = fb_gen(geo, fb_path, dur)
                    if ok and fb_path.exists():
                        out_path = fb_path
                        print(f" [fallback->{fb_type}]", end="", flush=True)
                        break
        elif not is_real:
            gen = BROLL_GENERATORS.get(btype)
            if gen:
                ok = gen(geo, out_path, dur)

        # Apply feature highlight overlay if beat has one
        if ok and out_path.exists() and beat.get("highlight"):
            if out_path.suffix.lower() == ".mp4":
                _apply_highlight_to_video(out_path, beat, geo)
            else:
                _apply_highlight_to_still(out_path, beat, geo)

        if ok:
            assets[bid] = out_path
            size_kb = out_path.stat().st_size / 1024 if out_path.exists() else 0
            print(f" [OK] {size_kb:.0f}KB")
        else:
            print(f" [SKIP] no generator for {btype}")

    print(f"  Total assets: {len(assets)}/{len(script.get('beats', []))}")
    return assets


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4: CAPTION BURN + CLIP ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

def still_to_video(still_path: Path, video_path: Path, duration: float) -> bool:
    """Convert a still image to a video with a very slight zoom-in for visual life."""
    total_n = int(duration * FPS)
    # Scale to 1.06x for slight zoom headroom
    pw = int(OUT_W * 1.06)
    ph = int(OUT_H * 1.06)
    prescale = video_path.parent / f"_pre_still_{still_path.stem}.jpg"
    if not run_ffmpeg([
        "ffmpeg", "-y", "-i", str(still_path),
        "-vf", f"scale={pw}:{ph}:force_original_aspect_ratio=increase,crop={pw}:{ph}",
        str(prescale)], timeout=30):
        # Fallback: just encode as-is without zoom
        return run_ffmpeg([
            "ffmpeg", "-y",
            "-loop", "1", "-framerate", str(FPS), "-i", str(still_path),
            "-vf", f"scale={OUT_W}:{OUT_H},format=yuv420p",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
            "-r", str(FPS), "-frames:v", str(total_n),
            str(video_path)], timeout=60)
    # Gentle zoom-in: crop shrinks from full 1.06x to 1x over duration
    vf = (
        f"crop=w='trunc({pw}*(1-n/{total_n}*0.04))':"
        f"h='trunc({ph}*(1-n/{total_n}*0.04))':"
        f"x='trunc(({pw}-trunc({pw}*(1-n/{total_n}*0.04)))/2)':"
        f"y='trunc(({ph}-trunc({ph}*(1-n/{total_n}*0.04)))/2)',"
        f"scale={OUT_W}:{OUT_H},format=yuv420p"
    )
    ok = run_ffmpeg([
        "ffmpeg", "-y",
        "-loop", "1", "-framerate", str(FPS), "-i", str(prescale),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
        "-r", str(FPS), "-frames:v", str(total_n),
        str(video_path)], timeout=60)
    prescale.unlink(missing_ok=True)
    return ok


def assemble_final(script: Dict, assets: Dict[int, Path], run_dir: Path,
                   vo_path: Optional[Path] = None,
                   whisper_segs: Optional[List[Dict]] = None) -> Optional[Path]:
    """Stage 4+5: Burn captions onto each clip, then concat into final Short."""
    print("\n[S4] Burning captions + assembling final video...")

    final_clips_dir = run_dir / "final_clips"
    final_clips_dir.mkdir(parents=True, exist_ok=True)

    ordered_clips: List[Path] = []

    for beat in script.get("beats", []):
        bid = beat["beat_id"]
        dur = beat.get("duration_sec", 5)

        asset_path = assets.get(bid)
        if not asset_path or not asset_path.exists():
            print(f"  Beat {bid}: [SKIP] no asset")
            continue

        clip_name = f"final_beat{bid:02d}.mp4"
        clip_path = final_clips_dir / clip_name

        is_video = asset_path.suffix.lower() == ".mp4"

        if is_video:
            # Stage 4 now keeps visuals clean; word-by-word captions are applied later from Whisper timestamps.
            ok = run_ffmpeg([
                "ffmpeg", "-y", "-i", str(asset_path),
                "-t", str(dur),
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-r", str(FPS), "-an", str(clip_path)
            ], timeout=120)

        else:
            # For still images, convert to motion video without pre-burned text.
            ok = still_to_video(asset_path, clip_path, dur)

        if ok and clip_path.exists():
            ordered_clips.append(clip_path)
            print(f"  Beat {bid}: [OK] {clip_path.name}")
        else:
            print(f"  Beat {bid}: [FAIL] caption/encode failed")

    if not ordered_clips:
        print("  [FAIL] No clips to assemble!")
        return None

    # Title card disabled — kills retention in first 2s with dead black screen.
    # The satellite zoom hook (beat 1) serves as the opener instead.
    # title_card = run_dir / "final_clips" / "_title.jpg"
    # title = script.get("title", "Geography Explained!")
    # _make_title_card(title, title_card)
    # title_vid = run_dir / "final_clips" / "final_beat00_title.mp4"
    # still_to_video(title_card, title_vid, 2.5)
    # if title_vid.exists():
    #     ordered_clips.insert(0, title_vid)
    # title_card.unlink(missing_ok=True)

    # Concat all clips with crossfade transitions
    final_out = run_dir / "final_short.mp4"
    print(f"\n[S5] Concatenating {len(ordered_clips)} clips with fade transitions...")

    FADE_DUR = 0.3  # seconds of crossfade between clips

    if len(ordered_clips) == 1:
        # Single clip — just copy
        import shutil
        shutil.copy2(str(ordered_clips[0]), str(final_out))
        ok = True
    elif len(ordered_clips) == 2:
        # Two clips — single xfade
        ok = run_ffmpeg([
            "ffmpeg", "-y",
            "-i", str(ordered_clips[0]),
            "-i", str(ordered_clips[1]),
            "-filter_complex",
            f"[0:v][1:v]xfade=transition=fade:duration={FADE_DUR}:offset={{}}".format(
                max(0.1, float(script['beats'][0].get('duration_sec', 4)) - FADE_DUR)),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-r", str(FPS), "-an",
            str(final_out)], timeout=180)
    else:
        # N clips — chain xfade filters
        inputs = []
        for p in ordered_clips:
            inputs.extend(["-i", str(p)])

        # Build xfade filter chain:
        # [0:v][1:v]xfade=...offset=O1[v1]; [v1][2:v]xfade=...offset=O2[v2]; ...
        # Offset for xfade i = (sum of durations of clips 0..i) - (i * FADE_DUR) - FADE_DUR
        # Because each xfade shortens the timeline by FADE_DUR.
        filter_parts = []

        # Get actual clip durations by probing each file with ffprobe
        clip_durations = []
        for cp in ordered_clips:
            try:
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(cp)],
                    capture_output=True, text=True, timeout=10)
                dur = float(probe.stdout.strip())
            except Exception:
                dur = 6.0
            clip_durations.append(dur)
        print(f"  Clip durations (probed): {[f'{d:.2f}s' for d in clip_durations]}")

        for i in range(len(ordered_clips) - 1):
            # Offset = sum of first (i+1) clip durations minus (i+1)*FADE_DUR
            # This is where the (i+1)-th xfade begins in the output timeline
            offset = sum(clip_durations[:i+1]) - (i + 1) * FADE_DUR
            offset = max(0.1, offset)

            in_a = f"[{i}:v]" if i == 0 else f"[v{i}]"
            in_b = f"[{i+1}:v]"
            out_label = f"[v{i+1}]"
            # Last xfade doesn't need output label for final output
            if i == len(ordered_clips) - 2:
                out_label = ""
            filter_parts.append(
                f"{in_a}{in_b}xfade=transition=fade:duration={FADE_DUR}:offset={offset:.2f}{out_label}"
            )

        filter_complex = ";".join(filter_parts)

        ok = run_ffmpeg([
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-r", str(FPS), "-an",
            str(final_out)], timeout=300)

    if not (ok and final_out.exists()):
        print("  [FAIL] Final assembly failed")
        return None

    mb = final_out.stat().st_size / 1048576
    print(f"  [OK] {final_out.name} ({mb:.1f} MB, silent)")

    # ── Stage 5b: Title overlay (prompt mode) ────────────────────────
    hook_question = script.get("hook_question") or script.get("title", "")
    if hook_question and script.get("_source") == "gemini_topic":
        print("\n[S5b] Burning title overlay...")
        titled_out = run_dir / "final_short_titled.mp4"
        if burn_title_overlay(final_out, hook_question, titled_out, display_duration=4.0):
            print(f"  [OK] Title overlay: '{hook_question[:50]}...'")
            final_out = titled_out
        else:
            print("  [WARN] Title overlay failed, continuing without it")

    # ── Captions from Whisper (passed in from main) ──────────────────
    if vo_path and whisper_segs:
        print("\n[S6c] Burning word-by-word captions from Whisper...")
        ass_text = generate_ass_captions_from_whisper(whisper_segs)
        captioned_final = run_dir / "final_short_captions.mp4"
        ass_path = run_dir / "captions.ass"
        if ass_text:
            ass_path.write_text(ass_text, encoding="utf-8-sig")
            if burn_ass_captions(final_out, ass_path, captioned_final) and captioned_final.exists():
                final_out = captioned_final
                print(f"  [OK] {captioned_final.name}")
            else:
                print("  [WARN] Whisper caption burn failed, using uncaptioned final_short.mp4")
        else:
            print("  [WARN] No Whisper word data for captions")

        print("\n[S7] Mixing audio...")
        import random
        music_files = list(MUSIC_DIR.glob("*.mp3"))
        music = random.choice(music_files) if music_files else None
        if music:
            print(f"  [OK] Selected random music: {music.name}")

        final_with_audio = mix_voiceover(final_out, vo_path, run_dir, music)
        if final_with_audio:
            return final_with_audio
        else:
            print("  Audio mix failed, returning silent video")

    return final_out


def _make_title_card(title: str, out_path: Path):
    """Generate a title card frame (dark bg, big text)."""
    img = Image.new("RGB", (OUT_W, OUT_H), (12, 12, 18))
    draw = ImageDraw.Draw(img)
    font = _font(64)

    # Word wrap
    words = title.split()
    lines = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 > 20:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        lines.append(cur)

    total_h = sum(draw.textbbox((0, 0), l, font=font)[3] - draw.textbbox((0, 0), l, font=font)[1] + 20
                  for l in lines)
    y = (OUT_H - total_h) // 2

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = (OUT_W - tw) // 2
        draw.text((x + 3, y + 3), line, font=font, fill=(0, 0, 0))
        draw.text((x, y), line, font=font, fill=(255, 255, 255))
        y += th + 20

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), "JPEG", quality=95)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def slugify(s):
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")[:40]


def main():
    parser = argparse.ArgumentParser(description="Geography Short Maker")
    parser.add_argument("--region", default="", help="Region name, e.g. 'Nova Scotia, Canada'")
    parser.add_argument("--prompt", default="",
                        help="Topic-driven prompt, e.g. 'This is the widest river in the world'")
    parser.add_argument("--resume", default="", help="Resume from existing run directory")
    parser.add_argument("--stage", default="all", choices=["all", "s1", "s2", "s3", "s4"],
                        help="Run specific stage or all")
    parser.add_argument("--disable-real-footage", action="store_true",
                        help="Disable YouTube real footage sourcing and use only generated map/satellite visuals")
    parser.add_argument("--voice", default="",
                        help="Path to a custom .mp3 or .wav file to clone for the voiceover")
    args = parser.parse_args()

    # Validate: need either --region or --prompt
    if not args.region and not args.prompt:
        parser.error("Either --region or --prompt is required")

    prompt_mode = bool(args.prompt)
    prompt_text = args.prompt.strip() if args.prompt else ""
    region = args.region.strip() if args.region else ""

    # For run directory naming
    run_label = prompt_text or region
    run_id = slugify(run_label)
    run_dir = RUNS_DIR / run_id if not args.resume else RUNS_DIR / args.resume
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  GEO SHORT MAKER")
    if prompt_mode:
        print(f"  Prompt: {prompt_text}")
    else:
        print(f"  Region: {region}")
    print(f"  Run:    {run_dir.name}")
    print(f"{'='*60}")

    # Stage 1: Script
    s1_path = run_dir / "s1_script.json"
    if args.stage in ("all", "s1") or not s1_path.exists():
        if prompt_mode:
            script = generate_topic_script(prompt_text, run_dir)
            # Extract region from Gemini output for geo stage
            if not region:
                region = script.get("region", "")
                print(f"  Extracted region: {region}")
        else:
            script = generate_script(region, run_dir)
    else:
        script = json.loads(s1_path.read_text(encoding="utf-8"))
        print(f"\n[S1] Loaded existing script ({len(script.get('beats', []))} beats)")
        if not region:
            region = script.get("region", "")

    # Stage 2: Geodata
    if not region:
        print("  [FATAL] No region available for geodata stage")
        return

    s2_path = run_dir / "s2_geodata.json"
    if args.stage in ("all", "s2") or not s2_path.exists():
        geo = gather_geo_data(region, run_dir, script)
    else:
        geo = json.loads(s2_path.read_text(encoding="utf-8"))
        print(f"\n[S2] Loaded existing geodata")
        # Rebuild rings from boundary if needed
        if "rings" not in geo:
            print("  Re-fetching boundary for pixel rings...")
            _, rings, _ = fetch_boundary(region)
            if rings:
                geo["rings"] = rings
                geo["pixel_rings"] = rings_to_pixels(
                    rings, geo["lat"], geo["lon"], geo["zoom"], geo["cols"], geo["rows"])
                geo["pixel_rings"] = [r for r in geo["pixel_rings"]
                                      if any(0 <= x <= OUT_W and 0 <= y <= OUT_H for x, y in r)]

    # Stage 2b: Audio-First — Generate voiceover BEFORE assets
    #           so clip durations match real spoken audio timing.
    vo_path = None
    whisper_segs = None
    if args.stage in ("all", "s3", "s4"):
        print("\n[S2b] Generating voiceover (audio-first pipeline)...")
        voice_path_ext = Path(args.voice) if args.voice else None
        vo_path = generate_voiceover(script, run_dir, voice_path=voice_path_ext)
        if vo_path:
            print("\n[S2c] Whisper alignment...")
            whisper_segs = run_whisper_alignment(vo_path)
            if whisper_segs:
                (run_dir / "whisper_segments.json").write_text(
                    json.dumps(whisper_segs, indent=2), encoding="utf-8")
                print("\n[S2d] Updating beat durations from spoken audio...")
                script = _update_beat_durations_from_whisper(script, whisper_segs)
                # Save updated script with real durations
                (run_dir / "s1_script.json").write_text(
                    json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            print("  [WARN] Voiceover generation failed, using Gemini-guessed durations")

    # Stage 3: Assets (now uses real audio durations if voiceover succeeded)
    if args.stage in ("all", "s3"):
        assets = generate_assets(
            script, geo, run_dir,
            region=region,
            allow_real_footage=not args.disable_real_footage,
        )
    else:
        # Scan existing
        clips_dir = run_dir / "clips"
        assets = {}
        if clips_dir.exists():
            for beat in script.get("beats", []):
                bid = beat["beat_id"]
                for ext in (".mp4", ".jpg"):
                    p = clips_dir / f"beat{bid:02d}_{beat['broll_type']}{ext}"
                    if p.exists():
                        assets[bid] = p
        print(f"\n[S3] Found {len(assets)} existing assets")

    # Stage 4+5: Captions + Assembly (receives voiceover + whisper data)
    if args.stage in ("all", "s4"):
        final = assemble_final(script, assets, run_dir,
                               vo_path=vo_path, whisper_segs=whisper_segs)
    else:
        final = None

    # Summary
    summary = {
        "region": region,
        "prompt": prompt_text if prompt_mode else None,
        "run_dir": str(run_dir),
        "script_beats": len(script.get("beats", [])),
        "assets_generated": len(assets),
        "final_video": str(final) if final else None,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  COMPLETE")
    print(f"  Beats:  {summary['script_beats']}")
    print(f"  Assets: {summary['assets_generated']}")
    if final:
        print(f"  Final:  {final.name}")
    print(f"  Dir:    {run_dir}")
    print(f"{'='*60}")
    
    # Extract and save YouTube metadata
    metadata = script.get("youtube_metadata", {})
    if metadata:
        meta_file = run_dir / "youtube_description.txt"
        title = metadata.get("title", f"{region} Geography")
        desc = metadata.get("description", "")
        tags = metadata.get("tags", [])
        
        content = f"Title: {title}\n\n"
        content += f"Description:\n{desc}\n\n"
        if tags:
            content += f"Tags:\n{', '.join(tags)}\n"
            
        meta_file.write_text(content, encoding="utf-8")
        print(f"  -> YouTube metadata saved to: {meta_file.name}\n")


if __name__ == "__main__":
    main()

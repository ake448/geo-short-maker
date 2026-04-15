"""
gemini.py — Stage 1: Gemini script generation + prompt templates.
"""
from __future__ import annotations

import json
import re
import textwrap
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict

from .config import GEMINI_API_KEY, GEMINI_MODEL, _SSL_CTX


def call_gemini(prompt: str, temperature: float = 0.75, tools: list | None = None) -> str:
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
        },
    }
    if not tools:
        payload["generationConfig"]["responseMimeType"] = "application/json"
    else:
        payload["tools"] = tools
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

- Total runtime: 55-72 seconds (target about 60-66 seconds)
- 9-12 beats (each beat = one narration line + one visual clip)
- Beat pacing target: each beat should be 3.0-6.5 seconds
- Beat 1 MUST be a curiosity-driven hook (question or shocking statement)
- Each narration line: 5-13 words MAX
- Narration should sound like a confident narrator, not a textbook. It should pair well with an Attenborough-style voice and dramatic background music.
- Every fact must be real and verifiable.
- Return ONLY valid JSON. Escape all quotes properly. No markdown outside the JSON block.

REQUIRED BEAT STRUCTURE (follow this closely):
  Beat 1 — HOOK: Open with a dramatic map visual (e.g., 3d_orbit, satellite_pan).
            Ask a provocative question or state a shocking fact.
    Beats 2-3 — SCOPE: Show geographic context (e.g., 3d_zoom, outline_reveal).
    Remaining beats — CORE_FACT: The MAJORITY of beats MUST use high-quality real footage (real_geography, real_city, native_animal).
               These must be cinematic, aesthetically pleasing shots (e.g., drone flights over terrain, clean cityscapes,
               skyline flyovers, dense downtown building passes, or dramatic neighborhood descents). 
                             IMPORTANT: Every real_* `visual_note` must be specific and include:
                             (1) shot type (drone top-down / low-altitude pass / night wide / macro / etc),
                             (2) exact subject visible,
                             (3) camera movement,
                             (4) cleanliness requirement: "clean, no text overlays, no logos, no watermarks".
               Use 3d_* or map_* b-roll sparingly here for geographical context.
    Second to last Beat — CONTRAST: Introduce a contrast, pivot, or surprising fact to build tension before the end.
    Beat Last — EXIT: Use a cinematic real_geography shot or 3d_orbit callback to the hook.

BLUE OUTLINE GUIDANCE:
- Suggest `outline_reveal` at most once in the entire beat list, usually in the SCOPE section.
- This creates the cinematic 3D blue outline reveal over satellite imagery.

IMPORTANT: Choose broll_type based ONLY on what the narration is saying — do NOT force any specific type
into any positional slot. If the narration for beat 3 is about a city, use real_city, not a map type.

Available broll_types:

  satellite_pan    = slow pan across terrain (use sparingly)
  map_highlight    = political map with region highlighted in color
  map_wipe         = map overlay wiping L-to-R over satellite (transitions only)
  outline_reveal   = cinematic 3D blue outline reveal on a curved satellite map (BEST for showing borders)
  terrain_map      = terrain/relief map centered on region
  real_city        = real footage of city streets / downtown / skyline (HIGHLY PREFERRED)
    real_geography   = real footage of nature/terrain/coast/mountains (HIGHLY PREFERRED)
    native_animal    = real footage of wildlife native to the region in natural habitat (HIGHLY PREFERRED)
  3d_orbit         = cinematic 3D satellite orbit around the region (BEST for establishing shots)
  3d_flyover       = cinematic 3D flyover at an oblique angle
  3d_zoom          = zooming from space down to the region in 3D
  3d_close_oblique = dramatic close-up tilted view looking across terrain (like Google Earth)
  zoom_to_region   = exponential tile-based zoom from space → region surface (cinematic, no 3D engine needed)
  chokepoint_flow  = animated trade-route arrows streaming through a strait or canal
  region_comparison = two regions drawn side-by-side at the same km scale (e.g. "Qatar vs Connecticut")
  size_comparison_overlay = Region B slides in over Region A at honest scale (iconic geography shorts technique)
  multi_region_reveal = sequential choropleth — multiple countries/regions light up one by one
  stat_counter_clip = full-clip animated number counting up to a stat, over glowing map background

Optional beat field:
  render_mode      = "day" | "night" | "auto"
                     Use "night" for beats about city lights, nighttime population clusters, or illuminated coastlines.
                     Use "day" for bright satellite or terrain context. Default is "auto".


SOURCING INTENT RULES (CRITICAL):
- NEVER output resolved URLs (`youtube_url`) or start timestamps.
- For every real_* beat, include BOTH a `search_intent` object AND a `youtube_queries` list.
- `youtube_queries` is a list of 4-6 simple search strings — exactly what a human would type into
  YouTube to find real footage of this beat's subject. Be specific and think about what actually
  exists on YouTube. Match the query to the EXACT beat subject AND the correct camera perspective:
    - traffic/sprawl beat → ["city name traffic", "city name highway drone", "city name aerial sprawl"]
    - gridlock/congestion/commute → ["city name traffic jam", "city name rush hour dashcam", "city name gridlock", "city name highway standstill"]
    - crime/rough area → ["city name hood", "city name ghetto", "neighborhood name", "city name abandoned"]
    - construction/infrastructure → ["city name highway construction", "city name roadwork", "city name bridge construction timelapse"]
    - culture/food/entertainment → ["city name beltline walk", "city name downtown street walk", "city name neighborhood", "city name food market"]
    - nature/terrain → ["region name drone aerial", "landmark name 4k", "state name nature wildlife"]
    - beach/coast → ["beach name drone", "city name beach aerial 4k"]
    - downtown/skyline → ["city name downtown drone", "city name skyline", "bridge name city name"]
    - specific landmark → use the actual landmark name + city + drone/aerial/footage
    - border/boundary topics → search BOTH countries + nearest major cities: ["country A drone 4k", "country B aerial", "nearest city drone", "border region name"]
  PERSPECTIVE RULE: Match camera angle to the beat's emotional tone.
    - Frustration/gridlock/congestion → dashcam, POV, street-level (not aerial — aerial makes traffic look smooth)
    - Beauty/scale/geography → drone/aerial
    - Culture/food/community → street walk, neighborhood, human scale
    - Construction/reality/problems → ground-level or timelapse, not drone
  Include local names, nicknames, bridge names, neighborhood names, highway numbers.
  Keep queries 2-5 words. No filler words. Think like a local searching for footage.
  CRITICAL RULES:
    - Search for FOOTAGE that exists on YouTube, not concepts or explanations. BAD: "Radcliffe Line map", "border explained", "geographic borders resolved". GOOD: "Bangladesh drone 4k", "Kolkata aerial", "India border village".
    - At least 2 of your 6 queries must be BROAD geographic queries that are guaranteed to have footage: "COUNTRYNAME drone 4k", "MAJORCITY aerial cinematic", "REGIONNAME landscape 4k".
    - NEVER search for maps, animations, infographics, or explainer videos — we need raw footage only.
- `search_intent.visual_description` must describe camera angle + subject + movement.
- `search_intent.required_geography` must name the SPECIFIC location (neighborhood/landmark/feature).
- `search_intent.geography_strictness` must be `strict` for exact-place beats, `loose` for regional/biome substitutes.
- `search_intent.fallback_allowed` must be one of `terrain_map|3d_orbit`.
- `search_intent.biome_hint` should be biome/environment wording (not city names), e.g. "southeastern US pine forest".

VISUAL VARIETY RULES:
- The MAJORITY of beats MUST use real_* broll types.
- At least 70% of beats should be real_* footage.
- Ensure `visual_note` for real_* beats includes shot type + exact subject + camera movement + "clean, no text overlays, no logos, no watermarks, high quality cinematic".
- Do not use the exact same `broll_type` (except real_* variants) two beats in a row.
- Do not use multiple map/orbit/outline beats back-to-back.
- Use map-style visuals sparingly: usually 1 early context moment and at most 1 later explanatory/payoff map moment.
- Avoid filler visuals: do NOT use generic stock concepts for crucial narrative beats.
- Do NOT use `real_concept`; convert explanatory beats into specific real_city or real_geography shots tied to the actual place.
- `map_wipe` is only allowed for geographic transitions, not event explanation beats.
- NOT EVERY BEAT NEEDS A DRONE SHOT. Use the camera level that matches what the script is saying.

OVERLAY RULES:
- Use `overlays` — a JSON array of 0, 1, or 2 overlay objects per beat (never more than 2).
- Most beats should have `"overlays": []` (empty).
- Add 1 overlay to CORE_FACT beats that state a verifiable statistic, ranking, or named reference worth visualizing.
- Optionally include a 2nd overlay on ~25% of fact-heavy beats, staggered 2-3 seconds after the first (use a different `at_sec`).
- Overlay types:
    - `context_photo` (alias `historical_photo`): provide `wikipedia_title` (exact Wikipedia article title) or `image_query`. Use `bw`/`sepia` only for historical content; otherwise `light` or `dark`.
    - `stat_counter`: animated statistic display. Pick stats that make viewers say "wait, really?" — surprising or counterintuitive beats only. Prefer comparisons ("3x more than France") over raw numbers.
      - `stat_category`: one of `big_number`, `percentage`, `ranking`, `comparison`, `measurement`, `year`, `ratio`
      - `stat_number`: the numeric value (e.g. 686, 45, 3, 1954)
      - `stat_prefix`: optional prefix (e.g. "#" for rankings)
      - `stat_suffix`: optional suffix (e.g. "%" for percentages, "x" for comparisons)
      - `stat_label`: short descriptor (e.g. "MIGRANT DEATHS", "SMARTEST DISTRICT IN GA")
      - `stat_context`: optional context line (e.g. "in 2022", "since 1998")
      - `compare_to`: for comparison category — what it's being compared to (e.g. "France", "the US average")
      - Rankings #1-5 get a slam animation (no counting). Higher rankings get a fast ticker. Big numbers count up. Percentages count then snap the %. Years and ratios reveal without counting.
    - `split_comparison`: before/after contrast. Provide `image_query_a`, `image_query_b`, `label_a`, `label_b`.
    - `flow_map`: movement beats (migration, trade routes). Apply on real_geography clips.
- Background clip for any overlay beat MUST be a moving video (real_* or 3d_*), never a still.
- `wikipedia_title` must be the exact page title (e.g. 'Hartsfield–Jackson Atlanta International Airport', not 'Atlanta airport').

Return STRICT JSON:
{{
  "region": "string",
  "location_focus": "string (the SPECIFIC mappable place — a proper noun like 'West Virginia', 'Bangladesh'. NEVER a common word.)",
  "geodata_query": "string (best search term for this place's map boundary — e.g. 'West Virginia', 'Bangladesh')",
  "title": "string (catchy, 3-6 words)",
  "total_duration_sec": number,
  "beats": [
    {{
      "beat_id": 1,
      "script_type": "HOOK|SCOPE|CORE_FACT|CONTRAST|PAYOFF|EXIT",
      "narration": "string (the spoken line)",
      "broll_type": "string (from the list above)",
      "duration_sec": number (5.0-8.0 seconds),
      "visual_note": "string (SPECIFIC description of what viewer sees — include 'clean, no text overlays' for real footage)",
            "youtube_queries": ["query 1", "query 2", "query 3", "query 4"],
            "search_intent": {{
                "visual_description": "string",
                "required_geography": "string",
                "geography_strictness": "strict|loose",
                "fallback_allowed": "terrain_map|3d_orbit",
                "biome_hint": "string"
            }},
      "caption_text": "string (key phrase, ALL CAPS, 2-5 words — visual emphasis only)",
      "overlays": [
        {{
          "type": "context_photo | stat_counter | split_comparison | flow_map",
          "at_sec": number,
          "style": "dark | light | bw | sepia",
          "wikipedia_title": "Exact Wikipedia article title — context_photo",
          "image_query": "fallback search query — context_photo",
          "caption": "label below photo — context_photo",
          "stat_category": "big_number|percentage|ranking|comparison|measurement|year|ratio — stat_counter",
          "stat_number": "number — stat_counter",
          "stat_prefix": "# — stat_counter ranking",
          "stat_suffix": "% or x — stat_counter",
          "stat_label": "SHORT LABEL — stat_counter",
          "stat_context": "in 2022 — stat_counter",
          "compare_to": "France — stat_counter comparison",
          "label": "sub-label — flow_map",
          "image_query_a": "BEFORE search — split_comparison",
          "image_query_b": "AFTER search — split_comparison",
          "label_a": "left label — split_comparison",
          "label_b": "right label — split_comparison"
        }}
      ]
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
  - The REAL geographic place this is about — a specific, mappable location (country, state, city, river, mountain, etc.). This must be a PROPER NOUN that exists on a map, NOT a word from the prompt title. For "Nobody wants to live in this state" → "West Virginia" or "Alaska", NOT "Nobody". For "The poorest reservation in America" → "Pine Ridge Indian Reservation", NOT "poorest".
  - The REGION this is about (e.g. "Quebec, Canada", "Arabian Peninsula", "Eastern Canada")
  - The SUBJECT being discussed (e.g. "St. Lawrence River", "Sahara Desert", "cities")
  - The HOOK QUESTION or HOOK FACT to open with

VIDEO PACING RULES:
- Target a total video length of 45-55 seconds.
- Average beat duration SHOULD be 3-5 seconds for a fast, modern pace.
- Avoid beats longer than 6 seconds; break them into smaller components if needed.
- Each beat MUST feel distinct and contribute a new visual or fact.
- Beat 1 MUST be a curiosity-driven hook (question or shocking statement)
- Each narration line: 8-18 words.
- Narration should be punchy, energetic, and highly engaging. It should pair well with a fast-paced YouTube Shorts narrator (Guy Michaels style). Do NOT use a slow, pausing, or textbook style.
- Every fact must be real and verifiable.
- Return ONLY valid JSON. Escape all quotes properly. No markdown outside the JSON block.

REFERENCE SHORT-FORM GRAMMAR:
- Open immediately. The first line should feel usable in the first 1-2 seconds.
- Use only one quick context/setup beat before moving into real footage.
- By beat 3, the video should usually be on real footage rather than another map angle.
- Map/orbit/outline visuals should be sparse and purposeful, not repeated.

REQUIRED BEAT STRUCTURE:
  Beat 1 — HOOK: Open with a dramatic map visual (e.g., 3d_orbit, satellite_pan). Restate the user's question/fact as a hook.
    Beats 2-3 — SCOPE: Show geographic context (e.g., 3d_zoom, map_highlight, outline_reveal).
    Remaining beats — CORE_FACT: The MAJORITY of beats MUST use high-quality real footage (real_geography, real_city, native_animal).
               These must be cinematic, aesthetically pleasing shots (e.g., drone flights over terrain, clean cityscapes,
               skyline flyovers, dense downtown building passes, or dramatic neighborhood descents).
                             IMPORTANT: Every real_* `visual_note` must be specific and include:
                             (1) shot type (drone top-down / low-altitude pass / night wide / macro / etc),
                             (2) exact subject visible,
                             (3) camera movement,
                             (4) cleanliness requirement: "clean, no text overlays, no logos, no watermarks".
               Use 3d_* or map_* b-roll sparingly here for geographical context.
               When discussing a SPECIFIC geographic feature (river, lake, etc.), include a "highlight" object in that beat to visually call it out.
    Second to last Beat — CONTRAST: Introduce a contrast, pivot, or surprising fact to build tension before the end.
    Beat Last+ — EXIT: Use a cinematic real_geography shot or 3d_orbit callback to the hook with a satisfying answer.

BLUE OUTLINE GUIDANCE:
- Suggest `outline_reveal` at most once in the entire beat list, and when used make it a blue boundary outline moment.

Available broll_types:

  satellite_pan    = slow pan across terrain (use sparingly)
  map_highlight    = political map with region highlighted in color
  map_wipe         = map overlay wiping L-to-R over satellite
  outline_reveal   = cinematic 3D blue outline reveal on a curved satellite map (BEST for showing borders)
  terrain_map      = terrain/relief map centered on region
  real_city        = real footage of city streets / downtown / skyline (HIGHLY PREFERRED)
    real_geography   = real footage of nature/terrain/coast/mountains (HIGHLY PREFERRED)
    native_animal    = real footage of wildlife native to the region in natural habitat (HIGHLY PREFERRED)
  3d_orbit         = cinematic 3D satellite orbit around the region (BEST for establishing shots)
  3d_flyover       = cinematic 3D flyover at an oblique angle
  3d_zoom          = zooming from space down to the region in 3D
  3d_close_oblique = dramatic close-up tilted view looking across terrain (like Google Earth)
  zoom_to_region   = exponential tile-based zoom from space → region surface (cinematic, no 3D engine needed)
  chokepoint_flow  = animated arrows streaming through a strait, canal, or border crossing
  region_comparison = two regions side-by-side at same km scale — requires beat.comparison object
  size_comparison_overlay = Region B slides in over Region A at honest scale — requires beat.comparison object
  multi_region_reveal = sequential choropleth: multiple countries light up one by one — requires beat.multi_regions list
  stat_counter_clip = animated number counting up over a map — requires beat.stat_value, stat_label, stat_unit
  comparison_map   = animated comparison overlay: highlights a sub-region vs the whole country
                     Requires a "comparison" object in the beat.

SOURCING INTENT RULES (CRITICAL):
- NEVER output resolved URLs (`youtube_url`) or start timestamps.
- For every real_* beat, include BOTH a `search_intent` object AND a `youtube_queries` list.
- `youtube_queries` is a list of 4-6 simple search strings — exactly what a human would type into
  YouTube to find real footage of this beat's subject. Be specific and think about what actually
  exists on YouTube. Match the query to the EXACT beat subject AND the correct camera perspective:
    - traffic/sprawl beat → ["city name traffic", "city name highway drone", "city name aerial sprawl"]
    - gridlock/congestion/commute → ["city name traffic jam", "city name rush hour dashcam", "city name gridlock", "city name highway standstill"]
    - crime/rough area → ["city name hood", "city name ghetto", "neighborhood name", "city name abandoned"]
    - construction/infrastructure → ["city name highway construction", "city name roadwork", "city name bridge construction timelapse"]
    - culture/food/entertainment → ["city name beltline walk", "city name downtown street walk", "city name neighborhood", "city name food market"]
    - nature/terrain → ["region name drone aerial", "landmark name 4k", "state name nature wildlife"]
    - beach/coast → ["beach name drone", "city name beach aerial 4k"]
    - downtown/skyline → ["city name downtown drone", "city name skyline", "bridge name city name"]
    - specific landmark → use the actual landmark name + city + drone/aerial/footage
    - border/boundary topics → search BOTH countries + nearest major cities: ["country A drone 4k", "country B aerial", "nearest city drone", "border region name"]
  PERSPECTIVE RULE: Match camera angle to the beat's emotional tone.
    - Frustration/gridlock/congestion → dashcam, POV, street-level (not aerial — aerial makes traffic look smooth)
    - Beauty/scale/geography → drone/aerial
    - Culture/food/community → street walk, neighborhood, human scale
    - Construction/reality/problems → ground-level or timelapse, not drone
  Include local names, nicknames, bridge names, neighborhood names, highway numbers.
  Keep queries 2-5 words. No filler words. Think like a local searching for footage.
  CRITICAL RULES:
    - Search for FOOTAGE that exists on YouTube, not concepts or explanations. BAD: "Radcliffe Line map", "border explained", "geographic borders resolved". GOOD: "Bangladesh drone 4k", "Kolkata aerial", "India border village".
    - At least 2 of your 6 queries must be BROAD geographic queries that are guaranteed to have footage: "COUNTRYNAME drone 4k", "MAJORCITY aerial cinematic", "REGIONNAME landscape 4k".
    - NEVER search for maps, animations, infographics, or explainer videos — we need raw footage only.
- `search_intent.visual_description` must describe camera angle + subject + movement.
- `search_intent.required_geography` must name the SPECIFIC location (neighborhood/landmark/feature).
- `search_intent.geography_strictness` must be `strict` for exact-place beats, `loose` for regional/biome substitutes.
- `search_intent.fallback_allowed` must be one of `terrain_map|3d_orbit`.
- `search_intent.biome_hint` should be biome/environment wording (not city names), e.g. "southeastern US pine forest".

VISUAL VARIETY RULES:
- The MAJORITY of beats MUST use real_* broll types.
- At least 70% of beats should be real_* footage.
- Ensure `visual_note` for real_* beats includes shot type + exact subject + camera movement + "clean, no text overlays, no logos, no watermarks, high quality cinematic".
- Do not use the exact same `broll_type` (except real_* variants) two beats in a row.
- Avoid filler visuals: do NOT use generic stock concepts for crucial narrative beats.
- Do NOT use `real_concept`; convert explanatory beats into specific real_city or real_geography shots tied to the actual place.
- `map_wipe` is only allowed for geographic transitions, not event explanation beats.
- NOT EVERY BEAT NEEDS A DRONE SHOT. Use the camera level that matches what the script is saying.

OVERLAY RULES:
- Use `overlays` — a JSON array of 0, 1, or 2 overlay objects per beat (never more than 2).
- Most beats should have `"overlays": []` (empty).
- Add 1 overlay to CORE_FACT beats that state a verifiable statistic, ranking, or named reference worth visualizing.
- Optionally include a 2nd overlay on ~25% of fact-heavy beats, staggered 2-3 seconds after the first (use a different `at_sec`).
- Overlay types:
    - `context_photo` (alias `historical_photo`): provide `wikipedia_title` (exact Wikipedia article title) or `image_query`. Use `bw`/`sepia` only for historical content; otherwise `light` or `dark`.
    - `stat_counter`: animated statistic display. Pick stats that make viewers say "wait, really?" — surprising or counterintuitive beats only. Prefer comparisons ("3x more than France") over raw numbers.
      - `stat_category`: one of `big_number`, `percentage`, `ranking`, `comparison`, `measurement`, `year`, `ratio`
      - `stat_number`: the numeric value (e.g. 686, 45, 3, 1954)
      - `stat_prefix`: optional prefix (e.g. "#" for rankings)
      - `stat_suffix`: optional suffix (e.g. "%" for percentages, "x" for comparisons)
      - `stat_label`: short descriptor (e.g. "MIGRANT DEATHS", "SMARTEST DISTRICT IN GA")
      - `stat_context`: optional context line (e.g. "in 2022", "since 1998")
      - `compare_to`: for comparison category — what it's being compared to (e.g. "France", "the US average")
      - Rankings #1-5 get a slam animation (no counting). Higher rankings get a fast ticker. Big numbers count up. Percentages count then snap the %. Years and ratios reveal without counting.
    - `split_comparison`: before/after contrast. Provide `image_query_a`, `image_query_b`, `label_a`, `label_b`.
    - `flow_map`: movement beats (migration, trade routes). Apply on real_geography clips.
- Background clip for any overlay beat MUST be a moving video (real_* or 3d_*), never a still.
- `wikipedia_title` must be the exact page title (e.g. 'Hartsfield–Jackson Atlanta International Airport', not 'Atlanta airport').

Return STRICT JSON:
{{
  "region": "string (the geographic region this is about)",
  "location_focus": "string (the SPECIFIC mappable place — a proper noun like 'West Virginia', 'Pine Ridge Indian Reservation', 'Bangladesh'. NEVER a common English word from the prompt title.)",
  "geodata_query": "string (the best search term for finding this place's boundary on a map — e.g. 'West Virginia', 'Bangladesh', 'Sahara Desert')",
  "subject": "string (the specific feature/topic being discussed)",
  "hook_question": "string (the opening question/statement)",
  "title": "string (catchy, 3-6 words)",
  "total_duration_sec": number,
  "beats": [
    {{
      "beat_id": 1,
      "script_type": "HOOK|SCOPE|CORE_FACT|CONTRAST|PAYOFF|EXIT",
      "narration": "string (the spoken line)",
      "broll_type": "string (from the list above)",
      "duration_sec": number (5.0-8.0 seconds),
      "visual_note": "string (SPECIFIC description of what viewer sees)",
            "youtube_queries": ["query 1", "query 2", "query 3", "query 4"],
            "search_intent": {{
                "visual_description": "string",
                "required_geography": "string",
                "geography_strictness": "strict|loose",
                "fallback_allowed": "terrain_map|3d_orbit",
                "biome_hint": "string"
            }},
      "caption_text": "string (key phrase, ALL CAPS, 2-5 words)",
      "render_mode": "optional: day|night|auto",
            "data_moment": {{
                "kind": "before_after|numeric",
                "label": "string (e.g., 'CRATER SIZE CHANGE')",
                "timepoint_a": "string (e.g., '2010')",
                "timepoint_b": "string (e.g., '2024')",
                "value_a": "string",
                "value_b": "string",
                "source_hint": "string (short citation or source name)"
            }} or null,
      "overlays": [
        {{
          "type": "context_photo | stat_counter | split_comparison | flow_map",
          "at_sec": number,
          "style": "dark | light | bw | sepia",
          "wikipedia_title": "Exact Wikipedia article title — context_photo",
          "image_query": "fallback search query — context_photo",
          "caption": "label below photo — context_photo",
          "stat_category": "big_number|percentage|ranking|comparison|measurement|year|ratio — stat_counter",
          "stat_number": "number — stat_counter",
          "stat_prefix": "# — stat_counter ranking",
          "stat_suffix": "% or x — stat_counter",
          "stat_label": "SHORT LABEL — stat_counter",
          "stat_context": "in 2022 — stat_counter",
          "compare_to": "France — stat_counter comparison",
          "label": "sub-label — flow_map",
          "image_query_a": "BEFORE search — split_comparison",
          "image_query_b": "AFTER search — split_comparison",
          "label_a": "left label — split_comparison",
          "label_b": "right label — split_comparison"
        }}
      ]
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


HOOK_CANDIDATE_PROMPT = textwrap.dedent("""\
You are turning a broad geography region into short-form video hook ideas.

Region: {region}

Return ONLY valid JSON:
{{
  "region": "string",
  "selected_hook": "string",
  "hook_candidates": [
    "string",
    "string",
    "string"
  ],
  "reason": "string"
}}

Rules:
- The selected_hook must be the strongest YouTube Shorts hook for this region.
- Prefer specific, curiosity-driven, visual hooks over broad overviews.
- Use one of these opening styles:
  - Why ...
  - This ...
  - How ...
  - The reason ...
- Hooks must imply a concrete geography story: river, border, city, wildlife, trade chokepoint, unusual climate, density, terrain, pollution, infrastructure, or historical anomaly.
- Avoid generic country-summary phrasing like "Why {region} matters" or "The geography of {region}".
- Keep each hook under 12 words.
""")


def _fallback_hook_candidates(region: str) -> Dict[str, Any]:
    region_clean = str(region or "").strip()
    hooks = [
        f"Why {region_clean} is so hard to map",
        f"This river system shapes {region_clean}",
        f"The reason {region_clean} feels so different",
    ]
    return {
        "region": region_clean,
        "selected_hook": hooks[0],
        "hook_candidates": hooks,
        "reason": "fallback",
    }


def generate_hook_candidates(region: str) -> Dict[str, Any]:
    region_clean = str(region or "").strip()
    if not region_clean:
        return _fallback_hook_candidates(region_clean)

    try:
        raw = call_gemini(HOOK_CANDIDATE_PROMPT.format(region=region_clean), temperature=0.65, tools=[{"googleSearch": {}}])
        parsed = extract_json(raw)
        selected = str(parsed.get("selected_hook", "") or "").strip()
        candidates = [str(x).strip() for x in parsed.get("hook_candidates", []) if str(x).strip()]
        if not selected and candidates:
            selected = candidates[0]
        if not candidates and selected:
            candidates = [selected]
        if selected:
            parsed["region"] = region_clean
            parsed["selected_hook"] = selected
            parsed["hook_candidates"] = candidates[:5]
            return parsed
    except Exception:
        pass
    return _fallback_hook_candidates(region_clean)


def _coarse_region_from_text(text: str) -> str:
    raw = re.sub(r"\s+", " ", str(text or "").strip())
    if not raw:
        return "the region"
    anchor = _extract_location_anchor(raw)
    if anchor:
        return anchor
    bad_starts = {"why", "how", "what", "when", "where", "this", "that", "these", "those", "the", "a", "an"}
    proper_phrases = [
        cleaned
        for phrase in re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", raw)
        for cleaned in [_clean_phrase(phrase)]
        if cleaned and cleaned.split()[0].lower() not in bad_starts
    ]
    if proper_phrases:
        return proper_phrases[-1].strip()
    return "the region"


def _extract_location_anchor(text: str) -> str:
    raw = re.sub(r"\s+", " ", str(text or "").strip())
    if not raw:
        return ""

    lower = raw.lower()
    bad_starts = {"why", "how", "what", "when", "where", "this", "that", "these", "those", "the", "a", "an"}
    generic_place_tokens = {
        "city", "region", "country", "state", "province", "county", "district",
        "town", "village", "place", "area", "location", "feature",
    }

    def _clean_phrase(candidate: str) -> str:
        words = [w for w in re.split(r"\s+", str(candidate or "").strip()) if w]
        while words and words[0].lower() in bad_starts:
            words.pop(0)
        phrase = " ".join(words).strip(" ,.-")
        if not phrase or phrase.lower() in generic_place_tokens:
            return ""
        return phrase

    stop_words = (
        " is ", " are ", " was ", " were ", " has ", " have ", " had ", " can ",
        " could ", " should ", " would ", " will ", " for ", " because ", " that ",
        " which ", " who ", " while ", " where ", " when ", " why ", " how ", " and ",
        " but ", " so ", " to ", " by ", " with ", " from ", " matters ", " survives ",
        " grows ", " provides ", " pollutes ", " polluted ",
    )
    for prep in (" in ", " near ", " around ", " across ", " through ", " along ", " at ", " on ", " of "):
        idx = lower.find(prep)
        if idx < 0:
            continue
        tail = raw[idx + len(prep):].strip(" ,.-")
        if not tail:
            continue
        cut = len(tail)
        tail_lower = tail.lower()
        for marker in stop_words:
            pos = tail_lower.find(marker)
            if pos >= 0:
                cut = min(cut, pos)
        candidate = tail[:cut].strip(" ,.-")
        candidate = re.sub(r"^(the|this|that)\s+", "", candidate, flags=re.IGNORECASE)
        proper_prefix = re.match(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})", candidate)
        if proper_prefix:
            phrase = _clean_phrase(proper_prefix.group(1))
            if phrase.split()[0].lower() not in bad_starts:
                return phrase
        candidate = _clean_phrase(candidate)
        if candidate and len(candidate) >= 3 and candidate.split()[0].lower() not in bad_starts:
            return candidate

    proper_phrases = [
        cleaned
        for phrase in re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", raw)
        for cleaned in [_clean_phrase(phrase)]
        if cleaned and cleaned.split()[0].lower() not in bad_starts
    ]
    if proper_phrases:
        return proper_phrases[-1].strip()
    capitalized_tokens = [
        token.strip()
        for token in re.findall(r"\b([A-Z][a-z]{2,})\b", raw)
        if token and token.lower() not in bad_starts and token.lower() not in generic_place_tokens
    ]
    if capitalized_tokens:
        return capitalized_tokens[-1]
    return ""


def _ensure_location_fields(script: Dict[str, Any], prompt: str = "") -> Dict[str, Any]:
    prompt_clean = re.sub(r"\s+", " ", str(prompt or script.get("_user_prompt") or "").strip())
    existing_region = str(script.get("region") or "").strip()
    subject = str(script.get("subject") or "").strip()
    hook = str(script.get("hook_question") or "").strip()
    generic_place_values = {"", "the region", "region", "city", "the city", "country", "state", "province", "county", "district", "none", "world", "the world", "earth", "global", "globe"}
    # Single words that Gemini sometimes picks from the prompt title instead of a real place
    _non_place_words = {
        "nobody", "everybody", "everyone", "someone", "something", "nothing",
        "anyone", "anything", "nowhere", "everywhere", "somebody", "people",
        "person", "man", "woman", "they", "them", "this", "that", "these",
        "those", "here", "there", "most", "least", "worst", "best", "biggest",
        "smallest", "poorest", "richest", "deadliest", "dangerous", "safest",
        "wants", "lives", "moved", "left", "abandoned", "forgotten", "lost",
        "border", "borders", "boundary", "line", "conflict", "war", "history",
        "mystery", "secret", "reason", "truth", "story", "problem", "crisis",
    }

    def _usable_place(value: Any) -> str:
        cleaned = re.sub(r"\s+", " ", str(value or "").strip())
        low = cleaned.lower()
        if low in generic_place_values:
            return ""
        # Reject single common English words that aren't place names
        if len(cleaned.split()) == 1 and low in _non_place_words:
            return ""
        return cleaned

    location_anchor = _usable_place(script.get("location_focus") or script.get("geodata_query") or "")
    if not location_anchor:
        for raw in (prompt_clean, subject, hook, existing_region):
            location_anchor = _usable_place(_extract_location_anchor(raw))
            if location_anchor:
                break

    if not existing_region or existing_region.lower() in {"the region", "region"}:
        inferred_region = _coarse_region_from_text(" ".join([prompt_clean, subject, hook, location_anchor]).strip())
        if inferred_region:
            script["region"] = inferred_region
            existing_region = inferred_region

    if not location_anchor:
        cities = script.get("cities") if isinstance(script.get("cities"), list) else []
        if cities:
            location_anchor = _usable_place(str((cities[0] or {}).get("name") or "").strip())

    if not location_anchor:
        location_anchor = _usable_place(existing_region)

    if location_anchor:
        script["location_focus"] = location_anchor
        script["geodata_query"] = _usable_place(script.get("geodata_query") or "") or location_anchor

    return script


def fallback_topic_script(prompt: str) -> Dict[str, Any]:
    prompt_clean = re.sub(r"\s+", " ", str(prompt or "").strip()).strip(" ?.")
    title = prompt_clean[:60] if prompt_clean else "Geography Short"
    hook = prompt_clean or "Why is this geography topic so important?"
    generic_region = _coarse_region_from_text(prompt_clean)
    location_anchor = _extract_location_anchor(prompt_clean) or generic_region
    topic_lower = prompt_clean.lower()
    landform_tokens = {
        "river", "lake", "mountain", "desert", "forest", "coast", "coastline",
        "island", "delta", "strait", "valley", "bay", "sea", "ocean",
        "volcano", "glacier", "reef", "waterfall",
    }
    urban_tokens = {
        "city", "capital", "downtown", "metro", "port", "harbor", "harbour",
        "street", "skyscraper", "urban", "sinking", "traffic",
    }
    looks_urban = any(tok in topic_lower for tok in urban_tokens)
    looks_landform = any(tok in topic_lower for tok in landform_tokens)
    hook_broll = "real_city" if looks_urban and not looks_landform else "real_geography"
    mid_broll = "real_city" if looks_urban else "real_geography"

    script = {
        "region": generic_region,
        "location_focus": location_anchor,
        "geodata_query": location_anchor,
        "subject": prompt_clean or "the topic",
        "hook_question": hook,
        "title": title[:50],
        "total_duration_sec": 46,
        "_source": "fallback_topic",
        "beats": [
            {
                "beat_id": 1,
                "narration": hook,
                "script_type": "HOOK",
                "broll_type": hook_broll,
                "duration_sec": 4.0,
                "visual_note": (
                    f"Cinematic opening shot tied to {location_anchor}, clean, no text overlays, no logos, "
                    "no watermarks, high quality cinematic"
                ),
                "caption_text": "WHY IT MATTERS",
            },
            {
                "beat_id": 2,
                "narration": "The answer starts with where this story happens.",
                "script_type": "SCOPE",
                "broll_type": "real_geography",
                "duration_sec": 4.2,
                "visual_note": (
                    f"Wide establishing shot around {location_anchor}, clean, no text overlays, no logos, "
                    "no watermarks, high quality cinematic"
                ),
                "caption_text": "START HERE",
            },
            {
                "beat_id": 3,
                "narration": "This geography story affects real places and real people.",
                "script_type": "CORE_FACT",
                "broll_type": mid_broll,
                "duration_sec": 4.8,
                "visual_note": f"Wide aerial drone shot of landscape tied to {prompt_clean}, clean, no text overlays, no logos, no watermarks, high quality cinematic",
                "caption_text": "REAL IMPACT",
            },
            {
                "beat_id": 4,
                "narration": "And that is why this place matters far beyond the map.",
                "script_type": "EXIT",
                "broll_type": hook_broll,
                "duration_sec": 4.6,
                "visual_note": f"Closing cinematic shot connected to {location_anchor}, clean, no text overlays, no logos, no watermarks, high quality cinematic",
                "caption_text": "BEYOND THE MAP",
            },
        ],
        "cities": [],
    }
    return _ensure_location_fields(script, prompt_clean)


def generate_topic_script(prompt: str, run_dir: Path) -> Dict[str, Any]:
    """Stage 1 (prompt mode): Call Gemini to produce topic-driven script + beat map."""
    print("\n[S1] Generating topic script via Gemini...")
    print(f"  Prompt: {prompt}")
    gemini_prompt = TOPIC_SCRIPT_PROMPT.format(prompt=prompt)

    try:
        raw = call_gemini(gemini_prompt, tools=[{"googleSearch": {}}])
        script = extract_json(raw)
        script["_source"] = "gemini_topic"
        script["_user_prompt"] = prompt
        script = _ensure_location_fields(script, prompt)
        print(f"  [OK] Region: {script.get('region', '?')}")
        print(f"  [OK] Subject: {script.get('subject', '?')}")
        print(f"  [OK] {len(script.get('beats', []))} beats, "
              f"{script.get('total_duration_sec', '?')}s total")
    except Exception as e:
        print(f"  [WARN] Gemini topic script failed: {e}")
        print("  Falling back to topic-based script...")
        script = fallback_topic_script(prompt)
        script["_user_prompt"] = prompt

    script = _ensure_location_fields(script, prompt)
    script = normalize_script_plan(script)

    path = run_dir / "s1_script.json"
    path.write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {path.name}")
    return script


def normalize_script_plan(script: Dict[str, Any], visual_style: str = "mixed") -> Dict[str, Any]:
    beats = script.get("beats", [])
    if not isinstance(beats, list):
        script["beats"] = []
        return script

    visual_style = str(visual_style or "mixed").strip().lower()
    primary_geo = str(
        script.get("location_focus")
        or script.get("geodata_query")
        or script.get("region")
        or ""
    ).strip()
    generic_geo_values = {"", "none", "n/a", "na", "city", "the city", "region", "the region", "country", "state", "province", "county", "district", "area", "location", "place", "world", "the world", "earth", "global", "globe"}
    # Single words that aren't real places (same set as _ensure_location_fields)
    _non_place_words_norm = {
        "nobody", "everybody", "everyone", "someone", "something", "nothing",
        "anyone", "anything", "nowhere", "everywhere", "somebody", "people",
        "person", "man", "woman", "they", "them", "this", "that", "these",
        "those", "here", "there", "most", "least", "worst", "best", "biggest",
        "smallest", "poorest", "richest", "deadliest", "dangerous", "safest",
        "wants", "lives", "moved", "left", "abandoned", "forgotten", "lost",
        "border", "borders", "boundary", "line", "conflict", "war", "history",
        "mystery", "secret", "reason", "truth", "story", "problem", "crisis",
    }

    def _is_generic_geo(value: str) -> bool:
        low = value.strip().lower()
        if low in generic_geo_values:
            return True
        if len(low.split()) == 1 and low in _non_place_words_norm:
            return True
        return False

    def _clean_required_geography(value: Any) -> str:
        cleaned = re.sub(r"\s+", " ", str(value or "").strip())
        if _is_generic_geo(cleaned):
            return primary_geo or str(script.get("region") or "").strip()
        if primary_geo and primary_geo.lower() in cleaned.lower():
            return primary_geo
        anchor = _extract_location_anchor(cleaned)
        if anchor and not _is_generic_geo(anchor):
            if any(sep in cleaned.lower() for sep in (" or ", ",")) or len(cleaned.split()) > len(anchor.split()) + 2:
                return anchor
        return cleaned

    def _remap_concept_type(beat: Dict[str, Any]) -> str:
        text = " ".join(
            [
                str(script.get("subject", "") or ""),
                str(beat.get("caption_text", "") or ""),
                str(beat.get("narration", "") or ""),
                str(beat.get("visual_note", "") or ""),
            ]
        ).lower()
        if any(tok in text for tok in ("city", "capital", "downtown", "street", "building", "traffic", "neighborhood", "urban", "subway", "market", "infrastructure", "construction", "road", "bridge", "housing", "skyline", "compacts", "cracked", "groundwater", "aquifer", "pumping")):
            return "real_city"
        if any(tok in text for tok in ("river", "lake", "coast", "shore", "mountain", "forest", "desert", "terrain", "valley", "lakebed", "soil", "clay", "wetland", "basin", "delta", "glacier", "ice")):
            return "real_geography"
        return "real_city" if any(tok in str(script.get("subject", "")).lower() for tok in ("city", "urban", "capital", "metro")) else "real_geography"

    def _digital_twin_type(original_type: str, beat: Dict[str, Any], index: int) -> str:
        script_type = str(beat.get("script_type", "") or "").upper()
        note = str(beat.get("visual_note", "") or "").lower()
        original = str(original_type or "")
        if original in ("real_city", "real_people", "native_animal"):
            return "3d_flyover"
        if original == "real_geography":
            if any(token in note for token in ("mountain", "coast", "valley", "river", "desert")):
                return "3d_close_oblique"
            return "3d_flyover"
        if original in ("dark_cutout", "comparison_map"):
            return "3d_orbit" if script_type in ("HOOK", "EXIT") else "3d_zoom"
        if original in ("map_highlight", "outline_reveal"):
            return "3d_zoom"
        if original in ("map_wipe", "satellite_pan", "google_earth_pan"):
            return "3d_flyover"
        if original == "terrain_map":
            return "3d_close_oblique"
        if original in ("wikipedia_image",):
            return "3d_zoom"
        if original in ("3d_orbit", "3d_flyover", "3d_zoom", "3d_close_oblique"):
            return original
        cycle = ["3d_orbit", "3d_zoom", "3d_flyover", "3d_close_oblique"]
        return cycle[index % len(cycle)]

    def _digital_twin_note(broll_type: str, region_name: str, beat: Dict[str, Any]) -> str:
        place = script.get("subject") or script.get("region") or region_name or "the region"
        if broll_type == "3d_orbit":
            return f"Cinematic digital twin orbital establishing shot over {place}"
        if broll_type == "3d_zoom":
            return f"Digital twin zoom from high altitude down toward {place}"
        if broll_type == "3d_close_oblique":
            return f"Low-altitude digital twin oblique view gliding across the terrain around {place}"
        return f"Cinematic digital twin flyover over {place}"

    def _is_real_type(broll_type: str) -> bool:
        return broll_type in ("real_city", "real_people", "real_geography", "native_animal")

    def _ensure_real_visual_note(beat: Dict[str, Any]) -> None:
        existing = str(beat.get("visual_note", "") or "").strip()
        required_clean = "clean, no text overlays, no logos, no watermarks, high quality cinematic"
        has_required_clean = all(tok in existing.lower() for tok in (
            "clean", "no text", "no logos", "no watermarks", "high quality"
        ))
        if existing and has_required_clean and len(existing.split()) >= 8:
            return
        # Build a proper cinematic visual description from beat context,
        # NOT from the narration text (which is useless as a search query).
        btype = str(beat.get("broll_type", "")).strip().lower()
        caption = str(beat.get("caption_text", "") or "").strip()
        region = str(script.get("region", "") or "").strip()
        subject = str(script.get("subject", "") or "").strip()
        cities = script.get("cities", [])
        primary_city = cities[0].get("name", "") if cities and isinstance(cities[0], dict) else ""
        area = primary_city or region or "the area"

        if btype == "real_city":
            beat["visual_note"] = (
                f"Drone aerial shot of downtown {area}, golden hour skyline with dense buildings, "
                f"slow forward camera movement, {required_clean}"
            )
        elif btype == "real_geography":
            beat["visual_note"] = (
                f"Wide aerial drone shot of landscape and terrain near {area}, "
                f"lush scenery with gentle forward camera movement, {required_clean}"
            )
        elif btype == "native_animal":
            beat["visual_note"] = (
                f"Native wildlife in natural habitat from {region}, "
                f"close-up cinematic shot with shallow depth of field, {required_clean}"
            )
        else:
            beat["visual_note"] = (
                f"Cinematic aerial drone shot of {area}, "
                f"slow forward camera movement revealing the landscape, {required_clean}"
            )

    def _has_data_moment(beat: Dict[str, Any]) -> bool:
        if isinstance(beat.get("data_moment"), dict) and beat.get("data_moment"):
            return True
        # Also detect stat-heavy narration (numbers, percentages, rankings)
        narr = str(beat.get("narration", "") or "").lower()
        import re as _re
        if _re.search(r'\b\d[\d,]*\s*(%|percent|per\s+capita|per\s+100)', narr):
            return True
        stat_tokens = ("rate", "rank", "highest", "lowest", "deadliest", "most dangerous",
                       "per capita", "homicide", "murder", "crime", "deaths", "killed",
                       "population", "average", "compared to", "times higher", "times more")
        return sum(1 for tok in stat_tokens if tok in narr) >= 2

    def _is_informational_video() -> bool:
        text = " ".join(
            [
                str(script.get("hook_question", "") or ""),
                str(script.get("subject", "") or ""),
                str(script.get("title", "") or ""),
                " ".join(str(b.get("narration", "") or "") for b in beats[: min(5, len(beats))]),
            ]
        ).lower()
        tokens = (
            "why", "reason", "history", "polluted", "border", "river", "trade",
            "founded", "provides", "millions", "climate", "wildlife", "city",
            "capital", "largest", "density", "infrastructure", "complicated",
        )
        return any(tok in text for tok in tokens)

    def _looks_explanatory_subject(text: str) -> bool:
        tl = str(text or "").strip().lower()
        if not tl:
            return False
        explanatory_tokens = (
            "importance", "history", "matters", "reason", "why", "how", "explained",
            "crisis", "problem", "issue", "sinking", "polluted", "pollution",
            "growing", "growth", "decline", "trade", "economy", "frontline",
            "battlefield", "border dispute", "geopolitical", "water crisis",
        )
        if "'s " in tl:
            return True
        return any(tok in tl for tok in explanatory_tokens)

    def _infer_wikipedia_title(beat: Dict[str, Any]) -> str:
        overlay = beat.get("overlay") if isinstance(beat.get("overlay"), dict) else {}
        subject = str(script.get("subject", "") or "").strip()
        location_focus = str(script.get("location_focus", "") or "").strip()
        subject_candidate = location_focus if _looks_explanatory_subject(subject) and location_focus else subject

        # Prioritize specific titles from narration/beat context over generic location
        narr_anchor = _extract_location_anchor(str(beat.get("narration", "") or ""))
        candidates = [
            str(overlay.get("wikipedia_title", "") or "").strip(),
            str(beat.get("wikipedia_title", "") or "").strip(),
            narr_anchor,  # specific place mentioned in this beat's narration
            subject_candidate,
            _extract_location_anchor(str(overlay.get("image_query", "") or "")),
            str(overlay.get("image_query", "") or "").strip(),
            location_focus,
            str(overlay.get("caption", "") or "").strip(),
            str(script.get("region", "") or "").strip(),
        ]
        for text in candidates:
            if not text:
                continue
            cleaned = re.sub(r"\s+", " ", text).strip(" .,-")
            if len(cleaned) >= 3 and not _looks_like_bad_wiki_title(cleaned):
                return cleaned[:120]
        return ""

    def _looks_like_bad_wiki_title(title_text: str) -> bool:
        t = str(title_text or "").strip()
        if not t:
            return False
        tl = t.lower()
        if len(t) > 72:
            return True
        if tl.startswith(("a ", "an ", "the ")):
            return True
        if " or " in tl:
            return True
        if "'s " in tl and _looks_explanatory_subject(t):
            return True
        noisy_phrases = (
            "image overlay", "archival photo", "illustration", "depicting",
            "depiction", "showing", "clean", "no text overlays", "drone"
        )
        return any(p in tl for p in noisy_phrases)

    def _normalize_overlay_type(overlay: Dict[str, Any]) -> str:
        otype = str(overlay.get("type", "") or "").strip().lower()
        if otype == "historical_photo":
            overlay["type"] = "context_photo"
            return "context_photo"
        return otype

    def _apply_overlay_marker_timing(beat: Dict[str, Any]) -> None:
        narration_raw = str(beat.get("narration", "") or "")
        if "^" not in narration_raw:
            return
        marker_idx = narration_raw.find("^")
        narration_clean = re.sub(r"\s+", " ", narration_raw.replace("^", "")).strip()
        beat["narration"] = narration_clean

        overlay = beat.get("overlay") if isinstance(beat.get("overlay"), dict) else None
        if not overlay:
            return

        duration = float(beat.get("duration_sec", 4.0) or 4.0)
        total_chars = max(1, len(narration_clean))
        before_chars = len(re.sub(r"\s+", " ", narration_raw[:marker_idx]).strip())
        ratio = max(0.0, min(1.0, before_chars / total_chars))
        latest_start = max(0.15, duration - 0.45)
        at_sec = max(0.15, min(latest_start, duration * ratio))
        beat["_overlay_marker_at_sec"] = round(at_sec, 2)
        overlay = beat.get("overlay") if isinstance(beat.get("overlay"), dict) else None
        if overlay:
            overlay["at_sec"] = round(at_sec, 2)

    def _build_context_photo_overlay(beat: Dict[str, Any]) -> Dict[str, Any]:
        existing = beat.get("overlay") if isinstance(beat.get("overlay"), dict) else {}
        existing_type = _normalize_overlay_type(existing) if existing else ""
        is_historical = _is_historical_context(beat)

        def _clean_query(text: str) -> str:
            q = str(text or "").strip()
            if not q:
                return ""
            q = re.sub(r"\bclean,?\s*no text overlays?.*$", "", q, flags=re.IGNORECASE)
            q = re.sub(r"\b(image overlay of|archival photo(?: or illustration)?(?: depicting)?|photo of|illustration of)\b", "", q, flags=re.IGNORECASE)
            q = q.replace("\"", "").replace("'", "")
            q = q.split(".")[0]
            q = q.split(",")[0]
            q = re.sub(r"\s+", " ", q).strip(" .,-")
            return q[:100]

        at_sec = existing.get("at_sec")
        if at_sec is None:
            at_sec = beat.get("_overlay_marker_at_sec", 0.8)

        title = str(existing.get("wikipedia_title", "") or beat.get("wikipedia_title", "") or "").strip()
        query = str(existing.get("image_query", "") or beat.get("image_query", "") or "").strip()
        if _looks_like_bad_wiki_title(title):
            if not is_historical and not query:
                query = _clean_query(title)
            title = ""
        if not title and not query and not is_historical:
            query = _clean_query(str(beat.get("visual_note", "") or ""))
        if not title and not query and not is_historical:
            query = _clean_query(str(beat.get("narration", "") or ""))
        if not title and not query:
            inferred = _infer_wikipedia_title(beat)
            if inferred and not _looks_like_bad_wiki_title(inferred):
                title = inferred
            elif inferred and not is_historical:
                query = _clean_query(inferred)

        caption = str(existing.get("caption", "") or beat.get("caption_text", "") or beat.get("caption", "") or "").strip()
        if not caption and existing_type == "timeline_marker":
            year = str(existing.get("year", "") or "").strip()
            event = str(existing.get("event_text", "") or "").strip()
            caption = " ".join(x for x in [event, year] if x).strip()

        style = str(existing.get("style", "") or beat.get("style", "") or "auto").strip().lower()
        credit = str(
            existing.get("image_credit", "")
            or existing.get("credit", "")
            or beat.get("image_credit", "")
            or beat.get("credit", "")
            or ""
        ).strip()
        if not credit:
            sources = script.get("youtube_metadata", {}).get("sources", [])
            if isinstance(sources, list) and sources:
                credit = str(sources[0]).strip()

        overlay: Dict[str, Any] = {
            "type": "context_photo",
            "at_sec": float(at_sec),
            "style": style or "auto",
        }
        if title:
            overlay["wikipedia_title"] = title
        if query:
            overlay["image_query"] = query
        if caption:
            overlay["caption"] = caption
        if credit:
            overlay["image_credit"] = credit
        return overlay

    def _is_historical_context(beat: Dict[str, Any]) -> bool:
        text = " ".join(
            [
                str(beat.get("narration", "") or ""),
                str(beat.get("caption_text", "") or ""),
                str(beat.get("visual_note", "") or ""),
                str(script.get("subject", "") or ""),
            ]
        ).lower()
        historical_tokens = (
            "history", "historical", "colonial", "empire", "independence",
            "founding", "founded", "revolution", "century", "war", "ancient",
            "medieval", "dynasty", "archival", "era", "year", "bc", "ad"
        )
        return any(tok in text for tok in historical_tokens)

    def _trim_to_target_beats(max_beats: int = 10) -> None:
        if len(beats) <= max_beats:
            return
        keep = {0, len(beats) - 1}
        for i, b in enumerate(beats):
            if _has_data_moment(b):
                keep.add(i)
                break

        def _score(idx: int, beat: Dict[str, Any]) -> int:
            btype = str(beat.get("broll_type", ""))
            script_type = str(beat.get("script_type", "") or "").upper()
            score = 0
            if _is_real_type(btype):
                score += 3
            if script_type in ("HOOK", "CORE_FACT", "EXIT"):
                score += 2
            if btype in ("map_wipe", "terrain_map"):
                score -= 2
            if btype.startswith("3d_"):
                score -= 1
            if beat.get("search_intent"):
                score += 1
            score += max(0, 12 - idx) // 6
            return score

        ranked = sorted(
            ((i, _score(i, b)) for i, b in enumerate(beats) if i not in keep),
            key=lambda x: x[1],
            reverse=True,
        )
        for i, _ in ranked:
            if len(keep) >= max_beats:
                break
            keep.add(i)
        filtered = [b for i, b in enumerate(beats) if i in keep]
        beats[:] = filtered

    def _reduce_map_saturation() -> None:
        if not beats:
            return
        # Include ALL satellite-render-based types — they all look like "maps" to viewers
        wide_map_types = {
            "satellite_pan", "map_wipe", "map_highlight", "terrain_map", "dark_cutout",
            "3d_orbit", "3d_flyover", "3d_close_oblique", "3d_zoom",
        }
        max_wide_maps = max(2, int(len(beats) * 0.25))
        wide_indices = [
            idx for idx, beat in enumerate(beats)
            if str(beat.get("broll_type", "")).strip().lower() in wide_map_types
        ]
        if len(wide_indices) > max_wide_maps:
            overflow = len(wide_indices) - max_wide_maps
            replacement_cycle = ["real_geography", "real_city", "native_animal"]
            cycle_idx = 0
            for idx in wide_indices:
                if overflow <= 0:
                    break
                if idx in (0, len(beats) - 1):
                    continue
                replacement = replacement_cycle[cycle_idx % len(replacement_cycle)]
                cycle_idx += 1
                beats[idx]["broll_type"] = replacement
                if replacement in ("real_geography", "real_city", "native_animal"):
                    _ensure_real_visual_note(beats[idx])
                overflow -= 1

    def _prefer_real_exit() -> None:
        if not beats:
            return
        last = beats[-1]
        last_type = str(last.get("broll_type", "")).strip().lower()
        if last_type in {"3d_orbit", "satellite_pan", "map_wipe", "terrain_map", "map_highlight", "dark_cutout"}:
            # Prefer real city for urban/county/city subjects; otherwise real geography.
            subj = str(script.get("subject", "") or "").lower()
            if any(tok in subj for tok in ("city", "county", "metro", "urban", "jacksonville", "gary")):
                last["broll_type"] = "real_city"
            else:
                last["broll_type"] = "real_geography"
            _ensure_real_visual_note(last)

    def _ensure_outline_moment() -> None:
        return

    def _convert_to_reference_real(idx: int) -> None:
        beat = beats[idx]
        for candidate in _preferred_real_variants(beat):
            if candidate not in {"real_geography", "real_city", "real_people", "native_animal"}:
                continue
            beat["broll_type"] = candidate
            if candidate == "native_animal":
                beat["visual_note"] = (
                    f"Native wildlife footage in natural habitat from {script.get('location_focus') or script.get('region') or 'the region'}, "
                    "clean, no text overlays, no logos, no watermarks, high quality cinematic"
                )
            else:
                _ensure_real_visual_note(beat)
            return

    def _enforce_reference_visual_flow() -> None:
        if len(beats) < 3:
            return
        map_family = {
            "satellite_pan", "map_wipe", "map_highlight", "terrain_map", "dark_cutout",
            "3d_orbit", "3d_flyover", "3d_close_oblique", "3d_zoom", "outline_reveal",
            "comparison_map", "google_earth_pan",
        }
        explanatory_maps = {"map_highlight", "outline_reveal", "comparison_map", "map_wipe"}

        if str(beats[1].get("broll_type", "")).strip().lower() in map_family and str(beats[2].get("broll_type", "")).strip().lower() in map_family:
            _convert_to_reference_real(2)

        later_map_indices = [
            idx for idx in range(3, max(len(beats) - 1, 3))
            if str(beats[idx].get("broll_type", "")).strip().lower() in map_family
        ]
        if len(later_map_indices) > 1:
            keep_idx = next(
                (idx for idx in later_map_indices if str(beats[idx].get("broll_type", "")).strip().lower() in explanatory_maps),
                later_map_indices[0],
            )
            for idx in later_map_indices:
                if idx != keep_idx:
                    _convert_to_reference_real(idx)

    def _preferred_real_variants(beat: Dict[str, Any]) -> List[str]:
        text = " ".join(
            [
                str(script.get("subject", "") or ""),
                str(beat.get("caption_text", "") or ""),
                str(beat.get("narration", "") or ""),
                str(beat.get("visual_note", "") or ""),
            ]
        ).lower()
        if any(tok in text for tok in ("city", "capital", "downtown", "street", "urban", "market", "harare", "metro")):
            return ["real_city", "real_people", "real_geography", "3d_flyover"]
        if any(tok in text for tok in ("animal", "wildlife", "species", "habitat", "elephant", "lion", "bird")):
            return ["native_animal", "real_geography", "3d_close_oblique", "3d_orbit"]
        if any(tok in text for tok in ("river", "lake", "delta", "coast", "mountain", "valley", "desert", "terrain", "forest")):
            return ["real_geography", "3d_close_oblique", "real_city", "3d_flyover"]
        return ["real_geography", "real_city", "native_animal", "3d_flyover"]

    def _allocate_context_overlays() -> None:
        if not beats or not _is_informational_video():
            return
        existing_indices = [
            idx for idx, beat in enumerate(beats)
            if isinstance(beat.get("overlay"), dict)
            and _normalize_overlay_type(beat["overlay"]) == "context_photo"
        ]
        max_overlays = 3
        target_overlays = 2 if len(beats) <= 10 else 3
        if len(existing_indices) >= target_overlays:
            if len(existing_indices) > target_overlays:
                ranked_existing = sorted(
                    existing_indices,
                    key=lambda idx: (
                        0 if str((beats[idx].get("overlay") or {}).get("wikipedia_title", "")).strip() else 1,
                        0 if _is_historical_context(beats[idx]) else 1,
                        idx,
                    ),
                )
                keep = set(ranked_existing[:target_overlays])
                for idx in existing_indices:
                    if idx not in keep:
                        beats[idx]["overlay"] = None
            return

        candidates: List[int] = []
        for idx, beat in enumerate(beats):
            if idx in existing_indices:
                continue
            if idx in (0, len(beats) - 1):
                continue
            btype = str(beat.get("broll_type", "") or "").strip().lower()
            if btype in {"real_city", "real_geography", "3d_flyover", "3d_orbit", "3d_zoom", "outline_reveal"}:
                # Any beat with facts, stats, historical context, or explanatory content
                stype = str(beat.get("script_type", "") or "").upper()
                if (_is_historical_context(beat) or _has_data_moment(beat)
                        or "why" in str(beat.get("narration", "")).lower()
                        or stype in ("CORE_FACT", "CONTRAST", "SCOPE")):
                    candidates.append(idx)
        for idx in candidates:
            if len(existing_indices) >= target_overlays or len(existing_indices) >= max_overlays:
                break
            beat = beats[idx]
            inferred = _infer_wikipedia_title(beat)
            if not inferred:
                continue
            beat["overlay"] = {
                "type": "context_photo",
                "at_sec": round(min(max(float(beat.get("duration_sec", 4.0)) * 0.58, 0.8), max(float(beat.get("duration_sec", 4.0)) - 0.8, 0.8)), 2),
                "style": "bw" if _is_historical_context(beat) else "light",
                "wikipedia_title": inferred,
                "caption": str(beat.get("caption_text", "") or inferred[:36]).strip(),
            }
            existing_indices.append(idx)

    total = len(beats)
    for i, beat in enumerate(beats):
        # Normalize new `overlays` list → internal `overlay` single dict for downstream logic.
        # (assembly.py handles both formats; normalizer internals use single `overlay`)
        if "overlays" in beat and not beat.get("overlay"):
            ovl_list = beat.get("overlays")
            if isinstance(ovl_list, list) and ovl_list and isinstance(ovl_list[0], dict):
                beat["overlay"] = ovl_list[0]

        dur = float(beat.get("duration_sec", 4))
        beat["duration_sec"] = max(4.0, min(8.0, dur))
        if primary_geo:
            # Override generic/vague geodata — setdefault won't fix "world"
            if _is_generic_geo(str(beat.get("location_focus", "") or "")):
                beat["location_focus"] = primary_geo
            if _is_generic_geo(str(beat.get("geodata_query", "") or "")):
                beat["geodata_query"] = primary_geo

        _apply_overlay_marker_timing(beat)

        if not beat.get("script_type"):
            if i == 0:
                beat["script_type"] = "HOOK"
            elif i == 1:
                beat["script_type"] = "SCOPE"
            elif i == total - 1:
                beat["script_type"] = "EXIT"
            elif i == 2:
                beat["script_type"] = "CORE_FACT"
            else:
                beat["script_type"] = "CORE_FACT"

        if str(beat.get("broll_type", "")) == "3d_curvature":
            beat["broll_type"] = "3d_orbit"

        # Fix cases where Gemini puts an overlay type in the broll_type field
        invalid_broll = str(beat.get("broll_type", "")).lower().strip()
        if invalid_broll in ["wikipedia_image", "historical_photo", "context_photo"]:
            beat["overlay"] = _build_context_photo_overlay(beat)
            beat["broll_type"] = "3d_orbit"
            for misplaced_key in ("wikipedia_title", "image_query", "image_url", "style", "credit", "image_credit"):
                beat.pop(misplaced_key, None)

        if str(beat.get("broll_type", "")).strip().lower() == "real_concept":
            beat["broll_type"] = _remap_concept_type(beat)

        if _is_real_type(str(beat.get("broll_type", ""))):
            _ensure_real_visual_note(beat)

            search_intent = beat.get("search_intent")
            if not isinstance(search_intent, dict):
                search_intent = {}
            visual_description = str(search_intent.get("visual_description") or beat.get("visual_note") or beat.get("caption_text") or beat.get("narration") or "").strip()
            required_geography = _clean_required_geography(search_intent.get("required_geography") or primary_geo or script.get("region") or "")
            geography_strictness = str(search_intent.get("geography_strictness") or "").strip().lower()
            if geography_strictness not in {"strict", "loose"}:
                btype_local = str(beat.get("broll_type", "")).strip().lower()
                geography_strictness = "strict" if btype_local in {"real_city", "real_people"} else "loose"
            fallback_allowed = str(search_intent.get("fallback_allowed") or "terrain_map").strip().lower()
            # Never allow "none" — a generated fallback is always better than a missing beat
            if fallback_allowed not in {"terrain_map", "3d_orbit"}:
                fallback_allowed = "terrain_map"
            biome_hint = str(search_intent.get("biome_hint") or beat.get("visual_note") or beat.get("narration") or "").strip()
            beat["search_intent"] = {
                "visual_description": visual_description,
                "required_geography": required_geography,
                "geography_strictness": geography_strictness,
                "fallback_allowed": fallback_allowed,
                "biome_hint": biome_hint,
            }
            beat.pop("youtube_url", None)
            beat.pop("youtube_start_time", None)

        overlay = beat.get("overlay") if isinstance(beat.get("overlay"), dict) else None
        if overlay:
            otype = _normalize_overlay_type(overlay)
        else:
            otype = ""

        if overlay and otype in ("timeline_marker", "animated_stat"):
            converted = _build_context_photo_overlay(beat)
            if converted.get("wikipedia_title") or converted.get("image_query"):
                beat["overlay"] = converted
                overlay = beat["overlay"]
                otype = "context_photo"
            else:
                beat["overlay"] = None
                overlay = None
                otype = ""

        if overlay and otype == "context_photo":
            has_title = str(overlay.get("wikipedia_title", "") or "").strip()
            has_query = str(overlay.get("image_query", "") or "").strip()
            historical = _is_historical_context(beat)
            if has_title and _looks_like_bad_wiki_title(has_title):
                overlay.pop("wikipedia_title", None)
                has_title = ""
                if not has_query and not historical:
                    inferred_bad = _infer_wikipedia_title(beat)
                    if inferred_bad and not _looks_like_bad_wiki_title(inferred_bad):
                        overlay["wikipedia_title"] = inferred_bad
                        has_title = inferred_bad
            if not has_title and not has_query:
                inferred = _infer_wikipedia_title(beat)
                if inferred and not _looks_like_bad_wiki_title(inferred):
                    overlay["wikipedia_title"] = inferred
                elif inferred and not historical:
                    overlay["image_query"] = inferred
            if historical and not str(overlay.get("wikipedia_title", "") or "").strip():
                beat["overlay"] = None
                overlay = None
                otype = ""
                continue
            if not str(overlay.get("wikipedia_title", "") or "").strip() and not str(overlay.get("image_query", "") or "").strip():
                beat["overlay"] = None
                overlay = None
                otype = ""
                continue
            style_value = str(overlay.get("style", "") or "").strip().lower()
            if not style_value or style_value == "auto":
                overlay["style"] = "bw" if _is_historical_context(beat) else "light"

        if visual_style == "digital_twin":
            remapped = _digital_twin_type(str(beat.get("broll_type", "")), beat, i)
            beat["broll_type"] = remapped
            beat["visual_note"] = _digital_twin_note(remapped, script.get("region", "the region"), beat)
            beat.pop("youtube_url", None)
            beat.pop("youtube_start_time", None)

    if visual_style != "digital_twin":
        _trim_to_target_beats(max_beats=10)
        _reduce_map_saturation()
        _ensure_outline_moment()
        _enforce_reference_visual_flow()
        _prefer_real_exit()
        _allocate_context_overlays()

    for i, beat in enumerate(beats):
        if i == 0:
            beat["duration_sec"] = min(float(beat.get("duration_sec", 3.6)), 3.6)
        elif i == 1:
            beat["duration_sec"] = min(float(beat.get("duration_sec", 4.0)), 4.0)
        elif i == 2:
            beat["duration_sec"] = min(float(beat.get("duration_sec", 4.7)), 4.7)
        else:
            beat["duration_sec"] = min(float(beat.get("duration_sec", 5.4)), 5.4)

    target_total = 46.0
    current_total = sum(float(b.get("duration_sec", 10.0)) for b in beats) if beats else 0.0
    if beats and current_total > 0 and (current_total < 40.0 or current_total > 54.0):
        scale = target_total / current_total
        raw_scaled = [float(beat.get("duration_sec", 10.0)) * scale for beat in beats]
        min_d, max_d = 4.0, 8.0
        lo, hi = min(raw_scaled), max(raw_scaled)
        if hi - lo > 1e-6:
            norm_scaled = [min_d + ((d - lo) / (hi - lo)) * (max_d - min_d) for d in raw_scaled]
        else:
            norm_scaled = [max(min_d, min(max_d, d)) for d in raw_scaled]
        for beat, nd in zip(beats, norm_scaled):
            beat["duration_sec"] = round(max(min_d, min(max_d, nd)), 2)

    # Prevent consecutive beats using the same geography-visual family
    satellite_based = {"satellite_pan", "map_wipe",
               "3d_orbit", "3d_flyover", "3d_zoom", "3d_close_oblique",
               "outline_reveal", "terrain_map", "map_highlight", "google_earth_pan", "dark_cutout"}
    real_cycle = ["real_geography", "real_city", "native_animal"]
    for i in range(1, len(beats)):
        prev_type = str(beats[i-1].get("broll_type", ""))
        curr_type = str(beats[i].get("broll_type", ""))
        if visual_style == "digital_twin":
            if prev_type == curr_type:
                twin_cycle = ["3d_orbit", "3d_zoom", "3d_flyover", "3d_close_oblique"]
                for candidate in twin_cycle:
                    if candidate != prev_type:
                        beats[i]["broll_type"] = candidate
                        beats[i]["visual_note"] = _digital_twin_note(candidate, script.get("region", "the region"), beats[i])
                        break
            continue
        if prev_type in satellite_based and curr_type in satellite_based:
            for candidate in _preferred_real_variants(beats[i]):
                if candidate == prev_type:
                    continue
                beats[i]["broll_type"] = candidate
                if candidate == "native_animal":
                    beats[i]["visual_note"] = (
                        f"Native wildlife footage in natural habitat from {script.get('region', 'the region')}, clean, no text overlays, no logos, no watermarks, high quality cinematic"
                    )
                elif candidate in {"real_city", "real_geography", "real_people"}:
                    _ensure_real_visual_note(beats[i])
                else:
                    beats[i]["visual_note"] = _digital_twin_note(candidate, script.get("region", "the region"), beats[i]) if candidate.startswith("3d_") else beats[i].get("visual_note", "")
                break

    for beat in beats:
        btype_local = str(beat.get("broll_type", "")).strip().lower()
        if btype_local == "real_concept":
            beat["broll_type"] = _remap_concept_type(beat)
            btype_local = str(beat.get("broll_type", "")).strip().lower()
        if _is_real_type(btype_local):
            _ensure_real_visual_note(beat)
            search_intent = beat.get("search_intent")
            if not isinstance(search_intent, dict):
                search_intent = {}
            visual_description = str(search_intent.get("visual_description") or beat.get("visual_note") or beat.get("caption_text") or beat.get("narration") or "").strip()
            required_geography = _clean_required_geography(search_intent.get("required_geography") or primary_geo or script.get("region") or "")
            geography_strictness = str(search_intent.get("geography_strictness") or "").strip().lower()
            if geography_strictness not in {"strict", "loose"}:
                geography_strictness = "strict" if btype_local in {"real_city", "real_people"} else "loose"
            fallback_allowed = str(search_intent.get("fallback_allowed") or "terrain_map").strip().lower()
            # Never allow "none" — a generated fallback is always better than a missing beat
            if fallback_allowed not in {"terrain_map", "3d_orbit"}:
                fallback_allowed = "terrain_map"
            biome_hint = str(search_intent.get("biome_hint") or beat.get("visual_note") or beat.get("narration") or "").strip()
            beat["search_intent"] = {
                "visual_description": visual_description,
                "required_geography": required_geography,
                "geography_strictness": geography_strictness,
                "fallback_allowed": fallback_allowed,
                "biome_hint": biome_hint,
            }
        elif "search_intent" in beat:
            beat.pop("search_intent", None)

    for i, beat in enumerate(beats, start=1):
        beat["beat_id"] = i
        beat.pop("_overlay_marker_at_sec", None)
        # Finalize: convert internal `overlay` (single, normalizer-processed) + any extra
        # items from the original `overlays` list (index 1+) into one canonical `overlays` list.
        _ovl_single = beat.pop("overlay", None)
        _ovl_list = beat.pop("overlays", None) or []
        if not isinstance(_ovl_list, list):
            _ovl_list = []
        merged = []
        if isinstance(_ovl_single, dict) and _ovl_single.get("type"):
            merged.append(_ovl_single)
        # overlays[0] was already promoted to `overlay` for normalization; only add [1:] here.
        for _ov in _ovl_list[1:]:
            if isinstance(_ov, dict) and _ov.get("type"):
                merged.append(_ov)
        beat["overlays"] = merged

    script["total_duration_sec"] = round(sum(float(b.get("duration_sec", 7)) for b in beats), 1)
    return script


def generate_script(region: str, run_dir: Path) -> Dict[str, Any]:
    """Stage 1: Call Gemini to produce the script + beat map."""
    print("\n[S1] Generating script via Gemini...")
    prompt = SCRIPT_PROMPT.format(region=region)

    try:
        raw = call_gemini(prompt, tools=[{"googleSearch": {}}])
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
                     "script_type": "CORE_FACT", "broll_type": "real_geography", "duration_sec": 18,
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
             "script_type": "CORE_FACT", "broll_type": "real_geography", "duration_sec": 12,
             "visual_note": f"Clean, no text overlays, cinematic aerial drone footage of {region} terrain and landscape",
             "caption_text": "NOWHERE ELSE"},
            {"beat_id": 3, "narration": f"And these are the cities that call it home.",
             "script_type": "CORE_FACT", "broll_type": "real_city", "duration_sec": 12,
             "visual_note": f"Clean, no text overlays, cinematic street level view of the largest city in {region}",
             "caption_text": "THE CITIES"},
            {"beat_id": 4, "narration": f"The people here have a way of life all their own.",
             "script_type": "CORE_FACT", "broll_type": "real_city", "duration_sec": 12,
             "visual_note": f"Clean, no text overlays, cinematic street level daily life footage in the main city of {region}",
             "caption_text": "CITY LIFE"},
            {"beat_id": 5, "narration": f"Look at how the coastline frames the whole region.",
             "script_type": "CORE_FACT", "broll_type": "real_geography", "duration_sec": 12,
             "visual_note": f"Clean, no text overlays, cinematic coastal cliffs or shoreline drone footage in {region}",
             "caption_text": "THE COASTLINE"},
            {"beat_id": 6, "narration": f"Now watch how the borders line up with the terrain.",
             "script_type": "CONTRAST", "broll_type": "outline_reveal", "duration_sec": 12,
             "visual_note": f"Boundary outline appearing over satellite of {region}",
             "caption_text": "THE BORDERS"},
            {"beat_id": 7, "narration": f"Now you see {region} differently.",
             "script_type": "EXIT", "broll_type": "dark_cutout", "duration_sec": 18,
             "visual_note": f"Region shape on dark grid — closing callback",
             "caption_text": "SEE IT NOW"},
        ],
        "cities": [],
    }

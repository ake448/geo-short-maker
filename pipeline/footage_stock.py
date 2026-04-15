"""
footage_stock.py — Intent-first YouTube sourcing and acceptance-gate orchestrators.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from .config import (
    _SSL_CTX, YTDLP_PATH,
    GEMINI_API_KEY, GEMINI_MODEL,
)
from .footage import (
    _yt_search_candidates, _yt_download_and_trim,
    _yt_query_for_beat, _extract_search_hint,
    _extract_middle_frame, _has_news_ticker_band, _has_burned_captions,
    _clean_place_anchor, _join_query_parts, _scene_terms_for_beat,
    _extract_place_aliases, _subject_query_hint,
)


_GENERIC_NATURE_TERMS = {
    "forest", "woods", "mountain", "ridge", "river", "fog", "rain", "soil", "clay",
    "wetland", "marsh", "coast", "shore", "sunrise", "sunset", "weather", "cloud",
    "kudzu", "appalachian", "foothill", "biome", "texture", "terrain",
}
_LOW_COVERAGE_TERMS = {
    "district", "county", "village", "rural", "homestead", "historical", "historic", "fort",
    "canal", "towpath", "strait coast", "restricted", "military", "adjacent", "airspace",
}
_TRUST_HINTS = ("official", "tourism", "city", "county", "gov", "government", "news", "tv", "drone")
_GENERIC_GEO_VALUES = {"", "none", "the region", "region", "city", "the city", "country", "state", "province", "county", "district", "area", "location", "place"}


def _append_diag(diag: Dict[str, Any], key: str, value: Any) -> None:
    diag.setdefault(key, [])
    diag[key].append(value)


def _safe_unlink(path: Path, retries: int = 6, delay: float = 0.4) -> None:
    """Unlink a file with retries — handles Windows file-lock race conditions."""
    for i in range(retries):
        try:
            path.unlink(missing_ok=True)
            return
        except OSError:
            if i < retries - 1:
                time.sleep(delay)
    # Last attempt — if still locked, leave it; don't crash the pipeline
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _ytdlp_available() -> bool:
    if not YTDLP_PATH:
        return False
    if "/" in YTDLP_PATH or "\\" in YTDLP_PATH:
        return Path(YTDLP_PATH).exists()
    return shutil.which(YTDLP_PATH) is not None


def _search_intent_for_beat(region: str, beat: Dict[str, Any], subject: str = "") -> Dict[str, str]:
    raw = beat.get("search_intent") if isinstance(beat.get("search_intent"), dict) else {}
    btype = str(beat.get("broll_type", "") or "").strip().lower()
    visual_description = str(
        raw.get("visual_description")
        or beat.get("visual_note")
        or beat.get("caption_text")
        or beat.get("narration")
        or ""
    ).strip()
    location_hint = str(
        beat.get("location_focus")
        or beat.get("geodata_query")
        or raw.get("location_focus")
        or raw.get("geodata_query")
        or region
        or ""
    ).strip()
    place_aliases = _extract_place_aliases(raw.get("required_geography"), location_hint, subject, region)
    required_geography = _clean_place_anchor(place_aliases[0] if place_aliases else (raw.get("required_geography") or location_hint or region or ""))
    if not required_geography:
        required_geography = _clean_place_anchor(location_hint or region or "")
    strictness = str(raw.get("geography_strictness") or "").strip().lower()
    if strictness not in {"strict", "loose"}:
        strictness = "strict" if btype in {"real_city", "real_people"} else "loose"
    fallback_allowed = str(raw.get("fallback_allowed") or "terrain_map").strip().lower()
    if fallback_allowed not in {"terrain_map", "3d_orbit", "none"}:
        fallback_allowed = "terrain_map"
    biome_hint = str(raw.get("biome_hint") or visual_description or "").strip()

    if not biome_hint and subject:
        biome_hint = subject

    return {
        "visual_description": visual_description,
        "required_geography": required_geography,
        "geography_strictness": strictness,
        "fallback_allowed": fallback_allowed,
        "biome_hint": biome_hint,
    }


def _beat_track(beat: Dict[str, Any], intent: Dict[str, str]) -> str:
    btype = str(beat.get("broll_type", "") or "").strip().lower()
    if btype in {"real_city", "real_people"}:
        return "location_specific"
    if btype == "native_animal":
        return "generic_nature"
    required_geo = _clean_place_anchor(intent.get("required_geography", ""))
    if btype == "real_geography" and required_geo:
        return "location_specific"
    text = f"{intent.get('visual_description', '')} {intent.get('biome_hint', '')}".lower()
    if any(term in text for term in _GENERIC_NATURE_TERMS):
        return "generic_nature"
    return "location_specific" if btype == "real_geography" else "generic_nature"


def _coverage_tier(region: str, intent: Dict[str, str], geo: Dict[str, Any], track: str) -> str:
    if track != "location_specific":
        return "B"
    target = f"{intent.get('required_geography', '')} {intent.get('visual_description', '')}".lower()
    if any(tok in target for tok in _LOW_COVERAGE_TERMS):
        return "C"

    cities = [str(c.get("name", "") or "").lower() for c in (geo.get("cities") or [])][:4]
    if cities and any(city and city in target for city in cities):
        return "A"

    region_l = str(region or "").lower()
    if any(tok in region_l for tok in ("district", "county", "rural", "village")):
        return "C"
    return "B"


def _trusted_channel_tokens(region: str, geo: Dict[str, Any]) -> List[str]:
    tokens: List[str] = []
    env_value = os.environ.get("GEO_TRUSTED_CHANNELS", "").strip()
    if env_value:
        tokens.extend([x.strip().lower() for x in env_value.split(",") if x.strip()])

    for city in (geo.get("cities") or [])[:3]:
        name = str(city.get("name", "") or "").strip().lower()
        if name:
            tokens.append(name)
    region_tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", str(region or "").lower()) if len(t) >= 4]
    tokens.extend(region_tokens[:3])

    out: List[str] = []
    seen = set()
    for token in tokens:
        if token and token not in seen:
            out.append(token)
            seen.add(token)
    return out


def _is_trusted_channel(candidate: Dict[str, Any], trusted_tokens: List[str]) -> bool:
    channel_text = f"{candidate.get('channel', '')} {candidate.get('uploader', '')}".lower()
    if not channel_text.strip():
        return False
    has_geo = any(tok in channel_text for tok in trusted_tokens if tok)
    has_hint = any(h in channel_text for h in _TRUST_HINTS)
    return has_geo and has_hint


def _query_variants_for_beat(
    region: str,
    beat: Dict[str, Any],
    intent: Dict[str, str],
    subject: str = "",
    extra_anchors: List[str] | None = None,
) -> List[str]:
    btype = str(beat.get("broll_type", "") or "").strip().lower()

    # Use Gemini-generated queries first — they're specific and human-like
    gemini_queries = beat.get("youtube_queries") or []
    if isinstance(gemini_queries, list):
        gemini_queries = [str(q).strip() for q in gemini_queries if str(q).strip()]

    base = _yt_query_for_beat(region, beat, subject)
    hint = _extract_search_hint(beat)
    short_hint = hint if 1 < len(hint.split()) <= 6 else ""
    visual = intent.get("visual_description", "")
    required_geo = _clean_place_anchor(intent.get("required_geography", "") or region)
    broader_region = _clean_place_anchor(region)
    place_candidates = _extract_place_aliases(
        beat.get("location_focus"),
        beat.get("geodata_query"),
        required_geo,
        subject,
        *(extra_anchors or []),
        broader_region,
    )
    if not place_candidates:
        place_candidates = [broader_region or region]
    primary_place = place_candidates[0]
    broad_places = [p for p in place_candidates[1:] if p.lower() != primary_place.lower()]
    scene_terms = _scene_terms_for_beat(beat, broll_type=btype)
    subject_hint = _subject_query_hint(subject, place_candidates)

    # Gemini queries lead; generated variants follow as fallback
    variants: List[str] = gemini_queries + [base]
    if subject_hint:
        variants.extend([
            _join_query_parts(primary_place, subject_hint, "4k"),
            _join_query_parts(primary_place, subject_hint),
        ])
    if short_hint and short_hint.lower() != subject_hint.lower():
        variants.extend([
            _join_query_parts(primary_place, short_hint, "4k"),
            _join_query_parts(primary_place, short_hint),
        ])
    for scene in scene_terms[:3]:
        variants.extend([
            _join_query_parts(primary_place, scene, "4k"),
            _join_query_parts(primary_place, scene),
        ])

    if btype == "real_city":
        variants.extend([
            _join_query_parts(primary_place, "skyline", "drone 4k"),
            _join_query_parts(primary_place, "downtown", "aerial 4k"),
            _join_query_parts(primary_place, "street view", "walk 4k"),
            _join_query_parts(primary_place, "city center"),
        ])
    elif btype == "real_people":
        variants.extend([
            _join_query_parts(primary_place, "street life", "4k"),
            _join_query_parts(primary_place, "people walking", "downtown"),
            _join_query_parts(primary_place, "market", "daily life"),
        ])
    elif btype == "real_geography":
        variants.extend([
            _join_query_parts(primary_place, "aerial", "4k"),
            _join_query_parts(primary_place, "drone", "4k"),
            _join_query_parts(primary_place, "landscape", "aerial"),
            _join_query_parts(primary_place, "mountains"),
        ])
        for scene in scene_terms[:3]:
            variants.extend([
                _join_query_parts(primary_place, scene, "drone 4k"),
                _join_query_parts(primary_place, scene, "drone"),
            ])
        for broad_place in broad_places[:2]:
            variants.extend([
                _join_query_parts(primary_place, broad_place, "aerial 4k"),
                _join_query_parts(broad_place, primary_place, "drone"),
            ])
    elif btype == "native_animal":
        species_terms = " ".join([bit for bit in re.split(r"[^a-zA-Z0-9]+", " ".join([subject, hint, visual])) if len(bit) >= 4][:4])
        variants.extend([
            _join_query_parts(primary_place, species_terms, "wildlife 4k"),
            _join_query_parts(primary_place, intent.get("biome_hint", ""), "habitat 4k"),
            _join_query_parts(primary_place, "national park wildlife 4k"),
        ])
    else:
        variants.append(_join_query_parts(primary_place, visual[:60], "4k"))

    # Broad geographic fallbacks — guaranteed to have footage on YouTube
    # Appended last so specific queries get priority, but these catch niche topics
    broad_places = set()
    for src in [region, broader_region, required_geo, primary_place] + list(place_candidates):
        cleaned = _clean_place_anchor(str(src or ""))
        if cleaned and len(cleaned) >= 3:
            broad_places.add(cleaned)
    for bp in list(broad_places)[:4]:
        variants.extend([
            _join_query_parts(bp, "drone aerial 4k"),
            _join_query_parts(bp, "cinematic 4k"),
            _join_query_parts(bp, "landscape drone"),
        ])
    # Also add broad queries for extra_anchors (cities from script)
    for anchor in (extra_anchors or [])[:3]:
        anchor_clean = _clean_place_anchor(str(anchor or ""))
        if anchor_clean and anchor_clean.lower() not in {p.lower() for p in broad_places}:
            variants.extend([
                _join_query_parts(anchor_clean, "drone 4k"),
                _join_query_parts(anchor_clean, "aerial cinematic"),
            ])

    unique: List[str] = []
    seen = set()
    for query in variants:
        q = " ".join(str(query or "").split())
        ql = q.lower()
        if q and ql not in seen:
            unique.append(q)
            seen.add(ql)
    return unique


def _gemini_geo_match(frame_path: Path, required_geography: str, strictness: str, track: str,
                      visual_description: str = "", broll_type: str = "") -> bool:
    if not GEMINI_API_KEY or not frame_path.exists():
        return True

    try:
        img_data = base64.b64encode(frame_path.read_bytes()).decode("utf-8")
    except Exception:
        return True

    mode = "exact place" if strictness == "strict" else "regional look"
    content_hint = ""
    if visual_description:
        content_hint = f" Expected content: {visual_description}."
    if broll_type:
        type_map = {
            "real_city": "urban/city scenery — streets, skyline, buildings",
            "real_geography": "natural landscape — terrain, aerial, drone, nature",
            "real_people": "people in public — street life, markets, crowds",
            "native_animal": "wildlife or animals in their natural habitat",
        }
        content_hint += f" Shot type required: {type_map.get(broll_type, broll_type)}."
    prompt = (
        "Decide if this video frame is acceptable for a geography documentary. "
        "Return JSON only: {\"geo_match\": true|false, \"reason\": \"short\"}. "
        f"Required geography: {required_geography}. Match mode: {mode}.{content_hint} "
        "Reject if: wrong geography, TV news tickers/overlays, product shots, food/cooking, "
        "indoor product reviews, unboxing, pets in domestic settings, or any content clearly "
        "unrelated to the geography or shot type described."
    )
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{urllib.parse.quote(GEMINI_MODEL)}:generateContent"
        f"?key={urllib.parse.quote(GEMINI_API_KEY)}"
    )
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_data}},
            ]
        }],
        "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45, context=_SSL_CTX) as res:
            raw = json.loads(res.read().decode("utf-8"))
        parts = raw.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "\n".join(p.get("text", "") for p in parts if p.get("text")).strip()
        parsed = json.loads(text) if text.startswith("{") else {}
        ok = bool(parsed.get("geo_match"))
        if not ok:
            print(" [reject: geo-mismatch]", end="", flush=True)
        return ok
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, KeyError, ValueError):
        return True


def _accept_candidate_clip(
    out_path: Path,
    source: str,
    intent: Dict[str, str],
    track: str,
    trusted_channel: bool,
    diagnostics: Dict[str, Any] | None = None,
    broll_type: str = "",
    geo_in_title: bool = False,
) -> bool:
    frame_path = _extract_middle_frame(out_path)
    if frame_path and _has_news_ticker_band(frame_path):
        print(" [reject:ticker]", end="", flush=True)
        if diagnostics is not None:
            _append_diag(diagnostics, "reject_reasons", "ticker")
        _safe_unlink(frame_path)
        _safe_unlink(out_path)
        return False
    if frame_path and _has_burned_captions(frame_path):
        print(" [reject:captions]", end="", flush=True)
        if diagnostics is not None:
            _append_diag(diagnostics, "reject_reasons", "burned_captions")
        _safe_unlink(frame_path)
        _safe_unlink(out_path)
        return False

    strictness = str(intent.get("geography_strictness") or "loose").lower()
    required_geo = str(intent.get("required_geography") or "")
    visual_description = str(intent.get("visual_description") or "")
    # Don't force strict — respect what the script requested. Forcing strict caused
    # Gemini to demand "exact place" match on every frame, rejecting most valid footage.
    # If the video title already matched the geographic keyword, that's strong enough
    # confirmation — skip Gemini vision check. Vision check rejects valid drone footage
    # because suburban streets don't have "Jacksonville" written on them.
    should_geo_check = ((not trusted_channel) or strictness == "strict") and not geo_in_title
    if should_geo_check and frame_path:
        if not _gemini_geo_match(frame_path, required_geo, strictness, track,
                                  visual_description=visual_description, broll_type=broll_type):
            if diagnostics is not None:
                _append_diag(diagnostics, "reject_reasons", "geo_mismatch")
            _safe_unlink(frame_path)
            _safe_unlink(out_path)
            return False

    if frame_path:
        _safe_unlink(frame_path)
    return True


def gen_real_youtube_clip(region: str, beat: Dict[str, Any], out_path: Path, duration: float, subject: str = "", geo: Dict[str, Any] = None, used_video_ids: set = None) -> bool:
    """Intent-first sourcing: geo-aware YouTube tiers, then validated stock fallback."""
    btype = str(beat.get("broll_type", "") or "").strip().lower()
    intent = _search_intent_for_beat(region, beat, subject)
    beat["search_intent"] = intent
    track = _beat_track(beat, intent)
    tier = _coverage_tier(region, intent, geo or {}, track)
    diagnostics: Dict[str, Any] = {
        "track": track,
        "tier": tier,
        "query_variants": [],
        "candidate_counts": [],
        "reject_reasons": [],
        "accepted_source_class": None,
        "fallback_reason": None,
    }
    beat["_sourcing_debug"] = diagnostics

    place_aliases = _extract_place_aliases(
        beat.get("location_focus"),
        beat.get("geodata_query"),
        intent.get("required_geography"),
        subject,
        region,
        *([c.get("name") for c in geo.get("cities", [])] if geo and geo.get("cities") else []),
    )
    strict_anchor = str(place_aliases[0] if place_aliases else (intent.get("required_geography") or (f"{subject} {region}".strip() if subject else region)))
    topic_hint = _extract_search_hint(beat)
    extra_anchors = place_aliases[1:5] if place_aliases else ([c.get("name") for c in geo.get("cities", [])] if geo and geo.get("cities") else None)

    print(f" [track:{track}|tier:{tier}]", end="", flush=True)

    query_variants = _query_variants_for_beat(region, beat, intent, subject, extra_anchors)
    trusted_tokens = _trusted_channel_tokens(region, geo or {})

    strict_modes = {
        "A": ["strict"],
        "B": ["strict", "medium", "relaxed"],
        "C": ["medium", "relaxed"],  # Skip strict for low-coverage — saves query budget
    }
    mode_list = strict_modes.get(tier, ["strict", "medium"])

    if _ytdlp_available() and track == "location_specific":
        _consecutive_empty_strict = 0
        for mode in mode_list:
            max_queries = 6 if mode == "strict" else 5
            for qidx, query in enumerate(query_variants[:max_queries]):
                print(f" [q{qidx+1}:{mode}]", end="", flush=True)
                _append_diag(diagnostics, "query_variants", {"mode": mode, "query": query})
                candidates = _yt_search_candidates(
                    query,
                    max_results=14 if mode == "strict" else 12,
                    region_anchor=strict_anchor,
                    topic_hint=topic_hint,
                    broll_type=btype,
                    extra_anchors=extra_anchors,
                    strictness=mode,
                )
                _append_diag(diagnostics, "candidate_counts", {"mode": mode, "query": query, "count": len(candidates)})
                candidates = sorted(
                    candidates,
                    key=lambda c: float(c.get("score") or 0.0) + (3.0 if _is_trusted_channel(c, trusted_tokens) else 0.0),
                    reverse=True,
                )
                if used_video_ids is not None:
                    candidates = [c for c in candidates if c.get("video_id") not in used_video_ids]
                print(f" [yt:{len(candidates)}]", end="", flush=True)
                # Early bail from strict: if first 3 queries return 0, skip to medium
                if mode == "strict" and len(candidates) == 0:
                    _consecutive_empty_strict += 1
                    if _consecutive_empty_strict >= 3:
                        print(" [strict-bail]", end="", flush=True)
                        break
                elif mode == "strict":
                    _consecutive_empty_strict = 0
                max_candidates = 6 if tier == "C" else (10 if mode == "strict" else 8)
                for i, cand in enumerate(candidates[:max_candidates]):
                    trusted = _is_trusted_channel(cand, trusted_tokens)
                    if trusted:
                        print(" [trusted]", end="", flush=True)
                    print(f".{i+1}", end="", flush=True)
                    seg_idx = (used_video_ids or {}).get(cand.get("video_id"), 0)
                    if _yt_download_and_trim(
                        cand,
                        out_path,
                        duration,
                        region_anchor=strict_anchor,
                        topic_hint=topic_hint,
                        broll_type=btype,
                        extra_anchors=extra_anchors,
                        strictness=mode,
                        segment_index=seg_idx,
                    ) and _accept_candidate_clip(out_path, "youtube", intent, track, trusted, diagnostics,
                                                  broll_type=btype, geo_in_title=bool(cand.get("geo_in_title"))):
                        diagnostics["accepted_source_class"] = "local_real" if mode in {"strict", "medium"} else "regional_real"
                        if used_video_ids is not None:
                            used_video_ids[cand.get("video_id")] = used_video_ids.get(cand.get("video_id"), 0) + 1
                        return True

    if track == "location_specific" and tier == "C" and intent.get("geography_strictness") == "strict":
        print(" [coverage-low: map fallback preferred]", end="", flush=True)
        diagnostics["fallback_reason"] = "coverage_low_strict"
        return False

    if track == "generic_nature" and _ytdlp_available():
        for qidx, query in enumerate(query_variants[:3]):
            print(f" [q{qidx+1}:loose]", end="", flush=True)
            _append_diag(diagnostics, "query_variants", {"mode": "relaxed", "query": query})
            candidates = _yt_search_candidates(
                query,
                max_results=10,
                region_anchor=strict_anchor,
                topic_hint=topic_hint,
                broll_type=btype,
                extra_anchors=extra_anchors,
                strictness="relaxed",
            )
            _append_diag(diagnostics, "candidate_counts", {"mode": "relaxed", "query": query, "count": len(candidates)})
            if used_video_ids:
                candidates = [c for c in candidates if c.get("video_id") not in used_video_ids]
            print(f" [yt:{len(candidates)}]", end="", flush=True)
            for i, cand in enumerate(candidates[:6]):
                print(f".{i+1}", end="", flush=True)
                if _yt_download_and_trim(
                    cand,
                    out_path,
                    duration,
                    region_anchor=strict_anchor,
                    topic_hint=topic_hint,
                    broll_type=btype,
                    extra_anchors=extra_anchors,
                    strictness="relaxed",
                ) and _accept_candidate_clip(out_path, "youtube", intent, track, trusted_channel=False, diagnostics=diagnostics, broll_type=btype):
                    diagnostics["accepted_source_class"] = "regional_real"
                    if used_video_ids is not None:
                        used_video_ids[cand.get("video_id")] = used_video_ids.get(cand.get("video_id"), 0) + 1
                    return True

    print(" [yt-only -> generated fallback]", end="", flush=True)
    diagnostics["fallback_reason"] = "youtube_exhausted"
    return False


def gen_wikipedia_image(geo: Dict, beat: Dict[str, Any], out_path: Path, duration: float) -> bool:
    """Fetch the main image from a Wikipedia page based on the wikipedia_title or visual_note."""
    article_title = beat.get("wikipedia_title", "").strip()
    if not article_title:
        article_title = beat.get("visual_note", "").strip()
    
    if not article_title:
        return False
    print(f" [wiki:'{article_title}']", end="", flush=True)
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
    page = next(iter(pages.values()))
    img_url = page.get("thumbnail", {}).get("source", "")
    if not img_url:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(img_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as resp:
            out_path.write_bytes(resp.read())
            return out_path.exists() and out_path.stat().st_size > 1024
    except Exception:
        out_path.unlink(missing_ok=True)
        return False

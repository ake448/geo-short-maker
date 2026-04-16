"""
footage.py — Real footage sourcing: YouTube and Wikipedia images,
             voiceover mixing.
"""
from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from .config import (
    _SSL_CTX, USER_AGENT, FPS, OUT_W, OUT_H,
    YTDLP_PATH, COOKIES_YT, REAL_MIN_SRC_SEC, REAL_MAX_SRC_SEC,
    BLUR_BG_FILTER, MUSIC_VOL_DB,
)
from .ffmpeg_utils import run_ffmpeg


_ANCHOR_STOPWORDS = {
    "the", "and", "with", "from", "into", "over", "under", "about", "this", "that",
    "what", "why", "how", "when", "where", "which", "who", "whose", "was", "were",
    "is", "are", "be", "been", "being", "of", "for", "in", "on", "to", "at", "by",
    "founding", "history", "historical", "treaty", "agreement", "timeline", "story",
    "fact", "facts", "explained", "explain", "documentary", "shorts", "short",
    "video", "footage", "cinematic", "aerial", "drone", "walk", "tour", "travel",
    "city", "cities", "people", "street", "landscape", "nature",
    "usa", "u_s_a", "us", "u_s", "united", "states", "country", "countries", "world", "earth",
}

_TITLE_HARD_BAD_TERMS = {
    "podcast", "interview", "news", "compilation", "slideshow", "top 10",
    "lecture", "reaction", "vlog", "lyrics", "music video", "highlights",
    "nba", "nfl", "mlb", "ufc", "boxing", "game recap", "reaction",
    "shutterstock", "getty", "istock", "storyblocks", "pond5", "preview", "watermark",
}

_BROLL_RELEVANCE_HINTS = {
    "real_city": ("city", "street", "downtown", "skyline", "walk", "urban"),
    "real_people": ("people", "street", "market", "crowd", "daily life", "walking"),
    "real_geography": ("drone", "aerial", "landscape", "mountain", "river", "coast", "lake", "nature"),
    "native_animal": ("wildlife", "animal", "habitat", "nature"),
}
_GENERIC_PLACE_MARKERS = {"", "none", "the region", "region", "city", "the city", "country", "state", "province", "county", "district", "area", "location", "place"}
_PLACE_TAIL_TOKENS = {"region", "area", "zone", "territory", "frontline", "corridor"}
_SUBJECT_QUERY_NOISE = {
    "importance", "history", "matters", "reason", "reasons", "why", "how", "explained",
    "explain", "topic", "fact", "facts", "story", "geography", "question", "answer",
    "issue", "problem", "crisis", "frontline", "battlefield", "geopolitical", "significance",
}


def _cleanup_ytdlp_sidecars(base_path: Path) -> None:
    """Remove yt-dlp/ffmpeg sidecar fragments for a given temp download stem."""
    try:
        parent = base_path.parent
        stem = base_path.stem
        for sibling in parent.glob(f"{stem}*"):
            if sibling == base_path:
                continue
            if sibling.is_file():
                try:
                    sibling.unlink()
                except OSError:
                    pass
    except OSError:
        pass


def _anchor_tokens(text: str) -> List[str]:
    raw = re.sub(r"[^a-zA-Z0-9]+", " ", str(text or "").lower())
    toks = [
        t for t in raw.split()
        if len(t) >= 4 and t not in _ANCHOR_STOPWORDS and not any(ch.isdigit() for ch in t)
    ]
    unique: List[str] = []
    seen = set()
    for t in toks:
        if t not in seen:
            unique.append(t)
            seen.add(t)
    return unique


def _clean_place_anchor(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "").replace("’", "'").strip(" ,.-"))
    cleaned = re.sub(r"^(the|this|that)\s+", "", cleaned, flags=re.IGNORECASE)
    poss = re.match(r"(.+?)'s\s+(.+)", cleaned)
    if poss:
        tail = poss.group(2).strip().lower()
        if any(term in tail for term in _SUBJECT_QUERY_NOISE):
            cleaned = poss.group(1).strip(" ,.-")
    words = cleaned.split()
    while words and words[-1].lower() in _PLACE_TAIL_TOKENS:
        words.pop()
    cleaned = " ".join(words).strip(" ,.-")
    if cleaned.lower() in _GENERIC_PLACE_MARKERS:
        return ""
    return cleaned


def _extract_place_aliases(*texts: str) -> List[str]:
    aliases: List[str] = []
    seen = set()

    def _add(value: str) -> None:
        cleaned = _clean_place_anchor(value)
        if not cleaned:
            return
        norm = cleaned.lower()
        if norm in seen:
            return
        aliases.append(cleaned)
        seen.add(norm)

    for text in texts:
        raw = re.sub(r"\s+", " ", str(text or "").replace("’", "'").strip())
        if not raw:
            continue
        for phrase in re.findall(r"\b([A-Z][A-Za-z0-9'\-]*(?:\s+[A-Z][A-Za-z0-9'\-]*){0,3})\b", raw):
            _add(phrase)
        for piece in re.split(r"\bor\b|/|;|\|", raw, flags=re.IGNORECASE):
            for subpiece in piece.split(","):
                _add(subpiece)
        _add(raw)
    return aliases


def _subject_query_hint(text: str, place_aliases: List[str] | None = None) -> str:
    raw = re.sub(r"\s+", " ", str(text or "").replace("’", "'").strip())
    if not raw:
        return ""
    lowered = raw.lower()
    for alias in place_aliases or []:
        if alias:
            lowered = re.sub(rf"\b{re.escape(alias.lower())}\b", " ", lowered)
    lowered = lowered.replace("'s", " ")
    tokens = [
        tok for tok in re.split(r"[^a-z0-9]+", lowered)
        if len(tok) >= 4 and tok not in _SUBJECT_QUERY_NOISE and tok not in _ANCHOR_STOPWORDS
    ]
    keepers: List[str] = []
    for tok in tokens:
        if tok not in keepers:
            keepers.append(tok)
    if not keepers:
        return ""
    return " ".join(keepers[:3]).strip()


def _join_query_parts(*parts: str) -> str:
    out: List[str] = []
    seen = set()
    for part in parts:
        cleaned = " ".join(str(part or "").split()).strip()
        if not cleaned:
            continue
        norm = cleaned.lower()
        if norm in seen:
            continue
        out.append(cleaned)
        seen.add(norm)
    return " ".join(out).strip()


def _scene_terms_for_beat(beat: Dict[str, Any], broll_type: str = "") -> List[str]:
    search_intent = beat.get("search_intent") if isinstance(beat.get("search_intent"), dict) else {}
    text = " ".join(
        [
            str(search_intent.get("visual_description") or ""),
            str(search_intent.get("biome_hint") or ""),
            str(beat.get("visual_note") or ""),
            str(beat.get("narration") or ""),
            str(beat.get("caption_text") or ""),
        ]
    ).lower()
    phrases: List[str] = []
    keyword_map = [
        (("sink", "subsidence", "subsid"), ["subsidence", "sinking streets", "land subsidence"]),
        (("groundwater", "aquifer", "pumped", "pumping", "extraction", "well"), ["groundwater", "water pumping", "aquifer", "water infrastructure"]),
        (("clay", "soil", "lakebed", "sediment"), ["clay soil", "lakebed", "construction site"]),
        (("crack", "damaged", "foundation", "settling"), ["cracked road", "building damage", "foundation cracks"]),
        (("flood", "flooding"), ["flooded streets", "urban flooding"]),
        (("skyline", "downtown", "urban", "street", "city"), ["skyline", "downtown", "street view", "aerial"]),
        (("river", "canal", "meltwater"), ["river", "valley river", "meltwater river"]),
        (("lake",), ["lake", "shoreline"]),
        (("glacier", "ice", "snowfield", "icefall"), ["glacier", "ice field", "snow mountains"]),
        (("mountain", "valley", "terrain", "pass", "ridge"), ["terrain", "landscape", "mountain pass"]),
        (("border", "disputed", "military", "battlefield"), ["border area", "high altitude", "mountain pass"]),
        (("construction", "infrastructure"), ["construction", "infrastructure"]),
        (("pollution", "sewage", "dirty water"), ["pollution", "dirty water", "sewage"]),
    ]
    for tokens, mapped in keyword_map:
        if any(tok in text for tok in tokens):
            phrases.extend(mapped)

    if broll_type == "real_geography":
        phrases = [p for p in phrases if p not in {"skyline", "downtown", "street view"}]
        phrases.extend(["terrain", "landscape", "drone"])
    elif broll_type == "real_city":
        phrases.extend(["downtown", "skyline", "street view"])
    elif broll_type == "real_people":
        phrases.extend(["street life", "people walking"])
    elif broll_type == "native_animal":
        phrases.extend(["wildlife", "natural habitat"])

    unique: List[str] = []
    seen = set()
    for phrase in phrases:
        cleaned = " ".join(str(phrase or "").split()).strip()
        if not cleaned:
            continue
        norm = cleaned.lower()
        if norm not in seen:
            unique.append(cleaned)
            seen.add(norm)
    return unique[:6]


def _title_passes_relevance(
    title: str,
    region_anchor: str = "",
    topic_hint: str = "",
    broll_type: str = "",
    extra_anchors: List[str] = None,
    strictness: str = "strict",
) -> bool:
    title_l = str(title or "").lower()
    if not title_l.strip():
        return False
    if any(term in title_l for term in _TITLE_HARD_BAD_TERMS):
        return False

    all_anchor_text = region_anchor
    if extra_anchors:
        all_anchor_text += " " + " ".join(extra_anchors)
        
    anchor_words = _anchor_tokens(all_anchor_text)
    hint_words = [w for w in _anchor_tokens(topic_hint) if w not in set(anchor_words)]
    broll = str(broll_type or "").strip().lower()
    fallback_kws = _BROLL_RELEVANCE_HINTS.get(broll, ())

    mode = str(strictness or "strict").strip().lower()
    if mode == "strict":
        if anchor_words and not any(word in title_l for word in anchor_words):
            return False
        if not anchor_words and hint_words and not any(word in title_l for word in hint_words[:8]):
            if fallback_kws and not any(k in title_l for k in fallback_kws):
                return False
    elif mode == "medium":
        relevance_words = list(anchor_words[:8]) + list(hint_words[:8]) + list(fallback_kws)
        if relevance_words and not any(w in title_l for w in relevance_words):
            return False
    else:  # relaxed
        relaxed_words = list(hint_words[:6]) + list(fallback_kws)
        if relaxed_words and not any(w in title_l for w in relaxed_words):
            return False

    return True


def _yt_probe_title(url: str) -> str:
    ytdlp_ok = False
    if YTDLP_PATH:
        if "/" in YTDLP_PATH or "\\" in YTDLP_PATH:
            ytdlp_ok = Path(YTDLP_PATH).exists()
        else:
            ytdlp_ok = shutil.which(YTDLP_PATH) is not None
    if not ytdlp_ok or not url:
        return ""
    cmd = [
        YTDLP_PATH,
        "--skip-download",
        "--print", "%(title)s",
        "--no-warnings", "--ignore-errors",
        url,
    ]
    if COOKIES_YT and Path(COOKIES_YT).exists():
        cmd[1:1] = ["--cookies", COOKIES_YT]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=35, check=False)
        title = (r.stdout or b"").decode("utf-8", errors="replace").strip().splitlines()
        return title[0].strip() if title else ""
    except Exception:
        return ""


def _yt_probe_metadata(url: str) -> Dict[str, str]:
    ytdlp_ok = False
    if YTDLP_PATH:
        if "/" in YTDLP_PATH or "\\" in YTDLP_PATH:
            ytdlp_ok = Path(YTDLP_PATH).exists()
        else:
            ytdlp_ok = shutil.which(YTDLP_PATH) is not None
    if not ytdlp_ok or not url:
        return {}

    cmd = [
        YTDLP_PATH,
        "--skip-download",
        "--print", "%(title)s|||%(description)s|||%(channel)s|||%(uploader)s",
        "--no-warnings", "--ignore-errors",
        url,
    ]
    if COOKIES_YT and Path(COOKIES_YT).exists():
        cmd[1:1] = ["--cookies", COOKIES_YT]

    try:
        r = subprocess.run(cmd, capture_output=True, timeout=35, check=False)
        raw = (r.stdout or b"").decode("utf-8", errors="replace").strip().splitlines()
        if not raw:
            return {}
        parts = raw[0].split("|||", 3)
        title = parts[0].strip() if len(parts) > 0 else ""
        description = parts[1].strip() if len(parts) > 1 else ""
        channel = parts[2].strip() if len(parts) > 2 else ""
        uploader = parts[3].strip() if len(parts) > 3 else ""
        return {
            "title": title,
            "description": description,
            "channel": channel,
            "uploader": uploader,
        }
    except Exception:
        return {}


def mix_voiceover(video_path: Path, voiceover_path: Path, run_dir: Path,
                  music_path: Optional[Path] = None) -> Optional[Path]:
    """Mix voiceover (and optional music) onto the silent video."""
    out_path = run_dir / "final_short_with_audio.mp4"
    if music_path and music_path.exists():
        print(f"  Mixing voiceover + music onto video...")
        ok = run_ffmpeg([
            "ffmpeg", "-y",
            "-i", str(video_path), "-i", str(voiceover_path), "-i", str(music_path),
            "-filter_complex",
            f"[1:a]aresample=44100[vo];"
            f"[2:a]aloop=loop=-1:size=2e+09,atrim=duration=120,"
            f"volume={MUSIC_VOL_DB}dB,aresample=44100[mu];"
            f"[vo][mu]amix=inputs=2:duration=shortest:dropout_transition=2:normalize=0[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
            str(out_path)
        ], timeout=120)
    else:
        print(f"  Mixing voiceover onto video...")
        ok = run_ffmpeg([
            "ffmpeg", "-y",
            "-i", str(video_path), "-i", str(voiceover_path),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
            str(out_path)
        ], timeout=120)
    if ok and out_path.exists():
        mb = out_path.stat().st_size / 1048576
        print(f"  [OK] {out_path.name} ({mb:.1f} MB)")
        return out_path
    else:
        print("  [FAIL] Audio mix failed")
        return None


def _extract_search_hint(beat: Dict[str, Any]) -> str:
    search_intent = beat.get("search_intent") if isinstance(beat.get("search_intent"), dict) else {}
    candidates = [
        search_intent.get("visual_description"),
        beat.get("youtube_query"),
        beat.get("visual_note"),
        beat.get("narration"),
        beat.get("caption_text"),
    ]
    noises = [
        "real footage related to:", "real footage of",
        "footage of", "montage of", "showing", "clip of", "video of",
        "people", "person", "the region", "in the background",
        "clean, no text overlays", "clean with no text", "no logos", "no overlays",
        "high quality cinematic", "cinematic", "4k", "60fps", "high quality",
        "drone footage of", "aerial view of", "drone shot of",
    ]
    for source in candidates:
        raw = str(source or "").split("^")[0]
        raw = re.split(r"[.!?]", raw, maxsplit=1)[0]
        if not raw.strip():
            continue
        for noise in noises:
            raw = re.sub(re.escape(noise), " ", raw, flags=re.IGNORECASE)
        raw = re.sub(r'\b(and|with|no|clean)\b', ' ', raw, flags=re.IGNORECASE)
        raw = re.sub(r'[,.\-;:!]', ' ', raw)
        cleaned = " ".join(raw.split()).strip()
        if not cleaned:
            continue
        words = cleaned.split()
        if source == beat.get("caption_text"):
            alpha_words = [w for w in words if re.search(r"[A-Za-z]", w)]
            if len(alpha_words) <= 3 or cleaned.upper() == cleaned:
                continue
        if len(words) >= 2:
            return " ".join(words[:10])[:90].strip()
    return ""


def _yt_query_for_beat(region: str, beat: Dict[str, Any], script_subject: str = "") -> str:
    """Build a concise, place-first YouTube search query for a beat."""
    # 1. Prefer Gemini-generated youtube_query if provided
    yt_query = beat.get("youtube_query")
    if yt_query and str(yt_query).strip():
        q = str(yt_query).strip()
        if "4k" not in q.lower():
            q += " 4k"
        return q

    # 2. Use highlight subject if available
    btype = str(beat.get("broll_type", "")).strip().lower()
    hint = _extract_search_hint(beat)
    search_intent = beat.get("search_intent") if isinstance(beat.get("search_intent"), dict) else {}
    place_aliases = _extract_place_aliases(
        beat.get("location_focus"),
        beat.get("geodata_query"),
        search_intent.get("required_geography"),
        region,
    )
    search_context = _clean_place_anchor(place_aliases[0] if place_aliases else (
        beat.get("location_focus")
        or beat.get("geodata_query")
        or search_intent.get("required_geography")
        or region
    ))
    if not search_context:
        search_context = _clean_place_anchor(region) or region
    scene_terms = _scene_terms_for_beat(beat, broll_type=btype)
    scene = scene_terms[0] if scene_terms else ""
    usable_hint = hint if 1 < len(hint.split()) <= 6 else ""
    subject_hint = _subject_query_hint(script_subject, place_aliases)
    highlight = beat.get("highlight")
    if highlight and isinstance(highlight, dict):
        subject = highlight.get("query", "")
        if subject:
            htype = highlight.get("type", "")
            if htype in ("river", "lake", "coastline"):
                return _join_query_parts(subject, search_context, "aerial drone 4k")
            elif htype in ("desert", "mountain"):
                return _join_query_parts(subject, search_context, "landscape drone 4k")
            else:
                return _join_query_parts(subject, search_context, "4k")

    # 3. Fallback: construct from broll_type + hint
    if btype == "real_city":
        return _join_query_parts(search_context, subject_hint or usable_hint or scene, "skyline aerial 4k")
    if btype == "real_people":
        return _join_query_parts(search_context, subject_hint or usable_hint or scene, "street life 4k")
    if btype == "native_animal":
        return _join_query_parts(search_context, usable_hint or scene, "native wildlife natural habitat 4k")
    if usable_hint:
        return _join_query_parts(search_context, subject_hint or usable_hint, "4k")
    return _join_query_parts(search_context, subject_hint or scene or "landscape", "drone aerial 4k")



def _time_to_seconds(time_str: str) -> float:
    """Convert HH:MM:SS or MM:SS to seconds."""
    if not time_str:
        return 0.0
    parts = str(time_str).strip().split(":")
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except ValueError:
        return 0.0


def _yt_download_exact(
    url: str,
    start_time: str,
    out_path: Path,
    duration_sec: float,
    region_anchor: str = "",
    topic_hint: str = "",
    broll_type: str = "",
    extra_anchors: List[str] = None,
    strictness: str = "strict",
) -> bool:
    """Download a precise chunk of a YouTube video using yt-dlp's --download-sections."""
    if not url:
        return False
        
    ytdlp_ok = False
    if YTDLP_PATH:
        if "/" in YTDLP_PATH or "\\" in YTDLP_PATH:
            ytdlp_ok = Path(YTDLP_PATH).exists()
        else:
            ytdlp_ok = shutil.which(YTDLP_PATH) is not None
    if not ytdlp_ok:
        return False
        
    if region_anchor or topic_hint:
        probed_title = _yt_probe_title(url)
        if probed_title and not _title_passes_relevance(
            probed_title,
            region_anchor=region_anchor,
            topic_hint=topic_hint,
            broll_type=broll_type,
            extra_anchors=extra_anchors,
            strictness=strictness,
        ):
            print(" (exact reject: off-topic)", end="", flush=True)
            return False

    start_sec = _time_to_seconds(start_time)
    # We download slightly more than needed, then let FFmpeg do the exact trim + blur
    end_sec = start_sec + duration_sec + 2.0 
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract just the video ID for a cleaner temp filename
    vid_match = re.search(r"v=([^&]+)", url)
    vid_id = vid_match.group(1) if vid_match else "exact"
    full_path = out_path.parent / f"_yt_exact_{vid_id}_full.mp4"
    
    # Note: --download-sections requires ffmpeg to be installed and available to yt-dlp
    dl_cmd = [
        YTDLP_PATH,
        "-f", "bv*[vcodec^=avc1][height<=1080]+ba[ext=m4a]/b[ext=mp4]/best",
        "--recode-video", "mp4", "--merge-output-format", "mp4",
        "--download-sections", f"*{start_sec}-{end_sec}",
        "-o", str(full_path), "--retries", "2", "--limit-rate", "5M",
        "--no-warnings", "--ignore-errors",
        url
    ]
    if COOKIES_YT and Path(COOKIES_YT).exists():
        dl_cmd[1:1] = ["--cookies", COOKIES_YT]
        
    print(f" [yt-dlp exact: {start_time}]", end="", flush=True)
    try:
        r = subprocess.run(dl_cmd, capture_output=True, timeout=180)
        if r.returncode != 0 and not full_path.exists():
            print(f" (yt-dlp fail)", end="", flush=True)
            _cleanup_ytdlp_sidecars(full_path)
            return False
    except Exception as e:
        print(f" (yt-dlp err: {e})", end="", flush=True)
        _cleanup_ytdlp_sidecars(full_path)
        return False

    if not full_path.exists() or full_path.stat().st_size < 10240:
        _cleanup_ytdlp_sidecars(full_path)
        return False

    # Now we process the downloaded chunk via FFmpeg to crop/blur/encode exactly to OUT_W/OUT_H
    ok = run_ffmpeg([
        "ffmpeg", "-y", "-i", str(full_path),
        "-t", str(max(5.0, min(8.0, float(duration_sec)))),
        "-filter_complex", BLUR_BG_FILTER,
        "-r", str(FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "21", "-an",
        str(out_path)
    ], timeout=180)
    
    full_path.unlink(missing_ok=True)
    _cleanup_ytdlp_sidecars(full_path)
    return ok


def _yt_search_candidates(
    query: str,
    max_results: int = 8,
    region_anchor: str = "",
    topic_hint: str = "",
    broll_type: str = "",
    extra_anchors: List[str] = None,
    strictness: str = "strict",
) -> List[Dict[str, Any]]:
    ytdlp_ok = False
    if YTDLP_PATH:
        if "/" in YTDLP_PATH or "\\" in YTDLP_PATH:
            ytdlp_ok = Path(YTDLP_PATH).exists()
        else:
            ytdlp_ok = shutil.which(YTDLP_PATH) is not None
    if not ytdlp_ok:
        return []
    cmd = [
        YTDLP_PATH, f"ytsearch{max_results}:{query}",
        "--flat-playlist",
        "--print", "%(id)s|||%(title)s|||%(duration)s|||%(view_count)s|||%(webpage_url)s|||%(channel)s|||%(uploader)s",
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
        "podcast", "interview", "compilation", "slideshow",
        "lecture", "reaction",
        "shutterstock", "getty", "istock", "dreamstime", "adobe stock",
        "footage farm", "videoblocks", "storyblocks", "pond5", "artgrid",
        "royalty free", "royalty-free", "watermark",
        "lyrics", "music video",
        "nba", "nfl", "mlb", "highlights reel",
    }
    # Initial search in requested strictness
    found = _process_yt_results(text, bad_terms, region_anchor, topic_hint, broll_type, extra_anchors, strictness)
    
    # Adaptive Fallback: If strict mode (default) found NOTHING, try again with medium
    if not found and strictness == "strict":
        print(f" (strict filter returned 0, attempting medium fallback)", end="", flush=True)
        found = _process_yt_results(text, bad_terms, region_anchor, topic_hint, broll_type, extra_anchors, "medium")
        
    return found

def _process_yt_results(text: str, bad_terms: set[str], region_anchor: str, topic_hint: str, 
                        broll_type: str, extra_anchors: List[str], strictness: str) -> List[Dict[str, Any]]:
    # Require at least one positive visual term — anything without these is not b-roll footage
    _VISUAL_TERMS = {
        "drone", "aerial", "footage", "walk", "walking", "street", "drive",
        "driving", "flight", "timelapse", "time lapse", "cam", "camera",
        "downtown", "skyline", "nature", "wildlife", "4k", "hd", "cinematic",
        "tour", "view", "neighborhood", "hood", "district",
        "beach", "river", "lake", "forest", "mountain", "highway", "traffic",
        "nightlife", "night", "helicopter", "gopro", "dashcam", "pov",
        "city", "metropolis", "capital", "urban", "scenery", "building", "architecture", "bridge",
    }
    
    # Extract geographic anchors from region context (not full query) to avoid over-constraining.
    anchor_text = region_anchor or ""
    if extra_anchors:
        anchor_text = f"{anchor_text} {' '.join(extra_anchors)}".strip()
    geo_kw = set(_anchor_tokens(anchor_text))
    strict_mode = str(strictness or "strict").strip().lower()
    out: List[Dict[str, Any]] = []
    
    for line in text.splitlines():
        parts = line.split("|||")
        if len(parts) < 5:
            continue
        vid, title, dur, views, url = parts[:5]
        channel = parts[5] if len(parts) > 5 else ""
        uploader = parts[6] if len(parts) > 6 else ""
        title_l = title.lower()
        if any(term in title_l for term in bad_terms):
            continue
        # Allowlist: require at least one positive visual term.
        if not any(term in title_l for term in _VISUAL_TERMS):
            continue
        # Geographic score bonus
        geo_in_title = bool(geo_kw and any(kw in title_l for kw in geo_kw))
        if not _title_passes_relevance(
            title,
            region_anchor=region_anchor,
            topic_hint=topic_hint,
            broll_type=broll_type,
            extra_anchors=extra_anchors,
            strictness=strict_mode,
        ):
            continue
        try: d = float(dur or 0)
        except Exception: d = 0
        try: v = float(views or 0)
        except Exception: v = 0
        score = 0.0
        if "4k" in title_l or "60fps" in title_l: score += 2.0
        if any(k in title_l for k in ("walk", "drone", "aerial", "street", "nature", "coast", "mountain")): score += 2.0
        if d >= 30: score += 1.0
        if v > 10000: score += 1.0
        if geo_in_title: score += 3.0
        out.append({
            "video_id": vid,
            "title": title,
            "duration": d,
            "views": v,
            "url": url,
            "channel": channel,
            "uploader": uploader,
            "score": score,
            "geo_in_title": geo_in_title,
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def _yt_download_and_trim(
    video: Dict[str, Any],
    out_path: Path,
    duration_sec: float,
    region_anchor: str = "",
    topic_hint: str = "",
    broll_type: str = "",
    extra_anchors: List[str] = None,
    strictness: str = "strict",
    segment_index: int = 0,
) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_path = out_path.parent / f"_yt_{video['video_id']}_full.mp4"
    dl_cmd = [
        YTDLP_PATH,
        "-f", "bv*[vcodec^=avc1][height<=1080]+ba[ext=m4a]/b[ext=mp4]/best",
        "--recode-video", "mp4", "--merge-output-format", "mp4",
        "-o", str(full_path), "--retries", "2", "--limit-rate", "2M",
        "--sleep-interval", "1", "--max-sleep-interval", "3",
        "--no-warnings", "--ignore-errors",
        video.get("url") or f"https://www.youtube.com/watch?v={video['video_id']}",
    ]
    if COOKIES_YT and Path(COOKIES_YT).exists():
        dl_cmd[1:1] = ["--cookies", COOKIES_YT]
    try:
        r = subprocess.run(dl_cmd, capture_output=True, timeout=120)
        if r.returncode != 0 or not full_path.exists():
            _cleanup_ytdlp_sidecars(full_path)
            return False
            
        # Minimum File Size Gate (500KB)
        if full_path.stat().st_size < 500000:
            print(f" (reject: corrupt/tiny download {full_path.stat().st_size/1024:.0f}KB)", end="", flush=True)
            full_path.unlink(missing_ok=True)
            _cleanup_ytdlp_sidecars(full_path)
            return False
    except Exception:
        _cleanup_ytdlp_sidecars(full_path)
        return False
        
    if not _title_passes_relevance(
        str(video.get("title", "") or ""),
        region_anchor=region_anchor,
        topic_hint=topic_hint,
        broll_type=broll_type,
        extra_anchors=extra_anchors,
        strictness=strictness,
    ):
        print(" [reject: off-topic candidate]", end="", flush=True)
        full_path.unlink(missing_ok=True)
        _cleanup_ytdlp_sidecars(full_path)
        return False

    target = max(5.0, min(8.0, float(duration_sec)))
    src_dur = max(0.0, float(video.get("duration") or 0.0))
    if src_dur > target + 6:
        # Spread segments across the source so reused videos show different content
        positions = [0.5, 0.25, 0.75, 0.1, 0.9]
        pos = positions[segment_index % len(positions)]
        start = max(2.0, src_dur * pos - target / 2)
        start = min(start, src_dur - target - 1)
    else:
        start = 1.0
    ok = run_ffmpeg([
        "ffmpeg", "-y", "-ss", str(start), "-i", str(full_path),
        "-t", str(target), "-filter_complex", BLUR_BG_FILTER,
        "-r", str(FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "21", "-an",
        str(out_path)
    ], timeout=180)
    full_path.unlink(missing_ok=True)
    _cleanup_ytdlp_sidecars(full_path)
    return ok


def _download_http_clip_and_trim(video_url: str, out_path: Path, duration_sec: float,
                                 src_duration: float = 0.0, tag: str = "src") -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_path = out_path.parent / f"_{tag}_full.mp4"
    req = urllib.request.Request(video_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=90, context=_SSL_CTX) as resp:
            full_path.write_bytes(resp.read())
    except Exception:
        return False
    if not full_path.exists() or full_path.stat().st_size < 10240:
        full_path.unlink(missing_ok=True)
        return False
    target = max(5.0, min(8.0, float(duration_sec)))
    src_dur = max(0.0, float(src_duration or 0.0))
    start = max(1.5, (src_dur - target) / 2) if src_dur > target + 6 else 0.5
    ok = run_ffmpeg([
        "ffmpeg", "-y", "-ss", str(start), "-i", str(full_path),
        "-t", str(target), "-filter_complex", BLUR_BG_FILTER,
        "-r", str(FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "21", "-an",
        str(out_path)
    ], timeout=180)
    full_path.unlink(missing_ok=True)
    return ok


def _extract_middle_frame(video_path: Path) -> Optional[Path]:
    if not video_path.exists():
        return None

    tmp_dir = Path(tempfile.gettempdir()) / "broll_geo_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    frame_path = tmp_dir / f"{video_path.stem}_mid.jpg"

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        duration = float(json.loads(probe.stdout).get("format", {}).get("duration", 0) or 0)
    except Exception:
        duration = 0.0

    timestamp = max(0.4, duration * 0.5) if duration > 0 else 0.8
    ok = run_ffmpeg(
        [
            "ffmpeg", "-y", "-ss", str(timestamp), "-i", str(video_path),
            "-vframes", "1", "-q:v", "4", str(frame_path),
        ],
        timeout=60,
    )
    if ok and frame_path.exists() and frame_path.stat().st_size > 1024:
        return frame_path
    frame_path.unlink(missing_ok=True)
    return None


def _extract_multi_frames(video_path: Path, count: int = 3) -> List[Path]:
    """Extract frames at 25%, 50%, 75% of the clip for multi-point validation."""
    if not video_path.exists():
        return []

    tmp_dir = Path(tempfile.gettempdir()) / "broll_geo_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        duration = float(json.loads(probe.stdout).get("format", {}).get("duration", 0) or 0)
    except Exception:
        duration = 0.0

    if duration <= 0:
        mid = _extract_middle_frame(video_path)
        return [mid] if mid else []

    positions = [0.25, 0.5, 0.75][:count]
    frames: List[Path] = []
    for i, frac in enumerate(positions):
        ts = max(0.2, duration * frac)
        fp = tmp_dir / f"{video_path.stem}_f{i}.jpg"
        ok = run_ffmpeg(
            [
                "ffmpeg", "-y", "-ss", str(ts), "-i", str(video_path),
                "-vframes", "1", "-q:v", "4", str(fp),
            ],
            timeout=30,
        )
        if ok and fp.exists() and fp.stat().st_size > 1024:
            frames.append(fp)
        else:
            fp.unlink(missing_ok=True)
    return frames


def _has_news_ticker_band(frame_path: Path) -> bool:
    try:
        with Image.open(str(frame_path)) as img_obj:
            img = img_obj.convert("L")
    except Exception:
        return False

    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape
    if h < 80 or w < 80:
        return False

    bottom = arr[int(h * 0.85):, :]
    top = arr[: max(1, int(h * 0.2)), :]
    if bottom.size == 0 or top.size == 0:
        return False

    bright_ratio = float((bottom > 210).mean())
    top_bright_ratio = float((top > 210).mean())

    grad_x = np.abs(np.diff(bottom, axis=1))
    grad_y = np.abs(np.diff(bottom, axis=0)) if bottom.shape[0] > 1 else np.zeros_like(bottom[:, :1])
    edge_x_ratio = float((grad_x > 35).mean()) if grad_x.size else 0.0
    edge_y_ratio = float((grad_y > 35).mean()) if grad_y.size else 0.0

    # Tickers usually create a dense, high-contrast horizontal text region.
    return (
        bright_ratio > 0.055
        and edge_x_ratio > 0.19
        and edge_y_ratio > 0.08
        and bright_ratio > top_bright_ratio * 1.5
    )


def _has_burned_captions(frame_path: Path) -> bool:
    """Detect burned-in captions/subtitles/text overlays in a video frame.

    Looks for high-contrast text-like regions (white/yellow text with dark
    outlines) in the bottom 40% and top 25% of the frame — where captions,
    titles, and channel watermarks typically appear.
    """
    try:
        with Image.open(str(frame_path)) as img_obj:
            img = img_obj.convert("RGB")
    except Exception:
        return False

    arr = np.asarray(img, dtype=np.float32)
    h, w, _ = arr.shape
    if h < 100 or w < 100:
        return False

    # Convert to grayscale for edge detection
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    # Check two zones where text overlays typically appear
    zones = [
        gray[int(h * 0.60):, :],   # bottom 40% — subtitles, captions
        gray[:int(h * 0.25), :],    # top 25% — titles, channel names
    ]

    for zone in zones:
        if zone.size == 0 or zone.shape[0] < 10:
            continue

        # High-contrast edge detection — text creates sharp edges
        grad_x = np.abs(np.diff(zone, axis=1))
        grad_y = np.abs(np.diff(zone, axis=0))

        # Text has many sharp edges (gradient > 50) in both directions
        sharp_x = float((grad_x > 50).mean()) if grad_x.size else 0.0
        sharp_y = float((grad_y > 50).mean()) if grad_y.size else 0.0

        # Very bright pixels (white/yellow text) in the zone
        bright = float((zone > 220).mean())

        # Text signature: lots of sharp horizontal edges + bright pixels
        # Typical natural footage: sharp_x < 0.08, bright < 0.15
        # Burned captions: sharp_x > 0.12, bright > 0.10, with vertical edges too
        if sharp_x > 0.12 and sharp_y > 0.06 and bright > 0.10:
            return True

        # Also catch dark-background caption boxes (semi-transparent bars)
        dark = float((zone < 35).mean())
        if dark > 0.25 and sharp_x > 0.10 and bright > 0.06:
            return True

    return False

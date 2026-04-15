"""
broll_overlays.py — Animated overlay sub-beats that composite onto a moving primary clip.

Each overlay type:
  animated_stat     — large number counting up + label text
  timeline_marker   — dramatic year/era + event text kinetic reveal
    context_photo     — centred contextual image/doc card on blurred background
                                             (historical_photo is accepted as a backward-compatible alias)
  flow_map          — animated arrow on satellite map showing movement/direction
  split_comparison  — side-by-side before/after images

Public API
----------
apply_overlay_to_clip(primary_clip, overlay_def, out_path, duration_sec, geo) -> bool
    Given an already-rendered primary clip and an overlay dict from the script, composites
    the overlay starting at overlay_def["at_sec"] and saves to out_path.

    The primary clip is always visible and playing behind the overlay.
    The region behind a stat/marker card is blurred (frosted glass), so footage still reads.
"""
from __future__ import annotations

import io
import math
import os
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .config import (
    OUT_W, OUT_H, FPS, _SSL_CTX, USER_AGENT, FONTS_DIR,
)
from .ffmpeg_utils import run_ffmpeg

# ── Font helpers ─────────────────────────────────────────────────────────────

def _font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        FONTS_DIR / "Montserrat-Bold.ttf",
        FONTS_DIR / "montserrat_bold.ttf",
        FONTS_DIR / "Outfit-Bold.ttf",
        Path("C:/Windows/Fonts/arialbd.ttf"),
    ]
    for p in candidates:
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                continue
    return ImageFont.load_default()


def _font_regular(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        FONTS_DIR / "Montserrat-Regular.ttf",
        FONTS_DIR / "Outfit-Regular.ttf",
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for p in candidates:
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                continue
    return ImageFont.load_default()


# ── Easing ────────────────────────────────────────────────────────────────────

def _ease_out(t: float) -> float:
    return 1 - (1 - t) ** 3


# ── Image fetch ───────────────────────────────────────────────────────────────

def _clean_search_text(text: str) -> str:
    cleaned = re.sub(r"[_\-]+", " ", str(text or ""))
    cleaned = re.sub(r"\.(jpg|jpeg|png|webp|svg)$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _query_from_url(url: str | None) -> str:
    if not url or "http" not in str(url):
        return ""
    try:
        parsed = urllib.parse.urlparse(str(url))
        base = Path(urllib.parse.unquote(parsed.path)).name
        return _clean_search_text(base)
    except Exception:
        return ""


def _fetch_image(url: str | None, query: str | None, cache_dir: Path, fallback_query: str | None = None) -> Optional[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if url and url.startswith("http"):
        safe = "".join(c if c.isalnum() else "_" for c in url[-40:])
        dest = cache_dir / f"img_{safe}.jpg"
        if dest.exists() and dest.stat().st_size > 1024:
            return dest
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=20, context=_SSL_CTX) as r:
                raw = r.read()
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            img.save(str(dest), "JPEG", quality=92)
            return dest
        except Exception:
            pass

    queries: List[str] = []
    for q in (query, fallback_query, _query_from_url(url)):
        qq = _clean_search_text(str(q or ""))
        if qq and qq not in queries:
            queries.append(qq)

    for q in queries:
        img = _wikimedia_search(q, cache_dir, region_anchor=fallback_query or "")
        if img:
            return img
    return None


def _fetch_wikipedia_pageimage(title: str, cache_dir: Path) -> Optional[Path]:
    """Fetch the primary thumbnail lead image from a Wikipedia article title."""
    if not title:
        return None
    safe_title = "".join(c if c.isalnum() else "_" for c in title[:40])
    dest = cache_dir / f"wp_{safe_title}.jpg"
    if dest.exists() and dest.stat().st_size > 1024:
        return dest
    try:
        enc = urllib.parse.quote(title)
        api = (
            f"https://en.wikipedia.org/w/api.php?action=query&prop=pageimages"
            f"&titles={enc}&pithumbsize=1080&redirects=1&format=json"
        )
        req = urllib.request.Request(api, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=10, context=_SSL_CTX) as r:
            import json
            data = json.loads(r.read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        for pg in pages.values():
            thumb = pg.get("thumbnail", {}).get("source", "")
            if thumb:
                req2 = urllib.request.Request(thumb, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req2, timeout=15, context=_SSL_CTX) as r2:
                    raw2 = r2.read()
                img = Image.open(io.BytesIO(raw2)).convert("RGB")
                img.save(str(dest), "JPEG", quality=92)
                return dest
    except Exception as e:
        print(f" [wiki lookup fail: {e}]", end="", flush=True)
    return None


def _wikimedia_search(query: str, cache_dir: Path, region_anchor: str = "") -> Optional[Path]:
    search_q = query
    if region_anchor and region_anchor.lower() not in query.lower():
        search_q = f"{query} {region_anchor}"

    safe_q = "".join(c if c.isalnum() else "_" for c in search_q[:40])
    dest = cache_dir / f"wiki_{safe_q}.jpg"
    if dest.exists() and dest.stat().st_size > 1024:
        return dest
    params = urllib.parse.urlencode({
        "action": "query", "format": "json",
        "generator": "search",
        "gsrnamespace": "6",
        "gsrsearch": f"filetype:bitmap {search_q}",
        "gsrlimit": "12",
        "prop": "imageinfo",
        "iiprop": "url|mime|size",
        "iiurlwidth": "1080",
    })
    api = f"https://commons.wikimedia.org/w/api.php?{params}"
    try:
        req = urllib.request.Request(api, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as r:
            import json
            data = json.loads(r.read().decode("utf-8"))
        pages = list((data.get("query", {}).get("pages", {}) or {}).values())
        query_tokens = {tok for tok in _clean_search_text(query).lower().split() if len(tok) > 2}

        def _score_page(page: Dict[str, Any]) -> int:
            title = str(page.get("title", "")).lower()
            token_hits = sum(1 for tok in query_tokens if tok in title)
            info = (page.get("imageinfo") or [{}])[0]
            width = int(info.get("width") or 0)
            height = int(info.get("height") or 0)
            size_bonus = 1 if min(width, height) >= 720 else 0
            year_bonus = 1 if re.search(r"(18|19|20)\d{2}|\d{4}s", title) else 0
            return token_hits * 3 + size_bonus + year_bonus

        pages.sort(key=_score_page, reverse=True)

        for p in pages:
            ii = (p.get("imageinfo") or [{}])[0]
            if not ii.get("mime", "").startswith("image/"):
                continue
            img_url = ii.get("thumburl") or ii.get("url")
            if not img_url:
                continue
            req2 = urllib.request.Request(img_url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req2, timeout=20, context=_SSL_CTX) as r2:
                raw = r2.read()
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            img.save(str(dest), "JPEG", quality=92)
            return dest
    except Exception:
        pass
    # Wikipedia fallback
    try:
        enc = urllib.parse.quote(query)
        api2 = (
            f"https://en.wikipedia.org/w/api.php?action=query&prop=pageimages"
            f"&titles={enc}&pithumbsize=1080&redirects=1&format=json"
        )
        req3 = urllib.request.Request(api2, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req3, timeout=10, context=_SSL_CTX) as r3:
            import json
            d2 = json.loads(r3.read().decode("utf-8"))
        pages2 = d2.get("query", {}).get("pages", {})
        for pg in pages2.values():
            thumb = pg.get("thumbnail", {}).get("source", "")
            if thumb:
                req4 = urllib.request.Request(thumb, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req4, timeout=15, context=_SSL_CTX) as r4:
                    raw4 = r4.read()
                img4 = Image.open(io.BytesIO(raw4)).convert("RGB")
                img4.save(str(dest), "JPEG", quality=92)
                return dest
    except Exception:
        pass
    return None


# ── Text wrap ─────────────────────────────────────────────────────────────────

def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int, draw: ImageDraw.ImageDraw) -> List[str]:
    words = text.split()
    lines, current = [], ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]


# ── Frame renderers ───────────────────────────────────────────────────────────

def _render_animated_stat_frames(
    value: str, unit: str, label: str, style: str, n_frames: int, tmp_dir: Path,
    stat_category: str = "", stat_prefix: str = "", stat_suffix: str = "",
    stat_context: str = "", compare_to: str = "",
) -> bool:
    """
    Category-aware stat counter with different animations per type.

    stat_category:
        big_number   — count-up with easing (686 deaths, 8000 lives)
        percentage   — count-up then "%" snaps on
        ranking      — #1-5: slam/punch-in, #6+: fast ticker
        comparison   — count "1x → 2x → 3x"
        measurement  — count-up with inline unit (1,954 MILES)
        year         — reveal, no counting
        ratio        — reveal with emphasis (1 in 4)
    """
    # Parse value
    clean = str(value).replace(",", "").replace(" ", "").strip()
    try:
        target = float(clean)
    except ValueError:
        target = 0.0
    is_int = "." not in clean and target == int(target) if target else True

    # Auto-detect category if not specified
    if not stat_category:
        if stat_prefix == "#" or "rank" in label.lower():
            stat_category = "ranking"
        elif stat_suffix == "%" or "percent" in unit.lower():
            stat_category = "percentage"
        elif any(u in unit.lower() for u in ("miles", "km", "feet", "meters", "sq")):
            stat_category = "measurement"
        elif compare_to or "x" in stat_suffix.lower() or "times" in label.lower():
            stat_category = "comparison"
        elif 1900 <= target <= 2100 and not unit:
            stat_category = "year"
        elif "in" in str(value) and not clean.replace(".", "").isdigit():
            stat_category = "ratio"
        else:
            stat_category = "big_number"

    # ── Style tokens ────────────────────────────────────────────────────────
    if style == "light":
        card_fill  = (255, 255, 255, 210)
        num_color  = (255, 80, 40)
        unit_color = (20, 20, 30)
        sub_color  = (70, 70, 90)
        border_col = (200, 200, 210, 200)
    else:
        card_fill  = (12, 12, 22, 225)
        num_color  = (255, 200, 50)
        unit_color = (240, 240, 255)
        sub_color  = (170, 170, 195)
        border_col = (60, 60, 90, 200)

    # ── Layout ───────────────────────────────────────────────────────────────
    card_w  = int(OUT_W * 0.88)
    padding = 52
    num_size   = min(180, card_w // 4)
    unit_size  = min(72, card_w // 7)
    label_size = min(48, card_w // 10)
    val_font   = _font(num_size)
    unit_font  = _font(unit_size)
    label_font = _font_regular(label_size)

    dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    num_h  = dummy.textbbox((0, 0), "0", font=val_font)[3]
    unit_h = dummy.textbbox((0, 0), "A", font=unit_font)[3] if (unit or stat_suffix or stat_context) else 0
    lab_h  = dummy.textbbox((0, 0), "A", font=label_font)[3] if label else 0

    card_h  = padding + num_h + (12 + unit_h if unit_h else 0) + (12 + lab_h if lab_h else 0) + padding
    card_x  = (OUT_W - card_w) // 2
    card_y  = (OUT_H - card_h) // 2

    fade_in_frames = max(1, n_frames // 5)
    count_dur_frames = int(FPS * 1.25)

    # ── Animation logic per category ─────────────────────────────────────────

    def _format_number(val: float) -> str:
        if is_int:
            return f"{int(round(val)):,}"
        return f"{val:,.1f}"

    def _count_ease(f: int) -> float:
        """Ease-out curve: fast start, decelerating."""
        t = min(1.0, f / max(1, count_dur_frames))
        return target * (t ** 0.45)

    def _display_at_frame(f: int) -> str:
        """Return the display string for a given frame, based on category."""
        if stat_category == "ranking":
            if target <= 5:
                # Slam: show final value immediately after fade-in
                return f"#{int(target)}"
            else:
                # Fast ticker: rapidly count down from a high number
                if f < count_dur_frames:
                    t = min(1.0, f / max(1, count_dur_frames))
                    # Overshoot then settle — start high, tick down
                    overshoot = target * 3
                    current = overshoot - (overshoot - target) * (t ** 0.3)
                    return f"#{int(max(target, current))}"
                return f"#{int(target)}"

        elif stat_category == "percentage":
            if f < count_dur_frames:
                current = _count_ease(f)
                return _format_number(current)
            return _format_number(target) + "%"

        elif stat_category == "comparison":
            if f < count_dur_frames:
                t = min(1.0, f / max(1, count_dur_frames))
                current = max(1, target * (t ** 0.5))
                return f"{int(round(current))}x"
            return f"{int(target)}x"

        elif stat_category == "year":
            return str(int(target)) if is_int else str(value)

        elif stat_category == "ratio":
            return str(value)

        elif stat_category == "measurement":
            if f < count_dur_frames:
                current = _count_ease(f)
                suffix = f" {stat_suffix}" if stat_suffix else (f" {unit.upper()}" if unit else "")
                return _format_number(current) + suffix
            suffix = f" {stat_suffix}" if stat_suffix else (f" {unit.upper()}" if unit else "")
            return _format_number(target) + suffix

        else:  # big_number
            if f < count_dur_frames:
                return _format_number(_count_ease(f))
            return _format_number(target)

    def _card_alpha_at_frame(f: int) -> float:
        """Fade-in envelope. Rankings #1-5 use a slam (overshoot bounce)."""
        if stat_category == "ranking" and target <= 5:
            # Slam: fast appear with slight overshoot
            if f < 3:
                return 0.0
            t = min(1.0, (f - 3) / max(1, fade_in_frames * 0.5))
            overshoot = 1.0 + 0.15 * math.sin(t * math.pi)
            return min(1.0, overshoot * t)
        elif stat_category in ("year", "ratio"):
            # Smooth reveal with upward slide (handled via y-offset below)
            return _ease_out(min(1.0, f / max(1, fade_in_frames)))
        else:
            return _ease_out(min(1.0, f / max(1, fade_in_frames)))

    def _y_offset_at_frame(f: int) -> int:
        """Slide-up for reveal animations."""
        if stat_category in ("year", "ratio"):
            t = min(1.0, f / max(1, fade_in_frames))
            return int(30 * (1.0 - _ease_out(t)))
        if stat_category == "ranking" and target <= 5:
            # Slam: slight downward bounce
            if f < 3:
                return -20
            t = min(1.0, (f - 3) / max(1, fade_in_frames * 0.5))
            return int(-8 * math.sin(t * math.pi * 1.5) * (1 - t))
        return 0

    # ── Render frames ────────────────────────────────────────────────────────
    # Flash frame index for count-up categories (white flash when number lands)
    flash_frame = count_dur_frames if stat_category not in ("year", "ratio", "ranking") or target > 5 else -1

    for f in range(n_frames):
        frame = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
        draw  = ImageDraw.Draw(frame)

        alpha_t = _card_alpha_at_frame(f)
        alpha   = int(alpha_t * 255)
        y_off   = _y_offset_at_frame(f)

        # Card background
        a_card = int(card_fill[3] * alpha_t)
        draw.rounded_rectangle(
            [card_x, card_y + y_off, card_x + card_w, card_y + card_h + y_off],
            radius=36,
            fill=(*card_fill[:3], a_card),
            outline=(*border_col[:3], int(border_col[3] * alpha_t)),
            width=2,
        )

        # Display string
        display = _display_at_frame(f)

        # Dynamic font sizing
        fnt = val_font
        while True:
            bbox = draw.textbbox((0, 0), display, font=fnt)
            if (bbox[2] - bbox[0]) <= card_w - padding * 2:
                break
            new_size = int(fnt.size * 0.9)
            if new_size < 40:
                break
            fnt = _font(new_size)

        # Number / main display
        nb = draw.textbbox((0, 0), display, font=fnt)
        nw, nh = nb[2] - nb[0], nb[3] - nb[1]
        nx = card_x + (card_w - nw) // 2
        ny = card_y + padding + y_off
        draw.text((nx, ny), display, font=fnt, fill=(*num_color, alpha))

        cursor_y = ny + nh + 12

        # Unit / suffix / context line
        unit_line = stat_context or unit
        # For percentage, don't show unit during count-up (% is in the number)
        if stat_category == "percentage" and f >= count_dur_frames:
            unit_line = stat_context or ""
        # For measurement, unit is inline in the number
        if stat_category == "measurement":
            unit_line = stat_context or ""
        # For comparison, show "MORE THAN {compare_to}"
        if stat_category == "comparison" and compare_to:
            unit_line = f"MORE THAN {compare_to.upper()}"
        if unit_line:
            ub = draw.textbbox((0, 0), unit_line.upper(), font=unit_font)
            uw = ub[2] - ub[0]
            draw.text((card_x + (card_w - uw) // 2, cursor_y + y_off),
                      unit_line.upper(), font=unit_font, fill=(*unit_color, alpha))
            cursor_y += (ub[3] - ub[1]) + 12

        # Label
        if label:
            lb = draw.textbbox((0, 0), label.upper(), font=label_font)
            lw = lb[2] - lb[0]
            draw.text((card_x + (card_w - lw) // 2, cursor_y + y_off),
                      label.upper(), font=label_font, fill=(*sub_color, alpha))

        # Flash on landing
        if f in (flash_frame, flash_frame + 1) and flash_frame > 0:
            flash = Image.new("RGBA", (OUT_W, OUT_H), (255, 255, 255, 90 if f == flash_frame else 45))
            frame = Image.alpha_composite(frame, flash)

        frame.save(str(tmp_dir / f"frame_{f:05d}.png"))

    return True


def _render_timeline_marker_frames(
    year: str, event_text: str, style: str, n_frames: int, tmp_dir: Path
) -> bool:
    fade_in  = max(1, n_frames // 6)
    slide_px = int(OUT_H * 0.06)
    if style == "light":
        bg_col   = (255, 255, 255, 200)
        yr_col   = (20, 20, 30)
        ev_col   = (60, 60, 80)
        line_col = (255, 80, 40)
    else:
        bg_col   = (10, 10, 20, 215)
        yr_col   = (255, 200, 50)
        ev_col   = (230, 230, 240)
        line_col = (255, 200, 50)

    card_w = int(OUT_W * 0.86)
    card_x = (OUT_W - card_w) // 2
    yr_font  = _font(min(160, OUT_H // 14))
    ev_font  = _font_regular(min(58, OUT_H // 36))

    for f in range(n_frames):
        frame = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
        draw  = ImageDraw.Draw(frame)
        t     = _ease_out(min(1.0, f / max(1, fade_in)))
        alpha = int(t * 255)
        dy    = int(slide_px * (1 - t))

        yr_bbox  = draw.textbbox((0, 0), year, font=yr_font)
        yr_h     = yr_bbox[3] - yr_bbox[1]
        ev_lines = _wrap_text(event_text.upper(), ev_font, card_w - 60, draw)
        ev_lh    = draw.textbbox((0, 0), "A", font=ev_font)[3]
        card_h   = yr_h + ev_lh * len(ev_lines) + 80
        card_y   = (OUT_H - card_h) // 2

        draw.rounded_rectangle(
            [card_x, card_y + dy, card_x + card_w, card_y + card_h + dy],
            radius=28, fill=(*bg_col[:3], int(bg_col[3] * t))
        )
        draw.rectangle(
            [card_x, card_y + dy, card_x + 8, card_y + card_h + dy],
            fill=(*line_col[:3], alpha)
        )
        draw.text((card_x + 28, card_y + 20 + dy), year, font=yr_font,
                  fill=(*yr_col[:3], alpha))
        ey = card_y + 20 + yr_h + 10 + dy
        for line in ev_lines:
            draw.text((card_x + 28, ey), line, font=ev_font,
                      fill=(*ev_col[:3], alpha))
            ey += ev_lh + 6

        frame.save(str(tmp_dir / f"frame_{f:05d}.png"))
    return True


def _render_historical_photo_frames(
    image_path: Path, caption: str, credit: str, style: str, n_frames: int, tmp_dir: Path
) -> bool:
    img = Image.open(str(image_path)).convert("RGB")
    if style == "bw":
        import numpy as np
        arr = np.array(img)
        g = (0.299 * arr[...,0] + 0.587 * arr[...,1] + 0.114 * arr[...,2]).astype(np.uint8)
        arr[...,0] = arr[...,1] = arr[...,2] = g
        img = Image.fromarray(arr)
    elif style == "sepia":
        import numpy as np
        arr = np.array(img).astype(float)
        r = np.clip(arr[...,0]*0.393 + arr[...,1]*0.769 + arr[...,2]*0.189, 0, 255)
        g = np.clip(arr[...,0]*0.349 + arr[...,1]*0.686 + arr[...,2]*0.168, 0, 255)
        b = np.clip(arr[...,0]*0.272 + arr[...,1]*0.534 + arr[...,2]*0.131, 0, 255)
        arr[...,0], arr[...,1], arr[...,2] = r, g, b
        img = Image.fromarray(arr.astype(np.uint8))

    card_w = int(OUT_W * 0.82)
    card_h = int(OUT_H * 0.50)
    card_x = (OUT_W - card_w) // 2
    card_y = (OUT_H - card_h) // 2

    img_ratio  = img.width / img.height
    card_ratio = card_w / (card_h - 80)
    if img_ratio > card_ratio:
        fit_w = card_w - 20
        fit_h = int(fit_w / img_ratio)
    else:
        fit_h = card_h - 80
        fit_w = int(fit_h * img_ratio)
    img_resized = img.resize((fit_w, fit_h), Image.LANCZOS)

    cap_font   = _font_regular(42)
    credit_font = _font_regular(26)
    fade_in    = max(1, n_frames // 5)
    pan_range  = min(30, fit_w // 10)

    for f in range(n_frames):
        frame = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
        draw  = ImageDraw.Draw(frame)
        t     = _ease_out(min(1.0, f / max(1, fade_in)))
        alpha = int(t * 255)

        draw.rounded_rectangle(
            [card_x - 12, card_y - 12, card_x + card_w + 12, card_y + card_h + 12],
            radius=8, fill=(0, 0, 0, int(185 * t))
        )
        draw.rounded_rectangle(
            [card_x, card_y, card_x + card_w, card_y + card_h],
            radius=6, fill=(240, 235, 225, int(255 * t))
        )
        pan_x  = int(pan_range * f / max(1, n_frames - 1))
        img_x  = card_x + (card_w - fit_w) // 2 - pan_range // 2 + pan_x
        img_y  = card_y + 10
        frame.paste(img_resized, (img_x, img_y))
        if caption:
            cbbox = draw.textbbox((0, 0), caption, font=cap_font)
            cw    = cbbox[2] - cbbox[0]
            draw.text((card_x + (card_w - cw) // 2, card_y + card_h - 86),
                      caption, font=cap_font, fill=(40, 40, 40, alpha))
        if credit:
            credit_text = f"Source: {credit}"
            rbbox = draw.textbbox((0, 0), credit_text, font=credit_font)
            rw = rbbox[2] - rbbox[0]
            draw.text((card_x + (card_w - rw) // 2, card_y + card_h - 44),
                      credit_text, font=credit_font, fill=(70, 70, 70, alpha))
        frame.save(str(tmp_dir / f"frame_{f:05d}.png"))
    return True


def _render_split_comparison_frames(
    img_a: Path, img_b: Path, label_a: str, label_b: str, n_frames: int, tmp_dir: Path
) -> bool:
    ia = Image.open(str(img_a)).convert("RGB").resize((OUT_W // 2, OUT_H), Image.LANCZOS)
    ib = Image.open(str(img_b)).convert("RGB").resize((OUT_W // 2, OUT_H), Image.LANCZOS)
    cap_font  = _font(52)
    fade_in   = max(1, n_frames // 4)
    wipe_frames = max(1, n_frames // 3)

    for f in range(n_frames):
        frame = Image.new("RGB", (OUT_W, OUT_H), (0, 0, 0))
        frame.paste(ia, (0, 0))
        wipe_t    = _ease_out(min(1.0, f / max(1, wipe_frames)))
        b_px      = int((OUT_W // 2) * wipe_t)
        if b_px > 0:
            frame.paste(ib.crop((0, 0, b_px, OUT_H)), (OUT_W // 2, 0))
        draw      = ImageDraw.Draw(frame)
        alpha_t   = _ease_out(min(1.0, f / max(1, fade_in)))
        alpha     = int(alpha_t * 255)
        draw.line([(OUT_W // 2, 0), (OUT_W // 2, OUT_H)], fill=(255, 255, 255), width=4)
        if label_a:
            draw.text((20, 40), label_a.upper(), font=cap_font, fill=(255, 255, 255))
        if label_b:
            bb = draw.textbbox((0, 0), label_b.upper(), font=cap_font)
            draw.text((OUT_W - (bb[2] - bb[0]) - 20, 40), label_b.upper(),
                      font=cap_font, fill=(255, 255, 0))
        frame.save(str(tmp_dir / f"frame_{f:05d}.png"))
    return True


def _render_flow_map_frames(
    base_map: "Image.Image", origin: Tuple[int, int], dest: Tuple[int, int],
    label: str, n_frames: int, tmp_dir: Path
) -> bool:
    arrow_frames = max(1, n_frames * 2 // 3)
    label_font   = _font(52)
    for f in range(n_frames):
        frame = base_map.copy().convert("RGBA")
        draw  = ImageDraw.Draw(frame)
        t     = _ease_out(min(1.0, f / max(1, arrow_frames)))
        cx    = int(origin[0] + (dest[0] - origin[0]) * t)
        cy    = int(origin[1] + (dest[1] - origin[1]) * t)
        draw.line([origin, (cx, cy)], fill=(255, 200, 50, 80), width=14)
        draw.line([origin, (cx, cy)], fill=(255, 200, 50, 200), width=4)
        if t > 0.05:
            dx = dest[0] - origin[0]; dy = dest[1] - origin[1]
            length = math.hypot(dx, dy) or 1
            ux, uy = dx / length, dy / length
            px, py = -uy, ux
            tip   = (cx, cy)
            base1 = (cx - int(ux * 28 + px * 14), cy - int(uy * 28 + py * 14))
            base2 = (cx - int(ux * 28 - px * 14), cy - int(uy * 28 - py * 14))
            draw.polygon([tip, base1, base2], fill=(255, 200, 50, 220))
        if t > 0.5 and label:
            lt  = min(1.0, (t - 0.5) * 2)
            la  = int(_ease_out(lt) * 255)
            mx  = (origin[0] + dest[0]) // 2
            my  = (origin[1] + dest[1]) // 2 - 40
            bb  = draw.textbbox((0, 0), label.upper(), font=label_font)
            draw.text((mx - (bb[2] - bb[0]) // 2, my), label.upper(), font=label_font,
                      fill=(255, 255, 255, la), stroke_width=3, stroke_fill=(0, 0, 0, la))
        frame.convert("RGB").save(str(tmp_dir / f"frame_{f:05d}.png"))
    return True


# ── Composite PNG sequence directly onto primary clip ─────────────────────────

def _composite_pngs_onto_clip(
    primary_clip: Path,
    tmp_dir: Path,
    n_frames: int,
    at_sec: float,
    out_path: Path,
    blur_bg: bool = False,
) -> bool:
    """
    Composite RGBA PNG frames directly onto primary_clip via FFmpeg.
    Uses FFmpeg's image2 concat demuxer (no intermediate video encode),
    so alpha transparency works correctly.

    blur_bg=True blurs the area of the primary clip behind the overlay card using
    a boxblur on the entire bg (frosted glass effect).
    """
    list_file = tmp_dir / "frames.txt"
    frame_dur = 1.0 / FPS
    pngs = sorted(tmp_dir.glob("frame_*.png"))
    if not pngs:
        return False
    with list_file.open("w") as lf:
        for png in pngs:
            lf.write(f"file '{png.as_posix()}'\n")
            lf.write(f"duration {frame_dur:.6f}\n")
        # Repeat last frame to prevent truncation
        lf.write(f"file '{pngs[-1].as_posix()}'\n")

    if blur_bg:
        # Build filter: blur the bg, then overlay the RGBA PNGs on the blurred bg,
        # then blend blurred-composite with the original clip using a luma mask
        # derived from the overlay alpha — so only the card area is blurred.
        filter_complex = (
            f"[0:v]split=2[orig][toblu];"
            f"[toblu]boxblur=luma_radius=25:luma_power=2[blurred];"
            f"[1:v]setpts=PTS+{at_sec}/TB[ovl];"
            f"[blurred][ovl]overlay=0:0:enable='gte(t,{at_sec})':format=auto[blended];"
            f"[orig][blended]blend=all_expr='if(gte(T\\,{at_sec})\\,B\\,A)'[out]"
        )
    else:
        filter_complex = (
            f"[1:v]setpts=PTS+{at_sec}/TB[ovl];"
            f"[0:v][ovl]overlay=0:0:enable='gte(t,{at_sec})':format=auto[out]"
        )

    return run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(primary_clip),
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "19",
        "-r", str(FPS), "-an",
        str(out_path)
    ], timeout=300)


# ── Public API ────────────────────────────────────────────────────────────────

def apply_overlay_to_clip(
    primary_clip: Path,
    overlay_def: Dict[str, Any],
    out_path: Path,
    duration_sec: float,
    geo: Optional[Dict] = None,
    cache_dir: Optional[Path] = None,
) -> bool:
    """
    Composite an animated overlay sub-beat onto the primary clip.

    overlay_def keys:
            type        — animated_stat | timeline_marker | context_photo |
                    split_comparison | flow_map
      at_sec      — when overlay appears (default 0.8)
      style       — dark | light | bw | sepia

      animated_stat:    value (full integer, e.g. "21000000"), unit, label
      timeline_marker:  year, event_text
    context_photo: image_url, image_query, caption, image_credit
      split_comparison: image_url_a / image_query_a, image_url_b / image_query_b,
                        label_a, label_b
      flow_map:         label (renders arrow on top of primary clip frame)
    """
    otype   = str(overlay_def.get("type", "")).strip().lower()
    if otype == "historical_photo":
        otype = "context_photo"
    at_sec  = float(overlay_def.get("at_sec", 0.8))
    style   = str(overlay_def.get("style", "dark")).strip().lower()

    if cache_dir is None:
        cache_dir = primary_clip.parent / "_overlay_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    overlay_dur = max(1.0, duration_sec - at_sec)
    n_frames    = max(2, int(overlay_dur * FPS))

    with tempfile.TemporaryDirectory(prefix="overlay_frames_") as tmp_str:
        tmp_dir   = Path(tmp_str)
        ok_frames = False
        blur_bg   = False   # whether to blur the primary clip bg region

        if otype in ("animated_stat", "stat_counter", "stat_counter_clip"):
            blur_bg  = True
            # Support both old schema (value/unit/label) and new (stat_number/stat_category/...)
            stat_val = str(overlay_def.get("stat_number") or overlay_def.get("stat_value") or overlay_def.get("value", "0"))
            ok_frames = _render_animated_stat_frames(
                value         = stat_val,
                unit          = str(overlay_def.get("stat_unit") or overlay_def.get("unit", "") or ""),
                label         = str(overlay_def.get("stat_label") or overlay_def.get("label", "") or ""),
                style         = style,
                n_frames      = n_frames,
                tmp_dir       = tmp_dir,
                stat_category = str(overlay_def.get("stat_category", "") or ""),
                stat_prefix   = str(overlay_def.get("stat_prefix", "") or ""),
                stat_suffix   = str(overlay_def.get("stat_suffix", "") or ""),
                stat_context  = str(overlay_def.get("stat_context", "") or ""),
                compare_to    = str(overlay_def.get("compare_to", "") or ""),
            )

        elif otype == "timeline_marker":
            blur_bg  = True
            ok_frames = _render_timeline_marker_frames(
                year       = str(overlay_def.get("year", "") or ""),
                event_text = str(overlay_def.get("event_text", "") or ""),
                style      = style,
                n_frames   = n_frames,
                tmp_dir    = tmp_dir,
            )

        elif otype == "context_photo":
            blur_bg = True
            historical_style = style in {"bw", "sepia"}
            wiki_title = str(overlay_def.get("wikipedia_title", "")).strip()
            if not wiki_title:
                wiki_title = str(overlay_def.get("image_query", "") or "").strip()
            if not wiki_title:
                wiki_title = str(overlay_def.get("caption", "") or "").strip()

            credit = str(overlay_def.get("image_credit") or overlay_def.get("credit") or "").strip()
            if not credit and str(overlay_def.get("wikipedia_title", "") or "").strip():
                credit = f"Wikipedia — {str(overlay_def.get('wikipedia_title')).strip()}"
            if not credit and str(overlay_def.get("image_url", "") or "").startswith("http"):
                try:
                    netloc = urllib.parse.urlparse(str(overlay_def.get("image_url"))).netloc.lower()
                    credit = re.sub(r"^www\.", "", netloc)
                except Exception:
                    credit = ""
            if not credit and str(overlay_def.get("image_query", "") or "").strip():
                credit = "Wikimedia Commons"

            region = str(geo.get("_region", "") or "").strip()
            img_path = _fetch_wikipedia_pageimage(wiki_title, cache_dir) if wiki_title else None
            if not img_path and wiki_title and region and region.lower() not in wiki_title.lower():
                # Try again with region anchor if specific title failed
                img_path = _fetch_wikipedia_pageimage(f"{wiki_title} ({region})", cache_dir)

            if not img_path and not historical_style:
                # Fallback for search
                img_path = _fetch_image(
                    overlay_def.get("image_url"),
                    overlay_def.get("image_query"),
                    cache_dir,
                    fallback_query=region,
                )
            if img_path:
                ok_frames = _render_historical_photo_frames(
                    image_path = img_path,
                    caption    = str(overlay_def.get("caption", "") or ""),
                    credit     = credit,
                    style      = style,
                    n_frames   = n_frames,
                    tmp_dir    = tmp_dir,
                )

        elif otype == "split_comparison":
            img_a = _fetch_image(
                overlay_def.get("image_url_a"),
                overlay_def.get("image_query_a"),
                cache_dir,
                fallback_query=str(overlay_def.get("label_a") or overlay_def.get("year_a") or ""),
            )
            img_b = _fetch_image(
                overlay_def.get("image_url_b"),
                overlay_def.get("image_query_b"),
                cache_dir,
                fallback_query=str(overlay_def.get("label_b") or overlay_def.get("year_b") or ""),
            )
            if img_a and img_b:
                ok_frames = _render_split_comparison_frames(
                    img_a    = img_a,
                    img_b    = img_b,
                    label_a  = str(overlay_def.get("label_a") or overlay_def.get("year_a") or "BEFORE"),
                    label_b  = str(overlay_def.get("label_b") or overlay_def.get("year_b") or "AFTER"),
                    n_frames = n_frames,
                    tmp_dir  = tmp_dir,
                )

        elif otype == "flow_map":
            base_frame_path = cache_dir / "_flow_base.png"
            run_ffmpeg([
                "ffmpeg", "-y", "-ss", "0", "-i", str(primary_clip),
                "-frames:v", "1", "-q:v", "2", str(base_frame_path)
            ], timeout=30)
            if base_frame_path.exists():
                base_img = Image.open(str(base_frame_path)).resize((OUT_W, OUT_H), Image.LANCZOS)
                ok_frames = _render_flow_map_frames(
                    base_map = base_img,
                    origin   = (int(OUT_W * 0.25), int(OUT_H * 0.35)),
                    dest     = (int(OUT_W * 0.72), int(OUT_H * 0.65)),
                    label    = str(overlay_def.get("label", "") or ""),
                    n_frames = n_frames,
                    tmp_dir  = tmp_dir,
                )

        if not ok_frames or not list(tmp_dir.glob("frame_*.png")):
            print(f"  [overlay:{otype}] frame render failed — skipping", flush=True)
            shutil.copy2(str(primary_clip), str(out_path))
            return True

        ok = _composite_pngs_onto_clip(
            primary_clip = primary_clip,
            tmp_dir      = tmp_dir,
            n_frames     = n_frames,
            at_sec       = at_sec,
            out_path     = out_path,
            blur_bg      = blur_bg,
        )

    if not ok or not out_path.exists():
        shutil.copy2(str(primary_clip), str(out_path))
    return True

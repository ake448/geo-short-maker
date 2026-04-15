"""
assembly.py — Stage 3-5: asset generation, highlight overlays, video assembly,
              title cards, caption burning, and audio mixing.
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw

from .config import OUT_W, OUT_H, FPS, MUSIC_DIR
from .ffmpeg_utils import run_ffmpeg
from .overlays import (
    _font, draw_feature_highlight, fetch_feature_geometry,
    burn_title_overlay,
)
from .captions import generate_ass_captions_from_whisper, burn_ass_captions
from .footage import mix_voiceover
from .footage_stock import gen_real_youtube_clip
from .broll_earth import BROLL_GENERATORS, _gen_3d, _HAS_3D_RENDERER
from .broll_overlays import apply_overlay_to_clip
from .branding import apply_branding
from .hook_card import burn_hook_text
from .color_grade import grade_clip

# Conditionally import gen_comparison_map
try:
    from gen import gen_comparison_map as _gen_comparison_map
except ImportError:
    _gen_comparison_map = None


def _apply_highlight_to_still(still_path: Path, beat: Dict, geo: Dict) -> bool:
    """If beat has a highlight object, fetch feature geometry and draw overlay."""
    return False
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
    """For video clips, overlay a highlight by rendering highlighted frames."""
    return False
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
    overlay_path = video_path.parent / f"_hl_{video_path.stem}.png"
    overlay_img = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay_img)
    for ring in feature_rings:
        pts = [(int(round(x)), int(round(y))) for x, y in ring]
        if len(pts) < 2: continue
        if is_line or len(pts) < 4:
            draw.line(pts, fill=(r_c, g_c, b_c, 40), width=18)
            draw.line(pts, fill=(r_c, g_c, b_c, 80), width=10)
            draw.line(pts, fill=(r_c, g_c, b_c, 220), width=4)
        else:
            if len(pts) >= 3: draw.polygon(pts, fill=(r_c, g_c, b_c, 80))
            draw.line(pts + [pts[0]], fill=(r_c, g_c, b_c, 200), width=3)
    overlay_img.save(str(overlay_path), "PNG")
    highlighted_path = video_path.parent / f"_hl_out_{video_path.stem}.mp4"
    ok = run_ffmpeg([
        "ffmpeg", "-y", "-i", str(video_path), "-i", str(overlay_path),
        "-filter_complex", "[0:v][1:v]overlay=0:0:format=auto",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-r", str(FPS), "-an", str(highlighted_path)
    ], timeout=180)
    overlay_path.unlink(missing_ok=True)
    if ok and highlighted_path.exists():
        shutil.move(str(highlighted_path), str(video_path))
        print(f" [highlight:{query}]", end="", flush=True)
        return True
    highlighted_path.unlink(missing_ok=True)
    return False


def generate_assets(script: Dict, geo: Dict, run_dir: Path, region: str,
                    allow_real_footage: bool = True,
                    default_render_mode: str = "auto") -> Dict[str, Path]:
    """Stage 3: Generate B-roll assets for each beat."""
    print("\n[S3] Generating B-roll assets...")
    clips_dir = run_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    assets = {}
    sourcing_report: List[Dict[str, Any]] = []
    used_video_ids: dict = {}  # video_id -> use count; enables timestamp variation on reuse
    non_real_fallbacks = {
        "google_earth_pan": ["3d_orbit", "satellite_pan", "map_wipe"],
        "satellite_pan": ["3d_orbit", "map_wipe"],
        "3d_flyover": ["3d_orbit", "satellite_pan"],
        "3d_zoom": ["3d_orbit", "satellite_pan"],
        "3d_close_oblique": ["3d_orbit", "satellite_pan"],
        "comparison_map": ["3d_flyover", "map_wipe"],
        "outline_reveal": ["3d_orbit", "satellite_pan"],
        "region_comparison": ["3d_orbit", "map_highlight"],
        "multi_region_reveal": ["map_highlight", "terrain_map"],
        "zoom_to_region": ["3d_zoom", "3d_orbit", "satellite_pan"],
        "size_comparison_overlay": ["3d_orbit", "map_highlight"],
        "chokepoint_flow": ["satellite_pan", "map_wipe"],
        "stat_counter_clip": ["map_highlight", "terrain_map"],
    }
    generated_type_counts: Dict[str, int] = {}

    def _ordered_fallbacks(candidates: List[str], previous_type: str) -> List[str]:
        ordered = sorted(
            candidates,
            key=lambda t: (
                1 if t == previous_type else 0,
                generated_type_counts.get(t, 0),
                1 if t == "3d_orbit" else 0,
            ),
        )
        return ordered

    for beat in script.get("beats", []):
        bid = beat["beat_id"]
        btype = beat["broll_type"]
        if script.get("location_focus") and not beat.get("location_focus"):
            beat["location_focus"] = script.get("location_focus")
        if script.get("geodata_query") and not beat.get("geodata_query"):
            beat["geodata_query"] = script.get("geodata_query")
        selected_type = btype
        dur = beat.get("duration_sec", 5)
        label = f"beat{bid:02d}_{btype}"
        is_video_type = btype in (
            "satellite_pan", "map_wipe", "outline_reveal", "dark_cutout",
            "real_city", "real_people", "real_geography", "native_animal",
            "3d_orbit", "3d_flyover", "3d_zoom", "3d_close_oblique", "comparison_map",
            "region_comparison", "multi_region_reveal",
            "zoom_to_region", "size_comparison_overlay", "chokepoint_flow",
            "stat_counter_clip",
        )
        ext = ".mp4" if is_video_type else ".jpg"
        out_path = clips_dir / f"{label}{ext}"
        existing = [p for p in clips_dir.glob(f"beat{bid:02d}_*")
                    if not p.name.startswith("_") and p.stat().st_size > 10240]
        if existing:
            assets[bid] = existing[0]
            sourcing_report.append({
                "beat_id": bid,
                "requested_type": btype,
                "selected_type": selected_type,
                "asset": str(existing[0]),
                "status": "cached",
                "sourcing_debug": beat.get("_sourcing_debug", {}),
            })
            size_kb = existing[0].stat().st_size / 1024
            print(f"  Beat {bid}: {btype} ({dur}s)... [CACHED] {size_kb:.0f}KB")
            continue
        print(f"  Beat {bid}: {btype} ({dur}s)...", end="", flush=True)
        geo["_beat"] = beat
        geo["_region"] = region
        geo["_beat_highlight"] = beat.get("highlight")
        geo["_beat_annotations"] = beat.get("annotations", [])
        geo["_beat_render_mode"] = str(beat.get("render_mode", default_render_mode) or default_render_mode).strip().lower()
        search_intent = beat.get("search_intent") if isinstance(beat.get("search_intent"), dict) else {}
        fallback_allowed = str(search_intent.get("fallback_allowed") or "terrain_map").strip().lower()
        geo_strictness = str(search_intent.get("geography_strictness") or "").strip().lower()
        ok = False
        is_real = btype in ("real_city", "real_people", "real_geography", "native_animal")
        if btype == "comparison_map" and _HAS_3D_RENDERER and _gen_comparison_map:
            comp_data = beat.get("comparison", {})
            if comp_data and comp_data.get("base") and comp_data.get("highlight"):
                result = _gen_comparison_map(beat_id=bid, comparison=comp_data,
                                             run_dir=str(run_dir), duration=dur, geo=geo)
                if result and Path(result).exists():
                    out_path = Path(result)
                    ok = True
            if not ok:
                ok = _gen_3d(geo, out_path, dur, "flyover")
        elif is_real and allow_real_footage:
            subject = script.get("subject", "")
            ok = gen_real_youtube_clip(region, beat, out_path, dur, subject, geo=geo, used_video_ids=used_video_ids)
        if is_real and not ok:
            allow_cross_retry = (
                allow_real_footage
                and geo_strictness != "strict"
                and btype not in ("real_city", "real_people")
            )
            if not ok:
                fallbacks = {
                    "real_geography": ["terrain_map", "satellite_pan", "cinematic_orbit", "zoom_to_region", "3d_flyover", "3d_close_oblique", "3d_orbit"],
                    "real_city":      ["cinematic_orbit", "zoom_to_region", "3d_close_oblique", "3d_flyover", "3d_orbit"],
                    "real_people":    ["3d_close_oblique", "3d_zoom", "satellite_pan", "3d_orbit"],
                    "native_animal":  ["real_geography", "zoom_to_region", "3d_flyover", "3d_close_oblique", "3d_orbit"],
                }
                if fallback_allowed == "3d_orbit":
                    fallbacks[btype] = ["3d_orbit"]
                elif fallback_allowed == "terrain_map":
                    fallbacks[btype] = ["terrain_map", "zoom_to_region", "3d_orbit", "3d_flyover"]
                elif fallback_allowed in BROLL_GENERATORS:
                    fallbacks[btype] = [fallback_allowed]
                prev_type = str(script.get("beats", [])[bid - 2].get("broll_type", "")) if bid > 1 and len(script.get("beats", [])) >= bid - 1 else ""
                fb_candidates = _ordered_fallbacks(fallbacks.get(btype, ["3d_flyover", "outline_reveal", "3d_orbit", "satellite_pan"]), prev_type)
                for fb_type in fb_candidates:
                    fb_gen = BROLL_GENERATORS.get(fb_type)
                    if fb_gen:
                        fb_ext = ".mp4" if fb_type in (
                            "satellite_pan", "map_wipe", "outline_reveal", "dark_cutout",
                            "3d_orbit", "3d_flyover", "3d_zoom", "3d_curvature", "3d_close_oblique",
                            "region_comparison", "multi_region_reveal",
                            "zoom_to_region", "size_comparison_overlay", "chokepoint_flow",
                            "stat_counter_clip") else ".jpg"
                        fb_path = clips_dir / f"beat{bid:02d}_{fb_type}{fb_ext}"
                        ok = fb_gen(geo, fb_path, dur)
                        if ok and fb_path.exists():
                            out_path = fb_path
                            selected_type = fb_type
                            if isinstance(beat.get("_sourcing_debug"), dict):
                                beat["_sourcing_debug"]["accepted_source_class"] = "generated_fallback"
                                beat["_sourcing_debug"]["fallback_reason"] = beat["_sourcing_debug"].get("fallback_reason") or "real_footage_not_found"
                            print(f" [fallback->{fb_type}]", end="", flush=True)
                            break
        elif not is_real:
            gen = BROLL_GENERATORS.get(btype)
            if gen:
                ok = gen(geo, out_path, dur)
            if not ok:
                for fb_type in non_real_fallbacks.get(btype, []):
                    fb_gen = BROLL_GENERATORS.get(fb_type)
                    if not fb_gen:
                        continue
                    fb_ext = ".mp4" if fb_type in (
                        "satellite_pan", "map_wipe", "outline_reveal", "dark_cutout",
                        "3d_orbit", "3d_flyover", "3d_zoom", "3d_close_oblique", "comparison_map",
                        "region_comparison", "multi_region_reveal",
                        "zoom_to_region", "size_comparison_overlay", "chokepoint_flow",
                        "stat_counter_clip") else ".jpg"
                    fb_path = clips_dir / f"beat{bid:02d}_{fb_type}{fb_ext}"
                    ok = fb_gen(geo, fb_path, dur)
                    if ok and fb_path.exists():
                        out_path = fb_path
                        if isinstance(beat.get("_sourcing_debug"), dict):
                            beat["_sourcing_debug"]["accepted_source_class"] = "generated_fallback"
                            beat["_sourcing_debug"]["fallback_reason"] = beat["_sourcing_debug"].get("fallback_reason") or "generator_fallback"
                        print(f" [fallback->{fb_type}]", end="", flush=True)
                        break
        if ok and out_path.exists() and beat.get("highlight"):
            if out_path.suffix.lower() == ".mp4":
                _apply_highlight_to_video(out_path, beat, geo)
            else:
                _apply_highlight_to_still(out_path, beat, geo)
        # Apply animated overlay(s) — supports both old `overlay` (dict) and new `overlays` (list)
        _ovl_raw = beat.get("overlays") or ([beat["overlay"]] if beat.get("overlay") else [])
        overlays_list = [o for o in _ovl_raw if isinstance(o, dict) and o.get("type")]
        for oi, overlay_def in enumerate(overlays_list):
            if ok and out_path.exists():
                ovl_out = clips_dir / f"beat{bid:02d}_{btype}_ovl{oi}.mp4"
                print(f" [overlay:{overlay_def.get('type')} @{overlay_def.get('at_sec', 0.8)}s]", end="", flush=True)
                apply_overlay_to_clip(
                    primary_clip = out_path,
                    overlay_def  = overlay_def,
                    out_path     = ovl_out,
                    duration_sec = float(dur),
                    geo          = geo,
                    cache_dir    = clips_dir / "_overlay_cache",
                )
                if ovl_out.exists() and ovl_out.stat().st_size > 10240:
                    out_path = ovl_out
        if ok:
            assets[bid] = out_path
            generated_type_counts[selected_type] = generated_type_counts.get(selected_type, 0) + 1
            if isinstance(beat.get("_sourcing_debug"), dict) and not beat["_sourcing_debug"].get("accepted_source_class"):
                beat["_sourcing_debug"]["accepted_source_class"] = "generated_fallback" if selected_type != btype or not is_real else "local_real"
            sourcing_report.append({
                "beat_id": bid,
                "requested_type": btype,
                "selected_type": selected_type,
                "asset": str(out_path),
                "status": "ok",
                "sourcing_debug": beat.get("_sourcing_debug", {}),
            })
            size_kb = out_path.stat().st_size / 1024 if out_path.exists() else 0
            print(f" [OK] {size_kb:.0f}KB")
        else:
            if isinstance(beat.get("_sourcing_debug"), dict) and not beat["_sourcing_debug"].get("fallback_reason"):
                beat["_sourcing_debug"]["fallback_reason"] = "no_generator_or_asset"
            sourcing_report.append({
                "beat_id": bid,
                "requested_type": btype,
                "selected_type": selected_type,
                "asset": None,
                "status": "failed",
                "sourcing_debug": beat.get("_sourcing_debug", {}),
            })
            print(f" [SKIP] no generator for {btype}")
    print(f"  Total assets: {len(assets)}/{len(script.get('beats', []))}")
    (run_dir / "asset_sourcing_report.json").write_text(
        json.dumps(sourcing_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return assets


def still_to_video(still_path: Path, video_path: Path, duration: float, seed: int = None) -> bool:
    """Convert a still image to a video with a randomized Ken Burns motion."""
    rng = random.Random(seed)
    mode = rng.choice(["zoom_center", "zoom_topleft", "zoom_bottomright", "pan"])
    rate = rng.uniform(0.03, 0.08)
    pan_dir = rng.choice([1, -1])

    total_n = int(duration * FPS)
    if mode == "pan":
        pw, ph = int(OUT_W * 1.10), OUT_H
    else:
        overscale = 1.0 + rate + 0.02
        pw, ph = int(OUT_W * overscale), int(OUT_H * overscale)

    prescale = video_path.parent / f"_pre_still_{still_path.stem}.jpg"
    if not run_ffmpeg([
        "ffmpeg", "-y", "-i", str(still_path),
        "-vf", f"scale={pw}:{ph}:force_original_aspect_ratio=increase,crop={pw}:{ph}",
        str(prescale)], timeout=30):
        return run_ffmpeg([
            "ffmpeg", "-y", "-loop", "1", "-framerate", str(FPS), "-i", str(still_path),
            "-vf", f"scale={OUT_W}:{OUT_H},format=yuv420p",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
            "-r", str(FPS), "-frames:v", str(total_n), str(video_path)], timeout=60)

    if mode == "zoom_center":
        vf = (
            f"crop=w='trunc({pw}*(1-n/{total_n}*{rate:.4f}))':"
            f"h='trunc({ph}*(1-n/{total_n}*{rate:.4f}))':"
            f"x='trunc(({pw}-trunc({pw}*(1-n/{total_n}*{rate:.4f})))/2)':"
            f"y='trunc(({ph}-trunc({ph}*(1-n/{total_n}*{rate:.4f})))/2)',"
            f"scale={OUT_W}:{OUT_H},format=yuv420p"
        )
    elif mode == "zoom_topleft":
        vf = (
            f"crop=w='trunc({pw}*(1-n/{total_n}*{rate:.4f}))':"
            f"h='trunc({ph}*(1-n/{total_n}*{rate:.4f}))':"
            f"x=0:y=0,"
            f"scale={OUT_W}:{OUT_H},format=yuv420p"
        )
    elif mode == "zoom_bottomright":
        vf = (
            f"crop=w='trunc({pw}*(1-n/{total_n}*{rate:.4f}))':"
            f"h='trunc({ph}*(1-n/{total_n}*{rate:.4f}))':"
            f"x='trunc({pw}*n/{total_n}*{rate:.4f})':"
            f"y='trunc({ph}*n/{total_n}*{rate:.4f})',"
            f"scale={OUT_W}:{OUT_H},format=yuv420p"
        )
    else:  # pan
        pan_px = pw - OUT_W
        x_expr = f"trunc({pan_px}*n/{total_n})" if pan_dir == 1 else f"trunc({pan_px}*(1-n/{total_n}))"
        vf = f"crop={OUT_W}:{OUT_H}:x='{x_expr}':y=0,format=yuv420p"

    ok = run_ffmpeg([
        "ffmpeg", "-y", "-loop", "1", "-framerate", str(FPS), "-i", str(prescale),
        "-vf", vf, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
        "-r", str(FPS), "-frames:v", str(total_n), str(video_path)], timeout=60)
    prescale.unlink(missing_ok=True)
    return ok


def _make_title_card(title: str, out_path: Path):
    """Generate a title card frame (dark bg, big text)."""
    img = Image.new("RGB", (OUT_W, OUT_H), (12, 12, 18))
    draw = ImageDraw.Draw(img)
    font = _font(64)
    words = title.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 20:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur: lines.append(cur)
    total_h = sum(draw.textbbox((0,0), l, font=font)[3] - draw.textbbox((0,0), l, font=font)[1] + 20 for l in lines)
    y = (OUT_H - total_h) // 2
    for line in lines:
        bbox = draw.textbbox((0,0), line, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        x = (OUT_W - tw) // 2
        draw.text((x+3, y+3), line, font=font, fill=(0,0,0))
        draw.text((x, y), line, font=font, fill=(255,255,255))
        y += th + 20
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), "JPEG", quality=95)


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
            print(f"  Beat {bid}: [SKIP] no asset"); continue
        clip_path = final_clips_dir / f"final_beat{bid:02d}.mp4"
        if asset_path.suffix.lower() == ".mp4":
            ok = run_ffmpeg([
                "ffmpeg", "-y", "-i", str(asset_path), "-t", str(dur),
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-r", str(FPS), "-an", str(clip_path)
            ], timeout=120)
        else:
            ok = still_to_video(asset_path, clip_path, dur, seed=bid)
        if ok and clip_path.exists():
            # Color grade each clip for unified look
            graded_path = final_clips_dir / f"graded_beat{bid:02d}.mp4"
            if grade_clip(clip_path, graded_path):
                clip_path.unlink(missing_ok=True)
                graded_path.rename(clip_path)

            ordered_clips.append(clip_path)
            print(f"  Beat {bid}: [OK] {clip_path.name}")
        else:
            print(f"  Beat {bid}: [FAIL] caption/encode failed")

    # Burn hook text onto the first clip
    if ordered_clips:
        beats = script.get("beats", [])
        hook_text = str(script.get("hook_question", "") or script.get("title", "") or "").strip()
        if hook_text and len(ordered_clips) >= 1:
            first_clip = ordered_clips[0]
            hooked_path = final_clips_dir / f"hooked_{first_clip.name}"
            if burn_hook_text(first_clip, hooked_path, hook_text):
                first_clip.unlink(missing_ok=True)
                hooked_path.rename(first_clip)

    if not ordered_clips:
        print("  [FAIL] No clips to assemble!"); return None

    final_out = run_dir / "final_short.mp4"
    print(f"\n[S5] Concatenating {len(ordered_clips)} clips with fade transitions...")
    FADE_DUR = 0.3
    if len(ordered_clips) == 1:
        shutil.copy2(str(ordered_clips[0]), str(final_out)); ok = True
    elif len(ordered_clips) == 2:
        ok = run_ffmpeg([
            "ffmpeg", "-y", "-i", str(ordered_clips[0]), "-i", str(ordered_clips[1]),
            "-filter_complex",
            f"[0:v][1:v]xfade=transition=fade:duration={FADE_DUR}:offset={{}}".format(
                max(0.1, float(script['beats'][0].get('duration_sec', 4)) - FADE_DUR)),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-r", str(FPS), "-an", str(final_out)], timeout=180)
    else:
        inputs = []
        for p in ordered_clips: inputs.extend(["-i", str(p)])
        clip_durations = []
        for cp in ordered_clips:
            try:
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(cp)],
                    capture_output=True, text=True, timeout=10)
                dur = float(probe.stdout.strip())
            except Exception: dur = 6.0
            clip_durations.append(dur)
        print(f"  Clip durations (probed): {[f'{d:.2f}s' for d in clip_durations]}")
        filter_parts = []
        for i in range(len(ordered_clips) - 1):
            offset = max(0.1, sum(clip_durations[:i+1]) - (i + 1) * FADE_DUR)
            in_a = f"[{i}:v]" if i == 0 else f"[v{i}]"
            in_b = f"[{i+1}:v]"
            out_label = "" if i == len(ordered_clips) - 2 else f"[v{i+1}]"
            filter_parts.append(f"{in_a}{in_b}xfade=transition=fade:duration={FADE_DUR}:offset={offset:.2f}{out_label}")
        ok = run_ffmpeg([
            "ffmpeg", "-y", *inputs,
            "-filter_complex", ";".join(filter_parts),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-r", str(FPS), "-an", str(final_out)], timeout=300)

        # Fallback: pairwise xfade if full chain OOMs
        if not (ok and final_out.exists()) and len(ordered_clips) > 3:
            print("  [WARN] Full xfade chain failed, falling back to pairwise concat...")
            tmp_dir = final_clips_dir / "_xfade_tmp"
            tmp_dir.mkdir(exist_ok=True)
            current = ordered_clips[0]
            cum_dur = clip_durations[0]
            for i in range(1, len(ordered_clips)):
                pair_out = tmp_dir / f"pair_{i:02d}.mp4"
                pair_offset = max(0.1, cum_dur - FADE_DUR)
                ok2 = run_ffmpeg([
                    "ffmpeg", "-y", "-i", str(current), "-i", str(ordered_clips[i]),
                    "-filter_complex",
                    f"[0:v][1:v]xfade=transition=fade:duration={FADE_DUR}:offset={pair_offset:.2f}",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                    "-r", str(FPS), "-an", str(pair_out)], timeout=120)
                if ok2 and pair_out.exists():
                    current = pair_out
                    cum_dur = cum_dur - FADE_DUR + clip_durations[i]
                else:
                    print(f"  [WARN] Pairwise xfade {i} failed, using simple concat")
                    # Last resort: concat demuxer (no crossfade)
                    concat_list = tmp_dir / "concat.txt"
                    concat_list.write_text(
                        "\n".join(f"file '{p}'" for p in ordered_clips), encoding="utf-8")
                    ok = run_ffmpeg([
                        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                        "-i", str(concat_list),
                        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                        "-r", str(FPS), "-an", str(final_out)], timeout=300)
                    break
            else:
                shutil.copy2(str(current), str(final_out))
                ok = True
            try:
                shutil.rmtree(str(tmp_dir), ignore_errors=True)
            except Exception:
                pass

    if not (ok and final_out.exists()):
        print("  [FAIL] Final assembly failed"); return None
    mb = final_out.stat().st_size / 1048576
    print(f"  [OK] {final_out.name} ({mb:.1f} MB, silent)")

    # Branding overlay (yellow border + Urban Vectors watermark)
    branded_out = run_dir / "final_short_branded.mp4"
    if apply_branding(final_out, branded_out):
        final_out = branded_out

    # Captions from Whisper
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
        music_files = list(MUSIC_DIR.glob("*.mp3"))
        music = random.choice(music_files) if music_files else None
        if music: print(f"  [OK] Selected random music: {music.name}")
        final_with_audio = mix_voiceover(final_out, vo_path, run_dir, music)
        if final_with_audio:
            return final_with_audio
        else:
            print("  Audio mix failed, returning silent video")
    return final_out

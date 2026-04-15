"""
runner.py — CLI/orchestration layer for the modular geo short pipeline.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set

from .assembly import assemble_final, generate_assets
from .audio import generate_voiceover, run_whisper_alignment, _update_beat_durations_from_whisper
from .config import OUT_W, OUT_H, RUNS_DIR, FINAL_EXPORT_DIR
from .gemini import generate_script, generate_topic_script, normalize_script_plan
from .geodata import (
    fetch_boundary, gather_geo_data, rings_to_pixels,
    script_requires_boundary,
)


ArgsLike = Optional[Sequence[str]]


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")[:40]


def parse_beat_selection(value: str, available_ids: Sequence[int]) -> Set[int]:
    available = set(int(v) for v in available_ids)
    if not value.strip():
        return set()

    selected: Set[int] = set()
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                start, end = end, start
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))

    invalid = sorted(selected - available)
    if invalid:
        raise ValueError(f"Invalid beat selection: {invalid}. Available beats: {sorted(available)}")
    return selected


def filter_script_beats(script: Dict[str, Any], selected_ids: Set[int]) -> Dict[str, Any]:
    if not selected_ids:
        return script
    filtered = copy.deepcopy(script)
    filtered["beats"] = [
        beat for beat in filtered.get("beats", [])
        if int(beat.get("beat_id", -1)) in selected_ids
    ]
    filtered["total_duration_sec"] = round(
        sum(float(beat.get("duration_sec", 0)) for beat in filtered.get("beats", [])),
        1,
    )
    return filtered


def _safe_write_json(path: Path, payload: Any) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if path.exists():
        try:
            if path.read_text(encoding="utf-8") == text:
                return
        except Exception:
            pass
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
    except OSError:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        print(f"  [WARN] Could not update cached file: {path.name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Geography Short Maker")
    parser.add_argument("--region", default="", help="Region name, e.g. 'Nova Scotia, Canada'")
    parser.add_argument(
        "--prompt",
        default="",
        help="Topic-driven prompt, e.g. 'This is the widest river in the world'",
    )
    parser.add_argument("--resume", default="", help="Resume from existing run directory")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "s1", "s2", "s3", "s4"],
        help="Run specific stage or all",
    )
    parser.add_argument(
        "--disable-real-footage",
        action="store_true",
        help="Disable YouTube real footage sourcing and use only generated map/satellite visuals",
    )
    parser.add_argument(
        "--voice",
        default="",
        help="Path to a custom .mp3 or .wav file to clone for the voiceover",
    )
    parser.add_argument(
        "--render-mode",
        default="auto",
        choices=["auto", "day", "night"],
        help="Default style for generated map and 3D geography shots",
    )
    parser.add_argument(
        "--beats",
        default="",
        help="Optional beat subset to render, e.g. '1,3-5'. Best used with --stage s3",
    )
    parser.add_argument(
        "--visual-style",
        default="mixed",
        choices=["mixed", "digital_twin"],
        help="Choose between the default mixed pipeline or a digital-twin-focused 3D visual plan",
    )
    parser.add_argument(
        "--require-online",
        action="store_true",
        help="Fail if Gemini or online voice generation fall back or are unavailable",
    )
    return parser


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.region and not args.prompt and not args.resume:
        raise ValueError("Either --region or --prompt is required")

    prompt_mode = bool(args.prompt)
    prompt_text = args.prompt.strip() if args.prompt else ""
    region = args.region.strip() if args.region else ""

    run_label = prompt_text or region
    run_id = slugify(run_label)
    run_dir = RUNS_DIR / run_id if not args.resume else RUNS_DIR / args.resume
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("  GEO SHORT MAKER")
    if prompt_mode:
        print(f"  Prompt: {prompt_text}")
    else:
        print(f"  Region: {region}")
    print(f"  Run:    {run_dir.name}")
    print(f"{'=' * 60}")

    pipeline_t0 = time.perf_counter()

    def _log_stage_timing(stage_label: str, stage_t0: float) -> None:
        stage_elapsed = time.perf_counter() - stage_t0
        cumulative_elapsed = time.perf_counter() - pipeline_t0
        print(f"  [TIMING] {stage_label}: {stage_elapsed:.2f}s | cumulative: {cumulative_elapsed:.2f}s")

    s1_path = run_dir / "s1_script.json"
    stage_t0 = time.perf_counter()
    if (args.stage in ("all", "s1") and not args.resume) or not s1_path.exists():
        if prompt_mode:
            script = generate_topic_script(prompt_text, run_dir)
            if not region:
                region = script.get("region", "")
                print(f"  Extracted region: {region}")
        else:
            script = generate_script(region, run_dir)
        script_source = str(script.get("_source") or "")
        if args.require_online:
            if prompt_mode and script_source != "gemini_topic":
                raise RuntimeError(
                    f"Online Stage 1 required Gemini topic output, got script source '{script_source or 'unknown'}'"
                )
            if not prompt_mode and script_source != "gemini":
                raise RuntimeError(
                    f"Online Stage 1 required Gemini regional output, got script source '{script_source or 'unknown'}'"
                )
    else:
        script = json.loads(s1_path.read_text(encoding="utf-8"))
        print(f"\n[S1] Loaded existing script ({len(script.get('beats', []))} beats)")
        if not region:
            region = script.get("region", "")
    _log_stage_timing("S1 script", stage_t0)

    stage_t0 = time.perf_counter()
    script = normalize_script_plan(script, visual_style=args.visual_style)

    available_beat_ids = [int(beat.get("beat_id", 0)) for beat in script.get("beats", [])]
    selected_beat_ids = parse_beat_selection(args.beats, available_beat_ids) if args.beats else set()
    active_script = filter_script_beats(script, selected_beat_ids)
    if selected_beat_ids:
        print(f"  Selected beats: {sorted(selected_beat_ids)}")
    _log_stage_timing("S1 normalize/select", stage_t0)

    if args.stage == "s1":
        summary = {
            "region": region,
            "prompt": prompt_text if prompt_mode else None,
            "run_dir": str(run_dir),
            "script_beats": len(active_script.get("beats", [])),
            "selected_beats": sorted(selected_beat_ids) if selected_beat_ids else None,
            "assets_generated": 0,
            "final_video": None,
            "exported_final_video": None,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n{'=' * 60}")
        print("  COMPLETE (S1 only)")
        print(f"  Beats:  {summary['script_beats']}")
        print(f"  Dir:    {run_dir}")
        print(f"{'=' * 60}")
        total_elapsed = time.perf_counter() - pipeline_t0
        print(f"  [TIMING] TOTAL pipeline: {total_elapsed:.2f}s")
        return summary

    if not region:
        raise RuntimeError("No region available for geodata stage")

    s2_path = run_dir / "s2_geodata.json"
    stage_t0 = time.perf_counter()
    if (args.stage in ("all", "s2") and not args.resume) or not s2_path.exists():
        geo = gather_geo_data(region, run_dir, script)
    else:
        geo = json.loads(s2_path.read_text(encoding="utf-8"))
        print("\n[S2] Loaded existing geodata")
        if "rings" not in geo and args.stage in ("all", "s2", "s3") and script_requires_boundary(script):
            print("  Re-fetching boundary for pixel rings...")
            _, rings, _ = fetch_boundary(region)
            if rings:
                geo["rings"] = rings
                geo["pixel_rings"] = rings_to_pixels(
                    rings,
                    geo["lat"],
                    geo["lon"],
                    geo["zoom"],
                    geo["cols"],
                    geo["rows"],
                )
                geo["pixel_rings"] = [
                    ring for ring in geo["pixel_rings"]
                    if any(0 <= x <= OUT_W and 0 <= y <= OUT_H for x, y in ring)
                ]
    _log_stage_timing("S2 geodata", stage_t0)

    vo_path = None
    whisper_segs = None
    if args.stage in ("all", "s4"):
        stage_t0 = time.perf_counter()
        print("\n[S2b] Generating voiceover (audio-first pipeline)...")
        voice_path_ext = Path(args.voice) if args.voice else None
        vo_path = generate_voiceover(active_script, run_dir, voice_path=voice_path_ext)
        if args.require_online and not vo_path:
            raise RuntimeError("Online Stage 2b required voiceover generation, but no voiceover.mp3 was produced")
        _log_stage_timing("S2b voiceover", stage_t0)
        if vo_path:
            stage_t0 = time.perf_counter()
            print("\n[S2c] Whisper alignment...")
            whisper_segs = run_whisper_alignment(vo_path)
            _log_stage_timing("S2c whisper", stage_t0)
            if whisper_segs:
                stage_t0 = time.perf_counter()
                _safe_write_json(run_dir / "whisper_segments.json", whisper_segs)
                print("\n[S2d] Updating beat durations from spoken audio...")
                active_script = _update_beat_durations_from_whisper(active_script, whisper_segs)
                if selected_beat_ids:
                    beat_updates = {
                        int(beat.get("beat_id", -1)): beat
                        for beat in active_script.get("beats", [])
                    }
                    for beat in script.get("beats", []):
                        updated = beat_updates.get(int(beat.get("beat_id", -1)))
                        if updated:
                            beat.update(updated)
                else:
                    script = active_script
                _safe_write_json(run_dir / "s1_script.json", script)
                _log_stage_timing("S2d duration update", stage_t0)
                if script.get("_overlong_audio"):
                    print("  [WARN] Script audio is over the default 45-55s target; tighten Stage 1 prompt/script.")
        else:
            print("  [WARN] Voiceover generation failed, using Gemini-guessed durations")

    stage_t0 = time.perf_counter()
    if args.stage in ("all", "s3"):
        assets = generate_assets(
            active_script,
            geo,
            run_dir,
            region=region,
            allow_real_footage=not args.disable_real_footage,
            default_render_mode=args.render_mode,
        )
    else:
        clips_dir = run_dir / "clips"
        assets = {}
        if clips_dir.exists():
            for beat in active_script.get("beats", []):
                bid = beat["beat_id"]
                for ext in (".mp4", ".jpg"):
                    clip_path = clips_dir / f"beat{bid:02d}_{beat['broll_type']}{ext}"
                    if clip_path.exists():
                        assets[bid] = clip_path
        print(f"\n[S3] Found {len(assets)} existing assets")
    _log_stage_timing("S3 assets", stage_t0)

    stage_t0 = time.perf_counter()
    if args.stage in ("all", "s4"):
        final = assemble_final(active_script, assets, run_dir, vo_path=vo_path, whisper_segs=whisper_segs)
    else:
        final = None
    _log_stage_timing("S4 assemble", stage_t0)

    exported_final: Optional[Path] = None
    if final:
        try:
            FINAL_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
            src_final = Path(final)
            base_name = f"{run_dir.name}_{src_final.name}"
            target = FINAL_EXPORT_DIR / base_name
            suffix = 2
            while target.exists():
                target = FINAL_EXPORT_DIR / f"{run_dir.name}_{src_final.stem}_{suffix}{src_final.suffix}"
                suffix += 1
            shutil.copy2(str(src_final), str(target))
            exported_final = target
            print(f"  -> Exported final video: {target}")
        except Exception as exc:
            print(f"  [WARN] Failed to export final video to central folder: {exc}")

    summary = {
        "region": region,
        "prompt": prompt_text if prompt_mode else None,
        "selected_hook": script.get("_selected_hook"),
        "run_dir": str(run_dir),
        "script_beats": len(active_script.get("beats", [])),
        "selected_beats": sorted(selected_beat_ids) if selected_beat_ids else None,
        "assets_generated": len(assets),
        "final_video": str(final) if final else None,
        "exported_final_video": str(exported_final) if exported_final else None,
        "overlong_audio": bool(script.get("_overlong_audio")),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print("  COMPLETE")
    print(f"  Beats:  {summary['script_beats']}")
    print(f"  Assets: {summary['assets_generated']}")
    if final:
        print(f"  Final:  {Path(final).name}")
    print(f"  Dir:    {run_dir}")
    print(f"{'=' * 60}")

    metadata = script.get("youtube_metadata", {})
    if metadata:
        stage_t0 = time.perf_counter()
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
        _log_stage_timing("Metadata export", stage_t0)

    total_elapsed = time.perf_counter() - pipeline_t0
    print(f"  [TIMING] TOTAL pipeline: {total_elapsed:.2f}s")

    return summary


def geoshortmaker(argv: ArgsLike = None) -> Dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.region and not args.prompt and not args.resume:
        parser.error("Either --region or --prompt is required")
    return run_pipeline(args)


def main(argv: ArgsLike = None) -> Dict[str, Any]:
    return geoshortmaker(argv)

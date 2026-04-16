
# GeoShortMaker

Automated pipeline that generates geography YouTube Shorts from a text prompt. Give it a topic like *"Nobody wants to live in this state"* and it produces a 45-60 second vertical video with narration, captions, b-roll footage, map visuals, and background music.



https://github.com/user-attachments/assets/e6d60657-fa19-4c36-9acf-a03148fb985f




## Pipeline Stages

1. **S1 — Script**: Gemini generates a beat-by-beat script with narration, visual types, and overlay instructions
2. **S2 — Geodata** : Fetches boundaries from Nominatim, satellite tiles from ESRI/Mapbox
3. **S2b-d — Voice**: Qwen voice cloning TTS generates narration, Whisper aligns word timestamps
4. **S3 — Assets**: Sources real YouTube footage (yt-dlp + Gemini Vision geo-matching) or generates 3D map visuals
5. **S4 — Assembly**: Burns captions, applies overlays (wiki photos, stat counters), color grading
6. **S5 — Concat**: Joins clips with crossfade transitions
7. **S6 — Captions**: Word-by-word animated captions from Whisper alignment
8. **S7 — Audio mix**: Layers voice + background music onto the final video

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in your API keys in .env
```

### System dependencies

- **FFmpeg** — must be on PATH
- **yt-dlp** — must be on PATH (or set `YTDLP_PATH` in `.env`)

## Usage

```bash
python geoshortmaker.py --prompt "India and Bangladesh have the most confusing border on Earth"
```

Custom voice:
```bash
python geoshortmaker.py --prompt "The poorest reservation in America" --voice voices/guy_michaels.mp3
```

Output lands in `runs/<prompt_slug>/` and a final export in `output/final_videos/`.

## Voices

The pipeline clones any voice from a short audio sample using Qwen TTS. Two voices are included:

| File | Style | Best for |
|------|-------|----------|
| `voices/david_attenborough.mp3` | Calm, authoritative nature documentary | Geography, nature, slow-paced topics (default) |
| `voices/guy_michaels.mp3` | Fast, energetic YouTube narrator | Rankings, lists, punchy viral topics |

### Using the included voices

With no `--voice` flag, the pipeline uses the Attenborough voice by default:
```bash
python geoshortmaker.py --prompt "Why is this river so dangerous"
```

To use a different included voice:
```bash
python geoshortmaker.py --prompt "Top 5 most dangerous cities" --voice voices/guy_michaels.mp3
```

### Adding your own voice

1. Get a clean audio clip of the voice you want (10-60 seconds, no background music/noise)
2. Save it as `.mp3` or `.wav` in the `voices/` folder
3. Pass it with `--voice`:

```bash
python geoshortmaker.py --prompt "Your topic" --voice voices/my_custom_voice.mp3
```

The first run with a new voice takes a few extra seconds — Qwen enrolls the voice and caches the voice ID for future runs. After that it reuses the cached clone automatically.

**Tips for good voice cloning:**
- Use a sample with clear speech, no music or background noise
- 15-30 seconds of audio is the sweet spot
- The sample should match the energy you want (calm sample = calm narration)
- Supported formats: `.mp3`, `.wav`, `.m4a`
- Max file size: 10MB

## API Keys & Costs

| Service | Env Variable | What it does | Free tier | Cost after free |
|---------|-------------|-------------|-----------|-----------------|
| **Gemini** | `GEMINI_API_KEY` | Script generation + footage geo-matching | 15 RPM / 1M tokens/day (Flash) | ~$0.01-0.05 per run |
| **DashScope (Qwen TTS)** | `DASHSCOPE_API_KEY` | Voice cloning + narration synthesis | Limited trial credits | ~$0.01-0.03 per run |
| **Mapbox** | `MAPBOX_TOKEN` | 3D satellite tiles for map visuals | 200k tile requests/month | $0.60 per 1k after |
| **YouTube (yt-dlp)** | None (no key needed) | Sources real footage clips | Free | Free |
| **Nominatim (OSM)** | None (no key needed) | Boundary/geocoding lookups | Free (1 req/sec) | Free |
| **ESRI ArcGIS** | None (no key needed) | Satellite imagery tiles | Free | Free |

**Typical cost per video: $0.02-0.10** depending on cache hits and footage sourcing attempts.

Optional keys in `.env.example` (Pexels, Pixabay, Cesium, Google Maps) are not currently used by the pipeline.

## Project Structure

```
geoshortmaker.py          # CLI entrypoint
pipeline/
  runner.py               # Orchestrates S1-S7
  gemini.py               # S1: Script generation + prompt templates
  geodata.py              # S2: Boundary fetch, tile download
  audio.py                # S2b-d: TTS + Whisper alignment
  footage_stock.py        # S3: YouTube sourcing + scoring
  footage.py              # S3: Clip validation + caption detection
  broll.py                # S3: Map visual generators (outline, orbit, flyover, etc.)
  broll_earth.py          # S3: Satellite-based visuals
  broll_overlays.py       # S4: Stat counters, wiki photos, overlays
  assembly.py             # S4-S7: FFmpeg encode, concat, audio mix
  captions.py             # S6: Word-by-word caption rendering
  branding.py             # Border + watermark overlay
  hook_card.py            # Hook text card for first 2 seconds
  color_grade.py          # Cinematic color grading filter
  config.py               # Env loading, constants, paths
generate_attenborough_audio.py  # Qwen voice cloning module
voices/
  david_attenborough.mp3      # Default narrator voice sample
  guy_michaels.mp3            # Energetic YouTube narrator voice sample
```

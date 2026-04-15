"""
config.py — Shared constants, env loading, and SSL setup for the geo short pipeline.
"""
from __future__ import annotations

import os
import ssl
import shutil
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent  # geography/


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
FINAL_EXPORT_DIR = Path(
    os.environ.get("GEO_FINAL_EXPORT_DIR", "").strip() or str(ROOT / "output" / "final_videos")
).resolve()
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
MUSIC_DIR = Path(os.environ.get("GEO_MUSIC_DIR", "").strip() or str(ROOT / "music")).resolve()
VOICES_DIR = ROOT.parent / "voices"
MUSIC_VOL_DB = -18  # duck music well below voice
FONTS_DIR = Path(os.environ.get("GEO_FONTS_DIR", str(ROOT.parent / "fonts"))).resolve()
CAPTION_FONT_NAME = os.environ.get("CAPTION_FONT_NAME", "Montserrat Bold").strip() or "Montserrat Bold"

# Real footage sourcing (YouTube)
_YTDLP_ENV = os.environ.get("YTDLP_PATH", "").strip()
_YTDLP_VENV_CANDIDATES = [
    ROOT.parent / "venv" / "Scripts" / "yt-dlp.exe",
    ROOT.parent / ".venv" / "Scripts" / "yt-dlp.exe",
]
def _ytdlp_path_usable(path_value: str) -> bool:
    if not path_value:
        return False
    if "/" in path_value or "\\" in path_value:
        return Path(path_value).exists()
    return shutil.which(path_value) is not None

if _YTDLP_ENV and _ytdlp_path_usable(_YTDLP_ENV):
    YTDLP_PATH = _YTDLP_ENV
else:
    _venv_ytdlp = next((p for p in _YTDLP_VENV_CANDIDATES if p.exists()), None)
    YTDLP_PATH = str(_venv_ytdlp) if _venv_ytdlp else (_YTDLP_ENV or "yt-dlp")

def _validate_cookies(path: str) -> str:
    """Return path only if file exists, is non-empty, and starts with a Netscape cookie header."""
    if not path:
        return ""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return ""
    try:
        first_line = p.read_bytes()[:64].decode("utf-8", errors="ignore").strip()
        if "Netscape" not in first_line and "HTTP Cookie" not in first_line:
            print(f"[WARN] cookies file at {path!r} is not Netscape format — skipping cookies")
            return ""
    except Exception:
        return ""
    return path

_COOKIES_ENV = os.environ.get("YOUTUBE_COOKIES_PATH", "").strip()
if _COOKIES_ENV:
    COOKIES_YT = _validate_cookies(_COOKIES_ENV)
else:
    _default_cookies = ROOT.parent / "youtube_cookies.txt"
    COOKIES_YT = _validate_cookies(str(_default_cookies))
REAL_MIN_SRC_SEC = 12
REAL_MAX_SRC_SEC = 240

# Blur-bar background filter for fitting non-9:16 footage into 9:16 frame.
BLUR_BG_FILTER = (
    f"split[bg_in][fg_in];"
    f"[bg_in]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
    f"crop={OUT_W}:{OUT_H},boxblur=25:5[bg];"
    f"[fg_in]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2:color=black@0[fg];"
    f"[bg][fg]overlay=0:0,format=yuv420p"
)

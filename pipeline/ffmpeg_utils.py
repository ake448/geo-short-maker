"""
ffmpeg_utils.py — Shared FFmpeg runner used by multiple modules.
"""
from __future__ import annotations

import subprocess


def run_ffmpeg(cmd, timeout=300):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            err = r.stderr.strip().split('\n')
            useful = [l for l in err if not l.startswith(('  configuration:', '  lib', '  built with'))]
            print(f"    [FFmpeg ERR] {'  '.join(useful[-5:])}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("    [FFmpeg TIMEOUT]")
        return False

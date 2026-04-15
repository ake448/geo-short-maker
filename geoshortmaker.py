#!/usr/bin/env python3
"""
geoshortmaker.py — preferred CLI entrypoint for the modular geography pipeline.
"""

from __future__ import annotations

from pipeline.runner import geoshortmaker, main

__all__ = ["geoshortmaker", "main"]


if __name__ == "__main__":
    geoshortmaker()

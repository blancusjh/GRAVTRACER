"""Shared helpers for the example scripts: import path and output dir.

All examples write their figures/data to ``<repo>/output/`` so that code
and generated artifacts stay separated (output/ is git-ignored).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if any((ROOT / "python/grayt").glob("_core*.so")) or any(
    (ROOT / "python/grayt").glob("_core*.pyd")
):
    sys.path.insert(0, str(ROOT / "python"))

OUTPUT = ROOT/"output"
OUTPUT.mkdir(exist_ok=True)

ASSETS = Path(__file__).resolve().parent/"assets"


def out(name: str) -> str:
    return str(OUTPUT/name)

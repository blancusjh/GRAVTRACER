"""Counterpart of Fig. 8 of arXiv:2202.00086: wall-clock time vs image
resolution for the three integrators (shadow scene, a = 0.98, r0 = 1000,
image plane [-8, 8]^2 — the same setup as their timing figure).

Usage: python validation/fig8_benchmark.py [--sizes 64 128 256 512]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import grayt

METHODS = ["rkdp45", "rkck45", "rkf45"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[64, 128, 256, 512])
    ap.add_argument("-o", "--output", default="validation/fig8_benchmark.png")
    args = ap.parse_args()

    bh = grayt.BlackHole(a=0.98)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for method in METHODS:
        times = []
        for n in args.sizes:
            cam = grayt.Camera(r=1000.0, theta=90.0, x=(-8, 8), y=(-8, 8),
                               resolution=(n, n))
            t0 = time.time()
            grayt.shadow(bh, cam, method=method, rtol=1e-10, atol=1e-12)
            times.append(time.time() - t0)
            print(f"{method} {n}x{n}: {times[-1]:.2f}s")
        ax.loglog([n*n for n in args.sizes], times, "o-", label=method)
    ax.set_xlabel("$N_x \\times N_y$")
    ax.set_ylabel("time [s]")
    ax.legend()
    ax.set_title("Render time vs resolution (OpenMP, all cores)")
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

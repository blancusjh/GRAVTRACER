"""Reproduce Fig. 6 of arXiv:2202.00086: numerical shadow of a Kerr black
hole (a = 0.98, equatorial observer at r0 = 1000) against the analytic
Bardeen curve, eqs. (13)-(14).

Usage: python validation/fig6_shadow.py [--res N]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import grayt
from grayt.plotting import plot_shadow


def bardeen_shadow(a, theta0):
    """Analytic shadow rim from spherical photon orbits (eqs. 13-14)."""
    r = np.linspace(1.0, 4.5, 40001)
    xi = -(r**3 - 3*r**2 + a*a*r + a*a)/(a*(r - 1.0))
    eta = -(r**3*(r**3 - 6*r**2 + 9*r - 4*a*a))/(a*a*(r - 1.0)**2)
    s, c = np.sin(theta0), np.cos(theta0)
    y2 = eta - c*c*(xi*xi/(s*s) - a*a)
    ok = (eta >= 0) & (y2 >= 0)
    # -xi/sin(theta) in Bardeen's convention already coincides with the
    # screen x used by grayt (flat shadow edge on the Doppler-bright side).
    x = -xi[ok]/s
    y = np.sqrt(y2[ok])
    order = np.argsort(np.arctan2(y, x - x.mean()))
    return (np.concatenate([x[order], x[order][::-1]]),
            np.concatenate([y[order], -y[order][::-1]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=625)
    ap.add_argument("-a", "--spin", type=float, default=0.98)
    ap.add_argument("-o", "--output", default="validation/fig6_shadow.png")
    args = ap.parse_args()

    bh = grayt.BlackHole(a=args.spin)
    cam = grayt.Camera(r=1000.0, theta=90.0, x=(-8, 8), y=(-8, 8),
                       resolution=(args.res, args.res))
    t0 = time.time()
    img = grayt.shadow(bh, cam, rtol=1e-10, atol=1e-12)
    print(f"rendered {args.res}x{args.res} in {time.time()-t0:.1f}s")

    ax = plot_shadow(img, analytic_xy=bardeen_shadow(args.spin,
                                                     np.deg2rad(90.0)))
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title(f"Shadow, $a = {args.spin}$ — numerical vs Bardeen")
    ax.figure.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"wrote {args.output}")

    # Quantitative check: distance between numerical edge and analytic rim
    xs = np.linspace(*img.extent[:2], img.status.shape[0], endpoint=False)
    row = img.status[:, img.status.shape[1]//2]
    edge = xs[row == grayt.STATUS_CAPTURED]
    ax_x, _ = bardeen_shadow(args.spin, np.deg2rad(90.0))
    print(f"numerical edge x-range: [{edge.min():.3f}, {edge.max():.3f}]")
    print(f"analytic  rim  x-range: [{ax_x.min():.3f}, {ax_x.max():.3f}]")


if __name__ == "__main__":
    main()

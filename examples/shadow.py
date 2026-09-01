"""Black hole shadow versus the analytic Bardeen curve.

The numerical shadow (captured rays) is compared with the analytic rim
built from spherical photon orbits. With the default parameters this
reproduces Fig. 6 of arXiv:2202.00086 (a = 0.98, equatorial observer at
r0 = 1000).

Usage: python examples/shadow.py [-a SPIN] [--res N] [--theta DEG]
"""
import argparse
import time

import numpy as np

from _common import out

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import grayt
from grayt.plotting import plot_shadow


def bardeen_shadow(a, theta0):
    """Analytic shadow rim from spherical photon orbits (Bardeen 1973)."""
    r = np.linspace(1.0, 4.5, 40001)
    xi = -(r**3 - 3*r**2 + a*a*r + a*a)/(a*(r - 1.0))
    eta = -(r**3*(r**3 - 6*r**2 + 9*r - 4*a*a))/(a*a*(r - 1.0)**2)
    s, c = np.sin(theta0), np.cos(theta0)
    y2 = eta - c*c*(xi*xi/(s*s) - a*a)
    ok = (eta >= 0) & (y2 >= 0)
    x = -xi[ok]/s
    y = np.sqrt(y2[ok])
    order = np.argsort(np.arctan2(y, x - x.mean()))
    return (np.concatenate([x[order], x[order][::-1]]),
            np.concatenate([y[order], -y[order][::-1]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--spin", type=float, default=0.98)
    ap.add_argument("--res", type=int, default=625)
    ap.add_argument("--r0", type=float, default=1000.0)
    ap.add_argument("--theta", type=float, default=90.0)
    ap.add_argument("--fov", type=float, default=8.0)
    ap.add_argument("-o", "--output", default=out("shadow.png"))
    args = ap.parse_args()

    bh = grayt.BlackHole(a=args.spin)
    cam = grayt.Camera(r=args.r0, theta=args.theta,
                       x=(-args.fov, args.fov), y=(-args.fov, args.fov),
                       resolution=(args.res, args.res))
    t0 = time.time()
    img = grayt.shadow(bh, cam, rtol=1e-10, atol=1e-12)
    print(f"rendered {args.res}x{args.res} in {time.time()-t0:.1f}s")

    analytic = None
    if abs(args.spin) > 1e-8:
        analytic = bardeen_shadow(args.spin, np.deg2rad(args.theta))
    ax = plot_shadow(img, analytic_xy=analytic)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title(f"Shadow, $a = {args.spin}$")
    ax.figure.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

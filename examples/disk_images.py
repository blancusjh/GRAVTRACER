"""Thin accretion disk images for a set of spins.

Page-Thorne emission, I_obs = g^3 I_em, all panels sharing the first
panel's normalization. With the default parameters this reproduces
Fig. 13 of arXiv:2202.00086 (a = 0, 0.5, 0.95 at 2048x1024 use
``--res 2048 1024``).

Usage: python examples/disk_images.py [--spins A ...] [--l0 L ...]
                                      [--res NX NY]
"""
import argparse
import time
from pathlib import Path

from _common import out

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import grayt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spins", type=float, nargs="+", default=[0.0, 0.5, 0.95])
    ap.add_argument("--l0", type=float, nargs="+", default=[2.8, 2.8, 1.8],
                    help="specific angular momentum per spin")
    ap.add_argument("--res", type=int, nargs=2, default=(1024, 512),
                    metavar=("NX", "NY"))
    ap.add_argument("--theta", type=float, default=85.0)
    ap.add_argument("--r-out", type=float, default=20.0)
    ap.add_argument("-o", "--output", default=out("disk_images.png"))
    ap.add_argument("--npz-dir", default=None,
                    help="also save raw maps per spin")
    args = ap.parse_args()
    if len(args.l0) != len(args.spins):
        ap.error("--l0 needs one value per spin")

    cam = grayt.Camera(r=1000.0, theta=args.theta, x=(-24, 24), y=(-12, 12),
                       resolution=tuple(args.res))
    images = []
    for a, l0 in zip(args.spins, args.l0):
        bh = grayt.BlackHole(a=a)
        disk = grayt.ThinDisk(r_in=None, r_out=args.r_out, l0=l0)
        t0 = time.time()
        img = grayt.render(bh, cam, disk, rtol=1e-8, atol=1e-10)
        print(f"a={a}: rendered {args.res[0]}x{args.res[1]} in "
              f"{time.time()-t0:.1f}s, max I={img.intensity.max():.3e}, "
              f"max|H|={img.herr.max():.1e}")
        if args.npz_dir:
            Path(args.npz_dir).mkdir(parents=True, exist_ok=True)
            img.save(Path(args.npz_dir)/f"disk_a{str(a).replace('.','')}.npz")
        images.append(img)

    norm = images[0].intensity.max()
    n = len(images)
    fig, axes = plt.subplots(n, 1, figsize=(9, 4*n), constrained_layout=True,
                             squeeze=False)
    for ax, a, img in zip(axes[:, 0], args.spins, images):
        img.plot(ax=ax, norm_to=norm, label=f"$a = {a:g}$")
        ax.set_ylabel("$y\\;[M]$")
    axes[-1, 0].set_xlabel("$x\\;[M]$")
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"wrote {args.output}")
    for a, img in zip(args.spins, images):
        print(f"a={a}: max intensity relative to first panel = "
              f"{img.intensity.max()/norm:.2f}")


if __name__ == "__main__":
    main()

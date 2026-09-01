"""Photograph a lambertian image through a Kerr spacetime.

An image is placed on a plane behind the black hole and photographed
with a camera on the other side: backward ray tracing samples the
texture wherever each camera ray lands (direction-independent emission,
the physically natural model for "images carry color, not direction").
The result shows the direct (distorted) view, the Einstein-ring
secondary images, and the shadow.

Usage: python examples/photograph.py [--image PATH] [--spin A]
                                     [--res NX NY]
"""
import argparse
import time

import numpy as np

from _common import out, ASSETS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import grayt

DEFAULT_IMAGE = ASSETS/"labore_et_constantia.jpg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(DEFAULT_IMAGE))
    ap.add_argument("--spin", type=float, default=0.9)
    ap.add_argument("--res", type=int, nargs=2, default=(900, 560),
                    metavar=("NX", "NY"))
    ap.add_argument("--distance", type=float, default=150.0,
                    help="source plane distance behind the hole [M]")
    ap.add_argument("--card-width", type=float, default=90.0)
    ap.add_argument("-o", "--output", default=out("photograph.png"))
    args = ap.parse_args()

    bh = grayt.BlackHole(a=args.spin)
    source = grayt.ImageSource(center=(-args.distance, 0.0, 0.0),
                               normal=(1.0, 0.0, 0.0),
                               up=(0.0, 0.0, 1.0),
                               width=args.card_width,
                               height=args.card_width, image=args.image)
    # match the card aspect ratio to the loaded image
    h, w = source.image.shape[:2]
    source.height = args.card_width*h/w
    ps = grayt.PhysicalSystem(spacetime=bh, sources=[source])
    sys3 = grayt.System(physical=ps, rtol=1e-8, atol=1e-10)

    # camera on the +x axis; FOV wide enough for card + Einstein ring
    cam = grayt.Camera(r=1000.0, theta=90.0, phi=0.0,
                       x=(-45.0, 45.0), y=(-28.0, 28.0),
                       resolution=tuple(args.res))
    t0 = time.time()
    photo = sys3.photograph(cam, 0, background=0.0)
    counts = {int(k): int(v)
              for k, v in zip(*np.unique(photo.status, return_counts=True))}
    print(f"photographed {args.res[0]}x{args.res[1]} in "
          f"{time.time()-t0:.1f}s, ray statuses: {counts}")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 5.6), constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1.6]})
    ax1.imshow(source.image)
    ax1.set_title("source image (lambertian card behind the hole)")
    ax1.axis("off")
    photo.plot(ax=ax2, label=f"$a = {args.spin}$")
    ax2.set_title("photograph through the Kerr spacetime")
    ax2.set_xlabel("$x\\;[M]$"); ax2.set_ylabel("$y\\;[M]$")
    fig.savefig(args.output, dpi=170, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

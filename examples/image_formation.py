"""Forward image projection through a Kerr spacetime (collimated mode).

A source image emits one ray per pixel, perpendicular to its plane (a
"projector"); rays are traced through the spacetime and accumulate on a
Screen. Central rays are captured, near-critical rays cross the optical
axis and form the caustic star. For photographing a lambertian source
with a camera, see examples/photograph.py.

Usage: python examples/image_formation.py [--image PATH] [--spin A]
"""
import argparse
import time

import numpy as np

from _common import out

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import grayt


def test_pattern(n=512):
    """Colored quadrants + grid + circle: enough structure to see the
    lensing distortion clearly."""
    img = np.zeros((n, n, 3))
    img[:n//2, :n//2] = (0.85, 0.15, 0.15)
    img[:n//2, n//2:] = (0.15, 0.6, 0.2)
    img[n//2:, :n//2] = (0.15, 0.3, 0.8)
    img[n//2:, n//2:] = (0.95, 0.85, 0.2)
    ii, jj = np.mgrid[0:n, 0:n]
    grid = (ii % (n//8) < 2) | (jj % (n//8) < 2)
    img[grid] = 1.0
    ring = np.abs(np.hypot(ii - n/2, jj - n/2) - n/3) < 2
    img[ring] = 0.0
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None,
                    help="path to an image to propagate (default: pattern)")
    ap.add_argument("--spin", type=float, default=0.9)
    ap.add_argument("--rays", type=int, default=300_000)
    ap.add_argument("-o", "--output", default=out("image_formation.png"))
    args = ap.parse_args()

    img_arr = args.image if args.image else test_pattern()

    bh = grayt.BlackHole(a=args.spin)
    source = grayt.ImageSource(center=(-60.0, 0.0, 0.0),
                               normal=(1.0, 0.0, 0.0),
                               up=(0.0, 0.0, 1.0),
                               width=40.0, height=40.0, image=img_arr,
                               emission="collimated")
    screen = grayt.Screen(center=(60.0, 0.0, 0.0),
                          normal=(1.0, 0.0, 0.0),
                          up=(0.0, 0.0, 1.0),
                          width=40.0, height=40.0, resolution=(256, 256))

    ps = grayt.PhysicalSystem(spacetime=bh, sources=[source])
    sys3 = grayt.System(physical=ps, screens=[screen],
                        rtol=1e-9, atol=1e-11)

    t0 = time.time()
    stats = sys3.form_image(0, 0, max_rays=args.rays, keep_sample_rays=40)
    print(f"propagated {stats['n_rays']} rays in {time.time()-t0:.1f}s: "
          f"{stats['status_counts']} -> {stats['n_on_screen']} on screen")

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(source.image)
    ax1.set_title("source image"); ax1.axis("off")
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(screen.image)
    ax2.set_title(f"formed image on screen ($a={args.spin}$)")
    ax2.axis("off")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    sys3.visualize3d(ax=ax3, max_rays=40, elev=16, azim=-72)
    lim = 75.0
    ax3.set_xlim(-lim, lim); ax3.set_ylim(-lim, lim); ax3.set_zlim(-lim, lim)
    ax3.set_title("scene (sample rays)")
    fig.savefig(args.output, dpi=170, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

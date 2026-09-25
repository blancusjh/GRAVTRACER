"""How a camera forms an image of an object behind a Kerr black hole.

A lambertian card (the bundled engraving) hangs behind the hole; a camera
on the far side traces one null geodesic per pixel backward in time until
it meets the card, falls through the horizon, or escapes. The figure shows
the scene in 3D with a sample of those rays, colored by what they reach,
next to the object and the image it forms on the camera.

Chief rays through three pixels are highlighted and numbered, and their
landing points are marked on the object and in the image. Rays 1 and 2 land
close to each other on the card but reach the camera from opposite sides of
the hole: they are a primary and a secondary image of the same point.

Usage: python examples/image_formation_scene.py [--spin A] [--res NX NY]
                                                  [-o PATH]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from _common import ASSETS, out

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import grayt
from grayt import style

CARD_X = -36.0          # the object plane, x = CARD_X
CAMERA_R = 50.0         # camera on the +x axis
FIELD = ((-19.0, 19.0), (-14.0, 14.0))   # image-plane extents [M]
BOX = ((-40.0, 54.0), (-34.0, 34.0), (-24.0, 24.0))   # drawn region [M]
SEPIA = np.array([1.0, 0.92, 0.80])      # warm paper tint for the card
PRIMARY = (-13.0, 2.0)  # pixel of chief ray 1; ray 2 is found to match it
THIRD = (15.0, 10.0)    # an unremarkable direct ray, for contrast
CHIEF_COLORS = ("#7fc8c0", "#b7a3e0", "#c9d77a")


def build(spin):
    bh = grayt.BlackHole(a=spin)
    card = grayt.ImageSource(center=(CARD_X, 0.0, 0.0), normal=(1.0, 0.0, 0.0),
                             up=(0.0, 0.0, 1.0), width=56.0, height=42.0,
                             image=str(ASSETS / "labore_et_constantia.jpg"))
    h, w = card.image.shape[:2]
    card.height = card.width * h / w
    card.image = np.clip(card.image * SEPIA, 0, 1)
    system = grayt.System(physical=grayt.PhysicalSystem(spacetime=bh,
                                                        sources=[card]),
                          rtol=1e-8, atol=1e-10)
    return bh, card, system


def clip_to_box(pts):
    """Keep the leading part of a path that stays inside BOX."""
    inside = np.all([(pts[:, i] >= lo) & (pts[:, i] <= hi)
                     for i, (lo, hi) in enumerate(BOX)], axis=0)
    out = np.nonzero(~inside)[0]
    return pts if not out.size else pts[:max(out[0], 2)]


def fade_out(pts, reach=26.0):
    """Stop a missing ray soon after its closest approach to the hole."""
    r = np.linalg.norm(pts, axis=1)
    k = int(np.argmin(r))
    far = np.nonzero(r[k:] > reach)[0]
    return clip_to_box(pts if not far.size else pts[:k + far[0] + 1])


def secondary_pixel(bh, camera, card, target):
    """Pixel on the far side of the hole whose ray lands nearest ``target``
    on the card: the secondary image of the primary pixel's object point."""
    x1, y1 = PRIMARY
    best = None
    for radius in np.linspace(6.5, 10.5, 17):
        for angle in np.linspace(-0.5, 0.5, 21):
            phi = np.arctan2(-y1, -x1) + angle
            x, y = radius * np.cos(phi), radius * np.sin(phi)
            _, kind, uv = trace_to_card(bh, camera, card, x, y)
            if kind == "card":
                d = np.hypot(uv[0] - target[0], uv[1] - target[1])
                if best is None or d < best[0]:
                    best = (d, x, y)
    return best[1], best[2]


def trace_to_card(bh, camera, card, x, y):
    """Backward geodesic from pixel (x, y), clipped at the object plane.

    Returns (points, kind, uv) with kind 'card', 'captured' or 'escaped'.
    """
    initial, pt, pphi = grayt.camera_ray(bh, camera, x, y)
    ray = grayt.trace(bh, initial, pt, pphi, h0=-0.1, lambda_max=400,
                      max_step=0.4, escape_radius=140, n_max=20_000)
    observer = grayt.bl_to_cart(camera.r, np.deg2rad(camera.theta),
                                np.deg2rad(camera.phi))
    pts = np.vstack((observer, ray.points))
    below = np.nonzero(pts[:, 0] <= CARD_X)[0]
    if below.size:
        k = below[0]
        a, b = pts[k - 1], pts[k]
        hit = a + (CARD_X - a[0]) / (b[0] - a[0]) * (b - a)
        pts = np.vstack((pts[:k], hit))
        u, v = card.to_local(hit)
        if abs(u[0]) <= card.width / 2 and abs(v[0]) <= card.height / 2:
            return pts, "card", (u[0], v[0])
        return pts, "escaped", None
    if ray.r[-1] < bh.capture_radius + 0.2:
        return pts, "captured", None
    return pts, "escaped", None


def polar_grid(ax, radii=(10, 20, 30), spokes=24, zorder=0.5):
    """Equatorial reference: rings of constant r and radial spokes."""
    phi = np.linspace(0, 2 * np.pi, 361)
    for r in radii:
        ax.plot(r * np.cos(phi), r * np.sin(phi), 0 * phi, color=style.AMBER,
                lw=0.5, alpha=0.3, zorder=zorder)
    for p in np.linspace(0, 2 * np.pi, spokes, endpoint=False):
        rr = np.array([4.0, radii[-1]])
        ax.plot(rr * np.cos(p), rr * np.sin(p), 0 * rr, color=style.AMBER,
                lw=0.35, alpha=0.18, zorder=zorder)


def horizon(ax, bh, zorder=6):
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 49), np.linspace(0, np.pi, 25))
    rh = bh.capture_radius
    ax.plot_surface(rh * np.cos(u) * np.sin(v), rh * np.sin(u) * np.sin(v),
                    rh * np.cos(v), color="#000000", edgecolor="#4a4640",
                    linewidth=0.15, shade=False, zorder=zorder)


def camera_glyph(ax, photo, zorder=9):
    """Frustum from the eye to a small image plane showing the photograph.

    Image x maps to world -y (library convention), image y to +z.
    """
    eye = np.array([CAMERA_R, 0.0, 0.0])
    depth, scale = 9.0, 9.0 / CAMERA_R
    (x0, x1), (y0, y1) = FIELD

    def plane(x, y):
        return eye + np.array([-depth, -x * scale, y * scale])

    corners = np.array([plane(x0, y0), plane(x1, y0), plane(x1, y1),
                        plane(x0, y1)])
    for c in corners:
        ax.plot(*np.vstack((eye, c)).T, color=style.INK, lw=0.6, alpha=0.8,
                zorder=zorder)
    # the photo as seen through the camera: rows top-down in +z, columns
    # left-to-right in +y (image x reversed)
    rgb = np.clip(photo.rgb, 0, 1).transpose(1, 0, 2)[::-1, ::-1]
    card = grayt.PlanarSurface(center=tuple(plane(0.5 * (x0 + x1),
                                                  0.5 * (y0 + y1))),
                               normal=(1, 0, 0), up=(0, 0, 1),
                               width=(x1 - x0) * scale,
                               height=(y1 - y0) * scale)
    style.textured_plane(ax, card, rgb, cells=80, zorder=zorder + 0.2)
    ax.scatter(*eye, s=10, color=style.INK, depthshade=False,
               zorder=zorder + 0.3)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spin", type=float, default=0.9)
    ap.add_argument("--res", type=int, nargs=2, default=(760, 560),
                    metavar=("NX", "NY"))
    ap.add_argument("-o", "--output", default=out("image_formation_scene.png"))
    args = ap.parse_args()

    bh, card, system = build(args.spin)
    camera = grayt.Camera(r=CAMERA_R, theta=90.0, phi=0.0, x=FIELD[0],
                          y=FIELD[1], resolution=tuple(args.res))
    t0 = time.time()
    photo = system.photograph(camera, 0, background=0.0)
    print(f"photograph {args.res[0]}x{args.res[1]} in {time.time() - t0:.1f}s")

    fan = [(x, y) for y in (-8.4, 0.0, 8.4)
           for x in np.linspace(-17, 17, 15)]
    fan += [(x, y) for x in (-2.8, 2.8) for y in np.linspace(-12, 12, 9)]
    rays = [trace_to_card(bh, camera, card, x, y) for x, y in fan]
    rays = [(clip_to_box(p) if k == "card" else fade_out(p), k, uv)
            for p, k, uv in rays]
    first = trace_to_card(bh, camera, card, *PRIMARY)
    pixels = [PRIMARY, secondary_pixel(bh, camera, card, first[2]), THIRD]
    chief = [trace_to_card(bh, camera, card, x, y) + (c,)
             for (x, y), c in zip(pixels, CHIEF_COLORS)]
    print("chief pixels:", [tuple(np.round(p, 2)) for p in pixels],
          "->", [tuple(np.round(c[2], 1)) for c in chief])

    with style.context():
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_axes((-0.04, -0.05, 0.78, 1.04), projection="3d",
                          computed_zorder=False)
        style.dark_3d(ax)
        polar_grid(ax)
        style.textured_plane(ax, card, card.image, cells=320, zorder=1)
        palette = {"card": style.AMBER, "captured": style.EMBER,
                   "escaped": "#6f8f8c"}
        for pts, kind, _ in rays:
            c = palette[kind]
            ax.plot(*pts.T, color=c, lw=2.4, alpha=0.05, zorder=3)
            ax.plot(*pts.T, color=c, lw=0.7 if kind != "escaped" else 0.5,
                    alpha=0.75 if kind != "escaped" else 0.3, zorder=3.1)
        horizon(ax, bh)
        for n, (pts, kind, uv, c) in enumerate(chief, start=1):
            pts = clip_to_box(pts) if kind != "card" else pts
            ax.plot(*pts.T, color=c, lw=3.5, alpha=0.18, zorder=7)
            ax.plot(*pts.T, color=c, lw=1.3, zorder=7.1)
            ax.scatter(*pts[-1], s=16, color=c, depthshade=False, zorder=7.2)
            ax.text(*(pts[-1] + (0, -1.5, 1.8)), str(n), color=c,
                    fontsize=12, ha="center", zorder=8)
        camera_glyph(ax, photo)
        ax.text(CAMERA_R, 0, -7, "camera", color=style.MUTED,
                fontsize=10, style="italic", ha="center", zorder=10)
        ax.text(CARD_X, 0, card.height / 2 + 4, "object", color=style.MUTED,
                fontsize=10, style="italic", ha="center", zorder=10)
        ax.plot([0, 0], [0, 0], [-2.2, -12], color=style.MUTED, lw=0.5,
                zorder=10)
        ax.text(0, 0, -14.5, f"Kerr hole, $a = {args.spin}$",
                color=style.MUTED, fontsize=10, style="italic", ha="center",
                zorder=10)
        ax.set(xlim=BOX[0], ylim=BOX[1], zlim=BOX[2])
        ax.set_box_aspect([hi - lo for lo, hi in BOX], zoom=1.3)
        ax.set_proj_type("persp", focal_length=0.5)
        ax.view_init(elev=24, azim=-62)

        # object and image panels
        (x0, x1), (y0, y1) = FIELD
        axo = fig.add_axes((0.70, 0.53, 0.26, 0.33))
        axo.imshow(card.image, extent=(-card.width / 2, card.width / 2,
                                       -card.height / 2, card.height / 2))
        axo.set_title("Object", fontsize=13)
        axo.set_xlabel("$u$ / M", labelpad=2)
        axo.set_ylabel("$v$ / M", labelpad=2)
        axi = fig.add_axes((0.70, 0.11, 0.26, 0.33))
        photo.plot(ax=axi)
        axi.invert_xaxis()           # as seen through the camera
        axi.set_title("Image", fontsize=13)
        axi.set_xlabel("$x$ / M", labelpad=2)
        axi.set_ylabel("$y$ / M", labelpad=2)
        for axis in (axo, axi):
            style.frame(axis)
        for n, ((pts, kind, uv, c), (x, y)) in enumerate(zip(chief, pixels),
                                                          start=1):
            axi.add_patch(Rectangle((x - 0.8, y - 0.8), 1.6, 1.6, fill=False,
                                    ec=c, lw=1.3))
            axi.text(x - 1.4, y + 1.1, str(n), color=c, fontsize=11,
                     weight="bold")
            if uv is not None:
                axo.plot(uv[0], uv[1], "o", mfc="none", mec=c, ms=8,
                         mew=1.4)
                axo.text(uv[0] + (1.8 if n != 2 else -3.4), uv[1] + 1.6,
                         str(n), color=c, fontsize=11, weight="bold")

        fig.text(0.035, 0.93, "Image formation in a Kerr spacetime",
                 fontsize=22, color=style.INK)
        fig.text(0.035, 0.895,
                 "Backward null geodesics from each camera pixel to an "
                 "illuminated plane behind the hole",
                 fontsize=12.5, color=style.MUTED, style="italic")
        handles = [plt.Line2D([], [], color=c, lw=1.4, label=l) for c, l in (
            (style.AMBER, "reaches the object"),
            (style.EMBER, "captured by the hole"),
            ("#6f8f8c", "misses the object"))]
        fig.legend(handles=handles, loc="lower left", ncol=3,
                   bbox_to_anchor=(0.03, 0.035), fontsize=11)
        fig.text(0.035, 0.018,
                 "Pseudo-Cartesian embedding of Boyer–Lindquist coordinates, "
                 "units of M. Rings in the equatorial plane every 10 M. "
                 "Numbered chief rays mark the same pixels in all panels.",
                 fontsize=9.5, color=style.MUTED)
        fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

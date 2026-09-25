"""Regenerate the README figures from GRAVTRACER and its bundled NASA sky.

Run from the repository root: python examples/readme_figures.py
The disk render is computed at publication resolution and may take a minute.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import grayt
from grayt import style


ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "docs" / "images"
BG, INK, MUTED = style.BG, style.INK, style.MUTED
style.use()


def disk_figure():
    spacetime = grayt.BlackHole(a=0.95)
    camera = grayt.Camera(
        r=100, theta=76, phi=90, x=(-40, 40), y=(-25, 25), resolution=(2240, 1400)
    )
    disk = grayt.SlabDisk(grayt.PageThorneDisk(spacetime, r_out=60),
                          lambda r, phi: 2.0 * (6.0 / r) ** 2)
    image = grayt.render_scene(
        spacetime,
        camera,
        disk,
        sky=grayt.CelestialSky.nasa_starmap(gain=1.4),
        exposure=240,
        tone="log",
        escape_radius=200,
        rtol=2e-9,
        atol=2e-11,
    )
    fig = plt.figure(figsize=(12.8, 8.4))
    ax = fig.add_axes((0.07, 0.11, 0.88, 0.75))
    ax.imshow(image.rgb.transpose(1, 0, 2), origin="lower", extent=image.extent)
    ax.set(xlabel="image-plane $x$ / M", ylabel="image-plane $y$ / M")
    style.frame(ax)
    fig.text(0.07, 0.945, "Kerr black hole, $a = 0.95$", color=INK,
             fontsize=21, va="top")
    fig.text(0.07, 0.895, "Translucent Page–Thorne disk at 76° inclination, "
             "with the Milky Way lensed into an Einstein ring behind it",
             color=MUTED, fontsize=12.5, style="italic", va="top")
    fig.text(0.07, 0.03, "Flux-conserving gray thin slab, $\\tau_\\perp = 2\\,(6M/r)^2$, "
             "transfer at every disk crossing. Log display over 2.5 decades; "
             "colors are a display mapping, not spectra. "
             "Sky: NASA SVS Deep Star Maps 2020.", color=MUTED, fontsize=9.5)
    fig.savefig(IMAGES / "kerr_disk.png", dpi=185)
    plt.close(fig)


def rays_figure():
    spacetime = grayt.BlackHole(a=0.8)
    camera = grayt.Camera(r=23, theta=65, phi=25,
                          x=(-8, 8), y=(-5, 5), resolution=(100, 70))
    rays = []
    observer = grayt.bl_to_cart(camera.r, np.deg2rad(camera.theta),
                                np.deg2rad(camera.phi))
    for y in (-4, 0, 4):
        for x in (-8, -6, -4, -2, 0, 2, 4, 6, 8):
            initial, pt, pphi = grayt.camera_ray(spacetime, camera, x, y)
            trajectory = grayt.trace(
                spacetime, initial, pt, pphi, h0=-0.1,
                lambda_max=70, max_step=0.35, escape_radius=26, n_max=4000
            )
            points = np.vstack((observer, trajectory.points))
            captured = trajectory.r[-1] < spacetime.capture_radius + 0.1
            near = trajectory.r.min() < 4.0 and not captured
            rays.append((points, "captured" if captured else "near" if near else "escaped"))

    colors = {"captured": style.EMBER, "near": style.AMBER,
              "escaped": style.TEAL}
    fig = plt.figure(figsize=(14, 7.8))
    for panel, (elev, azim) in enumerate(((27, -56), (48, 30)), start=1):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        style.dark_3d(ax)
        u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 48), np.linspace(0, np.pi, 28))
        rh = spacetime.capture_radius
        ax.plot_surface(rh * np.cos(u) * np.sin(v), rh * np.sin(u) * np.sin(v),
                        rh * np.cos(v), color="#000000", edgecolor="#4a4640",
                        linewidth=0.17, shade=False, zorder=5)
        phi = np.linspace(0, 2 * np.pi, 241)
        for r in (6, 12, 18, 24) if panel == 1 else (4, 8, 12):
            ax.plot(r * np.cos(phi), r * np.sin(phi), 0 * phi,
                    color=style.AMBER, lw=0.45, alpha=0.22)
        for points, kind in rays:
            if panel == 2:
                points = points[np.linalg.norm(points, axis=1) <= 12]
                if len(points) < 2:
                    continue
            color = colors[kind]
            ax.plot(*points.T, color=color, lw=3.0, alpha=0.10)
            ax.plot(*points.T, color=color, lw=1.15, alpha=0.88)
        if panel == 1:
            ax.scatter(*observer, s=24, color=INK, depthshade=False)
            ax.text(*(observer + (0, 0, 2.5)), "observer", color=MUTED,
                    fontsize=10, style="italic", ha="center")
            ax.set(xlim=(-26, 26), ylim=(-26, 26), zlim=(-18, 18))
        else:
            ax.set(xlim=(-12, 12), ylim=(-12, 12), zlim=(-9, 9))
        ax.set_box_aspect((1, 1, 0.78), zoom=1.42)
        ax.view_init(elev=elev, azim=azim)
        ax.text2D(0.06, 0.9, "Oblique view" if panel == 1
                  else "Close pass, $r < 12$ M", color=INK,
                  transform=ax.transAxes, fontsize=13)
    fig.text(0.055, 0.93, "Null geodesics around a Kerr black hole, $a = 0.8$",
             color=INK, fontsize=21)
    fig.text(0.055, 0.885, "Rays traced backward from one observer through "
             "a 9 × 3 grid of image-plane pixels", color=MUTED, fontsize=12.5,
             style="italic")
    legend = [Line2D([0], [0], color=c, lw=2, label=label) for c, label in (
        (style.TEAL, "escaped"), (style.AMBER, "strongly bent, $r_{\\min} < 4$ M"),
        (style.EMBER, "captured"))]
    fig.legend(handles=legend, loc="lower left", ncol=3,
               bbox_to_anchor=(0.05, 0.06), fontsize=11)
    fig.text(0.055, 0.03, "Boyer–Lindquist coordinates shown in a "
             "pseudo-Cartesian embedding, units of M. Equatorial rings mark "
             "constant $r$.", color=MUTED, fontsize=9.5)
    fig.savefig(IMAGES / "ray_trajectories_3d.png", dpi=185)
    plt.close(fig)


def scene_figure():
    import sys

    import image_formation_scene

    argv, sys.argv = sys.argv, ["image_formation_scene.py", "-o",
                                str(IMAGES / "image_formation_scene.png")]
    try:
        image_formation_scene.main()
    finally:
        sys.argv = argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("disk", "rays", "scene"),
                        help="regenerate one figure")
    args = parser.parse_args()
    IMAGES.mkdir(parents=True, exist_ok=True)
    jobs = {"disk": disk_figure, "rays": rays_figure, "scene": scene_figure}
    for name, job in jobs.items():
        if args.only is None or name == args.only:
            print(f"Rendering {name}...", flush=True)
            job()
            print(f"Saved {name}.", flush=True)


if __name__ == "__main__":
    main()

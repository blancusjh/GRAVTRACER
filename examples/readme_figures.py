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


ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "docs" / "images"
BG = "#000000"
INK = "#e3edf7"
MUTED = "#a4b4c8"


def disk_figure():
    spacetime = grayt.BlackHole(a=0.95)
    camera = grayt.Camera(
        r=100, theta=70, x=(-26, 26), y=(-16, 16), resolution=(900, 550)
    )
    image = grayt.render_scene(
        spacetime,
        camera,
        grayt.PageThorneDisk(spacetime),
        sky=grayt.CelestialSky.nasa_starmap(),
        exposure=5000,
        escape_radius=200,
        rtol=2e-9,
        atol=2e-11,
    )
    fig = plt.figure(figsize=(12.8, 7.8), facecolor=BG)
    ax = fig.add_axes((0.06, 0.13, 0.88, 0.78), facecolor=BG)
    ax.imshow(image.rgb.transpose(1, 0, 2), origin="lower", extent=image.extent)
    ax.set(xlabel="image-plane x / M", ylabel="image-plane y / M")
    ax.tick_params(colors=MUTED, length=0, labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    fig.text(0.06, 0.955, "KERR BLACK HOLE  /  a = 0.95", color=INK, fontsize=17, weight="bold", va="top")
    fig.text(0.06, 0.04, "Page–Thorne disk · 70° inclination · NASA SVS 2020 catalog sky", color=MUTED, fontsize=10)
    fig.savefig(IMAGES / "kerr_disk.png", dpi=185, facecolor=BG)
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

    fig = plt.figure(figsize=(14, 7.8), facecolor=BG)
    for panel, (elev, azim) in enumerate(((27, -56), (48, 30)), start=1):
        ax = fig.add_subplot(1, 2, panel, projection="3d", facecolor=BG)
        u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 48), np.linspace(0, np.pi, 28))
        rh = spacetime.capture_radius
        ax.plot_surface(rh * np.cos(u) * np.sin(v), rh * np.sin(u) * np.sin(v),
                        rh * np.cos(v), color="#000000", edgecolor="#36536e",
                        linewidth=0.17, shade=False, zorder=5)
        for points, kind in rays:
            if panel == 2:
                points = points[np.linalg.norm(points, axis=1) <= 12]
                if len(points) < 2:
                    continue
            color = {"captured": "#f39383", "near": "#f4ce80", "escaped": "#78cee2"}[kind]
            ax.plot(*points.T, color=color, lw=3.0, alpha=0.10)
            ax.plot(*points.T, color=color, lw=1.15, alpha=0.88)
        if panel == 1:
            ax.scatter(*observer, s=24, color="#f1f5fa", depthshade=False)
            ax.set(xlim=(-26, 26), ylim=(-26, 26), zlim=(-18, 18))
        else:
            ax.set(xlim=(-12, 12), ylim=(-12, 12), zlim=(-9, 9))
        ax.set_box_aspect((1, 1, 0.78), zoom=1.42)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.text2D(0.04, 0.93, "OBLIQUE VIEW" if panel == 1 else "CLOSE PASS  /  r < 12 M",
                  color=MUTED, transform=ax.transAxes, fontsize=10, weight="bold")
    fig.suptitle("NULL GEODESICS  /  KERR a = 0.8", color=INK, fontsize=18,
                 weight="bold", x=0.055, y=0.97, ha="left")
    legend = [Line2D([0], [0], color=c, lw=2.5, label=label) for c, label in (
        ("#78cee2", "escaped"), ("#f4ce80", "strongly bent"),
        ("#f39383", "captured"))]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
               labelcolor=INK, bbox_to_anchor=(0.5, 0.065), fontsize=10)
    fig.text(0.055, 0.025, "Traced null rays · Boyer–Lindquist coordinates shown in a pseudo-Cartesian embedding",
             color=MUTED, fontsize=9)
    fig.savefig(IMAGES / "ray_trajectories_3d.png", dpi=185, facecolor=BG)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("disk", "rays"),
                        help="regenerate one figure")
    args = parser.parse_args()
    IMAGES.mkdir(parents=True, exist_ok=True)
    jobs = {"disk": disk_figure, "rays": rays_figure}
    for name, job in jobs.items():
        if args.only is None or name == args.only:
            print(f"Rendering {name}...", flush=True)
            job()
            print(f"Saved {name}.", flush=True)


if __name__ == "__main__":
    main()

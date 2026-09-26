"""Generate a reproducible scientific gallery and actual camera-orbit videos.

Run: python examples/model_gallery.py --output output/models
Use --quick for a small smoke gallery; --skip-videos for stills only.
All comparisons use M=1. The sky uses the bundled NASA SVS star map;
colors are display mappings.
"""

from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import grayt
from grayt import style
from grayt.animation import VideoWriter

style.use()


INCLINATIONS = (20, 60, 84)
# Color is bolometric intensity on one fixed log scale for every panel
# (like EHT images), spanning the whole opaque disk so none of it is shown
# black. True-color photometry is shown separately in docs/images.
R_C, TAU_C = 8.0, 1e3   # spreading-disk scale radius [M], tau_perp(r_c)
DECADES = 7.0           # log color scale
FLOOR = 0.12            # gas that hides stars is never drawn darker than this

def models():
    cases = []
    for spin in (0.0, 0.5, 0.95):
        angles = INCLINATIONS
        st = grayt.BlackHole(spin)
        # Page-Thorne inside, Lynden-Bell & Pringle taper outside: opaque
        # where optically thick, fading as it cools; no edge.
        disk = grayt.spreading_disk(st, r_c=R_C, tau_c=TAU_C, r_trunc=4.4 * R_C)
        for angle in angles:
            cases.append(
                (
                    f"kerr_a{spin:g}_i{angle}",
                    f"Kerr a={spin:g} | inclination {angle} deg",
                    st,
                    disk,
                    None,
                    angle,
                    "Kerr / spreading thin disk, r_c=8 M / bolometric g^4 / log intensity scale",
                )
            )
    star = grayt.SphericalStar(radius=5.0)
    uniform = grayt.EmittingSurface(
        lambda th, ph, t: np.full_like(th, 3e-4), name="uniform static surface"
    )

    def spots(th, ph, t):
        dot = np.cos(th) * np.cos(0.8) + np.sin(th) * np.sin(0.8) * np.cos(ph - 0.5)
        return (
            3e-5 + 8e-4 * np.exp(-(1 - dot) / 0.035) + 4e-4 * np.exp(-(1 + dot) / 0.035)
        )

    spotted = grayt.EmittingSurface(spots, name="two prescribed static hot spots")
    cases.extend(
        [
            (
                "star_uniform",
                "Spherical star | R=5 M",
                star,
                None,
                uniform,
                70,
                "Schwarzschild exterior / prescribed surface",
            ),
            (
                "star_spots",
                "Spherical star | two hot spots",
                star,
                None,
                spotted,
                50,
                "Prescribed static hot spots; no stellar atmosphere",
            ),
        ]
    )
    for charge in (0.5, 0.8):
        st = grayt.ReissnerNordstrom(charge)
        disk = grayt.spreading_disk(st, r_c=12.0, tau_c=TAU_C, r_trunc=53.0, base=grayt.EmittingDisk(
            6.0,
            180.0,
            lambda r, ph, t: 2e-4 * (6 / r) ** 3 * (1 - np.sqrt(6 / r)),
            name="illustrative power-law emission",
            provenance={"warning": "not a self-consistent charged accretion solution"},
        ))
        cases.append(
            (
                f"charged_q{charge:g}",
                f"Charged black hole | Q/M={charge:g}",
                st,
                disk,
                None,
                70,
                "Theoretical comparison / prescribed emission / circular geodesics",
            )
        )
    for q in (0.3, 0.7):
        cases.append(
            (
                f"quadrupole_q{q:g}",
                f"Zipoy-Voorhees | q={q:g}",
                grayt.QMetric(q),
                None,
                None,
                70,
                "Naked-singularity comparison / celestial lensing only; ADM mass=1+q",
            )
        )
    return cases


def save_panel(image, path, title, note):
    fig, ax = plt.subplots(figsize=(10, 6))
    image.plot(ax)
    style.frame(ax)
    ax.set_title(title.replace(" | ", ",  "), pad=12)
    fig.text(0.12, 0.025, note, color=style.MUTED, fontsize=9,
             style="italic")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plate(thumbnails, path):
    """Two labelled blocks: Kerr spin x inclination, then other spacetimes."""
    kerr = {slug: rgb for slug, _, rgb in thumbnails if slug.startswith("kerr_")}
    captions = {"star_uniform": "Spherical star, $R = 5$ M",
                "star_spots": "Spherical star, two hot spots",
                "charged_q0.5": "Reissner–Nordström, $Q/M = 0.5$",
                "charged_q0.8": "Reissner–Nordström, $Q/M = 0.8$",
                "quadrupole_q0.3": "Zipoy–Voorhees, $q = 0.3$",
                "quadrupole_q0.7": "Zipoy–Voorhees, $q = 0.7$"}
    other = [(captions.get(slug, t), rgb) for slug, t, rgb in thumbnails
             if not slug.startswith("kerr_")]
    spins, angles = (0, 0.5, 0.95), INCLINATIONS
    ny, nx = thumbnails[0][2].shape[1], thumbnails[0][2].shape[0]
    size = (20, 9.4)
    w, gap, left, top = 0.163, 0.007, 0.06, 0.8
    h = w * size[0] / size[1] * ny / nx          # keep the camera aspect
    fig = plt.figure(figsize=size)

    def tile(x, y, rgb, caption=None):
        ax = fig.add_axes((x, y, w, h))
        ax.imshow(rgb.transpose(1, 0, 2), origin="lower", aspect="auto",
                  interpolation="antialiased")
        ax.set_xticks([]); ax.set_yticks([])
        style.frame(ax)
        if caption:
            ax.text(0.03, 0.05, caption, transform=ax.transAxes,
                    color=style.INK, fontsize=11,
                    bbox=dict(facecolor="black", edgecolor="none", alpha=0.7,
                              boxstyle="square,pad=0.3"))
        return ax

    for i, spin in enumerate(spins):
        y = top - (i + 1) * h - i * gap
        fig.text(left - 0.012, y + h / 2, f"$a = {spin:g}$", color=style.INK,
                 fontsize=14, ha="right", va="center")
        for j, angle in enumerate(angles):
            rgb = kerr.get(f"kerr_a{spin:g}_i{angle}")
            if rgb is not None:
                tile(left + j * (w + gap), y, rgb)
    for j, angle in enumerate(angles):
        fig.text(left + j * (w + gap) + w / 2, top + 0.012,
                 f"$i$ = {angle}°", color=style.INK, fontsize=14,
                 ha="center")
    x0 = left + 3 * (w + gap) + 0.03
    fig.text(left, top + 0.065, "Kerr black holes with thin accretion disks",
             color=style.MUTED, fontsize=13, style="italic")
    fig.text(x0, top + 0.065, "Other stationary spacetimes",
             color=style.MUTED, fontsize=13, style="italic")
    for k, (caption, rgb) in enumerate(other[:6]):
        i, j = divmod(k, 2)
        tile(x0 + j * (w + gap), top - (i + 1) * h - i * gap, rgb, caption)
    fig.text(left, 0.925, "Stationary spacetimes and the light they bend",
             color=style.INK, fontsize=26)
    fig.text(left, 0.03,
             "Color: bolometric intensity on one log scale (7 decades) for all "
             "panels, not true color. Spreading thin disks (Page–Thorne inside), "
             "gray slab transfer. Stars and charged holes: prescribed emission. "
             "Sky: NASA SVS Deep Star Maps 2020, lensed. Observers at $r = 400$ M.",
             color=style.MUTED, fontsize=10.5)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def ray_movie(path, frames=96):
    """Rotate a 3D plotting camera around fixed Kerr geodesics and a flat disk.

    This illustrates coordinate trajectories; it is not an observer photograph.
    The accompanying observer videos are generated by re-tracing each camera.
    """
    st = grayt.BlackHole(0.8)
    cam = grayt.Camera(
        r=45, theta=70, phi=25, x=(-15, 15), y=(-10, 10), resolution=(320, 180)
    )
    traces = []
    screen_points = [
        (x, y) for x in np.linspace(-12, 12, 5) for y in np.linspace(-6, 6, 5)
    ]
    for x, y in screen_points:
        y0, pt, pp = grayt.camera_ray(st, cam, x, y)
        tr = grayt.trace(
            st, y0, pt, pp, h0=-0.1, lambda_max=100, max_step=0.4, escape_radius=48
        )
        traces.append(
            np.vstack(
                (
                    grayt.geometry.bl_to_cart(
                        cam.r, np.deg2rad(cam.theta), np.deg2rad(cam.phi)
                    ),
                    tr.points,
                )
            )
        )
    np.savez_compressed(
        path.with_suffix(".npz"), **{f"ray_{i}": v for i, v in enumerate(traces)}
    )
    fig = plt.figure(figsize=(9.6, 7.2), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    style.dark_3d(ax, axes_visible=True)
    angle = np.linspace(0, 2 * np.pi, 100)
    radius = np.linspace(st.isco, 20, 14)
    aa, rr = np.meshgrid(angle, radius)
    ax.plot_surface(
        rr * np.cos(aa),
        rr * np.sin(aa),
        rr * 0,
        color=style.AMBER,
        alpha=0.2,
        linewidth=0,
    )
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 24))
    rh = st.horizon
    ax.plot_surface(
        rh * np.cos(u) * np.sin(v),
        rh * np.sin(u) * np.sin(v),
        rh * np.cos(v),
        color="#000000",
        edgecolor="#4a4640",
        linewidth=0.15,
    )
    for i, pts in enumerate(traces):
        ax.plot(
            *pts.T, color=plt.cm.YlOrBr(0.25 + 0.6 * i / len(traces)), lw=1.1, alpha=0.9
        )
    observer = traces[0][0]
    ax.scatter(*observer, color=style.INK, s=30)
    ax.text(*observer, "  observer", color=style.MUTED, fontsize=10,
            style="italic")
    ax.set(
        xlim=(-50, 50),
        ylim=(-50, 50),
        zlim=(-50, 50),
        xlabel="x / M",
        ylabel="y / M",
        zlabel="z / M",
    )
    ax.set_box_aspect((1, 1, 1))
    ax.set_title("Null geodesics around a flat disk, Kerr $a = 0.8$", pad=18)
    fig.text(
        0.09,
        0.045,
        "Spherical coordinate embedding; distances are not proper distances.",
        color=style.MUTED,
        fontsize=10,
        style="italic",
    )
    with VideoWriter(path, (960, 720), 24) as writer:
        for k in range(frames):
            ax.view_init(
                elev=24 + 10 * np.sin(2 * np.pi * k / frames), azim=360 * k / frames
            )
            fig.canvas.draw()
            writer.write(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            if k == 0:
                fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "frames": frames,
                "fps": 24,
                "metric": {"type": "Kerr", "a": 0.8},
                "kind": "3D coordinate-trajectory visualization",
                "units": "G=c=M=1",
                "ray_count": len(traces),
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/models"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-videos", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sky = grayt.CelestialSky.nasa_starmap(gain=1.4)
    plt.imsave(args.output / "celestial_map.png", sky.image)
    # phi=90 deg puts the Milky Way's core behind the hole, so the lensed
    # star field shows the curvature; the frame contains the Einstein ring.
    # The observer sits at 400 M, well outside the opaque 60 M disk, so the
    # disk does not fill the view; extents are direction cosines x r.
    camera = grayt.Camera(
        r=400,
        theta=70,
        phi=90,
        x=(-160, 160),
        y=(-100, 100),
        # 2x the displayed tile resolution: the plate's resampling averages
        # four rays per displayed pixel (anti-aliasing near the photon ring).
        resolution=(192, 120) if args.quick else (1280, 800),
    )
    # One exposure for all panels, from the brightest Kerr case, so panels
    # compare brightness physically.
    ref = grayt.BlackHole(0.95)
    peak = grayt.render_scene(
        ref, replace(camera, theta=60, resolution=(320, 200)),
        grayt.spreading_disk(ref, r_c=R_C, tau_c=TAU_C, r_trunc=4.4 * R_C),
        escape_radius=800).intensity.max()
    exposure = 1.0 / (0.5 * peak)
    records = []
    thumbnails = []
    for slug, title, st, disk, surface, theta, note in models():
        started = time.monotonic()
        image = grayt.render_scene(
            st,
            replace(camera, theta=theta),
            disk,
            surface=surface,
            sky=sky,
            exposure=exposure,
            tone="log",
            decades=DECADES,
            opaque_floor=FLOOR,
            rtol=2e-9,
            atol=2e-11,
            escape_radius=800,
        )
        image.save(args.output / f"{slug}.npz")
        save_panel(image, args.output / f"{slug}.png", title, note)
        counts = np.bincount(image.status.ravel(), minlength=4).tolist()
        records.append(
            {
                "name": slug,
                "title": title,
                "note": note,
                "seconds": time.monotonic() - started,
                "status_counts": counts,
                "metadata": image.meta,
            }
        )
        thumbnails.append((slug, title, image.rgb.copy()))
        print(f"{slug}: {counts}; {records[-1]['seconds']:.1f}s", flush=True)
    plate(thumbnails, args.output / "gallery.png")
    (args.output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    if not args.skip_videos:
        frames = 12 if args.quick else 96
        cam = replace(camera, resolution=(192, 120) if args.quick else (480, 300))
        for name, st, disk, surface in [
            (
                "kerr_camera_orbit",
                grayt.BlackHole(0.8),
                grayt.spreading_disk(grayt.BlackHole(0.8), r_c=R_C, tau_c=TAU_C,
                                     r_trunc=4.4 * R_C),
                None,
            ),
            ("stellar_camera_orbit", grayt.SphericalStar(5), None, next(m[4] for m in models() if m[0] == "star_spots")),
            ("quadrupole_camera_orbit", grayt.QMetric(0.5), None, None),
        ]:
            print(f"Rendering {name} ({frames} frames)", flush=True)
            grayt.render_movie(
                args.output / f"{name}.mp4",
                st,
                grayt.orbit_cameras(cam, frames),
                disk=disk,
                surface=surface,
                sky=sky,
                exposure=exposure,
                tone="log",
                decades=DECADES,
                opaque_floor=FLOOR,
                archive_every=max(1, frames // 4),
                coordinate_time_step=1 / 24,
                rtol=1e-8,
                atol=1e-10,
                escape_radius=800,
                progress=lambda i, n: (
                    print(f"  {i}/{n}", flush=True) if i % 12 == 0 else None
                ),
            )
        print("Rendering 3D trajectory movie", flush=True)
        ray_movie(args.output / "kerr_geodesics_3d.mp4", frames)
    print(f"Artifacts: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()

"""Generate a reproducible scientific gallery and actual camera-orbit videos.

Run: PYTHONPATH=python python examples/model_gallery.py --output output/models
Use --quick for a small smoke gallery; --skip-videos for stills only.
All comparisons use M=1. The sky is synthetic; colors are display mappings.
"""

from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
import json
import html
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import grayt
from grayt.animation import VideoWriter


def build_index(output):
    """Local, portable artifact browser; all assets remain relative files."""
    output = Path(output)
    interactive_path = output / "interactive_views.html"
    interactive = (
        interactive_path.read_text(encoding="utf-8")
        if interactive_path.exists()
        else ""
    )
    observer_path = output / "observer_view.html"
    observer = (
        observer_path.read_text(encoding="utf-8") if observer_path.exists() else ""
    )
    native = (output / "desktop_launcher.json").exists()
    if native and observer:
        observer = """<p>Use <strong>Open desktop viewer</strong> on an example to open its native window.
Press <strong>B</strong> for black → celestial map → grid. The browser may ask to open GRAVTRACER Viewer.</p>
<details id="browser-preview"><summary>Optional browser preview (CPU)</summary>
<iframe title="CPU browser preview" style="width:100%;height:1300px;border:0"></iframe></details>
<script>document.getElementById('browser-preview').addEventListener('toggle',function(){
if(this.open&&!this.dataset.loaded){this.dataset.loaded='true';this.querySelector('iframe').src='observer_preview.html';}});</script>"""
    if observer and interactive:
        # Coordinate plots are supplementary. Load their WebGL runtime only
        # when expanded, leaving the observer responsive on page startup.
        coordinates = (
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            "<style>body{margin:0;background:#090d16;color:#dce4ef;font:16px/1.6 system-ui}"
            ".interactive-plot{margin:18px 0;border-radius:12px;overflow:hidden}</style>"
            + interactive
            + "</html>"
        )
        payload = json.dumps(coordinates).replace("</", "<\\/")
        interactive = (
            '<details id="coordinate-diagrams"><summary>Coordinate diagrams and 3D emissivity</summary>'
            '<div id="coordinate-host"></div></details>'
            '<script id="coordinate-payload" type="application/json">'
            + payload
            + "</script>"
            '<script>document.getElementById("coordinate-diagrams").addEventListener("toggle",function(){'
            'if(!this.open||this.dataset.loaded)return;this.dataset.loaded="true";'
            'const frame=document.createElement("iframe");frame.title="Coordinate diagrams";'
            'frame.style="width:100%;height:1500px;border:0";'
            'frame.srcdoc=JSON.parse(document.getElementById("coordinate-payload").textContent);'
            'frame.addEventListener("load",()=>{frame.style.height=frame.contentDocument.documentElement.scrollHeight+"px";});'
            'document.getElementById("coordinate-host").append(frame);});</script>'
        )
    records = json.loads((output / "manifest.json").read_text())
    cards = []
    for row in records:
        name, title = row["name"], html.escape(row["title"])
        cards.append(
            f'<article><h2>{title}</h2><a href="{name}.png">'
            f'<img loading="lazy" src="{name}.png" alt="{title}"></a>'
            f'<p>{html.escape(row["note"])}</p><a href="{name}.npz">Raw ray and radiation maps</a>'
            + (
                f'<p><a class="desktop-launch" href="gravtracer://view/{name}">Open desktop viewer'
                + (" (GPU)" if name.startswith(("kerr_", "quadrupole_")) else " (CPU)")
                + "</a></p>"
                if native
                else ""
            )
            + (
                f' · <a href="#observer-view" data-observer-name="{name}">Move observer</a>'
                if observer and not native
                else ""
            )
            + "</article>"
        )
    for movie in sorted(output.glob("*.mp4")):
        title = html.escape(movie.stem.replace("_", " "))
        cards.append(
            f'<article><h2>{title}</h2><video controls loop preload="metadata" src="{movie.name}"></video>'
            f'<p><a href="{movie.with_suffix(".json").name}">Parameters and camera sequence</a></p>'
            + (
                f'<p><a class="desktop-launch" href="gravtracer://view/{movie.stem}">Open desktop viewer</a></p>'
                if native
                and movie.stem
                in (
                    "kerr_camera_orbit",
                    "stellar_camera_orbit",
                    "quadrupole_camera_orbit",
                )
                else ""
            )
            + "</article>"
        )
    for name, title in [
        ("volume_torus", "Prescribed 3D gray radiation field"),
        ("metric_convergence", "Metric interpolation convergence"),
    ]:
        if (output / f"{name}.png").exists():
            cards.append(
                f'<article><h2>{title}</h2><img src="{name}.png" alt="{title}"></article>'
            )
    page = (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GRAVTRACER — stationary models</title>
<style>body{background:#090d16;color:#dce4ef;font:16px/1.6 system-ui;margin:32px auto;max-width:1400px;padding:0 24px}
h1{font-size:36px}h2{font-size:19px}a{color:#8fcaff}img,video{width:100%;height:auto;border-radius:8px}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,360px),1fr));gap:24px}article{background:#131b28;padding:20px;border-radius:12px}
p{color:#b9c5d6}.intro{max-width:960px;margin-bottom:32px}
.interactive-section{margin:32px 0 48px}.interactive-section>h2{font-size:28px}
.interactive-plot{background:#0d1420;border:1px solid #29384c;border-radius:12px;overflow:hidden;margin:18px 0}
#coordinate-diagrams{margin:24px 0 40px}#coordinate-diagrams>summary{cursor:pointer;font-size:20px}
.desktop-launch{display:inline-block;padding:8px 14px;background:#243b53;border:1px solid #557b9c;border-radius:7px;text-decoration:none}
#browser-preview{margin:24px 0}summary{cursor:pointer}
</style>
<h1>Stationary spacetimes and light</h1><div class="intro">
<p>Kerr Page–Thorne disks, spherical stellar surfaces, and theoretical charged/quadrupolar comparisons.
The sky is synthetic; false colors use fixed exposure. All propagation is computed in the specified metric.</p>
<p>Observer movies retrace moving camera positions. The 3D geodesic movie is a coordinate visualization.
The volume example imports prescribed gray radiation coefficients, not a GRMHD solution.</p>
<p><a href="gallery.png">Full comparison sheet</a> · <a href="manifest.json">Scientific manifest</a> ·
<a href="celestial_map.png">Celestial map</a> · <a href="metric_convergence.json">Convergence measurements</a></p>
</div>"""
        + observer
        + interactive
        + "<main>"
        + "\n".join(cards)
        + "</main></html>\n"
    )
    (output / "index.html").write_text(page)


def models():
    cases = []
    for spin, angles in [(0.0, (30, 70, 85)), (0.5, (30, 70)), (0.95, (30, 70, 85))]:
        st = grayt.BlackHole(spin)
        disk = grayt.PageThorneDisk(st)
        for angle in angles:
            cases.append(
                (
                    f"kerr_a{spin:g}_i{angle}",
                    f"Kerr a={spin:g} | inclination {angle} deg",
                    st,
                    disk,
                    None,
                    angle,
                    "Kerr / Page-Thorne / Keplerian / bolometric",
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
        disk = grayt.EmittingDisk(
            6.0,
            20.0,
            lambda r, ph, t: 2e-4 * (6 / r) ** 3 * (1 - np.sqrt(6 / r)),
            name="illustrative power-law emission",
            provenance={"warning": "not a self-consistent charged accretion solution"},
        )
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
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#090d16")
    image.plot(ax)
    ax.set_title(title, color="white", loc="left", pad=14)
    ax.set_facecolor("black")
    ax.tick_params(colors="#aeb9c7")
    ax.xaxis.label.set_color("#aeb9c7")
    ax.yaxis.label.set_color("#aeb9c7")
    fig.text(0.12, 0.025, note, color="#aeb9c7", fontsize=8)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
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
    fig = plt.figure(figsize=(9.6, 7.2), dpi=100, facecolor="#090d16")
    ax = fig.add_subplot(111, projection="3d", facecolor="#090d16")
    angle = np.linspace(0, 2 * np.pi, 100)
    radius = np.linspace(st.isco, 20, 14)
    aa, rr = np.meshgrid(angle, radius)
    ax.plot_surface(
        rr * np.cos(aa),
        rr * np.sin(aa),
        rr * 0,
        color="#c36b2c",
        alpha=0.28,
        linewidth=0,
    )
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 24))
    rh = st.horizon
    ax.plot_surface(
        rh * np.cos(u) * np.sin(v),
        rh * np.sin(u) * np.sin(v),
        rh * np.cos(v),
        color="#10141e",
    )
    for i, pts in enumerate(traces):
        ax.plot(
            *pts.T, color=plt.cm.plasma(0.2 + 0.7 * i / len(traces)), lw=1.25, alpha=0.9
        )
    observer = traces[0][0]
    ax.scatter(*observer, color="white", s=35)
    ax.text(*observer, "  observer", color="white", fontsize=9)
    ax.set(
        xlim=(-50, 50),
        ylim=(-50, 50),
        zlim=(-50, 50),
        xlabel="x / M",
        ylabel="y / M",
        zlabel="z / M",
    )
    ax.set_box_aspect((1, 1, 1))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color("#c0cadd")
        axis.set_pane_color((0.06, 0.08, 0.12, 1))
    ax.tick_params(colors="#c0cadd")
    ax.set_title(
        "Kerr a=0.8 | null geodesics around a flat disk", color="white", pad=18
    )
    fig.text(
        0.09,
        0.045,
        "Spherical coordinate embedding; distances are not proper distances.",
        color="#b0bdcf",
        fontsize=10,
    )
    with VideoWriter(path, (960, 720), 24) as writer:
        for k in range(frames):
            ax.view_init(
                elev=24 + 10 * np.sin(2 * np.pi * k / frames), azim=360 * k / frames
            )
            fig.canvas.draw()
            writer.write(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            if k == 0:
                fig.savefig(path.with_suffix(".png"), facecolor=fig.get_facecolor())
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
    sky = grayt.CelestialSky.procedural(2048, 1024, seed=42, grid=True)
    plt.imsave(args.output / "celestial_map.png", sky.image)
    camera = grayt.Camera(
        r=100,
        theta=70,
        x=(-26, 26),
        y=(-16, 16),
        resolution=(192, 120) if args.quick else (640, 400),
    )
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
            exposure=5000,
            rtol=2e-9,
            atol=2e-11,
            escape_radius=200,
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
        thumbnails.append((title, image.rgb.copy()))
        print(f"{slug}: {counts}; {records[-1]['seconds']:.1f}s", flush=True)
    fig, axes = plt.subplots(4, 4, figsize=(20, 13), facecolor="#090d16")
    for ax, (title, rgb) in zip(axes.ravel(), thumbnails):
        ax.imshow(rgb.transpose(1, 0, 2), origin="lower")
        ax.set_title(title, color="white", fontsize=10)
        ax.axis("off")
    for ax in axes.ravel()[len(thumbnails) :]:
        ax.axis("off")
    fig.suptitle(
        "GRAVTRACER | stationary geometries and prescribed radiation",
        color="white",
        fontsize=19,
    )
    fig.text(
        0.04,
        0.025,
        "Kerr panels: Page-Thorne + Keplerian motion. Stellar and charged panels: prescribed emission. Sky: synthetic. Fixed display exposure.",
        color="#b9c3d2",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    fig.savefig(args.output / "gallery.png", dpi=130)
    plt.close(fig)
    (args.output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    if not args.skip_videos:
        frames = 12 if args.quick else 96
        cam = replace(camera, resolution=(192, 120) if args.quick else (480, 300))
        for name, st, disk, surface in [
            (
                "kerr_camera_orbit",
                grayt.BlackHole(0.8),
                grayt.PageThorneDisk(grayt.BlackHole(0.8)),
                None,
            ),
            ("stellar_camera_orbit", grayt.SphericalStar(5), None, models()[9][4]),
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
                exposure=5000,
                archive_every=max(1, frames // 4),
                coordinate_time_step=1 / 24,
                rtol=1e-8,
                atol=1e-10,
                escape_radius=200,
                progress=lambda i, n: (
                    print(f"  {i}/{n}", flush=True) if i % 12 == 0 else None
                ),
            )
        print("Rendering 3D trajectory movie", flush=True)
        ray_movie(args.output / "kerr_geodesics_3d.mp4", frames)
    build_index(args.output)
    print(f"Artifacts: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()

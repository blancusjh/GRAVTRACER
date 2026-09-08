"""Slower, antialiased Kerr observer orbit using axial symmetry exactly.

Only the stationary, axisymmetric Page–Thorne example is reused this way.
Inclinations are retraced; at fixed inclination an azimuth change adds a
constant to every endpoint phi, preserving disk emission and all momenta.
The sky is sampled at the new endpoint, not rotated as a flat photograph.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import grayt
import numpy as np
from grayt.animation import display_frame
from grayt.sky import compose_rgb
from model_gallery import build_index
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/stationary_models"))
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args()
    if (
        args.frames < 4
        or args.frames % 2
        or args.width < 16
        or args.width % 16
        or args.samples < 1
        or args.fps < 1
    ):
        parser.error(
            "use an even frame count, width divisible by 16, and positive samples/fps"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "kerr_camera_orbit.mp4"
    nx, ny = args.width, args.width * 5 // 8
    cam = grayt.Camera(r=100, theta=45, x=(-26, 26), y=(-16, 16), resolution=(nx, ny))
    cameras = list(grayt.orbit_cameras(cam, args.frames))
    st = grayt.BlackHole(0.8)
    disk = grayt.PageThorneDisk(st)
    sky = grayt.CelestialSky.procedural(2048, 1024, seed=42, grid=True)
    records = [None] * args.frames
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="gravtracer-orbit-") as tmp:
        for i in range(args.frames // 2 + 1):
            camera = replace(
                cameras[i], phi=0, resolution=(nx * args.samples, ny * args.samples)
            )
            image = grayt.render_scene(
                st,
                camera,
                disk,
                sky=sky,
                exposure=5000,
                escape_radius=200,
                rtol=1e-8,
                atol=1e-10,
            )
            for frame in sorted({i, (-i) % args.frames}):
                phi = np.deg2rad(cameras[frame].phi)
                rgb = compose_rgb(
                    image.intensity,
                    image.status,
                    image.theta_inf,
                    image.phi_inf + phi,
                    sky,
                    exposure=5000,
                )
                Image.fromarray(display_frame(rgb, args.samples)).save(
                    Path(tmp) / f"{frame:05d}.png"
                )
                meta = copy.deepcopy(image.meta)
                meta["camera"]["phi"] = cameras[frame].phi
                records[frame] = {
                    "frame": frame,
                    "metadata": meta,
                    "status_counts": np.bincount(
                        image.status.ravel(), minlength=4
                    ).tolist(),
                }
                if frame % (args.frames // 4) == 0:
                    ep = image.rays.endpoint.copy()
                    ep[..., 3] += phi
                    rays = replace(
                        image.rays,
                        endpoint=ep,
                        meta={
                            **image.rays.meta,
                            "camera": meta["camera"],
                        },
                    )
                    replace(image, rays=rays, rgb=rgb, meta=meta).save(
                        path.with_name(f"{path.stem}_{frame:04d}.npz")
                    )
            if i % 10 == 0:
                print(
                    f"{i}/{args.frames // 2} inclinations; {time.monotonic() - started:.0f}s",
                    flush=True,
                )
        temp_video = path.with_name(path.stem + ".partial.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-framerate",
                str(args.fps),
                "-i",
                str(Path(tmp) / "%05d.png"),
                "-c:v",
                "libx264",
                "-crf",
                "17",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(temp_video),
            ],
            check=True,
        )
        temp_video.replace(path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "fps": args.fps,
                "coordinate_time_step": 0.0,
                "display_resolution": [nx, ny],
                "supersampling": args.samples,
                "pixel_filter": "box average of display RGB; raw subrays preserved",
                "geometry_reuse": "axial isometry of stationary Kerr and axisymmetric Page-Thorne disk",
                "frames": records,
            },
            indent=2,
        )
        + "\n"
    )
    build_index(args.output)
    print(
        f"Saved {path}: {args.frames / args.fps:g}s, {args.samples**2} rays/pixel",
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Views from all around a Kerr black hole while its disk turns.

Frame k is what a stationary (ZAMO) observer at r = 400 M sees at
coordinate time t_k = k dt, placed at successive azimuths and inclinations
between 60 and 84 degrees, so the lensed Milky Way streams around the
Einstein ring. It is a sequence of observers, not one moving camera: a
single camera covering 360 degrees in this time would exceed light speed,
and a physically moving camera would also see aberration.
The disk is a viscously spreading thin disk (grayt.spreading_disk):
Page-Thorne inside, with the Lynden-Bell & Pringle outer taper, opaque
where optically thick and fading into transparency as it cools, with gray
slab transfer at every crossing. Color is bolometric intensity (g^4
boosted) on one fixed log scale for every frame, as in EHT images and
NASA SVS 13326; it is not true color (see examples/readme_figures.py).

To make the rotation visible, the disk carries knots of enhanced
dissipation advected with the Keplerian angular velocity of the same
circular-geodesic flow that sets the Doppler shifts (examples/
_disk_texture.py). Differential rotation shears them into arcs. This is
a prescribed pattern, not MHD.
Emission times include the light travel time along each ray, and the
sky is sampled along each ray's asymptotic direction.

Writes an MP4 and optionally a slowed-down GIF for the README (ffmpeg).

Usage: python examples/kerr_movie.py [--frames N] [--res NX NY] [--ss S]
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont

from _common import out

import grayt
from grayt import style
from grayt.animation import VideoWriter, display_frame

SPIN = 0.9
R_C, TAU_C = 5.0, 1e3   # spreading-disk scale radius [M], tau_perp(r_c)
DECADES = 8.0           # log color scale spanning all gas that blocks starlight
SKY_GAIN = 1.4          # display brightening of the (uncalibrated) star map
# The Milky Way's core lies behind the hole for a camera at phi = 90 deg.
# Start 120 deg earlier, over sparse sky, so the band sweeps in behind the
# hole about a third of the way through and is seen deforming as it enters.
PHI_START = -30.0
PHI_SLOWEST = 90.0      # the orbit eases to its slowest speed here
EASE = 0.65             # speed there is (1 - EASE) of the mean


def cameras(frames, resolution):
    camera = grayt.Camera(r=150.0, theta=72.0, phi=90.0, x=(-40.0, 40.0),
                          y=(-25.0, 25.0), resolution=resolution)
    u0 = (PHI_SLOWEST - PHI_START) / 360.0
    for k in range(frames):
        u = k / frames
        phase = 2 * np.pi * u
        # periodic easing: d(phi)/du = 360 (1 - EASE cos 2pi(u - u0))
        eased = u - EASE * (np.sin(2 * np.pi * (u - u0))
                            + np.sin(2 * np.pi * u0)) / (2 * np.pi)
        yield replace(camera, theta=float(72.0 + 12.0 * np.sin(phase)),
                      phi=float((PHI_START + 360.0 * eased) % 360.0))


def caption(frame, camera, t, font, small):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    w, h = image.size
    ink, muted = (236, 230, 218), (154, 149, 140)
    draw.text((0.03 * w, 0.035 * h), "Kerr black hole, a = 0.9", fill=ink,
              font=font)
    lines = ("color: bolometric intensity, log scale; sheared knots (prescribed)",
             "stationary observers at r = 150 M, one per frame")
    for n, line in enumerate(lines):
        draw.text((0.03 * w, 0.035 * h + 1.35 * font.size
                   + n * 1.3 * small.size), line, fill=muted, font=small)
    draw.text((0.03 * w, 0.93 * h),
              f"i = {camera.theta:4.1f}°    φ = {camera.phi:5.1f}°    "
              f"t = {t:5.0f} M", fill=muted, font=small)
    return np.asarray(image)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames", type=int, default=192)
    ap.add_argument("--res", type=int, nargs=2, default=(640, 400))
    ap.add_argument("--ss", type=int, default=2, help="supersampling")
    ap.add_argument("--dt", type=float, default=1.5,
                    help="observer coordinate time per frame [M]")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("-o", "--output", default=out("kerr_orbit.mp4"))
    ap.add_argument("--gif", default=None,
                    help="also write a GIF preview here (needs ffmpeg)")
    ap.add_argument("--gif-width", type=int, default=560)
    ap.add_argument("--gif-slowdown", type=float, default=1.5,
                    help="play the GIF this many times slower than the MP4")
    args = ap.parse_args()

    bh = grayt.BlackHole(SPIN)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _disk_texture import turbulent_disk

    disk = turbulent_disk(bh, r_c=R_C, tau_c=TAU_C)
    sky = grayt.CelestialSky.nasa_starmap(gain=SKY_GAIN)
    exposure = None         # fixed from the first frame: no flicker
    from matplotlib import font_manager
    face = font_manager.findfont(font_manager.FontProperties(family=style.SERIF))
    font = ImageFont.truetype(face, max(12, args.res[1] // 18))
    small = ImageFont.truetype(face, max(10, args.res[1] // 30))

    started = time.time()
    with VideoWriter(args.output, tuple(args.res), args.fps) as writer:
        for k, camera in enumerate(cameras(args.frames, tuple(args.res))):
            traced = replace(camera, resolution=tuple(n * args.ss
                                                      for n in args.res))
            t = k * args.dt
            image = grayt.render_scene(bh, traced, disk, observer_time=t,
                                       escape_radius=300.0)
            exposure = exposure or 1.0 / (0.5 * image.intensity.max())
            image.rgb = grayt.sky.compose_rgb(
                image.intensity, image.status, image.theta_inf,
                image.phi_inf, sky, exposure=exposure, tone="log",
                decades=DECADES, transmission=image.transmission)
            frame = caption(display_frame(image.rgb, args.ss), camera, t,
                            font, small)
            writer.write(frame)
            elapsed = time.time() - started
            print(f"frame {k + 1}/{args.frames}  {elapsed / (k + 1):.1f} s/frame",
                  flush=True)
    print(f"wrote {args.output}")
    if args.gif:
        write_gif(args.output, args.gif, args.gif_width, args.fps,
                  args.gif_slowdown)
        print(f"wrote {args.gif}")


def write_gif(mp4, gif, width, fps, slowdown):
    """Two-pass palette GIF from the MP4, played ``slowdown`` times slower.
    Every frame is kept; only the display time per frame grows."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("GIF export requires ffmpeg on PATH")
    Path(gif).parent.mkdir(parents=True, exist_ok=True)
    scale = f"fps={fps / slowdown:g},scale={width}:-2:flags=lanczos"
    subprocess.run(
        [ffmpeg, "-loglevel", "error", "-y", "-i", str(mp4), "-filter_complex",
         f"[0:v]setpts={slowdown:g}*PTS,{scale},split[a][b];"
         "[a]palettegen=max_colors=192:stats_mode=full[p];"
         "[b][p]paletteuse=dither=sierra2_4a", str(gif)],
        check=True)


if __name__ == "__main__":
    main()

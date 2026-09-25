"""Views from all around a Kerr black hole while its disk turns.

Frame k is what a stationary (ZAMO) observer at r = 100 M sees at
coordinate time t_k = k dt, placed at successive azimuths and inclinations
between 60 and 84 degrees, so the lensed Milky Way streams around the
Einstein ring. It is a sequence of observers, not one moving camera: a
single camera covering 360 degrees in this time would exceed light speed,
and a physically moving camera would also see aberration.
The disk is a translucent Page-Thorne slab (SlabDisk): opaque near the
ISCO, optically thin further out, so starlight passes through it and every
disk crossing of every ray is counted.

To make the rotation visible, the disk carries hot spots whose brightness
pattern is advected with the Keplerian angular velocity of the same
circular-geodesic flow that sets the Doppler shifts. Differential rotation
shears them into spiral arcs. This is a prescribed pattern, not MHD.
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
R_OUT = 60.0
EXPOSURE = 240.0        # white at I = 1/240; log tone over 2.5 decades
SKY_GAIN = 1.4
# The Milky Way's core lies behind the hole for a camera at phi = 90 deg.
# Start 120 deg earlier, over sparse sky, so the band sweeps in behind the
# hole about a third of the way through and is seen deforming as it enters.
PHI_START = -30.0
PHI_SLOWEST = 90.0      # the orbit eases to its slowest speed here
EASE = 0.65             # speed there is (1 - EASE) of the mean


def vertical_optical_depth(r, phi):
    """tau_perp = 2 (6 M / r)^2: opaque inside ~8 M, thin beyond ~20 M."""
    return 2.0 * (6.0 / r) ** 2


def hot_spot_disk(bh, n_spots=9, seed=3):
    """Page-Thorne source function times an advected hot-spot pattern."""
    base = grayt.PageThorneDisk(bh, r_out=R_OUT)
    rng = np.random.default_rng(seed)
    radii = rng.uniform(bh.isco + 1.0, 16.0, n_spots)
    phases = rng.uniform(0, 2 * np.pi, n_spots)
    amplitude = rng.uniform(0.8, 1.8, n_spots)

    def intensity(r, phi, t):
        omega = 1.0 / (r**1.5 + bh.a)          # prograde Kerr Keplerian
        pattern = np.zeros(np.shape(r))
        for rk, pk, ak in zip(radii, phases, amplitude):
            radial = np.exp(-(((r - rk) / (0.12 * rk + 0.4)) ** 2))
            angular = np.exp(4.0 * (np.cos(phi - pk - omega * t) - 1.0))
            pattern += ak * radial * angular
        return base.intensity(r, phi, t) * (1.0 + pattern)

    return grayt.EmittingDisk(
        base.r_in, base.r_out, intensity,
        name="Page-Thorne x hot spots advected with Keplerian Omega(r)",
        provenance={"spots": n_spots, "seed": seed,
                    "note": "prescribed brightness pattern; not MHD"})


def cameras(frames, resolution):
    camera = grayt.Camera(r=100.0, theta=72.0, phi=90.0, x=(-40.0, 40.0),
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
    draw.text((0.03 * w, 0.035 * h + 1.35 * font.size),
              "translucent Page–Thorne slab with Keplerian hot spots; "
              "stationary observers at r = 100 M",
              fill=muted, font=small)
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
    slab = grayt.SlabDisk(hot_spot_disk(bh), vertical_optical_depth)
    sky = grayt.CelestialSky.nasa_starmap(gain=SKY_GAIN)
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
            image = grayt.render_scene(bh, traced, slab, sky=sky,
                                       exposure=EXPOSURE, tone="log",
                                       observer_time=t,
                                       escape_radius=200.0)
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

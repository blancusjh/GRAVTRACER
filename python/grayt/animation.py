"""Reproducible camera paths and streaming MP4 output (external ffmpeg).

No frames are normalized individually: fixed exposure avoids artificial
brightness flicker. Scientific keyframes can be saved alongside the video.
"""

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np


def orbit_cameras(camera, frames=96, inclination=(45.0, 80.0), turns=1.0):
    """Smooth closed observer orbit; excludes the duplicated final frame.

    The observer remains a ZAMO at every position. This is a sequence of
    observer views, not a dynamical camera worldline or an added orbital boost.
    """
    if frames < 2 or not 0 < min(inclination) <= max(inclination) < 180:
        raise ValueError(
            "require >=2 frames and inclinations strictly between 0 and 180"
        )
    for k in range(frames):
        phase = 2 * np.pi * k / frames
        theta = (
            inclination[0] + (inclination[1] - inclination[0]) * (1 - np.cos(phase)) / 2
        )
        yield replace(
            camera, theta=float(theta), phi=float(camera.phi + 360 * turns * k / frames)
        )


class VideoWriter:
    """Stream uint8 RGB frames to ffmpeg; publish output only after success."""

    def __init__(self, path, resolution, fps=24):
        self.path = Path(path)
        self.width, self.height = resolution
        if min(resolution) < 2 or any(n % 2 for n in resolution) or fps <= 0:
            raise ValueError("video dimensions must be positive even integers; fps > 0")
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("MP4 export requires ffmpeg on PATH")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temp = self.path.with_name(self.path.stem + ".partial.mp4")
        self.log_path = self.path.with_suffix(".ffmpeg.log")
        self.log = self.log_path.open("wb")
        self.process = subprocess.Popen(
            [
                executable,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{self.width}x{self.height}",
                "-framerate",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "19",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.temp),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self.log,
        )
        self.count = 0

    def write(self, rgb):
        rgb = np.asarray(rgb)
        if rgb.shape != (self.height, self.width, 3) or rgb.dtype != np.uint8:
            raise ValueError("video frames must be uint8 (height,width,3)")
        try:
            self.process.stdin.write(np.ascontiguousarray(rgb).tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(f"ffmpeg failed; see {self.log_path}") from exc
        self.count += 1

    def __enter__(self):
        return self

    def __exit__(self, kind, value, tb):
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        if kind is not None:
            self.process.terminate()
        code = self.process.wait()
        self.log.close()
        if kind is None:
            if code or not self.count:
                raise RuntimeError(f"ffmpeg export failed; see {self.log_path}")
            self.temp.replace(self.path)


def render_movie(
    path,
    spacetime,
    cameras,
    *,
    disk=None,
    surface=None,
    sky=None,
    fps=24,
    exposure=1.0,
    supersampling=1,
    archive_every=0,
    coordinate_time_step=0.0,
    progress=None,
    **trace_options,
):
    """Render views, video, JSON manifest, and optional raw NPZ keyframes.

    Playback fps is independent of coordinate_time_step (in GM/c^3). By
    default every view samples the same observer coordinate time. With
    supersampling=N, trace N*N subpixels and average their display RGB.
    Saved scientific keyframes retain every subray; categories, momenta,
    and endpoints are never averaged across an occultation boundary.
    """
    from .scene import render_scene

    if not isinstance(supersampling, (int, np.integer)) or supersampling < 1:
        raise ValueError("supersampling must be a positive integer")
    cameras = list(cameras)
    if not np.isfinite(coordinate_time_step) or archive_every < 0:
        raise ValueError("require finite coordinate_time_step and archive_every >= 0")
    if not cameras or len({c.resolution for c in cameras}) != 1:
        raise ValueError("provide cameras with one shared resolution")
    path = Path(path)
    records = []
    with VideoWriter(path, cameras[0].resolution, fps) as writer:
        for i, camera in enumerate(cameras):
            traced_camera = replace(
                camera, resolution=tuple(n * supersampling for n in camera.resolution)
            )
            img = render_scene(
                spacetime,
                traced_camera,
                disk,
                surface=surface,
                sky=sky,
                exposure=exposure,
                observer_time=i * coordinate_time_step,
                **trace_options,
            )
            rgb = display_frame(img.rgb, supersampling)
            writer.write(rgb)
            records.append(
                {
                    "frame": i,
                    "metadata": img.meta,
                    "status_counts": np.bincount(
                        img.status.ravel(), minlength=4
                    ).tolist(),
                }
            )
            if archive_every and i % archive_every == 0:
                img.save(path.with_name(f"{path.stem}_{i:04d}.npz"))
            if progress:
                progress(i + 1, len(cameras))
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "fps": fps,
                "display_resolution": list(cameras[0].resolution),
                "supersampling": int(supersampling),
                "pixel_filter": "box average of display RGB; raw subrays preserved",
                "coordinate_time_step": coordinate_time_step,
                "frames": records,
            },
            indent=2,
        )
        + "\n"
    )
    return path


def display_frame(rgb, supersampling=1):
    """Convert (nx,ny,3) display colors to a top-down uint8 video frame.

    This averages visualization colors, not calibrated spectral radiance.
    Opaque geometry remains opaque for each subray.
    """
    if not isinstance(supersampling, (int, np.integer)) or supersampling < 1:
        raise ValueError("supersampling must be a positive integer")
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("RGB must have shape (nx,ny,3)")
    nx, ny, _ = rgb.shape
    if nx % supersampling or ny % supersampling:
        raise ValueError("RGB dimensions must be divisible by supersampling")
    if supersampling > 1:
        rgb = rgb.reshape(
            nx // supersampling, supersampling, ny // supersampling, supersampling, 3
        ).mean(axis=(1, 3))
    return np.round(np.clip(rgb.transpose(1, 0, 2)[::-1], 0, 1) * 255).astype(np.uint8)

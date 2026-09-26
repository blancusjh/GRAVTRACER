"""Compare live GPU rendering with the CPU reference at OSIRIS Fig. 13 settings.

Run after installing the viewer extra, without PYTHONPATH=python:
    python examples/viewer_review.py --output output/viewer_review

Writes a comparison figure, scientific maps, and timings. Timing samples vary
inclination to force fresh integrations; azimuth samples measure ray reuse.
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import grayt
from grayt.gpu import Renderer


def timed_render(renderer, camera):
    start = time.perf_counter()
    image = renderer.render(camera)
    return image, 1000 * (time.perf_counter() - start)


def timings(frames):
    rows = {}
    for resolution in [(128, 64), (512, 256), (1024, 512)]:
        camera = grayt.Camera(resolution=resolution)
        renderer = Renderer(grayt.BlackHole(.95), grayt.ThinDisk(l0=1.8),
                            resolution, rtol=1e-5, atol=1e-7)
        renderer.render(camera)  # exclude initialization/compilation
        fresh = [timed_render(renderer, dataclasses.replace(camera, theta=85 - .2 * (i + 1)))[1]
                 for i in range(frames)]
        renderer.render(camera)
        cached = [timed_render(renderer, dataclasses.replace(camera, phi=3 * (i + 1)))[1]
                  for i in range(frames)]
        rows[f"{resolution[0]}x{resolution[1]}"] = {
            "fresh_median_ms": float(np.median(fresh)),
            "azimuth_median_ms": float(np.median(cached)),
            "fresh_samples_ms": fresh, "azimuth_samples_ms": cached,
            "device": renderer.device_name, "precision": renderer.precision,
        }
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/viewer_review"))
    parser.add_argument("--res", type=int, nargs=2, default=(2048, 1024))
    parser.add_argument("--frames", type=int, default=12)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    camera = grayt.Camera(resolution=tuple(args.res))
    report = {"camera": dataclasses.asdict(camera), "timings": timings(args.frames), "spins": []}
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
    normalization = None
    for row, (spin, l0) in enumerate([(0., 2.8), (.5, 2.8), (.95, 1.8)]):
        spacetime, disk = grayt.BlackHole(spin), grayt.ThinDisk(l0=l0)
        start = time.perf_counter()
        cpu = grayt.render(spacetime, camera, disk, rtol=1e-8, atol=1e-10)
        cpu_ms = 1000 * (time.perf_counter() - start)
        renderer = Renderer(spacetime, disk, camera.resolution, rtol=1e-5, atol=1e-7)
        gpu, gpu_ms = timed_render(renderer, camera)
        if normalization is None:
            normalization = float(cpu.intensity.max())
        scale = float(cpu.intensity.max()) / normalization
        difference = gpu.intensity - cpu.intensity
        metrics = {
            "spin": spin, "l0": l0, "cpu_ms": cpu_ms, "gpu_ms": gpu_ms,
            "cpu_max": float(cpu.intensity.max()), "gpu_max": float(gpu.intensity.max()),
            "classification_disagreement_fraction": float(np.mean(cpu.status != gpu.status)),
            "intensity_rmse_relative_to_cpu_peak": float(np.sqrt(np.mean(difference**2)) / cpu.intensity.max()),
            "gpu_unresolved_rays": int(np.count_nonzero(gpu.status == 3)),
        }
        report["spins"].append(metrics)
        for col, (name, image) in enumerate([("CPU fp64", cpu), (f"GPU {renderer.precision}", gpu)]):
            ax = axes[row, col]
            plotted = ax.imshow(image.intensity.T / normalization, origin="lower", extent=camera.extent,
                                cmap="afmhot", vmin=0, vmax=scale, aspect="equal", interpolation="bilinear")
            ax.set_title(f"{name} | a={spin:g}, l₀={l0:g}, θ=85°")
            ax.set_xlabel("x / M")
            ax.set_ylabel("y / M")
            fig.colorbar(plotted, ax=ax, label="I / max(I at a=0)", fraction=.03, pad=.02)
            image.save(args.output / f"{name.split()[0].lower()}_a{spin:g}.npz")
        print(json.dumps(metrics), flush=True)
    fig.savefig(args.output / "osiris_comparison.png", dpi=150)
    plt.close(fig)
    (args.output / "measurements.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["timings"], indent=2), flush=True)
    print(f"Wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()

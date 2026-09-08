"""Export/import a prescribed 3D gray torus and check transfer convergence.

The snapshot is synthetic, not a GRMHD solution. It exercises the same data
interface that a simulation's radiation postprocessor can populate.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import grayt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/stationary_models"))
    parser.add_argument("--convergence-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bh = grayt.BlackHole(0.5)
    r = np.linspace(6, 22, 49)
    theta = np.linspace(0, np.pi, 49)
    phi = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    rr, th, ph = np.meshgrid(r, theta, phi, indexing="ij")
    profile = np.exp(-(((rr - 11) / 4) ** 2) - ((th - np.pi / 2) / 0.22) ** 2) * (
        1 + 0.2 * np.cos(3 * ph)
    )
    velocity = grayt.rotating_velocity(bh, rr, th, 0.6 / (rr**1.5 + bh.a))
    volume = grayt.VolumeGrid(
        r,
        theta,
        phi,
        profile * 2e-5,
        profile * 0.07,
        velocity,
        name="prescribed gray torus; not GRMHD",
    )
    volume.save(args.output / "radiation_snapshot.npz")
    volume = grayt.VolumeGrid.load(args.output / "radiation_snapshot.npz")
    camera = grayt.Camera(
        r=50, theta=75, x=(-24, 24), y=(-15, 15), resolution=(128, 80)
    )
    if args.convergence_only:
        camera = replace(camera, resolution=(32, 20))
        images = []
        steps = (0.6, 0.3, 0.15)
        for step in steps:
            images.append(
                grayt.render_volume(
                    bh,
                    camera,
                    volume,
                    max_step=step,
                    escape_radius=75,
                    exposure=12000,
                    max_steps=8000,
                )
            )
        reference = images[-1].intensity
        records = [
            {
                "max_affine_step": step,
                "relative_L1_to_finest": float(
                    np.sum(abs(img.intensity - reference)) / np.sum(reference)
                ),
                "failed_pixels": int(np.sum(img.status == 3)),
            }
            for step, img in zip(steps, images)
        ]
        (args.output / "volume_convergence.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )
        print(json.dumps(records, indent=2))
        return
    image = grayt.render_volume(
        bh,
        camera,
        volume,
        max_step=0.3,
        escape_radius=75,
        exposure=12000,
        max_steps=8000,
    )
    image.save(args.output / "volume_torus.npz")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    image.plot(axes[0])
    axes[0].set_title("Prescribed 3D gray torus | Kerr a=0.5")
    im = axes[1].imshow(
        image.optical_depth.T, origin="lower", extent=image.extent, cmap="magma"
    )
    axes[1].set_title("Integrated optical depth")
    fig.colorbar(im, ax=axes[1])
    fig.tight_layout()
    fig.savefig(args.output / "volume_torus.png", dpi=150)
    plt.close(fig)
    print(
        "Volume image status counts:",
        np.bincount(image.status.ravel(), minlength=4),
        flush=True,
    )


if __name__ == "__main__":
    main()

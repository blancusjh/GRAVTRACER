"""Measure metric-grid convergence against analytic Kerr; save JSON + plot."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import grayt
from grayt.emission import metric_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/stationary_models"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    exact = grayt.BlackHole(0.7)
    rh = exact.horizon
    camera = grayt.Camera(r=100, theta=70, x=(-12, 12), y=(-8, 8), resolution=(48, 32))
    reference = grayt.trace_bundle(exact, camera, rtol=1e-11, atol=1e-13)
    rows = []
    for nr in (96, 192, 384, 768):
        st = grayt.CustomMetric(
            lambda r, th: metric_samples(exact, r, th),
            r_min=rh + 1e-4,
            r_max=1000,
            inner_radius=rh + 0.01,
            radial_origin=rh,
            resolution=(nr, 129),
            name="Kerr interpolation convergence",
        )
        error = []
        for r, th in zip(np.geomspace(2.2, 100, 40), np.linspace(0.2, 2.9, 40)):
            error.append(
                np.max(
                    abs(st.metric_cov(r, th) - exact.metric_cov(r, th))
                    / np.maximum(abs(exact.metric_cov(r, th)), 1e-12)
                )
            )
        rays = grayt.trace_bundle(st, camera, rtol=1e-11, atol=1e-13)
        escaped = (rays.status == 0) & (reference.status == 0)
        # Angular separation of finite-sphere endpoint directions, invariant
        # to longitude wrapping (but still tied to the chosen coordinate sphere).
        t, p = rays.endpoint[escaped, 2], rays.endpoint[escaped, 3]
        tr, pr = reference.endpoint[escaped, 2], reference.endpoint[escaped, 3]
        a = np.stack((np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)), axis=-1)
        b = np.stack(
            (np.sin(tr) * np.cos(pr), np.sin(tr) * np.sin(pr), np.cos(tr)), axis=-1
        )
        separation = 2 * np.arcsin(np.clip(np.linalg.norm(a - b, axis=-1) / 2, 0, 1))
        rows.append(
            {
                "nr": nr,
                "nmu": 129,
                "metric_relative_max": float(max(error)),
                "endpoint_angle_p95_rad": float(np.quantile(separation, 0.95)),
                "status_mismatches": int(np.sum(rays.status != reference.status)),
                "escaped_hamiltonian_median": float(np.median(rays.herr[escaped])),
                "rtol": 1e-11,
                "atol": 1e-13,
            }
        )
    (args.output / "metric_convergence.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, label in zip(
        axes,
        ("metric_relative_max", "endpoint_angle_p95_rad"),
        ("Maximum relative metric error", "95th percentile endpoint angle / rad"),
    ):
        ax.loglog([r["nr"] for r in rows], [r[key] for r in rows], "o-")
        ax.set(xlabel="Radial grid samples", ylabel=label)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(
        "Sampled Kerr a=0.7 versus analytic Kerr | fixed integration tolerance"
    )
    fig.tight_layout()
    fig.savefig(args.output / "metric_convergence.png", dpi=150)
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

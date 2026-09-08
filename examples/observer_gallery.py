"""Embed an offline, live observer renderer into the scientific gallery.

The browser integrates null geodesics in float64 at arbitrary camera
inclinations, using analytic Kerr, RN, stellar-exterior and q metrics.
Reference images and their raw archives remain the scientific CPU output.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import grayt
import matplotlib
import numpy as np
from model_gallery import build_index, models
from PIL import Image


def browser_models():
    """Explicit supported browser presets; emission is separate from geometry."""
    presets = []
    cases = models()
    for spin in (0.0, 0.5, 0.8, 0.95):
        st = grayt.BlackHole(spin)
        disk = grayt.PageThorneDisk(st)
        r = np.linspace(disk.r_in, disk.r_out, 4000)
        aliases = [row[0] for row in cases if row[0].startswith(f"kerr_a{spin:g}_")]
        presets.append(
            {
                "id": f"kerr_a{spin:g}",
                "label": "Schwarzschild · a = 0"
                if spin == 0
                else f"Kerr · a = {spin:g}",
                "kind": "kerr",
                "parameter": spin,
                "capture": st.capture_radius + 0.01,
                "disk": {
                    "rIn": disk.r_in,
                    "rOut": disk.r_out,
                    "flux": disk.intensity(r, 0, 0).tolist(),
                },
                "aliases": aliases,
                "reference": f"kerr_a{spin:g}_i70.png"
                if spin != 0.8
                else "kerr_camera_orbit.mp4",
                "note": "Kerr geometry · opaque Page–Thorne disk · circular geodesic motion · bolometric g⁴ transfer",
            }
        )
    for slug, title, st, disk, surface, _, note in cases:
        if slug.startswith("kerr"):
            continue
        preset = {
            "id": slug,
            "label": title.replace("|", "·"),
            "aliases": [slug],
            "reference": slug + ".png",
            "note": note,
            "capture": st.capture_radius + (0.01 if st.mid in (1, 2) else 0),
        }
        if slug.startswith("star"):
            preset.update(
                kind="kerr",
                parameter=0.0,
                surface="spots" if slug.endswith("spots") else "uniform",
            )
        elif slug.startswith("charged"):
            r = np.linspace(disk.r_in, disk.r_out, 4000)
            preset.update(
                kind="rn",
                parameter=float(slug.split("_q")[1]),
                disk={
                    "rIn": disk.r_in,
                    "rOut": disk.r_out,
                    "flux": disk.intensity(r, 0, 0).tolist(),
                },
            )
        else:
            preset.update(kind="q", parameter=st.q)
        presets.append(preset)
    return presets


def safe_json(value):
    return json.dumps(value, separators=(",", ":")).replace("</", "<\\/")


def write_observer(output):
    assets = Path(grayt.__file__).parent / "web"
    sky = grayt.CelestialSky.procedural(2048, 1024, seed=42, grid=True)
    stream = io.BytesIO()
    Image.fromarray(np.round(255 * sky.image).astype(np.uint8)).save(
        stream, format="PNG"
    )
    config = {
        "models": browser_models(),
        "skyURL": "data:image/png;base64,"
        + base64.b64encode(stream.getvalue()).decode("ascii"),
        "palette": np.round(255 * matplotlib.colormaps["afmhot"](np.arange(256))[:, :3])
        .astype(int)
        .ravel()
        .tolist(),
    }
    worker = "\n".join(
        (assets / name).read_text()
        for name in ("metrics.js", "geodesics.js", "worker.js")
    )
    markup = """
<section id="observer-view" aria-label="Interactive rendered observer views">
<h2>Move the observer</h2>
<p>Drag across the image to orbit the object. Scroll to zoom. The disk and celestial sky
are traced together from your chosen viewpoint; the image sharpens when you stop moving.</p>
<div class="observer-toolbar">
<label>Model <select id="observer-model" aria-label="Observer model"></select></label>
<button id="observer-play" aria-pressed="false">Auto orbit</button>
<button id="observer-reset">Reset view</button><button id="observer-save" disabled>Save PNG</button>
</div>
<div class="observer-viewport">
<canvas id="observer-canvas" width="640" height="400" tabindex="0"
 aria-label="Lensed observer image. Drag or use arrow keys to orbit; scroll or use plus and minus to zoom."></canvas>
<img id="observer-poster" src="kerr_a0_i85.png" alt="Schwarzschild disk at inclination 85 degrees">
<span id="observer-position"></span>
</div>
<p id="observer-status" role="status" aria-live="polite">Starting the observer…</p>
<div class="observer-sliders">
<label>Inclination <input id="observer-inclination" aria-label="Inclination" type="range" min="5" max="175" step="0.1" value="85"><output id="observer-inclination-value">85°</output></label>
<label>Azimuth <input id="observer-azimuth" aria-label="Azimuth" type="range" min="0" max="359.9" step="0.1" value="0"><output id="observer-azimuth-value">0°</output></label>
</div>
<p class="observer-note" id="observer-note"></p>
<a id="observer-reference" href="kerr_a0_i85.png">Open reference render</a>
<details><summary>Display and model details</summary>
<div class="observer-toolbar">
<label>Disk / surface exposure <input id="observer-exposure" aria-label="Emission exposure" type="range" min="500" max="15000" step="100" value="5000"></label>
<label>Sky brightness <input id="observer-sky" aria-label="Sky brightness" type="range" min="0" max="1" step="0.02" value="1"></label>
<label>Quality <select id="observer-quality"><option value="standard">Standard</option><option value="detail">Detailed</option></select></label>
</div>
<p>The disk is a perfectly opaque, zero-thickness annulus ending at r = 20 M.
It therefore has a sharp silhouette against the sky. Its smooth, axisymmetric emission
does not change when only the observer's azimuth changes; stars behind it do move.
Four subrays per displayed pixel smooth sampling artifacts at the edge.</p>
<p>This is a stationary scene, sampled by ZAMO observers at r = 100 M. Browser geodesics
use float64 RKDP45 (preview tolerance 10⁻⁶, refined 10⁻⁸). Arbitrary inclinations
are retraced; azimuth changes reuse axial symmetry. The sky is a synthetic RGB texture
on the r = 200 M coordinate sphere. Emission colors and the sky brightness control are
display choices, not calibrated spectra. Charged and quadrupolar models are theoretical
comparisons. Imported metric tables and volume transfer use the Python renderer.</p>
</details>
</section>
"""
    page = (
        "<style>"
        + (assets / "observer.css").read_text()
        + "</style>"
        + markup
        + '<script id="observer-settings" type="application/json">'
        + safe_json(config)
        + "</script>"
        + '<script id="observer-worker-source" type="application/json">'
        + safe_json(worker)
        + "</script>"
        + "<script>"
        + (assets / "observer.js").read_text()
        + "</script>"
    )
    (output / "observer_view.html").write_text(page)
    (output / "observer_models.json").write_text(
        json.dumps(config["models"], indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/stationary_models"))
    args = parser.parse_args()
    write_observer(args.output)
    build_index(args.output)
    print(f"Embedded live observer: {(args.output / 'index.html').resolve()}")


if __name__ == "__main__":
    main()

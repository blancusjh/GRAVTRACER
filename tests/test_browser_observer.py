"""Independent browser/Fortran comparisons of geometry and radiation."""

import importlib.util
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import grayt
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def example(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "examples" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_browser_metric_formulas_are_current():
    assert (
        example("build_browser_metrics").generate()
        == (ROOT / "python/grayt/web/metrics.js").read_text()
    )


@pytest.mark.skipif(
    shutil.which("node") is None, reason="Node needed for browser parity"
)
@pytest.mark.parametrize(
    "kind,parameter,theta",
    [
        ("kerr", 0.0, 85),
        ("kerr", 0.95, 30),
        ("kerr", 0.8, 110),
        ("rn", 0.8, 70),
        ("q", 0.3, 85),
        ("star", 0.0, 50),
    ],
)
def test_browser_ray_paths_and_bolometric_radiation(kind, parameter, theta):
    disk, surface = None, None
    model = {"kind": kind, "parameter": parameter}
    if kind == "kerr":
        st = grayt.BlackHole(parameter)
        disk = grayt.PageThorneDisk(st)
    elif kind == "rn":
        st = grayt.ReissnerNordstrom(parameter)
        disk = grayt.EmittingDisk(
            6, 20, lambda r, ph, t: 2e-4 * (6 / r) ** 3 * (1 - np.sqrt(6 / r))
        )
    elif kind == "q":
        st = grayt.QMetric(parameter)
    else:
        st = grayt.SphericalStar(5)
        surface = grayt.EmittingSurface(lambda th, ph, t: np.ones_like(th))
        model.update(kind="kerr", surface="uniform")
    model["capture"] = st.capture_radius + (0.01 if st.mid in (1, 2) else 0)
    if disk:
        r = np.linspace(disk.r_in, disk.r_out, 4000)
        model["disk"] = {
            "rIn": disk.r_in,
            "rOut": disk.r_out,
            "flux": disk.intensity(r, 0, 0).tolist(),
        }
    cam = grayt.Camera(
        r=100, theta=theta, phi=47, x=(-26, 26), y=(-16, 16), resolution=(24, 16)
    )
    cpu = grayt.render_scene(
        st, cam, disk, surface=surface, escape_radius=200, rtol=1e-9, atol=1e-11
    )
    xs = -26 + 52 * (np.arange(24) + 0.5) / 24
    ys = -16 + 32 * (np.arange(16) + 0.5) / 16
    points = [[x, y] for x in xs for y in ys]
    driver = """
const fs=require('fs'),vm=require('vm');
for(const file of ['metrics','geodesics']) vm.runInThisContext(fs.readFileSync(process.argv[1]+'/'+file+'.js','utf8'));
const input=JSON.parse(fs.readFileSync(0,'utf8')),tracer=new GraytRayTracer(input.model);
const result=input.points.map(([x,y])=>{
  const ray=tracer.trace(x,y,input.camera,{rtol:1e-9,atol:1e-11});
  return {status:ray.status,y:Array.from(ray.y),intensity:tracer.emission(ray)};
});
process.stdout.write(JSON.stringify(result));
"""
    result = subprocess.run(
        ["node", "-e", driver, str(ROOT / "python/grayt/web")],
        input=json.dumps(
            {
                "model": model,
                "camera": {"r": 100, "theta": theta, "phi": 47},
                "points": points,
            }
        ),
        text=True,
        capture_output=True,
        check=True,
    )
    rows = json.loads(result.stdout)
    status = np.array([row["status"] for row in rows]).reshape(cam.resolution)
    endpoint = np.array([row["y"] for row in rows]).reshape(cam.resolution + (6,))
    intensity = np.array([row["intensity"] for row in rows]).reshape(cam.resolution)
    assert np.array_equal(status, cpu.status)
    valid = status != 3
    # Imported RN/star tables interpolate the analytic metric; their bound
    # includes that independent interpolation error, especially near caustics.
    tolerance = 2e-4 if kind in ("rn", "star") else 2e-6
    assert np.allclose(
        endpoint[valid], cpu.rays.endpoint[valid], rtol=tolerance, atol=tolerance
    )
    assert np.allclose(intensity, cpu.intensity, rtol=3e-5, atol=1e-11)


def test_axial_isometry_used_in_kerr_movie():
    st = grayt.BlackHole(0.8)
    cam = grayt.Camera(r=100, theta=80, resolution=(40, 24), x=(-26, 26), y=(-16, 16))
    disk = grayt.PageThorneDisk(st)
    sky = grayt.CelestialSky.procedural(128, 64, stars=30)
    kwargs = {
        "sky": sky,
        "exposure": 5000,
        "escape_radius": 200,
        "rtol": 1e-10,
        "atol": 1e-12,
    }
    a = grayt.render_scene(st, cam, disk, **kwargs)
    b = grayt.render_scene(st, replace(cam, phi=147), disk, **kwargs)
    from grayt.sky import compose_rgb

    rotated = compose_rgb(
        a.intensity,
        a.status,
        a.theta_inf,
        a.phi_inf + np.deg2rad(147),
        sky,
        exposure=5000,
    )
    assert np.array_equal(a.status, b.status)
    assert np.allclose(rotated, b.rgb, atol=3e-6)

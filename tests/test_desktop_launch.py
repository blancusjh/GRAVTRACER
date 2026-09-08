"""Native gallery links, allowlisted launches, and independent radiation."""

from pathlib import Path

import grayt
import numpy as np
import pytest


@pytest.mark.parametrize(
    "url",
    [
        "gravtracer://view/../../anything",
        "gravtracer://view/kerr_a0_i85?command=anything",
        "gravtracer://other/kerr_a0_i85",
        "https://view/kerr_a0_i85",
        "gravtracer://view/kerr_a0_i85#anything",
        "gravtracer://view/unknown",
    ],
)
def test_launcher_rejects_unrecognized_urls(monkeypatch, url):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from desktop_gallery import parse_url

    with pytest.raises(ValueError):
        parse_url(url, {"kerr_a0_i85"})


def test_gallery_presets_are_launchable(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from desktop_gallery import parse_url, presets
    from model_gallery import models

    available = presets()
    for row in models():
        assert parse_url("gravtracer://view/" + row[0], available) == row[0]
    assert available["kerr_camera_orbit"][2].a == 0.8


@pytest.mark.skipif(not grayt.gpu.available(), reason="OpenCL device required")
def test_native_page_thorne_matches_cpu_radiation():
    from grayt.desktop_scene import SceneRenderer

    st = grayt.BlackHole(0.8)
    disk = grayt.PageThorneDisk(st)
    camera = grayt.Camera(
        r=100, theta=70, x=(-26, 26), y=(-16, 16), resolution=(96, 60)
    )
    result = SceneRenderer(
        st, disk, camera.resolution, escape_radius=200, rtol=1e-5, atol=1e-7
    ).render(camera)
    cpu = grayt.render_scene(st, camera, disk, escape_radius=200)
    assert np.array_equal(result.status, cpu.status)
    hit = cpu.status == 2
    relative = abs(result.intensity[hit] - cpu.intensity[hit]) / np.maximum(
        cpu.intensity[hit], 1e-20
    )
    assert np.quantile(relative, 0.99) < 5e-4
    assert result.meta["escape_radius"] == 200
    assert "g^4" in result.meta["radiation"]


def test_native_cpu_surface_keeps_emission():
    from grayt.desktop_scene import SceneRenderer
    from grayt.viewer import compose_frame

    st = grayt.SphericalStar(5)
    surface = grayt.EmittingSurface(lambda th, ph, t: np.full_like(th, 3e-4))
    camera = grayt.Camera(r=100, theta=70, resolution=(24, 16), x=(-10, 10), y=(-6, 6))
    result = SceneRenderer(
        st, None, camera.resolution, surface=surface, escape_radius=200
    ).render(camera)
    frame = compose_frame(result, exposure=5000)
    assert frame[(result.status == 1).T].max() > 0.5


def test_native_disk_rejects_mismatched_metric():
    from grayt.desktop_scene import SceneRenderer

    with pytest.raises(ValueError, match="different Kerr"):
        SceneRenderer(
            grayt.BlackHole(0), grayt.PageThorneDisk(grayt.BlackHole(0.8)), (16, 12)
        )

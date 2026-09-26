"""Native settings integration: scene rebuilds, cached backgrounds, and errors."""

import sys
import os
from types import SimpleNamespace

import numpy as np
import pytest

import grayt

pytestmark = pytest.mark.skipif(
    not grayt.gpu.available() or (sys.platform.startswith("linux") and not os.getenv("DISPLAY")),
    reason="native display and OpenCL device required",
)


@pytest.fixture
def viewer():
    pytest.importorskip("PySide6")
    pytest.importorskip("vispy")
    from grayt.viewer import InteractiveViewer

    v = InteractiveViewer(
        grayt.BlackHole(.5), grayt.ThinDisk(),
        grayt.Camera(r=100, resolution=(96, 48)), preview_resolution=(48, 24),
        sky=grayt.CelestialSky.procedural(width=64, height=32, stars=100),
    )
    v.show()
    v._app.process_events()
    yield v
    v.canvas.close()
    v._app.process_events()


def test_menu_applies_spin_and_preserves_pending_edits(viewer):
    controls = viewer._settings
    renderer = viewer.renderer
    controls.fields["spin"].setValue(.8)
    viewer.state.orbit(4, 0)
    viewer.render()
    assert controls.fields["spin"].value() == .8
    assert controls.dirty
    controls.apply()
    assert viewer.spacetime.a == .8
    assert viewer.renderer is not renderer
    assert not controls.dirty
    assert controls.status.text() == "View updated."
    assert viewer._last_image.meta["azimuth_reused"] is False


def test_sky_image_and_gain_reuse_rays(viewer, tmp_path, monkeypatch):
    from PIL import Image

    path = tmp_path / "sky.png"
    Image.fromarray(np.full((16, 32, 3), 64, dtype=np.uint8)).save(path)
    old = viewer._last_image
    monkeypatch.setattr(viewer.renderer, "render", lambda _: pytest.fail("background retraced"))
    viewer.update_parameters(background="celestial", sky_path=str(path),
                             sky_longitude=45, sky_gain=2)
    assert viewer._last_image is old
    assert viewer.sky.longitude == 45
    assert viewer.sky.gain == 2
    assert np.allclose(viewer._last_frame[(old.status == 0).T], 128 / 255)
    disk = (old.status == 2).T
    assert np.any(disk)


def test_invalid_setting_keeps_scene_and_camera(viewer, tmp_path):
    old = (viewer.spacetime, viewer.disk, viewer.state.camera, viewer.renderer,
           viewer.sky, viewer._last_frame)
    for kwargs in ({"spin": 1}, {"r_out": 1}, {"radius": 1}, {"theta": 0},
                   {"brightness": -1}, {"sky_gain": 0},
                   {"spin": .8, "sky_path": str(tmp_path / "missing.png")}):
        with pytest.raises((ValueError, OSError)):
            viewer.update_parameters(**kwargs)
        assert (viewer.spacetime, viewer.disk, viewer.state.camera, viewer.renderer,
                viewer.sky, viewer._last_frame) == old


def test_hide_shortcut_and_disk_restore(viewer):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    viewer.canvas.native.setFocus()
    viewer._app.process_events()
    QTest.keyClick(viewer.canvas.native, Qt.Key.Key_H)
    viewer._app.process_events()
    assert not viewer._window.dock.isVisible()
    viewer._on_key_press(SimpleNamespace(key="H", handled=False))
    assert viewer._window.dock.isVisible()
    original = viewer.disk
    viewer.update_parameters(disk_on=False)
    assert viewer.disk is None
    viewer.update_parameters(disk_on=True)
    assert viewer.disk == original
    viewer._preview_timer.start()
    viewer.canvas.close()
    assert not viewer._window.isVisible()
    assert not viewer._preview_timer.running


def test_page_thorne_tracks_new_spin(viewer):
    disk = grayt.PageThorneDisk(viewer.spacetime, n=128)
    viewer.disk = viewer._disk_template = disk
    viewer.renderer, viewer.preview_renderer = viewer._make_renderers(
        viewer.spacetime, disk, viewer.state.camera)
    viewer.update_parameters(spin=.9, r_out=25)
    assert viewer.disk.r_in == pytest.approx(viewer.spacetime.isco)
    assert viewer.disk.r_out == 25
    assert viewer.disk.provenance["flux_samples"] == 128
    assert "g^4" in viewer._last_image.meta["radiation"]


def test_black_background_keeps_map_loading_lazy(viewer, monkeypatch):
    viewer.sky = None
    monkeypatch.setattr(grayt.CelestialSky, "nasa_starmap",
                        lambda **_: pytest.fail("black background loaded a sky image"))
    viewer.update_parameters(spin=.6, sky_longitude=90, sky_gain=3)
    assert viewer.sky is None
    assert viewer._sky_longitude == 90
    assert viewer._sky_gain == 3

"""Headless tests for viewer state and texture composition."""
import numpy as np
import pytest

import grayt
from grayt.viewer import CameraState, compose_frame


def _maps():
    shape = (4, 3)
    status = np.array(((0, 1, 2), (0, 0, 2),
                       (1, 3, 0), (2, 0, 1)), dtype=np.int32)
    intensity = np.zeros(shape)
    intensity[status == grayt.STATUS_DISK] = (0.2, 0.5, 1.0)
    return grayt.Image(
        intensity=intensity, g=np.zeros(shape), r_hit=np.zeros(shape),
        status=status, herr=np.zeros(shape),
        theta_inf=np.full(shape, np.pi/3),
        phi_inf=np.full(shape, np.pi/4), extent=(-2, 2, -1, 1))


def test_camera_state_orbit_zoom_reset_and_resolution():
    original = grayt.Camera(theta=85, phi=10, resolution=(400, 200))
    state = CameraState(original)
    state.orbit(dx=20, dy=-40)
    assert state.camera.theta == pytest.approx(75.0)
    assert state.camera.phi == pytest.approx(5.0)
    width = state.camera.x[1] - state.camera.x[0]
    state.zoom(steps=1)
    assert state.camera.x[1] - state.camera.x[0] < width
    assert state.at_resolution((80, 40)).resolution == (80, 40)
    assert state.reset() == original


def test_camera_state_keeps_away_from_coordinate_poles():
    state = CameraState(grayt.Camera(theta=90))
    assert state.orbit(0, -10_000).theta == 3.2
    assert state.orbit(0, 10_000).theta == 176.8


@pytest.mark.parametrize("mode", ["composite", "intensity", "lensing",
                                  "shadow"])
def test_compose_frame_is_rgb_texture(mode):
    image = _maps()
    frame = compose_frame(image, mode)
    assert frame.shape == (3, 4, 3)
    assert frame.dtype == np.float32
    assert np.isfinite(frame).all()
    assert frame.min() >= 0.0 and frame.max() <= 1.0
    # Captured rays remain black in every mode.
    x, y = np.argwhere(image.status == grayt.STATUS_CAPTURED)[0]
    assert np.all(frame[y, x] == 0.0)


def test_composite_contains_disk_and_lensed_background():
    image = _maps()
    frame = compose_frame(image, "composite")
    x_disk, y_disk = np.argwhere(image.status == grayt.STATUS_DISK)[-1]
    x_sky, y_sky = np.argwhere(image.status == grayt.STATUS_ESCAPED)[0]
    assert frame[y_disk, x_disk].max() > 0.5
    assert frame[y_sky, x_sky].max() > 0.0


def test_compose_frame_rejects_unknown_mode():
    with pytest.raises(ValueError, match="viewer mode"):
        compose_frame(_maps(), "infrared")


def test_view_cli_builds_scene_without_importing_matplotlib(monkeypatch):
    from grayt import cli

    called = {}

    def fake_view(spacetime, disk, camera, **kwargs):
        called.update(spacetime=spacetime, disk=disk, camera=camera,
                      kwargs=kwargs)

    monkeypatch.setattr(grayt, "view", fake_view)
    result = cli._main(("view", "-a", "0.7", "--theta", "70",
                        "--phi", "15", "--res", "320", "160",
                        "--preview-res", "80", "40", "--no-disk",
                        "--mode", "lensing"))

    assert result == 0
    assert called["spacetime"].a == pytest.approx(0.7)
    assert called["disk"] is None
    assert called["camera"].resolution == (320, 160)
    assert called["kwargs"]["preview_resolution"] == (80, 40)
    assert called["kwargs"]["mode"] == "lensing"


def test_view_cli_defaults_to_physical_intensity(monkeypatch):
    from grayt import cli

    called = {}

    def fake_view(_spacetime, _disk, _camera, **kwargs):
        called.update(kwargs)

    monkeypatch.setattr(grayt, "view", fake_view)
    assert cli._main(("view", "--res", "32", "16",
                      "--preview-res", "16", "8")) == 0
    assert called["mode"] == "intensity"

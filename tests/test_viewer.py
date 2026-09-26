"""Headless tests for viewer state and texture composition."""

import numpy as np
import pytest

import grayt
from grayt.viewer import CameraState, compose_frame, _hot


def _maps():
    shape = (4, 3)
    status = np.array(((0, 1, 2), (0, 0, 2), (1, 3, 0), (2, 0, 1)), dtype=np.int32)
    intensity = np.zeros(shape)
    intensity[status == grayt.STATUS_DISK] = (0.2, 0.5, 1.0)
    return grayt.Image(
        intensity=intensity,
        g=np.zeros(shape),
        r_hit=np.zeros(shape),
        status=status,
        herr=np.zeros(shape),
        theta_inf=np.full(shape, np.pi / 3),
        phi_inf=np.full(shape, np.pi / 4),
        extent=(-2, 2, -1, 1),
    )


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


@pytest.mark.parametrize("mode", ["composite", "intensity", "lensing", "shadow"])
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
    frame = compose_frame(image, "composite", background="grid")
    x_disk, y_disk = np.argwhere(image.status == grayt.STATUS_DISK)[-1]
    x_sky, y_sky = np.argwhere(image.status == grayt.STATUS_ESCAPED)[0]
    assert frame[y_disk, x_disk].max() > 0.5
    assert frame[y_sky, x_sky].max() > 0.0


def test_backgrounds_preserve_emission_and_use_lensed_sky():
    image = _maps()
    sky = grayt.CelestialSky(np.full((16, 32, 3), [0.2, 0.4, 0.6]))
    black = compose_frame(image)
    celestial = compose_frame(image, background="celestial", sky=sky)
    grid = compose_frame(image, background="grid")
    escaped = (image.status == 0).T
    disk = (image.status == 2).T
    assert np.all(black[escaped] == 0)
    assert np.allclose(celestial[escaped], [0.2, 0.4, 0.6])
    assert np.array_equal(black[disk], celestial[disk])
    assert np.array_equal(black[disk], grid[disk])


def test_b_cycles_background_without_retracing():
    from types import SimpleNamespace
    from grayt.viewer import InteractiveViewer

    viewer = InteractiveViewer.__new__(InteractiveViewer)
    viewer.background = "black"
    viewer._last_image = None
    event = SimpleNamespace(key="B", handled=False)
    for expected in ("celestial", "grid", "black"):
        viewer._on_key_press(event)
        assert viewer.background == expected
        assert event.handled


def test_vispy_key_objects_trigger_background_switch():
    from types import SimpleNamespace
    keys = pytest.importorskip("vispy.util.keys")
    from grayt.viewer import InteractiveViewer
    viewer = InteractiveViewer.__new__(InteractiveViewer)
    viewer.background = "black"
    viewer._last_image = None
    for key, expected in [("b", "celestial"), ("B", "grid"), ("b", "black")]:
        event = SimpleNamespace(key=keys.Key(key), handled=False)
        viewer._on_key_press(event)
        assert viewer.background == expected
        assert event.handled


def test_compose_frame_rejects_unknown_mode():
    with pytest.raises(ValueError, match="viewer mode"):
        compose_frame(_maps(), "infrared")


def test_palette_matches_reference_afmhot():
    import matplotlib

    values = np.linspace(0, 1, 256)
    assert np.allclose(_hot(values), matplotlib.colormaps["afmhot"](values)[:, :3])


def test_fixed_normalization_survives_missing_preview_peak():
    full = _maps()
    preview = _maps()
    preview.intensity[preview.intensity == 1] = 0.5
    reference = compose_frame(full, "intensity", norm_to=1.0)
    coarse = compose_frame(preview, "intensity", norm_to=1.0)
    unchanged = (full.intensity == preview.intensity).T
    assert np.array_equal(reference[unchanged], coarse[unchanged])


def test_brightness_changes_display_without_changing_raw_intensity():
    image = _maps()
    saved = image.intensity.copy()
    normal = compose_frame(image, "intensity", norm_to=1.0)
    bright = compose_frame(image, "intensity", norm_to=1.0, brightness=2)
    assert (bright >= normal).all()
    assert (bright > normal).any()
    assert np.array_equal(saved, image.intensity)


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_display_brightness_rejected(value):
    with pytest.raises(ValueError, match="brightness"):
        compose_frame(_maps(), brightness=value)


def test_drag_events_coalesce_and_release_refines_latest_camera():
    from types import SimpleNamespace
    from grayt.viewer import InteractiveViewer

    class Timer:
        running = False
        starts = 0

        def start(self):
            self.running = True
            self.starts += 1

        def stop(self):
            self.running = False

    viewer = InteractiveViewer.__new__(InteractiveViewer)
    viewer.state = CameraState(grayt.Camera(theta=85, phi=0))
    viewer._drag_pos = np.array([0, 0])
    viewer._preview_timer = Timer()
    viewer._refine_timer = Timer()
    renders = []
    viewer.render = lambda refine: renders.append((viewer.state.camera, refine))
    for i in range(1, 21):
        viewer._on_mouse_move(SimpleNamespace(pos=(i, -i), handled=False))
    assert not renders
    assert viewer._preview_timer.starts == 1
    viewer._on_preview_timer(None)
    assert len(renders) == 1
    assert renders[0][0].theta == 80
    assert renders[0][0].phi == 355
    assert renders[0][1] is False
    viewer._on_mouse_release(SimpleNamespace(button=1, handled=False))
    assert renders[-1][1] is True
    assert not viewer._preview_timer.running
    assert not viewer._refine_timer.running


def test_view_cli_builds_scene_without_importing_matplotlib(monkeypatch):
    from grayt import cli

    called = {}

    def fake_view(spacetime, disk, camera, **kwargs):
        called.update(spacetime=spacetime, disk=disk, camera=camera, kwargs=kwargs)

    monkeypatch.setattr(grayt, "view", fake_view)
    result = cli._main(
        (
            "view",
            "-a",
            "0.7",
            "--theta",
            "70",
            "--phi",
            "15",
            "--res",
            "320",
            "160",
            "--preview-res",
            "80",
            "40",
            "--no-disk",
            "--mode",
            "lensing",
        )
    )

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
    assert cli._main(("view", "--res", "32", "16", "--preview-res", "16", "8")) == 0
    assert called["mode"] == "intensity"


def test_view_cli_quality_and_display_defaults(monkeypatch):
    from grayt import cli

    called = {}

    def fake_view(_spacetime, _disk, camera, **kwargs):
        called.update(camera=camera, **kwargs)

    monkeypatch.setattr(grayt, "view", fake_view)
    assert cli._main(("view", "--brightness", "2", "--norm", "0.01")) == 0
    assert called["camera"].resolution == (1024, 512)
    assert called["preview_resolution"] == (512, 256)
    assert called["brightness"] == 2
    assert called["norm_to"] == 0.01

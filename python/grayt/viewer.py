"""Interactive, GPU-rendered observer view built on VisPy.

VisPy and a GUI backend are optional dependencies.  They are imported only
when :func:`view` or :class:`InteractiveViewer` is used, so importing
``grayt`` remains lightweight and headless-safe.
"""

from __future__ import annotations

import copy
import dataclasses
import time
from dataclasses import dataclass, field

import numpy as np

from .instruments import Camera
from .matter import ThinDisk
from .results import Image
from .spacetime import Spacetime

_MODES = ("composite", "intensity", "lensing", "shadow")
_BACKGROUNDS = ("black", "celestial", "grid")
_MIN_THETA = 3.2  # stay clear of the polar coordinate singularity


@dataclass
class CameraState:
    """Mutable interaction state around an immutable :class:`Camera`."""

    camera: Camera
    _initial: Camera = field(init=False, repr=False)

    def __post_init__(self):
        self._initial = self.camera

    def orbit(self, dx: float, dy: float, sensitivity: float = 0.25) -> Camera:
        """Orbit by a mouse displacement in pixels."""
        theta = np.clip(
            self.camera.theta + sensitivity * dy, _MIN_THETA, 180.0 - _MIN_THETA
        )
        phi = (self.camera.phi - sensitivity * dx) % 360.0
        self.camera = dataclasses.replace(
            self.camera, theta=float(theta), phi=float(phi)
        )
        return self.camera

    def zoom(self, steps: float, sensitivity: float = 0.15) -> Camera:
        """Scale both image-plane axes around their current centers."""
        factor = float(np.exp(-sensitivity * steps))

        def scaled(bounds):
            center = 0.5 * (bounds[0] + bounds[1])
            half = np.clip(0.5 * (bounds[1] - bounds[0]) * factor, 0.25, 1000.0)
            return (float(center - half), float(center + half))

        self.camera = dataclasses.replace(
            self.camera, x=scaled(self.camera.x), y=scaled(self.camera.y)
        )
        return self.camera

    def reset(self) -> Camera:
        self.camera = self._initial
        return self.camera

    def at_resolution(self, resolution: tuple[int, int]) -> Camera:
        return dataclasses.replace(self.camera, resolution=tuple(resolution))


def _hot(values: np.ndarray) -> np.ndarray:
    """Continuous ``afmhot`` map, matching the paper-style plotting palette."""
    return np.stack(
        (
            np.clip(2.0 * values, 0.0, 1.0),
            np.clip(2.0 * values - 0.5, 0.0, 1.0),
            np.clip(2.0 * values - 1.0, 0.0, 1.0),
        ),
        axis=-1,
    )


def compose_frame(
    image: Image,
    mode: str = "composite",
    *,
    background="black",
    sky=None,
    exposure=None,
    norm_to=None,
    brightness=1.0,
) -> np.ndarray:
    """Turn ray-tracing maps into an RGB ``(ny, nx, 3)`` display texture.

    Background selection is independent of emission display mode. The
    default is black; celestial samples the supplied map along escaped rays;
    grid retains the original diagnostic pattern. Shadow is monochrome.
    """
    if mode not in _MODES:
        raise ValueError(f"unknown viewer mode {mode!r}; use {', '.join(_MODES)}")
    if background not in _BACKGROUNDS:
        raise ValueError(f"unknown background {background!r}")
    if not np.isfinite(brightness) or brightness <= 0:
        raise ValueError("brightness must be finite and positive")
    if norm_to is not None and (not np.isfinite(norm_to) or norm_to <= 0):
        raise ValueError("norm_to must be finite and positive")

    status = image.status
    escaped = status == 0
    disk = status == 2
    rgb = np.zeros(status.shape + (3,), dtype=np.float32)

    if mode != "shadow" and background == "celestial":
        if sky is None:
            raise ValueError("celestial background requires a sky map")
        rgb[escaped] = sky.sample(image.theta_inf[escaped], image.phi_inf[escaped])
    elif mode != "shadow" and background == "grid":
        theta = np.arccos(np.clip(np.cos(image.theta_inf), -1.0, 1.0))
        phi = np.mod(image.phi_inf, 2.0 * np.pi)
        top = theta < np.pi / 2.0
        east = phi < np.pi
        colors = np.array(
            (
                (0.05, 0.16, 0.08),
                (0.22, 0.06, 0.06),
                (0.05, 0.08, 0.22),
                (0.20, 0.16, 0.04),
            ),
            dtype=np.float32,
        )
        rgb[escaped & top & east] = colors[0]
        rgb[escaped & top & ~east] = colors[1]
        rgb[escaped & ~top & east] = colors[2]
        rgb[escaped & ~top & ~east] = colors[3]

        # Fifteen-degree latitude/longitude lines make gravitational
        # deflection visible while keeping the texture cheap to regenerate.
        grid = escaped & (
            (np.abs(np.sin(12.0 * theta)) < 0.045)
            | (np.abs(np.sin(12.0 * phi)) < 0.045)
        )
        rgb[grid] = (0.72, 0.72, 0.72)
    elif mode == "shadow":
        rgb[escaped] = (0.82, 0.82, 0.82)

    if mode in ("composite", "intensity") and np.any(disk):
        values = np.maximum(image.intensity, 0.0)
        if exposure is None:
            scale = norm_to if norm_to is not None else float(values[disk].max())
            normalized = np.clip(brightness * values / (scale or 1.0), 0.0, 1.0)
            rgb[disk] = _hot(normalized[disk])
    # Stellar surfaces have status=1; their positive intensity distinguishes
    # them from the dark absorbing boundary of a black hole.
    if mode in ("composite", "intensity") and exposure is not None:
        import matplotlib

        emitting = image.intensity > 0
        rgb[emitting] = matplotlib.colormaps["afmhot"](
            -np.expm1(-brightness * exposure * image.intensity[emitting])
        )[..., :3]
    rgb[status == 3] = (1, 0, 1)

    # GRAVTRACER maps are (nx, ny); image textures are rows then columns.
    return np.ascontiguousarray(rgb.transpose(1, 0, 2))


def _load_vispy(backend: str | None):
    try:
        from vispy import app, scene
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "the interactive viewer needs VisPy and PySide6; install gravtracer[viewer]"
        ) from exc
    try:
        app.use_app(backend or "pyside6")
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "VisPy could not start a GUI backend; install "
            "gravtracer[viewer] (VisPy + PySide6)"
        ) from exc
    return app, scene


class InteractiveViewer:
    """Desktop observer view with live GPU re-rendering.

    Drag with the left mouse button to orbit in ``(theta, phi)`` and use the
    wheel to zoom the image plane.  Dragging and wheel bursts use the preview
    renderer; release/idle refines at the final resolution.  Keys: ``R``
    reset, ``M`` cycle display mode, ``B`` cycle black/celestial/grid,
    ``Space`` refine, ``S`` save a PNG,
    ``+``/``-`` adjust display brightness, ``H`` hide/show settings,
    and ``Escape`` close.
    Azimuth-only motion reuses traced rays at full resolution. Other motion
    requests are coalesced into previews, followed by refinement when idle.
    """

    def __init__(
        self,
        spacetime: Spacetime,
        disk: ThinDisk | None = None,
        camera: Camera | None = None,
        preview_resolution: tuple[int, int] = (512, 256),
        mode: str = "intensity",
        precision: str = "auto",
        device: int | None = None,
        rtol: float = 1e-5,
        atol: float = 1e-7,
        max_steps: int = 500_000,
        backend: str | None = None,
        background: str = "black",
        sky=None,
        surface=None,
        exposure=None,
        escape_radius=None,
        norm_to: float | None = None,
        brightness: float = 1.0,
        window_size: tuple[int, int] = (1100, 650),
    ):
        if mode not in _MODES:
            raise ValueError(f"unknown viewer mode {mode!r}; use {', '.join(_MODES)}")
        if camera is None:
            camera = Camera()

        self._window = None
        self._settings = None
        self._disk_template = disk
        self.surface = surface
        self.escape_radius = escape_radius
        self._auto_norm = norm_to is None
        self.spacetime = spacetime
        self.disk = disk
        self.state = CameraState(camera)
        self.mode = mode
        if background not in _BACKGROUNDS:
            raise ValueError(f"unknown background {background!r}")
        self.background = background
        self.sky = sky
        self._sky_longitude = getattr(sky, "longitude", 180.0)
        self._sky_gain = getattr(sky, "gain", 1.0)
        self.exposure = exposure
        if not np.isfinite(brightness) or brightness <= 0:
            raise ValueError("brightness must be finite and positive")
        if norm_to is not None and (not np.isfinite(norm_to) or norm_to <= 0):
            raise ValueError("norm_to must be finite and positive")
        self.norm_to = norm_to
        self.brightness = brightness
        self.preview_resolution = tuple(
            min(p, f) for p, f in zip(preview_resolution, camera.resolution)
        )
        self._renderer_options = dict(
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
            constraint_monitor=False,
            precision=precision,
            device=device,
        )
        self.renderer, self.preview_renderer = self._make_renderers(spacetime, disk, camera)
        if self.exposure is None and self.renderer.__class__.__name__ == "SceneRenderer":
            self.exposure = 5000

        self._app, scene = _load_vispy(backend)
        try:
            self.canvas = scene.SceneCanvas(
                title="GRAVTRACER interactive observer",
                keys="interactive",
                bgcolor="black",
                size=window_size,
                show=False,
            )
        except Exception as exc:  # pragma: no cover - GUI environment
            raise RuntimeError(
                "VisPy could not create a window; install or select a working "
                "GUI backend (gravtracer[viewer] includes PySide6)"
            ) from exc
        self._view = self.canvas.central_widget.add_view()
        self._view.camera = scene.PanZoomCamera(aspect=1, interactive=False)
        nx, ny = camera.resolution
        initial = np.zeros((ny, nx, 3), dtype=np.float32)
        self._visual = scene.visuals.Image(
            initial, interpolation="bilinear", parent=self._view.scene
        )
        from vispy.visuals.transforms import STTransform

        self._visual.transform = STTransform()

        self._drag_pos = None
        self._last_image: Image | None = None
        self._last_frame = initial
        self._last_render_camera = None
        self._preview_timer = self._app.Timer(
            interval=1 / 30, connect=self._on_preview_timer, start=False
        )
        self._refine_timer = self._app.Timer(
            interval=0.15, connect=self._on_refine_timer, start=False
        )
        self.canvas.events.mouse_press.connect(self._on_mouse_press)
        self.canvas.events.mouse_move.connect(self._on_mouse_move)
        self.canvas.events.mouse_release.connect(self._on_mouse_release)
        self.canvas.events.mouse_wheel.connect(self._on_mouse_wheel)
        self.canvas.events.key_press.connect(self._on_key_press)
        self.canvas.events.close.connect(self._on_close)

        if self._app.use_app().backend_name.lower() == "pyside6":
            from .viewer_controls import ViewerWindow

            self._window = ViewerWindow(self)
            self._settings = self._window.controls
        self.render(refine=True)

    def _make_renderers(self, spacetime, disk, camera):
        from .gpu import Renderer

        options = dict(self._renderer_options)
        if (spacetime.mid not in (1, 2) or self.surface is not None or
                (disk is not None and not isinstance(disk, ThinDisk))):
            from .desktop_scene import SceneRenderer

            Renderer = SceneRenderer
            options.update(surface=self.surface, escape_radius=self.escape_radius)
        elif self.escape_radius is not None:
            options["escape_radius"] = self.escape_radius
        renderer = Renderer(spacetime, disk, camera.resolution, **options)
        preview = (renderer if self.preview_resolution == camera.resolution else
                   Renderer(spacetime, disk, self.preview_resolution, **options))
        return renderer, preview

    def update_parameters(self, *, spin=None, q=None, disk_on=None, r_out=None,
                          l0=None, radius=None, theta=None, phi=None, fov_x=None,
                          fov_y=None, background=None, sky_path=None,
                          sky_longitude=None, sky_gain=None, brightness=None):
        """Apply native settings atomically; display-only changes reuse ray maps."""
        from pathlib import Path
        from .emission import PageThorneDisk
        from .spacetime import BlackHole, QMetric
        from .sky import CelestialSky

        numeric = (spin, q, r_out, l0, radius, theta, phi, fov_x, fov_y,
                   sky_longitude, sky_gain, brightness)
        if any(value is not None and not np.isfinite(value) for value in numeric):
            raise ValueError("Parameters must be finite")
        spacetime = self.spacetime
        if spin is not None:
            if not isinstance(spacetime, BlackHole) or abs(spin) >= 1:
                raise ValueError("Spin must be between -1 and 1 (nonextremal Kerr)")
            spacetime = BlackHole(spin)
        if q is not None:
            if not isinstance(spacetime, QMetric):
                raise ValueError("Quadrupole is only available for the q-metric")
            spacetime = QMetric(q)
        geometry_changed = spacetime != self.spacetime
        template = self.disk or self._disk_template
        supported = isinstance(spacetime, BlackHole) and (
            template is None or isinstance(template, (ThinDisk, PageThorneDisk)))
        if not supported and (disk_on is not None or r_out is not None or l0 is not None
                              or (geometry_changed and template is not None)):
            raise ValueError("This custom disk keeps its original physical parameters")
        enabled = self.disk is not None if disk_on is None else disk_on
        disk = self.disk
        if supported:
            if template is None and enabled:
                template = ThinDisk()
            if isinstance(template, PageThorneDisk):
                outer = template.r_out if r_out is None else r_out
                if geometry_changed or outer != template.r_out:
                    template = PageThorneDisk(spacetime, r_out=outer,
                                              n=template.provenance["flux_samples"])
            elif isinstance(template, ThinDisk):
                template = dataclasses.replace(template,
                    r_out=template.r_out if r_out is None else r_out,
                    l0=template.l0 if l0 is None else l0)
                inner = template.r_in if template.r_in is not None else spacetime.isco
                if template.r_out <= inner:
                    raise ValueError("Disk outer radius must exceed its inner edge / ISCO")
            disk = template if enabled else None
        changed = geometry_changed or disk != self.disk
        camera = self.state.camera
        values = {name: value for name, value in
                  (("r", radius), ("theta", theta), ("phi", phi)) if value is not None}
        for axis, half in (("x", fov_x), ("y", fov_y)):
            if half is not None:
                if not 0.25 <= half <= 1000:
                    raise ValueError("View half-size must be between 0.25 and 1000 M")
                center = sum(getattr(camera, axis)) / 2
                values[axis] = (center - half, center + half)
        camera = dataclasses.replace(camera, **values)
        if not _MIN_THETA <= camera.theta <= 180 - _MIN_THETA:
            raise ValueError("Inclination must be between 3.2 and 176.8 degrees")
        if camera.r <= spacetime.capture_radius:
            raise ValueError("Observer must be outside the absorbing boundary")
        if self.escape_radius is not None and camera.r >= self.escape_radius:
            raise ValueError("Observer must be inside the escape sphere")
        sky = copy.copy(self.sky)
        if sky_path is not None:
            sky = (CelestialSky(sky_path, name=Path(sky_path).name) if sky_path
                   else CelestialSky.nasa_starmap())
        background = self.background if background is None else background
        longitude = (getattr(sky, "longitude", self._sky_longitude)
                     if sky_longitude is None else sky_longitude)
        gain = getattr(sky, "gain", self._sky_gain) if sky_gain is None else sky_gain
        if gain <= 0:
            raise ValueError("Background brightness must be positive")
        if sky is None and background == "celestial":
            sky = CelestialSky.nasa_starmap(longitude=longitude, gain=gain)
        if sky is not None:
            sky.longitude, sky.gain = longitude, gain
        brightness = self.brightness if brightness is None else brightness
        renderers = self._make_renderers(spacetime, disk, camera) if changed else (
            self.renderer, self.preview_renderer)
        started = time.perf_counter()
        image = (self._last_image if not changed and camera == self._last_render_camera
                 else renderers[0].render(camera))
        norm = self.norm_to
        if self._auto_norm and changed and self.exposure is None:
            norm = float(image.intensity.max()) or 1.0
        frame = compose_frame(image, self.mode, background=background, sky=sky,
                              exposure=self.exposure, norm_to=norm, brightness=brightness)
        # All validation, tracing, and file loading complete before changing state.
        self._preview_timer.stop()
        self._refine_timer.stop()
        self.spacetime, self.disk, self._disk_template = spacetime, disk, template
        self.state.camera = camera
        self.renderer, self.preview_renderer = renderers
        self.sky, self.background = sky, background
        self._sky_longitude, self._sky_gain = longitude, gain
        self.norm_to, self.brightness = norm, brightness
        self._present(image, camera, frame, "settings", started)
        return image

    def _title(self, elapsed: float, quality: str):
        cam = self.state.camera
        parameter = (
            f"a={self.spacetime.a:g}" if hasattr(self.spacetime, "a")
            else f"q={self.spacetime.q:g}" if hasattr(self.spacetime, "q") else ""
        )
        ny, nx = self._last_frame.shape[:2]
        self.canvas.title = (
            f"GRAVTRACER | {parameter} theta={cam.theta:.1f} deg  phi={cam.phi:.1f} deg  "
            f"{quality} {nx}x{ny} {1000.0 * elapsed:.1f} ms | {self.renderer.device_name} | "
            f"brightness={self.brightness:g} [+/-] | "
            f"background={self.background} [B]"
        )
        if self._window is not None:
            self._window.setWindowTitle(self.canvas.title)
        if self._settings is not None:
            self._settings.sync()

    def render(self, refine: bool = True) -> Image:
        """Render and display the current camera; return its map bundle."""
        renderer = self.renderer if refine else self.preview_renderer
        # A full-quality cached image remains useful during azimuth-only motion.
        # Do not downgrade it to a preview just because a drag is in progress.
        previous = self._last_render_camera
        current = self.state.camera
        if (previous is not None and previous.resolution == self.renderer.resolution
                and self._last_image.meta.get("backend") == "gpu"):
            if dataclasses.replace(previous, phi=current.phi) == current:
                renderer = self.renderer
        camera = self.state.at_resolution(renderer.resolution)
        t0 = time.perf_counter()
        image = renderer.render(camera)
        if self.norm_to is None and self.exposure is None:
            # Calibrate once from the initial refined image. Resampling a narrow
            # Doppler peak must not change brightness on every preview frame.
            self.norm_to = float(image.intensity.max()) or 1.0
        frame = self._compose(image)
        quality = "cached" if image.meta.get("azimuth_reused") else (
            "final" if renderer is self.renderer else "preview"
        )
        self._present(image, camera, frame, quality, t0)
        return image

    def _present(self, image, camera, frame, quality, started):
        self._visual.set_data(frame)
        ny, nx = frame.shape[:2]
        # Image-plane units preserve the projection through resolution changes.
        self._visual.transform.scale = (
            (camera.x[1] - camera.x[0]) / nx,
            (camera.y[1] - camera.y[0]) / ny,
        )
        self._visual.transform.translate = (camera.x[0], camera.y[0])
        self._view.camera.set_range(x=camera.x, y=camera.y, margin=0)
        self._last_image = image
        self._last_frame = frame
        self._last_render_camera = camera
        self._title(time.perf_counter() - started, quality)
        self.canvas.update()

    def _compose(self, image):
        if self.background == "celestial" and self.sky is None:
            from .sky import CelestialSky

            self.sky = CelestialSky.nasa_starmap(
                longitude=self._sky_longitude, gain=self._sky_gain)
        return compose_frame(
            image,
            self.mode,
            background=self.background,
            sky=self.sky,
            exposure=self.exposure,
            norm_to=self.norm_to,
            brightness=self.brightness,
        )

    def cycle_background(self):
        """Repaint the cached rays; changing a background never retraces them."""
        self.background = _BACKGROUNDS[(_BACKGROUNDS.index(self.background) + 1) % 3]
        if self._last_image is not None:
            started = time.perf_counter()
            self._last_frame = self._compose(self._last_image)
            self._visual.set_data(self._last_frame)
            self._title(time.perf_counter() - started, "background")
            self.canvas.update()
        return self.background

    def _on_mouse_press(self, event):
        if event.button == 1:
            self._drag_pos = np.asarray(event.pos, dtype=float)
            event.handled = True

    def _on_mouse_move(self, event):
        if self._drag_pos is None:
            return
        pos = np.asarray(event.pos, dtype=float)
        delta = pos - self._drag_pos
        self._drag_pos = pos
        if np.any(delta):
            self.state.orbit(delta[0], delta[1])
            self._schedule_preview()
            self._refine_timer.stop()
            self._refine_timer.start()
        event.handled = True

    def _on_mouse_release(self, event):
        if event.button == 1 and self._drag_pos is not None:
            self._drag_pos = None
            self._preview_timer.stop()
            self._refine_timer.stop()
            self.render(refine=True)
            event.handled = True

    def _on_mouse_wheel(self, event):
        delta = event.delta
        steps = float(delta[1] if hasattr(delta, "__len__") else delta)
        self.state.zoom(steps)
        self._schedule_preview()
        self._refine_timer.stop()
        self._refine_timer.start()
        event.handled = True

    def _schedule_preview(self):
        # Keep only the most recent camera state in each 30 Hz event batch.
        # Rendering every queued mouse event makes the window lag behind input.
        if not self._preview_timer.running:
            self._preview_timer.start()

    def _on_preview_timer(self, _event):
        self._preview_timer.stop()
        self.render(refine=False)

    def _on_close(self, _event):
        self._preview_timer.stop()
        self._refine_timer.stop()
        if self._window is not None and not self._window.closing:
            self._window.close()

    def _on_refine_timer(self, _event):
        self._preview_timer.stop()
        self._refine_timer.stop()
        self.render(refine=True)

    def _on_key_press(self, event):
        # VisPy Key.__str__ returns "<Key 'B'>", not "B". Use the key's
        # name so real backend events and plain-string callers agree.
        key = getattr(event.key, "name", str(event.key)).upper()
        if key == "H" and self._window is not None:
            self._window.toggle.trigger()
        elif key == "R":
            self.state.reset()
            self.render(refine=True)
        elif key == "M":
            self.mode = _MODES[(_MODES.index(self.mode) + 1) % len(_MODES)]
            self.render(refine=True)
        elif key == "B":
            self.cycle_background()
        elif key in ("+", "=", "PLUS", "-", "MINUS"):
            factor = 0.8 if key in ("-", "MINUS") else 1.25
            self.brightness = float(np.clip(self.brightness * factor, 1 / 32, 32))
            if self._last_image is not None:
                started = time.perf_counter()
                self._last_frame = self._compose(self._last_image)
                self._visual.set_data(self._last_frame)
                self._title(time.perf_counter() - started, "brightness")
                self.canvas.update()
        elif key in ("SPACE", "ENTER", "RETURN"):
            self.render(refine=True)
        elif key == "S":
            self.save()
        elif key in ("ESCAPE", "ESC"):
            self.canvas.close()
        else:
            return
        event.handled = True

    def save(self, path: str | None = None) -> str:
        """Save the currently displayed RGB frame and return its path."""
        if path is None:
            path = time.strftime("gravtracer_view_%Y%m%d_%H%M%S.png")
        import matplotlib.image as mpimg

        mpimg.imsave(path, self._last_frame, origin="lower")
        self.canvas.title = f"saved {path}"
        return path

    def show(self):
        native = self._window if self._window is not None else self.canvas.native
        native.show()
        if hasattr(native, "raise_"):
            native.raise_()
            native.activateWindow()
        return self

    def run(self):
        self.show()
        self._app.run()
        return self


def view(
    spacetime: Spacetime,
    disk: ThinDisk | None = None,
    camera: Camera | None = None,
    *,
    run: bool = True,
    **kwargs,
) -> InteractiveViewer:
    """Open an interactive GR camera view.

    Set ``run=False`` to integrate the returned viewer into an existing GUI
    event loop.  See :class:`InteractiveViewer` for controls and options.
    """
    viewer = InteractiveViewer(spacetime, disk, camera, **kwargs)
    return viewer.run() if run else viewer.show()


__all__ = ["CameraState", "InteractiveViewer", "compose_frame", "view"]

"""Interactive, GPU-rendered observer view built on VisPy.

VisPy and a GUI backend are optional dependencies.  They are imported only
when :func:`view` or :class:`InteractiveViewer` is used, so importing
``grayt`` remains lightweight and headless-safe.
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field

import numpy as np

from .instruments import Camera
from .matter import ThinDisk
from .results import Image
from .spacetime import Spacetime

_MODES = ("composite", "intensity", "lensing", "shadow")
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
        theta = np.clip(self.camera.theta + sensitivity*dy,
                        _MIN_THETA, 180.0 - _MIN_THETA)
        phi = (self.camera.phi - sensitivity*dx) % 360.0
        self.camera = dataclasses.replace(
            self.camera, theta=float(theta), phi=float(phi))
        return self.camera

    def zoom(self, steps: float, sensitivity: float = 0.15) -> Camera:
        """Scale both image-plane axes around their current centers."""
        factor = float(np.exp(-sensitivity*steps))

        def scaled(bounds):
            center = 0.5*(bounds[0] + bounds[1])
            half = np.clip(0.5*(bounds[1] - bounds[0])*factor, 0.25, 1000.0)
            return (float(center - half), float(center + half))

        self.camera = dataclasses.replace(
            self.camera, x=scaled(self.camera.x), y=scaled(self.camera.y))
        return self.camera

    def reset(self) -> Camera:
        self.camera = self._initial
        return self.camera

    def at_resolution(self, resolution: tuple[int, int]) -> Camera:
        return dataclasses.replace(self.camera, resolution=tuple(resolution))


def _hot(values: np.ndarray) -> np.ndarray:
    """Small dependency-free approximation to the ``afmhot`` color map."""
    return np.stack((np.clip(3.0*values, 0.0, 1.0),
                     np.clip(3.0*values - 1.0, 0.0, 1.0),
                     np.clip(3.0*values - 2.0, 0.0, 1.0)), axis=-1)


def compose_frame(image: Image, mode: str = "composite") -> np.ndarray:
    """Turn ray-tracing maps into an RGB ``(ny, nx, 3)`` display texture.

    ``composite`` overlays Page--Thorne disk emission on a colored celestial
    grid. ``lensing`` shows only that grid, ``intensity`` only disk emission,
    and ``shadow`` gives a monochrome escaped/captured classification.
    """
    if mode not in _MODES:
        raise ValueError(f"unknown viewer mode {mode!r}; use {', '.join(_MODES)}")

    status = image.status
    escaped = status == 0
    disk = status == 2
    rgb = np.zeros(status.shape + (3,), dtype=np.float32)

    if mode in ("composite", "lensing"):
        theta = np.arccos(np.clip(np.cos(image.theta_inf), -1.0, 1.0))
        phi = np.mod(image.phi_inf, 2.0*np.pi)
        top = theta < np.pi/2.0
        east = phi < np.pi
        colors = np.array(((0.05, 0.16, 0.08), (0.22, 0.06, 0.06),
                           (0.05, 0.08, 0.22), (0.20, 0.16, 0.04)),
                          dtype=np.float32)
        rgb[escaped & top & east] = colors[0]
        rgb[escaped & top & ~east] = colors[1]
        rgb[escaped & ~top & east] = colors[2]
        rgb[escaped & ~top & ~east] = colors[3]

        # Fifteen-degree latitude/longitude lines make gravitational
        # deflection visible while keeping the texture cheap to regenerate.
        grid = escaped & ((np.abs(np.sin(12.0*theta)) < 0.045) |
                          (np.abs(np.sin(12.0*phi)) < 0.045))
        rgb[grid] = (0.72, 0.72, 0.72)
    elif mode == "shadow":
        rgb[escaped] = (0.82, 0.82, 0.82)

    if mode in ("composite", "intensity") and np.any(disk):
        values = np.maximum(image.intensity, 0.0)
        positive = values[disk & (values > 0.0)]
        scale = float(np.quantile(positive, 0.995)) if positive.size else 1.0
        normalized = np.clip(values/(scale or 1.0), 0.0, 1.0)
        rgb[disk] = _hot(normalized[disk])

    # GRAVTRACER maps are (nx, ny); image textures are rows then columns.
    return np.ascontiguousarray(rgb.transpose(1, 0, 2))


def _load_vispy(backend: str | None):
    try:
        from vispy import app, scene
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "the interactive viewer needs VisPy and PySide6; install "
            "gravtracer[viewer]") from exc
    try:
        app.use_app(backend) if backend else app.use_app()
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "VisPy could not start a GUI backend; install "
            "gravtracer[viewer] (VisPy + PySide6)") from exc
    return app, scene


class InteractiveViewer:
    """Desktop observer view with live GPU re-rendering.

    Drag with the left mouse button to orbit in ``(theta, phi)`` and use the
    wheel to zoom the image plane.  Dragging and wheel bursts use the preview
    renderer; release/idle refines at the final resolution.  Keys: ``R``
    reset, ``M`` cycle display mode, ``Space`` refine, ``S`` save a PNG,
    and ``Escape`` close.
    """

    def __init__(self, spacetime: Spacetime, disk: ThinDisk | None = None,
                 camera: Camera | None = None,
                 preview_resolution: tuple[int, int] = (128, 64),
                 mode: str = "intensity", precision: str = "auto",
                 device: int | None = None, rtol: float = 1e-5,
                 atol: float = 1e-7, max_steps: int = 500_000,
                 backend: str | None = None,
                 window_size: tuple[int, int] = (1100, 650)):
        if mode not in _MODES:
            raise ValueError(
                f"unknown viewer mode {mode!r}; use {', '.join(_MODES)}")
        if camera is None:
            camera = Camera()

        from .gpu import Renderer

        self.spacetime = spacetime
        self.disk = disk
        self.state = CameraState(camera)
        self.mode = mode
        self.preview_resolution = tuple(preview_resolution)
        renderer_args = dict(
            rtol=rtol, atol=atol, max_steps=max_steps,
            constraint_monitor=False, precision=precision, device=device)
        self.renderer = Renderer(
            spacetime, disk, camera.resolution, **renderer_args)
        self.preview_renderer = Renderer(
            spacetime, disk, self.preview_resolution, **renderer_args)

        self._app, scene = _load_vispy(backend)
        try:
            self.canvas = scene.SceneCanvas(
                title="GRAVTRACER interactive observer", keys="interactive",
                bgcolor="black", size=window_size, show=False)
        except Exception as exc:  # pragma: no cover - GUI environment
            raise RuntimeError(
                "VisPy could not create a window; install or select a working "
                "GUI backend (gravtracer[viewer] includes PySide6)") from exc
        self._view = self.canvas.central_widget.add_view()
        self._view.camera = scene.PanZoomCamera(aspect=1, interactive=False)
        nx, ny = camera.resolution
        initial = np.zeros((ny, nx, 3), dtype=np.float32)
        self._visual = scene.visuals.Image(
            initial, interpolation="bilinear", parent=self._view.scene)
        self._view.camera.set_range(x=(0, nx), y=(0, ny), margin=0)

        self._drag_pos = None
        self._last_image: Image | None = None
        self._last_frame = initial
        self._refine_timer = self._app.Timer(
            interval=0.15, connect=self._on_refine_timer, start=False)
        self.canvas.events.mouse_press.connect(self._on_mouse_press)
        self.canvas.events.mouse_move.connect(self._on_mouse_move)
        self.canvas.events.mouse_release.connect(self._on_mouse_release)
        self.canvas.events.mouse_wheel.connect(self._on_mouse_wheel)
        self.canvas.events.key_press.connect(self._on_key_press)

        self.render(refine=True)

    def _title(self, elapsed: float, quality: str):
        cam = self.state.camera
        self.canvas.title = (
            f"GRAVTRACER | theta={cam.theta:.1f} deg  phi={cam.phi:.1f} deg  "
            f"{quality} {1000.0*elapsed:.1f} ms | mode={self.mode}")

    def render(self, refine: bool = True) -> Image:
        """Render and display the current camera; return its map bundle."""
        renderer = self.renderer if refine else self.preview_renderer
        camera = self.state.at_resolution(renderer.resolution)
        t0 = time.perf_counter()
        image = renderer.render(camera)
        frame = compose_frame(image, self.mode)
        self._visual.set_data(frame)
        ny, nx = frame.shape[:2]
        self._view.camera.set_range(x=(0, nx), y=(0, ny), margin=0)
        self._last_image = image
        self._last_frame = frame
        self._title(time.perf_counter() - t0, "final" if refine else "preview")
        self.canvas.update()
        return image

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
            self.render(refine=False)
        event.handled = True

    def _on_mouse_release(self, event):
        if event.button == 1 and self._drag_pos is not None:
            self._drag_pos = None
            self.render(refine=True)
            event.handled = True

    def _on_mouse_wheel(self, event):
        delta = event.delta
        steps = float(delta[1] if hasattr(delta, "__len__") else delta)
        self.state.zoom(steps)
        self.render(refine=False)
        self._refine_timer.stop()
        self._refine_timer.start()
        event.handled = True

    def _on_refine_timer(self, _event):
        self._refine_timer.stop()
        self.render(refine=True)

    def _on_key_press(self, event):
        key = str(event.key).upper()
        if key == "R":
            self.state.reset()
            self.render(refine=True)
        elif key == "M":
            self.mode = _MODES[(_MODES.index(self.mode) + 1) % len(_MODES)]
            self.render(refine=True)
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
        self.canvas.show()
        return self

    def run(self):
        self.show()
        self._app.run()
        return self


def view(spacetime: Spacetime, disk: ThinDisk | None = None,
         camera: Camera | None = None, *, run: bool = True,
         **kwargs) -> InteractiveViewer:
    """Open an interactive GR camera view.

    Set ``run=False`` to integrate the returned viewer into an existing GUI
    event loop.  See :class:`InteractiveViewer` for controls and options.
    """
    viewer = InteractiveViewer(spacetime, disk, camera, **kwargs)
    return viewer.run() if run else viewer.show()


__all__ = ["CameraState", "InteractiveViewer", "compose_frame", "view"]

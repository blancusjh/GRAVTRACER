"""Instruments: things that measure light — cameras (backward ray
sources) and screens (forward-projection detectors)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import PlanarSurface


@dataclass(frozen=True)
class Camera:
    """Observer image plane: the source of (backward-traced) rays.

    Parameters
    ----------
    r, theta, phi : observer position in Boyer-Lindquist-like
        coordinates. **theta and phi are in DEGREES** (a value below
        ~3.2 is almost certainly radians by mistake and is rejected).
    x, y : image-plane extents in units of M (paper convention: the
        Doppler-approaching side of a prograde disk appears at x < 0).
    resolution : (nx, ny) pixels.
    """

    r: float = 1000.0
    theta: float = 85.0
    phi: float = 0.0
    x: tuple[float, float] = (-24.0, 24.0)
    y: tuple[float, float] = (-12.0, 12.0)
    resolution: tuple[int, int] = (512, 256)

    def __post_init__(self):
        nx, ny = self.resolution
        if not (isinstance(nx, (int, np.integer)) and
                isinstance(ny, (int, np.integer)) and nx > 0 and ny > 0):
            raise ValueError(
                f"resolution must be positive integers, got {self.resolution}")
        if self.x[0] >= self.x[1] or self.y[0] >= self.y[1]:
            raise ValueError("image-plane ranges must be increasing")
        if self.r <= 0:
            raise ValueError("observer radius must be positive")
        if 0.0 < self.theta < 3.2 and self.theta != round(self.theta):
            raise ValueError(
                f"theta = {self.theta} looks like radians; Camera.theta "
                "is in degrees (e.g. theta=85)")

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (self.x[0], self.x[1], self.y[0], self.y[1])


@dataclass
class Screen(PlanarSurface):
    """Detector surface where formed images appear.

    Accumulates ray hits into a pixel buffer; ``image`` averages the
    colors deposited in each bin (black where nothing arrived).
    """

    resolution: tuple = (256, 256)

    def __post_init__(self):
        super().__post_init__()
        nu, nv = self.resolution
        if nu <= 0 or nv <= 0:
            raise ValueError("Screen resolution must be positive")
        self.reset()

    def reset(self):
        nu, nv = self.resolution
        self._rgb = np.zeros((nu, nv, 3))
        self._count = np.zeros((nu, nv), dtype=int)

    def add_hits(self, points_cart, colors):
        """Bin Cartesian hit points into screen pixels."""
        if len(points_cart) == 0:
            return 0
        u, v = self.to_local(points_cart)
        nu, nv = self.resolution
        iu = np.floor((u/self.width + 0.5)*nu).astype(int)
        iv = np.floor((v/self.height + 0.5)*nv).astype(int)
        ok = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv)
        np.add.at(self._rgb, (iu[ok], iv[ok]), colors[ok])
        np.add.at(self._count, (iu[ok], iv[ok]), 1)
        return int(ok.sum())

    @property
    def image(self):
        """(H, W, 3) formed image, rows top-down for imshow."""
        img = np.where(self._count[..., None] > 0,
                       self._rgb/np.maximum(self._count[..., None], 1), 0.0)
        return np.clip(img.transpose(1, 0, 2)[::-1], 0.0, 1.0)

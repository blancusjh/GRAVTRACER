"""Matter and optical objects: things that live in the spacetime and
emit light (the physical content of a ``PhysicalSystem``)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import PlanarSurface


@dataclass(frozen=True)
class ThinDisk:
    """Equatorial thin accretion disk.

    Emission follows the Page & Thorne (1974) time-averaged flux; matter
    moves on circular orbits with constant specific angular momentum
    ``l0``. ``r_in=None`` means the ISCO of the black hole.

    The model (ISCO, Keplerian flux integral, redshift) is defined for
    the Kerr spacetime only; rendering a disk around any other
    spacetime raises at the API layer.
    """

    r_in: float | None = None
    r_out: float = 20.0
    l0: float = 2.8
    model: str = "page-thorne"

    def __post_init__(self):
        if self.model != "page-thorne":
            raise ValueError(f"unknown disk model: {self.model!r}")
        if self.r_out <= 0:
            raise ValueError("r_out must be positive")
        if self.r_in is not None and self.r_in >= self.r_out:
            raise ValueError("r_in must be smaller than r_out")


@dataclass
class ImageSource(PlanarSurface):
    """A loaded image placed on a plane; a source of light.

    ``image`` is an (H, W, 3) float array in [0, 1] or a path to load.
    ``emission``: ``"lambertian"`` (default — pixels emit isotropically,
    photographed by backward tracing) or ``"collimated"`` (one forward
    ray per pixel along the normal, projected onto a Screen).
    """

    image: object = None
    emission: str = "lambertian"

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.image, (str, bytes)):
            import matplotlib.image as mpimg
            arr = mpimg.imread(self.image)
            if arr.dtype.kind in "ui":
                arr = arr/255.0
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=-1)
            self.image = np.asarray(arr[..., :3], float)
        elif self.image is None:
            raise ValueError("ImageSource requires an image array or path")
        if self.emission not in ("lambertian", "collimated"):
            raise ValueError(f"unknown emission model {self.emission!r}; "
                             "use 'lambertian' or 'collimated'")

    def sample_rays(self, max_rays=40_000):
        """Subsample pixels -> (positions (N,3), colors (N,3), uv).

        Pixel row 0 is the top of the image; it maps to +e2 (``up``).
        """
        h, w = self.image.shape[:2]
        stride = max(1, int(np.ceil(np.sqrt(h*w/max_rays))))
        ii, jj = np.meshgrid(np.arange(0, h, stride),
                             np.arange(0, w, stride), indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()
        u = ((jj + 0.5)/w - 0.5)*self.width
        v = (0.5 - (ii + 0.5)/h)*self.height
        pos = self.to_world(u, v)
        colors = self.image[ii, jj]
        return pos, colors, (u, v)

    def sample_color(self, u, v, background=0.0):
        """Texture lookup at in-plane coordinates (u, v); points off the
        card return the background color. Rows map top-down to +e2."""
        u = np.atleast_1d(u)
        v = np.atleast_1d(v)
        h, w = self.image.shape[:2]
        jj = np.floor((u/self.width + 0.5)*w).astype(int)
        ii = np.floor((0.5 - v/self.height)*h).astype(int)
        ok = (ii >= 0) & (ii < h) & (jj >= 0) & (jj < w)
        out = np.full((len(u), 3), background, dtype=float)
        out[ok] = self.image[ii[ok], jj[ok]]
        return out, ok

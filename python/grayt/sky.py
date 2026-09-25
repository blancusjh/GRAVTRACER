"""Equirectangular celestial maps on the tracer's finite escape sphere.

Maps are display RGB, not calibrated radiance or an astronomical catalogue.
Longitude is phi=0 at column zero, increasing to the right; north is row zero.
"""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np


@dataclass
class CelestialSky:
    image: object
    longitude: float = 0.0
    name: str = "celestial map"
    source: str | None = None

    def __post_init__(self):
        if not np.isfinite(self.longitude):
            raise ValueError("sky longitude must be finite")
        if isinstance(self.image, (str, Path)):
            from PIL import Image

            self.image = np.asarray(Image.open(self.image).convert("RGB"), float) / 255
        self.image = np.array(self.image, dtype=float, copy=True)
        if (
            self.image.ndim != 3
            or self.image.shape[2] != 3
            or min(self.image.shape[:2]) < 2
            or not np.isfinite(self.image).all()
            or np.any(self.image < 0)
            or np.any(self.image > 1)
        ):
            raise ValueError(
                "sky image must be finite display RGB (height,width,3) in [0,1]"
            )
        self.image.setflags(write=False)
        self.digest = hashlib.sha256(self.image.tobytes()).hexdigest()

    def sample(self, theta, phi):
        # Convert via the spherical direction to handle theta outside [0,pi].
        th = np.arccos(np.clip(np.cos(theta), -1, 1))
        ph = np.arctan2(np.sin(theta) * np.sin(phi), np.sin(theta) * np.cos(phi))
        h, w = self.image.shape[:2]
        x = np.mod((ph - np.deg2rad(self.longitude)) / (2 * np.pi), 1) * w - 0.5
        y = np.clip(th / np.pi * h - 0.5, 0, h - 1)
        ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
        sx, sy = (x - ix)[..., None], (y - iy)[..., None]
        top = (1 - sx) * self.image[iy, ix % w] + sx * self.image[iy, (ix + 1) % w]
        bottom = (1 - sx) * self.image[
            np.minimum(iy + 1, h - 1), ix % w
        ] + sx * self.image[np.minimum(iy + 1, h - 1), (ix + 1) % w]
        return (1 - sy) * top + sy * bottom

    def metadata(self):
        result = {
            "name": self.name,
            "sha256": self.digest,
            "longitude_deg": self.longitude,
            "projection": "equirectangular",
            "values": "display RGB; uncalibrated",
        }
        if self.source:
            result["source"] = self.source
        return result

    @classmethod
    def nasa_starmap(cls, longitude=180.0):
        """Bundled NASA SVS Deep Star Maps image, mapped to the escape sphere.

        The catalog-based image is a visual background, not a calibrated
        radiance field. Its celestial coordinates are not aligned to a
        particular black-hole coordinate frame.
        """
        from PIL import Image

        path = Path(__file__).resolve().parent / "assets" / "nasa_starmap_2020_4k.jpg"
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=float) / 255
        # NASA's right ascension increases leftward; grayt's phi increases
        # rightward. Put the source map's central RA=0 at phi=0 as well.
        return cls(
            rgb[:, ::-1],
            longitude=longitude,
            name="NASA SVS Deep Star Maps 2020",
            source="https://svs.gsfc.nasa.gov/4851/",
        )

    @classmethod
    def procedural(cls, width=2048, height=1024, seed=42, stars=9000, grid=False):
        """Deterministic synthetic star field with a tilted diffuse band.

        This is a visualization background, NOT a measured Milky Way map.
        Broad star kernels reduce aliasing; supersample rendered images for
        quantitative magnification measurements of compact sources.
        """
        if min(width, height) < 16 or stars < 0:
            raise ValueError("require sky dimensions >=16 and nonnegative star count")
        rng = np.random.default_rng(seed)
        th = (np.arange(height) + 0.5) * np.pi / height
        ph = (np.arange(width) + 0.5) * 2 * np.pi / width
        mu = np.cos(th)[:, None] * 0.65 + np.sin(th)[:, None] * np.sin(ph)[
            None, :
        ] * np.sqrt(1 - 0.65**2)
        band = np.exp(-((mu / 0.12) ** 2))
        structure = (
            0.65 + 0.2 * np.cos(11 * ph)[None, :] + 0.15 * np.sin(23 * ph + 5 * mu)
        )
        rgb = np.full((height, width, 3), 0.004)
        rgb += (band * structure)[..., None] * np.array([0.12, 0.15, 0.23])
        rows = np.clip(
            (np.arccos(rng.uniform(-1, 1, stars)) / np.pi * height).astype(int),
            0,
            height - 1,
        )
        cols = rng.integers(width, size=stars)
        colors = np.array([[0.55, 0.72, 1.0], [1.0, 0.8, 0.52], [0.9, 0.92, 1.0]])
        light = (
            colors[rng.integers(3, size=stars)]
            * rng.uniform(0.15, 0.95, stars)[:, None]
        )
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                weight = np.exp(-(dx * dx + dy * dy) / 1.5)
                np.add.at(
                    rgb,
                    (np.clip(rows + dy, 0, height - 1), (cols + dx) % width),
                    light * weight,
                )
        if grid:
            lat = np.abs(np.sin(12 * th))[:, None] < 0.025
            lon = np.abs(np.sin(12 * ph))[None, :] < 0.025
            rgb[lat | lon] += np.array([0.05, 0.13, 0.14])
        return cls(np.clip(rgb, 0, 1), name=f"synthetic sky seed={seed}; grid={grid}")


def compose_rgb(
    intensity, status, theta, phi, sky=None, *, exposure=1.0, cmap="afmhot"
):
    """Display-only tone mapping; raw bolometric intensity is never modified."""
    import matplotlib

    if not np.isfinite(exposure) or exposure <= 0:
        raise ValueError("exposure must be positive and finite")
    rgb = np.zeros(status.shape + (3,))
    escaped = status == 0
    if sky is not None:
        rgb[escaped] = sky.sample(theta[escaped], phi[escaped])
    emitting = intensity > 0
    value = -np.expm1(-exposure * intensity[emitting])
    rgb[emitting] = matplotlib.colormaps[cmap](value)[..., :3]
    # Failed rays must remain recognizable rather than silently become a shadow.
    rgb[status == 3] = (1.0, 0.0, 1.0)
    return rgb

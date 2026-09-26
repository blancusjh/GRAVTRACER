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
    gain: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.longitude):
            raise ValueError("sky longitude must be finite")
        if not np.isfinite(self.gain) or self.gain <= 0:
            raise ValueError("sky gain must be positive and finite")
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
        rgb = (1 - sy) * top + sy * bottom
        return rgb if self.gain == 1.0 else np.clip(self.gain * rgb, 0, 1)

    def metadata(self):
        result = {
            "name": self.name,
            "sha256": self.digest,
            "longitude_deg": self.longitude,
            "projection": "equirectangular",
            "values": "display RGB; uncalibrated",
            "display_gain": self.gain,
        }
        if self.source:
            result["source"] = self.source
        return result

    @classmethod
    def nasa_starmap(cls, longitude=180.0, gain=1.0):
        """Bundled NASA SVS Deep Star Maps image, mapped to the escape sphere.

        ``gain`` brightens the display map (clipped at white) so faint,
        lensed stars stay visible next to a bright disk; it is display only.

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
            gain=gain,
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
    intensity, status, theta, phi, sky=None, *, exposure=1.0, cmap="afmhot",
    transmission=None, tone="exp", decades=2.5, opaque_floor=0.0,
):
    """Display-only tone mapping; raw bolometric intensity is never modified.

    ``tone="exp"`` maps 1 - exp(-exposure I) onto the colormap. ``tone="log"``
    maps log10(exposure I) over ``decades`` below white (exposure I = 1),
    which keeps Doppler-beamed and dim sides of a disk readable at once.

    ``opaque_floor`` (needs ``transmission``): emitting gas that blocks a
    fraction 1 - T of the sky is shown at colormap value at least
    opaque_floor * (1 - T). Thermal gas that hides starlight outshines it
    by many decades, so drawing it black (below the tone range) would
    invert the true brightness order; the floor vanishes exactly where the
    gas turns transparent, so it adds no edge.

    With ``transmission`` (per pixel, 0..1) the sky is dimmed by it and the
    emission color is added on top, as for light passing through a
    semi-transparent emitter; without it emission replaces the sky.
    """
    import matplotlib

    if not np.isfinite(exposure) or exposure <= 0:
        raise ValueError("exposure must be positive and finite")
    if tone not in ("exp", "log"):
        raise ValueError("tone must be 'exp' or 'log'")
    if not np.isfinite(decades) or decades <= 0:
        raise ValueError("decades must be positive and finite")
    rgb = np.zeros(status.shape + (3,))
    escaped = status == 0
    if sky is not None:
        rgb[escaped] = sky.sample(theta[escaped], phi[escaped])
    emitting = intensity > 0
    if tone == "exp":
        value = -np.expm1(-exposure * intensity[emitting])
    else:
        value = np.clip(
            1 + np.log10(exposure * intensity[emitting]) / decades, 0, 1)
    if opaque_floor and transmission is not None:
        blocked = 1.0 - np.asarray(transmission, float)[emitting]
        value = np.maximum(value, opaque_floor * blocked)
    color = matplotlib.colormaps[cmap](value)[..., :3]
    if transmission is None:
        rgb[emitting] = color
    else:
        rgb *= np.asarray(transmission, float)[..., None]
        rgb[emitting] = np.clip(rgb[emitting] + color, 0, 1)
    # Failed rays must remain recognizable rather than silently become a shadow.
    rgb[status == 3] = (1.0, 0.0, 1.0)
    return rgb

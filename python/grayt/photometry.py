"""Physical photometry: disk and sky in one calibrated radiance image.

The ray tracer works in geometrized units (G = c = M = 1). This module puts
a scene on an absolute scale so that disk and stars can be compared:

* A black-hole mass and accretion rate fix the physical disk flux,
  F_phys = F_code * Mdot c^2 / (4 pi r_g^2), where F_code is the traced
  bolometric emission normalized to Mdot/(4 pi) = 1 (the Page-Thorne
  convention). Each emitting element radiates a blackbody at the effective
  temperature sigma T^4 = pi I_em (optionally color-corrected by f_col).
  The observer receives B_nu(g T): specific intensity transforms as
  I_nu / nu^3, so a redshift g turns a blackbody at T into one at g T.
* The celestial map is converted to linear radiance and scaled so that its
  all-sky mean matches a V-band surface brightness (default 23 mag/arcsec^2,
  roughly the mean integrated starlight; the map's absolute calibration is
  approximate to a factor of a few).
* Colors come from integrating spectra against the CIE 1931 2-degree
  color-matching functions (multi-lobe fit of Wyman, Sloan & Shirley 2013),
  then converting XYZ to linear sRGB. Planckian chromaticities match the
  CIE locus to ~0.003 in (x, y) above 2000 K and ~0.01 near 1000 K.

Display: one global, monotonic tone curve maps luminance to lightness,
piecewise linear in log10(Y) through (floor, 0), (knee, 0.35),
(white / 10^disk_decades, 0.5) and (white, 1). Brighter is always
brighter; nothing is scaled per object. The segments give the starlight,
the nearly empty range between the brightest stars and the dimmest disk
rim, and the disk's own top decades their own share of lightness, so
faint stars and a disk ~10^14 times brighter per unit solid angle share
one picture, as in astronomical HDR color images.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

H = 6.62607015e-34        # J s
KB = 1.380649e-23         # J / K
C = 2.99792458e8          # m / s
G = 6.67430e-11           # m^3 / (kg s^2)
SIGMA = 5.670374419e-8    # W / (m^2 K^4)
MSUN = 1.98847e30         # kg
L_EDD_PER_MSUN = 1.2566e31  # W, electron scattering, hydrogen

_LAMBDA = np.arange(360.0, 831.0, 1.0)          # nm


def _g(x, mu, s1, s2):
    return np.exp(-0.5 * ((x - mu) / np.where(x < mu, s1, s2)) ** 2)


def cie_xyz_bar(lam_nm):
    """CIE 1931 2-degree color-matching functions, Wyman et al. (2013) fit."""
    x = (1.056 * _g(lam_nm, 599.8, 37.9, 31.0)
         + 0.362 * _g(lam_nm, 442.0, 16.0, 26.7)
         - 0.065 * _g(lam_nm, 501.1, 20.4, 26.2))
    y = (0.821 * _g(lam_nm, 568.8, 46.9, 40.5)
         + 0.286 * _g(lam_nm, 530.9, 16.3, 31.1))
    z = (1.217 * _g(lam_nm, 437.0, 11.8, 36.0)
         + 0.681 * _g(lam_nm, 459.0, 26.0, 13.8))
    return np.stack((x, y, z), axis=-1)


_CMF = cie_xyz_bar(_LAMBDA)
_Y_INTEGRAL = float(np.trapezoid(_CMF[:, 1], _LAMBDA))   # ~106.9 nm

# linear sRGB <-> CIE XYZ (D65)
XYZ_TO_RGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                       [-0.9692660, 1.8760108, 0.0415560],
                       [0.0556434, -0.2040259, 1.0572252]])
RGB_TO_XYZ = np.linalg.inv(XYZ_TO_RGB)


def planck_lambda(lam_nm, temperature):
    """Blackbody B_lambda in W m^-2 sr^-1 nm^-1."""
    lam = np.asarray(lam_nm, float) * 1e-9
    t = np.asarray(temperature, float)[..., None]
    x = H * C / (lam * KB * t)
    with np.errstate(over="ignore"):
        b = 2 * H * C**2 / lam**5 / np.expm1(x)
    return b * 1e-9


_T_GRID = np.logspace(np.log10(200.0), 8.0, 1400)
_XYZ_GRID = np.trapezoid(planck_lambda(_LAMBDA, _T_GRID)[..., None] * _CMF,
                         _LAMBDA, axis=-2)
_LOG_XYZ_GRID = np.log(np.maximum(_XYZ_GRID, 1e-300))


def blackbody_xyz(temperature):
    """CIE XYZ (W m^-2 sr^-1, CMF-weighted) of a blackbody; shape (...,3).

    Tabulated from 200 K to 1e8 K and interpolated in log-log; colder
    bodies are black in the visible, hotter ones follow Rayleigh-Jeans.
    """
    t = np.asarray(temperature, float)
    out = np.zeros(t.shape + (3,))
    ok = t >= _T_GRID[0]
    tt = np.minimum(t[ok], _T_GRID[-1])
    logt = np.log(tt)
    for k in range(3):
        out[ok, k] = np.exp(np.interp(logt, np.log(_T_GRID),
                                      _LOG_XYZ_GRID[:, k]))
    hot = t > _T_GRID[-1]
    out[hot] *= (t[hot] / _T_GRID[-1])[..., None]      # Rayleigh-Jeans
    return out


def srgb_to_linear(v):
    v = np.asarray(v, float)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v):
    v = np.clip(np.asarray(v, float), 0, 1)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055)


def surface_brightness_to_y(mu_v):
    """V-band surface brightness (mag/arcsec^2) -> CIE Y radiance.

    Uses the V zero point F_lambda = 3.63e-11 W m^-2 nm^-1 near 550 nm and
    approximates the spectrum as flat across the Y response.
    """
    arcsec2 = (np.pi / 180 / 3600) ** 2
    i_lambda = 3.63e-11 * 10 ** (-0.4 * mu_v) / arcsec2
    return i_lambda * _Y_INTEGRAL


def kerr_efficiency(spin):
    """Radiative efficiency 1 - E_ISCO of a prograde thin disk."""
    from .spacetime import BlackHole

    r = BlackHole(spin).isco
    e = (r**1.5 - 2 * r**0.5 + spin) / (
        r**0.75 * np.sqrt(r**1.5 - 3 * r**0.5 + 2 * spin))
    return 1.0 - e


@dataclass(frozen=True)
class Photometry:
    """Absolute scale for a traced scene.

    mass_msun, eddington_ratio: black-hole mass and L / L_Edd; the
    accretion rate is Mdot = L / (eta c^2) with the thin-disk efficiency
    eta of ``spin``. f_col: color-correction factor (observed spectrum
    B(f_col T) / f_col^4, same bolometric flux); 1 means a pure blackbody.
    sky_mu_v: all-sky mean V surface brightness of the celestial map.
    """

    mass_msun: float = 1e8
    eddington_ratio: float = 0.1
    spin: float = 0.0
    f_col: float = 1.0
    sky_mu_v: float = 23.0

    def __post_init__(self):
        if not (self.mass_msun > 0 and self.eddington_ratio > 0
                and self.f_col >= 1 and np.isfinite(self.sky_mu_v)):
            raise ValueError("invalid photometric parameters")

    @property
    def mdot(self):
        """Accretion rate in kg/s."""
        luminosity = self.eddington_ratio * L_EDD_PER_MSUN * self.mass_msun
        return luminosity / (kerr_efficiency(self.spin) * C**2)

    @property
    def intensity_scale(self):
        """Code bolometric intensity -> W m^-2 sr^-1."""
        r_g = G * self.mass_msun * MSUN / C**2
        return self.mdot * C**2 / (4 * np.pi * r_g**2)

    def temperature(self, emitted_intensity_code):
        """Effective temperature (K) of an emitter of code intensity I_em."""
        i = np.asarray(emitted_intensity_code, float) * self.intensity_scale
        return (np.pi * np.maximum(i, 0) / SIGMA) ** 0.25

    def observed_xyz(self, g, emitted_intensity_code, weight=1.0):
        """XYZ radiance received from an emitter with redshift factor g."""
        t = self.temperature(emitted_intensity_code) * self.f_col
        xyz = blackbody_xyz(np.asarray(g) * t) / self.f_col**4
        return xyz * np.asarray(weight, float)[..., None]

    def metadata(self):
        return {"mass_msun": self.mass_msun,
                "eddington_ratio": self.eddington_ratio,
                "spin_for_efficiency": self.spin,
                "mdot_kg_s": self.mdot, "f_col": self.f_col,
                "sky_mean_mu_v": self.sky_mu_v,
                "cmf": "CIE 1931 2deg, Wyman-Sloan-Shirley 2013 fit"}


def sky_xyz(sky, theta, phi, mu_v=23.0):
    """Calibrated XYZ radiance of a CelestialSky along (theta, phi).

    The display map is linearized (sRGB transfer undone; its original tone
    compression is not) and scaled so its solid-angle-weighted mean Y equals
    ``surface_brightness_to_y(mu_v)``.
    """
    if getattr(sky, "gain", 1.0) != 1.0:
        raise ValueError("photometric calibration needs the map without "
                         "display gain (CelestialSky gain=1)")
    linear = srgb_to_linear(np.asarray(sky.image, float))
    y_map = linear @ RGB_TO_XYZ[1]
    h = y_map.shape[0]
    weight = np.sin((np.arange(h) + 0.5) * np.pi / h)[:, None]
    mean_y = float(np.sum(y_map * weight) / (np.sum(weight) * y_map.shape[1]))
    scale = surface_brightness_to_y(mu_v) / mean_y
    rgb = srgb_to_linear(sky.sample(theta, phi))
    return (rgb @ RGB_TO_XYZ.T) * scale


def tone_curve(y, floor, knee, white, disk_decades=4.0):
    """Global monotonic lightness in [0, 1], piecewise linear in log10(Y)."""
    logy = np.log10(np.maximum(y, 1e-300))
    xs = [np.log10(floor), np.log10(knee)]
    vs = [0.0, 0.35]
    bottom = np.log10(white) - disk_decades
    if bottom > xs[-1]:
        xs.append(bottom)
        vs.append(0.5)
    xs.append(np.log10(white))
    vs.append(1.0)
    if np.any(np.diff(xs) <= 0):
        raise ValueError("tone levels must increase: floor < knee < white")
    return np.interp(logy, xs, vs)


def compose(xyz, floor, knee, white, disk_decades=4.0):
    """XYZ radiance image -> display sRGB, preserving chromaticity."""
    y = np.maximum(xyz[..., 1], 1e-300)
    lightness = tone_curve(y, floor, knee, white, disk_decades)
    rgb = xyz @ XYZ_TO_RGB.T
    rgb = np.maximum(rgb, 0) / y[..., None]
    # scale the color to the target luminance in linear light, then encode
    target = srgb_to_linear(lightness)
    rgb_lin = rgb * target[..., None]
    peak = np.max(rgb_lin, axis=-1, keepdims=True)
    rgb_lin = np.where(peak > 1, rgb_lin / np.maximum(peak, 1e-300), rgb_lin)
    return linear_to_srgb(rgb_lin)


def photometric_render(image, photometry, sky=None, *, floor=None, knee=None,
                       white=None, disk_decades=4.0):
    """Calibrated XYZ and display RGB for a rendered scene.

    ``image`` must come from ``render_scene`` (it carries the emission
    layers). Defaults: floor = 0.05 x mean sky, knee = 1000 x mean sky
    (bright stars), white = 99.5th percentile of the emission luminance;
    the disk's top ``disk_decades`` get the upper half of the lightness.
    Pass the returned levels back in to keep a movie's exposure fixed.
    Returns (xyz (nx,ny,3), rgb (nx,ny,3), levels dict).
    """
    layers = getattr(image, "layers", None)
    if layers is None:
        raise ValueError("image has no emission layers; render with render_scene")
    shape = image.status.shape
    xyz = np.zeros(shape + (3,))
    emission = np.zeros(shape + (3,))
    for mask, g, source, weight in layers:
        emission[mask] += photometry.observed_xyz(g, source, weight)
    xyz += emission
    sky_mean = surface_brightness_to_y(photometry.sky_mu_v)
    if sky is not None:
        esc = image.status == 0
        transmission = getattr(image, "transmission", None)
        background = sky_xyz(sky, image.theta_inf[esc], image.phi_inf[esc],
                             photometry.sky_mu_v)
        if transmission is not None:
            background *= transmission[esc][:, None]
        xyz[esc] += background
    disk_y = emission[..., 1][emission[..., 1] > 0]
    levels = {
        "floor": floor or 0.05 * sky_mean,
        "knee": knee or 1000.0 * sky_mean,
        "white": white or (float(np.quantile(disk_y, 0.995)) if disk_y.size
                           else 1e6 * sky_mean),
        "disk_decades": disk_decades,
        "sky_mean_Y": sky_mean,
    }
    rgb = compose(xyz, levels["floor"], levels["knee"], levels["white"],
                  disk_decades)
    return xyz, rgb, levels

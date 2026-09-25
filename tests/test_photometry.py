"""Physical photometry and the spreading-disk model."""
import numpy as np
import pytest

import grayt
from grayt import photometry as P


def test_temperature_scale_matches_the_newtonian_thin_disk():
    """Far out, sigma T^4 = 3 G M Mdot / (8 pi R^3) (Shakura & Sunyaev)."""
    bh = grayt.BlackHole(0.0)
    disk = grayt.PageThorneDisk(bh, r_out=3000.0, n=8000)
    ph = P.Photometry(mass_msun=1e8, eddington_ratio=0.1, spin=0.0)
    r = 2500.0
    t = ph.temperature(disk.intensity(np.array([r]), 0, 0))[0]
    m = 1e8 * P.MSUN
    radius = r * P.G * m / P.C**2
    t_ss = (3 * P.G * m * ph.mdot / (8 * np.pi * P.SIGMA * radius**3)
            * (1 - np.sqrt(6 / r))) ** 0.25
    assert t == pytest.approx(t_ss, rel=0.01)   # remaining GR terms ~ M/r


# CIE 1931 Planckian locus reference points
@pytest.mark.parametrize("temperature, xy", [(2000.0, (0.5267, 0.4133)),
                                             (3000.0, (0.4369, 0.4041)),
                                             (6500.0, (0.3135, 0.3236)),
                                             (10000.0, (0.2807, 0.2884))])
def test_planckian_chromaticity(temperature, xy):
    """The analytic CMF fit is good to ~3e-3 in (x, y) above 2000 K."""
    xyz = P.blackbody_xyz(np.array([temperature]))[0]
    assert xyz[:2] / xyz.sum() == pytest.approx(xy, abs=3e-3)


def test_redshift_turns_a_blackbody_at_t_into_one_at_gt():
    ph = P.Photometry()
    source = np.array([1e-3])
    t = ph.temperature(source)
    for g in (0.5, 1.3):
        np.testing.assert_allclose(ph.observed_xyz(np.array([g]), source),
                                   P.blackbody_xyz(g * t), rtol=1e-12)


def test_tone_curve_is_global_and_monotonic():
    y = np.logspace(-12, 12, 2001)
    v = P.tone_curve(y, floor=1e-9, knee=1e-4, white=1e8)
    assert np.all(np.diff(v) >= 0) and v[0] == 0 and v[-1] == 1


def test_spreading_disk_fades_inside_the_computed_domain():
    bh = grayt.BlackHole(0.95)
    ph = P.Photometry(1e8, 0.1, spin=0.95)
    disk = grayt.spreading_disk(bh)
    r = np.linspace(disk.r_in, disk.r_out, 20000)
    tau = disk.tau_perp(r, 0)
    y = P.blackbody_xyz(ph.temperature(disk.source_function(r, 0, 0)))[:, 1]
    y *= -np.expm1(-tau)
    sky = P.surface_brightness_to_y(23.0)
    last_visible = r[np.nonzero(y > 0.01 * sky)[0][-1]]
    assert last_visible < 0.9 * disk.r_out
    assert tau[r < 4 * 8.0].min() > 10          # opaque where it is bright
    # energy not radiated by the thin tail is negligible
    i = disk.disk.intensity(r, 0, 0)
    lost = np.trapezoid(i * r * (tau < 1), r) / np.trapezoid(i * r, r)
    assert lost < 1e-4


def test_render_scene_photometry_is_display_only():
    bh = grayt.BlackHole(0.9)
    cam = grayt.Camera(r=400, theta=70, phi=90, x=(-80, 80), y=(-50, 50),
                       resolution=(40, 25))
    disk = grayt.spreading_disk(bh)
    sky = grayt.CelestialSky.procedural(width=256, height=128, stars=300)
    plain = grayt.render_scene(bh, cam, disk, sky=sky, escape_radius=800)
    phot = grayt.render_scene(bh, cam, disk, sky=sky, escape_radius=800,
                              photometry=P.Photometry(spin=0.9))
    np.testing.assert_array_equal(plain.intensity, phot.intensity)
    assert "photometry" in phot.meta["display"]
    again = grayt.render_scene(bh, cam, disk, sky=sky, escape_radius=800,
                               photometry=P.Photometry(spin=0.9),
                               levels=phot.meta["display"])
    np.testing.assert_allclose(again.rgb, phot.rgb)


def test_scene_white_balance_neutralizes_the_emission():
    """Bradford adaptation maps the adapting white exactly to D65."""
    white = P.blackbody_xyz(np.array([1e5]))[0]
    cat = P.adaptation_matrix(white)
    np.testing.assert_allclose(cat @ (white / white[1]), P.D65, rtol=1e-10)
    rgb = P.XYZ_TO_RGB @ (cat @ white)
    assert np.ptp(rgb / rgb.max()) < 1e-3     # neutral: R = G = B

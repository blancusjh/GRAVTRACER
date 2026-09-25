"""SlabDisk: gray transfer through a see-through equatorial slab."""
import numpy as np
import pytest

import grayt


@pytest.fixture(scope="module")
def setup():
    bh = grayt.BlackHole(0.9)
    disk = grayt.PageThorneDisk(bh, r_out=20.0)
    cam = grayt.Camera(r=100, theta=75, x=(-26, 26), y=(-16, 16),
                       resolution=(48, 30))
    return bh, disk, cam


def test_opaque_limit_reproduces_the_opaque_disk(setup):
    bh, disk, cam = setup
    opaque = grayt.render_scene(bh, cam, disk, escape_radius=200)
    slab = grayt.render_scene(bh, cam, grayt.SlabDisk(disk, 1e6),
                              escape_radius=200)
    hit = opaque.status == 2
    assert hit.any()
    np.testing.assert_allclose(slab.intensity[hit], opaque.intensity[hit],
                               rtol=1e-12)
    assert np.all(slab.intensity[~hit] == 0)
    assert np.all(slab.status[~hit] == opaque.status[~hit])


def test_vacuum_limit_leaves_the_sky_untouched(setup):
    bh, disk, cam = setup
    sky = grayt.CelestialSky.procedural(width=256, height=128, stars=400)
    bare = grayt.render_scene(bh, cam, None, sky=sky, escape_radius=200)
    thin = grayt.render_scene(bh, cam, grayt.SlabDisk(disk, 1e-12,
                                                      conserve_flux=False),
                              sky=sky, escape_radius=200)
    assert np.all(thin.status == bare.status)
    assert thin.intensity.max() < 1e-9 * disk.intensity(np.array([6.0]), 0, 0)[0]
    np.testing.assert_allclose(thin.rgb, bare.rgb, atol=1e-9)


def test_translucent_slab_counts_several_crossings(setup):
    bh, disk, cam = setup
    image = grayt.render_scene(bh, cam, grayt.SlabDisk(disk, 0.3),
                               escape_radius=200)
    counts = image.meta["crossing_counts"]
    assert counts[1] > 0 and counts[2] > 0
    assert image.optical_depth.max() > 0
    assert set(np.unique(image.status)) <= {0, 1}


def test_log_tone_is_display_only(setup):
    bh, disk, cam = setup
    slab = grayt.SlabDisk(disk, 1.0)
    a = grayt.render_scene(bh, cam, slab, exposure=200, escape_radius=200)
    b = grayt.render_scene(bh, cam, slab, exposure=200, tone="log",
                           escape_radius=200)
    np.testing.assert_array_equal(a.intensity, b.intensity)
    assert not np.allclose(a.rgb, b.rgb)


@pytest.mark.parametrize("tau", [1e-4, 0.05, 1.0, 20.0])
def test_flux_conserving_slab_emits_the_disk_flux(setup, tau):
    """Emergent flux per face, 2 pi int S (1 - e^(-tau/mu)) mu dmu, equals
    the opaque disk's pi I_disk for any vertical optical depth."""
    _, disk, _ = setup
    slab = grayt.SlabDisk(disk, tau)
    r = np.array([5.0, 10.0])
    source = slab.source_function(r, 0.0, 0.0)
    mu = np.linspace(1e-9, 1, 400_001)
    flux = 2 * np.pi * source * np.trapezoid(
        -np.expm1(-tau / mu) * mu, mu)
    np.testing.assert_allclose(flux, np.pi * disk.intensity(r, 0, 0),
                               rtol=1e-5)


def test_sky_uses_the_asymptotic_direction():
    """Escape-sphere radius must not change where a ray points."""
    bh = grayt.BlackHole(0.95)
    cam = grayt.Camera(r=100, theta=76, phi=90, x=(-40, 40), y=(-25, 25),
                       resolution=(40, 25))
    near = grayt.render_scene(bh, cam, None, escape_radius=200)
    far = grayt.render_scene(bh, cam, None, escape_radius=20000)

    def unit(img):
        t, p = img.theta_inf, img.phi_inf
        return np.stack((np.sin(t) * np.cos(p), np.sin(t) * np.sin(p),
                         np.cos(t)), axis=-1)

    esc = (near.status == 0) & (far.status == 0)
    cos = np.sum(unit(near)[esc] * unit(far)[esc], axis=-1)
    assert np.degrees(np.arccos(np.clip(cos, -1, 1))).max() < 0.02

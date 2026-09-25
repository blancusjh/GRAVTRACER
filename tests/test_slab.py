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
    thin = grayt.render_scene(bh, cam, grayt.SlabDisk(disk, 1e-12), sky=sky,
                              escape_radius=200)
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

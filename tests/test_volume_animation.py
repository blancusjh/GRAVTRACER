from dataclasses import replace
import shutil
import numpy as np
import pytest
import grayt
from grayt.volume import integrate_gray
from grayt.metrics import spherical_components


def test_gray_transfer_matches_homogeneous_slab():
    # I = (j/alpha)(1-exp(-alpha L)), including foreground ordering.
    j, alpha, length = 2.0, 0.3, 4.0
    for n in (1, 7, 100):
        intensity, tau = integrate_gray(
            np.full(n, j), np.full(n, alpha), np.ones(n), np.full(n, length / n)
        )
        assert intensity == pytest.approx(
            j / alpha * (1 - np.exp(-alpha * length)), rel=1e-13
        )
        assert tau == pytest.approx(alpha * length)
    intensity, _ = integrate_gray([1, 0], [0, 10], [1, 1], [1, 1])
    assert intensity == pytest.approx(1.0)  # absorber behind emitter cannot dim it


def test_gray_vacuum_limit_and_redshift():
    intensity, tau = integrate_gray([2, 2], [0, 0], [0.5, 0.5], [3, 1])
    assert intensity == pytest.approx(0.5**4 * 8)
    assert tau == 0


def test_snapshot_interpolation_and_archive(tmp_path):
    r = np.linspace(6, 12, 8)
    th = np.linspace(0, np.pi, 9)
    ph = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    rr, tt, pp = np.meshgrid(r, th, ph, indexing="ij")
    j = 2 * rr + 0.5 * tt
    a = j * 0 + 0.1
    u = np.zeros(j.shape + (4,))
    u[..., 0] = 1 / np.sqrt(1 - 2 / rr)
    grid = grayt.VolumeGrid(r, th, ph, j, a, u)
    st = grayt.BlackHole(0)
    point = np.array([8.2])
    angle = np.array([1.1])
    jj, aa, uu = grid.sample(st, point, angle, np.array([2 * np.pi - 0.1]), 0)
    assert jj == pytest.approx(2 * point + 0.5 * angle)
    assert aa == pytest.approx(0.1)
    assert uu[:, 0] == pytest.approx(1 / np.sqrt(1 - 2 / point))
    path = tmp_path / "volume.npz"
    grid.save(path)
    restored = grayt.VolumeGrid.load(path)
    assert restored.provenance == grid.provenance


def test_imported_surface_archive(tmp_path):
    r = np.linspace(6, 20, 40)
    disk = grayt.TabulatedDisk(r, r**-3, r**-1.5)
    path = tmp_path / "disk.npz"
    disk.save(path)
    other = grayt.TabulatedDisk.load(path)
    assert other.metadata() == disk.metadata()
    assert other.intensity(np.array([8]), 0, 0) == pytest.approx(np.interp(8, r, r**-3))


def test_volume_ray_in_flat_space(tmp_path):
    flat = grayt.CustomMetric(
        lambda r, t: spherical_components(r, t, np.ones_like(r)),
        r_min=0.01,
        r_max=20,
        inner_radius=0.1,
        resolution=(128, 17),
        name="Minkowski",
    )
    cam = grayt.Camera(r=5, theta=90, x=(-0.1, 0.1), y=(-0.1, 0.1), resolution=(1, 1))

    def sample(st, r, th, ph, t):
        u = np.zeros(r.shape + (4,))
        u[:, 0] = 1
        return np.full_like(r, 2), np.full_like(r, 0.3), u

    volume = grayt.EmittingVolume(1, 3, sample)
    img = grayt.render_volume(flat, cam, volume, max_step=0.005, rtol=1e-10, atol=1e-12)
    expected = 2 / 0.3 * (1 - np.exp(-0.3 * 2))
    assert img.intensity[0, 0] == pytest.approx(expected, rel=0.006)
    assert img.optical_depth[0, 0] == pytest.approx(0.6, rel=0.006)
    path = tmp_path / "result.npz"
    img.save(path)
    assert np.array_equal(grayt.VolumeImage.load(path).optical_depth, img.optical_depth)


def test_observer_path_is_smooth_and_excludes_duplicate():
    camera = grayt.Camera(resolution=(16, 12))
    path = list(grayt.orbit_cameras(camera, frames=12))
    assert len(path) == 12
    assert path[0].phi == 0
    assert path[-1].phi == 330
    assert path[0].theta == 45
    assert path[6].theta == 80


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_video_export_and_manifest(tmp_path):
    cam = grayt.Camera(r=30, theta=60, resolution=(16, 12), x=(-8, 8), y=(-6, 6))
    path = tmp_path / "orbit.mp4"
    grayt.render_movie(
        path, grayt.BlackHole(0), [cam, replace(cam, phi=30)], archive_every=1
    )
    assert path.stat().st_size > 500
    assert path.with_suffix(".json").exists()
    assert (tmp_path / "orbit_0000.npz").exists()

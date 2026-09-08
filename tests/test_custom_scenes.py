"""Physical invariants, imported geometry convergence, and archive round trips."""

from dataclasses import replace
import numpy as np
import pytest
import grayt
from grayt.metrics import spherical_components

CAM = grayt.Camera(r=100, theta=70, x=(-12, 12), y=(-8, 8), resolution=(48, 32))


def test_public_keyword_arguments_preserved():
    r, flux = grayt.flux_profile(black_hole=grayt.BlackHole(0), n=64)
    assert len(r) == 64 and flux.max() > 0
    image = grayt.render_scene(spacetime=grayt.BlackHole(0), camera=CAM)
    assert image.status.shape == CAM.resolution


def test_custom_metric_and_derivatives():
    st = grayt.ReissnerNordstrom(0)
    exact = grayt.BlackHole(0)
    for r, th in [(3.1, 0.4), (8.0, 1.2), (100.0, 2.7)]:
        gu, dr, dt = st.metric_contra(r, th)
        ref, _, _ = exact.metric_contra(r, th)
        assert np.allclose(gu, ref, rtol=2e-6, atol=1e-12)
        eps = 1e-5
        assert np.allclose(
            dr,
            (st.metric_contra(r + eps, th)[0] - st.metric_contra(r - eps, th)[0])
            / (2 * eps),
            rtol=2e-6,
            atol=1e-9,
        )

        assert np.allclose(
            dt,
            (st.metric_contra(r, th + eps)[0] - st.metric_contra(r, th - eps)[0])
            / (2 * eps),
            rtol=2e-6,
            atol=1e-9,
        )


def test_rotating_metric_table_and_ray_agreement():
    from grayt.emission import metric_samples

    exact = grayt.BlackHole(0.7)
    rh = exact.horizon
    table = grayt.CustomMetric(
        lambda r, th: metric_samples(exact, r, th),
        r_min=rh + 1e-4,
        r_max=1000,
        radial_origin=rh,
        inner_radius=rh + 0.01,
        resolution=(768, 129),
        name="sampled Kerr",
    )
    for r, th in [(2.2, 0.5), (6.0, 1.2), (40.0, 2.8)]:
        gd = table.metric_cov(r, th)
        gu, _, _ = table.metric_contra(r, th)
        cov = np.array([[gd[0], gd[1]], [gd[1], gd[4]]])
        contra = np.array([[gu[0], gu[1]], [gu[1], gu[4]]])
        assert np.allclose(cov @ contra, np.eye(2), atol=1e-12)
        assert np.allclose(gd, exact.metric_cov(r, th), rtol=2e-5)
    ref = grayt.trace_bundle(exact, CAM, rtol=1e-10, atol=1e-12)
    out = grayt.trace_bundle(table, CAM, rtol=1e-10, atol=1e-12)
    assert np.array_equal(ref.status, out.status)
    both = out.status == 0
    assert (
        np.quantile(abs(out.endpoint[both, 2:4] - ref.endpoint[both, 2:4]), 0.99) < 2e-3
    )


def test_grid_refinement_reduces_metric_error():
    exact = grayt.BlackHole(0).metric_contra(7.321, 1.123)[0]
    errors = []
    for nr in [64, 128, 256]:
        st = grayt.ReissnerNordstrom(0, resolution=(nr, 17))
        errors.append(np.max(abs(st.metric_contra(7.321, 1.123)[0] - exact)))
    assert errors[-1] < errors[0] / 10


def test_custom_schwarzschild_rays_agree_with_analytic():
    ref = grayt.trace_bundle(grayt.BlackHole(0), CAM, rtol=1e-10, atol=1e-12)
    out = grayt.trace_bundle(grayt.ReissnerNordstrom(0), CAM, rtol=1e-10, atol=1e-12)
    assert np.array_equal(ref.status, out.status)
    ok = out.status == 0
    assert np.max(abs(out.endpoint[ok, 2:4] - ref.endpoint[ok, 2:4])) < 2e-3
    assert np.median(out.herr[ok]) < 1e-7


def test_metric_activation_and_archive(tmp_path):
    first, second = grayt.ReissnerNordstrom(0.2), grayt.ReissnerNordstrom(0.8)
    a = first.metric_cov(5, 1)
    assert not np.allclose(a, second.metric_cov(5, 1))
    assert np.array_equal(a, first.metric_cov(5, 1))
    path = tmp_path / "metric.npz"
    first.save(path)
    restored = grayt.TabulatedMetric.load(path)
    assert restored.digest == first.digest
    assert np.array_equal(a, restored.metric_cov(5, 1))


def test_custom_camera_is_null():
    st = grayt.ReissnerNordstrom(0.7)
    y, pt, pp = grayt.camera_ray(st, CAM, 4.0, 3.0)
    gu, _, _ = st.metric_contra(y[1], y[2])
    h = (
        gu[0] * pt**2
        + 2 * gu[1] * pt * pp
        + gu[2] * y[4] ** 2
        + gu[3] * y[5] ** 2
        + gu[4] * pp**2
    )
    assert abs(h) < 1e-13


def test_stellar_surface_redshift_and_boundary():
    st = grayt.SphericalStar(5)
    source = grayt.EmittingSurface(lambda th, ph, t: np.ones_like(th))
    img = grayt.render_scene(st, CAM, surface=source, rtol=1e-10, atol=1e-12)
    hits = img.status == 1
    assert hits.any()
    assert np.max(abs(img.r_hit[hits] - 5)) < 1e-8
    expected = np.sqrt((1 - 2 / 5) / (1 - 2 / CAM.r))
    assert np.allclose(img.g[hits], expected, rtol=3e-6)
    assert np.allclose(img.intensity[hits], expected**4, rtol=1e-5)


def test_imported_stellar_boundary_retains_emission(tmp_path):
    star = grayt.SphericalStar(5)
    path = tmp_path / "star.npz"
    star.save(path)
    restored = grayt.SphericalStar.load(path)
    assert restored.boundary_kind == "surface"
    surface = grayt.EmittingSurface(lambda th, ph, t: 1.0)
    image = grayt.render_scene(restored, CAM, surface=surface)
    assert image.intensity.max() > 0


def test_emission_changes_intensity_not_paths():
    st = grayt.BlackHole(0.5)
    one = grayt.EmittingDisk(6, 20, lambda r, p, t: np.ones_like(r))
    two = replace(one, intensity=lambda r, p, t: np.full_like(r, 2))
    a = grayt.render_scene(st, CAM, one)
    b = grayt.render_scene(st, CAM, two)
    assert np.array_equal(a.rays.endpoint, b.rays.endpoint)
    assert np.allclose(b.intensity, 2 * a.intensity)
    assert np.allclose(a.g, b.g)


def test_page_thorne_keplerian_motion():
    st = grayt.BlackHole(0.8)
    r = np.array([4.0, 8.0, 15.0])
    th = np.full(3, np.pi / 2)
    u = grayt.circular_velocity(st, r, th, 0)
    assert np.allclose(u[:, 3] / u[:, 0], 1 / (r**1.5 + st.a), rtol=1e-8)
    disk = grayt.PageThorneDisk(st)
    assert disk.intensity(np.array([st.isco]), 0, 0)[0] == 0
    with pytest.raises(ValueError, match="different"):
        grayt.render_scene(grayt.BlackHole(0.4), CAM, disk)


def test_bad_emitter_rejected():
    disk = grayt.EmittingDisk(
        6,
        20,
        lambda r, p, t: r * 0 + 1,
        four_velocity=lambda st, r, th, ph: np.array([1.0, 0, 0, 0]),
    )
    with pytest.raises(ValueError, match="normalized"):
        grayt.render_scene(grayt.BlackHole(0), CAM, disk)


def test_sky_wrapping_and_orientation():
    texture = np.zeros((16, 32, 3))
    texture[:8, :, 0] = 1
    texture[8:, :, 2] = 1
    sky = grayt.CelestialSky(texture)
    assert np.allclose(sky.sample(0.2, 0.7), [1, 0, 0])
    assert np.allclose(sky.sample(3.0, 0.7), [0, 0, 1])
    assert np.allclose(sky.sample(1.0, 1e-10), sky.sample(1.0, 2 * np.pi + 1e-10))
    assert np.allclose(sky.sample(-0.3, 0.5), sky.sample(0.3, 0.5 + np.pi))


def test_scene_archive_and_failure_display(tmp_path):
    img = grayt.render_scene(
        grayt.BlackHole(0),
        CAM,
        sky=grayt.CelestialSky.procedural(128, 64, stars=30),
        max_steps=1,
    )
    assert (img.status == 3).all()
    assert np.allclose(img.rgb, [1, 0, 1])
    path = tmp_path / "image.npz"
    img.save(path)
    restored = grayt.SceneImage.load(path)
    assert np.array_equal(img.rays.endpoint, restored.rays.endpoint)
    assert img.meta == restored.meta


def test_domain_and_backend_rejections():
    st = grayt.ReissnerNordstrom(0.2, r_max=200)
    with pytest.raises(ValueError, match="domain"):
        st.metric_cov(300, 1.0)
    with pytest.raises(ValueError, match="margin"):
        grayt.trace_bundle(st, CAM, escape_radius=195)
    with pytest.raises(ValueError, match="cpu"):
        grayt.render(st, CAM, backend="gpu")
    with pytest.raises(ValueError, match="unregistered"):
        grayt.shadow(grayt.Spacetime(), CAM)


def test_analytic_callback_and_invalid_signature():
    st = grayt.CustomMetric(
        lambda r, t: spherical_components(r, t, 1 - 2 / r),
        r_min=2.0001,
        r_max=1000,
        radial_origin=2,
        inner_radius=2.01,
    )
    assert st.metric_cov(8, 1)[0] == pytest.approx(-0.75, rel=1e-6)
    with pytest.raises(ValueError, match="Lorentzian"):
        grayt.CustomMetric(
            lambda r, t: np.ones(r.shape + (5,)), r_min=1, r_max=100, inner_radius=2
        )

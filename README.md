# kerr-raytracer

Backward ray tracing of null geodesics around Kerr black holes, replicating
the results of *OSIRIS: A New Code for Ray Tracing Around Compact Objects*
(Velásquez-Cadavid et al., arXiv:2202.00086, Eur. Phys. J. C).

**Architecture:** modern Fortran core (Hamiltonian geodesic integration,
adaptive RKDP45/RKCK45/RKF45, OpenMP over pixels) wrapped with f2py and
driven by the Python package **`grayt`** (configuration, API, CLI, plotting).

## Build

Requires `gfortran`, `meson`, `ninja`, numpy ≥ 1.26.

```sh
make            # compiles src/*.f90 -> python/grayt/_core.*.so
make test       # pytest validation suite
```

## Usage

```python
import sys; sys.path.insert(0, "python")   # or add to PYTHONPATH
import grayt

bh   = grayt.BlackHole(a=0.95)                          # spacetime
cam  = grayt.Camera(r=1000, theta=85, x=(-24, 24),      # source of rays
                    y=(-12, 12), resolution=(1024, 512))
disk = grayt.ThinDisk(l0=1.8, r_out=20)                 # matter geometry

img = grayt.render(bh, cam, disk)   # Image: .intensity .g .r_hit .status ...
img.plot(label="$a=0.95$")
img.save("a095.npz")
```

Other entry points: `grayt.shadow(bh, cam)`, `grayt.trace(...)` for single
geodesics with constraint monitoring, `grayt.flux_profile(bh)` for the
Page–Thorne emission profile, `grayt.Scene.from_yaml("configs/fig13_a0.yml")`.

Scene layer (`grayt.system`): `PhysicalSystem` (spacetime + objects +
rays) and `System` (physics + cameras + screens + 3D visualization).
Image formation with a loaded picture:

```python
src = grayt.ImageSource(center=(-150, 0, 0), normal=(1, 0, 0),
                        width=90, height=68, image="picture.jpg")  # lambertian
sys3 = grayt.System(physical=grayt.PhysicalSystem(black_hole=bh,
                                                  sources=[src]))
photo = sys3.photograph(grayt.Camera(x=(-45, 45), y=(-28, 28),
                                     resolution=(900, 560)), src)
photo.plot()
```

Emission models: `"lambertian"` (default — photographed by backward
tracing) and `"collimated"` (forward projection onto a `grayt.Screen`
via `System.form_image`).

CLI:

```sh
PYTHONPATH=python python3 -m grayt render configs/fig13_a095.yml -o a095.png
PYTHONPATH=python python3 -m grayt shadow -a 0.98 -o shadow.png
```

## Validation against the paper

| Check | Result |
|---|---|
| ISCO radii (a = 0, 0.5, 0.95) | 6.000, 4.233, 1.937 (exact) |
| Camera initial conditions | null to ~1e-16 |
| Constraint drift, Figs. 4–5 orbits (rtol 1e-11) | RKDP45 ~1e-10 (best), CK/F45 ~1e-9 |
| Shadow vs analytic Bardeen rim, a = 0.98 (Fig. 6) | within 1 pixel |
| Page–Thorne flux, a = 0 | F(isco) = 0, peak at r = 9.55 |
| Thin-disk images (Fig. 13) | `examples/disk_images.py` |
| Weak-field deflection (b = 50) | 4M/b + 15πM²/4b² to < 2% |

The example scripts are general-purpose (spin, resolution, geometry as
CLI flags); their *defaults* reproduce the paper's figures. All outputs
go to `output/` (git-ignored), keeping code and artifacts separate.

| Script | Defaults reproduce |
|---|---|
| `examples/rays3d.py` | Fig. 1 (3D geodesics) |
| `examples/constraint_drift.py` | Figs. 4–5 |
| `examples/shadow.py` | Fig. 6 |
| `examples/benchmark.py` | Fig. 8 |
| `examples/lensing_sphere.py` | Fig. 12 |
| `examples/disk_images.py` | Fig. 13 (`--res 2048 1024`) |
| `examples/image_formation.py` | forward (collimated) projection demo |
| `examples/photograph.py` | lambertian imaging of a loaded picture |

### Errata found in the paper (as printed)

Documented where implemented in the code:

1. **Eq. (7)**: `P^t = A_t[p_t + (g_tφ/g_φφ)L]` disagrees with the paper's
   own base-change matrix; the term needs a minus sign, otherwise camera
   initial conditions are not null for a ≠ 0 (`src/raytracer.f90`).
2. **Eq. (18)**: the numerator `g_tφ + g_φφ l0` must be `g_tφ + g_tt l0`;
   as printed, Ω → −l0 in the Schwarzschild limit, which is superluminal
   at large r (`src/disk_model.f90`).
3. **Eq. (22)**: the printed g equals ν_em/ν_obs; the redshift factor used
   in I_obs = g³ I_em is its inverse (`src/disk_model.f90`).

## Status

- [x] M0 planning + skeleton & build (f2py/meson, `.f2py_f2cmap`)
- [x] M1 geodesics + integrators (Figs. 4–5 constraint behaviour)
- [x] M2 camera & shadow vs Bardeen (Fig. 6)
- [x] M3 celestial-sphere lensing (Fig. 12)
- [x] M4 thin accretion disk (Fig. 13)
- [x] Scene layer: 3D viewer, image formation (`photograph`/`form_image`),
      `Screen`/`ImageSource`/`PhysicalSystem`/`System`
- [ ] M5 q-metric, time-like geodesics

# GRAVTRACER

Relativistic ray tracing around compact objects: shadows, thin accretion
disks, gravitational lensing, image formation, and orbit visualization
in **Kerr** and **q-metric (Zipoy–Voorhees)** spacetimes.

Born as a replication of *OSIRIS: A New Code for Ray Tracing Around
Compact Objects* (Velásquez-Cadavid et al., arXiv:2202.00086,
Eur. Phys. J. C) — every figure of the paper is reproduced by the
example scripts — and extended into a general laboratory.

**Architecture:** modern Fortran core (Hamiltonian geodesics, one shared
adaptive RKDP45/RKCK45/RKF45 stepper + event-bisection machinery,
OpenMP over rays) wrapped with f2py, driven by the Python package
**`grayt`**.

**Ontology:**

```
Spacetime (BlackHole | QMetric)        the geometry
  └─ PhysicalSystem (+ ThinDisk, ImageSource)     the physics
       └─ System (+ Camera, Screen, rays, experiments)   the laboratory
Results: Image, Photograph, Trajectory, Ray
```

## Install

```sh
uv pip install .                                   # regular install
uv pip install meson-python numpy ninja meson      # then, for editable:
uv pip install -e . --no-build-isolation
uv pip install '.[viewer]'                         # GPU + desktop viewer
```

(`--no-build-isolation` for editable installs is the standard
meson-python requirement; plain `pip install .` works too. Needs
`gfortran`.) This provides the `gravtracer` console command.

Development fallback without pip: `make` compiles the extension in-tree
(`python/grayt/`), then `PYTHONPATH=python`; `make test` runs pytest.

## Usage

```python
import grayt

bh   = grayt.BlackHole(a=0.95)                          # spacetime
cam  = grayt.Camera(r=1000, theta=85, x=(-24, 24),      # theta in DEGREES
                    y=(-12, 12), resolution=(1024, 512))
disk = grayt.ThinDisk(l0=1.8, r_out=20)                 # matter (Kerr-only)

img = grayt.render(bh, cam, disk)   # Image: .intensity .g .r_hit .status ...
img.plot(label="$a=0.95$")

qm = grayt.QMetric(q=1.0)           # naked singularity, ADM mass 1+q
grayt.shadow(qm, cam).plot()
```

The laboratory layer composes multi-instrument scenes:

```python
src = grayt.ImageSource(center=(-150, 0, 0), normal=(1, 0, 0),
                        width=90, image="picture.jpg")   # lambertian
lab = grayt.System(physical=grayt.PhysicalSystem(spacetime=bh,
                                                 sources=[src]))
photo = lab.photograph(grayt.Camera(x=(-45, 45), y=(-28, 28),
                                    resolution=(900, 560)))
photo.plot()
lab.visualize3d()                    # 3D scene with traced rays
```

Emission models: `"lambertian"` (default — photographed by backward
tracing) and `"collimated"` (forward projection onto a `Screen` via
`System.form_image`). Single geodesics: `grayt.trace` (photons or
massive particles via `grayt.orbit_ic`), plotted with
`grayt.plot_orbits_2d`.

CLI (YAML scenes; schema in `System.from_yaml.__doc__`):

```sh
gravtracer render configs/fig13_a095.yml -o a095.png
gravtracer shadow -a 0.98 -o shadow.png
gravtracer view -a 0.95
```

### GPU backend

`render`/`shadow` can run on any OpenCL device — Apple Silicon GPUs
(via Apple's OpenCL-on-Metal) and NVIDIA/AMD/Intel — with one work-item
per pixel (`pip install gravtracer[gpu]`, i.e. pyopencl):

```python
img = grayt.render(bh, cam, disk, backend="gpu")   # ~80x an M4's CPU cores
grayt.gpu.devices()                                # enumerate devices

# Reuse allocations and the disk flux table across changing cameras:
renderer = grayt.gpu.Renderer(bh, disk, cam.resolution)
next_img = renderer.render(cam)
```

Precision follows the hardware: fp64 where supported (NVIDIA/AMD), fp32
on Apple GPUs (no double-precision hardware); on fp32 the tolerances are
clamped to `rtol>=1e-5, atol>=1e-7` and the result is image-quality
(status maps match the CPU reference; hit radii/angles agree to ~1e-3).
`precision="fp32"` also speeds up NVIDIA cards considerably. The Fortran
CPU core (`backend="cpu"`, default) remains the double-precision
reference; the GPU backend implements the `rkdp45` integrator only.

### Interactive viewer

The optional VisPy viewer turns the GPU renderer into a live observer view:

```python
grayt.view(bh, disk, cam)  # install gravtracer[viewer] first
```

Left-drag orbits the observer in inclination/azimuth and the mouse wheel
zooms the image plane. Interaction renders at a preview resolution and
automatically refines on release/idle. The default view shows only the
physical disk intensity. `M` cycles intensity, lensing, shadow, and the
optional composite overlay with its colored celestial grid. `R` resets,
Space refines, `S` saves the current frame, and Escape closes the window.
The CLI exposes resolution, preview resolution, precision, OpenCL device,
disk, and display-mode options through `gravtracer view --help`.

The integrator's maximum step grows conservatively in the weak-curvature far
field (`max(25, min(0.1 r, 100))`); the adaptive error test is unchanged, and
the same rule is used by the Fortran reference and OpenCL kernel.

## Validation

| Check | Result |
|---|---|
| ISCO radii (a = 0, 0.5, 0.95) | 6.000, 4.233, 1.937 (exact) |
| Camera initial conditions | null to ~1e-16 (Kerr and q-metric) |
| Constraint drift, Figs. 4–5 orbits (rtol 1e-11) | RKDP45 ~1e-10 (best) |
| Shadow vs analytic Bardeen rim, a = 0.98 (Fig. 6) | within 1 pixel |
| Page–Thorne flux, a = 0 | F(isco) = 0, peak at r = 9.55 |
| Weak-field deflection (b = 50) | 4M/b + 15πM²/4b² to < 2% |
| q-metric | q = 0 ≡ Schwarzschild to round-off; shadow scales with ADM mass 1+q |

58 tests: `make test` (OpenCL cases skip when no device is available).
Example scripts (outputs go to git-ignored
`output/`); defaults reproduce the paper's figures:

| Script | Defaults reproduce |
|---|---|
| `examples/rays3d.py` | Fig. 1 (3D geodesics) |
| `examples/constraint_drift.py` | Figs. 4–5 |
| `examples/shadow.py` | Fig. 6 |
| `examples/benchmark.py` | Fig. 8 |
| `examples/lensing_sphere.py` | Fig. 12 |
| `examples/disk_images.py` | Fig. 13 (`--res 2048 1024`) |
| `examples/qmetric.py` | Appendix A / Fig. 14 (quadrupole physics) |
| `examples/orbits2d.py` | 2D orbit projections (Figs. 3/14 style) |
| `examples/image_formation.py` | forward (collimated) projection |
| `examples/photograph.py` | lambertian imaging of a loaded picture |

A demo notebook lives at `notebooks/grayt_demo.ipynb`.

### Errata found in the paper (as printed)

1. **Eq. (7)**: the (g_tφ/g_φφ)L term in 𝒫^t needs a minus sign (their
   own base-change matrix has it right); otherwise camera initial
   conditions are not null for a ≠ 0 (`src/raytracer.f90`).
2. **Eq. (18)**: the numerator `g_tφ + g_φφ l0` must be `g_tφ + g_tt l0`;
   as printed, Ω → −l0 in the Schwarzschild limit (`src/disk_model.f90`).
3. **Eq. (22)**: the printed g equals ν_em/ν_obs; the redshift factor in
   I_obs = g³ I_em is its inverse (`src/disk_model.f90`).
4. **Eq. (A.1)**: sign slip in the spatial block of the q-metric; the
   standard Zipoy–Voorhees form is used (`src/q_metric.f90`).

## Review & roadmap

A three-way code review (architecture/physics, CLI user, notebook user)
with the action plan and future-extension roadmap lives in
[docs/REVIEW.md](docs/REVIEW.md).

## Status

- [x] M0–M4: build, geodesics, shadow, lensing, thin disk (paper Figs. 1–13)
- [x] Scene layer: 3D viewer, image formation (photograph/form_image)
- [x] Structural review fixes: Spacetime→PhysicalSystem→System hierarchy,
      shared Fortran adaptive-step/event machinery, input validation
- [x] q-metric spacetime (Appendix A) + Fig. 14-style orbit physics
- [x] Packaging: `uv pip install -e .` (meson-python), `gravtracer` CLI
- [ ] Next: Keplerian/g⁴ disk toggles, physical (redshifted) photograph,
      volumetric radiative transfer — see docs/REVIEW.md

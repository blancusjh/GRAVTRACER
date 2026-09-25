# Stationary metrics, radiation models, and celestial maps

GRAVTRACER separates geometry, emitting matter, observer configuration, and
display. It propagates test photons; it does not solve Einstein's equations,
fluid dynamics, or the formation of an accretion disk.

## Supported scope

The geometric core assumes a **stationary, axisymmetric, circular metric** in
coordinates `(t,r,theta,phi)`, signature `(-+++)`, with only these independent
covariant components:

```text
g_tt, g_tphi, g_rr, g_thetatheta, g_phiphi
```

Components depend on `(r,theta)`. The conserved covariant momenta are `p_t`
and `p_phi`; the shared Fortran Hamiltonian integrator evolves the remaining
six variables. This is not support for arbitrary time-dependent metrics,
nonaxisymmetric geometry, or coordinates with `g_tr`, `g_rtheta`, etc.

| Geometry | Implementation | Radiation options |
|---|---|---|
| Kerr, including Schwarzschild | Analytic Fortran; legacy GPU also available | Legacy disk, new Page–Thorne disk, custom surfaces/volumes |
| Zipoy–Voorhees / q-metric | Analytic Fortran; legacy GPU also available | Custom disk if physically admissible, volume, sky |
| Spherical stellar exterior | Sampled Schwarzschild exterior, surface boundary | Surface emission, external disks/volumes, sky |
| Reissner–Nordström | Sampled analytic charged exterior, nonextremal | Prescribed disks/volumes, sky |
| User callback / imported table | Bicubic Hermite inverse-metric interpolation | Prescribed disks/volumes, sky |

New scene and volume rendering use the **double-precision CPU backend**.
Custom metric tables are read-only during OpenMP tracing. A reentrant Python
lock serializes public calls which access Fortran's global metric/flux tables.
For concurrent independent scenes use separate processes. Direct `_core`
calls bypass that protection and are an internal interface.

Units are `G=c=M=1`; a radius of 5 means `5 GM/c^2`. Camera angles are degrees;
metric callbacks, endpoints, and emission callbacks use radians. In the
q-metric implementation, the parameter mass is 1 and ADM mass is `1+q`;
unrescaled q-metric panels are **not comparisons at equal ADM mass**.

## A physically motivated disk and a celestial sphere

```python
import grayt

bh = grayt.BlackHole(a=0.8)
disk = grayt.PageThorneDisk(bh, r_out=20)
sky = grayt.CelestialSky.nasa_starmap()
camera = grayt.Camera(r=100, theta=70, x=(-26, 26), y=(-16, 16),
                     resolution=(640, 400))
image = grayt.render_scene(bh, camera, disk, sky=sky,
                           escape_radius=200, exposure=5000)
image.plot()
image.save("observation.npz")
```

`PageThorneDisk` uses a zero-torque ISCO Page–Thorne flux profile, circular
geodesic emitter motion, and `I_em=F/pi` for isotropic emission into one
hemisphere. It propagates bolometric intensity with `I_obs=g^4 I_em`.
Accretion normalization is the legacy `Mdot/(4*pi)=1`. Exact extremal spin is
excluded from this model because the ISCO flux formulas require a limiting
treatment.

`ThinDisk` retains the historical constant-specific-angular-momentum emitter
motion and `g^3 F` rendering used to reproduce the original paper. It is not
silently redefined. Both models are Kerr-only. A PageThorneDisk stores its Kerr
parameters and rejects reuse with a different spin.

`CelestialSky.nasa_starmap()` loads the bundled NASA Scientific Visualization
Studio [Deep Star Maps](https://svs.gsfc.nasa.gov/3895/) image, based on star
catalogs. `CelestialSky("map.png")` accepts a user-provided equirectangular RGB map.
The bundled loader reverses NASA's leftward right-ascension axis and centers
RA=0 at `phi=0` for display. This is a coordinate convention, not an
astrometric alignment with the black hole.

For user maps, north is row zero; longitude phi=0 is at the left edge,
increasing rightward.
Sampling is bilinear with periodic longitude. `longitude` rotates the map in
degrees. NASA's map is a display image; its celestial axes have no specified
alignment with a black hole's coordinates. The optional procedural sky remains
a deterministic illustration, **not a measured Milky Way map or catalog**.

Sky maps are sampled at the finite coordinate escape sphere, not the
asymptotic momentum direction. Change `escape_radius` to check this effect.
The RGB values are uncalibrated display colors; no spectral redshift or
absolute luminosity is inferred for them. Disk/surface raw intensities remain
separate. Display uses an explicit fixed exposure and `1-exp(-exposure*I)`;
false color is not an inferred temperature. Failed rays appear magenta.

## Custom analytic geometry

```python
import numpy as np
from grayt.metrics import spherical_components

def metric(r, theta):
    f = 1 - 2/r + 0.4**2/r**2
    return spherical_components(r, theta, f)

rh = 1 + np.sqrt(1 - 0.4**2)
st = grayt.CustomMetric(
    metric, name="charged exterior", radial_origin=rh,
    r_min=rh+1e-4, r_max=3000, inner_radius=rh+0.01,
    resolution=(768, 129),
    provenance={"charge": 0.4, "mass": 1.0})
image = grayt.render_scene(st, camera, sky=sky, escape_radius=200)
st.save("metric.npz")
```

The callback broadcasts over NumPy arrays and returns shape `(...,5)`.
It is evaluated once to build a table; **there is no Python callback inside
the RK loop**. No recompilation is required. Scalar Python loops can be used
inside a callback to adapt a legacy metric provider, at a setup-time cost.

`TabulatedMetric` imports the same covariant data directly. Its radial grid
is uniform in `log(r-radial_origin)` and its angular grid is uniform in
`mu=cos(theta)` from -1 to 1. At the polar endpoints supply the limiting
values evaluated with `theta=acos(clip(mu,-1+1e-10,1-1e-10))`. A numerical
simulation with a different mesh must first be resampled into this explicit
format. No arbitrary AMR, HDF5, or GRMHD-code-specific reader is claimed.
Metric archives load as `TabulatedMetric`, including archives created from
analytic subclasses. An explicit `boundary_kind="surface"` permits emission
at the inner radius; SphericalStar sets and saves this tag automatically.

Internally, the inverse angular components are regularized by `r^2` and
`r^2 sin^2(theta)`. Bicubic Hermite interpolation returns both the inverse
metric and **derivatives of that same interpolant**, preserving the
Hamiltonian relationship. Covariant values are obtained by inversion of the
interpolated inverse metric, keeping the camera tetrad consistent with it.
Table construction uses fourth-order centered slopes in the interior and
second-order edge slopes. This is a numerical approximation to the supplied
metric; it is not itself a newly verified solution of Einstein's equations.

Provide radial margin below the absorbing boundary and above the escape
sphere for RK stages. The API requires at least 5% outer margin. Evaluation
outside the table yields a rejected step, never extrapolated geometry. Rays
that exhaust their step budget are explicitly failed. Coordinate poles and
singular charts remain limitations; traversable wormholes and interior
horizon crossing require different boundary/chart support.

## Prescribed disks and stellar surfaces

```python
disk = grayt.EmittingDisk(
    r_in=6, r_out=20,
    intensity=lambda r, phi, t: 1e-4*(6/r)**3,
    name="power-law emitting annulus")
```

This model is illustrative. `intensity(r,phi,t)` gives the local bolometric
specific intensity. The default emitter velocity is a positive-angular-
velocity circular geodesic calculated from metric derivatives. It checks
that the orbit exists and is timelike; it does not solve disk equilibrium
or establish orbital stability. The user chooses an appropriate inner edge.

For another motion prescription provide
`four_velocity(metric,r,theta,phi) -> (...,4)` in coordinate components.
The renderer checks future direction and `g(u,u)=-1`. For example,
`grayt.rotating_velocity(metric,r,theta,omega)` normalizes a chosen circular
motion. Invalid/superluminal emitters raise an error instead of being clamped.

`TabulatedDisk(radius, intensity, omega)` imports a radial emission/velocity
table with linear interpolation. `save()` and `load()` use portable NPZ.
Callbacks may use phi and emission time for nonaxisymmetric brightness, even
though spacetime remains stationary and axisymmetric. Emission time is
observer time minus the positive coordinate light-travel time.

```python
star = grayt.SphericalStar(radius=5)
surface = grayt.EmittingSurface(lambda theta, phi, t: 3e-4)
image = grayt.render_scene(star, camera, surface=surface, sky=sky)
```

The stellar geometry is Schwarzschild **outside** the surface; no interior,
equation of state, atmosphere, or rotational quadrupole is modeled. Surface
hits are refined to the specified radius. Brightness patterns and material
velocities are separate inputs.

## A first volume-transfer workflow

`EmittingVolume` accepts `sample(metric,r,theta,phi,t) -> (j,alpha,u)`:

- `j`: comoving bolometric intensity emitted per unit proper path length;
- `alpha`: gray absorption coefficient per unit proper path length;
- `u`: normalized coordinate four-velocity of the emitting fluid.

`VolumeGrid` stores these fields on an increasing radial/theta grid and
periodic uniform phi grid. Its arrays have shape `(nr,ntheta,nphi)`, with a
final four-component axis for velocity. Trilinear interpolation renormalizes
the velocity. NPZ imports preserve axes, coefficients, velocity, and name.
The matter distribution can be three-dimensional and nonaxisymmetric.

```python
volume = grayt.VolumeGrid.load("radiation_snapshot.npz")
image = grayt.render_volume(bh, camera, volume, max_step=0.2,
                            exposure=5000, escape_radius=200)
image.save("volume_image.npz")
tau = image.optical_depth
```

This is a **CPU reference implementation**: one sampled Fortran trajectory
per pixel followed by midpoint transfer quadrature. On each segment it uses
`ds=E_em |d lambda|`, `g=E_obs/E_em`, foreground optical depth, and exact
constant-coefficient absorption. Bolometric emission transforms with `g^4`.
Reduce `max_step` independently of geodesic tolerances to check transfer
convergence. No opaque disk/surface is combined with this volume path yet.

Raw GRMHD density, pressure and magnetic field are **not radiation
coefficients**. A microphysics/electron-temperature prescription must produce
the imported `j` and `alpha`. Frequency-resolved synchrotron transfer,
scattering, polarization, plasma refraction, time-dependent snapshots, and
an evolving metric are outside this implementation. Optional sky attenuation
is an illustrative RGB composite, not calibrated bolometric background flux.

## Scientific validation and archives

Independently refine:

1. Metric table resolution, comparing analytic solutions where available.
2. Geodesic integration tolerances and maximum steps.
3. Image resolution, especially near critical curves and small stars.
4. Escape-sphere radius and field-of-view conventions.
5. Volume sampling step and radiation grid, when using volume transfer.

`SceneImage` preserves endpoints, conserved momenta, status, Hamiltonian
monitor, intensity, frequency shift, and RGB. `VolumeImage` also preserves
optical depth; its `g` array is zero because a volume has no single emitter
redshift. `r_hit` exposes endpoint radius even for escaped/failed rays: mask
using status before interpreting it as an emitting radius. Status 1 means an
inner-boundary hit (horizon buffer **or** stellar surface).

NPZ archives use JSON metadata and `allow_pickle=False`. Metric/radiation
tables carry content hashes; callbacks are not serialized. Keep the source
script, data archives, and Git commit together for reproduction. Coordinate
time delay and pseudo-Cartesian ray plots are coordinate-dependent quantities.
The legacy camera's stored covector is past-oriented while its integration
uses negative steps; the new transfer functions explicitly account for that
convention. The measured observer energy is normalized to 1.

Tests cover metric inversion and derivative consistency, analytic versus
sampled Schwarzschild rays, grid refinement, null camera initial conditions,
stellar-surface redshift, emission/path independence, circular Kerr velocities,
gray slab transfer, a flat-spacetime emitting shell, archives, sky seams, and
video export. Near-horizon Hamiltonian error is more sensitive to the
coordinate singularity than error on escaped rays; inspect both separately.

## Gallery, videos, and CLI

```sh
PYTHONPATH=python python examples/model_gallery.py --output output/stationary_models
gravtracer render configs/celestial_kerr.yml -o output/celestial.png --npz
PYTHONPATH=python python examples/validate_custom_metrics.py
PYTHONPATH=python python examples/volume_snapshot.py --convergence-only
```

The gallery generates 14 scientific panels, a comparison sheet, the bundled
NASA celestial map, raw NPZ results, and a JSON manifest. It prioritizes Kerr
Page–Thorne disks and spherical stellar surfaces; charged and quadrupolar
geometries are explicitly labeled theoretical comparisons. Colors use a
fixed exposure. The q-metric examples have differing ADM mass, stated on the
panels. No claims of astrophysical inference are made from these images.

Three MP4s retrace views along smoothly varying observer positions. Each
position is a ZAMO observer; the sequence is not a camera worldline with an
orbital Doppler boost. A fourth video rotates a 3D plotting camera around
fixed geodesics. That video is a coordinate embedding, not an observer's
photograph. MP4 output requires `ffmpeg` on PATH. Camera sequences, rendering
parameters and raw keyframes accompany the videos. `--quick` reduces image
size and frame count; `--skip-videos` produces only stills.
The generated `index.html` provides a local browser for all artifacts.

The gallery supports native desktop windows and optional rendered browser views.

For native desktop windows on macOS, first run
`PYTHONPATH=python python examples/install_desktop_launcher.py` from the
repository, using the environment containing the viewer dependencies.
This registers `gravtracer://view/<preset>` with a local application in
`output/GRAVTRACER Viewer.app`. Then regenerate the page using the command
below. The gallery opens each selected example in a separate native window;
the browser may ask to open the registered application. No server is needed.
Re-run installation after moving the repository or Python environment.

Native windows start on black. **B** cycles black, the same NASA celestial
map used in the gallery, and the original colored diagnostic grid. Background
changes recolor cached rays without retracing. Kerr Page–Thorne views use
OpenCL geometry with Keplerian bolometric g^4 emission and the same escape
sphere as the gallery. q-metric views also use OpenCL. Imported metric and
stellar views use the CPU scene renderer, labeled in the gallery and window.
With the launcher installed, the CPU browser preview is collapsed and only
starts when expanded. Launch errors are logged in `output/desktop_viewer.log`.

```sh
PYTHONPATH=python python examples/observer_gallery.py --output output/stationary_models
```

Drag the rendered disk/sky image to change the observer's inclination and
azimuth; scroll to change the field of view. Model selection, sliders,
automatic azimuth rotation, PNG export, keyboard controls, and display
exposure are available. Each model card links to its observer view. The
default is the Schwarzschild disk at 85 degrees. The viewer works directly
from `file://`, with embedded JavaScript, sky texture, and flux tables, and
does not need a running Python server, OpenCL, or an internet connection.

Web Workers integrate float64 Hamiltonian null geodesics with RKDP45 and
dense event location. Preview/refined tolerances are 1e-6/1e-8 (absolute
tolerance is 1/100 of relative tolerance). Kerr and q-metric expressions
are generated from the existing OpenCL source by
`examples/build_browser_metrics.py`; browser parity tests compare complete
rays and bolometric emission against Fortran. RN and stellar exteriors use
their analytic formulas in the browser. The Python reference renderer uses
the imported metric tables for these examples. Arbitrary metric tables and
volume radiation transfer remain Python workflows.

At fixed observer radius, inclination and field of view, axial symmetry
allows ray reuse: add the azimuth offset to endpoint phi and reevaluate the
sky or surface pattern. Inclination and field-of-view changes retrace the
rays; no viewpoint interpolation or image morphing is used. Refinement
averages four subray display colors per output pixel. These browser images
are visualizations; the gallery's NPZ archives retain the scientific maps.

`examples/validate_observer_gallery.py` exercises the controls, PNG export,
optional coordinate diagrams, and mobile layout with Playwright in offline
mode. It writes screenshots and `observer_validation.json`. Install
Playwright and its Chromium browser, or pass `--chrome /path/to/Chrome` to
use an existing installation. The physics comparisons are in
`tests/test_browser_observer.py` and require Node for JavaScript execution.

The disk/sky edge is intentionally an occultation boundary: the example
disk is an opaque, zero-thickness annulus cut off at r=20 M. Adjacent rays
may terminate on the disk or reach the sky. The smooth axisymmetric disk
does not reveal azimuthal camera motion through moving emission features,
whereas background stars do move. Disk false colors and sky RGB have no
common calibrated spectrum. A physical atmosphere or gradual optical-depth
transition requires an appropriate radiation model, rather than blending
background light through an opaque disk.

For a slower Kerr movie with reduced spatial aliasing:

```sh
OMP_NUM_THREADS=8 PYTHONPATH=python python examples/refine_kerr_movie.py \
  --output output/stationary_models --frames 360 --fps 30 --samples 2
```

This replaces the four-second preview with a 12-second orbit, changing
azimuth by one degree per frame and using four rays per displayed pixel.
It reuses axial symmetry only for stationary Kerr with an axisymmetric
Page–Thorne disk. Frame metadata and full subray keyframes accompany the
movie. `render_movie(..., supersampling=2)` enables the same pixel filter
for other models, with full retracing at every camera. RGB is averaged
after display tone mapping; this is not a calibrated detector response.

Add supplementary interactive coordinate diagrams with:

```sh
uv pip install '.[interactive]'
PYTHONPATH=python python examples/interactive_gallery.py --output output/stationary_models
```

The ray viewer switches between five metrics and supports orbit, zoom, pan,
reset, and layer visibility. A second viewer shows emissivity isosurfaces
from `radiation_snapshot.npz` when that snapshot is present. These are
coordinate visualizations; the displayed vacuum rays continue through the
reference disk plane. The HTML embeds Plotly and the plotted data, so opening
`index.html` via `file://` works without a server or an internet connection.
With an observer viewer present, the diagrams load when their disclosure
is expanded. Rebuilding the gallery index preserves both interactive sections.
The plotted datasets are also exported as `geodesic-view.json` and
`volume-view.json`.

```python
cameras = grayt.orbit_cameras(camera, frames=96, inclination=(45,80))
grayt.render_movie("orbit.mp4", bh, cameras, disk=grayt.PageThorneDisk(bh),
                   sky=sky, exposure=5000, archive_every=24)
```

## Literature context

- Page & Thorne, *Disk-Accretion onto a Black Hole. Time-Averaged Structure of
  Accretion Disk* (1974), [DOI](https://doi.org/10.1086/152990).
- Velásquez-Cadavid et al., *OSIRIS: A New Code for Ray Tracing Around Compact
  Objects*, [arXiv:2202.00086](https://arxiv.org/abs/2202.00086): legacy model.
- Quevedo et al., *Quadrupolar gravitational fields described by the q-metric*,
  [arXiv:1310.5339](https://arxiv.org/abs/1310.5339).
- Eiroa et al., *Reissner-Nordstrom black hole lensing*,
  [arXiv:gr-qc/0203049](https://arxiv.org/abs/gr-qc/0203049).

Hartle–Thorne, Kerr–Newman, Johannsen, boson-star and wormhole models are not
built-in additions in this branch. Compatible exteriors can be supplied
through the custom interface; topology, chart, boundary and radiation
assumptions must still be checked for each model.

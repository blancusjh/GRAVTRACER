# GRAVTRACER

Fast, simple and reliable light tracer in curved spaces.

GRAVTRACER traces light around compact objects to render black-hole shadows,
accretion disks, lensing, stellar surfaces, and celestial skies. A Fortran
geodesic core powers the Python package `grayt` and the `gravtracer` CLI.

![A lensed Page–Thorne disk around a spinning Kerr black hole](docs/images/kerr_disk.png)

*Kerr black hole, spin 0.95, viewed at 70°. The background uses
[NASA SVS Deep Star Maps 2020](https://svs.gsfc.nasa.gov/4851/); disk colors are a
display mapping, not measured spectra.*

## Install

Requires Python 3.10+ and a Fortran compiler such as `gfortran`.

```sh
python -m pip install .
```

For development, install the build tools first, then use an editable install:

```sh
python -m pip install meson-python numpy ninja meson
python -m pip install -e . --no-build-isolation
```

Optional extras: `python -m pip install '.[viewer]'` for the OpenCL desktop
viewer, or `python -m pip install '.[interactive]'` for browser gallery tools.

## Quick start

```python
import grayt

black_hole = grayt.BlackHole(a=0.8)
camera = grayt.Camera(r=100, theta=70, resolution=(640, 400))
disk = grayt.PageThorneDisk(black_hole)
image = grayt.render_scene(
    black_hole, camera, disk,
    sky=grayt.CelestialSky.nasa_starmap(),
    exposure=5000,
)
image.plot()
image.save("observation.npz")
```

Camera angles are in degrees; `image.save` keeps raw ray and radiation maps
with scene provenance. For a command-line render:

```sh
gravtracer render configs/celestial_kerr.yml -o celestial.png --npz
```

## What it supports

- Kerr and Zipoy–Voorhees geometries, plus stationary axisymmetric metric
  tables and spherical stellar or charged exteriors.
- Thin disks, Page–Thorne disks, prescribed surfaces, gray volume radiation,
  and celestial image maps. The legacy thin-disk model reproduces the
  [OSIRIS paper](https://arxiv.org/abs/2202.00086).
- Double-precision CPU rendering; optional OpenCL rendering and a live
  desktop viewer for the supported Kerr and q-metric scenes.
- Scientific `.npz` archives, reproducible galleries, and camera movies.

## Gallery and ray paths

![Gallery of GRAVTRACER black holes, stars, and other stationary models](docs/images/model_gallery.png)

*Kerr holes arranged by spin and inclination, beside stellar, charged, and
Zipoy–Voorhees exteriors, all against the catalog sky. Disk and surface colors
are display mappings.*

![A camera photographs an engraving behind a Kerr black hole through traced null geodesics](docs/images/image_formation_scene.png)

*How an image forms. Each camera pixel sends a null geodesic backward until it
meets the illuminated card, falls into the hole, or escapes. Rays 1 and 2 land
on almost the same point of the engraving but pass on opposite sides of the
hole: they are its primary and secondary images. Run
`python examples/image_formation_scene.py` to reproduce it.*

![Two 3D views of Kerr light-ray trajectories](docs/images/ray_trajectories_3d.png)

*The same traced null rays from two angles. Colors distinguish escaped,
strongly bent, and captured rays. Positions use a pseudo-Cartesian embedding
of Boyer–Lindquist coordinates.*

Generate the gallery yourself with:

```sh
python examples/model_gallery.py --output output/stationary_models --skip-videos
```

Every figure uses one house style: a black field and STIX serif type. Library
plots apply it automatically; call `grayt.style.use()` to style your own
matplotlib figures the same way.

See [custom models and scientific workflows](docs/custom_models.md) for model
assumptions, import formats, and gallery controls. The
[demo notebook](notebooks/grayt_demo.ipynb) provides a guided example.

## Validation and license

Install `.[test]`, then run `make test` for physics and regression checks.
OpenCL cases skip when no compatible device is available. See
[validation and paper errata](docs/validation.md) for reference results.
GRAVTRACER is released under the [MIT License](LICENSE).

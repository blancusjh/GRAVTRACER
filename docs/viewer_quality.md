# Live viewer quality and the OSIRIS comparison

## Why the bright region is compact

[OSIRIS, section 5 and Figure 13](https://link.springer.com/article/10.1140/epjc/s10052-022-10054-0)
shows a broad bright region for spin a=0 and a much smaller, more intense
region for a=0.95. The live CLI viewer starts at a=0.95. The approaching side
of the orbiting disk is Doppler boosted; gravity bends the back of the disk
into the upper arc. Yellow and white represent higher intensity on a chosen
display scale, rather than a measured visible color or temperature.

Match the spin and angular momentum before comparing images:

```sh
gravtracer view --spin 0 --l0 2.8 --gui-backend pyside6
gravtracer view --spin 0.5 --l0 2.8 --gui-backend pyside6
gravtracer view --spin 0.95 --l0 1.8 --gui-backend pyside6
```

The paper uses observer radius 1000 M, inclination 85 degrees, image-plane
limits x=[-24,24] and y=[-12,12], a disk from the ISCO to 20 M, and 2048×1024
pixels. The viewer uses the same default geometry, with lower resolutions
for interaction. The figure's caption has a spin inconsistency: its last
panel and the main text identify a=0.95, while one caption list says 0.998.

The CLI retains the paper replication model: Page–Thorne radial emission,
prescribed constant specific angular momentum, and the g³ intensity factor.
The independent `PageThorneDisk` scene model instead uses circular geodesic
motion and bolometric g⁴ emission. Its images are a different comparison.
See the existing [paper errata](validation.md#errata-in-the-osiris-paper-as-printed)
for the corrected velocity and redshift expressions.

## Display and interaction changes

The native PySide6 settings panel is shown initially. **H** or the **Settings**
button toggles it. **Apply** updates spin (or q), observer position, image-plane
half-width and half-height, disk visibility, outer radius and angular momentum.
Page–Thorne scene disks rebuild their flux profile and ISCO when spin changes;
custom disk models keep their prescribed physics. The panel also selects a
sky image, rotates it, and changes its display gain. Sky image, rotation and
brightness changes reuse the current ray maps. Invalid settings leave the
current scene intact and show an explanation beside **Apply**.


The old moving preview used only 128×64 rays, enlarged to fill the desktop
window. The new default is 512×256: sixteen times as many samples. Idle
frames refine to 1024×512. `--res` and `--preview-res` remain configurable.

The palette now matches the `afmhot` map used by the static reference plots.
Normalization is calibrated from the initial refined image and held fixed
while moving. This avoids brightness changes when a coarse preview happens
to miss a narrow intensity peak. `--norm VALUE` sets an explicit physical
intensity scale for comparisons across scenes.

Use **+ / −** or `--brightness 2` to raise or lower display gain. This changes
RGB presentation only; the underlying intensity and redshift maps are kept.
The initial default gain is 1, with linear intensity normalization. Display
coordinates use the image-plane extents, so different preview dimensions
and nonsquare pixel sampling do not distort the projection.

Camera events are coalesced into 30 Hz preview requests. Pausing for 150 ms
or releasing the mouse triggers refinement. At fixed radius, inclination
and field of view, axial symmetry allows existing rays to be reused for
azimuth changes. Disk intensity, classification and hit radius stay fixed;
escaped endpoint azimuths are translated before sampling the sky. A cached
refined frame retains its resolution during azimuth motion. Changes in
inclination, radius or field of view trigger new integrations.

The installed Fortran build previously selected C's OpenMP dependency,
leaving Fortran integration serial on this Mac. The build now selects the
Fortran dependency, compiles the physics in a separate static library and
links the extension with the Fortran compiler. This enables CPU parallelism
while keeping incompatible OpenMP flags away from Apple's C compiler.

## Academic acceleration approaches

[Bruneton (2020)](https://ebruneton.github.io/black_hole_shader/paper.pdf)
uses precomputed beam intersections and careful filtering to obtain fast,
high-quality Schwarzschild images. Its spherical-symmetry reductions do
not directly apply to spinning Kerr black holes. The present improvement
uses axial symmetry, without introducing interpolated trajectory tables.

[Gelles et al. (2021)](https://arxiv.org/abs/2103.07417)
demonstrate adaptive ray tracing: trace extra rays where estimated image
error is high, especially near sharp features. This is a possible next
algorithmic extension; it is not implemented by the uniform-resolution
preview changes here. Smooth interpolation alone cannot recover an
unresolved photon ring or validate its intensity.

## Reproduce the comparison

From the installed Python environment, run without `PYTHONPATH=python`:

```sh
OMP_NUM_THREADS=8 python examples/viewer_review.py --output output/viewer_review
```

This writes `osiris_comparison.png`, separate CPU/GPU raw NPZ maps and
`measurements.json`. The image compares all three spins at the paper's
2048×1024 resolution with the same palette and color limits. Each colorbar
uses intensity relative to the a=0 CPU peak. Timing samples change inclination
to force integrations; the azimuth timings measure reuse. Initialization
and kernel compilation are excluded from those timing samples.

Measured on the Apple M4 on 2026-09-25 (renderer times, before RGB composition
and window drawing):

| Resolution | Fresh inclination, median | Reused azimuth, median |
|---|---:|---:|
| 128×64, previous preview size | 9.85 ms | 0.032 ms |
| 512×256, new preview size | 22.88 ms | 0.263 ms |
| 1024×512, new refined size | 58.12 ms | 1.59 ms |

At 2048×1024, the a=0 CPU render took 92.4 seconds in the previous serial
build and 15.7 seconds with the corrected build and eight OpenMP threads.
The GPU render took 0.23 seconds. Across the three spins, CPU/GPU pixel
classification disagreement was below 0.002%, and intensity RMS error was
below 0.05% of each CPU image's peak. These measurements describe the tested
camera and scenes; frame times vary with view direction and background.

Apple M4 GPU rendering uses single precision. More samples improve spatial
detail but do not make its geodesics equivalent to the double-precision CPU
solver. Small classification differences and edge artifacts remain possible.
The exported comparison quantifies them separately from display changes.

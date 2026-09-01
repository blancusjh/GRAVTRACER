# `grayt` from a notebook: an interactive-user review

Reviewer context: graduate student in gravitational physics, evaluating the
package purely as a Jupyter/IPython user via the public API. Companion
notebook: `notebooks/grayt_demo.ipynb` (all snippets below were actually
executed; timings on a 10-core Apple-silicon laptop, Python 3.14).

## 1. API first impressions

Strongly positive. The core mental model — *compose three frozen dataclasses
and call `render`* — is learnable in one minute from the module docstring:

```python
img = grayt.render(grayt.BlackHole(a=0.95),
                   grayt.Camera(resolution=(384, 192)),
                   grayt.ThinDisk(l0=1.8))
img.plot()
```

Every object prints a clean dataclass repr (`BlackHole(a=0.9)`,
`ThinDisk(r_in=None, r_out=20.0, l0=2.8, model='page-thorne')`), all
defaults are physically sensible (observer at r=1000 M, theta=85°), and the
result is a plain dataclass of numpy arrays rather than an opaque handle.
A 384×192 disk render takes ~5 s; a 450×280 photograph ~7 s; single geodesics
are milliseconds — genuinely interactive speeds.

The two-layer design (`api` for pixel physics, `system` for Cartesian scene
composition) is coherent, but the seam shows: the layers use different
coordinate conventions (BL + degrees vs pseudo-Cartesian vectors) and
different result types (`Image` vs `Photograph`), and nothing in `help()`
tells you which layer you are supposed to start in.

## 2. What worked well interactively

- **Composability of plots.** Everything that draws accepts `ax=` and
  returns the axes: `img.plot(ax=...)`, `plot_orbits_2d(..., ax=...)`,
  `visualize3d(ax=...)`. Building a 2×2 comparison figure "just worked".
- **Raw arrays are first-class.** `img.g`, `img.r_hit`, `img.status`,
  `img.herr`, `img.theta_inf` are documented fields; masking by
  `img.status == grayt.STATUS_DISK` and histogramming `g` needed no digging.
  The named `STATUS_*` constants beat magic integers.
- **`trace` returns a dict** with self-describing keys
  (`lambda, t, r, theta, phi, p_r, p_theta, herr`) — no class to learn, and
  the per-step constraint error is right there for a quality check.
- **`plot_orbits_2d` polymorphism**: trace dicts, `Ray` objects, or bare
  `(N,3)` arrays, each optionally `(orbit, label)`; horizon drawn for free.
- **Physics validation for free**: `flux_profile` starts exactly at
  `bh.isco`; `r_hit` on a default disk bottoms out at the ISCO; blueshifted
  pixels (`g` up to 1.21) sit on the paper's approaching side. The code
  earns trust quickly.
- **Frozen dataclasses + `dataclasses.replace(cam, resolution=(96,48))`**
  make parameter scans pleasant, and `img.meta` records the full
  render configuration.

## 3. Friction log (ranked by severity)

**F1 — `Camera.theta` silently accepts radians (units trap). Severity: high.**
```python
img = grayt.render(grayt.BlackHole(a=0.9),
                   grayt.Camera(theta=np.pi/2, resolution=(96, 48)),
                   grayt.ThinDisk(l0=1.8))
# no warning: 1.5708 is taken as 1.57 DEGREES -> near-polar view,
# a plausible-looking but physically wrong image
```
Degrees are documented in the `Camera` docstring, but `trace`'s `y0` wants
theta in **radians** while `orbit_ic(theta0=...)` is degrees again. Mixed
conventions inside one package, with no runtime hint, is the most likely
source of silent wrong science for a new user.

**F2 — Physically inconsistent scenes render silently. Severity: high.**
```python
bh0 = grayt.BlackHole(a=0.0)                      # ISCO = 6
img = grayt.render(bh0, grayt.Camera(resolution=(96, 48)),
                   grayt.ThinDisk(r_out=3.0))     # disk entirely inside ISCO
img.intensity.max()   # -> 0.0, zero disk hits, no error, no warning
```
`r_in=None` defaults to the ISCO (good), which here lies *outside* `r_out`,
so the annulus is empty. Nothing complains at construction or render time —
you get a black image and have to debug it yourself.

**F3 — Degenerate camera passes render, explodes later in plotting.
Severity: medium.**
```python
img = grayt.render(grayt.BlackHole(a=0.9), grayt.Camera(resolution=(0, 0)),
                   grayt.ThinDisk())              # "succeeds", (0,0) arrays
img.plot()
# ValueError: zero-size array to reduction operation maximum which has no
# identity        <- raised inside plotting.py, nothing mentions the camera
```
Negative resolutions similarly reach f2py. Validation belongs in
`Camera.__post_init__`.

**F4 — Unknown integrator name leaks a bare `KeyError`. Severity: medium.**
```python
grayt.render(bh, cam, method="rk4")   # KeyError: 'rk4'
```
The valid set (`rkdp45`, `rkck45`, `rkf45`) appears nowhere in the message or
in `help(grayt.render)`; you must read `api.py` to find `_METHODS`.

**F5 — No progress feedback on long calls. Severity: medium.**
`render` and `System.photograph` are silent for their whole runtime (5–10 s
at demo sizes, minutes at paper sizes of 1024×512+). In a notebook you
cannot tell a hung kernel from a slow render; there is no `verbose=` or
callback, and nothing reports the OpenMP thread count.

**F6 — Import ceremony. Severity: low/medium.**
The package is not installable (`pip install -e .` — no `pyproject.toml`),
so every notebook starts with
`sys.path.insert(0, ".../kerr-raytracer/python")`, with a hardcoded absolute
path if the notebook does not live in the repo root.

**F7 — `repr(img)` dumps ~3 kB of array text. Severity: low.**
Typing `img` in a cell prints the full dataclass repr of seven arrays.
A one-line summary (`Image(384x192, a=0.95, 32% disk, 4% captured)`) would
be far kinder; same applies to `Photograph` and `Ray`.

**F8 — f2py errors leak through `trace`. Severity: low.**
```python
grayt.trace(bh, [0, 100, 1.57], p_t=-1.0, p_phi=2.0)
# ValueError: _core._core.raytracer.trace_geodesic: failed to create array
# from the 2nd argument `y0` -- 0-th dimension must be fixed to 6 but got 3
```
The message does state the shape, but the `_core._core.raytracer` prefix is
internal machinery; a Python-side check could say
"`y0 must be (t, r, theta, phi, p_r, p_theta)`".

**F9 — Small discoverability/orientation nits. Severity: low.**
`plot_image`, `plot_shadow`, `plot_lensing` are not exported at top level
(only `plot_orbits_2d` is), so tab-completion never reveals the lensing/shadow
plots that reproduce the paper's Figs. 6–12. `flux_profile` returns a bare
tuple with unstated flux units. `Image` maps are `(nx, ny)` so every direct
`imshow` needs a `.T` (the docstring does warn). `Image.save/load` round-trips
`meta` through `repr()` and drops it on load.

## 4. Docstrings and discoverability

Above average for research code. Highlights: the package docstring opens
with a runnable quick-start; `Camera` has a real numpydoc parameter section
(including the degrees warning and the paper's x-axis sign convention);
`render` cites the exact equation of arXiv:2202.00086 it implements;
`system.py`'s module docstring explains the layer model and the
lambertian/collimated emission physics; `System.photograph` honestly
documents its infinite-plane approximation. `help(grayt.render)` shows a
fully type-hinted signature with defaults — you can drive the main entry
point from `help()` alone.

Gaps: valid `method` strings are undocumented; flux units are unstated;
`Image.plot`'s options (`norm_to`, `vmax`, `cmap`) are only documented on the
underlying `plot_image`, which isn't exported; dataclass `help()` output
buries the good prose under ~60 lines of `__delattr__`/`__dataclass_fields__`
boilerplate (unavoidable, but a `Parameters` section on *every* class would
help — `ThinDisk` doesn't document `l0`'s admissible range, and `l0`'s physical
meaning vs the paper's ℓ₀ takes a moment to pin down). There is no
top-level `grayt.__doc__` pointer to the examples directory, which is where
the best usage patterns actually live.

## 5. Top 5 concrete recommendations

1. **Validate at construction time.** `Camera.__post_init__`: require
   positive integer resolution, `x[0] < x[1]`, `0 <= theta <= 180` with a
   pointed message ("theta is in degrees; got 1.57 — did you pass
   radians?"). `render`: raise (or at least `warnings.warn`) when
   `disk.r_in`/ISCO ≥ `disk.r_out`, and replace the `_METHODS[method]`
   `KeyError` with `ValueError(f"method must be one of {sorted(_METHODS)}")`.
2. **One angle convention.** Accept degrees everywhere in the public API
   (or provide `theta_deg=`/`np.deg2rad` symmetry), and document each
   angle's units in each docstring line where it appears — especially
   `trace`'s `y0` (radians) vs `Camera.theta`/`orbit_ic.theta0` (degrees).
3. **Progress + timing feedback.** A `progress=True` default that prints
   rows-completed (trivial from the OpenMP loop, or a Python-side chunked
   driver) and a final "`73k pixels in 4.6 s, 8 threads`" line; same for
   `photograph` and `propagate_to_screen`.
4. **Make it installable and finish the namespace.** Ship a minimal
   `pyproject.toml` so `pip install -e .` replaces the `sys.path` dance;
   export `plot_shadow`/`plot_lensing`/`plot_image` at top level so
   tab-completion reveals the paper-figure toolkit.
5. **Compact reprs and a `summary()`.** Custom `__repr__` for
   `Image`/`Photograph`/`Ray` (shape, spin, status fractions), plus e.g.
   `img.summary()` printing status counts, `g` range and max constraint
   error — exactly what one checks after every render anyway.

Bottom line: the physics core is fast, well-validated and pleasant to drive;
nearly all friction lives in the last centimeter of input validation, unit
consistency and feedback, which is cheap to fix.

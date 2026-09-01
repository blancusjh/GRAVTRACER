# Consolidated code review — kerr-raytracer / grayt

Date: 2026-08-31. Three independent passes:

1. [Architecture & physics review](reviews/architecture_review.md) —
   core-developer pass over the Fortran and Python source.
2. [CLI user review](reviews/cli_user_review.md) — a fresh user driving
   the code from the terminal (README → make test → CLI → examples).
3. [Notebook user review](reviews/notebook_user_review.md) — an
   interactive user authoring and executing
   [`notebooks/grayt_demo.ipynb`](../notebooks/grayt_demo.ipynb).

## Where the three reviews agree

**Strengths (triangulated):**

- The README is *accurate*: every documented command ran verbatim on
  first try in the CLI pass. The quick-start alone produces a correct
  image.
- Physics self-validates from the outside: flux profile starts exactly
  at the ISCO, r_hit bottoms out at the ISCO, blueshift appears on the
  paper's approaching side, shadows match the Bardeen rim.
- Fast: 384×192 disk render ~3–5 s; whole demo notebook ~15 s; tests
  16 s.
- Reproducibility touches praised twice: `.npz` outputs embed the full
  scene parameters; the errata section documenting three equation bugs
  in the published paper is "exemplary".
- The three-dataclass compose-and-render API and the raw per-pixel
  maps (`g`, `r_hit`, `status`, `herr`) are the ergonomic highlights.

**Weaknesses (triangulated), ranked:**

| # | Finding | Seen by | Severity |
|---|---|---|---|
| 1 | Silent bad inputs: `Camera(theta=np.pi/2)` read as 1.57°; `ThinDisk(r_out=3)` inside ISCO renders empty with no warning; `resolution=(0,0)` accepted, fails later opaquely; `method="rk4"` leaks a bare KeyError | notebook, arch | high |
| 2 | Mixed angle conventions: `Camera.theta`/`orbit_ic` in degrees, `trace` y0 in radians | notebook | high |
| 3 | No packaging: `sys.path` ceremony, no pyproject/console script/`--version`, despite PLAN promising `pip install -e .` | all three | high |
| 4 | CLI shows raw tracebacks for routine user errors (good messages buried) | CLI | medium |
| 5 | Fortran control flow duplicated 3× (adaptive step; event bisection) — where future bugs will come from | arch | medium |
| 6 | `Scene` vs `System` compete for the same ontological slot; instruments split across modules; three result shapes with no common protocol | arch | medium |
| 7 | Scene YAML schema documented nowhere; PLAN.md drifted from reality | CLI | medium |
| 8 | No progress feedback on multi-second renders; threading (OMP_NUM_THREADS) undocumented | CLI, notebook | low |
| 9 | Eq.-(11) camera ICs implemented twice (Fortran + Python) — verified bit-identical today, still a divergence hazard | arch | low |
| 10 | `Image.save/load` drops `meta`; `plot_shadow`/`plot_lensing` not exported; ~3 kB reprs | notebook | low |

## Agreed action plan

- **P0 (cheap insurance):** constructor-time validation across the
  dataclasses (positive resolutions, r_out vs ISCO warning, spin range
  already done) with a radians-vs-degrees hint; friendly method-name
  error listing valid integrators.
- **P1 (adoption):** pyproject + meson-python packaging with a `grayt`
  console script; CLI try/except for one-line errors; document the
  YAML schema; pick one angle convention (degrees at the API surface,
  stated everywhere); simple progress feedback + threads note.
- **P2 (structure):** extract the shared Fortran adaptive-step/event
  routines; fold Scene into System; add a `Trajectory` result type and
  a common result protocol; single-source the camera ICs; refresh
  PLAN.md.

## Future extensions the code deserves

Ordered by (scientific payoff) / (effort), given the current
architecture:

1. **q-metric spacetime** (paper Appendix A) — completes the
   replication and forces the healthy `Spacetime` abstraction (Fortran
   metric selector + Python base class). Everything downstream (shadow,
   disk, photograph) then works for naked singularities for free.
2. **Self-consistent disk toggles** — `velocity="keplerian"` and
   bolometric `g⁴`: one-line physics options that make the model
   internally consistent beyond the paper's choices.
3. **Physical photograph** — apply the redshift factor to photographed
   sources: g³ intensity dimming plus spectral color shifting
   (RGB → shifted blackbody approximation). Turns the demo into a
   quantitative instrument.
4. **Volumetric radiative transfer** — integrate dI/dλ = j − αI along
   rays (the event machinery already exists): optically thin tori,
   Polish doughnuts, and orbiting hotspots → light curves and movies.
   This is the step from "images" to "observables".
5. **Angular emission models n_k(θ,φ)** — as backward-tracing weights
   at the hit point (never as forward ray fans), completing the
   image-formation idea at full generality and negligible cost.
6. **Adaptive supersampling** near the critical curve, where the
   winding divergence makes single-pixel sampling alias (the speckle
   band); refine pixels whose neighbors disagree in status.
7. **Time-dependent scenes** — hotspot orbits, disk variability,
   camera trajectories; the movie pipeline is just a loop plus
   provenance metadata.
8. **Polarization** — parallel-transport the polarization vector
   (Walker–Penrose constant in Kerr) and add polarized transfer;
   aligns the code with EHT-era observables.
9. **Performance tier** — the RHS is tiny and pixel-parallel: an
   OpenACC/GPU port or MPI tiling when paper-scale (2048×1024)
   parameter surveys become routine.
10. **Community packaging** — pip wheels, CI running the validation
    suite, a docs site with the notebook as the tutorial; the code is
    unusually well-validated for release, packaging is the only gap.

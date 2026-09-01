# Architecture & physics review

Reviewer: core developer pass (self-review with fresh eyes), 2026-08-31.
Scope: `src/*.f90`, `python/grayt/*`, tests, examples, build.

## Verdict

The physics core is solid and honestly validated; the layering
(Fortran = geodesic engine, Python = semantics) is the right cut and has
already proven extensible (image formation landed without touching the
integrator). The main debts are: duplicated control flow in the Fortran
driver routines, an ontology that grew two "top-level" concepts
(`Scene` vs `System`), result types without a common shape, and
almost no input validation. None of these are bugs today; all of them
are where bugs will come from as the code grows.

## 1. Physics groundedness

Solid and traceable to first principles:

- Hamiltonian formulation with conserved (p_t, p_phi); the constraint
  |H| is monitored per ray and tested (drift ~1e-10 with RKDP45).
- Camera tetrad is ZAMO-based and produces exactly null ICs (~1e-16);
  the sign error in the paper's eq. (7) is corrected and documented.
- Validation is against *independent* analytics: Bardeen shadow rim,
  ISCO/horizon closed forms, weak-field deflection 4M/b + 15πM²/4b²
  (matched < 2%), Page–Thorne flux with F(isco)=0 and peak at 9.55M.

Physically debatable choices — all inherited from the paper and worth a
toggle rather than a rewrite:

- **Kinematics/emission inconsistency:** disk velocity uses constant
  l0 (eq. 18) while Page–Thorne flux assumes Keplerian orbits. This is
  what the paper does, and we replicate it; a `velocity="keplerian"`
  option would make the model self-consistent.
- **g³ vs g⁴:** eq. (19) applies to specific intensity; a bolometric
  option (g⁴) is a one-line addition worth exposing.
- **Photograph mode is purely geometric:** colors are sampled with no
  redshift factor at all (no g³ dimming, no Doppler color shift).
  Correct as a "where does the light come from" map; not a physical
  photograph. Should be stated in the docstring and offered as a
  `redshift=` option later.
- **Pseudo-Cartesian embedding** (x = r sinθcosφ …) for planes and 3D
  plots is exact only asymptotically. Documented; fine while sources
  and screens sit at r >> M.

## 2. Fortran core

Strengths: single-responsibility modules with clean dependency order
(KIND → METRIC → EOM → RK → DISK → RAYTRACER); everything callable in
the hot loop is PURE and thread-safe; analytic metric derivatives are
FD-tested; the f2py surface is deliberately narrow (`only:` list).

Debts, in priority order:

1. **The adaptive-step loop exists three times** (TRACE_RAY,
   TRACE_TO_PLANE, TRACE_GEODESIC) and **event bisection three times**
   (REFINE_CROSSING, REFINE_RADIUS, inline in TRACE_TO_PLANE). The
   pattern is identical: step → error-control → check scalar event
   functions → bisect the accepted step. Extract one ADVANCE_ADAPTIVE
   step routine and one REFINE_EVENT(g(Y)) bisection; the three drivers
   become thin loops over event lists. This is the single highest-value
   refactor: today a fix to step control must be applied in three
   places.
2. **Hardcoded numerics:** HMIN/HMAX/CAPTURE_BUFFER, the flux-table
   size (4000), and the bisection depth (48) are module PARAMETERs
   invisible from Python. HMAX=25 interacts with event detection
   (a step can in principle contain two equatorial crossings and be
   missed — unreachable in practice because error control shrinks steps
   in the strong field where the disk lives, but the invariant is
   implicit). Promote them to arguments with defaults, or at least one
   SETTINGS module documented as the contract.
3. **RENDER_IMAGE returns seven parallel arrays.** Fine for f2py, but
   the Fortran-side "what is a result" concept is smeared across the
   argument list. Acceptable; document the column meaning in one place.
4. Cosmetic: TRACE_GEODESIC's `DO N = 1, 100*NMAX` retry budget is
   magic; name it.

## 3. Python ontology

Current concept graph, annotated:

```
api.py:    BlackHole  Camera  ThinDisk  Image   render()/shadow()/trace()
scene.py:  Scene  (YAML → render)            ← top-level concept #1
system.py: PlanarSurface → ImageSource, Screen
           Ray, PhysicalSystem, System       ← top-level concept #2
           Photograph
```

Findings:

1. **`Scene` and `System` compete for the same ontological slot**
   ("the whole setup"). Scene predates System and only covers the
   backward-render path. Resolution: Scene should *construct* a System
   (`Scene.from_yaml(...).to_system()`), or be absorbed as
   `System.from_yaml`. One laboratory, one serialization of it.
2. **Instruments are split across modules.** Camera (an instrument)
   lives in `api.py` beside BlackHole (physics), while Screen lives in
   `system.py`. A `spacetime / matter / instruments / results / lab`
   module split would make the crystal axes visible in the file layout
   itself.
3. **Three result shapes with no common protocol:** `Image`
   (dataclass, .plot/.save), `Photograph` (dataclass, .plot only),
   and `trace()`'s plain dict. A minimal shared protocol (.plot(),
   .save(), .meta) plus a `Trajectory` dataclass would let downstream
   code treat results uniformly.
4. **`PhysicalSystem.rays` mixes history into ontology.** Traced rays
   are *records of experiments*, not part of the physical system's
   definition. Harmless now; move to System (the laboratory keeps lab
   notebooks) when convenient.
5. **`BlackHole` conflates "spacetime" with "Kerr parameters".** The
   q-metric extension (paper Appendix A) will force a `Spacetime` base
   with Kerr as one implementation, mirrored by a metric-selector in
   Fortran. Do the abstraction when the second metric arrives, not
   before — but know it's coming.
6. **Eq. (11) is implemented twice** (Fortran CAMERA_INIT; vectorized
   in System.photograph). Verified bit-identical today (max diff 0.0
   over random pixels), including the sign correction — but the two
   copies can drift. Either add a bundled CAMERA_INIT_MANY in Fortran
   or make the Python version the single source and feed the Fortran
   render ICs directly.
7. **Validation is thin at the edges** (verified empirically):
   `Camera(resolution=(0,0))` is accepted silently;
   `ThinDisk(r_out=3)` inside the ISCO renders an empty disk with no
   warning; nothing checks `l0` against the timelike condition, where a
   bad value would produce NaNs via the redshift square root. A small
   `__post_init__` validation pass across the dataclasses is cheap
   insurance.

## 4. Tests, docs, build

- Physics coverage is good (21 tests, all meaningful). Gaps: no test
  touches the CLI, `Scene.from_yaml`, the plotting module, or the
  degenerate inputs above. One smoke test each would do.
- README is accurate and the errata section is genuinely valuable.
  PLAN.md has drifted slightly (says "M5 remaining" items that partly
  landed); refresh it or fold it into README.
- Build: the Makefile + `.f2py_f2cmap` works but is bespoke. A
  `pyproject.toml` with meson-python would give `pip install -e .`,
  wheels, and CI-ability — the single biggest step toward other people
  using this.

## 5. Prioritized actions

1. Extract the shared adaptive-step + event-refinement routines
   (Fortran).  [correctness insurance]
2. Merge Scene into System; move Camera next to Screen. [ontology]
3. Dataclass validation + a Trajectory result type. [robustness]
4. Single-source the camera ICs. [maintenance]
5. pyproject/meson-python packaging + CI running `make test`. [reach]

# Development Plan — Kerr Ray Tracer (OSIRIS replication)

**Goal:** replicate the results of *OSIRIS: A New Code for Ray Tracing Around
Compact Objects* (Velásquez-Cadavid et al., arXiv:2202.00086, EPJ C), with
Fig. 13 (thin accretion disk around Kerr black holes, a = 0, 0.5, 0.95) as the
flagship target.

**Approach:** hybrid **modern Fortran core + Python interface**, matching the
original (OSIRIS is itself Fortran) while providing the ergonomic layer the
paper's code lacks.

---

## 1. Physics specification (extracted from the paper)

Conventions: geometrized units G = c = 1, M_BH = 1, signature (−,+,+,+),
Boyer–Lindquist coordinates {t, r, θ, φ}.

### 1.1 Equations of motion (Sec. 2.1)
Hamiltonian formulation for null geodesics, H = ½ g^{μν} p_μ p_ν = 0.
Evolved variables: (t, r, θ, φ, p_r, p_θ); p_t and p_φ are conserved
(stationarity + axisymmetry). The RHS only needs the contravariant metric
components g^{tt}, g^{tφ}, g^{rr}, g^{θθ}, g^{φφ} and their r- and
θ-derivatives — this defines the **metric module interface** and makes the
code extensible to any stationary axisymmetric metric (e.g. the q-metric of
Appendix A).

### 1.2 Backward ray tracing & camera (Secs. 2.2–2.3)
- Observer at (t₀, r₀, θ₀, φ₀); image plane with impact parameters
  x = −β r₀, y = α r₀ (small-angle approximation, eq. 4).
- Initial momenta from the observer tetrad Λ_μ^ν (eq. 7) and the null
  constraint, giving eq. (11):
  p_θ = y √g_θθ / r₀,  E = 1/A_t + x g_tφ /(r₀ √g_φφ),
  L = −x √g_φφ / r₀,  p_r = √( g_rr [1 − (𝒫^θ)² − (𝒫^φ)²] ),
  with A_t = √( g_φφ / (g_tφ² − g_tt g_φφ) ).
- Integrate **backwards** from the image plane; classify rays as
  captured (r < r_H + δ_r), escaped (r > r_cs), or disk-hit.

### 1.3 Integrators (Sec. 3)
Four adaptive schemes, selectable at runtime: **RKDP45 (default — keeps
|H| < 10⁻¹¹, the paper's best)**, RKF45, RKCK45, Bulirsch–Stoer.
Diagnostic: Hamiltonian constraint error = |H_num| (eq. 12), tracked per ray.

### 1.4 Kerr metric (Sec. 4.3)
Eq. (15): Σ = r² + a²cos²θ, Δ = r² − 2Mr + a², horizon r_H = M + √(M² − a²).

### 1.5 Thin accretion disk (Sec. 5)
- Equatorial, circular orbits (u^r = u^θ = 0); angular velocity from constant
  specific angular momentum l₀ (eq. 18):
  Ω := u^φ/u^t = −(g_tφ + g_φφ l₀) / (g_φφ + g_tφ l₀).
- Redshift factor (eq. 22):
  g = p_t (1 + Ω p_φ/p_t) (−g_tt − g_φφ Ω² − 2Ω g_tφ)^{−1/2}
- Observed intensity: **I_obs = g³ I_em** (eq. 19).
- Emission model: Page & Thorne (1974) time-averaged flux F(r) [ref 30].
- Disk extent: r_in = r_isco(a), r_out = 20.

### 1.6 Fig. 13 parameters (the target)
| Parameter | Value |
|---|---|
| Observer | r₀ = 1000, θ₀ = 85°, φ₀ = 0, t₀ = 0 |
| Spins | a = 0, 0.5, 0.95 (caption says 0.998 but panel labels say 0.95 — use labels) |
| l₀ | 2.8 (a = 0 and 0.5), 1.8 (a = 0.95) |
| Disk | r_isco ≤ r ≤ 20 |
| Image plane | x ∈ [−24, 24], y ∈ [−12, 12] |
| Resolution | 2048 × 1024 |
| Normalization | a = 0 panel normalized to max 1; others in same units (max ≈ 3.9, ≈ 50) |
| Colormap | matplotlib `afmhot`-like |

---

## 2. Architecture

```
kerr-raytracer/
├── src/                      # Fortran core (modern Fortran, 2008+)
│   ├── kind_params.f90       # real64 kinds, constants
│   ├── metrics/
│   │   ├── metric_base.f90   # interface: g^{μν}(r,θ) + derivatives
│   │   ├── kerr.f90          # Kerr in BL coords (a=0 → Schwarzschild)
│   │   └── qmetric.f90       # (later) Appendix A q-metric
│   ├── hamiltonian.f90       # geodesic RHS, constraint monitor
│   ├── integrators.f90       # RKDP45 (default), RKF45, RKCK45, BS
│   ├── camera.f90            # tetrad, image-plane initial conditions (eq. 11)
│   ├── disk.f90              # r_isco, Ω(l₀), redshift g, Page–Thorne flux
│   └── raytrace.f90          # per-ray driver + OpenMP loop over pixels
├── python/grayt/             # Python package (thin, ergonomic)
│   ├── __init__.py           # high-level API: trace(), render()
│   ├── config.py             # dataclass/YAML scene description
│   ├── plotting.py           # figure reproduction (afmhot, colorbars)
│   └── cli.py                # `grayt render scene.yml`, `grayt shadow ...`
├── configs/                  # fig6_shadow.yml, fig13_a0.yml, ...
├── tests/                    # pytest: constraint norms, ISCO values,
│                             #   shadow vs analytic Bardeen curve (eqs 13–14)
├── validation/               # scripts reproducing paper figures 4–7, 12, 13
├── PLAN.md
└── README.md
```

**Build:** f2py via meson (`meson-python`), so `pip install -e .` compiles the
Fortran with OpenMP. Fallback: plain `make` + `f2py` command.

**Interface design (the "smart interface"):**
```python
from grayt import Scene, render
scene = Scene(metric="kerr", a=0.95, observer=dict(r=1000, theta=85),
              disk=dict(model="page-thorne", l0=1.8, r_out=20),
              camera=dict(x=(-24, 24), y=(-12, 12), n=(2048, 1024)))
img = render(scene)          # returns ndarray + metadata
```
plus a CLI: `grayt render configs/fig13_a095.yml -o fig13_a095.png`.
Every run stores parameters + git hash in the output metadata for
reproducibility.

---

## 3. Milestones (each ends with a validation against the paper)

- **M0 — Skeleton & build.** Repo layout, meson/f2py build compiling a
  trivial module callable from Python, pytest wired up, CI-ready.
- **M1 — Geodesics.** Kerr metric + Hamiltonian RHS + RKDP45. Validate:
  single orbits (escape/fall, a = 0.98) with |H| < 10⁻¹¹ → reproduce
  Figs. 4–5 constraint curves.
- **M2 — Camera & shadow.** Image-plane initial conditions + ray
  classification. Validate: a = 0.98 shadow vs analytic Bardeen curve
  (eqs. 13–14) → reproduce Fig. 6; constraint maps → Fig. 7.
- **M3 — Lensing (optional but cheap).** Celestial-sphere 4-color test →
  reproduce Fig. 12. Strong qualitative check on the geodesic bending.
- **M4 — Thin disk (flagship).** Disk intersection, Ω(l₀), redshift g,
  Page–Thorne flux, I_obs = g³ I_em → **reproduce Fig. 13** for
  a = 0, 0.5, 0.95 at 2048×1024, matching morphology and colorbar ranges.
- **M5 — Polish & extend.** CLI/docs, OpenMP scaling benchmark (compare
  Fig. 8 timings), remaining integrators, q-metric (Appendix A),
  time-like geodesics (Fig. 14).

## 4. Key numbers for tests

- r_isco: 6.0 (a=0), ≈ 4.2330 (a=0.5), ≈ 1.9372 (a=0.95) [prograde]
- r_H: 2.0 (a=0), ≈ 1.8660 (a=0.5), ≈ 1.3122 (a=0.95)
- Schwarzschild photon sphere: r = 3, shadow critical impact parameter √27 ≈ 5.196
- Hamiltonian constraint with RKDP45: |H| ≲ 10⁻¹¹ along rays

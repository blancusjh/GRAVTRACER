# Validation and paper errata

Run `make test` for the physics and regression suite. OpenCL cases skip when
no compatible device is available. These reference results come from the
original Kerr and q-metric validation work:

| Check | Reference result |
|---|---|
| ISCO radii (a = 0, 0.5, 0.95) | 6.000, 4.233, 1.937 |
| Camera initial conditions | Null to ~1e-16 (Kerr and q-metric) |
| Constraint drift, Figs. 4–5 orbits (rtol 1e-11) | RKDP45 ~1e-10 |
| Shadow vs analytic Bardeen rim, a = 0.98 (Fig. 6) | Within 1 pixel |
| Page–Thorne flux, a = 0 | F(ISCO) = 0, peak at r = 9.55 |
| Weak-field deflection (b = 50) | 4M/b + 15πM²/4b² to < 2% |
| q-metric | q = 0 ≡ Schwarzschild to round-off; shadow scales with ADM mass 1+q |

For imported metrics, radiation models, and native viewer comparisons, see the
[scientific validation notes](custom_models.md#scientific-validation-and-archives).

## Errata in the OSIRIS paper as printed

1. **Eq. (7):** The `(g_tφ/g_φφ)L` term in `𝒫^t` needs a minus sign. The
   printed sign makes camera initial conditions non-null for nonzero spin;
   the paper's base-change matrix has the correct sign.
2. **Eq. (18):** The numerator `g_tφ + g_φφ l0` must be
   `g_tφ + g_tt l0`. The printed version gives `Ω → −l0` in the
   Schwarzschild limit.
3. **Eq. (22):** The printed `g` equals `ν_em/ν_obs`; the redshift factor
   in `I_obs = g³ I_em` is its inverse.
4. **Eq. (A.1):** The spatial block of the q-metric has a sign error.
   GRAVTRACER uses the standard Zipoy–Voorhees form.

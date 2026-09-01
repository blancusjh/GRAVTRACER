# CLI User-Experience Review — kerr-raytracer / `grayt`

*Reviewer: computational astrophysicist, evaluating purely as a terminal user.*
*Date: 2026-08-31. Machine: macOS (Darwin 25), 10 cores, Homebrew Python 3.14,
OpenMP enabled. The compiled extension (`python/grayt/_core.cpython-314-darwin.so`)
was already present; I did not rebuild it, so `make` itself is untested here.*

---

## 1. First contact with the docs

`README.md` is short, honest, and unusually substantive for a research code.
The physics claims are front-loaded (validation table with ISCO radii, Bardeen
rim agreement, constraint drift), the architecture is stated in one sentence,
and — genuinely rare — there is an **errata section documenting three bugs in
the published paper** (Eqs. 7, 18, 22 of arXiv:2202.00086) with pointers to
where each fix lives in the Fortran. That alone made me trust the code more.

What worked as documented:

- The Python quickstart snippet runs **verbatim** (I only lowered the
  resolution). `sys.path.insert(0, "python")` plus `import grayt` just works.
- Both CLI one-liners from the README work as printed.
- The example-scripts table accurately maps scripts to paper figures, and the
  scripts really do take CLI flags as promised.
- The unchecked "M5 q-metric, time-like geodesics" status box is honest.

Where I had to guess or where docs diverge from reality:

- **The Scene YAML format is documented nowhere.** The README shows
  `grayt.Scene.from_yaml("configs/fig13_a0.yml")` and stops. To learn the
  schema I had to read the three files in `configs/` and then
  `python/grayt/scene.py` + `api.py` to find out which keys exist, that
  `r_in: isco` is a magic string, and what `integrator: {method, rtol, atol}`
  accepts. A newcomer without source access is stuck at "copy a config and
  mutate it blindly."
- **PLAN.md has drifted badly from reality** (acceptable for a planning doc,
  but confusing if read as documentation, which its prominence invites):
  the promised tree (`src/metrics/metric_base.f90`, `hamiltonian.f90`,
  `camera.f90`, `config.py`, a `validation/` directory) does not match the
  actual flat `src/` (`kerr_metric.f90`, `geodesic_eom.f90`, `raytracer.f90`,
  ...); the promised Bulirsch–Stoer integrator does not exist (only the three
  RK45 variants); the promised `pip install -e .` via meson-python does not
  exist (**no `pyproject.toml`/`setup.py` anywhere**); and "every run stores
  parameters + git hash in the output metadata" is only half true — the `.npz`
  `meta` field stores the full parameter dict (good) but no git hash.
- README says "Requires gfortran, meson, ninja" but the Makefile just calls
  `f2py`; true only indirectly (numpy ≥ 1.26 f2py uses a meson backend).
  Worth one clarifying sentence.
- Minor: `examples/image_formation.png` sits inside `examples/`, contradicting
  "All outputs go to `output/`."

## 2. What I ran and what happened

All commands from the repo root. Wall times from `/usr/bin/time` (`real`).

| Command | Result | Wall time |
|---|---|---|
| `make test` | 21 passed, 0 failed (`pytest -q`: 15.30 s) | 16.2 s |
| `PYTHONPATH=python python3 -m grayt render configs/fig13_a0.yml --res 384 192 -o output/review_cli_a0.png` | "wrote output/review_cli_a0.png in 3.1s"; image is physically correct (lensed disk, secondary image below, photon ring visible) | 3.7 s |
| `PYTHONPATH=python python3 -m grayt shadow -a 0.7 --res 300 300 -o output/review_cli_shadow.png` | correct D-shaped, frame-dragging-displaced silhouette; plot is bare (no title/spin annotation) | 5.1 s |
| `python3 examples/shadow.py -a 0.6 --res 300` | "rendered 300x300 in 11.2s", wrote `output/shadow.png`; two NumPy `RuntimeWarning: divide by zero` lines on startup | 12.7 s |
| `python3 examples/disk_images.py --spins 0.3 --l0 2.5 --res 384 192` | "a=0.3: rendered 384x192 in 5.6s, max I=4.895e-04, max\|H\|=1.7e-05", wrote `output/disk_images.png` | 7.2 s |
| `python3 examples/orbits2d.py` | wrote `output/orbits2d.png` | 5.9 s |
| README quickstart (Python API, a=0.95, 512×256) | rendered, `.npz` saved with full metadata | 9.3 s |
| same render as row 2 with `OMP_NUM_THREADS=1` | 16.1 s vs 3.1 s → **~5× speedup on 10 cores**; OpenMP works and the env var is honored (but undocumented) | 17.1 s |

Notably, the example scripts run **without** setting `PYTHONPATH` — they
self-locate the package. Nice touch, and better than the README implies.

Deliberate misuse (error-message quality):

- `python3 -m grayt render configs/nonexistent.yml` → full Python traceback
  ending in `FileNotFoundError: [Errno 2] No such file or directory:
  'configs/nonexistent.yml'`. Exit code 1 (correct), but 12 lines of frames
  through `runpy` for a typo'd path is hostile.
- `python3 -m grayt shadow -a 1.5` → traceback ending in
  `ValueError: spin must satisfy |a| <= 1, got 1.5`. The validation message
  itself is excellent — it is just buried under a traceback.
- YAML with a wrong key (`spin:` instead of `a:`) → traceback ending in
  `TypeError: BlackHole.__init__() got an unexpected keyword argument 'spin'`.
  Decipherable, but no hint of valid keys or which config section failed.
- `python3 -m grayt render` (no config) and bare `python3 -m grayt` → proper
  argparse usage + "the following arguments are required: ...", exit 2. Good.
- No `PYTHONPATH` → bare `No module named grayt`; fine, the README covers it.

## 3. Friction log (ranked by severity)

1. **[High] Raw tracebacks for routine user errors.** Bad config path, out of
   range spin, and unknown YAML keys all dump full tracebacks (see quotes
   above). `cli.py:main()` has no try/except around expected failure modes.
   Exit codes are right; presentation is not.
2. **[High] Undocumented YAML schema.** The central CLI workflow
   (`grayt render <config.yml>`) has zero schema documentation; users must
   reverse-engineer `configs/*.yml` and read `scene.py`/`api.py` source
   (e.g. to discover `r_in: isco` or the `integrator:` block).
3. **[Medium] No packaging.** No `pyproject.toml` despite PLAN.md promising
   `pip install -e .`; no `grayt` console entry point, so every invocation is
   `PYTHONPATH=python python3 -m grayt ...`. `grayt.__version__` is `"0.1.0"`
   but there is no `--version` flag.
4. **[Medium] Missing CLI conveniences.** No `--threads` (OMP_NUM_THREADS
   works — ~5× on 10 cores — but no doc mentions it); no progress indicator
   (irrelevant at 384×192, but the paper-reproduction resolution is 2048×1024
   and would leave the terminal silent for minutes); `render` can override
   only `--res` — not spin, `l0`, inclination, or integrator; no
   verbosity/quiet control.
5. **[Medium] Flag inconsistencies between the CLI and examples.**
   `grayt shadow --res NX NY` (two ints) vs `examples/shadow.py --res RES`
   (one int) — I hit this immediately. The `shadow` subcommand's `-a`, `-o`,
   `--res` flags have no help strings at all; `orbits2d.py` exposes only `-o`.
6. **[Low] Self-contradicting help text.** `grayt render -o` says "output PNG
   (and .npz alongside)" yet the `.npz` is only written when `--npz` is also
   passed.
7. **[Low] Output naming.** CLI defaults (`grayt_out.png`, `shadow.png`) land
   in the CWD while examples write to `output/`; `examples/shadow.py` always
   writes `output/shadow.png`, so a spin scan silently overwrites itself.
   The shadow PNG has no title, spin label, or axis annotations.
8. **[Low] Cosmetic warnings.** `examples/shadow.py` emits two
   `RuntimeWarning: divide by zero encountered in divide` (lines 28–29,
   analytic Bardeen rim at r = 1) on every run.
9. **[Low] PLAN.md drift** (details in §1): phantom `validation/` directory,
   phantom Bulirsch–Stoer integrator, renamed files, missing git-hash
   metadata. Harmless individually; collectively erodes trust in the docs.

## 4. What genuinely impressed me

- **Everything the README shows actually runs, verbatim, first try** — the
  quickstart, both CLI lines, and the examples. That is far above the norm
  for paper-replication codes.
- **Speed.** A full thin-disk render (backward-traced null geodesics from
  r = 1000, adaptive RKDP45) at 384×192 in 3.1 s; a 512×256 a = 0.95 disk in
  ~9 s; near-ideal use of the machine (5× on 10 cores without any user
  action). The whole 21-test physics validation suite runs in 15 s.
- **The output is right.** Photon ring and secondary disk image where they
  belong; correctly displaced D-shaped shadow at a = 0.7; per-run diagnostics
  (`max|H|=1.7e-05`) printed by the examples.
- **Reproducibility plumbing.** The `.npz` output carries `intensity`, `g`,
  `r_hit`, `status`, `herr`, `phi_inf`/`theta_inf`, `extent`, and a `meta`
  string with the *complete* scene parameters. You can reconstruct a run from
  its artifact.
- **The paper-errata section.** Documenting three sign/index errors in the
  published equations, with the code locations of the fixes, is exemplary
  scientific software practice.
- Examples that self-insert the package path, take real flags, and default to
  paper-figure parameters — the "general-purpose defaults reproduce the
  paper" design is exactly right.

## 5. Top 5 recommendations

1. **Catch expected errors in `cli.py` and print one-line messages** (e.g.
   `grayt: error: config file not found: configs/nonexistent.yml`), exiting
   1; hide tracebacks behind `--debug`. The good validation messages
   (`spin must satisfy |a| <= 1, got 1.5`) already exist — surface them.
2. **Add a `pyproject.toml`** (meson-python, as PLAN.md already envisioned)
   so `pip install -e .` builds the Fortran and installs a `grayt` console
   script; add `--version` wired to `grayt.__version__`.
3. **Document the Scene YAML schema** — a ~20-line annotated example in the
   README (every key, defaults, `r_in: isco`, integrator choices) — and let
   `render` override more than resolution: `--a`, `--l0`, `--theta`,
   `--method` mirroring `--res`.
4. **Unify and finish the flag surface**: same `--res` convention in CLI and
   examples, help strings on every flag (the `shadow` subcommand has none),
   fix the misleading `-o` help text, default CLI outputs into `output/` with
   parameter-tagged names (`shadow_a0.70.png`).
5. **Add `--threads N` and a progress line** for large renders (row-level
   OpenMP progress or a simple percentage), and document `OMP_NUM_THREADS`
   plus the measured ~5×/10-core scaling in the README. While there, update
   or clearly label PLAN.md as historical and silence the two divide-by-zero
   warnings in `examples/shadow.py`.

---

*Files produced during this review: `output/review_cli_a0.png`,
`output/review_cli_shadow.png`, `output/review_api_a095.npz`,
`output/review_npz_test.png` + `.npz`, and (via examples) `output/shadow.png`,
`output/disk_images.png`, `output/orbits2d.png`. Note: several `demo_*.png`
files appeared in `output/` during the session from a process outside this
review; they are not attributable to any command run here (no reference to
"demo" exists anywhere in the repo).*

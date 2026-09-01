# kerr-raytracer

Backward ray tracing of null geodesics around Kerr black holes, replicating
the results of *OSIRIS: A New Code for Ray Tracing Around Compact Objects*
(Velásquez-Cadavid et al., arXiv:2202.00086, Eur. Phys. J. C).

Flagship target: images of a thin accretion disk (Page–Thorne emission,
I_obs = g³ I_em) around black holes with spin a = 0, 0.5, 0.95 (Fig. 13 of
the paper).

**Design:** hybrid architecture — a modern Fortran core (Hamiltonian geodesic
integration, OpenMP-parallel over image pixels) wrapped with f2py, driven by a
Python package (`grayt`) providing configuration, a high-level API, a CLI, and
plotting.

See [PLAN.md](PLAN.md) for the full physics specification, architecture, and
milestones.

## Status

- [x] M0 planning — physics spec extracted from the paper
- [ ] M0 skeleton & build system
- [ ] M1 geodesics + integrators (Figs. 4–5)
- [ ] M2 camera & shadow (Figs. 6–7)
- [ ] M3 celestial-sphere lensing (Fig. 12)
- [ ] M4 thin accretion disk (Fig. 13)
- [ ] M5 polish, benchmarks, extensions

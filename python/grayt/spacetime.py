"""Spacetimes: the geometric layer of the ontology.

A ``Spacetime`` names a metric implemented in the Fortran core (module
SPACETIME) via an integer id ``mid`` and a parameter vector ``par``.
Analytic registered metrics use the Fortran dispatcher. CustomMetric and
TabulatedMetric (grayt.metrics) import compatible stationary axisymmetric
geometry without recompilation. Radiation and boundary choices are separate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _core

MID_KERR = 1
MID_QMETRIC = 2


class Spacetime:
    """Base class for spacetimes (geometrized units G = c = M = 1).

    Subclasses define ``mid`` (Fortran metric id) and ``par`` (length-4
    parameter vector) plus a ``capture_radius`` below which rays count
    as captured.
    """

    mid: int = 0

    @property
    def par(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def capture_radius(self) -> float:
        raise NotImplementedError

    def metric_cov(self, r, theta):
        """Covariant components (tt, t-phi, rr, thth, phph) at (r, theta)."""
        return _core.spacetime.metric_cov(self.mid, self.par, r, theta)

    def metric_contra(self, r, theta):
        """Contravariant components and their r/theta derivatives."""
        return _core.spacetime.metric_contra(self.mid, self.par, r, theta)


@dataclass(frozen=True)
class BlackHole(Spacetime):
    """Kerr black hole with dimensionless spin ``a`` (Boyer-Lindquist)."""

    a: float = 0.0
    mid = MID_KERR

    def __post_init__(self):
        if not -1.0 <= self.a <= 1.0:
            raise ValueError(f"spin must satisfy |a| <= 1, got {self.a}")

    @property
    def par(self) -> np.ndarray:
        return np.array([self.a, 0.0, 0.0, 0.0])

    @property
    def horizon(self) -> float:
        """Outer event horizon radius r_H = 1 + sqrt(1 - a^2)."""
        return float(_core.raytracer.get_horizon(self.a))

    @property
    def isco(self) -> float:
        """Prograde innermost stable circular orbit radius."""
        return float(_core.raytracer.get_isco(self.a))

    @property
    def capture_radius(self) -> float:
        return self.horizon


@dataclass(frozen=True)
class QMetric(Spacetime):
    """q-metric (Zipoy-Voorhees): static, axisymmetric vacuum solution
    with quadrupole ``q`` (Appendix A of arXiv:2202.00086). q = 0 is
    Schwarzschild; for -1 < q < -1 + sqrt(3/2), q /= 0, the source is a
    naked singularity. The surface r = 2m is singular for q /= 0; rays
    are captured just outside it.
    """

    q: float = 0.0
    mid = MID_QMETRIC

    def __post_init__(self):
        if self.q <= -1.0:
            raise ValueError(f"quadrupole must satisfy q > -1, got {self.q}")

    @property
    def par(self) -> np.ndarray:
        return np.array([self.q, 0.0, 0.0, 0.0])

    @property
    def horizon(self) -> float:
        """The singular surface r = 2m (an event horizon only for q = 0);
        exposed under this name for plotting helpers."""
        return 2.0

    @property
    def capture_radius(self) -> float:
        return 2.0

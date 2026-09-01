"""Geometric helpers shared by matter and instruments.

Surfaces live in the pseudo-Cartesian embedding
x = r sin(th) cos(ph), y = r sin(th) sin(ph), z = r cos(th) (exact only
asymptotically; keep sources and screens at r >> M). The spin axis is +z.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _core


def bl_to_cart(r, theta, phi):
    """Boyer-Lindquist / spherical-like -> pseudo-Cartesian embedding."""
    s = np.sin(theta)
    return np.stack([r*s*np.cos(phi), r*s*np.sin(phi), r*np.cos(theta)],
                    axis=-1)


def cart_to_bl(p):
    """Pseudo-Cartesian point -> (r, theta, phi)."""
    x, y, z = p
    r = np.sqrt(x*x + y*y + z*z)
    return r, np.arccos(np.clip(z/r, -1, 1)), np.arctan2(y, x)


def null_momentum(spacetime, point_cart, direction_cart):
    """Null 4-momentum at ``point_cart`` moving along ``direction_cart``.

    The spatial direction is expressed in the ZAMO orthonormal frame
    (coordinate direction mapped through the embedding Jacobian), and
    the photon energy in that frame is normalized to 1. Returns
    ``(y0, p_t, p_phi)`` ready for the Fortran tracer. Works for any
    registered ``Spacetime``.
    """
    r, th, ph = cart_to_bl(np.asarray(point_cart, float))
    d = np.asarray(direction_cart, float)
    d = d/np.linalg.norm(d)

    st, ct = np.sin(th), np.cos(th)
    cp, sp = np.cos(ph), np.sin(ph)
    jac = np.array([[st*cp, r*ct*cp, -r*st*sp],
                    [st*sp, r*ct*sp, r*st*cp],
                    [ct, -r*st, 0.0]])
    dr, dth, dph = np.linalg.solve(jac, d)

    gd = spacetime.metric_cov(r, th)
    g_tt, g_tp, g_rr, g_thth, g_pp = gd
    n = np.array([np.sqrt(g_rr)*dr, np.sqrt(g_thth)*dth, np.sqrt(g_pp)*dph])
    n = n/np.linalg.norm(n)

    gu, _, _ = spacetime.metric_contra(r, th)
    alpha = 1.0/np.sqrt(-gu[0])
    u_t = np.sqrt(-gu[0])            # ZAMO u^t
    u_p = -alpha*gu[1]               # ZAMO u^phi
    p_up_t = u_t
    p_up_r = n[0]/np.sqrt(g_rr)
    p_up_th = n[1]/np.sqrt(g_thth)
    p_up_ph = u_p + n[2]/np.sqrt(g_pp)

    p_t = g_tt*p_up_t + g_tp*p_up_ph
    p_phi = g_tp*p_up_t + g_pp*p_up_ph
    p_r = g_rr*p_up_r
    p_th = g_thth*p_up_th
    y0 = np.array([0.0, r, th, ph, p_r, p_th])
    return y0, float(p_t), float(p_phi)


@dataclass
class PlanarSurface:
    """Finite rectangular plane: ``center`` + span along ``(e1, e2)``,
    where e1 = up x normal (width direction) and e2 completes the basis."""

    center: tuple = (0.0, 0.0, 0.0)
    normal: tuple = (1.0, 0.0, 0.0)
    up: tuple = (0.0, 0.0, 1.0)
    width: float = 40.0
    height: float = 40.0

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("surface width/height must be positive")
        n = np.asarray(self.normal, float)
        self._n = n/np.linalg.norm(n)
        u = np.asarray(self.up, float)
        e1 = np.cross(u, self._n)
        self._e1 = e1/np.linalg.norm(e1)
        self._e2 = np.cross(self._n, self._e1)
        self._c = np.asarray(self.center, float)
        self._d = float(self._n @ self._c)

    @property
    def plane(self):
        """(normal, d) with plane equation n.x = d."""
        return self._n, self._d

    def to_local(self, points):
        """Cartesian points -> in-plane coordinates (u, v)."""
        rel = np.atleast_2d(points) - self._c
        return rel @ self._e1, rel @ self._e2

    def to_world(self, u, v):
        return (self._c + np.multiply.outer(u, self._e1) +
                np.multiply.outer(v, self._e2))

    def corners(self):
        w, h = 0.5*self.width, 0.5*self.height
        return np.array([self.to_world(su*w, sv*h)
                         for su, sv in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])

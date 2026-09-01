"""Plotting helpers reproducing the visual style of arXiv:2202.00086."""
from __future__ import annotations

import numpy as np


def plot_image(img, ax=None, cmap="afmhot", norm_to=None, colorbar=True,
               label=None, vmax=None):
    """Show an intensity map (paper Fig. 13 style).

    ``norm_to``: value that maps to 1.0 (e.g. the max of the a=0 panel);
    default is the image's own maximum.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.2))
    data = img.intensity.T  # -> (ny, nx), origin lower
    scale = norm_to if norm_to else (data.max() or 1.0)
    im = ax.imshow(data/scale, origin="lower", extent=img.extent,
                   cmap=cmap, vmin=0.0, vmax=vmax, aspect="equal",
                   interpolation="bilinear")
    if colorbar:
        ax.figure.colorbar(im, ax=ax, pad=0.02)
    if label:
        ax.text(0.04, 0.08, label, transform=ax.transAxes, color="black",
                fontsize=13, bbox=dict(facecolor="white", alpha=0.9,
                                       boxstyle="round,pad=0.3"))
    return ax


def plot_shadow(img, ax=None, analytic_xy=None):
    """Black captured region on grey background (paper Fig. 6 style);
    optionally overlay the analytic Bardeen rim as a red curve."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    from .api import STATUS_CAPTURED
    mask = (img.status == STATUS_CAPTURED).T.astype(float)
    ax.imshow(mask, origin="lower", extent=img.extent, cmap="binary",
              vmin=0, vmax=1, aspect="equal")
    if analytic_xy is not None:
        ax.plot(analytic_xy[0], analytic_xy[1], "r-", lw=1.2,
                label="Analytic solution")
        ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_orbits_2d(orbits, black_hole=None, plane="xy", ax=None,
                   colors=None, lw=0.9, legend=True):
    """2D projection of ray/particle orbits (paper Fig. 3/14 style).

    ``orbits``: list of entries, each one of
      - the dict returned by ``grayt.trace`` (keys r, theta, phi),
      - a ``grayt.Ray`` (Cartesian ``points``),
      - an (N, 3) Cartesian array,
    optionally wrapped as ``(orbit, label)``.
    ``plane``: "xy" (equatorial projection), "xz" or "yz".
    The event horizon of ``black_hole`` is drawn as a circle at the
    origin.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 5.2))
    comp = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[plane]

    for k, entry in enumerate(orbits):
        label = None
        orbit = entry
        if isinstance(entry, tuple):
            orbit, label = entry
        if isinstance(orbit, dict):
            from .system import bl_to_cart
            pts = bl_to_cart(orbit["r"], orbit["theta"], orbit["phi"])
        elif hasattr(orbit, "points"):
            pts = orbit.points
        else:
            pts = np.asarray(orbit)
        color = colors[k] if colors else None
        ax.plot(pts[:, comp[0]], pts[:, comp[1]], lw=lw, color=color,
                label=label)

    if black_hole is not None:
        th = np.linspace(0, 2*np.pi, 200)
        rh = black_hole.horizon
        ax.fill(rh*np.cos(th), rh*np.sin(th), facecolor="white",
                edgecolor="black", lw=1.2, zorder=3)
    ax.set_aspect("equal")
    labels = {"xy": ("$x$", "$y$"), "xz": ("$x$", "$z$"),
              "yz": ("$y$", "$z$")}[plane]
    ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1])
    if legend and any(isinstance(e, tuple) and e[1] for e in orbits):
        ax.legend(loc="lower right", fontsize=8)
    return ax


def plot_lensing(img, ax=None, mesh_deg=6.0):
    """Celestial-sphere quadrant coloring (paper Figs. 9-12).

    Escaped rays are colored by the quadrant of the sphere they strike;
    captured rays are black. The latitude/longitude mesh (``mesh_deg``
    spacing) is drawn as contour lines of the continuous escape-direction
    fields, which stays smooth even where the deflection gradient is
    steep (a per-pixel band test aliases into dashes there).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    from .api import STATUS_ESCAPED

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    escaped = img.status == STATUS_ESCAPED
    # Fold theta smoothly into [0, pi]; keep phi continuous (unwrapped)
    # for contouring and reduce it only for the quadrant test.
    th_fold = np.arccos(np.clip(np.cos(img.theta_inf), -1.0, 1.0))
    ph = np.mod(img.phi_inf, 2*np.pi)

    # Quadrants (Fig. 9): top (theta < pi/2): green then red with phi;
    # bottom: blue then yellow.
    quad = np.zeros_like(th_fold, dtype=int)
    top = th_fold < np.pi/2
    east = ph < np.pi
    quad[top & east] = 1      # green
    quad[top & ~east] = 2     # red
    quad[~top & east] = 3     # blue
    quad[~top & ~east] = 4    # yellow
    quad[~escaped] = 0        # captured/disk/failed -> black

    colors = ["black", "#2ca02c", "#d62728", "#1f77b4", "#ffdf22"]
    ax.imshow(quad.T, origin="lower", extent=img.extent,
              cmap=ListedColormap(colors), vmin=0, vmax=4, aspect="equal",
              interpolation="nearest")

    # Mesh as contours of the (masked) escape-direction fields.
    step = np.deg2rad(mesh_deg)
    nx, ny = th_fold.shape
    xs = np.linspace(img.extent[0], img.extent[1], nx, endpoint=False) + \
        0.5*(img.extent[1] - img.extent[0])/nx
    ys = np.linspace(img.extent[2], img.extent[3], ny, endpoint=False) + \
        0.5*(img.extent[3] - img.extent[2])/ny
    for field in (th_fold, img.phi_inf):
        f = np.ma.masked_where(~escaped, field)
        lo = np.floor(f.min()/step)*step
        hi = np.ceil(f.max()/step)*step
        levels = np.arange(lo, hi + 0.5*step, step)
        ax.contour(xs, ys, f.T, levels=levels, colors="black",
                   linewidths=0.4, antialiased=True)
    return ax

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


def plot_lensing(img, ax=None, mesh_deg=6.0):
    """Celestial-sphere quadrant coloring (paper Figs. 9-12).

    Escaped rays are colored by the quadrant of the sphere they strike;
    captured rays are black. A black mesh with ``mesh_deg`` spacing in
    latitude/longitude conveys the distortion.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    from .api import STATUS_ESCAPED

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    th = np.mod(img.theta_inf, 2*np.pi)
    th = np.where(th > np.pi, 2*np.pi - th, th)  # fold to [0, pi]
    ph = np.mod(img.phi_inf, 2*np.pi)

    # Quadrants (Fig. 9): top (theta < pi/2): green then red with phi;
    # bottom: blue then yellow.
    quad = np.zeros_like(th, dtype=int)
    top = th < np.pi/2
    east = ph < np.pi
    quad[top & east] = 1      # green
    quad[top & ~east] = 2     # red
    quad[~top & east] = 3     # blue
    quad[~top & ~east] = 4    # yellow

    # Mesh lines of constant latitude/longitude
    step = np.deg2rad(mesh_deg)
    on_mesh = ((np.mod(th, step) < 0.15*step) |
               (np.mod(ph, step) < 0.15*step))
    quad[on_mesh] = 5         # black mesh line
    quad[img.status != STATUS_ESCAPED] = 0  # captured/disk/failed -> black

    colors = ["black", "#2ca02c", "#d62728", "#1f77b4", "#ffdf22", "black"]
    ax.imshow(quad.T, origin="lower", extent=img.extent,
              cmap=ListedColormap(colors), vmin=0, vmax=5, aspect="equal",
              interpolation="nearest")
    return ax

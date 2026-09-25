"""House style for GRAVTRACER figures: black field, classical serif type.

Every library plot that creates its own figure is drawn inside
:func:`context`, so figures look the same whether they come from
``img.plot()``, ``System.visualize3d()`` or the example scripts. Call
:func:`use` once to apply the style to your own matplotlib figures too.

The typeface is STIX (a Times-like face shipped with matplotlib, so the
look is identical on every machine), with STIX mathtext so symbols such
as $r_{\\rm ISCO}$ match the surrounding text.
"""
from __future__ import annotations

from contextlib import contextmanager

from cycler import cycler

BG = "#000000"        # figure and axes field
INK = "#ece6da"       # titles, primary text (warm paper white)
MUTED = "#9a958c"     # tick labels, captions
FAINT = "#3a3835"     # spines, grid, frames
AMBER = "#f2b85a"     # highlighted rays, instrument outlines
EMBER = "#e0643a"     # captured rays
TEAL = "#7fc8c0"      # escaping rays

SERIF = ["STIXGeneral", "STIX Two Text", "Times New Roman", "Liberation Serif",
         "DejaVu Serif"]

RC = {
    "figure.facecolor": BG,
    "figure.edgecolor": BG,
    "savefig.facecolor": BG,
    "savefig.edgecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": FAINT,
    "axes.labelcolor": MUTED,
    "axes.titlecolor": INK,
    "axes.titlesize": 13,
    "axes.titleweight": "normal",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.labelsize": 11,
    "axes.linewidth": 0.6,
    "axes.grid": False,
    "axes.prop_cycle": cycler(
        color=[AMBER, TEAL, EMBER, "#b7a3e0", "#c9d77a", "#e8e2d4"]),
    "grid.color": FAINT,
    "grid.linewidth": 0.5,
    "xtick.color": FAINT,
    "ytick.color": FAINT,
    "xtick.labelcolor": MUTED,
    "ytick.labelcolor": MUTED,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "text.color": INK,
    "font.family": "serif",
    "font.serif": SERIF,
    "font.size": 11,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": True,
    "legend.frameon": False,
    "legend.labelcolor": INK,
    "legend.fontsize": 10,
    "image.cmap": "afmhot",
    "savefig.dpi": 170,
}


def use():
    """Apply the GRAVTRACER style to all subsequent matplotlib figures."""
    import matplotlib as mpl

    mpl.rcParams.update(RC)


@contextmanager
def context():
    """Temporarily apply the style (used internally by plotting helpers)."""
    import matplotlib as mpl

    with mpl.rc_context(RC):
        yield


def label(ax, text, loc=(0.03, 0.05), size=11):
    """Small instrument-style annotation in a corner of ``ax``."""
    return ax.text(*loc, text, transform=ax.transAxes, color=INK,
                   fontsize=size, ha="left", va="bottom",
                   bbox=dict(facecolor=BG, edgecolor=FAINT, lw=0.6,
                             alpha=0.75, boxstyle="square,pad=0.35"))


def frame(ax):
    """Thin hairline frame and inward ticks for an image axis."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(FAINT)
        spine.set_linewidth(0.6)
    ax.tick_params(which="both", top=True, right=True)
    return ax


def dark_3d(ax, axes_visible=False):
    """Black panes, no default grey box; optional hairline axes."""
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0, 0, 0, 0))
        axis.pane.set_edgecolor(FAINT)
        axis._axinfo["grid"].update(color=FAINT, linewidth=0.4)
        axis.label.set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    if not axes_visible:
        ax.set_axis_off()
    return ax


def textured_plane(ax, surface, image, cells=220, alpha=1.0, zorder=1,
                   edge=INK):
    """Paint ``image`` onto a :class:`~grayt.PlanarSurface` in a 3D axis.

    Row 0 of the image is drawn at the ``up`` edge, column 0 at ``-e1``,
    matching :meth:`grayt.ImageSource.sample_color`. The texture is
    box-averaged to at most ``cells`` facets along its longer side.
    """
    import numpy as np

    image = np.asarray(image, float)
    h, w = image.shape[:2]
    step = max(1, int(np.ceil(max(h, w) / cells)))
    hh, ww = h // step * step, w // step * step
    tex = image[:hh, :ww, :3].reshape(hh // step, step, ww // step, step, 3)
    tex = np.clip(tex.mean(axis=(1, 3)), 0, 1)
    nv, nu = tex.shape[:2]
    u = np.linspace(-surface.width / 2, surface.width / 2, nu + 1)
    v = np.linspace(surface.height / 2, -surface.height / 2, nv + 1)
    xyz = surface.to_world(*np.meshgrid(u, v))
    colors = np.ones((nv + 1, nu + 1, 4))
    colors[:nv, :nu, :3] = tex
    colors[..., 3] = alpha
    ax.plot_surface(*np.moveaxis(xyz, -1, 0), facecolors=colors,
                    rstride=1, cstride=1, shade=False, linewidth=0,
                    antialiased=False, zorder=zorder)
    if edge is not None:
        c = surface.corners()
        ax.plot(*np.vstack((c, c[:1])).T, color=edge, lw=0.6,
                zorder=zorder + 0.1)

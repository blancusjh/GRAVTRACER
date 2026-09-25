"""House plotting style: black field, serif type, textured scene cards."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex

import grayt
from grayt import style


def _tiny_image():
    return grayt.Image(
        intensity=np.ones((8, 4)), g=np.ones((8, 4)), r_hit=np.zeros((8, 4)),
        status=np.full((8, 4), 2), herr=np.zeros((8, 4)),
        theta_inf=np.zeros((8, 4)), phi_inf=np.zeros((8, 4)),
        extent=(-1, 1, -1, 1))


def test_owned_figures_use_the_house_style_without_leaking_it():
    before = dict(matplotlib.rcParams)
    ax = _tiny_image().plot(label="$a=0$")
    assert to_hex(ax.figure.get_facecolor()) == style.BG
    assert ax.title.get_fontfamily() == ["serif"]
    assert dict(matplotlib.rcParams) == before
    plt.close("all")


def test_helpers_leave_a_caller_axis_alone():
    fig, ax = plt.subplots(facecolor="white")
    grayt.plot_image(_tiny_image(), ax, colorbar=False)
    assert to_hex(fig.get_facecolor()) == "#ffffff"
    plt.close(fig)


def test_visualize3d_paints_sources_with_their_image():
    image = np.zeros((6, 9, 3))
    image[..., 0] = 1.0
    source = grayt.ImageSource(center=(-40, 0, 0), normal=(1, 0, 0),
                               width=9, height=6, image=image)
    system = grayt.System(physical=grayt.PhysicalSystem(
        spacetime=grayt.BlackHole(a=0.5), sources=[source]))
    ax = system.visualize3d(show_rays=False)
    faces = np.vstack([c.get_facecolor() for c in ax.collections
                       if len(c.get_facecolor()) == 54])
    assert np.allclose(faces[:, :3], (1, 0, 0))
    plt.close("all")

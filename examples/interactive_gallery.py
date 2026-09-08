"""Embed offline interactive 3D scenes into an existing model gallery.

PYTHONPATH=python python examples/interactive_gallery.py --output output/stationary_models
Requires the optional `interactive` extra (Plotly). No web server is needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

import grayt
from grayt.geometry import bl_to_cart

BACKGROUND = "#0d1420"
CONFIG = {
    "responsive": True,
    "displaylogo": False,
    "scrollZoom": True,
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


def scene_layout(radius=50):
    """Equal coordinate scales and a z-up orbit camera."""
    axis = dict(
        range=[-radius, radius],
        backgroundcolor=BACKGROUND,
        gridcolor="#29384c",
        zerolinecolor="#475b76",
        showbackground=True,
        tickfont=dict(size=11),
    )
    return dict(
        xaxis=dict(title="x / M", **axis),
        yaxis=dict(title="y / M", **axis),
        zaxis=dict(title="z / M", **axis),
        aspectmode="cube",
        dragmode="orbit",
        camera=dict(eye=dict(x=1.2, y=1.5, z=0.85), up=dict(x=0, y=0, z=1)),
    )


def layout(radius=50):
    return dict(
        template="plotly_dark",
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        font=dict(color="#dce4ef", family="system-ui, sans-serif"),
        height=620,
        margin=dict(l=0, r=0, t=85, b=25),
        scene=scene_layout(radius),
        legend=dict(orientation="h", x=0, y=0, yanchor="bottom"),
        uirevision="preserve-viewpoint",
    )


def sphere(radius, *, star=False):
    phi, theta = np.meshgrid(np.linspace(0, 2 * np.pi, 49), np.linspace(0, np.pi, 33))
    xyz = bl_to_cart(radius, theta, phi)
    if star:
        dot = np.cos(theta) * np.cos(0.8) + np.sin(theta) * np.sin(0.8) * np.cos(
            phi - 0.5
        )
        color = np.exp(-(1 - dot) / 0.035) + 0.5 * np.exp(-(1 + dot) / 0.035)
        scale = [[0, "#8b351c"], [0.4, "#ed8625"], [1, "#fff1a1"]]
    else:
        color = np.zeros_like(theta)
        scale = [[0, "#202a3a"], [1, "#202a3a"]]
    return go.Surface(
        x=xyz[..., 0],
        y=xyz[..., 1],
        z=xyz[..., 2],
        surfacecolor=color,
        cmin=0,
        cmax=1,
        colorscale=scale,
        showscale=False,
        showlegend=True,
        name="Stellar surface" if star else "Inner boundary",
        hoverinfo="name",
        lighting=dict(ambient=0.7, diffuse=0.6),
    )


def annulus(inner, outer):
    phi, radius = np.meshgrid(
        np.linspace(0, 2 * np.pi, 81), np.linspace(inner, outer, 12)
    )
    return go.Surface(
        x=radius * np.cos(phi),
        y=radius * np.sin(phi),
        z=np.zeros_like(radius),
        surfacecolor=np.zeros_like(radius),
        colorscale=[[0, "#cf772b"], [1, "#cf772b"]],
        opacity=0.28,
        showscale=False,
        showlegend=True,
        name="Disk reference plane",
        hoverinfo="name",
    )


def ray_groups(spacetime, camera):
    """Sample vacuum null geodesics with the library's shared Fortran solver."""
    coordinates = {
        "Captured / surface hit": [[], [], []],
        "Escaping rays": [[], [], []],
    }
    colors = {"Captured / surface hit": "#ffb454", "Escaping rays": "#69d4ff"}
    for x in np.linspace(-12, 12, 5):
        for y in np.linspace(-8, 8, 5):
            initial, pt, pp = grayt.camera_ray(spacetime, camera, x, y)
            ray = grayt.trace(
                spacetime,
                initial,
                pt,
                pp,
                h0=-0.1,
                lambda_max=140,
                max_step=0.45,
                escape_radius=48,
                rtol=1e-9,
                atol=1e-11,
            )
            if not len(ray.r):
                raise RuntimeError("No samples returned for an interactive ray")
            boundary = spacetime.capture_radius + (
                0.01 if spacetime.mid in (1, 2) else 0
            )
            if ray.r[-1] <= boundary:
                key = "Captured / surface hit"
            elif ray.r[-1] >= 48:
                key = "Escaping rays"
            else:
                raise RuntimeError("Interactive ray exhausted its integration budget")
            points = np.vstack(
                (
                    bl_to_cart(
                        camera.r, np.deg2rad(camera.theta), np.deg2rad(camera.phi)
                    ),
                    ray.points,
                )
            )
            for axis, values in enumerate(coordinates[key]):
                values.extend(np.round(points[:, axis], 6).tolist() + [None])
    return [
        go.Scatter3d(
            x=xyz[0],
            y=xyz[1],
            z=xyz[2],
            mode="lines",
            name=name,
            line=dict(color=colors[name], width=3),
            connectgaps=False,
            hovertemplate="x=%{x:.2f} M<br>y=%{y:.2f} M<br>z=%{z:.2f} M<extra>%{fullData.name}</extra>",
        )
        for name, xyz in coordinates.items()
    ]


def geodesic_view():
    """One WebGL scene with five selectable, independently traced metrics."""
    camera = grayt.Camera(r=45, theta=70, phi=25)
    cases = [
        ("Kerr a=0.8", grayt.BlackHole(0.8)),
        ("Schwarzschild", grayt.BlackHole(0)),
        ("Spherical star R=5 M", grayt.SphericalStar(5)),
        ("Reissner–Nordström Q/M=0.8", grayt.ReissnerNordstrom(0.8)),
        ("Zipoy–Voorhees q=0.5; ADM mass=1.5", grayt.QMetric(0.5)),
    ]
    fig = go.Figure()
    groups = []
    observer = bl_to_cart(camera.r, np.deg2rad(camera.theta), np.deg2rad(camera.phi))
    for label, spacetime in cases:
        start = len(fig.data)
        fig.add_trace(
            sphere(
                spacetime.capture_radius,
                star=isinstance(spacetime, grayt.SphericalStar),
            )
        )
        if isinstance(spacetime, grayt.BlackHole):
            fig.add_trace(annulus(spacetime.isco, 20))
        fig.add_traces(ray_groups(spacetime, camera))
        fig.add_trace(
            go.Scatter3d(
                x=[observer[0]],
                y=[observer[1]],
                z=[observer[2]],
                mode="markers+text",
                text=["Observer"],
                textposition="top center",
                name="Observer",
                marker=dict(size=5, color="white"),
                showlegend=False,
            )
        )
        groups.append((label, start, len(fig.data)))
    buttons = []
    for label, start, end in groups:
        visible = [start <= i < end for i in range(len(fig.data))]
        buttons.append(
            dict(
                label=label,
                method="update",
                args=[
                    {"visible": visible},
                    {"title.text": label},
                ],
            )
        )
    for i, trace in enumerate(fig.data):
        trace.visible = groups[0][1] <= i < groups[0][2]
    fig.update_layout(
        **layout(),
        title=dict(text=cases[0][0], x=0.03, y=0.97),
        updatemenus=[
            dict(
                buttons=buttons,
                x=1,
                xanchor="right",
                y=1.07,
                yanchor="top",
                bgcolor="#23334a",
                bordercolor="#536984",
                font=dict(color="#dce4ef"),
            )
        ],
    )
    return fig


def volume_view(path):
    """Resample the imported emissivity onto a Cartesian grid for isosurfaces."""
    volume = grayt.VolumeGrid.load(path)
    spacetime = grayt.BlackHole(0.5)  # this gallery's archived snapshot geometry
    axis = np.linspace(-24, 24, 41)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    radius = np.sqrt(x * x + y * y + z * z)
    inside = (radius >= volume.r_in) & (radius <= volume.r_out)
    values = np.zeros_like(radius)
    theta = np.arccos(np.clip(z[inside] / radius[inside], -1, 1))
    j, _, _ = volume.sample(
        spacetime, radius[inside], theta, np.arctan2(y[inside], x[inside]), 0
    )
    values[inside] = j
    maximum = float(values.max())
    fig = go.Figure(
        go.Isosurface(
            x=x.ravel(),
            y=y.ravel(),
            z=z.ravel(),
            value=values.ravel(),
            isomin=0.08 * maximum,
            isomax=0.95 * maximum,
            surface_count=5,
            opacity=0.35,
            colorscale="Inferno",
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorbar=dict(
                title="Local emissivity", thickness=14, len=0.7, tickformat=".1e"
            ),
            name="Prescribed emissivity",
            hovertemplate="j=%{value:.2e}<extra>Imported gray torus</extra>",
        )
    )
    fig.add_trace(sphere(spacetime.horizon))
    fig.update_layout(
        **layout(26), title=dict(text="Imported 3D emissivity | Kerr a=0.5", x=0.03)
    )
    return fig


def write_interactive_views(output):
    """Write an HTML fragment with one bundled runtime and embedded datasets."""
    output = Path(output)
    views = [("geodesic-view", geodesic_view())]
    if (output / "radiation_snapshot.npz").exists():
        views.append(("volume-view", volume_view(output / "radiation_snapshot.npz")))
    chunks = [
        """<section id="interactive-views" class="interactive-section">
<h2>Coordinate diagrams</h2>
<p>Drag to rotate · scroll or pinch to zoom · use the toolbar to pan or reset.
Choose a metric from the menu and click legend entries to show or hide layers.</p>
<p>These are coordinate views of computed vacuum geodesics. The translucent disk is a geometric reference;
these rays continue through its plane. Axis lengths are coordinate distances in units of M.</p>
"""
    ]
    for i, (identifier, figure) in enumerate(views):
        if identifier == "volume-view":
            chunks.append(
                "<p>The volume view shows imported local emissivity, with threshold surfaces that reveal its 3D structure.</p>"
            )
        chunks.append('<div class="interactive-plot">')
        chunks.append(
            pio.to_html(
                figure,
                full_html=False,
                include_plotlyjs=(i == 0),
                include_mathjax=False,
                config=CONFIG,
                div_id=identifier,
            )
        )
        chunks.append("</div>")
        # Export the plotted data independently of the presentation.
        figure.write_json(output / f"{identifier}.json")
    chunks.append("</section>")
    (output / "interactive_views.html").write_text("\n".join(chunks), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/stationary_models"))
    args = parser.parse_args()
    if not (args.output / "manifest.json").is_file():
        parser.error("generate the model gallery first")
    write_interactive_views(args.output)
    from model_gallery import build_index

    build_index(args.output)
    print(f"Embedded interactive views: {(args.output / 'index.html').resolve()}")


if __name__ == "__main__":
    main()

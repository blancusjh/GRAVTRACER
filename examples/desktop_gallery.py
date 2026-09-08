"""Open a native viewer for one allowlisted gallery example.

The macOS URL launcher passes gravtracer://view/<preset> here. URLs cannot
supply code, paths, shell options, or arbitrary renderer parameters.
"""

import argparse
from urllib.parse import urlsplit

import grayt
from model_gallery import models


def presets():
    cases = {row[0]: row for row in models()}
    cases["kerr_camera_orbit"] = (
        "kerr_camera_orbit",
        "Kerr camera orbit",
        grayt.BlackHole(0.8),
        grayt.PageThorneDisk(grayt.BlackHole(0.8)),
        None,
        45,
        "Kerr / Page-Thorne",
    )
    cases["stellar_camera_orbit"] = replace_case(
        cases["star_spots"], "stellar_camera_orbit"
    )
    cases["quadrupole_camera_orbit"] = (
        "quadrupole_camera_orbit",
        "Quadrupole camera orbit",
        grayt.QMetric(0.5),
        None,
        None,
        45,
        "q-metric / sky",
    )
    return cases


def replace_case(case, name):
    return (name, *case[1:5], 45, case[6])


def parse_url(url, allowed):
    parsed = urlsplit(url)
    name = parsed.path.removeprefix("/")
    if (
        parsed.scheme != "gravtracer"
        or parsed.netloc != "view"
        or parsed.query
        or parsed.fragment
        or name not in allowed
    ):
        raise ValueError("unknown GRAVTRACER viewer link")
    return name


def open_example(name):
    _, _, st, disk, surface, inclination, _ = presets()[name]
    cpu = st.mid not in (1, 2) or surface is not None
    camera = grayt.Camera(
        r=100,
        theta=inclination,
        x=(-26, 26),
        y=(-16, 16),
        resolution=(480, 300) if cpu else (960, 600),
    )
    viewer = grayt.view(
        st,
        disk,
        camera,
        surface=surface,
        exposure=5000,
        escape_radius=200,
        background="black",
        mode="composite",
        backend="pyside6",
        preview_resolution=(80, 50) if cpu else (256, 160),
        run=False,
    )
    print(
        f"Opened {name} | {viewer.renderer.device_name} | background=black; B cycles backgrounds",
        flush=True,
    )
    return viewer.run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url")
    group.add_argument("--model")
    args = parser.parse_args()
    allowed = presets()
    try:
        name = parse_url(args.url, allowed) if args.url else args.model
        if name not in allowed:
            raise ValueError("unknown gallery example")
    except ValueError as error:
        parser.error(str(error))
    open_example(name)


if __name__ == "__main__":
    main()

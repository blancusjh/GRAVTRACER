"""Command-line interface: ``python -m grayt render scene.yml -o out.png``."""
from __future__ import annotations

import argparse
import sys
import time


def main(argv=None):
    try:
        return _main(argv)
    except (ValueError, FileNotFoundError, KeyError, TypeError,
            RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _main(argv=None):
    parser = argparse.ArgumentParser(
        prog="gravtracer",
        description="GRAVTRACER: relativistic ray tracing around compact objects")
    from . import __version__
    parser.add_argument("--version", action="version",
                        version=f"gravtracer {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_render = sub.add_parser("render", help="render a scene from a YAML config")
    p_render.add_argument("config", help="scene YAML file")
    p_render.add_argument("-o", "--output", default="grayt_out.png",
                          help="output PNG (and .npz alongside)")
    p_render.add_argument("--res", type=int, nargs=2, metavar=("NX", "NY"),
                          help="override resolution")
    p_render.add_argument("--norm", type=float, default=None,
                          help="intensity value mapped to 1.0")
    p_render.add_argument("--label", default=None, help="panel label text")
    p_render.add_argument("--npz", action="store_true",
                          help="also save raw maps to .npz")

    p_shadow = sub.add_parser("shadow", help="render the black hole shadow")
    p_shadow.add_argument("-a", "--spin", type=float, default=0.98)
    p_shadow.add_argument("-o", "--output", default="shadow.png")
    p_shadow.add_argument("--res", type=int, nargs=2, default=(625, 625),
                          metavar=("NX", "NY"))
    p_shadow.add_argument("--fov", type=float, default=8.0,
                          help="half-size of the image plane in M")
    p_shadow.add_argument("--r0", type=float, default=1000.0)
    p_shadow.add_argument("--theta", type=float, default=90.0,
                          help="observer inclination in degrees")

    p_view = sub.add_parser("view", help="open the interactive GPU camera view")
    p_view.add_argument("-a", "--spin", type=float, default=0.95)
    p_view.add_argument("--theta", type=float, default=85.0,
                        help="observer inclination in degrees")
    p_view.add_argument("--phi", type=float, default=0.0,
                        help="observer azimuth in degrees")
    p_view.add_argument("--r0", type=float, default=1000.0,
                        help="observer radius in M")
    p_view.add_argument("--fov", type=float, nargs=2, default=(24.0, 12.0),
                        metavar=("X", "Y"),
                        help="image-plane half-width and half-height in M")
    p_view.add_argument("--res", type=int, nargs=2, default=(512, 256),
                        metavar=("NX", "NY"), help="final resolution")
    p_view.add_argument("--preview-res", type=int, nargs=2,
                        default=(128, 64), metavar=("NX", "NY"),
                        help="resolution used while interacting")
    p_view.add_argument("--disk-out", type=float, default=20.0,
                        help="outer disk radius in M")
    p_view.add_argument("--l0", type=float, default=1.8,
                        help="disk specific angular momentum")
    p_view.add_argument("--no-disk", action="store_true",
                        help="show a bare black-hole shadow")
    p_view.add_argument("--mode", choices=("composite", "intensity",
                                            "lensing", "shadow"),
                        default="intensity")
    p_view.add_argument("--precision", choices=("auto", "fp32", "fp64"),
                        default="auto")
    p_view.add_argument("--device", type=int, default=None,
                        help="OpenCL device index (see grayt.gpu.devices())")
    p_view.add_argument("--gui-backend", default=None,
                        help="VisPy GUI backend (default: auto; pyside6 recommended)")
    p_view.add_argument("--background", choices=("black", "celestial", "grid"), default="black",
                        help="initial background; B cycles black, celestial, grid")
    p_view.add_argument("--sky", help="optional equirectangular celestial map image")

    args = parser.parse_args(argv)

    t0 = time.time()
    if args.command == "render":
        import matplotlib
        matplotlib.use("Agg")
        from . import System

        sys3 = System.from_yaml(args.config)
        if args.res:
            import dataclasses
            sys3.cameras[0] = dataclasses.replace(sys3.cameras[0],
                                                  resolution=tuple(args.res))
        img = sys3.render()
        st = sys3.physical.spacetime
        default_label = (f"$a = {st.a:g}$" if hasattr(st, "a")
                         else f"$q = {st.q:g}$" if hasattr(st, "q")
                         else getattr(st, "name", type(st).__name__))
        label = args.label or default_label
        from .scene import SceneImage
        if isinstance(img, SceneImage):
            if args.norm is not None:
                raise ValueError("RGB scenes use YAML exposure; --norm is for legacy intensity maps")
            ax = img.plot()
            ax.set_title(label)
        else:
            ax = img.plot(norm_to=args.norm, label=label)
        ax.figure.savefig(args.output, dpi=200, bbox_inches="tight")
        if args.npz:
            img.save(args.output.rsplit(".", 1)[0] + ".npz")
    elif args.command == "shadow":
        import matplotlib
        matplotlib.use("Agg")
        from . import BlackHole, Camera, shadow as render_shadow

        bh = BlackHole(a=args.spin)
        cam = Camera(r=args.r0, theta=args.theta, x=(-args.fov, args.fov),
                     y=(-args.fov, args.fov), resolution=tuple(args.res))
        img = render_shadow(bh, cam)
        from .plotting import plot_shadow
        ax = plot_shadow(img)
        ax.figure.savefig(args.output, dpi=200, bbox_inches="tight")

    elif args.command == "view":
        from . import BlackHole, Camera, ThinDisk, view

        bh = BlackHole(a=args.spin)
        cam = Camera(
            r=args.r0, theta=args.theta, phi=args.phi,
            x=(-args.fov[0], args.fov[0]),
            y=(-args.fov[1], args.fov[1]), resolution=tuple(args.res))
        disk = None if args.no_disk else ThinDisk(
            r_out=args.disk_out, l0=args.l0)
        from .sky import CelestialSky
        sky = CelestialSky(args.sky) if args.sky else None
        view(bh, disk, cam, preview_resolution=tuple(args.preview_res),
             mode=args.mode, precision=args.precision, device=args.device,
             backend=args.gui_backend, background=args.background, sky=sky)
        return 0

    print(f"wrote {args.output} in {time.time() - t0:.1f}s", file=sys.stderr)
    return 0

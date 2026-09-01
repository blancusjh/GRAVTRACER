"""Command-line interface: ``python -m grayt render scene.yml -o out.png``."""
from __future__ import annotations

import argparse
import sys
import time


def main(argv=None):
    try:
        return _main(argv)
    except (ValueError, FileNotFoundError, KeyError, TypeError) as exc:
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

    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import BlackHole, Camera, System, shadow as render_shadow

    t0 = time.time()
    if args.command == "render":
        sys3 = System.from_yaml(args.config)
        if args.res:
            import dataclasses
            sys3.cameras[0] = dataclasses.replace(sys3.cameras[0],
                                                  resolution=tuple(args.res))
        img = sys3.render()
        st = sys3.physical.spacetime
        default_label = (f"$a = {st.a:g}$" if hasattr(st, "a")
                         else f"$q = {st.q:g}$")
        label = args.label or default_label
        ax = img.plot(norm_to=args.norm, label=label)
        ax.figure.savefig(args.output, dpi=200, bbox_inches="tight")
        if args.npz:
            img.save(args.output.rsplit(".", 1)[0] + ".npz")
    elif args.command == "shadow":
        bh = BlackHole(a=args.spin)
        cam = Camera(r=args.r0, theta=args.theta, x=(-args.fov, args.fov),
                     y=(-args.fov, args.fov), resolution=tuple(args.res))
        img = render_shadow(bh, cam)
        from .plotting import plot_shadow
        ax = plot_shadow(img)
        ax.figure.savefig(args.output, dpi=200, bbox_inches="tight")

    print(f"wrote {args.output} in {time.time() - t0:.1f}s", file=sys.stderr)
    return 0

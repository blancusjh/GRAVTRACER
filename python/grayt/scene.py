"""Declarative scene description (YAML) for the CLI and reproducibility."""
from __future__ import annotations

from dataclasses import dataclass

from .api import BlackHole, Camera, ThinDisk, render


@dataclass
class Scene:
    black_hole: BlackHole
    camera: Camera
    disk: ThinDisk | None = None
    method: str = "rkdp45"
    rtol: float = 1e-8
    atol: float = 1e-10

    @classmethod
    def from_yaml(cls, path) -> "Scene":
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f)
        bh = BlackHole(**cfg.get("black_hole", {}))
        cam_cfg = dict(cfg.get("camera", {}))
        for key in ("x", "y", "resolution"):
            if key in cam_cfg:
                cam_cfg[key] = tuple(cam_cfg[key])
        cam = Camera(**cam_cfg)
        disk = None
        if "disk" in cfg and cfg["disk"] is not None:
            disk_cfg = dict(cfg["disk"])
            if disk_cfg.get("r_in") == "isco":
                disk_cfg["r_in"] = None
            disk = ThinDisk(**disk_cfg)
        integ = cfg.get("integrator", {})
        return cls(black_hole=bh, camera=cam, disk=disk,
                   method=integ.get("method", "rkdp45"),
                   rtol=float(integ.get("rtol", 1e-8)),
                   atol=float(integ.get("atol", 1e-10)))

    def render(self, **overrides):
        kwargs = dict(method=self.method, rtol=self.rtol, atol=self.atol)
        kwargs.update(overrides)
        return render(self.black_hole, self.camera, self.disk, **kwargs)

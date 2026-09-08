"""Register a macOS native-viewer URL handler for the local HTML gallery.

Run with the Python environment containing grayt, OpenCL, VisPy and PySide6.
Re-run after moving the repository or its Python environment.
"""

import argparse
import json
import plistlib
import subprocess
import sys
import tempfile
from pathlib import Path


def install(destination):
    if sys.platform != "darwin":
        raise RuntimeError("this URL-handler installer is for macOS")
    root = Path(__file__).resolve().parents[1]
    launcher = root / "examples/desktop_gallery.py"
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    # AppleScript's quoted form performs shell quoting at runtime. The URL
    # is passed as one argument and validated by desktop_gallery.parse_url.
    def literal(value):
        return json.dumps(str(value))

    script = f"""
on open location targetURL
    set pythonPath to {literal(sys.executable)}
    set launcherPath to {literal(launcher)}
    set modulePath to {literal(root / "python")}
    set logPath to {literal(root / "output/desktop_viewer.log")}
    do shell script "PYTHONPATH=" & quoted form of modulePath & " " & quoted form of pythonPath & " " & quoted form of launcherPath & " --url " & quoted form of targetURL & " >> " & quoted form of logPath & " 2>&1 &"
end open location
"""
    with tempfile.TemporaryDirectory(prefix="gravtracer-launcher-") as temporary:
        source = Path(temporary) / "launcher.applescript"
        source.write_text(script)
        subprocess.run(["osacompile", "-o", str(destination), str(source)], check=True)
    plist_path = destination / "Contents/Info.plist"
    with plist_path.open("rb") as stream:
        plist = plistlib.load(stream)
    plist.update(
        {
            "CFBundleIdentifier": "org.gravtracer.desktop-viewer",
            "CFBundleName": "GRAVTRACER Viewer",
            "CFBundleURLTypes": [
                {
                    "CFBundleURLName": "GRAVTRACER viewer",
                    "CFBundleURLSchemes": ["gravtracer"],
                }
            ],
        }
    )
    with plist_path.open("wb") as stream:
        plistlib.dump(plist, stream)
    subprocess.run(["codesign", "--force", "--sign", "-", str(destination)], check=True)
    registrar = "/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
    subprocess.run([registrar, "-f", str(destination)], check=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--app", type=Path, default=Path("output/GRAVTRACER Viewer.app")
    )
    args = parser.parse_args()
    app = install(args.app)
    gallery = Path(__file__).resolve().parents[1] / "output/stationary_models"
    gallery.mkdir(parents=True, exist_ok=True)
    (gallery / "desktop_launcher.json").write_text(
        json.dumps({"app": str(app), "scheme": "gravtracer"}, indent=2)
    )
    print(f"Registered {app}")

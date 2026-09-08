"""Exercise the generated file:// observer with Playwright, fully offline.

Optional tooling: pip install playwright, then playwright install chromium;
or pass --chrome /path/to/an/existing/Chrome executable. Writes screenshots
and a JSON report into the generated gallery directory.
"""

import argparse
import base64
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def validate(output, chrome=None):
    errors, external = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=chrome,
            headless=True,
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
        )
        context = browser.new_context(
            viewport={"width": 1280, "height": 1000},
            offline=True,
            device_scale_factor=1,
        )
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "request",
            lambda request: (
                external.append(request.url)
                if request.url.startswith(("http:", "https:"))
                else None
            ),
        )
        page.goto((output / "index.html").as_uri(), wait_until="domcontentloaded")
        native_links = page.locator("a.desktop-launch").evaluate_all(
            "links=>links.map(a=>a.href)"
        )
        if native_links:
            assert len(native_links) >= 14
            assert all(link.startswith("gravtracer://view/") for link in native_links)
            assert not page.workers  # the optional CPU preview has not started
            page.goto(
                (output / "observer_preview.html").as_uri(),
                wait_until="domcontentloaded",
            )

        def ready():
            page.wait_for_function(
                "document.getElementById('observer-view').dataset.rendering==='false' && "
                "document.getElementById('observer-status').textContent.startsWith('Ready')",
                timeout=60000,
            )

        def state():
            return page.eval_on_selector("#observer-view", "e=>({...e.dataset})")

        def snapshot(name):
            data = page.eval_on_selector(
                "#observer-canvas",
                "c=>c.toDataURL('image/png').split(',')[1]",
            )
            (output / name).write_bytes(base64.b64decode(data))

        ready()
        snapshot("observer_reference_check.png")
        page.locator("#observer-view").scroll_into_view_if_needed()
        page.screenshot(path=str(output / "observer_desktop_preview.png"))
        box = page.locator("#observer-canvas").bounding_box()
        x, y = box["x"] + box["width"] * 0.6, box["y"] + box["height"] * 0.5

        def drag(dx, dy):
            page.mouse.move(x, y)
            page.mouse.down()
            page.mouse.move(x + dx, y + dy, steps=10)
            page.mouse.up()
            ready()

        drag(60, 0)
        assert abs(float(state()["phi"]) - 345) < 0.1
        snapshot("observer_azimuth_check.png")
        drag(0, -50)
        assert abs(float(state()["theta"]) - 75) < 0.1
        page.mouse.wheel(0, -200)
        ready()
        assert page.locator("#observer-position").inner_text().endswith("1.22×")
        page.locator("#observer-model").select_option(label="Kerr · a = 0.95")
        ready()
        assert state()["model"] == "kerr_a0.95"
        page.locator("#observer-play").click()
        page.wait_for_timeout(1000)
        page.locator("#observer-play").click()
        ready()
        assert float(state()["phi"]) != 345
        with page.expect_download() as download:
            page.locator("#observer-save").click()
        assert download.value.suggested_filename.startswith("kerr_a0.95_i75.0_phi")
        page.locator("#observer-reset").click()
        ready()
        assert state()["theta"] == "85"
        if native_links:
            page.locator("#observer-model").select_option(
                label="Spherical star · two hot spots"
            )
        else:
            page.locator('[data-observer-name="star_spots"]').click()
        ready()
        assert state()["model"] == "star_spots"
        for label in ("Charged black hole · Q/M=0.8", "Zipoy-Voorhees · q=0.3"):
            page.locator("#observer-model").select_option(label=label)
            ready()
        diagrams = bool(page.locator("#coordinate-diagrams").count())
        if diagrams:
            assert page.locator("#coordinate-host iframe").count() == 0
            page.locator("#coordinate-diagrams > summary").click()
            frame = (
                page.locator("#coordinate-host iframe").element_handle().content_frame()
            )
            frame.wait_for_load_state("domcontentloaded")
            for identifier in ("geodesic-view", "volume-view"):
                if frame.locator("#" + identifier).count():
                    frame.wait_for_function(
                        "id=>!!document.getElementById(id)?._fullLayout?.scene?._scene",
                        arg=identifier,
                        timeout=60000,
                    )
            page.locator("#coordinate-diagrams > summary").click()
        page.set_viewport_size({"width": 390, "height": 844})
        page.locator("#observer-view").scroll_into_view_if_needed()
        page.screenshot(path=str(output / "observer_mobile_preview.png"))
        assert page.evaluate("document.documentElement.scrollWidth<=window.innerWidth")
        assert not errors, errors
        assert not external, external
        report = {
            "offline": True,
            "errors": errors,
            "external_requests": external,
            "controls": [
                "azimuth drag",
                "inclination drag",
                "zoom",
                "model selection",
                "auto orbit",
                "PNG download",
                "reset",
                "native gallery links" if native_links else "gallery card navigation",
            ],
            "lazy_coordinate_plots": diagrams,
            "native_links": native_links,
            "mobile_no_overflow": True,
            "last_state": state(),
        }
        (output / "observer_validation.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        browser.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/stationary_models"))
    parser.add_argument("--chrome", help="use an existing Chromium/Chrome executable")
    args = parser.parse_args()
    print(json.dumps(validate(args.output.resolve(), args.chrome), indent=2))

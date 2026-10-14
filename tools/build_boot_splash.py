"""Writes the boot splash of the Raspberry Pi (`pi/plymouth/dab-radio`, installed by
`pi/install.sh --splash`) from `ui.touch.splash`, the picture the radio shows while it
starts: the images of a dot, the axes and the name, and in the script the table of the
layout and the dots, between its two marks. The rest of the script, the animation, is
written by hand.

Run when the picture changes:

    uv run python tools/build_boot_splash.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PyQt6.QtGui import QGuiApplication

from ui.touch import splash

THEME = Path(__file__).resolve().parent.parent / "pi" / "plymouth" / "dab-radio"
SCRIPT = THEME / "dab-radio.script"


def main() -> None:
    _app = QGuiApplication(sys.argv[:1])
    for name, image in splash.splash_images().items():
        path = THEME / name
        if not image.save(str(path), "PNG"):
            raise SystemExit(f"cannot write {path}")
        print(f"{path} ({image.width()} x {image.height()})")
    script = SCRIPT.read_text(encoding="utf-8")
    SCRIPT.write_text(splash.with_plymouth_table(script), encoding="utf-8", newline="\n")
    print(f"{SCRIPT} ({len(splash.constellation())} dots)")


if __name__ == "__main__":
    main()

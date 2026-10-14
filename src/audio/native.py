"""Finding the decoder libraries (FAAD2, libmpg123) that `audio.aac` and `audio.mp2`
load through ctypes.

On Windows `ctypes.util.find_library` searches nothing but the PATH. The DLLs live in
the Scripts folder of the virtual environment, which is on the PATH under `uv run` or in
an activated environment, but not when the environment's python.exe is started directly,
as the run button of VS Code does. FAAD2 was then missing without any notice in the GUI
and no service had sound, so that folder is searched explicitly.
"""

from __future__ import annotations

import ctypes.util
import sys
from pathlib import Path


def find_library(*names: str) -> str | None:
    """Path or name of the first library found under one of `names` (without prefix
    "lib" on Linux and without ".dll"), for `ctypes.CDLL`; None if there is none."""
    for name in names:
        found = ctypes.util.find_library(name)
        if found is not None:
            return found
    scripts = Path(sys.prefix) / "Scripts"
    for name in names:
        candidate = scripts / f"{name}.dll"
        if candidate.is_file():
            return str(candidate)
    return None

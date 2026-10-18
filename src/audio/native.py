"""Finding the decoder libraries (FAAD2, libmpg123) that `audio.aac` and `audio.mp2`
load through ctypes.

On Windows `ctypes.util.find_library` searches nothing but the PATH. The DLLs that
`native\\build_codecs.ps1` builds lie in the folder `native` next to `src`, like
`native\\rtlsdr.dll`; DLLs obtained elsewhere may lie in the Scripts folder of the
virtual environment, which is on the PATH under `uv run` or in an activated environment,
but not when the environment's python.exe is started directly (the run button of VS
Code). FAAD2 would then be missing without any notice in the GUI and no service would
have sound, so both folders are searched explicitly.
"""

from __future__ import annotations

import ctypes.util
import sys
from pathlib import Path

NATIVE_DIRECTORY = Path(__file__).resolve().parents[2] / "native"
"""Where `native\\build_codecs.ps1` puts the DLLs it builds."""


def find_library(*names: str) -> str | None:
    """Path or name of the first library found under one of `names` (without prefix
    "lib" on Linux and without ".dll"), for `ctypes.CDLL`; None if there is none.

    The PATH comes first, then the folder `native`, then the Scripts folder of the
    environment."""
    for name in names:
        found = ctypes.util.find_library(name)
        if found is not None:
            return found
    for folder in (NATIVE_DIRECTORY, Path(sys.prefix) / "Scripts"):
        for name in names:
            candidate = folder / f"{name}.dll"
            if candidate.is_file():
                return str(candidate)
    return None

"""Imports a TMC location table into the radio's data directory, once per computer.

The table is a location code list (LCL) in the exchange format of ISO 14819-3, as the
responsible authority of a country publishes it, downloaded as a ZIP or unpacked into a
folder:

    uv run python tools/import_tmc_locations.py C:\\Users\\...\\lcl-download.zip

Tried so far only with the table of Germany, which ARD TPEG refers to (BASt, country D,
table 1). The files have to be UTF-8, as the BASt writes them. The data directory holds
one table: a second import replaces the first.

Keeps of it only what the traffic map needs (`dab.tmc_locations`): the points with
their coordinates, roads, names and neighbours. Neither the download nor the file
written may go into the repository (terms of the BASt; other authorities set their
own).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from build_road_network import data_directory

from dab.tmc_locations import TMC_LOCATIONS_FILENAME, TmcLocationTable


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="the LCL download (ZIP or folder)")
    parser.add_argument("--output", type=Path, help="file to write instead of the data directory")
    arguments = parser.parse_args(argv[1:])
    started = time.monotonic()
    table = TmcLocationTable.read_exchange(arguments.source)
    output = arguments.output or data_directory() / TMC_LOCATIONS_FILENAME
    output.parent.mkdir(parents=True, exist_ok=True)
    table.save(output)
    print(
        f"Wrote {output}: table {table.country}/{table.table} version {table.version}, "
        f"{len(table)} points, {output.stat().st_size / 1e6:.1f} MB "
        f"in {time.monotonic() - started:.1f} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

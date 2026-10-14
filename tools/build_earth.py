"""Builds `src/ui/assets/earth.npz`, the outlines the 3D view (`ui.globe_gl`) draws the
Earth from. Taken over unchanged from the gnss-receiver project, the file as well.

The source is Natural Earth (naturalearthdata.com, public domain) at 1:50 million: the
land, the large lakes and the borders between countries. The coordinates are stored in
hundredths of a degree (1.1 km on the ground, finer than the data) as int16, longitude
and latitude, with the length of every ring or line next to them: 95 000 points, 350 kB.

Run once, when the data is to change:

    uv run python tools/build_earth.py
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np

# A fixed release, so that the same file comes out again.
SOURCE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/"
LAND = "ne_50m_land.geojson"
LAKES = "ne_50m_lakes.geojson"
BORDERS = "ne_50m_admin_0_boundary_lines_land.geojson"
# Lakes up to this rank: the Caspian Sea, the Great Lakes, Lake Victoria and the like.
MAX_LAKE_RANK = 1
UNITS_PER_DEGREE = 100

OUTPUT = Path(__file__).resolve().parent.parent / "src" / "ui" / "assets" / "earth.npz"


def _features(name: str) -> list[dict]:
    with urllib.request.urlopen(SOURCE + name, timeout=60) as response:
        return json.load(response)["features"]


def _parts(geometry: dict) -> list[list]:
    """The rings of a polygon or the lines of a line string, whatever the nesting."""
    kind, coordinates = geometry["type"], geometry["coordinates"]
    if kind in ("Polygon", "MultiLineString"):
        return coordinates
    if kind == "MultiPolygon":
        return [ring for polygon in coordinates for ring in polygon]
    if kind == "LineString":
        return [coordinates]
    raise ValueError(f"unexpected geometry {kind}")


def _pack(features: list[dict], minimum: int) -> tuple[np.ndarray, np.ndarray]:
    """All points as int16 (N, 2) and the number of points of every part. Points that
    fall together after rounding are dropped, and parts with fewer than `minimum` left."""
    points, lengths = [], []
    for feature in features:
        for part in _parts(feature["geometry"]):
            rounded = np.round(np.array(part)[:, :2] * UNITS_PER_DEGREE).astype(np.int16)
            keep = np.ones(len(rounded), dtype=bool)
            keep[1:] = np.any(rounded[1:] != rounded[:-1], axis=1)
            rounded = rounded[keep]
            if len(rounded) >= minimum:
                points.append(rounded)
                lengths.append(len(rounded))
    return np.concatenate(points), np.array(lengths, dtype=np.int32)


def main() -> None:
    land, land_rings = _pack(_features(LAND), minimum=4)
    lakes = [f for f in _features(LAKES) if f["properties"]["scalerank"] <= MAX_LAKE_RANK]
    lake, lake_rings = _pack(lakes, minimum=4)
    border, border_lines = _pack(_features(BORDERS), minimum=2)
    np.savez_compressed(
        OUTPUT,
        land=land,
        land_rings=land_rings,
        lakes=lake,
        lake_rings=lake_rings,
        borders=border,
        border_lines=border_lines,
    )
    for name, points, parts in (
        ("land", land, land_rings),
        ("lakes", lake, lake_rings),
        ("borders", border, border_lines),
    ):
        print(f"{name}: {len(points)} points in {len(parts)} parts")
    print(f"{OUTPUT.name}: {OUTPUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()

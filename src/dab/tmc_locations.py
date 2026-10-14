"""The TMC location table (Location Code List, LCL) that TPEG's TMC locations refer to:
for every location code its coordinates, its road, its name and its neighbours along
the road (ISO 14819-3, exchange format).

ARD TPEG locates every traffic flow message (TFP) and part of its traffic events (TEC)
by TMC alone: table 1 of Germany (country code D, extended country code E0), version 22.0.
The BASt publishes the table; its terms do not allow it in the repository, so
`tools/import_tmc_locations.py` takes the download (ZIP or folder) on each computer and
keeps only what is needed here in a file in the data directory.

A TMC location is a point and an extent: the number of further points in the direction
its direction bit names, following the neighbours of the table. Checked against the
messages of ARD TPEG that carry an OpenLR line as well (7A, 06.10.2026): of 217 with an
extent, 207 extend that way, and in 223 of 234 the primary point lies at the end of the
OpenLR line, where the traffic flows to.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dab.tpeg_messages import TmcLocation
from i18n import tr, tr_noop

TMC_LOCATIONS_FILENAME = "tmc_locations.npz"
FORMAT_VERSION = 1

# Point types (the German names of SUBTYPES, as the table has them) and how a traffic
# report names a point of the type; German reports shorten them ("AS Remscheid").
# Translators: a junction of a motorway or major road, by its name; German traffic reports
# write "AS {name}" (Anschlussstelle).
_JUNCTION = tr_noop("{name} junction")
_SHORT_KINDS = {
    "Autobahnanschlussstelle": _JUNCTION,
    "Anschlussstelle": _JUNCTION,
    # Translators: a motorway interchange (Autobahnkreuz) by its name, German "AK {name}".
    "Autobahnkreuz": tr_noop("{name} interchange"),
    # Translators: a three-way motorway interchange (Autobahndreieck) by its name, German
    # "AD {name}".
    "Autobahndreieck": tr_noop("{name} triangle"),
}
_ROAD_NUMBER = re.compile(r"^([A-Z]+)\s*(\d.*)$")


def road_text(number: str) -> str:
    """ "A1" → "A 1", as roads are written in Germany; anything else as it is."""
    match = _ROAD_NUMBER.match(number.strip())
    return f"{match.group(1)} {match.group(2)}" if match else number.strip()


@dataclass(frozen=True)
class TmcPoint:
    code: int
    longitude: float
    latitude: float
    road: str | None  # "A 1", "B 6"
    name: str | None  # "Remscheid"
    kind: str | None  # "Autobahnanschlussstelle", "Kreuzung", ...
    junction: str | None  # the junction's number on the road, "95b"

    def label(self) -> str:
        """As a traffic report names it: "Remscheid junction", "Hannover-Ost interchange",
        "Laatzen"."""
        # Translators: a TMC location without a name, by its code.
        name = self.name or tr("Location {code}").format(code=self.code)
        short = _SHORT_KINDS.get(self.kind or "")
        return tr(short).format(name=name) if short else name


class TmcLocationTable:
    """The points of one table, by code."""

    def __init__(
        self,
        country: str,
        extended_country: str,
        table: int,
        version: str,
        points: list[TmcPoint],
        negative: dict[int, int],
        positive: dict[int, int],
    ) -> None:
        self.country = country  # "D"
        self.extended_country = extended_country  # "E0"
        self.table = table
        self.version = version  # "22.0"
        self._points = {p.code: p for p in points}
        self._negative = negative
        self._positive = positive

    def __len__(self) -> int:
        return len(self._points)

    def point(self, code: int) -> TmcPoint | None:
        return self._points.get(code)

    def serves(self, location: TmcLocation) -> bool:
        """Whether a message's location refers to this table."""
        country = format(location.country, "X")
        if country != self.country or location.table != self.table:
            return False
        ecc = location.extended_country_code
        return ecc is None or format(ecc, "02X") == self.extended_country

    def stretch(self, location: TmcLocation) -> list[TmcPoint]:
        """The primary point and the points of the extent, in that order; empty if the
        table does not know the primary point. Stops early where the table has no
        further neighbour."""
        point = self._points.get(location.location)
        if point is None:
            return []
        neighbours = self._positive if location.positive_direction else self._negative
        points = [point]
        for _ in range(location.extent):
            following = neighbours.get(points[-1].code)
            if following is None or following not in self._points:
                break
            points.append(self._points[following])
        return points

    # --- Exchange format ------------------------------------------------------------

    @classmethod
    def read_exchange(cls, source: Path) -> TmcLocationTable:
        """Reads the tables of the exchange format from a ZIP or a folder (searched for
        POINTS.DAT); ValueError if they are not there."""
        files = _ExchangeFiles(source)
        country_row = files.rows("COUNTRIES.DAT")[0]
        dataset = files.rows("LOCATIONDATASETS.DAT")[0]
        names = {int(r["NID"]): r["NAME"] for r in files.rows("NAMES.DAT") if r["NAME"]}
        kinds = {
            (r["CLASS"], r["TCD"], r["STCD"]): r["SNATDESC"] for r in files.rows("SUBTYPES.DAT")
        }
        roads = {int(r["LCD"]): r["ROADNUMBER"] for r in files.rows("ROADS.DAT")}
        segments = {}
        for r in files.rows("SEGMENTS.DAT"):
            number = r["ROADNUMBER"] or roads.get(_integer(r["ROA_LCD"]) or -1, "")
            segments[int(r["LCD"])] = number
        points = []
        for r in files.rows("POINTS.DAT"):
            if not r["XCOORD"] or not r["YCOORD"]:
                continue
            number = roads.get(_integer(r["ROA_LCD"]) or -1) or segments.get(
                _integer(r["SEG_LCD"]) or -1
            )
            points.append(
                TmcPoint(
                    int(r["LCD"]),
                    int(r["XCOORD"]) / 1e5,
                    int(r["YCOORD"]) / 1e5,
                    road_text(number) if number else None,
                    names.get(_integer(r["N1ID"]) or -1),
                    kinds.get((r["CLASS"], r["TCD"], r["STCD"])),
                    r["JUNCTIONNUMBER"] or None,
                )
            )
        negative, positive = {}, {}
        for r in files.rows("POFFSETS.DAT"):
            code = int(r["LCD"])
            if r["NEG_OFF_LCD"]:
                negative[code] = int(r["NEG_OFF_LCD"])
            if r["POS_OFF_LCD"]:
                positive[code] = int(r["POS_OFF_LCD"])
        return cls(
            country_row["CCD"],
            country_row["ECC"],
            int(dataset["TABCD"]),
            dataset["VERSION"],
            points,
            negative,
            positive,
        )

    # --- Own file -----------------------------------------------------------------------

    def save(self, path: Path) -> None:
        points = list(self._points.values())
        strings: list[str] = [""]
        index: dict[str, int] = {"": 0}

        def ref(text: str | None) -> int:
            if not text:
                return 0
            if text not in index:
                index[text] = len(strings)
                strings.append(text)
            return index[text]

        codes = np.array([p.code for p in points], dtype=np.int32)
        np.savez_compressed(
            path,
            version=np.array(FORMAT_VERSION),
            meta=np.array([self.country, self.extended_country, str(self.table), self.version]),
            code=codes,
            xy=np.array([(p.longitude, p.latitude) for p in points], dtype=np.float64).reshape(
                -1, 2
            ),
            texts=np.array(
                [(ref(p.road), ref(p.name), ref(p.kind), ref(p.junction)) for p in points],
                dtype=np.int32,
            ).reshape(-1, 4),
            negative=np.array([self._negative.get(c, -1) for c in codes], dtype=np.int32),
            positive=np.array([self._positive.get(c, -1) for c in codes], dtype=np.int32),
            strings=np.array(strings, dtype=str),
        )

    @classmethod
    def load(cls, path: Path) -> TmcLocationTable:
        with np.load(path, allow_pickle=False) as data:
            if int(data["version"]) != FORMAT_VERSION:
                raise ValueError(f"TMC location format {int(data['version'])}")
            country, extended, table, version = (str(s) for s in data["meta"])
            strings = [str(s) or None for s in data["strings"]]
            points = [
                TmcPoint(int(code), float(x), float(y), *(strings[i] for i in texts))
                for code, (x, y), texts in zip(data["code"], data["xy"], data["texts"])
            ]
            negative = {int(c): int(n) for c, n in zip(data["code"], data["negative"]) if n >= 0}
            positive = {int(c): int(n) for c, n in zip(data["code"], data["positive"]) if n >= 0}
        return cls(country, extended, int(table), version, points, negative, positive)


def load_if_present(data_directory: Path) -> TmcLocationTable | None:
    """The table imported into the data directory, None if there is none (or it is of
    an older format)."""
    path = data_directory / TMC_LOCATIONS_FILENAME
    if not path.exists():
        return None
    try:
        return TmcLocationTable.load(path)
    except OSError, ValueError, KeyError:
        return None


def _integer(text: str) -> int | None:
    return int(text) if text and text.strip().lstrip("-").isdigit() else None


class _ExchangeFiles:
    """The .DAT files of the exchange format in a ZIP or under a folder."""

    def __init__(self, source: Path) -> None:
        self._zip: zipfile.ZipFile | None = None
        self._paths: dict[str, str | Path] = {}
        if source.is_file():
            self._zip = zipfile.ZipFile(source)
            members = self._zip.namelist()
            for member in members:
                self._paths.setdefault(Path(member).name.upper(), member)
        else:
            for path in source.rglob("*.DAT"):
                self._paths.setdefault(path.name.upper(), path)
        if "POINTS.DAT" not in self._paths:
            raise ValueError(f"no TMC location table (POINTS.DAT) in {source}")

    def rows(self, name: str) -> list[dict[str, str]]:
        where = self._paths.get(name)
        if where is None:
            raise ValueError(f"{name} missing from the TMC location table")
        raw = self._zip.read(where) if self._zip is not None else Path(where).read_bytes()
        text = raw.decode("utf-8-sig")
        return [dict(row) for row in csv.DictReader(io.StringIO(text), delimiter=";")]

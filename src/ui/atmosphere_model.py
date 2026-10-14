"""What the SSRZ corrections of GEPOS say about the ionosphere and the troposphere over
Germany, without a widget: the reference stations of the grids, the latest message of
each kind, and from them the fields a map shows (`AtmosphereModel`).

`ui.gnss_model.GnssModel` keeps one for every surface; the desktop's "Atmosphere" tab
(`ui.atmosphere_tab`) and the touch surface's page of the same name draw its fields.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from dab.atmosphere import (
    corrected_zenith_delays,
    day_of_year,
    model_zenith_delays,
    summed_scale_factors,
)
from dab.ssrz import (
    GPS_EPOCH_POSIX,
    TROPOSPHERE_DRY,
    TROPOSPHERE_TOTAL,
    TROPOSPHERE_WET,
    GlobalVtec,
    Grid,
    GridDefinitions,
    GridIonosphere,
    GridTroposphere,
    LeapSecond,
    RegionalIonosphere,
    RegionalTroposphere,
    SsrzMetadata,
    SsrzTime,
    StreamIdentification,
    sort_satellites,
)
from i18n import tr, tr_context, tr_plural

# The map: Germany with a margin, rasters in steps of 0.1 degrees.
MAP_WEST, MAP_EAST = 5.0, 16.0
MAP_SOUTH, MAP_NORTH = 47.0, 55.5
RASTER_STEP = 0.1
# Between the stations the rasters interpolate (inverse distance squared); farther than
# this from every station with a value they stay empty. GEPOS stations are 40 to 70 km
# apart.
STATION_REACH_KM = 55.0
_KM_PER_DEGREE = 111.2
# Corrections older than this (by the ensemble clock) are shown as out of date.
STALE_SECONDS = 120.0
# After new corrections the maps are redrawn at most this often: they come in a burst of
# some 40 messages every 30 s, and a drawing costs a few 10 ms in the GUI thread.
REDRAW_SECONDS = 5.0
# GPS time is ahead of UTC by this until ZM024 says otherwise.
DEFAULT_LEAP_SECONDS = 18

# SSRZ provider IDs (Geo++ SSRZ format 1.1.3, table 15.6).
_PROVIDERS = {
    33: "BKG",
    34: "ZSS (SAPOS)",
    35: "TERIA",
    36: "BEV",
    37: "IGN",
    38: "NLSC",
    39: "06-GPS",
}

# Orientation on the map: large cities (latitude, longitude). Place names as the maps
# beneath them have them (the roads of OpenStreetMap, the TMC location table): German,
# in every language of the surface.
CITIES = (
    ("Berlin", 52.52, 13.40),
    ("Hamburg", 53.55, 9.99),
    ("München", 48.14, 11.58),
    ("Köln", 50.94, 6.96),
    ("Frankfurt", 50.11, 8.68),
    ("Stuttgart", 48.78, 9.18),
    ("Leipzig", 51.34, 12.37),
    ("Hannover", 52.37, 9.73),
    ("Dresden", 51.05, 13.74),
    ("Nürnberg", 49.45, 11.08),
    ("Bremen", 53.08, 8.80),
    ("Rostock", 54.09, 12.10),
)

# What the ionosphere map shows: the global model, or the gridded correction of one
# satellite (then its name).
IONOSPHERE_GLOBAL = None
# What the troposphere map shows.
TROPOSPHERE_VIEW_WET = "zwd"
TROPOSPHERE_VIEW_TOTAL = "ztd"
TROPOSPHERE_VIEW_FACTOR = "factor"


@dataclass(frozen=True)
class Station:
    grid_id: int
    index: int  # within the grid
    latitude: float
    longitude: float
    height: float


@dataclass(frozen=True)
class Field:
    """What a map shows: a raster over the map area (rows south to north), values at
    the stations, and how to colour them."""

    raster: np.ndarray | None
    values: np.ndarray  # per station, NaN: none
    levels: tuple[float, float]
    colour_map: str
    label: str
    unit_format: str  # format of one value with its unit, e.g. "{value:.2f} TECU"


def raster_axes() -> tuple[np.ndarray, np.ndarray]:
    latitudes = np.arange(MAP_SOUTH, MAP_NORTH + RASTER_STEP / 2, RASTER_STEP)
    longitudes = np.arange(MAP_WEST, MAP_EAST + RASTER_STEP / 2, RASTER_STEP)
    return latitudes, longitudes


class _Interpolator:
    """Spreads values at the stations over the raster: inverse distance squared, empty
    farther than `STATION_REACH_KM` from every station with a value."""

    def __init__(self, stations: list[Station]) -> None:
        latitudes, longitudes = raster_axes()
        grid_lat, grid_lon = np.meshgrid(latitudes, longitudes, indexing="ij")
        lat = np.array([s.latitude for s in stations])
        lon = np.array([s.longitude for s in stations])
        dy = (grid_lat[..., None] - lat) * _KM_PER_DEGREE
        dx = (grid_lon[..., None] - lon) * _KM_PER_DEGREE * np.cos(np.radians(grid_lat[..., None]))
        squared = (dx * dx + dy * dy).reshape(-1, len(stations))
        self._weights = (1.0 / np.maximum(squared, 1.0)).astype(np.float32)
        self._near = squared <= STATION_REACH_KM**2
        self.shape = grid_lat.shape

    def __call__(self, values: np.ndarray) -> np.ndarray:
        valid = np.isfinite(values)
        if not np.any(valid):
            return np.full(self.shape, np.nan)
        weights = self._weights[:, valid]
        raster = (weights @ values[valid].astype(np.float32)) / weights.sum(axis=1)
        raster[~self._near[:, valid].any(axis=1)] = np.nan
        return raster.reshape(self.shape)


def _symmetric(values: np.ndarray, minimum: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    extent = max(float(np.max(np.abs(finite))) if finite.size else 0.0, minimum)
    return -extent, extent


def _range(values: np.ndarray, minimum_span: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 0.0, minimum_span
    low, high = float(np.min(finite)), float(np.max(finite))
    if high - low < minimum_span:
        middle = (low + high) / 2
        low, high = middle - minimum_span / 2, middle + minimum_span / 2
    return low, high


def decimal_point() -> str:
    """The decimal separator of the numbers on the GNSS pages, as the language writes it."""
    # Translators: the decimal separator of the numbers on the GNSS pages ("0.97 %");
    # German writes a comma there.
    return tr_context("decimal separator", ".")


def decimal(value: float, spec: str) -> str:
    """A number with the decimal separator of the language (`decimal_point`)."""
    return format(value, spec).replace(".", decimal_point())


def grouped(value: float) -> str:
    """A whole number with its thousands grouped as the language writes it: 20,183."""
    # Translators: the separator of the thousands in large numbers ("20,183 km");
    # German writes a full stop.
    return format(value, ",.0f").replace(",", tr_context("thousands separator", ","))


class AtmosphereModel:
    """The latest SSRZ messages about the atmosphere and the stations of their grids.

    `add` takes every result of the SSRZ decoder; the fields are computed when asked for,
    at the time given (seconds since the epoch, by the ensemble clock). The corrections
    stay after the decoding stops; `received` tells how old they are.
    """

    def __init__(self) -> None:
        self.grids: dict[int, Grid] = {}
        self.stations: list[Station] = []
        self._interpolator: _Interpolator | None = None
        self.grid_ionosphere: dict[int, GridIonosphere] = {}
        self.grid_troposphere: dict[int, GridTroposphere] = {}
        self.regional_ionosphere: RegionalIonosphere | None = None
        self.global_vtec: GlobalVtec | None = None
        self.regional_troposphere: RegionalTroposphere | None = None
        self.metadata: SsrzMetadata | None = None
        self.stream: StreamIdentification | None = None
        self.leap: LeapSecond | None = None
        self.time: SsrzTime | None = None
        self.received: float | None = None  # ensemble time of the last correction

    def add(self, result: object, now: float) -> bool:
        """Takes what an SSRZ message said (`dsp.pipeline.DspWorker.ssrz_ready`); True
        if a map would change."""
        if isinstance(result, GridDefinitions):
            changed = False
            for grid in result.grids:
                if self.grids.get(grid.grid_id) != grid:
                    self.grids[grid.grid_id] = grid
                    changed = True
            if changed:
                self._rebuild_stations()
            return changed
        if isinstance(result, SsrzMetadata):
            self.metadata = result
            for grid in result.grids:
                self.grids[grid.grid_id] = grid
            if result.grids:
                self._rebuild_stations()
        elif isinstance(result, SsrzTime):
            self.time = result
            return False
        elif isinstance(result, StreamIdentification):
            self.stream = result
            return False
        elif isinstance(result, LeapSecond):
            self.leap = result
            return False
        elif isinstance(result, GridIonosphere):
            self.grid_ionosphere[result.grid_id] = result
        elif isinstance(result, GridTroposphere):
            self.grid_troposphere[result.grid_id] = result
        elif isinstance(result, RegionalIonosphere):
            self.regional_ionosphere = result
        elif isinstance(result, GlobalVtec):
            self.global_vtec = result
        elif isinstance(result, RegionalTroposphere):
            self.regional_troposphere = result
        else:
            return False
        self.received = now
        return True

    def is_stale(self, now: float) -> bool:
        return self.received is not None and now - self.received > STALE_SECONDS

    # --- Fields --------------------------------------------------------------------

    def _rebuild_stations(self) -> None:
        self.stations = [
            Station(grid.grid_id, index, lat, lon, height)
            for grid in sorted(self.grids.values(), key=lambda g: g.grid_id)
            for index, (lat, lon, height) in enumerate(
                zip(grid.latitudes, grid.longitudes, grid.heights)
            )
        ]
        self._interpolator = _Interpolator(self.stations) if self.stations else None
        # Corrections for a grid that has since changed no longer fit its points.
        for store in (self.grid_ionosphere, self.grid_troposphere):
            for grid_id in [g for g, item in store.items() if not self._fits(item)]:
                del store[grid_id]

    def _fits(self, item: GridIonosphere | GridTroposphere) -> bool:
        grid = self.grids.get(item.grid_id)
        return grid is not None and grid.iod == item.grid_iod

    def _per_station(self, values_of: Callable[[int], np.ndarray | None]) -> np.ndarray:
        """Values per station from the latest message of each grid; `values_of` maps
        a message to its values for the grid's points, or None."""
        result = np.full(len(self.stations), np.nan)
        offset = 0
        for grid in sorted(self.grids.values(), key=lambda g: g.grid_id):
            count = grid.point_count
            values = values_of(grid.grid_id)
            if values is not None and len(values) == count:
                result[offset : offset + count] = values
            offset += count
        return result

    def leap_seconds(self) -> int:
        return self.leap.leap_seconds if self.leap is not None else DEFAULT_LEAP_SECONDS

    def _gps_time(self, item_time: float | None, now: float) -> float:
        """GPS time of a correction; without one the last time tag (ZM009), else the
        clock."""
        if item_time is not None:
            return item_time
        if self.time is not None and self.time.gps_time is not None:
            return self.time.gps_time
        return now - GPS_EPOCH_POSIX + self.leap_seconds()

    def ionosphere_field(self, satellite: str | None, now: float) -> Field | None:
        """The global model (`IONOSPHERE_GLOBAL`) or the gridded residual of a satellite."""
        if satellite is IONOSPHERE_GLOBAL:
            model = self.global_vtec
            if model is None or not model.valid:
                return None
            gps_time = self._gps_time(model.gps_time, now)
            latitudes, longitudes = raster_axes()
            grid_lat, grid_lon = np.meshgrid(latitudes, longitudes, indexing="ij")
            raster = model.vtec(grid_lat, grid_lon, gps_time)
            values = model.vtec(
                np.array([s.latitude for s in self.stations]),
                np.array([s.longitude for s in self.stations]),
                gps_time,
            )
            return Field(
                raster,
                values,
                _range(raster, 2.0),
                "viridis",
                "VTEC (TECU)",
                tr("VTEC {value:.1f} TECU (global model)"),
            )

        def residuals(grid_id: int) -> np.ndarray | None:
            item = self.grid_ionosphere.get(grid_id)
            if item is None or satellite not in item.satellites:
                return None
            return item.values[item.satellites.index(satellite)]

        values = self._per_station(residuals)
        raster = None if self._interpolator is None else self._interpolator(values)
        return Field(
            raster,
            values,
            _symmetric(values, 0.1),
            "CET-D1A",
            tr("Residual correction {satellite} (TECU)").format(satellite=satellite),
            tr("{value:+.2f} TECU vertical"),
        )

    def troposphere_field(self, view: str, now: float) -> Field | None:
        """The wet or total zenith delay (`TROPOSPHERE_VIEW_WET`, `..._TOTAL`) or the
        wet scale factor sent (`..._FACTOR`)."""
        if not self.stations or not self.grid_troposphere:
            return None
        latitude = np.array([s.latitude for s in self.stations])
        longitude = np.array([s.longitude for s in self.stations])
        height = np.array([s.height for s in self.stations])
        height = np.where(np.isfinite(height), height, 0.0)
        wet_factor = self._per_station(self._wet_factors)
        if view == TROPOSPHERE_VIEW_FACTOR:
            values = wet_factor * 100.0
            raster = None if self._interpolator is None else self._interpolator(values)
            return Field(
                raster,
                values,
                _symmetric(values, 5.0),
                "CET-D1A",
                tr("Wet scale factor (%)"),
                tr("Wet scale factor {value:+.1f} %"),
            )
        posix = self._gps_time(None, now) + GPS_EPOCH_POSIX - self.leap_seconds()
        model = model_zenith_delays(latitude, longitude, height, day_of_year(posix))
        factors = summed_scale_factors(
            latitude, longitude, height, regional=self.regional_troposphere
        )
        factors[TROPOSPHERE_WET] = factors.get(TROPOSPHERE_WET, 0.0) + wet_factor
        delays = corrected_zenith_delays(model, factors)
        if view == TROPOSPHERE_VIEW_WET:
            values = delays.wet * 100.0
            label, unit = "ZWD (cm)", tr("wet {value:.1f} cm at the zenith")
        else:
            values = delays.total * 100.0
            label, unit = "ZTD (cm)", tr("total {value:.1f} cm at the zenith")
        raster = None if self._interpolator is None else self._interpolator(values)
        return Field(raster, values, _range(values, 2.0), "viridis", label, unit)

    def _wet_factors(self, grid_id: int) -> np.ndarray | None:
        item = self.grid_troposphere.get(grid_id)
        if item is None:
            return None
        rows = item.components.get(TROPOSPHERE_WET)
        return None if rows is None else rows[0]

    def ionosphere_satellites(self) -> list[str]:
        """The satellites the ionosphere is described for, regionally or in a grid."""
        names: set[str] = set()
        for item in self.grid_ionosphere.values():
            names.update(item.satellites)
        if self.regional_ionosphere is not None:
            names.update(self.regional_ionosphere.satellites)
        return list(sort_satellites(names))

    # --- Texts ---------------------------------------------------------------------

    @staticmethod
    def station_tip(station: Station) -> str:
        height = "" if math.isnan(station.height) else f", {station.height:.0f} m"
        return tr("Grid {grid}, station {station}\n{latitude}° N, {longitude}° E{height}").format(
            grid=station.grid_id,
            station=station.index + 1,
            latitude=decimal(station.latitude, ".3f"),
            longitude=decimal(station.longitude, ".3f"),
            height=height,
        )

    def describe(self) -> str:
        """What the stream, the metadata and the regional troposphere say, a sentence a
        line."""
        lines = []
        stream = self.stream
        if stream is not None:
            provider = _PROVIDERS.get(stream.provider_id) or tr("unknown")
            lines.append(
                tr("Provider {number} ({provider}), stream {stream}.").format(
                    number=stream.provider_id, provider=provider, stream=stream.stream_id
                )
            )
        if self.grids:
            lines.append(
                tr_plural(
                    "{count} grid with {stations} stations in total (ZM013).",
                    "{count} grids with {stations} stations in total (ZM013).",
                    len(self.grids),
                ).format(count=len(self.grids), stations=len(self.stations))
            )
        metadata = self.metadata
        if metadata is not None:
            decoded = sum(block.decoded for block in metadata.blocks)
            lines.append(
                tr(
                    "Metadata (ZM012, issue {iod}): {blocks} blocks, {decoded} of them evaluated."
                ).format(iod=metadata.iod, blocks=len(metadata.blocks), decoded=decoded)
            )
            vtec = metadata.global_vtec
            if vtec is not None:
                lines.append(
                    tr(
                        "Global model: spherical harmonics up to degree {degree}, order "
                        "{order}, layer at {height:.0f} km, resolution {resolution} TECU."
                    ).format(
                        degree=vtec.degree,
                        order=vtec.order,
                        height=vtec.layer_height_km,
                        resolution=decimal(vtec.resolution, "g"),
                    )
                )
            regional = metadata.regional_ionosphere
            if regional is not None and regional.origin is not None:
                lat, lon, _h = regional.origin
                lines.append(
                    tr(
                        "Regional ionosphere: {count} coefficients per satellite around "
                        "{latitude}° N {longitude}° E."
                    ).format(
                        count=len(regional.coefficients),
                        latitude=decimal(lat, ".1f"),
                        longitude=decimal(lon, ".1f"),
                    )
                )
            gridded = metadata.grid_ionosphere
            if gridded is not None:
                lines.append(
                    tr("Gridded ionosphere: resolution {resolution} TECU.").format(
                        resolution=decimal(gridded.resolution, "g")
                    )
                )
        lines.extend(self._describe_troposphere())
        time_line = self._describe_time()
        if time_line:
            lines.append(time_line)
        return "\n".join(lines)

    def _describe_troposphere(self) -> list[str]:
        lines = []
        regional = self.regional_troposphere
        if regional is not None:
            for region in regional.regions:
                definition = region.definition
                lat, lon, height = definition.origin
                dry = region.components.get(TROPOSPHERE_DRY)
                if dry is not None and dry.size:
                    lines.append(
                        tr(
                            "Regional troposphere around {latitude}° N {longitude}° E, "
                            "{height:.0f} m: dry component {factor} % relative to the model."
                        ).format(
                            latitude=decimal(lat, ".1f"),
                            longitude=decimal(lon, ".1f"),
                            height=height,
                            factor=decimal(dry[0] * 100, "+.2f"),
                        )
                    )
                if region.mapping is not None and definition.mapping is not None:
                    lines.append(
                        tr("Improved mapping function below {elevation}° elevation.").format(
                            elevation=definition.mapping.max_elevation
                        )
                    )
        metadata = self.metadata
        gridded = None if metadata is None else metadata.grid_troposphere
        if gridded is not None:
            names = {
                # Translators: the components of the troposphere.
                TROPOSPHERE_DRY: tr("dry"),
                TROPOSPHERE_WET: tr("wet"),
                TROPOSPHERE_TOTAL: tr("total"),
            }
            parts = ", ".join(
                tr("{component} (resolution {resolution} %)").format(
                    component=names[c], resolution=decimal(r * 100, "g")
                )
                for c, r, _p in gridded.components
            )
            lines.append(tr("Gridded troposphere: {components}.").format(components=parts))
        return lines

    def _describe_time(self) -> str:
        parts = []
        if self.time is not None and self.time.gps_time is not None:
            posix = self.time.gps_time + GPS_EPOCH_POSIX - self.leap_seconds()
            parts.append(
                tr("Time tag (ZM009): {time} UTC").format(
                    time=time.strftime("%H:%M:%S", time.gmtime(posix))
                )
            )
        if self.leap is not None:
            parts.append(
                tr("GPS time {seconds} s ahead of UTC").format(seconds=self.leap.leap_seconds)
            )
        return (", ".join(parts) + ".") if parts else ""

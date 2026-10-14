"""Tests for SSRZ (`dab.ssrz`): prefix and Rice codes, the grid chains, the gridded data
predictor, the spherical harmonics and real messages of GEPOS (atmosphere, clock, orbit
and signal biases), checked against the reference decoder of Geo++."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dab.crc import crc16_ccitt
from dab.data_group import parse_data_group
from dab.rtcm import RtcmStreamDecoder
from dab.ssrz import (
    TROPOSPHERE_DRY,
    TROPOSPHERE_WET,
    GlobalVtec,
    GridDefinitions,
    GridIonosphere,
    GridTroposphere,
    HighRateCorrections,
    LeapSecond,
    LowRateCorrections,
    RegionalIonosphere,
    RegionalTroposphere,
    SatelliteGroups,
    SsrzDecoder,
    SsrzMetadata,
    SsrzTime,
    StreamIdentification,
    _Bits,
    normalised_legendre,
    predict_chain,
    reconstruct_chain,
    sort_satellites,
)
from tests.ssrz_vectors import (
    ZM001_HIGH_RATE,
    ZM002_LOW_RATE,
    ZM003_GRID_IONOSPHERE,
    ZM004_GRID_TROPOSPHERE,
    ZM005_REGIONAL_IONOSPHERE,
    ZM006_GLOBAL_VTEC,
    ZM007_REGIONAL_TROPOSPHERE,
    ZM009_TIME,
    ZM010_STREAM_ID,
    ZM011_SATELLITE_GROUPS,
    ZM012_METADATA,
    ZM013_GRID_3,
    ZM024_LEAP_SECOND,
)

NAN = math.nan


def _message(frame: bytes) -> bytes:
    return frame[3:-3]


def _bits(text: str) -> _Bits:
    """A reader over the bits written as text ("11 000 ..."), padded with zeros."""
    clean = text.replace(" ", "")
    clean += "0" * (-len(clean) % 8)
    return _Bits(int(clean, 2).to_bytes(len(clean) // 8, "big"))


# Table 6.2 of the format description. Its rows 23 and 24 are one too high (their
# codes are 24 and 25); the stream agrees with the rule: ZM024 is 11 111 1101.
@pytest.mark.parametrize(
    ("code", "value"),
    [
        ("00", 0),
        ("10", 2),
        ("11 000", 3),
        ("11 110", 9),
        ("11 111 0000", 10),
        ("11 111 0011", 13),
        ("11 111 1101", 23),
        ("11 111 1111 11110", 55),
        ("11 111 1111 11111 00000", 56),
    ],
)
def test_prefix_code_2_1_5(code: str, value: int) -> None:
    bits = _bits(code)
    assert bits.prefix(2, 1, 5) == value
    assert bits.position == len(code.replace(" ", ""))


@pytest.mark.parametrize(
    ("code", "first", "value"),
    [
        ("11 01", 2, 4),  # PC(2,0)
        ("11 11 11 00", 2, 9),
        ("1111111110", 1, 9),  # PC(1,0): unary
        ("0", 1, 0),
        ("11111 00011", 5, 34),  # PC(5,0)
    ],
)
def test_prefix_code_constant_blocks(code: str, first: int, value: int) -> None:
    assert _bits(code).prefix(first) == value


def test_rice_values_with_sign_and_negative_zero() -> None:
    # Sign, quotient in unary, remainder in p = 2 bits: +(2 * 4 + 1), -(0 * 4 + 3), -0.
    bits = _bits("0 110 01  1 0 11  1 0 00")
    assert bits.rice(2) == 9
    assert bits.rice(2) == -3
    assert bits.rice(2) is None
    # Without remainder bits (p = 0) the value is the quotient.
    assert _bits("1 1110").rice(0) == -3
    # The variant of the global VTEC coefficients: quotient, sign, remainder.
    assert _bits("10 1 11").rice_quotient_first(2) == -7


def test_sign_magnitude_and_two_complement() -> None:
    assert _bits("1 0000000101").sign_magnitude(11) == -5
    assert _bits("0 0000000101").sign_magnitude(11) == 5
    assert _bits("1" * 32).int32() == -1


def test_satellites_are_sorted_by_gnss_id_then_number() -> None:
    assert sort_satellites(["C01", "E12", "G32", "R02", "G05"]) == (
        "G05",
        "G32",
        "R02",
        "E12",
        "C01",
    )


# --- Grid chains -----------------------------------------------------------------------


def test_equilateral_triangle_left_and_right_of_the_baseline() -> None:
    """ds = ss = 0 is an equilateral triangle. The baseline points east (lambda grows);
    left of it is north (phi grows), in the (phi, lambda) plane of eqs. 12.7 to 12.13."""
    first, side = (0.9, 0.1), 0.01
    for left, sign in ((True, 1), (False, -1)):
        points = reconstruct_chain(
            first, [(0.0, side), (0.0, 0.0)], [(True,), (left,), (False,), (False,)]
        )
        phi, lam = points[2]
        assert phi == pytest.approx(0.9 + sign * side * math.sqrt(3) / 2, abs=1e-12)
        assert lam == pytest.approx(0.1 + side / 2, abs=1e-12)


def test_chain_takes_the_next_baseline_from_the_list() -> None:
    """Two equilateral triangles: point 2 north of the baseline 0 -> 1 (east), which
    adds the baseline from its left base point, 0 -> 2. Point 3 takes that one (the
    next in the list) and lies left of it: north-west of point 0."""
    side = 0.01
    height = side * math.sqrt(3) / 2
    points = reconstruct_chain(
        (0.0, 0.0),
        [(0.0, side), (0.0, 0.0), (0.0, 0.0)],
        [(True, False), (True, True), (True, False), (False, False)],
    )
    # Point 2: equilateral, north of 0 -> 1.
    assert points[2] == pytest.approx((height, side / 2))
    # Point 3 over the added baseline 0 -> 2, on its left: to the north-west.
    phi, lam = points[3]
    assert math.hypot(phi, lam) == pytest.approx(side)
    assert math.hypot(phi - height, lam - side / 2) == pytest.approx(side)
    assert lam < 0


def test_chain_without_baseline_to_use_is_an_error() -> None:
    with pytest.raises(ValueError):
        reconstruct_chain(
            (0.0, 0.0), [(0.0, 0.01), (0.0, 0.0)], [(False,), (True,), (False,), (False,)]
        )


def test_gridded_data_predictor() -> None:
    """Table 12.1: the first valid value directly, the next two predicted by it, then
    x_b + x_c - x_a; an invalid value keeps its prediction for later points."""
    # (a, b, c) for the points 3, 4 and 5; point 3 is only the third of the series.
    points = ((0, 0, 0), (3, 2, 1), (4, 3, 2))
    values = [NAN, 1.0, 0.5, -0.25, NAN, 0.125]
    result = predict_chain(values, points)
    # Point 0 invalid: the series starts at point 1.
    assert math.isnan(result[0])
    assert result[1] == 1.0
    assert result[2] == 1.5  # 1.0 + 0.5
    assert result[3] == 0.75  # 1.0 - 0.25
    # Point 4 is invalid: its prediction x_2 + x_1 - x_3 = 1.75 stands in for it later.
    assert math.isnan(result[4])
    # Point 5: x_3 + x_2 - x_4 = 0.75 + 1.5 - 1.75, plus 0.125.
    assert result[5] == pytest.approx(0.625)


# --- Spherical harmonics ---------------------------------------------------------------


def test_normalised_legendre_functions() -> None:
    x = np.array([0.3, -0.8])
    c = np.sqrt(1 - x * x)
    p = normalised_legendre(2, 2, x)
    np.testing.assert_allclose(p[0][0], 1.0)
    np.testing.assert_allclose(p[1][0], math.sqrt(3) * x)
    np.testing.assert_allclose(p[1][1], math.sqrt(3) * c)
    np.testing.assert_allclose(p[2][0], math.sqrt(5) / 2 * (3 * x * x - 1))
    np.testing.assert_allclose(p[2][1], math.sqrt(15) * x * c)
    np.testing.assert_allclose(p[2][2], math.sqrt(15) / 2 * c * c)


# --- Real messages of GEPOS ------------------------------------------------------------


def _decoder_with_metadata() -> SsrzDecoder:
    decoder = SsrzDecoder()
    for frame in (ZM009_TIME, ZM011_SATELLITE_GROUPS, ZM012_METADATA, ZM013_GRID_3):
        assert decoder.push(_message(frame)) is not None
    return decoder


def test_time_stream_and_leap_second() -> None:
    decoder = SsrzDecoder()
    time_tag = decoder.push(_message(ZM009_TIME))
    assert isinstance(time_tag, SsrzTime)
    assert (time_tag.definition, time_tag.week, time_tag.seconds) == (6, 2438, 239995)
    # Tuesday, 29.09.2026, 18:39:55 GPS time.
    assert time_tag.gps_time == 2438 * 604800 + 239995
    stream = decoder.push(_message(ZM010_STREAM_ID))
    assert stream == StreamIdentification(version=1, provider_id=33, stream_id=0)
    leap = decoder.push(_message(ZM024_LEAP_SECOND))
    assert leap == LeapSecond(year=2026, month=12, leap_seconds=18, change=0)
    # The time tag of the corrections resolves against the last ZM009: 18:40:00.
    assert decoder.resolve_time_tag(600) == 2438 * 604800 + 240000
    assert decoder.resolve_time_tag(590) == 2438 * 604800 + 239990


def test_satellite_groups() -> None:
    groups = SsrzDecoder().push(_message(ZM011_SATELLITE_GROUPS))
    assert isinstance(groups, SatelliteGroups)
    # One group each, all satellites of GPS, GLONASS, Galileo and BeiDou (GNSS mode).
    assert len(groups.low_rate) == len(groups.high_rate) == 1
    satellites = groups.low_rate[0]
    assert len(satellites) == 32 + 24 + 36 + 62
    assert satellites[:2] == ("G01", "G02")
    assert satellites[32] == "R01"
    assert satellites[-1] == "C62"


def test_metadata() -> None:
    metadata = SsrzDecoder().push(_message(ZM012_METADATA))
    assert isinstance(metadata, SsrzMetadata)
    assert metadata.iod == 0
    assert [(b.type_number, b.tag, b.bits, b.decoded) for b in metadata.blocks] == [
        (1, 1, 57, True),
        (2, 1, 618, True),
        (3, 1, 29, True),
        (4, 2, 230, True),
        (5, 1, 289, True),
        (6, 3, 246, True),
        (7, 3, 144, True),
    ]
    assert metadata.low_rate_tag == 1
    assert metadata.low_rate_timing.interval == 30
    high = metadata.high_rate
    assert high.timing.interval == 5
    assert high.clock.resolutions == (pytest.approx(1e-4),)
    assert high.clock.blocks[0].gnss == ("G", "R", "E", "J", "S", "C", "I")
    assert high.orbit is None
    low = metadata.low_rate
    assert low.iod_bits == dict.fromkeys("GREJSCI", 10)
    assert low.clock.resolutions == (pytest.approx(0.01),)
    assert low.orbit.resolutions == (pytest.approx(1e-4),) * 3
    assert low.orbit.blocks[0].p0 == (0, 0, 0)
    assert low.velocity is None
    assert low.code_bias_resolution == pytest.approx(1e-4)
    (block,) = low.code_bias_blocks
    assert block.signals == {"G": ("2W", "5Q"), "R": ("2P", "2C"), "E": ("5Q", "7Q"), "C": ("6I",)}
    assert low.code_bias_reference == {"G": ("1C",), "R": ("1C",), "E": ("1C",), "C": ("2I",)}
    assert low.phase_bias_signals == {
        "G": ("1C", "2W", "5Q"),
        "R": ("1C", "2P", "2C"),
        "E": ("1C", "5Q", "7Q"),
        "C": ("2I", "6I"),
    }
    assert (low.phase_bias_cycle_range, low.phase_bias_bits, low.continuity_bits) == (10, 16, 4)
    assert low.phase_bias_resolution == pytest.approx(10 / 65536)
    assert not low.satellite_ionosphere
    assert metadata.grid_ionosphere.resolution == pytest.approx(0.01)
    assert metadata.grid_ionosphere.p0 == 0
    troposphere = metadata.grid_troposphere
    assert troposphere.components == ((TROPOSPHERE_WET, pytest.approx(1e-5), 0),)
    assert troposphere.model.model_id == 6
    regional = metadata.regional_ionosphere
    assert regional.coefficients == ((0, 0), (0, 1), (1, 0))
    assert regional.origin == (51.5, 10.0, 250.0)
    assert regional.layer_height_km == 400.0
    vtec = metadata.global_vtec
    assert (vtec.coefficient_count, vtec.degree, vtec.order) == (9, 2, 2)
    assert vtec.resolution == pytest.approx(1e-4)
    assert (vtec.phase_time, vtec.layer_height_km) == (50400.0, 400.0)
    (region,) = metadata.regional_troposphere.regions
    assert region.origin == (51.5, 10.0, 250.0)
    assert (region.horizontal_scale, region.vertical_scale) == (700e3, 2000.0)
    (dry,) = region.components
    assert (dry.component, dry.max_horizontal_order, dry.max_vertical_order) == (
        TROPOSPHERE_DRY,
        0,
        0,
    )
    assert region.mapping.parameter_count == 2
    assert region.mapping.max_elevation == 20


def test_grid_definition() -> None:
    definitions = SsrzDecoder().push(_message(ZM013_GRID_3))
    assert isinstance(definitions, GridDefinitions)
    (grid,) = definitions.grids
    assert (grid.grid_id, grid.iod, grid.point_count) == (3, 0, 13)
    assert grid.resolution == pytest.approx(1.4e-5)
    # The reference decoder in radians: points 0, 1 and 12 of the chain.
    np.testing.assert_allclose(
        np.radians(grid.latitudes[[0, 1, 12]]), [0.9235100, 0.9178400, 0.9375097], atol=1e-7
    )
    np.testing.assert_allclose(
        np.radians(grid.longitudes[[0, 1, 12]]), [0.1611960, 0.1456000, 0.1259053], atol=1e-7
    )
    np.testing.assert_array_equal(
        grid.heights, [92, 82, 63, 87, 71, 73, 110, 150, 81, 57, 44, 54, 48]
    )


def test_corrections_wait_for_their_metadata() -> None:
    decoder = SsrzDecoder()
    assert decoder.push(_message(ZM004_GRID_TROPOSPHERE)) is None
    assert decoder.push(_message(ZM003_GRID_IONOSPHERE)) is None
    statistics = decoder.statistics()
    assert statistics.waiting == {4: 1, 3: 1}
    assert statistics.errors == {}
    # Not an SSRZ message at all.
    assert decoder.push(b"\x3f\xb3\x18") is None


def test_low_rate_satellites() -> None:
    decoder = _decoder_with_metadata()
    low_rate = decoder.push(_message(ZM002_LOW_RATE))
    assert isinstance(low_rate, LowRateCorrections)
    assert (low_rate.time_tag, low_rate.iod) == (600, 0)
    assert low_rate.gps_time == 2438 * 604800 + 240000
    assert len(low_rate.satellites) == 34
    assert low_rate.satellites[:3] == ("G06", "G11", "G12")
    assert low_rate.satellites[-2:] == ("C35", "C39")


def test_low_rate_corrections() -> None:
    """Values of the reference decoder for time tag 600."""
    low_rate = _decoder_with_metadata().push(_message(ZM002_LOW_RATE))
    row = low_rate.satellites.index("G06")
    assert low_rate.ephemeris_iods["G06"] == 81
    assert low_rate.clock[row, 0] == pytest.approx(-3.19)
    np.testing.assert_allclose(low_rate.orbit[row], [-0.1920, 0.5190, -0.6664], atol=1e-9)
    assert low_rate.velocity is None
    assert low_rate.code_biases["G06"] == pytest.approx({"1C": 0.0, "2W": -2.0456, "5Q": -0.0853})
    phase = low_rate.phase_biases["G06"]
    assert phase["1C"].cycles == pytest.approx(8.9925, abs=1e-4)
    assert phase["5Q"].cycles == pytest.approx(9.0390, abs=1e-4)
    assert {bias.continuity for bias in phase.values()} == {0}
    # Missing values: G12 has no 5Q.
    assert "5Q" not in low_rate.code_biases["G12"]
    assert "5Q" not in low_rate.phase_biases["G12"]
    assert low_rate.ephemeris_iods["R02"] == 87
    assert low_rate.code_biases["R02"] == pytest.approx({"1C": 0.0, "2P": -0.5559, "2C": -0.5733})
    row = low_rate.satellites.index("C33")
    assert low_rate.clock[row, 0] == pytest.approx(11.82)
    assert low_rate.code_biases["C33"] == pytest.approx({"2I": 0.0, "6I": -13.4290})
    assert low_rate.phase_biases["C33"]["6I"].cycles == pytest.approx(0.1263, abs=1e-4)


def test_high_rate_clock_on_top_of_the_low_rate_one() -> None:
    decoder = _decoder_with_metadata()
    decoder.push(_message(ZM002_LOW_RATE))
    high_rate = decoder.push(_message(ZM001_HIGH_RATE))
    assert isinstance(high_rate, HighRateCorrections)
    assert high_rate.time_tag == 605
    assert len(high_rate.satellites) == 34
    clock = dict(zip(high_rate.satellites, high_rate.clock, strict=True))
    assert clock["G06"] == pytest.approx(0.0146)
    assert clock["R21"] == pytest.approx(-0.0169)
    assert clock["C39"] == pytest.approx(-0.0071)
    assert high_rate.radial is None


def test_high_rate_waits_for_the_low_rate_satellites() -> None:
    assert _decoder_with_metadata().push(_message(ZM001_HIGH_RATE)) is None


def test_continuity_indicator_is_unary_up_to_its_length() -> None:
    bits = _bits("0 10 110 1111 1111")
    assert [bits.unary_limited(4) for _ in range(5)] == [0, 1, 2, 4, 4]
    assert bits.remaining() == 2  # padding to whole bytes


def _decoder_with_low_rate() -> SsrzDecoder:
    decoder = _decoder_with_metadata()
    decoder.push(_message(ZM002_LOW_RATE))
    return decoder


def test_gridded_ionosphere() -> None:
    result = _decoder_with_low_rate().push(_message(ZM003_GRID_IONOSPHERE))
    assert isinstance(result, GridIonosphere)
    assert (result.time_tag, result.grid_id, result.grid_iod) == (600, 3, 0)
    assert len(result.satellites) == 34
    assert result.values.shape == (34, 13)
    assert not result.predicted
    rows = dict(zip(result.satellites, result.values))
    # fmt: off
    expected = {
        "G06": [-0.17, -0.18, NAN, -0.11, -0.26, -0.21, -0.17, -0.15, -0.22, -0.15, -0.18, -0.07, -0.11],
        "E31": [-0.06, 0.05, NAN, -0.20, 0.26, 0.12, 0.16, -0.03, 0.26, 0.31, 0.36, 0.29, 0.30],
        "C39": [0.01, -0.01, NAN, 0.00, -0.04, -0.04, 0.09, 0.06, 0.04, -0.12, -0.11, -0.16, -0.15],
    }
    # fmt: on
    for satellite, values in expected.items():
        np.testing.assert_allclose(rows[satellite], values, atol=1e-9)


def test_gridded_troposphere() -> None:
    result = _decoder_with_metadata().push(_message(ZM004_GRID_TROPOSPHERE))
    assert isinstance(result, GridTroposphere)
    assert (result.time_tag, result.grid_id) == (600, 3)
    (wet,) = result.components[TROPOSPHERE_WET]
    # fmt: off
    np.testing.assert_allclose(
        wet,
        [0.34850, 0.29220, NAN, 0.36241, 0.24851, 0.38908, 0.06243, 0.16801, 0.07795,
         0.16398, 0.26833, 0.16073, 0.15922],
        atol=1e-9,
    )
    # fmt: on


def test_regional_ionosphere() -> None:
    result = _decoder_with_low_rate().push(_message(ZM005_REGIONAL_IONOSPHERE))
    assert isinstance(result, RegionalIonosphere)
    assert result.orders == ((0, 0), (0, 1), (1, 0))
    rows = dict(zip(result.satellites, result.coefficients))
    np.testing.assert_allclose(rows["G06"], [-1.347, -0.909, -0.734], atol=1e-9)
    np.testing.assert_allclose(rows["E31"], [0.143, 0.265, -0.638], atol=1e-9)
    np.testing.assert_allclose(rows["C39"], [1.860, 0.406, -1.138], atol=1e-9)


def test_global_vtec() -> None:
    decoder = _decoder_with_metadata()
    result = decoder.push(_message(ZM006_GLOBAL_VTEC))
    assert isinstance(result, GlobalVtec)
    assert result.valid
    assert (result.degree, result.order) == (2, 2)
    # C_n0, C_n1, C_22 and S_n1, S_22 as the reference decoder lists them.
    np.testing.assert_allclose(result.cosine[:, 0], [-8.1478, 33.3845, -19.2824], atol=1e-9)
    np.testing.assert_allclose(result.cosine[1:, 1], [-74.6353, 22.0997], atol=1e-9)
    assert result.cosine[2, 2] == pytest.approx(13.4947)
    np.testing.assert_allclose(result.sine[1:, 1], [0.7125, -0.9224], atol=1e-9)
    assert result.sine[2, 2] == pytest.approx(27.4107)

    # Eq. 11.25 written out for degree and order 2 at one pierce point.
    latitude, longitude = 51.0, 10.0
    gps_time = result.gps_time
    sun = math.radians(longitude) + ((gps_time % 86400) - 50400) / 86400 * 2 * math.pi
    s, c = math.sin(math.radians(latitude)), math.cos(math.radians(latitude))
    legendre = {
        (0, 0): 1.0,
        (1, 0): math.sqrt(3) * s,
        (1, 1): math.sqrt(3) * c,
        (2, 0): math.sqrt(5) / 2 * (3 * s * s - 1),
        (2, 1): math.sqrt(15) * s * c,
        (2, 2): math.sqrt(15) / 2 * c * c,
    }
    expected = sum(
        (result.cosine[n, m] * math.cos(m * sun) + result.sine[n, m] * math.sin(m * sun)) * p
        for (n, m), p in legendre.items()
    )
    assert result.vtec(latitude, longitude, gps_time) == pytest.approx(expected)
    # Evening in Germany: some 8 TECU.
    assert 5.0 < expected < 12.0


def test_regional_troposphere() -> None:
    result = _decoder_with_metadata().push(_message(ZM007_REGIONAL_TROPOSPHERE))
    assert isinstance(result, RegionalTroposphere)
    (region,) = result.regions
    np.testing.assert_allclose(region.components[TROPOSPHERE_DRY], [0.00968], atol=1e-12)
    np.testing.assert_allclose(region.mapping, [0.000885, -0.005531], atol=1e-12)
    # Order 0: the same scale factor everywhere.
    factor = region.scale_factor(TROPOSPHERE_DRY, [47.5, 54.0], [8.0, 13.0], [500.0, 20.0])
    np.testing.assert_allclose(factor, [0.00968, 0.00968])
    assert region.scale_factor(TROPOSPHERE_WET, 51.0, 10.0, 0.0) is None


def _wrap(data: bytes, continuity: int) -> bytes:
    """A TDC data group (type 0, with CRC) around `data`."""
    raw = bytes([0x40, continuity << 4]) + data
    return raw + (crc16_ccitt(raw) ^ 0xFFFF).to_bytes(2, "big")


def test_rtcm_stream_passes_ssrz_on() -> None:
    decoder = RtcmStreamDecoder()
    frames = (
        ZM009_TIME + ZM011_SATELLITE_GROUPS + ZM012_METADATA,
        ZM013_GRID_3 + ZM002_LOW_RATE,
        ZM004_GRID_TROPOSPHERE + ZM003_GRID_IONOSPHERE,
    )
    results = []
    for index, data in enumerate(frames):
        assert decoder.push(parse_data_group(_wrap(data, index))) == []
        results += decoder.take_ssrz()
    assert [type(r) for r in results] == [
        SsrzTime,
        SatelliteGroups,
        SsrzMetadata,
        GridDefinitions,
        LowRateCorrections,
        GridTroposphere,
        GridIonosphere,
    ]
    assert decoder.take_ssrz() == []
    assert decoder.statistics(1.0).ssrz.decoded == {9: 1, 11: 1, 12: 1, 13: 1, 2: 1, 4: 1, 3: 1}

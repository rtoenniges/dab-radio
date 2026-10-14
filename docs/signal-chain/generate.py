"""Generates the block diagram of the signal processing as an HTML page.

Run from the project directory:

    uv run python docs/signal-chain/generate.py

writes `docs/signal-chain/signal-chain.html`: a self-contained page with two SVG
figures (block diagram and true-to-scale frame layout) that runs in the browser
without further files; only the fonts come from Google Fonts. With
`--fragment PATH` only the page content without the `<html>`/`<head>` scaffold is
produced instead, for environments that supply the scaffold themselves.

The blocks are placed by hand, every coordinate is in the code below, and the texts
describe the state of the modules under `src/`. If the processing chain changes,
this script has to follow. Texts that, by estimated width, do not fit into their
block are reported by the script as warnings. The page is English, like the user
interface and the code. The file is excluded from `ruff format` on purpose: the
compact, hand-aligned calls are the layout.
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
from datetime import datetime
from itertools import pairwise
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "template.html"
DEFAULT_OUT = HERE / "signal-chain.html"

# Separates the head part (title, fonts, styles) from the page content in the template.
BODY_MARKER = "<!-- BODY -->"

# Superscript text in block labels: "e^{−j2π·Δf·t}".
_SUP = re.compile(r"\^\{([^}]*)\}")


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def plain(s: str) -> str:
    return _SUP.sub(lambda m: m.group(1), s)


def rich(s: str) -> str:
    """Typesets `^{...}` as superscript tspans."""
    out, pos, shifted = [], 0, False
    for m in _SUP.finditer(s):
        before = esc(s[pos : m.start()])
        out.append(f'<tspan dy="4.5">{before}</tspan>' if shifted else before)
        out.append(f'<tspan dy="-4.5" font-size="8.5">{esc(m.group(1))}</tspan>')
        shifted, pos = True, m.end()
    rest = esc(s[pos:])
    out.append(f'<tspan dy="4.5">{rest or "&#8203;"}</tspan>' if shifted else rest)
    return "".join(out)


def text_width(s: str, size: float, k: float) -> float:
    """Rough text width: character count times mean character width (`k` in em)."""
    return len(plain(s)) * size * k


class Svg:
    """Collects the SVG elements of one figure. The looks come from the CSS classes
    in the template so that both colour schemes (light/dark) apply."""

    def __init__(self, marker_prefix: str) -> None:
        self.parts: list[str] = []
        self.warnings: list[str] = []
        self.size: tuple[float, float] = (0, 0)  # viewBox width and height
        self._prefix = marker_prefix

    def add(self, s: str) -> None:
        self.parts.append(s)

    def render(self, width: int, height: int, aria_label: str, extra_class: str = "") -> str:
        cls = f"diagram {extra_class}".strip()
        return (
            f'<svg class="{cls}" viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="{esc(aria_label)}">{self._markers()}{"".join(self.parts)}</svg>'
        )

    def _markers(self) -> str:
        # IDs of their own per figure: all SVGs live in the same document.
        return "<defs>" + "".join(
            f'<marker id="{self._prefix}-{kind}" viewBox="0 0 10 10" refX="9.5" refY="5" '
            'markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" '
            f'orient="auto-start-reverse"><path d="M0,0.8 L10,5 L0,9.2 z" class="d-ah-{kind}"/>'
            "</marker>"
            for kind in ("flow", "ctrl")
        ) + "</defs>"

    def text(self, x, y, s, cls, anchor="start"):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{rich(s)}</text>')

    def arrow(self, pts, kind="flow", head=True):
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        marker = f' marker-end="url(#{self._prefix}-{kind})"' if head else ""
        self.add(f'<path d="{d}" class="d-{kind}"{marker}/>')

    def dot(self, x, y, kind="flow"):
        self.add(f'<circle cx="{x}" cy="{y}" r="3.2" class="d-dot-{kind}"/>')

    def label(self, x, y, s, anchor="middle", kind="flow"):
        self.text(x, y, s, "d-lbl" + (" d-lbl-ctrl" if kind == "ctrl" else ""), anchor)

    def badge(self, x, yc, label, kind="disp", align="right") -> float:
        """Badge "visible in the GUI" (disp) or "adjustable" (ctrl)."""
        w = text_width(label, 10.5, 0.5) + 24
        x0 = x - w if align == "right" else x
        if kind == "disp":
            glyph = (
                f'<rect x="{x0 + 7:.1f}" y="{yc - 4.5:.1f}" width="9" height="6.5" rx="1" class="g"/>'
                f'<path d="M{x0 + 9.5:.1f},{yc + 4.5:.1f} h4" class="g"/>'
            )
        else:
            glyph = (
                f'<path d="M{x0 + 6.5:.1f},{yc - 2.5:.1f} h10 M{x0 + 6.5:.1f},{yc + 2.5:.1f} h10" '
                'class="g"/>'
                f'<circle cx="{x0 + 9.5:.1f}" cy="{yc - 2.5:.1f}" r="1.8" class="gf"/>'
                f'<circle cx="{x0 + 14:.1f}" cy="{yc + 2.5:.1f}" r="1.8" class="gf"/>'
            )
        self.add(
            f'<g class="d-badge d-badge-{kind}" data-d="{esc(label)}" tabindex="0">'
            f'<rect x="{x0:.1f}" y="{yc - 8.5:.1f}" width="{w:.1f}" height="17" rx="2.5"/>'
            f"{glyph}"
            f'<text x="{x0 + 20.5:.1f}" y="{yc + 3.7:.1f}">{esc(label)}</text></g>'
        )
        return w

    def block(self, x, y, w, title, lines=(), h=58, disp=(), ctrl=None) -> None:
        """Processing block: title, detail lines, display badges at the top right,
        control badge below the bottom edge."""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" class="d-block"/>')
        n = len(lines)
        ty = y + (h - (13 + 14 * n)) / 2 + 13
        cx = x + w / 2
        if text_width(title, 14, 0.5) > w - 12:
            self.warnings.append(f"Title too wide: {title!r}")
        self.text(cx, ty, title, "d-title", "middle")
        for i, line in enumerate(lines):
            if text_width(line, 11.5, 0.46) > w - 12:
                self.warnings.append(f"Line too wide: {line!r}")
            self.text(cx, ty + 16 + 14 * i, line, "d-detail", "middle")
        used = 0.0
        for label in reversed(disp):
            used += self.badge(x + w - 6 - used, y, label, "disp") + 4
        if used - 4 > w - 12:
            self.warnings.append(f"Display badges too wide in {title!r}")
        if ctrl and self.badge(x + 6, y + h + 6, ctrl, "ctrl", align="left") > w - 12:
            self.warnings.append(f"Control badge too wide in {title!r}")

    def band(self, x, y, w, h, kind) -> None:
        """Background band of a thread."""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" class="d-band-{kind}"/>')

    def band_label(self, x, y, title, module="", badge=None) -> None:
        self.text(x, y, title, "d-band-label")
        mx = x + text_width(title, 12, 0.62) + 12
        if module:
            self.text(mx, y, module, "d-mod")
        if badge:
            self.badge(mx + text_width(module, 10.5, 0.6) + 14, y - 4, badge, "disp", align="left")

    def group(self, x, y, w, h, title, module="") -> None:
        """Processing stage: dashed frame, name on the left, modules on the right."""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" class="d-group"/>')
        self.text(x + 12, y + 17, title, "d-group-label")
        if module:
            self.text(x + w - 12, y + 17, module, "d-mod", "end")
            if x + 24 + text_width(title, 11.5, 0.62) > x + w - 12 - text_width(module, 10.5, 0.6):
                self.warnings.append(f"Group label too wide: {title!r}")


# ---------------------------------------------------------------------------
# Figure 1: block diagram
# ---------------------------------------------------------------------------
def build_block_diagram() -> Svg:
    s = Svg("b")
    width = 1400

    # Bands: one per thread; the DSP thread reaches down into the PAD row at the
    # bottom left, with the audio thread next to it on the right.
    s.band(0, 0, 620, 146, "hw")
    s.band(636, 0, 764, 146, "sdr")
    s.band(0, 178, 1400, 1214, "dsp")
    s.band(0, 1382, 492, 292, "dsp")
    s.band(508, 1408, 892, 266, "audio")

    # --- Hardware and SDR thread ------------------------------------------
    s.band_label(16, 22, "HARDWARE", "Antenna and RTL-SDR stick (or RSPduo, see caption)")
    s.band_label(652, 22, "SDR THREAD", "RtlSdrWorker (RspDuoWorker) · sdr/worker.py")

    s.add('<path d="M54,44 L86,44 L70,66 Z" class="d-sym"/><path d="M70,66 V87" class="d-flow"/>')
    s.text(70, 112, "Antenna", "d-title", "middle")
    s.text(70, 127, "Band III, 174–240 MHz", "d-detail", "middle")

    s.group(140, 32, 460, 104, "RTL-SDR STICK")
    s.block(156, 58, 200, "Tuner (e.g. R820T)",
            ["LNA, amplification (Gain/AGC)", "Mixing to intermediate frequency"],
            ctrl="Channel · Gain/AGC · PPM")
    s.block(390, 58, 198, "RTL2832U", ["8-bit ADC, mixing to 0 Hz", "Decimation to 2.048 MSps"])
    s.block(690, 58, 190, "librtlsdr", ["asynchronous: rtlsdr_read_async", "gapless sample stream"])
    s.block(920, 58, 230, "RtlSdrWorker",
            ["Blocks of 393 216 samples", "= 2 frames = 192 ms, complex64"])

    s.arrow([(70, 87), (156, 87)])
    s.label(113, 80, "RF")
    s.arrow([(356, 87), (390, 87)])
    s.label(373, 80, "IF")
    s.arrow([(588, 87), (690, 87)])
    s.label(639, 80, "USB 2.0")
    s.label(639, 103, "I/Q, 8 bits each")
    s.arrow([(880, 87), (920, 87)])

    s.arrow([(1035, 116), (1035, 162), (24, 162), (24, 363), (150, 363)])
    s.label(530, 157, "Qt signal samples_ready: every IQ block lands in the queue "
                      "of the DSP thread")

    # --- Preprocessing and frame synchronisation ---------------------------
    s.band_label(16, 200, "DSP THREAD", "DspWorker · dsp/pipeline.py", badge="Load")
    s.group(36, 212, 848, 196, "PREPROCESSING & FRAME SYNCHRONISATION",
            "dsp/iq_correction.py · dsp/spectrum.py · dsp/sync.py")
    s.block(48, 240, 170, "Overload", ["Share of |I| or |Q| > 0.99", "measured on the raw signal"],
            disp=["Overload"])
    s.block(290, 240, 170, "Spectrum", ["Welch: 2 048-point FFT, Hann", "averaged per IQ block"],
            disp=["Spectrum"])
    s.block(150, 334, 180, "DC/IQ correction",
            ["DC offset, IQ errors", "estimated blindly, default off"],
            disp=["Receiver errors"], ctrl="DC · IQ on/off")
    s.block(414, 334, 180, "Frame buffer", ["Concatenates blocks", "no frame is lost"])
    s.block(628, 334, 230, "Null symbol search",
            ["Power minimum over 2 656 samples", "first whole frame, then ±1 024"],
            disp=["Power envelope"])
    s.dot(110, 363)
    s.arrow([(110, 363), (110, 298)])
    s.arrow([(330, 363), (414, 363)])
    s.dot(372, 363)
    s.arrow([(372, 363), (372, 298)])
    s.arrow([(594, 363), (628, 363)])

    s.arrow([(700, 392), (700, 440), (24, 440), (24, 603), (590, 603)])
    s.label(362, 435, "Frame: 196 608 samples = 96 ms, starting at the null symbol")

    s.arrow([(1262, 574), (1262, 424), (820, 424), (820, 392)], kind="ctrl")
    s.label(830, 419, "10 frames in a row without a reliable PRS → new search", anchor="start", kind="ctrl")

    # --- OFDM demodulator ---------------------------------------------------
    s.group(36, 452, 1358, 290, "OFDM DEMODULATOR",
            "dsp/ofdm.py · dsp/timing.py · dsp/channel.py · dsp/tii.py")
    # Same order as in dsp.ofdm.demodulate_frame_symbols: the fractional part
    # first, because the guard correlation does not see whole kHz at all (a
    # rotation by a full 2π per millisecond), while the carrier mask only works
    # unambiguously on carriers sitting cleanly in their bins.
    s.block(48, 480, 150, "Δf fine", ["Guard correlation, ±500 Hz", "blind to whole kHz"])
    s.block(222, 480, 150, "Δf integer", ["Shift carrier mask", "whole kHz, ±64 kHz"])
    s.block(396, 480, 150, "Frame position", ["PRS cross-correlation", "unreliable: last values"],
            disp=["Frequency offset"])
    s.arrow([(198, 509), (222, 509)])
    s.arrow([(372, 509), (396, 509)])
    s.dot(70, 603)
    s.arrow([(70, 603), (70, 538)])
    s.arrow([(546, 509), (610, 509), (610, 574)], kind="ctrl")
    s.label(618, 552, "Δf, offset", anchor="start", kind="ctrl")

    # No fine timing correction between FFT and D-QPSK: its phase ramp across the
    # carriers is the same in every symbol and cancels in z(l) · z*(l−1).
    mx = [590, 806, 1022, 1238]
    s.block(mx[0], 574, 142, "Δf correction", ["Adjust frame position", "× e^{−j2π·Δf·t}"])
    s.block(mx[1], 574, 142, "FFT per symbol", ["2 048 points, no guard", "→ 1 536 carriers"],
            disp=["FFT output"])
    s.block(mx[2], 574, 142, "D-QPSK", ["Differential detection", "z(l) · z*(l−1)"])
    s.block(mx[3], 574, 142, "Residual phase", ["z⁴ estimate per symbol", "and rotate it out"],
            disp=["Constellation"])
    for a, b in pairwise(mx):
        s.arrow([(a + 142, 603), (b, 603)])

    s.block(48, 668, 190, "Transmitter ID (TII)", ["FFT of the null symbol", "averaged over 32 frames"],
            disp=["Transmitters (TII)"])
    s.arrow([(70, 603), (70, 668)])
    s.label(78, 648, "Null symbol", anchor="start")
    # Below the FFT, from which the PRS spectrum comes.
    cx = [806, 968, 1130]
    s.block(cx[0], 668, 142, "Channel estimate", ["PRS × reference*", "per carrier"])
    s.arrow([(877, 632), (877, 668)])
    s.label(885, 653, "PRS (l = 1)", anchor="start")
    s.block(cx[1], 668, 142, "Impulse response", ["IFFT, Hann window", "−250 … +750 µs"],
            disp=["Impulse response"])
    s.block(cx[2], 668, 142, "Path shares", ["Power outside", "±1.5 µs of the main path"],
            disp=["SFN share"])
    for a, b in pairwise(cx):
        s.arrow([(a + 142, 697), (b, 697)])

    s.arrow([(1309, 632), (1309, 772), (24, 772), (24, 906), (48, 906)])
    s.label(660, 767, "Data symbols l = 2–76, 1 536 D-QPSK points each "
                      "(without a selected station or data service only the first 16)")

    # --- Channel decoding FIC + MSC ----------------------------------------
    s.group(36, 788, 1358, 220, "CHANNEL DECODING",
            "dsp/deinterleave.py · dsp/qpsk.py · dsp/fic.py · dsp/msc.py · "
            "dsp/time_interleave.py · dsp/eep.py · dsp/uep.py · dsp/convolutional.py · dab/fig.py")
    s.block(48, 877, 190, "Frequency deinterleaving",
            ["Π(i) = 13·Π(i−1) + 511", "mod 2 048, ETSI 14.6"])
    s.block(262, 877, 150, "QPSK demapping", ["Soft bits: Re and Im", "1 point → 2 values"])
    s.arrow([(238, 906), (262, 906)])
    s.arrow([(412, 906), (450, 906)], head=False)
    s.arrow([(450, 851), (450, 961)], head=False)
    s.dot(450, 906)
    s.arrow([(450, 851), (530, 851)])
    s.arrow([(450, 961), (530, 961)])
    s.label(490, 845, "FIC · l = 2–4")
    s.label(490, 955, "MSC · l = 5–76")

    fx = [530, 704, 878, 1052, 1226]
    s.block(fx[0], 822, 154, "Depuncturing", ["PI 16, PI 15, PI X", "4 × 2 304 → 3 096 bits"])
    s.block(fx[1], 822, 154, "Viterbi decoder", ["Soft decision, K = 7", "64 states, numba JIT"])
    s.block(fx[2], 822, 154, "Descrambling", ["PRBS x⁹ + x⁵ + 1", "→ 4 × 768 bits"])
    s.block(fx[3], 822, 154, "CRC-16", ["12 FIBs of 256 bits", "faulty ones discarded"])
    s.block(fx[4], 822, 154, "FIG evaluation", ["FIG 0/0–3, 0/8, 0/10, 0/13", "0/14, 0/17–19, 0/21, FIG 1"],
            disp=["Station list"])
    s.block(fx[0], 932, 154, "Subchannel selection", ["Start + size in CU", "from 4 CIFs of 55 296 bits"],
            disp=["Raw MSC bits"], ctrl="Station")
    # The data service is decoded by a second instance of the same chain.
    s.badge(fx[0] + 6 + text_width("Station", 10.5, 0.5) + 28, 932 + 58 + 6,
            "Data service", "ctrl", align="left")
    s.block(fx[1], 932, 154, "Time deinterleaver", ["Convolutional over 16 CIFs", "= 384 ms delay"])
    s.block(fx[2], 932, 154, "Depuncturing", ["EEP (DAB+), UEP (DAB)", "2 or 4 sections + PI X"])
    s.block(fx[3], 932, 154, "Viterbi decoder", ["Soft decision, K = 7", "once per CIF, numba JIT"])
    s.block(fx[4], 932, 154, "Descrambling", ["PRBS x⁹ + x⁵ + 1", "→ 24 · bitrate bits"],
            disp=["Decoded bits"])
    for a, b in pairwise(fx):
        s.arrow([(a + 154, 851), (b, 851)])
        s.arrow([(a + 154, 961), (b, 961)])

    s.arrow([(1303, 880), (1303, 906), (548, 906), (548, 932)], kind="ctrl")
    s.dot(898, 906, kind="ctrl")
    s.arrow([(898, 906), (898, 932)], kind="ctrl")
    s.label(560, 901, "Station selection + FIG 0/1, 0/2 (data service: + FIG 0/3, announcement: + FIG 0/19) → "
                      "start address, size and protection profile of the subchannel", anchor="start", kind="ctrl")

    s.arrow([(1303, 990), (1303, 1034), (24, 1034), (24, 1117), (48, 1117)])
    s.label(660, 1029, "Logical frames: 24 · bitrate bits per 24 ms – onwards as DAB+ (63) "
                       "or DAB (0), depending on ASCTy")

    # --- Data service in packet mode (second chain) --------------------------
    s.dot(1270, 1034)
    s.arrow([(1270, 1034), (1270, 1084)])
    s.label(1278, 1049, "Data service", anchor="start")
    # Blocks narrowed by 8 on the left: room for the branch from the data groups to
    # the two TDC streams, past the carousel. The three applications get one detail
    # line each to fit under one another.
    s.group(1176, 1054, 218, 326, "PACKET MODE", "dsp/packet.py")
    s.block(1196, 1084, 186, "Packet FEC", ["RS(204, 188), t = 8, per row",
                                            "Table 12 × 204, column by column"], h=54,
            disp=["FEC frame"])
    s.block(1196, 1148, 186, "Packets & data groups", ["24–96 bytes, CRC-16, address",
                                                       "First/Last → MSC data group"], h=54,
            disp=["Packet statistics"])
    s.block(1196, 1212, 186, "MOT carousel, SPI", ["Files; SPI → programme schedule"], h=44,
            disp=["Carousel", "Programme guide"])
    s.block(1196, 1266, 186, "RTCM 3 (GEPOS)", ["CRC-24Q, orbit data, SSRZ, NTRIP"], h=44,
            disp=["Sky", "RTCM", "SSRZ"])
    s.block(1196, 1320, 186, "TPEG (ARD TPEG)", ["SNI, TEC, TFP; TMC and OpenLR"], h=44,
            disp=["Messages", "Map"])
    s.arrow([(1204, 1138), (1204, 1148)])
    s.arrow([(1204, 1202), (1204, 1212)])
    s.arrow([(1196, 1190), (1185, 1190), (1185, 1342), (1196, 1342)])
    s.dot(1185, 1288)
    s.arrow([(1185, 1288), (1196, 1288)])
    # Shortened to fit the width of the group (the full list ran past the figure).
    s.text(1188, 1376, "dab/{spi,rtcm,ssrz,tpeg,…}.py", "d-mod")

    # --- DAB+ superframe ---------------------------------------------------
    s.group(36, 1054, 1104, 142, "DAB+ SUPERFRAME", "dsp/superframe.py · dsp/reed_solomon.py")
    px = [48, 272, 496, 720, 944]
    s.block(px[0], 1088, 180, "Assemble superframe",
            ["5 logical frames = 120 ms", "Bits → 15 · bitrate bytes"])
    s.block(px[1], 1088, 180, "Reed-Solomon",
            ["RS(120, 110), numba JIT", "up to 5 byte errors per word"])
    s.block(px[2], 1088, 180, "Fire code sync",
            ["CRC over the header bytes", "finds the superframe boundary"],
            disp=["Audio frames"])
    s.block(px[3], 1088, 180, "Read header", ["Format byte: rate, SBR, PS", "Start addresses of the AUs"])
    s.block(px[4], 1088, 180, "Audio units", ["CRC-16 per AU", "only valid ones pass on"],
            disp=["Frame layout"])
    for a, b in pairwise(px):
        s.arrow([(a + 180, 1117), (b, 1117)])
    s.arrow([(586, 1146), (586, 1168), (138, 1168), (138, 1146)], kind="ctrl")
    s.label(150, 1184, "Fire code wrong → shift the grid by one logical frame (24 ms)",
            anchor="start", kind="ctrl")

    # --- DAB audio frames (MP2) --------------------------------------------
    s.dot(24, 1117)
    s.arrow([(24, 1117), (24, 1273), (48, 1273)])
    s.group(36, 1210, 1104, 142, "DAB AUDIO FRAMES (MP2)", "dsp/mp2_frame.py")
    s.block(px[0], 1244, 180, "Frame search", ["Sync word, Layer II", "Bitrate of the subchannel"])
    s.block(px[1], 1244, 180, "Header CRC", ["CRC-16 over bit allocation", "and scale factor selection"],
            disp=["Audio frames"])
    s.block(px[2], 1244, 180, "Follow the grid", ["48 kHz: 1 frame every 24 ms", "24 kHz: every 48 ms"])
    s.block(px[3], 1244, 180, "Split off PAD", ["F-PAD, ScF-CRC, X-PAD", "X-PAD length from the CIs"],
            disp=["Frame layout"])
    s.block(px[4], 1244, 180, "MP2 frames", ["no Reed-Solomon", "wrong CRC → silence"])
    for a, b in pairwise(px):
        s.arrow([(a + 180, 1273), (b, 1273)])
    s.arrow([(586, 1302), (586, 1324), (138, 1324), (138, 1302)], kind="ctrl")
    s.label(150, 1340, "5 frames without a valid header → frame search starts over",
            anchor="start", kind="ctrl")

    s.arrow([(1124, 1117), (1160, 1117), (1160, 1378), (24, 1378), (24, 1541), (48, 1541)])
    s.arrow([(1124, 1273), (1160, 1273)], head=False)
    s.dot(1160, 1273)
    s.dot(520, 1378)
    s.arrow([(520, 1378), (520, 1541), (540, 1541)])
    s.dot(520, 1469)
    s.arrow([(520, 1469), (540, 1469)])
    s.label(840, 1373, "Audio units (DAB+) or MP2 frames (DAB), CRC checked")
    s.label(530, 1403, "Qt signals superframe_ready, mp2_frame_ready", anchor="start")

    # --- PAD (DSP thread) and audio thread ---------------------------------
    s.group(36, 1402, 444, 256, "PROGRAMME-ASSOCIATED DATA", "dab/{pad,dls,mot,journaline}.py")
    s.block(48, 1512, 160, "PAD reader", ["DAB+: DSE of the AU", "DAB: end of the frame"])
    s.arrow([(208, 1541), (236, 1541)], head=False)
    s.arrow([(236, 1469), (236, 1613)], head=False)
    s.dot(236, 1541)
    s.arrow([(236, 1469), (270, 1469)])
    s.arrow([(236, 1541), (270, 1541)])
    s.arrow([(236, 1613), (270, 1613)])
    s.block(270, 1440, 196, "Dynamic Label", ["Segments, toggle bit, CRC", "DL Plus: title, artist …"],
            disp=["Dynamic Label", "DL Plus"])
    s.block(270, 1512, 196, "MOT decoder", ["Data groups → header + body", "complete image → slideshow"],
            disp=["Slideshow"])
    s.block(270, 1584, 196, "Journaline", ["Data groups → JML objects",
                                            "AppTy and entry point: FIG 0/13"],
            disp=["Journaline pages"])

    s.band_label(536, 1430, "AUDIO THREAD",
                 "AudioPlayer · audio/player.py · audio/aac.py · audio/mp2.py")
    s.block(540, 1440, 220, "AAC decoder (FAAD2)",
            ["HE-AAC v2 via ctypes", "960 samples per audio unit"])
    s.block(540, 1512, 220, "MP2 decoder (mpg123)",
            ["MPEG Layer II via ctypes", "1 152 samples per frame"])
    s.block(800, 1476, 180, "Playback buffer", ["Prebuffer 0.5 s", "at most 1 s latency"],
            disp=["Audio level", "Audio"])
    s.block(1020, 1476, 170, "QAudioSink", ["16 bits, 24 to 48 kHz", "to the sound card"],
            ctrl="Audio · Volume")
    s.arrow([(760, 1469), (780, 1469), (780, 1505), (800, 1505)])
    s.arrow([(760, 1541), (780, 1541), (780, 1505)], head=False)
    s.dot(780, 1505)
    s.label(790, 1498, "PCM")
    s.arrow([(980, 1505), (1020, 1505)])
    s.arrow([(1190, 1505), (1232, 1505)])
    s.add(
        '<path d="M1234,1497 h9 v16 h-9 z M1243,1497 L1257,1486 V1524 L1243,1513" class="d-sym"/>'
        '<path d="M1263,1496 q6,9 0,18 M1270,1490 q10,15 0,30" class="d-flow"/>'
    )
    s.text(1252, 1548, "Loudspeaker", "d-detail", "middle")

    # --- GUI thread: where the badges from above end up in the GUI ---------
    gui_y = 1690
    # Two rows of panels: ten do not fit next to each other into the 1 400.
    panel_rows = [
        [
            ("Status lines", "disp", ["Frequency offset", "SFN share", "Overload",
                                      "Receiver errors", "Load"], 242),
            ("“Signal” tab", "disp",
             ["Spectrum", "Power envelope", "Constellation", "Impulse response"], 170),
            ("“OFDM” tab", "disp", ["FFT output"], 124),
            ("“Station” tab", "disp",
             ["Raw MSC bits", "Decoded bits", "Frame layout", "Audio level"], 170),
            ("“Programme” tab", "disp", ["Dynamic Label", "DL Plus", "Slideshow"], 154),
            ("“Journaline” tab", "disp", ["Journaline pages"], 146),
            ("Side panel", "disp",
             ["Station list", "Transmitters (TII)", "Audio frames", "Audio"], 170),
        ],
        [
            ("“Data services” tab", "disp", ["FEC frame", "Packet statistics", "Carousel"], 180),
            ("“Programme guide” tab", "disp", ["Programme guide"], 180),
            # The sub-tabs of "GNSS"; as "… tab" they would not fit into the row.
            ("GNSS › Satellites", "disp", ["Sky", "RTCM"], 130),
            ("GNSS › Atmosphere", "disp", ["SSRZ"], 150),
            ("GNSS › Corrections", "disp", ["SSRZ"], 150),
            ("“Traffic” tab", "disp", ["Messages", "Map"], 150),
            ("Controls", "ctrl",
             ["Channel · Gain/AGC · PPM", "DC · IQ on/off", "Station", "Data service",
              "Audio · Volume"], 250),
        ],
    ]
    gap = 12
    row_layouts, max_h = [], 0
    for panels in panel_rows:
        x = (width - sum(p[3] for p in panels) - gap * (len(panels) - 1)) / 2
        layouts = []
        for title, kind, chips, pw in panels:
            rows, cx, row = [], 0.0, []
            for chip in chips:
                cw = text_width(chip, 10.5, 0.5) + 24
                if row and cx + cw > pw - 24:
                    rows.append(row)
                    row, cx = [], 0.0
                row.append((chip, cx))
                cx += cw + 6
            rows.append(row)
            max_h = max(max_h, 36 + len(rows) * 24)
            layouts.append((title, kind, rows, pw, x))
            x += pw + gap
        row_layouts.append(layouts)

    gui_h = 40 + len(row_layouts) * (max_h + gap) - gap + 18
    s.band(0, gui_y, width, gui_h, "gui")
    s.band_label(16, gui_y + 22, "GUI THREAD",
                 "MainWindow on ReceiverController · ui/ · plots with pyqtgraph, drawn by a timer")
    for index, layouts in enumerate(row_layouts):
        py = gui_y + 40 + index * (max_h + gap)
        for title, kind, rows, pw, px0 in layouts:
            s.add(f'<rect x="{px0:.1f}" y="{py}" width="{pw}" height="{max_h}" rx="5" '
                  'class="d-panel"/>')
            s.text(px0 + 12, py + 20, title, "d-panel-title")
            for r, row in enumerate(rows):
                for chip, cx in row:
                    s.badge(px0 + 12 + cx, py + 38 + r * 24, chip, kind, align="left")

    s.size = (width, gui_y + gui_h)
    return s


# ---------------------------------------------------------------------------
# Figure 2: layout of a transmission frame (to scale)
# ---------------------------------------------------------------------------
# Mode I in samples at 2.048 MSps, as in dab.constants.MODE_I.
NULL_SYMBOL = 2656
SYMBOL = 2552  # 504 guard + 2 048 useful part
GUARD = 504
FRAME = NULL_SYMBOL + 76 * SYMBOL  # 196 608 = 96 ms
CIF_SYMBOLS = 18

# The symbol the magnifier enlarges. l = 44 is placed so that the magnifier cone
# passes between the axis labels 48 and 64 ms.
ZOOM_SYMBOL = 44


def build_frame_figure() -> Svg:
    f = Svg("r")
    x0, x1 = 40.0, 1360.0

    def xs(sample: float) -> float:
        return x0 + sample / FRAME * (x1 - x0)

    def symbol_start(l_index: int) -> int:
        """First sample of OFDM symbol l (l = 1 is the PRS)."""
        return NULL_SYMBOL + (l_index - 1) * SYMBOL

    by, bh = 110, 44
    prs0, fic0, msc0 = symbol_start(1), symbol_start(2), symbol_start(5)
    cif = CIF_SYMBOLS * SYMBOL

    # --- Frame -------------------------------------------------------------
    f.add(f'<rect x="{xs(0):.1f}" y="{by}" width="{xs(NULL_SYMBOL) - xs(0):.1f}" height="{bh}" '
          'class="f-null"/>')
    f.add(f'<rect x="{xs(prs0):.1f}" y="{by}" width="{xs(fic0) - xs(prs0):.1f}" height="{bh}" '
          'class="f-prs"/>')
    f.add(f'<rect x="{xs(fic0):.1f}" y="{by}" width="{xs(msc0) - xs(fic0):.1f}" height="{bh}" '
          'class="f-fic"/>')
    for k in (1, 2):
        t = xs(fic0 + k * SYMBOL)
        f.add(f'<path d="M{t:.1f},{by} v{bh}" class="f-tick-fic"/>')
    for k in range(4):
        a, b = msc0 + k * cif, msc0 + (k + 1) * cif
        f.add(f'<rect x="{xs(a):.1f}" y="{by}" width="{xs(b) - xs(a):.1f}" height="{bh}" '
              'class="f-msc"/>')
        for j in range(1, CIF_SYMBOLS):
            t = xs(a + j * SYMBOL)
            f.add(f'<path d="M{t:.1f},{by} v6 M{t:.1f},{by + bh} v-6" class="f-tick"/>')
        first = 5 + CIF_SYMBOLS * k
        cx = (xs(a) + xs(b)) / 2
        f.text(cx, by + 19, f"CIF {k} · l = {first}–{first + CIF_SYMBOLS - 1}", "d-title", "middle")
        f.text(cx, by + 35, "18 symbols · 55 296 bits", "d-detail", "middle")

    # Staircase labels for the narrow sections at the start of the frame.
    for lx, top, ty, head, rest in [
        (xs(NULL_SYMBOL / 2), 34, 30, "Null symbol",
         "2 656 samples = 1.3 ms transmission pause · frame synchronisation, TII carriers"),
        (xs(prs0 + SYMBOL / 2), 56, 52, "Phase reference symbol (PRS, l = 1)",
         "known phases · frame position, channel estimate, D-QPSK start"),
        (xs(fic0 + 1.5 * SYMBOL), 78, 74, "FIC, l = 2–4",
         "12 FIBs · station list and position of the subchannels"),
    ]:
        f.add(f'<path d="M{lx:.1f},{top} V{by}" class="f-leader"/>')
        f.add(f'<text x="{lx + 7:.1f}" y="{ty}" class="d-detail">'
              f'<tspan class="f-head">{esc(head)}</tspan>  {esc(rest)}</text>')
    f.add(f'<path d="M{xs(msc0):.1f},{by - 8} v-6 H{x1} v6" class="f-leader"/>')
    f.add(f'<text x="{x1}" y="{by - 20}" class="d-detail" text-anchor="end">'
          '<tspan class="f-head">MSC, l = 5–76</tspan>  four CIFs of 864 CU × 64 bits · '
          "a subchannel occupies the same CUs in every CIF</text>")

    # --- Magnifier on a single MSC symbol ----------------------------------
    z0, z1 = xs(symbol_start(ZOOM_SYMBOL)), xs(symbol_start(ZOOM_SYMBOL + 1))
    iy, ih, iw = 260, 38, 560
    i0 = (z0 + z1) / 2 - iw / 2
    i1 = i0 + iw
    g1 = i0 + GUARD / SYMBOL * iw  # end of the guard interval
    tail = i1 - GUARD / SYMBOL * iw  # start of the last 504 samples
    zb = by + bh + 4

    # The cone first, so that time axis and frame lie on top of it.
    f.add(f'<path d="M{z0:.1f},{zb} L{z1:.1f},{zb} L{i1:.1f},{iy} L{i0:.1f},{iy} Z" class="f-cone"/>')

    ay = by + bh + 12
    f.add(f'<path d="M{x0},{ay} H{x1}" class="f-axis"/>')
    for t in (0, 16, 32, 48, 64, 80, 96):
        x = x0 + t / 96 * (x1 - x0)
        f.add(f'<path d="M{x:.1f},{ay} v5" class="f-axis"/>')
        f.text(x, ay + 18, f"{t} ms" if t in (0, 96) else str(t), "d-detail", "middle")

    f.add(f'<rect x="{z0:.1f}" y="{by - 4}" width="{z1 - z0:.1f}" height="{bh + 8}" '
          'class="f-zoomsym"/>')

    f.add(f'<rect x="{i0:.1f}" y="{iy}" width="{g1 - i0:.1f}" height="{ih}" class="f-copy"/>')
    f.add(f'<rect x="{g1:.1f}" y="{iy}" width="{i1 - g1:.1f}" height="{ih}" class="f-useful"/>')
    f.add(f'<rect x="{tail:.1f}" y="{iy}" width="{i1 - tail:.1f}" height="{ih}" class="f-copy"/>')
    f.add(f'<rect x="{i0:.1f}" y="{iy}" width="{iw}" height="{ih}" class="f-outline"/>')
    f.add(f'<path d="M{g1:.1f},{iy} v{ih}" class="f-outline"/>')
    f.add(f'<path d="M{tail:.1f},{iy} v{ih}" class="f-divider"/>')

    f.text((i0 + g1) / 2, iy + 16, "Guard", "d-title", "middle")
    f.text((i0 + g1) / 2, iy + 30, "504 · 246 µs", "d-detail", "middle")
    f.text((g1 + tail) / 2, iy + 16, "Useful part: 2 048 samples = 1 ms", "d-title", "middle")
    f.text((g1 + tail) / 2, iy + 30, "FFT → 1 536 carriers, 1 kHz apart", "d-detail", "middle")
    f.text((tail + i1) / 2, iy + 16, "End", "d-title", "middle")
    f.text((tail + i1) / 2, iy + 30, "504 samples", "d-detail", "middle")
    f.label(i0 - 12, iy + 23, f"Symbol l = {ZOOM_SYMBOL}, enlarged", anchor="end")

    # The guard interval is a copy of the end of the symbol.
    copy_y = iy + ih + 18
    f.arrow([((tail + i1) / 2, iy + ih), ((tail + i1) / 2, copy_y),
             ((i0 + g1) / 2, copy_y), ((i0 + g1) / 2, iy + ih)])
    f.label((i0 + i1) / 2, copy_y + 18, "Copy: the guard interval repeats the end of the "
                                        "useful part. Their correlation yields Δf fine.")

    f.size = (1400, copy_y + 30)
    return f


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def code_state() -> str:
    """Date and commit for the footer."""
    today = f"{datetime.now().astimezone():%Y-%m-%d}"
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=HERE, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return f"Generated on {today}"
    return f"Generated on {today} at commit {commit}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fragment", type=Path, metavar="PATH",
                        help="write only the page content without the <html>/<head> scaffold to PATH")
    args = parser.parse_args()

    blocks = build_block_diagram()
    frame = build_frame_figure()
    fig1 = blocks.render(
        *blocks.size,
        "Block diagram of the signal processing: antenna, RTL-SDR stick, SDR thread, "
        "preprocessing and frame synchronisation, OFDM demodulator, channel decoding of FIC "
        "and MSC, DAB+ superframe or DAB audio frames (MP2), programme-associated data, "
        "audio thread and user interface",
    )
    fig2 = frame.render(
        *frame.size,
        "True-to-scale layout of a DAB transmission frame in Mode I: null symbol, "
        "phase reference symbol, three FIC symbols and 72 MSC symbols in four CIFs, 96 ms "
        f"in total; below it symbol l = {ZOOM_SYMBOL} enlarged with guard interval and useful part",
        "frame",
    )

    head, body = TEMPLATE.read_text(encoding="utf-8").split(BODY_MARKER)
    body = body.replace("{{FIG1}}", fig1).replace("{{FIG2}}", fig2)
    body = body.replace("{{GENERATED}}", esc(code_state()))

    if args.fragment:
        out, page = args.fragment, head.strip() + "\n\n" + body.strip() + "\n"
    else:
        out = DEFAULT_OUT
        page = (
            '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"{head.strip()}\n</head>\n<body>\n{body.strip()}\n</body>\n</html>\n"
        )
    out.write_text(page, encoding="utf-8")
    print(f"written: {out}")
    for warning in blocks.warnings + frame.warnings:
        print("Warning:", warning)


if __name__ == "__main__":
    main()

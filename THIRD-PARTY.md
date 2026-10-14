# Third-party material

The DAB+ radio is licensed under the GNU General Public License, version 3 or (at your
option) any later version; the full text is in [LICENSE](LICENSE). This file lists what
it contains or uses that comes from others, and under which terms.

## Contained in this repository

| Where | What | Terms |
|---|---|---|
| `src/ui/assets/fonts/` | Source Sans 3 (3.052R, Regular and Bold) by Adobe, unmodified | SIL Open Font License 1.1, see `LICENSE.md` beside the fonts; the fonts are not covered by the GPL |
| `src/ui/assets/earth.npz` | Coastlines and borders from Natural Earth, built by `tools/build_earth.py` | Public domain |
| `src/dab/gmf.py` | The 440 coefficients and the constants of the Global Mapping Function (Böhm, Niell, Tregoning, Schuh 2006), taken from the author's routine `gmf.f` (Johannes Böhm, TU Wien, revision of 2011-07-21, vmf.geo.tuwien.ac.at/codes/gmf.f); the code is written from the papers | The routine carries no licence text; its author, Johannes Böhm, permitted their use in this project and their publication under GPL-3.0-or-later (e-mail, October 2026) |
| `src/dsp/uep.py` | Puncturing table of unequal error protection, taken from welle.io (`uep-protection.cpp`) and corrected in one entry | welle.io: GPL-2.0-or-later |
| `docs/screenshots/` | Screenshots of the radio as it received channel 5C on 2026-10-11; they show the logos, slides, programme data and Journaline pages the stations broadcast, which belong to the broadcasters and appear only to illustrate the program | Not covered by the GPL |
| `native/rtlsdr-raw-io.patch`, `native/rtlsdr-r82xx-if-agc.patch` | Patches to rtl-sdr 2.0.3 (osmocom); the second carries register values of librtlsdr/librtlsdr 0.8.0 | rtl-sdr and librtlsdr: GPL-2.0-or-later |

## Data used but not contained

- **Road network:** `tools/build_road_network.py` builds it from OpenStreetMap
  data. © OpenStreetMap contributors, Open Database License 1.0. Every map that shows it
  credits the source. The built file stays on the machine that built it.
- **TMC location tables:** the location code lists (LCL) in the exchange format of
  ISO 14819-3, which the responsible authority of each country publishes under its own
  terms (for Germany the Bundesanstalt für Straßenwesen, BASt, whose terms do not allow
  passing the table on). Each user obtains the table from that authority and imports it
  with `tools/import_tmc_locations.py`. No table goes into this repository, neither as
  downloaded nor as imported; the tests use a made-up table (`tests/tmc_table.py`).

## Libraries used at run time, not contained

Python packages, installed by `uv sync`:

| Package | License |
|---|---|
| PyQt6 | GPL-3.0-only |
| PyQt6-Qt6 (Qt 6) | LGPL-3.0 |
| PyQt6-sip | BSD-2-Clause |
| pyrtlsdr | GPL-3.0-or-later |
| pyrtlsdrlib | MIT (bundles builds of librtlsdr, GPL-2.0-or-later) |
| NumPy | BSD-3-Clause |
| numba, llvmlite | BSD-2-Clause; llvmlite also Apache-2.0 WITH LLVM-exception |
| pyqtgraph | MIT |
| paho-mqtt | EPL-2.0 OR BSD-3-Clause (used under BSD-3-Clause, the Eclipse Distribution License 1.0) |
| segno | BSD-3-Clause |

System libraries, loaded through ctypes:

| Library | Used for | License |
|---|---|---|
| FAAD2 (`libfaad2`, `faad.dll`) | AAC decoding of DAB+ | GPL-2.0-or-later |
| mpg123 (`libmpg123-0`) | MPEG Audio Layer II decoding of DAB | LGPL-2.1 |
| rtl-sdr 2.0.3 with libusb 1.0.30 (`native/rtlsdr.dll`, built by `native/build_rtlsdr.ps1`) | RTL-SDR stick on Windows | GPL-2.0-or-later; libusb LGPL-2.1-or-later |
| SDRplay API 3.15 | SDRplay RSPduo, optional | Proprietary (SDRplay Ltd.); installed by the user from sdrplay.com |

A binary distribution that bundles any of these libraries has to meet their licenses as
well: for the GPL and LGPL ones, their source code goes with it, including the patches
in `native/`. The SDRplay API is not to be bundled.

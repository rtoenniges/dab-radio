# Changelog

All notable changes to the radio. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versions follow
[Semantic Versioning](https://semver.org/). Each version is also published as a
[release on GitHub](https://github.com/rtoenniges/dab-radio/releases).

## [Unreleased]

## [1.3.1] - 2026-10-16

### Fixed

- Touch UI, stations page: during the first scan, with no stations known yet, the
  filters slid down onto the spinner and the scan button moved up. The layout now stays
  in place.

## [1.3.0] - 2026-10-16

### Changed

- Touch UI, settings page: the eight categories stand in a list on the left, each with
  its current state beneath its name (device and server, the output playing, network
  and band, headphones and their battery, recording folder and free space, whether the
  recorder runs, the MQTT connection). The settings of the chosen category stand on
  cards on the right, grouped under headings, with an explanation under each setting.
- A setting with several choices shows the one in effect; a tap opens a sheet with all
  of them. Switches are toggles.
- The recording delays are applied to all stations with one row, “Apply to all
  stations”, for both delays at once.
- “Back” on the equalizer and on the recordings returns to the settings when they were
  opened from there.

### Fixed

- `install.sh` on a fresh Raspberry Pi OS Lite: `--splash` failed while rebuilding the
  initramfs, because Plymouth’s hook needs fontconfig, which the package does not
  depend on; and Qt’s media backend was missing `libXrandr`. Both are installed now.

## [1.2.0] - 2026-10-15

### Added

- A fourth device, **RTL_TCP**: an RTL-SDR stick on another computer, served by
  `rtl_tcp` from rtl-sdr (rsp_tcp and the Android app “RTL-SDR driver” speak the same
  protocol). Host and port are set on both surfaces; gain, AGC and ppm correction work
  as with a local stick, and the channel scan works the same way.
- For RTL_TCP the radio asks the system for a large receive buffer, so that the audio
  keeps playing while the computer is busy for a moment. On Linux this needs
  `net.core.rmem_max` to allow 16 MiB (see Troubleshooting in the README).

## [1.1.1] - 2026-10-15

### Fixed

- Loud noise after the audio buffer had to drop a backlog, while the displays and the
  decoded stream stayed clean. It lasted until the next drop, and was heard with the
  RTL-SDR stick and often with an rtl_tcp server over Wi-Fi. The buffer now drops whole
  audio frames only.

## [1.1.0] - 2026-10-15

### Added

- `native/build_codecs.ps1` builds the audio decoders for Windows, `faad.dll`
  (FAAD2 2.11.4) and `mpg123.dll` (libmpg123 1.33.7), from their original sources, so
  that they no longer have to be found elsewhere. The radio finds them in `native/`.

## [1.0.0] - 2026-10-14

The first public release.

- Reception in Band III with an RTL-SDR stick or one tuner of an SDRplay RSPduo, and a
  scan that keeps the station lists of all channels.
- The whole decoding chain in Python with NumPy and numba, every step shown: spectrum,
  null symbol search, constellation, the FFT of each symbol, the error correction,
  transmitter identification and SFN share.
- DAB+ (AAC) and DAB (MPEG Audio Layer II), with an equalizer for all stations or per
  station.
- Data services: MOT slideshow, Dynamic Label with DL Plus, programme guide with station
  logos, Journaline, announcements (also from other ensembles), ARD TPEG traffic
  messages on a list and a map, and the GNSS corrections of GEPOS.
- A Touch UI for the Raspberry Pi with the Touch Display 2, set up by `pi/install.sh`:
  Wi-Fi and Bluetooth headphones on the screen, recording of the last titles to a USB
  drive with a lossless cutter, a dark and a light theme, a boot splash.
- Remote control over MQTT with Home Assistant discovery and notifications with buttons.
- A desktop window for signal analysis on the PC.
- English and German.

[Unreleased]: https://github.com/rtoenniges/dab-radio/compare/v1.3.1...HEAD
[1.3.1]: https://github.com/rtoenniges/dab-radio/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/rtoenniges/dab-radio/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/rtoenniges/dab-radio/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/rtoenniges/dab-radio/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/rtoenniges/dab-radio/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/rtoenniges/dab-radio/releases/tag/v1.0.0

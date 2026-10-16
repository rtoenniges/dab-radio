# Changelog

All notable changes to the radio. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versions follow
[Semantic Versioning](https://semver.org/). Each version is also published as a
[release on GitHub](https://github.com/rtoenniges/dab-radio/releases).

## [Unreleased]

### Changed

- Devices: The RSPduo is one device; its tuner is a setting of its own
- MQTT: The device names the tuner of the RSPduo, e.g. "RSPduo(2)"

### Fixed

- MQTT: The device changed only with the next start of reception

## [1.3.1] - 2026-10-16

### Fixed

- Touch UI: Stations page layout jumped during the first scan

## [1.3.0] - 2026-10-16

### Changed

- Touch UI: Settings as a list of categories, each with its current state
- Touch UI: Settings grouped on cards, with a short explanation under each
- Touch UI: Choices open in a sheet; switches are toggles
- Touch UI: One row applies both recording delays to all stations
- Touch UI: "Back" from the equalizer and recordings returns to the settings

### Fixed

- install.sh: Boot splash failed on a fresh Raspberry Pi OS Lite
- install.sh: Qt media backend was missing libxrandr2

## [1.2.0] - 2026-10-15

### Added

- Devices: New device RTL_TCP, an RTL-SDR stick on another computer served by rtl_tcp
- RTL_TCP: Host and port on both surfaces; gain, AGC and ppm as with a local stick
- RTL_TCP: Large receive buffer, so the audio keeps playing through short load peaks

## [1.1.1] - 2026-10-15

### Fixed

- Audio: Loud noise after the audio buffer dropped a backlog

## [1.1.0] - 2026-10-15

### Added

- Windows: `native/build_codecs.ps1` builds the audio decoders (FAAD2, mpg123) from source

## [1.0.0] - 2026-10-14

The first public release.

### Added

- Reception: RTL-SDR stick and SDRplay RSPduo, scan of all Band III channels
- Signal: The whole decoding chain in Python, every step shown
- Audio: DAB+ and DAB, five-band equalizer for all stations or per station
- Data services: Slideshow, Dynamic Label with DL Plus, programme guide, Journaline
- Data services: Announcements, also from other ensembles
- Data services: ARD TPEG traffic messages and GEPOS GNSS corrections
- Touch UI: Raspberry Pi with the Touch Display 2, set up by `pi/install.sh`
- Touch UI: Wi-Fi and Bluetooth headphones set up on the screen
- Touch UI: Recording of the last titles to a USB drive, with a lossless cutter
- Touch UI: Dark and light theme, boot splash
- MQTT: Remote control with Home Assistant discovery and notifications
- Desktop: Window for signal analysis
- Languages: English and German

[Unreleased]: https://github.com/rtoenniges/dab-radio/compare/v1.3.1...HEAD
[1.3.1]: https://github.com/rtoenniges/dab-radio/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/rtoenniges/dab-radio/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/rtoenniges/dab-radio/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/rtoenniges/dab-radio/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/rtoenniges/dab-radio/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/rtoenniges/dab-radio/releases/tag/v1.0.0

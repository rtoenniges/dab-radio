"""The surface for the Raspberry Pi Touch Display 2 (7 inch), used in landscape.

Its panel is natively portrait (720 × 1280); the operating system rotates picture and
touch input, so this surface only ever sees a screen of 1280 × 720. It is built on the
same `ui.receiver_controller.ReceiverController` as `ui.main_window.MainWindow` and
holds no receiver logic of its own. Start it with `uv run python src/main.py --touch`.

`window.TouchWindow` holds a navigation rail (the buttons of the pages, the load, the
symbols of WLAN and Bluetooth while they are on, clock and date) and five pages: `radio_page` (what is
playing; a long press on the slideshow leads to `display_choice`, what to show in its
place, a tap to `slide_gallery`, the slides received from the station), `stations_page` (every known station as a tile, and the scan),
`services_page` (sub-menus: the data services of the ensemble in `data_services_view`,
the programme guide in `guide_view` on `ui.guide_model`, GNSS in `gnss_view` on
`ui.gnss_model` and Traffic in `traffic_view` on `ui.traffic_model`, their pages in a
`pager` to swipe through, the map zoomed with two fingers by `gestures`), `signal_page` (spectrum,
constellation, impulse response, figures; the panels the radio page can show as well in
`signal_panels`) and `settings_page` (with the listener's
position, entered on the `number_pad`, and the times after which `backlight` dims the
screen and switches it off). The station catalogue behind the station page
and ◀ ▶ is `stations.StationDirectory`; `quality` judges the reception for the bars,
`theme` holds the dark and the light scheme and the typeface, `icons` draws the symbols and `widgets`
the controls made for a finger.
"""

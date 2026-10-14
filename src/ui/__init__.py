"""The PyQt6 user interface.

`ui.receiver_controller.ReceiverController` runs the receiver without a single widget:
device, tuning, scan, service selection, audio, the worker threads and the stored
settings. A surface builds its widgets on it, calls its methods and follows its signals;
`ui.main_window.MainWindow` is the one for a desktop screen. Each tab of the plot area is
a widget of its own (`signal_tab`, `ofdm_tab`, `service_tab`, `programme_tab`, ...) that
only receives data through plain method calls. Shared plot helpers live in `ui.plots`.

All text the user sees is marked for translation (`i18n`): English in the code, other
languages from a catalog. Comments and identifiers are English.
"""

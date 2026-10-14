#!/bin/sh
# Starts the DAB+ radio on the Raspberry Pi Touch Display 2 in the kiosk compositor
# cage (Raspberry Pi OS Lite, from the console): sh pi/start-touch.sh
# Further arguments go to src/main.py.
#
# `video=...,rotate=` in cmdline.txt only turns the console, cage sets the mode itself;
# so wlr-randr (package wlr-randr) turns the output inside cage before the radio starts,
# and the radio sees 1280 x 720. The output is the DSI connector with a display
# connected (DSI-1 or DSI-2, after the socket; pi/touch-display.sh), looked up once cage
# has opened the displays. DAB_RADIO_OUTPUT names another (a name wlr-randr lists, such
# as HDMI-A-1); for the unit through `sudo systemctl edit dab-radio`, which pi/install.sh
# leaves alone. Without either the output stays as it is.
#
# The pointer is put into the bottom right corner, where its arrow lies all but off
# the screen: cage shows it until the radio hides it, and again for a moment whenever
# an input device comes or goes (ui.touch.window.rehide_cursor). A Wayland client
# cannot move the pointer itself (QCursor.setPos does nothing there); wlrctl (package
# wlrctl) does it through cage's virtual pointer, and is skipped where it is missing.
#
# The radio is the one program playing here, so it sets the volume of the sound system
# (--system-volume): a loudspeaker stands at 100 % and the radio attenuates, a Bluetooth
# device with a volume of its own takes the level. Started any other way, the radio
# leaves the system's volume alone.
#
# The radio runs from the Python environment of the checkout (uv sync, or pi/install.sh,
# makes it), not through `uv run`, which checks the environment at every start and
# would write to it: in /opt/dab-radio it belongs to root.
cd "$(dirname "$0")/.." || exit 1
if [ ! -x .venv/bin/python ]; then
    echo "start-touch: no Python environment in $(pwd)/.venv; uv sync or pi/install.sh makes it" >&2
    exit 1
fi
exec /usr/bin/cage -s -- sh -c '
    . pi/touch-display.sh
    output=${DAB_RADIO_OUTPUT:-$(dsi_output)}
    if [ -n "$output" ]; then
        wlr-randr --output "$output" --transform "$DISPLAY_TRANSFORM" || exit 1
    else
        echo "start-touch: no DSI display connected; the output is not turned" >&2
    fi
    if command -v wlrctl >/dev/null 2>&1; then wlrctl pointer move 5000 5000; fi
    exec .venv/bin/python src/main.py --touch --fullscreen --system-volume "$@"
' sh "$@"

# The Raspberry Pi Touch Display 2 as pi/start-touch.sh and pi/install.sh see it;
# sourced by both (. pi/touch-display.sh), never run on its own.
#
# The panel is built upright (720 x 1280) and used in landscape: wlr-randr turns the
# output by DISPLAY_TRANSFORM inside cage (pi/start-touch.sh), the touch input turns
# with it through pi/99-dab-radio-touch-rotation.rules (its matrix belongs to this
# value), and pi/install.sh --splash writes the same rotation into cmdline.txt for
# Plymouth and the console.
#
# The display may sit on either DSI socket of the Pi 5, CAM/DISP 0 or CAM/DISP 1; the
# firmware finds it on both without a line in config.txt (display_auto_detect=1, as
# Raspberry Pi OS ships it). The kernel names the connector after the socket: DSI-1 on
# CAM/DISP 0, DSI-2 on CAM/DISP 1 (RP1's dsi1). So the connector is looked up, never
# written down.
DISPLAY_TRANSFORM=270
# The one mode of the panel, as /sys/class/drm/<connector>/modes lists it, and the
# refresh rate that goes with it in the kernel's video=.
DISPLAY_MODE=720x1280
DISPLAY_REFRESH=60

# The name of the first DSI connector with a display connected, as wlr-randr and the
# kernel's video= take it (DSI-1, DSI-2); nothing where there is none.
dsi_output() {
    for dsi_dir in /sys/class/drm/card*-DSI-*; do
        if [ -f "$dsi_dir/status" ] && [ "$(cat "$dsi_dir/status")" = connected ]; then
            dsi_dir=${dsi_dir##*/}
            echo "${dsi_dir#card*-}"
            return 0
        fi
    done
    return 0
}

#!/bin/sh
# Brings the touch of the Raspberry Pi Touch Display 2 back when the kernel found no
# touchscreen at boot. Its controller (Goodix, address 0x5d on the I2C bus of the DSI
# connector) now and then does not answer when the driver probes it (dmesg:
# "Goodix-TS 11-005d: I2C communication failure: -121", "probe with driver Goodix-TS
# failed"), mostly after a warm reboot that left it as it was; then there is no input
# device at all, whatever cage, the udev rules or the radio do. Binding the driver
# again brings it back without a reboot.
#
# Runs as root: by hand `sudo sh pi/rebind-touch.sh`, or from pi/dab-radio.service as
# ExecStartPre before cage starts (cage takes the touchscreen by hotplug all the same,
# but a radio that starts with it is better). Nothing to do where a touchscreen exists;
# never fails, so that the radio starts without touch rather than not at all.
#   sh pi/rebind-touch.sh [tries]   (default 5, a second apart)
DRIVER=/sys/bus/i2c/drivers/Goodix-TS
TRIES=${1:-5}

has_touchscreen() {
    # What libinput would take: an input device udev tagged as a touchscreen.
    for device in /sys/class/input/event*; do
        [ -e "$device" ] || continue
        if udevadm info --query=property --path="$device" 2>/dev/null \
            | grep -q '^ID_INPUT_TOUCHSCREEN=1$'; then
            return 0
        fi
    done
    return 1
}

# The Goodix devices of the device tree (modalias "of:N...Cgoodix,gt...") that have no
# driver bound: those whose probe failed.
unbound_goodix() {
    for device in /sys/bus/i2c/devices/*; do
        [ -f "$device/modalias" ] || continue
        grep -qi goodix "$device/modalias" 2>/dev/null || continue
        [ -e "$device/driver" ] && continue
        basename "$device"
    done
}

if has_touchscreen; then
    exit 0
fi
if [ ! -d "$DRIVER" ]; then
    echo "rebind-touch: no touchscreen and no Goodix-TS driver; nothing to bind"
    exit 0
fi

try=1
while [ "$try" -le "$TRIES" ]; do
    for id in $(unbound_goodix); do
        echo "rebind-touch: no touchscreen; binding $id to Goodix-TS (try $try of $TRIES)"
        echo "$id" > "$DRIVER/bind" 2>/dev/null
    done
    # udev tags the new input device a moment after the driver registered it.
    sleep 1
    if has_touchscreen; then
        echo "rebind-touch: touchscreen is back"
        exit 0
    fi
    try=$((try + 1))
done
echo "rebind-touch: still no touchscreen after $TRIES tries; see dmesg | grep -i goodix"
exit 0

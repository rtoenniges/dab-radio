#!/bin/sh
# Sets up a Raspberry Pi (Raspberry Pi OS Lite, 64 bit) for the DAB+ radio on the Touch
# Display 2, in one go, and brings it up to date: every step looks first and changes
# only what differs, so running it twice is safe.
#
#   curl -fsSL <URL of pi/install.sh> | sudo sh [-s -- OPTIONS]   (from the web)
#   sudo sh pi/install.sh [OPTIONS]                               (from a checkout)
#   sudo sh /opt/dab-radio/pi/install.sh --update [OPTIONS]       (the newest version)
#   sh pi/install.sh --dry-run [OPTIONS]                          (only look)
#
# Options: --dry-run, --update, --from URL, --user NAME, --no-autostart, --splash,
# --no-splash, --verbose (see below).
#
# The radio is installed in /opt/dab-radio, a git checkout that belongs to root, and
# runs as an account of its own, dabradio, with its home in /var/lib/dab-radio: an
# ordinary account without password or login shell (not a system account: its login
# session on tty1 starts PipeWire as any user's does). --user NAME runs the radio as an
# existing account instead. Where the checkout in /opt/dab-radio comes from:
#
#   - Piped from the web: the script clones the repository (--from URL, or the one it
#     is published in) into /opt/dab-radio, or brings the checkout there to the newest
#     commit of the repository's default branch, and goes on from there.
#   - From a checkout elsewhere, such as one to develop in: the commit that checkout is
#     on goes to /opt/dab-radio (changes not committed do not), and the script goes on
#     from there. The checkout stays as it is.
#   - From /opt/dab-radio itself: what is there; with --update first the newest commit
#     of its branch, from the repository it came from (--from URL: from another one,
#     which it then follows).
#
# What it does then, in this order:
#
#   1. apt packages: the kiosk compositor cage with wlr-randr and wlrctl, PipeWire
#      with its Pulse interface, Bluetooth plugin and the Pulse tools (pactl, through
#      which the radio sets the volume of the sink), BlueZ, NetworkManager, UDisks2
#      with ntfs-3g, the audio decoders libfaad2 and libmpg123, the libraries the Qt
#      of the PyQt6 wheel loads from the system (EGL, GL, xkbcommon, fontconfig, D-Bus,
#      Wayland, libpulse, XRandR), and the udev rule of the rtl-sdr package for the
#      stick; with the boot splash Plymouth and fontconfig.
#   2. the account: dabradio where it is missing, in the groups video (backlight),
#      render (GPU), plugdev (USB stick and RTL-SDR), netdev (WLAN), bluetooth and
#      audio. The data of the account the radio ran as before (the user of the unit in
#      place, or else the one who ran sudo) is copied over once, as far as the new
#      account has none: the data directory (settings, station lists, programme guides,
#      the data files of the traffic map) and the state of WirePlumber (volumes and
#      profiles of the outputs). In the settings the folder of the recordings moves from
#      /media/<former account>/ to /media/<account>/, where UDisks2 mounts the USB stick
#      for the account that mounts it. The radio stops for the copy; the old data stays
#      where it is.
#   3. the udev rules (touch rotation, backlight) and polkit rules (NetworkManager,
#      UDisks2) from pi/, and a modprobe file that keeps the kernel's DVB driver off
#      the RTL-SDR stick, unless one is there already.
#   4. Bluetooth: Experimental = true in /etc/bluetooth/main.conf, for the battery of
#      headphones, and WirePlumber's Bluetooth side only in the radio's account
#      (pi/50-dab-radio-no-bluetooth.conf for every account, pi/50-dab-radio-bluetooth.conf
#      in the radio's home, which turns it on again there, without seat monitoring, so
#      that connected headphones keep their sound when the unit restarts).
#   5. uv in /usr/local/bin and `uv sync` in /opt/dab-radio, without the development
#      tools (Python 3.14 into /opt/dab-radio/.python, the packages into .venv; needs
#      the network), the radio's own modules compiled to bytecode, since the radio may
#      not write there, then a check that the Qt modules the radio imports and the
#      plugins it loads under cage find their system libraries.
#   6. the systemd unit pi/dab-radio.service with the account, the path and the UID of
#      this Pi, enabled for the next boot (not started now: it takes tty1, where this
#      may run).
#   7. only with --splash: the boot splash, Plymouth with the theme pi/plymouth/dab-radio
#      (the constellation locking in, as the radio shows it while it starts), the
#      initramfs of every kernel rebuilt with it, and in cmdline.txt `quiet splash
#      plymouth.ignore-serial-consoles` and the rotation of the display as its panel
#      orientation, on the DSI connector it is on (pi/touch-display.sh): on the Pi here
#      video=DSI-2:720x1280@60,panel_orientation=right_side_up, in place of any video=
#      word of a DSI connector (the console turns by rotate= or by this, Plymouth only
#      by this). It needs the display connected. A copy of cmdline.txt as it was stays
#      beside it (cmdline.txt.dab-radio.bak).
#      --no-splash takes quiet, splash and plymouth.ignore-serial-consoles out again and
#      leaves video= (the console stays turned); Plymouth stays installed and then shows
#      the console as before.
#      With neither option, a splash that is on (Plymouth's theme is dab-radio and
#      cmdline.txt says splash) stays on: its theme is brought up to date, as an update
#      may change the pictures, and cmdline.txt stays as it is.
#
# The console shows a line per step, what it did or found, and at the end what is left
# to do; a step that takes long shows that it runs. Everything else goes into a log:
# what each step looked at, the output of apt, git and uv, the diffs of a dry run. The
# log is /var/log/dab-radio-install.log for root (root:adm, 640), and
# ~/.local/state/dab-radio-install.log for a dry run without root; each run is added
# with its time. Should a step fail, the console shows the end of the log. --verbose
# shows all of it on the console instead, and writes no log. Colours only on a terminal
# and without NO_COLOR; on the Linux console, whose font lacks the marks, plain ASCII.
#
# --dry-run changes nothing and says what a run would do: the commit it would install,
# the packages apt would install (apt's own simulation, against the package lists as
# they are), the account and groups, the data it would copy, the files that differ (as
# a diff, in the log), what `uv sync` would change, and whether the unit is in place
# and enabled. It needs no root: sh pi/install.sh --dry-run from a checkout, or piped,
# curl ... | sh -s -- --dry-run, which clones into a temporary directory to look from
# there. Without root it cannot look into /etc/polkit-1/rules.d (root:polkitd, 750) or
# the radio's home (700) and says so rather than calling files missing; sudo with
# --dry-run looks there too and changes nothing either.
#
# Not touched: /boot/firmware/config.txt (the sound HAT; the display needs nothing
# there, the firmware finds it on either DSI socket), cmdline.txt but with --splash or
# --no-splash, the SDRplay API (proprietary installer from sdrplay.com, only for the
# RSPduo), libinput-tools for checking touch (`libinput debug-events`, see README.md)
# and the data files the traffic pages want (road_network.npz, tmc_locations.npz,
# built on the PC and copied into the radio's data directory); the end of the run says
# which of them are missing.
#
# The end of the run also says what is left to make it all take effect. A reboot, when
# the run changed something that takes effect only then: packages apt installed or
# upgraded, the account of the radio, groups, udev rules, the DVB blacklist, WirePlumber,
# a unit enabled anew, the boot splash. Otherwise a restart of the radio is enough: the
# checkout in /opt/dab-radio (code, scripts in pi/, the Python packages of uv sync) and
# a changed unit take effect then, and polkit and BlueZ (restarted here) need nothing
# more. A run that changed nothing says "Nothing to do". A dry run says which of these
# a run would end with.
set -eu
# usermod, useradd, update-initramfs and plymouth-set-default-theme live in /usr/sbin,
# which a user's PATH lacks: a dry run without root found no Plymouth theme and would
# have rebuilt the initramfs.
PATH=$PATH:/usr/sbin:/sbin
# What lands in /opt/dab-radio has to be readable by the radio's account, whatever
# umask root has.
umask 022
# Nothing may wait for an answer: the question would be in the log, unseen. git asks
# for no password, and apt keeps changed configuration files (see step_packages).
export GIT_TERMINAL_PROMPT=0

# Where the radio is installed, the account it runs as and that account's home.
INSTALL_DIR=/opt/dab-radio
RADIO_USER=dabradio
RADIO_HOME=/var/lib/dab-radio
# The repository a piped run clones without --from.
SOURCE_URL=https://github.com/rtoenniges/dab-radio.git
UNIT=/etc/systemd/system/dab-radio.service
UV=/usr/local/bin/uv

AUTOSTART=1
DRY_RUN=0
SPLASH=
UPDATE=0
VERBOSE=0
USER_NAME=
FROM_URL=
# The arguments stay in "$@", for the run in /opt/dab-radio this one may hand over to.
expect=
for arg in "$@"; do
    if [ -n "$expect" ]; then
        case $expect in
            user) USER_NAME=$arg ;;
            from) FROM_URL=$arg ;;
        esac
        expect=
        continue
    fi
    case $arg in
        --dry-run) DRY_RUN=1 ;;
        --no-autostart) AUTOSTART=0 ;;
        --splash) SPLASH=on ;;
        --no-splash) SPLASH=off ;;
        --update) UPDATE=1 ;;
        -v|--verbose) VERBOSE=1 ;;
        --user) expect=user ;;
        --from) expect=from ;;
        -h|--help)
            if [ -f "$0" ]; then
                sed -n '2,/^set -eu/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
            else
                echo "install: the options are described at the head of pi/install.sh"
            fi
            exit 0
            ;;
        *) echo "install: unknown argument $arg" >&2; exit 2 ;;
    esac
done
if [ -n "$expect" ]; then
    echo "install: --$expect needs a value" >&2
    exit 2
fi

# The checkout this script runs from; none when it is piped into sh, where $0 is the
# shell.
REPO=
if [ -f "$0" ] && [ -f "$(dirname "$0")/../src/main.py" ]; then
    REPO=$(cd "$(dirname "$0")/.." && pwd -P)
fi
INSTALLED=$(cd "$INSTALL_DIR" 2>/dev/null && pwd -P || echo "$INSTALL_DIR")

# --- output --------------------------------------------------------------------------
# The console (fd 3) gets a line per step and, at the end, what is left to do; the log
# (fd 1 and 2) gets everything else: what each step looked at (say), the output of apt,
# git and uv, the diffs of a dry run. With --verbose fd 1 and 2 stay on the console and
# there is no log (LOG=none). A run this one hands over to finds DAB_RADIO_LOG in its
# environment and goes on with the same console and log.
OUTPUT=0
REPORTED=0
SHOW_LOG_END=1
STEP=
STEP_OPEN=0
STEP_NOTES=
STEP_START=0
SPIN_PID=
SPIN_TEXT=
TODO=
changes=0
unseen=0
# Whether the run changed anything (a dry run: would), the checkout of the run that
# handed over to this one included; without, there is nothing to restart.
ANY_CHANGE=${DAB_RADIO_CHANGED:-0}

say() { printf '%s\n' "$*"; }
console() { printf '%s\n' "$*" >&3; }

# The marks and colours for the console: colours on a terminal without NO_COLOR, the
# marks in UTF-8 but on the Linux console, whose font lacks them.
setup_marks() {
    TTY=0
    if [ -t 3 ] && [ "${TERM:-dumb}" != dumb ]; then
        TTY=1
    fi
    esc=$(printf '\033')
    GREEN='' YELLOW='' RED='' DIM='' BOLD='' RESET='' CLEAR_EOL=''
    if [ "$TTY" -eq 1 ]; then
        CLEAR_EOL="${esc}[K"
        if [ -z "${NO_COLOR:-}" ]; then
            GREEN="${esc}[32m" YELLOW="${esc}[33m" RED="${esc}[31m"
            DIM="${esc}[2m" BOLD="${esc}[1m" RESET="${esc}[0m"
        fi
    fi
    case ${LC_ALL:-${LC_CTYPE:-${LANG:-}}} in
        *[Uu][Tt][Ff]-8*|*[Uu][Tt][Ff]8*) utf8=1 ;;
        *) utf8=0 ;;
    esac
    [ "${TERM:-}" != linux ] || utf8=0
    if [ "$utf8" -eq 1 ]; then
        MARK_OK=✓ MARK_WOULD=• MARK_WARN=! MARK_FAIL=✗ MARK_WAIT=· BULLET=• DOTS=… SEP=·
        SPIN_FRAMES='⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏'
    else
        MARK_OK=+ MARK_WOULD=- MARK_WARN=! MARK_FAIL=x MARK_WAIT=. BULLET=- DOTS=... SEP=-
        SPIN_FRAMES='| / - \'
    fi
}

# Where the log goes: root's in /var/log (root:adm, 640, as the system's logs), a dry
# run's without root in the user's state directory, else a temporary file. It keeps
# the last 5000 lines of the runs before; LOG_START is the line before this run's.
open_log() {
    if [ "$(id -u)" -eq 0 ]; then
        LOG=/var/log/dab-radio-install.log
    else
        LOG=${XDG_STATE_HOME:-$HOME/.local/state}/dab-radio-install.log
    fi
    if ! { mkdir -p "${LOG%/*}" && touch "$LOG"; } 2>/dev/null; then
        LOG=$(mktemp)
    fi
    if [ "$(id -u)" -eq 0 ] && [ "$LOG" = /var/log/dab-radio-install.log ]; then
        chgrp adm "$LOG" 2>/dev/null || true
        chmod 640 "$LOG"
    fi
    if [ "$(wc -c < "$LOG")" -gt 1048576 ]; then
        kept=$(mktemp)
        tail -n 5000 "$LOG" > "$kept"
        cat "$kept" > "$LOG"
        rm -f "$kept"
    fi
    LOG_START=$(wc -l < "$LOG")
}

setup_output() {
    if [ -n "${DAB_RADIO_LOG:-}" ]; then
        LOG=$DAB_RADIO_LOG
        LOG_START=${DAB_RADIO_LOG_START:-0}
        [ "$LOG" != none ] || VERBOSE=1
        setup_marks
        OUTPUT=1
        trap 'on_exit $?' EXIT
        trap 'exit 130' INT TERM
        echo
        echo "----- $(date '+%F %T') going on: $0 $*"
        return
    fi
    exec 3>&1
    LOG_START=0
    if [ "$VERBOSE" -eq 1 ]; then
        LOG=none
    else
        open_log
        exec >>"$LOG" 2>&1
        echo
        echo "===== $(date '+%F %T') $0 $* (uid $(id -u), from ${REPO:-a pipe})"
    fi
    DAB_RADIO_LOG=$LOG
    DAB_RADIO_LOG_START=$LOG_START
    export DAB_RADIO_LOG DAB_RADIO_LOG_START
    setup_marks
    OUTPUT=1
    trap 'on_exit $?' EXIT
    trap 'exit 130' INT TERM
    title="${BOLD}DAB+ radio install${RESET}"
    if [ "$DRY_RUN" -eq 1 ]; then
        title="$title $SEP dry run, nothing is changed"
    fi
    console "$title"
    [ "$LOG" = none ] || console "${DIM}Log: $LOG${RESET}"
    console ""
}

# A step: its line shows that it runs (on a terminal) and then what it did; notes
# stand below it, and the log gets the same between the step's own lines.
step_begin() { # step_begin NAME
    STEP=$1
    STEP_NOTES=
    STEP_START=$(date +%s)
    changes=0
    unseen=0
    echo
    echo "== $STEP"
    if [ "$TTY" -eq 1 ] && [ "$VERBOSE" -eq 0 ]; then
        printf '  %s %-12s %s' "$DIM$MARK_WAIT$RESET" "$STEP" "$DIM$DOTS$RESET" >&3
        STEP_OPEN=1
    fi
}

step_end() { # step_end ok|would|warn|fail TEXT
    case $1 in
        ok) mark=$GREEN$MARK_OK$RESET ;;
        would) mark=$BOLD$MARK_WOULD$RESET ;;
        warn) mark=$YELLOW$MARK_WARN$RESET ;;
        *) mark=$RED$MARK_FAIL$RESET ;;
    esac
    took=$(( $(date +%s) - STEP_START ))
    after=
    if [ "$took" -ge 10 ]; then
        after=" $DIM($(duration "$took"))$RESET"
    fi
    if [ "$STEP_OPEN" -eq 1 ]; then
        printf '\r%s' "$CLEAR_EOL" >&3
    fi
    printf '  %s %-12s %s%s\n' "$mark" "$STEP" "$2" "$after" >&3
    # With --verbose the console has the line already.
    [ "$LOG" = none ] || echo "-> $2"
    if [ -n "$STEP_NOTES" ]; then
        printf '%s\n' "$STEP_NOTES" >&3
    fi
    [ "$changes" -eq 0 ] || ANY_CHANGE=1
    STEP_OPEN=0
    STEP=
}

# A line below the step's own; WARN in yellow, else dim.
note() { # note TEXT [warn]
    echo "note: $1"
    if [ "${2:-}" = warn ]; then
        line="                 $YELLOW$1$RESET"
    else
        line="                 $DIM$1$RESET"
    fi
    STEP_NOTES="${STEP_NOTES:+$STEP_NOTES
}$line"
}

# Something to do by hand, for the end of the run.
todo() { # todo TEXT
    TODO="${TODO:+$TODO
}$1"
}

duration() { # duration SECONDS
    if [ "$1" -lt 60 ]; then
        echo "$1 s"
    else
        echo "$(($1 / 60)) min $(($1 % 60)) s"
    fi
}

plural() { # plural COUNT WORD
    if [ "$1" -eq 1 ]; then
        echo "1 $2"
    else
        echo "$1 ${2}s"
    fi
}

spinner() {
    while :; do
        for frame in $SPIN_FRAMES; do
            printf '\r  %s %-12s %s%s' "$frame" "$STEP" "$DIM$SPIN_TEXT$RESET" "$CLEAR_EOL" >&3
            sleep 0.1
        done
    done
}

stop_spinner() {
    [ -n "$SPIN_PID" ] || return 0
    kill "$SPIN_PID" 2>/dev/null || true
    wait "$SPIN_PID" 2>/dev/null || true
    SPIN_PID=
    printf '\r  %s %-12s %s%s' "$DIM$MARK_WAIT$RESET" "$STEP" "$DIM$DOTS$RESET" "$CLEAR_EOL" >&3
}

# Runs a command that takes a while, with TEXT and a spinner on the step's line
# meanwhile, and nothing on its standard input. Only for programs and plain shell
# functions: inside it, set -e does not stop a function at its first failure.
working() { # working TEXT COMMAND...
    SPIN_TEXT=$1
    shift
    printf '$ %s\n' "$*"
    if [ "$STEP_OPEN" -eq 1 ]; then
        spinner &
        SPIN_PID=$!
    fi
    working_status=0
    "$@" </dev/null || working_status=$?
    stop_spinner
    return "$working_status"
}

# The step that is open ends with TEXT as its failure, and below it the end of the log.
fail() { # fail TEXT
    REPORTED=1
    if [ "$OUTPUT" -eq 0 ]; then
        echo "install: $1" >&2
        return
    fi
    stop_spinner
    echo "error: $1"
    if [ -n "$STEP" ]; then
        step_end fail "$1"
    else
        console "  $RED$MARK_FAIL$RESET $1"
    fi
    if [ "$SHOW_LOG_END" -eq 1 ] && [ "$LOG" != none ]; then
        console ""
        console "${DIM}The end of the log:$RESET"
        tail -n "+$((LOG_START + 1))" "$LOG" | grep -v '^error: \|^-> ' | tail -n 12 \
            | sed "s/^/    /" >&3
        console ""
        console "The whole log: $LOG"
    fi
}

# Stops the run: die after something failed, refuse when it cannot even start (no
# root, a wrong option); only die shows the end of the log.
die() {
    fail "$*"
    exit 1
}
refuse() {
    SHOW_LOG_END=0
    fail "$*"
    exit 2
}

on_exit() {
    if [ "$1" -ne 0 ] && [ "$REPORTED" -eq 0 ]; then
        if [ "$1" -eq 130 ]; then
            SHOW_LOG_END=0
            fail "interrupted"
        else
            fail "a command failed (exit status $1)"
        fi
    fi
}

# --- helpers -------------------------------------------------------------------------
# Runs a command that changes something; in a dry run only says what it would run.
run() {
    if [ "$DRY_RUN" -eq 1 ]; then
        say "would run: $*"
    else
        printf '$ %s\n' "$*"
        "$@"
    fi
}

# What the run changed (or, in a dry run, would change) that takes effect only after a
# reboot; empty when a restart of the radio is enough.
reboot_for=
needs_reboot() { # needs_reboot WHAT
    case ", $reboot_for, " in
        *", $1, "*) ;;
        *) reboot_for="${reboot_for:+$reboot_for, }$1" ;;
    esac
}

# Whether a dry run without root cannot see a file: the nearest directory on its path
# that is there may not be searched (/etc/polkit-1/rules.d is root:polkitd, 750, the
# radio's home 700), or the file not read.
hidden() { # hidden PATH
    [ "$(id -u)" -ne 0 ] || return 1
    hidden_dir=${1%/*}
    while [ -n "$hidden_dir" ] && [ ! -d "$hidden_dir" ]; do
        hidden_dir=${hidden_dir%/*}
    done
    if [ -n "$hidden_dir" ] && [ ! -x "$hidden_dir" ]; then
        return 0
    fi
    [ -e "$1" ] && [ ! -r "$1" ]
}

# Installs a file where the one in place differs (in a dry run: shows the difference)
# and notes in put_changed whether it did; the step counts it among its changes, or
# among the files it cannot see.
put_changed=0
put() { # put SOURCE DESTINATION
    if [ -f "$2" ] && cmp -s "$1" "$2"; then
        say "$2 unchanged"
        return
    fi
    if [ "$DRY_RUN" -eq 1 ] && hidden "$2"; then
        say "cannot see $2 without root; it may be there already (sudo with --dry-run looks)"
        unseen=$((unseen + 1))
        return
    fi
    put_changed=1
    changes=$((changes + 1))
    if [ "$DRY_RUN" -eq 0 ]; then
        install -m 644 "$1" "$2"
        say "$2 installed"
    elif [ -f "$2" ]; then
        say "would replace $2 with this change:"
        diff -u --label "$2 (installed)" --label "$2 (new)" "$2" "$1" || true
    else
        say "would install $2 (not there yet)"
    fi
}

# The end of a step that installs files: what it changed, or would.
end_with_changes() {
    if [ "$changes" -eq 0 ]; then
        kind=ok
        text=unchanged
    elif [ "$DRY_RUN" -eq 1 ]; then
        kind=would
        text="would make $(plural "$changes" change) (see the log)"
    else
        kind=ok
        text="updated ($(plural "$changes" change))"
    fi
    if [ "$unseen" -eq 1 ]; then
        text="$text; 1 file needs root to check"
    elif [ "$unseen" -gt 1 ]; then
        text="$text; $unseen files need root to check"
    fi
    step_end "$kind" "$text"
}

# Makes the directories of PATH, relative to the home of the radio's account, where
# they are missing, each the account's own.
user_dir() { # user_dir PATH
    user_dir_path=$USER_HOME
    user_dir_rest=$1/
    while [ -n "$user_dir_rest" ]; do
        user_dir_path=$user_dir_path/${user_dir_rest%%/*}
        user_dir_rest=${user_dir_rest#*/}
        if [ ! -d "$user_dir_path" ]; then
            run install -d -o "$USER_NAME" -g "$USER_GROUP" -m 755 "$user_dir_path"
        fi
    done
}

# git in the checkout DIR, whoever it belongs to: /opt/dab-radio is root's, and a dry
# run reads it as a user; a checkout to install from may be a user's, and this runs as
# root. git would refuse a repository of another owner.
git_in() { # git_in DIR ARGS...
    git_in_dir=$1
    shift
    git -c safe.directory='*' -C "$git_in_dir" "$@"
}

# The option that lets git fetch from SOURCE where it is the path of a checkout of
# another owner (root fetching from a user's, a user's dry run from /opt/dab-radio):
# the upload-pack on that side refuses it, and git_in's -c does not reach it, git
# clears the configuration of the command line for the side it serves from. sudo's
# SUDO_UID only covers a checkout of the one who ran sudo. Empty for a URL.
serve_option() { # serve_option SOURCE
    case $1 in
        /*) echo "--upload-pack=git -c 'safe.directory=*' upload-pack" ;;
    esac
}

# The default branch of the repository at URL; empty where it does not say.
default_branch() { # default_branch URL
    git ls-remote --symref "$1" HEAD 2>/dev/null \
        | sed -n 's|^ref: refs/heads/\(.*\)[[:space:]][[:space:]]*HEAD$|\1|p'
}

# The version of the radio in the checkout DIR, and its commit: 1.0.0 (1a2b3c4).
version_of() { # version_of DIR
    version=$(sed -n 's/^version = "\(.*\)"$/\1/p' "$1/pyproject.toml" 2>/dev/null | head -n 1)
    commit=$(git_in "$1" rev-parse --short HEAD 2>/dev/null || echo '?')
    echo "${version:-?} ($commit)"
}

# Brings /opt/dab-radio to REF of SOURCE (a URL or the path of a checkout), on BRANCH
# (empty: the commit alone, without a branch), making the checkout there first. ORIGIN
# is where --update fetches from later. Only the newest commit comes over, not the
# history.
put_checkout() { # put_checkout SOURCE REF ORIGIN BRANCH
    if [ ! -d "$INSTALL_DIR/.git" ]; then
        if [ -e "$INSTALL_DIR" ] && [ -n "$(ls -A "$INSTALL_DIR" 2>/dev/null)" ]; then
            refuse "$INSTALL_DIR is there but is no git checkout; move it away and run again"
        fi
        mkdir -p "$INSTALL_DIR"
        git -c init.defaultBranch=master init -q "$INSTALL_DIR"
    fi
    if git_in "$INSTALL_DIR" remote get-url origin >/dev/null 2>&1; then
        git_in "$INSTALL_DIR" remote set-url origin "$3"
    else
        git_in "$INSTALL_DIR" remote add origin "$3"
    fi
    serve=$(serve_option "$1")
    working "fetching $2" git_in "$INSTALL_DIR" fetch --depth 1 ${serve:+"$serve"} "$1" "$2" \
        || die "cannot fetch $2 from $1"
    if [ -n "$4" ]; then
        git_in "$INSTALL_DIR" checkout -q -f -B "$4" FETCH_HEAD
        git_in "$INSTALL_DIR" config "branch.$4.remote" origin
        git_in "$INSTALL_DIR" config "branch.$4.merge" "refs/heads/$4"
    else
        git_in "$INSTALL_DIR" checkout -q -f --detach FETCH_HEAD
    fi
    say "$INSTALL_DIR is at $(git_in "$INSTALL_DIR" log -1 --format='%h %s')"
}

# Goes on with the copy of this script in /opt/dab-radio, which the run has just put
# there: same console, same log, the checkout done.
hand_over() {
    say "going on with $INSTALL_DIR/pi/install.sh"
    DAB_RADIO_HANDOVER=1 DAB_RADIO_CHANGED=$ANY_CHANGE exec sh "$INSTALL_DIR/pi/install.sh" "$@"
}

# uv for the environment in /opt/dab-radio: Python goes there too, not into root's home,
# which the radio's account cannot read.
uv_env() { # uv_env UV ARGS...
    env UV_PYTHON_INSTALL_DIR="$INSTALL_DIR/.python" \
        UV_PYTHON_BIN_DIR="$INSTALL_DIR/.python/bin" \
        UV_PROJECT_ENVIRONMENT="$INSTALL_DIR/.venv" "$@"
}

# --- piped from the web --------------------------------------------------------------
# curl ... | sudo sh: the checkout in /opt/dab-radio comes first, then its own copy of
# this script goes on. A dry run looks from a temporary clone instead.
bootstrap() {
    setup_output "$@"
    url=${FROM_URL:-$SOURCE_URL}
    if [ "$DRY_RUN" -eq 1 ]; then
        step_begin Download
        [ -n "$url" ] || refuse "from which repository? curl ... | sh -s -- --dry-run --from URL"
        command -v git >/dev/null 2>&1 || refuse "a dry run needs git: sudo apt install git"
        look=$(mktemp -d)
        serve=$(serve_option "$url")
        if ! working "cloning $url" git clone -q --depth 1 ${serve:+--no-local "$serve"} "$url" "$look/dab-radio"; then
            rm -rf "$look"
            die "cannot clone $url"
        fi
        step_end ok "$url, into a temporary directory to look from"
        status=0
        DAB_RADIO_URL=$url sh "$look/dab-radio/pi/install.sh" "$@" || status=$?
        rm -rf "$look"
        # The run there has shown its failure.
        [ "$status" -eq 0 ] || REPORTED=1
        exit "$status"
    fi
    step_begin Checkout
    [ -n "$url" ] || refuse "from which repository? curl ... | sudo sh -s -- --from URL"
    [ "$(id -u)" -eq 0 ] || refuse "run as root: curl ... | sudo sh"
    if ! command -v git >/dev/null 2>&1; then
        working "updating the package lists" apt-get update -q
        working "installing git" env DEBIAN_FRONTEND=noninteractive \
            apt-get install -y -q git ca-certificates
    fi
    branch=$(default_branch "$url")
    before=$(git_in "$INSTALL_DIR" rev-parse HEAD 2>/dev/null || true)
    put_checkout "$url" "${branch:-HEAD}" "$url" "$branch"
    [ "$(git_in "$INSTALL_DIR" rev-parse HEAD)" = "$before" ] || changes=1
    step_end ok "$INSTALL_DIR, version $(version_of "$INSTALL_DIR"), from $url"
    hand_over "$@"
}

# --- 0. the checkout in /opt/dab-radio -----------------------------------------------
# From a checkout elsewhere its commit goes there, and the copy of this script there
# goes on; a dry run goes on here, with the files of this checkout. From /opt/dab-radio
# with --update the newest commit of its branch comes first.
step_checkout() {
    # The run before put the checkout in place and showed it.
    if [ -n "${DAB_RADIO_HANDOVER:-}" ]; then
        return
    fi
    step_begin Checkout
    if [ "$REPO" != "$INSTALLED" ]; then
        [ "$UPDATE" -eq 0 ] \
            || refuse "--update is for the installed checkout: sudo sh $INSTALL_DIR/pi/install.sh --update (from a checkout, its commit is installed)"
        # A piped dry run hands its --from on to the clone it looks from.
        [ -z "$FROM_URL" ] || [ -n "${DAB_RADIO_URL:-}" ] \
            || refuse "--from goes with a piped run or with --update"
        git_in "$REPO" rev-parse -q --verify HEAD >/dev/null 2>&1 \
            || refuse "$REPO is no git checkout with a commit"
        source=${DAB_RADIO_URL:-$REPO}
        if [ -n "$(git_in "$REPO" --no-optional-locks status --porcelain --untracked-files=no)" ]; then
            say "$REPO has changes that are not committed; only its commit is installed"
            note "changes in $REPO that are not committed are not installed" warn
        fi
        if [ "$DRY_RUN" -eq 1 ]; then
            say "the files to install are those of $REPO"
            if [ ! -d "$INSTALL_DIR/.git" ]; then
                changes=1
                step_end would "would put $INSTALL_DIR in place, version $(version_of "$REPO"), from $source"
            elif [ "$(git_in "$INSTALL_DIR" rev-parse -q --verify HEAD 2>/dev/null || true)" \
                = "$(git_in "$REPO" rev-parse HEAD)" ]; then
                step_end ok "$INSTALL_DIR, version $(version_of "$REPO"), as $source already"
            else
                changes=1
                step_end would "would bring $INSTALL_DIR from $(version_of "$INSTALL_DIR") to $(version_of "$REPO"), from $source"
            fi
            return
        fi
        branch=$(git_in "$REPO" symbolic-ref -q --short HEAD || true)
        origin=$(git_in "$REPO" remote get-url origin 2>/dev/null || echo "$REPO")
        before=$(git_in "$INSTALL_DIR" rev-parse HEAD 2>/dev/null || true)
        put_checkout "$REPO" HEAD "$origin" "$branch"
        [ "$(git_in "$INSTALL_DIR" rev-parse HEAD)" = "$before" ] || changes=1
        step_end ok "$INSTALL_DIR, version $(version_of "$INSTALL_DIR"), from $source"
        hand_over "$@"
    fi
    if [ "$UPDATE" -eq 0 ]; then
        [ -z "$FROM_URL" ] || refuse "--from goes with --update here"
        step_end ok "$INSTALL_DIR, version $(version_of "$INSTALL_DIR")"
        return
    fi
    origin=${FROM_URL:-$(git_in "$INSTALL_DIR" remote get-url origin 2>/dev/null || true)}
    [ -n "$origin" ] || refuse "$INSTALL_DIR names no repository to update from; --from URL names one"
    branch=$(git_in "$INSTALL_DIR" symbolic-ref -q --short HEAD || true)
    if [ "$DRY_RUN" -eq 1 ]; then
        theirs=$(git ls-remote "$origin" "${branch:+refs/heads/}${branch:-HEAD}" 2>/dev/null \
            | head -n 1 | cut -f 1)
        ours=$(git_in "$INSTALL_DIR" rev-parse HEAD 2>/dev/null || true)
        if [ -z "$theirs" ]; then
            step_end warn "cannot reach ${branch:-HEAD} of $origin"
        elif [ "$theirs" = "$ours" ]; then
            step_end ok "$INSTALL_DIR, version $(version_of "$INSTALL_DIR"), the newest of $origin"
        else
            changes=1
            step_end would "would bring $INSTALL_DIR from $(version_of "$INSTALL_DIR") to $(echo "$theirs" | cut -c 1-7), from $origin"
        fi
        return
    fi
    before=$(git_in "$INSTALL_DIR" rev-parse HEAD 2>/dev/null || true)
    put_checkout "$origin" "${branch:-HEAD}" "$origin" "$branch"
    if [ "$(git_in "$INSTALL_DIR" rev-parse HEAD)" = "$before" ]; then
        step_end ok "$INSTALL_DIR, version $(version_of "$INSTALL_DIR"), the newest of $origin"
    else
        changes=1
        step_end ok "$INSTALL_DIR updated to version $(version_of "$INSTALL_DIR"), from $origin"
    fi
    hand_over "$@"
}

# The account the radio runs as: --user NAME, an existing one, or its own, made in
# step 2 where it is missing (NEW_USER; its UID comes with it).
NEW_USER=0
resolve_account() {
    USER_NAME=${USER_NAME:-$RADIO_USER}
    [ "$USER_NAME" != root ] || refuse "not as root: the radio runs as an account of its own"
    if getent passwd "$USER_NAME" >/dev/null; then
        USER_HOME=$(getent passwd "$USER_NAME" | cut -d: -f6)
        [ -n "$USER_HOME" ] && [ -d "$USER_HOME" ] || refuse "no home directory for $USER_NAME"
        USER_UID=$(id -u "$USER_NAME")
        USER_GROUP=$(id -gn "$USER_NAME")
        say "account $USER_NAME (uid $USER_UID, home $USER_HOME), installed in $INSTALL_DIR"
    elif [ "$USER_NAME" = "$RADIO_USER" ]; then
        NEW_USER=1
        USER_HOME=$RADIO_HOME
        USER_UID="(new)"
        USER_GROUP=$USER_NAME
        say "account $USER_NAME (new, home $USER_HOME), installed in $INSTALL_DIR"
    else
        refuse "no account $USER_NAME; --user names an existing one, without it the radio gets its own, $RADIO_USER"
    fi
}

# --- 1. packages ---------------------------------------------------------------------
# Checked against a set-up Pi on 09.10.2026 (dry run): apt wanted only rtl-sdr; Qt
# loaded without libopengl0, so it is not listed. A fresh Raspberry Pi OS Lite lacked
# libxrandr2 (16.10.2026): Qt's media backend, the ffmpeg plugin, links it, and
# QMediaDevices loads that backend.
dpkg_state() { dpkg-query -W -f '${binary:Package} ${Version} ${db:Status-Abbrev}\n'; }

step_packages() {
    step_begin Packages
    PACKAGES="
        cage wlr-randr wlrctl
        pipewire pipewire-pulse wireplumber libspa-0.2-bluetooth pulseaudio-utils bluez
        network-manager udisks2 ntfs-3g
        libfaad2 libmpg123-0
        libpulse0 libegl1 libgl1 libgles2 libxkbcommon0 libfontconfig1 libfreetype6
        libdbus-1-3 libwayland-client0 libwayland-cursor0 libwayland-egl1 fonts-dejavu-core
        libxrandr2
        rtl-sdr
        git curl ca-certificates
    "
    # Debian's initramfs hook of Plymouth calls fc-match and fc-cache for any theme
    # with pictures, but plymouth does not depend on fontconfig: on a fresh Raspberry
    # Pi OS Lite the hook stopped with status 127 (16.10.2026).
    case $SPLASH in
        on|theme) PACKAGES="$PACKAGES plymouth fontconfig" ;;
    esac
    if [ "$DRY_RUN" -eq 1 ]; then
        say "asking apt what it would install (package lists as they are; a run updates them first)"
        out=$(mktemp)
        # shellcheck disable=SC2086
        if apt-get install -s -q $PACKAGES > "$out" 2>&1; then
            cat "$out"
            wanted=$(sed -n 's/^Inst \([^ ]*\).*/\1/p' "$out" | tr '\n' ' ')
            count=$(sed -n '/^Inst /p' "$out" | wc -l)
            if [ "$count" -gt 0 ]; then
                say "would install: $wanted"
                step_end would "would install or upgrade $(plural "$count" package) (see the log)"
                needs_reboot packages
            else
                step_end ok "all installed"
            fi
        else
            cat "$out"
            step_end warn "apt cannot resolve the package list (see the log)"
            needs_reboot "packages (apt could not tell)"
        fi
        rm -f "$out"
        return
    fi
    export DEBIAN_FRONTEND=noninteractive
    working "updating the package lists" apt-get update -q
    # Whatever apt installs or upgrades (an upgrade of PipeWire or BlueZ among the listed
    # ones, too) shows in dpkg's state, without reading apt's messages. A configuration
    # file changed here stays as it is, rather than dpkg asking.
    before=$(mktemp)
    after=$(mktemp)
    dpkg_state > "$before"
    # shellcheck disable=SC2086
    working "installing (a few minutes the first time)" apt-get install -y -q \
        -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold $PACKAGES
    dpkg_state > "$after"
    count=$(diff "$before" "$after" | sed -n '/^>/p' | wc -l)
    rm -f "$before" "$after"
    if [ "$count" -gt 0 ]; then
        needs_reboot packages
        step_end ok "$(plural "$count" package) installed or upgraded"
    else
        step_end ok "all installed"
    fi
}

# --- 2. the account ------------------------------------------------------------------
# Whether the radio's account is in GROUP; an account a dry run would only make is in
# none.
in_group() { # in_group GROUP
    if [ "$NEW_USER" -eq 1 ] && [ "$DRY_RUN" -eq 1 ]; then
        return 1
    fi
    id -nG "$USER_NAME" | tr ' ' '\n' | grep -qx "$1"
}

step_account() {
    step_begin Account
    if [ "$NEW_USER" -eq 1 ]; then
        [ ! -e "$RADIO_HOME" ] \
            || refuse "$RADIO_HOME is there, but there is no account $RADIO_USER; move it away and run again"
        run useradd --create-home --home-dir "$RADIO_HOME" --shell /usr/sbin/nologin \
            --comment "DAB+ radio" "$USER_NAME"
        if [ "$DRY_RUN" -eq 0 ]; then
            USER_UID=$(id -u "$USER_NAME")
            USER_GROUP=$(id -gn "$USER_NAME")
            say "account $USER_NAME made (uid $USER_UID, home $USER_HOME)"
        fi
        needs_reboot "the radio's account"
    fi
    # audio for the sound card, which the session on tty1 gets through logind as well.
    added=
    for group in video render plugdev netdev bluetooth audio; do
        if ! getent group "$group" >/dev/null; then
            say "no group $group on this system; skipped"
            note "no group $group on this system" warn
        elif in_group "$group"; then
            say "$USER_NAME is in $group"
        else
            run usermod -aG "$group" "$USER_NAME"
            added="${added:+$added, }$group"
            needs_reboot groups
        fi
    done
    if [ "$NEW_USER" -eq 1 ] && [ "$DRY_RUN" -eq 1 ]; then
        step_end would "would make $USER_NAME (home $USER_HOME), in $added"
    elif [ "$NEW_USER" -eq 1 ]; then
        step_end ok "$USER_NAME made (uid $USER_UID, home $USER_HOME)"
    elif [ -n "$added" ] && [ "$DRY_RUN" -eq 1 ]; then
        step_end would "$USER_NAME (uid $USER_UID), would add it to $added"
    elif [ -n "$added" ]; then
        step_end ok "$USER_NAME (uid $USER_UID), added to $added"
    else
        step_end ok "$USER_NAME (uid $USER_UID, home $USER_HOME)"
    fi
}

# The data of the account the radio ran as before, once: what of it the radio's account
# lacks. That account is the unit's (User=), or else the one who ran sudo; without one
# there is no step to show. Sets DATA_DIR to where the radio's data will be.
DATA_DIR=
step_data() {
    DATA_DIR=$USER_HOME/.local/share/dab-radio
    take_over_data
    # Where the data files of the traffic map go, before the radio first made it.
    if [ "$DRY_RUN" -eq 0 ] && [ ! -e "$USER_HOME/.local/share/dab-radio" ]; then
        user_dir .local/share/dab-radio
    fi
}

take_over_data() {
    former=
    if [ -f "$UNIT" ]; then
        former=$(sed -n 's/^User=//p' "$UNIT" | head -n 1)
    fi
    former=${former:-${SUDO_USER:-}}
    if [ -z "$former" ] || [ "$former" = "$USER_NAME" ] || [ "$former" = root ]; then
        return
    fi
    former_home=$(getent passwd "$former" | cut -d: -f6)
    [ -n "$former_home" ] || return 0
    step_begin Data
    if hidden "$former_home/.local/share/dab-radio"; then
        step_end warn "cannot see the data of $former without root (sudo with --dry-run looks)"
        return
    fi
    data=
    if [ -d "$former_home/.local/share/dab-radio" ]; then
        data=.local/share/dab-radio
    fi
    copy=
    what=
    for item in $data .local/state/wireplumber; do
        [ -d "$former_home/$item" ] || continue
        if [ "$item" = "$data" ]; then
            name="settings and data"
        else
            name="WirePlumber's state"
        fi
        if [ "$NEW_USER" -eq 0 ] && hidden "$USER_HOME/$item"; then
            say "cannot see $USER_HOME without root, nor whether it lacks $item of $former"
            unseen=$((unseen + 1))
            continue
        fi
        if [ -e "$USER_HOME/$item" ]; then
            say "$USER_HOME/$item is there; $former_home/$item not copied"
            continue
        fi
        copy="$copy $item"
        what="${what:+$what, }$name"
    done
    if [ -z "$copy" ]; then
        if [ "$unseen" -gt 0 ]; then
            step_end warn "cannot see whether $USER_HOME lacks the data of $former without root"
        else
            step_end ok "nothing left to take over from $former"
        fi
        return
    fi
    # In the settings: the folder of the recordings, on the USB stick, which UDisks2
    # mounts under /media/<account>/ for the account that mounts it. QSettings quotes a
    # value with a comma in it.
    media="s#=\(\"\{0,1\}\)/media/$former/#=\1/media/$USER_NAME/#"
    settings=$data/settings.ini
    needs_reboot "the radio's account"
    if [ "$DRY_RUN" -eq 1 ]; then
        for item in $copy; do
            say "would copy $former_home/$item to $USER_HOME/$item"
        done
        if [ -n "$data" ] && grep -q "=\"\{0,1\}/media/$former/" "$former_home/$settings" 2>/dev/null; then
            say "would move the folder of the recordings in the settings from /media/$former/ to /media/$USER_NAME/"
            note "the folder of the recordings would follow the USB drive to /media/$USER_NAME/"
        fi
        case " $copy " in
            *" $data "*) DATA_DIR=$former_home/$data ;;
        esac
        step_end would "would take over from $former: $what"
        return
    fi
    if systemctl is-active -q dab-radio.service 2>/dev/null; then
        systemctl stop dab-radio.service
        say "the radio stopped while its data is copied; it starts again at the reboot"
        note "the radio stopped for it; it starts again at the reboot"
    fi
    for item in $copy; do
        mkdir -p "$USER_HOME/${item%/*}"
        cp -a "$former_home/$item" "$USER_HOME/$item"
        say "$former_home/$item copied to $USER_HOME/$item"
    done
    chown -R "$USER_NAME:$USER_GROUP" "$USER_HOME/.local"
    if [ -n "$data" ] && [ -f "$USER_HOME/$settings" ] \
        && grep -q "=\"\{0,1\}/media/$former/" "$USER_HOME/$settings"; then
        sed -i "$media" "$USER_HOME/$settings"
        say "the folder of the recordings moved in the settings from /media/$former/ to /media/$USER_NAME/"
        note "the folder of the recordings follows the USB drive to /media/$USER_NAME/"
    fi
    say "the data of $former stays where it was"
    step_end ok "taken over from $former: $what"
}

# --- 3. rules ------------------------------------------------------------------------
step_rules() {
    step_begin Rules
    # The udev rules apply when the devices come, at the boot.
    put_changed=0
    put "$REPO/pi/99-dab-radio-touch-rotation.rules" /etc/udev/rules.d/99-dab-radio-touch-rotation.rules
    put "$REPO/pi/99-dab-radio-backlight.rules" /etc/udev/rules.d/99-dab-radio-backlight.rules
    if [ "$put_changed" -eq 1 ]; then
        run udevadm control --reload-rules
        needs_reboot "udev rules"
    fi
    # The polkit rules matter for a radio started over SSH; from the unit it runs in a
    # local, active session, where polkit's defaults already allow most of it. polkitd
    # watches its rules directory and reloads them by itself.
    put "$REPO/pi/50-dab-radio-networkmanager.rules" /etc/polkit-1/rules.d/50-dab-radio-networkmanager.rules
    put "$REPO/pi/50-dab-radio-udisks2.rules" /etc/polkit-1/rules.d/50-dab-radio-udisks2.rules

    # The kernel's DVB driver would take the RTL2832U before librtlsdr (the one bundled
    # with pyrtlsdrlib; the system's only brings the udev rule) can open it. A blacklist
    # that is already there counts, whatever its name (blacklist-rtlsdr.conf, made by
    # hand, on the Pi here).
    existing=$(grep -rls '^blacklist[[:space:]][[:space:]]*dvb_usb_rtl28xxu' /etc/modprobe.d/ 2>/dev/null \
        | head -n 1 || true)
    if [ -n "$existing" ]; then
        say "DVB driver already blacklisted in $existing"
    else
        blacklist=$(mktemp)
        cat > "$blacklist" <<'EOF'
# dab-radio: keep the kernel's DVB driver off the RTL-SDR stick.
blacklist dvb_usb_rtl28xxu
blacklist rtl2832
blacklist rtl2830
EOF
        put_changed=0
        put "$blacklist" /etc/modprobe.d/dab-radio-rtl-sdr.conf
        rm -f "$blacklist"
        [ "$put_changed" -eq 0 ] || needs_reboot "DVB blacklist"
    fi
    end_with_changes
}

# --- 4. Bluetooth --------------------------------------------------------------------
step_bluetooth() {
    step_begin Bluetooth
    conf=/etc/bluetooth/main.conf
    if [ ! -f "$conf" ]; then
        say "$conf missing; Experimental not set"
        note "no $conf, so no battery levels of headphones" warn
    elif grep -Eq '^[[:space:]]*Experimental[[:space:]]*=[[:space:]]*true' "$conf"; then
        say "$conf has Experimental = true"
    elif [ "$DRY_RUN" -eq 1 ]; then
        say "would set Experimental = true in $conf and restart bluetooth"
        changes=$((changes + 1))
    else
        if grep -Eq '^[[:space:]]*#?[[:space:]]*Experimental[[:space:]]*=' "$conf"; then
            sed -i -E 's/^[[:space:]]*#?[[:space:]]*Experimental[[:space:]]*=.*/Experimental = true/' "$conf"
        elif grep -q '^\[General\]' "$conf"; then
            sed -i '/^\[General\]/a Experimental = true' "$conf"
        else
            printf '\n[General]\nExperimental = true\n' >> "$conf"
        fi
        say "$conf: Experimental = true set"
        changes=$((changes + 1))
        if systemctl is-active -q bluetooth; then
            systemctl restart bluetooth
            say "bluetooth restarted"
            note "bluetoothd restarted"
        fi
    fi

    # WirePlumber's Bluetooth side only in the radio's account. The PipeWire of every
    # account starts with its user manager, at an SSH login too, and would register A2DP
    # endpoints of its own at BlueZ: headphones could end up with it rather than with
    # the radio. WirePlumber reads the fragments in a user's home after those in /etc;
    # the radio's turns Bluetooth on again, without seat monitoring, so that connected
    # headphones keep their sound through a restart of the unit. It reads them when it
    # starts: at the reboot.
    wp_conf=/etc/wireplumber/wireplumber.conf.d
    user_wp=.config/wireplumber/wireplumber.conf.d
    if [ ! -d "$wp_conf" ]; then
        run mkdir -p "$wp_conf"
    fi
    if [ "$DRY_RUN" -eq 0 ]; then
        user_dir "$user_wp"
    fi
    put_changed=0
    put "$REPO/pi/50-dab-radio-no-bluetooth.conf" "$wp_conf/50-dab-radio-no-bluetooth.conf"
    put "$REPO/pi/50-dab-radio-bluetooth.conf" "$USER_HOME/$user_wp/50-dab-radio-bluetooth.conf"
    [ "$put_changed" -eq 0 ] || needs_reboot WirePlumber
    end_with_changes
}

# --- 5. uv and the Python environment ------------------------------------------------
step_python() {
    step_begin Python
    if [ -x "$UV" ]; then
        say "uv is $("$UV" --version)"
    elif [ "$DRY_RUN" -eq 1 ]; then
        say "would install uv into ${UV%/*} with the installer from astral.sh"
        note "would install uv into ${UV%/*}"
    else
        working "installing uv" sh -c 'curl -LsSf https://astral.sh/uv/install.sh \
            | env UV_INSTALL_DIR="$1" UV_NO_MODIFY_PATH=1 sh' sh "${UV%/*}"
        [ -x "$UV" ] || die "uv did not land in $UV"
        note "uv $("$UV" --version | cut -d ' ' -f 2) installed into ${UV%/*}"
    fi
    out=$(mktemp)
    if [ "$DRY_RUN" -eq 1 ]; then
        # Any uv will do for a look: the one of a run, or the user's own.
        uv=$UV
        if [ ! -x "$uv" ]; then
            uv=$(command -v uv || true)
        fi
        if [ -z "$uv" ] && [ -x "$HOME/.local/bin/uv" ]; then
            uv=$HOME/.local/bin/uv
        fi
        if [ -z "$uv" ]; then
            step_end would "no uv to look with; a run puts Python 3.14 and the packages into $INSTALL_DIR"
        elif (cd "$REPO" && uv_env "$uv" sync --dry-run --locked --no-dev --no-python-downloads) \
            > "$out" 2>&1; then
            cat "$out"
            if grep -q '^Would make no changes' "$out"; then
                step_end ok "environment up to date"
            else
                changes=1
                step_end would "would update the environment in $INSTALL_DIR (see the log)"
            fi
        else
            cat "$out"
            step_end would "uv cannot tell from here (no Python 3.14 in $INSTALL_DIR/.python yet, or no access without root); a run brings what is missing"
        fi
        rm -f "$out"
        return
    fi
    cd "$INSTALL_DIR"
    status=0
    working "uv sync (Python 3.14 and the packages; a while the first time)" \
        uv_env "$UV" sync --locked --no-dev --compile-bytecode > "$out" 2>&1 || status=$?
    cat "$out"
    [ "$status" -eq 0 ] || die "uv sync failed"
    if grep -Eq '^ *(Installed|Uninstalled) [0-9]' "$out"; then
        state="environment updated"
        changes=1
    else
        state="environment up to date"
    fi
    rm -f "$out"
    # The radio's account may not write here, so Python would compile the radio's
    # modules anew at every start; numba keeps its cache in the account's home then.
    working "compiling the radio's modules" "$INSTALL_DIR/.venv/bin/python" -m compileall -q "$INSTALL_DIR/src"
    step_end ok "$("$INSTALL_DIR/.venv/bin/python" -V 2>&1), $state"
}

# Qt comes with the PyQt6 wheel but loads EGL, GL, fontconfig, D-Bus, Wayland and
# libpulse from the system; a missing one shows up here, not at the first start. Only
# what the radio loads is checked: the modules it imports and the plugins cage makes it
# use. The wheel also carries plugins of modules it does not ship (QML, WebEngine, SQL
# drivers), the X11 platform and the GStreamer media backend, whose libraries are
# missing without harm: of the media backends Qt loads its default, ffmpeg (the
# GStreamer one only with QT_MEDIA_BACKEND=gstreamer).
step_qt() {
    step_begin Qt
    qt=$(find "$INSTALL_DIR/.venv" -type d -path '*/PyQt6/Qt6' -print -quit 2>/dev/null || true)
    if [ -z "$qt" ]; then
        say "no Qt in $INSTALL_DIR/.venv yet; the check of its libraries comes after uv sync"
        step_end would "checked once $INSTALL_DIR/.venv is there"
        return
    fi
    missing=$(
        for lib in "$qt"/lib/libQt6Core.so.6 "$qt"/lib/libQt6Gui.so.6 \
            "$qt"/lib/libQt6Widgets.so.6 "$qt"/lib/libQt6DBus.so.6 \
            "$qt"/lib/libQt6Multimedia.so.6 "$qt"/lib/libQt6OpenGL.so.6 \
            "$qt"/plugins/platforms/libqwayland*.so "$qt"/plugins/wayland-*/*.so \
            "$qt"/plugins/multimedia/libffmpegmediaplugin.so "$qt"/plugins/imageformats/*.so; do
            [ -e "$lib" ] || continue
            ldd "$lib" 2>/dev/null | grep 'not found' || true
        done | sort -u
    )
    if [ -n "$missing" ]; then
        say "Qt misses these; install the packages that provide them:"
        echo "$missing"
        libs=$(echo "$missing" | sed 's/^[[:space:]]*\([^ ]*\).*/\1/' | tr '\n' ' ')
        todo "Qt: install the packages that provide $libs"
        step_end warn "misses system libraries: $libs"
    else
        step_end ok "finds all its system libraries"
    fi
}

# --- 6. autostart --------------------------------------------------------------------
# The example unit keeps its shape; only account, paths and UID are those of this Pi.
step_unit() {
    step_begin Autostart
    if [ "$AUTOSTART" -eq 0 ]; then
        if [ -f "$UNIT" ]; then
            step_end ok "$UNIT left as it is (--no-autostart)"
        else
            step_end ok "left out (--no-autostart)"
        fi
        return
    fi
    # A session of another account on tty1 before: its PipeWire goes with the reboot.
    if [ -f "$UNIT" ] && [ "$(sed -n 's/^User=//p' "$UNIT" | head -n 1)" != "$USER_NAME" ]; then
        needs_reboot "the radio's account"
    fi
    rendered=$(mktemp)
    sed -e "s|^User=.*|User=$USER_NAME|" \
        -e "s|^WorkingDirectory=.*|WorkingDirectory=$INSTALL_DIR|" \
        -e "s|^ExecStartPre=+.*|ExecStartPre=+/bin/sh $INSTALL_DIR/pi/rebind-touch.sh|" \
        -e "s|^ExecStart=.*|ExecStart=/bin/sh $INSTALL_DIR/pi/start-touch.sh|" \
        -e "s|XDG_RUNTIME_DIR=/run/user/[0-9]*|XDG_RUNTIME_DIR=/run/user/$USER_UID|" \
        "$REPO/pi/dab-radio.service" > "$rendered"
    there=0
    [ ! -f "$UNIT" ] || there=1
    put_changed=0
    put "$rendered" "$UNIT"
    rm -f "$rendered"
    # A changed unit takes effect when the radio restarts.
    if [ "$put_changed" -eq 1 ]; then
        run systemctl daemon-reload
    fi
    if systemctl is-enabled -q dab-radio 2>/dev/null; then
        say "dab-radio.service is enabled"
        enabled=enabled
    else
        run systemctl enable -q dab-radio
        needs_reboot autostart
        enabled="enabled for the next boot"
    fi
    if [ "$put_changed" -eq 0 ]; then
        step_end ok "dab-radio.service unchanged, $enabled"
    elif [ "$DRY_RUN" -eq 1 ] && [ "$there" -eq 1 ]; then
        step_end would "would update dab-radio.service (diff in the log), $enabled"
    elif [ "$DRY_RUN" -eq 1 ]; then
        step_end would "would install dab-radio.service, $enabled"
    elif [ "$there" -eq 1 ]; then
        step_end ok "dab-radio.service updated, $enabled"
    else
        step_end ok "dab-radio.service installed, $enabled"
    fi
}

# --- 7. boot splash ------------------------------------------------------------------
# Plymouth with pi/plymouth/dab-radio, whose last picture the radio shows while it
# starts (src/ui/touch/splash.py). The display of the Pi 5 comes up only when its driver
# loads from the root file system, 5.0 s after power on; the splash shows from there
# until the unit starts at 8.3 s and ends Plymouth (measured 10.10.2026, without it).
# Plymouth turns its picture by the panel orientation of the display, never by the
# rotate= the console turns by (fbcon adds the two); so the rotation of the output
# (pi/touch-display.sh) goes into cmdline.txt as panel_orientation= in place of rotate=,
# on the DSI connector the display is on. The video= words of the DSI connectors are the
# script's: the first becomes the one of the display, any further one goes (left from
# the other socket, before the cable moved); those of other connectors stay.
#
# Where cmdline.txt is, and with neither --splash nor --no-splash, whether a splash is
# on: then it stays on and gets the theme of this checkout, cmdline.txt stays as it is
# (SPLASH=theme). Found before the packages, which bring what its initramfs needs.
find_splash() {
    CMDLINE=
    for dir in "$(/usr/lib/raspberrypi-sys-mods/get_fw_loc 2>/dev/null || true)" \
        /boot/firmware /boot; do
        if [ -n "$dir" ] && [ -f "$dir/cmdline.txt" ]; then
            CMDLINE=$dir/cmdline.txt
            break
        fi
    done
    if [ -z "$SPLASH" ] && [ -n "$CMDLINE" ] \
        && [ "$(plymouth-set-default-theme 2>/dev/null || true)" = dab-radio ] \
        && tr ' ' '\n' < "$CMDLINE" | grep -qx splash; then
        SPLASH=theme
    fi
}

step_splash() {
    step_begin "Boot splash"
    cmdline=$CMDLINE
    if [ "$SPLASH" = theme ]; then
        say "the boot splash is on; its theme is brought up to date (--no-splash turns it off)"
    fi
    if [ -z "$SPLASH" ]; then
        step_end ok "off (--splash turns it on)"
        return
    fi
    [ -n "$cmdline" ] || die "no cmdline.txt in /boot/firmware or /boot"

    if [ "$SPLASH" = on ]; then
        . "$REPO/pi/touch-display.sh"
        output=$(dsi_output)
        [ -n "$output" ] || die "no DSI display connected (/sys/class/drm/card*-DSI-*/status); connect the Touch Display 2 and run again"
        # wlr-randr turns counter-clockwise like the kernel's rotate=; a panel whose
        # right side is up is turned by 270 degrees, as drm_client_rotation() does it.
        case $DISPLAY_TRANSFORM in
            0) orientation=normal ;;
            90) orientation=left_side_up ;;
            180) orientation=upside_down ;;
            270) orientation=right_side_up ;;
            *) die "pi/touch-display.sh: DISPLAY_TRANSFORM=$DISPLAY_TRANSFORM is not 0, 90, 180 or 270" ;;
        esac
        # With the mode, as the Pi here has run since 10.10.2026, where the connector
        # lists the one of the Touch Display 2; another panel keeps its own.
        if grep -qx "$DISPLAY_MODE" /sys/class/drm/card*-"$output"/modes 2>/dev/null; then
            video="video=$output:$DISPLAY_MODE@$DISPLAY_REFRESH,panel_orientation=$orientation"
            say "the Touch Display 2 is on $output"
        else
            video="video=$output:panel_orientation=$orientation"
            say "the display on $output does not list $DISPLAY_MODE, so it is not the Touch Display 2; video= without a mode"
            note "the display on $output is not the Touch Display 2; video= without a mode" warn
        fi
    fi

    # The words of the command line with the splash on or off; the others stay where
    # they are, missing ones are added at the end.
    if [ "$SPLASH" != theme ]; then
        new_cmdline=$(mktemp)
        words=
        has_quiet=0
        has_splash=0
        has_serial=0
        has_video=0
        set -f
        for word in $(cat "$cmdline"); do
            case $word in
                quiet) has_quiet=1; [ "$SPLASH" = on ] || continue ;;
                splash) has_splash=1; [ "$SPLASH" = on ] || continue ;;
                plymouth.ignore-serial-consoles) has_serial=1; [ "$SPLASH" = on ] || continue ;;
                video=DSI-*)
                    if [ "$SPLASH" = on ]; then
                        [ "$has_video" -eq 0 ] || continue
                        has_video=1
                        word=$video
                    fi
                    ;;
            esac
            words="$words $word"
        done
        set +f
        if [ "$SPLASH" = on ]; then
            [ "$has_video" -eq 1 ] || words="$words $video"
            [ "$has_quiet" -eq 1 ] || words="$words quiet"
            [ "$has_splash" -eq 1 ] || words="$words splash"
            [ "$has_serial" -eq 1 ] || words="$words plymouth.ignore-serial-consoles"
        fi
        printf '%s\n' "${words# }" > "$new_cmdline"
        backup=$cmdline.dab-radio.bak
        if ! cmp -s "$new_cmdline" "$cmdline" && [ "$DRY_RUN" -eq 0 ] && [ ! -f "$backup" ]; then
            cp "$cmdline" "$backup"
            say "$cmdline as it was saved as $backup"
            note "the old $cmdline stays as $backup"
        fi
        put_changed=0
        put "$new_cmdline" "$cmdline"
        rm -f "$new_cmdline"
        [ "$put_changed" -eq 0 ] || needs_reboot "boot splash"
    fi

    if [ "$SPLASH" = off ]; then
        say "--no-splash: Plymouth and its theme stay installed and show the console"
        if [ "$changes" -eq 0 ]; then
            step_end ok "off"
        elif [ "$DRY_RUN" -eq 1 ]; then
            step_end would "would turn it off (diff in the log)"
        else
            step_end ok "turned off; Plymouth shows the console"
        fi
        return
    fi
    theme=/usr/share/plymouth/themes/dab-radio
    if [ ! -d "$theme" ]; then
        run mkdir -p "$theme"
    fi
    put_changed=0
    for file in "$REPO"/pi/plymouth/dab-radio/*; do
        put "$file" "$theme/$(basename "$file")"
    done
    rebuild=$put_changed
    current=$(plymouth-set-default-theme 2>/dev/null || true)
    if [ "$current" = dab-radio ]; then
        say "Plymouth's theme is dab-radio"
    else
        run plymouth-set-default-theme dab-radio
        changes=$((changes + 1))
        rebuild=1
    fi
    # plymouthd starts in the initramfs and keeps the theme it found there; a run
    # that stopped before rebuilding it left the theme out.
    initrd=/boot/initrd.img-$(uname -r)
    if [ "$rebuild" -eq 0 ] && [ -r "$initrd" ] && ! lsinitramfs "$initrd" 2>/dev/null \
        | grep -q 'plymouth/themes/dab-radio/dab-radio\.script$'; then
        rebuild=1
    fi
    # Plymouth's hook warns there that label-pango.so is missing (plymouth-label); the
    # theme draws pictures only, no text.
    if [ "$rebuild" -eq 1 ]; then
        needs_reboot "boot splash"
        if [ "$DRY_RUN" -eq 1 ]; then
            say "would run: update-initramfs -u -k all"
        else
            working "rebuilding the initramfs (a minute or two)" update-initramfs -u -k all
        fi
    else
        say "the initramfs has the theme"
    fi
    if [ "$changes" -eq 0 ] && [ "$rebuild" -eq 0 ]; then
        step_end ok "on, theme up to date"
    elif [ "$DRY_RUN" -eq 1 ]; then
        step_end would "would bring it up to date, initramfs too (see the log)"
    else
        step_end ok "on, brought up to date with the initramfs"
    fi
}

# --- what remains by hand ------------------------------------------------------------
report() {
    if ! ls /usr/local/lib/libsdrplay_api.so.3* >/dev/null 2>&1; then
        todo "RSPduo: install the SDRplay API 3.15 for ARM64 from sdrplay.com (its service must run)"
    fi
    data=$USER_HOME/.local/share/dab-radio
    if [ "$DRY_RUN" -eq 1 ] && hidden "$DATA_DIR/road_network.npz"; then
        todo "Traffic map: whether its data files are in $DATA_DIR needs root to check"
    else
        missing=
        for file in road_network.npz tmc_locations.npz; do
            [ -f "$DATA_DIR/$file" ] || missing="${missing:+$missing }$file"
        done
        if [ -n "$missing" ]; then
            todo "Traffic map: copy $missing from the PC, as the radio's own:
      sudo install -o $USER_NAME -g $USER_GROUP -m 644 $missing $data/"
        fi
    fi
    # A reboot only for what takes effect then; otherwise the new version (the checkout,
    # its Python packages, a changed unit) takes effect when the radio starts again.
    if [ -n "$reboot_for" ]; then
        if [ "$DRY_RUN" -eq 1 ]; then
            todo "A reboot, for the changes to $reboot_for"
        else
            todo "Reboot, for the changes to $reboot_for: sudo reboot"
        fi
    elif [ -f "$UNIT" ] && systemctl is-active -q dab-radio 2>/dev/null; then
        [ "$ANY_CHANGE" -eq 0 ] || todo "Restart the radio: sudo systemctl restart dab-radio.service"
    elif [ -f "$UNIT" ] && [ "$AUTOSTART" -eq 1 ]; then
        todo "Start the radio: sudo systemctl start dab-radio.service"
    elif [ "$ANY_CHANGE" -eq 1 ]; then
        todo "Start the radio anew, as $USER_NAME: sh $INSTALL_DIR/pi/start-touch.sh"
    fi
    echo
    echo "== to do"
    printf '%s\n' "${TODO:-nothing}"
    console ""
    if [ -z "$TODO" ]; then
        console "${BOLD}Nothing to do${RESET}"
    else
        if [ "$DRY_RUN" -eq 1 ]; then
            console "${BOLD}A run would leave to do:${RESET}"
        else
            console "${BOLD}Still to do:${RESET}"
        fi
        printf '%s\n' "$TODO" | sed "s/^\([^ ]\)/  $BULLET \1/" >&3
    fi
    console ""
    [ "$LOG" = none ] || console "${DIM}Details: $LOG${RESET}"
}

install_radio() {
    setup_output "$@"
    if [ "$(id -u)" -ne 0 ] && [ "$DRY_RUN" -eq 0 ]; then
        refuse "run as root: sudo sh $0 (only --dry-run needs no root)"
    fi
    step_checkout "$@"
    resolve_account
    find_splash
    step_packages
    step_account
    step_data
    step_rules
    step_bluetooth
    step_python
    step_qt
    step_unit
    step_splash
    report
}

# Only here, at the end: piped into sh, the shell has read the whole script by now, so a
# download cut short runs nothing.
if [ -z "$REPO" ]; then
    bootstrap "$@"
else
    install_radio "$@"
fi

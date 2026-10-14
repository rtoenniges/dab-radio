"""The WLAN of the Raspberry Pi, through nmcli, the command line of NetworkManager.

Raspberry Pi OS (from Bookworm on) manages its networks with NetworkManager, as the
network symbol of its desktop does. `WifiManager` runs one nmcli command at a time as a
`QProcess` (`net.command`), so that the surface never waits for it: a scan takes a few
seconds, connecting up to `CONNECT_WAIT_S`. The output is read in terse mode (`-t`), with
the messages in English (`LC_ALL=C.UTF-8`) so that they can be recognised.

NetworkManager asks polkit whether a program may scan, connect or change a profile. By
default polkit allows that only to the active desktop session, and asks even there for
a password; a radio started over SSH or by systemd may not do it at all.
`pi/50-dab-radio-networkmanager.rules` lets the members of the group netdev do what the
radio needs; `WifiManager.permissions_missing` tells whether it is in place.

The passwords stay with NetworkManager, in profiles only root can read; the radio keeps
none. A new password is handed to nmcli on its command line, as raspi-config does.

The IPv4 settings of a profile (DHCP or an address set by hand, and the DNS servers in
either case) are read with the profile and written with `WifiManager.apply_ipv4`.

A profile may be locked to one frequency band (`WifiManager.set_band`), for a network
that sends on 2.4 GHz and 5 GHz under one name. NetworkManager knows no preference,
only the lock (`802-11-wireless.band`); without it the Wi-Fi chooses. On the Raspberry
Pi, Wi-Fi and Bluetooth share one chip and its antenna, and on 2.4 GHz the same band
too: Bluetooth headphones then share their radio time with the Wi-Fi.
"""

from __future__ import annotations

import ipaddress
import shutil
import string
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from i18n import tr
from net.command import KILLED, NOT_STARTED, CommandResult, CommandRunner, Done

NMCLI = "nmcli"
# How long nmcli waits for a connection to come up before it gives up (exit code 3).
CONNECT_WAIT_S = 30
# For a new network longer: NetworkManager waits 45 s for DHCP before it reports that
# the network gives no address, and the radio keeps the profile then (`gave_no_address`),
# so that the address can be set by hand; nmcli must not give up before.
CONNECT_NEW_WAIT_S = 60
# A command that has not finished by then is killed; connecting has its own wait above.
COMMAND_TIMEOUT_MS = 30_000
CONNECT_TIMEOUT_MS = (CONNECT_WAIT_S + 15) * 1000
CONNECT_NEW_TIMEOUT_MS = (CONNECT_NEW_WAIT_S + 15) * 1000
# While the sub-page is shown: state, address and the networks in reach (from the last
# scan of NetworkManager, which scans by itself).
POLL_INTERVAL_MS = 5000
# In the background, from the start of the radio (`WifiManager.watch`): so that the
# sub-page has everything when it is opened, and for the symbol in the navigation rail
# (whether the WLAN is on and connected). Four short nmcli commands each time.
BACKGROUND_POLL_INTERVAL_MS = 30_000

# The exit codes of nmcli that mean something here (nmcli(1), "Exit Status"); those of
# a command that did not run to its end are in `net.command`.
EXIT_TIMEOUT = 3
EXIT_NOT_RUNNING = 8
EXIT_NOT_FOUND = 10

# The type of a WLAN profile in `nmcli connection show`.
WIFI_CONNECTION_TYPES = ("802-11-wireless", "wifi")
# What the radio asks NetworkManager to do (`nmcli general permissions`).
REQUIRED_PERMISSIONS = (
    "org.freedesktop.NetworkManager.enable-disable-wifi",
    "org.freedesktop.NetworkManager.network-control",
    "org.freedesktop.NetworkManager.settings.modify.system",
    "org.freedesktop.NetworkManager.wifi.scan",
)

SSID_MAX_BYTES = 32
PASSWORD_LENGTH = (8, 63)
# A WPA key may also be given as such: 64 hex digits instead of a passphrase.
PSK_HEX_LENGTH = 64

# The prefix lengths a subnet may have here, and the one of nearly every home network.
PREFIXES = range(8, 31)
DEFAULT_PREFIX = 24
# How a list of DNS servers is written for the surface; commas alone separate as well.
DNS_SEPARATOR = ", "

# The values of `802-11-wireless.band`: none (the Wi-Fi chooses), 2.4 GHz or 5 GHz.
BAND_AUTOMATIC = ""
BAND_2_4_GHZ = "bg"
BAND_5_GHZ = "a"
BANDS = (BAND_AUTOMATIC, BAND_2_4_GHZ, BAND_5_GHZ)
# Where the bands lie, in MHz. 6 GHz (Wi-Fi 6E) NetworkManager cannot lock, and the chip
# of the Raspberry Pi does not reach it.
_BAND_RANGES = ((BAND_2_4_GHZ, 2400, 2500), (BAND_5_GHZ, 4900, 5925))

_RADIO = ["radio", "wifi"]
_DEVICE_STATUS = ["-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "device", "status"]
_CONNECTIONS = ["-t", "-f", "UUID,TYPE,NAME", "connection", "show"]
_PERMISSIONS = ["-t", "-f", "PERMISSION,VALUE", "general", "permissions"]
# One "key:value" line each, values unescaped (unlike -g; seen with nmcli 1.52).
_PROFILE_FIELDS = (
    "802-11-wireless.ssid,802-11-wireless.band,"
    "ipv4.method,ipv4.addresses,ipv4.gateway,ipv4.dns,ipv4.ignore-auto-dns"
)
_PROFILE = ["-t", "-f", _PROFILE_FIELDS, "connection", "show", "uuid"]
_ACTIVE_IPV4 = ["-t", "-f", "IP4,DHCP4", "device", "show"]
_NETWORKS = ["-t", "-f", "IN-USE,SIGNAL,FREQ,SECURITY,SSID", "device", "wifi", "list"]


@dataclass(frozen=True)
class Ipv4Settings:
    """How a profile comes by its IPv4 address: by DHCP, or as set here (`manual`).
    `dns` replaces the servers of the router with DHCP; empty, the router's count then.
    By hand, they are the only ones (none where empty)."""

    manual: bool = False
    address: str = ""
    prefix: int = DEFAULT_PREFIX
    gateway: str = ""
    dns: tuple[str, ...] = ()

    def comparable(self) -> Ipv4Settings:
        """With DHCP, what is set by hand does not count."""
        return self if self.manual else Ipv4Settings(dns=self.dns)


@dataclass(frozen=True)
class ActiveIpv4:
    """The IPv4 configuration of the connection up now. `dns` are the servers the
    router gave by DHCP, or those in use without DHCP."""

    address: str = ""
    prefix: int = 0
    gateway: str = ""
    dns: tuple[str, ...] = ()


@dataclass(frozen=True)
class Network:
    """A network in reach: the access point connected to, or else the strongest of its
    access points, and the bands of all."""

    ssid: str
    signal: int  # percent
    security: str  # as nmcli names it: "WPA2", "WPA1 WPA2", "WPA3" …, "" when open
    in_use: bool = False
    bands: frozenset[str] = frozenset()  # `BAND_2_4_GHZ`, `BAND_5_GHZ`
    band_in_use: str = BAND_AUTOMATIC  # that of the access point connected to

    @property
    def enterprise(self) -> bool:
        """WPA Enterprise (802.1X) asks for a user and certificates, which the radio
        cannot enter."""
        return "802.1X" in self.security

    @property
    def needs_password(self) -> bool:
        # OWE ("Enhanced Open") encrypts without a password.
        return self.security not in ("", "OWE") and not self.enterprise


@dataclass(frozen=True)
class SavedNetwork:
    """A WLAN profile of NetworkManager. Its name is not always the network's: the
    Raspberry Pi Imager calls the one it makes "preconfigured"."""

    uuid: str
    name: str
    ssid: str
    ipv4: Ipv4Settings = Ipv4Settings()
    band: str = BAND_AUTOMATIC  # the band it is locked to


@dataclass(frozen=True)
class DeviceStatus:
    device: str  # "wlan0"
    state: str  # "connected", "disconnected", "connecting (configuring)", "unavailable" …
    connection: str  # the name of the active profile, "" without one

    @property
    def connected(self) -> bool:
        return self.state == "connected"

    @property
    def connecting(self) -> bool:
        return self.state.startswith("connecting")


def find_nmcli() -> str | None:
    """The path of nmcli; None where there is none (Windows, or Linux without
    NetworkManager)."""
    return shutil.which(NMCLI)


def split_terse(line: str) -> list[str]:
    """The fields of a line of nmcli's terse output: separated by colons, a colon or
    backslash within a value escaped by a backslash."""
    fields: list[str] = []
    field: list[str] = []
    escaped = False
    for char in line:
        if escaped:
            field.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == ":":
            fields.append("".join(field))
            field = []
        else:
            field.append(char)
    fields.append("".join(field))
    return fields


def _value(text: str) -> str:
    """nmcli writes "--" for a value that is not there."""
    return "" if text == "--" else text


def band_of(frequency: str) -> str:
    """The band of a frequency as nmcli writes it ("2412 MHz"); `BAND_AUTOMATIC` (none)
    for one outside 2.4 GHz and 5 GHz."""
    try:
        mhz = int(frequency.split()[0])
    except IndexError, ValueError:
        return BAND_AUTOMATIC
    return next((band for band, low, high in _BAND_RANGES if low <= mhz < high), BAND_AUTOMATIC)


def band_name(band: str) -> str:
    """The name of a band for the surface; "Automatic" for none."""
    if band == BAND_2_4_GHZ:
        return tr("2.4 GHz")
    if band == BAND_5_GHZ:
        return tr("5 GHz")
    return tr("Automatic")


def parse_networks(output: str) -> list[Network]:
    """The networks in `nmcli -t -f IN-USE,SIGNAL,FREQ,SECURITY,SSID device wifi list`,
    the strongest first. nmcli lists every access point, so a network with several of
    them or on both bands once for each: here it is listed once, with the strongest, and
    with the bands of all. The network connected to has the signal of the access point
    connected to: its bars are those of the connection (on 2026-10-11 the Pi was
    connected over 5 GHz with 40 %, and showed the 70 % of 2.4 GHz). Hidden networks
    (without a name) are left out; they are entered by hand."""
    networks: dict[str, Network] = {}
    bands: dict[str, set[str]] = {}
    # Band and signal of the access point connected to.
    in_use: dict[str, tuple[str, int]] = {}
    for line in output.splitlines():
        fields = split_terse(line)
        if len(fields) != 5:
            continue
        mark, signal, frequency, security, ssid = fields
        ssid = _value(ssid)
        if not ssid:
            continue
        try:
            strength = int(signal)
        except ValueError:
            strength = 0
        band = band_of(frequency)
        bands.setdefault(ssid, set())
        if band:
            bands[ssid].add(band)
        if mark.strip() == "*":
            in_use[ssid] = (band, strength)
        known = networks.get(ssid)
        if known is None or strength > known.signal:
            networks[ssid] = Network(ssid, strength, _value(security))
    found = []
    for network in networks.values():
        network = replace(network, bands=frozenset(bands[network.ssid]))
        if network.ssid in in_use:
            band, strength = in_use[network.ssid]
            network = replace(network, signal=strength, in_use=True, band_in_use=band)
        found.append(network)
    return sorted(found, key=lambda n: (-n.signal, n.ssid.casefold()))


def parse_wifi_connections(output: str) -> list[tuple[str, str]]:
    """UUID and name of the WLAN profiles in `nmcli -t -f UUID,TYPE,NAME connection
    show`."""
    profiles = []
    for line in output.splitlines():
        fields = split_terse(line)
        if len(fields) == 3 and fields[1] in WIFI_CONNECTION_TYPES:
            profiles.append((fields[0], fields[2]))
    return profiles


def parse_device_status(output: str) -> DeviceStatus | None:
    """The first WLAN device in `nmcli -t -f DEVICE,TYPE,STATE,CONNECTION device status`;
    None without one. "wifi-p2p" (Wi-Fi Direct beside wlan0) is not one."""
    for line in output.splitlines():
        fields = split_terse(line)
        if len(fields) == 4 and fields[1] == "wifi":
            return DeviceStatus(fields[0], fields[2], _value(fields[3]))
    return None


def parse_fields(output: str) -> dict[str, str]:
    """Key and value of each line of `nmcli -t -f … show`: "ipv4.method:auto",
    "IP4.DNS[1]:192.168.2.1". The value is not escaped there, so it is all after the
    first colon (a key has none)."""
    fields = {}
    for line in output.splitlines():
        key, colon, value = line.partition(":")
        if colon:
            fields[key] = _value(value)
    return fields


def _split_list(text: str) -> list[str]:
    """A list as nmcli writes one, "a,b" or "a, b"."""
    return [part.strip() for part in text.split(",") if part.strip()]


def _address_and_prefix(text: str) -> tuple[str, int]:
    """The address and the prefix length of "192.168.2.20/24"."""
    address, _slash, prefix = text.strip().partition("/")
    try:
        return address, int(prefix)
    except ValueError:
        return address, DEFAULT_PREFIX


def parse_profile(output: str) -> tuple[str, Ipv4Settings, str]:
    """The network, the IPv4 settings and the band locked to of a WLAN profile
    (`_PROFILE`). Of several addresses set by hand the first counts: the surface sets
    one."""
    fields = parse_fields(output)
    addresses = _split_list(fields.get("ipv4.addresses", ""))
    address, prefix = _address_and_prefix(addresses[0]) if addresses else ("", DEFAULT_PREFIX)
    settings = Ipv4Settings(
        manual=fields.get("ipv4.method") == "manual",
        address=address,
        prefix=prefix,
        gateway=fields.get("ipv4.gateway", ""),
        dns=tuple(_split_list(fields.get("ipv4.dns", ""))),
    )
    band = fields.get("802-11-wireless.band", BAND_AUTOMATIC)
    return (
        fields.get("802-11-wireless.ssid", ""),
        settings,
        band if band in BANDS else BAND_AUTOMATIC,
    )


def parse_active_ipv4(output: str) -> ActiveIpv4:
    """Address, gateway and DNS servers of a connected device (`_ACTIVE_IPV4`). The
    servers are those DHCP gave, where it did: with servers of its own the profile shows
    others in IP4.DNS."""
    fields = parse_fields(output)
    address, prefix = _address_and_prefix(fields.get("IP4.ADDRESS[1]", ""))
    options = {}
    for key, value in fields.items():
        if key.startswith("DHCP4.OPTION"):
            name, _equals, option = value.partition("=")
            options[name.strip()] = option.strip()
    dns = options.get("domain_name_servers", "").split()
    if not dns:
        dns = [value for key, value in fields.items() if key.startswith("IP4.DNS[")]
    return ActiveIpv4(address, prefix if address else 0, fields.get("IP4.GATEWAY", ""), tuple(dns))


def parse_permissions(output: str) -> dict[str, str]:
    """Permission and value ("yes", "auth", "no") from `nmcli -t -f PERMISSION,VALUE
    general permissions`."""
    permissions = {}
    for line in output.splitlines():
        fields = split_terse(line)
        if len(fields) == 2:
            permissions[fields[0]] = fields[1]
    return permissions


def permissions_missing(permissions: dict[str, str]) -> bool:
    """Whether NetworkManager would refuse the radio something, or ask for a password."""
    return any(permissions.get(name) != "yes" for name in REQUIRED_PERMISSIONS)


def valid_ssid(text: str) -> bool:
    """A network's name is 1 to 32 bytes long."""
    return 0 < len(text.encode()) <= SSID_MAX_BYTES


def valid_password(text: str) -> bool:
    """A WPA passphrase of 8 to 63 characters, or the key itself as 64 hex digits."""
    size = len(text.encode())
    if size == PSK_HEX_LENGTH:
        return all(char in string.hexdigits for char in text)
    return PASSWORD_LENGTH[0] <= size <= PASSWORD_LENGTH[1]


def signal_bars(signal: int) -> int:
    """The percentage of nmcli as bars 0 to 4, a bar for each started quarter."""
    return min(max((signal + 24) // 25, 0), 4)


def valid_ipv4(text: str) -> bool:
    """Four numbers from 0 to 255 between dots, without leading zeros."""
    try:
        ipaddress.IPv4Address(text)
    except ValueError:
        return False
    return True


def parse_dns_list(text: str) -> tuple[str, ...] | None:
    """The servers of a list separated by commas; None where one is no IPv4 address."""
    servers = tuple(_split_list(text))
    return servers if all(valid_ipv4(server) for server in servers) else None


def valid_dns_list(text: str) -> bool:
    """IPv4 addresses separated by commas, or nothing (the router's servers)."""
    return parse_dns_list(text) is not None and not text.rstrip().endswith(",")


def subnet_mask(prefix: int) -> str:
    """The prefix length as the mask written out: 24 gives "255.255.255.0"."""
    return str(ipaddress.IPv4Network(f"0.0.0.0/{prefix}").netmask)


# The fields `ipv4_problem` names.
ADDRESS = "address"
GATEWAY = "gateway"


def ipv4_problem(settings: Ipv4Settings) -> tuple[str, str] | None:
    """What keeps addresses set by hand from working, beyond what each field shows by
    itself, and in which field (`ADDRESS`, `GATEWAY`); None if nothing does."""
    if not settings.manual or not valid_ipv4(settings.address):
        return None
    interface = ipaddress.IPv4Interface(f"{settings.address}/{settings.prefix}")
    subnet = str(interface.network)
    # /31 and /32 have neither an address of their own nor a broadcast.
    if settings.prefix < 31:
        if interface.ip == interface.network.network_address:
            return ADDRESS, tr("{address} is the address of the subnet {subnet} itself.").format(
                address=settings.address, subnet=subnet
            )
        if interface.ip == interface.network.broadcast_address:
            return ADDRESS, tr("{address} is the broadcast address of the subnet {subnet}.").format(
                address=settings.address, subnet=subnet
            )
    if valid_ipv4(settings.gateway):
        gateway = ipaddress.IPv4Address(settings.gateway)
        if gateway == interface.ip:
            return GATEWAY, tr("The gateway cannot be the radio's own address.")
        if gateway not in interface.network:
            return GATEWAY, tr("The gateway is not in the subnet {subnet}.").format(subnet=subnet)
    return None


def ipv4_arguments(settings: Ipv4Settings) -> list[str]:
    """The properties of `nmcli connection modify` for these settings. Back to DHCP,
    what was set by hand goes: NetworkManager would keep an address as a second one."""
    if settings.manual:
        properties = {
            "ipv4.method": "manual",
            "ipv4.addresses": f"{settings.address}/{settings.prefix}",
            "ipv4.gateway": settings.gateway,
            "ipv4.ignore-auto-dns": "no",
        }
    else:
        properties = {
            "ipv4.method": "auto",
            "ipv4.addresses": "",
            "ipv4.gateway": "",
            # Servers of its own replace the router's rather than coming before them.
            "ipv4.ignore-auto-dns": "yes" if settings.dns else "no",
        }
    properties["ipv4.dns"] = ",".join(settings.dns)
    return [part for pair in properties.items() for part in pair]


def gave_no_address(result: CommandResult) -> bool:
    """Whether a connection failed only for want of an address: the network took the
    password, but no DHCP server answered (NetworkManager's reason 5)."""
    return "ip configuration could not be reserved" in result.error.lower()


def error_text(result: CommandResult) -> str:
    """What went wrong, for the surface."""
    detail = result.error.strip().splitlines()[-1] if result.error.strip() else ""
    lowered = result.error.lower()
    if result.exit_code == NOT_STARTED:
        return tr("nmcli cannot be started.")
    if "not authorized" in lowered or "insufficient privileges" in lowered:
        return tr("No permission: on the Pi the polkit rule for the radio is missing.")
    if "secrets were required" in lowered:
        return tr("Wrong password.")
    if result.exit_code in (EXIT_TIMEOUT, KILLED):
        return tr("Timeout: the network did not answer in time.")
    if result.exit_code == EXIT_NOT_FOUND or "no network with ssid" in lowered:
        return tr("Network not found.")
    if result.exit_code == EXIT_NOT_RUNNING:
        return tr("NetworkManager is not running.")
    if detail:
        return tr("nmcli reports: {detail}").format(detail=detail.removeprefix("Error: "))
    return tr("nmcli failed (code {code}).").format(code=result.exit_code)


class WifiManager(QObject):
    """State and networks of the WLAN, and what can be done with it.

    `watch` reads everything once and keeps the state current in the background from
    then on; `start` reads everything again, scanning, and polls faster while the
    surface shows it, `stop` goes back to the background. Every command waits for the
    one before; `activity` tells what an action in progress does. `changed` follows
    every action and comes twice for a reading: once the state (radio, device, address
    and, when asked for, the profiles) is known, which is quick, and again with the
    networks, which may wait for a scan, and the permissions.
    """

    changed = pyqtSignal()
    failed = pyqtSignal(str)  # what went wrong, for the surface

    def __init__(self, parent: QObject | None = None, runner: CommandRunner | None = None) -> None:
        super().__init__(parent)
        if runner is None:
            program = find_nmcli()
            runner = CommandRunner(program, self) if program else None
        self._runner = runner
        self._queue: deque[tuple[list[str], int, Done]] = deque()
        self._running = False

        self.enabled = False
        self.device: DeviceStatus | None = None
        self.active_ipv4 = ActiveIpv4()
        self.networks: list[Network] = []
        self.saved: list[SavedNetwork] = []
        self.permissions_missing = False
        self.activity = ""  # e.g. "Connecting to “Gast” …"
        # The profiles of new networks that took the password but gave no address by
        # DHCP: kept, so that the address can be set by hand (`apply_ipv4`).
        self.without_dhcp: set[str] = set()

        # One reading at a time; one asked for during it follows it.
        self._reading = False
        self._read_again: dict[str, object] | None = None
        self._rescan = "no"
        self._read_saved = False
        self._read_permissions = False
        self._found_saved: list[SavedNetwork] = []

        self._watching = False  # polling in the background
        self._shown = False  # the sub-page is shown: polling faster
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._poll)

    @property
    def available(self) -> bool:
        """Whether there is an nmcli to ask (not on Windows)."""
        return self._runner is not None

    @property
    def connected_ssid(self) -> str:
        """The name of the network connected to; "" without one."""
        if self.device is None or not self.device.connected:
            return ""
        # The profile before the list: the list may be from before the connection.
        profile = next((s for s in self.saved if s.name == self.device.connection), None)
        if profile is not None:
            return profile.ssid
        in_use = next((n.ssid for n in self.networks if n.in_use), "")
        return in_use or self.device.connection

    @property
    def address(self) -> str:
        """The IPv4 address of the connection up now; "" without one."""
        return self.active_ipv4.address

    @property
    def connected_band(self) -> str:
        """The band of the connection up now, from the last list of networks;
        `BAND_AUTOMATIC` where it is not known."""
        ssid = self.connected_ssid
        network = next((n for n in self.networks if n.ssid == ssid and n.in_use), None)
        return network.band_in_use if network is not None else BAND_AUTOMATIC

    def saved_network(self, ssid: str) -> SavedNetwork | None:
        return next((s for s in self.saved if s.ssid == ssid), None)

    def active_profile(self) -> SavedNetwork | None:
        """The profile of the connection up now."""
        if self.device is None or not self.device.connected:
            return None
        return next((s for s in self.saved if s.name == self.device.connection), None)

    def watch(self) -> None:
        """Reads everything once, so that the sub-page has it when opened, and keeps
        the state current in the background from then on. NetworkManager's last scan
        does: a scan at the start would get in the way of the connection coming up."""
        if not self.available:
            return
        self._watching = True
        self.read(saved=True, permissions=True)
        if not self._shown:
            self._poll_timer.start(BACKGROUND_POLL_INTERVAL_MS)

    def start(self) -> None:
        """Reads everything, scanning if the last scan is old, and then the state every
        few seconds (while the sub-page is shown)."""
        if not self.available:
            return
        self._shown = True
        self.read(rescan="auto", saved=True, permissions=True)
        self._poll_timer.start(POLL_INTERVAL_MS)

    def stop(self) -> None:
        """Back to the background, or to nothing (the sub-page is hidden)."""
        self._shown = False
        if self._watching:
            self._poll_timer.start(BACKGROUND_POLL_INTERVAL_MS)
        else:
            self._poll_timer.stop()

    def shutdown(self) -> None:
        """Ends polling and kills a command still running (on closing the window)."""
        self._watching = False
        self.stop()
        self._queue.clear()
        if self._runner is not None:
            self._runner.stop()

    def scan(self) -> None:
        """Lets NetworkManager look for networks now, and reads them."""
        self.read(rescan="yes")

    # --- Reading ----------------------------------------------------------------------

    def read(self, rescan: str = "no", saved: bool = False, permissions: bool = False) -> None:
        """Reads radio, device, address and networks; with `saved` also the profiles,
        with `permissions` what polkit allows. `rescan` as in `nmcli device wifi list
        --rescan`: "no", "auto" (if the last scan is older than 30 s) or "yes"."""
        if not self.available:
            return
        if self._reading:
            again = self._read_again or {"rescan": "no", "saved": False, "permissions": False}
            if rescan != "no":
                again["rescan"] = rescan
            again["saved"] = again["saved"] or saved
            again["permissions"] = again["permissions"] or permissions
            self._read_again = again
            return
        self._reading = True
        self._rescan = rescan
        self._read_saved = saved
        self._read_permissions = permissions
        self._run(_RADIO, self._on_radio)

    def _poll(self) -> None:
        if not self.activity:
            self.read()

    def _on_radio(self, result: CommandResult) -> None:
        if result.ok:
            self.enabled = result.output.strip() == "enabled"
        self._run(_DEVICE_STATUS, self._on_device_status)

    def _on_device_status(self, result: CommandResult) -> None:
        self.device = parse_device_status(result.output) if result.ok else None
        if self.device is not None and self.device.connected:
            self._run([*_ACTIVE_IPV4, self.device.device], self._on_active_ipv4)
        else:
            self.active_ipv4 = ActiveIpv4()
            self._read_profiles()

    def _on_active_ipv4(self, result: CommandResult) -> None:
        self.active_ipv4 = parse_active_ipv4(result.output) if result.ok else ActiveIpv4()
        self._read_profiles()

    def _read_profiles(self) -> None:
        if self._read_saved:
            self._run(_CONNECTIONS, self._on_connections)
        else:
            self._on_state_read()

    def _on_connections(self, result: CommandResult) -> None:
        self._found_saved = []
        self._read_settings(parse_wifi_connections(result.output) if result.ok else [])

    def _read_settings(self, profiles: list[tuple[str, str]]) -> None:
        """The network and the IPv4 settings of each profile, one after the other:
        `connection show` lists neither."""
        if not profiles:
            self.saved = self._found_saved
            self._on_state_read()
            return
        (uuid, name), rest = profiles[0], profiles[1:]

        def on_settings(result: CommandResult) -> None:
            if result.ok:
                ssid, ipv4, band = parse_profile(result.output)
                if ssid:
                    self._found_saved.append(SavedNetwork(uuid, name, ssid, ipv4, band))
            self._read_settings(rest)

        self._run([*_PROFILE, uuid], on_settings)

    def _on_state_read(self) -> None:
        """Reports the state before the networks: a scan may take seconds, and the
        symbol in the navigation rail and the row of the connection need only the
        state (the name of the network comes from the profile, read before)."""
        # A network without DHCP that has come up since (by hand) is one no more.
        active = self.active_profile()
        if active is not None:
            self.without_dhcp.discard(active.uuid)
        self.without_dhcp &= {profile.uuid for profile in self.saved}
        listing = self.enabled and self.device is not None
        if not listing:
            self.networks = []
        self.changed.emit()
        if not listing:
            self._read_rights()
            return
        self._run([*_NETWORKS, "--rescan", self._rescan], self._on_networks)

    def _on_networks(self, result: CommandResult) -> None:
        if result.ok:
            self.networks = parse_networks(result.output)
        elif self._rescan == "yes":
            # Asked for by a tap; a failed reading in passing stays quiet.
            self.failed.emit(error_text(result))
        self._read_rights()

    def _read_rights(self) -> None:
        if self._read_permissions:
            self._run(_PERMISSIONS, self._on_permissions)
        else:
            self._read_done()

    def _on_permissions(self, result: CommandResult) -> None:
        if result.ok:
            self.permissions_missing = permissions_missing(parse_permissions(result.output))
        self._read_done()

    def _read_done(self) -> None:
        self._reading = False
        self.changed.emit()
        again, self._read_again = self._read_again, None
        if again is not None:
            self.read(**again)

    # --- Actions ----------------------------------------------------------------------

    def set_enabled(self, enabled: bool) -> None:
        activity = tr("Switching Wi-Fi on …") if enabled else tr("Switching Wi-Fi off …")
        self._act(activity, ["radio", "wifi", "on" if enabled else "off"])

    def connect_saved(self, profile: SavedNetwork) -> None:
        self._act(
            tr("Connecting to “{name}” …").format(name=profile.ssid),
            ["-w", str(CONNECT_WAIT_S), "connection", "up", "uuid", profile.uuid],
            CONNECT_TIMEOUT_MS,
        )

    def disconnect(self) -> None:
        if self.device is not None:
            self._act(tr("Disconnecting …"), ["device", "disconnect", self.device.device])

    def forget(self, profile: SavedNetwork) -> None:
        self._act(
            tr("Forgetting “{name}” …").format(name=profile.ssid),
            ["connection", "delete", "uuid", profile.uuid],
        )

    def connect_new(self, ssid: str, password: str = "", hidden: bool = False) -> None:
        """Connects to a network without a profile; NetworkManager makes one and keeps
        it. A hidden network does not give its name, so it is looked for by name."""
        self._begin(tr("Connecting to “{name}” …").format(name=ssid))
        arguments = ["-w", str(CONNECT_NEW_WAIT_S), "device", "wifi", "connect", ssid]
        if password:
            arguments += ["password", password]
        if hidden:
            arguments += ["hidden", "yes"]
        if self.device is not None:
            arguments += ["ifname", self.device.device]

        def connect(before: CommandResult) -> None:
            known = {uuid for uuid, _name in parse_wifi_connections(before.output)}
            self._run(
                arguments,
                lambda result: self._on_connected_new(ssid, known, result),
                CONNECT_NEW_TIMEOUT_MS,
            )

        # The profiles before, to know afterwards which one nmcli made.
        self._run(_CONNECTIONS, connect)

    def _on_connected_new(self, ssid: str, known: set[str], result: CommandResult) -> None:
        """nmcli keeps the profile it made even when the network refused it (a wrong
        password), and NetworkManager would try it again and again: it goes then,
        unless the network only gave no address by DHCP."""
        if result.ok:
            self._end()
            return
        keep = gave_no_address(result)
        if not keep:
            self.failed.emit(error_text(result))

        def settle(listed: CommandResult) -> None:
            made = [uuid for uuid, _name in parse_wifi_connections(listed.output)]
            made = [uuid for uuid in made if uuid not in known]
            if not keep:
                for uuid in made:
                    self._run(["connection", "delete", "uuid", uuid], _ignore)
            else:
                self.without_dhcp.update(made)
                self.failed.emit(
                    tr(
                        "“{name}” gives no address by DHCP. The network stays saved: tap it "
                        "and set the address under “IP settings”."
                    ).format(name=ssid)
                )
            self._after_queue(self._end)

        self._run(_CONNECTIONS, settle)

    def apply_ipv4(self, profile: SavedNetwork, settings: Ipv4Settings) -> None:
        """Writes `settings` into the profile. The connection up now takes them at once:
        `device reapply` keeps the association, `connection up` follows where
        NetworkManager refuses it. A network that gave no address by DHCP is connected
        with them."""
        active = self.active_profile()
        is_active = active is not None and active.uuid == profile.uuid
        retry = profile.uuid in self.without_dhcp
        self.without_dhcp.discard(profile.uuid)
        self._begin(tr("Applying the IP settings of “{name}” …").format(name=profile.ssid))
        connect = ["-w", str(CONNECT_WAIT_S), "connection", "up", "uuid", profile.uuid]

        def reapplied(result: CommandResult) -> None:
            if result.ok:
                self._end()
            else:
                self._run(connect, self._finish, CONNECT_TIMEOUT_MS)

        def modified(result: CommandResult) -> None:
            if not result.ok:
                self._finish(result)
            elif is_active and self.device is not None:
                self._run(["device", "reapply", self.device.device], reapplied)
            elif retry:
                self._run(connect, self._finish, CONNECT_TIMEOUT_MS)
            else:
                self._end()

        modify = ["connection", "modify", "uuid", profile.uuid, *ipv4_arguments(settings)]
        self._run(modify, modified)

    def set_band(self, profile: SavedNetwork, band: str) -> None:
        """Locks the profile to `band`, or lets the Wi-Fi choose (`BAND_AUTOMATIC`).
        NetworkManager takes a band on connecting: the connection up now is made anew
        where it is on another band than the one locked. Where it does not come up there
        (the network not in reach on that band), the profile goes back to the band it had
        and connects so, rather than leaving the radio without Wi-Fi."""
        if band == profile.band:
            return
        active = self.active_profile()
        is_active = active is not None and active.uuid == profile.uuid
        reconnect = is_active and band != BAND_AUTOMATIC and band != self.connected_band
        self._begin(tr("Setting the frequency band of “{name}” …").format(name=profile.ssid))
        connect = ["-w", str(CONNECT_WAIT_S), "connection", "up", "uuid", profile.uuid]

        def modify(value: str) -> list[str]:
            return ["connection", "modify", "uuid", profile.uuid, "802-11-wireless.band", value]

        def connected(result: CommandResult) -> None:
            if result.ok:
                self._end()
                return
            self.failed.emit(
                tr("“{name}” did not connect over {band} and stays at “{before}”.").format(
                    name=profile.ssid, band=band_name(band), before=band_name(profile.band)
                )
            )
            self._run(modify(profile.band), _ignore)
            self._run(connect, lambda _result: self._end(), CONNECT_TIMEOUT_MS)

        def modified(result: CommandResult) -> None:
            if result.ok and reconnect:
                self._run(connect, connected, CONNECT_TIMEOUT_MS)
            else:
                self._finish(result)

        self._run(modify(band), modified)

    def _act(
        self, activity: str, arguments: list[str], timeout_ms: int = COMMAND_TIMEOUT_MS
    ) -> None:
        self._begin(activity)
        self._run(arguments, self._finish, timeout_ms)

    def _finish(self, result: CommandResult) -> None:
        """Ends an action with its last command, reporting what went wrong."""
        if not result.ok:
            self.failed.emit(error_text(result))
        self._end()

    def _begin(self, activity: str) -> None:
        self.activity = activity
        self.changed.emit()

    def _end(self) -> None:
        self.activity = ""
        self.changed.emit()
        self.read(saved=True)

    # --- Queue ------------------------------------------------------------------------

    def _run(self, arguments: list[str], done: Done, timeout_ms: int = COMMAND_TIMEOUT_MS) -> None:
        self._queue.append((arguments, timeout_ms, done))
        self._next()

    def _after_queue(self, then: Callable[[], None]) -> None:
        """Calls `then` once every command queued so far has run."""
        self._queue.append(([], 0, lambda _result: then()))
        self._next()

    def _next(self) -> None:
        if self._running or not self._queue or self._runner is None:
            return
        arguments, timeout_ms, done = self._queue.popleft()
        if not arguments:  # a mark of `_after_queue`
            done(CommandResult(0, ""))
            self._next()
            return
        self._running = True

        def finished(result: CommandResult) -> None:
            self._running = False
            done(result)
            self._next()

        self._runner.run(arguments, timeout_ms, finished)


def _ignore(_result: CommandResult) -> None:
    pass

"""An nmcli that answers from a table, for `net.wifi.WifiManager`, and the answers of a
Raspberry Pi connected to a WLAN, in nmcli's terse format."""

from __future__ import annotations

from collections.abc import Callable

from net.command import CommandResult

RADIO = ("radio", "wifi")
DEVICE_STATUS = ("-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "device", "status")
ACTIVE_IPV4 = ("-t", "-f", "IP4,DHCP4", "device", "show", "wlan0")
LIST = ("-t", "-f", "IN-USE,SIGNAL,FREQ,SECURITY,SSID", "device", "wifi", "list", "--rescan")
CONNECTIONS = ("-t", "-f", "UUID,TYPE,NAME", "connection", "show")
PERMISSIONS = ("-t", "-f", "PERMISSION,VALUE", "general", "permissions")
PROFILE_FIELDS = (
    "802-11-wireless.ssid,802-11-wireless.band,"
    "ipv4.method,ipv4.addresses,ipv4.gateway,ipv4.dns,ipv4.ignore-auto-dns"
)

UUID_HOME = "5f1c0a2e-0d7b-4c51-9b7e-3a3f0f5d1a01"
UUID_HIDDEN = "8d2e4b6a-1c3f-4e5a-8b7c-9d0e1f2a3b02"
UUID_NEW = "c4a1b2c3-d4e5-4f60-8a7b-1c2d3e4f5a03"


def profile_command(uuid: str) -> tuple[str, ...]:
    return ("-t", "-f", PROFILE_FIELDS, "connection", "show", "uuid", uuid)


def profile(
    ssid: str,
    method: str = "auto",
    addresses: str = "",
    gateway: str = "",
    dns: str = "",
    ignore_auto_dns: str = "no",
    band: str = "",
) -> str:
    """A profile as `nmcli -t -f … connection show` prints it: one "key:value" line
    each, the values unescaped (a colon in the name stays as it is)."""
    return (
        f"802-11-wireless.ssid:{ssid}\n"
        f"802-11-wireless.band:{band}\n"
        f"ipv4.method:{method}\n"
        f"ipv4.dns:{dns}\n"
        f"ipv4.addresses:{addresses}\n"
        f"ipv4.gateway:{gateway}\n"
        f"ipv4.ignore-auto-dns:{ignore_auto_dns}\n"
    )


# What a Raspberry Pi shows (shortened): the address and gateway the router gave, and
# its DNS servers both in use and as the DHCP option.
ACTIVE_IPV4_DHCP = (
    "IP4.ADDRESS[1]:192.168.178.42/24\n"
    "IP4.GATEWAY:192.168.178.1\n"
    "IP4.ROUTE[1]:dst = 192.168.178.0/24, nh = 0.0.0.0, mt = 600\n"
    "IP4.DNS[1]:192.168.178.1\n"
    "IP4.DNS[2]:192.168.178.2\n"
    "IP4.DOMAIN[1]:fritz.box\n"
    "DHCP4.OPTION[1]:broadcast_address = 192.168.178.255\n"
    "DHCP4.OPTION[2]:dhcp_client_identifier = 01:2c:cf:67:aa:bb:cc\n"
    "DHCP4.OPTION[3]:domain_name_servers = 192.168.178.1 192.168.178.2\n"
    "DHCP4.OPTION[4]:routers = 192.168.178.1\n"
    "DHCP4.OPTION[5]:subnet_mask = 255.255.255.0\n"
)


# Every access point once: the home network on both bands, connected over 2.4 GHz, a
# neighbour whose name has a colon (escaped), an open one, a company network with 802.1X
# and a hidden one.
WIFI_LIST = (
    " :82:5180 MHz:WPA2:FRITZ!Box 7590 XY\n"
    "*:64:2437 MHz:WPA2:FRITZ!Box 7590 XY\n"
    " :55:2462 MHz:WPA1 WPA2:Nachbar\\:WLAN\n"
    " :40:2412 MHz::Gast\n"
    " :30:5500 MHz:WPA2 802.1X:Firma\n"
    " :20:2437 MHz:WPA2:\n"
)
# The profile of the Raspberry Pi Imager and one of a hidden network, beside a cable.
CONNECTIONS_LIST = (
    f"{UUID_HOME}:802-11-wireless:preconfigured\n"
    f"{UUID_HIDDEN}:802-11-wireless:Keller\n"
    "0b1c2d3e-4f50-4617-8a9b-0c1d2e3f4a05:802-3-ethernet:Wired connection 1\n"
    "lo-0000:loopback:lo\n"
)
PERMISSIONS_DENIED = (
    "org.freedesktop.NetworkManager.enable-disable-wifi:no\n"
    "org.freedesktop.NetworkManager.network-control:auth\n"
    "org.freedesktop.NetworkManager.settings.modify.system:auth\n"
    "org.freedesktop.NetworkManager.wifi.scan:auth\n"
)
PERMISSIONS_GRANTED = PERMISSIONS_DENIED.replace(":no", ":yes").replace(":auth", ":yes")


def connected_answers() -> dict[tuple[str, ...], object]:
    return {
        RADIO: "enabled\n",
        DEVICE_STATUS: (
            "wlan0:wifi:connected:preconfigured\n"
            "p2p-dev-wlan0:wifi-p2p:disconnected:\n"
            "eth0:ethernet:unavailable:\n"
            "lo:loopback:connected (externally):lo\n"
        ),
        ACTIVE_IPV4: ACTIVE_IPV4_DHCP,
        (*LIST, "no"): WIFI_LIST,
        (*LIST, "auto"): WIFI_LIST,
        (*LIST, "yes"): WIFI_LIST,
        CONNECTIONS: CONNECTIONS_LIST,
        profile_command(UUID_HOME): profile("FRITZ!Box 7590 XY"),
        # The cellar's network has its address set by hand.
        profile_command(UUID_HIDDEN): profile(
            "Keller", "manual", "192.168.178.60/24", "192.168.178.1", "192.168.178.1,1.1.1.1"
        ),
        PERMISSIONS: PERMISSIONS_GRANTED,
    }


class FakeNmcli:
    """Answers at once. An answer is the output, a `CommandResult`, a function of the
    arguments, or a list of these taken in turn (the last one stays). Commands without
    an answer succeed without output."""

    def __init__(self, answers: dict[tuple[str, ...], object] | None = None) -> None:
        self.answers = answers if answers is not None else connected_answers()
        self.commands: list[list[str]] = []
        self.held: list[tuple[list[str], Callable[[CommandResult], None]]] = []
        self.holding = False

    def run(self, arguments: list[str], timeout_ms: int, done) -> None:
        self.commands.append(list(arguments))
        if self.holding:
            self.held.append((list(arguments), done))
            return
        done(self.answer(arguments))

    def release(self) -> None:
        """Answers the commands held back, as nmcli would after a while."""
        self.holding = False
        while self.held:
            arguments, done = self.held.pop(0)
            done(self.answer(arguments))

    def answer(self, arguments: list[str]) -> CommandResult:
        answer = self.answers.get(tuple(arguments), "")
        if isinstance(answer, list):
            answer = answer.pop(0) if len(answer) > 1 else answer[0]
        if callable(answer):
            answer = answer(arguments)
        if isinstance(answer, str):
            answer = CommandResult(0, answer)
        return answer

    def stop(self) -> None:
        self.held.clear()

    def commands_starting(self, *words: str) -> list[list[str]]:
        return [c for c in self.commands if c[: len(words)] == list(words)]

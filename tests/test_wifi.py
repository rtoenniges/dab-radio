"""The WLAN through nmcli (`net.wifi`): reading its terse output, and the commands
`WifiManager` gives for what the surface asks."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from net import wifi
from net.command import KILLED, NOT_STARTED, CommandResult
from net.wifi import (
    ADDRESS,
    BACKGROUND_POLL_INTERVAL_MS,
    BAND_2_4_GHZ,
    BAND_5_GHZ,
    BAND_AUTOMATIC,
    GATEWAY,
    POLL_INTERVAL_MS,
    ActiveIpv4,
    Ipv4Settings,
    Network,
    WifiManager,
    band_of,
    error_text,
    gave_no_address,
    ipv4_arguments,
    ipv4_problem,
    parse_active_ipv4,
    parse_device_status,
    parse_dns_list,
    parse_networks,
    parse_permissions,
    parse_profile,
    parse_wifi_connections,
    permissions_missing,
    signal_bars,
    split_terse,
    subnet_mask,
    valid_dns_list,
    valid_ipv4,
    valid_password,
    valid_ssid,
)
from tests.fake_nmcli import (
    ACTIVE_IPV4_DHCP,
    CONNECTIONS,
    CONNECTIONS_LIST,
    DEVICE_STATUS,
    LIST,
    PERMISSIONS,
    PERMISSIONS_DENIED,
    UUID_HIDDEN,
    UUID_HOME,
    UUID_NEW,
    WIFI_LIST,
    FakeNmcli,
    profile,
    profile_command,
)

NO_ADDRESS = (
    "Error: Connection activation failed: (5) IP configuration could not be reserved "
    "(no available address, timeout, etc.).\n"
)


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def test_escaped_colons_and_backslashes_stay_in_their_field():
    assert split_terse("*:55:WPA2:Nachbar\\:WLAN") == ["*", "55", "WPA2", "Nachbar:WLAN"]
    assert split_terse("a\\\\b::c") == ["a\\b", "", "c"]


def test_a_network_is_listed_once_with_its_strongest_access_point():
    networks = parse_networks(WIFI_LIST)

    assert [n.ssid for n in networks] == ["FRITZ!Box 7590 XY", "Nachbar:WLAN", "Gast", "Firma"]
    home, neighbour, guest, company = networks
    # The signal of the access point connected to (64), not of the strongest (82).
    assert home == Network(
        "FRITZ!Box 7590 XY",
        64,
        "WPA2",
        in_use=True,
        bands=frozenset({BAND_2_4_GHZ, BAND_5_GHZ}),
        band_in_use=BAND_2_4_GHZ,
    )
    assert neighbour.bands == {BAND_2_4_GHZ} and company.bands == {BAND_5_GHZ}
    assert neighbour.needs_password and not neighbour.in_use
    assert not guest.needs_password
    assert company.enterprise and not company.needs_password


def test_the_access_point_in_use_gives_its_network_its_signal_even_when_weaker():
    # As on the Pi on 2026-10-11: connected over 5 GHz, the 2.4 GHz of the same access
    # point stronger; the bars are those of the connection.
    networks = parse_networks(" :70:2412 MHz:WPA2:Heim\n*:40:5180 MHz:WPA2:Heim\n")
    assert networks == [
        Network("Heim", 40, "WPA2", True, frozenset({BAND_2_4_GHZ, BAND_5_GHZ}), BAND_5_GHZ)
    ]


def test_the_profiles_of_the_wlan_are_told_from_the_others():
    assert parse_wifi_connections(CONNECTIONS_LIST) == [
        (UUID_HOME, "preconfigured"),
        (UUID_HIDDEN, "Keller"),
    ]


def test_the_wlan_device_is_found_and_wifi_direct_is_not_it():
    status = parse_device_status(
        "p2p-dev-wlan0:wifi-p2p:disconnected:\nwlan0:wifi:connecting (configuring):Heim\n"
    )
    assert status.device == "wlan0"
    assert status.connecting and not status.connected
    assert status.connection == "Heim"
    assert parse_device_status("eth0:ethernet:connected:Kabel\n") is None


def test_the_connection_up_shows_its_address_gateway_and_the_routers_dns():
    active = parse_active_ipv4(ACTIVE_IPV4_DHCP)
    assert active == ActiveIpv4(
        "192.168.178.42", 24, "192.168.178.1", ("192.168.178.1", "192.168.178.2")
    )
    assert parse_active_ipv4("") == ActiveIpv4()


def test_the_dns_servers_in_use_count_where_dhcp_gave_none():
    output = "IP4.ADDRESS[1]:10.0.0.5/8\nIP4.GATEWAY:\nIP4.DNS[1]:9.9.9.9\nIP4.DNS[2]:1.1.1.1\n"
    assert parse_active_ipv4(output) == ActiveIpv4("10.0.0.5", 8, "", ("9.9.9.9", "1.1.1.1"))


def test_a_profile_gives_its_network_and_ipv4_settings():
    assert parse_profile(profile("Nachbar:WLAN")) == (
        "Nachbar:WLAN",
        Ipv4Settings(),
        BAND_AUTOMATIC,
    )
    ssid, manual, _band = parse_profile(
        profile("Keller", "manual", "192.168.178.60/24, 10.0.0.2/8", "192.168.178.1", "1.1.1.1")
    )
    assert ssid == "Keller"
    assert manual == Ipv4Settings(True, "192.168.178.60", 24, "192.168.178.1", ("1.1.1.1",))
    # DNS servers of its own with DHCP.
    _ssid, dhcp, _band = parse_profile(
        profile("Heim", dns="9.9.9.9,1.1.1.1", ignore_auto_dns="yes")
    )
    assert dhcp == Ipv4Settings(dns=("9.9.9.9", "1.1.1.1"))
    # Locked to a band.
    assert parse_profile(profile("Heim", band="a"))[2] == BAND_5_GHZ
    assert parse_profile(profile("Heim", band="bg"))[2] == BAND_2_4_GHZ


def test_addresses_and_lists_of_dns_servers_are_checked():
    assert valid_ipv4("192.168.2.20") and valid_ipv4("0.0.0.0")
    assert not valid_ipv4("192.168.2") and not valid_ipv4("192.168.2.256")
    assert not valid_ipv4("192.168.02.1") and not valid_ipv4("")
    assert parse_dns_list("192.168.2.1, 1.1.1.1") == ("192.168.2.1", "1.1.1.1")
    assert parse_dns_list("9.9.9.9,1.1.1.1") == ("9.9.9.9", "1.1.1.1")
    assert parse_dns_list("") == ()
    assert parse_dns_list("192.168.2.1, 1.1") is None
    assert valid_dns_list("") and valid_dns_list("1.1.1.1, 9.9.9.9")
    assert not valid_dns_list("1.1.1.1, ")
    assert subnet_mask(24) == "255.255.255.0" and subnet_mask(20) == "255.255.240.0"


@pytest.mark.parametrize(
    ("settings", "problem"),
    [
        (Ipv4Settings(True, "192.168.2.20", 24, "192.168.2.1"), None),
        (Ipv4Settings(True, "192.168.2.20", 24, ""), None),
        (Ipv4Settings(True, "", 24, "192.168.1.1"), None),  # not complete yet
        (Ipv4Settings(False, "192.168.2.0", 24, "10.0.0.1"), None),  # DHCP
        (
            Ipv4Settings(True, "192.168.2.20", 24, "192.168.1.1"),
            (GATEWAY, "The gateway is not in the subnet 192.168.2.0/24."),
        ),
        (
            Ipv4Settings(True, "192.168.2.20", 24, "192.168.2.20"),
            (GATEWAY, "The gateway cannot be the radio's own address."),
        ),
        (
            Ipv4Settings(True, "192.168.2.0", 24, "192.168.2.1"),
            (ADDRESS, "192.168.2.0 is the address of the subnet 192.168.2.0/24 itself."),
        ),
        (
            Ipv4Settings(True, "192.168.2.255", 24, ""),
            (ADDRESS, "192.168.2.255 is the broadcast address of the subnet 192.168.2.0/24."),
        ),
        # A wider subnet takes the gateway in.
        (Ipv4Settings(True, "192.168.2.20", 16, "192.168.1.1"), None),
    ],
)
def test_addresses_set_by_hand_must_fit_together(settings, problem):
    assert ipv4_problem(settings) == problem


def test_settings_become_properties_of_the_profile():
    manual = Ipv4Settings(True, "192.168.2.20", 24, "192.168.2.1", ("192.168.2.1", "1.1.1.1"))
    assert ipv4_arguments(manual) == [
        *("ipv4.method", "manual", "ipv4.addresses", "192.168.2.20/24"),
        *("ipv4.gateway", "192.168.2.1", "ipv4.ignore-auto-dns", "no"),
        *("ipv4.dns", "192.168.2.1,1.1.1.1"),
    ]
    # Back to DHCP, what was set by hand goes; servers of its own replace the router's.
    assert ipv4_arguments(Ipv4Settings(dns=("9.9.9.9",))) == [
        *("ipv4.method", "auto", "ipv4.addresses", "", "ipv4.gateway", ""),
        *("ipv4.ignore-auto-dns", "yes", "ipv4.dns", "9.9.9.9"),
    ]
    assert ipv4_arguments(Ipv4Settings())[-4:] == ["ipv4.ignore-auto-dns", "no", "ipv4.dns", ""]


def test_a_network_that_gives_no_address_is_told_from_other_failures():
    assert gave_no_address(CommandResult(4, "", NO_ADDRESS))
    refusal = "Error: Connection activation failed: (7) Secrets were required, but not provided.\n"
    assert not gave_no_address(CommandResult(4, "", refusal))


def test_anything_but_yes_counts_as_missing_permission():
    assert permissions_missing(parse_permissions(PERMISSIONS_DENIED))
    granted = PERMISSIONS_DENIED.replace(":no", ":yes").replace(":auth", ":yes")
    assert not permissions_missing(parse_permissions(granted))


def test_names_and_passwords_are_checked_as_wpa_wants_them():
    assert valid_ssid("Gast") and not valid_ssid("")
    assert valid_ssid("ü" * 16) and not valid_ssid("ü" * 17)  # 32 bytes, 34 bytes
    assert not valid_password("1234567")
    assert valid_password("12345678") and valid_password("x" * 63)
    assert not valid_password("x" * 65)
    assert valid_password("0123456789abcdef" * 4)  # the key itself
    assert not valid_password("x" * 64)


def test_bars_count_started_quarters():
    assert [signal_bars(s) for s in (0, 1, 25, 26, 50, 75, 76, 100)] == [0, 1, 1, 2, 2, 3, 4, 4]


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (
            CommandResult(
                4,
                "",
                "Error: Connection activation failed: (7) Secrets were required, but not provided.\n",
            ),
            "Wrong password.",
        ),
        (
            CommandResult(
                1,
                "",
                "Error: Failed to add/activate new connection: Not authorized to control networking.\n",
            ),
            "No permission: on the Pi the polkit rule for the radio is missing.",
        ),
        (
            CommandResult(10, "", "Error: No network with SSID 'Gast' found.\n"),
            "Network not found.",
        ),
        (
            CommandResult(3, "", "Error: Timeout expired (30 seconds)\n"),
            "Timeout: the network did not answer in time.",
        ),
        (
            CommandResult(KILLED, ""),
            "Timeout: the network did not answer in time.",
        ),
        (CommandResult(NOT_STARTED, ""), "nmcli cannot be started."),
        (
            CommandResult(8, "", "Error: NetworkManager is not running.\n"),
            "NetworkManager is not running.",
        ),
        (CommandResult(1, "", "Error: something new.\n"), "nmcli reports: something new."),
    ],
)
def test_errors_of_nmcli_are_explained_to_the_user(result, message):
    assert error_text(result) == message


@pytest.fixture
def nmcli() -> FakeNmcli:
    return FakeNmcli()


@pytest.fixture
def manager(nmcli) -> WifiManager:
    manager = WifiManager(runner=nmcli)
    yield manager
    manager.shutdown()


def test_starting_reads_state_networks_profiles_and_permissions(manager, nmcli):
    changes = []
    manager.changed.connect(lambda: changes.append((manager.connected_ssid, len(manager.networks))))

    manager.start()

    assert manager.enabled
    assert manager.device.device == "wlan0" and manager.device.connected
    assert manager.address == "192.168.178.42"
    assert [n.ssid for n in manager.networks][:2] == ["FRITZ!Box 7590 XY", "Nachbar:WLAN"]
    # The imager names its profile "preconfigured": the network comes from the profile.
    assert [(s.name, s.ssid) for s in manager.saved] == [
        ("preconfigured", "FRITZ!Box 7590 XY"),
        ("Keller", "Keller"),
    ]
    assert manager.connected_ssid == "FRITZ!Box 7590 XY"
    assert not manager.permissions_missing
    # The last scan is used unless it is old.
    assert [
        "-t",
        "-f",
        "IN-USE,SIGNAL,FREQ,SECURITY,SSID",
        "device",
        "wifi",
        "list",
        "--rescan",
        "auto",
    ] in nmcli.commands
    # Two reports: the state with the profiles (which name the network) before the
    # networks, which may wait for the scan; the profiles are read before the list.
    assert changes == [("FRITZ!Box 7590 XY", 0), ("FRITZ!Box 7590 XY", 4)]
    assert nmcli.commands.index(list(CONNECTIONS)) < nmcli.commands.index([*LIST, "auto"])


def test_watching_reads_everything_once_and_polls_slowly_until_the_page_is_shown(manager, nmcli):
    manager.watch()

    assert manager.connected_ssid == "FRITZ!Box 7590 XY"
    assert [s.ssid for s in manager.saved] == ["FRITZ!Box 7590 XY", "Keller"]
    assert not manager.permissions_missing
    # NetworkManager's last scan does: a scan at the start would get in the way of the
    # connection coming up.
    assert [c[-1] for c in nmcli.commands_starting(*LIST)] == ["no"]
    assert manager._poll_timer.isActive()
    assert manager._poll_timer.interval() == BACKGROUND_POLL_INTERVAL_MS

    manager.start()
    assert manager._poll_timer.interval() == POLL_INTERVAL_MS
    assert [c[-1] for c in nmcli.commands_starting(*LIST)] == ["no", "auto"]
    manager.stop()
    assert manager._poll_timer.isActive()
    assert manager._poll_timer.interval() == BACKGROUND_POLL_INTERVAL_MS
    manager.shutdown()
    assert not manager._poll_timer.isActive()


def test_a_reading_in_the_background_notices_the_wlan_switched_off_elsewhere(manager, nmcli):
    manager.watch()
    changes = []
    manager.changed.connect(lambda: changes.append(manager.enabled))

    # Off, NetworkManager reports the device as unavailable.
    nmcli.answers[tuple(wifi._RADIO)] = "disabled\n"
    nmcli.answers[DEVICE_STATUS] = "wlan0:wifi:unavailable:\n"
    manager._poll()

    assert changes == [False, False]
    assert manager.networks == []
    assert manager.connected_ssid == ""
    # Nothing to list while off.
    assert [c[-1] for c in nmcli.commands_starting(*LIST)] == ["no"]


def test_the_name_of_the_network_comes_from_the_profile_before_the_list(manager, nmcli):
    manager.watch()
    # Another network since the list was read: the list still marks the old one.
    nmcli.answers[DEVICE_STATUS] = "wlan0:wifi:connected:Keller\n"
    changes = []
    manager.changed.connect(lambda: changes.append(manager.connected_ssid))

    manager.read()

    assert changes == ["Keller", "Keller"]


def test_missing_permissions_are_noticed(manager, nmcli):
    nmcli.answers[PERMISSIONS] = PERMISSIONS_DENIED
    manager.start()
    assert manager.permissions_missing


def test_without_nmcli_nothing_is_run(monkeypatch):
    monkeypatch.setattr(wifi, "find_nmcli", lambda: None)
    manager = WifiManager()
    manager.start()
    assert not manager.available
    assert manager.device is None


def test_a_reading_asked_for_during_one_follows_it(manager, nmcli):
    nmcli.holding = True
    manager.read()
    manager.read(rescan="yes", saved=True)
    manager.read()
    assert nmcli.commands == [["radio", "wifi"]]

    nmcli.release()

    assert len(nmcli.commands_starting("radio", "wifi")) == 2
    lists = [c[-1] for c in nmcli.commands_starting(*LIST[:3])]
    assert lists == ["no", "yes"]
    assert len(nmcli.commands_starting(*CONNECTIONS)) == 1


def test_polling_waits_while_an_action_runs(manager, nmcli):
    manager.start()
    nmcli.holding = True
    manager.connect_saved(manager.saved[1])
    count = len(nmcli.commands)

    manager._poll()

    assert len(nmcli.commands) == count
    assert manager.activity == "Connecting to “Keller” …"


def test_a_saved_network_is_connected_by_its_profile(manager, nmcli):
    manager.start()
    manager.connect_saved(manager.saved[1])
    assert ["-w", "30", "connection", "up", "uuid", UUID_HIDDEN] in nmcli.commands
    assert manager.activity == ""


def test_a_new_network_is_connected_with_its_password_on_the_device(manager, nmcli):
    manager.start()
    manager.connect_new("Nachbar:WLAN", "geheim123")
    assert [
        "-w",
        "60",
        "device",
        "wifi",
        "connect",
        "Nachbar:WLAN",
        "password",
        "geheim123",
        "ifname",
        "wlan0",
    ] in nmcli.commands


def test_a_hidden_network_is_looked_for_by_name(manager, nmcli):
    manager.start()
    manager.connect_new("Keller 2", hidden=True)
    connect = nmcli.commands_starting("-w", "60", "device", "wifi", "connect")
    assert connect == [
        ["-w", "60", "device", "wifi", "connect", "Keller 2", "hidden", "yes", "ifname", "wlan0"]
    ]


def _new_network(nmcli: FakeNmcli, password: str, answer: CommandResult | str) -> None:
    """nmcli makes a profile for "Nachbar:WLAN" and answers the connection so."""
    after = CONNECTIONS_LIST + f"{UUID_NEW}:802-11-wireless:Nachbar\\:WLAN\n"
    nmcli.answers[CONNECTIONS] = [CONNECTIONS_LIST, after, after]
    nmcli.answers[profile_command(UUID_NEW)] = profile("Nachbar:WLAN")
    connect = ("-w", "60", "device", "wifi", "connect", "Nachbar:WLAN", "password", password)
    nmcli.answers[(*connect, "ifname", "wlan0")] = answer


def test_a_new_network_keeps_the_profile_nmcli_made(manager, nmcli):
    manager.start()
    _new_network(nmcli, "geheim123", "Device 'wlan0' successfully activated.\n")

    manager.connect_new("Nachbar:WLAN", "geheim123")

    assert nmcli.commands_starting("connection", "modify") == []
    assert nmcli.commands_starting("connection", "delete") == []
    assert manager.activity == ""


def test_a_refused_new_network_leaves_no_profile_behind(manager, nmcli):
    manager.start()
    refusal = "Error: Connection activation failed: (7) Secrets were required, but not provided.\n"
    _new_network(nmcli, "falsch123", CommandResult(4, "", refusal))
    nmcli.answers[CONNECTIONS][-1] = CONNECTIONS_LIST  # deleted again
    failures = []
    manager.failed.connect(failures.append)

    manager.connect_new("Nachbar:WLAN", "falsch123")

    assert failures == ["Wrong password."]
    assert nmcli.commands_starting("connection", "delete") == [
        ["connection", "delete", "uuid", UUID_NEW]
    ]
    assert manager.activity == ""
    assert [s.uuid for s in manager.saved] == [UUID_HOME, UUID_HIDDEN]


def test_a_new_network_without_dhcp_stays_saved_for_an_address_by_hand(manager, nmcli):
    manager.start()
    _new_network(nmcli, "geheim123", CommandResult(4, "", NO_ADDRESS))
    failures = []
    manager.failed.connect(failures.append)

    manager.connect_new("Nachbar:WLAN", "geheim123")

    assert nmcli.commands_starting("connection", "delete") == []
    assert failures == [
        (
            "“Nachbar:WLAN” gives no address by DHCP. The network stays saved: tap it and "
            "set the address under “IP settings”."
        )
    ]
    assert manager.without_dhcp == {UUID_NEW}
    assert manager.activity == ""


def test_ip_settings_are_written_and_the_connection_up_takes_them_at_once(manager, nmcli):
    manager.start()
    home = manager.saved[0]
    manual = Ipv4Settings(True, "192.168.178.20", 24, "192.168.178.1", ("192.168.178.1",))

    manager.apply_ipv4(home, manual)

    modify = ["connection", "modify", "uuid", UUID_HOME, *ipv4_arguments(manual)]
    assert modify in nmcli.commands
    assert nmcli.commands.index(["device", "reapply", "wlan0"]) > nmcli.commands.index(modify)
    assert nmcli.commands_starting("-w", "30", "connection", "up") == []
    assert manager.activity == ""


def test_the_connection_is_made_anew_where_networkmanager_will_not_reapply(manager, nmcli):
    manager.start()
    refused = CommandResult(1, "", "Error: Reapplying connection failed: Can't reapply.\n")
    nmcli.answers[("device", "reapply", "wlan0")] = refused
    failures = []
    manager.failed.connect(failures.append)

    manager.apply_ipv4(manager.saved[0], Ipv4Settings(dns=("9.9.9.9",)))

    assert ["-w", "30", "connection", "up", "uuid", UUID_HOME] in nmcli.commands
    assert failures == []


def test_settings_of_a_network_not_up_wait_for_its_next_connection(manager, nmcli):
    manager.start()
    manager.apply_ipv4(manager.saved[1], Ipv4Settings())
    assert nmcli.commands_starting("connection", "modify", "uuid", UUID_HIDDEN)
    assert nmcli.commands_starting("device", "reapply") == []
    assert nmcli.commands_starting("-w", "30", "connection", "up") == []


def test_a_network_without_dhcp_is_connected_with_the_address_set_by_hand(manager, nmcli):
    manager.start()
    manager.without_dhcp.add(UUID_HIDDEN)
    manager.apply_ipv4(manager.saved[1], Ipv4Settings(True, "10.0.0.5", 8))
    assert ["-w", "30", "connection", "up", "uuid", UUID_HIDDEN] in nmcli.commands
    assert manager.without_dhcp == set()


def test_settings_that_cannot_be_written_are_reported(manager, nmcli):
    manager.start()
    denied = CommandResult(1, "", "Error: Failed to modify connection: Not authorized.\n")
    modify = ("connection", "modify", "uuid", UUID_HOME, *ipv4_arguments(Ipv4Settings()))
    nmcli.answers[modify] = denied
    failures = []
    manager.failed.connect(failures.append)

    manager.apply_ipv4(manager.saved[0], Ipv4Settings())

    assert failures == ["No permission: on the Pi the polkit rule for the radio is missing."]
    assert nmcli.commands_starting("device", "reapply") == []


def test_forgetting_and_switching_off(manager, nmcli):
    manager.start()
    manager.forget(manager.saved[0])
    manager.set_enabled(False)
    manager.disconnect()
    assert ["connection", "delete", "uuid", UUID_HOME] in nmcli.commands
    assert ["radio", "wifi", "off"] in nmcli.commands
    assert ["device", "disconnect", "wlan0"] in nmcli.commands


def test_a_scan_that_fails_is_reported_one_in_passing_is_not(manager, nmcli):
    denied = CommandResult(1, "", "Error: Scanning not allowed: not authorized.\n")
    nmcli.answers[(*LIST, "yes")] = denied
    nmcli.answers[(*LIST, "no")] = denied
    failures = []
    manager.failed.connect(failures.append)

    manager.read()
    assert failures == []
    manager.scan()
    assert failures == ["No permission: on the Pi the polkit rule for the radio is missing."]


# --- The frequency band -----------------------------------------------------------------


def test_frequencies_fall_into_their_band():
    assert band_of("2412 MHz") == band_of("2484 MHz") == BAND_2_4_GHZ
    assert band_of("5180 MHz") == band_of("5825 MHz") == BAND_5_GHZ
    # 6 GHz NetworkManager cannot lock; nothing readable is no band either.
    assert band_of("5955 MHz") == band_of("") == band_of("--") == BAND_AUTOMATIC


def test_the_profiles_and_the_connection_tell_their_band(manager, nmcli):
    nmcli.answers[profile_command(UUID_HIDDEN)] = profile("Keller", band="a")
    manager.start()
    assert [s.band for s in manager.saved] == [BAND_AUTOMATIC, BAND_5_GHZ]
    assert manager.connected_band == BAND_2_4_GHZ


def _band_modify(uuid: str, band: str) -> list[str]:
    return ["connection", "modify", "uuid", uuid, "802-11-wireless.band", band]


def test_locking_the_connection_up_to_another_band_connects_anew(manager, nmcli):
    manager.start()

    manager.set_band(manager.saved[0], BAND_5_GHZ)

    modify = _band_modify(UUID_HOME, "a")
    up = ["-w", "30", "connection", "up", "uuid", UUID_HOME]
    assert nmcli.commands.index(up) > nmcli.commands.index(modify)
    assert manager.activity == ""


def test_a_band_the_connection_is_on_or_automatic_needs_no_new_connection(manager, nmcli):
    nmcli.answers[profile_command(UUID_HOME)] = profile("FRITZ!Box 7590 XY", band="a")
    manager.start()

    manager.set_band(manager.saved[0], BAND_2_4_GHZ)  # it is on 2.4 GHz already
    manager.set_band(manager.saved[0], BAND_AUTOMATIC)  # the Wi-Fi chooses next time
    manager.set_band(manager.saved[1], BAND_5_GHZ)  # not up: for its next connection

    assert _band_modify(UUID_HOME, "bg") in nmcli.commands
    assert _band_modify(UUID_HOME, "") in nmcli.commands
    assert _band_modify(UUID_HIDDEN, "a") in nmcli.commands
    assert nmcli.commands_starting("-w", "30", "connection", "up") == []


def test_the_band_a_profile_has_is_not_written_again(manager, nmcli):
    manager.start()
    manager.set_band(manager.saved[0], BAND_AUTOMATIC)
    assert nmcli.commands_starting("connection", "modify") == []


def test_a_band_the_network_does_not_connect_over_is_taken_back(manager, nmcli):
    manager.start()
    up = ("-w", "30", "connection", "up", "uuid", UUID_HOME)
    not_found = CommandResult(
        4, "", "Error: Connection activation failed: The Wi-Fi network could not be found.\n"
    )
    nmcli.answers[up] = [not_found, ""]
    failures = []
    manager.failed.connect(failures.append)

    manager.set_band(manager.saved[0], BAND_5_GHZ)

    # Locked, tried, unlocked again and connected as before.
    assert nmcli.commands_starting("connection", "modify") == [
        _band_modify(UUID_HOME, "a"),
        _band_modify(UUID_HOME, ""),
    ]
    assert nmcli.commands_starting(*up) == [list(up), list(up)]
    assert failures == ["“FRITZ!Box 7590 XY” did not connect over 5 GHz and stays at “Automatic”."]
    assert manager.activity == ""

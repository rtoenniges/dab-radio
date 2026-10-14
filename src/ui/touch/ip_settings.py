"""The dialog "IP settings" of a saved Wi-Fi network (opened from `ui.touch.wifi_view`):
whether the network gives the radio its IPv4 address by DHCP or the address is set here,
and which DNS servers the radio asks, in either case.

With DHCP the dialog shows what the connection up now got from the router; switching
to "Manual" starts from that. Addresses come in over the number pad
(`NumberPad.open_address`), the prefix of the subnet steps. Nothing is written before
"Apply"; then `WifiManager.apply_ipv4` takes over."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from i18n import tr, tr_noop
from net.wifi import (
    ADDRESS,
    DEFAULT_PREFIX,
    DNS_SEPARATOR,
    GATEWAY,
    PREFIXES,
    ActiveIpv4,
    Ipv4Settings,
    SavedNetwork,
    ipv4_problem,
    parse_dns_list,
    subnet_mask,
    valid_dns_list,
    valid_ipv4,
)
from ui.touch.number_pad import NumberPad
from ui.touch.overlay import Overlay
from ui.touch.theme import TOUCH_TARGET
from ui.touch.widgets import ChoiceButtons, ElidedLabel, Stepper, repolish, set_role

PANEL_WIDTH = 860
LABEL_WIDTH = 160
# What the connection got, where there is none.
NOTHING = "–"
# Translators: beneath the DNS servers in the dialog "IP settings", and on the number
# pad while they are typed.
DNS_HINT = tr_noop("Several separated by commas")
NOT_SET = tr_noop("Not set")


def _prefix_text(prefix: object) -> str:
    return f"/{prefix}"


class IpSettingsDialog(Overlay):
    """Rows "Address", "Subnet", "Gateway" and "DNS" beneath the choice between DHCP
    and "Manual". With DHCP the first three show what the connection got, read only;
    by hand they are buttons for the number pad and a stepper. "Apply" waits for a
    change that works: a problem across the fields (`net.wifi.ipv4_problem`) shows in
    red beneath them and marks its field."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._profile: SavedNetwork | None = None
        self._active = ActiveIpv4()
        self._saved = Ipv4Settings()
        self._on_apply: Callable[[Ipv4Settings], None] | None = None
        self._number_pad: NumberPad | None = None
        self._manual = False
        self._address = ""
        self._prefix = DEFAULT_PREFIX
        self._gateway = ""
        self._dns: tuple[str, ...] = ()
        # The servers of the router, filled in on switching to "Manual": they go again
        # on switching back, unless they were changed.
        self._dns_from_router = False

        self.panel.setFixedWidth(PANEL_WIDTH)
        title = set_role(QLabel(tr("IP settings")), "large")
        self._network = set_role(ElidedLabel(), "small")
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(title)
        titles.addWidget(self._network)
        self._mode = ChoiceButtons([(False, tr("Automatic (DHCP)")), (True, tr("Manual"))])
        self._mode.chosen.connect(self._on_mode_chosen)
        header = QHBoxLayout()
        header.setSpacing(16)
        header.addLayout(titles, stretch=1)
        header.addWidget(self._mode)

        self._address_value = set_role(QLabel(), "large")
        self._address_button = set_role(QPushButton(), "value")
        self._address_button.clicked.connect(self._edit_address)
        self._subnet_value = set_role(QLabel(), "large")
        self._prefix_stepper = Stepper(list(PREFIXES), _prefix_text, value_width=TOUCH_TARGET)
        self._prefix_stepper.value_changed.connect(self._on_prefix_changed)
        self._mask = set_role(QLabel(), "muted")
        self._gateway_value = set_role(QLabel(), "large")
        self._gateway_button = set_role(QPushButton(), "value")
        self._gateway_button.clicked.connect(self._edit_gateway)
        self._dns_button = set_role(QPushButton(), "value")
        self._dns_button.clicked.connect(self._edit_dns)
        dns_hint = set_role(QLabel(tr(DNS_HINT)), "small")

        rows = (
            (ADDRESS, tr("Address"), (self._address_value, self._address_button)),
            ("subnet", tr("Subnet"), (self._subnet_value, self._prefix_stepper, self._mask)),
            (GATEWAY, tr("Gateway"), (self._gateway_value, self._gateway_button)),
            ("dns", "DNS", (self._dns_button,)),
        )
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(12)
        labels: dict[str, QLabel] = {}
        for row, (field, name, widgets) in enumerate(rows):
            labels[field] = set_role(QLabel(name), "muted")
            labels[field].setFixedWidth(LABEL_WIDTH)
            grid.addWidget(labels[field], row, 0)
            content = QHBoxLayout()
            content.setSpacing(12)
            for widget in widgets:
                content.addWidget(widget)
            content.addStretch()
            grid.addLayout(content, row, 1)
            # The same height with DHCP (text) and by hand (buttons).
            grid.setRowMinimumHeight(row, TOUCH_TARGET)
        grid.addWidget(dns_hint, len(rows), 1)
        # The fields a problem across them may lie in (`ipv4_problem`).
        self._fields = {
            ADDRESS: (labels[ADDRESS], self._address_button),
            GATEWAY: (labels[GATEWAY], self._gateway_button),
        }

        self._error = set_role(QLabel(), "error")
        self._error.setWordWrap(True)
        cancel = QPushButton(tr("Cancel"))
        cancel.clicked.connect(self.dismiss)
        self._apply = set_role(QPushButton(tr("Apply")), "accent")
        self._apply.clicked.connect(self._on_apply_clicked)
        # The error between the buttons: there is room, and the dialog keeps its size.
        buttons = QHBoxLayout()
        buttons.setSpacing(16)
        buttons.addWidget(cancel)
        buttons.addWidget(self._error, stretch=1)
        buttons.addWidget(self._apply)

        layout = QVBoxLayout(self.panel)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)
        layout.addLayout(header)
        layout.addLayout(grid)
        layout.addLayout(buttons)

    def number_pad(self) -> NumberPad:
        """The pad over the whole window, over this dialog; made on first use."""
        if self._number_pad is None:
            self._number_pad = NumberPad(self.parentWidget())
        return self._number_pad

    def open(
        self,
        profile: SavedNetwork,
        active: ActiveIpv4 | None,
        on_apply: Callable[[Ipv4Settings], None],
    ) -> None:
        """Shows the settings of `profile`; `active` is what its connection got, None
        where it is not the one up now. `on_apply` gets the settings taken over."""
        self._profile = profile
        self._active = active or ActiveIpv4()
        self._on_apply = on_apply
        saved = profile.ipv4
        self._saved = saved.comparable()
        self._manual = saved.manual
        if saved.manual:
            self._address, self._prefix, self._gateway = saved.address, saved.prefix, saved.gateway
        else:
            self._address = self._active.address
            self._prefix = self._active.prefix or DEFAULT_PREFIX
            self._gateway = self._active.gateway
        # As the stepper can show it.
        self._prefix = min(max(self._prefix, PREFIXES[0]), PREFIXES[-1])
        self._dns = saved.dns
        self._dns_from_router = False
        self._network.setText(profile.ssid)
        self._mode.set_current(self._manual)
        self._show()
        self.show_over()

    def dismiss(self) -> None:
        self._on_apply = None
        super().dismiss()

    def settings(self) -> Ipv4Settings:
        """What the dialog shows now."""
        if not self._manual:
            return Ipv4Settings(dns=self._dns)
        return Ipv4Settings(True, self._address, self._prefix, self._gateway, self._dns)

    def problem(self) -> tuple[str, str] | None:
        return ipv4_problem(self.settings())

    def can_apply(self) -> bool:
        settings = self.settings()
        complete = not settings.manual or valid_ipv4(settings.address)
        return complete and self.problem() is None and settings != self._saved

    # --- Showing ------------------------------------------------------------------

    def _show(self) -> None:
        manual = self._manual
        active = self._active
        for value in (self._address_value, self._subnet_value, self._gateway_value):
            value.setVisible(not manual)
        for editor in (self._address_button, self._prefix_stepper, self._gateway_button):
            editor.setVisible(manual)

        self._address_value.setText(active.address or NOTHING)
        self._subnet_value.setText(_prefix_text(active.prefix) if active.address else NOTHING)
        self._gateway_value.setText(active.gateway or NOTHING)
        self._set_value(self._address_button, self._address, tr(NOT_SET))
        self._prefix_stepper.set_value(self._prefix)
        self._set_value(self._gateway_button, self._gateway, tr("No gateway"))
        prefix = self._prefix if manual else active.prefix if active.address else 0
        self._mask.setText(subnet_mask(prefix) if prefix else "")
        self._mask.setVisible(bool(prefix))

        if manual:
            without = tr("No DNS server")
        elif active.dns:
            without = tr("From the router: {servers}").format(
                servers=DNS_SEPARATOR.join(active.dns)
            )
        else:
            without = tr("From the router")
        self._set_value(self._dns_button, DNS_SEPARATOR.join(self._dns), without)

        problem = self.problem()
        for field, widgets in self._fields.items():
            invalid = problem is not None and problem[0] == field
            for widget in widgets:
                if bool(widget.property("invalid")) != invalid:
                    widget.setProperty("invalid", invalid)
                    repolish(widget)
        self._error.setText(problem[1] if problem else "")
        self._apply.setEnabled(self.can_apply())

    @staticmethod
    def _set_value(button: QPushButton, text: str, placeholder: str) -> None:
        """The value, or in grey what holds without one."""
        button.setText(text or placeholder)
        if bool(button.property("placeholder")) != (not text):
            button.setProperty("placeholder", not text)
            repolish(button)

    # --- Changing -----------------------------------------------------------------

    def _on_mode_chosen(self, manual: object) -> None:
        self._manual = bool(manual)
        # By hand without DNS servers no name resolves: the router's are a start.
        if self._manual and not self._dns and self._active.dns:
            self._dns, self._dns_from_router = self._active.dns, True
        elif not self._manual and self._dns_from_router:
            self._dns, self._dns_from_router = (), False
        self._show()

    def _on_prefix_changed(self, prefix: object) -> None:
        self._prefix = int(prefix)
        self._show()

    def _edit_address(self) -> None:
        self.number_pad().open_address(
            tr("Address"),
            self._address,
            self._set_address,
            check=valid_ipv4,
            hint=tr("IPv4 address, e.g. 192.168.2.20"),
        )

    def _set_address(self, text: str) -> None:
        self._address = text
        self._show()

    def _edit_gateway(self) -> None:
        self.number_pad().open_address(
            tr("Gateway"),
            self._gateway,
            self._set_gateway,
            check=lambda text: not text or valid_ipv4(text),
            hint=tr("IPv4 address; empty for none"),
        )

    def _set_gateway(self, text: str) -> None:
        self._gateway = text
        self._show()

    def _edit_dns(self) -> None:
        self.number_pad().open_address(
            tr("DNS servers"),
            DNS_SEPARATOR.join(self._dns),
            self._set_dns,
            check=valid_dns_list,
            hint=tr(DNS_HINT),
            several=True,
        )

    def _set_dns(self, text: str) -> None:
        self._dns = parse_dns_list(text) or ()
        self._dns_from_router = False
        self._show()

    def _on_apply_clicked(self) -> None:
        settings, on_apply = self.settings(), self._on_apply
        applicable = self.can_apply()
        self.dismiss()
        if applicable and on_apply is not None:
            on_apply(settings)

"""The data services of the ensemble received, each on a card with a switch: every one
is decoded by default (`ReceiverController`), one switched off stays off on its channel."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QScroller,
    QVBoxLayout,
    QWidget,
)

from dab.data_services import DataComponent
from dab.mot_directory import MotCarouselStatus
from i18n import tr, tr_noop
from ui.data_tab import describe_applications, why_not_decodable
from ui.receiver_controller import ReceiverController
from ui.touch.widgets import set_role

_NO_RECEPTION = tr_noop(
    "The data services an ensemble carries are listed in its FIC. They appear here as soon "
    "as a station is received."
)
_FOOTER = tr_noop(
    "New data services are decoded automatically; those switched off stay off on their channel."
)


def _description(component: DataComponent) -> str:
    parts = [describe_applications(component)]
    subchannel = component.subchannel
    if subchannel is not None:
        parts.append(
            tr("Subchannel {id}, {bitrate} kbit/s").format(
                id=subchannel.subchannel_id, bitrate=subchannel.bitrate
            )
        )
    elif component.subchannel_id is not None:
        parts.append(tr("Subchannel {id}").format(id=component.subchannel_id))
    return " · ".join(parts)


class _Card(QWidget):
    """One data service: name, application and subchannel, state, and the switch."""

    def __init__(self, component: DataComponent, on_toggled) -> None:
        super().__init__()
        set_role(self, "panel")
        # A QWidget subclass paints the background of the style sheet only when asked.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.scid = component.scid
        self.name = set_role(QLabel(component.label or tr("Unnamed")), "large")
        self.description = set_role(QLabel(_description(component)), "muted")
        self.description.setWordWrap(True)
        self.state = set_role(QLabel(), "muted")
        self.state.setWordWrap(True)
        self.switch = set_role(QPushButton(), "switch")
        self.switch.setCheckable(True)
        self.switch.toggled.connect(lambda on: self.switch.setText(tr("On") if on else tr("Off")))
        self.switch.clicked.connect(lambda on: on_toggled(component.scid, on))
        layout = QGridLayout(self)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setHorizontalSpacing(16)
        layout.setVerticalSpacing(4)
        layout.addWidget(self.name, 0, 0)
        layout.addWidget(self.description, 1, 0)
        layout.addWidget(self.state, 2, 0)
        layout.addWidget(self.switch, 0, 1, 3, 1, alignment=Qt.AlignmentFlag.AlignVCenter)
        layout.setColumnStretch(0, 1)


class DataServicesView(QWidget):
    def __init__(self, controller: ReceiverController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._cards: dict[int, _Card] = {}
        self._shown: list[tuple] = []  # what the cards were made from
        self._carousels: dict[int, MotCarouselStatus] = {}  # by SCId

        self._title = set_role(QLabel(), "large")
        all_on = QPushButton(tr("All on"))
        all_on.clicked.connect(lambda: controller.set_all_data_components_enabled(True))
        all_off = QPushButton(tr("All off"))
        all_off.clicked.connect(lambda: controller.set_all_data_components_enabled(False))
        self._all_buttons = (all_on, all_off)
        header = QHBoxLayout()
        header.setSpacing(12)
        header.addWidget(self._title, stretch=1)
        header.addWidget(all_on)
        header.addWidget(all_off)

        self._card_layout = QVBoxLayout()
        self._card_layout.setContentsMargins(0, 0, 0, 0)
        self._card_layout.setSpacing(12)
        self._card_layout.addStretch()
        cards = QWidget()
        cards.setLayout(self._card_layout)
        self._area = QScrollArea()
        self._area.setWidget(cards)
        self._area.setWidgetResizable(True)
        self._area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        QScroller.grabGesture(
            self._area.viewport(), QScroller.ScrollerGestureType.LeftMouseButtonGesture
        )
        self._empty = set_role(QLabel(tr(_NO_RECEPTION)), "large")
        self._empty.setWordWrap(True)
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        layout.addLayout(header)
        layout.addWidget(self._area, stretch=1)
        layout.addWidget(self._empty, stretch=1)
        layout.addWidget(set_role(QLabel(tr(_FOOTER)), "small"))

        controller.data_components_changed.connect(self._refresh)
        controller.dsp_worker.mot_carousel_ready.connect(self._on_carousel)
        controller.live_reset.connect(self._carousels.clear)
        self._refresh()

    def _refresh(self) -> None:
        controller = self._controller
        components = controller.data_components
        channel = controller.list_channel
        label = controller.live_ensemble_label
        where = " · ".join(part for part in (channel, label) if part)
        title = tr("Data services")
        self._title.setText(f"{title} · {where}" if where and components else title)
        shown = [(c.scid, c.label, c.decodable, _description(c)) for c in components]
        if shown != self._shown:
            self._shown = shown
            self._fill(components)
        for component in components:
            self._show_state(component)
        self._area.setVisible(bool(components))
        self._empty.setVisible(not components)
        for button in self._all_buttons:
            button.setEnabled(any(c.decodable for c in components))

    def _fill(self, components: list[DataComponent]) -> None:
        for card in self._cards.values():
            self._card_layout.removeWidget(card)
            card.deleteLater()
        self._cards = {}
        for index, component in enumerate(components):
            card = _Card(component, self._controller.set_data_component_enabled)
            self._card_layout.insertWidget(index, card)
            self._cards[component.scid] = card

    def _show_state(self, component: DataComponent) -> None:
        card = self._cards.get(component.scid)
        if card is None:
            return
        controller = self._controller
        switched_off = controller.is_data_component_switched_off(component)
        card.switch.setEnabled(component.decodable)
        card.switch.setChecked(component.decodable and not switched_off)
        card.switch.setText(tr("On") if card.switch.isChecked() else tr("Off"))
        if not component.decodable:
            text = why_not_decodable(component)
        elif switched_off:
            if controller.list_channel:
                text = tr("off · stays off on {channel} until switched on again").format(
                    channel=controller.list_channel
                )
            else:
                text = tr("off · stays off on this frequency until switched on again")
        elif not controller.is_data_component_decoding(component.scid):
            text = tr("starting …")
        else:
            carousel = self._carousels.get(component.scid)
            if carousel is not None and carousel.directory_objects is not None:
                text = tr("decoding · carousel {complete} of {total} files").format(
                    complete=carousel.complete_objects, total=carousel.directory_objects
                )
            else:
                text = tr("decoding")
        card.state.setText(text)

    def _on_carousel(self, scid: int, status: MotCarouselStatus) -> None:
        self._carousels[scid] = status
        component = next((c for c in self._controller.data_components if c.scid == scid), None)
        if component is not None:
            self._show_state(component)

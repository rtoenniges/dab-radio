"""The services page: sub-menus at the top, first the data services of the ensemble,
then one per service: the programme guide, GNSS (what GEPOS carries) and Traffic
(ARD TPEG).

A service's menu is only offered while the channel received carries the service (its
user application in FIG 0/13), except for those always shown: the list of data
services, and the programme guide, which also shows the guides stored for other
channels. A menu whose service is being decoded right now is marked green. The three
service menus have pages to swipe through; a second tap on the chip of the menu shown
takes it back to its first page.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from dab.data_services import (
    USER_APPLICATION_GEPOS,
    USER_APPLICATION_SPI,
    USER_APPLICATION_TPEG,
    DataComponent,
)
from dab.timecode import ReferenceClock
from i18n import tr, tr_noop
from ui.globe_gl import GlobeRenderThread
from ui.gnss_model import GnssModel
from ui.guide_model import GuideModel
from ui.receiver_controller import ReceiverController
from ui.touch.data_services_view import DataServicesView
from ui.touch.gnss_view import GnssView
from ui.touch.guide_view import GuideView
from ui.touch.theme import DARK, THEMES, Theme
from ui.touch.traffic_view import TrafficView
from ui.touch.widgets import repolish, set_role
from ui.traffic_model import TrafficModel

DATA_SERVICES = "data"
GUIDE = "guide"
GNSS = "gnss"
TRAFFIC = "traffic"

# The dot on a menu whose service is being decoded, and the gap to the name after it.
DECODING_DOT = 16
DOT_GAP = 8


@dataclass(frozen=True)
class Menu:
    key: str
    name: str  # marked with `tr_noop`, translated where shown
    # The user application of the service (FIG 0/13); None for the list of data services.
    app_type: int | None
    always: bool  # offered even if the channel received does not carry the service


MENUS = (
    Menu(DATA_SERVICES, tr_noop("Data services"), None, always=True),
    Menu(GUIDE, tr_noop("Programme guide"), USER_APPLICATION_SPI, always=True),
    Menu(GNSS, "GNSS", USER_APPLICATION_GEPOS, always=False),
    Menu(TRAFFIC, tr_noop("Traffic"), USER_APPLICATION_TPEG, always=False),
)


def menu_offered(menu: Menu, components: list[DataComponent]) -> bool:
    """Whether the menu is offered for the ensemble with these data components."""
    return menu.always or any(menu.app_type in c.app_types for c in components)


def menu_decoding(menu: Menu, components: list[DataComponent], decoding: set[int]) -> bool:
    """Whether the service of the menu is being decoded (SCIds in `decoding`)."""
    if menu.app_type is None:
        return False
    return any(menu.app_type in c.app_types and c.scid in decoding for c in components)


def _dot(color: QColor, ring: QColor) -> QIcon:
    """A green dot with a ring that sets it off from the blue of a chosen menu too; the
    picture is wider than the dot, which makes the gap to the name."""
    pixmap = QPixmap(DECODING_DOT + DOT_GAP, DECODING_DOT)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(ring)
    painter.drawEllipse(0, 0, DECODING_DOT, DECODING_DOT)
    painter.setBrush(color)
    painter.drawEllipse(2, 2, DECODING_DOT - 4, DECODING_DOT - 4)
    painter.end()
    return QIcon(pixmap)


def _theme_dot(theme: Theme) -> QIcon:
    return _dot(theme.color("success"), theme.color("on_accent"))


class ServicesPage(QWidget):
    def __init__(
        self,
        controller: ReceiverController,
        guide_model: GuideModel,
        gnss_model: GnssModel,
        traffic_model: TrafficModel,
        clock: ReferenceClock,
        parent: QWidget | None = None,
        globe_renderer: GlobeRenderThread | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._dot = _theme_dot(THEMES[DARK])
        self._data_view = DataServicesView(controller)
        self._guide_view = GuideView(controller, guide_model, clock)
        self._gnss_view = GnssView(controller, gnss_model, globe_renderer=globe_renderer)
        self._traffic_view = TrafficView(controller, traffic_model)
        views = {
            DATA_SERVICES: self._data_view,
            GUIDE: self._guide_view,
            GNSS: self._gnss_view,
            TRAFFIC: self._traffic_view,
        }

        self._stack = QStackedWidget()
        self._menu_group = QButtonGroup(self)
        self._menu_group.setExclusive(True)
        self._buttons: dict[str, QPushButton] = {}
        menu_row = QHBoxLayout()
        menu_row.setSpacing(12)
        for index, menu in enumerate(MENUS):
            button = set_role(QPushButton(tr(menu.name)), "chip")
            button.setCheckable(True)
            button.setIconSize(QSize(DECODING_DOT + DOT_GAP, DECODING_DOT))
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            button.setProperty("decoding", False)
            self._menu_group.addButton(button, index)
            self._buttons[menu.key] = button
            self._stack.addWidget(views[menu.key])
            menu_row.addWidget(button)
        menu_row.addStretch()
        self._menu_group.idClicked.connect(lambda index: self._on_menu_clicked(MENUS[index].key))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(16)
        layout.addLayout(menu_row)
        layout.addWidget(self._stack, stretch=1)

        controller.data_components_changed.connect(self._update_menus)
        self.show_view(DATA_SERVICES)
        self._update_menus()

    def _on_menu_clicked(self, key: str) -> None:
        """A tap on the chip of the view shown takes that view back to its first page."""
        if key == self.current_view():
            self.show_first_page()
        self.show_view(key)

    def show_first_page(self) -> None:
        """The view shown back to the first page of its pager, with the pages sliding
        by; nothing for a view without one."""
        pager = getattr(self._stack.currentWidget(), "pager", None)
        if pager is not None:
            pager.set_current_index(0, animated=True)

    def show_view(self, key: str) -> None:
        index = next(i for i, menu in enumerate(MENUS) if menu.key == key)
        self._stack.setCurrentIndex(index)
        self._buttons[key].setChecked(True)
        self.render()

    def show_guide(self, sid: int) -> None:
        """The programme guide at the schedule of the service, today, the programme
        running at the top."""
        self._guide_view.show_running(sid)
        self.show_view(GUIDE)

    def current_view(self) -> str:
        return MENUS[self._stack.currentIndex()].key

    def menu_button(self, key: str) -> QPushButton:
        return self._buttons[key]

    @property
    def gnss_view(self) -> GnssView:
        return self._gnss_view

    @property
    def traffic_view(self) -> TrafficView:
        return self._traffic_view

    def render(self) -> None:
        """Draws the plots of the menu shown; called by the window at its plot rate."""
        if self.current_view() == GUIDE:
            self._guide_view.render()
        elif self.current_view() == GNSS:
            self._gnss_view.render()
        elif self.current_view() == TRAFFIC:
            self._traffic_view.render()

    def apply_theme(self, theme: Theme) -> None:
        self._dot = _theme_dot(theme)
        self._guide_view.apply_theme(theme)
        self._gnss_view.apply_theme(theme)
        self._traffic_view.apply_theme(theme)
        self._update_menus()

    def _update_menus(self) -> None:
        """Offers the menus the channel received carries and marks those decoded."""
        controller = self._controller
        components = controller.data_components
        decoding = {c.scid for c in components if controller.is_data_component_decoding(c.scid)}
        for menu in MENUS:
            button = self._buttons[menu.key]
            button.setVisible(menu_offered(menu, components))
            active = menu_decoding(menu, components, decoding)
            if button.property("decoding") != active:
                button.setProperty("decoding", active)
                repolish(button)
            button.setIcon(self._dot if active else QIcon())
        if not self._buttons[self.current_view()].isVisibleTo(self):
            self.show_view(DATA_SERVICES)

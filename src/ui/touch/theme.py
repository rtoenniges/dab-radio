"""The colour schemes of the touch surface, dark and light, and the sizes that make it
usable with a finger.

The display has about 210 pixels per inch; `TOUCH_TARGET` (72 px) is a good 8.5 mm,
a comfortable size for a fingertip. Fonts are given in pixels, not points, and in a
typeface shipped with the surface (`touch_font`), so that they come out the same on the
display and on a PC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pyqtgraph as pg
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QGuiApplication

logger = logging.getLogger(__name__)

# Smallest edge of anything that is tapped.
TOUCH_TARGET = 72
NAV_WIDTH = 112
# The corners of a panel, and of the slide with the shade over it.
PANEL_RADIUS = 16
# A station tile: four side by side fill the 1120 px of the station page.
TILE_WIDTH = 268
TILE_HEIGHT = 164
# Every tile has this border, invisible but on the station playing: a border only there
# would make that tile bigger than the others.
TILE_BORDER = 3
# A tile of the choice of what the radio page shows in the place of the slideshow
# (`ui.touch.display_choice`): as wide as a station tile, and high enough for a picture
# in the shape of the slide with the name beneath it.
DISPLAY_TILE_HEIGHT = 232
# A thumbnail of the slide gallery (`ui.touch.slide_gallery`), two beside the slide.
THUMBNAIL_TILE_WIDTH = 216
THUMBNAIL_TILE_HEIGHT = 196
# How far the content of a row swiped sideways, or of a page swiped up and down, fades
# into the background at an edge where more of it lies beyond
# (`ui.touch.widgets.EdgeFades`).
EDGE_FADE_WIDTH = 64
# The square to tap of the star on the corner of the station's logo (`ui.touch.radio_page`),
# a little larger than the badge it shows.
BADGE_SIZE = 64
# A key of the number pad (`ui.touch.number_pad`) and of the keyboard
# (`ui.touch.keyboard`), whose eleven keys side by side fit the 1280 px of the screen.
KEY_WIDTH = 96
FONT_PX = 20
TITLE_FONT_PX = 34
LARGE_FONT_PX = 24
SMALL_FONT_PX = 16

# The typeface, Adobe's Source Sans 3 (SIL Open Font License, beside the files). Left to
# the system, the Raspberry Pi drew DejaVu Sans, about 12 % wider than Segoe UI on
# Windows, and cut "Einstellungen" (German) in the rail to "Eins…ngen" (08.10.2026). Source
# Sans 3 runs a little narrower than Segoe UI, so what fits on the PC fits on the display.
FONT_FAMILY = "Source Sans 3"
FONT_FILES = tuple(
    Path(__file__).parent.parent / "assets" / "fonts" / name
    for name in ("SourceSans3-Regular.ttf", "SourceSans3-Bold.ttf")
)

# Key of the setting holding the scheme, and its values.
SETTINGS_THEME = "touch/theme"
DARK = "dark"
LIGHT = "light"


@dataclass(frozen=True)
class Theme:
    key: str
    background: str  # behind everything
    surface: str  # navigation rail, panels
    raised: str  # buttons and tiles
    pressed: str
    outline: str
    text: str
    muted: str  # secondary text
    accent: str
    on_accent: str  # text on the accent colour
    accent_soft: str  # the tile of the station playing, the page shown
    danger: str
    on_danger: str
    success: str  # a service being decoded
    notice: str  # a running announcement
    on_notice: str

    def color(self, name: str) -> QColor:
        return QColor(getattr(self, name))


THEMES = {
    DARK: Theme(
        key=DARK,
        background="#101418",
        surface="#182027",
        raised="#232d37",
        pressed="#33414f",
        outline="#3a4654",
        text="#e8edf2",
        muted="#93a1af",
        accent="#4fc3f7",
        on_accent="#06141c",
        accent_soft="#143447",
        danger="#e05252",
        on_danger="#ffffff",
        success="#5bd17a",
        notice="#f0a020",
        on_notice="#101418",
    ),
    LIGHT: Theme(
        key=LIGHT,
        background="#eef1f5",
        surface="#ffffff",
        raised="#dfe5ec",
        pressed="#c9d2dc",
        outline="#b9c3ce",
        text="#15202b",
        muted="#5b6875",
        accent="#0277bd",
        on_accent="#ffffff",
        accent_soft="#d3e8f5",
        danger="#c62828",
        on_danger="#ffffff",
        success="#2e7d32",
        notice="#f0a020",
        on_notice="#15202b",
    ),
}


def touch_font() -> QFont:
    """The application's font in the typeface of the touch surface, to be set with
    `QApplication.setFont` before the window is built; registers the typeface with Qt
    when its database does not have it, which it checks every time: once registered,
    the typeface was gone again from the database between two tests on the offscreen
    platform (08.10.2026), and a flag of this module would not have known. A file that
    cannot be loaded leaves the system's typeface in its place."""
    if FONT_FAMILY not in QFontDatabase.families():
        for path in FONT_FILES:
            if QFontDatabase.addApplicationFont(str(path)) < 0:
                logger.warning("Cannot load the font %s, the system's font stands in", path)
    font = QGuiApplication.font()
    font.setFamily(FONT_FAMILY)
    return font


def stylesheet(theme: Theme) -> str:
    """The style sheet of the whole touch window. Widgets pick their look through the
    dynamic property `role`; a button with `active` set tells that what it leads to is
    in effect without being a switch (the equalizer beside the loudspeaker, see
    `ui.touch.radio_page`)."""
    t = theme
    # Over the slide while a finger rests on it (`ui.touch.radio_page`).
    text = theme.color("text")
    shade = f"rgba({text.red()}, {text.green()}, {text.blue()}, 36)"
    # The background, transparent: where an edge fade begins.
    background = theme.color("background")
    clear = f"rgba({background.red()}, {background.green()}, {background.blue()}, 0)"
    return f"""
    QWidget {{
        background: {t.background};
        color: {t.text};
        font-size: {FONT_PX}px;
    }}
    QWidget#navRail, QWidget[role="panel"] {{
        background: {t.surface};
        border-radius: {PANEL_RADIUS}px;
    }}
    QWidget#navRail {{ border-radius: 0px; }}
    QLabel, QWidget[role="transparent"] {{ background: transparent; }}
    QLabel[role="title"] {{ font-size: {TITLE_FONT_PX}px; font-weight: bold; }}
    QLabel[role="large"] {{ font-size: {LARGE_FONT_PX}px; }}
    QLabel[role="headline"] {{ font-size: {LARGE_FONT_PX}px; font-weight: bold; }}
    QLabel[role="muted"] {{ color: {t.muted}; }}
    QLabel[role="small"] {{ color: {t.muted}; font-size: {SMALL_FONT_PX}px; }}
    QLabel[role="heading"] {{ color: {t.muted}; font-size: {SMALL_FONT_PX}px; font-weight: bold; }}
    QLabel[role="slide"] {{
        background: {t.surface};
        border-radius: {PANEL_RADIUS}px;
        color: {t.muted};
    }}
    QWidget[role="shade"] {{ background: {shade}; border-radius: {PANEL_RADIUS}px; }}
    QWidget[role="fadeLeft"] {{
        background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 {t.background}, stop: 1 {clear});
    }}
    QWidget[role="fadeRight"] {{
        background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 {clear}, stop: 1 {t.background});
    }}
    QWidget[role="fadeTop"] {{
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
            stop: 0 {t.background}, stop: 1 {clear});
    }}
    QWidget[role="fadeBottom"] {{
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
            stop: 0 {clear}, stop: 1 {t.background});
    }}
    QLabel#errorBanner {{
        background: {t.danger};
        color: {t.on_danger};
        border-radius: 12px;
        padding: 12px 20px;
    }}
    QPushButton, QToolButton {{
        background: {t.raised};
        color: {t.text};
        border: none;
        border-radius: 12px;
        min-height: {TOUCH_TARGET}px;
        min-width: {TOUCH_TARGET}px;
        padding: 0px 18px;
    }}
    QPushButton:pressed, QToolButton:pressed {{ background: {t.pressed}; }}
    QPushButton:checked, QToolButton:checked {{
        background: {t.accent};
        color: {t.on_accent};
    }}
    QPushButton:disabled, QToolButton:disabled {{
        background: {t.surface};
        color: {t.outline};
    }}
    QToolButton[active="true"] {{ background: {t.accent_soft}; color: {t.accent}; }}
    QToolButton[role="nav"] {{
        background: transparent;
        border-radius: 12px;
        font-size: {SMALL_FONT_PX}px;
        padding: 6px 0px;
        min-width: 80px;
    }}
    QToolButton[role="nav"]:checked {{ background: {t.accent_soft}; color: {t.accent}; }}
    QToolButton[role="badge"], QToolButton[role="badge"]:checked,
    QToolButton[role="badge"]:pressed {{
        background: transparent;
        border: none;
        padding: 0px;
        min-width: {BADGE_SIZE}px;
        max-width: {BADGE_SIZE}px;
        min-height: {BADGE_SIZE}px;
        max-height: {BADGE_SIZE}px;
    }}
    QToolButton[role="tile"] {{
        padding: 12px 8px;
        font-size: {FONT_PX}px;
        border: {TILE_BORDER}px solid transparent;
        min-width: {TILE_WIDTH - 2 * (8 + TILE_BORDER)}px;
        max-width: {TILE_WIDTH - 2 * (8 + TILE_BORDER)}px;
        min-height: {TILE_HEIGHT - 2 * (12 + TILE_BORDER)}px;
        max-height: {TILE_HEIGHT - 2 * (12 + TILE_BORDER)}px;
    }}
    QToolButton[role="display"] {{
        padding: 10px 8px;
        border: {TILE_BORDER}px solid transparent;
        min-width: {TILE_WIDTH - 2 * (8 + TILE_BORDER)}px;
        max-width: {TILE_WIDTH - 2 * (8 + TILE_BORDER)}px;
        min-height: {DISPLAY_TILE_HEIGHT - 2 * (10 + TILE_BORDER)}px;
        max-height: {DISPLAY_TILE_HEIGHT - 2 * (10 + TILE_BORDER)}px;
    }}
    QToolButton[role="thumb"] {{
        padding: 4px;
        font-size: {SMALL_FONT_PX}px;
        border: {TILE_BORDER}px solid transparent;
        min-width: {THUMBNAIL_TILE_WIDTH - 2 * (4 + TILE_BORDER)}px;
        max-width: {THUMBNAIL_TILE_WIDTH - 2 * (4 + TILE_BORDER)}px;
        min-height: {THUMBNAIL_TILE_HEIGHT - 2 * (4 + TILE_BORDER)}px;
        max-height: {THUMBNAIL_TILE_HEIGHT - 2 * (4 + TILE_BORDER)}px;
    }}
    QToolButton[role="tile"][current="true"], QToolButton[role="display"][current="true"],
    QToolButton[role="thumb"][current="true"] {{
        background: {t.accent_soft};
        color: {t.accent};
        border: {TILE_BORDER}px solid {t.accent};
    }}
    QPushButton[role="chip"] {{
        min-height: 50px;
        border-radius: 28px;
        padding: 0px 22px;
        border: 3px solid transparent;
    }}
    QPushButton[role="chip"][decoding="true"] {{ border-color: {t.success}; }}
    QPushButton[role="danger"] {{ background: {t.danger}; color: {t.on_danger}; }}
    QPushButton[role="notice"] {{ background: {t.notice}; color: {t.on_notice}; }}
    QPushButton[role="step"] {{ font-size: {TITLE_FONT_PX}px; padding: 0px; }}
    QSlider {{ background: transparent; min-height: {TOUCH_TARGET}px; }}
    QSlider::groove:horizontal {{
        height: 10px;
        border-radius: 5px;
        background: {t.outline};
    }}
    QSlider::sub-page:horizontal {{ border-radius: 5px; background: {t.accent}; }}
    QSlider::handle:horizontal {{
        width: 40px;
        height: 40px;
        margin: -15px 0px;
        border-radius: 20px;
        background: {t.accent};
    }}
    QPushButton[role="switch"] {{ min-width: 120px; }}
    QListWidget {{
        background: {t.surface};
        border: none;
        border-radius: 16px;
        padding: 8px;
        outline: none;
    }}
    QListWidget::item {{
        min-height: {TOUCH_TARGET - 16}px;
        padding: 4px 8px;
        border-radius: 10px;
    }}
    QListWidget::item:disabled {{ color: {t.muted}; font-size: {SMALL_FONT_PX}px; }}
    QListWidget::item:selected {{ background: {t.accent_soft}; color: {t.accent}; }}
    QWidget[role="legend"] {{ background: transparent; border-radius: 10px; }}
    QWidget[role="legend"][pressed="true"] {{ background: {t.pressed}; }}
    QWidget[role="programme"] {{ background: transparent; border-radius: 10px; }}
    QWidget[role="programme"][running="true"] {{ background: {t.accent_soft}; }}
    QWidget[role="programme"][running="true"] QLabel {{ color: {t.accent}; }}
    QWidget[role="notice"] {{ background: {t.notice}; border-radius: 16px; }}
    QWidget[role="notice"] QLabel {{ color: {t.on_notice}; }}
    QWidget[role="notice"] QLabel[role="large"] {{ font-weight: bold; }}
    QWidget[role="notice"][alarm="true"] {{ background: {t.danger}; }}
    QWidget[role="notice"][alarm="true"] QLabel {{ color: {t.on_danger}; }}
    QWidget[role="level"] {{ background: {t.accent}; border-radius: 3px; }}
    QWidget[role="level"][level="warning"] {{ background: {t.notice}; }}
    QWidget[role="level"][level="alarm"] {{ background: {t.danger}; }}
    QScrollArea {{ border: none; }}
    QScrollBar {{ width: 0px; height: 0px; }}
    QTableView {{
        background: {t.surface};
        border: none;
        border-radius: 16px;
        padding: 4px 8px;
        gridline-color: transparent;
        outline: none;
    }}
    QTableView::item {{ padding: 0px 6px; border-bottom: 1px solid {t.background}; }}
    QTableView::item:selected {{ background: {t.accent_soft}; color: {t.accent}; }}
    QHeaderView, QHeaderView::section, QTableCornerButton::section {{
        background: {t.surface};
        color: {t.muted};
        border: none;
    }}
    QHeaderView::section {{
        font-size: {SMALL_FONT_PX}px;
        padding: 8px 6px;
        border-bottom: 2px solid {t.outline};
    }}
    QWidget[role="scrim"] {{ background: rgba(0, 0, 0, 150); }}
    QPushButton[role="key"] {{
        font-size: {TITLE_FONT_PX}px;
        padding: 0px;
        min-width: {KEY_WIDTH}px;
        max-width: {KEY_WIDTH}px;
        min-height: {TOUCH_TARGET}px;
        max-height: {TOUCH_TARGET}px;
    }}
    QPushButton[role="key"][locked="true"] {{ border: 3px solid {t.on_accent}; }}
    QLabel[role="field"] {{
        background: {t.raised};
        border-radius: 12px;
        padding: 8px 16px;
        font-size: {LARGE_FONT_PX}px;
    }}
    QWidget[role="row"] {{ background: transparent; border-radius: 10px; }}
    QWidget[role="row"][selected="true"] {{ background: {t.accent_soft}; }}
    QWidget[role="row"][selected="true"] QLabel {{ color: {t.accent}; }}
    QWidget[role="row"][pressed="true"] {{ background: {t.pressed}; }}
    QPushButton[role="accent"] {{ background: {t.accent}; color: {t.on_accent}; }}
    QPushButton[role="accent"]:disabled {{ background: {t.surface}; color: {t.outline}; }}
    QLabel[role="error"] {{ color: {t.danger}; font-size: {SMALL_FONT_PX}px; }}
    QLabel[role="accent"] {{ color: {t.accent}; }}
    QLabel[role="danger"] {{ color: {t.danger}; }}
    QWidget[role="row"][dimmed="true"] QLabel {{ color: {t.outline}; }}
    QLabel[alert="true"] {{ color: {t.danger}; font-weight: bold; }}
    QPushButton[role="value"] {{ font-size: {LARGE_FONT_PX}px; min-width: 200px; }}
    QPushButton[role="value"][placeholder="true"] {{ color: {t.muted}; }}
    QPushButton[role="value"][invalid="true"] {{
        border: 3px solid {t.danger};
        padding: 0px 15px;
        min-height: {TOUCH_TARGET - 6}px;
    }}
    QLabel[role="muted"][invalid="true"] {{ color: {t.danger}; }}
    QLabel[role="logo"] {{ background: #ffffff; border-radius: 10px; padding: 6px 10px; }}
    """


def style_plot(plot: pg.PlotWidget, theme: Theme) -> None:
    """Colours a plot like the panel it sits on; touch neither pans nor zooms it."""
    plot.setBackground(theme.surface)
    for name in ("left", "bottom", "top", "right"):
        axis = plot.getAxis(name)
        axis.setPen(pg.mkPen(theme.outline))
        axis.setTextPen(pg.mkPen(theme.muted))
    plot.setMouseEnabled(x=False, y=False)
    plot.setMenuEnabled(False)
    plot.hideButtons()

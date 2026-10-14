"""The Journaline of the station playing (ETSI TS 102 979), without widgets, so far for
`ui.touch`: the pages received, whether FIG 0/13 announces Journaline for the station
at all, and its point of entry.

Journaline travels in the X-PAD of the station, like its dynamic label and slideshow:
only the station selected can be decoded, and a change of station voids everything.
Pages past their timeout (clause 5.3.2.2) are removed by a timer.
"""

from __future__ import annotations

import time

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from dab.journaline import (
    ROOT_OBJECT_ID,
    Change,
    JmlObject,
    JournalineParameters,
    JournalineStore,
)
from ui.receiver_controller import ReceiverController

# Interval at which pages past their timeout are removed. Timeouts count minutes or
# quarter hours (clause 5.3.2.2); ten seconds are ample.
EXPIRY_TICK_MS = 10_000


class JournalineModel(QObject):
    # The object IDs of pages new, changed or gone; None: everything may have changed
    # (another station, Journaline announced or no longer, another point of entry).
    changed = pyqtSignal(object)

    def __init__(self, controller: ReceiverController, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._store = JournalineStore()
        self._parameters: JournalineParameters | None = None

        self._expiry_timer = QTimer(self)
        self._expiry_timer.setInterval(EXPIRY_TICK_MS)
        self._expiry_timer.timeout.connect(self._expire)
        self._expiry_timer.start()

        dsp = controller.dsp_worker
        dsp.journaline_object_ready.connect(self._on_object)
        dsp.journaline_parameters_ready.connect(self._on_parameters)
        controller.service_reset.connect(self._reset)

    @property
    def announced(self) -> bool:
        """Whether FIG 0/13 announces Journaline for the station playing."""
        return self._parameters is not None

    @property
    def entry(self) -> int:
        """The page to start from, as FIG 0/13 names it (clause 8.1.2); usually the
        main menu, object 0x0000."""
        if self._parameters is None:
            return ROOT_OBJECT_ID
        return self._parameters.entry_object

    def get(self, object_id: int) -> JmlObject | None:
        """The page in its newest version; None while not received or past its
        timeout."""
        return self._store.get(object_id, time.time())

    def objects(self) -> list[JmlObject]:
        """All pages received and valid, in ascending order of their IDs."""
        return self._store.objects(time.time())

    def received_at(self, object_id: int) -> float | None:
        """When the page was received last (POSIX time)."""
        return self._store.received_at(object_id)

    def _on_object(self, jml_object: JmlObject) -> None:
        if self._store.add(jml_object, time.time()) is not Change.UNCHANGED:
            self.changed.emit(frozenset((jml_object.object_id,)))

    def _on_parameters(self, parameters: JournalineParameters | None) -> None:
        if parameters != self._parameters:
            self._parameters = parameters
            self.changed.emit(None)

    def _reset(self) -> None:
        """The pages belong to the station before; the pipeline reports anew whether
        the next one announces Journaline."""
        if self._parameters is None and not len(self._store):
            return
        self._store.clear()
        self._parameters = None
        self.changed.emit(None)

    def _expire(self) -> None:
        expired = self._store.expire(time.time())
        if expired:
            self.changed.emit(frozenset(expired))

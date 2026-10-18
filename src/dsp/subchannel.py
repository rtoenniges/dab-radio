"""The channel decoding of one subchannel: from its capacity units in the CIFs to the
logical frames (ETSI EN 300 401, clauses 5.3, 11 and 12).

Every subchannel is decoded on its own: its bits are cut out of each CIF, time
deinterleaved over 16 CIFs and then freed from convolutional code, puncturing and
energy dispersal (EEP or UEP). The receiver runs this once for the audio of the selected
service and once more for each subchannel carrying a data service being decoded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dab.fig import Subchannel
from dsp.eep import decode_eep_logical_frame
from dsp.msc import extract_subchannel_bits
from dsp.time_interleave import INTERLEAVE_DEPTH, TimeDeinterleaver
from dsp.uep import decode_uep_logical_frame


@dataclass
class SubchannelFrames:
    """What one transmission frame (4 CIFs) yielded for a subchannel."""

    raw_bits: list[np.ndarray] = field(default_factory=list)  # soft bits per CIF, still coded
    logical_frames: list[np.ndarray] = field(default_factory=list)  # decoded info bits
    # True if the subchannel changed (another one, or another size) and the chain started
    # afresh; whatever the caller built from earlier logical frames is then worthless.
    restarted: bool = False


class SubchannelDecoder:
    """Time deinterleaving and convolutional decoding of one subchannel.

    The time deinterleaver needs an uninterrupted sequence of CIFs. After a gap the
    caller calls `reset`; the first `INTERLEAVE_DEPTH` CIFs after that are withheld,
    since they still contain the zeros of the unfilled delay stages.
    """

    def __init__(self) -> None:
        self._deinterleaver: TimeDeinterleaver | None = None
        self._config: tuple[int, int] | None = None  # (SubChId, fragment size)
        self._fill = 0  # CIFs since the last reset

    def reset(self) -> None:
        if self._config is not None:
            self._deinterleaver = TimeDeinterleaver(self._config[1])
        self._fill = 0

    def forget(self) -> None:
        """Drops the subchannel altogether (the service is no longer decoded)."""
        self._deinterleaver = None
        self._config = None
        self._fill = 0

    def push(self, cifs: list[np.ndarray], subchannel: Subchannel) -> SubchannelFrames:
        frames = SubchannelFrames()
        frames.raw_bits = [
            extract_subchannel_bits(cif, subchannel.start_address, subchannel.size) for cif in cifs
        ]

        # A service/subchannel change needs a fresh time deinterleaver, whose ring
        # buffer would otherwise hold data of another subchannel that no longer line up
        # in time.
        fragment_size = subchannel.size * 64
        config = (subchannel.subchannel_id, fragment_size)
        if self._config != config:
            self._config = config
            self.reset()
            frames.restarted = True

        # Important: ALWAYS feed the actually demodulated bits here, even when
        # `demodulate_frame_symbols` rated this frame unreliable. Zero bits in their
        # place, to preserve only the ring buffer phase of the time deinterleaver,
        # make audio decoding practically impossible: through the convolutional
        # structure of the deinterleaver every block of zeros corrupts 1/16 of the
        # bits in EACH of the next 16 CIFs (see dsp.time_interleave.TimeDeinterleaver),
        # so that a share of unreliable frames leaves practically no Reed-Solomon
        # codeword correctable. The actually demodulated bits, on the other hand, are
        # mostly only partially wrong even in "unreliable" frames, exactly the kind of
        # bit errors Viterbi and Reed-Solomon decoding are meant to absorb; as soft
        # bits they additionally tell the Viterbi decoder how little to trust them.
        deinterleaved = [self._deinterleaver.push(bits) for bits in frames.raw_bits]

        # The first `INTERLEAVE_DEPTH` CIFs after a reset still contain the zeros of
        # the unfilled delay stages and are therefore necessarily garbage. Decoding
        # and passing them on would only feed error counters and the frame searches
        # with guaranteed worthless data (~384 ms start-up, see TimeDeinterleaver).
        primed = []
        for bits in deinterleaved:
            if self._fill < INTERLEAVE_DEPTH:
                self._fill += 1
            else:
                primed.append(bits)

        # Classic DAB services mostly use UEP, DAB+ services and packet mode data
        # always EEP; the protection is a property of the subchannel.
        decode = decode_uep_logical_frame if subchannel.is_uep else decode_eep_logical_frame
        frames.logical_frames = [decode(bits, subchannel) for bits in primed]
        return frames

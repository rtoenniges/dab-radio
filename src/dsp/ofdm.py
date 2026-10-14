"""OFDM demodulation of one DAB transmission frame.

Frequency offset estimation and correction, PRS-based frame alignment, the FFT per
symbol and the differential (D-QPSK) demodulation of the data symbols. `dsp.timing`
holds the phase reference symbol and the PRS cross-correlation for the frame alignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dab.constants import TransmissionMode
from dsp.timing import find_integer_symbol_offset, generate_reference_symbol


def estimate_frequency_offset(
    frame_samples: np.ndarray,
    fft_length: int,
    guard_length: int,
    symbol_length: int,
    sample_rate: float,
) -> float:
    """Estimates the carrier frequency offset from the cyclic prefix correlation.

    The guard interval of every OFDM symbol is an exact copy of the last `guard_length`
    samples of the useful part. A carrier frequency offset rotates the phase between
    the two copies in proportion to their distance in time (`fft_length` samples). The
    correlation is averaged over all symbols in the buffer (coherent summation).

    Unambiguous range: +/- sample_rate / (2 * fft_length).

    Raises:
        ValueError: if the buffer does not hold a complete symbol.
    """
    num_symbols = len(frame_samples) // symbol_length
    if num_symbols == 0:
        raise ValueError("Too few samples for a frequency offset estimate.")

    correlation_sum = 0j
    for i in range(num_symbols):
        start = i * symbol_length
        prefix = frame_samples[start : start + guard_length]
        tail = frame_samples[start + fft_length : start + fft_length + guard_length]
        correlation_sum += np.vdot(prefix, tail)

    return float(np.angle(correlation_sum) * sample_rate / (2 * np.pi * fft_length))


# Block length of the rotation phasor in `correct_frequency_offset`.
_ROTATION_BLOCK = 2048


def correct_frequency_offset(
    samples: np.ndarray, freq_offset_hz: float, sample_rate: float
) -> np.ndarray:
    """Rotates the samples back against the estimated carrier frequency offset.

    The phasor exp(-j·2π·f·n/fs) is not evaluated sample by sample: a complex
    exponential over a whole frame (~196 000 samples) cost more than all of the FFTs.
    Instead it is assembled from the rotation within one block and the rotation at the
    start of every block, since exp(a + b) = exp(a) · exp(b). That is one exponential
    per block and per sample within a block, and one multiplication per sample.
    """
    step = -2 * np.pi * freq_offset_hz / sample_rate
    num_blocks = -(-len(samples) // _ROTATION_BLOCK)
    within_block = np.exp(1j * step * np.arange(_ROTATION_BLOCK))
    block_starts = np.exp(1j * step * _ROTATION_BLOCK * np.arange(num_blocks))
    rotated = np.outer(block_starts, within_block).ravel()[: len(samples)]
    rotated *= samples
    return rotated


def estimate_integer_frequency_offset(
    samples: np.ndarray, mode: TransmissionMode, max_bins: int = 64
) -> int:
    """Estimates an integer carrier frequency offset in multiples of the subcarrier
    spacing (exactly 1 kHz per FFT bin in DAB mode I).

    The cyclic prefix correlation (`estimate_frequency_offset`) can only resolve
    offsets within +/- sample_rate/(2*fft_length); a larger offset that is a whole
    number of bins aliases to a small value there and goes unnoticed. Such a residual
    offset shifts every subcarrier by whole FFT bins: the constellation stays
    deceptively clean (every bin still evolves consistently over time), but frequency
    deinterleaving and bit demapping read the wrong bins. It is detected by sliding the
    ideal rectangular carrier occupation across an averaged power spectrum; the best
    match gives the offset.

    Returns:
        Integer bin offset (in kHz for mode I, since 1 bin = 1 kHz).
    """
    num_symbols = min(24, len(samples) // mode.symbol_length)
    if num_symbols == 0:
        return 0

    symbols = samples[: num_symbols * mode.symbol_length].reshape(num_symbols, mode.symbol_length)
    spectra = demodulate_symbol(symbols, mode.fft_length, mode.guard_length)
    power = np.sum(np.abs(spectra) ** 2, axis=0)

    # The occupation mask is a rectangle over bins center-half..center+half with a hole
    # at DC. Its overlap with the power spectrum at shift s is therefore the power
    # summed over that range moved by s, minus the power at the moved DC bin: a
    # difference of two prefix sums instead of a product over all bins per shift. The
    # spectrum is appended to itself so that a range wrapping around the end (as the
    # circular shift of the mask would) stays contiguous.
    center = mode.fft_length // 2
    half = mode.num_carriers // 2
    width = 2 * half + 1
    prefix = np.concatenate(([0.0], np.cumsum(np.concatenate((power, power)))))

    shifts = np.arange(-max_bins, max_bins + 1)
    starts = (center - half + shifts) % mode.fft_length
    scores = prefix[starts + width] - prefix[starts] - power[(center + shifts) % mode.fft_length]
    return int(shifts[np.argmax(scores)])


def demodulate_symbol(symbol_samples: np.ndarray, fft_length: int, guard_length: int) -> np.ndarray:
    """FFT demodulation of an OFDM symbol; the guard interval is discarded.

    Also accepts several symbols at once, one per row: the transform always runs along
    the last axis, each row on its own.
    """
    useful_part = symbol_samples[..., guard_length : guard_length + fft_length]
    return np.fft.fftshift(np.fft.fft(useful_part, axis=-1), axes=-1)


def extract_used_carriers(spectrum: np.ndarray, num_carriers: int) -> np.ndarray:
    """Extracts the subcarriers actually in use from the FFT spectrum.

    DAB uses `num_carriers` subcarriers symmetrically around the (unused) centre
    frequency (DC): half below, half above. `spectrum` must already be centred with
    `fftshift` (DC in the middle).

    Also accepts several spectra at once, one per row (as `demodulate_symbol` delivers
    them).
    """
    center = spectrum.shape[-1] // 2
    half = num_carriers // 2
    lower = spectrum[..., center - half : center]
    upper = spectrum[..., center + 1 : center + 1 + half]
    return np.concatenate([lower, upper], axis=-1)


def differential_demodulate(current: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """D-QPSK differential demodulation: phase difference to the previous OFDM symbol,
    per subcarrier."""
    return current * np.conj(reference)


def correct_residual_phase(differential_symbol: np.ndarray) -> np.ndarray:
    """Removes a common residual phase from a differentially demodulated symbol (e.g.
    from frequency offset drift within the observation window, which changes slightly
    between consecutive symbols and therefore does not cancel exactly in the
    difference).

    The four ideal D-QPSK points lie at +/-45° and +/-135°; without a residual phase
    their fourth power therefore points to 180° for all of them. A deviation from 180°
    in the averaged z^4 reveals the common residual phase, which is estimated per
    symbol and rotated out.

    Also accepts all symbols of a frame at once, one per row; each row then gets its
    own residual phase. z^4 is formed by squaring twice: `**4` on complex numbers goes
    through the general complex power and took most of the time here.
    """
    unit = differential_symbol / (np.abs(differential_symbol) + 1e-12)
    squared = unit * unit
    phi = np.angle(-np.mean(squared * squared, axis=-1, keepdims=True)) / 4.0
    return differential_symbol * np.exp(-1j * phi)


# Minimum ratio of correlation peak to mean (see `dsp.timing.find_integer_symbol_offset`)
# for the PRS alignment of a frame to be trusted. Pure noise gives 3.2 (median, at most
# 4.7 in 2000 trials); ~3 was also observed during multipath dropouts, where the
# alignment was wrong. A real match lies far above, even for a weak signal: measured
# 50-90 on channel 9D at 6-9 dB SNR, ~250 on the strong 5C.
MIN_INTEGER_OFFSET_CONFIDENCE = 5.0

# This many symbols from the start of the frame suffice for estimating the integer
# frequency offset and the sample alignment (see `estimate_integer_frequency_offset`,
# which averages at most 24 symbols anyway).
_INTEGER_OFFSET_PROBE_SYMBOLS = 24


@dataclass(frozen=True)
class DemodulatedFrame:
    """Everything `demodulate_frame_symbols` recovers from one transmission frame."""

    # D-QPSK constellation points per data symbol, each `mode.num_carriers` long in
    # physical carrier order. Index 0 is OFDM symbol l=2, the first of the three FIC
    # symbols.
    symbols: list[np.ndarray]
    # Total carrier frequency offset that was estimated and corrected: the fractional
    # part from the guard interval correlation plus whole subcarrier spacings from the
    # spectral mask correlation. Useful as a display of the current deviation, e.g.
    # for calibrating the RTL-SDR's ppm setting by hand.
    frequency_offset_hz: float
    # Integer sample correction of the frame start that was actually applied
    # (possibly the fallback, see `reliable`).
    sample_correction: int
    # False when the PRS was not found with certainty and the sync parameters fell back
    # to the previous trustworthy values. That means not only an uncertain alignment
    # but often simply a signal dropout for this one frame (e.g. brief destructive
    # multipath in an SFN) that the correct parameters could not heal either. The
    # demodulated bits are still fed into the time deinterleaver (see
    # `dsp.subchannel.SubchannelDecoder.push` for why); a run of unreliable frames
    # makes the pipeline re-acquire frame sync.
    #
    # Deliberately a statement about synchronisation only, not about the signal
    # quality: a weak but decodable signal must not count as lost. An earlier version
    # also rated frames by how tightly their D-QPSK points cluster; on channel 9D at
    # ~9 dB SNR that failed every frame, the MSC chain was restarted every 10 frames
    # and half of the audio was lost, although Reed-Solomon corrected every superframe
    # once the restarts were gone. Noise is for the error correction to deal with.
    reliable: bool
    # Channel per subcarrier (physical order), estimated from the received against the
    # known PRS at exactly the window position used for demodulation. The path the
    # receiver synchronised to therefore sits at delay 0 to within a fraction of a
    # sample (see `dsp.channel`).
    channel_estimate: np.ndarray
    # Unmodified FFT output, one row per symbol starting at the PRS (row 0 = l=1), all
    # `fft_length` bins with DC in the middle: unused edge bins included, before the
    # differential demodulation. For display.
    spectra: np.ndarray


def demodulate_frame_symbols(
    samples: np.ndarray,
    null_start: int,
    mode: TransmissionMode,
    sample_rate: float,
    max_data_symbols: int = 16,
    fallback_sample_correction: int = 0,
    fallback_freq_offset: float = 0.0,
) -> DemodulatedFrame | None:
    """Demodulates the D-QPSK constellation points of a DAB frame, per data symbol.

    Uses the frame start already known from the null symbol search, estimates and
    corrects the carrier frequency offset from the guard interval, corrects a
    remaining integer sample misalignment of the frame start by PRS cross-correlation
    (the null symbol search is only approximately accurate) and demodulates the phase
    reference symbol (PRS) and the following data symbols by FFT. The data symbols are
    then demodulated differentially against the previous symbol on the same
    subcarrier, which is how DAB's D-QPSK modulation works. A remaining fraction of a
    sample of timing error needs no correction: its phase ramp across the carriers is
    the same in every symbol and cancels in that difference.

    `fallback_sample_correction` is used when the PRS cross-correlation
    (`dsp.timing.find_integer_symbol_offset`) does not deliver a sufficiently certain
    detection (confidence below `MIN_INTEGER_OFFSET_CONFIDENCE`), e.g. during brief
    multipath dropouts in an SFN such as the German Bundesmux. Live measurements
    showed that blindly taking an uncertain value (confidence ~3 instead of the usual
    ~9-10) misaligns the frame by up to several hundred samples and produces a
    constellation rotated by ~45° across all carriers, unusable for the whole frame:
    clearly visible as a brief loss of sync (collapsed constellation cloud,
    "fluttering" frequency offset display). Since the correction actually needed
    hardly changes between consecutive frames (only through slow clock drift), the
    last certain correction is a far better estimate than a single uncertain
    measurement. In that case the frequency offset falls back to
    `fallback_freq_offset` as well: the same disturbance affects both estimators, so
    the guard interval estimate of this frame is untrustworthy too. The caller passes
    the returned `sample_correction` and `frequency_offset_hz` back in for the next
    frame; `FrameDemodulator` does exactly that.

    Returns:
        The demodulated frame, or None if there are not enough samples for at least
        one data symbol after the null symbol.
    """

    def slice_frame(start: int) -> tuple[np.ndarray, int] | None:
        available = len(samples) - start
        symbols = min(
            max_data_symbols,
            mode.symbols_per_frame - 1,
            available // mode.symbol_length - 1,
        )
        if symbols < 1 or start < 0:
            return None
        total_symbols = symbols + 1  # + phase reference symbol (PRS)
        return samples[start : start + total_symbols * mode.symbol_length], total_symbols

    frame_start = null_start + mode.null_symbol_length
    sliced = slice_frame(frame_start)
    if sliced is None:
        return None
    frame_samples, num_symbols = sliced

    freq_offset = estimate_frequency_offset(
        frame_samples, mode.fft_length, mode.guard_length, mode.symbol_length, sample_rate
    )

    # The two estimates that follow only need the start of the frame, not all of it.
    # Rotating out the frequency offset used to run up to four times over all ~196 000
    # samples of a frame, 16 % of the total compute time according to profiling. The
    # excerpts start at the frame start, so they yield exactly the same values as the
    # full frame would.
    bin_width_hz = sample_rate / mode.fft_length
    probe_symbols = min(_INTEGER_OFFSET_PROBE_SYMBOLS, num_symbols)
    probe = correct_frequency_offset(
        frame_samples[: probe_symbols * mode.symbol_length], freq_offset, sample_rate
    )

    # Integer residual frequency offset (multiples of the subcarrier spacing), which
    # the cyclic prefix correlation cannot resolve by design (see
    # estimate_integer_frequency_offset). Must be corrected BEFORE the PRS-based sample
    # alignment, otherwise its cross-correlation would work on the wrong carriers too.
    integer_bins = estimate_integer_frequency_offset(probe, mode)
    if integer_bins != 0:
        freq_offset += integer_bins * bin_width_hz
        probe = correct_frequency_offset(
            frame_samples[: probe_symbols * mode.symbol_length], freq_offset, sample_rate
        )

    # Coarse integer sample alignment: the null symbol search only approximates the
    # frame start. The PRS cross-correlation (see find_integer_symbol_offset) finds the
    # actual start of the useful part and the frame start is corrected accordingly,
    # except when the detection is uncertain (see the docstring): then the last
    # certain correction is kept.
    integer_offset, confidence = find_integer_symbol_offset(
        probe[: mode.fft_length], mode.num_carriers, mode.fft_length
    )
    frame_reliable = confidence >= MIN_INTEGER_OFFSET_CONFIDENCE
    if frame_reliable:
        sample_correction = integer_offset - mode.guard_length
    else:
        # Uncertain detection (e.g. brief multipath in an SFN): then the cyclic prefix
        # estimate of freq_offset is typically unreliable for this frame as well (the
        # same disturbance affects both estimators). Fall back completely to the last
        # certain values, not only for the sample alignment, otherwise a corrupted
        # freq_offset would remain in effect.
        freq_offset = fallback_freq_offset
        sample_correction = fallback_sample_correction
    if sample_correction != 0:
        frame_start += sample_correction
        sliced = slice_frame(frame_start)
        if sliced is None:
            return None
        frame_samples, num_symbols = sliced

    # Only now, with the final frequency offset and the final frame position, once
    # over the whole frame.
    corrected = correct_frequency_offset(frame_samples, freq_offset, sample_rate)

    # FFT per symbol: the symbols are separated in the time domain, not in the
    # spectrum. The frame is laid out on the grid of the frame position in rows of one
    # symbol each (guard + useful part), and every row gets its own FFT over exactly
    # the useful part. The FFT then separates the carriers.
    spectra = demodulate_symbol(
        corrected.reshape(num_symbols, mode.symbol_length), mode.fft_length, mode.guard_length
    )

    carriers = extract_used_carriers(spectra, mode.num_carriers)
    channel_estimate = carriers[0] * np.conj(generate_reference_symbol(mode.num_carriers))

    # No fine timing correction: a fractional sample offset of the FFT window turns
    # each carrier by an angle that grows with k but is the same in every symbol of the
    # frame, so it drops out of z(l) · z*(l-1) exactly. The channel estimate above keeps
    # it on purpose: estimated from the phase slope across the carriers, it would be
    # biased by echoes towards their centre of power and shift the impulse response
    # (see `dsp.channel`) by that error.
    #
    # All symbols of the frame at once, one per row: each row against the one before it
    # (row 0 is the PRS), instead of a loop over the symbols with one small NumPy call
    # after another.
    differential = differential_demodulate(carriers[1:], carriers[:-1])
    constellation_points = list(correct_residual_phase(differential))

    return DemodulatedFrame(
        symbols=constellation_points,
        frequency_offset_hz=freq_offset,
        sample_correction=sample_correction,
        reliable=frame_reliable,
        channel_estimate=channel_estimate,
        spectra=spectra,
    )


class FrameDemodulator:
    """Demodulates frame after frame, carrying the last trustworthy sync parameters
    over as the fallback for the next one (see `demodulate_frame_symbols`).

    Shared by reception (`dsp.pipeline.DspWorker`) and the channel scanner
    (`dsp.scan.EnsembleProbe`).
    """

    def __init__(self, mode: TransmissionMode, sample_rate: float) -> None:
        self._mode = mode
        self._sample_rate = sample_rate
        self._last_sample_correction = 0
        self._last_frequency_offset = 0.0

    def reset(self) -> None:
        """Forgets the fallback values, e.g. after a channel change."""
        self._last_sample_correction = 0
        self._last_frequency_offset = 0.0

    def demodulate(
        self, samples: np.ndarray, null_start: int, max_data_symbols: int
    ) -> DemodulatedFrame | None:
        frame = demodulate_frame_symbols(
            samples,
            null_start=null_start,
            mode=self._mode,
            sample_rate=self._sample_rate,
            max_data_symbols=max_data_symbols,
            fallback_sample_correction=self._last_sample_correction,
            fallback_freq_offset=self._last_frequency_offset,
        )
        if frame is not None:
            self._last_sample_correction = frame.sample_correction
            self._last_frequency_offset = frame.frequency_offset_hz
        return frame

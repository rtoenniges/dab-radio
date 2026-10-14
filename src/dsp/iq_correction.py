"""Blind estimation and correction of DC offset and IQ imbalance."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Smoothing factor of the block-wise estimates (exponential moving average). The
# estimated quantities are physical properties of the tuner and change slowly at most
# (temperature, gain switching), while each single estimate from one read block is
# slightly noisy. 0.2 follows a real change within a few blocks but clearly smooths the
# noise of the individual estimate.
DEFAULT_SMOOTHING = 0.2


@dataclass(frozen=True)
class IqCorrectionReport:
    """The receiver errors currently estimated, for the GUI.

    Each value is None while the corresponding correction is switched off.
    """

    dc_offset: float | None  # magnitude of the DC offset, in units of the IQ samples
    quadrature_error_deg: float | None  # deviation of the I/Q phase from 90°
    gain_imbalance_db: float | None  # gain of I relative to Q


class IqCorrector:
    """Removes DC offset and IQ imbalance from the IQ stream.

    Both are errors of the receiver, not of the signal:

    * **DC offset**: the tuner's direct-conversion mixer produces a constant DC
      component (self-mixing of the local oscillator, ADC offset). In the spectrum it
      shows as a narrow spike exactly at the centre of the band.
    * **IQ imbalance**: the I and Q branches never have exactly the same gain and are
      never exactly 90° apart. This creates an attenuated mirror image of the signal:
      what lies at -f additionally appears, weakly, at +f.

    Both estimates are "blind", i.e. they need no known test signal. They only assume
    that the undistorted signal is circularly symmetric (I and Q uncorrelated and
    equally strong), which holds very well for DAB's OFDM signal of 1536 independently
    modulated subcarriers.

    What this means for DAB specifically: the subcarrier at the band centre (k=0) is
    not used by DAB (ETSI EN 300 401, clause 14.5), so a DC offset hits no used
    carrier. And since a DAB ensemble fills the whole band symmetrically around the
    centre frequency, the mirror image falls onto the signal itself and acts as extra
    noise, typically 30-40 dB below the signal. The correction therefore mainly
    improves the display (spectrum, constellation) and the signal-to-noise ratio at
    weak reception; a service that already decodes without errors does not get better.

    The corrector is stateful: the estimates are smoothed exponentially across read
    blocks (see `DEFAULT_SMOOTHING`).
    """

    def __init__(self, smoothing: float = DEFAULT_SMOOTHING) -> None:
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must lie in (0, 1].")
        self._smoothing = smoothing
        self._dc: complex | None = None
        self._quadrature_error: float | None = None  # share of I leaking into Q
        self._gain_ratio: float | None = None  # gain ratio I/Q

    @property
    def dc_offset(self) -> complex:
        """Most recently estimated DC offset (0 until something has been estimated)."""
        return 0j if self._dc is None else self._dc

    @property
    def quadrature_error_deg(self) -> float:
        """Most recently estimated quadrature error in degrees (deviation from 90°)."""
        return (
            0.0
            if self._quadrature_error is None
            else math.degrees(math.asin(max(-1.0, min(1.0, self._quadrature_error))))
        )

    @property
    def gain_imbalance_db(self) -> float:
        """Most recently estimated gain imbalance between I and Q in dB."""
        if self._gain_ratio is None or self._gain_ratio <= 0.0:
            return 0.0
        return 20.0 * math.log10(self._gain_ratio)

    def reset(self) -> None:
        """Discards the estimates (e.g. after a channel or gain change)."""
        self._dc = None
        self._quadrature_error = None
        self._gain_ratio = None

    def _smooth(self, previous: complex | None, measured: complex) -> complex:
        if previous is None:
            return measured
        return previous + self._smoothing * (measured - previous)

    def process(
        self, samples: np.ndarray, *, remove_dc: bool = True, correct_iq: bool = True
    ) -> np.ndarray:
        """Corrects one block of IQ samples.

        Regardless of the switches, the estimation always runs on the DC-free signal,
        because the moments `E[I*Q]` and `E[I^2]` would otherwise be dominated by the
        DC component. Only what is requested is then applied.

        Args:
            samples: complex IQ samples.
            remove_dc: subtract the DC offset.
            correct_iq: compensate the IQ imbalance.

        Returns:
            The corrected block. With both corrections off, `samples` is returned
            unchanged (no copy).
        """
        if not remove_dc and not correct_iq:
            return samples
        if samples.size == 0:
            return samples

        self._dc = self._smooth(self._dc, complex(samples.mean()))
        centred = samples - self._dc

        if correct_iq:
            self._update_iq_estimate(centred)

        corrected = centred if remove_dc else samples
        if not correct_iq or self._quadrature_error is None:
            return corrected

        # Gram-Schmidt orthogonalisation: subtract the share of I leaking into Q, then
        # bring Q to the same power as I.
        in_phase = corrected.real
        quadrature = (corrected.imag - self._quadrature_error * in_phase) * self._gain_ratio
        return in_phase + 1j * quadrature

    def _update_iq_estimate(self, centred: np.ndarray) -> None:
        in_phase = centred.real
        quadrature = centred.imag
        power_i = float(np.mean(in_phase * in_phase))
        if power_i <= 0.0:
            return  # silent block: keep the previous estimate

        quadrature_error = float(np.mean(in_phase * quadrature)) / power_i
        orthogonal = quadrature - quadrature_error * in_phase
        power_q = float(np.mean(orthogonal * orthogonal))
        if power_q <= 0.0:
            return
        gain_ratio = math.sqrt(power_i / power_q)

        self._quadrature_error = self._smooth(self._quadrature_error, quadrature_error)
        self._gain_ratio = self._smooth(self._gain_ratio, gain_ratio)

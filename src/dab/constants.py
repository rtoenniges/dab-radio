"""OFDM parameters of the DAB transmission modes and the receiver's sample rate."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TransmissionMode:
    """OFDM parameters of a DAB transmission mode (ETSI EN 300 401, table 3).

    All lengths are multiples of the elementary period T = 1/2 048 000 s, which at a
    sample rate of 2.048 MSps is simply a number of samples.
    """

    name: str
    fft_length: int  # T_u: useful symbol length (= number of OFDM subcarrier bins)
    guard_length: int  # T_g: guard interval (cyclic prefix) length
    num_carriers: int  # K: number of used subcarriers
    symbols_per_frame: int  # L: OFDM symbols per frame, not counting the null symbol
    null_symbol_length: int  # T_NULL

    @property
    def symbol_length(self) -> int:
        """T_s: total length of a regular OFDM symbol (guard interval + useful part)."""
        return self.fft_length + self.guard_length

    @property
    def frame_length(self) -> int:
        """T_F: total length of a transmission frame in samples."""
        return self.null_symbol_length + self.symbols_per_frame * self.symbol_length


# Mode I is the transmission mode used for terrestrial DAB+ in Europe (Band III,
# including Germany).
MODE_I = TransmissionMode(
    name="I",
    fft_length=2048,
    guard_length=504,
    num_carriers=1536,
    symbols_per_frame=76,
    null_symbol_length=2656,
)

SAMPLE_RATE = 2_048_000.0

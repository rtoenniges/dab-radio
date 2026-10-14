"""DAB signal processing: from IQ samples to decoded bits.

Everything here is implemented natively in NumPy (with numba for the two hot loops,
Viterbi and Reed-Solomon) so that every intermediate result can be visualised. The
entry point during reception is `dsp.pipeline.DspWorker`; the channel scanner uses the
same functions through `dsp.scan.EnsembleProbe`.
"""

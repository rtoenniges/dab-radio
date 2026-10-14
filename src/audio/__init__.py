"""Audio output stage: AAC decoding and playback.

Deliberately separate from `dsp`: everything under `dsp` is the self-implemented DAB
signal processing that is meant to be made visible. The actual AAC decoding, on the
other hand, uses an established external library (FAAD2), because rebuilding an audio
codec lies outside the project's goal.
"""

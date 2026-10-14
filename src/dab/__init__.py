"""DAB data layer: everything that interprets bits the DSP chain has already recovered.

The modules in this package know the DAB standards (FIC/FIG, PAD, MOT, slideshow,
character sets) but nothing about the IQ signal. They take bits and bytes from `dsp`
and turn them into ensembles, services, texts and images.
"""

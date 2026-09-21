"""Sharp-wave ripple and multiunit-burst detection from LFP and spikes.

Detectors take ``time``, the signal, ``speed``, and ``sampling_frequency`` and
return one DataFrame row per event; see the README's "Choosing a detector"
table for how they differ, and its "Published parameter values" table for the
range each threshold, duration and smoothing width takes in the literature and
where this package's defaults sit in it.
"""

from ripple_detection.core import (
    estimate_noise_threshold,
    filter_ripple_band,
    gaussian_smooth,
    get_envelope,
    get_multiunit_population_firing_rate,
    merge_close_events,
    minimum_sample_count,
    nearest_sample_index,
    normalize_signal,
    normalize_signal_manually,
    require_overlap,
    ripple_bandpass_filter,
    sample_count_within,
)
from ripple_detection.detectors import (
    Carey_candidate_detector,
    Karlsson_ripple_detector,
    Kay_ripple_detector,
    Long_sharp_wave_ripple_detector,
    Roumis_ripple_detector,
    Shvartsman_ripple_detector,
    Yu_ripple_detector,
    Zugaro_ripple_detector,
    get_Kay_ripple_consensus_trace,
    get_Yu_ripple_consensus_trace,
    multiunit_HSE_detector,
)
from ripple_detection.simulate import simulate_LFP, simulate_time

try:
    from ripple_detection._version import __version__
except ImportError:
    __version__ = "unknown"

__all__ = [
    "Carey_candidate_detector",
    "Karlsson_ripple_detector",
    "Kay_ripple_detector",
    "Long_sharp_wave_ripple_detector",
    "Roumis_ripple_detector",
    "Shvartsman_ripple_detector",
    "Yu_ripple_detector",
    "Zugaro_ripple_detector",
    "__version__",
    "estimate_noise_threshold",
    "filter_ripple_band",
    "gaussian_smooth",
    "get_Kay_ripple_consensus_trace",
    "get_Yu_ripple_consensus_trace",
    "get_envelope",
    "get_multiunit_population_firing_rate",
    "merge_close_events",
    "minimum_sample_count",
    "multiunit_HSE_detector",
    "nearest_sample_index",
    "normalize_signal",
    "normalize_signal_manually",
    "require_overlap",
    "ripple_bandpass_filter",
    "sample_count_within",
    "simulate_LFP",
    "simulate_time",
]

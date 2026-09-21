"""The package's public surface: what ``from ripple_detection import *`` gives you."""

import ripple_detection

DETECTORS = [
    "Kay_ripple_detector",
    "Karlsson_ripple_detector",
    "Roumis_ripple_detector",
    "Shvartsman_ripple_detector",
    "Yu_ripple_detector",
    "Zugaro_ripple_detector",
    "Long_sharp_wave_ripple_detector",
    "Carey_candidate_detector",
    "multiunit_HSE_detector",
]
HELPERS = [
    "filter_ripple_band",
    "ripple_bandpass_filter",
    "get_envelope",
    "gaussian_smooth",
    "normalize_signal",
    "normalize_signal_manually",
    "estimate_noise_threshold",
    "get_Kay_ripple_consensus_trace",
    "get_Yu_ripple_consensus_trace",
    "get_multiunit_population_firing_rate",
    "merge_close_events",
    "minimum_sample_count",
    "nearest_sample_index",
    "sample_count_within",
    "simulate_LFP",
    "simulate_time",
]


def test_all_lists_every_detector_and_documented_helper():
    assert set(DETECTORS + HELPERS) <= set(ripple_detection.__all__)


def test_every_name_in_all_is_importable():
    for name in ripple_detection.__all__:
        assert hasattr(ripple_detection, name), name


def test_all_has_no_duplicates():
    assert len(ripple_detection.__all__) == len(set(ripple_detection.__all__))

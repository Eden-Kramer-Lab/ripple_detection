"""The package's public surface: ``__all__`` is the contract, pinned exactly."""

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
REGISTRY = [
    "DETECTORS",
    "DetectorSpec",
    "SignalKind",
    "MULTIUNIT",
    "RAW_LFP_PAIR",
    "RIPPLE_BAND_LFP",
    "get_detector",
]
HELPERS = [
    "DEFAULT_RIPPLE_BAND",
    "DEFAULT_TRANSITION_WIDTH",
    "filter_ripple_band",
    "ripple_bandpass_filter",
    "get_envelope",
    "gaussian_smooth",
    "normalize_signal",
    "normalize_signal_manually",
    "estimate_noise_threshold",
    "noise_threshold_diagnostics",
    "NoiseThresholdDiagnostics",
    "get_Kay_ripple_consensus_trace",
    "get_Yu_ripple_consensus_trace",
    "get_multiunit_population_firing_rate",
    "load_literature_parameters",
    "exclude_close_events",
    "merge_close_events",
    "require_overlap",
    "minimum_sample_count",
    "sample_count_within",
    "simulate_LFP",
    "simulate_multichannel_LFP",
    "simulate_sharp_wave_ripple_pair",
    "simulate_multiunit",
    "simulate_session",
    "SimulatedSession",
    "simulate_time",
    "__version__",
]


def test_all_is_exactly_the_documented_surface():
    """An accidental export, or a lost one, fails here."""
    assert sorted(ripple_detection.__all__) == sorted(DETECTORS + REGISTRY + HELPERS)


def test_every_name_in_all_is_importable():
    for name in ripple_detection.__all__:
        assert hasattr(ripple_detection, name), name


def test_all_has_no_duplicates():
    assert len(ripple_detection.__all__) == len(set(ripple_detection.__all__))

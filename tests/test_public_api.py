"""The package's public surface: ``__all__`` is the contract, pinned exactly."""

import numpy as np
import pytest

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


class TestCallsWrittenFor1x:
    """A call written for 1.x, by a person or a language model trained on it,
    fails with the 2.0 change behind it and the call to write instead."""

    @pytest.fixture
    def inputs(self):
        time = np.arange(15_000) / 1500
        lfps = np.random.default_rng(0).standard_normal((15_000, 2))
        return time, ripple_detection.filter_ripple_band(lfps, 1500), np.zeros(15_000)

    def test_filter_without_a_rate(self):
        with pytest.raises(TypeError, match=r"filter_ripple_band.*required since 2\.0"):
            ripple_detection.filter_ripple_band(np.zeros((6000, 2)))

    def test_a_removed_keyword(self, inputs):
        with pytest.raises(TypeError, match=r"removed in 2.0: a time range is a mask"):
            ripple_detection.Kay_ripple_detector(
                *inputs, 1500, normalization_time_range=(0, 5)
            )

    def test_positional_tunables(self, inputs):
        with pytest.raises(
            TypeError, match=r"keyword-only.*speed_threshold=4.0, minimum_duration=0.015"
        ):
            ripple_detection.Kay_ripple_detector(*inputs, 1500, 4.0, 0.015)

    def test_a_near_miss_keyword(self, inputs):
        with pytest.raises(TypeError, match="did you mean close_ripple_threshold"):
            ripple_detection.Kay_ripple_detector(*inputs, 1500, close_event_threshold=0.05)

    def test_normalize_signal_with_time(self, inputs):
        time, lfps, _ = inputs
        with pytest.raises(TypeError, match="no longer takes time"):
            ripple_detection.normalize_signal(lfps[:, 0], time)

    def test_the_noise_generator_keyword(self):
        from ripple_detection.simulate import pink

        with pytest.raises(TypeError, match="renamed rng"):
            pink(100, state=np.random.RandomState(0))

    def test_a_renamed_keyword_names_its_own_replacement(self, inputs):
        """The Carey and Shvartsman tunables 1.x named differently."""
        time, lfps, speed = inputs
        multiunit = np.zeros((len(time), 3))
        with pytest.raises(TypeError, match=r"edge_threshold was renamed low_threshold"):
            ripple_detection.Carey_candidate_detector(
                time, lfps, multiunit, speed, 1500, edge_threshold=1.0
            )
        with pytest.raises(TypeError, match=r"peak_threshold was renamed high_threshold"):
            ripple_detection.Carey_candidate_detector(
                time, lfps, multiunit, speed, 1500, peak_threshold=3.0
            )
        with pytest.raises(
            TypeError, match=r"participation_threshold was split.*minimum_participating"
        ):
            ripple_detection.Shvartsman_ripple_detector(
                *inputs, 1500, participation_threshold=2
            )
        with pytest.raises(TypeError, match=r"pass normalization_method='manual'"):
            ripple_detection.Shvartsman_ripple_detector(
                *inputs, 1500, manual_normalization=True
            )
        with pytest.raises(TypeError, match=r"elec_baselines was renamed channel_baselines"):
            ripple_detection.Shvartsman_ripple_detector(
                *inputs, 1500, elec_baselines=[0.0, 0.0]
            )

    def test_a_hint_applies_only_where_the_replacement_exists(self, inputs):
        """`state` was renamed `rng` on the noise generators; the detectors and
        simulators take `random_state`, so they must not claim otherwise."""
        time, lfps, speed = inputs
        with pytest.raises(TypeError, match="takes no state") as info:
            ripple_detection.Long_sharp_wave_ripple_detector(time, lfps, speed, 1500, state=0)
        assert "rng" not in str(info.value)
        with pytest.raises(TypeError, match="takes no edge_threshold") as info:
            ripple_detection.Kay_ripple_detector(*inputs, 1500, edge_threshold=1.0)
        assert "low_threshold" not in str(info.value)
        with pytest.raises(TypeError, match="takes no manual_normalization") as info:
            ripple_detection.Kay_ripple_detector(*inputs, 1500, manual_normalization=True)
        assert "'manual'" not in str(info.value), "Kay has no manual method"

    def test_a_wrapped_detector_keeps_its_signature_and_name(self):
        import inspect

        detector = ripple_detection.Kay_ripple_detector
        assert detector.__name__ == "Kay_ripple_detector"
        assert list(inspect.signature(detector).parameters)[:4] == [
            "time",
            "filtered_lfps",
            "speed",
            "sampling_frequency",
        ]


class TestLlmsTxt:
    """llms.txt, the short map for language models, stays true to the code."""

    TEXT = (__import__("pathlib").Path(__file__).parents[1] / "llms.txt").read_text()

    def test_its_example_runs(self):
        import re

        (block,) = re.findall(r"```python\n(.*?)```", self.TEXT, re.DOTALL)
        namespace: dict[str, object] = {}
        exec(block, namespace)
        assert len(namespace["events"]) > 0

    def test_it_names_every_registered_detector(self):
        missing = [name for name in ripple_detection.DETECTORS if f"`{name}`" not in self.TEXT]
        assert missing == []

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
    "exclude_movement",
    "exclude_movement_by_majority",
    "merge_close_events",
    "require_overlap",
    "segment_boolean_series",
    "threshold_by_zscore",
    "minimum_sample_count",
    "sample_count_within",
    "simulate_LFP",
    "simulate_multichannel_LFP",
    "simulate_sharp_wave_ripple_pair",
    "simulate_multiunit",
    "simulate_session",
    "SimulatedSession",
    "simulate_time",
    "pink",
    "white",
    "brown",
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
            TypeError, match=r"1.x's order.*speed_threshold=4.0, minimum_duration=0.015"
        ):
            ripple_detection.Kay_ripple_detector(*inputs, 1500, 4.0, 0.015)

    def test_positional_values_are_named_in_the_1x_order(self):
        """HSE's 1.x flag sat sixth, before normalization_method; the hint
        follows that order, not 2.0's, and says the flag was removed."""
        time = np.arange(3000) / 1500
        spikes = np.zeros((3000, 3))
        with pytest.raises(
            TypeError,
            match=r"use_speed_threshold_for_zscore=True, normalization_method='zscore'.*"
            r"use_speed_threshold_for_zscore was removed",
        ):
            ripple_detection.multiunit_HSE_detector(
                time, spikes, np.zeros(3000), 1500, 4.0, 0.015, 2.0, 0.015, 0.0, True, "zscore"
            )

    def test_a_function_new_in_2_does_not_guess_a_1x_order(self):
        time = np.arange(3000) / 1500
        with pytest.raises(TypeError, match=r"pass the 1 extra value\(s\) by name"):
            ripple_detection.Zugaro_ripple_detector(
                time, np.zeros((3000, 2)), np.zeros(3000), 1500, 4.0
            )

    WRAPPED = (
        *(getattr(ripple_detection, name) for name in DETECTORS),
        ripple_detection.filter_ripple_band,
        ripple_detection.normalize_signal,
        ripple_detection.simulate_LFP,
        ripple_detection.simulate_multichannel_LFP,
        ripple_detection.simulate_sharp_wave_ripple_pair,
        ripple_detection.simulate_multiunit,
        ripple_detection.simulate_session,
    )

    @pytest.mark.parametrize("function", WRAPPED, ids=lambda function: function.__name__)
    def test_every_public_entry_point_explains_a_bad_call(self, function):
        assert hasattr(function, "__wrapped__")
        with pytest.raises(TypeError, match=rf"^{function.__name__}\(\): .*not_a_parameter"):
            function(not_a_parameter=1)

    @pytest.mark.parametrize("name", ["pink", "white", "brown"])
    def test_the_noise_functions_explain_a_bad_call(self, name):
        from ripple_detection import simulate

        with pytest.raises(TypeError, match=rf"^{name}\(\): .*not_a_parameter"):
            getattr(simulate, name)(100, not_a_parameter=1)

    def test_only_the_unknown_keyword_is_reported(self, inputs):
        with pytest.raises(TypeError) as raised:
            ripple_detection.Kay_ripple_detector(
                *inputs, 1500, speed_threshold=4.0, close_event_threshold=0.05
            )
        assert "takes no close_event_threshold" in str(raised.value)
        assert "speed_threshold;" not in str(raised.value)

    def test_a_near_miss_keyword(self, inputs):
        with pytest.raises(TypeError, match="did you mean close_ripple_threshold"):
            ripple_detection.Kay_ripple_detector(*inputs, 1500, close_event_threshold=0.05)

    def test_normalize_signal_with_time(self, inputs):
        time, lfps, _ = inputs
        with pytest.raises(TypeError, match="no longer takes time"):
            ripple_detection.normalize_signal(lfps[:, 0], time)

    def test_a_random_state_passed_by_position(self):
        from ripple_detection.simulate import pink

        with pytest.raises(TypeError, match="not a RandomState"):
            pink(100, np.random.RandomState(0))

    def test_the_noise_generator_keyword(self):
        from ripple_detection.simulate import pink

        with pytest.raises(TypeError, match="renamed rng"):
            pink(100, state=np.random.RandomState(0))

    def test_the_pre_release_seed_keyword(self, inputs):
        time, _, speed = inputs
        with pytest.raises(TypeError, match="random_state was renamed rng"):
            ripple_detection.simulate_session(time, [1.0], random_state=0)
        with pytest.raises(TypeError, match="random_state was renamed rng"):
            ripple_detection.Long_sharp_wave_ripple_detector(
                time, np.zeros((len(time), 2)), speed, 1500, random_state=0
            )

    def test_every_seed_is_called_rng(self):
        """One name for a seed or Generator across the package."""
        import inspect

        from ripple_detection.simulate import brown, pink, white

        functions = [
            getattr(ripple_detection, name)
            for name in ripple_detection.__all__
            if inspect.isfunction(getattr(ripple_detection, name))
        ]
        seed_names = {"rng", "random_state", "seed", "state"}
        seeded = {
            (function.__name__, name)
            for function in [*functions, pink, white, brown]
            for name in inspect.signature(function).parameters
            if name in seed_names
        }
        assert seeded == {
            (function, "rng")
            for function in (
                "Long_sharp_wave_ripple_detector",
                "simulate_LFP",
                "simulate_multichannel_LFP",
                "simulate_multiunit",
                "simulate_session",
                "simulate_sharp_wave_ripple_pair",
                "pink",
                "white",
                "brown",
            )
        }

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
        for old, new in (
            ("spike_kernel_sigma", "spike_smoothing_sigma"),
            ("baseline_sigma", "baseline_smoothing_sigma"),
            ("state_minimum_length", "minimum_state_duration"),
        ):
            with pytest.raises(TypeError, match=rf"{old} was renamed {new}"):
                ripple_detection.Carey_candidate_detector(
                    time, lfps, multiunit, speed, 1500, **{old: 0.05}
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
        with pytest.raises(TypeError, match=r"raw_lfps was renamed raw_lfp_pair"):
            ripple_detection.Long_sharp_wave_ripple_detector(
                time=time,
                raw_lfps=np.zeros((len(time), 2)),
                speed=speed,
                sampling_frequency=1500,
            )
        with pytest.raises(TypeError, match=r"elec_baselines was renamed channel_baselines"):
            ripple_detection.Shvartsman_ripple_detector(
                *inputs, 1500, elec_baselines=[0.0, 0.0]
            )

    def test_a_hint_applies_only_where_the_replacement_exists(self, inputs):
        """`state` was renamed `rng`; a detector that draws no random numbers
        takes neither, so it must not claim otherwise."""
        with pytest.raises(TypeError, match="takes no state") as info:
            ripple_detection.Kay_ripple_detector(*inputs, 1500, state=0)
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

    def test_the_migration_guide_is_the_one_every_document_points_to(self):
        root = __import__("pathlib").Path(__file__).parents[1]
        assert (root / "MIGRATING.md").is_file()
        for document in ("README.md", "CHANGELOG.md", "llms.txt"):
            assert "MIGRATING.md" in (root / document).read_text(), document

    def test_it_names_every_registered_detector(self):
        missing = [name for name in ripple_detection.DETECTORS if f"`{name}`" not in self.TEXT]
        assert missing == []

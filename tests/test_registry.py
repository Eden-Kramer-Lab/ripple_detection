"""The registry that lets a caller resolve a detector by name."""

import dataclasses
import inspect

import numpy as np
import pytest

import ripple_detection
from ripple_detection import (
    DETECTORS,
    MULTIUNIT,
    RAW_LFP,
    RIPPLE_BAND_LFP,
    DetectorSpec,
    Kay_ripple_detector,
    get_detector,
)


def _exported_detectors():
    """Every detector the package exports, by the only definition that holds:
    a public name ending in ``_detector`` that is defined in the detectors
    module. ``get_detector`` ends the same way but lives in the registry."""
    return {
        name
        for name in ripple_detection.__all__
        if name.endswith("_detector")
        and getattr(getattr(ripple_detection, name), "__module__", "").startswith(
            "ripple_detection.detectors"
        )
    }


def test_every_exported_detector_is_registered():
    """A detector added to the package cannot be left out of the registry."""
    assert _exported_detectors() == set(DETECTORS)


def test_the_registry_holds_every_detector_the_package_has():
    """Nine of them, so an accidental deletion from either side is visible."""
    assert len(DETECTORS) == 9
    assert len(_exported_detectors()) == 9


def test_every_registered_name_resolves_to_that_detector():
    for name, spec in DETECTORS.items():
        assert spec.name == name
        assert spec.detector is getattr(ripple_detection, name)


def test_get_detector_returns_the_spec():
    spec = get_detector("Kay_ripple_detector")

    assert spec.detector is ripple_detection.Kay_ripple_detector
    assert spec.inputs == ("ripple_band_lfp",)


def test_unknown_name_raises_and_names_the_alternatives():
    with pytest.raises(KeyError, match="Kay_ripple_detector") as raised:
        get_detector("Kay_detector")
    assert "\\n" not in repr(raised.value), "a KeyError shows its message by repr"


@pytest.mark.parametrize(
    ("name", "inputs"),
    [
        ("Kay_ripple_detector", ("ripple_band_lfp",)),
        ("Karlsson_ripple_detector", ("ripple_band_lfp",)),
        ("Roumis_ripple_detector", ("ripple_band_lfp",)),
        ("Shvartsman_ripple_detector", ("ripple_band_lfp",)),
        ("Yu_ripple_detector", ("ripple_band_lfp",)),
        ("Zugaro_ripple_detector", ("ripple_band_lfp",)),
        ("Long_sharp_wave_ripple_detector", ("raw_lfp",)),
        ("Carey_candidate_detector", ("ripple_band_lfp", "multiunit")),
        ("multiunit_HSE_detector", ("multiunit",)),
    ],
)
def test_each_detector_declares_what_it_needs(name, inputs):
    """The point of the registry: which signal a detector takes, not just its name.

    Long takes raw LFP in the slot where the ripple-band detectors take
    filtered LFP, so a caller that resolves by name alone would feed it
    filtered data; its required ``sharp_wave_lfp`` makes that call fail.
    """
    assert get_detector(name).inputs == inputs


@pytest.mark.parametrize(
    ("name", "keyword_inputs"),
    [
        ("Long_sharp_wave_ripple_detector", {"sharp_wave_lfp": "raw_lfp"}),
        ("Carey_candidate_detector", {"theta_lfp": "raw_lfp"}),
        ("Kay_ripple_detector", {}),
        ("multiunit_HSE_detector", {}),
    ],
)
def test_each_detector_declares_its_keyword_signals(name, keyword_inputs):
    """Signals passed by name are signals, not tunables."""
    spec = get_detector(name)
    assert dict(spec.keyword_inputs) == keyword_inputs
    assert not set(keyword_inputs) & set(spec.parameters)


def test_the_sharp_wave_channel_is_required_and_the_theta_channel_is_not():
    assert get_detector("Long_sharp_wave_ripple_detector").required_keyword_inputs == (
        "sharp_wave_lfp",
    )
    assert get_detector("Carey_candidate_detector").required_keyword_inputs == ()


def test_declared_inputs_match_the_signature_in_kind_and_position():
    """The spec's promise is positional: time, the listed signals in order,
    speed, sampling_frequency. A mislabeled kind (Long as ripple-band) or a
    swapped pair (Carey's spikes before its LFP) fails here, independently of
    the hand-written table above."""
    kind_of_parameter = {
        "filtered_lfps": RIPPLE_BAND_LFP,
        "raw_lfp": RAW_LFP,
        "multiunit": MULTIUNIT,
    }
    for name, spec in DETECTORS.items():
        parameters = list(inspect.signature(spec.detector).parameters)
        n_signals = len(spec.inputs)

        assert parameters[0] == "time", name
        assert parameters[n_signals + 1 : n_signals + 3] == ["speed", "sampling_frequency"], (
            name
        )
        assert (
            tuple(kind_of_parameter[p] for p in parameters[1 : n_signals + 1]) == spec.inputs
        )


def test_registry_is_read_only():
    """A caller cannot register a detector by mutating the mapping."""
    with pytest.raises(TypeError):
        DETECTORS["Kay_ripple_detector"] = None


def test_spec_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        get_detector("Kay_ripple_detector").inputs = ("multiunit",)


class TestSpecConstruction:
    """A spec is checked when it is built, not only by the test that walks the
    built-in table, so a hand-made spec cannot name the wrong signal parameters."""

    def test_a_list_of_inputs_is_stored_as_a_tuple(self):
        spec = DetectorSpec(Kay_ripple_detector, [RIPPLE_BAND_LFP])
        assert spec.inputs == (RIPPLE_BAND_LFP,)

    def test_an_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="not signal kinds"):
            DetectorSpec(Kay_ripple_detector, ("ripple_lfp",))

    def test_inputs_that_do_not_match_the_signature_raise(self):
        with pytest.raises(ValueError, match="takes time, the signals, speed"):
            DetectorSpec(Kay_ripple_detector, (RIPPLE_BAND_LFP, MULTIUNIT))

    def test_a_keyword_input_the_detector_does_not_take_raises(self):
        with pytest.raises(ValueError, match="theta_lfp is not a keyword-only parameter"):
            DetectorSpec(Kay_ripple_detector, (RIPPLE_BAND_LFP,), {"theta_lfp": RAW_LFP})

    def test_an_undeclared_required_keyword_raises(self):
        """Every tunable has a default, so a keyword without one must be a
        declared signal."""
        from ripple_detection import Long_sharp_wave_ripple_detector

        with pytest.raises(ValueError, match="sharp_wave_lfp has no default"):
            DetectorSpec(Long_sharp_wave_ripple_detector, (RAW_LFP,))


class TestCheckInputs:
    """The spec checks what an array can show about a signal: count, shape, and
    for spikes that the values are counts. It does not judge filtered against raw."""

    def test_raw_lfp_is_one_channel(self):
        spec = get_detector("Long_sharp_wave_ripple_detector")
        spec.check_inputs(np.zeros(600), sharp_wave_lfp=np.zeros(600))
        spec.check_inputs(np.zeros((600, 1)), sharp_wave_lfp=np.zeros((600, 1)))
        with pytest.raises(ValueError, match="one channel"):
            spec.check_inputs(np.zeros((600, 2)), sharp_wave_lfp=np.zeros(600))

    def test_a_required_keyword_signal_must_be_given(self):
        spec = get_detector("Long_sharp_wave_ripple_detector")
        with pytest.raises(ValueError, match="needs sharp_wave_lfp"):
            spec.check_inputs(np.zeros(600))

    def test_an_optional_keyword_signal_is_checked_when_given(self):
        spec = get_detector("Carey_candidate_detector")
        lfps, counts = np.zeros((600, 3)), np.zeros((600, 4))
        spec.check_inputs(lfps, counts)
        spec.check_inputs(lfps, counts, theta_lfp=np.zeros(600))
        with pytest.raises(ValueError, match="one channel"):
            spec.check_inputs(lfps, counts, theta_lfp=np.zeros((600, 2)))

    def test_an_unknown_keyword_signal_raises(self):
        with pytest.raises(ValueError, match="takes no keyword signal theta_lfp"):
            get_detector("Kay_ripple_detector").check_inputs(
                np.zeros((600, 2)), theta_lfp=np.zeros(600)
            )

    def test_spike_counts_must_be_non_negative_whole_numbers(self):
        spec = get_detector("multiunit_HSE_detector")
        counts = np.random.default_rng(0).poisson(0.1, (600, 4)).astype(float)
        counts[10, 0] = np.nan  # missing is allowed
        spec.check_inputs(counts)
        with pytest.raises(ValueError, match="whole numbers"):
            spec.check_inputs(counts * 0.5)
        with pytest.raises(ValueError, match="whole numbers"):
            spec.check_inputs(-counts)

    def test_signal_count_and_dimensions(self):
        spec = get_detector("Carey_candidate_detector")
        lfps, counts = np.zeros((600, 3)), np.zeros((600, 4))
        spec.check_inputs(lfps, counts)
        with pytest.raises(ValueError, match="takes 2 signal"):
            spec.check_inputs(lfps)
        with pytest.raises(ValueError, match="2-D"):
            spec.check_inputs(lfps[:, 0], counts)

    def test_ripple_band_lfp_of_any_content_passes(self):
        """Raw and filtered LFP have the same shape; the check makes no claim."""
        get_detector("Kay_ripple_detector").check_inputs(
            np.random.default_rng(0).standard_normal((600, 4))
        )


class TestParameters:
    def test_parameters_are_the_keyword_only_tunables_with_defaults(self):
        spec = get_detector("Kay_ripple_detector")
        assert spec.parameters["zscore_threshold"] == 2.0
        assert spec.parameters["speed_threshold"] == 4.0
        assert "time" not in spec.parameters
        assert "sampling_frequency" not in spec.parameters

    def test_every_detector_runs_with_its_default_parameters_spelled_out(self):
        """The mapping is exactly what **parameters needs."""
        for spec in DETECTORS.values():
            defaults = spec.parameters
            assert all(value is not inspect.Parameter.empty for value in defaults.values()), (
                spec.name
            )
            spec.check_parameters(defaults)

    def test_signal_parameters_follow_inputs(self):
        assert get_detector("Carey_candidate_detector").signal_parameters == (
            "filtered_lfps",
            "multiunit",
        )
        assert get_detector("Long_sharp_wave_ripple_detector").signal_parameters == (
            "raw_lfp",
        )

    def test_a_signal_new_in_2_is_named_after_its_kind(self):
        """So `signal_parameters`, `inputs` and the SimulatedSession field
        agree; only 1.x's `filtered_lfps` keeps a name of its own."""
        from ripple_detection import SimulatedSession

        fields = set(SimulatedSession.__dataclass_fields__)
        for spec in DETECTORS.values():
            for kind, parameter in zip(spec.inputs, spec.signal_parameters, strict=True):
                if kind != RIPPLE_BAND_LFP:
                    assert parameter == kind, spec.name
                    assert parameter in fields, spec.name
            for parameter in spec.required_keyword_inputs:
                assert parameter in fields, spec.name

    def test_check_parameters_names_the_unknown_key(self):
        with pytest.raises(ValueError, match="does not take z_score_threshold"):
            get_detector("Kay_ripple_detector").check_parameters({"z_score_threshold": 3.0})

    def test_every_detector_has_parameter_checks_named_as_its_parameters(self):
        """A checker is passed the parameters it names, so a name that drifted
        from the detector's would be checked against nothing: every one it
        requires must be the detector's."""
        from ripple_detection.registry import _PARAMETER_CHECKS

        for spec in DETECTORS.values():
            check = _PARAMETER_CHECKS[spec.detector]
            required = {
                name
                for name, parameter in inspect.signature(check).parameters.items()
                if parameter.default is inspect.Parameter.empty
            }
            assert required <= set(spec.parameters), (
                spec.name,
                required - set(spec.parameters),
            )

    def test_check_parameters_suggests_the_parameter_meant(self):
        with pytest.raises(ValueError, match="did you mean zscore_threshold"):
            get_detector("Kay_ripple_detector").check_parameters({"z_score_threshold": 3.0})

    def test_check_parameters_explains_an_argument_removed_in_2(self):
        with pytest.raises(ValueError, match=r"removed in 2\.0[\s\S]*normalization_mask"):
            get_detector("Kay_ripple_detector").check_parameters(
                {"normalization_time_range": (0.0, 10.0)}
            )

    @pytest.mark.parametrize(
        "name",
        [name for name, spec in DETECTORS.items() if "minimum_duration" in spec.parameters],
    )
    def test_check_parameters_catches_milliseconds_given_as_seconds(self, name):
        with pytest.raises(ValueError, match=r"For 15 ms pass 0\.015"):
            get_detector(name).check_parameters({"minimum_duration": 15})

    @pytest.mark.parametrize(
        ("name", "parameters"),
        [
            ("Kay_ripple_detector", {"zscore_threshold": "2"}),
            ("Zugaro_ripple_detector", {"high_threshold": "5"}),
            ("Yu_ripple_detector", {"smoothing_sigma": None}),
            ("Long_sharp_wave_ripple_detector", {"minimum_separation": "0.05"}),
            ("Carey_candidate_detector", {"spike_cap": "2"}),
            ("multiunit_HSE_detector", {"close_event_threshold": "0"}),
        ],
    )
    def test_check_parameters_catches_a_value_that_is_no_number(self, name, parameters):
        with pytest.raises(TypeError, match="must be a number"):
            get_detector(name).check_parameters(parameters)

    @pytest.mark.parametrize(
        ("name", "parameters", "match"),
        [
            ("Kay_ripple_detector", {"speed_threshold": -1.0}, "speed_threshold"),
            ("Karlsson_ripple_detector", {"maximum_duration": 0.01}, "below minimum"),
            ("Roumis_ripple_detector", {"close_ripple_threshold": 50.0}, "50.0 ms"),
            ("Shvartsman_ripple_detector", {"normalization_method": "manual"}, "needs"),
            ("Zugaro_ripple_detector", {"low_threshold": 6.0}, "above high_threshold"),
            ("Long_sharp_wave_ripple_detector", {"sharp_wave_percentile": 100.0}, "(0, 100)"),
            ("Carey_candidate_detector", {"threshold_method": "median"}, "threshold_method"),
            ("multiunit_HSE_detector", {"smoothing_sigma": 15.0}, "15.0 ms"),
        ],
    )
    def test_check_parameters_runs_each_detector_s_range_checks(self, name, parameters, match):
        with pytest.raises(ValueError, match=match):
            get_detector(name).check_parameters(parameters)


@pytest.fixture(scope="module")
def results():
    """Every registered detector's result on one simulated session."""
    from ripple_detection import filter_ripple_band
    from ripple_detection.simulate import simulate_session, simulate_time

    time = simulate_time(45_000, 1500)
    session = simulate_session(time, [5.0, 10.0, 15.0, 20.0, 25.0], rng=0)
    signals = {
        RIPPLE_BAND_LFP: filter_ripple_band(session.lfps, 1500),
        RAW_LFP: session.raw_lfp,
        MULTIUNIT: session.multiunit,
    }
    return {
        name: spec.detector(
            time,
            *(signals[kind] for kind in spec.inputs),
            session.speed,
            1500,
            **{name: getattr(session, name) for name in spec.required_keyword_inputs},
        )
        for name, spec in DETECTORS.items()
    }


class TestDescribe:
    """describe() is plain data a pipeline or a language model can read, and a
    test holds it to the code so it cannot drift."""

    @pytest.mark.parametrize("name", list(DETECTORS))
    def test_round_trips_through_json(self, name):
        import json

        description = get_detector(name).describe()
        assert json.loads(json.dumps(description)) == description

    @pytest.mark.parametrize("name", list(DETECTORS))
    def test_positional_arguments_follow_the_signature(self, name):
        spec = get_detector(name)
        described = [argument["name"] for argument in spec.describe()["positional"]]
        assert described == list(inspect.signature(spec.detector).parameters)[: len(described)]
        assert described[-2:] == ["speed", "sampling_frequency"]

    @pytest.mark.parametrize("name", list(DETECTORS))
    def test_every_tunable_has_a_unit_and_a_description(self, name):
        parameters = get_detector(name).describe()["parameters"]
        assert list(parameters) == list(get_detector(name).parameters)
        assert all(entry["description"] for entry in parameters.values())

    @pytest.mark.parametrize("name", list(DETECTORS))
    def test_keyword_signals_are_described_with_their_kind(self, name):
        spec = get_detector(name)
        described = spec.describe()["keyword_signals"]
        assert [entry["name"] for entry in described] == list(spec.keyword_inputs)
        for entry in described:
            assert entry["kind"] == spec.keyword_inputs[entry["name"]]
            assert entry["required"] == (entry["name"] in spec.required_keyword_inputs)
            assert entry["description"]

    def test_the_call_names_a_required_keyword_signal(self):
        call = get_detector("Long_sharp_wave_ripple_detector").describe()["call"]
        assert call == (
            "Long_sharp_wave_ripple_detector(time, raw_lfp, speed, sampling_frequency, "
            "sharp_wave_lfp=sharp_wave_lfp, **parameters)"
        )

    def test_no_description_is_left_without_a_parameter(self):
        from ripple_detection import _descriptions

        used = {parameter for spec in DETECTORS.values() for parameter in spec.parameters}
        assert set(_descriptions.PARAMETERS) == used
        for detector, parameter in _descriptions.OVERRIDES:
            assert parameter in DETECTORS[detector].parameters
        for detector, signal in _descriptions.SIGNAL_ROLES:
            spec = DETECTORS[detector]
            assert signal in (*spec.signal_parameters, *spec.keyword_inputs)

    @pytest.mark.parametrize("name", list(DETECTORS))
    def test_columns_are_what_the_detector_returns_in_order(self, name, results):
        columns = get_detector(name).describe()["returns"]["columns"]
        assert list(columns) == list(results[name].columns)
        assert results[name].index.name == "event_number"

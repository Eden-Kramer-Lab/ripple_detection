"""The registry that lets a caller resolve a detector by name."""

import dataclasses
import inspect

import numpy as np
import pytest

import ripple_detection
from ripple_detection import (
    DETECTORS,
    MULTIUNIT,
    RAW_LFP_PAIR,
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
    with pytest.raises(KeyError, match="Kay_ripple_detector"):
        get_detector("Kay_detector")


@pytest.mark.parametrize(
    ("name", "inputs"),
    [
        ("Kay_ripple_detector", ("ripple_band_lfp",)),
        ("Karlsson_ripple_detector", ("ripple_band_lfp",)),
        ("Roumis_ripple_detector", ("ripple_band_lfp",)),
        ("Shvartsman_ripple_detector", ("ripple_band_lfp",)),
        ("Yu_ripple_detector", ("ripple_band_lfp",)),
        ("Zugaro_ripple_detector", ("ripple_band_lfp",)),
        ("Long_sharp_wave_ripple_detector", ("raw_lfp_pair",)),
        ("Carey_candidate_detector", ("ripple_band_lfp", "multiunit")),
        ("multiunit_HSE_detector", ("multiunit",)),
    ],
)
def test_each_detector_declares_what_it_needs(name, inputs):
    """The point of the registry: which signal a detector takes, not just its name.

    Long takes raw two-channel LFP through a signature identical to the
    ripple-band detectors', so a caller that resolves by name alone would feed
    it filtered data and get plausible nonsense.
    """
    assert get_detector(name).inputs == inputs


def test_declared_inputs_match_the_signature_in_kind_and_position():
    """The spec's promise is positional: time, the listed signals in order,
    speed, sampling_frequency. A mislabeled kind (Long as ripple-band) or a
    swapped pair (Carey's spikes before its LFP) fails here, independently of
    the hand-written table above."""
    kind_of_parameter = {
        "filtered_lfps": RIPPLE_BAND_LFP,
        "raw_lfps": RAW_LFP_PAIR,
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


class TestCheckInputs:
    """The spec checks what an array can show about a signal: count, shape, and
    for spikes that the values are counts. It does not judge filtered against raw."""

    def test_raw_pair_needs_two_channels(self):
        spec = get_detector("Long_sharp_wave_ripple_detector")
        spec.check_inputs(np.zeros((600, 2)))
        with pytest.raises(ValueError, match="two channels"):
            spec.check_inputs(np.zeros((600, 1)))

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
            "raw_lfps",
        )

    def test_check_parameters_names_the_unknown_key(self):
        with pytest.raises(ValueError, match="does not take z_score_threshold"):
            get_detector("Kay_ripple_detector").check_parameters({"z_score_threshold": 3.0})


@pytest.fixture(scope="module")
def results():
    """Every registered detector's result on one simulated session."""
    from ripple_detection import filter_ripple_band
    from ripple_detection.simulate import simulate_session, simulate_time

    time = simulate_time(45_000, 1500)
    session = simulate_session(time, [5.0, 10.0, 15.0, 20.0, 25.0], random_state=0)
    signals = {
        RIPPLE_BAND_LFP: filter_ripple_band(session.lfps, 1500),
        RAW_LFP_PAIR: session.raw_lfp_pair,
        MULTIUNIT: session.multiunit,
    }
    return {
        name: spec.detector(
            time, *(signals[kind] for kind in spec.inputs), session.speed, 1500
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

    def test_no_description_is_left_without_a_parameter(self):
        from ripple_detection import _descriptions

        used = {parameter for spec in DETECTORS.values() for parameter in spec.parameters}
        assert set(_descriptions.PARAMETERS) == used
        for detector, parameter in _descriptions.OVERRIDES:
            assert parameter in DETECTORS[detector].parameters

    @pytest.mark.parametrize("name", list(DETECTORS))
    def test_columns_are_what_the_detector_returns_in_order(self, name, results):
        columns = get_detector(name).describe()["returns"]["columns"]
        assert list(columns) == list(results[name].columns)
        assert results[name].index.name == "event_number"

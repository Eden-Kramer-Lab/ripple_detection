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
        counts = np.random.RandomState(0).poisson(0.1, (600, 4)).astype(float)
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
            np.random.RandomState(0).randn(600, 4)
        )


class TestParameters:
    def test_parameters_are_the_keyword_only_tunables_with_defaults(self):
        spec = get_detector("Kay_ripple_detector")
        assert spec.parameters["zscore_threshold"] == 2.0
        assert spec.parameters["speed_threshold"] == 4.0
        assert "time" not in spec.parameters and "sampling_frequency" not in spec.parameters

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

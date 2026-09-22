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
    filter_ripple_band,
    get_detector,
)
from ripple_detection.simulate import brown


def _exported_detectors():
    """Every detector the package exports, by the only definition that holds:
    a public name ending in ``_detector`` that is defined in the detectors
    module. ``get_detector`` ends the same way but lives in the registry."""
    return {
        name
        for name in ripple_detection.__all__
        if name.endswith("_detector")
        and getattr(getattr(ripple_detection, name), "__module__", "")
        == "ripple_detection.detectors"
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
    """The spec can tell whether what a caller has is what the detector takes:
    the two silent mismatches, raw for filtered and filtered for raw, raise."""

    FS = 1500

    @pytest.fixture
    def raw(self):
        state = np.random.RandomState(0)
        return np.column_stack([brown(6000, state), brown(6000, state)])

    @pytest.fixture
    def filtered(self, raw):
        return filter_ripple_band(raw, self.FS)

    def test_ripple_band_detectors_accept_filtered_and_reject_raw(self, raw, filtered):
        spec = get_detector("Kay_ripple_detector")
        spec.check_inputs(filtered, sampling_frequency=self.FS)
        with pytest.raises(ValueError, match="looks unfiltered"):
            spec.check_inputs(raw, sampling_frequency=self.FS)

    def test_raw_pair_detector_accepts_raw_and_rejects_filtered(self, raw, filtered):
        spec = get_detector("Long_sharp_wave_ripple_detector")
        spec.check_inputs(raw, sampling_frequency=self.FS)
        with pytest.raises(ValueError, match="looks band-pass filtered"):
            spec.check_inputs(filtered, sampling_frequency=self.FS)
        with pytest.raises(ValueError, match="two channels"):
            spec.check_inputs(raw[:, :1], sampling_frequency=self.FS)

    def test_spike_counts_must_be_non_negative_whole_numbers(self):
        spec = get_detector("multiunit_HSE_detector")
        counts = np.random.RandomState(0).poisson(0.1, (6000, 4)).astype(float)
        counts[10, 0] = np.nan  # missing is allowed
        spec.check_inputs(counts, sampling_frequency=self.FS)
        with pytest.raises(ValueError, match="whole numbers"):
            spec.check_inputs(counts * 0.5, sampling_frequency=self.FS)
        with pytest.raises(ValueError, match="whole numbers"):
            spec.check_inputs(-counts, sampling_frequency=self.FS)

    def test_signal_count_and_dimensions(self, filtered):
        spec = get_detector("Carey_candidate_detector")
        counts = np.zeros((6000, 4))
        spec.check_inputs(filtered, counts, sampling_frequency=self.FS)
        with pytest.raises(ValueError, match="takes 2 signal"):
            spec.check_inputs(filtered, sampling_frequency=self.FS)
        with pytest.raises(ValueError, match="2-D"):
            spec.check_inputs(filtered[:, 0], counts, sampling_frequency=self.FS)

    def test_missing_samples_are_left_out_of_the_judgement(self, filtered):
        filtered[1000:1500] = np.nan
        get_detector("Kay_ripple_detector").check_inputs(filtered, sampling_frequency=self.FS)

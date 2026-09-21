"""The registry that lets a caller resolve a detector by name."""

import dataclasses
import inspect

import pytest

import ripple_detection
from ripple_detection import DETECTORS, get_detector


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


def test_declared_inputs_match_the_signature():
    """One entry per signal the detector actually takes, in order."""
    signal_arguments = {"filtered_lfps", "lfp", "multiunit"}
    for name, spec in DETECTORS.items():
        parameters = list(inspect.signature(spec.detector).parameters)
        signals = [p for p in parameters if p in signal_arguments]

        assert len(signals) == len(spec.inputs), name


def test_registry_is_read_only():
    """A caller cannot register a detector by mutating the mapping."""
    with pytest.raises(TypeError):
        DETECTORS["Kay_ripple_detector"] = None


def test_spec_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        get_detector("Kay_ripple_detector").name = "other"

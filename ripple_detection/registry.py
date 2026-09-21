"""Resolve a detector by name, and say what signal it needs.

A caller that stores a detector's name rather than importing it, such as a
database-backed pipeline, needs two things from this package: the callable
behind a name, and what that callable expects to be fed. The second matters
because the detectors do not all take the same input through the same
signature. ``Long_sharp_wave_ripple_detector`` takes **raw** two-channel LFP
through a signature identical to the ripple-band detectors', so resolving by
name alone lets a caller hand it filtered data and get plausible nonsense.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd

from ripple_detection.detectors import (
    Carey_candidate_detector,
    Karlsson_ripple_detector,
    Kay_ripple_detector,
    Long_sharp_wave_ripple_detector,
    Roumis_ripple_detector,
    Shvartsman_ripple_detector,
    Yu_ripple_detector,
    Zugaro_ripple_detector,
    multiunit_HSE_detector,
)

RIPPLE_BAND_LFP = "ripple_band_lfp"
"""``(n_time, n_channels)`` LFP filtered to the ripple band."""

RAW_LFP_PAIR = "raw_lfp_pair"
"""``(n_time, 2)`` **unfiltered** LFP: a pyramidal-layer channel and a
stratum radiatum channel, in that order."""

MULTIUNIT = "multiunit"
"""``(n_time, n_units)`` spike counts or indicators."""


@dataclass(frozen=True)
class DetectorSpec:
    """A detector and what it needs to be given.

    Attributes
    ----------
    name : str
        The detector's name, the same string used as its registry key.
    detector : callable
        The detector itself. Every one takes ``time``, its signals, ``speed``
        and ``sampling_frequency`` positionally, in that order, and returns one
        DataFrame row per event.
    inputs : tuple of str
        The signals the detector takes, in the order it takes them, each one
        of :data:`RIPPLE_BAND_LFP`, :data:`RAW_LFP_PAIR` or :data:`MULTIUNIT`.

    """

    name: str
    detector: Callable[..., pd.DataFrame]
    inputs: tuple[str, ...]


def _spec(detector: Callable[..., pd.DataFrame], *inputs: str) -> DetectorSpec:
    return DetectorSpec(name=detector.__name__, detector=detector, inputs=inputs)


DETECTORS: Mapping[str, DetectorSpec] = MappingProxyType(
    {
        spec.name: spec
        for spec in (
            _spec(Kay_ripple_detector, RIPPLE_BAND_LFP),
            _spec(Karlsson_ripple_detector, RIPPLE_BAND_LFP),
            _spec(Roumis_ripple_detector, RIPPLE_BAND_LFP),
            _spec(Shvartsman_ripple_detector, RIPPLE_BAND_LFP),
            _spec(Yu_ripple_detector, RIPPLE_BAND_LFP),
            _spec(Zugaro_ripple_detector, RIPPLE_BAND_LFP),
            _spec(Long_sharp_wave_ripple_detector, RAW_LFP_PAIR),
            _spec(Carey_candidate_detector, RIPPLE_BAND_LFP, MULTIUNIT),
            _spec(multiunit_HSE_detector, MULTIUNIT),
        )
    }
)
"""Every detector in the package, keyed by name. Read-only."""


def get_detector(name: str) -> DetectorSpec:
    """Look up a detector by name.

    Parameters
    ----------
    name : str
        A key of :data:`DETECTORS`.

    Returns
    -------
    spec : DetectorSpec
        The detector and the signals it takes.

    Raises
    ------
    KeyError
        If no detector has that name. The message lists the ones that do.

    Examples
    --------
    >>> from ripple_detection import get_detector
    >>> spec = get_detector("Kay_ripple_detector")
    >>> spec.inputs
    ('ripple_band_lfp',)

    Check that a detector takes what you have before calling it:

    >>> spec.inputs == ("ripple_band_lfp",)
    True

    """
    try:
        return DETECTORS[name]
    except KeyError:
        known = "\n  ".join(sorted(DETECTORS))
        raise KeyError(f"No detector named {name!r}. This package has:\n  {known}") from None

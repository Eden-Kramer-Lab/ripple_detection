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
from typing import Literal

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

SignalKind = Literal["ripple_band_lfp", "raw_lfp_pair", "multiunit"]
"""The kinds of signal a detector can take. The three values below are the
only ones, so a misspelled kind is a type error at the registration site."""

RIPPLE_BAND_LFP: SignalKind = "ripple_band_lfp"
"""``(n_time, n_channels)`` LFP filtered to the ripple band."""

RAW_LFP_PAIR: SignalKind = "raw_lfp_pair"
"""``(n_time, 2)`` **unfiltered** LFP: a pyramidal-layer channel and a
stratum radiatum channel, in that order."""

MULTIUNIT: SignalKind = "multiunit"
"""``(n_time, n_units)`` spike counts or indicators."""


@dataclass(frozen=True)
class DetectorSpec:
    """A detector and what it needs to be given.

    Attributes
    ----------
    detector : callable
        The detector itself. Every one takes ``time``, its signals, ``speed``
        and ``sampling_frequency`` positionally, in that order, followed by
        keyword parameters that tune it. Every one returns a DataFrame with
        one row per event, indexed by ``event_number`` from 1, holding the
        columns listed under "Output Format" in the README; some add columns
        of their own.
    inputs : tuple of SignalKind
        The required positional signals, in the order the detector takes
        them. Optional signals such as ``Carey_candidate_detector``'s
        ``theta_lfp`` are keyword parameters and are not listed.

    """

    detector: Callable[..., pd.DataFrame]
    inputs: tuple[SignalKind, ...]

    @property
    def name(self) -> str:
        """The detector's name, which is also its key in :data:`DETECTORS`."""
        return self.detector.__name__


def _spec(detector: Callable[..., pd.DataFrame], *inputs: SignalKind) -> DetectorSpec:
    return DetectorSpec(detector=detector, inputs=inputs)


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
    Check that a detector takes what you have before calling it:

    >>> from ripple_detection import RIPPLE_BAND_LFP, get_detector
    >>> spec = get_detector("Kay_ripple_detector")
    >>> spec.inputs == (RIPPLE_BAND_LFP,)
    True

    """
    try:
        return DETECTORS[name]
    except KeyError:
        known = "\n  ".join(sorted(DETECTORS))
        raise KeyError(f"No detector named {name!r}. This package has:\n  {known}") from None

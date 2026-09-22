"""Resolve a detector by name, and say what signal it needs.

A caller that stores a detector's name rather than importing it, such as a
database-backed pipeline, needs two things from this package: the callable
behind a name, and what that callable expects to be fed. The second matters
because the detectors do not all take the same input through the same
signature. ``Long_sharp_wave_ripple_detector`` takes **raw** two-channel LFP
through a signature identical to the ripple-band detectors', so resolving by
name alone lets a caller hand it filtered data and get plausible nonsense.
:meth:`DetectorSpec.check_inputs` catches that before the call.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ripple_detection.core import low_frequency_variance_fraction
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

LOW_FREQUENCY_CUTOFF = 100.0
"""Hz. Raw LFP holds nearly all of its variance below this; ripple-band LFP
holds almost none. :meth:`DetectorSpec.check_inputs` tells the two apart by
the fraction of variance below it, with :data:`RAW_LFP_MINIMUM_LOW_FRACTION`
as the dividing line."""

RAW_LFP_MINIMUM_LOW_FRACTION = 0.5
"""Least fraction of variance below :data:`LOW_FREQUENCY_CUTOFF` for a
channel to count as raw LFP; a ripple-band channel must stay below it."""


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

    def check_inputs(self, *signals: ArrayLike, sampling_frequency: float) -> None:
        """Raise if the signals are not what this detector takes.

        Each signal is checked against its declared kind. A ripple-band LFP
        must hold less than half of each channel's variance below 100 Hz, and
        a raw LFP pair must hold at least half there and have two channels,
        which tells filtered from unfiltered input; spike counts must be
        non-negative integers. Missing samples (NaN) are left out.

        Parameters
        ----------
        *signals : array_like
            The signals, in the order of :attr:`inputs`.
        sampling_frequency : float
            Sampling rate in Hz, needed to place the 100 Hz cutoff.

        Raises
        ------
        ValueError
            If the number of signals is wrong, a signal is not 2-D, a
            ripple-band signal looks unfiltered, a raw pair looks filtered or
            does not have two channels, or spike counts are negative or not
            whole numbers. The message names the detector, the signal and
            what was found.

        Examples
        --------
        >>> import numpy as np
        >>> from ripple_detection import filter_ripple_band, get_detector
        >>> raw = np.cumsum(np.random.randn(6000, 2), axis=0)  # brown noise
        >>> spec = get_detector("Long_sharp_wave_ripple_detector")
        >>> spec.check_inputs(raw, sampling_frequency=1500)  # raw: fine
        >>> try:
        ...     spec.check_inputs(filter_ripple_band(raw, 1500), sampling_frequency=1500)
        ... except ValueError as error:
        ...     print("looks band-pass filtered" in str(error))
        True

        """
        if len(signals) != len(self.inputs):
            raise ValueError(
                f"{self.name} takes {len(self.inputs)} signal(s), {self.inputs}, "
                f"got {len(signals)}."
            )
        for position, (kind, signal) in enumerate(zip(self.inputs, signals, strict=True)):
            array = np.asarray(signal, dtype=float)
            what = f"{self.name} takes {kind} as signal {position + 1}"
            if array.ndim != 2:
                raise ValueError(
                    f"{what}, a 2-D array (n_time, n_channels), got shape {array.shape}."
                )
            if kind == MULTIUNIT:
                finite = array[np.isfinite(array)]
                if np.any(finite < 0) or np.any(finite != np.round(finite)):
                    raise ValueError(
                        f"{what}: spike counts or indicators, non-negative whole numbers, "
                        "but the array holds other values."
                    )
                continue
            if kind == RAW_LFP_PAIR and array.shape[1] != 2:
                raise ValueError(
                    f"{what}: two channels, the ripple channel then the sharp-wave "
                    f"channel, got {array.shape[1]}."
                )
            low = low_frequency_variance_fraction(
                array, sampling_frequency, LOW_FREQUENCY_CUTOFF
            )
            percent = ", ".join(f"{100 * f:.0f}%" for f in low)
            if kind == RIPPLE_BAND_LFP and np.any(low >= RAW_LFP_MINIMUM_LOW_FRACTION):
                raise ValueError(
                    f"{what}, LFP filtered to the ripple band, but channel(s) hold "
                    f"{percent} of their variance below {LOW_FREQUENCY_CUTOFF:.0f} Hz, which "
                    "looks unfiltered. Pass the output of filter_ripple_band."
                )
            if kind == RAW_LFP_PAIR and np.any(low < RAW_LFP_MINIMUM_LOW_FRACTION):
                raise ValueError(
                    f"{what}, unfiltered LFP, but channel(s) hold only {percent} of their "
                    f"variance below {LOW_FREQUENCY_CUTOFF:.0f} Hz, which looks band-pass "
                    "filtered. Pass the raw signal; this detector filters it itself."
                )


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

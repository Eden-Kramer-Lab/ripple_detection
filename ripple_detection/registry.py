"""Resolve a detector by name, and say what signal it needs.

A caller that stores a detector's name rather than importing it, such as a
database-backed pipeline, needs two things from this package: the callable
behind a name, and what that callable expects to be fed. The second matters
because the detectors do not all take the same input through the same
signature. ``Long_sharp_wave_ripple_detector`` takes **raw** two-channel LFP
through a signature identical to the ripple-band detectors', so resolving by
name alone lets a caller hand it filtered data and get plausible nonsense.
:attr:`DetectorSpec.inputs` says which it needs; :meth:`DetectorSpec.check_inputs`
verifies what an array can show, the shapes, but not whether it is filtered.
"""

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

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

    @property
    def signal_parameters(self) -> tuple[str, ...]:
        """Names of the detector's signal parameters, one per entry of
        :attr:`inputs` and in the same order, for callers that pass signals
        by keyword."""
        names = list(inspect.signature(self.detector).parameters)
        return tuple(names[1 : 1 + len(self.inputs)])

    @property
    def parameters(self) -> dict[str, object]:
        """The detector's tunable parameters and their defaults.

        Everything after ``sampling_frequency``; every one is keyword-only
        and has a default. A pipeline that stores a detector's parameters as
        a mapping can build the default mapping from this and validate a
        stored one with :meth:`check_parameters`.
        """
        signature = inspect.signature(self.detector)
        return {
            name: parameter.default
            for name, parameter in signature.parameters.items()
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        }

    def check_parameters(self, parameters: Mapping[str, object]) -> None:
        """Raise if ``parameters`` holds a name this detector does not take.

        Parameters
        ----------
        parameters : Mapping[str, object]
            Tunables to pass as ``**parameters``.

        Raises
        ------
        ValueError
            Naming the unknown keys and the detector's parameters.

        Examples
        --------
        >>> from ripple_detection import get_detector
        >>> spec = get_detector("Kay_ripple_detector")
        >>> spec.parameters["zscore_threshold"]
        2.0
        >>> spec.check_parameters({"zscore_threshold": 3.0})
        >>> try:
        ...     spec.check_parameters({"z_score_threshold": 3.0})
        ... except ValueError as error:
        ...     print(str(error).split(";")[0])
        Kay_ripple_detector does not take z_score_threshold

        """
        unknown = sorted(set(parameters) - set(self.parameters))
        if unknown:
            raise ValueError(
                f"{self.name} does not take {', '.join(unknown)}; its parameters are "
                f"{', '.join(self.parameters)}."
            )

    def check_inputs(self, *signals: ArrayLike) -> None:
        """Raise if the signals do not have the shape this detector takes.

        Checks what an array can show: the number of signals, that each is
        2-D, that a raw LFP pair has two channels, and that spike counts are
        non-negative whole numbers. It cannot tell raw LFP from ripple-band
        LFP; both are ``(n_time, n_channels)`` floats, and no property of the
        numbers settles it for every recording. That remains the caller's
        responsibility, which is why :attr:`inputs` states it.

        Parameters
        ----------
        *signals : array_like
            The signals, in the order of :attr:`inputs`.

        Raises
        ------
        ValueError
            If the number of signals is wrong, a signal is not 2-D, a raw pair
            does not have two channels, or spike counts are negative or not
            whole numbers. The message names the detector and the signal.

        Examples
        --------
        >>> import numpy as np
        >>> from ripple_detection import get_detector
        >>> spec = get_detector("Long_sharp_wave_ripple_detector")
        >>> spec.check_inputs(np.zeros((6000, 2)))
        >>> try:
        ...     spec.check_inputs(np.zeros((6000, 4)))
        ... except ValueError as error:
        ...     print(error)
        Long_sharp_wave_ripple_detector takes raw_lfp_pair as signal 1: two channels, the ripple channel then the sharp-wave channel, got 4.

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
            if kind == RAW_LFP_PAIR and array.shape[1] != 2:
                raise ValueError(
                    f"{what}: two channels, the ripple channel then the sharp-wave "
                    f"channel, got {array.shape[1]}."
                )
            if kind == MULTIUNIT:
                finite = array[np.isfinite(array)]
                if np.any(finite < 0) or np.any(finite != np.round(finite)):
                    raise ValueError(
                        f"{what}: spike counts or indicators, non-negative whole numbers, "
                        "but the array holds other values."
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

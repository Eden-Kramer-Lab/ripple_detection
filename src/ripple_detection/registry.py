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
from typing import Literal, get_args

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ripple_detection import _descriptions
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
from ripple_detection.detectors._validation import _validate_multiunit

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

    Raises
    ------
    ValueError
        If an input is not a :data:`SignalKind`, or the detector's positional
        parameters are not ``time``, one per input, ``speed`` and
        ``sampling_frequency``, so that :attr:`signal_parameters` would name
        the wrong ones.

    Notes
    -----
    :attr:`name` is the key a pipeline stores, so it is a public contract: a
    detector renamed in a later release keeps its old name here as an alias.

    """

    detector: Callable[..., pd.DataFrame]
    inputs: tuple[SignalKind, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(self.inputs))
        unknown = [kind for kind in self.inputs if kind not in get_args(SignalKind)]
        if unknown:
            msg = f"{unknown} are not signal kinds; the kinds are {get_args(SignalKind)}."
            raise ValueError(msg)
        positional = [
            parameter.name
            for parameter in inspect.signature(self.detector).parameters.values()
            if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        n_signals = len(self.inputs)
        if (
            len(positional) != n_signals + 3
            or positional[0] != "time"
            or positional[-2:] != ["speed", "sampling_frequency"]
        ):
            msg = (
                f"{self.detector.__name__} takes {positional} positionally; a detector "
                f"with {n_signals} signal(s) takes time, the signals, speed and "
                "sampling_frequency."
            )
            raise ValueError(msg)

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

    def describe(self) -> dict[str, object]:
        """Everything a caller needs to use this detector, as plain data.

        A dict that ``json.dumps`` accepts, for a pipeline or a language model
        choosing and configuring a detector without reading its docstring:
        the positional arguments with their shapes and units, every tunable
        with its default, unit and meaning, and the columns of the result.
        Units are ``"s"``, ``"Hz"``, ``"cm/s"``, ``"SD"`` (standard deviations
        of a normalized trace) and so on, or ``""`` for none.

        Returns
        -------
        description : dict
            ``name``, ``summary``, ``call``, ``positional`` (list of dicts with
            ``name``, ``shape``, ``unit``, ``description``, and ``kind`` for a
            signal), ``parameters`` (name -> ``default``, ``unit``,
            ``description``), and ``returns`` (``index`` and ``columns``,
            column -> description, in the order the detector returns them).

        Examples
        --------
        >>> import json
        >>> from ripple_detection import get_detector
        >>> description = get_detector("Kay_ripple_detector").describe()
        >>> description["parameters"]["minimum_duration"]["unit"]
        's'
        >>> [argument["name"] for argument in description["positional"]]
        ['time', 'filtered_lfps', 'speed', 'sampling_frequency']
        >>> json.loads(json.dumps(description)) == description
        True

        """
        signal_names = self.signal_parameters
        positional: list[dict[str, str]] = []
        for name in ("time", *signal_names, "speed", "sampling_frequency"):
            if name in signal_names:
                kind = self.inputs[signal_names.index(name)]
                shape, unit, text = _descriptions.SIGNALS[kind]
                positional.append(
                    {
                        "name": name,
                        "kind": kind,
                        "shape": shape,
                        "unit": unit,
                        "description": text,
                    }
                )
            else:
                shape, unit, text = _descriptions.POSITIONAL[name]
                positional.append(
                    {"name": name, "shape": shape, "unit": unit, "description": text}
                )
        parameters: dict[str, dict[str, object]] = {}
        for name, default in self.parameters.items():
            unit, text = _descriptions.OVERRIDES.get(
                (self.name, name), _descriptions.PARAMETERS[name]
            )
            parameters[name] = {
                "default": list(default) if isinstance(default, tuple) else default,
                "unit": unit,
                "description": text,
            }
        columns = {**_descriptions.COLUMNS, **_descriptions.EXTRA_COLUMNS.get(self.name, {})}
        columns.update(
            {
                column: text
                for (detector, column), text in _descriptions.COLUMN_OVERRIDES.items()
                if detector == self.name
            }
        )
        return {
            "name": self.name,
            "summary": (inspect.getdoc(self.detector) or "").split("\n", 1)[0],
            "call": f"{self.name}({', '.join(item['name'] for item in positional)}, **parameters)",
            "positional": positional,
            "parameters": parameters,
            "returns": {"index": "event_number", "columns": columns},
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
            msg = (
                f"{self.name} does not take {', '.join(unknown)}; its parameters are "
                f"{', '.join(self.parameters)}."
            )
            raise ValueError(msg)

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
            msg = (
                f"{self.name} takes {len(self.inputs)} signal(s), {self.inputs}, "
                f"got {len(signals)}."
            )
            raise ValueError(msg)
        for position, (kind, signal) in enumerate(zip(self.inputs, signals, strict=True)):
            array = np.asarray(signal, dtype=float)
            what = f"{self.name} takes {kind} as signal {position + 1}"
            if array.ndim != 2:
                msg = f"{what}, a 2-D array (n_time, n_channels), got shape {array.shape}."
                raise ValueError(msg)
            if kind == RAW_LFP_PAIR and array.shape[1] != 2:
                msg = (
                    f"{what}: two channels, the ripple channel then the sharp-wave "
                    f"channel, got {array.shape[1]}."
                )
                raise ValueError(msg)
            if kind == MULTIUNIT:
                _validate_multiunit(array, what)


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
        msg = f"No detector named {name!r}. This package has:\n  {known}"
        raise KeyError(msg) from None

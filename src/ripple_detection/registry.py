"""Resolve a detector by name, and say what signal it needs.

A caller that stores a detector's name rather than importing it, such as a
database-backed pipeline, needs two things from this package: the callable
behind a name, and what that callable expects to be fed. The second matters
because the detectors do not all take the same input through the same
signature. ``Long_sharp_wave_ripple_detector`` takes **raw** LFP in the slot
where the ripple-band detectors take filtered LFP, and a second raw channel,
``sharp_wave_lfp``, by name. :attr:`DetectorSpec.inputs` and
:attr:`DetectorSpec.keyword_inputs` say what each needs;
:meth:`DetectorSpec.check_inputs` verifies what an array can show, the shapes,
but not whether it is filtered.
"""

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
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

SignalKind = Literal["ripple_band_lfp", "raw_lfp", "multiunit"]
"""The kinds of signal a detector can take. The three values below are the
only ones, so a misspelled kind is a type error at the registration site."""

RIPPLE_BAND_LFP: SignalKind = "ripple_band_lfp"
"""``(n_time, n_channels)`` LFP filtered to the ripple band."""

RAW_LFP: SignalKind = "raw_lfp"
"""``(n_time,)`` or ``(n_time, 1)`` **unfiltered** LFP from one channel."""

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
        The positional signals, in the order the detector takes them.
    keyword_inputs : Mapping[str, SignalKind]
        The signals the detector takes by name, such as
        ``Long_sharp_wave_ripple_detector``'s ``sharp_wave_lfp`` (required)
        and ``Carey_candidate_detector``'s ``theta_lfp`` (optional). They are
        signals, not tunables, so :attr:`parameters` leaves them out.

    Raises
    ------
    ValueError
        If an input is not a :data:`SignalKind`; if the detector's positional
        parameters are not ``time``, one per input, ``speed`` and
        ``sampling_frequency``, so that :attr:`signal_parameters` would name
        the wrong ones; if a keyword input is not a keyword-only parameter;
        or if a keyword-only parameter without a default is not a keyword
        input, since every tunable has a default.

    Notes
    -----
    :attr:`name` is the key a pipeline stores, so it is a public contract: a
    detector renamed in a later release keeps its old name here as an alias.

    """

    detector: Callable[..., pd.DataFrame]
    inputs: tuple[SignalKind, ...]
    keyword_inputs: Mapping[str, SignalKind] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "keyword_inputs", MappingProxyType(dict(self.keyword_inputs)))
        kinds = [*self.inputs, *self.keyword_inputs.values()]
        unknown = [kind for kind in kinds if kind not in get_args(SignalKind)]
        if unknown:
            msg = f"{unknown} are not signal kinds; the kinds are {get_args(SignalKind)}."
            raise ValueError(msg)
        parameters = inspect.signature(self.detector).parameters
        keyword_only = {
            name: parameter
            for name, parameter in parameters.items()
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        }
        for name in self.keyword_inputs:
            if name not in keyword_only:
                msg = f"{name} is not a keyword-only parameter of {self.detector.__name__}."
                raise ValueError(msg)
        for name, parameter in keyword_only.items():
            if (
                parameter.default is inspect.Parameter.empty
                and name not in self.keyword_inputs
            ):
                msg = (
                    f"{name} has no default, so it must be declared in keyword_inputs; "
                    "every tunable has one."
                )
                raise ValueError(msg)
        positional = [
            parameter.name
            for parameter in parameters.values()
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
    def required_keyword_inputs(self) -> tuple[str, ...]:
        """The :attr:`keyword_inputs` a call must pass."""
        parameters = inspect.signature(self.detector).parameters
        return tuple(
            name
            for name in self.keyword_inputs
            if parameters[name].default is inspect.Parameter.empty
        )

    @property
    def parameters(self) -> dict[str, object]:
        """The detector's tunable parameters and their defaults.

        Every keyword-only parameter but the :attr:`keyword_inputs`; every one
        has a default. A pipeline that stores a detector's parameters as a
        mapping can build the default mapping from this and validate a stored
        one with :meth:`check_parameters`.
        """
        signature = inspect.signature(self.detector)
        return {
            name: parameter.default
            for name, parameter in signature.parameters.items()
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY
            and name not in self.keyword_inputs
        }

    def describe(self) -> dict[str, object]:
        """Everything a caller needs to use this detector, as plain data.

        A dict that ``json.dumps`` accepts, for a pipeline or a language model
        choosing and configuring a detector without reading its docstring:
        the positional arguments and the signals passed by name with their
        shapes and units, every tunable with its default, unit and meaning,
        and the columns of the result.
        Units are ``"s"``, ``"Hz"``, ``"cm/s"``, ``"SD"`` (standard deviations
        of a normalized trace) and so on, or ``""`` for none.

        Returns
        -------
        description : dict
            ``name``, ``summary``, ``call``, ``positional`` (list of dicts with
            ``name``, ``shape``, ``unit``, ``description``, and ``kind`` for a
            signal), ``keyword_signals`` (list of dicts with ``name``,
            ``kind``, ``shape``, ``unit``, ``description`` and ``required``),
            ``parameters`` (name -> ``default``, ``unit``, ``description``),
            and ``returns`` (``index`` and ``columns``, column -> description,
            in the order the detector returns them).

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
                        "description": _descriptions.SIGNAL_ROLES.get((self.name, name), text),
                    }
                )
            else:
                shape, unit, text = _descriptions.POSITIONAL[name]
                positional.append(
                    {"name": name, "shape": shape, "unit": unit, "description": text}
                )
        required = self.required_keyword_inputs
        keyword_signals: list[dict[str, object]] = []
        for name, kind in self.keyword_inputs.items():
            shape, unit, _ = _descriptions.SIGNALS[kind]
            keyword_signals.append(
                {
                    "name": name,
                    "kind": kind,
                    "shape": shape,
                    "unit": unit,
                    "description": _descriptions.SIGNAL_ROLES[(self.name, name)],
                    "required": name in required,
                }
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
        arguments = [
            *(str(item["name"]) for item in positional),
            *(f"{name}={name}" for name in required),
            "**parameters",
        ]
        return {
            "name": self.name,
            "summary": (inspect.getdoc(self.detector) or "").split("\n", 1)[0],
            "call": f"{self.name}({', '.join(arguments)})",
            "positional": positional,
            "keyword_signals": keyword_signals,
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

    def check_inputs(self, *signals: ArrayLike, **keyword_signals: ArrayLike) -> None:
        """Raise if the signals do not have the shape this detector takes.

        Checks what an array can show: the number of signals, that the
        required keyword signals are there, that multichannel signals are 2-D
        and a raw LFP is one channel, and that spike counts are non-negative
        whole numbers. It cannot tell raw LFP from ripple-band LFP; both are
        floats of the same shape, and no property of the numbers settles it
        for every recording. That remains the caller's responsibility, which
        is why :attr:`inputs` states it.

        Parameters
        ----------
        *signals : array_like
            The signals, in the order of :attr:`inputs`.
        **keyword_signals : array_like
            The signals passed by name, :attr:`keyword_inputs`.

        Raises
        ------
        ValueError
            If the number of signals is wrong, a required keyword signal is
            missing or an unknown one given, a multichannel signal is not 2-D,
            a raw LFP is not one channel, or spike counts are negative or not
            whole numbers. The message names the detector and the signal.

        Examples
        --------
        >>> import numpy as np
        >>> from ripple_detection import get_detector
        >>> spec = get_detector("Long_sharp_wave_ripple_detector")
        >>> spec.check_inputs(np.zeros(6000), sharp_wave_lfp=np.zeros(6000))
        >>> try:
        ...     spec.check_inputs(np.zeros((6000, 2)), sharp_wave_lfp=np.zeros(6000))
        ... except ValueError as error:
        ...     print(error)
        Long_sharp_wave_ripple_detector takes raw_lfp as raw_lfp: one channel, shape (n_time,) or (n_time, 1), got shape (6000, 2).

        """
        if len(signals) != len(self.inputs):
            msg = (
                f"{self.name} takes {len(self.inputs)} signal(s), {self.inputs}, "
                f"got {len(signals)}."
            )
            raise ValueError(msg)
        unknown = sorted(set(keyword_signals) - set(self.keyword_inputs))
        if unknown:
            msg = (
                f"{self.name} takes no keyword signal {', '.join(unknown)}; its keyword "
                f"signals are {list(self.keyword_inputs) or 'none'}."
            )
            raise ValueError(msg)
        missing = [
            name for name in self.required_keyword_inputs if name not in keyword_signals
        ]
        if missing:
            msg = f"{self.name} needs {', '.join(missing)}, passed by name."
            raise ValueError(msg)
        named = [
            *zip(self.signal_parameters, self.inputs, signals, strict=True),
            *(
                (name, self.keyword_inputs[name], signal)
                for name, signal in keyword_signals.items()
            ),
        ]
        for name, kind, signal in named:
            array = np.asarray(signal, dtype=float)
            what = f"{self.name} takes {kind} as {name}"
            if kind == RAW_LFP:
                if array.ndim not in (1, 2) or (array.ndim == 2 and array.shape[1] != 1):
                    msg = (
                        f"{what}: one channel, shape (n_time,) or (n_time, 1), "
                        f"got shape {array.shape}."
                    )
                    raise ValueError(msg)
            elif array.ndim != 2:
                msg = f"{what}, a 2-D array (n_time, n_channels), got shape {array.shape}."
                raise ValueError(msg)
            if kind == MULTIUNIT:
                _validate_multiunit(array, what)


def _spec(
    detector: Callable[..., pd.DataFrame],
    *inputs: SignalKind,
    **keyword_inputs: SignalKind,
) -> DetectorSpec:
    return DetectorSpec(detector=detector, inputs=inputs, keyword_inputs=keyword_inputs)


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
            _spec(Long_sharp_wave_ripple_detector, RAW_LFP, sharp_wave_lfp=RAW_LFP),
            _spec(Carey_candidate_detector, RIPPLE_BAND_LFP, MULTIUNIT, theta_lfp=RAW_LFP),
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
        msg = f"No detector named {name!r}. This package has {', '.join(sorted(DETECTORS))}."
        raise KeyError(msg) from None

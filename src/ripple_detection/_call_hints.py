"""Explain a call that does not fit a function's signature, in terms of what
changed in 2.0.

Code written for 1.x, by a person or by a language model trained on it,
fails in 2.0 with Python's generic ``TypeError``: an unexpected keyword, too
many positional arguments, a missing one. The wrapper here binds the call to
the signature first and, when that fails, re-raises with the 2.0 change that
explains it and the call to write instead. A call that binds runs the
function unchanged.
"""

import difflib
import functools
import inspect
from collections.abc import Callable, Mapping
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

_TIME_RANGE_IS_A_MASK = (
    "was removed in 2.0: a time range is a mask, "
    "normalization_mask=(time >= start) & (time <= end)"
)

REMOVED_ARGUMENTS: dict[tuple[str, str], str] = {
    **{
        (function, "normalization_time_range"): _TIME_RANGE_IS_A_MASK
        for function in (
            "Kay_ripple_detector",
            "Karlsson_ripple_detector",
            "Roumis_ripple_detector",
            "multiunit_HSE_detector",
            "normalize_signal",
        )
    },
    ("multiunit_HSE_detector", "use_speed_threshold_for_zscore"): (
        "was removed in 2.0: pass normalization_mask=speed <= speed_threshold"
    ),
    ("normalize_signal", "time"): (
        "was removed in 2.0: call normalize_signal(data, method, normalization_mask), "
        "with a time range as normalization_mask=(time >= start) & (time <= end)"
    ),
    **{
        (function, "state"): (
            "was renamed rng in 2.0, and takes a seed or a numpy.random.Generator "
            "rather than a RandomState"
        )
        for function in ("pink", "white", "brown")
    },
}
"""Keywords a 1.x release accepted and 2.0 does not, by the function that took
them: what happened to each and what to pass instead. Checked against the
signatures of every release from 1.0.0 to 1.7.1. Only names a release shipped
belong here: a reader upgrades from a release, not from a development commit,
so a name that changed between 1.7.1 and 2.0 gets no history (``SAME_ROLE``
and the spelling match still point it at the parameter to use)."""

POSITIONAL_ORDER_1X = {
    **dict.fromkeys(
        ("Kay_ripple_detector", "Karlsson_ripple_detector", "Roumis_ripple_detector"),
        (
            "speed_threshold",
            "minimum_duration",
            "zscore_threshold",
            "smoothing_sigma",
            "close_ripple_threshold",
            "normalization_method",
            "normalization_mask",
            "normalization_time_range",
        ),
    ),
    "multiunit_HSE_detector": (
        "speed_threshold",
        "minimum_duration",
        "zscore_threshold",
        "smoothing_sigma",
        "close_event_threshold",
        "use_speed_threshold_for_zscore",
        "normalization_method",
        "normalization_mask",
        "normalization_time_range",
    ),
}
"""The order 1.7 took each detector's tunables in after ``sampling_frequency``,
so the values of a positional 1.x call can be named as that call meant them;
2.0's keyword order differs from it (HSE's flag sat sixth)."""

REQUIRED_SINCE_2 = {
    ("filter_ripple_band", "sampling_frequency"): (
        "is required since 2.0; 1.x assumed 1500 Hz whatever the data's rate. Pass the "
        "rate the samples were recorded at, in Hz"
    ),
}
"""Arguments a 1.x function let a caller leave out, by function and name."""

NORMALIZE_SIGNAL_WITHOUT_TIME = (
    "normalize_signal no longer takes time (removed in 2.0); call "
    "normalize_signal(data, method, normalization_mask), with a time range as "
    "normalization_mask=(time >= start) & (time <= end)."
)
"""What to write instead of 1.x's ``normalize_signal(data, time, ...)``."""

TOO_MANY_POSITIONAL_1X = {"normalize_signal": NORMALIZE_SIGNAL_WITHOUT_TIME}
"""The note for a 1.x function called with more positional arguments than
2.0 takes, where the extra one was removed from the middle rather than made
keyword-only."""

SAME_ROLE = (
    (
        "zscore_threshold",
        "low_threshold",
        "high_threshold",
        "percentile",
        "sharp_wave_thresholds",
        "ripple_thresholds",
    ),
    ("minimum_duration", "minimum_sharp_wave_duration", "minimum_ripple_duration"),
    ("maximum_duration", "maximum_sharp_wave_duration"),
    (
        "close_ripple_threshold",
        "close_event_threshold",
        "minimum_inter_ripple_interval",
        "minimum_separation",
    ),
    ("smoothing_sigma", "ripple_smoothing_sigma", "smoothing_window"),
    ("rng", "random_state", "seed", "state"),
)
"""Parameters that play one role under different names in different detectors,
or in other libraries: the detection threshold, the duration limits, the rule
for close events, the smoothing width, the seed. A keyword from elsewhere is
matched to this function's by role, not by spelling, which sent
``zscore_threshold`` to ``speed_threshold``."""


def explain_call_errors(function: Callable[P, R]) -> Callable[P, R]:
    """Re-raise a call that does not bind to ``function``'s signature with the
    2.0 change behind it. ``inspect.signature`` and ``__name__`` still report
    ``function``'s own."""
    signature = inspect.signature(function)
    name = getattr(function, "__name__", repr(function))

    @functools.wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            signature.bind(*args, **kwargs)
        except TypeError as error:
            raise TypeError(_explain(name, signature, args, kwargs, error)) from None
        return function(*args, **kwargs)

    return wrapper


def _explain(
    name: str,
    signature: inspect.Signature,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    error: TypeError,
) -> str:
    parameters = signature.parameters
    positional = [
        parameter
        for parameter in parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    keyword_only = [
        parameter.name
        for parameter in parameters.values()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    notes = []
    for key in kwargs:
        if key in parameters:
            continue
        if (name, key) in REMOVED_ARGUMENTS:
            notes.append(f"{key} {REMOVED_ARGUMENTS[name, key]}.")
            continue
        roles = [group for group in SAME_ROLE if key in group]
        same_role = [
            parameter for group in roles for parameter in group if parameter in parameters
        ]
        if roles:
            notes.append(
                f"{name} takes no {key}; did you mean {' or '.join(same_role)}?"
                if same_role
                else f"{name} takes no {key}; its parameters are {', '.join(parameters)}."
            )
            continue
        close = difflib.get_close_matches(key, list(parameters), n=1, cutoff=0.6)
        notes.append(
            f"{name} takes no {key}; did you mean {close[0]}?"
            if close
            else f"{name} takes no {key}; its parameters are {', '.join(parameters)}."
        )
    if len(args) > len(positional) and name in TOO_MANY_POSITIONAL_1X:
        notes.append(TOO_MANY_POSITIONAL_1X[name])
    if len(args) > len(positional) and keyword_only:
        extra = args[len(positional) :]
        order = POSITIONAL_ORDER_1X.get(name)
        if order is None:
            notes.append(
                f"Every argument after {positional[-1].name} is keyword-only; pass the "
                f"{len(extra)} extra value(s) by name."
            )
        else:
            named = ", ".join(
                f"{parameter}={_short(value)}"
                for parameter, value in zip(order, extra, strict=False)
            )
            notes.append(
                f"Since 2.0 every argument after {positional[-1].name} is keyword-only; "
                f"in 1.x's order the extra values were {named}. Pass each by name."
            )
            notes.extend(
                f"{parameter} {REMOVED_ARGUMENTS[name, parameter]}."
                for parameter in order[: len(extra)]
                if (name, parameter) in REMOVED_ARGUMENTS
            )
    missing = [
        parameter.name
        for parameter in positional[len(args) :]
        if parameter.default is inspect.Parameter.empty and parameter.name not in kwargs
    ]
    since_2 = [
        f"{parameter} {REQUIRED_SINCE_2[name, parameter]}."
        for parameter in missing
        if (name, parameter) in REQUIRED_SINCE_2
    ]
    if missing:
        notes.extend(
            since_2
            or [f"{name} takes {', '.join(item.name for item in positional)} positionally."]
        )
    return f"{name}(): {error}." + (f" {' '.join(notes)}" if notes else "")


def _short(value: object) -> str:
    """A value as it would be written in a call, or its type if that is long."""
    text = repr(value)
    return text if len(text) <= 30 else f"<{type(value).__name__}>"

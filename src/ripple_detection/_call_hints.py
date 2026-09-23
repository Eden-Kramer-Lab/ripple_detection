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

REMOVED_ARGUMENTS: dict[str, tuple[str | None, str]] = {
    "normalization_time_range": (
        None,
        (
            "was removed in 2.0: a time range is a mask, "
            "normalization_mask=(time >= start) & (time <= end)"
        ),
    ),
    "use_speed_threshold_for_zscore": (
        None,
        "was removed in 2.0: pass normalization_mask=speed <= speed_threshold",
    ),
    "state": (
        "rng",
        (
            "was renamed rng in 2.0, and takes a seed or a numpy.random.Generator "
            "rather than a RandomState"
        ),
    ),
    "random_state": ("rng", "was renamed rng before 2.0 was released"),
    "raw_lfps": ("raw_lfp_pair", "was renamed raw_lfp_pair before 2.0 was released"),
    "edge_threshold": ("low_threshold", "was renamed low_threshold before 2.0 was released"),
    "peak_threshold": ("high_threshold", "was renamed high_threshold before 2.0 was released"),
    "participation_threshold": (
        "minimum_participating_channels",
        (
            "was split before 2.0 was released into minimum_participating_channels, a "
            "count, and "
            "minimum_participating_fraction, a fraction of the channels in [0, 1]"
        ),
    ),
    "manual_normalization": (
        "channel_baselines",  # Shvartsman's alone; Kay has normalization_method too
        (
            "was removed before 2.0 was released: pass normalization_method='manual' with "
            "channel_baselines and channel_deviations"
        ),
    ),
    "elec_baselines": (
        "channel_baselines",
        "was renamed channel_baselines before 2.0 was released",
    ),
    "elec_deviations": (
        "channel_deviations",
        "was renamed channel_deviations before 2.0 was released",
    ),
}
"""Keywords that 1.x, or the development versions before 2.0 (the Carey and
Shvartsman names, Long's signal and the simulators' seed), accepted and 2.0
does not: the parameter that replaces each, or None for one with no
replacement, and what to pass instead. A note applies only to a function that
has the replacement, so a keyword one function renamed is not explained that
way on a function that never took it."""

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
    "sampling_frequency": (
        "is required since 2.0; 1.x assumed 1500 Hz whatever the data's rate. Pass the "
        "rate the samples were recorded at, in Hz"
    ),
}
"""Arguments 1.x let a caller leave out."""


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
        if key in REMOVED_ARGUMENTS:
            replacement, note = REMOVED_ARGUMENTS[key]
            if replacement is None or replacement in parameters:
                notes.append(f"{key} {note}.")
                continue
        close = difflib.get_close_matches(key, list(parameters), n=1, cutoff=0.6)
        notes.append(
            f"{name} takes no {key}; did you mean {close[0]}?"
            if close
            else f"{name} takes no {key}; its parameters are {', '.join(parameters)}."
        )
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
                f"{parameter} {REMOVED_ARGUMENTS[parameter][1]}."
                for parameter in order[: len(extra)]
                if parameter in REMOVED_ARGUMENTS
            )
    notes.extend(
        f"{parameter.name} {REQUIRED_SINCE_2[parameter.name]}."
        for parameter in positional[len(args) :]
        if parameter.default is inspect.Parameter.empty
        and parameter.name not in kwargs
        and parameter.name in REQUIRED_SINCE_2
    )
    return f"{name}(): {error}." + (f" {' '.join(notes)}" if notes else "")


def _short(value: object) -> str:
    """A value as it would be written in a call, or its type if that is long."""
    text = repr(value)
    return text if len(text) <= 30 else f"<{type(value).__name__}>"

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

REMOVED_ARGUMENTS = {
    "normalization_time_range": (
        "was removed in 2.0: a time range is a mask, "
        "normalization_mask=(time >= start) & (time <= end)"
    ),
    "use_speed_threshold_for_zscore": (
        "was removed in 2.0: pass normalization_mask=speed <= speed_threshold"
    ),
    "state": (
        "was renamed rng in 2.0, and takes a seed or a numpy.random.Generator rather "
        "than a RandomState"
    ),
}
"""Keywords 1.x accepted that 2.0 does not, and what to pass instead."""

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
            notes.append(f"{key} {REMOVED_ARGUMENTS[key]}.")
            continue
        close = difflib.get_close_matches(key, list(parameters), n=1, cutoff=0.6)
        notes.append(
            f"{name} takes no {key}; did you mean {close[0]}?"
            if close
            else f"{name} takes no {key}; its parameters are {', '.join(parameters)}."
        )
    if len(args) > len(positional) and keyword_only:
        extra = args[len(positional) :]
        named = ", ".join(
            f"{parameter}={_short(value)}"
            for parameter, value in zip(keyword_only, extra, strict=False)
        )
        notes.append(
            f"Since 2.0 every argument after {positional[-1].name} is keyword-only; "
            f"in the order of the keywords the extra values would be {named}. "
            "Pass each by name."
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

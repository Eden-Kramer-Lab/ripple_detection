"""Benchmark configurations of the packaged literature methods.

Each ``RecipeConfig`` names a method of ``ripple_detection.literature_methods``
by its function name, the options it runs with and the expression of a
simulated network event it is to be headlined against. ``run_recipe`` calls the
installed method through ``run_method``; nothing here reimplements one.

``make_recording`` builds the method's ``Recording`` from a simulated session
with ``Recording.from_arrays``, the measured-data path, under one input policy
(``INPUT_POLICY``): the recording holds exactly the inputs the method declares
(its ``list_methods()`` requirements), taken from the session's observations,
its known unit labels and running bouts, never its truth tables. Where a method needs
an input the simulation has no counterpart for (a sleep state, a template, an
external ripple inventory), the policy supplies a stated stand-in, and every
stand-in a configuration relies on is listed in its ``assumptions``.
``EXCLUSIONS`` gives the reason for each catalog method not configured.
"""

from __future__ import annotations

import functools
import inspect
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

import ripple_detection as rd
from ripple_detection import literature_methods
from ripple_detection.core import FloatArray
from ripple_detection.literature_methods import (
    Recording,
    bounds,
    check_method,
    list_methods,
    run_method,
)

Params = tuple[tuple[str, Any], ...]

# The truth a configuration is headlined against: an expression of the latent
# event, or "network" (the union of its expressions).
PRIMARY_EXPRESSIONS = (*rd.EXPRESSIONS, "network")

INPUT_POLICY = "simulated_awake_session"

# The external ripple inventory for methods that require one (Yang 2024,
# Grosmark 2016): the package's stated stand-in for their unspecified ripple
# detector, the public Zugaro detector on the first channel. Never truth.
EXTERNAL_RIPPLES: dict[str, Any] = {
    "detector": "Zugaro_ripple_detector",
    "channel": 0,
    "band": (130.0, 200.0),
    "options": {
        "low_threshold": 2.0,
        "high_threshold": 5.0,
        "maximum_duration": 0.2,
        "speed_threshold": np.inf,
    },
    "columns": ("start_time", "end_time", "peak_time"),
}

# Carey 2019's manually selected example ripples: the package's stand-in, the
# largest events of the public Kay detector at its defaults on every channel.
EXAMPLE_RIPPLES: dict[str, Any] = {
    "detector": "Kay_ripple_detector",
    "band": (150.0, 250.0),
    "options": {},
    "n_examples": 5,
    "rank_by": "max_zscore",
}

# What each input the policy supplies is, as input_policy records it. The
# values themselves are in each result's attrs["inputs"] and
# attrs["behavior_intervals"].
_REST = (
    "rest: the recorded samples outside session.running_intervals, the simulator's "
    "known running bouts"
)
_SOURCES: dict[str, Any] = {
    "lfps": "session.lfps: every pyramidal-layer channel, the ripple channel first",
    "sharp_wave_lfp": "session.sharp_wave_lfp: the stratum radiatum channel",
    "multiunit": "session.multiunit: every unit",
    "speed": "session.speed",
    "place_cells": "units whose session.unit_types is 'place' (known labels)",
    "pyramidal": "units whose session.unit_types is 'place' or 'pyramidal' (known labels)",
    "templates": "one template: the place_cells selection",
    "sleep_intervals": _REST,
    "baseline_intervals": _REST,
    "behavior_intervals": _REST,
    "reference_lfp": "zeros",
    "external_ripples": EXTERNAL_RIPPLES,
    "example_ripples": EXAMPLE_RIPPLES,
}

# The benchmark choice behind each supplied input that is not an observation.
_ASSUMPTIONS = {
    "place_cells": (
        "place_cells: every unit the simulator labels 'place', also standing in for "
        "any narrower selection the method names, such as one template's, one "
        "directional template's, one probe sequence's, block-specific or "
        "place-responsive cells, since the simulator has no place fields or "
        "trajectories"
    ),
    "pyramidal": "pyramidal: every unit the simulator labels 'place' or 'pyramidal'",
    "templates": (
        "templates: one template of every place unit, since the simulator has no "
        "place fields or trajectories"
    ),
    "sleep_intervals": (
        "sleep_intervals: rest, the samples outside the simulator's known running "
        "bouts, stands in for the sleep state, since the simulated sessions are awake"
    ),
    "baseline_intervals": (
        "baseline_intervals: rest, the samples outside the simulator's known running "
        "bouts, stands in for the normalization epoch"
    ),
    "behavior_intervals": (
        "behavior_intervals: rest, the samples outside the simulator's known running "
        "bouts, stands in for the eligible epochs: simulated events occur only at rest, "
        "and the simulator has no position"
    ),
    "reference_lfp": (
        "reference_lfp: zeros, so nothing is subtracted, as examples/literature_recipes.py "
        "does: the simulation has no reference electrode"
    ),
    "external_ripples": (
        "external_ripples: Zugaro_ripple_detector with the settings input_policy "
        "records, the package's stated assumption for this paper's unspecified ripple "
        "detector; not the simulation's truth"
    ),
    "example_ripples": (
        "example_ripples: the largest Kay_ripple_detector events, as input_policy "
        "records, the package's stand-in for manual selection; not the simulation's truth"
    ),
}

_CONFIG_ID = re.compile(r"[a-z0-9_]+(\.[a-z0-9_]+)?")

# Call arguments that are not method options: the input policy supplies them.
_NOT_OPTIONS = frozenset({"rec", "behavior_intervals"})


def _is_option_pairs(options: object) -> bool:
    """Whether ``options`` is a tuple of (str, value) pairs."""
    return isinstance(options, tuple) and all(
        isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[0], str)
        for pair in options
    )


@dataclass(frozen=True)
class RecipeConfig:
    """One benchmark configuration of a packaged literature method.

    Attributes
    ----------
    config_id : str
        Unique, stable identifier: the method name, or ``"{method}.{label}"``
        for a protocol variant, the label of ``[a-z0-9_]``.
    method : str
        Exact function name from ``list_methods()``.
    primary_expression : str
        The truth the method is to be headlined against: ``"ripple"``,
        ``"sharp_wave"``, ``"burst"`` or ``"network"``.
    options : tuple of (str, object) pairs
        Method options, including ``stage`` where the method takes one;
        each name once, never ``rec`` or ``behavior_intervals``, and each
        value hashable (a scalar or a tuple), so the configuration is.
    input_policy : str
        The simulation-to-``Recording`` policy. Defaults to ``""``; it must
        be ``INPUT_POLICY`` for ``make_recording`` to build a recording.
    assumptions : tuple of str
        Benchmark choices absent from the source: the policy's stand-ins
        for the inputs this method needs, and demonstration values for
        options the paper does not report.

    Raises
    ------
    ValueError
        The identifier is neither the method name nor
        ``"{method}.{label}"``, the expression is unknown, or ``options`` is
        not a tuple of pairs as described.
    """

    config_id: str
    method: str
    primary_expression: str
    options: Params = ()
    input_policy: str = ""
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _CONFIG_ID.fullmatch(self.config_id) or not (
            self.config_id == self.method or self.config_id.startswith(f"{self.method}.")
        ):
            msg = (
                f"config_id {self.config_id!r} must be the method name or "
                f"'{self.method}.<label>', the label of a-z, 0-9 and '_'."
            )
            raise ValueError(msg)
        if self.primary_expression not in PRIMARY_EXPRESSIONS:
            msg = (
                f"primary_expression {self.primary_expression!r} is not one of "
                f"{', '.join(PRIMARY_EXPRESSIONS)}."
            )
            raise ValueError(msg)
        if not _is_option_pairs(self.options):
            msg = f"options must be a tuple of (str, value) pairs; got {self.options!r}."
            raise ValueError(msg)
        names = [name for name, _ in self.options]
        for name, value in self.options:
            if names.count(name) > 1:
                msg = f"options names {name!r} more than once."
                raise ValueError(msg)
            if name in _NOT_OPTIONS:
                msg = (
                    f"{name!r} is not a method option: the recording and the call's "
                    "behavior_intervals come from the input policy."
                )
                raise ValueError(msg)
            try:
                hash(value)
            except TypeError:
                msg = f"option {name!r} must be hashable (a scalar or a tuple); got {value!r}."
                raise ValueError(msg) from None


@functools.cache
def _catalog() -> dict[str, dict[str, Any]]:
    """``list_methods()`` rows by method name."""
    return {row["name"]: row for row in list_methods().to_dict("records")}


def _entry(method: str) -> dict[str, Any]:
    catalog = _catalog()
    if method not in catalog:
        msg = f"Unknown literature method {method!r}; see list_methods() for every name."
        raise KeyError(msg)
    return catalog[method]


def _resolved(method: str, options: Params) -> dict[str, Any]:
    """Every option of the method: the configured value, else its default.
    A required option the configuration does not give is left out."""
    _entry(method)
    signature = inspect.signature(getattr(literature_methods, method))
    call = signature.bind_partial(**dict(options))
    call.apply_defaults()
    return {name: value for name, value in call.arguments.items() if name not in _NOT_OPTIONS}


def _json_ready(value: Any, *, non_finite_as_none: bool = False) -> Any:
    """``value`` in JSON's types: arrays and tuples become lists, NumPy
    scalars Python ones, and a non-finite float None (the package's attrs
    convention) or its ``repr``, such as "inf"."""
    if isinstance(value, dict):
        return {
            str(key): _json_ready(item, non_finite_as_none=non_finite_as_none)
            for key, item in value.items()
        }
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_ready(item, non_finite_as_none=non_finite_as_none) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None if non_finite_as_none else repr(value)
    return value


def resolved_options(config: RecipeConfig) -> dict[str, Any]:
    """The options a call of ``config`` runs with, defaults filled in.

    Parameters
    ----------
    config : RecipeConfig

    Returns
    -------
    options : dict
        Every option of the public method, the configured value or its
        signature's default, in JSON types by the package's convention for
        ``attrs["options"]`` (tuples and arrays as lists, NumPy scalars as
        Python numbers, non-finite values as None), so it equals what a
        successful call records. Known without running, so a failed call
        can record them too. A per-sample array option, which the package
        records by its hash, would be written out in full.

    Raises
    ------
    KeyError
        Unknown method.
    TypeError
        An option the method does not take.
    """
    resolved: dict[str, Any] = _json_ready(
        _resolved(config.method, config.options), non_finite_as_none=True
    )
    return resolved


def policy_inputs(config: RecipeConfig) -> tuple[str, ...]:
    """The inputs the policy supplies for ``config``: its declared requirements.

    Parameters
    ----------
    config : RecipeConfig

    Returns
    -------
    inputs : tuple of str
        ``Recording.from_arrays`` inputs, and ``"behavior_intervals"`` when
        the call takes eligible epochs, in declaration order. A requirement
        applies when every option in its ``when`` has that value in the
        resolved options. Its ``unless`` never lapses: the input it names
        would have to be supplied, and the policy supplies only declared
        inputs (Krause 2022 declares no ``external_ripples``, so it gets
        ``lfps`` and ``speed``). Options are not inputs.

    Raises
    ------
    KeyError
        Unknown method.
    TypeError
        An option the method does not take.
    """
    return _inputs(config.method, config.options)


def _inputs(method: str, options: Params) -> tuple[str, ...]:
    """``policy_inputs`` of a method with these options."""
    requirements = _entry(method)["requirements"]
    resolved = _resolved(method, options)
    return tuple(
        dict.fromkeys(
            requirement["input"]
            for requirement in requirements
            if requirement["kind"] != "option"
            and all(resolved.get(option) == value for option, value in requirement["when"])
        )
    )


def rest_intervals(session: rd.SimulatedSession) -> FloatArray:
    """The session's rest: its samples outside every running bout.

    Parameters
    ----------
    session : SimulatedSession

    Returns
    -------
    intervals : ndarray, shape (n_intervals, 2)
        Sorted, disjoint ``[start, end]`` intervals on recorded timestamps,
        bounds included: the first and last sample of each stretch outside
        ``session.running_intervals``.

    Raises
    ------
    ValueError
        The running bouts cover every sample, so there is no rest to stand
        in for a method's sleep, baseline or eligible epochs.
    """
    running = rd.intervals_to_mask(session.time, session.running_intervals)
    rest = rd.state_intervals(running.astype(float), session.time, 0.5)
    if not len(rest):
        msg = (
            "The session has no rest: its running bouts cover every sample, and "
            "rest stands in for sleep, baseline and eligible epochs."
        )
        raise ValueError(msg)
    return rest


def external_ripples(session: rd.SimulatedSession) -> FloatArray:
    """The external ripple inventory ``EXTERNAL_RIPPLES`` describes.

    Parameters
    ----------
    session : SimulatedSession

    Returns
    -------
    ripples : ndarray, shape (n_ripples, 3)
        Start, end and peak time of each ripple, in seconds.
    """
    filtered = rd.filter_ripple_band(
        session.lfps,
        session.sampling_frequency,
        band=EXTERNAL_RIPPLES["band"],
        time=session.time,
    )
    channel = EXTERNAL_RIPPLES["channel"]
    events = rd.get_detector(EXTERNAL_RIPPLES["detector"]).detector(
        session.time,
        filtered[:, channel : channel + 1],
        session.speed,
        session.sampling_frequency,
        **EXTERNAL_RIPPLES["options"],
    )
    return np.asarray(events[list(EXTERNAL_RIPPLES["columns"])], dtype=float)


def example_ripples(session: rd.SimulatedSession) -> FloatArray:
    """The example ripples ``EXAMPLE_RIPPLES`` describes.

    Parameters
    ----------
    session : SimulatedSession

    Returns
    -------
    examples : ndarray, shape (n_examples, 2)
        Start and end of each example, largest ``max_zscore`` first; fewer
        rows when the detector finds fewer events.
    """
    filtered = rd.filter_ripple_band(
        session.lfps,
        session.sampling_frequency,
        band=EXAMPLE_RIPPLES["band"],
        time=session.time,
    )
    events = rd.get_detector(EXAMPLE_RIPPLES["detector"]).detector(
        session.time,
        filtered,
        session.speed,
        session.sampling_frequency,
        **EXAMPLE_RIPPLES["options"],
    )
    return bounds(events.nlargest(EXAMPLE_RIPPLES["n_examples"], EXAMPLE_RIPPLES["rank_by"]))


def make_recording(session: rd.SimulatedSession, config: RecipeConfig) -> Recording:
    """Build the ``Recording`` a configuration runs on, under ``INPUT_POLICY``.

    Parameters
    ----------
    session : SimulatedSession
        A simulated session with unit types, such as
        ``simulate_network_session`` returns.
    config : RecipeConfig

    Returns
    -------
    recording : Recording
        ``Recording.from_arrays`` of exactly ``policy_inputs(config)`` (bar
        ``behavior_intervals``, which go to each call), at the session's
        rate. A selection of units the session does not label is empty and
        ``check_method`` reports it; nothing stands in for it.

    Raises
    ------
    ValueError
        ``config.input_policy`` is not ``INPUT_POLICY``, or its
        ``assumptions`` are not those its method and options imply.
    KeyError
        Unknown method.
    """
    _check_provenance(config)
    place = np.flatnonzero(session.unit_types == "place")
    sources: dict[str, Callable[[], Any]] = {
        "lfps": lambda: session.lfps,
        "sharp_wave_lfp": lambda: session.sharp_wave_lfp,
        "multiunit": lambda: session.multiunit,
        "speed": lambda: session.speed,
        "place_cells": lambda: place,
        "pyramidal": lambda: np.flatnonzero(
            np.isin(session.unit_types, ("place", "pyramidal"))
        ),
        "templates": lambda: (place,) if len(place) else (),
        "sleep_intervals": lambda: rest_intervals(session),
        "baseline_intervals": lambda: rest_intervals(session),
        "reference_lfp": lambda: np.zeros_like(session.time),
        "external_ripples": lambda: external_ripples(session),
        "example_ripples": lambda: example_ripples(session),
    }
    inputs = {
        name: sources[name]() for name in policy_inputs(config) if name != "behavior_intervals"
    }
    return Recording.from_arrays(session.time, session.sampling_frequency, **inputs)


def behavior_intervals(
    session: rd.SimulatedSession, config: RecipeConfig
) -> FloatArray | None:
    """The eligible epochs the policy passes to a call of ``config``.

    Parameters
    ----------
    session : SimulatedSession
    config : RecipeConfig

    Returns
    -------
    intervals : ndarray, shape (n_intervals, 2), or None
        ``rest_intervals(session)`` when the method declares
        ``behavior_intervals``, else None: epochs a method does not ask for
        would drop events.
    """
    if "behavior_intervals" in policy_inputs(config):
        return rest_intervals(session)
    return None


def run_recipe(
    config: RecipeConfig, recording: Recording, behavior_intervals: FloatArray | None = None
) -> pd.DataFrame:
    """Run a configured method through the package's ``run_method``.

    Parameters
    ----------
    config : RecipeConfig
    recording : Recording
        Usually ``make_recording(session, config)``.
    behavior_intervals : ndarray, shape (n_intervals, 2), optional
        The call's eligible epochs, as ``run_method`` takes them.

    Returns
    -------
    events : pandas.DataFrame
        ``run_method``'s result, attrs and diagnostics untouched.

    Raises
    ------
    KeyError, TypeError, ValueError
        As ``run_method``: an unknown method, an option it does not take or
        a required one missing, or inputs it lacks. Never an empty result
        in their place.
    """
    return run_method(
        config.method, recording, behavior_intervals=behavior_intervals, **dict(config.options)
    )


def check_recipe(
    config: RecipeConfig, recording: Recording, behavior_intervals: FloatArray | None = None
) -> list[str]:
    """What a ``run_recipe`` call would lack, from the package's ``check_method``.

    Parameters
    ----------
    config : RecipeConfig
    recording : Recording
    behavior_intervals : ndarray, shape (n_intervals, 2), optional
        The call's eligible epochs, as ``run_recipe`` would pass them;
        usually ``behavior_intervals(session, config)``.

    Returns
    -------
    problems : list of str
        One line per problem, starting with the input or option it names;
        empty when the call can run.
    """
    return check_method(
        config.method, recording, behavior_intervals=behavior_intervals, **dict(config.options)
    )


def input_policy(config: RecipeConfig) -> dict[str, Any]:
    """The inputs a configuration receives and where each comes from.

    Parameters
    ----------
    config : RecipeConfig

    Returns
    -------
    policy : dict
        JSON-ready: the policy ``name``, the source of each supplied
        ``Recording`` input, the source of the call's ``behavior_intervals``
        (None when not passed), and where the per-session values are
        recorded. A detector behind an input is recorded with every tunable
        of its signature, the configured value or the default. A non-finite
        setting is written as its ``repr`` ("inf"), since None is already a
        setting there (no limit, no mask); the method options in
        ``resolved_options`` instead follow the package's attrs, which write
        it as None.

    Raises
    ------
    ValueError
        ``config.input_policy`` is not ``INPUT_POLICY``, or its
        ``assumptions`` are not those its method and options imply.
    """
    _check_provenance(config)
    inputs = policy_inputs(config)
    sources = {name: _SOURCES[name] for name in inputs if name != "behavior_intervals"}
    for name, source in sources.items():
        if isinstance(source, dict):
            defaults = rd.get_detector(source["detector"]).parameters
            sources[name] = {**source, "options": {**defaults, **source["options"]}}
    return {
        "name": config.input_policy,
        "recording": _json_ready(sources),
        "behavior_intervals": _SOURCES["behavior_intervals"]
        if "behavior_intervals" in inputs
        else None,
        "values": "each result's attrs['inputs'] and attrs['behavior_intervals']",
    }


def method_record(config: RecipeConfig) -> dict[str, str]:
    """One flat, all-string record of a configuration.

    Parameters
    ----------
    config : RecipeConfig

    Returns
    -------
    record : dict of str to str
        In this order: ``method`` (``"recipe:{config_id}"``), ``setting``
        (``"literature"``), ``doi``, ``role`` and ``inventory`` (from
        ``list_methods()``), ``stage`` (the configured stage, or
        ``"detection"`` for a method without a stage option),
        ``primary_expression``, ``resolved_options``, ``input_policy`` and
        ``assumptions`` (as JSON), and ``interpretation`` (from
        ``list_methods()``).

    Raises
    ------
    ValueError
        ``config.input_policy`` is not ``INPUT_POLICY``, or its
        ``assumptions`` are not those its method and options imply.
    """
    _check_provenance(config)
    entry = _entry(config.method)
    options = resolved_options(config)
    return {
        "method": f"recipe:{config.config_id}",
        "setting": "literature",
        "doi": entry["doi"],
        "role": entry["role"],
        "inventory": entry["inventory"],
        "stage": options.get("stage", "detection"),
        "primary_expression": config.primary_expression,
        "resolved_options": json.dumps(options, sort_keys=True, allow_nan=False),
        "input_policy": json.dumps(input_policy(config), sort_keys=True, allow_nan=False),
        "assumptions": json.dumps(list(config.assumptions)),
        "interpretation": entry["interpretation"],
    }


def _assumptions(method: str, options: Params) -> tuple[str, ...]:
    """The benchmark choices a configuration of ``method`` with ``options``
    relies on: the stand-in for each input the policy supplies that is not
    an observation, and each unreported option set to a demonstration value."""
    requirements = _entry(method)["requirements"]
    meanings = {requirement["input"]: requirement["meaning"] for requirement in requirements}
    assumptions = [
        _ASSUMPTIONS[name].replace(":", f', for "{meanings[name]}":', 1)
        if meanings[name]
        else _ASSUMPTIONS[name]
        for name in _inputs(method, options)
        if name in _ASSUMPTIONS
    ]
    unreported = {
        requirement["input"]
        for requirement in requirements
        if requirement["kind"] == "option" and requirement["measured_only"]
    }
    assumptions += [
        f"{name}={value!r}: unreported in the paper; the package's demonstration value"
        for name, value in options
        if name in unreported
    ]
    return tuple(assumptions)


def _check_provenance(config: RecipeConfig) -> None:
    """Raise unless ``config`` names ``INPUT_POLICY`` and states the
    assumptions its method and options imply, so no record is stale."""
    if config.input_policy != INPUT_POLICY:
        msg = (
            f"{config.config_id} names input policy {config.input_policy!r}; "
            f"expected {INPUT_POLICY!r}. Build configurations with configure()."
        )
        raise ValueError(msg)
    expected = _assumptions(config.method, config.options)
    if config.assumptions != expected:
        msg = (
            f"{config.config_id}'s assumptions are not those its method and options "
            f"imply: {list(expected)}. Build configurations with configure()."
        )
        raise ValueError(msg)


def configure(
    method: str, primary_expression: str, *options: tuple[str, Any], label: str = ""
) -> RecipeConfig:
    """A configuration under ``INPUT_POLICY`` with the assumptions it implies.

    Parameters
    ----------
    method : str
        Exact function name from ``list_methods()``.
    primary_expression : str
        ``"ripple"``, ``"sharp_wave"``, ``"burst"`` or ``"network"``.
    *options : (str, object) pairs
        Method options.
    label : str, optional
        A protocol variant's label; the id is then ``"{method}.{label}"``.

    Returns
    -------
    config : RecipeConfig
        Its ``assumptions`` state the stand-in for each input the policy
        supplies that is not an observation, and each unreported option set
        to the package's demonstration value.

    Raises
    ------
    KeyError
        Unknown method.
    TypeError
        An option the method does not take.
    ValueError
        As ``RecipeConfig``.
    """
    return RecipeConfig(
        f"{method}.{label}" if label else method,
        method,
        primary_expression,
        tuple(options),
        INPUT_POLICY,
        _assumptions(method, tuple(options)),
    )


_DETECTION = ("stage", "detection")

# Primary expressions follow what each implemented inventory's events require:
# "ripple" for LFP ripple or SWR events (a participation count or spiking veto
# on those events is a gate, not a burst detection), "burst" for population
# events alone, "network" where the events join an LFP ripple or SWR detection
# with a population-burst detection.
RECIPES: tuple[RecipeConfig, ...] = (
    configure("mallory_2025", "burst"),
    configure("widloski_2025", "ripple"),
    configure("yang_2024", "network"),
    configure("huelin_gorriz_2023", "network"),
    # Long's SWR detector on the pyramidal and radiatum channels, then a
    # pyramidal spiking veto near the ripple peak.
    configure("harvey_2023_code", "ripple", _DETECTION),
    # Difference-of-Gaussians ripples overlapping radiatum sharp waves; the
    # detection stage applies no spiking criterion.
    configure("harvey_2023_text", "ripple", _DETECTION),
    configure("liu_2023", "network"),
    configure("tirole_2022", "network"),
    configure("bush_2022", "burst"),
    configure("berners_lee_2022", "burst"),
    # SWRs trimmed to the stretch of place-cell activity inside them.
    configure("krause_2022", "network"),
    configure("mou_2022", "burst", _DETECTION),
    configure("berners_lee_2021", "ripple"),
    configure("denovellis_2021", "ripple"),
    configure("gillespie_2021", "ripple"),
    configure("michon_2021", "network"),
    # The implemented candidate stage is population-only; the GMM split on
    # ripple power that makes the paper's output "SWR+MUA" is not reproduced.
    configure("igata_2021", "burst"),
    configure("gridchyn_2020", "burst"),
    configure("kaefer_2020", "ripple"),
    configure("bhattarai_2020", "network"),
    configure(
        "stella_2019",
        "ripple",
        ("frequencies", (150.0, 170.0, 190.0, 210.0, 230.0, 250.0)),
        ("cycles", 7.0),
    ),
    configure("xu_2019", "burst"),
    configure("farooq_2019_neuron", "burst"),
    configure("farooq_2019_science", "burst"),
    configure("chenani_2019", "burst"),
    configure("michon_2019", "network"),
    configure("liu_2019", "burst"),
    configure("shin_2019", "ripple", _DETECTION),
    configure("carey_2019", "network"),
    configure("muessig_2019", "network"),
    configure("drieu_2018", "burst", _DETECTION),
    configure("maboudi_2018", "burst"),
    configure("olafsdottir_2017", "burst"),
    configure("olafsdottir_2017", "burst", ("analysis", "trajectory"), label="trajectory"),
    configure("wu_2017", "burst", _DETECTION),
    configure("yamamoto_2017", "network"),
    configure("tang_2017", "ripple"),
    configure("grosmark_2016", "network", _DETECTION),
    configure("ambrose_2016", "ripple"),
    configure("jadhav_2016", "ripple", _DETECTION),
    configure("olafsdottir_2016", "burst"),
    configure("silva_2015", "burst"),
    configure("olafsdottir_2015", "burst"),
    configure(
        "olafsdottir_2015",
        "burst",
        ("minimum_active_units", 7),
        label="bayesian_candidates",
    ),
    configure("pfeiffer_2015", "ripple"),
    configure("wu_2014", "burst"),
    # Ripple-power windows; >= 3 active units and >= 5 spikes is a gate.
    configure("wikenheiser_2013", "ripple", ("window_anchor", "samples")),
    configure("pfeiffer_2013", "burst"),
    configure("carr_2012", "ripple", _DETECTION),
    configure("bendor_2012", "burst"),
    # A gate, not an event definition (role "candidate_gate"); scored as ripples.
    configure("gupta_2010", "ripple"),
    configure("karlsson_2009", "ripple"),
    configure("davidson_2009", "burst"),
    configure("diba_2007", "burst"),
    configure("ji_2007", "burst", _DETECTION),
    configure("foster_2006", "burst"),
    configure("lee_2002", "burst"),
    configure("nadasdy_1999", "ripple", ("rms_window", 0.004), ("bound_threshold", 0.0)),
    configure("kudrimoti_1999", "ripple", ("threshold_sd", 3.0)),
    # Zugaro's FindRipples branch with the same spiking veto as harvey_2023_code.
    configure("harvey_2023_no_radiatum", "ripple", _DETECTION),
    configure("mallory_2025_ripples", "ripple"),
    configure("igata_2021_ripples", "ripple"),
    configure("wu_2014_ripples", "ripple"),
    configure("pfeiffer_2013_ripples", "ripple"),
    configure("davidson_2009_ripples", "ripple"),
    configure("ji_2007_ripples", "ripple"),
    configure("lee_2002_ripples", "ripple"),
    configure("foster_2006_ripples", "ripple"),
    configure("widloski_2025_bursts", "burst"),
    configure("krause_2022_hse", "burst"),
    configure("denovellis_2021_mua", "burst"),
    configure("gillespie_2021_mua", "burst"),
    configure("maboudi_2018_open_field", "burst"),
    configure("muessig_2019_ripples", "ripple"),
    # Ripples with >= 5 active place cells: a participation gate.
    configure("bhattarai_2020_ripples", "ripple"),
    configure("farooq_2019_science_awake", "burst"),
    configure("liu_2019_awake", "burst"),
)

_UNRESAMPLED = (
    "the method requires this rate, the simulated sessions are 1500 Hz, and "
    "resampling is not part of the input policy"
)
_NO_VALUE = "no value is assumed"
_UNREPORTED_RMS = (
    "rms_window, bound_threshold: required options the paper does not report, "
    f"nor do the Csicsvari et al. 1999 methods it cites; {_NO_VALUE}"
)

EXCLUSIONS: dict[str, str] = {
    "bush_2022_ripples": f"input sampled at 4800 Hz: {_UNRESAMPLED}",
    "olafsdottir_2017_ripples": f"input sampled at 1200 Hz: {_UNRESAMPLED}",
    "gridchyn_2020_ripples": _UNREPORTED_RMS,
    "xu_2019_ripples": _UNREPORTED_RMS,
    "farooq_2019_neuron_ripples": (
        "threshold, bound_threshold, smoothing_sigma: required options the paper "
        f"does not report; {_NO_VALUE}"
    ),
    "farooq_2019_science_ripples": (
        "power_measure, bound_threshold: required options (the power definition and "
        f"the event bounds) the paper does not report; {_NO_VALUE}"
    ),
    "chenani_2019_hfe": (
        "ar_coefficients: required AR(2) coefficients fitted per channel, whose fit "
        f"convention the paper does not specify; {_NO_VALUE}"
    ),
    "liu_2019_ripples": (
        f"smoothing_sigma: a required option the paper does not report; {_NO_VALUE}"
    ),
    "liu_2019_ripple_frames": (
        "smoothing_sigma: a required option (of the ripple power it builds on) the "
        f"paper does not report; {_NO_VALUE}"
    ),
    "drieu_2018_ripples": (
        "signal_measure: a required option, since the paper leaves amplitude or power "
        f"unresolved; {_NO_VALUE}"
    ),
    "diba_2007_ripples": (
        "rms_window: a required option the paper does not report. The 1.6 ms window "
        "of the Csicsvari et al. 1999b methods it cites is not assumed: the package "
        "lists RMS windows among settings its source audit did not establish"
    ),
}

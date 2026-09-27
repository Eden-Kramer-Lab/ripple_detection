"""Benchmark configurations of the packaged literature methods.

Each ``RecipeConfig`` names a method of ``ripple_detection.literature_methods``
by its function name, the options it runs with and the expression of a
simulated network event it is headlined against. ``run_recipe`` calls the
installed method through ``run_method``; nothing here reimplements one.

``make_recording`` builds the method's ``Recording`` from a simulated session
with ``Recording.from_arrays``, the measured-data path, under one input policy
(``INPUT_POLICY``): the recording holds exactly the inputs the method declares
(its ``list_methods()`` requirements), taken from the session's observations
and its known unit labels, never from its truth tables. Where a method needs
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

# What each input the policy supplies is, as recorded in methods.csv. The
# values themselves are in each result's attrs["inputs"] and
# attrs["behavior_intervals"].
_REST = "rest: the complement of session.running_intervals within the recording"
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
        "place_cells: every unit the simulator labels 'place'; they also stand in "
        "for any narrower selection the method names (one template's, one "
        "directional template's, one probe sequence's, block-specific or "
        "place-responsive cells), since the simulator has no place fields or "
        "trajectories"
    ),
    "pyramidal": "pyramidal: every unit the simulator labels 'place' or 'pyramidal'",
    "templates": (
        "templates: one template of every place unit, since the simulator has no "
        "place fields or trajectories"
    ),
    "sleep_intervals": (
        "sleep_intervals: rest (outside the running bouts) stands in for the sleep "
        "state, since the simulated sessions are awake"
    ),
    "baseline_intervals": "baseline_intervals: rest stands in for the normalization epoch",
    "behavior_intervals": (
        "behavior_intervals: rest stands in for the eligible epochs; position-defined "
        "epochs have no simulated counterpart, and simulated events occur only at rest"
    ),
    "reference_lfp": "reference_lfp: zeros, since the simulated channels share no reference",
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

_CONFIG_ID = re.compile(r"[a-z0-9_.]+")


@dataclass(frozen=True)
class RecipeConfig:
    """One benchmark configuration of a packaged literature method.

    Attributes
    ----------
    config_id : str
        Unique, stable identifier: the method name, or ``"{method}.{label}"``
        for a protocol variant. Only ``[a-z0-9_.]``.
    method : str
        Exact function name from ``list_methods()``.
    primary_expression : str
        The truth the method is headlined against: ``"ripple"``,
        ``"sharp_wave"``, ``"burst"`` or ``"network"``.
    options : tuple of (str, object) pairs
        Method options, including ``stage`` where the method takes one;
        scalars or tuples, so the configuration is hashable.
    input_policy : str
        The simulation-to-``Recording`` policy, ``INPUT_POLICY``.
    assumptions : tuple of str
        Benchmark choices absent from the source: the policy's stand-ins
        for the inputs this method needs, and demonstration values for
        options the paper does not report.

    Raises
    ------
    ValueError
        The identifier has other characters or does not start with the
        method name, or the expression is unknown.
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
                f"'{self.method}.<label>', using only a-z, 0-9, '_' and '.'."
            )
            raise ValueError(msg)
        if self.primary_expression not in PRIMARY_EXPRESSIONS:
            msg = (
                f"primary_expression {self.primary_expression!r} is not one of "
                f"{', '.join(PRIMARY_EXPRESSIONS)}."
            )
            raise ValueError(msg)


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


def _resolved(config: RecipeConfig) -> dict[str, Any]:
    """Every option of the method: the configured value, else its default.
    A required option the configuration does not give is left out."""
    signature = inspect.signature(getattr(literature_methods, config.method))
    call = signature.bind_partial(**dict(config.options))
    call.apply_defaults()
    return {
        name: value
        for name, value in call.arguments.items()
        if name not in {"rec", "behavior_intervals"}
    }


def _json_ready(value: Any) -> Any:
    """``value`` in JSON's types: tuples become lists, non-finite floats
    their ``repr`` ("inf")."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
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
        signature's default, in JSON types (tuples as lists), as a
        successful call's ``attrs["options"]`` records them. Known without
        running, so a failed call can record them too.

    Raises
    ------
    KeyError
        Unknown method.
    TypeError
        An option the method does not take.
    """
    _entry(config.method)
    resolved: dict[str, Any] = _json_ready(_resolved(config))
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
    requirements = _entry(config.method)["requirements"]
    options = _resolved(config)
    return tuple(
        dict.fromkeys(
            requirement["input"]
            for requirement in requirements
            if requirement["kind"] != "option"
            and all(options.get(option) == value for option, value in requirement["when"])
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
    events = rd.Zugaro_ripple_detector(
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
    events = rd.Kay_ripple_detector(
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
        ``config.input_policy`` is not ``INPUT_POLICY``.
    KeyError
        Unknown method.
    """
    if config.input_policy != INPUT_POLICY:
        msg = (
            f"{config.config_id} names input policy {config.input_policy!r}; "
            f"expected {INPUT_POLICY!r}."
        )
        raise ValueError(msg)
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
        ``Recording`` input (external detectors with their settings), the
        source of the call's ``behavior_intervals`` (None when not passed),
        and where the per-session values are recorded.
    """
    inputs = policy_inputs(config)
    return {
        "name": config.input_policy,
        "recording": _json_ready(
            {name: _SOURCES[name] for name in inputs if name != "behavior_intervals"}
        ),
        "behavior_intervals": _SOURCES["behavior_intervals"]
        if "behavior_intervals" in inputs
        else None,
        "values": "each result's attrs['inputs'] and attrs['behavior_intervals']",
    }


def method_record(config: RecipeConfig) -> dict[str, str]:
    """A configuration's row of ``methods.csv``, without ``session_id``.

    Parameters
    ----------
    config : RecipeConfig

    Returns
    -------
    record : dict of str to str
        ``method`` (``"recipe:{config_id}"``), ``setting``
        (``"literature"``), ``doi``, ``role``, ``inventory`` (from
        ``list_methods()``), ``stage``, ``primary_expression``,
        ``resolved_options``, ``input_policy`` and ``assumptions`` (JSON),
        and ``interpretation``, in that order.
    """
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


def _configure(
    method: str, primary_expression: str, *options: tuple[str, Any], label: str = ""
) -> RecipeConfig:
    """A configuration under ``INPUT_POLICY``, its assumptions derived from
    the inputs the policy supplies and the unreported options it sets."""
    config = RecipeConfig(
        f"{method}.{label}" if label else method,
        method,
        primary_expression,
        tuple(options),
        INPUT_POLICY,
    )
    meanings = {
        requirement["input"]: requirement["meaning"]
        for requirement in _entry(method)["requirements"]
    }
    assumptions = [
        _ASSUMPTIONS[name].replace(":", f" ({meanings[name]}):", 1)
        if meanings[name]
        else _ASSUMPTIONS[name]
        for name in policy_inputs(config)
        if name in _ASSUMPTIONS
    ]
    unreported = {
        requirement["input"]
        for requirement in _entry(method)["requirements"]
        if requirement["kind"] == "option" and requirement["measured_only"]
    }
    assumptions += [
        f"{name}={value!r}: unreported in the paper; the package's demonstration value"
        for name, value in options
        if name in unreported
    ]
    return RecipeConfig(
        config.config_id,
        method,
        primary_expression,
        config.options,
        INPUT_POLICY,
        tuple(assumptions),
    )


_DETECTION = ("stage", "detection")

# Primary expressions follow what each implemented inventory's events require:
# "ripple" for LFP ripple or SWR events (a participation count or spiking veto
# on those events is a gate, not a burst detection), "burst" for population
# events alone, "network" where the events join an LFP ripple or SWR detection
# with a population-burst detection.
RECIPES: tuple[RecipeConfig, ...] = (
    _configure("mallory_2025", "burst"),
    _configure("widloski_2025", "ripple"),
    _configure("yang_2024", "network"),
    _configure("huelin_gorriz_2023", "network"),
    # Long's SWR detector with a pyramidal spiking veto near the ripple peak.
    _configure("harvey_2023_code", "ripple", _DETECTION),
    _configure("harvey_2023_text", "ripple", _DETECTION),
    _configure("liu_2023", "network"),
    _configure("tirole_2022", "network"),
    _configure("bush_2022", "burst"),
    _configure("berners_lee_2022", "burst"),
    # SWRs trimmed to the stretch of place-cell activity inside them.
    _configure("krause_2022", "network"),
    _configure("mou_2022", "burst", _DETECTION),
    _configure("berners_lee_2021", "ripple"),
    _configure("denovellis_2021", "ripple"),
    _configure("gillespie_2021", "ripple"),
    _configure("michon_2021", "network"),
    # The implemented candidate stage is population-only; the GMM split on
    # ripple power that makes the paper's output "SWR+MUA" is not reproduced.
    _configure("igata_2021", "burst"),
    _configure("gridchyn_2020", "burst"),
    _configure("kaefer_2020", "ripple"),
    _configure("bhattarai_2020", "network"),
    _configure(
        "stella_2019",
        "ripple",
        ("frequencies", (150.0, 170.0, 190.0, 210.0, 230.0, 250.0)),
        ("cycles", 7.0),
    ),
    _configure("xu_2019", "burst"),
    _configure("farooq_2019_neuron", "burst"),
    _configure("farooq_2019_science", "burst"),
    _configure("chenani_2019", "burst"),
    _configure("michon_2019", "network"),
    _configure("liu_2019", "burst"),
    _configure("shin_2019", "ripple", _DETECTION),
    _configure("carey_2019", "network"),
    _configure("muessig_2019", "network"),
    _configure("drieu_2018", "burst", _DETECTION),
    _configure("maboudi_2018", "burst"),
    _configure("olafsdottir_2017", "burst"),
    _configure("olafsdottir_2017", "burst", ("analysis", "trajectory"), label="trajectory"),
    _configure("wu_2017", "burst", _DETECTION),
    _configure("yamamoto_2017", "network"),
    _configure("tang_2017", "ripple"),
    _configure("grosmark_2016", "network", _DETECTION),
    _configure("ambrose_2016", "ripple"),
    _configure("jadhav_2016", "ripple", _DETECTION),
    _configure("olafsdottir_2016", "burst"),
    _configure("silva_2015", "burst"),
    _configure("olafsdottir_2015", "burst"),
    _configure(
        "olafsdottir_2015",
        "burst",
        ("minimum_active_units", 7),
        label="bayesian_candidates",
    ),
    _configure("pfeiffer_2015", "ripple"),
    _configure("wu_2014", "burst"),
    # Ripple-power windows; >= 3 active units and >= 5 spikes is a gate.
    _configure("wikenheiser_2013", "ripple", ("window_anchor", "samples")),
    _configure("pfeiffer_2013", "burst"),
    _configure("carr_2012", "ripple", _DETECTION),
    _configure("bendor_2012", "burst"),
    # A gate, not an event definition (role "candidate_gate"); scored as ripples.
    _configure("gupta_2010", "ripple"),
    _configure("karlsson_2009", "ripple"),
    _configure("davidson_2009", "burst"),
    _configure("diba_2007", "burst"),
    _configure("ji_2007", "burst", _DETECTION),
    _configure("foster_2006", "burst"),
    _configure("lee_2002", "burst"),
    _configure("nadasdy_1999", "ripple", ("rms_window", 0.004), ("bound_threshold", 0.0)),
    _configure("kudrimoti_1999", "ripple", ("threshold_sd", 3.0)),
    # Zugaro's FindRipples branch with the same spiking veto as harvey_2023_code.
    _configure("harvey_2023_no_radiatum", "ripple", _DETECTION),
    _configure("mallory_2025_ripples", "ripple"),
    _configure("igata_2021_ripples", "ripple"),
    _configure("wu_2014_ripples", "ripple"),
    _configure("pfeiffer_2013_ripples", "ripple"),
    _configure("davidson_2009_ripples", "ripple"),
    _configure("ji_2007_ripples", "ripple"),
    _configure("lee_2002_ripples", "ripple"),
    _configure("foster_2006_ripples", "ripple"),
    _configure("widloski_2025_bursts", "burst"),
    _configure("krause_2022_hse", "burst"),
    _configure("denovellis_2021_mua", "burst"),
    _configure("gillespie_2021_mua", "burst"),
    _configure("maboudi_2018_open_field", "burst"),
    _configure("muessig_2019_ripples", "ripple"),
    # Ripples with >= 5 active place cells: a participation gate.
    _configure("bhattarai_2020_ripples", "ripple"),
    _configure("farooq_2019_science_awake", "burst"),
    _configure("liu_2019_awake", "burst"),
)

_UNRESAMPLED = (
    "the method requires this rate, the simulated sessions are 1500 Hz, and "
    "resampling is not part of the input policy"
)
_NO_VALUE = "no value is assumed"

EXCLUSIONS: dict[str, str] = {
    "bush_2022_ripples": f"input sampled at 4800 Hz: {_UNRESAMPLED}",
    "olafsdottir_2017_ripples": f"input sampled at 1200 Hz: {_UNRESAMPLED}",
    "gridchyn_2020_ripples": (
        "rms_window, bound_threshold: required options the paper does not report, "
        f"nor do the Csicsvari et al. 1999 methods it cites; {_NO_VALUE}"
    ),
    "xu_2019_ripples": (
        "rms_window, bound_threshold: required options the paper does not report, "
        f"nor do the Csicsvari et al. 1999 methods it cites; {_NO_VALUE}"
    ),
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

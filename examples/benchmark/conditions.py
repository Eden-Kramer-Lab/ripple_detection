"""Simulation conditions of the detector benchmark.

``REFERENCE`` holds every keyword of the three simulator calls that make a
benchmark session, at its reference value, in four sections: ``"session"``
(the recording's length and rate), ``"events"`` (``draw_network_events``),
``"non_events"`` (``draw_non_events``) and ``"render"``
(``simulate_network_session``). A ``Condition`` changes some of them, each named
by a dotted key: ``"events.ripple_snr"``, or ``"non_events.rates.emg"`` for one
entry of a mapping. ``conditions()`` lists the benchmark's conditions: the
reference, one factor at a time, and two crossed pairs of factors;
``select_conditions`` reads a command line's selection of them.
``REFERENCE_REVISIONS`` records every change to a ``REFERENCE`` value after it
was first set, with its reason and evidence.

``simulate_condition`` renders replicate ``k`` of a condition, and
``simulate_parameters`` of a saved parameter set (``parameters_from_json``).
Replicate ``k`` has the same seed in every condition (common random numbers),
so conditions are compared replicate by replicate.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

import ripple_detection as rd
from ripple_detection.core import FloatArray

Params = tuple[tuple[str, Any], ...]

# Every value is the simulator's own default, the reference whose sources are
# in draw_network_events' Notes; a mapping whose default is None is written out.
# channel_gains stays None (1 on every channel): the channel count varies.
REFERENCE: dict[str, dict[str, Any]] = {
    "session": {
        "duration_s": 600.0,
        "sampling_frequency": 1500.0,
    },
    "events": {
        "event_rate": 0.3,
        "type_probabilities": {
            "swr": 0.55,
            "weak_ripple": 0.15,
            "burst_only": 0.10,
            "ripple_doublet": 0.10,
            "sharp_wave_only": 0.10,
        },
        "ripple_duration": (0.042, 0.21),  # revised: see REFERENCE_REVISIONS
        "ripple_skew": (0.5, 0.7),
        "ripple_frequency": (160.0, 220.0),
        "ripple_chirp": (0.0, 30.0),
        "ripple_snr": (2.5, 6.0),
        "weak_ripple_snr": (1.2, 2.2),
        "sharp_wave_duration": (0.04, 0.12),
        "sharp_wave_amplitude": (3.0, 8.0),
        "sharp_wave_lag": 0.01,
        "burst_duration_ratio": (1.0, 1.5),
        "burst_lag": 0.01,
        "burst_gain": 34.0,  # revised: see REFERENCE_REVISIONS
        "participation": (0.2, 0.6),
        "weak_participation": (0.02, 0.1),
        "burst_only_duration": (0.05, 0.3),
        "doublet_interval": (0.06, 0.12),
        "minimum_separation": 0.05,
        "strength_correlation": 0.0,
        "envelope_power": 2,
    },
    "non_events": {
        "rates": {"spike_leakage": 2.0, "emg": 1.0, "fast_gamma": 2.0, "theta_burst": 6.0},
        "n_channels": 4,
        "spike_leakage_units": (1, 3),
        "spike_leakage_spikes": (3, 8),
        "spike_leakage_isi": (0.003, 0.006),
        "spike_leakage_amplitude": 2.0,
        "emg_duration": (0.05, 0.5),
        "emg_amplitude": 1.5,
        "fast_gamma_frequency": (60.0, 100.0),
        "fast_gamma_band": (60.0, 100.0),
        "fast_gamma_duration": (0.05, 0.15),
        "fast_gamma_snr": (1.5, 4.0),
        "theta_burst_units": (5, 15),
        "theta_burst_duration": (0.1, 0.3),
        "theta_burst_gain": 10.0,
    },
    "render": {
        "n_channels": 4,
        "unit_counts": {"place": 40, "pyramidal": 10, "interneuron": 10},
        "baseline_rate": {
            "place": (0.1, 0.5),
            "pyramidal": (0.5, 1.5),
            "interneuron": (8.0, 15.0),
        },
        "channel_gains": None,
        "spatial_profile": "global",
        "channel_occupancy": 1.0,
        "channel_gain_range": (1.0, 1.0),
        "channel_delay": 0.0,
        "shared_noise_fraction": 0.5,
        "noise_type": "pink",
        "noise_amplitude": 1.3,
        "noise_log_amplitude": 0.0,
        "noise_modulation_period": 60.0,
        "sharp_wave_leak": 0.3,
        "ripple_leak": 0.3,
        "interneuron_gain": 3.0,
        "spike_model": "poisson",
        "refractory_period": 0.002,
        "peak_speed": 30.0,
        "theta_amplitude": 4.0,
        "delta_amplitude": 4.0,
    },
}


@dataclass(frozen=True)
class ReferenceRevision:
    """A change to a ``REFERENCE`` value after it was first set.

    Attributes
    ----------
    key : str
        The value's dotted key, such as ``"events.ripple_duration"``.
    previous, revised : object
        The value before and after the revision.
    reason : str
        Why it changed.
    evidence : str
        What shows the change is needed, such as a calibration or a source.
    """

    key: str
    previous: Any
    revised: Any
    reason: str
    evidence: str


# Every revision of a REFERENCE value, oldest first; the simulator validation
# report lists them.
REFERENCE_REVISIONS: tuple[ReferenceRevision, ...] = (
    ReferenceRevision(
        key="events.ripple_duration",
        previous=(0.03, 0.15),
        revised=(0.042, 0.21),
        reason=(
            "The nominal 30-150 ms span read Buzsaki 2015's ripple durations, whose "
            "convention is unstated, as a span at three side scales. Measured by the "
            "ripple_duration target's convention (a 17 ms RMS of the 100-250 Hz band above "
            "the noise mean plus 2 SD), those spans gave swr ripples a median of 34 ms, "
            "below the target's 40-60 ms. The range is scaled by the smallest factor, on a "
            "0.1 grid from 1.0 to 2.0, whose median lies inside the target by at least two "
            "bootstrap standard errors: 1.4. SNR and every other value are unchanged."
        ),
        evidence=(
            "Calibration on the reference with replicates 20000-20019, separate from the "
            "report's, 600 s each, no detector run; median of the swr ripples' durations "
            "that cross (standard error): x1.0 33.3 ms (0.64), x1.3 40.0 (0.71), "
            "x1.4 42.7 (0.82), x2.0 56.0 (1.16). The first report, on 5 replicates, gave "
            "34 ms."
        ),
    ),
    ReferenceRevision(
        key="events.burst_gain",
        previous=40.0,
        revised=34.0,
        reason=(
            "The gain of 40 was assumed, taken from examples/literature_recipes.py. After "
            "the ripple span revision, longer bursts put the pyramidal_ripple_gain ratio "
            "at the target's upper bound (reference 9.96, coupled 10.00, quartic 10.17 "
            "against 5-10). The gain is set, on a grid of whole numbers from 30 to 40, to "
            "the value whose reference ratio is closest to the source's reported mean, "
            "8.6 (Csicsvari et al. 1999, p. 278): 34."
        ),
        evidence=(
            "Calibration on the reference with the revised spans, replicates "
            "20000-20019, 600 s each, no detector run, the ratio pooled as the report "
            "pools it: gain 33 8.25, 34 8.64, 35 8.96, 40 10.21. The 20-replicate report "
            "with gain 40 gave 9.96."
        ),
    ),
)

# The label of the reference's place among a factor's levels.
REFERENCE_LEVEL = "reference"

# One factor at a time: factor -> label -> overrides, every level in the
# designed order, the reference's (REFERENCE_LEVEL, no overrides) in place.
_GRID: dict[str, dict[str, dict[str, Any]]] = {
    "ripple_snr": {
        "low": {"events.ripple_snr": (1.5, 3.0)},
        REFERENCE_LEVEL: {},
        "high": {"events.ripple_snr": (4.0, 10.0)},
    },
    "participation": {
        "low": {"events.participation": (0.05, 0.2)},
        REFERENCE_LEVEL: {},
        "high": {"events.participation": (0.5, 0.9)},
    },
    "n_units": {
        "30": {"render.unit_counts": {"place": 20, "pyramidal": 5, "interneuron": 5}},
        REFERENCE_LEVEL: {},
        "120": {"render.unit_counts": {"place": 80, "pyramidal": 20, "interneuron": 20}},
    },
    "n_channels": {
        "1": {"render.n_channels": 1, "non_events.n_channels": 1},
        REFERENCE_LEVEL: {},
        "16": {"render.n_channels": 16, "non_events.n_channels": 16},
    },
    "shared_noise_fraction": {
        "0.2": {"render.shared_noise_fraction": 0.2},
        REFERENCE_LEVEL: {},
        "0.8": {"render.shared_noise_fraction": 0.8},
    },
    "noise_type": {
        REFERENCE_LEVEL: {},
        "brown": {"render.noise_type": "brown"},
    },
    "event_rate": {
        "0.15": {"events.event_rate": 0.15},
        REFERENCE_LEVEL: {},
        "0.6": {"events.event_rate": 0.6},
    },
    "type_mix": {
        REFERENCE_LEVEL: {},
        "swr_only": {"events.type_probabilities": {"swr": 1.0}},
        "hard": {
            "events.type_probabilities": {
                "swr": 0.25,
                "weak_ripple": 0.3,
                "burst_only": 0.15,
                "ripple_doublet": 0.15,
                "sharp_wave_only": 0.15,
            }
        },
    },
    "burst_lag": {
        "0.0": {"events.burst_lag": 0.0},
        REFERENCE_LEVEL: {},
        "0.03": {"events.burst_lag": 0.03},
    },
    "ripple_chirp": {
        "none": {"events.ripple_chirp": (0.0, 0.0)},
        REFERENCE_LEVEL: {},
    },
    "spike_leakage_rate": {
        "0": {"non_events.rates.spike_leakage": 0.0},
        REFERENCE_LEVEL: {},
        "6": {"non_events.rates.spike_leakage": 6.0},
    },
    "emg_rate": {
        "0": {"non_events.rates.emg": 0.0},
        REFERENCE_LEVEL: {},
        "3": {"non_events.rates.emg": 3.0},
    },
    "fast_gamma_rate": {
        "0": {"non_events.rates.fast_gamma": 0.0},
        REFERENCE_LEVEL: {},
        "6": {"non_events.rates.fast_gamma": 6.0},
    },
    "theta_burst_rate": {
        "0": {"non_events.rates.theta_burst": 0.0},
        REFERENCE_LEVEL: {},
        "18": {"non_events.rates.theta_burst": 18.0},
    },
    "slow_amplitude": {
        "0": {"render.theta_amplitude": 0.0, "render.delta_amplitude": 0.0},
        REFERENCE_LEVEL: {},
        "8": {"render.theta_amplitude": 8.0, "render.delta_amplitude": 8.0},
    },
}

# The simulator's alternative models, one at a time, as _GRID: the reference first.
_ALTERNATIVES: dict[str, dict[str, dict[str, Any]]] = {
    "strength_correlation": {
        REFERENCE_LEVEL: {},
        "coupled": {"events.strength_correlation": 0.6},
    },
    "spatial_profile": {
        REFERENCE_LEVEL: {},
        "local": {
            "render.spatial_profile": "local",
            "render.channel_occupancy": 0.5,
            "render.channel_gain_range": (0.5, 1.0),
            "render.channel_delay": 0.002,
        },
    },
    "noise_modulation": {
        REFERENCE_LEVEL: {},
        "varying": {
            "render.noise_log_amplitude": 0.35,
            "render.noise_modulation_period": 60.0,
        },
    },
    "fast_gamma_band": {
        REFERENCE_LEVEL: {},
        "nearby": {
            "non_events.fast_gamma_frequency": (90.0, 140.0),
            "non_events.fast_gamma_band": (90.0, 140.0),
        },
    },
    "spike_model": {
        REFERENCE_LEVEL: {},
        "refractory": {
            "render.spike_model": "refractory",
            "render.refractory_period": 0.002,
        },
    },
    "envelope_power": {
        REFERENCE_LEVEL: {},
        "quartic": {"events.envelope_power": 4},
    },
}

# Pairs of factors crossed at every level; the cells not already in the grid.
_CROSSED = (("ripple_snr", "participation"), ("ripple_snr", "spike_leakage_rate"))

_FIRST_SEED = 20260924
_REST_DURATION = (20.0, 40.0)  # seconds, before each bout
_BOUT_DURATION = (10.0, 20.0)  # seconds
_FINAL_REST = 5.0  # the least rest after the last bout, seconds


@dataclass(frozen=True)
class Condition:
    """One simulation condition of the benchmark.

    Frozen, but not hashable in general: ``params`` may hold a mapping (such
    as ``render.unit_counts``), so key a collection of conditions by
    ``condition_id``.

    Attributes
    ----------
    condition_id : str
        ``"reference"``, ``"{factor}={label}"`` for one factor, or
        ``"{factor1}={label1},{factor2}={label2}"`` for a crossed cell; only
        ``[A-Za-z0-9_.=,-]``, so it can name a directory.
    factor : str
        ``"reference"``, the factor's name, or ``"{factor1},{factor2}"``.
    level : str
        ``"reference"``, the level's label, or ``"{label1},{label2}"``.
    params : tuple of (str, object) pairs
        The values that differ from ``REFERENCE``, by dotted key
        (``"events.ripple_snr"``, ``"non_events.rates.emg"``), in order.
    """

    condition_id: str
    factor: str
    level: str
    params: Params = ()


def conditions() -> tuple[Condition, ...]:
    """Every condition of the benchmark, in a fixed order.

    Returns
    -------
    conditions : tuple of Condition
        43: the reference; each factor's other levels in turn (28); the
        simulator's six alternative models (``strength_correlation=coupled``,
        ``spatial_profile=local``, ``noise_modulation=varying``,
        ``fast_gamma_band=nearby``, ``spike_model=refractory``,
        ``envelope_power=quartic``); and the cells of ``ripple_snr`` crossed
        with ``participation`` and with ``spike_leakage_rate`` whose levels
        both differ from the reference (8). A factor that sets several
        values lists them all; a rate replaces one entry of
        ``non_events.rates``, the others keeping their reference values.
    """
    found = [Condition("reference", "reference", "reference")]
    for factor, levels in (*_GRID.items(), *_ALTERNATIVES.items()):
        found.extend(
            Condition(f"{factor}={label}", factor, label, tuple(overrides.items()))
            for label, overrides in levels.items()
            if label != REFERENCE_LEVEL
        )
    for first, second in _CROSSED:
        found.extend(
            Condition(
                f"{first}={label1},{second}={label2}",
                f"{first},{second}",
                f"{label1},{label2}",
                (*overrides1.items(), *overrides2.items()),
            )
            for label1, overrides1 in _GRID[first].items()
            for label2, overrides2 in _GRID[second].items()
            if REFERENCE_LEVEL not in (label1, label2)
        )
    # a copy, so a caller changing a mapping value changes no later call's
    return copy.deepcopy(tuple(found))


def factor_levels(factor: str) -> tuple[str, ...]:
    """Every level of a one-factor ``factor``, in its designed order.

    Parameters
    ----------
    factor : str
        A factor of the grid (``"ripple_snr"``) or an alternative model
        (``"spike_model"``), as a condition's ``factor`` names it.

    Returns
    -------
    levels : tuple of str
        The labels, the reference's in place as ``"reference"`` (not its
        value): ``("low", "reference", "high")`` for ``ripple_snr``,
        ``("reference", "brown")`` for ``noise_type``, ``("none",
        "reference")`` for ``ripple_chirp``; the reference first for each
        alternative model.

    Raises
    ------
    ValueError
        ``factor`` is not a one-factor factor (``"reference"`` and a crossed
        pair are not).
    """
    levels = {**_GRID, **_ALTERNATIVES}
    if factor not in levels:
        msg = f"Unknown factor {factor!r}; use one of {', '.join(levels)}."
        raise ValueError(msg)
    return tuple(levels[factor])


def select_conditions(text: str) -> tuple[Condition, ...]:
    """Conditions by ``"all"`` or comma-separated ids, in the benchmark's order.

    A crossed cell's id holds a comma itself, so the longest known id is taken
    at each position: ``"ripple_snr=low,participation=low"`` is the crossed
    cell, never its two one-factor conditions.

    Parameters
    ----------
    text : str
        ``"all"``, or condition ids separated by commas (spaces around an id
        are ignored), as a command line's ``--conditions`` takes them.

    Returns
    -------
    selected : tuple of Condition
        Each named condition once, in ``conditions()`` order.

    Raises
    ------
    ValueError
        An id is not a condition's, or ``text`` names none.
    """
    everything = conditions()
    if text.strip() == "all":
        return everything
    known = {condition.condition_id for condition in everything}
    tokens = [token.strip() for token in text.split(",") if token.strip()]
    if not tokens:
        msg = f"No condition id in {text!r}; use 'all' or ids of conditions()."
        raise ValueError(msg)
    chosen, position = set(), 0
    while position < len(tokens):
        for end in range(len(tokens), position, -1):
            candidate = ",".join(tokens[position:end])
            if candidate in known:
                chosen.add(candidate)
                position = end
                break
        else:
            msg = f"Unknown condition {tokens[position]!r}; use 'all' or ids of conditions()."
            raise ValueError(msg)
    return tuple(c for c in everything if c.condition_id in chosen)


def session_seed(replicate: int) -> int:
    """The seed of replicate ``replicate``, the same in every condition.

    Parameters
    ----------
    replicate : int
        From 0.

    Returns
    -------
    seed : int
        ``20260924 + replicate``.
    """
    return _FIRST_SEED + replicate


def running_schedule(duration_s: float, rng: np.random.Generator) -> FloatArray:
    """Running bouts separated by rest, starting and ending at rest.

    Rest of ``U(20, 40)`` s and a bout of ``U(10, 20)`` s alternate from the
    start; the first bout that would leave less than 5 s of rest before the
    end is dropped, and with it every later one. So a session shorter than
    35 s has no bout.

    Parameters
    ----------
    duration_s : float
        The session's length, seconds.
    rng : numpy.random.Generator
        Draws each rest, then its bout, until a bout is dropped.

    Returns
    -------
    bouts : ndarray, shape (n_bouts, 2)
        Start and end of each bout, seconds from the session's first sample,
        sorted and not overlapping.

    Raises
    ------
    ValueError
        ``duration_s`` is not finite.
    """
    if not np.isfinite(duration_s):
        msg = f"duration_s must be finite, got {duration_s}."
        raise ValueError(msg)
    bouts = []
    end = 0.0
    while True:
        start = end + rng.uniform(*_REST_DURATION)
        end = start + rng.uniform(*_BOUT_DURATION)
        if duration_s - end < _FINAL_REST:
            break
        bouts.append((start, end))
    return np.array(bouts, dtype=float).reshape(-1, 2)


def _set(parameters: dict[str, dict[str, Any]], key: str, value: Any) -> None:
    """Set ``parameters`` at the dotted ``key``, which must already be there."""
    section, *path = key.split(".")
    if section not in parameters:
        msg = f"Unknown section {section!r} in {key!r}; use {', '.join(parameters)}."
        raise ValueError(msg)
    target: Any = parameters[section]
    for name in path[:-1]:
        target = target.get(name) if isinstance(target, dict) else None
    if not path or not isinstance(target, dict) or path[-1] not in target:
        msg = f"Unknown parameter {key!r}: REFERENCE has no such entry."
        raise ValueError(msg)
    target[path[-1]] = copy.deepcopy(value)


def resolve(
    condition: Condition, overrides: Mapping[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    """Every simulation parameter of ``condition``.

    Parameters
    ----------
    condition : Condition
    overrides : mapping of str to object, optional
        Further values by dotted key, applied after the condition's, such as
        ``{"session.duration_s": 60.0}`` for a short session.

    Returns
    -------
    parameters : dict of str to dict
        A deep copy of ``REFERENCE`` with the condition's ``params`` and then
        ``overrides`` applied: the four sections, every keyword in each.

    Raises
    ------
    ValueError
        A dotted key names a section or an entry ``REFERENCE`` lacks.
    """
    parameters = copy.deepcopy(REFERENCE)
    for key, value in (*condition.params, *(overrides or {}).items()):
        _set(parameters, key, value)
    return parameters


def resolved_json(condition: Condition, overrides: Mapping[str, Any] | None = None) -> str:
    """``resolve(condition, overrides)`` as JSON, keys sorted, for saving and
    hashing.

    Parameters
    ----------
    condition : Condition
    overrides : mapping of str to object, optional
        As in ``resolve``.

    Returns
    -------
    text : str
        Tuples are written as lists and ``None`` as ``null``.

    Raises
    ------
    ValueError
        As ``resolve``, or a value is not finite.
    """
    return json.dumps(resolve(condition, overrides), sort_keys=True, allow_nan=False)


# Mapping-valued keywords a condition replaces whole, whose entries may differ
# from REFERENCE's.
_REPLACED_WHOLE = frozenset({"events.type_probabilities"})


def _dotted_keys(parameters: Mapping[str, Any], prefix: str = "") -> set[str]:
    """Every section, keyword and mapping entry of ``parameters`` by dotted
    key, the entries of ``_REPLACED_WHOLE`` aside."""
    keys = set()
    for name, value in parameters.items():
        key = f"{prefix}{name}"
        keys.add(key)
        if isinstance(value, Mapping) and key not in _REPLACED_WHOLE:
            keys |= _dotted_keys(value, f"{key}.")
    return keys


def parameters_from_json(text: str) -> dict[str, dict[str, Any]]:
    """Parameters saved as JSON, as ``resolve`` returns them.

    The inverse of ``resolved_json``, for a saved specification such as a
    run's ``conditions.csv`` ``params``: each list becomes a tuple again (every
    list-like value of ``REFERENCE`` is one), each mapping stays a mapping.

    The saved set must have exactly ``REFERENCE``'s sections, keywords and the
    entries of each mapping-valued keyword, since the simulator reads an entry
    left out as a value of its own: a rate (``non_events.rates``) or a unit
    count (``render.unit_counts``) as 0, a baseline range
    (``render.baseline_rate``) as its default. The conditions change those
    entry by entry, so a missing one was lost. ``events.type_probabilities``
    is the exception: a condition replaces the mixture whole
    (``type_mix=swr_only`` saves ``{"swr": 1.0}``), and a type it leaves out
    never occurs, so its entries are taken as saved.

    Parameters
    ----------
    text : str
        JSON of the four sections, every keyword in each.

    Returns
    -------
    parameters : dict of str to dict
        By section, as ``resolve`` gives them; ``simulate_parameters`` takes it.

    Raises
    ------
    ValueError
        A section, keyword or mapping entry ``REFERENCE`` has is missing, or
        one it lacks is there.
    """

    def restored(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: restored(item) for key, item in value.items()}
        if isinstance(value, list):
            return tuple(restored(item) for item in value)
        return value

    parameters: dict[str, dict[str, Any]] = restored(json.loads(text))
    expected, found = _dotted_keys(REFERENCE), _dotted_keys(parameters)
    if found != expected:
        msg = (
            f"The saved parameters lack {', '.join(sorted(expected - found)) or 'nothing'}; "
            f"have {', '.join(sorted(found - expected)) or 'nothing else'}."
        )
        raise ValueError(msg)
    return parameters


def simulate_condition(
    condition: Condition, replicate: int, overrides: Mapping[str, Any] | None = None
) -> rd.SimulatedSession:
    """Replicate ``replicate`` of ``condition``, simulated.

    ``simulate_parameters(resolve(condition, overrides), replicate)``.

    Parameters
    ----------
    condition : Condition
    replicate : int
        From 0.
    overrides : mapping of str to object, optional
        As in ``resolve``, such as ``{"session.duration_s": 60.0}``.

    Returns
    -------
    session : SimulatedSession
        As ``simulate_network_session`` returns it, with the non-events.

    Raises
    ------
    ValueError
        As ``resolve``, or a simulator function rejects a value.
    """
    return simulate_parameters(resolve(condition, overrides), replicate)


def simulate_parameters(
    parameters: Mapping[str, Mapping[str, Any]], replicate: int
) -> rd.SimulatedSession:
    """Replicate ``replicate`` of a condition's resolved parameters, simulated.

    The session's timestamps start at 0, ``duration_s`` long at
    ``sampling_frequency``. It is built in four stages: the running schedule
    (``running_schedule``), the network events (``draw_network_events``), the
    non-events (``draw_non_events``) and the rendering
    (``simulate_network_session``), each given the parameters of its section,
    the schedule and the sampling rate.

    Randomness: one generator seeded with ``session_seed(replicate)`` draws
    four seeds at once, one per stage in that order, and each stage draws
    only from a generator of its own seed. So a stage's random stream depends
    on the replicate alone, never on how many draws another stage made:
    replicate ``k`` has the same schedule in every condition of one duration,
    and its later stages start from the same streams. Within a stage the
    simulator's own rules apply (see each function's Notes and ``rng``): a
    factor that changes no draw count, such as a size, the strength
    correlation, the envelope power, the spatial profile, the noise
    modulation, the gamma band or the spike model, leaves the event times,
    the units' baseline rates and the per-unit participant draws as they
    are, while one that changes a draw count (the event rate, a rate of
    non-events, the channel or unit count) or which events fit changes the
    draws after it in that stage and in what depends on its table.

    Parameters
    ----------
    parameters : mapping of str to mapping
        Every section and keyword, as ``resolve`` or ``parameters_from_json``
        gives them.
    replicate : int
        From 0.

    Returns
    -------
    session : SimulatedSession
        As ``simulate_network_session`` returns it, with the non-events.

    Raises
    ------
    ValueError
        A simulator function rejects a value.
    """
    duration = parameters["session"]["duration_s"]
    rate = parameters["session"]["sampling_frequency"]
    time = rd.simulate_time(round(duration * rate), rate)
    schedule, events, non_events, render = (
        np.random.default_rng(seed)
        for seed in np.random.default_rng(session_seed(replicate)).integers(
            np.iinfo(np.int64).max, size=4
        )
    )
    running = running_schedule(duration, schedule)
    network_events = rd.draw_network_events(
        time,
        running_intervals=running,
        rng=events,
        sampling_frequency=rate,
        **parameters["events"],
    )
    other_events = rd.draw_non_events(
        time,
        running_intervals=running,
        rng=non_events,
        sampling_frequency=rate,
        **parameters["non_events"],
    )
    return rd.simulate_network_session(
        time,
        network_events,
        non_events=other_events,
        running_intervals=running,
        rng=render,
        sampling_frequency=rate,
        **parameters["render"],
    )

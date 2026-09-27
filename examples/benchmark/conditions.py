"""Simulation conditions of the detector benchmark.

``REFERENCE`` holds every keyword of the three simulator calls that make a
benchmark session, at its reference value, in four sections: ``"session"``
(the recording's length and rate), ``"events"`` (``draw_network_events``),
``"non_events"`` (``draw_non_events``) and ``"render"``
(``simulate_network_session``). A ``Condition`` changes some of them, each named
by a dotted key: ``"events.ripple_snr"``, or ``"non_events.rates.emg"`` for one
entry of a mapping. ``conditions()`` lists the benchmark's conditions: the
reference, one factor at a time, and two crossed pairs of factors.

``simulate_condition`` renders replicate ``k`` of a condition. Replicate ``k``
has the same seed in every condition (common random numbers), so conditions
are compared replicate by replicate.
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
        "ripple_duration": (0.03, 0.15),
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
        "burst_gain": 40.0,
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

# One factor at a time, in order: factor -> label -> overrides, for each level
# other than the reference.
_GRID: dict[str, dict[str, dict[str, Any]]] = {
    "ripple_snr": {
        "low": {"events.ripple_snr": (1.5, 3.0)},
        "high": {"events.ripple_snr": (4.0, 10.0)},
    },
    "participation": {
        "low": {"events.participation": (0.05, 0.2)},
        "high": {"events.participation": (0.5, 0.9)},
    },
    "n_units": {
        "30": {"render.unit_counts": {"place": 20, "pyramidal": 5, "interneuron": 5}},
        "120": {"render.unit_counts": {"place": 80, "pyramidal": 20, "interneuron": 20}},
    },
    "n_channels": {
        "1": {"render.n_channels": 1, "non_events.n_channels": 1},
        "16": {"render.n_channels": 16, "non_events.n_channels": 16},
    },
    "shared_noise_fraction": {
        "0.2": {"render.shared_noise_fraction": 0.2},
        "0.8": {"render.shared_noise_fraction": 0.8},
    },
    "noise_type": {
        "brown": {"render.noise_type": "brown"},
    },
    "event_rate": {
        "0.15": {"events.event_rate": 0.15},
        "0.6": {"events.event_rate": 0.6},
    },
    "type_mix": {
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
        "0.03": {"events.burst_lag": 0.03},
    },
    "ripple_chirp": {
        "none": {"events.ripple_chirp": (0.0, 0.0)},
    },
    "spike_leakage_rate": {
        "0": {"non_events.rates.spike_leakage": 0.0},
        "6": {"non_events.rates.spike_leakage": 6.0},
    },
    "emg_rate": {
        "0": {"non_events.rates.emg": 0.0},
        "3": {"non_events.rates.emg": 3.0},
    },
    "fast_gamma_rate": {
        "0": {"non_events.rates.fast_gamma": 0.0},
        "6": {"non_events.rates.fast_gamma": 6.0},
    },
    "theta_burst_rate": {
        "0": {"non_events.rates.theta_burst": 0.0},
        "18": {"non_events.rates.theta_burst": 18.0},
    },
    "slow_amplitude": {
        "0": {"render.theta_amplitude": 0.0, "render.delta_amplitude": 0.0},
        "8": {"render.theta_amplitude": 8.0, "render.delta_amplitude": 8.0},
    },
}

# The simulator's alternative models, one at a time, as _GRID.
_ALTERNATIVES: dict[str, dict[str, dict[str, Any]]] = {
    "strength_correlation": {
        "coupled": {"events.strength_correlation": 0.6},
    },
    "spatial_profile": {
        "local": {
            "render.spatial_profile": "local",
            "render.channel_occupancy": 0.5,
            "render.channel_gain_range": (0.5, 1.0),
            "render.channel_delay": 0.002,
        },
    },
    "noise_modulation": {
        "varying": {
            "render.noise_log_amplitude": 0.35,
            "render.noise_modulation_period": 60.0,
        },
    },
    "fast_gamma_band": {
        "nearby": {
            "non_events.fast_gamma_frequency": (90.0, 140.0),
            "non_events.fast_gamma_band": (90.0, 140.0),
        },
    },
    "spike_model": {
        "refractory": {
            "render.spike_model": "refractory",
            "render.refractory_period": 0.002,
        },
    },
    "envelope_power": {
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
        )
    # a copy, so a caller changing a mapping value changes no later call's
    return copy.deepcopy(tuple(found))


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


def simulate_condition(
    condition: Condition, replicate: int, overrides: Mapping[str, Any] | None = None
) -> rd.SimulatedSession:
    """Replicate ``replicate`` of ``condition``, simulated.

    The session's timestamps start at 0, ``duration_s`` long at
    ``sampling_frequency``. It is built in four stages: the running schedule
    (``running_schedule``), the network events (``draw_network_events``), the
    non-events (``draw_non_events``) and the rendering
    (``simulate_network_session``), each given the resolved parameters of its
    section, the schedule and the sampling rate.

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
    parameters = resolve(condition, overrides)
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

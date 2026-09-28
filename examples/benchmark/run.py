"""Run the detector benchmark: simulate each condition, run every method, score it.

For each replicate of each condition (``conditions.py``), the runner simulates the
session, runs the package's nine detectors at their defaults and along their
threshold sweeps (``THRESHOLD_SWEEPS``) and every configured literature method
(``recipe_configs.RECIPES``), and scores each against the session's truth windows
of every expression at every level of ``MATCH_IOU_LEVELS``. A method that raises is
recorded in ``failures.csv`` and the run goes on; a missing (session, method,
setting) is a failure, never zero events. An error in the runner's own code, such as
building a recipe's inputs or scoring a result, stops the run. A method's warnings
change nothing: each is recorded in ``warnings.csv``.

Usage, from the repository root (see README.md, "Running the benchmark")::

    uv run python examples/benchmark/run.py --run-name NAME [--conditions all|ID,ID]
        [--replicates N] [--duration S] [--workers N] [--resume] [--smoke]
        [--combine] [--validation-report PATH]

Every run that simulates needs ``--validation-report``, a ready simulator
validation report (``validate_simulator.py``) covering the selected conditions'
resolved parameters; it is checked before any method runs. ``--combine`` alone
rebuilds ``combined/`` and needs none.

Outputs, under ``examples/benchmark/output/<run_name>/`` (git-ignored). Tables are
CSV, ``.csv.gz`` for the large ones; times are seconds, signed errors are detected
minus truth (negative: early). Read a table with ``read_table``, which keeps text as
text: pandas' own reader makes a ``setting`` of ``"3.0"``, or a ``level`` of
``"30"``, a number when no other row of the column says otherwise, and an empty
``doi`` a NaN.

Run files, written when the run starts:

- ``manifest.json``: ``run_name``, ``git_commit``, ``package_version``,
  ``numpy_version``, ``scipy_version``, ``command``, ``started``, ``finished``
  (null until the run ends), ``n_workers`` (the processes the run, or its latest
  resumption, used: at most one per session left to run).
- ``run_spec.json``: what ``--resume`` checks: ``conditions`` (each condition's
  parameters after overrides such as ``--duration``), ``replicates`` and ``seeds``
  per condition, ``methods`` (each method and setting's record, as in
  ``methods.csv``), ``scoring`` (``MATCH_IOU_LEVELS``, ``TRUTH_FRACTIONS``, the
  expressions), ``package_version``, ``git_commit`` (``"<hash>-dirty"`` when
  ``src/`` or ``examples/benchmark/`` has uncommitted changes) and ``validation_report``
  (``path``, ``sha256`` of its spec.json, ``simulation_fingerprint``,
  ``target_table_hash``).
- ``conditions.csv``: ``condition_id``, ``factor``, ``level``, ``params`` (JSON of the
  full parameter set after the condition's and the command line's overrides).

Condition files, in ``conditions/<condition_id>/``, written into
``conditions/<condition_id>.partial/`` and renamed into place after ``done.json``:

- ``done.json``: ``files``, each other file's path relative to the directory with
  its ``rows`` (data rows of a table, null for a JSON file) and ``sha256``.
- ``sessions.csv.gz``, one row per session: ``session_id``
  (``"{condition_id}/{replicate}"``), ``condition_id``, ``replicate``, ``seed``,
  ``duration_s``, ``rest_s`` (outside every running bout), ``event_time_s`` (the
  union of the network windows at fraction 0.1), ``n_events_<type>`` per
  ``EVENT_TYPES`` and ``n_non_events_<type>`` per ``NON_EVENT_TYPES``,
  ``simulate_s``, ``detect_s`` (every method, summary and score).
- ``truth.csv.gz``, one row per component of the latent event table and per
  non-event: ``session_id``, ``table`` (``"event"`` or ``"non_event"``), ``id``,
  ``type``, ``expression``, ``component``, then the event table's other columns
  (``center_time``, ``rise_sigma``, ``decay_sigma``, ``envelope_power``,
  ``amplitude``, ``frequency_start``, ``frequency_end``, ``participation``,
  ``n_participants``) and the non-event table's (``frequency``, ``snr_band_low``,
  ``snr_band_high``, ``channel``, ``n_units``, ``n_spikes``, ``isi``), empty where a
  column is not the row's table's. ``load_truth`` restores both tables.
- ``truth_counts.csv.gz``, one row per truth window of each expression at fraction
  0.1: ``session_id``, ``expression`` (``ripple``, ``sharp_wave``, ``burst``,
  ``network``), ``row`` (its position in ``truth_windows(events, 0.1, expression)``,
  the rows matching uses), ``n_active_units`` and ``n_active_principal`` (units,
  and place or pyramidal units, with a spike in the window).
- ``ripple_channels.csv.gz``: ``session_id``, then every column of
  ``SimulatedSession.ripple_channels`` (``event_id``, ``component``, ``channel``,
  ``gain``, ``delay_s``).
- ``units.csv.gz``: ``session_id``, ``unit``, ``unit_type``, ``baseline_rate``
  (spikes/s, the simulator's drawn baseline).
- ``methods.csv``, one row per session, method and setting run: ``session_id``,
  ``method``, ``setting``, ``doi``, ``role``, ``inventory``, ``stage``,
  ``primary_expression``, ``resolved_options`` (JSON), ``input_policy`` (JSON),
  ``assumptions`` (JSON), ``interpretation``. ``method`` is a registry name or
  ``"recipe:<config_id>"``; ``setting`` is ``"default"``, a swept value as
  ``repr(float(value))``, or ``"literature"``. A detector's ``doi``, ``role``,
  ``inventory`` and ``interpretation`` are empty and its stage ``"detection"``.
- ``events.csv.gz``, one row per detected event: ``session_id``, ``method``,
  ``setting``, ``event_index`` (the event's row position in its result, as
  matching indexes it), ``start_time``, ``end_time``, ``peak_time``,
  ``n_active_units``, ``n_active_principal``.
- ``results/<method_slug>__<setting>.csv.gz`` and ``.json``: each result complete,
  indexed by ``event_number``: ``session_id``, then every column the method
  returned, in its order; a column of tuples (Shvartsman's ``participants``) is
  written as JSON lists. The JSON holds, per session (one with no events too), each
  column's dtype, its ``tuple_columns`` and the result's ``attrs`` (recipes: the
  package's provenance; detectors: ``method``, ``options`` and
  ``ripple_detection_version``). ``method_slug`` is ``method`` with ``:`` replaced
  by ``--``. ``load_results`` reads it back exactly.
- ``metrics.csv.gz``, one row per session, method, setting, expression and
  ``minimum_iou``: ``session_id``, ``method``, ``setting``, ``expression``,
  ``minimum_iou``, ``n_reference``, ``n_detected``, ``n_matched``, ``recall``,
  ``precision``, ``f1``, ``false_positives_per_minute`` (the unmatched detections,
  wherever they lie, over the minutes of the session outside every network window
  at 0.1), ``median_iou``, ``median_coverage``, ``median_temporal_precision``, then,
  against the truth windows at ``f`` percent of the peak (matched at 10),
  ``median_onset_error_<f>`` and ``median_offset_error_<f>`` for ``f`` in 10, 25,
  50, then ``median_abs_onset_error_<f>`` and ``median_abs_offset_error_<f>`` for
  ``f`` in 10, 25, 50, and ``n_split``, ``n_merged``.
- ``failures.csv``, one row per failed call: ``session_id``, ``method``,
  ``setting``, ``error`` (``"{type}: {message}"``, at most 200 characters).
- ``warnings.csv``, one row per warning a call issued, failed calls included, each
  one even when a line repeats it: ``session_id``, ``method``, ``setting``,
  ``category`` (the warning's class name), ``message`` (at most 200 characters).

``combined/`` holds the same tables concatenated over the finished conditions,
with ``results/<condition_id>/`` copied from each, and ``manifest.json``:
``included`` and ``missing``, the run's conditions (``conditions.csv``) it holds
and those it leaves out, not finished. It is derived, rebuilt by ``--combine`` at
any time (into ``combined.partial/``, renamed into place) and at the end of a run,
and never read by ``--resume``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import functools
import gzip
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time as wall_clock
import warnings
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from conditions import (
    TRUTH_FRACTIONS,
    Condition,
    conditions,
    differing_keys,
    resolve,
    resolved_json,
    select_conditions,
    session_seed,
    simulate_condition,
)
from recipe_configs import (
    RECIPES,
    RecipeConfig,
    _json_ready,
    behavior_intervals,
    make_recording,
    method_record,
    run_recipe,
)
from validate_simulator import (
    DEFAULT_REPLICATES,
    ReportNotReady,
    hash_files,
    peak_rss_bytes,
    replace_directory,
    versions,
)
from validate_simulator import require_ready_report as _require_report

import ripple_detection as rd
from ripple_detection.core import FloatArray

MATCH_IOU_LEVELS = (0.0, 0.2, 0.5)
EXPRESSIONS = (*rd.EXPRESSIONS, "network")

_ZSCORE_SWEEP = ("zscore_threshold", (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0))
THRESHOLD_SWEEPS: dict[str, tuple[str, tuple[Any, ...]]] = {
    "Kay_ripple_detector": _ZSCORE_SWEEP,
    "Karlsson_ripple_detector": _ZSCORE_SWEEP,
    "Roumis_ripple_detector": _ZSCORE_SWEEP,
    "Shvartsman_ripple_detector": _ZSCORE_SWEEP,
    "multiunit_HSE_detector": _ZSCORE_SWEEP,
    "Zugaro_ripple_detector": ("high_threshold", (2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)),
    "Carey_candidate_detector": ("high_threshold", (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)),
    "Yu_ripple_detector": ("percentile", (99.0, 99.5, 99.9, 99.95, 99.99, 99.995, 99.999)),
    "Long_sharp_wave_ripple_detector": (
        "peak_thresholds",
        (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0),
    ),
}

# The truth each detector is headlined against; recipes carry their own.
DETECTOR_EXPRESSION: dict[str, str] = {
    "Kay_ripple_detector": "ripple",
    "Karlsson_ripple_detector": "ripple",
    "Roumis_ripple_detector": "ripple",
    "Shvartsman_ripple_detector": "ripple",
    "Yu_ripple_detector": "ripple",
    "Zugaro_ripple_detector": "ripple",
    "Long_sharp_wave_ripple_detector": "ripple",
    "Carey_candidate_detector": "network",
    "multiunit_HSE_detector": "burst",
}

# Replicates per condition unless --replicates says otherwise.
REFERENCE_REPLICATES = 20
REPLICATES = 10

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"

# Long's sweep value v sets both of its (bound, peak) threshold pairs to (0.5, v).
_LONG_BOUND = 0.5
_PRINCIPAL = ("place", "pyramidal")
_ERROR_LENGTH = 200
_RESULT_INDEX = "event_number"
# gzip without a timestamp, so the same rows give the same bytes
_GZIP: dict[str, Any] = {"method": "gzip", "mtime": 0}
# What run_spec.json records of the validation report.
_REPORT_IDENTITY = ("path", "sha256", "simulation_fingerprint", "target_table_hash")

SESSION_COLUMNS = (
    "session_id",
    "condition_id",
    "replicate",
    "seed",
    "duration_s",
    "rest_s",
    "event_time_s",
    *(f"n_events_{kind}" for kind in rd.EVENT_TYPES),
    *(f"n_non_events_{kind}" for kind in rd.NON_EVENT_TYPES),
    "simulate_s",
    "detect_s",
)
# The latent tables' columns, from the simulator's own empty tables.
_TABLE_DTYPES = {
    field.name: dict(field.default_factory().dtypes)  # type: ignore[misc]
    for field in dataclasses.fields(rd.SimulatedSession)
    if field.name in ("events", "non_events", "ripple_channels")
}
_TRUTH_KEYS = {
    "events": ("event_id", "event_type"),
    "non_events": ("non_event_id", "non_event_type"),
}
TRUTH_COLUMNS = (
    "session_id",
    "table",
    "id",
    "type",
    "expression",
    "component",
    *dict.fromkeys(
        name
        for table in ("events", "non_events")
        for name in _TABLE_DTYPES[table]
        if name not in (*_TRUTH_KEYS[table], "expression", "component")
    ),
)
TRUTH_COUNT_COLUMNS = (
    "session_id",
    "expression",
    "row",
    "n_active_units",
    "n_active_principal",
)
RIPPLE_CHANNEL_COLUMNS = ("session_id", *_TABLE_DTYPES["ripple_channels"])
UNIT_COLUMNS = ("session_id", "unit", "unit_type", "baseline_rate")
METHOD_COLUMNS = (
    "session_id",
    "method",
    "setting",
    "doi",
    "role",
    "inventory",
    "stage",
    "primary_expression",
    "resolved_options",
    "input_policy",
    "assumptions",
    "interpretation",
)
EVENT_COLUMNS = (
    "session_id",
    "method",
    "setting",
    "event_index",
    "start_time",
    "end_time",
    "peak_time",
    "n_active_units",
    "n_active_principal",
)
_PERCENTS = tuple(round(100 * fraction) for fraction in TRUTH_FRACTIONS)
SCORE_COLUMNS = (
    "expression",
    "minimum_iou",
    "n_reference",
    "n_detected",
    "n_matched",
    "recall",
    "precision",
    "f1",
    "false_positives_per_minute",
    "median_iou",
    "median_coverage",
    "median_temporal_precision",
    *(
        f"median_{kind}_error_{percent}"
        for percent in _PERCENTS
        for kind in ("onset", "offset")
    ),
    *(
        f"median_abs_{kind}_error_{percent}"
        for percent in _PERCENTS
        for kind in ("onset", "offset")
    ),
    "n_split",
    "n_merged",
)
METRIC_COLUMNS = ("session_id", "method", "setting", *SCORE_COLUMNS)
FAILURE_COLUMNS = ("session_id", "method", "setting", "error")
WARNING_COLUMNS = ("session_id", "method", "setting", "category", "message")
CONDITION_COLUMNS = ("condition_id", "factor", "level", "params")


@dataclasses.dataclass(frozen=True)
class Table:
    """A table the runner writes.

    Attributes
    ----------
    columns : tuple of str
        In order.
    text : frozenset of str
        The columns that hold text in every row, ``""`` included.
    """

    columns: tuple[str, ...]
    text: frozenset[str]


_KEY = ("session_id", "method", "setting")
# Every table a condition directory holds, by file name.
TABLES: dict[str, Table] = {
    "sessions.csv.gz": Table(SESSION_COLUMNS, frozenset({"session_id", "condition_id"})),
    "truth.csv.gz": Table(TRUTH_COLUMNS, frozenset({"session_id", "table", "type"})),
    "truth_counts.csv.gz": Table(TRUTH_COUNT_COLUMNS, frozenset({"session_id", "expression"})),
    "ripple_channels.csv.gz": Table(RIPPLE_CHANNEL_COLUMNS, frozenset({"session_id"})),
    "units.csv.gz": Table(UNIT_COLUMNS, frozenset({"session_id", "unit_type"})),
    "methods.csv": Table(METHOD_COLUMNS, frozenset(METHOD_COLUMNS)),
    "events.csv.gz": Table(EVENT_COLUMNS, frozenset(_KEY)),
    "metrics.csv.gz": Table(METRIC_COLUMNS, frozenset({*_KEY, "expression"})),
    "failures.csv": Table(FAILURE_COLUMNS, frozenset(FAILURE_COLUMNS)),
    "warnings.csv": Table(WARNING_COLUMNS, frozenset(WARNING_COLUMNS)),
}
# The run's table of its conditions.
CONDITIONS_TABLE = Table(CONDITION_COLUMNS, frozenset(CONDITION_COLUMNS))

# (method, setting, prepare): prepare builds the call's inputs and returns the
# method's call itself.
MethodCall = tuple[str, str, Callable[[], Callable[[], pd.DataFrame]]]


@dataclasses.dataclass
class SessionOutput:
    """Everything one session writes.

    Attributes
    ----------
    sessions, truth, truth_counts, ripple_channels, units, methods, events,
    metrics, failures, warnings : pandas.DataFrame
        The session's rows of each table, with the columns in the module
        docstring.
    results : dict of (str, str) to pandas.DataFrame
        Each successful call's result, complete with its ``attrs``, by
        ``(method, setting)``.
    runtimes : dict of (str, str) to float
        Seconds each call took, failed calls included, by ``(method, setting)``.
    """

    sessions: pd.DataFrame
    truth: pd.DataFrame
    truth_counts: pd.DataFrame
    ripple_channels: pd.DataFrame
    units: pd.DataFrame
    methods: pd.DataFrame
    events: pd.DataFrame
    metrics: pd.DataFrame
    failures: pd.DataFrame
    warnings: pd.DataFrame
    results: dict[tuple[str, str], pd.DataFrame]
    runtimes: dict[tuple[str, str], float]


def default_replicates(condition_id: str) -> int:
    """A condition's replicates unless ``--replicates`` says otherwise:
    ``REFERENCE_REPLICATES`` for the reference, ``REPLICATES`` for every other."""
    return REFERENCE_REPLICATES if condition_id == "reference" else REPLICATES


def setting_label(value: Any) -> str:
    """The ``setting`` of a swept value: ``repr(float(value))``, such as "2.0"."""
    return repr(float(value))


def _detector_options(name: str, setting: str) -> dict[str, Any]:
    """The keyword arguments of detector ``name`` at ``setting``: none at its
    defaults; the swept value otherwise, Long's alias expanded."""
    if setting == "default":
        return {}
    parameter, values = THRESHOLD_SWEEPS[name]
    value = next(float(value) for value in values if setting_label(value) == setting)
    if parameter == "peak_thresholds":
        return {
            "sharp_wave_thresholds": (_LONG_BOUND, value),
            "ripple_thresholds": (_LONG_BOUND, value),
        }
    return {parameter: value}


def _detector_settings() -> list[tuple[str, str]]:
    return [
        (name, setting)
        for name in rd.DETECTORS
        for setting in ("default", *map(setting_label, THRESHOLD_SWEEPS[name][1]))
    ]


def _resolved_detector_options(name: str, setting: str) -> dict[str, Any]:
    """Every tunable of the detector at ``setting``, in JSON types; a
    non-finite value is written as its repr ("inf"), since None is itself a
    setting of these detectors."""
    options = {**rd.get_detector(name).parameters, **_detector_options(name, setting)}
    resolved: dict[str, Any] = _json_ready(options)
    return resolved


_SIGNAL_SOURCES = {
    rd.RIPPLE_BAND_LFP: "session.lfps, every channel, filtered by filter_ripple_band "
    "(150-250 Hz)",
    rd.RAW_LFP: "channel 0 of session.lfps, unfiltered",
    rd.MULTIUNIT: "session.multiunit: every unit",
}
_KEYWORD_SOURCES = {"sharp_wave_lfp": "session.sharp_wave_lfp", "theta_lfp": None}


def _detector_policy(name: str) -> dict[str, Any]:
    """Where each signal of detector ``name`` comes from; a keyword signal the
    runner does not pass (Carey's theta_lfp) is None."""
    spec = rd.get_detector(name)
    return {
        "name": "detector_signals",
        "positional": {
            parameter: _SIGNAL_SOURCES[kind]
            for parameter, kind in zip(spec.signal_parameters, spec.inputs, strict=True)
        },
        "keyword": {
            parameter: _KEYWORD_SOURCES[parameter] for parameter in spec.keyword_inputs
        },
        "speed": "session.speed",
    }


def _detector_record(name: str, setting: str) -> dict[str, str]:
    """A detector's ``methods.csv`` row, ``session_id`` aside."""
    return {
        "method": name,
        "setting": setting,
        "doi": "",
        "role": "",
        "inventory": "",
        "stage": "detection",
        "primary_expression": DETECTOR_EXPRESSION[name],
        "resolved_options": json.dumps(
            _resolved_detector_options(name, setting), sort_keys=True, allow_nan=False
        ),
        "input_policy": json.dumps(_detector_policy(name), sort_keys=True, allow_nan=False),
        "assumptions": "[]",
        "interpretation": "",
    }


@functools.cache
def _records() -> tuple[dict[str, str], ...]:
    return (
        *(_detector_record(name, setting) for name, setting in _detector_settings()),
        *(method_record(config) for config in RECIPES),
    )


def method_records(methods: Collection[tuple[str, str]] | None = None) -> list[dict[str, str]]:
    """The ``methods.csv`` row of each method and setting the runner runs.

    Parameters
    ----------
    methods : collection of (method, setting) pairs, optional
        Only these; default every one.

    Returns
    -------
    records : list of dict of str to str
        In run order: each detector at its defaults then along its sweep, in
        registry order, then each recipe in ``RECIPES`` order; the columns of
        ``methods.csv`` but ``session_id``.

    Raises
    ------
    ValueError
        A pair the runner does not run.
    """
    records = [dict(record) for record in _records()]
    if methods is None:
        return records
    known = {(record["method"], record["setting"]) for record in records}
    unknown = sorted(set(methods) - known)
    if unknown:
        msg = f"The runner runs no {unknown}; use the (method, setting) of method_records()."
        raise ValueError(msg)
    return [record for record in records if (record["method"], record["setting"]) in methods]


def _detector_call(
    detector: Callable[[], pd.DataFrame], attrs: dict[str, Any]
) -> Callable[[], Callable[[], pd.DataFrame]]:
    """``detector``'s call, its result's ``attrs`` set to ``attrs``; its
    signals are ready, so preparing it builds nothing."""

    def call() -> pd.DataFrame:
        result = detector()
        result.attrs = attrs
        return result

    return lambda: call


def _integer_counts(session: rd.SimulatedSession) -> rd.SimulatedSession:
    """``session`` with its spike counts as int16 when that holds them exactly.
    ``Recording.from_arrays`` checks integer counts at once rather than chunk by
    chunk, and makes the same float64 copy of either."""
    counts = session.multiunit.astype(np.int16)
    if not np.array_equal(counts, session.multiunit):
        return session
    # integer on purpose, for make_recording alone; detectors keep the floats
    return dataclasses.replace(session, multiunit=cast(FloatArray, counts))


def _recipe_call(
    config: RecipeConfig,
    session: rd.SimulatedSession,
    counted: Callable[[], rd.SimulatedSession],
) -> Callable[[], Callable[[], pd.DataFrame]]:
    """Preparing builds the recording and the eligible epochs; the call runs
    the method on them."""

    def prepare() -> Callable[[], pd.DataFrame]:
        recording = make_recording(counted(), config)
        intervals = behavior_intervals(session, config)
        return functools.partial(run_recipe, config, recording, intervals)

    return prepare


def method_calls(session: rd.SimulatedSession) -> list[MethodCall]:
    """Every call the runner makes on a session, not yet made.

    Parameters
    ----------
    session : SimulatedSession
        A network session, such as ``simulate_condition`` returns.

    Returns
    -------
    calls : list of (method, setting, prepare)
        In ``method_records`` order. ``prepare()`` builds the call's inputs,
        the benchmark's own code, and returns the method's call, which takes
        no arguments and returns its result: each detector at its defaults
        and at each point of its sweep, with the signals of its registry kinds
        (the ripple band from ``filter_ripple_band``; Long's
        ``sharp_wave_lfp``; Carey without ``theta_lfp``), its result's
        ``attrs`` set to its name, resolved options and the package version;
        then each recipe, whose preparation builds its recording
        (``make_recording``, with the spike counts as integers) and the
        policy's ``behavior_intervals``, and whose call runs the method on
        them.
    """
    fs = session.sampling_frequency
    filtered = rd.filter_ripple_band(session.lfps, fs)
    signals = {
        rd.RIPPLE_BAND_LFP: filtered,
        rd.RAW_LFP: session.raw_lfp,
        rd.MULTIUNIT: session.multiunit,
    }
    calls: list[MethodCall] = []
    for name, setting in _detector_settings():
        spec = rd.get_detector(name)
        keywords = (
            {"sharp_wave_lfp": session.sharp_wave_lfp} if spec.required_keyword_inputs else {}
        )
        detector = functools.partial(
            spec.detector,
            session.time,
            *(signals[kind] for kind in spec.inputs),
            session.speed,
            fs,
            **keywords,
            **_detector_options(name, setting),
        )
        attrs = {
            "method": name,
            "options": _resolved_detector_options(name, setting),
            "ripple_detection_version": rd.__version__,
        }
        calls.append((name, setting, _detector_call(detector, attrs)))

    @functools.cache
    def counted() -> rd.SimulatedSession:
        return _integer_counts(session)

    calls.extend(
        (f"recipe:{config.config_id}", "literature", _recipe_call(config, session, counted))
        for config in RECIPES
    )
    return calls


def _interval_union(bounds: np.ndarray[Any, Any]) -> float:
    """Total length of the union of ``[start, end]`` rows, shape (n, 2)."""
    union = rd.merge_close_events(bounds[np.argsort(bounds[:, 0], kind="stable")])
    return float(np.sum(union[:, 1] - union[:, 0]))


def truth_window_sets(events: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, ...]]:
    """Each expression's truth windows at each of ``TRUTH_FRACTIONS``.

    Parameters
    ----------
    events : pandas.DataFrame
        A latent event table, ``SimulatedSession.events``.

    Returns
    -------
    windows : dict of str to tuple of pandas.DataFrame
        By expression (``EXPRESSIONS``), ``truth_windows(events, fraction,
        expression)`` at each fraction in order, rows in the same order.
    """
    return {
        expression: tuple(
            rd.truth_windows(events, fraction, expression) for fraction in TRUTH_FRACTIONS
        )
        for expression in EXPRESSIONS
    }


def _median(values: pd.Series[float] | np.ndarray[Any, Any]) -> float:
    """The median, NaN when empty."""
    values = np.asarray(values, dtype=float)
    return float(np.median(values)) if len(values) else np.nan


def score_events(
    windows: Mapping[str, Sequence[pd.DataFrame | np.ndarray[Any, Any]]],
    detected: pd.DataFrame,
    minutes_outside: float,
) -> pd.DataFrame:
    """Score detected events against the truth of every expression.

    Parameters
    ----------
    windows : mapping of str to sequence of pandas.DataFrame or ndarray
        By expression, its truth windows at each of ``TRUTH_FRACTIONS``, as
        ``truth_window_sets`` gives, or their ``[start_time, end_time]`` as
        arrays of shape (n_windows, 2); matching uses the first.
    detected : pandas.DataFrame
        The events, with ``start_time`` and ``end_time``.
    minutes_outside : float
        Minutes of the session outside every network window at fraction 0.1,
        the time false positives are counted over.

    Returns
    -------
    scores : pandas.DataFrame
        One row per expression and ``minimum_iou`` in ``MATCH_IOU_LEVELS``,
        columns ``SCORE_COLUMNS``: ``match_events(windows[expression][0],
        detected, minimum_iou=level)`` summarized, with the pairs' onset and
        offset errors against the windows at each fraction from
        ``boundary_errors``, signed and absolute medians. Split and merged
        events count any overlap, so they are the same at every level.
    """
    rows = []
    detected_bounds = _bounds(detected)
    for expression, sets in windows.items():
        for level in MATCH_IOU_LEVELS:
            matching = rd.match_events(sets[0], detected_bounds, minimum_iou=level)
            pairs = matching.pairs
            row: dict[str, Any] = {
                "expression": expression,
                "minimum_iou": level,
                "n_reference": len(matching.reference),
                "n_detected": len(matching.detected),
                "n_matched": len(pairs),
                "recall": matching.recall,
                "precision": matching.precision,
                "f1": matching.f1,
                "false_positives_per_minute": len(matching.unmatched_detected)
                / minutes_outside,
                "median_iou": _median(pairs.iou),
                "median_coverage": _median(pairs.coverage),
                "median_temporal_precision": _median(pairs.temporal_precision),
            }
            for position, (percent, truth) in enumerate(zip(_PERCENTS, sets, strict=True)):
                # the pairs hold their errors against the windows they matched
                errors = pairs if position == 0 else matching.boundary_errors(truth)
                for kind in ("onset", "offset"):
                    signed = errors[f"{kind}_error"].to_numpy()
                    row[f"median_{kind}_error_{percent}"] = _median(signed)
                    row[f"median_abs_{kind}_error_{percent}"] = _median(np.abs(signed))
            row["n_split"] = len(matching.split_reference)
            row["n_merged"] = len(matching.merged_detected)
            rows.append(row)
    return pd.DataFrame(rows, columns=list(SCORE_COLUMNS))


def active_counts(
    bounds: np.ndarray[Any, Any], session: rd.SimulatedSession
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Units with a spike in each event: all, and place or pyramidal ones.

    ``count_spikes_in_events`` on the samples inside some event only, which
    are every sample it reads, so its whole-array check of the counts costs
    little per method. An event holding no sample (one lying between two
    samples, as some methods' bounds may) has no active unit.

    Parameters
    ----------
    bounds : ndarray, shape (n_events, 2)
        ``[start_time, end_time]`` of each event, closed.
    session : SimulatedSession

    Returns
    -------
    n_active_units, n_active_principal : ndarray of int, shape (n_events,)

    Raises
    ------
    ValueError
        As ``count_spikes_in_events``, such as counts that are not whole.
    """
    time = session.time
    first = np.searchsorted(time, bounds[:, 0], side="left")
    last = np.searchsorted(time, bounds[:, 1], side="right")
    holds = last > first
    step = np.zeros(len(time) + 1, dtype=np.int64)
    np.add.at(step, first[holds], 1)
    np.add.at(step, last[holds], -1)
    inside = np.cumsum(step[:-1]) > 0
    counts = np.zeros((len(bounds), session.multiunit.shape[1]), dtype=np.int64)
    if holds.any():
        counts[holds] = rd.count_spikes_in_events(
            bounds[holds], session.multiunit[inside], time[inside]
        )
    principal = np.isin(session.unit_types, _PRINCIPAL)
    return (counts > 0).sum(axis=1), (counts[:, principal] > 0).sum(axis=1)


def _bounds(frame: pd.DataFrame) -> np.ndarray[Any, Any]:
    return np.asarray(frame[["start_time", "end_time"]], dtype=float).reshape(-1, 2)


def _truth(session: rd.SimulatedSession, session_id: str) -> pd.DataFrame:
    """The session's latent event and non-event tables as ``truth.csv`` rows."""
    parts = []
    for table, label in (("events", "event"), ("non_events", "non_event")):
        frame = getattr(session, table)
        identifier, kind = _TRUTH_KEYS[table]
        parts.append(
            frame.rename(columns={identifier: "id", kind: "type"}).assign(
                session_id=session_id, table=label
            )
        )
    return pd.concat(parts, ignore_index=True).reindex(columns=list(TRUTH_COLUMNS))


def _table_frame(rows: Iterable[dict[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(columns))


def evaluate_session(
    session: rd.SimulatedSession,
    session_id: str,
    methods: Collection[tuple[str, str]] | None = None,
) -> SessionOutput:
    """Run and score every method on a simulated session.

    Parameters
    ----------
    session : SimulatedSession
        A network session with unit types, such as ``simulate_condition``
        returns.
    session_id : str
    methods : collection of (method, setting) pairs, optional
        Only these of ``method_calls``; default every one.

    Returns
    -------
    output : SessionOutput
        Its ``sessions`` row has ``session_id``, the session's times and
        counts and ``detect_s``; ``run_session`` adds the condition, replicate,
        seed and simulation time. A call's warnings, its inputs' included,
        are recorded as ``warnings`` rows and change nothing else; a method
        that raises gives a ``failures`` row and no events, results or
        metrics rows.

    Raises
    ------
    ValueError
        A (method, setting) pair the runner does not run.
    Exception
        Whatever building a call's inputs (a recipe's recording and eligible
        epochs, the integer spike counts), summarizing or scoring a result
        raises: those steps are the benchmark's own, so an error there is a
        bug to fix, never a method's failure.
    """
    started = wall_clock.perf_counter()
    fs = session.sampling_frequency
    duration = len(session.time) / fs
    # as arrays once, for every call's scores and the truth's active units
    windows = {
        expression: tuple(_bounds(frame) for frame in frames)
        for expression, frames in truth_window_sets(session.events).items()
    }
    event_time = _interval_union(windows["network"][0])
    minutes_outside = (duration - event_time) / 60
    running = np.sum(np.diff(session.running_intervals, axis=1))

    records = method_records(methods)
    selected = {(record["method"], record["setting"]) for record in records}
    calls = [call for call in method_calls(session) if methods is None or call[:2] in selected]
    events, metrics, failures, warned, results, runtimes = [], [], [], [], {}, {}
    for method, setting, prepare in calls:
        call_started = wall_clock.perf_counter()
        key = {"session_id": session_id, "method": method, "setting": setting}
        result, failed, caught = _make_call(prepare, key)
        failures += failed
        warned += caught
        if result is not None:
            events.append(_event_rows(result, key, session))
            scores = score_events(windows, result, minutes_outside)
            metrics.append(scores.assign(**key)[list(METRIC_COLUMNS)])
            results[method, setting] = result
        runtimes[method, setting] = wall_clock.perf_counter() - call_started

    truth_counts = _truth_counts(windows, session, session_id)
    event_types = session.events.drop_duplicates("event_id")["event_type"]
    non_event_types = session.non_events["non_event_type"]
    sessions = {
        "session_id": session_id,
        "duration_s": duration,
        "rest_s": duration - running,
        "event_time_s": event_time,
        **{f"n_events_{kind}": int((event_types == kind).sum()) for kind in rd.EVENT_TYPES},
        **{
            f"n_non_events_{kind}": int((non_event_types == kind).sum())
            for kind in rd.NON_EVENT_TYPES
        },
        "detect_s": wall_clock.perf_counter() - started,
    }
    ran = [call[:2] for call in calls]
    return SessionOutput(
        sessions=pd.DataFrame([sessions]),
        truth=_truth(session, session_id),
        truth_counts=truth_counts,
        ripple_channels=session.ripple_channels.assign(session_id=session_id)[
            list(RIPPLE_CHANNEL_COLUMNS)
        ],
        units=pd.DataFrame(
            {
                "session_id": session_id,
                "unit": np.arange(session.multiunit.shape[1]),
                "unit_type": session.unit_types,
                "baseline_rate": session.baseline_rates,
            },
            columns=list(UNIT_COLUMNS),
        ),
        methods=_table_frame(
            (
                {"session_id": session_id, **record}
                for record in records
                if (record["method"], record["setting"]) in ran
            ),
            METHOD_COLUMNS,
        ),
        events=_concat(events, EVENT_COLUMNS),
        metrics=_concat(metrics, METRIC_COLUMNS),
        failures=_table_frame(failures, FAILURE_COLUMNS),
        warnings=_table_frame(warned, WARNING_COLUMNS),
        results=results,
        runtimes=runtimes,
    )


def _make_call(
    prepare: Callable[[], Callable[[], pd.DataFrame]], key: Mapping[str, str]
) -> tuple[pd.DataFrame | None, list[dict[str, str]], list[dict[str, str]]]:
    """Build one call's inputs and make it, recording its warnings.

    Returns its result, None if the method raised; its ``failures`` row, if
    it did; and a ``warnings`` row per warning, each with ``key``'s
    ``session_id``, ``method`` and ``setting``. An error building the inputs
    is raised.
    """
    result, failed = None, []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # the inputs are the benchmark's own: an error building them raises
        call = prepare()
        try:
            result = call()
        except Exception as error:  # a method's failure is data, never the run's
            failed.append({**key, "error": f"{type(error).__name__}: {error}"[:_ERROR_LENGTH]})
        del call  # a recipe's recording, freed before the next one is built
    warned = [
        {
            **key,
            "category": warning.category.__name__,
            "message": str(warning.message)[:_ERROR_LENGTH],
        }
        for warning in caught
    ]
    return result, failed, warned


def _event_rows(
    result: pd.DataFrame, key: Mapping[str, str], session: rd.SimulatedSession
) -> pd.DataFrame:
    """A result's ``events.csv`` rows, with the units active in each event."""
    n_units, n_principal = active_counts(_bounds(result), session)
    peak = result.get("peak_time", pd.Series(np.nan, index=result.index))
    return pd.DataFrame(
        {
            **key,
            "event_index": np.arange(len(result)),
            "start_time": result["start_time"].to_numpy(dtype=float),
            "end_time": result["end_time"].to_numpy(dtype=float),
            "peak_time": peak.to_numpy(dtype=float),
            "n_active_units": n_units,
            "n_active_principal": n_principal,
        },
        columns=list(EVENT_COLUMNS),
    )


def _truth_counts(
    windows: Mapping[str, Sequence[np.ndarray[Any, Any]]],
    session: rd.SimulatedSession,
    session_id: str,
) -> pd.DataFrame:
    """The units active in each truth window at fraction 0.1, the
    ``truth_counts.csv`` rows of every expression."""
    frames = []
    for expression in EXPRESSIONS:
        truth = windows[expression][0]
        n_units, n_principal = active_counts(truth, session)
        frames.append(
            pd.DataFrame(
                {
                    "session_id": session_id,
                    "expression": expression,
                    "row": np.arange(len(truth)),
                    "n_active_units": n_units,
                    "n_active_principal": n_principal,
                },
                columns=list(TRUTH_COUNT_COLUMNS),
            )
        )
    return _concat(frames, TRUTH_COUNT_COLUMNS)


def _concat(frames: list[pd.DataFrame], columns: Sequence[str]) -> pd.DataFrame:
    """The rows of ``frames``, in order, with ``columns``; empty frames add
    nothing (and would make pandas warn about their dtypes)."""
    filled = [frame for frame in frames if len(frame)]
    if not filled:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(filled, ignore_index=True)[list(columns)]


def run_session(
    condition: Condition,
    replicate: int,
    methods: Collection[tuple[str, str]] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> SessionOutput:
    """Simulate replicate ``replicate`` of ``condition`` and run every method on it.

    Parameters
    ----------
    condition : Condition
    replicate : int
        From 0; the session's seed is ``session_seed(replicate)``.
    methods : collection of (method, setting) pairs, optional
        Only these; default every one.
    overrides : mapping of str to object, optional
        As in ``conditions.resolve``, such as ``{"session.duration_s": 60.0}``.

    Returns
    -------
    output : SessionOutput
        As ``evaluate_session``, the session id ``"{condition_id}/{replicate}"``,
        with the condition, replicate, seed and simulation time in its
        ``sessions`` row.
    """
    started = wall_clock.perf_counter()
    session = simulate_condition(condition, replicate, overrides)
    simulate_s = wall_clock.perf_counter() - started
    session_id = f"{condition.condition_id}/{replicate}"
    output = evaluate_session(session, session_id, methods)
    output.sessions = output.sessions.assign(
        condition_id=condition.condition_id,
        replicate=replicate,
        seed=session_seed(replicate),
        simulate_s=simulate_s,
    )[list(SESSION_COLUMNS)]
    return output


# Writing and reading


def _write_table(frame: pd.DataFrame, path: Path) -> int:
    """Write ``frame`` without its index, gzip-compressed for a ``.gz`` path
    with no timestamp, so the same rows give the same bytes."""
    compression = _GZIP if path.suffix == ".gz" else None
    frame.to_csv(path, index=False, compression=compression)
    return len(frame)


def read_table(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read a table the runner wrote, its text as text.

    Parameters
    ----------
    path : str or path-like
        A table of a condition or of ``combined/``, by its file name in
        ``TABLES``, or ``conditions.csv``.

    Returns
    -------
    table : pandas.DataFrame
        The table's text columns (``Table.text``) as ``str``, an empty field
        as ``""``; every other column as pandas infers it, an empty field as
        NaN, floats exactly as written. So ``truth.csv.gz``'s ``expression``
        and ``component``, which only event rows have, are NaN on non-event
        rows, as is every column of the other table.

    Raises
    ------
    ValueError
        The file name is not one of the runner's tables.
    """
    name = Path(path).name
    table = CONDITIONS_TABLE if name == "conditions.csv" else TABLES.get(name)
    if table is None:
        msg = f"{name} is not a table the runner writes; use conditions.csv or {list(TABLES)}."
        raise ValueError(msg)
    return pd.read_csv(
        path,
        dtype=dict.fromkeys(table.text, str),
        keep_default_na=False,
        na_values={column: [""] for column in table.columns if column not in table.text},
        float_precision="round_trip",
    )


def result_stem(method: str, setting: str) -> str:
    """``results/`` file name without its suffix: ``"{method_slug}__{setting}"``."""
    return f"{method.replace(':', '--')}__{setting}"


def _tuple_columns(result: pd.DataFrame) -> list[str]:
    """The columns of ``result`` holding a tuple in every row, such as
    Shvartsman's ``participants``, which CSV cannot hold as they are."""
    return [
        column
        for column in result.columns
        if len(result) and result[column].map(lambda value: isinstance(value, tuple)).all()
    ]


def _write_results(
    directory: Path, method: str, setting: str, results: Sequence[tuple[str, pd.DataFrame]]
) -> dict[str, int | None]:
    """One method and setting's results of every session, in ``save_events``'
    conventions; returns each file's row count."""
    stem = result_stem(method, setting)
    tuples = {session_id: _tuple_columns(result) for session_id, result in results}
    # sessions without events add no rows (and would make pandas warn)
    frames = [
        result.assign(
            session_id=session_id,
            **{
                column: result[column].map(lambda value: json.dumps(_json_ready(value)))
                for column in tuples[session_id]
            },
        )
        for session_id, result in results
        if len(result)
    ] or [results[0][1].assign(session_id=results[0][0])]
    table = pd.concat(frames)
    table = table[["session_id", *(c for c in table.columns if c != "session_id")]]
    table.to_csv(directory / f"{stem}.csv.gz", index_label=_RESULT_INDEX, compression=_GZIP)
    sidecar = {
        "saved_with_ripple_detection_version": rd.__version__,
        "sessions": {
            session_id: {
                "columns": {column: str(dtype) for column, dtype in result.dtypes.items()},
                "tuple_columns": tuples[session_id],
                "attrs": result.attrs,
            }
            for session_id, result in results
        },
    }
    (directory / f"{stem}.json").write_text(
        json.dumps(sidecar, indent=2, allow_nan=False, ensure_ascii=False)
    )
    return {f"{stem}.csv.gz": len(table), f"{stem}.json": None}


def load_results(path: str | os.PathLike[str]) -> dict[str, pd.DataFrame]:
    """Read one ``results/`` file back, session by session.

    Parameters
    ----------
    path : str or path-like
        The ``.csv.gz`` file; its ``.json`` sidecar must sit beside it.

    Returns
    -------
    results : dict of str to pandas.DataFrame
        By ``session_id``, in the order written: the result as the method
        returned it, indexed by ``event_number``, with its saved dtypes and
        ``attrs`` (JSON types).
    """
    table = Path(path)
    sidecar = json.loads(table.with_name(table.name.replace(".csv.gz", ".json")).read_text())
    # the C parser's default float conversion can be off by an ulp
    frame = pd.read_csv(table, index_col=_RESULT_INDEX, float_precision="round_trip")
    results = {}
    for session_id, saved in sidecar["sessions"].items():
        rows = frame[frame["session_id"] == session_id]
        decoded = {
            column: rows[column].map(lambda text: tuple(json.loads(text))).astype(object)
            for column in saved["tuple_columns"]
        }
        result = rows.assign(**decoded)[list(saved["columns"])].astype(saved["columns"])
        result.index = result.index.astype("int64")
        result.attrs = saved["attrs"]
        results[session_id] = result
    return results


def load_truth(path: str | os.PathLike[str]) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Read ``truth.csv.gz`` back into each session's latent tables.

    Parameters
    ----------
    path : str or path-like

    Returns
    -------
    tables : dict of str to (pandas.DataFrame, pandas.DataFrame)
        By ``session_id``, in the order written: the event table and the
        non-event table, with the simulator's columns and dtypes, so
        ``truth_windows`` gives what it gave on the session's own.
    """
    frame = pd.read_csv(path, float_precision="round_trip")
    tables = {}
    for session_id, rows in frame.groupby("session_id", sort=False):
        parts = []
        for table, label in (("events", "event"), ("non_events", "non_event")):
            identifier, kind = _TRUTH_KEYS[table]
            dtypes = _TABLE_DTYPES[table]
            part = rows[rows["table"] == label].rename(
                columns={"id": identifier, "type": kind}
            )
            parts.append(part[list(dtypes)].astype(dtypes).reset_index(drop=True))
        tables[str(session_id)] = (parts[0], parts[1])
    return tables


def write_condition(
    directory: str | os.PathLike[str], outputs: Sequence[SessionOutput]
) -> None:
    """Write a condition's sessions into ``directory``, ``done.json`` last.

    Parameters
    ----------
    directory : str or path-like
        Created; usually ``conditions/<condition_id>.partial``.
    outputs : sequence of SessionOutput
        The condition's sessions, in the order to write them.
    """
    root = Path(directory)
    (root / "results").mkdir(parents=True)
    rows: dict[str, int | None] = {}
    for name, table in TABLES.items():
        frames = [getattr(output, name.split(".")[0]) for output in outputs]
        rows[name] = _write_table(_concat(frames, table.columns), root / name)
    keys = dict.fromkeys(key for output in outputs for key in output.results)
    for method, setting in keys:
        found = [
            (output.sessions["session_id"].iloc[0], output.results[method, setting])
            for output in outputs
            if (method, setting) in output.results
        ]
        written = _write_results(root / "results", method, setting, found)
        rows.update({f"results/{name}": count for name, count in written.items()})
    files = {
        name: {"rows": rows[name], "sha256": digest}
        for name, digest in hash_files(root).items()
    }
    (root / "done.json").write_text(json.dumps({"files": files}, indent=2, sort_keys=True))


def condition_is_finished(directory: str | os.PathLike[str]) -> bool:
    """Whether ``directory`` holds a condition written in full.

    Parameters
    ----------
    directory : str or path-like

    Returns
    -------
    finished : bool
        ``done.json`` exists and lists exactly the directory's other files,
        each with its SHA-256.
    """
    root = Path(directory)
    try:
        files = json.loads((root / "done.json").read_text())["files"]
        recorded = {name: entry["sha256"] for name, entry in files.items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
    return hash_files(root, exclude={"done.json"}) == recorded


def _finish_condition(
    conditions_directory: Path, condition_id: str, outputs: Sequence[SessionOutput]
) -> None:
    """Write a condition into its ``.partial`` directory and rename it into place."""
    with replace_directory(conditions_directory / condition_id) as partial:
        write_condition(partial, outputs)


def _concat_csv(sources: Sequence[Path], target: Path) -> None:
    """Concatenate CSV files, gzip-compressed when ``target``'s suffix is
    ``.gz`` as the sources then are, byte for byte under one header."""
    gzipped = target.suffix == ".gz"
    header = None
    with target.open("wb") as raw, contextlib.ExitStack() as stack:
        out = (
            stack.enter_context(gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0))
            if gzipped
            else raw
        )
        for source in sources:
            with gzip.open(source, "rb") if gzipped else source.open("rb") as file:
                first = file.readline()
                if header is None:
                    header = first
                    out.write(first)
                elif first != header:
                    msg = f"{source} has other columns than {sources[0]}."
                    raise ValueError(msg)
                shutil.copyfileobj(file, out)


def combine(run_directory: str | os.PathLike[str]) -> Path:
    """Build ``combined/`` from the run's finished conditions.

    Parameters
    ----------
    run_directory : str or path-like
        ``examples/benchmark/output/<run_name>``.

    Returns
    -------
    combined : pathlib.Path
        Rebuilt from scratch in ``combined.partial/`` and renamed into place:
        each table the finished conditions' tables concatenated in
        condition-id order, ``results/<condition_id>/`` a copy of each one's
        ``results/``, and ``manifest.json`` naming the run's conditions
        included and those missing. A condition of ``conditions.csv`` without
        a valid ``done.json`` is left out, and printed.

    Raises
    ------
    ValueError
        The directory holds no run (no ``conditions.csv``), or no condition
        has finished.
    """
    root = Path(run_directory)
    try:
        listed = pd.read_csv(root / "conditions.csv", usecols=["condition_id"], dtype=str)
    except FileNotFoundError:
        msg = f"No run at {root}: it has no conditions.csv."
        raise ValueError(msg) from None
    finished = [
        condition_id
        for condition_id in sorted(listed.condition_id)
        if condition_is_finished(root / "conditions" / condition_id)
    ]
    missing = sorted(set(listed.condition_id) - set(finished))
    if not finished:
        msg = f"No finished condition under {root / 'conditions'}."
        raise ValueError(msg)
    if missing:
        print(
            f"combined/ leaves out {len(missing)} of the run's {len(listed)} conditions, "
            f"not finished: {', '.join(missing)}"
        )
    combined = root / "combined"
    with replace_directory(combined) as partial:
        (partial / "results").mkdir()
        for name in TABLES:
            _concat_csv([root / "conditions" / c / name for c in finished], partial / name)
        for condition_id in finished:
            shutil.copytree(
                root / "conditions" / condition_id / "results",
                partial / "results" / condition_id,
            )
        (partial / "manifest.json").write_text(
            json.dumps({"included": finished, "missing": missing}, indent=2)
        )
    return combined


# The command line


def _git_commit(directory: str | os.PathLike[str] = HERE) -> str:
    """The checkout's commit: ``HEAD``'s hash, ``"<hash>-dirty"`` when the
    working tree has changes under ``src/`` or ``examples/benchmark/``
    (untracked files included), ``"unknown"`` outside a git checkout."""

    def git(*arguments: str) -> str:
        found = subprocess.run(
            ["git", *arguments], cwd=directory, capture_output=True, text=True, check=True
        )
        return found.stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        # ":/" names a path from the top of the checkout, wherever git runs
        changed = git("status", "--porcelain", "--", ":/src", ":/examples/benchmark")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if changed else commit


def run_specification(
    resolved: Mapping[str, Mapping[str, Any]],
    replicates: Mapping[str, int],
    report: Mapping[str, Any],
    methods: Collection[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """The resolved specification a run writes to ``run_spec.json``.

    Parameters
    ----------
    resolved : mapping of str to mapping
        Each selected condition's parameters after the command line's
        overrides (``conditions.resolve``), by condition id.
    replicates : mapping of str to int
        Replicate count by condition id.
    report : mapping of str to object
        The validation report's record, as ``require_ready_report`` gives
        it; its path, hash and fingerprints are recorded.
    methods : collection of (method, setting) pairs, optional
        As in ``method_records``.

    Returns
    -------
    spec : dict
        In JSON types, as the file holds it; the keys are in the module
        docstring.
    """
    spec = {
        "conditions": dict(resolved),
        "replicates": dict(replicates),
        "seeds": {
            condition_id: [session_seed(replicate) for replicate in range(count)]
            for condition_id, count in replicates.items()
        },
        "methods": {
            f"{record['method']} {record['setting']}": record
            for record in method_records(methods)
        },
        "scoring": {
            "match_iou_levels": MATCH_IOU_LEVELS,
            "truth_fractions": TRUTH_FRACTIONS,
            "expressions": EXPRESSIONS,
        },
        "package_version": rd.__version__,
        "git_commit": _git_commit(),
        "validation_report": {key: report[key] for key in _REPORT_IDENTITY},
    }
    loaded: dict[str, Any] = json.loads(json.dumps(spec, allow_nan=False))
    return loaded


def _clear_unfinished(conditions_directory: Path) -> None:
    """Delete each interrupted (``.partial``) or unverifiable condition
    directory, printing it and why."""
    for path in sorted(conditions_directory.glob("*")):
        if not path.is_dir():
            continue
        if path.name.endswith(".partial"):
            reason = "interrupted while written"
        elif not (path / "done.json").exists():
            reason = "it has no done.json"
        elif not condition_is_finished(path):
            reason = "its files do not match done.json"
        else:
            continue
        print(f"Deleting {path}: {reason}")
        shutil.rmtree(path)


def _run_one(
    condition: Condition,
    replicate: int,
    methods: Collection[tuple[str, str]] | None,
    overrides: Mapping[str, Any],
) -> tuple[str, int, SessionOutput]:
    return (
        condition.condition_id,
        replicate,
        run_session(condition, replicate, methods, overrides),
    )


def _available_memory() -> int | None:
    """Memory available to new processes in bytes, None where unknown."""
    try:
        if platform.system() == "Darwin":
            report = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, check=True
            ).stdout
            page = int(report.split("page size of ")[1].split()[0])
            pages = {
                line.split(":")[0]: int(line.split(":")[1].strip().rstrip("."))
                for line in report.splitlines()[1:]
                if ":" in line
            }
            names = ("Pages free", "Pages inactive", "Pages speculative")
            return page * sum(pages.get(name, 0) for name in names)
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError, subprocess.CalledProcessError):
        return None
    return None


def _smoke_report(
    output: SessionOutput,
    condition_directory: Path,
    requested_workers: int,
    validation_sessions: Sequence[Mapping[str, Any]],
) -> str:
    """The smoke run's measurements, the full grid's extrapolation, the full
    validation's from the report's own sessions (``validation_sessions``, as
    ``require_ready_report`` gives them, each with ``seconds`` and
    ``peak_rss_bytes``),
    and the decision rules' verdicts, as printed lines."""
    lines = ["Per-method runtime (s):"]
    lines += [
        f"  {method} {setting}: {seconds:.3f}"
        for (method, setting), seconds in output.runtimes.items()
    ]
    session = output.sessions.iloc[0]
    session_s = float(session["simulate_s"] + session["detect_s"])
    peak = peak_rss_bytes()
    lines += [
        (
            f"Simulate: {session['simulate_s']:.1f} s; detect and score: "
            f"{session['detect_s']:.1f} s"
        ),
        f"Peak resident memory: {peak / 2**30:.2f} GiB",
        "Rows and bytes per table:",
    ]
    done = json.loads((condition_directory / "done.json").read_text())["files"]
    sizes = {name: (condition_directory / name).stat().st_size for name in done}
    lines += [
        f"  {name}: {entry['rows']} rows, {sizes[name]} bytes"
        for name, entry in done.items()
        if "/" not in name
    ]
    results_bytes = sum(size for name, size in sizes.items() if name.startswith("results/"))
    n_results = sum(1 for name in done if name.startswith("results/") and name.endswith(".gz"))
    lines.append(f"  results/: {n_results} tables and their sidecars, {results_bytes} bytes")
    n_sessions = sum(default_replicates(c.condition_id) for c in conditions())
    total_bytes = n_sessions * sum(sizes.values())
    events_bytes = n_sessions * sizes["events.csv.gz"]
    cpu_hours = n_sessions * session_s / 3600
    n_conditions = len(conditions())
    n_validation = n_conditions * DEFAULT_REPLICATES
    validation_s = float(np.mean([s["seconds"] for s in validation_sessions]))
    validation_peak = max(float(s["peak_rss_bytes"]) for s in validation_sessions)
    validation_hours = n_validation * validation_s / 3600
    both = cpu_hours + validation_hours
    lines += [
        (
            f"Full grid at this duration: {n_sessions} sessions, {cpu_hours:.1f} CPU hours, "
            f"{cpu_hours / requested_workers:.1f} h on {requested_workers} workers, "
            f"{total_bytes / 2**30:.2f} GiB written ({events_bytes / 2**30:.2f} GiB of "
            "events.csv.gz)"
        ),
        (
            f"Validation of the full grid at this duration: {n_validation} sessions "
            f"({n_conditions} conditions x {DEFAULT_REPLICATES} replicates), "
            f"{validation_s:.1f} s "
            f"and at most {validation_peak / 2**30:.2f} GiB each (the report's "
            f"{len(validation_sessions)} sessions): {validation_hours:.1f} CPU hours, "
            f"{validation_hours / requested_workers:.1f} h on {requested_workers} workers"
        ),
        (
            f"Validation and benchmark together: {both:.1f} CPU hours, "
            f"{both / requested_workers:.1f} h on {requested_workers} workers"
        ),
        "Decision rules:",
    ]
    if session_s > 300:
        lines.append(
            f"  a session takes {session_s:.0f} s > 300 s: halve duration_s and double "
            "the replicates (a new validation report first)"
        )
    else:
        lines.append(f"  a session takes {session_s:.0f} s <= 300 s: keep duration_s")
    if events_bytes > 2 * 10**9:
        lines.append("  events pass 2 GB: write sweep events only for the reference condition")
    else:
        lines.append("  events stay under 2 GB: write every condition's events")
    free_cores = (os.cpu_count() or 1) - round(os.getloadavg()[0])
    available = _available_memory()
    limits = [requested_workers, free_cores - 1]
    memory_limit = "unknown"
    if available is not None:
        limits.append(int(0.7 * available // peak))
        memory_limit = str(limits[-1])
    lines.append(
        f"  workers = min(requested {requested_workers}, free cores - 1 = {free_cores - 1}, "
        f"0.7 x available memory / peak = {memory_limit}) = {max(1, min(limits))}"
    )
    return "\n".join(lines)


def run_benchmark(
    run_name: str,
    *,
    validation_report: str | os.PathLike[str],
    condition_ids: Sequence[str] | None = None,
    replicates: int | None = None,
    duration: float | None = None,
    workers: int | None = None,
    resume: bool = False,
    smoke: bool = False,
    methods: Collection[tuple[str, str]] | None = None,
    command: str = "",
) -> Path:
    """Run the benchmark, or resume it, into ``OUTPUT / run_name``.

    Parameters
    ----------
    run_name : str
    validation_report : str or path-like
        The simulator validation report's spec.json; checked against the
        selected conditions' parameters before any method runs.
    condition_ids : sequence of str, optional
        Default every condition; run in ``conditions()`` order, each once.
    replicates : int, optional
        Per condition; default ``REFERENCE_REPLICATES`` for the reference and
        ``REPLICATES`` for every other.
    duration : float, optional
        Session length in seconds; default the conditions'.
    workers : int, optional
        Processes; default ``os.cpu_count() - 1``. With one, sessions run in
        this process.
    resume : bool, optional
        Continue the run in place: the specification must equal the saved
        one, finished conditions are kept and every other runs again.
    smoke : bool, optional
        The reference condition, one replicate, in this process; prints the
        measurements, the full grid's extrapolation at ``workers``, the full
        validation's (every condition at the validator's default replicates,
        from the runtimes the report records) and the decision rules.
    methods : collection of (method, setting) pairs, optional
        Only these; default every one.
    command : str, optional
        The command line, for ``manifest.json``.

    Returns
    -------
    run_directory : pathlib.Path

    Raises
    ------
    SystemExit
        A condition id is unknown, the report is not ready or does not cover
        the conditions, the git commit is unknown, the run exists and
        ``resume`` is not set, or, resuming, the working tree has changes
        (``"-dirty"``) or the saved specification differs from this one
        (naming the differing keys). Nothing is written or deleted before
        these checks.
    Exception
        Whatever a session raises outside a method's call, at once: with
        workers, the queued sessions are cancelled first. Finished conditions
        stay, and ``resume`` runs the rest.
    """
    requested_workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if smoke:
        condition_ids, replicates, workers = ["reference"], 1, 1
    try:
        selected = select_conditions(",".join(condition_ids) if condition_ids else "all")
    except ValueError as error:
        raise SystemExit(str(error)) from None
    counts = {
        c.condition_id: replicates or default_replicates(c.condition_id) for c in selected
    }
    overrides = {} if duration is None else {"session.duration_s": float(duration)}
    resolved = {c.condition_id: resolve(c, overrides) for c in selected}
    try:
        report = _require_report(validation_report, resolved)
    except ReportNotReady as error:  # its message names the report and why
        raise SystemExit(str(error)) from None
    spec = run_specification(resolved, counts, report, methods)
    _require_known_commit(spec["git_commit"], resume=resume)

    root = OUTPUT / run_name
    conditions_directory = root / "conditions"
    manifest = _start_or_resume(
        root, spec, selected, overrides, resume=resume, run_name=run_name, command=command
    )
    pending = [c for c in selected if not (conditions_directory / c.condition_id).exists()]
    tasks = [(c, replicate) for c in pending for replicate in range(counts[c.condition_id])]
    n_workers = min(workers or requested_workers, max(1, len(tasks)))
    (root / "manifest.json").write_text(
        json.dumps({**manifest, "n_workers": n_workers}, indent=2)
    )
    last = _execute(conditions_directory, tasks, counts, methods, overrides, n_workers)

    combine(root)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if smoke and last is not None:
        print(
            _smoke_report(
                last, conditions_directory / "reference", requested_workers, report["sessions"]
            )
        )
    return root


def _require_known_commit(commit: str, *, resume: bool) -> None:
    """Refuse an unknown commit, and a resumption from a dirty tree."""
    if commit == "unknown":
        msg = (
            "The git commit is unknown (not a git checkout, or no git): a run records "
            "the commit of the code it ran, so it cannot start or resume."
        )
        raise SystemExit(msg)
    if resume and commit.endswith("-dirty"):
        msg = (
            "Resume accepts only committed, clean code: src/ or examples/benchmark/ has "
            "changes, which the commit does not identify. Commit them and start a new "
            "run, or set them aside."
        )
        raise SystemExit(msg)


def _start_or_resume(
    root: Path,
    spec: Mapping[str, Any],
    selected: Sequence[Condition],
    overrides: Mapping[str, Any],
    *,
    resume: bool,
    run_name: str,
    command: str,
) -> dict[str, Any]:
    """Resume the run at ``root``, or start it, and return its manifest.

    Resuming, the saved specification must equal ``spec``, and interrupted
    or unverifiable conditions are deleted; starting, the run files are
    written. Raises SystemExit when there is no run to resume, the saved
    specification differs, or a run to start exists.
    """
    if resume:
        try:
            saved = json.loads((root / "run_spec.json").read_text())
        except OSError:
            msg = f"No run to resume at {root}."
            raise SystemExit(msg) from None
        differences = differing_keys(saved, spec)
        if differences:
            msg = f"The saved run specification differs at: {', '.join(differences)}."
            raise SystemExit(msg)
        _clear_unfinished(root / "conditions")
        manifest: dict[str, Any] = json.loads((root / "manifest.json").read_text())
        return manifest
    if root.exists():
        msg = f"{root} exists; pass --resume to continue it, or another --run-name."
        raise SystemExit(msg)
    (root / "conditions").mkdir(parents=True)
    found = versions()
    manifest = {
        "run_name": run_name,
        "git_commit": spec["git_commit"],
        "package_version": found["ripple_detection"],
        "numpy_version": found["numpy"],
        "scipy_version": found["scipy"],
        "command": command,
        "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "finished": None,
    }
    (root / "run_spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True))
    _write_table(
        pd.DataFrame(
            [
                {
                    "condition_id": c.condition_id,
                    "factor": c.factor,
                    "level": c.level,
                    "params": resolved_json(c, overrides),
                }
                for c in selected
            ],
            columns=list(CONDITIONS_TABLE.columns),
        ),
        root / "conditions.csv",
    )
    return manifest


def _execute(
    conditions_directory: Path,
    tasks: Sequence[tuple[Condition, int]],
    counts: Mapping[str, int],
    methods: Collection[tuple[str, str]] | None,
    overrides: Mapping[str, Any],
    n_workers: int,
) -> SessionOutput | None:
    """Run every (condition, replicate) of ``tasks``, in this process or on
    ``n_workers`` processes, writing each condition once its ``counts``
    sessions are in; return the last session finished, None if none ran.
    The first error stops the run: no queued session starts."""
    collected: dict[str, dict[int, SessionOutput]] = {c.condition_id: {} for c, _ in tasks}
    last: SessionOutput | None = None

    def finish(condition_id: str, replicate: int, output: SessionOutput) -> None:
        nonlocal last
        collected[condition_id][replicate] = output
        last = output
        if len(collected[condition_id]) == counts[condition_id]:
            outputs = collected.pop(condition_id)
            _finish_condition(
                conditions_directory, condition_id, [outputs[r] for r in sorted(outputs)]
            )

    if n_workers == 1:
        for condition, replicate in tasks:
            finish(*_run_one(condition, replicate, methods, overrides))
        return last
    pool = ProcessPoolExecutor(max_workers=n_workers)
    try:
        futures = [
            pool.submit(_run_one, condition, replicate, methods, overrides)
            for condition, replicate in tasks
        ]
        for future in as_completed(futures):
            finish(*future.result())
    except BaseException:
        # stop at the first failure: no queued session starts, and the
        # error is raised without waiting for the running ones
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()
    return last


def main(argv: Sequence[str] | None = None) -> None:
    """The command line; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--conditions",
        default="all",
        help="'all' or ids, comma-separated (a crossed cell's id holds its own comma)",
    )
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--duration", type=float, help="session length, seconds")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--validation-report")
    args = parser.parse_args(argv)
    if args.combine:
        others = [
            f"--{name.replace('_', '-')}"
            for name in ("replicates", "duration", "workers", "resume", "smoke")
            if getattr(args, name) not in (None, False)
        ]
        if others or args.conditions != "all" or args.validation_report:
            parser.error(f"--combine takes only --run-name, not {' '.join(others) or 'more'}.")
        try:
            print(combine(OUTPUT / args.run_name))
        except ValueError as error:
            raise SystemExit(str(error)) from None
        return
    if args.validation_report is None:
        parser.error("--validation-report is required for every run that simulates.")
    try:
        selected = select_conditions(args.conditions)
    except ValueError as error:
        parser.error(str(error))
    if args.smoke and (args.conditions != "all" or args.replicates is not None):
        parser.error(
            "--smoke runs the reference condition once; drop --conditions and --replicates."
        )
    run_benchmark(
        args.run_name,
        validation_report=args.validation_report,
        condition_ids=[condition.condition_id for condition in selected],
        replicates=args.replicates,
        duration=args.duration,
        workers=args.workers,
        resume=args.resume,
        smoke=args.smoke,
        command=shlex.join(
            [Path(sys.argv[0]).name, *(sys.argv[1:] if argv is None else argv)]
        ),
    )


if __name__ == "__main__":
    main()

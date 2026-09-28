"""Analyze a finished benchmark run: what each method finds and misses, what its
false positives are, how methods agree and how their boundaries differ.

Usage, from the repository root (see README.md, "Analysing a run")::

    uv run python examples/benchmark/analyze.py --run-name NAME [--workers N]
        [--run-directory PATH] [--results-directory PATH]

It reads the run's ``combined/`` and ``conditions.csv``
(``examples/benchmark/output/<run_name>/`` unless ``--run-directory`` says
otherwise; ``run.py``'s docstring lists every column) and rebuilds
``examples/benchmark/results/<run_name>/``: a CSV per analysis in ``ANALYSES``, a
PNG for each with a figure (all but ``failures``, ``operating_differences``,
``appendix_expressions``, the ``appendix_curves_<expression>`` and
``model_sensitivity_orders``; none for an empty table), ``candidate_trends.csv``
and ``summary.md``, which names each file with one sentence on what it shows and
lists the methods that failed. ``trends.md`` and ``spot_checks/``, written there by
hand, are carried over. No file may pass ``SIZE_LIMIT`` (1 MB): the command stops
before writing one, leaving the previous results as they were.

What is analysed. Most tables read the reference condition's sessions and the rows
whose ``setting`` is ``"default"`` or ``"literature"`` (``main_rows``): each detector
at its defaults and every recipe. The runner stores events, not pairs, so every
session is matched again (``match_run``, ``--workers`` processes): one to one
(``match_events``, IoU 0: any overlap) against the truth windows at 10 % of the
peak, errors also against those at 25 and 50 %. A method is headlined against its
primary expression (``methods.csv``), and in the appendix against every
expression; comparisons of all pairs of methods use the network truth, the one every
method is scored on, but for timing and error correlations of two methods sharing a
primary expression. The operating curves, points, differences, held-out thresholds
and appendix curves read the detectors' sweeps too, in the reference alone
(``load_scores``). Robustness and model sensitivity read every condition, model
sensitivity each detector's sweep in the reference and the alternative models too.

Failures. A session, method and setting the run should hold and has no scores for
is a failure, never zero events (``load_run``). Every per-method table has a row for
every method, one that never ran included, with ``n_sessions``, the sessions its
numbers pool, and ``n_failures``; a table of pairs ``n_failures_a`` and
``n_failures_b``; a table across conditions ``n_replicates``, ``n_dropped`` and
``n_failures``; the curves and points ``n_sessions`` and ``n_dropped`` (a
comparison or a curve pools only the units on which the method ran in every
condition and at every setting it compares).

Intervals and tests. Every interval is a 95 % percentile interval from
``paired_bootstrap`` (2000 resamples, seed 0): within a condition a resample draws
sessions with replacement, one draw shared by every method, so the methods stay
paired; across conditions it draws replicates, a replicate's sessions sharing its
seed in every condition. A pooled ratio or median is resampled whole; a difference
between two methods is summarized per session (the sessions are the independent
units), its estimate and interval the mean of those per-session values and its
p-value ``sign_flip_test``'s, two-sided, over them. A change between conditions is
the value pooled over the replicates minus the reference's pooled value, while its
p-value is over each replicate's own change.

Signs and units. Times and errors are seconds (the figures show milliseconds). A
signed error is detected minus truth: negative, early. A difference between two
methods is A minus B, A the method named first (by name): negative, A earlier, or
for absolute errors, A closer to the truth. A change between conditions is the other
condition minus the reference.

The tables, each built by the function of the same name, whose docstring lists its
columns, except: ``failures`` (``failure_counts``), ``paired_timing_<expression>``
(``paired_timing``, one per primary expression), ``robustness_<measure>`` and
``robustness_crossed_<measure>`` (``robustness`` and ``robustness_crossed``, their
columns after the factor and levels ``CHANGE_COLUMNS``, ``paired_changes``'), the
changes and orders of ``model_sensitivity`` (``SENSITIVITY_COLUMNS``,
``ORDER_COLUMNS``), ``operating_differences`` (``DIFFERENCE_COLUMNS``),
``appendix_curves_<expression>`` (``expression_curves``) and ``candidate_trends``
(``TREND_COLUMNS``).
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import io
import itertools
import json
import operator
import os
import shutil
import time as wall_clock
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from conditions import (
    ALTERNATIVES,
    REFERENCE_LEVEL,
    TRUTH_FRACTIONS,
    factor_levels,
    running_schedule,
    stage_seeds,
)
from conditions import conditions as benchmark_conditions
from numpy.typing import ArrayLike
from recipe_configs import RECIPES
from run import (
    _KEY,
    _PRINCIPAL,
    MATCH_IOU_LEVELS,
    OUTPUT,
    THRESHOLD_SWEEPS,
    _bounds,
    _concat,
    load_truth,
    read_table,
    setting_label,
    truth_window_sets,
)
from run import _PERCENTS as PERCENTS
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from validate_simulator import replace_directory

import ripple_detection as rd
from ripple_detection.evaluate import COMPARISON_COLUMNS
from ripple_detection.literature_methods import list_methods

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent.parent
RESULTS = HERE / "results"

# Bytes a results file may hold.
SIZE_LIMIT = 1_000_000
N_RESAMPLES = 2000
SEED = 0
LEVEL = 0.95
# Differences within this of the observed statistic count as at least as
# large, so an exact tie is not lost to rounding.
_TIE = 1e-12
# Up to this many sessions the sign-flip null is enumerated.
_EXACT_UP_TO = 16

# The settings of the main analyses: detectors at their defaults, and recipes.
MAIN_SETTINGS = ("default", "literature")
REFERENCE_CONDITION = "reference"

# The catalog's output of a method whose events are time points, and the two
# scoring rules.
POINT_OUTPUT = "ripple peaks"
PEAK_CONTAINMENT = "peak_containment"
INTERVAL = "interval"

# The label of a false positive that overlaps no truth window.
BACKGROUND = "background"
# The type that has two or three ripples, for split and merge rates.
DOUBLET = "ripple_doublet"
WINDOW_COLUMNS = ("session_id", "expression", "row", "id", "type", "start_time", "end_time")
ERROR_COLUMNS = tuple(
    f"{kind}_error_{percent}" for percent in PERCENTS for kind in ("onset", "offset")
)
PAIR_COLUMNS = (
    *_KEY,
    "expression",
    "minimum_iou",
    "truth_row",
    "event_index",
    "iou",
    "coverage",
    "temporal_precision",
    *ERROR_COLUMNS,
)
OVERLAP_COLUMNS = (
    *_KEY,
    "subset",
    "n_truth",
    "n_split",
    "n_detected",
    "n_merged",
)
FALSE_POSITIVE_COLUMNS = (
    *_KEY,
    "event_index",
    "start_time",
    "end_time",
    "label",
)
SESSION_COMPARISON_COLUMNS = ("session_id", "truth_expression", *COMPARISON_COLUMNS)
CONSENSUS_COLUMNS = ("session_id", "row", "type", "n_methods", "n_methods_run")
GROUP_COLUMNS = ("session_id", "n_methods", "n_events", "start_time", "end_time")
POINT_COLUMNS = (
    *_KEY,
    "expression",
    "n_reference",
    "n_detected",
    "n_matched",
)
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
QUANTILE_NAMES = ("q05", "q25", "median", "q75", "q95")
OVERLAP_MEASURES = ("iou", "coverage", "temporal_precision")
# The order of expressions in a table: the joint event first.
EXPRESSION_ORDER = ("network", *rd.EXPRESSIONS)
# compare_detectors' columns the pair tables average over sessions.
AGREEMENT = ("jaccard", "jaccard_true", "jaccard_false", "jaccard_truth_ids")
DIFFERENCES = (
    "median_onset_difference",
    "median_offset_difference",
    "fraction_a_earlier_onset",
    "fraction_a_earlier_offset",
)
CORRELATIONS = ("onset_error_correlation", "offset_error_correlation")
PAIRED_TIMING_COLUMNS = (
    "expression",
    "method_a",
    "method_b",
    "fraction",
    "n_shared",
    "n_sessions",
    "n_sessions_without",
    "jaccard_truth_ids",
    *(
        f"{boundary}_{measure}_{part}"
        for boundary in ("onset", "offset")
        for measure in ("signed", "absolute")
        for part in ("pooled", "estimate", "low", "high", "p", "n_dropped")
    ),
)


def paired_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], pd.Series],
    *,
    key: str,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
    level: float = LEVEL,
) -> pd.DataFrame:
    """Percentile intervals from resampling sessions or replicates.

    Each resample draws the values of ``key`` with replacement, one draw
    shared by every row with that value: ``key="session_id"`` within one
    condition, where every method's rows of a session come together, and
    ``key="replicate"`` whenever the statistic compares conditions, since a
    replicate's sessions share a seed in every condition and drawing the
    replicate keeps them together. A value drawn twice is two draws: in a
    resample, every row's ``session_id`` and ``replicate`` are text, the
    original value with ``"#<k>"`` appended, ``k`` the draw's position, so
    a statistic grouping by either keeps both copies (and must not index its
    result by them).

    Parameters
    ----------
    frame : pandas.DataFrame
        With ``session_id`` and ``replicate`` columns.
    statistic : callable
        ``statistic(frame)`` gives a Series of estimates, each resample's
        with the same index as the estimate's (NaN for an entry a resample
        cannot give).
    key : {"session_id", "replicate"}
        The column whose values are drawn.
    n_resamples : int, optional
    seed : int, optional
        Of ``numpy.random.default_rng``; the draws depend only on the seed and
        the number of values of ``key``.
    level : float, optional
        The interval's coverage, in (0, 1).

    Returns
    -------
    intervals : pandas.DataFrame
        Indexed as the statistic's Series: ``estimate``, the statistic of
        ``frame`` itself, and ``low`` and ``high``, the resamples'
        ``(1 - level) / 2`` and ``(1 + level) / 2`` quantiles (NaN draws left
        out).

    Raises
    ------
    ValueError
        ``key`` is neither ``"session_id"`` nor ``"replicate"``, or a
        resample's statistic is not indexed as the estimate is (a statistic
        indexed by the drawn sessions, say).
    """
    if key not in ("session_id", "replicate"):
        msg = f"key must be 'session_id' or 'replicate', got {key!r}."
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    values = frame[key].unique()
    positions = frame.groupby(key, sort=False).indices
    # each value's rows, and their session and replicate labels as indices
    # into the few distinct labels, so relabelling a draw costs little
    labels = {}
    for value in values:
        rows = positions[value]
        sessions, session_code = np.unique(
            frame["session_id"].astype(str).to_numpy()[rows], return_inverse=True
        )
        replicates, replicate_code = np.unique(
            frame["replicate"].astype(str).to_numpy()[rows], return_inverse=True
        )
        labels[value] = (rows, sessions, session_code, replicates, replicate_code)
    estimate = statistic(frame)
    draws = []
    for _ in range(n_resamples):
        pick = rng.choice(values, size=values.size, replace=True)
        rows, session_ids, replicate_ids = [], [], []
        for k, value in enumerate(pick):
            taken, sessions, session_code, replicates, replicate_code = labels[value]
            rows.append(taken)
            session_ids.append(
                np.array([f"{s}#{k}" for s in sessions], dtype=object)[session_code]
            )
            replicate_ids.append(
                np.array([f"{r}#{k}" for r in replicates], dtype=object)[replicate_code]
            )
        resampled = frame.iloc[np.concatenate(rows)].reset_index(drop=True)
        resampled["session_id"] = np.concatenate(session_ids)
        resampled["replicate"] = np.concatenate(replicate_ids)
        draw = statistic(resampled)
        if not draw.index.equals(estimate.index):
            msg = (
                "A resample's statistic index is not the estimate's: the statistic "
                "must not be indexed by the drawn sessions or replicates."
            )
            raise ValueError(msg)
        draws.append(draw)
    table = pd.DataFrame(draws)
    alpha = (1 - level) / 2
    return pd.DataFrame(
        {"estimate": estimate, "low": table.quantile(alpha), "high": table.quantile(1 - alpha)}
    )


def sign_flip_test(
    differences: ArrayLike, *, n_resamples: int = 10_000, seed: int = SEED
) -> float:
    """Two-sided paired test that the mean difference over units is 0.

    Parameters
    ----------
    differences : array_like, shape (n_units,)
        One paired difference per unit (a session, or a replicate across
        conditions), each finite: pair the units where both values exist
        first, and report how many were dropped.
    n_resamples : int, optional
        Random sign vectors when there are more than 16 units.
    seed : int, optional

    Returns
    -------
    p_value : float
        The fraction of sign flips whose absolute mean is at least the
        observed one: every flip, exactly, for up to 16 units; else
        ``(k + 1) / (n_resamples + 1)`` over random flips, never 0. NaN for
        no units.

    Raises
    ------
    ValueError
        A difference is not finite.
    """
    d = np.asarray(differences, dtype=float)
    if not np.isfinite(d).all():
        msg = "sign_flip_test needs finite paired differences; drop incomplete pairs first."
        raise ValueError(msg)
    if d.size == 0:
        return float("nan")
    observed = abs(d.mean())
    if d.size <= _EXACT_UP_TO:
        null = np.abs((_sign_vectors(d.size, 0, 0) * d).mean(axis=1))
        return float((null >= observed - _TIE).mean())
    null = np.abs((_sign_vectors(d.size, n_resamples, seed) * d).mean(axis=1))
    return float(((null >= observed - _TIE).sum() + 1) / (n_resamples + 1))


@functools.cache
def _sign_vectors(n_units: int, n_resamples: int, seed: int) -> np.ndarray[Any, Any]:
    """``sign_flip_test``'s sign vectors, shape (n_flips, n_units): every one
    for up to 16 units (``n_resamples`` and ``seed`` 0), else ``n_resamples``
    drawn with ``seed``. Kept for the next call of the same size, read-only."""
    if n_units <= _EXACT_UP_TO:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n_units)))
    else:
        signs = np.random.default_rng(seed).choice((-1.0, 1.0), size=(n_resamples, n_units))
    signs.flags.writeable = False
    return signs


def is_held_out(replicate: int) -> bool:
    """Whether a replicate is held out from choosing a threshold.

    Membership is by replicate id, the same in every condition: odd ids are
    held out, even ids calibrate, so replicate ``k`` never calibrates in one
    condition while being held out in another.

    Parameters
    ----------
    replicate : int

    Returns
    -------
    held_out : bool
    """
    return replicate % 2 == 1


def write_result(path: str | Path, content: bytes) -> None:
    """Write one results file, refusing one over ``SIZE_LIMIT`` bytes.

    Parameters
    ----------
    path : str or path-like
    content : bytes

    Raises
    ------
    ValueError
        ``content`` is larger than ``SIZE_LIMIT``; nothing is written.
    """
    if len(content) > SIZE_LIMIT:
        msg = (
            f"{Path(path).name} would be {len(content):,} bytes, over the "
            f"{SIZE_LIMIT:,}-byte limit of a results file."
        )
        raise ValueError(msg)
    Path(path).write_bytes(content)


# Point inventories


@functools.cache
def point_methods() -> frozenset[str]:
    """The methods whose events are time points, scored by peak containment.

    A configured literature method whose catalog entry (``list_methods()``)
    gives ``output`` ``"ripple peaks"``: every event it returns is one time
    point, which no interval rule can credit. Decided by the catalog, never by
    the events' lengths: a method whose intervals happen to be one sample
    long is still an interval method.

    Returns
    -------
    methods : frozenset of str
        ``"recipe:<config_id>"`` names.
    """
    outputs = list_methods().set_index("name")["output"]
    return frozenset(
        f"recipe:{config.config_id}"
        for config in RECIPES
        if outputs[config.method] == POINT_OUTPUT
    )


def scoring_rule(method: str) -> str:
    """How a method's events are matched to the truth.

    Parameters
    ----------
    method : str
        A registry name or ``"recipe:<config_id>"``.

    Returns
    -------
    rule : {"peak_containment", "interval"}
        ``PEAK_CONTAINMENT`` for the methods in ``point_methods``,
        ``INTERVAL`` for every other.
    """
    return PEAK_CONTAINMENT if method in point_methods() else INTERVAL


def _by_intervals(frame: pd.DataFrame, column: str = "method") -> pd.DataFrame:
    """The rows of ``frame`` whose method is scored by intervals."""
    return frame[~frame[column].isin(point_methods())]


def match_peaks(windows: ArrayLike, peaks: ArrayLike) -> np.ndarray[Any, Any]:
    """Pair truth windows one to one with the time points inside them.

    A point matches a window that contains it, bounds included, to the
    timestamps' rounding: 8 units in the last place of the largest
    magnitude, the room ``match_events`` gives a minimum IoU and ties
    between overlaps. Windows are taken in
    order of their ends, each given the earliest unused point inside it:
    for points in intervals this gives the most pairs there can be.

    Parameters
    ----------
    windows : array_like, shape (n_windows, 2)
        ``[start, end]`` of each window.
    peaks : array_like, shape (n_peaks,)

    Returns
    -------
    pairs : ndarray of int, shape (n_pairs, 2)
        The window row and the point's position of each pair, by window row.
    """
    windows = np.asarray(windows, dtype=float).reshape(-1, 2)
    peaks = np.asarray(peaks, dtype=float).reshape(-1)
    scale = max((float(np.abs(a).max()) for a in (windows, peaks) if a.size), default=0.0)
    tolerance = 8 * float(np.spacing(scale))
    order = np.argsort(peaks, kind="stable")
    ordered = peaks[order]
    # the next unused point at or after each position (a union-find, halving paths)
    following = np.arange(len(peaks) + 1)

    def unused(position: int) -> int:
        while following[position] != position:
            following[position] = following[following[position]]
            position = int(following[position])
        return position

    pairs = []
    for row in np.lexsort((windows[:, 0], windows[:, 1])):
        start, end = windows[row]
        position = unused(int(np.searchsorted(ordered, start - tolerance, side="left")))
        if position < len(peaks) and ordered[position] <= end + tolerance:
            pairs.append((int(row), int(order[position])))
            following[position] = position + 1
    return np.array(sorted(pairs), dtype=int).reshape(-1, 2)


def event_times(events: pd.DataFrame) -> np.ndarray[Any, Any]:
    """Each event's time: its ``peak_time``, else its bounds' midpoint.

    Parameters
    ----------
    events : pandas.DataFrame
        ``events.csv`` rows.

    Returns
    -------
    times : ndarray, shape (n_events,)
    """
    peak = events["peak_time"].to_numpy(dtype=float)
    middle = (
        events["start_time"].to_numpy(dtype=float) + events["end_time"].to_numpy(float)
    ) / 2
    times: np.ndarray[Any, Any] = np.where(np.isfinite(peak), peak, middle)
    return times


def _matched_rows(
    windows: np.ndarray[Any, Any], rows: pd.DataFrame, point: bool
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Each pair of a truth window, shape (n_windows, 2), and one of a
    method's events (``events.csv`` rows, in order) at IoU 0: by peak
    containment (``match_peaks``) for a ``point`` method, else one to one
    by overlap (``match_events``). The window rows and event positions."""
    if point:
        pairs = match_peaks(windows, event_times(rows))
        return pairs[:, 0], pairs[:, 1]
    found = rd.match_events(windows, _bounds(rows)).pairs
    return found["reference_index"].to_numpy(), found["detected_index"].to_numpy()


# Loading a run


def main_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows the main analyses read: detectors at their defaults, and every recipe.

    Parameters
    ----------
    frame : pandas.DataFrame
        With a ``setting`` column, such as ``methods.csv`` or ``metrics.csv.gz``.

    Returns
    -------
    rows : pandas.DataFrame
        Those whose ``setting`` is ``"default"`` or ``"literature"``
        (``MAIN_SETTINGS``); a swept value's are left out.
    """
    return frame[frame["setting"].isin(MAIN_SETTINGS)]


@dataclasses.dataclass(frozen=True)
class RunTables:
    """The tables of a run the analyses read, for some sessions and settings.

    Attributes
    ----------
    sessions : pandas.DataFrame
        The ``sessions.csv.gz`` rows read.
    methods : pandas.DataFrame
        One row per method and setting of those sessions' ``methods.csv``,
        sorted: ``method``, ``setting``, ``primary_expression``, ``scoring``
        (``scoring_rule``).
    ran : pandas.DataFrame
        One row per session, method and setting with scores (rows of
        ``metrics.csv.gz``): ``session_id``, ``method``, ``setting``.
    failures : pandas.DataFrame
        One row per session and (``methods``) method and setting without
        scores: those columns and ``error``, the runner's ``failures.csv``
        message, ``""`` where it recorded none. A missing result is a
        failure, never zero events.
    events, truth_counts : pandas.DataFrame
        Those tables' rows of the sessions, methods and settings read
        (``truth_counts`` has no method).
    truth : dict of str to (pandas.DataFrame, pandas.DataFrame)
        By ``session_id``, its latent event and non-event tables, as
        ``run.load_truth`` restores them.
    conditions : tuple of str
        The conditions of the sessions read, in the run's order.
    settings : tuple of str or None
        The settings read; None for every setting, sweeps included.
    """

    sessions: pd.DataFrame
    methods: pd.DataFrame
    ran: pd.DataFrame
    failures: pd.DataFrame
    events: pd.DataFrame
    truth_counts: pd.DataFrame
    truth: dict[str, tuple[pd.DataFrame, pd.DataFrame]]
    conditions: tuple[str, ...]
    settings: tuple[str, ...] | None

    @functools.cached_property
    def intervals(self) -> RunTables:
        """These tables of the interval methods alone: ``point_methods`` left
        out of ``methods``, ``ran``, ``failures`` and ``events``, as every
        table of bounds, overlap, timing or agreement reads them."""
        return dataclasses.replace(
            self,
            methods=_by_intervals(self.methods),
            ran=_by_intervals(self.ran),
            failures=_by_intervals(self.failures),
            events=_by_intervals(self.events),
        )


def _selection(
    frame: pd.DataFrame, sessions: Collection[str], settings: Collection[str] | None
) -> pd.Series:
    """Whether each row is of ``sessions`` and, when given, of ``settings``."""
    keep = frame["session_id"].isin(sessions)
    if settings is not None and "setting" in frame.columns:
        keep &= frame["setting"].isin(settings)
    return keep


def _methods_table(listed: pd.DataFrame) -> pd.DataFrame:
    """One row per method and setting of ``methods.csv`` rows, sorted:
    ``method``, ``setting``, ``primary_expression``, ``scoring``."""
    methods = (
        listed.drop_duplicates(["method", "setting"])[
            ["method", "setting", "primary_expression"]
        ]
        .sort_values(["method", "setting"])
        .reset_index(drop=True)
    )
    return methods.assign(scoring=methods["method"].map(scoring_rule))


def _missing(sessions: pd.DataFrame, methods: pd.DataFrame, ran: pd.DataFrame) -> pd.DataFrame:
    """Every session, method and setting without scores: ``session_id``,
    ``method``, ``setting``. Every session should hold every method and
    setting, so one missing failed, never found zero events."""
    expected = sessions[["session_id"]].merge(methods[["method", "setting"]], how="cross")
    missing = expected.merge(ran, on=list(_KEY), how="left", indicator=True)
    return missing[missing["_merge"] == "left_only"][list(_KEY)].reset_index(drop=True)


def load_run(
    directory: str | os.PathLike[str],
    *,
    conditions: Collection[str] | None = (REFERENCE_CONDITION,),
    settings: Collection[str] | None = MAIN_SETTINGS,
) -> RunTables:
    """Read a run's tables for some conditions and settings.

    Parameters
    ----------
    directory : str or path-like
        The run's ``combined/`` (or one condition's directory).
    conditions : collection of str, optional
        Condition ids to read; default the reference condition. None: all.
    settings : collection of str, optional
        Settings to read; default ``MAIN_SETTINGS``, the main analyses'.
        None: all, sweeps included.

    Returns
    -------
    tables : RunTables

    Raises
    ------
    ValueError
        A condition the directory holds no session of.
    """
    root = Path(directory)
    sessions = read_table(root / "sessions.csv.gz")
    if conditions is not None:
        unknown = sorted(set(conditions) - set(sessions["condition_id"]))
        if unknown:
            msg = f"{root} holds no session of the conditions {unknown}."
            raise ValueError(msg)
        sessions = sessions[sessions["condition_id"].isin(conditions)].reset_index(drop=True)
    ids = set(sessions["session_id"])

    def selected(rows: pd.DataFrame) -> pd.Series:
        return _selection(rows, ids, settings)

    methods = _methods_table(read_table(root / "methods.csv", keep=selected))
    ran = (
        read_table(root / "metrics.csv.gz", columns=_KEY, keep=selected)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    recorded = read_table(root / "failures.csv", keep=selected)
    failures = (
        _missing(sessions, methods, ran)
        .merge(
            recorded.drop_duplicates(list(_KEY))[[*_KEY, "error"]], on=list(_KEY), how="left"
        )
        .fillna({"error": ""})
    )
    truth = {
        session_id: tables
        for session_id, tables in load_truth(root / "truth.csv.gz").items()
        if session_id in ids
    }
    return RunTables(
        sessions=sessions,
        methods=methods,
        ran=ran,
        failures=failures,
        events=read_table(root / "events.csv.gz", keep=selected),
        truth_counts=read_table(
            root / "truth_counts.csv.gz", keep=lambda rows: _selection(rows, ids, None)
        ),
        truth=truth,
        conditions=tuple(dict.fromkeys(sessions["condition_id"])),
        settings=None if settings is None else tuple(settings),
    )


def failure_counts(tables: RunTables) -> pd.DataFrame:
    """How often each method failed on the sessions read.

    Parameters
    ----------
    tables : RunTables

    Returns
    -------
    counts : pandas.DataFrame
        One row per method and setting (``tables.methods``' order):
        ``method``, ``setting``, ``primary_expression``, ``scoring``,
        ``n_sessions`` (the sessions it has scores on, which every analysis
        pools), ``n_failures`` (those it has none on) and ``error``, the first
        failure's message.
    """
    ran = tables.ran.groupby(["method", "setting"]).size().rename("n_sessions")
    failed = tables.failures.groupby(["method", "setting"])
    counts = tables.methods[["method", "setting", "primary_expression", "scoring"]].join(
        ran, on=["method", "setting"]
    )
    counts = counts.join(failed.size().rename("n_failures"), on=["method", "setting"])
    counts = counts.join(failed["error"].first(), on=["method", "setting"])
    return counts.fillna({"n_sessions": 0, "n_failures": 0, "error": ""}).astype(
        {"n_sessions": int, "n_failures": int}
    )


# Matching every session again


@dataclasses.dataclass(frozen=True)
class Matches:
    """Each session's events matched to its truth again, as the runner did.

    Every table has ``session_id`` first. Matching is one-to-one
    (``match_events``) against the truth windows at 10 % of the peak;
    errors are against the windows at each of ``TRUTH_FRACTIONS``, detected
    minus truth, in seconds.

    Attributes
    ----------
    windows : pandas.DataFrame
        One row per truth window of each expression (``network``, ``ripple``,
        ``sharp_wave``, ``burst``) at 10 %:
        ``expression``, ``row`` (its position, as ``truth_row`` gives it),
        ``id`` (the latent event), ``type`` (its event type), ``start_time``,
        ``end_time``.
    pairs : pandas.DataFrame
        One row per matched pair of each method and setting, expression and
        ``minimum_iou`` (every expression at 0; the method's primary
        expression and the network at every level matched):
        ``truth_row``, ``event_index`` (``events.csv``'s),
        ``iou``, ``coverage``, ``temporal_precision`` and
        ``{onset,offset}_error_{10,25,50}``.
    overlaps : pandas.DataFrame
        Split and merge counts against each method's primary expression, any
        overlap counting, for ``subset`` ``"all"`` and ``"ripple_doublet"``:
        ``n_truth`` windows, ``n_split`` of them overlapped by two or more
        events; ``n_detected`` events (for the doublets, those overlapping a
        doublet's window), ``n_merged`` of them overlapping two or more
        windows.
    false_positives : pandas.DataFrame
        One row per event matching no window of its method's primary
        expression at ``minimum_iou`` 0: ``event_index``, ``start_time``,
        ``end_time`` and ``label``, the window it overlaps longest
        (``label_by_overlap``) among every event component's,
        ``"<event_type>:<expression>"``, and every non-event's, its type;
        ``"background"`` for none.
    comparisons : pandas.DataFrame
        ``compare_detectors`` on the main methods (one setting each, the
        first method of a pair first by name), ``truth_expression``
        ``"network"`` for every pair, and each primary expression but network
        for the pairs sharing it.
    consensus : pandas.DataFrame
        One row per network window: ``row``, ``type``, ``n_methods`` (main
        methods that matched it) and ``n_methods_run`` (main methods with
        scores on the session).
    false_positive_groups : pandas.DataFrame
        The main methods' false positives joined by overlap into connected
        groups, one row per group: ``n_methods`` it spans, ``n_events``,
        ``start_time``, ``end_time``.
    points : pandas.DataFrame
        One row per method and setting in ``point_methods`` and expression
        (``EXPRESSIONS``), scored by peak containment (``match_peaks``)
        against that expression's windows at 10 %: ``expression``,
        ``n_reference``, ``n_detected``, ``n_matched``. These methods are in
        no other table: an interval rule cannot credit a point.
    """

    windows: pd.DataFrame
    pairs: pd.DataFrame
    overlaps: pd.DataFrame
    false_positives: pd.DataFrame
    comparisons: pd.DataFrame
    consensus: pd.DataFrame
    false_positive_groups: pd.DataFrame
    points: pd.DataFrame


_MATCH_COLUMNS = {
    "windows": WINDOW_COLUMNS,
    "pairs": PAIR_COLUMNS,
    "overlaps": OVERLAP_COLUMNS,
    "false_positives": FALSE_POSITIVE_COLUMNS,
    "comparisons": SESSION_COMPARISON_COLUMNS,
    "consensus": CONSENSUS_COLUMNS,
    "false_positive_groups": GROUP_COLUMNS,
    "points": POINT_COLUMNS,
}


def label_windows(events: pd.DataFrame, non_events: pd.DataFrame) -> pd.DataFrame:
    """Every truth window a false positive can be labelled by, at 10 % of the peak.

    Parameters
    ----------
    events, non_events : pandas.DataFrame
        A session's latent event and non-event tables.

    Returns
    -------
    windows : pandas.DataFrame
        ``start_time``, ``end_time`` and ``label``: each event component's
        window labelled ``"<event_type>:<expression>"``, by expression
        (``ripple``, ``sharp_wave``, ``burst``), then each non-event's
        labelled by its type.
    """
    fraction = TRUTH_FRACTIONS[0]
    parts = []
    for expression in rd.EXPRESSIONS:
        windows = rd.truth_windows(events, fraction, expression)
        parts.append(windows.assign(label=windows["type"].astype(str) + f":{expression}"))
    windows = rd.truth_windows(non_events, fraction)
    parts.append(windows.assign(label=windows["type"].astype(str)))
    return _concat(parts, ("start_time", "end_time", "label"))


def _connected_groups(bounds: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Each interval's connected group under positive-length overlap, in
    order of the groups' starts; shape (n,)."""
    order = np.argsort(bounds[:, 0], kind="stable")
    start, end = bounds[order, 0], bounds[order, 1]
    new = np.ones(len(order), dtype=bool)
    new[1:] = start[1:] >= np.maximum.accumulate(end)[:-1]
    groups = np.empty(len(order), dtype=int)
    groups[order] = np.cumsum(new) - 1
    return groups


def match_session(
    session_id: str,
    events: pd.DataFrame,
    truth: tuple[pd.DataFrame, pd.DataFrame],
    ran: Sequence[tuple[str, str]],
    primary: Mapping[tuple[str, str], str],
    levels: Sequence[float] = (0.0,),
    points: Collection[str] = (),
) -> Matches:
    """Match one session's events to its truth again.

    Parameters
    ----------
    session_id : str
    events : pandas.DataFrame
        The session's ``events.csv`` rows.
    truth : (pandas.DataFrame, pandas.DataFrame)
        Its latent event and non-event tables.
    ran : sequence of (method, setting)
        The methods and settings with scores on the session, events or not.
    primary : mapping of (method, setting) to str
        Each one's primary expression.
    levels : sequence of float, optional
        The ``minimum_iou`` levels of ``pairs`` against each method's primary
        expression and the network, the ones read at levels other than 0;
        0 is always among them, and the only level of the other expressions.
    points : collection of str, optional
        Methods whose events are time points (``point_methods``): scored by
        peak containment in ``points`` and left out of every other table.

    Returns
    -------
    matches : Matches
        The session's rows.

    Raises
    ------
    ValueError
        A method has two main settings, so the main methods cannot be
        compared by name.
    """
    event_table, non_event_table = truth
    levels = tuple(dict.fromkeys((0.0, *levels)))
    sets = truth_window_sets(event_table)
    truth_bounds = {
        expression: [_bounds(frame) for frame in frames] for expression, frames in sets.items()
    }
    windows = _concat(
        [
            frames[0].assign(
                session_id=session_id, expression=expression, row=np.arange(len(frames[0]))
            )
            for expression, frames in sets.items()
        ],
        WINDOW_COLUMNS,
    )
    labels = label_windows(event_table, non_event_table)
    by_method = dict(tuple(events.groupby(["method", "setting"], sort=False)))
    pairs, overlaps, false_positives, peaks = [], [], [], []
    detected = {}
    for method, setting in ran:
        rows = by_method.get((method, setting), events.iloc[:0]).sort_values("event_index")
        bounds, index = _bounds(rows), rows["event_index"].to_numpy()
        key = {"session_id": session_id, "method": method, "setting": setting}
        if method in points:
            for expression, references in truth_bounds.items():
                peaks.append(
                    {
                        **key,
                        "expression": expression,
                        "n_reference": len(references[0]),
                        "n_detected": len(rows),
                        "n_matched": len(_matched_rows(references[0], rows, point=True)[0]),
                    }
                )
            continue
        detected[method, setting] = bounds
        expression = primary[method, setting]
        at_zero = {}
        for against, references in truth_bounds.items():
            for level in levels if against in (expression, "network") else (0.0,):
                matching = rd.match_events(references[0], bounds, minimum_iou=level)
                if level == 0:
                    at_zero[against] = matching
                found = matching.pairs
                errors = {
                    f"{kind}_error_{percent}": (
                        found
                        if position == 0
                        else matching.boundary_errors(references[position])
                    )[f"{kind}_error"].to_numpy()
                    for position, percent in enumerate(PERCENTS)
                    for kind in ("onset", "offset")
                }
                pairs.append(
                    pd.DataFrame(
                        {
                            **key,
                            "expression": against,
                            "minimum_iou": level,
                            "truth_row": found["reference_index"].to_numpy(),
                            "event_index": index[found["detected_index"].to_numpy()],
                            "iou": found["iou"].to_numpy(),
                            "coverage": found["coverage"].to_numpy(),
                            "temporal_precision": found["temporal_precision"].to_numpy(),
                            **errors,
                        },
                        columns=list(PAIR_COLUMNS),
                    )
                )
        reference = truth_bounds[expression][0]
        matching = at_zero[expression]
        unmatched = matching.unmatched_detected
        false_positives.append(
            pd.DataFrame(
                {
                    **key,
                    "event_index": index[unmatched],
                    "start_time": bounds[unmatched, 0],
                    "end_time": bounds[unmatched, 1],
                    "label": rd.label_by_overlap(bounds[unmatched], labels).to_numpy(),
                },
                columns=list(FALSE_POSITIVE_COLUMNS),
            )
        )
        doublet = (sets[expression][0]["type"] == DOUBLET).to_numpy()
        # the events overlapping a doublet's window
        near = rd.match_events(reference[doublet], bounds).detected_overlaps > 0
        overlaps.append(
            pd.DataFrame(
                [
                    {
                        **key,
                        "subset": "all",
                        "n_truth": len(reference),
                        "n_split": int((matching.reference_overlaps >= 2).sum()),
                        "n_detected": len(bounds),
                        "n_merged": int((matching.detected_overlaps >= 2).sum()),
                    },
                    {
                        **key,
                        "subset": DOUBLET,
                        "n_truth": int(doublet.sum()),
                        "n_split": int((matching.reference_overlaps[doublet] >= 2).sum()),
                        "n_detected": int(near.sum()),
                        "n_merged": int((matching.detected_overlaps[near] >= 2).sum()),
                    },
                ],
                columns=list(OVERLAP_COLUMNS),
            )
        )
    unmatched_events = _concat(false_positives, FALSE_POSITIVE_COLUMNS)
    comparisons, consensus, groups = _compare_main(
        session_id,
        detected,
        primary,
        truth_bounds,
        sets["network"][0]["type"].to_numpy(),
        unmatched_events,
    )
    return Matches(
        windows=windows,
        pairs=_concat(pairs, PAIR_COLUMNS),
        overlaps=_concat(overlaps, OVERLAP_COLUMNS),
        false_positives=unmatched_events,
        comparisons=comparisons,
        consensus=consensus,
        false_positive_groups=groups,
        points=pd.DataFrame(peaks, columns=list(POINT_COLUMNS)),
    )


def _compare_main(
    session_id: str,
    detected: Mapping[tuple[str, str], np.ndarray[Any, Any]],
    primary: Mapping[tuple[str, str], str],
    truth_bounds: Mapping[str, Sequence[np.ndarray[Any, Any]]],
    network_types: np.ndarray[Any, Any],
    false_positives: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The main methods compared with each other on one session: their
    ``compare_detectors`` rows, which of them found each network event, and
    their false positives' connected groups."""
    main = sorted(key for key in detected if key[1] in MAIN_SETTINGS)
    names = [method for method, _ in main]
    if len(set(names)) < len(names):
        msg = f"A method has two main settings on {session_id}: {main}."
        raise ValueError(msg)
    events = {method: detected[method, setting] for method, setting in main}
    expressions = {method: primary[method, setting] for method, setting in main}
    parts = [
        rd.compare_detectors(events, truth=truth_bounds["network"][0]).assign(
            truth_expression="network"
        )
    ]
    for expression in sorted(set(expressions.values()) - {"network"}):
        group = {
            method: events[method] for method in names if expressions[method] == expression
        }
        parts.append(
            rd.compare_detectors(group, truth=truth_bounds[expression][0]).assign(
                truth_expression=expression
            )
        )
    comparisons = _concat(
        [part.assign(session_id=session_id) for part in parts], SESSION_COMPARISON_COLUMNS
    )
    found = rd.consensus_counts(events, truth_bounds["network"][0])
    consensus = pd.DataFrame(
        {
            "session_id": session_id,
            "row": np.arange(len(found)),
            "type": network_types,
            "n_methods": found["n_methods"].to_numpy(),
            "n_methods_run": len(events),
        },
        columns=list(CONSENSUS_COLUMNS),
    )
    shown = false_positives[false_positives["setting"].isin(MAIN_SETTINGS)]
    labels = _connected_groups(_bounds(shown))
    groups = (
        shown.assign(group=labels)
        .groupby("group")
        .agg(
            n_methods=("method", "nunique"),
            n_events=("method", "size"),
            start_time=("start_time", "min"),
            end_time=("end_time", "max"),
        )
        .assign(session_id=session_id)
    )
    return comparisons, consensus, groups.reset_index(drop=True)[list(GROUP_COLUMNS)]


def match_run(
    tables: RunTables, *, workers: int = 1, levels: Sequence[float] = (0.0,)
) -> Matches:
    """Match every session of a run again, one process per session.

    Parameters
    ----------
    tables : RunTables
    workers : int, optional
        Processes; 1 matches in this one.
    levels : sequence of float, optional
        ``minimum_iou`` levels of the pairs; 0 is always among them.

    Returns
    -------
    matches : Matches
        Every session's rows, in ``tables.sessions``' order.
    """
    primary = {
        (method, setting): expression
        for method, setting, expression in tables.methods[
            ["method", "setting", "primary_expression"]
        ].itertuples(index=False)
    }
    events = dict(tuple(tables.events.groupby("session_id", sort=False)))
    ran = {
        session_id: list(rows[["method", "setting"]].itertuples(index=False, name=None))
        for session_id, rows in tables.ran.groupby("session_id", sort=False)
    }
    session_ids = list(tables.sessions["session_id"])
    arguments = (
        session_ids,
        [events.get(session_id, tables.events.iloc[:0]) for session_id in session_ids],
        [tables.truth[session_id] for session_id in session_ids],
        [ran.get(session_id, []) for session_id in session_ids],
    )
    points = set(tables.methods.loc[tables.methods["scoring"] == PEAK_CONTAINMENT, "method"])
    match = functools.partial(match_session, primary=primary, levels=levels, points=points)
    if workers == 1:
        sessions = list(map(match, *arguments))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            sessions = list(pool.map(match, *arguments))
    return Matches(
        **{
            name: _concat([getattr(session, name) for session in sessions], columns)
            for name, columns in _MATCH_COLUMNS.items()
        }
    )


# Intervals over groups


class GroupStatistic:
    """Statistics of every group of a frame's rows at once, and their
    resamples over units.

    Attributes
    ----------
    columns : tuple of str
        The frame's columns it reads.
    """

    columns: tuple[str, ...]

    def __call__(
        self, codes: np.ndarray[Any, Any], frame: pd.DataFrame, n_groups: int
    ) -> np.ndarray[Any, Any]:
        """The statistics of each group.

        Parameters
        ----------
        codes : ndarray of int, shape (n_rows,)
            Each row's group, from 0.
        frame : pandas.DataFrame
            The rows, with ``columns``.
        n_groups : int

        Returns
        -------
        statistics : ndarray, shape (n_statistics, n_groups)
        """
        raise NotImplementedError

    def resampled(
        self,
        codes: np.ndarray[Any, Any],
        units: np.ndarray[Any, Any],
        frame: pd.DataFrame,
        n_groups: int,
        picks: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """The statistics of each resample of units, as ``paired_bootstrap``
        computes them on the resample's rows.

        Parameters
        ----------
        codes : ndarray of int, shape (n_rows,)
        units : ndarray of int, shape (n_rows,)
            Each row's unit, from 0.
        frame : pandas.DataFrame
        n_groups : int
        picks : ndarray of int, shape (n_resamples, n_units)
            The units each resample draws, in order (``resample_picks``).

        Returns
        -------
        statistics : ndarray, shape (n_resamples, n_statistics, n_groups)
        """
        raise NotImplementedError


class _Sums(GroupStatistic):
    """Statistics of sums over each group's rows: ``terms(frame)`` gives the
    values summed, each shape (n_rows,), by name, and ``combine(sums)`` the
    statistics, shape (n_statistics, ...), from each term's sums of any shape.

    A resample's sums are added in the order its units are drawn, each unit's
    rows in the frame's order summed first: ``paired_bootstrap``'s sums to the
    bit where a unit has one row per group or the terms are whole numbers.
    """

    def __init__(
        self,
        columns: Sequence[str],
        terms: Callable[[pd.DataFrame], dict[str, np.ndarray[Any, Any]]],
        combine: Callable[[Mapping[str, np.ndarray[Any, Any]]], np.ndarray[Any, Any]],
    ) -> None:
        self.columns = tuple(columns)
        self.terms = terms
        self.combine = combine

    def __call__(
        self, codes: np.ndarray[Any, Any], frame: pd.DataFrame, n_groups: int
    ) -> np.ndarray[Any, Any]:
        return self.combine(
            {
                name: np.bincount(codes, weights=values, minlength=n_groups)
                for name, values in self.terms(frame).items()
            }
        )

    def resampled(
        self,
        codes: np.ndarray[Any, Any],
        units: np.ndarray[Any, Any],
        frame: pd.DataFrame,
        n_groups: int,
        picks: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        sums = {}
        for name, values in self.terms(frame).items():
            per_unit = np.zeros((picks.shape[1], n_groups))
            np.add.at(per_unit, (units, codes), values)
            total = np.zeros((len(picks), n_groups))
            for drawn in picks.T:
                total += per_unit[drawn]
            sums[name] = total
        return np.moveaxis(self.combine(sums), 0, 1)


class _Medians(GroupStatistic):
    """Each column's median over each group's values, NaN left out and NaN
    for a group with none."""

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = tuple(columns)

    def __call__(
        self, codes: np.ndarray[Any, Any], frame: pd.DataFrame, n_groups: int
    ) -> np.ndarray[Any, Any]:
        medians = frame[list(self.columns)].groupby(codes).median().reindex(range(n_groups))
        return np.asarray(medians, dtype=float).T

    def resampled(
        self,
        codes: np.ndarray[Any, Any],
        units: np.ndarray[Any, Any],
        frame: pd.DataFrame,
        n_groups: int,
        picks: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        weights = _pick_counts(picks)
        found = []
        for column in self.columns:
            values = frame[column].to_numpy(dtype=float)
            known = ~np.isnan(values)
            medians = WeightedMedians(values[known], codes[known], n_groups)
            drawn = units[known]
            found.append([medians(counts[drawn]) for counts in weights])
        return np.stack(found, axis=1)


def _ratio_of_sums(*ratios: tuple[str, str]) -> GroupStatistic:
    """Each (numerator, denominator) column pair's pooled ratio per group,
    NaN where the denominator sums to 0."""
    columns = tuple(dict.fromkeys(itertools.chain.from_iterable(ratios)))

    def combine(sums: Mapping[str, np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
        return np.array([_ratio(sums[top], sums[bottom]) for top, bottom in ratios])

    def terms(frame: pd.DataFrame) -> dict[str, np.ndarray[Any, Any]]:
        return {column: frame[column].to_numpy(dtype=float) for column in columns}

    return _Sums(columns, terms, combine)


def _means(*columns: str) -> GroupStatistic:
    """Each column's mean per group over its finite values, NaN for none."""

    def terms(frame: pd.DataFrame) -> dict[str, np.ndarray[Any, Any]]:
        found = {}
        for column in columns:
            values = frame[column].to_numpy(dtype=float)
            finite = np.isfinite(values)
            found[f"{column} total"] = np.where(finite, values, 0.0)
            found[f"{column} count"] = finite.astype(float)
        return found

    def combine(sums: Mapping[str, np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
        return np.array([_ratio(sums[f"{c} total"], sums[f"{c} count"]) for c in columns])

    return _Sums(columns, terms, combine)


def _medians(*columns: str) -> GroupStatistic:
    """Each column's median per group over its values, NaN for none."""
    return _Medians(columns)


def grouped_intervals(
    frame: pd.DataFrame,
    by: Sequence[str],
    statistic: GroupStatistic,
    names: Sequence[str],
    *,
    n_resamples: int = N_RESAMPLES,
) -> pd.DataFrame:
    """A statistic of each group, with its paired-bootstrap interval over sessions.

    Parameters
    ----------
    frame : pandas.DataFrame
        With ``session_id``, the ``by`` columns and the statistic's columns.
    by : sequence of str
        The columns whose values name a group.
    statistic : GroupStatistic
        Of shape ``(len(names), n_groups)``.
    names : sequence of str
        The statistics' names.
    n_resamples : int, optional

    Returns
    -------
    intervals : pandas.DataFrame
        One row per group, sorted by ``by``: the ``by`` columns, then each
        name's estimate, ``<name>_low`` and ``<name>_high`` (95 %, over
        ``paired_bootstrap``'s resamples of sessions, ``resample_picks``,
        every group of a resample from the same sessions).
    """
    grouped = frame.groupby(list(by), sort=True)
    keys = grouped.size().reset_index()[list(by)]
    n_groups = len(keys)
    for name in names:
        keys[name] = keys[f"{name}_low"] = keys[f"{name}_high"] = np.nan
    if not n_groups:
        return keys
    codes = grouped.ngroup().to_numpy()
    rows = frame[list(statistic.columns)]
    # the sessions in the order paired_bootstrap lists them
    units, sessions = pd.factorize(frame["session_id"], sort=False)
    picks = resample_picks(len(sessions), n_resamples=n_resamples)
    estimate = statistic(codes, rows, n_groups)
    resampled = statistic.resampled(codes, units, rows, n_groups, picks)
    low, high = percentile_intervals(resampled.reshape(n_resamples, -1))
    for position, name in enumerate(names):
        block = slice(position * n_groups, (position + 1) * n_groups)
        keys[name] = estimate[position]
        keys[f"{name}_low"] = low[block]
        keys[f"{name}_high"] = high[block]
    return keys


def _pooled_ratios(
    frame: pd.DataFrame,
    by: Sequence[str],
    ratios: Mapping[str, tuple[str, str]],
    counts: Sequence[str],
    sums: Sequence[str] = (),
    *,
    n_resamples: int,
) -> pd.DataFrame:
    """Each group's totals and pooled ratios, with intervals.

    Parameters
    ----------
    frame : pandas.DataFrame
        With ``session_id``, the ``by`` columns and those summed.
    by : sequence of str
    ratios : mapping of str to (str, str)
        Each ratio's name and its numerator and denominator columns.
    counts : sequence of str
        Columns summed as whole numbers.
    sums : sequence of str, optional
        Columns summed as they are.
    n_resamples : int

    Returns
    -------
    pooled : pandas.DataFrame
        One row per group, sorted by ``by``: the ``by`` columns, ``counts``
        and ``sums`` summed over the group's rows, then each ratio's
        ``<name>`` (the numerator's total over the denominator's, NaN where
        that is 0), ``<name>_low`` and ``<name>_high`` (``grouped_intervals``).
    """
    intervals = grouped_intervals(
        frame, by, _ratio_of_sums(*ratios.values()), list(ratios), n_resamples=n_resamples
    )
    totals = frame.groupby(list(by))[[*counts, *sums]].sum()
    return (
        totals.astype(dict.fromkeys(counts, int)).reset_index().merge(intervals, on=list(by))
    )


_DETECTION_COUNTS = ("n_reference", "n_detected", "n_matched")


def _detection_rates(
    frame: pd.DataFrame, by: Sequence[str], tables: RunTables, *, n_resamples: int
) -> pd.DataFrame:
    """Recall, precision and false positives per minute of each group of
    per-session counts (``_DETECTION_COUNTS``), pooled with intervals
    (``_pooled_ratios``), the counts and ``minutes`` outside every network
    window summed."""
    frame = frame.assign(
        n_unmatched=frame["n_detected"] - frame["n_matched"],
        minutes=frame["session_id"].map(_minutes_outside(tables.sessions)).to_numpy(),
    )
    ratios = {
        "recall": ("n_matched", "n_reference"),
        "precision": ("n_matched", "n_detected"),
        "false_positives_per_minute": ("n_unmatched", "minutes"),
    }
    return _pooled_ratios(
        frame, by, ratios, _DETECTION_COUNTS, ("minutes",), n_resamples=n_resamples
    )


def _method_grid(
    methods: pd.DataFrame, keys: Mapping[str, Sequence[Any]] | None = None
) -> pd.DataFrame:
    """Every method and setting of ``methods``, in order, with every
    combination of ``keys``' values, in their order."""
    grid = methods[["method", "setting"]].drop_duplicates().reset_index(drop=True)
    for column, values in (keys or {}).items():
        grid = grid.merge(pd.DataFrame({column: list(values)}), how="cross")
    return grid


def _on_grid(
    frame: pd.DataFrame, grid: pd.DataFrame, zero: Sequence[str] = ()
) -> pd.DataFrame:
    """``frame``'s columns on every row of ``grid``, in its order: a row
    ``frame`` lacks has its counts (``zero``) 0 and every other value
    missing, so a method, or a pair, that never ran stays in the table."""
    full = grid.merge(frame, on=list(grid.columns), how="left")
    full[list(zero)] = full[list(zero)].fillna(0).astype(int)
    return full[list(frame.columns)]


def _per_method(
    frame: pd.DataFrame, tables: RunTables, grid: pd.DataFrame, zero: Sequence[str] = ()
) -> pd.DataFrame:
    """A per-method table on every row of ``grid`` (``_on_grid``), with
    each method's ``primary_expression``, ``n_sessions`` and ``n_failures``
    (``failure_counts``) last."""
    counts = failure_counts(tables)[
        ["method", "setting", "primary_expression", "n_sessions", "n_failures"]
    ]
    return _on_grid(frame, grid, zero).merge(counts, on=["method", "setting"], how="left")


def _expected_pairs(tables: RunTables, expression: str | None = None) -> pd.DataFrame:
    """Every pair of main interval methods (of one primary expression when
    given), the first first by name: ``method_a``, ``method_b``."""
    main = tables.intervals.methods
    if expression is not None:
        main = main[main["primary_expression"] == expression]
    return pd.DataFrame(
        list(itertools.combinations(sorted(main["method"]), 2)),
        columns=["method_a", "method_b"],
    )


def _primary_pairs(tables: RunTables, matches: Matches) -> pd.DataFrame:
    """The pairs of each method against its primary expression, at IoU 0."""
    pairs = matches.pairs[matches.pairs["minimum_iou"] == 0]
    primary = tables.methods[["method", "setting", "primary_expression"]].rename(
        columns={"primary_expression": "expression"}
    )
    return pairs.merge(primary, on=["method", "setting", "expression"])


def recall(
    tables: RunTables,
    matches: Matches,
    scored: pd.DataFrame,
    *,
    n_resamples: int = N_RESAMPLES,
) -> pd.DataFrame:
    """Each method's pooled recall against some expressions, with intervals.

    Parameters
    ----------
    tables : RunTables
    matches : Matches
    scored : pandas.DataFrame
        ``method``, ``setting`` and ``expression``: which recalls.
    n_resamples : int, optional

    Returns
    -------
    recall : pandas.DataFrame
        One row per row of ``scored`` with scores: those columns, ``n_truth``
        and ``n_found`` (truth windows at 10 % and those matched, at IoU 0,
        over the sessions the method has scores on), ``recall`` (their ratio),
        ``recall_low`` and ``recall_high``.
    """
    n_truth = matches.windows.groupby(["session_id", "expression"]).size().rename("n_truth")
    found = matches.pairs[matches.pairs["minimum_iou"] == 0]
    n_found = found.groupby([*_KEY, "expression"]).size().rename("n_found")
    frame = tables.ran.merge(scored, on=["method", "setting"])
    frame = frame.join(n_truth, on=["session_id", "expression"]).join(
        n_found, on=[*_KEY, "expression"]
    )
    return _pooled_ratios(
        frame.fillna({"n_truth": 0, "n_found": 0}),
        ["method", "setting", "expression"],
        {"recall": ("n_found", "n_truth")},
        ["n_truth", "n_found"],
        n_resamples=n_resamples,
    )


# Every condition, against the primary expression

COUNT_COLUMNS = (
    *_KEY,
    "minimum_iou",
    "n_reference",
    "n_detected",
    "n_matched",
)
ERROR_ROW_COLUMNS = (
    *_KEY,
    "minimum_iou",
    "onset_error",
    "offset_error",
)
EXPRESSION_COUNT_COLUMNS = (*COUNT_COLUMNS[:3], "expression", *COUNT_COLUMNS[3:])
PARTICIPATION_COLUMNS = (*_KEY, "n_events", "principal_fraction")
_EVENT_READ = (
    *_KEY,
    "event_index",
    "start_time",
    "end_time",
    "peak_time",
    "n_active_principal",
)


@dataclasses.dataclass(frozen=True)
class ConditionScores:
    """Every condition's scores against each method's primary expression.

    Attributes
    ----------
    sessions : pandas.DataFrame
        One row per session: ``session_id``, ``condition_id``, ``replicate``,
        ``duration_s``, ``rest_s``, ``event_time_s`` and ``minutes`` (outside
        every network window at 10 %, the time false positives are counted
        over).
    conditions : pandas.DataFrame
        The run's ``conditions.csv``: ``condition_id``, ``factor``, ``level``.
    methods : pandas.DataFrame
        One row per method and setting: ``method``, ``setting``,
        ``primary_expression``, ``scoring``.
    counts : pandas.DataFrame
        One row per session, method, setting and ``minimum_iou`` with scores
        (``COUNT_COLUMNS``): ``n_reference``, ``n_detected``, ``n_matched``
        against the primary expression's windows at 10 %: the runner's
        ``metrics.csv`` for an interval method, peak containment
        (``match_peaks``) for a point method, whose ``minimum_iou`` is NaN
        (no IoU applies): ``methods``' ``scoring`` is the column to tell them
        apart by.
    errors : pandas.DataFrame
        One row per matched pair of an interval method against its primary
        expression (``ERROR_ROW_COLUMNS``): ``onset_error`` and
        ``offset_error`` against the windows at 10 %, detected minus truth,
        in seconds; ``session_id``, ``method`` and ``setting`` categorical.
        The main settings at ``minimum_iou`` 0 in every session; in the
        reference condition's sessions, which the operating curves read,
        every setting at every level of ``MATCH_IOU_LEVELS`` (model
        sensitivity reads the other conditions' sweeps from ``counts``).
    participation : pandas.DataFrame
        One row per session and main setting of an interval method with
        scores: ``n_events`` and ``principal_fraction``, the sum over its
        events of the fraction of place and pyramidal units active in each.
    failures : pandas.DataFrame
        One row per session, method and setting without scores:
        ``session_id``, ``method``, ``setting``.
    expression_counts : pandas.DataFrame
        ``counts`` against every expression, not only the primary
        (``EXPRESSION_COUNT_COLUMNS``): the runner's ``metrics.csv`` rows of
        the reference condition's sessions, every interval method and
        setting at every level, for the appendix's curves.
    """

    sessions: pd.DataFrame
    conditions: pd.DataFrame
    methods: pd.DataFrame
    counts: pd.DataFrame
    errors: pd.DataFrame
    participation: pd.DataFrame
    failures: pd.DataFrame
    expression_counts: pd.DataFrame = dataclasses.field(
        default_factory=lambda: pd.DataFrame(columns=list(EXPRESSION_COUNT_COLUMNS))
    )


def score_primary(
    session_id: str,
    events: pd.DataFrame,
    event_table: pd.DataFrame,
    ran: Sequence[tuple[str, str, str]],
    levels: Sequence[float],
    categories: Mapping[str, Sequence[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One session's pairs against each method's primary expression.

    Parameters
    ----------
    session_id : str
    events : pandas.DataFrame
        The session's ``events.csv`` rows.
    event_table : pandas.DataFrame
        Its latent event table.
    ran : sequence of (method, setting, primary expression)
        The methods and settings to score.
    levels : sequence of float
        The ``minimum_iou`` levels of an interval method's pairs.
    categories : mapping of str to sequence of str
        The categories of ``session_id``, ``method`` and ``setting`` in the
        errors.

    Returns
    -------
    errors : pandas.DataFrame
        ``ERROR_ROW_COLUMNS``, one row per pair of an interval method.
    points : pandas.DataFrame
        ``COUNT_COLUMNS``, one row per point method, ``minimum_iou`` NaN.
    """
    windows = {
        expression: _bounds(rd.truth_windows(event_table, TRUTH_FRACTIONS[0], expression))
        for expression in {expression for _, _, expression in ran}
    }
    by_method = dict(tuple(events.groupby(["method", "setting"], sort=False)))
    errors, points = [], []
    for method, setting, expression in ran:
        rows = by_method.get((method, setting), events.iloc[:0]).sort_values("event_index")
        reference = windows[expression]
        if method in point_methods():
            points.append(
                {
                    "session_id": session_id,
                    "method": method,
                    "setting": setting,
                    "minimum_iou": np.nan,
                    "n_reference": len(reference),
                    "n_detected": len(rows),
                    "n_matched": len(_matched_rows(reference, rows, point=True)[0]),
                }
            )
            continue
        bounds = _bounds(rows)
        for level in levels:
            pairs = rd.match_events(reference, bounds, minimum_iou=level).pairs
            errors.append(
                pd.DataFrame(
                    {
                        "method": method,
                        "setting": setting,
                        "minimum_iou": level,
                        "onset_error": pairs["onset_error"].to_numpy(dtype=float),
                        "offset_error": pairs["offset_error"].to_numpy(dtype=float),
                    }
                )
            )
    found = _concat(errors, ERROR_ROW_COLUMNS[1:]).assign(session_id=session_id)
    for column in ("session_id", "method", "setting"):
        found[column] = pd.Categorical(found[column], categories=categories[column])
    found = found.astype({"minimum_iou": float, "onset_error": float, "offset_error": float})
    return found[list(ERROR_ROW_COLUMNS)], pd.DataFrame(points, columns=list(COUNT_COLUMNS))


def load_scores(run_directory: str | os.PathLike[str], *, workers: int = 1) -> ConditionScores:
    """Read every condition's scores against the primary expressions.

    Parameters
    ----------
    run_directory : str or path-like
        ``examples/benchmark/output/<run_name>``: its ``conditions.csv`` and
        ``combined/`` are read.
    workers : int, optional
        Processes matching sessions again; 1 matches in this one.

    Returns
    -------
    scores : ConditionScores
    """
    root = Path(run_directory)
    combined = root / "combined"
    sessions = read_table(combined / "sessions.csv.gz")[
        ["session_id", "condition_id", "replicate", "duration_s", "rest_s", "event_time_s"]
    ]
    sessions = sessions.assign(minutes=_minutes_outside(sessions).to_numpy())
    conditions = read_table(root / "conditions.csv")[["condition_id", "factor", "level"]]
    methods = _methods_table(read_table(combined / "methods.csv"))
    metrics = read_table(combined / "metrics.csv.gz", columns=EXPRESSION_COUNT_COLUMNS)
    primary = methods[["method", "setting", "primary_expression"]].rename(
        columns={"primary_expression": "expression"}
    )
    reference = set(
        sessions.loc[sessions["condition_id"] == REFERENCE_CONDITION, "session_id"]
    )
    expression_counts = _by_intervals(metrics[metrics["session_id"].isin(reference)])[
        list(EXPRESSION_COUNT_COLUMNS)
    ].reset_index(drop=True)
    metrics = metrics.merge(primary, on=["method", "setting", "expression"])
    ran = metrics[list(_KEY)].drop_duplicates().reset_index(drop=True)
    failures = _missing(sessions, methods, ran)

    # the reference's sweeps, for the curves, and every condition's main settings
    events = read_table(
        combined / "events.csv.gz",
        columns=_EVENT_READ,
        keep=lambda rows: (
            rows["setting"].isin(MAIN_SETTINGS) | rows["session_id"].isin(reference)
        ),
    )
    truth = load_truth(combined / "truth.csv.gz")
    scored = ran.merge(primary, on=["method", "setting"])
    scored = scored[
        scored["setting"].isin(MAIN_SETTINGS) | scored["session_id"].isin(reference)
    ]
    to_score = {
        session_id: list(rows[["method", "setting", "expression"]].itertuples(False, None))
        for session_id, rows in scored.groupby("session_id", sort=False)
    }
    by_session = dict(tuple(events.groupby("session_id", sort=False)))
    session_ids = [s for s in sessions["session_id"] if s in to_score]
    categories = {
        "session_id": list(sessions["session_id"]),
        "method": sorted(set(methods["method"])),
        "setting": sorted(set(methods["setting"])),
    }
    arguments = (
        session_ids,
        [by_session.get(s, events.iloc[:0]) for s in session_ids],
        [truth[s][0] for s in session_ids],
        [to_score[s] for s in session_ids],
        [MATCH_IOU_LEVELS if s in reference else (0.0,) for s in session_ids],
    )
    score = functools.partial(score_primary, categories=categories)
    if workers == 1:
        found = list(map(score, *arguments))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            found = list(pool.map(score, *arguments))
    errors = _concat([session_errors for session_errors, _ in found], ERROR_ROW_COLUMNS)
    points = _concat([session_points for _, session_points in found], COUNT_COLUMNS)
    interval = _by_intervals(metrics)[list(COUNT_COLUMNS)]
    counts = _concat([interval, points], COUNT_COLUMNS).astype(
        {"n_reference": int, "n_detected": int, "n_matched": int, "minimum_iou": float}
    )
    return ConditionScores(
        sessions=sessions,
        conditions=conditions,
        methods=methods,
        counts=counts,
        errors=errors,
        participation=_participation(events, ran, read_table(combined / "units.csv.gz")),
        failures=failures,
        expression_counts=expression_counts,
    )


def _participation(
    events: pd.DataFrame, ran: pd.DataFrame, units: pd.DataFrame
) -> pd.DataFrame:
    """Each session and main setting of an interval method: its events and
    the sum of their fractions of principal units active."""
    principal = units[units["unit_type"].isin(_PRINCIPAL)].groupby("session_id").size()
    main = _by_intervals(main_rows(events))
    fraction = main["n_active_principal"].to_numpy(dtype=float) / main["session_id"].map(
        principal
    ).to_numpy(dtype=float)
    sums = (
        main[list(_KEY)]
        .assign(n_events=1, principal_fraction=fraction)
        .groupby(list(_KEY))[["n_events", "principal_fraction"]]
        .sum()
    )
    table = _by_intervals(main_rows(ran)).join(sums, on=list(_KEY))
    return (
        table.fillna({"n_events": 0, "principal_fraction": 0.0})
        .astype({"n_events": int})[list(PARTICIPATION_COLUMNS)]
        .reset_index(drop=True)
    )


# Resampling with weights: the draws of paired_bootstrap, as counts per unit


def resample_picks(
    n_units: int, *, n_resamples: int = N_RESAMPLES, seed: int = SEED
) -> np.ndarray[Any, Any]:
    """The units each resample of ``paired_bootstrap`` draws, in order.

    ``paired_bootstrap`` draws the values of its key, in the order
    ``frame[key].unique()`` lists them, with replacement; these are the
    positions it draws, so a statistic of the resamples can be computed
    without building each resample's frame.

    Parameters
    ----------
    n_units : int
    n_resamples : int, optional
    seed : int, optional

    Returns
    -------
    picks : ndarray of int, shape (n_resamples, n_units)
    """
    rng = np.random.default_rng(seed)
    picks = [rng.choice(n_units, size=n_units, replace=True) for _ in range(n_resamples)]
    return np.array(picks, dtype=int).reshape(n_resamples, n_units)


def resample_weights(
    n_units: int, *, n_resamples: int = N_RESAMPLES, seed: int = SEED
) -> np.ndarray[Any, Any]:
    """How often each unit is drawn in each resample of ``paired_bootstrap``.

    ``resample_picks`` as counts: a unit drawn ``k`` times counts ``k``
    times. A statistic that pools over units is then the same computed with
    these counts as weights.

    Parameters
    ----------
    n_units : int
    n_resamples : int, optional
    seed : int, optional

    Returns
    -------
    weights : ndarray, shape (n_resamples, n_units)
        Whole numbers; each row sums to ``n_units``.
    """
    return _pick_counts(resample_picks(n_units, n_resamples=n_resamples, seed=seed))


def _pick_counts(picks: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """How often each row of ``resample_picks``' draws each unit."""
    weights = np.zeros(picks.shape)
    np.add.at(weights, (np.arange(len(picks))[:, np.newaxis], picks), 1.0)
    return weights


class WeightedMedians:
    """The median of each group of values, each value counted a whole number
    of times.

    Parameters
    ----------
    values : array_like, shape (n_values,)
        Finite.
    groups : array_like of int, shape (n_values,)
        Each value's group, in ``[0, n_groups)``.
    n_groups : int
    """

    def __init__(self, values: ArrayLike, groups: ArrayLike, n_groups: int) -> None:
        values = np.asarray(values, dtype=float)
        groups = np.asarray(groups, dtype=int)
        self.order = np.lexsort((values, groups))
        self.values = values[self.order]
        ordered = groups[self.order]
        self.starts = np.searchsorted(ordered, np.arange(n_groups), side="left")
        self.ends = np.searchsorted(ordered, np.arange(n_groups), side="right")

    def __call__(self, counts: ArrayLike) -> np.ndarray[Any, Any]:
        """The medians, each value counted ``counts`` times.

        Parameters
        ----------
        counts : array_like, shape (n_values,)
            Whole numbers.

        Returns
        -------
        medians : ndarray, shape (n_groups,)
            The mean of the two middle values of an even count; NaN for a
            group counting nothing.
        """
        medians = np.full(len(self.starts), np.nan)
        if not len(self.values):
            return medians
        running = np.concatenate([[0.0], np.cumsum(np.asarray(counts, float)[self.order])])
        before, total = running[self.starts], running[self.ends] - running[self.starts]
        found = total > 0
        middle = []
        for position in (np.floor((total - 1) / 2), np.floor(total / 2)):
            index = np.searchsorted(running[1:], before + position, side="right")
            middle.append(self.values[np.minimum(index, len(self.values) - 1)])
        medians[found] = ((middle[0] + middle[1]) / 2)[found]
        return medians


def percentile_intervals(
    draws: ArrayLike, level: float = LEVEL
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """``paired_bootstrap``'s interval from a matrix of resampled statistics.

    Parameters
    ----------
    draws : array_like, shape (n_resamples, n_statistics)
    level : float, optional

    Returns
    -------
    low, high : ndarray, shape (n_statistics,)
        The ``(1 - level) / 2`` and ``(1 + level) / 2`` quantiles of each
        column, NaN draws left out.
    """
    alpha = (1 - level) / 2
    quantiles = pd.DataFrame(np.asarray(draws, dtype=float)).quantile([alpha, 1 - alpha])
    low, high = quantiles.to_numpy(dtype=float, copy=True)
    return low, high


# The analyses


def detection_profile(
    tables: RunTables, matches: Matches, *, n_resamples: int = N_RESAMPLES
) -> pd.DataFrame:
    """Recall per event type against the network truth, per method.

    Parameters
    ----------
    tables : RunTables
    matches : Matches
    n_resamples : int, optional

    Returns
    -------
    profile : pandas.DataFrame
        One row per method, setting and event type (``EVENT_TYPES`` order):
        ``method``, ``setting``, ``event_type``, ``n_true`` (network events
        of that type in the sessions the method has scores on), ``n_found``
        (those it matched, IoU 0), ``recall``, ``recall_low``,
        ``recall_high``, ``primary_expression``, ``n_sessions``,
        ``n_failures``.
    """
    network = matches.windows[matches.windows["expression"] == "network"]
    n_true = network.groupby(["session_id", "type"]).size().rename("n_true")
    found = matches.pairs[
        (matches.pairs["expression"] == "network") & (matches.pairs["minimum_iou"] == 0)
    ].merge(
        network[["session_id", "row", "type"]],
        left_on=["session_id", "truth_row"],
        right_on=["session_id", "row"],
    )
    n_found = found.groupby([*_KEY, "type"]).size().rename("n_found")
    frame = tables.intervals.ran.merge(pd.DataFrame({"type": rd.EVENT_TYPES}), how="cross")
    frame = frame.join(n_true, on=["session_id", "type"]).join(n_found, on=[*_KEY, "type"])
    profile = _pooled_ratios(
        frame.fillna({"n_true": 0, "n_found": 0}),
        ["method", "setting", "type"],
        {"recall": ("n_found", "n_true")},
        ["n_true", "n_found"],
        n_resamples=n_resamples,
    ).rename(columns={"type": "event_type"})
    grid = _method_grid(tables.intervals.methods, {"event_type": rd.EVENT_TYPES})
    return _per_method(profile, tables, grid, ("n_true", "n_found"))


def _false_positive_labels(matches: Matches) -> list[str]:
    """Every label a false positive can have: each event type's components
    that occur, in ``EVENT_TYPES`` and ``EXPRESSIONS`` order, each non-event
    type, and ``"background"``."""
    windows = matches.windows[matches.windows["expression"] != "network"]
    present = set(windows["type"].astype(str) + ":" + windows["expression"].astype(str))
    components = [
        f"{kind}:{expression}"
        for kind in rd.EVENT_TYPES
        for expression in rd.EXPRESSIONS
        if f"{kind}:{expression}" in present
    ]
    return [*components, *rd.NON_EVENT_TYPES, BACKGROUND]


def false_positive_classes(
    tables: RunTables, matches: Matches, *, n_resamples: int = N_RESAMPLES
) -> pd.DataFrame:
    """What each method's false positives overlap.

    A false positive is an event matching no truth window of the method's
    primary expression (IoU 0). It is labelled by the window it overlaps
    longest among every event component's (``"<event_type>:<expression>"``)
    and every non-event's (its type), ``"background"`` for none.

    Parameters
    ----------
    tables : RunTables
    matches : Matches
    n_resamples : int, optional

    Returns
    -------
    classes : pandas.DataFrame
        One row per method, setting and label, every label included:
        ``method``, ``setting``, ``label``, ``n_events`` (its false
        positives with that label), ``n_unmatched`` (all its false
        positives), ``fraction`` (their pooled ratio), ``fraction_low``,
        ``fraction_high``, ``primary_expression``, ``n_sessions``,
        ``n_failures``.
    """
    labels = _false_positive_labels(matches)
    counted = matches.false_positives.groupby([*_KEY, "label"]).size()
    frame = tables.intervals.ran.merge(pd.DataFrame({"label": labels}), how="cross")
    frame = frame.join(counted.rename("n_events"), on=[*_KEY, "label"]).fillna({"n_events": 0})
    frame["n_unmatched"] = frame.groupby(list(_KEY))["n_events"].transform("sum")
    counts = ["n_events", "n_unmatched"]
    classes = _pooled_ratios(
        frame,
        ["method", "setting", "label"],
        {"fraction": ("n_events", "n_unmatched")},
        counts,
        n_resamples=n_resamples,
    )
    grid = _method_grid(tables.intervals.methods, {"label": labels})
    return _per_method(classes, tables, grid, counts)


def consensus(tables: RunTables, matches: Matches) -> pd.DataFrame:
    """How many methods found each true event, and how many each group of
    overlapping false positives spans.

    Parameters
    ----------
    tables : RunTables
    matches : Matches

    Returns
    -------
    consensus : pandas.DataFrame
        One row per ``kind``, ``event_type`` and ``n_methods`` that occurs:
        for ``kind`` ``"true_event"``, network events of each type
        (``EVENT_TYPES`` order) that ``n_methods`` of the main methods
        matched (IoU 0); for ``"false_positive_group"`` (``event_type``
        ``"all"``), connected groups of overlapping false positives, each
        unmatched against its method's primary expression, spanning
        ``n_methods`` methods. ``count``, ``fraction`` (of the kind and
        type's), ``n_methods_compared`` (the main methods) and
        ``n_failed_calls`` (sessions and methods without scores, which
        found nothing here).
    """
    true = (
        matches.consensus.groupby(["type", "n_methods"])
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={"type": "event_type"})
        .assign(kind="true_event")
    )
    groups = (
        matches.false_positive_groups.groupby("n_methods")
        .size()
        .rename("count")
        .reset_index()
        .assign(kind="false_positive_group", event_type="all")
    )
    table = _concat([true, groups], ["kind", "event_type", "n_methods", "count"])
    table["fraction"] = table["count"] / table.groupby(["kind", "event_type"])[
        "count"
    ].transform("sum")
    rank = {kind: position for position, kind in enumerate((*rd.EVENT_TYPES, "all"))}
    table = (
        table.assign(_rank=table["event_type"].map(rank))
        .sort_values(["_rank", "n_methods"], kind="stable")
        .drop(columns="_rank")
        .reset_index(drop=True)
    )
    table["n_methods_compared"] = len(tables.intervals.methods)
    table["n_failed_calls"] = len(tables.intervals.failures)
    return table


def splits_and_merges(
    tables: RunTables, matches: Matches, *, n_resamples: int = N_RESAMPLES
) -> pd.DataFrame:
    """How often each method splits a true event or merges several.

    Parameters
    ----------
    tables : RunTables
    matches : Matches
    n_resamples : int, optional

    Returns
    -------
    rates : pandas.DataFrame
        One row per method, setting and ``subset`` (``"all"``, then
        ``"ripple_doublet"``), against the method's primary expression, any
        overlap counting: ``n_truth``, ``n_split`` (windows overlapped by two
        or more events), ``split_rate`` (pooled ratio) with ``_low`` and
        ``_high``; ``n_detected`` (for the doublets, events overlapping a
        doublet's window), ``n_merged`` (events overlapping two or more
        windows), ``merge_rate`` with ``_low`` and ``_high``;
        ``primary_expression``, ``n_sessions``, ``n_failures``.
    """
    by = ["method", "setting", "subset"]
    counts = ["n_truth", "n_split", "n_detected", "n_merged"]
    rates = _pooled_ratios(
        matches.overlaps,
        by,
        {"split_rate": ("n_split", "n_truth"), "merge_rate": ("n_merged", "n_detected")},
        counts,
        n_resamples=n_resamples,
    )
    columns = [
        *by,
        "n_truth",
        "n_split",
        "split_rate",
        "split_rate_low",
        "split_rate_high",
        "n_detected",
        "n_merged",
        "merge_rate",
        "merge_rate_low",
        "merge_rate_high",
    ]
    grid = _method_grid(tables.intervals.methods, {"subset": ("all", DOUBLET)})
    return _per_method(rates[columns], tables, grid, counts)


def _minutes_outside(sessions: pd.DataFrame) -> pd.Series:
    """Each session's minutes outside every network window at 10 %, the time
    false positives are counted over, by ``session_id``."""
    minutes = (sessions["duration_s"] - sessions["event_time_s"]).to_numpy(dtype=float) / 60
    return pd.Series(minutes, index=sessions["session_id"].to_numpy(), name="minutes")


def point_inventories(
    tables: RunTables, matches: Matches, *, n_resamples: int = N_RESAMPLES
) -> pd.DataFrame:
    """Recall, precision and false positives of the methods that return points.

    Scored by peak containment (``match_peaks``) against the primary
    expression's windows at 10 %, never pooled with an interval score: no
    IoU, coverage, boundary or timing measure exists for a point.

    Parameters
    ----------
    tables : RunTables
    matches : Matches
    n_resamples : int, optional

    Returns
    -------
    points : pandas.DataFrame
        One row per method and setting in ``point_methods``: ``method``,
        ``setting``, ``scoring`` (``"peak_containment"``), ``n_reference``,
        ``n_detected``, ``n_matched``, ``minutes`` (outside every network
        window, pooled over the sessions it has scores on), then ``recall``,
        ``precision`` and ``false_positives_per_minute``, each pooled with
        ``_low`` and ``_high``; ``primary_expression``, ``n_sessions``,
        ``n_failures``.
    """
    primary = tables.methods[["method", "setting", "primary_expression"]].rename(
        columns={"primary_expression": "expression"}
    )
    own = matches.points.merge(primary, on=["method", "setting", "expression"])
    table = _detection_rates(own, ["method", "setting"], tables, n_resamples=n_resamples)
    grid = _method_grid(tables.methods[tables.methods["scoring"] == PEAK_CONTAINMENT])
    table = _per_method(table, tables, grid, _DETECTION_COUNTS).fillna({"minutes": 0.0})
    table.insert(2, "scoring", PEAK_CONTAINMENT)
    return table


def _with_pair_failures(frame: pd.DataFrame, tables: RunTables) -> pd.DataFrame:
    """A table of method pairs with each method's ``n_failures_a`` and
    ``n_failures_b`` added (main methods, one setting each)."""
    failed = failure_counts(tables).set_index("method")["n_failures"]
    return frame.assign(
        n_failures_a=frame["method_a"].map(failed).to_numpy(),
        n_failures_b=frame["method_b"].map(failed).to_numpy(),
    )


def _session_means(
    tables: RunTables,
    comparisons: pd.DataFrame,
    columns: Sequence[str],
    *,
    n_resamples: int,
    expected: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Each pair's per-session ``compare_detectors`` values averaged over the
    sessions both methods have scores on (NaN sessions left out), with
    intervals; ``n_sessions`` counts those sessions, and
    ``<column>_n_sessions`` those with a finite value, the ones each mean
    pools (a correlation is NaN below 3 shared events). Every pair of
    ``expected`` (``method_a``, ``method_b``, ``truth_expression``) has a
    row, a pair never compared its values missing and ``n_sessions`` 0;
    default every pair of main interval methods against the network truth."""
    by = ["method_a", "method_b", "truth_expression"]
    means = grouped_intervals(
        comparisons, by, _means(*columns), list(columns), n_resamples=n_resamples
    )
    counted = [f"{column}_n_sessions" for column in columns]
    finite = pd.DataFrame(
        np.isfinite(comparisons[list(columns)].to_numpy(dtype=float)), columns=counted
    )
    n_finite = finite.groupby([comparisons[column].to_numpy() for column in by]).sum()
    n_finite.index.names = by
    means = means.join(comparisons.groupby(by).size().rename("n_sessions"), on=by).join(
        n_finite, on=by
    )
    if expected is None:
        expected = _expected_pairs(tables).assign(truth_expression="network")
    means = _on_grid(means, expected[by], ["n_sessions", *counted])
    order = [
        f"{column}{part}"
        for column in columns
        for part in ("", "_low", "_high", "_n_sessions")
    ]
    return _with_pair_failures(means[[*by, *order, "n_sessions"]], tables)


def pairwise_agreement(
    tables: RunTables, matches: Matches, *, n_resamples: int = N_RESAMPLES
) -> pd.DataFrame:
    """How much each pair of main methods agrees, against the network truth.

    Parameters
    ----------
    tables : RunTables
    matches : Matches
    n_resamples : int, optional

    Returns
    -------
    agreement : pandas.DataFrame
        One row per pair, the first method first by name:
        ``method_a``, ``method_b``, ``truth_expression`` (``"network"``,
        the truth every method is scored on), then for ``jaccard`` (their
        events matched one-to-one), ``jaccard_true`` and ``jaccard_false``
        (among their events that matched a network event, and that matched
        none) and ``jaccard_truth_ids`` (the network events both found over
        those either found): the mean over sessions of ``compare_detectors``'
        value, ``<name>_low``, ``<name>_high`` and ``<name>_n_sessions`` (the
        sessions with a finite value, which the mean pools); then
        ``n_sessions`` (both have scores), ``n_failures_a``, ``n_failures_b``.
    """
    network = matches.comparisons[matches.comparisons["truth_expression"] == "network"]
    return _session_means(tables, network, AGREEMENT, n_resamples=n_resamples)


def agreement_linkage(methods: Sequence[str], jaccard: pd.Series) -> np.ndarray[Any, Any]:
    """Average linkage of methods on ``1 - jaccard``.

    Parameters
    ----------
    methods : sequence of str
        The leaves, in order.
    jaccard : pandas.Series
        Indexed by ``(method_a, method_b)``; a pair missing, or NaN, is at
        distance 1.

    Returns
    -------
    tree : ndarray, shape (n_methods - 1, 4)
        ``scipy.cluster.hierarchy.linkage``'s; empty for fewer than two
        methods.
    """
    position = {method: index for index, method in enumerate(methods)}
    distance = np.ones((len(methods), len(methods)))
    np.fill_diagonal(distance, 0.0)
    for (a, b), value in jaccard.items():
        distance[position[a], position[b]] = distance[position[b], position[a]] = (
            1.0 - value if np.isfinite(value) else 1.0
        )
    if len(methods) < 2:
        return np.empty((0, 4))
    tree: np.ndarray[Any, Any] = linkage(squareform(distance, checks=False), method="average")
    return tree


def agreement_dendrogram(tables: RunTables, matches: Matches) -> pd.DataFrame:
    """Main methods clustered by agreement: average linkage on ``1 - jaccard``.

    ``jaccard`` is each pair's mean over sessions, as in
    ``pairwise_agreement``; a pair never compared, or never with an event,
    is at distance 1.

    Parameters
    ----------
    tables : RunTables
    matches : Matches

    Returns
    -------
    dendrogram : pandas.DataFrame
        ``scipy.cluster.hierarchy.linkage``'s tree as rows: first each
        method as a leaf (``node`` 0 .. n - 1 in name order, ``method``,
        ``left`` and ``right`` -1, ``distance`` 0, ``size`` 1), then each
        merge (``node`` n, n + 1, ..., ``method`` ``""``, the two nodes it
        joins, their distance and the leaves under it).
    """
    methods = sorted(tables.intervals.methods["method"])
    network = matches.comparisons[matches.comparisons["truth_expression"] == "network"]
    tree = agreement_linkage(
        methods, network.groupby(["method_a", "method_b"])["jaccard"].mean()
    )
    leaves = pd.DataFrame(
        {
            "node": np.arange(len(methods)),
            "method": methods,
            "left": -1,
            "right": -1,
            "distance": 0.0,
            "size": 1,
        }
    )
    if not len(tree):
        return leaves
    merges = pd.DataFrame(
        {
            "node": len(methods) + np.arange(len(tree)),
            "method": "",
            "left": tree[:, 0].astype(int),
            "right": tree[:, 1].astype(int),
            "distance": tree[:, 2],
            "size": tree[:, 3].astype(int),
        }
    )
    return pd.concat([leaves, merges], ignore_index=True)


def _sign_flips(
    frame: pd.DataFrame, by: Sequence[str], columns: Sequence[str]
) -> pd.DataFrame:
    """Each group's ``sign_flip_test`` p-value of each column over its
    sessions with a finite value, as ``<column>_p``, and the sessions left
    out as not finite, ``<column>_n_dropped``."""
    rows = []
    for key, group in frame.groupby(list(by), sort=True):
        row = dict(zip(by, key, strict=True))
        for column in columns:
            values = group[column].to_numpy(dtype=float)
            finite = np.isfinite(values)
            row[f"{column}_p"] = sign_flip_test(values[finite])
            row[f"{column}_n_dropped"] = int((~finite).sum())
        rows.append(row)
    names = [f"{column}{part}" for column in columns for part in ("_p", "_n_dropped")]
    return pd.DataFrame(rows, columns=[*by, *names])


def method_differences(
    tables: RunTables, matches: Matches, *, n_resamples: int = N_RESAMPLES
) -> pd.DataFrame:
    """How each pair of main methods' matched events differ in timing.

    Parameters
    ----------
    tables : RunTables
    matches : Matches
    n_resamples : int, optional

    Returns
    -------
    differences : pandas.DataFrame
        One row per pair, as ``pairwise_agreement``'s: for
        ``median_onset_difference`` and ``median_offset_difference`` (the
        median over their matched events of A's start, or end, minus B's, in
        seconds: negative, A earlier) and ``fraction_a_earlier_onset`` and
        ``fraction_a_earlier_offset`` (the matched events where A's is
        strictly earlier): the mean over sessions, ``<name>_low``,
        ``<name>_high`` and ``<name>_n_sessions`` (finite sessions);
        ``median_onset_difference_p`` and ``median_offset_difference_p``,
        ``sign_flip_test`` on the per-session medians, with ``_n_dropped``,
        the sessions left out of it as not finite; ``n_sessions``,
        ``n_failures_a``, ``n_failures_b``.
    """
    network = matches.comparisons[matches.comparisons["truth_expression"] == "network"]
    means = _session_means(tables, network, DIFFERENCES, n_resamples=n_resamples)
    by = ["method_a", "method_b", "truth_expression"]
    tests = _sign_flips(network, by, DIFFERENCES[:2])
    return means.merge(tests, on=by, how="left")


def error_correlations(
    tables: RunTables, matches: Matches, *, n_resamples: int = N_RESAMPLES
) -> pd.DataFrame:
    """Whether two main methods err together on the true events both found.

    Two methods sharing a primary expression are timed against that
    expression's truth, as ``paired_timing`` times them; any other pair
    against the network truth, the one every method is scored on. The
    network truth's correlations are beside every pair's as well.

    Parameters
    ----------
    tables : RunTables
    matches : Matches
    n_resamples : int, optional

    Returns
    -------
    correlations : pandas.DataFrame
        One row per pair, the first method first by name: ``method_a``,
        ``method_b``, ``truth_expression`` (the pair's shared primary
        expression, else ``"network"``); for ``onset_error_correlation`` and
        ``offset_error_correlation`` (Spearman's correlation of their signed
        errors against that truth's windows over the events both found, NaN
        below 3) the mean over sessions with one, ``<name>_low``,
        ``<name>_high`` and ``<name>_n_sessions`` (the sessions with one);
        the same against the network truth, each column prefixed
        ``network_``; ``n_sessions`` (both have scores), ``n_failures_a``,
        ``n_failures_b``.
    """
    by = ["method_a", "method_b", "truth_expression"]
    primary = tables.intervals.methods.set_index("method")["primary_expression"]
    expected = _expected_pairs(tables)
    first = expected["method_a"].map(primary).to_numpy()
    shared = first == expected["method_b"].map(primary).to_numpy()
    expected["truth_expression"] = np.where(shared, first, "network")
    chosen = matches.comparisons.merge(expected, on=by)
    own = _session_means(
        tables, chosen, CORRELATIONS, n_resamples=n_resamples, expected=expected
    )
    network = matches.comparisons[matches.comparisons["truth_expression"] == "network"]
    against = _session_means(tables, network, CORRELATIONS, n_resamples=n_resamples)
    columns = [c for c in against.columns if c.startswith(CORRELATIONS)]
    beside = against[["method_a", "method_b", *columns]].rename(
        columns={column: f"network_{column}" for column in columns}
    )
    last = ["n_sessions", "n_failures_a", "n_failures_b"]
    kept = [column for column in own.columns if column not in last]
    return own[kept].merge(beside, on=["method_a", "method_b"], how="left").join(own[last])


def _quantiles(frame: pd.DataFrame, by: Sequence[str], columns: Sequence[str]) -> pd.DataFrame:
    """Each column's pooled 5, 25, 50, 75 and 95 % quantiles and count per
    group, one row per group and column (``measure``)."""
    parts = []
    for column in columns:
        grouped = frame.groupby(list(by))[column]
        found = pd.DataFrame(
            {
                name: grouped.quantile(q)
                for q, name in zip(QUANTILES, QUANTILE_NAMES, strict=True)
            }
        )
        parts.append(found.assign(measure=column, n_pairs=grouped.size()).reset_index())
    return _concat(parts, [*by, "measure", "n_pairs", *QUANTILE_NAMES])


def overlap_quality(
    tables: RunTables, matches: Matches, *, n_resamples: int = N_RESAMPLES
) -> pd.DataFrame:
    """How much each method's matched events overlap their truth.

    Parameters
    ----------
    tables : RunTables
    matches : Matches
    n_resamples : int, optional

    Returns
    -------
    quality : pandas.DataFrame
        One row per method, setting and ``measure`` (``iou``, ``coverage``:
        the fraction of the true event found, ``temporal_precision``: the
        fraction of the event that is true), over the pairs matched against
        the method's primary expression (IoU 0), pooled over sessions:
        ``n_pairs``, ``q05``, ``q25``, ``median``, ``q75``, ``q95``,
        ``median_low`` and ``median_high`` (the pooled median's interval),
        the method's ``recall`` against that expression with ``recall_low``
        and ``recall_high``, ``primary_expression``, ``n_sessions``,
        ``n_failures``.
    """
    pairs = _primary_pairs(tables, matches)
    by = ["method", "setting"]
    quality = _quantiles(pairs, by, OVERLAP_MEASURES)
    medians = grouped_intervals(
        pairs,
        by,
        _medians(*OVERLAP_MEASURES),
        OVERLAP_MEASURES,
        n_resamples=n_resamples,
    )
    quality = quality.merge(
        _long_intervals(medians, by, OVERLAP_MEASURES), on=[*by, "measure"]
    )
    primary = tables.intervals.methods[["method", "setting", "primary_expression"]].rename(
        columns={"primary_expression": "expression"}
    )
    found = recall(tables, matches, primary, n_resamples=n_resamples)
    quality = quality.merge(
        found[[*by, "recall", "recall_low", "recall_high"]], on=by, how="left"
    )
    grid = _method_grid(tables.intervals.methods, {"measure": OVERLAP_MEASURES})
    return _per_method(quality, tables, grid, ("n_pairs",))


def _long_intervals(
    wide: pd.DataFrame, by: Sequence[str], names: Sequence[str]
) -> pd.DataFrame:
    """``grouped_intervals``' medians, one row per group and name
    (``measure``): ``median_low`` and ``median_high``."""
    return _concat(
        [
            wide[[*by, f"{name}_low", f"{name}_high"]]
            .rename(columns={f"{name}_low": "median_low", f"{name}_high": "median_high"})
            .assign(measure=name)
            for name in names
        ],
        [*by, "measure", "median_low", "median_high"],
    )


def _errors(pairs: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Each pair's signed and absolute onset and offset errors at every
    truth fraction, as ``<boundary>_<measure>_<percent>`` columns."""
    columns = {}
    for percent in PERCENTS:
        for boundary in ("onset", "offset"):
            signed = pairs[f"{boundary}_error_{percent}"].to_numpy(dtype=float)
            columns[f"{boundary}_signed_{percent}"] = signed
            columns[f"{boundary}_absolute_{percent}"] = np.abs(signed)
    return pairs.assign(**columns), list(columns)


def _split_error_names(frame: pd.DataFrame) -> pd.DataFrame:
    """``measure`` ``<boundary>_<measure>_<percent>`` as ``fraction``,
    ``boundary`` and ``measure`` columns."""
    parts = frame["measure"].str.split("_")
    return frame.assign(
        fraction=parts.str[2].astype(int) / 100, boundary=parts.str[0], measure=parts.str[1]
    )


def boundary_errors(
    tables: RunTables, matches: Matches, *, n_resamples: int = N_RESAMPLES
) -> pd.DataFrame:
    """Each method's onset and offset errors against the truth.

    Against the method's primary expression, and for a method whose primary
    expression is the network event (ripple and burst joined) against the
    ripple and the burst windows as well. Pairs are matched at 10 % of the
    peak (IoU 0) and their errors measured against the windows at 10, 25
    and 50 %.

    Parameters
    ----------
    tables : RunTables
    matches : Matches
    n_resamples : int, optional

    Returns
    -------
    errors : pandas.DataFrame
        One row per method, setting, ``expression``, ``fraction`` (0.1,
        0.25, 0.5), ``boundary`` (``onset``, ``offset``) and ``measure``
        (``signed``: detected minus truth, negative early; ``absolute``), in
        seconds, pooled over sessions: ``n_pairs``, ``q05``, ``q25``,
        ``median``, ``q75``, ``q95``, ``iqr``, ``median_low`` and
        ``median_high``, and beside every median the method's ``recall``
        against the expression, with ``recall_low`` and ``recall_high``, since
        a method that finds only easy events can time them better;
        ``primary_expression``, ``n_sessions``, ``n_failures``.
    """
    # each method against its primary expression, the joint event's against its
    # ripple and burst too, in EXPRESSION_ORDER
    listed = tables.intervals.methods[["method", "setting", "primary_expression"]]
    scored = pd.DataFrame(
        [
            (method, setting, expression)
            for method, setting, primary in listed.itertuples(index=False)
            for expression in (
                (primary, "ripple", "burst") if primary == "network" else (primary,)
            )
        ],
        columns=["method", "setting", "expression"],
    )
    pairs = matches.pairs[matches.pairs["minimum_iou"] == 0].merge(
        scored, on=["method", "setting", "expression"]
    )
    pairs, names = _errors(pairs)
    by = ["method", "setting", "expression"]
    errors = _quantiles(pairs, by, names)
    medians = grouped_intervals(pairs, by, _medians(*names), names, n_resamples=n_resamples)
    errors = errors.merge(_long_intervals(medians, by, names), on=[*by, "measure"])
    errors["iqr"] = errors["q75"] - errors["q25"]
    found = recall(tables, matches, scored, n_resamples=n_resamples)
    errors = errors.merge(
        found[[*by, "recall", "recall_low", "recall_high"]], on=by, how="left"
    )
    columns = [
        *by,
        "fraction",
        "boundary",
        "measure",
        "n_pairs",
        "recall",
        "recall_low",
        "recall_high",
        *QUANTILE_NAMES,
        "iqr",
        "median_low",
        "median_high",
    ]
    # every method scored, each expression and measure, whether it ran or not
    measures = _split_error_names(pd.DataFrame({"measure": names}))
    grid = scored.merge(measures[["fraction", "boundary", "measure"]], how="cross")
    return _per_method(_split_error_names(errors)[columns], tables, grid, ("n_pairs",))


def paired_timing(
    tables: RunTables,
    matches: Matches,
    expression: str,
    *,
    n_resamples: int = N_RESAMPLES,
) -> pd.DataFrame:
    """Two methods' errors on the true events both found, paired.

    For each pair of main methods whose primary expression is
    ``expression``, on the truth windows of that expression both matched
    (IoU 0) in a session: A's error minus B's, signed (``signed``) and in
    absolute value (``absolute``: negative, A closer to the truth), at each
    truth fraction.

    Parameters
    ----------
    tables : RunTables
    matches : Matches
    expression : str
        A primary expression.
    n_resamples : int, optional

    Returns
    -------
    timing : pandas.DataFrame
        One row per pair (A first by name) and ``fraction``: ``expression``,
        ``method_a``, ``method_b``, ``fraction``, ``n_shared`` (the true
        events both found, over the sessions), ``n_sessions`` (sessions with
        one), ``n_sessions_without`` (sessions both have scores on without
        one: left out), ``jaccard_truth_ids`` (mean over sessions: the true
        events both found over those either found); then for each
        ``<boundary>_<measure>`` (``onset_signed``, ``onset_absolute``,
        ``offset_signed``, ``offset_absolute``), in seconds: ``_pooled``,
        the median difference over the shared events; ``_estimate``, the
        mean over sessions of each session's median difference, with
        ``_low`` and ``_high``; ``_p``, ``sign_flip_test`` on those
        per-session medians, and ``_n_dropped``, the sessions left out of it
        as not finite; then ``n_failures_a``, ``n_failures_b``.
    """
    members = tables.intervals.methods
    members = members[members["primary_expression"] == expression]["method"]
    comparisons = matches.comparisons[
        (matches.comparisons["truth_expression"] == expression)
        & matches.comparisons["method_a"].isin(members)
        & matches.comparisons["method_b"].isin(members)
    ]
    pairs = matches.pairs[
        (matches.pairs["minimum_iou"] == 0)
        & (matches.pairs["expression"] == expression)
        & matches.pairs["method"].isin(members)
    ][["session_id", "method", "truth_row", *ERROR_COLUMNS]]
    shared = pairs.merge(pairs, on=["session_id", "truth_row"], suffixes=("_a", "_b"))
    shared = shared[shared["method_a"] < shared["method_b"]]
    differences = {}
    for percent in PERCENTS:
        for boundary in ("onset", "offset"):
            a = shared[f"{boundary}_error_{percent}_a"].to_numpy(dtype=float)
            b = shared[f"{boundary}_error_{percent}_b"].to_numpy(dtype=float)
            differences[f"{boundary}_signed_{percent}"] = a - b
            differences[f"{boundary}_absolute_{percent}"] = np.abs(a) - np.abs(b)
    names = list(differences)
    shared = shared[["session_id", "method_a", "method_b"]].assign(**differences)
    pair = ["method_a", "method_b"]
    pooled = shared.groupby(pair)[names].median()
    n_shared = shared.groupby(pair).size().rename("n_shared")
    per_session = shared.groupby([*pair, "session_id"])[names].median().reset_index()
    estimates = grouped_intervals(
        per_session, pair, _means(*names), names, n_resamples=n_resamples
    ).set_index(pair)
    tests = _sign_flips(per_session, pair, names).set_index(pair)
    summary = (
        comparisons.groupby(pair)
        .agg(n_run=("session_id", "size"), jaccard_truth_ids=("jaccard_truth_ids", "mean"))
        .join(n_shared)
        .join(per_session.groupby(pair).size().rename("n_sessions"))
        # every pair of the expression's methods, one never compared included
        .reindex(pd.MultiIndex.from_frame(_expected_pairs(tables, expression)))
        .fillna({"n_run": 0, "n_shared": 0, "n_sessions": 0})
    )
    # each pair's values of every stem at every percent, a pair missing NaN
    found = pooled.add_suffix("_pooled").join([estimates, tests]).reindex(summary.index)

    def by_fraction(stem: str, suffix: str) -> np.ndarray[Any, Any]:
        """A column at each percent in turn, pair by pair."""
        columns = [f"{stem}_{percent}{suffix}" for percent in PERCENTS]
        values: np.ndarray[Any, Any] = found[columns].to_numpy(dtype=float).ravel()
        return values

    stems = [f"{b}_{m}" for b in ("onset", "offset") for m in ("signed", "absolute")]
    n_percents = len(PERCENTS)
    counts = summary[["n_shared", "n_sessions", "n_run"]].astype(int)
    columns: dict[str, Any] = {
        "expression": expression,
        "method_a": np.repeat(summary.index.get_level_values("method_a"), n_percents),
        "method_b": np.repeat(summary.index.get_level_values("method_b"), n_percents),
        "fraction": np.tile([percent / 100 for percent in PERCENTS], len(summary)),
        "n_shared": np.repeat(counts["n_shared"].to_numpy(), n_percents),
        "n_sessions": np.repeat(counts["n_sessions"].to_numpy(), n_percents),
        "n_sessions_without": np.repeat(
            (counts["n_run"] - counts["n_sessions"]).to_numpy(), n_percents
        ),
        "jaccard_truth_ids": np.repeat(summary["jaccard_truth_ids"].to_numpy(), n_percents),
    }
    for stem in stems:
        for part, suffix in (
            ("pooled", "_pooled"),
            ("estimate", ""),
            ("low", "_low"),
            ("high", "_high"),
            ("p", "_p"),
        ):
            columns[f"{stem}_{part}"] = by_fraction(stem, suffix)
        dropped = np.nan_to_num(by_fraction(stem, "_n_dropped"), nan=0.0)
        columns[f"{stem}_n_dropped"] = dropped.astype(int)
    timing = pd.DataFrame(columns)[list(PAIRED_TIMING_COLUMNS)]
    return _with_pair_failures(timing, tables)


# Operating curves

# False positives per minute at which recall and errors are read off a curve.
FP_TARGETS = (0.5, 1.0, 2.0, 5.0)
# What a curve gives at a target, in this order.
AT_TARGET = ("recall", "median_onset_error", "median_offset_error")


def _at_fp_rates(
    fp_rate: ArrayLike,
    recall: ArrayLike,
    values: ArrayLike,
    targets: ArrayLike,
    floor: float,
) -> np.ndarray[Any, Any]:
    """``at_fp_rate`` on arrays, every target at once: ``values``, shape
    (n_settings, n_columns), read off at each target, shape (n_targets,
    n_columns)."""
    fp_rate = np.asarray(fp_rate, dtype=float)
    recall = np.asarray(recall, dtype=float)
    values = np.asarray(values, dtype=float).reshape(len(fp_rate), -1)
    targets = np.log(np.asarray(targets, dtype=float))
    found = np.full((len(targets), values.shape[1]), np.nan)
    if not len(fp_rate):
        return found
    x = np.log(np.maximum(fp_rate, floor))
    # FP rate up, then recall down (NaN last), then threshold order
    order = np.lexsort((np.arange(len(x)), -recall, x))
    ranked = x[order]
    repeated = (ranked[1:] == ranked[:-1]) | (np.isnan(ranked[1:]) & np.isnan(ranked[:-1]))
    kept = order[np.concatenate([[True], ~repeated])]
    xs = x[kept]
    inside = (xs[0] <= targets) & (targets <= xs[-1])
    for column in range(values.shape[1]):
        found[inside, column] = np.interp(targets[inside], xs, values[kept, column])
    return found


def at_fp_rate(
    curve: pd.DataFrame, target: float, floor: float, columns: Sequence[str]
) -> pd.Series:
    """Read a curve's columns off at a false-positive rate.

    Settings with the same floored false-positive rate keep one row, the best
    recall (ties: the first in threshold order), whole; every column is then
    interpolated linearly in log false-positive rate between the same two
    bracketing rows, so recall and the boundary errors describe the same
    settings.

    Parameters
    ----------
    curve : pandas.DataFrame
        One row per setting of one method, condition, expression and minimum
        IoU, in threshold order, with ``fp_rate``, ``recall`` and ``columns``.
    target : float
        False positives per minute.
    floor : float
        Half of 1 / the total non-event minutes, the estimate's resolution:
        a rate of 0 counts as this.
    columns : sequence of str

    Returns
    -------
    found : pandas.Series
        Indexed by ``columns``; NaN outside the curve's range of rates.
    """
    found = _at_fp_rates(
        curve["fp_rate"], curve["recall"], curve[list(columns)], [target], floor
    )
    return pd.Series(found[0], index=list(columns))


class Pool:
    """Counts and errors of some groups over some units, pooled with weights.

    A unit is what a bootstrap resamples (a session, or a replicate across
    conditions); a group is what a statistic is of (a method and setting,
    say). ``pool(weights)`` sums each count column over the units, each unit
    counted its weight, and takes each error column's median over the pairs
    of the units, each pair counted its unit's weight: with weights of 1 the
    plain pooled values, with ``resample_weights``' rows ``paired_bootstrap``'s
    resamples.

    Parameters
    ----------
    counts : pandas.DataFrame
        One row per unit and group at most, with the count columns.
    count_units, count_groups : array_like of int, shape (n_count_rows,)
        Each row's unit and group, -1 for a row left out.
    errors : pandas.DataFrame
        One row per pair, with the error columns.
    error_units, error_groups : array_like of int, shape (n_error_rows,)
    n_units, n_groups : int
    sums : sequence of str
        Count columns.
    medians : sequence of str
        Error columns.
    """

    def __init__(
        self,
        counts: pd.DataFrame,
        count_units: ArrayLike,
        count_groups: ArrayLike,
        errors: pd.DataFrame,
        error_units: ArrayLike,
        error_groups: ArrayLike,
        n_units: int,
        n_groups: int,
        sums: Sequence[str],
        medians: Sequence[str] = (),
    ) -> None:
        units, groups = np.asarray(count_units), np.asarray(count_groups)
        keep = (units >= 0) & (groups >= 0)
        self.sums = {}
        for column in sums:
            dense = np.zeros((n_units, n_groups))
            np.add.at(dense, (units[keep], groups[keep]), counts[column].to_numpy(float)[keep])
            self.sums[column] = dense
        units, groups = np.asarray(error_units), np.asarray(error_groups)
        keep = (units >= 0) & (groups >= 0)
        self.error_units = units[keep]
        self.medians = {
            column: WeightedMedians(
                errors[column].to_numpy(float)[keep], groups[keep], n_groups
            )
            for column in medians
        }

    def __call__(self, weights: ArrayLike) -> dict[str, np.ndarray[Any, Any]]:
        """Each column pooled, shape (n_groups,), for weights of shape (n_units,)."""
        weights = np.asarray(weights, dtype=float)
        found = {column: weights @ dense for column, dense in self.sums.items()}
        for column, median in self.medians.items():
            found[column] = median(weights[self.error_units])
        return found


def _ratio(top: np.ndarray[Any, Any], bottom: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """``top / bottom``, NaN where ``bottom`` is 0 (or NaN)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(bottom > 0, top / np.where(bottom > 0, bottom, 1.0), np.nan)


def _rates(pooled: Mapping[str, np.ndarray[Any, Any]]) -> dict[str, np.ndarray[Any, Any]]:
    """Recall, precision and false positives per minute of pooled counts."""
    matched = pooled["n_matched"]
    return {
        "recall": _ratio(matched, pooled["n_reference"]),
        "precision": _ratio(matched, pooled["n_detected"]),
        "fp_rate": _ratio(pooled["n_detected"] - matched, pooled["minutes"]),
    }


_COUNTED = ("n_reference", "n_detected", "n_matched", "minutes", "ran")
_ERROR_MEASURES = ("onset_error", "offset_error", "abs_onset_error", "abs_offset_error")
# A method without rows: counts and errors as the pools read them.
_NO_COUNTS = pd.DataFrame(
    columns=["session_id", "setting", "replicate", "condition_id", *_COUNTED], dtype=float
)
_NO_ERRORS = pd.DataFrame(
    columns=["session_id", "setting", "replicate", "condition_id", *_ERROR_MEASURES],
    dtype=float,
)


def _codes(
    frame: pd.DataFrame, columns: Sequence[str], index: pd.Index
) -> np.ndarray[Any, Any]:
    """Each row's position in ``index`` by ``columns``, -1 where absent."""
    if len(columns) == 1:
        return np.asarray(index.get_indexer(frame[columns[0]].astype(object)))
    keys = pd.MultiIndex.from_arrays([frame[column].astype(object) for column in columns])
    return np.asarray(index.get_indexer(keys))


def _session_counts(
    scores: ConditionScores, level: float | None, sessions: Collection[str] | None = None
) -> pd.DataFrame:
    """``scores.counts`` at one ``minimum_iou`` (point methods at every
    level: they have none), of some sessions (default all), with each
    session's ``condition_id``, ``replicate``, ``minutes`` and ``ran`` (1)."""
    counts = scores.counts
    if sessions is not None:
        counts = counts[counts["session_id"].isin(sessions)]
    if level is not None:
        counts = counts[(counts["minimum_iou"] == level) | counts["minimum_iou"].isna()]
    listed = scores.sessions.set_index("session_id")
    return counts.assign(
        condition_id=counts["session_id"].map(listed["condition_id"]),
        replicate=counts["session_id"].map(listed["replicate"]),
        minutes=counts["session_id"].map(listed["minutes"]),
        ran=1.0,
    )


def _session_errors(
    scores: ConditionScores, level: float, sessions: Collection[str] | None = None
) -> pd.DataFrame:
    """``scores.errors`` at one ``minimum_iou``, of some sessions (default
    all), with each session's ``condition_id`` and ``replicate``, and
    absolute errors."""
    errors = scores.errors
    if sessions is not None:
        errors = errors[errors["session_id"].isin(sessions)]
    errors = errors[errors["minimum_iou"] == level]
    listed = scores.sessions.set_index("session_id")
    session_ids = errors["session_id"].astype(object)
    return errors.assign(
        condition_id=session_ids.map(listed["condition_id"]).to_numpy(),
        replicate=session_ids.map(listed["replicate"]).to_numpy(),
        abs_onset_error=errors["onset_error"].abs(),
        abs_offset_error=errors["offset_error"].abs(),
    )


def _sweep_settings(method: str, methods: pd.DataFrame) -> list[str]:
    """A detector's swept settings that ``methods`` lists, in threshold
    order; none for a recipe."""
    if method not in THRESHOLD_SWEEPS:
        return []
    listed = set(methods.loc[methods["method"] == method, "setting"])
    labels = (setting_label(value) for value in THRESHOLD_SWEEPS[method][1])
    return [label for label in labels if label in listed]


def _by_method(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """A frame's rows by ``method``."""
    return {
        str(method): rows
        for method, rows in frame.groupby(frame["method"].astype(object), sort=False)
    }


def _curve_pool(
    counts: pd.DataFrame,
    errors: pd.DataFrame,
    unit: str,
    units: Sequence[Any],
    settings: Sequence[str],
    medians: Sequence[str] = ("onset_error", "offset_error"),
) -> Pool:
    """A ``Pool`` of one method's settings (the groups, in order) over units
    named by the ``unit`` column, from that method's counts and errors."""
    unit_index, setting_index = pd.Index(units), pd.Index(settings)
    return Pool(
        counts,
        _codes(counts, [unit], unit_index),
        _codes(counts, ["setting"], setting_index),
        errors,
        _codes(errors, [unit], unit_index),
        _codes(errors, ["setting"], setting_index),
        len(unit_index),
        len(setting_index),
        _COUNTED,
        medians,
    )


def _complete_units(
    counts: pd.DataFrame, unit: str, keys: Sequence[str], cells: Collection[Any]
) -> set[Any]:
    """The values of ``unit`` whose rows hold every one of ``cells``.

    Parameters
    ----------
    counts : pandas.DataFrame
        One row per unit and cell the method has scores on.
    unit : str
        The column of the units (``session_id`` or ``replicate``).
    keys : sequence of str
        The columns naming a cell: ``["setting"]``, say, or
        ``["condition_id", "setting"]``.
    cells : collection
        The cells a unit must hold all of, each a value of ``keys`` (a
        tuple for several keys).

    Returns
    -------
    units : set
        The units on which the method ran in every cell: the only ones a
        comparison or a curve across the cells may pool, so that every cell
        is pooled over the same units.
    """
    if counts.empty or not len(cells):
        return set()
    labels = (
        counts[keys[0]].astype(object)
        if len(keys) == 1
        else pd.Series(list(zip(*(counts[key].astype(object) for key in keys), strict=True)))
    )
    inside = labels.isin(list(cells)).to_numpy()
    held = (
        pd.DataFrame(
            {"unit": counts[unit].to_numpy()[inside], "cell": labels.to_numpy()[inside]}
        )
        .drop_duplicates()
        .groupby("unit")
        .size()
    )
    return set(held.index[held == len(cells)])


def _only_complete(
    frame: pd.DataFrame,
    unit: str,
    keys: Sequence[str],
    complete: Mapping[tuple[Any, ...], Collection[Any]],
) -> pd.DataFrame:
    """``frame``'s rows of each cell of ``complete`` (a value of ``keys``, as
    a tuple) whose ``unit`` is one of that cell's (``_complete_units``), and
    its rows of any other cell."""
    columns = [frame[key].astype(object) for key in keys]
    held = [(*cell, value) for cell, values in complete.items() for value in values]
    keep = ~pd.MultiIndex.from_arrays(columns).isin(list(complete))
    if held:
        rows = pd.MultiIndex.from_arrays([*columns, frame[unit].astype(object)])
        keep |= rows.isin(held)
    return frame[keep]


def _sweep_inputs(
    counts: Mapping[str, pd.DataFrame],
    errors: Mapping[str, pd.DataFrame],
    method: str,
    settings: Sequence[str],
) -> tuple[set[Any], pd.DataFrame, pd.DataFrame]:
    """The sessions on which every one of a method's ``settings`` ran, and
    its counts and errors (``_by_method``'s) with those settings' rows of
    those sessions alone, so that every setting is pooled over the same."""
    mine = counts.get(method, _NO_COUNTS)
    complete = _complete_units(mine, "session_id", ["setting"], settings)
    cells = {(setting,): complete for setting in settings}
    return (
        complete,
        _only_complete(mine, "session_id", ["setting"], cells),
        _only_complete(errors.get(method, _NO_ERRORS), "session_id", ["setting"], cells),
    )


def _read_off(
    pooled: Mapping[str, np.ndarray[Any, Any]],
    targets: Sequence[float],
    columns: Sequence[str],
) -> np.ndarray[Any, Any]:
    """A pooled curve's ``columns`` (``recall`` or the pool's) at each target
    (``at_fp_rate``), shape (n_targets, n_columns), over the settings with
    scores; the floor is half of 1 / the most minutes of any of them."""
    held = pooled["ran"] > 0
    if not held.any():
        return np.full((len(targets), len(columns)), np.nan)
    rates = _rates(pooled)
    values = np.column_stack([{**pooled, **rates}[column][held] for column in columns])
    return _at_fp_rates(
        rates["fp_rate"][held],
        rates["recall"][held],
        values,
        targets,
        0.5 / pooled["minutes"][held].max(),
    )


def _condition_sessions(scores: ConditionScores, condition: str) -> list[str]:
    return list(
        scores.sessions.loc[scores.sessions["condition_id"] == condition, "session_id"]
    )


def _detectors(scores: ConditionScores) -> list[str]:
    """The detectors swept in the run, by name."""
    return sorted(set(THRESHOLD_SWEEPS) & set(scores.methods["method"]))


def _primary_of(scores: ConditionScores) -> pd.Series:
    """Each method's primary expression, by ``method``."""
    return scores.methods.drop_duplicates("method").set_index("method")["primary_expression"]


def _failure_counts(
    scores: ConditionScores, sessions: Collection[str], by: Sequence[str]
) -> pd.Series:
    """The failed calls on ``sessions``, counted by ``by``: of ``method``,
    ``setting`` and ``condition_id``."""
    failed = scores.failures[scores.failures["session_id"].isin(sessions)]
    conditions = scores.sessions.set_index("session_id")["condition_id"]
    failed = failed.assign(condition_id=failed["session_id"].map(conditions).to_numpy())
    return failed.groupby(list(by)).size().rename("n_failures")


def _sweep_failures(failed: pd.Series, method: str, settings: Sequence[str]) -> int:
    """A sweep's failed calls, of ``_failure_counts`` by method and setting."""
    return int(sum(failed.get((method, setting), 0) for setting in settings))


def operating_curves(
    scores: ConditionScores, *, condition: str = REFERENCE_CONDITION
) -> pd.DataFrame:
    """Recall against false positives per minute along each detector's sweep.

    Against each method's primary expression, pooled over the condition's
    sessions: at each setting, ``recall`` is the matched truth windows over
    all, and ``false_positives_per_minute`` the unmatched events over the
    minutes outside every network window. A sweep is pooled over the sessions
    on which every one of its settings ran, so every point of a curve is
    over the same sessions. Every interval method's main setting is a point
    too, over the sessions it ran: a detector's default, each recipe's own.
    Point methods have no interval score and are not here.

    Parameters
    ----------
    scores : ConditionScores
    condition : str, optional

    Returns
    -------
    curves : pandas.DataFrame
        One row per interval method, setting and ``minimum_iou``, by
        ``minimum_iou``, then method, then ``kind`` (``"sweep"`` in threshold
        order, ``"default"``, ``"recipe"``): ``method``, ``setting``,
        ``kind``, ``threshold`` (the swept value; NaN for a main setting),
        ``minimum_iou``,
        ``primary_expression``, ``n_sessions`` (pooled), ``n_dropped`` (those
        it ran on but another setting of the sweep did not, left out),
        ``n_reference``, ``n_detected``, ``n_matched``, ``minutes``, ``recall``,
        ``false_positives_per_minute``, ``median_onset_error``,
        ``median_offset_error``, ``median_abs_onset_error``,
        ``median_abs_offset_error`` (the matched pairs', detected minus truth
        at 10 % of the peak, seconds) and ``n_failures``.
    """
    sessions = _condition_sessions(scores, condition)
    methods = _by_intervals(scores.methods)
    rows = []
    for level in MATCH_IOU_LEVELS:
        counts = _by_method(_session_counts(scores, level, sessions))
        errors = _by_method(_session_errors(scores, level, sessions))
        for method, own in methods.groupby("method", sort=True):
            sweep = _sweep_settings(method, own)
            settings = [*sweep, *sorted(set(own["setting"]) - set(sweep))]
            mine = counts.get(method, _NO_COUNTS)
            ran = mine.groupby(mine["setting"].astype(object)).size()
            _, own_counts, own_errors = _sweep_inputs(counts, errors, method, sweep)
            pooled = _curve_pool(
                own_counts, own_errors, "session_id", sessions, settings, _ERROR_MEASURES
            )(np.ones(len(sessions)))
            rates = _rates(pooled)
            for position, setting in enumerate(settings):
                if not ran.get(setting, 0):
                    continue
                swept = setting in sweep
                rows.append(
                    {
                        "method": method,
                        "setting": setting,
                        "kind": "sweep"
                        if swept
                        else ("default" if setting == "default" else "recipe"),
                        "threshold": float(setting) if swept else np.nan,
                        "minimum_iou": level,
                        "n_sessions": int(pooled["ran"][position]),
                        "n_dropped": int(ran[setting] - pooled["ran"][position]),
                        **{
                            column: int(pooled[column][position])
                            for column in ("n_reference", "n_detected", "n_matched")
                        },
                        "minutes": pooled["minutes"][position],
                        "recall": rates["recall"][position],
                        "false_positives_per_minute": rates["fp_rate"][position],
                        **{
                            f"median_{column}": pooled[column][position]
                            for column in _ERROR_MEASURES
                        },
                    }
                )
    curves = pd.DataFrame(rows)
    if curves.empty:
        return curves
    curves.insert(5, "primary_expression", curves["method"].map(_primary_of(scores)))
    failed = _failure_counts(scores, sessions, ["method", "setting"])
    curves = curves.join(failed, on=["method", "setting"]).fillna({"n_failures": 0})
    return curves.astype({"n_failures": int})


def operating_points(
    scores: ConditionScores,
    *,
    condition: str = REFERENCE_CONDITION,
    targets: Sequence[float] = FP_TARGETS,
    n_resamples: int = N_RESAMPLES,
) -> pd.DataFrame:
    """Each detector's recall and errors at target false-positive rates.

    ``at_fp_rate`` on the detector's sweep (``operating_curves``' rows of
    kind ``"sweep"``) against its primary expression, pooled over the
    sessions on which every setting of the sweep ran, with 95 % intervals
    from resampling the condition's sessions (``paired_bootstrap``'s
    draws): each resample pools its sessions into a curve and reads it off
    again, the setting chosen afresh. A target outside a curve's range of
    rates is NaN, never the nearest end.

    Parameters
    ----------
    scores : ConditionScores
    condition : str, optional
    targets : sequence of float, optional
        False positives per minute.
    n_resamples : int, optional

    Returns
    -------
    points : pandas.DataFrame
        One row per detector, ``minimum_iou`` and target: ``method``,
        ``primary_expression``, ``minimum_iou``, ``fp_target``, then for
        ``recall``, ``median_onset_error`` and ``median_offset_error``
        (seconds, detected minus truth at 10 %) the estimate, ``_low`` and
        ``_high`` (over the resamples whose curve reaches the target; none
        where the estimate is NaN); ``attained``, the fraction of resamples
        whose curve reaches it; ``n_sessions``, the sessions every setting
        was pooled over; ``n_dropped``, the condition's other sessions (some
        setting failed there); ``n_failures`` (the sweep's failed calls).
    """
    sessions = _condition_sessions(scores, condition)
    weights = resample_weights(len(sessions), n_resamples=n_resamples)
    primary = _primary_of(scores)
    failed = _failure_counts(scores, sessions, ["method", "setting"])
    # the pool's columns read off, in AT_TARGET's order
    columns = ("recall", "onset_error", "offset_error")
    rows = []
    for level in MATCH_IOU_LEVELS:
        counts = _by_method(_session_counts(scores, level, sessions))
        errors = _by_method(_session_errors(scores, level, sessions))
        for method in _detectors(scores):
            settings = _sweep_settings(method, scores.methods)
            complete, own_counts, own_errors = _sweep_inputs(counts, errors, method, settings)
            pool = _curve_pool(own_counts, own_errors, "session_id", sessions, settings)
            estimate = _read_off(pool(np.ones(len(sessions))), targets, columns)
            draws = np.array([_read_off(pool(w), targets, columns) for w in weights])
            low, high = _conditional_intervals(estimate, draws)
            attained = np.isfinite(draws[:, :, 0]).mean(axis=0)
            n_failures = _sweep_failures(failed, method, settings)
            for position, target in enumerate(targets):
                row: dict[str, Any] = {
                    "method": method,
                    "primary_expression": primary[method],
                    "minimum_iou": level,
                    "fp_target": target,
                }
                for column, name in enumerate(AT_TARGET):
                    row[name] = estimate[position, column]
                    row[f"{name}_low"] = low[position, column]
                    row[f"{name}_high"] = high[position, column]
                row["attained"] = attained[position]
                row["n_sessions"] = len(complete)
                row["n_dropped"] = len(sessions) - len(complete)
                row["n_failures"] = n_failures
                rows.append(row)
    return pd.DataFrame(rows)


def _conditional_intervals(
    estimate: np.ndarray[Any, Any], draws: np.ndarray[Any, Any]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Percentile intervals of resampled values shaped like ``estimate``,
    over the resamples where each is defined; NaN wherever the estimate is
    (a target the full data cannot reach has no interval)."""
    low, high = percentile_intervals(draws.reshape(len(draws), -1))
    low, high = low.reshape(estimate.shape), high.reshape(estimate.shape)
    missing = ~np.isfinite(estimate)
    low[missing] = high[missing] = np.nan
    return low, high


def choose_setting(recall: ArrayLike, fp_rate: ArrayLike, target: float) -> int | None:
    """The setting a threshold recommendation takes at a false-positive rate.

    Parameters
    ----------
    recall, fp_rate : array_like, shape (n_settings,)
        In threshold order.
    target : float

    Returns
    -------
    position : int or None
        The setting with the best recall among those at or below ``target``
        (ties: the first in threshold order); None when none is.
    """
    recall = np.asarray(recall, dtype=float)
    allowed = np.asarray(fp_rate, dtype=float) <= target
    allowed &= np.isfinite(recall)
    if not allowed.any():
        return None
    return int(np.argmax(np.where(allowed, recall, -np.inf)))


def held_out_thresholds(
    scores: ConditionScores,
    *,
    condition: str = REFERENCE_CONDITION,
    targets: Sequence[float] = FP_TARGETS,
    n_resamples: int = N_RESAMPLES,
) -> pd.DataFrame:
    """A threshold per detector and target, chosen and judged on separate replicates.

    On the calibration replicates (``is_held_out`` false: even ids) each
    detector's sweep is pooled against its primary expression at IoU 0 and
    ``choose_setting`` picks a setting; its recall, false positives per
    minute and median errors are reported on the held-out replicates (odd
    ids) alone, with intervals from resampling those sessions. Both pool
    only the sessions on which every setting of the sweep ran. The curves
    stay descriptive: this is the number a recommendation may quote.

    Parameters
    ----------
    scores : ConditionScores
    condition : str, optional
    targets : sequence of float, optional
    n_resamples : int, optional

    Returns
    -------
    thresholds : pandas.DataFrame
        One row per detector and target: ``method``, ``primary_expression``,
        ``fp_target``, ``setting`` (``""`` when no setting is at or below the
        target on the calibration replicates, and every value NaN),
        ``calibration_recall``, ``calibration_fp_rate``,
        ``n_calibration_sessions``, then ``recall``,
        ``false_positives_per_minute``, ``median_onset_error`` and
        ``median_offset_error`` on the held-out sessions, each with ``_low``
        and ``_high``; ``n_held_out_sessions``, ``held_out_replicates``
        (space-separated, those pooled), ``n_dropped`` (the condition's
        sessions left out of both, some setting having failed there) and
        ``n_failures`` (the sweep's, on the condition).
    """
    listed = scores.sessions[scores.sessions["condition_id"] == condition]
    held = listed["replicate"].map(is_held_out).to_numpy(dtype=bool)
    calibration = list(listed.loc[~held, "session_id"])
    held_out = list(listed.loc[held, "session_id"])
    weights = resample_weights(len(held_out), n_resamples=n_resamples)
    counts = _by_method(_session_counts(scores, 0.0, listed["session_id"]))
    errors = _by_method(_session_errors(scores, 0.0, listed["session_id"]))
    primary = _primary_of(scores)
    failed = _failure_counts(scores, list(listed["session_id"]), ["method", "setting"])
    measures = (
        "recall",
        "false_positives_per_minute",
        "median_onset_error",
        "median_offset_error",
    )
    rows = []
    for method in _detectors(scores):
        settings = _sweep_settings(method, scores.methods)
        complete, own_counts, own_errors = _sweep_inputs(counts, errors, method, settings)
        calibrated = _curve_pool(own_counts, own_errors, "session_id", calibration, settings)
        chosen = _rates(calibrated(np.ones(len(calibration))))
        judged = _curve_pool(own_counts, own_errors, "session_id", held_out, settings)

        def measured(pooled: Mapping[str, np.ndarray[Any, Any]], position: int) -> list[float]:
            rates = _rates(pooled)
            return [
                rates["recall"][position],
                rates["fp_rate"][position],
                pooled["onset_error"][position],
                pooled["offset_error"][position],
            ]

        for target in targets:
            position = choose_setting(chosen["recall"], chosen["fp_rate"], target)
            row: dict[str, Any] = {
                "method": method,
                "primary_expression": primary[method],
                "fp_target": target,
                "setting": "" if position is None else settings[position],
                "calibration_recall": np.nan
                if position is None
                else chosen["recall"][position],
                "calibration_fp_rate": np.nan
                if position is None
                else chosen["fp_rate"][position],
                "n_calibration_sessions": len(complete & set(calibration)),
            }
            if position is None:
                estimate = low = high = np.full(len(measures), np.nan)
            else:
                estimate = np.array(measured(judged(np.ones(len(held_out))), position))
                low, high = percentile_intervals(
                    [measured(judged(w), position) for w in weights]
                )
            for column, name in enumerate(measures):
                row[name] = estimate[column]
                row[f"{name}_low"] = low[column]
                row[f"{name}_high"] = high[column]
            pooled = listed[held & listed["session_id"].isin(complete).to_numpy()]
            row["n_held_out_sessions"] = len(pooled)
            row["held_out_replicates"] = " ".join(str(r) for r in pooled["replicate"])
            row["n_dropped"] = len(listed) - len(complete)
            row["n_failures"] = _sweep_failures(failed, method, settings)
            rows.append(row)
    return pd.DataFrame(rows)


# Across conditions, paired by replicate

# What a main setting is measured by across conditions: pooled over the
# replicates, against the primary expression at IoU 0.
MEASURES = (
    "recall",
    "precision",
    "false_positives_per_minute",
    "median_onset_error",
    "median_offset_error",
    "participation",
)
# The measures a point method has: no pair has bounds.
POINT_MEASURES = ("recall", "precision", "false_positives_per_minute")
ROBUSTNESS_MEASURES = ("recall", "precision", "median_onset_error")
CHANGE_COLUMNS = (
    "condition_id",
    "method",
    "setting",
    "primary_expression",
    "scoring",
    "measure",
    "n_replicates",
    "value",
    "value_low",
    "value_high",
    "change",
    "change_low",
    "change_high",
    "change_p",
    "n_paired",
    "n_dropped",
    "n_failures",
)
# A method whose recall moves more than this across a factor's levels is listed.
RECALL_CHANGE = 0.1


# The error column each median measure is of.
_MEDIAN_OF = {"median_onset_error": "onset_error", "median_offset_error": "offset_error"}


def _measured(
    pooled: Mapping[str, np.ndarray[Any, Any]], measures: Sequence[str]
) -> np.ndarray[Any, Any]:
    """Some of ``MEASURES`` of pooled counts and errors, shape (n_measures,
    n_groups); ``pooled`` holds the medians they need."""
    rates = _rates(pooled)
    rates = {
        "recall": rates["recall"],
        "precision": rates["precision"],
        "false_positives_per_minute": rates["fp_rate"],
        "participation": _ratio(pooled["principal_fraction"], pooled["n_events"]),
    }
    return np.array(
        [pooled[_MEDIAN_OF[name]] if name in _MEDIAN_OF else rates[name] for name in measures]
    )


def _main_methods(scores: ConditionScores) -> pd.DataFrame:
    """The main settings: ``method``, ``setting``, ``primary_expression``,
    ``scoring``."""
    return main_rows(scores.methods).reset_index(drop=True)


def condition_pool(
    scores: ConditionScores,
    condition_ids: Sequence[str],
    replicates: Sequence[int],
    medians: Sequence[str] = ("onset_error", "offset_error"),
) -> tuple[Pool, pd.MultiIndex, pd.Series]:
    """The main settings in some conditions, pooled over shared replicates.

    A main setting is pooled only over the replicates on which it ran in
    every one of the conditions, so its values in all of them, and their
    differences, are over the same replicates.

    Parameters
    ----------
    scores : ConditionScores
    condition_ids : sequence of str
    replicates : sequence of int
        The units: a replicate's session in every condition shares its seed,
        so drawing it keeps the conditions paired.
    medians : sequence of str, optional
        The errors whose medians the pool takes, of ``onset_error`` and
        ``offset_error`` (signed, at IoU 0).

    Returns
    -------
    pool : Pool
        Counts (with participation), and the ``medians``' errors, of each
        group.
    groups : pandas.MultiIndex
        ``(condition_id, method, setting)``, every condition with every main
        setting.
    paired : pandas.Series
        By ``(method, setting)``: the replicates pooled, those it ran on in
        every condition.
    """
    main = _main_methods(scores)
    groups = pd.MultiIndex.from_tuples(
        [
            (condition, method, setting)
            for condition in condition_ids
            for method, setting in main[["method", "setting"]].itertuples(index=False)
        ],
        names=["condition_id", "method", "setting"],
    )
    sessions = scores.sessions[scores.sessions["condition_id"].isin(condition_ids)]
    sessions = set(sessions.loc[sessions["replicate"].isin(replicates), "session_id"])
    counts = main_rows(_session_counts(scores, 0.0, sessions)).merge(
        scores.participation, on=list(_KEY), how="left"
    )
    participation = ["n_events", "principal_fraction"]
    counts[participation] = counts[participation].astype(float).fillna(0.0)
    errors = main_rows(_session_errors(scores, 0.0, sessions))
    # each main setting's replicates with scores in every condition
    complete = {
        (method, setting): _complete_units(rows, "replicate", ["condition_id"], condition_ids)
        for (method, setting), rows in counts.groupby(["method", "setting"])
    }
    paired = pd.Series(
        [
            len(complete.get(key, ()))
            for key in main[["method", "setting"]].itertuples(False, None)
        ],
        index=pd.MultiIndex.from_frame(main[["method", "setting"]]),
        dtype=int,
    )

    # each main setting's rows of its complete replicates alone
    kept = {**dict.fromkeys(paired.index, ()), **complete}
    counts = _only_complete(counts, "replicate", ["method", "setting"], kept)
    errors = _only_complete(errors, "replicate", ["method", "setting"], kept)
    units = pd.Index(replicates)
    keys = ["condition_id", "method", "setting"]
    pool = Pool(
        counts,
        _codes(counts, ["replicate"], units),
        _codes(counts, keys, groups),
        errors,
        _codes(errors, ["replicate"], units),
        _codes(errors, keys, groups),
        len(units),
        len(groups),
        (*_COUNTED, "n_events", "principal_fraction"),
        medians,
    )
    return pool, groups, paired


def paired_changes(
    scores: ConditionScores,
    condition_ids: Sequence[str],
    reference: str,
    *,
    measures: Sequence[str] = MEASURES,
    n_resamples: int = N_RESAMPLES,
) -> pd.DataFrame:
    """Each main setting's measures in some conditions, and their changes from one.

    Over the replicates every condition has, a replicate's sessions in each
    condition sharing its seed (common random numbers), and of those only
    the ones on which the main setting ran in every condition
    (``condition_pool``): each value is pooled over them, and each change is
    the pooled value minus the reference condition's pooled value, both from
    the same resamples of replicates (``paired_bootstrap`` with
    ``key="replicate"``); its p-value is ``sign_flip_test``'s over the
    per-replicate changes (each replicate's value alone minus its reference
    value alone) where both are defined.

    Parameters
    ----------
    scores : ConditionScores
    condition_ids : sequence of str
        Present in the run; ``reference`` among them.
    reference : str
    measures : sequence of str, optional
        Of ``MEASURES``; a point method gets only ``POINT_MEASURES``.
    n_resamples : int, optional

    Returns
    -------
    changes : pandas.DataFrame
        One row per condition, main setting and measure (``CHANGE_COLUMNS``):
        ``n_replicates`` (pooled), ``value`` with ``_low`` and ``_high``;
        ``change`` (value minus the reference's) with ``_low``, ``_high``
        and ``_p``, ``n_paired`` (replicates in the test); ``n_dropped``
        (replicates the conditions share left out because the method failed
        on one of them); ``n_failures`` (the condition's sessions without the
        method's scores, of the shared replicates). Errors are seconds,
        detected minus truth at 10 %;
        participation is the mean fraction of place and pyramidal units
        active in the method's events.
    """
    listed = scores.sessions[scores.sessions["condition_id"].isin(condition_ids)]
    replicates = _shared_replicates(scores, condition_ids)
    medians = [_MEDIAN_OF[name] for name in measures if name in _MEDIAN_OF]
    pool, groups, paired = condition_pool(scores, condition_ids, replicates, medians)
    base = groups.get_indexer(
        pd.MultiIndex.from_arrays(
            [
                [reference] * len(groups),
                groups.get_level_values("method"),
                groups.get_level_values("setting"),
            ]
        )
    )

    def statistic(weights: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        values = _measured(pool(weights), measures)
        return np.stack([values, values - values[:, base]])

    estimate = statistic(np.ones(len(replicates)))
    draws = np.array(
        [statistic(w) for w in resample_weights(len(replicates), n_resamples=n_resamples)]
    )
    low, high = _conditional_intervals(estimate, draws)
    alone = np.array([statistic(w) for w in np.eye(len(replicates))])[:, 1]
    main = _main_methods(scores).set_index(["method", "setting"])
    kept = listed[listed["replicate"].isin(replicates)]["session_id"]
    n_failed = _failure_counts(scores, kept, ["condition_id", "method", "setting"])
    rows = []
    for position, (condition, method, setting) in enumerate(groups):
        scoring = main.loc[(method, setting), "scoring"]
        for index, measure in enumerate(measures):
            if scoring == PEAK_CONTAINMENT and measure not in POINT_MEASURES:
                continue
            per_replicate = alone[:, index, position]
            finite = per_replicate[np.isfinite(per_replicate)]
            rows.append(
                {
                    "condition_id": condition,
                    "method": method,
                    "setting": setting,
                    "primary_expression": main.loc[(method, setting), "primary_expression"],
                    "scoring": scoring,
                    "measure": measure,
                    "n_replicates": int(paired[method, setting]),
                    "value": estimate[0, index, position],
                    "value_low": low[0, index, position],
                    "value_high": high[0, index, position],
                    "change": estimate[1, index, position],
                    "change_low": low[1, index, position],
                    "change_high": high[1, index, position],
                    "change_p": sign_flip_test(finite),
                    "n_paired": len(finite),
                    "n_dropped": len(replicates) - int(paired[method, setting]),
                    "n_failures": int(n_failed.get((condition, method, setting), 0)),
                }
            )
    return pd.DataFrame(rows, columns=list(CHANGE_COLUMNS))


def _condition_ids(scores: ConditionScores) -> dict[tuple[str, str], str]:
    """The id of each condition of the run with sessions, by its
    ``conditions.csv`` factor and level."""
    present = set(scores.sessions["condition_id"])
    listed = scores.conditions[["condition_id", "factor", "level"]]
    return {
        (factor, level): condition
        for condition, factor, level in listed.itertuples(index=False)
        if condition in present
    }


def _cell(factors: Sequence[str], levels: Sequence[str]) -> tuple[str, str]:
    """The ``conditions.csv`` factor and level of the condition setting each
    factor to its level: the reference's own when every level is the
    reference, else those of the factors moved from it, joined by commas."""
    moved = [
        (f, level)
        for f, level in zip(factors, levels, strict=True)
        if level != REFERENCE_LEVEL
    ]
    if not moved:
        return REFERENCE_CONDITION, REFERENCE_LEVEL
    return ",".join(f for f, _ in moved), ",".join(level for _, level in moved)


def _level_conditions(scores: ConditionScores, factor: str) -> list[tuple[str, str]]:
    """A one-factor factor's (level, condition id) in its levels' order,
    those the run holds."""
    ids = _condition_ids(scores)
    cells = ((level, _cell([factor], [level])) for level in factor_levels(factor))
    return [(level, ids[cell]) for level, cell in cells if cell in ids]


def robustness(
    scores: ConditionScores,
    *,
    measures: Sequence[str] = ROBUSTNESS_MEASURES,
    n_resamples: int = N_RESAMPLES,
) -> pd.DataFrame:
    """Each main setting's recall, precision and onset error along each factor.

    For each one-factor factor of the run's conditions (the grid's and the
    alternative models'), ``paired_changes`` over its levels, the reference
    condition in place, on the replicates they share.

    Parameters
    ----------
    scores : ConditionScores
    measures : sequence of str, optional
    n_resamples : int, optional

    Returns
    -------
    robustness : pandas.DataFrame
        ``factor``, ``level`` (``factor_levels``' order), then
        ``CHANGE_COLUMNS``, each change from the reference level.
    """
    factors = [
        factor
        for factor in dict.fromkeys(scores.conditions["factor"])
        if factor != REFERENCE_CONDITION and "," not in factor
    ]
    parts = []
    for factor in factors:
        levels = _level_conditions(scores, factor)
        if len(levels) < 2 or REFERENCE_CONDITION not in dict(levels).values():
            continue
        changes = paired_changes(
            scores,
            [condition for _, condition in levels],
            REFERENCE_CONDITION,
            measures=measures,
            n_resamples=n_resamples,
        )
        level_of = {condition: level for level, condition in levels}
        changes.insert(0, "level", changes["condition_id"].map(level_of))
        changes.insert(0, "factor", factor)
        parts.append(changes)
    return _concat(parts, ["factor", "level", *CHANGE_COLUMNS])


def robustness_crossed(
    scores: ConditionScores,
    *,
    measures: Sequence[str] = ROBUSTNESS_MEASURES,
    n_resamples: int = N_RESAMPLES,
) -> pd.DataFrame:
    """Each main setting's measures over the cells of each crossed pair of factors.

    A cell is a level of each factor: both at the reference is the reference
    condition, one at the reference that factor's one-factor condition, and
    neither the crossed condition. ``paired_changes`` over the cells the run
    holds, from the reference.

    Parameters
    ----------
    scores : ConditionScores
    measures : sequence of str, optional
    n_resamples : int, optional

    Returns
    -------
    crossed : pandas.DataFrame
        ``factors`` (``"<first>,<second>"``), ``level_1``, ``level_2``, then
        ``CHANGE_COLUMNS``.
    """
    ids = _condition_ids(scores)
    pairs = [factor for factor in dict.fromkeys(scores.conditions["factor"]) if "," in factor]
    parts = []
    for pair in pairs:
        factors = pair.split(",")
        cells = {}
        for levels in itertools.product(*(factor_levels(factor) for factor in factors)):
            cell = _cell(factors, levels)
            if cell in ids:
                cells[ids[cell]] = levels
        if REFERENCE_CONDITION not in cells or len(cells) < 2:
            continue
        changes = paired_changes(
            scores,
            list(cells),
            REFERENCE_CONDITION,
            measures=measures,
            n_resamples=n_resamples,
        )
        firsts = {condition: levels[0] for condition, levels in cells.items()}
        seconds = {condition: levels[1] for condition, levels in cells.items()}
        changes.insert(0, "level_2", changes["condition_id"].map(seconds))
        changes.insert(0, "level_1", changes["condition_id"].map(firsts))
        changes.insert(0, "factors", pair)
        parts.append(changes)
    return _concat(parts, ["factors", "level_1", "level_2", *CHANGE_COLUMNS])


def recall_changes(table: pd.DataFrame, threshold: float = RECALL_CHANGE) -> pd.DataFrame:
    """The methods whose recall moves more than ``threshold`` across a factor.

    Parameters
    ----------
    table : pandas.DataFrame
        ``robustness``' rows.
    threshold : float, optional

    Returns
    -------
    changes : pandas.DataFrame
        One row per factor and main setting whose pooled recall, over the
        factor's levels, spans more than ``threshold``: ``factor``,
        ``method``, ``setting``, ``scoring``, ``lowest`` and ``highest`` (the
        levels), ``recall_lowest``, ``recall_highest``, ``span``; largest span
        first.
    """
    columns = [
        "factor",
        "method",
        "setting",
        "scoring",
        "lowest",
        "highest",
        "recall_lowest",
        "recall_highest",
        "span",
    ]
    recall_rows = table[(table["measure"] == "recall") & table["value"].astype(float).notna()]
    rows = []
    for (factor, method, setting), group in recall_rows.groupby(
        ["factor", "method", "setting"], sort=False
    ):
        low, high = group.loc[group["value"].idxmin()], group.loc[group["value"].idxmax()]
        span = high["value"] - low["value"]
        if span > threshold:
            rows.append(
                [
                    factor,
                    method,
                    setting,
                    scoring_rule(str(method)),
                    low["level"],
                    high["level"],
                    low["value"],
                    high["value"],
                    span,
                ]
            )
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("span", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


# Model sensitivity

# The simulator's alternative models: (factor, level, condition id).
MODEL_ALTERNATIVES = tuple(
    (condition.factor, condition.level, condition.condition_id)
    for condition in benchmark_conditions()
    if condition.factor in ALTERNATIVES
)
SENSITIVITY_COLUMNS = (
    "alternative",
    "status",
    "method",
    "setting",
    "primary_expression",
    "scoring",
    "measure",
    "fp_target",
    "n_replicates",
    "reference_value",
    "value",
    "change",
    "change_low",
    "change_high",
    "change_p",
    "n_paired",
    "n_dropped",
    "n_failures",
)
ORDER_COLUMNS = (
    "alternative",
    "status",
    "fp_target",
    "primary_expression",
    "method_a",
    "method_b",
    "n_replicates",
    "n_dropped",
    "n_failures_a",
    "n_failures_b",
    "setting_a",
    "setting_b",
    "reference_difference",
    "reference_low",
    "reference_high",
    "alternative_difference",
    "alternative_low",
    "alternative_high",
    "supported",
    "reversed",
    "p_reversed",
)
# Relative and absolute room within which a validation statistic is unchanged.
_RELATIVE_ROOM, _ABSOLUTE_ROOM = 0.01, 0.001


@dataclasses.dataclass(frozen=True)
class SweepRecalls:
    """Detectors' recalls at target false-positive rates, read off their sweeps.

    Attributes
    ----------
    estimate : ndarray, shape (n_conditions, n_detectors, n_targets)
    draws : ndarray, shape (n_resamples, n_conditions, n_detectors, n_targets)
        ``paired_bootstrap``'s resamples of replicates, as weights.
    alone : ndarray, shape (n_replicates, n_conditions, n_detectors, n_targets)
        Each replicate's own curves read off.
    replicates : list of set
        Per detector, the replicates pooled: those on which every setting of
        its sweep ran in every condition, so that the conditions are paired.
    nearest : ndarray of str, shape (n_conditions, n_detectors, n_targets)
        The swept setting whose pooled false-positive rate is nearest each
        target in log rate (``""`` for a curve with none), where a spot check
        of the operating point looks.
    """

    estimate: np.ndarray[Any, Any]
    draws: np.ndarray[Any, Any]
    alone: np.ndarray[Any, Any]
    replicates: list[set[Any]]
    nearest: np.ndarray[Any, Any]

    def pair(self, a: int, b: int) -> SweepRecalls:
        """Two detectors' slices, A (``a``) at index 0 and B at 1."""
        return SweepRecalls(
            self.estimate[:, [a, b]],
            self.draws[:, :, [a, b]],
            self.alone[:, :, [a, b]],
            [self.replicates[a], self.replicates[b]],
            self.nearest[:, [a, b]],
        )


def _nearest_settings(
    pool: Pool, settings: Sequence[str], n_units: int, targets: Sequence[float]
) -> list[str]:
    """The setting of a sweep pool whose false-positive rate, pooled over
    every unit, is nearest each target in log rate; ``""`` for none."""
    pooled = pool(np.ones(n_units))
    held = np.flatnonzero(pooled["ran"] > 0)
    if not len(held):
        return [""] * len(targets)
    rates = _rates(pooled)["fp_rate"][held]
    floor = 0.5 / pooled["minutes"][held].max()
    distance = np.abs(
        np.log(np.maximum(rates, floor))[:, None] - np.log(np.asarray(targets))[None, :]
    )
    return [settings[held[position]] for position in np.argmin(distance, axis=0)]


def _sweep_recalls(
    scores: ConditionScores,
    conditions: Sequence[str],
    replicates: Sequence[int],
    detectors: Sequence[str],
    targets: Sequence[float],
    n_resamples: int,
) -> SweepRecalls:
    """Each detector's recall at each target in each condition, over the
    shared ``replicates`` on which its whole sweep ran in every condition."""
    sessions = scores.sessions[
        scores.sessions["condition_id"].isin(conditions)
        & scores.sessions["replicate"].isin(replicates)
    ]
    counts = _by_method(_session_counts(scores, 0.0, sessions["session_id"]))
    pools: list[list[Pool]] = [[] for _ in conditions]
    paired = []
    nearest = np.full((len(conditions), len(detectors), len(targets)), "", dtype=object)
    for d, detector in enumerate(detectors):
        settings = _sweep_settings(detector, scores.methods)
        own = counts.get(detector, _NO_COUNTS)
        cells = [(condition, setting) for condition in conditions for setting in settings]
        complete = _complete_units(own, "replicate", ["condition_id", "setting"], cells)
        paired.append(complete)
        own = own[own["replicate"].isin(list(complete)).to_numpy()]
        for c, (row, condition) in enumerate(zip(pools, conditions, strict=True)):
            mine = own[(own["condition_id"] == condition).to_numpy()]
            row.append(_curve_pool(mine, _NO_ERRORS, "replicate", replicates, settings, ()))
            nearest[c, d] = _nearest_settings(row[-1], settings, len(replicates), targets)

    def statistic(weights: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.array(
            [
                [_read_off(pool(weights), targets, ["recall"])[:, 0] for pool in row]
                for row in pools
            ]
        )

    weights = resample_weights(len(replicates), n_resamples=n_resamples)
    return SweepRecalls(
        estimate=statistic(np.ones(len(replicates))),
        draws=np.array([statistic(w) for w in weights]),
        alone=np.array([statistic(w) for w in np.eye(len(replicates))]),
        replicates=paired,
        nearest=nearest,
    )


def _same_primary_pairs(
    scores: ConditionScores,
    conditions: Sequence[str],
    detectors: Sequence[str],
    found: SweepRecalls,
    targets: Sequence[float],
    n_resamples: int,
) -> Iterator[tuple[int, int, SweepRecalls]]:
    """Each pair of ``detectors`` sharing a primary expression, A first by
    name, and their recalls at the targets over the replicates both were
    pooled over: ``found`` (``_sweep_recalls`` of every detector) sliced when
    their replicates are the same, else pooled again over those the two
    share, so a difference between them is paired."""
    primary = _primary_of(scores)
    for a, b in itertools.combinations(range(len(detectors)), 2):
        if primary[detectors[a]] != primary[detectors[b]]:
            continue
        if found.replicates[a] == found.replicates[b]:
            yield a, b, found.pair(a, b)
            continue
        shared = sorted(found.replicates[a] & found.replicates[b])
        pair = [detectors[a], detectors[b]]
        yield a, b, _sweep_recalls(scores, conditions, shared, pair, targets, n_resamples)


def _paired_test(
    difference: np.ndarray[Any, Any], draws: np.ndarray[Any, Any], alone: np.ndarray[Any, Any]
) -> tuple[
    np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]
]:
    """A paired difference's interval and test.

    Parameters
    ----------
    difference : ndarray, any shape
    draws : ndarray, shape (n_resamples, *difference.shape)
        Its resampled values.
    alone : ndarray, shape (n_units, *difference.shape)
        Each unit's own.

    Returns
    -------
    low, high : ndarray, shaped as ``difference``
        ``_conditional_intervals``'.
    p : ndarray, shaped as ``difference``
        ``sign_flip_test`` over the units whose own difference is finite;
        NaN where ``difference`` is not.
    n_paired : ndarray of int, shaped as ``difference``
        Those units; 0 where ``difference`` is not finite.
    """
    low, high = _conditional_intervals(difference, draws)
    p = np.full(difference.shape, np.nan)
    n_paired = np.zeros(difference.shape, dtype=int)
    for index in zip(*np.nonzero(np.isfinite(difference)), strict=True):
        own = alone[(slice(None), *index)]
        finite = own[np.isfinite(own)]
        p[index], n_paired[index] = sign_flip_test(finite), len(finite)
    return low, high, p, n_paired


def _status(value: ArrayLike, n_failures: ArrayLike, otherwise: str) -> Any:
    """A comparison's status: ``"compared"`` where ``value`` is finite, else
    ``"failed"`` where the methods have failed calls, else ``otherwise``."""
    value = np.asarray(value, dtype=float)
    failed = np.asarray(n_failures, dtype=float) > 0
    status = np.where(np.isfinite(value), "compared", np.where(failed, "failed", otherwise))
    return status.item() if status.ndim == 0 else status


def _shared_replicates(scores: ConditionScores, conditions: Sequence[str]) -> list[int]:
    listed = scores.sessions[scores.sessions["condition_id"].isin(conditions)]
    found = [set(rows["replicate"]) for _, rows in listed.groupby("condition_id")]
    return sorted(set.intersection(*found)) if len(found) == len(conditions) else []


def model_sensitivity(
    scores: ConditionScores,
    *,
    targets: Sequence[float] = FP_TARGETS,
    n_resamples: int = N_RESAMPLES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How each result moves under each of the simulator's alternative models.

    Each alternative is paired with the reference by replicate (the
    replicates both hold; a replicate's sessions share its seed): each main
    setting's ``MEASURES`` (``paired_changes``), and each detector's recall
    at the target false-positive rates, read off its sweep in each condition
    (``at_fp_rate``) and compared only where both curves reach the target.
    Each is pooled only over the replicates on which the method ran in both
    conditions (a detector: every setting of its sweep), a pair of detectors
    over the replicates both ran on. A comparison left without a value has
    ``status`` ``"failed"`` when the method (either detector of an order)
    failed on some session of the two conditions, else ``"unattainable"``
    for a target a curve does not reach; failures are never read as an
    unreachable target. An alternative the run lacks has rows too, every
    value missing (``"not run"``): the reference alone says nothing about it.

    The orders are those of detectors sharing a primary expression, by
    recall at a common target, in the reference and in the alternative, from
    the same resamples of replicates.

    Parameters
    ----------
    scores : ConditionScores
    targets : sequence of float, optional
    n_resamples : int, optional

    Returns
    -------
    changes : pandas.DataFrame
        ``SENSITIVITY_COLUMNS``: per alternative (its condition id), main
        setting and ``measure`` (``MEASURES``; a point method's
        ``POINT_MEASURES`` only), or per detector ``"recall_at_fp"`` with
        ``fp_target``, whose ``setting`` is ``"sweep"`` (the whole sweep,
        read off at the target); ``status`` (``"compared"``, ``"failed"``,
        ``"unattainable"``, ``"not run"``); ``scoring``, the rule the row's
        counts come from: filter on it, not on ``setting`` or a missing
        ``minimum_iou``, to keep point methods (``"peak_containment"``)
        apart. Then the reference's and the alternative's values over the
        ``n_replicates`` pooled, ``change`` (alternative minus reference)
        with its interval and sign-flip p-value over the replicates where
        both exist (``n_paired``), ``n_dropped`` (shared replicates left out
        for the method's failures) and ``n_failures`` (the method's failed
        calls, every setting of a sweep, in both conditions, on the shared
        replicates).
    orders : pandas.DataFrame
        ``ORDER_COLUMNS``: per alternative, target and pair of detectors
        with the same primary expression (A first by name), ``status`` as
        above, the ``n_replicates`` both were pooled over (``n_dropped`` left
        out), each detector's failed calls in both conditions
        (``n_failures_a``, ``n_failures_b``), each detector's swept setting
        whose false-positive rate in the alternative is nearest the target
        (``setting_a``, ``setting_b``: where a spot check looks), their
        recall differences (A minus B) in the reference and in the
        alternative with intervals;
        ``supported``, the reference interval excludes 0; ``reversed``, the
        two estimates have opposite signs; ``p_reversed``, the fraction of
        resamples (where both are defined) in which they do.
    """
    main = _main_methods(scores)
    detectors = _detectors(scores)
    primary = _primary_of(scores)
    present = set(scores.sessions["condition_id"])
    changes, orders = [], []
    for _, _, alternative in MODEL_ALTERNATIVES:
        if alternative not in present or REFERENCE_CONDITION not in present:
            skipped = [
                {
                    "alternative": alternative,
                    "status": "not run",
                    "method": row.method,
                    "setting": row.setting,
                    "primary_expression": row.primary_expression,
                    "scoring": row.scoring,
                    "measure": measure,
                    "fp_target": np.nan,
                }
                for row in main.itertuples(index=False)
                for measure in (MEASURES if row.scoring == INTERVAL else POINT_MEASURES)
            ]
            changes.append(pd.DataFrame(skipped, columns=list(SENSITIVITY_COLUMNS)))
            continue
        pair = [REFERENCE_CONDITION, alternative]
        found = paired_changes(scores, pair, REFERENCE_CONDITION, n_resamples=n_resamples)
        reference = found[found["condition_id"] == REFERENCE_CONDITION]
        found = found[found["condition_id"] == alternative].merge(
            reference[["method", "setting", "measure", "value", "n_failures"]].rename(
                columns={"value": "reference_value", "n_failures": "reference_failures"}
            ),
            on=["method", "setting", "measure"],
        )
        # failures in either condition, of the replicates both hold
        found["n_failures"] += found.pop("reference_failures")
        found["status"] = _status(found["change"], found["n_failures"], "compared")
        changes.append(
            found.rename(columns={"condition_id": "alternative"}).assign(fp_target=np.nan)[
                list(SENSITIVITY_COLUMNS)
            ]
        )
        replicates = _shared_replicates(scores, pair)
        listed = scores.sessions[
            scores.sessions["condition_id"].isin(pair)
            & scores.sessions["replicate"].isin(replicates)
        ]
        failed = _failure_counts(scores, listed["session_id"], ["method", "setting"])
        sweep_failures = {
            detector: _sweep_failures(
                failed, detector, _sweep_settings(detector, scores.methods)
            )
            for detector in detectors
        }
        found = _sweep_recalls(scores, pair, replicates, detectors, targets, n_resamples)
        estimate, draws, alone = found.estimate, found.draws, found.alone
        complete = found.replicates
        difference = estimate[1] - estimate[0]
        low, high, p, n_paired = _paired_test(
            difference, draws[:, 1] - draws[:, 0], alone[:, 1] - alone[:, 0]
        )
        rows = []
        for d, detector in enumerate(detectors):
            for t, target in enumerate(targets):
                rows.append(
                    {
                        "alternative": alternative,
                        "status": _status(
                            difference[d, t], sweep_failures[detector], "unattainable"
                        ),
                        "method": detector,
                        "setting": "sweep",
                        "primary_expression": primary[detector],
                        "scoring": INTERVAL,
                        "measure": "recall_at_fp",
                        "fp_target": target,
                        "n_replicates": len(complete[d]),
                        "reference_value": estimate[0, d, t],
                        "value": estimate[1, d, t],
                        "change": difference[d, t],
                        "change_low": low[d, t],
                        "change_high": high[d, t],
                        "change_p": p[d, t],
                        "n_paired": n_paired[d, t],
                        "n_dropped": len(replicates) - len(complete[d]),
                        "n_failures": sweep_failures[detector],
                    }
                )
        changes.append(pd.DataFrame(rows, columns=list(SENSITIVITY_COLUMNS)))
        pairs = {
            (a, b): recalls
            for a, b, recalls in _same_primary_pairs(
                scores, pair, detectors, found, targets, n_resamples
            )
        }
        orders.append(
            _orders(
                alternative,
                detectors,
                primary,
                targets,
                pairs,
                len(replicates),
                sweep_failures,
            )
        )
    return (
        _concat(changes, SENSITIVITY_COLUMNS),
        _concat(orders, ORDER_COLUMNS),
    )


def _orders(
    alternative: str,
    detectors: Sequence[str],
    primary: pd.Series,
    targets: Sequence[float],
    pairs: Mapping[tuple[int, int], SweepRecalls],
    n_shared: int,
    failures: Mapping[str, int],
) -> pd.DataFrame:
    """The pairs of detectors sharing a primary expression, ordered by recall
    at each target in the reference (index 0) and the alternative (1), each
    pair over the replicates both were pooled over (``_pair_recalls``), of
    ``n_shared`` the conditions share; ``failures``, each detector's failed
    calls in the two conditions."""
    rows = []
    for (a, b), found in pairs.items():
        paired = found.replicates[0]
        observed = found.estimate[:, 0] - found.estimate[:, 1]
        resampled = found.draws[:, :, 0] - found.draws[:, :, 1]
        low, high = _conditional_intervals(observed, resampled)
        for t, target in enumerate(targets):
            both = np.isfinite(resampled[:, 0, t]) & np.isfinite(resampled[:, 1, t])
            flips = np.sign(resampled[both, 0, t]) * np.sign(resampled[both, 1, t]) < 0
            n_failures = failures[detectors[a]] + failures[detectors[b]]
            rows.append(
                {
                    "alternative": alternative,
                    "status": _status(
                        observed[0, t] + observed[1, t], n_failures, "unattainable"
                    ),
                    "fp_target": target,
                    "primary_expression": primary[detectors[a]],
                    "method_a": detectors[a],
                    "method_b": detectors[b],
                    "n_replicates": len(paired),
                    "n_dropped": n_shared - len(paired),
                    "n_failures_a": failures[detectors[a]],
                    "n_failures_b": failures[detectors[b]],
                    "setting_a": found.nearest[1, 0, t],
                    "setting_b": found.nearest[1, 1, t],
                    "reference_difference": observed[0, t],
                    "reference_low": low[0, t],
                    "reference_high": high[0, t],
                    "alternative_difference": observed[1, t],
                    "alternative_low": low[1, t],
                    "alternative_high": high[1, t],
                    "supported": bool(low[0, t] > 0 or high[0, t] < 0),
                    "reversed": bool(observed[0, t] * observed[1, t] < 0),
                    "p_reversed": flips.mean() if both.any() else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=list(ORDER_COLUMNS))


DIFFERENCE_COLUMNS = (
    "primary_expression",
    "fp_target",
    "method_a",
    "method_b",
    "setting_a",
    "setting_b",
    "recall_a",
    "recall_b",
    "reached_a",
    "reached_b",
    "difference",
    "difference_low",
    "difference_high",
    "difference_p",
    "n_paired",
    "n_replicates",
    "n_dropped",
)


def operating_differences(
    scores: ConditionScores,
    *,
    condition: str = REFERENCE_CONDITION,
    targets: Sequence[float] = FP_TARGETS,
    n_resamples: int = N_RESAMPLES,
) -> pd.DataFrame:
    """Paired differences in recall at the targets between detectors sharing
    a primary expression.

    Each detector's recall is read off its sweep (``at_fp_rate``, IoU 0),
    pooled over the condition's sessions both detectors ran every setting
    on; the interval is from the same resamples of sessions for both
    (``resample_weights``), the p-value ``sign_flip_test``'s over each
    session's difference, its own curves read off alone.

    Parameters
    ----------
    scores : ConditionScores
    condition : str, optional
    targets : sequence of float, optional
    n_resamples : int, optional

    Returns
    -------
    differences : pandas.DataFrame
        ``DIFFERENCE_COLUMNS``: per pair (A first by name) and target,
        ``setting_a`` and ``setting_b`` (each detector's swept setting whose
        false-positive rate is nearest the target, where a spot check
        looks), ``recall_a`` and ``recall_b``, ``reached_a`` and ``reached_b``
        (whether each curve reaches the target; a recall it does not reach
        is NaN, never the curve's end), ``difference`` (A minus B) with
        ``_low``, ``_high`` and ``_p``, ``n_paired`` (sessions whose own
        curves both reach it), ``n_replicates`` (sessions pooled) and
        ``n_dropped`` (the condition's other sessions).
    """
    detectors = _detectors(scores)
    primary = _primary_of(scores)
    replicates = sorted(
        scores.sessions.loc[scores.sessions["condition_id"] == condition, "replicate"]
    )
    found = _sweep_recalls(scores, [condition], replicates, detectors, targets, n_resamples)
    rows = []
    for a, b, own in _same_primary_pairs(
        scores, [condition], detectors, found, targets, n_resamples
    ):
        estimate, draws, alone, paired = own.estimate, own.draws, own.alone, own.replicates[0]
        difference = estimate[0, 0] - estimate[0, 1]
        low, high, p, n_paired = _paired_test(
            difference, draws[:, 0, 0] - draws[:, 0, 1], alone[:, 0, 0] - alone[:, 0, 1]
        )
        for t, target in enumerate(targets):
            rows.append(
                {
                    "primary_expression": primary[detectors[a]],
                    "fp_target": target,
                    "method_a": detectors[a],
                    "method_b": detectors[b],
                    "setting_a": own.nearest[0, 0, t],
                    "setting_b": own.nearest[0, 1, t],
                    "recall_a": estimate[0, 0, t],
                    "recall_b": estimate[0, 1, t],
                    "reached_a": bool(np.isfinite(estimate[0, 0, t])),
                    "reached_b": bool(np.isfinite(estimate[0, 1, t])),
                    "difference": difference[t],
                    "difference_low": low[t],
                    "difference_high": high[t],
                    "difference_p": p[t],
                    "n_paired": n_paired[t],
                    "n_replicates": len(paired),
                    "n_dropped": len(replicates) - len(paired),
                }
            )
    return pd.DataFrame(rows, columns=list(DIFFERENCE_COLUMNS))


def validation_changes(checks: pd.DataFrame) -> pd.DataFrame:
    """The validation report's target statistics each alternative model moves.

    Parameters
    ----------
    checks : pandas.DataFrame
        The report's ``checks.csv``: ``check``, ``kind``, ``condition_id``,
        ``statistic``, ``observed``.

    Returns
    -------
    changed : pandas.DataFrame
        One row per alternative and target check whose pooled statistic is
        not within 1 % (and 0.001) of the reference's: ``alternative``,
        ``check``, ``statistic``, ``reference``, ``observed``.
    """
    columns = ["alternative", "check", "statistic", "reference", "observed"]
    targets = checks[checks["kind"] == "target"]
    reference = targets[targets["condition_id"] == REFERENCE_CONDITION].set_index("check")
    rows = []
    for _, _, alternative in MODEL_ALTERNATIVES:
        own = targets[targets["condition_id"] == alternative]
        for row in own.itertuples(index=False):
            if row.check not in reference.index:
                continue
            before = float(reference.loc[row.check, "observed"])
            after = float(row.observed)
            if (
                np.isfinite(before)
                and np.isfinite(after)
                and np.isclose(after, before, rtol=_RELATIVE_ROOM, atol=_ABSOLUTE_ROOM)
            ):
                continue
            rows.append([alternative, row.check, row.statistic, before, after])
    return pd.DataFrame(rows, columns=columns)


def model_sensitivity_statements(
    changes: pd.DataFrame, orders: pd.DataFrame, validation: pd.DataFrame | None = None
) -> list[str]:
    """What survives each alternative model and what depends on it, as Markdown.

    A reference order counts as a statement when its interval excludes 0
    (``supported``): it survives an alternative when the alternative's
    interval excludes 0 on the same side, loses its support when it
    includes 0, and reverses when the estimates' signs differ; it cannot be
    compared when the alternative's difference is missing, counted apart by
    why (a detector failed, or its curve does not reach the target there).
    Orders a detector's failures leave without a reference difference are
    counted too, as untested. Nothing is pooled across alternatives.

    Parameters
    ----------
    changes, orders : pandas.DataFrame
        ``model_sensitivity``'s tables.
    validation : pandas.DataFrame, optional
        ``validation_changes``' table; None (the default) for a report not
        read, which each line says rather than reading it as nothing moved.

    Returns
    -------
    lines : list of str
        One bullet per alternative, then its reversed orders.
    """
    lines = []
    for _, _, alternative in MODEL_ALTERNATIVES:
        own = changes[changes["alternative"] == alternative]
        if own.empty or (own["status"] == "not run").all():
            lines.append(
                f"- `{alternative}`: not in this run, so no statement is tested against it; "
                "the reference's results say nothing about it."
            )
            continue
        observed = "the report was not read, so what this alternative changes in it is unknown"
        if validation is not None:
            observed = "no target statistic moved by more than 1 %"
            moved = validation[validation["alternative"] == alternative]
            if len(moved):
                observed = "; ".join(
                    f"{row.check} {row.reference:.4g} to {row.observed:.4g}"
                    for row in moved.itertuples(index=False)
                )
        mine = orders[(orders["alternative"] == alternative) & orders["supported"]]
        same = np.sign(mine["reference_difference"])
        survive = int(
            (
                ((same > 0) & (mine["alternative_low"] > 0))
                | ((same < 0) & (mine["alternative_high"] < 0))
            ).sum()
        )
        reversed_ = mine[mine["reversed"]]
        missing = mine["alternative_difference"].isna()
        failed_orders = int((missing & (mine["status"] == "failed")).sum())
        lost = len(mine) - survive - len(reversed_) - int(missing.sum())
        # orders without a reference difference at all, for a detector's failures
        untested = orders[
            (orders["alternative"] == alternative)
            & ~orders["supported"]
            & (orders["status"] == "failed")
        ]
        recall = own[own["measure"] == "recall"]
        compared = recall[recall["change"].notna()]
        moved_recall = compared[(compared["change_low"] > 0) | (compared["change_high"] < 0)]
        targets = own[own["measure"] == "recall_at_fp"]
        lines.append(
            f"- `{alternative}` (validation: {observed}): of {len(mine)} reference orders "
            f"of detectors by recall at a common false-positive rate that their intervals "
            f"support, {survive} keep that support, {lost} lose it, {len(reversed_)} "
            f"reverse and {int(missing.sum())} cannot be compared here ({failed_orders} for "
            f"a failure, {int(missing.sum()) - failed_orders} out of reach) and "
            f"{len(untested)} more are untested because a detector failed; "
            f"{len(moved_recall)} of {len(compared)} main settings' recall compared moves "
            "with an interval excluding 0 "
            f"({int((recall['status'] == 'failed').sum())} failed); "
            f"{int((targets['status'] == 'unattainable').sum())} detector targets are out of "
            f"reach in one condition and {int((targets['status'] == 'failed').sum())} "
            "missing for failures, all left missing."
        )
        lines += [
            f"  - reversed at {row.fp_target:g}/min: `{row.method_a}` minus "
            f"`{row.method_b}` {row.reference_difference:+.3f} "
            f"({row.reference_low:+.3f}, {row.reference_high:+.3f}) in the reference, "
            f"{row.alternative_difference:+.3f} ({row.alternative_low:+.3f}, "
            f"{row.alternative_high:+.3f}) here; reversed in {row.p_reversed:.0%} of resamples"
            for row in reversed_.itertuples(index=False)
        ]
    return lines


# Matching sensitivity


def matching_sensitivity(
    tables: RunTables,
    matches: Matches,
    points: pd.DataFrame | None = None,
    *,
    n_resamples: int = N_RESAMPLES,
) -> pd.DataFrame:
    """The main interval methods' scores at every minimum IoU.

    The headline matches at IoU 0 (any overlap); here the same pairs are
    formed again requiring IoU 0.2 and 0.5, against each method's primary
    expression: the counts and their ratios, the IoU distribution behind
    them, the median absolute errors, recall by event type against the
    network truth, and each detector's recall at the target false-positive
    rates. Methods are ranked by recall among those sharing a primary
    expression, so an order that holds only at IoU 0 shows.

    Parameters
    ----------
    tables : RunTables
    matches : Matches
        Matched at every level of ``MATCH_IOU_LEVELS``.
    points : pandas.DataFrame, optional
        ``operating_points``' table, for the recall at each target.
    n_resamples : int, optional

    Returns
    -------
    sensitivity : pandas.DataFrame
        One row per interval method, setting and ``minimum_iou``:
        ``primary_expression``, ``n_reference``, ``n_detected``,
        ``n_matched``; ``recall``, ``precision`` and ``f1`` (pooled:
        ``2 n_matched / (n_reference + n_detected)``), each with ``_low`` and
        ``_high``; ``iou_q25``, ``median_iou``, ``iou_q75``;
        ``median_abs_onset_error`` and ``median_abs_offset_error`` (10 %,
        seconds); ``recall_<event_type>`` against the network truth;
        ``recall_at_<target>`` (detectors); ``rank`` (by recall, 1 best,
        among methods of the same primary expression at that level);
        ``n_sessions``, ``n_failures``.
    """
    primary = tables.intervals.methods[["method", "setting", "primary_expression"]]
    pairs = matches.pairs.merge(
        primary.rename(columns={"primary_expression": "expression"}),
        on=["method", "setting", "expression"],
    )
    n_truth = (
        matches.windows.groupby(["session_id", "expression"]).size().rename("n_reference")
    )
    n_events = tables.events.groupby(list(_KEY)).size().rename("n_detected")
    frame = tables.intervals.ran.merge(primary, on=["method", "setting"])
    frame = frame.join(n_truth, on=["session_id", "primary_expression"]).join(
        n_events, on=list(_KEY)
    )
    frame = frame.fillna({"n_reference": 0, "n_detected": 0})
    by = ["method", "setting", "minimum_iou"]
    parts = []
    for level in MATCH_IOU_LEVELS:
        at_level = pairs[pairs["minimum_iou"] == level]
        found = at_level.groupby(list(_KEY)).size().rename("n_matched")
        counted = (
            frame.join(found, on=list(_KEY)).fillna({"n_matched": 0}).assign(minimum_iou=level)
        )
        parts.append(counted)
    counted = pd.concat(parts, ignore_index=True)
    counted = counted.assign(
        twice_matched=2 * counted["n_matched"],
        n_either=counted["n_reference"] + counted["n_detected"],
    )
    ratios = {
        "recall": ("n_matched", "n_reference"),
        "precision": ("n_matched", "n_detected"),
        "f1": ("twice_matched", "n_either"),
    }
    table = _pooled_ratios(counted, by, ratios, _DETECTION_COUNTS, n_resamples=n_resamples)
    grouped = pairs.assign(
        abs_onset=pairs["onset_error_10"].abs(), abs_offset=pairs["offset_error_10"].abs()
    ).groupby(by)
    table = table.join(
        pd.DataFrame(
            {
                "iou_q25": grouped["iou"].quantile(0.25),
                "median_iou": grouped["iou"].median(),
                "iou_q75": grouped["iou"].quantile(0.75),
                "median_abs_onset_error": grouped["abs_onset"].median(),
                "median_abs_offset_error": grouped["abs_offset"].median(),
            }
        ),
        on=by,
    )
    network = matches.windows[matches.windows["expression"] == "network"]
    typed = matches.pairs[matches.pairs["expression"] == "network"].merge(
        network[["session_id", "row", "type"]],
        left_on=["session_id", "truth_row"],
        right_on=["session_id", "row"],
    )
    n_true = network.groupby(["session_id", "type"]).size().rename("n_true").reset_index()
    for kind in rd.EVENT_TYPES:
        exists = n_true[n_true["type"] == kind].set_index("session_id")["n_true"]
        found = typed[typed["type"] == kind].groupby([*_KEY, "minimum_iou"]).size()
        rows = counted[[*_KEY, "minimum_iou"]].assign(
            n_true=counted["session_id"].map(exists).fillna(0).to_numpy(),
            n_found=found.reindex(pd.MultiIndex.from_frame(counted[[*_KEY, "minimum_iou"]]))
            .fillna(0)
            .to_numpy(),
        )
        sums = rows.groupby(by)[["n_found", "n_true"]].sum()
        ratio = _ratio(sums["n_found"].to_numpy(float), sums["n_true"].to_numpy(float))
        table = table.join(pd.Series(ratio, index=sums.index, name=f"recall_{kind}"), on=by)
    if points is not None and len(points):
        for target in FP_TARGETS:
            at = points.loc[points["fp_target"] == target, ["method", "minimum_iou", "recall"]]
            table = table.merge(
                at.rename(columns={"recall": f"recall_at_{target:g}"}),
                on=["method", "minimum_iou"],
                how="left",
            )
    grid = _method_grid(primary, {"minimum_iou": MATCH_IOU_LEVELS})
    table = _per_method(table, tables, grid, ("n_reference", "n_detected", "n_matched"))
    rank = table.groupby(["minimum_iou", "primary_expression"])["recall"].rank(
        ascending=False, method="min"
    )
    table.insert(table.columns.get_loc("primary_expression"), "rank", rank.astype("Int64"))
    return table


def order_changes(sensitivity: pd.DataFrame) -> pd.DataFrame:
    """The methods whose rank by recall moves away from IoU 0.

    Parameters
    ----------
    sensitivity : pandas.DataFrame
        ``matching_sensitivity``' table.

    Returns
    -------
    changes : pandas.DataFrame
        One row per method whose rank at some other level differs from its
        rank at IoU 0: ``method``, ``setting``, ``primary_expression``, then
        ``rank_<level>`` at each level.
    """
    ranks = sensitivity.pivot_table(
        index=["method", "setting", "primary_expression"],
        columns="minimum_iou",
        values="rank",
        aggfunc="first",
    )
    if ranks.empty:
        return pd.DataFrame(columns=["method", "setting", "primary_expression"])
    moved = (ranks.ne(ranks[0.0], axis=0)).any(axis=1)
    ranks.columns = [f"rank_{level:g}" for level in ranks.columns]
    return ranks[moved].reset_index()


# Rates and participation

STATES = ("rest", "running")
SELECTIONS = {"all": "n_active_units", "principal": "n_active_principal"}


def session_bouts(sessions: pd.DataFrame) -> dict[str, np.ndarray[Any, Any]]:
    """Each session's running bouts, drawn again from its replicate's seed.

    ``running_schedule`` with the schedule stage's seed (``stage_seeds``),
    as the simulation drew it, in seconds from the session's first sample
    (a run's sessions start at 0).

    Parameters
    ----------
    sessions : pandas.DataFrame
        ``session_id``, ``replicate``, ``duration_s`` and ``rest_s``.

    Returns
    -------
    bouts : dict of str to ndarray, shape (n_bouts, 2)

    Raises
    ------
    ValueError
        A session's rest is not its duration less the bouts drawn again, so
        the schedule is not the one the run simulated.
    """
    bouts = {}
    for row in sessions.itertuples(index=False):
        rng = np.random.default_rng(stage_seeds(int(row.replicate))[0])
        drawn = running_schedule(float(row.duration_s), rng)
        rest = float(row.duration_s) - float(np.sum(np.diff(drawn, axis=1)))
        if not np.isclose(rest, float(row.rest_s), rtol=0, atol=1e-6):
            msg = (
                f"{row.session_id}: the running schedule drawn again leaves {rest} s of "
                f"rest, the run recorded {row.rest_s} s."
            )
            raise ValueError(msg)
        bouts[row.session_id] = drawn
    return bouts


def rates_by_state(
    tables: RunTables,
    *,
    bouts: Mapping[str, np.ndarray[Any, Any]] | None = None,
    n_resamples: int = N_RESAMPLES,
) -> pd.DataFrame:
    """Each main method's events per minute at rest and while running.

    An event is placed by its ``peak_time``, else its bounds' midpoint
    (``event_times``); one on a bout's start or end, to the timestamps'
    rounding, is running (``intervals_to_mask``). Beside each rate, the true rate in that state: network
    events per minute of rest (they occur at rest only), and theta bursts, the
    non-events of running, per minute of running.

    Parameters
    ----------
    tables : RunTables
    bouts : mapping of str to ndarray, optional
        Each session's running bouts; default ``session_bouts``.
    n_resamples : int, optional

    Returns
    -------
    rates : pandas.DataFrame
        One row per method, setting and ``state`` (``"rest"``, ``"running"``):
        ``scoring``, ``n_events``, ``minutes`` (pooled over the sessions it
        has scores on), ``rate`` with ``_low`` and ``_high``, ``true_events``
        and ``true_rate`` over the same sessions, ``primary_expression``,
        ``n_sessions``, ``n_failures``.
    """
    bouts = session_bouts(tables.sessions) if bouts is None else bouts
    durations = tables.sessions.set_index("session_id")["duration_s"]
    frames = []
    for session_id, events in tables.events.groupby("session_id", sort=False):
        running = rd.intervals_to_mask(event_times(events), bouts[session_id])
        frames.append(
            events[list(_KEY)].assign(rest=(~running).astype(int), running=running.astype(int))
        )
    placed = _concat(frames, [*_KEY, "rest", "running"])
    counted = placed.groupby(list(_KEY))[["rest", "running"]].sum()
    truth = []
    for session_id, (events, non_events) in tables.truth.items():
        running_minutes = float(np.sum(np.diff(bouts[session_id], axis=1))) / 60
        truth.append(
            {
                "session_id": session_id,
                "rest_minutes": durations[session_id] / 60 - running_minutes,
                "running_minutes": running_minutes,
                "rest_true": events["event_id"].nunique(),
                "running_true": int((non_events["non_event_type"] == "theta_burst").sum()),
            }
        )
    frame = tables.ran.join(counted, on=list(_KEY)).fillna({"rest": 0, "running": 0})
    frame = frame.merge(pd.DataFrame(truth), on="session_id")
    parts = []
    for state in STATES:
        part = frame.assign(
            state=state,
            n_events=frame[state],
            minutes=frame[f"{state}_minutes"],
            true_events=frame[f"{state}_true"],
        )
        parts.append(part[[*_KEY, "state", "n_events", "minutes", "true_events"]])
    long = pd.concat(parts, ignore_index=True)
    by = ["method", "setting", "state"]
    rates = _pooled_ratios(
        long,
        by,
        {"rate": ("n_events", "minutes")},
        ["n_events", "true_events"],
        ["minutes"],
        n_resamples=n_resamples,
    )
    rates["true_rate"] = _ratio(
        rates["true_events"].to_numpy(float), rates["minutes"].to_numpy(float)
    )
    columns = [*by, "n_events", "minutes", "rate", "rate_low", "rate_high"]
    grid = _method_grid(tables.methods, {"state": STATES})
    rates = _per_method(
        rates[[*columns, "true_events", "true_rate"]],
        tables,
        grid,
        ("n_events", "true_events"),
    ).fillna({"minutes": 0.0})
    rates.insert(3, "scoring", rates["method"].map(scoring_rule))
    return rates


def _burst_participants(tables: RunTables) -> pd.DataFrame:
    """Every truth event with a burst: ``session_id``, ``id`` and its latent
    ``n_participants`` (recruited cells, silent ones included)."""
    parts = [
        events.loc[events["expression"] == "burst", ["event_id", "n_participants"]]
        .drop_duplicates("event_id")
        .rename(columns={"event_id": "id"})
        .assign(session_id=session_id)
        for session_id, (events, _) in tables.truth.items()
    ]
    return _concat(parts, ["session_id", "id", "n_participants"])


def _ratio_of_means() -> GroupStatistic:
    """Per group, the mean of the matched events' participants over the
    mean of all events'."""
    columns = ("matched_sum", "matched_n", "all_sum", "all_n")

    def terms(frame: pd.DataFrame) -> dict[str, np.ndarray[Any, Any]]:
        return {column: frame[column].to_numpy(dtype=float) for column in columns}

    def combine(sums: Mapping[str, np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
        matched = _ratio(sums["matched_sum"], sums["matched_n"])
        return _ratio(matched, _ratio(sums["all_sum"], sums["all_n"]))[np.newaxis]

    return _Sums(columns, terms, combine)


def participation_bias(
    tables: RunTables, matches: Matches, *, n_resamples: int = N_RESAMPLES
) -> pd.DataFrame:
    """Which true events each method finds, by their recruited cells.

    Among the truth events with a burst, the latent ``n_participants``
    (recruited cells, some of which fire no spike) of those the method
    matched against its primary expression (IoU 0), against those of all
    of them. Reported on its own, never subtracted from an observed count.
    Interval methods only.

    Parameters
    ----------
    tables : RunTables
    matches : Matches
    n_resamples : int, optional

    Returns
    -------
    bias : pandas.DataFrame
        One row per interval method and setting: ``n_matched_events`` and
        ``n_events`` (truth events with a burst, over the sessions it has
        scores on), ``mean_matched`` and ``mean_all`` (their mean
        ``n_participants``), ``ratio_of_means`` with ``_low`` and ``_high``,
        ``ks_statistic`` (``scipy.stats.ks_2samp`` of the matched events'
        against all events' counts, pooled), ``primary_expression``,
        ``n_sessions``, ``n_failures``.
    """
    from scipy.stats import ks_2samp

    bursts = _burst_participants(tables)
    pairs = _primary_pairs(tables, matches)
    windows = matches.windows[["session_id", "expression", "row", "id"]]
    found = pairs.merge(
        windows,
        left_on=["session_id", "expression", "truth_row"],
        right_on=["session_id", "expression", "row"],
    )[[*_KEY, "id"]].drop_duplicates()
    found = found.merge(bursts, on=["session_id", "id"])
    matched = found.groupby(list(_KEY))["n_participants"].agg(
        matched_sum="sum", matched_n="size"
    )
    every = bursts.groupby("session_id")["n_participants"].agg(all_sum="sum", all_n="size")
    frame = tables.intervals.ran.join(matched, on=list(_KEY)).join(every, on="session_id")
    frame = frame.fillna(0.0)
    by = ["method", "setting"]
    intervals = grouped_intervals(
        frame,
        by,
        _ratio_of_means(),
        ["ratio_of_means"],
        n_resamples=n_resamples,
    )
    rows = []
    for (method, setting), group in frame.groupby(by, sort=True):
        sessions = set(group["session_id"])
        own = found[(found["method"] == method) & (found["setting"] == setting)]
        every_count = bursts.loc[bursts["session_id"].isin(sessions), "n_participants"]
        mine = own.loc[own["session_id"].isin(sessions), "n_participants"]
        rows.append(
            {
                "method": method,
                "setting": setting,
                "n_matched_events": len(mine),
                "n_events": len(every_count),
                "mean_matched": mine.mean() if len(mine) else np.nan,
                "mean_all": every_count.mean() if len(every_count) else np.nan,
                "ks_statistic": (
                    float(ks_2samp(mine, every_count).statistic)
                    if len(mine) and len(every_count)
                    else np.nan
                ),
            }
        )
    bias = pd.DataFrame(
        rows,
        columns=[
            *by,
            "n_matched_events",
            "n_events",
            "mean_matched",
            "mean_all",
            "ks_statistic",
        ],
    ).merge(intervals, on=by)
    columns = [
        *by,
        "n_matched_events",
        "n_events",
        "mean_matched",
        "mean_all",
        "ratio_of_means",
        "ratio_of_means_low",
        "ratio_of_means_high",
        "ks_statistic",
    ]
    grid = _method_grid(tables.intervals.methods)
    return _per_method(bias[columns], tables, grid, ("n_matched_events", "n_events"))


def boundary_effect(
    tables: RunTables, matches: Matches, *, n_resamples: int = N_RESAMPLES
) -> pd.DataFrame:
    """What a method's bounds do to the units counted active in its events.

    For each pair matched against the method's primary expression (IoU 0):
    the units with a spike within the detected bounds minus those within the
    matched truth window of that expression at 10 % (``truth_counts.csv``),
    both counted by ``count_spikes_in_events`` over the same units. Silent
    recruits, interneurons and background spikes count on both sides, so a
    difference is the bounds' alone: zero when they agree.

    Parameters
    ----------
    tables : RunTables
    matches : Matches
    n_resamples : int, optional

    Returns
    -------
    effect : pandas.DataFrame
        One row per interval method, setting and ``selection`` (``"all"``
        units; ``"principal"``, place and pyramidal): ``n_pairs``,
        ``mean_difference`` with ``_low`` and ``_high``,
        ``median_difference``, ``mean_detected`` and ``mean_truth`` (the
        counts), ``primary_expression``, ``n_sessions``, ``n_failures``.
    """
    pairs = _primary_pairs(tables, matches)
    detected = tables.events[[*_KEY, "event_index", *SELECTIONS.values()]]
    pairs = pairs.merge(detected, on=[*_KEY, "event_index"])
    truth = tables.truth_counts.rename(
        columns={column: f"truth_{column}" for column in SELECTIONS.values()}
    )
    pairs = pairs.merge(
        truth,
        left_on=["session_id", "expression", "truth_row"],
        right_on=["session_id", "expression", "row"],
    )
    parts = [
        pairs[[*_KEY]].assign(
            selection=selection,
            detected=pairs[column].to_numpy(float),
            truth=pairs[f"truth_{column}"].to_numpy(float),
            difference=(pairs[column] - pairs[f"truth_{column}"]).to_numpy(float),
        )
        for selection, column in SELECTIONS.items()
    ]
    long = _concat(parts, [*_KEY, "selection", "detected", "truth", "difference"])
    by = ["method", "setting", "selection"]
    intervals = grouped_intervals(
        long,
        by,
        _means("difference"),
        ["mean_difference"],
        n_resamples=n_resamples,
    )
    summary = long.groupby(by).agg(
        n_pairs=("difference", "size"),
        median_difference=("difference", "median"),
        mean_detected=("detected", "mean"),
        mean_truth=("truth", "mean"),
    )
    effect = intervals.join(summary, on=by)
    columns = [
        *by,
        "n_pairs",
        "mean_difference",
        "mean_difference_low",
        "mean_difference_high",
        "median_difference",
        "mean_detected",
        "mean_truth",
    ]
    grid = _method_grid(tables.intervals.methods, {"selection": tuple(SELECTIONS)})
    return _per_method(effect[columns], tables, grid, ("n_pairs",))


# Appendix: every expression

# The appendix's pair medians: (name, the pair column, absolute).
_APPENDIX_MEDIANS = (
    ("median_iou", "iou", False),
    ("median_onset_error", "onset_error_10", False),
    ("median_abs_onset_error", "onset_error_10", True),
    ("median_offset_error", "offset_error_10", False),
    ("median_abs_offset_error", "offset_error_10", True),
)


def appendix_expressions(
    tables: RunTables, matches: Matches, *, n_resamples: int = N_RESAMPLES
) -> pd.DataFrame:
    """Each main method's scores against every expression, not only its primary.

    Matched at IoU 0 against each expression's truth windows at 10 % of the
    peak, as the headline is against the primary one; a point method by
    peak containment (``match_peaks``), with no bound or overlap measure.

    Parameters
    ----------
    tables : RunTables
    matches : Matches
    n_resamples : int, optional

    Returns
    -------
    appendix : pandas.DataFrame
        One row per main method, setting and ``expression`` (``network``,
        ``ripple``, ``sharp_wave``, ``burst``): ``scoring``, ``primary``
        (whether it is the method's primary expression), ``n_reference``,
        ``n_detected``, ``n_matched``, ``minutes`` (outside every network
        window), then ``recall``, ``precision`` and
        ``false_positives_per_minute``, each pooled over sessions with
        ``_low`` and ``_high``; ``n_pairs``, and ``median_iou``,
        ``median_onset_error``, ``median_abs_onset_error``,
        ``median_offset_error`` and ``median_abs_offset_error`` (signed:
        detected minus truth, seconds, at 10 %), each pooled over the pairs
        with ``_low`` and ``_high`` (NaN for a point method); then
        ``primary_expression``, ``n_sessions``, ``n_failures``.
    """
    by = ["method", "setting", "expression"]
    counts = ["n_reference", "n_detected", "n_matched"]
    n_reference = matches.windows.groupby(["session_id", "expression"]).size()
    n_detected = tables.events.groupby(list(_KEY)).size()
    found = matches.pairs[matches.pairs["minimum_iou"] == 0]
    n_matched = found.groupby([*by, "session_id"]).size()
    frame = tables.intervals.ran.merge(
        pd.DataFrame({"expression": EXPRESSION_ORDER}), how="cross"
    )
    frame = frame.join(n_reference.rename("n_reference"), on=["session_id", "expression"])
    frame = frame.join(n_detected.rename("n_detected"), on=list(_KEY))
    frame = frame.join(n_matched.rename("n_matched"), on=[*by, "session_id"])
    frame = _concat(
        [frame.fillna(dict.fromkeys(counts, 0)), matches.points],
        ["session_id", *by, *counts],
    ).astype(dict.fromkeys(counts, int))
    appendix = _detection_rates(frame, by, tables, n_resamples=n_resamples)
    pairs = found.assign(
        **{
            name: np.abs(found[column]) if absolute else found[column]
            for name, column, absolute in _APPENDIX_MEDIANS
        }
    )
    names = [name for name, _, _ in _APPENDIX_MEDIANS]
    medians = grouped_intervals(
        pairs, by, _medians(*names), names, n_resamples=n_resamples
    ).join(pairs.groupby(by).size().rename("n_pairs"), on=by)
    ordered = ["n_pairs", *(f"{n}{part}" for n in names for part in ("", "_low", "_high"))]
    appendix = appendix.merge(medians[[*by, *ordered]], on=by, how="left")
    appendix["n_pairs"] = appendix["n_pairs"].fillna(0)
    grid = _method_grid(tables.methods, {"expression": EXPRESSION_ORDER})
    appendix = _per_method(appendix, tables, grid, counts).fillna({"minutes": 0.0})
    appendix.insert(3, "scoring", appendix["method"].map(scoring_rule))
    appendix.insert(4, "primary", appendix["primary_expression"] == appendix["expression"])
    appendix.loc[appendix["scoring"] == PEAK_CONTAINMENT, "n_pairs"] = np.nan
    return appendix


def expression_curves(
    scores: ConditionScores, expression: str, *, condition: str = REFERENCE_CONDITION
) -> pd.DataFrame:
    """``operating_curves`` against one expression, whatever each method's primary.

    Parameters
    ----------
    scores : ConditionScores
    expression : str
        One of ``network``, ``ripple``, ``sharp_wave``, ``burst``.
    condition : str, optional
        Default the reference condition, the one ``scores.expression_counts``
        holds.

    Returns
    -------
    curves : pandas.DataFrame
        ``operating_curves``' rows and columns against ``expression``
        (``expression`` inserted after ``primary_expression``), without the
        matched pairs' median errors: every setting of every interval
        method, its counts, recall, false positives per minute and
        ``precision``, at every minimum IoU.
    """
    counts = scores.expression_counts
    own = counts[counts["expression"] == expression].drop(columns="expression")
    view = dataclasses.replace(
        scores,
        counts=own.reset_index(drop=True),
        errors=scores.errors.iloc[:0],
    )
    curves = operating_curves(view, condition=condition)
    if curves.empty:
        return curves
    curves = curves.drop(columns=[f"median_{column}" for column in _ERROR_MEASURES])
    curves.insert(6, "expression", expression)
    at = curves.columns.get_loc("false_positives_per_minute")
    precision = _ratio(
        curves["n_matched"].to_numpy(float), curves["n_detected"].to_numpy(float)
    )
    curves.insert(at + 1, "precision", precision)
    return curves


# Figures (matplotlib is imported only inside them)

_FONT = 5
# Seconds to the milliseconds the figures show.
_MS = 1000.0


def _grid(
    table: pd.DataFrame,
    rows: str,
    columns: str,
    value: str,
    row_order: Sequence[str],
    column_order: Sequence[str],
) -> np.ndarray[Any, Any]:
    """``table``'s ``value`` as a (row, column) array in the given orders,
    NaN where no row gives one."""
    grid = np.full((len(row_order), len(column_order)), np.nan)
    row = pd.Index(row_order).get_indexer(table[rows])
    column = pd.Index(column_order).get_indexer(table[columns])
    keep = (row >= 0) & (column >= 0)
    grid[row[keep], column[keep]] = table[value].to_numpy(dtype=float)[keep]
    return grid


def _pair_grid(
    table: pd.DataFrame, value: str, methods: Sequence[str], *, antisymmetric: bool
) -> np.ndarray[Any, Any]:
    """A pair table's ``value`` as a method-by-method array, row A and column
    B; the transpose holds B against A (negated for a difference)."""
    upper = _grid(table, "method_a", "method_b", value, methods, methods)
    lower = _grid(table, "method_b", "method_a", value, methods, methods)
    return np.where(np.isnan(upper), -lower if antisymmetric else lower, upper)


def _heatmap(
    axis: Axes,
    values: np.ndarray[Any, Any],
    rows: Sequence[str],
    columns: Sequence[str],
    title: str,
    **style: Any,
) -> AxesImage:
    image = axis.imshow(values, aspect="auto", interpolation="nearest", **style)
    axis.set_xticks(range(len(columns)), list(columns), rotation=90, fontsize=_FONT)
    axis.set_yticks(range(len(rows)), list(rows), fontsize=_FONT)
    axis.set_title(title, fontsize=8)
    return image


def _boxes(axis: Axes, stats: pd.DataFrame, scale: float = 1.0, color: str = "C0") -> None:
    """Horizontal boxes, one row of ``stats`` per box from the top: whiskers
    from ``q05`` to ``q95``, the box ``q25`` to ``q75``, a tick at the
    median."""
    y = np.arange(len(stats))[::-1]
    axis.hlines(y, stats["q05"] * scale, stats["q95"] * scale, color="0.5", linewidth=0.6)
    axis.barh(
        y,
        (stats["q75"] - stats["q25"]) * scale,
        left=stats["q25"] * scale,
        height=0.6,
        color=color,
        alpha=0.5,
    )
    axis.plot(stats["median"] * scale, y, "|", color="k", markersize=4)
    axis.tick_params(labelsize=_FONT)


def _tall(n_rows: int) -> float:
    """Inches of figure height for ``n_rows`` labelled rows."""
    return 1.5 + 0.12 * n_rows


def plot_detection_profile(profile: pd.DataFrame) -> Figure:
    """``detection_profile``'s recall, method by event type.

    Parameters
    ----------
    profile : pandas.DataFrame
        ``detection_profile``' table.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    methods = list(dict.fromkeys(profile["method"]))
    kinds = list(dict.fromkeys(profile["event_type"]))
    figure, axis = plt.subplots(figsize=(5, _tall(len(methods))))
    image = _heatmap(
        axis,
        _grid(profile, "method", "event_type", "recall", methods, kinds),
        methods,
        kinds,
        "Recall against the network truth, by event type",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    figure.colorbar(image, ax=axis, shrink=0.3)
    return figure


def plot_false_positive_classes(classes: pd.DataFrame) -> Figure:
    """``false_positive_classes``' fractions, stacked per method.

    Parameters
    ----------
    classes : pandas.DataFrame
        ``false_positive_classes``' table.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    methods = list(dict.fromkeys(classes["method"]))
    shown = classes.groupby("label", sort=False)["n_events"].sum()
    labels = list(shown[shown > 0].index)
    fractions = _grid(classes, "method", "label", "fraction", methods, labels)
    figure, axis = plt.subplots(figsize=(7, _tall(len(methods))))
    colors = plt.get_cmap("tab20")(np.arange(len(labels)) % 20)
    y = np.arange(len(methods))[::-1]
    left = np.zeros(len(methods))
    for position, label in enumerate(labels):
        width = np.nan_to_num(fractions[:, position])
        axis.barh(y, width, left=left, color=colors[position], label=label, height=0.8)
        left += width
    axis.set_yticks(y, methods, fontsize=_FONT)
    axis.set_xlabel("fraction of the method's false positives", fontsize=7)
    axis.set_title("What false positives overlap (primary expression, IoU 0)", fontsize=8)
    axis.legend(fontsize=_FONT, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return figure


def _leaf_order(agreement: pd.DataFrame) -> list[str]:
    """The methods of a pair table in its dendrogram's leaf order."""
    from scipy.cluster.hierarchy import leaves_list

    methods = sorted(set(agreement["method_a"]) | set(agreement["method_b"]))
    tree = agreement_linkage(methods, agreement.set_index(["method_a", "method_b"])["jaccard"])
    if not len(tree):
        return methods
    return [methods[leaf] for leaf in leaves_list(tree)]


def plot_pairwise_agreement(agreement: pd.DataFrame) -> Figure:
    """``pairwise_agreement``'s four Jaccard indices, methods in the
    dendrogram's leaf order.

    Parameters
    ----------
    agreement : pandas.DataFrame
        ``pairwise_agreement``' table.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    methods = _leaf_order(agreement)
    size = 2 + 0.09 * len(methods)
    figure, axes = plt.subplots(2, 2, figsize=(2 * size, 2 * size))
    for axis, name in zip(axes.flat, AGREEMENT, strict=True):
        image = _heatmap(
            axis,
            _pair_grid(agreement, name, methods, antisymmetric=False),
            methods,
            methods,
            f"{name} (mean over sessions)",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
    figure.colorbar(image, ax=axes, shrink=0.3)
    return figure


def plot_agreement_dendrogram(dendrogram: pd.DataFrame) -> Figure:
    """``agreement_dendrogram``'s tree.

    Parameters
    ----------
    dendrogram : pandas.DataFrame
        ``agreement_dendrogram``' table.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram as draw

    leaves = dendrogram[dendrogram["left"] < 0]
    merges = dendrogram[dendrogram["left"] >= 0]
    figure, axis = plt.subplots(figsize=(6, _tall(len(leaves))))
    if len(merges):
        tree = merges[["left", "right", "distance", "size"]].to_numpy(dtype=float)
        draw(
            tree,
            labels=list(leaves["method"]),
            orientation="left",
            ax=axis,
            leaf_font_size=_FONT,
            color_threshold=0,
            above_threshold_color="0.3",
        )
    axis.set_xlabel("1 - jaccard (average linkage)", fontsize=7)
    axis.set_title("Methods clustered by agreement against the network truth", fontsize=8)
    return figure


def plot_consensus(table: pd.DataFrame) -> Figure:
    """``consensus``: methods per true event by type, and per group of false
    positives.

    Parameters
    ----------
    table : pandas.DataFrame
        ``consensus``' table.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    true = table[table["kind"] == "true_event"]
    for kind, rows in true.groupby("event_type", sort=False):
        left.plot(rows["n_methods"], rows["fraction"], ".-", label=kind, markersize=3)
    left.set_xlabel("methods that found it", fontsize=7)
    left.set_ylabel("fraction of the type's true events", fontsize=7)
    left.set_title("True events", fontsize=8)
    left.legend(fontsize=_FONT)
    groups = table[table["kind"] == "false_positive_group"]
    right.bar(groups["n_methods"], groups["count"], color="0.4")
    right.set_yscale("log")
    right.set_xlabel("methods a group of overlapping false positives spans", fontsize=7)
    right.set_ylabel("groups", fontsize=7)
    right.set_title("False positives", fontsize=8)
    for axis in (left, right):
        axis.tick_params(labelsize=_FONT)
    return figure


def plot_overlap_quality(quality: pd.DataFrame) -> Figure:
    """``overlap_quality``'s distributions, one panel per measure.

    Parameters
    ----------
    quality : pandas.DataFrame
        ``overlap_quality``' table.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    methods = list(dict.fromkeys(quality["method"]))
    figure, axes = plt.subplots(1, 3, figsize=(10, _tall(len(methods))), sharey=True)
    for axis, measure in zip(axes, OVERLAP_MEASURES, strict=True):
        rows = quality[quality["measure"] == measure].set_index("method").loc[methods]
        _boxes(axis, rows)
        axis.set_xlim(0, 1)
        axis.set_title(measure, fontsize=8)
    axes[0].set_yticks(np.arange(len(methods))[::-1], methods, fontsize=_FONT)
    figure.suptitle("Matched pairs against the primary expression: 5-95 % and IQR", fontsize=8)
    return figure


def plot_boundary_errors(errors: pd.DataFrame) -> Figure:
    """``boundary_errors`` in ms: boxes at 10 % of the peak, the medians at
    25 % (triangle) and 50 % (square).

    Parameters
    ----------
    errors : pandas.DataFrame
        ``boundary_errors``' table.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    keys = errors[["method", "expression"]].drop_duplicates()
    labels = [
        f"{method} ({expression})" for method, expression in keys.itertuples(index=False)
    ]
    panels = [(b, m) for m in ("signed", "absolute") for b in ("onset", "offset")]
    figure, axes = plt.subplots(1, 4, figsize=(13, _tall(len(labels))), sharey=True)
    y = np.arange(len(labels))[::-1]
    for axis, (boundary, measure) in zip(axes, panels, strict=True):
        rows = errors[(errors["boundary"] == boundary) & (errors["measure"] == measure)]
        at = {
            fraction: rows[rows["fraction"] == fraction].merge(keys, how="right")
            for fraction in TRUTH_FRACTIONS
        }
        _boxes(axis, at[TRUTH_FRACTIONS[0]], scale=_MS)
        for fraction, marker in zip(TRUTH_FRACTIONS[1:], ("^", "s"), strict=True):
            axis.plot(at[fraction]["median"] * _MS, y, marker, markersize=2, color="C3")
        if measure == "signed":
            axis.axvline(0, color="0.7", linewidth=0.6)
        shown = "detected - truth" if measure == "signed" else "|detected - truth|"
        axis.set_title(f"{boundary}, {measure} (ms; {shown})", fontsize=8)
    axes[0].set_yticks(y, labels, fontsize=_FONT)
    return figure


def plot_paired_timing(timing: pd.DataFrame) -> Figure:
    """``paired_timing``'s mean paired differences at 10 % of the peak, in ms:
    row A minus column B.

    Parameters
    ----------
    timing : pandas.DataFrame
        ``paired_timing``' table.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    shown = timing[timing["fraction"] == TRUTH_FRACTIONS[0]]
    methods = sorted(set(shown["method_a"]) | set(shown["method_b"]))
    size = 2 + 0.12 * len(methods)
    figure, axes = plt.subplots(2, 2, figsize=(2 * size, 2 * size))
    for axis, stem in zip(
        axes.flat,
        ("onset_signed", "offset_signed", "onset_absolute", "offset_absolute"),
        strict=True,
    ):
        values = _pair_grid(shown, f"{stem}_estimate", methods, antisymmetric=True) * _MS
        limit = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
        image = _heatmap(
            axis,
            values,
            methods,
            methods,
            f"{stem} (ms, A - B)",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        figure.colorbar(image, ax=axis, shrink=0.5)
    return figure


def plot_method_differences(differences: pd.DataFrame) -> Figure:
    """``method_differences``' median start and end differences, in ms:
    row A minus column B.

    Parameters
    ----------
    differences : pandas.DataFrame
        ``method_differences``' table.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    methods = sorted(set(differences["method_a"]) | set(differences["method_b"]))
    size = 2 + 0.09 * len(methods)
    figure, axes = plt.subplots(1, 2, figsize=(2 * size, size))
    for axis, name in zip(axes, DIFFERENCES[:2], strict=True):
        values = _pair_grid(differences, name, methods, antisymmetric=True) * _MS
        limit = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
        image = _heatmap(
            axis,
            values,
            methods,
            methods,
            f"{name} (ms, A - B)",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        figure.colorbar(image, ax=axis, shrink=0.5)
    return figure


def plot_error_correlations(correlations: pd.DataFrame) -> Figure:
    """``error_correlations``' mean correlations, method by method.

    Parameters
    ----------
    correlations : pandas.DataFrame
        ``error_correlations``' table.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    methods = sorted(set(correlations["method_a"]) | set(correlations["method_b"]))
    size = 2 + 0.09 * len(methods)
    figure, axes = plt.subplots(1, 2, figsize=(2 * size, size))
    for axis, name in zip(axes, CORRELATIONS, strict=True):
        image = _heatmap(
            axis,
            _pair_grid(correlations, name, methods, antisymmetric=False),
            methods,
            methods,
            name,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
    figure.colorbar(image, ax=axes, shrink=0.5)
    return figure


def plot_splits_and_merges(rates: pd.DataFrame) -> Figure:
    """``splits_and_merges``' rates with their intervals, overall and on
    doublets.

    Parameters
    ----------
    rates : pandas.DataFrame
        ``splits_and_merges``' table.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    methods = list(dict.fromkeys(rates["method"]))
    figure, axes = plt.subplots(1, 2, figsize=(9, _tall(len(methods))), sharey=True)
    y = np.arange(len(methods))[::-1]
    for axis, rate in zip(axes, ("split_rate", "merge_rate"), strict=True):
        for offset, (subset, color) in enumerate((("all", "C0"), (DOUBLET, "C1"))):
            rows = rates[rates["subset"] == subset].set_index("method").loc[methods]
            error = np.abs(
                rows[[f"{rate}_low", f"{rate}_high"]].to_numpy().T - rows[rate].to_numpy()
            )
            axis.errorbar(
                rows[rate],
                y + 0.2 * offset,
                xerr=error,
                fmt=".",
                color=color,
                label=subset,
                markersize=3,
                elinewidth=0.6,
            )
        axis.set_title(rate.replace("_", " "), fontsize=8)
        axis.tick_params(labelsize=_FONT)
    axes[0].set_yticks(y, methods, fontsize=_FONT)
    axes[1].legend(fontsize=_FONT)
    return figure


def _dots(
    axis: Axes,
    rows: pd.DataFrame,
    column: str,
    y: np.ndarray[Any, Any],
    *,
    scale: float = 1.0,
    **style: Any,
) -> None:
    """``column`` of each row at height ``y``, with its ``_low``-``_high``
    interval as a horizontal bar."""
    value = rows[column].to_numpy(dtype=float) * scale
    bounds = rows[[f"{column}_low", f"{column}_high"]].to_numpy(dtype=float).T * scale
    error = np.abs(np.nan_to_num(bounds - value, nan=0.0))
    axis.errorbar(value, y, xerr=error, fmt=".", markersize=3, elinewidth=0.6, **style)
    axis.tick_params(labelsize=_FONT)


def plot_point_inventories(points: pd.DataFrame) -> Figure:
    """``point_inventories``' recall, precision and false positives per minute.

    Parameters
    ----------
    points : pandas.DataFrame
        ``point_inventories``' table.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    columns = ("recall", "precision", "false_positives_per_minute")
    figure, axes = plt.subplots(1, 3, figsize=(9, _tall(len(points))), sharey=True)
    y = np.arange(len(points))[::-1]
    for axis, column in zip(axes, columns, strict=True):
        _dots(axis, points, column, y, color="C0")
        axis.set_title(f"{column.replace('_', ' ')} (peak containment)", fontsize=8)
    axes[0].set_yticks(y, list(points["method"]), fontsize=_FONT)
    return figure


_EXPRESSION_COLORS = {"ripple": "C0", "sharp_wave": "C3", "burst": "C2", "network": "C7"}


def plot_operating_curves(curves: pd.DataFrame) -> Figure:
    """``operating_curves`` at IoU 0: one panel per primary expression, each
    detector's sweep a line (its default an open circle), each recipe a grey
    dot, the target rates dotted.

    Parameters
    ----------
    curves : pandas.DataFrame
        ``operating_curves``' table.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    shown = curves[curves["minimum_iou"] == 0]
    expressions = [e for e in EXPRESSION_ORDER if e in set(shown["primary_expression"])]
    figure, axes = plt.subplots(
        1, len(expressions), figsize=(4 * len(expressions), 4), squeeze=False
    )
    for axis, expression in zip(axes[0], expressions, strict=True):
        own = shown[shown["primary_expression"] == expression]
        floor = 0.5 / own["minutes"].max()
        fp = np.maximum(own["false_positives_per_minute"], floor)
        recipes = own["kind"] == "recipe"
        axis.scatter(
            fp[recipes], own.loc[recipes, "recall"], s=6, color="0.6", label="recipes"
        )
        for position, (method, rows) in enumerate(
            own[own["kind"] != "recipe"].groupby("method", sort=True)
        ):
            color = f"C{position % 10}"
            sweep = rows[rows["kind"] == "sweep"].sort_values("threshold")
            axis.plot(
                np.maximum(sweep["false_positives_per_minute"], floor),
                sweep["recall"],
                ".-",
                color=color,
                markersize=3,
                linewidth=0.8,
                label=method,
            )
            default = rows[rows["kind"] == "default"]
            axis.scatter(
                np.maximum(default["false_positives_per_minute"], floor),
                default["recall"],
                s=18,
                facecolors="none",
                edgecolors=color,
            )
        for target in FP_TARGETS:
            axis.axvline(target, color="0.8", linestyle=":", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_ylim(0, 1)
        axis.set_xlabel("false positives per minute (0 drawn at the resolution)", fontsize=7)
        axis.set_ylabel(f"recall against {expression}", fontsize=7)
        axis.set_title(f"primary expression {expression}", fontsize=8)
        axis.tick_params(labelsize=_FONT)
        axis.legend(fontsize=_FONT, loc="lower right")
    return figure


def plot_operating_points(points: pd.DataFrame) -> Figure:
    """``operating_points``: each detector's recall at the targets, one panel
    per minimum IoU.

    Parameters
    ----------
    points : pandas.DataFrame
        ``operating_points``' table.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    levels = list(dict.fromkeys(points["minimum_iou"]))
    methods = sorted(set(points["method"]))
    figure, axes = plt.subplots(1, len(levels), figsize=(4 * len(levels), 3.5), sharey=True)
    for axis, level in zip(np.atleast_1d(axes), levels, strict=True):
        own = points[points["minimum_iou"] == level]
        for position, method in enumerate(methods):
            rows = own[own["method"] == method]
            x = np.log2(rows["fp_target"].to_numpy(float)) + 0.04 * (
                position - len(methods) / 2
            )
            error = np.abs(
                np.nan_to_num(
                    rows[["recall_low", "recall_high"]].to_numpy(float).T
                    - rows["recall"].to_numpy(float),
                    nan=0.0,
                )
            )
            axis.errorbar(
                x,
                rows["recall"],
                yerr=error,
                fmt=".-",
                markersize=3,
                linewidth=0.6,
                label=method,
            )
        axis.set_xticks(np.log2(FP_TARGETS), [f"{t:g}" for t in FP_TARGETS], fontsize=_FONT)
        axis.set_xlabel("false positives per minute", fontsize=7)
        axis.set_title(f"recall at the target, minimum IoU {level:g}", fontsize=8)
        axis.set_ylim(0, 1)
        axis.tick_params(labelsize=_FONT)
    np.atleast_1d(axes)[-1].legend(fontsize=_FONT, loc="lower right")
    return figure


def plot_held_out_thresholds(thresholds: pd.DataFrame) -> Figure:
    """``held_out_thresholds``: held-out recall (with its interval) against
    the calibration recall of the chosen setting, per target.

    Parameters
    ----------
    thresholds : pandas.DataFrame
        ``held_out_thresholds``' table.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    targets = list(dict.fromkeys(thresholds["fp_target"]))
    methods = sorted(set(thresholds["method"]))
    figure, axes = plt.subplots(
        1, len(targets), figsize=(3 * len(targets), _tall(len(methods))), sharey=True
    )
    y = np.arange(len(methods))[::-1]
    for axis, target in zip(np.atleast_1d(axes), targets, strict=True):
        rows = thresholds[thresholds["fp_target"] == target].set_index("method").loc[methods]
        _dots(axis, rows, "recall", y, color="C0", label="held out")
        axis.scatter(
            rows["calibration_recall"],
            y,
            s=12,
            facecolors="none",
            edgecolors="C1",
            label="calibration",
        )
        axis.set_title(f"{target:g} per minute", fontsize=8)
        axis.set_xlim(0, 1)
    np.atleast_1d(axes)[0].set_yticks(y, methods, fontsize=_FONT)
    np.atleast_1d(axes)[-1].legend(fontsize=_FONT)
    return figure


def plot_robustness(table: pd.DataFrame) -> Figure:
    """One measure of ``robustness``: a panel per factor, the measure against
    the factor's levels, one line per main setting (coloured by primary
    expression), the reference level in place.

    Parameters
    ----------
    table : pandas.DataFrame
        One measure's rows of ``robustness``.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    factors = list(dict.fromkeys(table["factor"]))
    columns = 6
    rows_of_panels = int(np.ceil(len(factors) / columns))
    figure, axes = plt.subplots(
        rows_of_panels, columns, figsize=(2.4 * columns, 2.2 * rows_of_panels), squeeze=False
    )
    measure = str(table["measure"].iloc[0])
    scale = _MS if "error" in measure else 1.0
    for axis, factor in zip(axes.flat, factors, strict=False):
        own = table[table["factor"] == factor]
        levels = list(dict.fromkeys(own["level"]))
        for (_, _), rows in own.groupby(["method", "setting"], sort=False):
            x = [levels.index(level) for level in rows["level"]]
            axis.plot(
                x,
                rows["value"] * scale,
                "-",
                color=_EXPRESSION_COLORS.get(str(rows["primary_expression"].iloc[0]), "0.5"),
                linewidth=0.5,
                alpha=0.6,
            )
        axis.set_xticks(range(len(levels)), levels, fontsize=_FONT)
        axis.set_title(factor, fontsize=7)
        axis.tick_params(labelsize=_FONT)
    for axis in list(axes.flat)[len(factors) :]:
        axis.set_visible(False)
    unit = " (ms, detected - truth)" if scale != 1.0 else ""
    figure.suptitle(
        f"{measure}{unit} by factor level; "
        "blue ripple, grey network, green burst, red sharp wave",
        fontsize=8,
    )
    figure.tight_layout()
    return figure


def plot_robustness_crossed(table: pd.DataFrame) -> Figure:
    """One measure of ``robustness_crossed``: per crossed pair, the change
    from the reference in every cell, a row per main setting.

    Parameters
    ----------
    table : pandas.DataFrame
        One measure's rows of ``robustness_crossed``.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    pairs = list(dict.fromkeys(table["factors"]))
    measure = str(table["measure"].iloc[0])
    scale = _MS if "error" in measure else 1.0
    methods = list(dict.fromkeys(table["method"] + " (" + table["setting"] + ")"))
    figure, axes = plt.subplots(
        1, len(pairs), figsize=(5 * len(pairs), _tall(len(methods))), squeeze=False
    )
    for axis, pair in zip(axes[0], pairs, strict=True):
        own = table[table["factors"] == pair].assign(
            row=lambda f: f["method"] + " (" + f["setting"] + ")",
            cell=lambda f: f["level_1"] + " / " + f["level_2"],
        )
        cells = list(dict.fromkeys(own["cell"]))
        values = _grid(
            own.assign(change=own["change"] * scale), "row", "cell", "change", methods, cells
        )
        limit = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
        image = _heatmap(
            axis,
            values,
            methods,
            cells,
            f"{pair}: {measure} change from reference",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        figure.colorbar(image, ax=axis, shrink=0.3)
    return figure


def plot_rates_by_state(rates: pd.DataFrame) -> Figure:
    """``rates_by_state``: each method's rate at rest and while running, the
    true rates as vertical lines.

    Parameters
    ----------
    rates : pandas.DataFrame
        ``rates_by_state``' table.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    methods = list(dict.fromkeys(rates["method"]))
    figure, axes = plt.subplots(1, 2, figsize=(9, _tall(len(methods))), sharey=True)
    y = np.arange(len(methods))[::-1]
    for axis, state in zip(axes, STATES, strict=True):
        rows = rates[rates["state"] == state].set_index("method").loc[methods]
        _dots(axis, rows, "rate", y, color="C0")
        axis.axvline(rows["true_rate"].median(), color="C3", linewidth=0.8)
        axis.set_title(f"events per minute, {state} (red: true)", fontsize=8)
    axes[0].set_yticks(y, methods, fontsize=_FONT)
    return figure


def plot_participation_bias(bias: pd.DataFrame) -> Figure:
    """``participation_bias``' ratio of means, per method, 1 marked.

    Parameters
    ----------
    bias : pandas.DataFrame
        ``participation_bias``' table.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5, _tall(len(bias))))
    y = np.arange(len(bias))[::-1]
    _dots(axis, bias, "ratio_of_means", y, color="C0")
    axis.axvline(1.0, color="0.6", linewidth=0.8)
    axis.set_yticks(y, list(bias["method"]), fontsize=_FONT)
    axis.set_title("Recruited cells of matched events over all events' (mean)", fontsize=8)
    return figure


def plot_boundary_effect(effect: pd.DataFrame) -> Figure:
    """``boundary_effect``' mean differences, per method and unit selection.

    Parameters
    ----------
    effect : pandas.DataFrame
        ``boundary_effect``' table.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    methods = list(dict.fromkeys(effect["method"]))
    figure, axes = plt.subplots(1, 2, figsize=(9, _tall(len(methods))), sharey=True)
    y = np.arange(len(methods))[::-1]
    for axis, selection in zip(axes, SELECTIONS, strict=True):
        rows = effect[effect["selection"] == selection].set_index("method").loc[methods]
        _dots(axis, rows, "mean_difference", y, color="C0")
        axis.axvline(0.0, color="0.6", linewidth=0.8)
        axis.set_title(f"{selection} units active: detected bounds - truth window", fontsize=8)
    axes[0].set_yticks(y, methods, fontsize=_FONT)
    return figure


def plot_matching_sensitivity(sensitivity: pd.DataFrame) -> Figure:
    """``matching_sensitivity``: recall and precision at each minimum IoU.

    Parameters
    ----------
    sensitivity : pandas.DataFrame
        ``matching_sensitivity``' table.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    methods = list(dict.fromkeys(sensitivity["method"]))
    figure, axes = plt.subplots(1, 2, figsize=(9, _tall(len(methods))), sharey=True)
    y = np.arange(len(methods))[::-1]
    for axis, column in zip(axes, ("recall", "precision"), strict=True):
        for position, level in enumerate(dict.fromkeys(sensitivity["minimum_iou"])):
            rows = (
                sensitivity[sensitivity["minimum_iou"] == level]
                .set_index("method")
                .loc[methods]
            )
            _dots(
                axis,
                rows,
                column,
                y + 0.2 * position,
                color=f"C{position}",
                label=f"IoU {level:g}",
            )
        axis.set_title(column, fontsize=8)
        axis.set_xlim(0, 1)
    axes[0].set_yticks(y, methods, fontsize=_FONT)
    axes[1].legend(fontsize=_FONT)
    return figure


def plot_model_sensitivity(changes: pd.DataFrame) -> Figure:
    """``model_sensitivity``: per alternative model, each main setting's
    change in recall and each detector's at 1 per minute (triangles), with
    intervals.

    Parameters
    ----------
    changes : pandas.DataFrame
        ``model_sensitivity``' table.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    alternatives = list(dict.fromkeys(changes["alternative"]))
    recall = changes[changes["measure"] == "recall"]
    methods = list(dict.fromkeys(recall["method"] + " (" + recall["setting"] + ")"))
    figure, axes = plt.subplots(
        1,
        len(alternatives),
        figsize=(2.6 * len(alternatives), _tall(len(methods))),
        sharey=True,
        squeeze=False,
    )
    y = np.arange(len(methods))[::-1]
    for axis, alternative in zip(axes[0], alternatives, strict=True):
        own = recall[recall["alternative"] == alternative]
        own = own.set_index(own["method"] + " (" + own["setting"] + ")").reindex(methods)
        _dots(axis, own, "change", y, color="C0")
        at_one = changes[
            (changes["alternative"] == alternative)
            & (changes["measure"] == "recall_at_fp")
            & (changes["fp_target"] == 1.0)
        ].set_index("method")
        rows = [
            methods.index(f"{m} (default)")
            for m in at_one.index
            if f"{m} (default)" in methods
        ]
        shown = at_one.loc[[m for m in at_one.index if f"{m} (default)" in methods]]
        axis.scatter(shown["change"], y[rows] + 0.3, marker="^", s=8, color="C1")
        axis.axvline(0.0, color="0.6", linewidth=0.8)
        axis.set_title(f"{alternative}\nrecall change", fontsize=7)
    axes[0][0].set_yticks(y, methods, fontsize=_FONT)
    return figure


# Candidate trends

# The tables candidate_trends and summary.md read, by name.
ROBUSTNESS_RECALL = "robustness_recall"
MODEL_CHANGES = "model_sensitivity"
MODEL_ORDERS = "model_sensitivity_orders"
MATCHING = "matching_sensitivity"
PARTICIPATION_BIAS = "participation_bias"
BOUNDARY_EFFECT = "boundary_effect"
OPERATING_POINTS = "operating_points"
OPERATING_DIFFERENCES = "operating_differences"
# The candidates' file stem, and what is written by hand in a results
# directory, kept when it is rebuilt.
CANDIDATES = "candidate_trends"
TRENDS = "trends.md"
SPOT_CHECKS = "spot_checks"
# Names no analysis may take: the command's own steps and files.
RESERVED = ("load", "match", "scores", CANDIDATES)

TREND_COLUMNS = (
    "kind",
    "statement",
    "source",
    "method",
    "scoring",
    "condition_id",
    "value",
    "low",
    "high",
    "p",
    "spot_condition",
    "spot_method_a",
    "spot_setting_a",
    "spot_method_b",
    "spot_setting_b",
    "spot_selection",
)
# Candidates of each kind kept, largest first.
_TRENDS_PER_KIND = 12


def _excludes(frame: pd.DataFrame, low: str, high: str, value: float = 0.0) -> pd.Series:
    """Rows whose interval lies wholly on one side of ``value``."""
    return (frame[low] > value) | (frame[high] < value)


def _scored(scoring: str) -> str:
    """The words a statement about a point method adds: its scoring rule."""
    return " by peak containment" if scoring == PEAK_CONTAINMENT else ""


def _spot(
    condition: str,
    selection: str,
    method_a: str,
    setting_a: str,
    method_b: str = "",
    setting_b: str = "",
) -> dict[str, str]:
    """A trend's spot-check columns: where to look, one method and its
    setting per column."""
    return {
        "spot_condition": condition,
        "spot_method_a": method_a,
        "spot_setting_a": setting_a,
        "spot_method_b": method_b,
        "spot_setting_b": setting_b,
        "spot_selection": selection,
    }


def candidate_trends(results: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Candidate trend statements, each with the rows that support it.

    Not conclusions: each is a pattern in the tables worth stating only after
    its underlying events have been looked at (``spot_check``), and after
    checking that it does not come from failures, empty sweeps or a unit
    error. Each names where to look, for ``select_events``: a condition, one
    or two methods each with its setting (a main setting's own; for a trend
    at a false-positive target, the swept setting whose rate is nearest the
    target), and which events (``"missed"`` or ``"found"`` truth events of
    the first method's primary expression, or its ``"false_positive"``
    events).

    Parameters
    ----------
    results : mapping of str to pandas.DataFrame
        The analyses' tables by name, as ``analyze_run`` writes them.

    Returns
    -------
    trends : pandas.DataFrame
        ``TREND_COLUMNS``: ``kind``, a templated ``statement``, its
        ``source`` table, the ``method`` it is about and its ``scoring``
        (a point method's statement says it is by peak containment), the
        ``condition_id``, the ``value`` with its interval and p-value where
        the table has them, and the spot check to draw
        (``spot_condition``, ``spot_method_a`` and ``spot_setting_a``,
        ``spot_method_b`` and ``spot_setting_b`` (``""`` for one method),
        ``spot_selection``). Largest effects first within a kind.
    """
    rows: list[dict[str, Any]] = []

    def add(
        frame: pd.DataFrame,
        order: pd.Series,
        make: Callable[[Any], dict[str, Any]],
        per: str | None = None,
    ) -> None:
        """The largest of ``order`` first; with ``per``, the largest of each
        value of that column first, so one condition cannot fill the list."""
        ranked = frame.assign(_order=order.to_numpy()).sort_values(
            "_order", ascending=False, kind="stable"
        )
        if per is not None:
            rank = ranked.groupby(per, sort=False).cumcount()
            ranked = ranked.assign(_rank=rank.to_numpy()).sort_values(
                ["_rank", "_order"], ascending=[True, False], kind="stable"
            )
        rows.extend(make(row) for row in ranked.head(_TRENDS_PER_KIND).itertuples(index=False))

    robust = results.get(ROBUSTNESS_RECALL, pd.DataFrame())
    if len(robust):
        moved = robust[
            (robust["level"] != REFERENCE_LEVEL)
            & _excludes(robust, "change_low", "change_high")
        ]
        add(
            moved,
            moved["change"].abs(),
            lambda r: {
                "kind": "robustness",
                "statement": (
                    f"{r.method}'s recall against {r.primary_expression}"
                    f"{_scored(r.scoring)} changes by {r.change:+.3f} ({r.change_low:+.3f}, "
                    f"{r.change_high:+.3f}) from the reference to {r.condition_id}."
                ),
                "source": ROBUSTNESS_RECALL,
                "method": r.method,
                "scoring": r.scoring,
                "condition_id": r.condition_id,
                "value": r.change,
                "low": r.change_low,
                "high": r.change_high,
                "p": r.change_p,
                **_spot(
                    r.condition_id, "missed" if r.change < 0 else "found", r.method, r.setting
                ),
            },
            per="condition_id",
        )
    orders = results.get(MODEL_ORDERS, pd.DataFrame())
    if len(orders):
        flipped = orders[orders["supported"] & orders["reversed"]]
        add(
            flipped,
            flipped["p_reversed"]
            + (flipped["alternative_difference"] - flipped["reference_difference"]).abs(),
            lambda r: {
                "kind": "model_order_reversal",
                "statement": (
                    f"At {r.fp_target:g} false positives a minute, {r.method_a} minus "
                    f"{r.method_b} in recall is {r.reference_difference:+.3f} in the "
                    f"reference and {r.alternative_difference:+.3f} under {r.alternative} "
                    f"(reversed in {r.p_reversed:.0%} of resamples)."
                ),
                "source": MODEL_ORDERS,
                "method": f"{r.method_a} {r.method_b}",
                "scoring": INTERVAL,
                "condition_id": r.alternative,
                "value": r.alternative_difference,
                "low": r.alternative_low,
                "high": r.alternative_high,
                "p": np.nan,
                **_spot(
                    r.alternative, "missed", r.method_a, r.setting_a, r.method_b, r.setting_b
                ),
            },
        )
    model = results.get(MODEL_CHANGES, pd.DataFrame())
    if len(model):
        moved = model[
            (model["measure"] == "recall")
            & (model["status"] == "compared")
            & _excludes(model, "change_low", "change_high")
        ]
        add(
            moved,
            moved["change"].abs(),
            lambda r: {
                "kind": "model_change",
                "statement": (
                    f"{r.method}'s recall{_scored(r.scoring)} changes by {r.change:+.3f} "
                    f"({r.change_low:+.3f}, {r.change_high:+.3f}) under {r.alternative}."
                ),
                "source": MODEL_CHANGES,
                "method": r.method,
                "scoring": r.scoring,
                "condition_id": r.alternative,
                "value": r.change,
                "low": r.change_low,
                "high": r.change_high,
                "p": r.change_p,
                **_spot(
                    r.alternative, "missed" if r.change < 0 else "found", r.method, r.setting
                ),
            },
            per="alternative",
        )
    sensitivity = results.get(MATCHING, pd.DataFrame())
    if len(sensitivity):
        ranks = order_changes(sensitivity)
        last = [column for column in ranks.columns if column.startswith("rank_")][-1:]
        if last:
            level = last[0].removeprefix("rank_")
            ranks = ranks.rename(columns={last[0]: "rank_last"})
            move = (ranks["rank_last"] - ranks["rank_0"]).abs()
            moved = ranks[move >= 3]
            add(
                moved,
                move[move >= 3],
                lambda r: {
                    "kind": "matching_rank",
                    "statement": (
                        f"{r.method}'s rank by recall among {r.primary_expression} methods "
                        f"moves from {r.rank_0} at IoU 0 to {r.rank_last} at {level} "
                        "(descriptive: ranks carry no interval or test)."
                    ),
                    "source": MATCHING,
                    "method": r.method,
                    "scoring": INTERVAL,
                    "condition_id": REFERENCE_CONDITION,
                    "value": float(r.rank_last - r.rank_0),
                    "low": np.nan,
                    "high": np.nan,
                    "p": np.nan,
                    **_spot(REFERENCE_CONDITION, "found", r.method, r.setting),
                },
            )
    bias = results.get(PARTICIPATION_BIAS, pd.DataFrame())
    if len(bias):
        biased = bias[_excludes(bias, "ratio_of_means_low", "ratio_of_means_high", 1.0)]
        add(
            biased,
            np.log(biased["ratio_of_means"]).abs(),
            lambda r: {
                "kind": "participation_bias",
                "statement": (
                    f"The true events {r.method} finds recruit {r.ratio_of_means:.2f} "
                    f"({r.ratio_of_means_low:.2f}, {r.ratio_of_means_high:.2f}) times as "
                    "many cells on average as all true events."
                ),
                "source": PARTICIPATION_BIAS,
                "method": r.method,
                "scoring": INTERVAL,
                "condition_id": REFERENCE_CONDITION,
                "value": r.ratio_of_means,
                "low": r.ratio_of_means_low,
                "high": r.ratio_of_means_high,
                "p": np.nan,
                **_spot(REFERENCE_CONDITION, "missed", r.method, r.setting),
            },
        )
    effect = results.get(BOUNDARY_EFFECT, pd.DataFrame())
    if len(effect):
        shifted = effect[
            (effect["selection"] == "principal")
            & _excludes(effect, "mean_difference_low", "mean_difference_high")
        ]
        add(
            shifted,
            shifted["mean_difference"].abs(),
            lambda r: {
                "kind": "boundary_effect",
                "statement": (
                    f"{r.method}'s bounds change the principal units counted active in a "
                    f"true event by {r.mean_difference:+.2f} ({r.mean_difference_low:+.2f}, "
                    f"{r.mean_difference_high:+.2f}) on average."
                ),
                "source": BOUNDARY_EFFECT,
                "method": r.method,
                "scoring": INTERVAL,
                "condition_id": REFERENCE_CONDITION,
                "value": r.mean_difference,
                "low": r.mean_difference_low,
                "high": r.mean_difference_high,
                "p": np.nan,
                **_spot(REFERENCE_CONDITION, "found", r.method, r.setting),
            },
        )
    points = results.get(OPERATING_POINTS, pd.DataFrame())
    differences = results.get(OPERATING_DIFFERENCES, pd.DataFrame())
    if len(points):
        at_one = points[(points["minimum_iou"] == 0) & (points["fp_target"] == 1.0)]
        for expression, own in at_one.groupby("primary_expression", sort=True):
            reached = own.dropna(subset=["recall"]).sort_values(
                "recall", ascending=False, kind="stable"
            )
            if len(reached) < 2:
                continue
            first, second = reached.iloc[0], reached.iloc[1]
            unreached = sorted(own.loc[own["recall"].isna(), "method"])
            value = low = high = p = np.nan
            n_paired = 0
            settings = {first.method: "", second.method: ""}
            if len(differences):
                pair = differences[
                    (differences["fp_target"] == 1.0)
                    & differences["method_a"].isin([first.method, second.method])
                    & differences["method_b"].isin([first.method, second.method])
                ]
                if len(pair):
                    found = pair.iloc[0]
                    # the leader minus the next, whichever is first by name
                    sign = 1.0 if found.method_a == first.method else -1.0
                    value = sign * found.difference
                    low, high = sorted(
                        (sign * found.difference_low, sign * found.difference_high)
                    )
                    p, n_paired = found.difference_p, int(found.n_paired)
                    settings = {
                        found.method_a: found.setting_a,
                        found.method_b: found.setting_b,
                    }
            statement = (
                f"At 1 false positive a minute against {expression}, {first.method} has "
                f"the highest recall, {first.recall:.3f} ({first.recall_low:.3f}, "
                f"{first.recall_high:.3f}); next {second.method}, {second.recall:.3f} "
                f"({second.recall_low:.3f}, {second.recall_high:.3f}); {first.method} "
                f"minus {second.method} {value:+.3f} ({low:+.3f}, {high:+.3f}), sign-flip "
                f"p {p:.3g} over {n_paired} sessions, paired."
            )
            if unreached:
                verb = "does" if len(unreached) == 1 else "do"
                statement += (
                    f" {', '.join(unreached)} {verb} not reach 1 false positive a minute."
                )
            rows.append(
                {
                    "kind": "operating_order",
                    "statement": statement,
                    "source": OPERATING_DIFFERENCES,
                    "method": first.method,
                    "scoring": INTERVAL,
                    "condition_id": REFERENCE_CONDITION,
                    "value": value,
                    "low": low,
                    "high": high,
                    "p": p,
                    **_spot(
                        REFERENCE_CONDITION,
                        "missed",
                        first.method,
                        settings[first.method],
                        second.method,
                        settings[second.method],
                    ),
                }
            )
    return pd.DataFrame(rows, columns=list(TREND_COLUMNS))


# Spot checks

SPOT_SELECTIONS = ("missed", "found", "false_positive")
SPOT_COLUMNS = ("session_id", "start_time", "end_time", "label")
# Events drawn per spot check.
SPOT_EVENTS = 6


def select_from(
    tables: RunTables,
    method: str,
    setting: str,
    selection: str,
    *,
    event_type: str | None = None,
) -> pd.DataFrame:
    """The events of one kind a method's result holds, in some sessions.

    Parameters
    ----------
    tables : RunTables
        Holding the method and setting.
    method, setting : str
    selection : {"missed", "found", "false_positive"}
        Truth windows of the method's primary expression at 10 % it matched
        no event of (IoU 0; peak containment for a point method), those it
        matched, or its events matching no window.
    event_type : str, optional
        Keep only truth windows of this event type, or false positives
        labelled with it (``label_by_overlap`` against every component and
        non-event; ``"background"`` for none).

    Returns
    -------
    selected : pandas.DataFrame
        ``session_id``, ``start_time``, ``end_time`` and ``label`` (the
        window's event type, or the false positive's label), by session and
        time.

    Raises
    ------
    ValueError
        An unknown selection, or a method and setting the tables lack.
    """
    if selection not in SPOT_SELECTIONS:
        msg = f"selection must be one of {SPOT_SELECTIONS}, got {selection!r}."
        raise ValueError(msg)
    listed = tables.methods[
        (tables.methods["method"] == method) & (tables.methods["setting"] == setting)
    ]
    if listed.empty:
        msg = f"The tables hold no {method} ({setting})."
        raise ValueError(msg)
    expression = listed["primary_expression"].iloc[0]
    ran = tables.ran[(tables.ran["method"] == method) & (tables.ran["setting"] == setting)]
    events = tables.events[
        (tables.events["method"] == method) & (tables.events["setting"] == setting)
    ]
    by_session = dict(tuple(events.groupby("session_id", sort=False)))
    parts = []
    for session_id in ran["session_id"]:
        event_table, non_event_table = tables.truth[session_id]
        truth = rd.truth_windows(event_table, TRUTH_FRACTIONS[0], expression)
        rows = by_session.get(session_id, events.iloc[:0]).sort_values("event_index")
        matched_truth, matched_events = _matched_rows(
            _bounds(truth), rows, method in point_methods()
        )
        if selection == "false_positive":
            unmatched = np.setdiff1d(np.arange(len(rows)), matched_events)
            bounds = _bounds(rows)[unmatched]
            labels = rd.label_by_overlap(bounds, label_windows(event_table, non_event_table))
            chosen = pd.DataFrame(bounds, columns=["start_time", "end_time"]).assign(
                label=labels.to_numpy()
            )
        else:
            matched = np.isin(np.arange(len(truth)), matched_truth)
            keep = matched if selection == "found" else ~matched
            chosen = truth.loc[keep, ["start_time", "end_time"]].assign(
                label=truth.loc[keep, "type"].astype(str).to_numpy()
            )
        parts.append(chosen.assign(session_id=session_id))
    selected = _concat(parts, SPOT_COLUMNS)
    if event_type is not None:
        kept = selected["label"].str.split(":").str[0] == event_type
        selected = selected[kept]
    return selected.reset_index(drop=True)


def select_events(
    run_directory: str | os.PathLike[str],
    condition_id: str,
    method: str,
    setting: str,
    selection: str,
    *,
    event_type: str | None = None,
) -> pd.DataFrame:
    """``select_from`` on one condition of a run, read from its ``combined/``.

    Parameters
    ----------
    run_directory : str or path-like
    condition_id, method, setting, selection : str
    event_type : str, optional

    Returns
    -------
    selected : pandas.DataFrame
    """
    tables = load_run(
        Path(run_directory) / "combined", conditions=[condition_id], settings=[setting]
    )
    return select_from(tables, method, setting, selection, event_type=event_type)


def spot_check(
    run_directory: str | os.PathLike[str],
    results_directory: str | os.PathLike[str],
    name: str,
    selected: pd.DataFrame,
    methods: Sequence[tuple[str, str]],
    *,
    n_events: int = SPOT_EVENTS,
    seed: int = SEED,
) -> Path:
    """Draw some of the events behind a trend, from their sessions simulated again.

    Each session is simulated again from the run's saved parameters and seed
    (``spot_check.load_session``, which refuses a seed that is not the
    replicate's), and each event drawn with the ripple-band and radiatum
    signals, the spikes, the truth windows of every expression at every
    fraction and the methods' events (``spot_check.draw_window``).

    Parameters
    ----------
    run_directory : str or path-like
        ``examples/benchmark/output/<run_name>``.
    results_directory : str or path-like
        ``examples/benchmark/results/<run_name>``; the figure goes to its
        ``spot_checks/<name>.png``, at most ``SIZE_LIMIT`` bytes.
    name : str
        The figure's stem.
    selected : pandas.DataFrame
        ``select_events``' rows.
    methods : sequence of (method, setting)
        Whose events are drawn.
    n_events : int, optional
        Events drawn, chosen at random (``seed``) when there are more.
    seed : int, optional

    Returns
    -------
    path : pathlib.Path

    Raises
    ------
    ValueError
        No event is selected, or the figure would be over ``SIZE_LIMIT``.
    """
    import matplotlib.pyplot as plt
    from spot_check import _MARGIN, draw_window, load_session, session_failures

    if selected.empty:
        msg = f"{name}: no event is selected."
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    picks = np.sort(
        rng.choice(len(selected), size=min(n_events, len(selected)), replace=False)
    )
    chosen = selected.iloc[picks]
    n_rows = (len(chosen) + 1) // 2
    figure = plt.figure(figsize=(12, 4.5 * n_rows), layout="constrained")
    figure.suptitle(f"{name}: {len(chosen)} of {len(selected)} selected events", fontsize=10)
    blocks = figure.subfigures(n_rows, 2, squeeze=False).ravel()
    for block in blocks[len(chosen) :]:
        block.set_visible(False)
    position = 0
    for session_id, rows in chosen.groupby("session_id", sort=False):
        session, found = load_session(Path(run_directory), str(session_id))
        failed = session_failures(Path(run_directory), str(session_id))
        filtered = rd.filter_ripple_band(session.lfps, session.sampling_frequency)
        windows = truth_window_sets(session.events)
        for row in rows.itertuples(index=False):
            draw_window(
                blocks[position],
                session,
                filtered,
                windows,
                found,
                row.start_time - _MARGIN,
                row.end_time + _MARGIN,
                f"{session_id}, {row.label}, {row.start_time:.3f}-{row.end_time:.3f} s",
                methods,
                failed,
            )
            position += 1
    directory = Path(results_directory) / SPOT_CHECKS
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    write_result(path, _png(figure))
    return path


# The command line


@dataclasses.dataclass(frozen=True)
class Inputs:
    """What the analyses read.

    Attributes
    ----------
    tables : RunTables
        The reference condition's main settings, as ``load_run`` reads them
        by default: anything else is refused.
    matches : Matches
        Its sessions matched again at every level of ``MATCH_IOU_LEVELS``.
    scores : ConditionScores
        Every condition, against the primary expressions.
    validation : pandas.DataFrame or None
        ``validation_changes`` of the run's simulator validation report;
        None when it could not be read.
    validation_problem : str
        Why it could not be read (``_validation``), ``""`` when it was.
    n_resamples : int
        The bootstrap resamples of every interval.
    cache : dict
        Tables several analyses share, computed once.
    """

    tables: RunTables
    matches: Matches
    scores: ConditionScores
    validation: pd.DataFrame | None
    validation_problem: str = ""
    n_resamples: int = N_RESAMPLES
    cache: dict[Any, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse tables that are not the reference's main settings alone,
        which every analysis of ``tables`` describes itself as."""
        settings = self.tables.settings
        if (
            self.tables.conditions != (REFERENCE_CONDITION,)
            or settings is None
            or set(settings) != set(MAIN_SETTINGS)
        ):
            msg = (
                "The analyses of the reference need the reference condition's main "
                f"settings alone; the tables hold the conditions {self.tables.conditions} "
                f"and the settings {settings or 'all'}."
            )
            raise ValueError(msg)


AnalysisTable = Callable[[Inputs], pd.DataFrame]


def _of_reference(function: Callable[..., pd.DataFrame]) -> AnalysisTable:
    """An analysis of the reference condition's tables and matches."""
    return lambda inputs: function(
        inputs.tables, inputs.matches, n_resamples=inputs.n_resamples
    )


def _of_scores(
    function: Callable[..., Any], select: Callable[[Any], pd.DataFrame] | None = None
) -> AnalysisTable:
    """An analysis of every condition's scores, computed once for the
    tables read from it (``select``: default the result itself)."""

    def table(inputs: Inputs) -> pd.DataFrame:
        if function not in inputs.cache:
            inputs.cache[function] = function(inputs.scores, n_resamples=inputs.n_resamples)
        found = inputs.cache[function]
        return found if select is None else select(found)

    return table


def _measure_rows(measure: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """A table's rows of one ``measure``."""
    return lambda table: table[table["measure"] == measure].reset_index(drop=True)


def _expression_curves(expression: str) -> AnalysisTable:
    """``expression_curves`` against one expression, of the reference."""
    return lambda inputs: expression_curves(inputs.scores, expression)


def _matching(inputs: Inputs) -> pd.DataFrame:
    points = _of_scores(operating_points)(inputs)
    return matching_sensitivity(
        inputs.tables, inputs.matches, points, n_resamples=inputs.n_resamples
    )


@dataclasses.dataclass(frozen=True)
class Analysis:
    """One analysis: its table, the figure drawn from it, and what each shows.

    Attributes
    ----------
    name : str
        The files' stem: ``<name>.csv`` and ``<name>.png``.
    table : callable
        ``table(inputs)``: the analysis's table from an ``Inputs``.
    description : str
        One sentence on what the table shows, for ``summary.md``.
    figure : callable or None
        ``figure(table)``: a matplotlib Figure; None for no figure.
    figure_description : str
        One sentence on what the figure shows; given exactly when ``figure``
        is.

    Raises
    ------
    ValueError
        A name in ``RESERVED``, or a figure without its description (or a
        description without a figure). ``analyze_run`` refuses two analyses
        of one name.
    """

    name: str
    table: AnalysisTable
    description: str
    figure: Callable[[pd.DataFrame], Figure] | None = None
    figure_description: str = ""

    def __post_init__(self) -> None:
        """Refuse a name the command uses itself, and a figure without its
        sentence for ``summary.md`` (or a sentence without a figure)."""
        if self.name in RESERVED:
            msg = f"{self.name!r} is reserved: the command's own steps are {RESERVED}."
            raise ValueError(msg)
        if self.figure is not None and not self.figure_description:
            msg = f"{self.name}: a figure needs its description for summary.md."
            raise ValueError(msg)
        if self.figure is None and self.figure_description:
            msg = f"{self.name}: a figure description without a figure."
            raise ValueError(msg)


_ROBUSTNESS_NAMES = {
    "recall": "recall",
    "precision": "precision",
    "median_onset_error": "onset",
}

ANALYSES: tuple[Analysis, ...] = (
    Analysis(
        "failures",
        lambda inputs: failure_counts(inputs.tables),
        "Each method's sessions with scores and failures (a missing result, never zero "
        "events), with the first error and its scoring rule.",
    ),
    Analysis(
        "point_inventories",
        _of_reference(point_inventories),
        "Recall, precision and false positives per minute of the methods that return "
        "time points, scored by peak containment and never pooled with interval scores.",
        plot_point_inventories,
        "Those three with their intervals.",
    ),
    Analysis(
        "detection_profile",
        _of_reference(detection_profile),
        "Recall per event type against the network truth, per method, pooled over sessions.",
        plot_detection_profile,
        "Recall as a heatmap, method by event type.",
    ),
    Analysis(
        "false_positive_classes",
        _of_reference(false_positive_classes),
        "What each method's false positives (unmatched against its primary expression) "
        "overlap longest: an event type's component, a non-event or nothing.",
        plot_false_positive_classes,
        "Those fractions stacked per method.",
    ),
    Analysis(
        "pairwise_agreement",
        _of_reference(pairwise_agreement),
        "Agreement of every pair of methods against the network truth: Jaccard of their "
        "events, of their true and of their false events, and of the true events found.",
        plot_pairwise_agreement,
        "The four indices as heatmaps, methods in the dendrogram's order.",
    ),
    Analysis(
        "agreement_dendrogram",
        lambda inputs: agreement_dendrogram(inputs.tables, inputs.matches),
        "Methods clustered by average linkage on 1 - Jaccard.",
        plot_agreement_dendrogram,
        "The dendrogram.",
    ),
    Analysis(
        "consensus",
        lambda inputs: consensus(inputs.tables, inputs.matches),
        "How many methods found each true event, by type, and how many methods each "
        "group of overlapping false positives spans.",
        plot_consensus,
        "Both distributions.",
    ),
    Analysis(
        "overlap_quality",
        _of_reference(overlap_quality),
        "IoU, coverage and temporal precision of each method's matched pairs against its "
        "primary expression, with its recall.",
        plot_overlap_quality,
        "Their distributions per method.",
    ),
    Analysis(
        "boundary_errors",
        _of_reference(boundary_errors),
        "Signed and absolute onset and offset errors (detected minus truth) against the "
        "truth at 10, 25 and 50 % of the peak, each median with its pair count and the "
        "method's recall.",
        plot_boundary_errors,
        "Their distributions at 10 % and the medians at 25 and 50 %, in ms.",
    ),
    *(
        Analysis(
            f"paired_timing_{expression}",
            _of_reference(functools.partial(paired_timing, expression=expression)),
            f"For methods whose primary expression is {expression}, each pair's error "
            "differences (A minus B) on the true events both found, with a sign-flip test.",
            plot_paired_timing,
            "The mean paired differences at 10 % as heatmaps, in ms.",
        )
        for expression in ("ripple", "burst", "network")
    ),
    Analysis(
        "method_differences",
        _of_reference(method_differences),
        "How every pair of methods' matched events differ in start and end (A minus B), "
        "and how often A's comes first.",
        plot_method_differences,
        "The median differences as heatmaps, in ms.",
    ),
    Analysis(
        "error_correlations",
        _of_reference(error_correlations),
        "Spearman correlation of every pair of methods' signed errors on the true events "
        "both found, against their shared primary expression's truth (else the network "
        "truth), with the network truth's beside it.",
        plot_error_correlations,
        "The correlations as heatmaps.",
    ),
    Analysis(
        "splits_and_merges",
        _of_reference(splits_and_merges),
        "How often each method splits a true event or merges several, overall and on "
        "ripple doublets.",
        plot_splits_and_merges,
        "Both rates with their intervals.",
    ),
    Analysis(
        "operating_curves",
        lambda inputs: operating_curves(inputs.scores),
        "Recall against false positives per minute at every setting of each detector's "
        "sweep, its default and each recipe, against the primary expression, at every "
        "minimum IoU, with the matched pairs' median errors.",
        plot_operating_curves,
        "The curves at IoU 0 by primary expression, recipes as grey points on them.",
    ),
    Analysis(
        OPERATING_POINTS,
        _of_scores(operating_points),
        "Each detector's recall and median onset and offset errors read off its sweep at "
        "0.5, 1, 2 and 5 false positives per minute, with intervals; missing where the "
        "curve does not reach the target.",
        plot_operating_points,
        "Recall at each target, per minimum IoU.",
    ),
    Analysis(
        OPERATING_DIFFERENCES,
        _of_scores(operating_differences),
        "For each pair of detectors sharing a primary expression, the difference in recall "
        "(A minus B) at each target rate, paired by session, with its interval and "
        "sign-flip test; missing where a curve does not reach the target.",
    ),
    Analysis(
        "held_out_thresholds",
        _of_scores(held_out_thresholds),
        "Per detector and target, the setting chosen on the even replicates and its "
        "recall, false positives and errors on the odd (held-out) replicates alone.",
        plot_held_out_thresholds,
        "Held-out recall beside the calibration recall of the chosen setting.",
    ),
    *(
        Analysis(
            f"robustness_{name}",
            _of_scores(robustness, _measure_rows(measure)),
            f"Each method's {measure.replace('_', ' ')} at every level of each factor, "
            "and its change from the reference level, paired by replicate.",
            plot_robustness,
            "One panel per factor, one line per method.",
        )
        for measure, name in _ROBUSTNESS_NAMES.items()
    ),
    *(
        Analysis(
            f"robustness_crossed_{name}",
            _of_scores(robustness_crossed, _measure_rows(measure)),
            f"Each method's {measure.replace('_', ' ')} in every cell of the two crossed "
            "pairs of factors, and its change from the reference, paired by replicate.",
            plot_robustness_crossed,
            "The changes as a heatmap per pair, a row per method.",
        )
        for measure, name in _ROBUSTNESS_NAMES.items()
    ),
    Analysis(
        "rates_by_state",
        lambda inputs: rates_by_state(inputs.tables, n_resamples=inputs.n_resamples),
        "Each method's events per minute at rest and while running, beside the true "
        "rates (network events at rest, theta bursts while running).",
        plot_rates_by_state,
        "Both rates per method, the true rates marked.",
    ),
    Analysis(
        PARTICIPATION_BIAS,
        _of_reference(participation_bias),
        "The recruited cells of the true events each method finds against those of all "
        "true events with a burst: ratio of means and KS statistic.",
        plot_participation_bias,
        "The ratios with their intervals.",
    ),
    Analysis(
        BOUNDARY_EFFECT,
        _of_reference(boundary_effect),
        "Units active within the detected bounds minus within the matched truth window, "
        "for all units and principal ones.",
        plot_boundary_effect,
        "The mean differences with their intervals.",
    ),
    Analysis(
        MATCHING,
        _matching,
        "Recall, precision, F1, the IoU distribution, median absolute errors, recall by "
        "event type and at the target rates, and ranks, at minimum IoU 0, 0.2 and 0.5.",
        plot_matching_sensitivity,
        "Recall and precision at each minimum IoU.",
    ),
    Analysis(
        "appendix_expressions",
        _of_reference(appendix_expressions),
        "Every main method against every expression (network, ripple, sharp wave, burst) "
        "at IoU 0, not only its primary one: recall, precision, false positives per "
        "minute, median IoU and median signed and absolute onset and offset errors, with "
        "intervals; point methods by peak containment.",
    ),
    *(
        Analysis(
            f"appendix_curves_{expression}",
            _expression_curves(expression),
            f"Every interval method's recall, precision and false positives per minute at "
            f"every setting and minimum IoU against the {expression.replace('_', ' ')} "
            "truth, whatever its primary expression.",
        )
        for expression in EXPRESSION_ORDER
    ),
    Analysis(
        MODEL_CHANGES,
        _of_scores(model_sensitivity, operator.itemgetter(0)),
        "Each result's change under each of the simulator's six alternative models, "
        "paired with the reference by replicate; unreachable targets stay missing.",
        plot_model_sensitivity,
        "Recall changes per alternative, detectors at 1 per minute as triangles.",
    ),
    Analysis(
        MODEL_ORDERS,
        _of_scores(model_sensitivity, operator.itemgetter(1)),
        "Orders of detectors by recall at common false-positive rates in the reference and "
        "under each alternative model, with intervals and the share of resamples reversed.",
    ),
)
# Figures are saved at this resolution.
_DPI = 100
FLOAT_FORMAT = "%.6g"


def _png(figure: Figure) -> bytes:
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=_DPI, bbox_inches="tight")
    plt.close(figure)
    return buffer.getvalue()


def _point_lines(tables: RunTables) -> list[str]:
    """``summary.md``'s account of the point methods and of interval methods
    with events of one sample."""
    points = sorted(
        set(tables.methods.loc[tables.methods["scoring"] == PEAK_CONTAINMENT, "method"])
    )
    events = tables.intervals.events
    single = sorted(set(events.loc[events["end_time"] <= events["start_time"], "method"]))
    lines = [
        (
            "Point inventories ("
            + (", ".join(f"`{method}`" for method in points) or "none in this run")
            + f'; the catalog\'s output "{POINT_OUTPUT}") return one time point per event, '
            "which no interval rule can credit. They are scored by peak containment: a "
            "point matches a truth window that contains it, one to one, the most pairs. "
            "Only recall, precision and false positives per minute are reported for them "
            "(`point_inventories.csv` against the primary expression, "
            "`appendix_expressions.csv` against every expression, and rows marked "
            "`peak_containment` in `rates_by_state.csv`, the robustness and model "
            "sensitivity tables and the recall changes below), never pooled with interval "
            "scores; they are left out of the detection profile, false positive classes, "
            "agreement and its dendrogram, consensus, overlap, boundary errors, paired "
            "timing, method differences, error correlations, splits and merges, the "
            "operating curves, points, differences and held-out thresholds, the appendix "
            "curves, participation bias, the boundary effect and matching sensitivity."
        )
    ]
    if single:
        lines.append("")
        lines.append(
            "Interval methods whose events can be one sample long ("
            + ", ".join(f"`{method}`" for method in single)
            + ") keep the interval rule: the catalog, not the events' lengths, decides."
        )
    return lines


def _summary(
    run_name: str,
    tables: RunTables,
    files: Sequence[tuple[str, str]],
    results: Mapping[str, pd.DataFrame],
    validation: pd.DataFrame | None,
    scores: ConditionScores | None = None,
    validation_problem: str = "",
    trends_written: bool = False,
    run_directory: str | None = None,
) -> str:
    """``summary.md``: what was analysed, the conventions, each file with
    its sentence, the failures (of the reference, and of every condition
    when ``scores`` is given), and the lists the analyses call for."""
    main = tables.methods
    counts = failure_counts(tables)
    failed = counts[counts["n_failures"] > 0]
    conditions = ", ".join(tables.sessions["condition_id"].drop_duplicates())
    lines = [
        f"# Benchmark results: {run_name}",
        "",
        (
            f"`analyze.py` on `{run_directory or f'examples/benchmark/output/{run_name}'}"
            f"/combined/`: {len(tables.sessions)} sessions of {conditions}, {len(main)} "
            "methods (detectors at their defaults, every recipe), unless a file says "
            "otherwise (the operating curves, points, differences and thresholds and the "
            "appendix curves read the reference's sweeps; robustness and model sensitivity "
            "every condition)."
        ),
        "",
        (
            "Events are matched one to one to the truth windows at 10 % of the peak (IoU "
            "0), each method against its primary expression unless a file says otherwise. "
            "Times are seconds; a signed error is detected minus truth (negative: early), a "
            "difference between methods A minus B, A named first, a change between "
            "conditions the other condition minus the reference. Intervals are 95 % "
            "paired-bootstrap intervals over sessions within a condition and over "
            "replicates across conditions; p-values are two-sided sign-flip tests over the "
            "same units."
        ),
        "",
        *_point_lines(tables),
        "",
        "## Files",
        "",
        *(f"- `{name}`: {sentence}" for name, sentence in files),
        "",
        "## Failures",
        "",
        (
            "Every table counts each method's failures (`n_failures`): a session without "
            "its scores is a failure, never zero events, and the numbers pool the sessions "
            "it ran."
        ),
        "",
    ]
    if failed.empty:
        lines.append("No method failed on the reference condition's sessions.")
    else:
        lines += [
            f"- `{row.method}` ({row.setting}): {row.n_failures} of "
            f"{row.n_sessions + row.n_failures} sessions; {row.error}"
            for row in failed.itertuples()
        ]
    if scores is not None:
        grouped = _failure_counts(
            scores, scores.sessions["session_id"], ["method", "setting", "condition_id"]
        )
        lines += [
            "",
            (
                f"Across every condition, {grouped.sum()} calls failed (sweeps included), "
                "by method, setting and condition:"
            ),
            "",
        ]
        lines += [
            f"- `{method}` ({setting}), `{condition}`: {count} sessions"
            for (method, setting, condition), count in grouped.items()
        ] or ["None."]
    robust = results.get(ROBUSTNESS_RECALL)
    if robust is not None:
        moved = recall_changes(robust)
        lines += [
            "",
            f"## Recall changing by more than {RECALL_CHANGE:g} across a factor",
            "",
            (
                "Pooled recall against the primary expression at each level, over the "
                "replicates every level shares on which the method ran in every level "
                "(`robustness_recall.csv`, which has each change's interval)."
            ),
            "",
        ]
        lines += [
            f"- `{row.factor}`: `{row.method}` ({row.setting}) {row.recall_lowest:.3f} at "
            f"{row.lowest} to {row.recall_highest:.3f} at {row.highest}"
            + ("," + _scored(row.scoring) if row.scoring == PEAK_CONTAINMENT else "")
            for row in moved.itertuples(index=False)
        ] or ["None."]
    sensitivity = results.get(MATCHING)
    if sensitivity is not None:
        ranks = order_changes(sensitivity)
        levels = [column for column in ranks.columns if column.startswith("rank_")]
        lines += [
            "",
            "## Order changes with the minimum IoU",
            "",
            "Rank by recall among methods of the same primary expression (1 best, ties "
            "sharing the best rank) at "
            + ", ".join(level.removeprefix("rank_") for level in levels)
            + " (`matching_sensitivity.csv`).",
            "",
        ]
        lines += [
            f"- `{row['method']}` ({row['primary_expression']}): "
            + ", ".join(str(row[level]) for level in levels)
            for row in ranks.to_dict("records")
        ] or ["None."]
    changes = results.get(MODEL_CHANGES)
    orders = results.get(MODEL_ORDERS)
    if changes is not None and orders is not None:
        lines += [
            "",
            "## Model sensitivity",
            "",
            (
                "Each alternative model against the reference, paired by replicate "
                "(`model_sensitivity.csv`, `model_sensitivity_orders.csv`). A statement here "
                "is a reference order of detectors by recall at a common false-positive rate "
                "that its interval supports; it survives an alternative when that "
                "alternative's interval supports the same order. The alternatives are not "
                "pooled: there is no overall winner across them, and one factor at a time "
                "does not establish robustness to combinations of assumptions. Recipes have "
                "one setting each, so their changes are reported but not ordered. The "
                "validation report's changed target statistics are listed beside each."
            ),
            "",
            *(
                [
                    (
                        f"The validation report was not read ({validation_problem}): which "
                        "target statistics each alternative moves is unknown, not unchanged."
                    ),
                    "",
                ]
                if validation is None
                else []
            ),
            *model_sensitivity_statements(changes, orders, validation),
            "",
            (
                "Under `spatial_profile=local` timing errors are measured from the latent "
                "anchor, the component's centre on its anchor channel; the other channels' "
                "delays are part of that comparison. Under `fast_gamma_band=nearby` gamma "
                "bursts at 90-140 Hz are non-events by this benchmark's declared taxonomy, "
                "not by any physiological claim. Attribution computed on the reference alone "
                "remains conditional on the reference simulator."
            ),
        ]
    lines += [
        "",
        "## Held-out thresholds",
        "",
        (
            "No threshold is recommended here. `held_out_thresholds.csv` gives, per detector "
            "and target rate, the setting chosen on the even replicates and its performance "
            "on the odd ones alone: the numbers a recommendation would quote. The operating "
            "curves stay descriptive."
        ),
        "",
        "## Trends and spot checks",
        "",
        (
            "The trends stated so far, each with what its spot check showed, are in "
            f"[{TRENDS}]({TRENDS})."
            if trends_written
            else f"No `{TRENDS}` has been written yet: no trend is stated."
        ),
        "",
        (
            f"`{CANDIDATES}.csv` lists candidates with their evidence rows; each is stated "
            f"(in `{TRENDS}`, with a sentence on what its spot check showed) only once its "
            f"underlying events have been looked at (a figure in `{SPOT_CHECKS}/`) and it is "
            "checked not to come from failures, empty sweeps or a unit error. "
            f"`{TRENDS}` and `{SPOT_CHECKS}/` are written by hand and carried over when "
            "`analyze.py` rebuilds this directory; every other file is rebuilt."
        ),
    ]
    return "\n".join(lines) + "\n"


def _shown(path: Path) -> str:
    """``path`` from the repository's root when it is inside it, else whole."""
    resolved = path.resolve()
    if resolved.is_relative_to(REPOSITORY):
        return resolved.relative_to(REPOSITORY).as_posix()
    return resolved.as_posix()


def _validation(run_directory: Path) -> tuple[pd.DataFrame | None, str]:
    """``validation_changes`` of the report the run was checked against.

    ``run_spec.json`` names the report's ``spec.json`` (a path from the
    repository, or absolute) and its SHA-256; the spec lists its
    ``checks.csv``'s. Both files must be there and be those files.

    Parameters
    ----------
    run_directory : pathlib.Path

    Returns
    -------
    changes : pandas.DataFrame or None
        None when the report cannot be read: then nothing is known about
        what the alternatives change in it, which is not the same as nothing
        changing.
    problem : str
        Why it was not read; ``""`` when it was.
    """
    from validate_simulator import file_sha256, manifest_problems

    run_spec = run_directory / "run_spec.json"
    if not run_spec.exists():
        return None, f"no run_spec.json in {run_directory}"
    try:
        identity = json.loads(run_spec.read_text()).get("validation_report") or {}
        if not identity.get("path"):
            return None, "run_spec.json names no validation report"
        spec = REPOSITORY / str(identity["path"])
        if not spec.exists():
            return None, f"{spec} is missing"
        if file_sha256(spec) != identity.get("sha256"):
            return None, f"{spec} differs from the one the run was checked against"
        listed = json.loads(spec.read_text()).get("artifacts", {}).get("checks.csv")
        if listed is None:
            return None, f"{spec} does not list checks.csv"
        problems = manifest_problems(spec.parent, {"checks.csv": listed})
        if problems:
            return None, f"{spec.parent}: {problems[0]}"
        return validation_changes(pd.read_csv(spec.with_name("checks.csv"))), ""
    except (OSError, ValueError, KeyError) as error:
        return None, f"the report could not be read: {type(error).__name__}: {error}"


def analyze_run(
    run_directory: str | os.PathLike[str],
    results_directory: str | os.PathLike[str],
    *,
    workers: int = 1,
    figures: bool = True,
    analyses: Sequence[Analysis] = ANALYSES,
    n_resamples: int = N_RESAMPLES,
) -> dict[str, float]:
    """Run every analysis on a run and write its results.

    Parameters
    ----------
    run_directory : str or path-like
        ``examples/benchmark/output/<run_name>``; its ``combined/`` and
        ``conditions.csv`` are read.
    results_directory : str or path-like
        Rebuilt from scratch (in ``<name>.partial``, renamed into place):
        ``<name>.csv`` per analysis, ``<name>.png`` per figure (none for an
        empty table), ``candidate_trends.csv`` and ``summary.md``. What is
        written there by hand, ``trends.md`` and ``spot_checks/``, is copied
        into the rebuilt directory.
    workers : int, optional
        Processes for matching the sessions again.
    figures : bool, optional
        Draw the figures (needs matplotlib).
    analyses : sequence of Analysis, optional
    n_resamples : int, optional
        The bootstrap resamples of every interval.

    Returns
    -------
    seconds : dict of str to float
        Wall time of loading, matching, loading every condition's scores,
        each analysis's table and each figure.

    Raises
    ------
    ValueError
        Two analyses share a name, or a file would be over ``SIZE_LIMIT``:
        nothing is written in place.
    """
    names = [analysis.name for analysis in analyses]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        msg = f"The analysis names repeat: {repeated}."
        raise ValueError(msg)
    root = Path(run_directory)
    seconds = {}
    started = wall_clock.perf_counter()
    tables = load_run(root / "combined")
    seconds["load"] = wall_clock.perf_counter() - started
    started = wall_clock.perf_counter()
    matches = match_run(
        tables, workers=max(1, min(workers, len(tables.sessions))), levels=MATCH_IOU_LEVELS
    )
    seconds["match"] = wall_clock.perf_counter() - started
    started = wall_clock.perf_counter()
    scores = load_scores(root, workers=workers)
    seconds["scores"] = wall_clock.perf_counter() - started
    inputs = Inputs(tables, matches, scores, *_validation(root), n_resamples=n_resamples)
    if figures:
        import matplotlib as mpl

        mpl.use("Agg")
    files, results = [], {}
    previous = Path(results_directory)
    with replace_directory(previous) as partial:
        # the hand-written trends and their spot checks outlive a rebuild
        if (previous / SPOT_CHECKS).is_dir():
            shutil.copytree(previous / SPOT_CHECKS, partial / SPOT_CHECKS)
        if (previous / TRENDS).is_file():
            shutil.copy2(previous / TRENDS, partial / TRENDS)
        for analysis in analyses:
            started = wall_clock.perf_counter()
            table = analysis.table(inputs)
            seconds[analysis.name] = wall_clock.perf_counter() - started
            results[analysis.name] = table
            text = table.to_csv(index=False, float_format=FLOAT_FORMAT)
            write_result(partial / f"{analysis.name}.csv", text.encode())
            files.append((f"{analysis.name}.csv", analysis.description))
            if figures and analysis.figure is not None and len(table):
                started = wall_clock.perf_counter()
                write_result(partial / f"{analysis.name}.png", _png(analysis.figure(table)))
                seconds[f"{analysis.name}.png"] = wall_clock.perf_counter() - started
                files.append((f"{analysis.name}.png", analysis.figure_description))
        trends = candidate_trends(results)
        text = trends.to_csv(index=False, float_format=FLOAT_FORMAT)
        write_result(partial / f"{CANDIDATES}.csv", text.encode())
        files.append(
            (
                f"{CANDIDATES}.csv",
                (
                    "Candidate trends drawn from the tables, each with its evidence and the "
                    "spot check to draw; none is a conclusion until its events have been "
                    "looked at."
                ),
            )
        )
        summary = _summary(
            root.name,
            tables,
            files,
            results,
            inputs.validation,
            scores,
            inputs.validation_problem,
            (partial / TRENDS).is_file(),
            _shown(root),
        )
        write_result(partial / "summary.md", summary.encode())
    return seconds


def main(argv: Sequence[str] | None = None) -> None:
    """The command line; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="processes matching sessions again (default: one fewer than the cores)",
    )
    parser.add_argument(
        "--run-directory",
        help="the run's directory (default: examples/benchmark/output/<run-name>)",
    )
    parser.add_argument(
        "--results-directory",
        help="where to write (default: examples/benchmark/results/<run-name>)",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1.")
    run_directory = Path(args.run_directory or OUTPUT / args.run_name)
    results = Path(args.results_directory or RESULTS / args.run_name)
    try:
        seconds = analyze_run(run_directory, results, workers=args.workers)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    for step, taken in seconds.items():
        print(f"{step}: {taken:.1f} s")
    print(results)


if __name__ == "__main__":
    main()

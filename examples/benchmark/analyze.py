"""Analyze a finished benchmark run: what each method finds and misses, what its
false positives are, how methods agree and how their boundaries differ.

Usage, from the repository root (see README.md, "Analysing a run")::

    uv run python examples/benchmark/analyze.py --run-name NAME [--workers N]
        [--run-directory PATH] [--results-directory PATH]

It reads the run's ``combined/`` (``examples/benchmark/output/<run_name>/`` unless
``--run-directory`` says otherwise; ``run.py``'s docstring lists every column) and
rebuilds ``examples/benchmark/results/<run_name>/``: per analysis in ``ANALYSES``
one CSV and one PNG, and ``summary.md``, which names each file with one sentence on
what it shows and lists the methods that failed. No file may pass ``SIZE_LIMIT``
(1 MB): the command stops before writing one, leaving the previous results as they
were.

What is analysed. The reference condition's sessions and the rows whose
``setting`` is ``"default"`` or ``"literature"`` (``main_rows``): each detector at
its defaults and every recipe. The runner stores events, not pairs, so every
session is matched again (``match_run``, ``--workers`` processes): one to one
(``match_events``, IoU 0: any overlap) against the truth windows at 10 % of the
peak, errors also against those at 25 and 50 %. A method is headlined against its
primary expression (``methods.csv``); comparisons of all pairs of methods use the
network truth, the one every method is scored on.

Failures. A session, method and setting the run should hold and has no scores for
is a failure, never zero events (``load_run``). Every per-method table carries
``n_sessions``, the sessions its numbers pool, and ``n_failures``; a table of pairs
``n_failures_a`` and ``n_failures_b``.

Intervals and tests. Every interval is a 95 % percentile interval from
``paired_bootstrap`` over sessions (2000 resamples, seed 0): a resample draws
sessions with replacement, one draw shared by every method, so the methods stay
paired. A pooled ratio or median is resampled whole; a difference between two
methods is summarized per session (the sessions are the independent units), and
its estimate and interval are the mean of those per-session values and its
p-value ``sign_flip_test``'s, two-sided, over them.

Signs and units. Times and errors are seconds (the figures show milliseconds). A
signed error is detected minus truth: negative, early. A difference between two
methods is A minus B, A the method named first (by name): negative, A earlier, or
for absolute errors, A closer to the truth.

The tables, each built by the function of the same name, whose docstring lists its
columns: ``failures`` (``failure_counts``), ``detection_profile``,
``false_positive_classes``, ``pairwise_agreement``, ``agreement_dendrogram``,
``consensus``, ``overlap_quality``, ``boundary_errors``,
``paired_timing_<expression>`` (``paired_timing``, one per primary expression),
``method_differences``, ``error_correlations`` and ``splits_and_merges``.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import io
import itertools
import os
import time as wall_clock
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from conditions import TRUTH_FRACTIONS
from numpy.typing import ArrayLike
from recipe_configs import RECIPES
from run import OUTPUT, TABLES, _concat, load_truth, read_table, truth_window_sets
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
_KEY = ["session_id", "method", "setting"]
# Rows of a large table read at once.
_CHUNK_ROWS = 1_000_000

# The catalog's output of a method whose events are time points, and the two
# scoring rules.
POINT_OUTPUT = "ripple peaks"
PEAK_CONTAINMENT = "peak_containment"
INTERVAL = "interval"

PERCENTS = tuple(round(100 * fraction) for fraction in TRUTH_FRACTIONS)
# The label of a false positive that overlaps no truth window.
BACKGROUND = "background"
# The type that has two or three ripples, for split and merge rates.
DOUBLET = "ripple_doublet"
WINDOW_COLUMNS = ("session_id", "expression", "row", "id", "type", "start_time", "end_time")
ERROR_COLUMNS = tuple(
    f"{kind}_error_{percent}" for percent in PERCENTS for kind in ("onset", "offset")
)
PAIR_COLUMNS = (
    "session_id",
    "method",
    "setting",
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
    "session_id",
    "method",
    "setting",
    "subset",
    "n_truth",
    "n_split",
    "n_detected",
    "n_merged",
)
FALSE_POSITIVE_COLUMNS = (
    "session_id",
    "method",
    "setting",
    "event_index",
    "start_time",
    "end_time",
    "label",
)
SESSION_COMPARISON_COLUMNS = ("session_id", "truth_expression", *COMPARISON_COLUMNS)
CONSENSUS_COLUMNS = ("session_id", "row", "type", "n_methods", "n_methods_run")
GROUP_COLUMNS = ("session_id", "n_methods", "n_events", "start_time", "end_time")
POINT_COLUMNS = ("session_id", "method", "setting", "n_reference", "n_detected", "n_matched")
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
        for part in ("pooled", "estimate", "low", "high", "p")
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
    replicate keeps them together. A value drawn twice is two draws: its
    rows' ``session_id`` and ``replicate`` become text with ``"#<k>"``
    appended, ``k`` the draw's position, so a statistic grouping by either
    keeps both copies.

    Parameters
    ----------
    frame : pandas.DataFrame
        With ``session_id`` and ``replicate`` columns.
    statistic : callable
        ``statistic(frame)`` gives a Series of estimates, each resample's
        with the same index (an entry a resample lacks is NaN there).
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
    """
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
        draws.append(statistic(resampled))
    table = pd.DataFrame(draws)
    alpha = (1 - level) / 2
    return pd.DataFrame(
        {"estimate": estimate, "low": table.quantile(alpha), "high": table.quantile(1 - alpha)}
    )


def sign_flip_test(
    differences: ArrayLike, *, n_resamples: int = 10_000, seed: int = SEED
) -> float:
    """Two-sided paired test that the mean difference over sessions is 0.

    Parameters
    ----------
    differences : array_like, shape (n_sessions,)
        One paired difference per session, each finite: pair the sessions
        where both values exist first, and report how many were dropped.
    n_resamples : int, optional
        Random sign vectors when there are more than 16 sessions.
    seed : int, optional

    Returns
    -------
    p_value : float
        The fraction of sign flips whose absolute mean is at least the
        observed one: every flip, exactly, for up to 16 sessions; else
        ``(k + 1) / (n_resamples + 1)`` over random flips, never 0. NaN for
        no sessions.

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
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=d.size)))
        null = np.abs((signs * d).mean(axis=1))
        return float((null >= observed - _TIE).mean())
    signs = np.random.default_rng(seed).choice((-1.0, 1.0), size=(n_resamples, d.size))
    null = np.abs((signs * d).mean(axis=1))
    return float(((null >= observed - _TIE).sum() + 1) / (n_resamples + 1))


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
    timestamps' rounding (8 units in the last place of the largest
    magnitude, as ``match_events`` judges a touch). Windows are taken in
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
    return np.where(np.isfinite(peak), peak, middle)


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
        sorted: ``method``, ``setting``, ``primary_expression``, ``role``,
        ``scoring`` (``scoring_rule``).
    ran : pandas.DataFrame
        One row per session, method and setting with scores: ``session_id``,
        ``method``, ``setting``.
    failures : pandas.DataFrame
        One row per session and (``methods``) method and setting without
        scores: those columns and ``error``, the runner's ``failures.csv``
        message, ``""`` where it recorded none. A missing result is a
        failure, never zero events.
    metrics, events, truth_counts : pandas.DataFrame
        Those tables' rows of the sessions, methods and settings read
        (``truth_counts`` has no method).
    truth : dict of str to (pandas.DataFrame, pandas.DataFrame)
        By ``session_id``, its latent event and non-event tables, as
        ``run.load_truth`` restores them.
    """

    sessions: pd.DataFrame
    methods: pd.DataFrame
    ran: pd.DataFrame
    failures: pd.DataFrame
    metrics: pd.DataFrame
    events: pd.DataFrame
    truth_counts: pd.DataFrame
    truth: dict[str, tuple[pd.DataFrame, pd.DataFrame]]


def _selected(
    frame: pd.DataFrame, sessions: Collection[str], settings: Collection[str] | None
) -> pd.DataFrame:
    """The rows of ``sessions`` and, when given, of ``settings``."""
    keep = frame["session_id"].isin(sessions)
    if settings is not None and "setting" in frame.columns:
        keep &= frame["setting"].isin(settings)
    return frame[keep].reset_index(drop=True)


def _read_selected(
    path: Path, sessions: Collection[str], settings: Collection[str] | None
) -> pd.DataFrame:
    """A runner table's rows of some sessions and settings, read chunk by
    chunk with ``read_table``'s conventions (text as text, floats exactly as
    written), so a whole large table is never held at once."""
    table = TABLES[path.name]
    reader = pd.read_csv(
        path,
        chunksize=_CHUNK_ROWS,
        dtype=dict.fromkeys(table.text, str),
        keep_default_na=False,
        na_values={column: [""] for column in table.columns if column not in table.text},
        float_precision="round_trip",
    )
    chunks = [_selected(chunk, sessions, settings) for chunk in reader]
    # chunks left empty add nothing (and would make pandas warn about dtypes)
    kept = [chunk for chunk in chunks if len(chunk)] or chunks[:1]
    return pd.concat(kept, ignore_index=True)


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
    listed = _selected(read_table(root / "methods.csv"), ids, settings)
    methods = (
        listed.drop_duplicates(["method", "setting"])[
            ["method", "setting", "primary_expression", "role"]
        ]
        .sort_values(["method", "setting"])
        .reset_index(drop=True)
    )
    methods["scoring"] = methods["method"].map(scoring_rule)
    metrics = _read_selected(root / "metrics.csv.gz", ids, settings)
    ran = metrics[_KEY].drop_duplicates().reset_index(drop=True)
    # every session should hold every method and setting: one without scores failed
    expected = sessions[["session_id"]].merge(methods[["method", "setting"]], how="cross")
    missing = expected.merge(ran, on=_KEY, how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"][_KEY]
    recorded = _selected(read_table(root / "failures.csv"), ids, settings)
    failures = missing.merge(
        recorded.drop_duplicates(_KEY)[[*_KEY, "error"]], on=_KEY, how="left"
    ).fillna({"error": ""})
    truth = {
        session_id: tables
        for session_id, tables in load_truth(root / "truth.csv.gz").items()
        if session_id in ids
    }
    return RunTables(
        sessions=sessions,
        methods=methods,
        ran=ran,
        failures=failures.reset_index(drop=True),
        metrics=metrics,
        events=_read_selected(root / "events.csv.gz", ids, settings),
        truth_counts=_selected(read_table(root / "truth_counts.csv.gz"), ids, None),
        truth=truth,
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
        One row per truth window of each expression (``EXPRESSIONS``) at 10 %:
        ``expression``, ``row`` (its position, as ``truth_row`` gives it),
        ``id`` (the latent event), ``type`` (its event type), ``start_time``,
        ``end_time``.
    pairs : pandas.DataFrame
        One row per matched pair of each method and setting, expression and
        ``minimum_iou``: ``truth_row``, ``event_index`` (``events.csv``'s),
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
        One row per method and setting in ``point_methods``, scored by peak
        containment (``match_peaks``) against its primary expression's
        windows at 10 %: ``n_reference``, ``n_detected``, ``n_matched``. These
        methods are in no other table: an interval rule cannot credit a point.
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


def _bounds(frame: pd.DataFrame) -> np.ndarray[Any, Any]:
    return np.asarray(frame[["start_time", "end_time"]], dtype=float).reshape(-1, 2)


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
        The ``minimum_iou`` levels of ``pairs``; 0 is always among them.
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
            reference = truth_bounds[primary[method, setting]][0]
            found = match_peaks(reference, event_times(rows))
            peaks.append(
                {
                    **key,
                    "n_reference": len(reference),
                    "n_detected": len(rows),
                    "n_matched": len(found),
                }
            )
            continue
        detected[method, setting] = bounds
        for expression, references in truth_bounds.items():
            for level in levels:
                matching = rd.match_events(references[0], bounds, minimum_iou=level)
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
                            "expression": expression,
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
        expression = primary[method, setting]
        reference = truth_bounds[expression][0]
        matching = rd.match_events(reference, bounds)
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
    comparisons, consensus, groups = _compare_main(
        session_id,
        detected,
        primary,
        truth_bounds,
        sets["network"][0]["type"].to_numpy(),
        _concat(false_positives, FALSE_POSITIVE_COLUMNS),
    )
    return Matches(
        windows=windows,
        pairs=_concat(pairs, PAIR_COLUMNS),
        overlaps=_concat(overlaps, OVERLAP_COLUMNS),
        false_positives=_concat(false_positives, FALSE_POSITIVE_COLUMNS),
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

# A statistic of every group at once: (group codes, rows, number of groups)
# to an array of shape (n_statistics, n_groups).
GroupStatistic = Callable[[np.ndarray[Any, Any], pd.DataFrame, int], np.ndarray[Any, Any]]


def _ratio_of_sums(*ratios: tuple[str, str]) -> GroupStatistic:
    """Each (numerator, denominator) column pair's pooled ratio per group,
    NaN where the denominator sums to 0."""

    def statistic(
        codes: np.ndarray[Any, Any], frame: pd.DataFrame, n_groups: int
    ) -> np.ndarray[Any, Any]:
        sums = {
            column: np.bincount(
                codes, weights=frame[column].to_numpy(dtype=float), minlength=n_groups
            )
            for column in dict.fromkeys(itertools.chain.from_iterable(ratios))
        }
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.array([sums[top] / sums[bottom] for top, bottom in ratios])

    return statistic


def _means(*columns: str) -> GroupStatistic:
    """Each column's mean per group over its finite values, NaN for none."""

    def statistic(
        codes: np.ndarray[Any, Any], frame: pd.DataFrame, n_groups: int
    ) -> np.ndarray[Any, Any]:
        means = []
        for column in columns:
            values = frame[column].to_numpy(dtype=float)
            finite = np.isfinite(values)
            total = np.bincount(codes[finite], weights=values[finite], minlength=n_groups)
            count = np.bincount(codes[finite], minlength=n_groups)
            with np.errstate(invalid="ignore", divide="ignore"):
                means.append(total / count)
        return np.array(means)

    return statistic


def _medians(*columns: str) -> GroupStatistic:
    """Each column's median per group over its values, NaN for none."""

    def statistic(
        codes: np.ndarray[Any, Any], frame: pd.DataFrame, n_groups: int
    ) -> np.ndarray[Any, Any]:
        medians = frame[list(columns)].groupby(codes).median().reindex(range(n_groups))
        return np.asarray(medians, dtype=float).T

    return statistic


def grouped_intervals(
    frame: pd.DataFrame,
    by: Sequence[str],
    statistic: GroupStatistic,
    names: Sequence[str],
    columns: Sequence[str],
    *,
    n_resamples: int = N_RESAMPLES,
) -> pd.DataFrame:
    """A statistic of each group, with its paired-bootstrap interval over sessions.

    Parameters
    ----------
    frame : pandas.DataFrame
        With ``session_id``, ``replicate``, the ``by`` columns and ``columns``.
    by : sequence of str
        The columns whose values name a group.
    statistic : callable
        ``statistic(codes, rows, n_groups)``: an array of shape
        ``(len(names), n_groups)``, the statistics of each group, ``codes``
        numbering the groups of ``rows`` from 0.
    names : sequence of str
        The statistics' names.
    columns : sequence of str
        The columns ``statistic`` reads.
    n_resamples : int, optional

    Returns
    -------
    intervals : pandas.DataFrame
        One row per group, sorted by ``by``: the ``by`` columns, then each
        name's estimate, ``<name>_low`` and ``<name>_high`` (95 %,
        ``paired_bootstrap`` with ``key="session_id"``, every group of a
        resample from the same sessions).
    """
    grouped = frame.groupby(list(by), sort=True)
    keys = grouped.size().reset_index()[list(by)]
    n_groups = len(keys)
    for name in names:
        keys[name] = keys[f"{name}_low"] = keys[f"{name}_high"] = np.nan
    if not n_groups:
        return keys
    rows = frame[["session_id", "replicate", *columns]].assign(
        _group=grouped.ngroup().to_numpy()
    )

    def flat(resampled: pd.DataFrame) -> pd.Series:
        codes = resampled["_group"].to_numpy()
        return pd.Series(statistic(codes, resampled, n_groups).ravel())

    found = paired_bootstrap(rows, flat, key="session_id", n_resamples=n_resamples)
    for position, name in enumerate(names):
        block = found.iloc[position * n_groups : (position + 1) * n_groups]
        keys[name] = block["estimate"].to_numpy()
        keys[f"{name}_low"] = block["low"].to_numpy()
        keys[f"{name}_high"] = block["high"].to_numpy()
    return keys


def _with_replicate(frame: pd.DataFrame, tables: RunTables) -> pd.DataFrame:
    """``frame`` with each session's ``replicate``."""
    replicates = tables.sessions.set_index("session_id")["replicate"]
    return frame.assign(replicate=frame["session_id"].map(replicates).to_numpy())


def _with_failures(frame: pd.DataFrame, tables: RunTables) -> pd.DataFrame:
    """A per-method table with each method's ``primary_expression``,
    ``n_sessions`` and ``n_failures`` (``failure_counts``) added."""
    counts = failure_counts(tables)[
        ["method", "setting", "primary_expression", "n_sessions", "n_failures"]
    ]
    return frame.merge(counts, on=["method", "setting"], how="left")


def _in_order(frame: pd.DataFrame, column: str, order: Sequence[str]) -> pd.DataFrame:
    """``frame`` sorted by method, setting and then ``column`` in ``order``."""
    rank = {value: position for position, value in enumerate(order)}
    return (
        frame.assign(_rank=frame[column].map(rank))
        .sort_values(["method", "setting", "_rank"], kind="stable")
        .drop(columns="_rank")
        .reset_index(drop=True)
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
    frame = _with_replicate(frame.fillna({"n_truth": 0, "n_found": 0}), tables)
    by = ["method", "setting", "expression"]
    intervals = grouped_intervals(
        frame,
        by,
        _ratio_of_sums(("n_found", "n_truth")),
        ["recall"],
        ["n_found", "n_truth"],
        n_resamples=n_resamples,
    )
    totals = frame.groupby(by)[["n_truth", "n_found"]].sum().astype(int).reset_index()
    return totals.merge(intervals, on=by)


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
    frame = _by_intervals(tables.ran).merge(
        pd.DataFrame({"type": rd.EVENT_TYPES}), how="cross"
    )
    frame = frame.join(n_true, on=["session_id", "type"]).join(n_found, on=[*_KEY, "type"])
    frame = _with_replicate(frame.fillna({"n_true": 0, "n_found": 0}), tables)
    by = ["method", "setting", "type"]
    intervals = grouped_intervals(
        frame,
        by,
        _ratio_of_sums(("n_found", "n_true")),
        ["recall"],
        ["n_found", "n_true"],
        n_resamples=n_resamples,
    )
    totals = frame.groupby(by)[["n_true", "n_found"]].sum().astype(int).reset_index()
    profile = totals.merge(intervals, on=by).rename(columns={"type": "event_type"})
    return _in_order(_with_failures(profile, tables), "event_type", rd.EVENT_TYPES)


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
    frame = _by_intervals(tables.ran).merge(pd.DataFrame({"label": labels}), how="cross")
    frame = frame.join(counted.rename("n_events"), on=[*_KEY, "label"]).fillna({"n_events": 0})
    frame["n_unmatched"] = frame.groupby(_KEY)["n_events"].transform("sum")
    frame = _with_replicate(frame, tables)
    by = ["method", "setting", "label"]
    intervals = grouped_intervals(
        frame,
        by,
        _ratio_of_sums(("n_events", "n_unmatched")),
        ["fraction"],
        ["n_events", "n_unmatched"],
        n_resamples=n_resamples,
    )
    totals = frame.groupby(by)[["n_events", "n_unmatched"]].sum().astype(int).reset_index()
    classes = totals.merge(intervals, on=by)
    return _in_order(_with_failures(classes, tables), "label", labels)


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
    table["n_methods_compared"] = len(_by_intervals(main_rows(tables.methods)))
    table["n_failed_calls"] = len(_by_intervals(main_rows(tables.failures)))
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
    frame = _with_replicate(matches.overlaps, tables)
    by = ["method", "setting", "subset"]
    counts = ["n_truth", "n_split", "n_detected", "n_merged"]
    intervals = grouped_intervals(
        frame,
        by,
        _ratio_of_sums(("n_split", "n_truth"), ("n_merged", "n_detected")),
        ["split_rate", "merge_rate"],
        counts,
        n_resamples=n_resamples,
    )
    totals = frame.groupby(by)[counts].sum().astype(int).reset_index()
    rates = totals.merge(intervals, on=by)
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
    return _in_order(_with_failures(rates[columns], tables), "subset", ("all", DOUBLET))


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
    counts = ["n_reference", "n_detected", "n_matched"]
    frame = matches.points.assign(
        n_unmatched=matches.points["n_detected"] - matches.points["n_matched"],
        minutes=matches.points["session_id"].map(_minutes_outside(tables.sessions)),
    )
    frame = _with_replicate(frame, tables)
    by = ["method", "setting"]
    intervals = grouped_intervals(
        frame,
        by,
        _ratio_of_sums(
            ("n_matched", "n_reference"),
            ("n_matched", "n_detected"),
            ("n_unmatched", "minutes"),
        ),
        ["recall", "precision", "false_positives_per_minute"],
        [*counts, "n_unmatched", "minutes"],
        n_resamples=n_resamples,
    )
    totals = frame.groupby(by)[[*counts, "minutes"]].sum().reset_index()
    table = totals.astype(dict.fromkeys(counts, int)).merge(intervals, on=by)
    table.insert(2, "scoring", PEAK_CONTAINMENT)
    return _with_failures(table, tables)


def _with_pair_failures(frame: pd.DataFrame, tables: RunTables) -> pd.DataFrame:
    """A table of method pairs with each method's ``n_failures_a`` and
    ``n_failures_b`` added (main methods, one setting each)."""
    failed = main_rows(failure_counts(tables)).set_index("method")["n_failures"]
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
) -> pd.DataFrame:
    """Each pair's per-session ``compare_detectors`` values averaged over the
    sessions both methods have scores on (NaN sessions left out), with
    intervals; ``n_sessions`` counts those sessions."""
    frame = _with_replicate(comparisons, tables)
    by = ["method_a", "method_b", "truth_expression"]
    means = grouped_intervals(
        frame, by, _means(*columns), list(columns), list(columns), n_resamples=n_resamples
    )
    n_sessions = frame.groupby(by).size().rename("n_sessions")
    means = means.join(n_sessions, on=by)
    return _with_pair_failures(means, tables)


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
        value, ``<name>_low`` and ``<name>_high``; then ``n_sessions`` (both
        have scores), ``n_failures_a``, ``n_failures_b``.
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
    methods = sorted(_by_intervals(main_rows(tables.methods))["method"])
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
    sessions with a finite value, as ``<column>_p``."""
    rows = []
    for key, group in frame.groupby(list(by), sort=True):
        row = dict(zip(by, key, strict=True))
        for column in columns:
            values = group[column].to_numpy(dtype=float)
            row[f"{column}_p"] = sign_flip_test(values[np.isfinite(values)])
        rows.append(row)
    return pd.DataFrame(rows, columns=[*by, *(f"{column}_p" for column in columns)])


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
        strictly earlier): the mean over sessions, ``<name>_low`` and
        ``<name>_high``; ``median_onset_difference_p`` and
        ``median_offset_difference_p``, ``sign_flip_test`` on the
        per-session medians; ``n_sessions``, ``n_failures_a``,
        ``n_failures_b``.
    """
    network = matches.comparisons[matches.comparisons["truth_expression"] == "network"]
    means = _session_means(tables, network, DIFFERENCES, n_resamples=n_resamples)
    by = ["method_a", "method_b", "truth_expression"]
    tests = _sign_flips(network, by, DIFFERENCES[:2])
    return means.merge(tests, on=by, how="left")


def error_correlations(
    tables: RunTables, matches: Matches, *, n_resamples: int = N_RESAMPLES
) -> pd.DataFrame:
    """Whether two main methods err together on the network events both found.

    Parameters
    ----------
    tables : RunTables
    matches : Matches
    n_resamples : int, optional

    Returns
    -------
    correlations : pandas.DataFrame
        One row per pair, as ``pairwise_agreement``'s: for
        ``onset_error_correlation`` and ``offset_error_correlation``
        (Spearman's correlation of their signed errors against the network
        windows over the events both found, NaN below 3) the mean over
        sessions with one, ``<name>_low`` and ``<name>_high``;
        ``n_sessions``, ``n_failures_a``, ``n_failures_b``.
    """
    network = matches.comparisons[matches.comparisons["truth_expression"] == "network"]
    return _session_means(tables, network, CORRELATIONS, n_resamples=n_resamples)


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
    pairs = _with_replicate(_primary_pairs(tables, matches), tables)
    by = ["method", "setting"]
    quality = _quantiles(pairs, by, OVERLAP_MEASURES)
    medians = grouped_intervals(
        pairs,
        by,
        _medians(*OVERLAP_MEASURES),
        OVERLAP_MEASURES,
        OVERLAP_MEASURES,
        n_resamples=n_resamples,
    )
    quality = quality.merge(
        _long_intervals(medians, by, OVERLAP_MEASURES), on=[*by, "measure"]
    )
    primary = _by_intervals(tables.methods)[
        ["method", "setting", "primary_expression"]
    ].rename(columns={"primary_expression": "expression"})
    found = recall(tables, matches, primary, n_resamples=n_resamples)
    quality = quality.merge(
        found[[*by, "recall", "recall_low", "recall_high"]], on=by, how="left"
    )
    return _in_order(_with_failures(quality, tables), "measure", OVERLAP_MEASURES)


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
    parts = frame["measure"].str.split("_", expand=True)
    return frame.assign(
        fraction=parts[2].astype(int) / 100, boundary=parts[0], measure=parts[1]
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
    scored = _by_intervals(tables.methods)[["method", "setting", "primary_expression"]].rename(
        columns={"primary_expression": "expression"}
    )
    joint = scored[scored["expression"] == "network"]
    scored = pd.concat(
        [scored, *(joint.assign(expression=expression) for expression in ("ripple", "burst"))],
        ignore_index=True,
    )
    pairs = matches.pairs[matches.pairs["minimum_iou"] == 0].merge(
        scored, on=["method", "setting", "expression"]
    )
    pairs, names = _errors(_with_replicate(pairs, tables))
    by = ["method", "setting", "expression"]
    errors = _quantiles(pairs, by, names)
    medians = grouped_intervals(
        pairs, by, _medians(*names), names, names, n_resamples=n_resamples
    )
    errors = errors.merge(_long_intervals(medians, by, names), on=[*by, "measure"])
    errors["iqr"] = errors["q75"] - errors["q25"]
    found = recall(tables, matches, scored, n_resamples=n_resamples)
    errors = errors.merge(
        found[[*by, "recall", "recall_low", "recall_high"]], on=by, how="left"
    )
    errors = _split_error_names(errors)
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
    errors = _with_failures(errors[columns], tables)
    rank = {expression: position for position, expression in enumerate(EXPRESSION_ORDER)}
    return (
        errors.assign(_rank=errors["expression"].map(rank))
        .sort_values(
            ["method", "setting", "_rank", "fraction", "boundary", "measure"],
            ascending=[True, True, True, True, False, False],
            kind="stable",
        )
        .drop(columns="_rank")
        .reset_index(drop=True)
    )


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
        ``_low`` and ``_high``; and ``_p``, ``sign_flip_test`` on those
        per-session medians; then ``n_failures_a``, ``n_failures_b``.
    """
    members = _by_intervals(main_rows(tables.methods))
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
        & matches.pairs["setting"].isin(MAIN_SETTINGS)
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
    per_session = _with_replicate(
        shared.groupby([*pair, "session_id"])[names].median().reset_index(), tables
    )
    estimates = grouped_intervals(
        per_session, pair, _means(*names), names, names, n_resamples=n_resamples
    ).set_index(pair)
    tests = _sign_flips(per_session, pair, names).set_index(pair)
    summary = (
        comparisons.groupby(pair)
        .agg(n_run=("session_id", "size"), jaccard_truth_ids=("jaccard_truth_ids", "mean"))
        .join(n_shared)
        .join(per_session.groupby(pair).size().rename("n_sessions"))
        .fillna({"n_shared": 0, "n_sessions": 0})
    )
    rows = []
    for (a, b), found in summary.iterrows():
        for percent in PERCENTS:
            row: dict[str, Any] = {
                "expression": expression,
                "method_a": a,
                "method_b": b,
                "fraction": percent / 100,
                "n_shared": int(found["n_shared"]),
                "n_sessions": int(found["n_sessions"]),
                "n_sessions_without": int(found["n_run"] - found["n_sessions"]),
                "jaccard_truth_ids": found["jaccard_truth_ids"],
            }
            for boundary in ("onset", "offset"):
                for measure in ("signed", "absolute"):
                    name = f"{boundary}_{measure}_{percent}"
                    stem = f"{boundary}_{measure}"
                    known = (a, b) in estimates.index
                    row[f"{stem}_pooled"] = pooled.loc[(a, b), name] if known else np.nan
                    for part in ("estimate", "low", "high"):
                        column = name if part == "estimate" else f"{name}_{part}"
                        row[f"{stem}_{part}"] = (
                            estimates.loc[(a, b), column] if known else np.nan
                        )
                    row[f"{stem}_p"] = tests.loc[(a, b), f"{name}_p"] if known else np.nan
            rows.append(row)
    return _with_pair_failures(pd.DataFrame(rows, columns=PAIRED_TIMING_COLUMNS), tables)


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
    """``detection_profile``'s recall, method by event type."""
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
    """``false_positive_classes``' fractions, stacked per method."""
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
    dendrogram's leaf order."""
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
    """``agreement_dendrogram``'s tree."""
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
    positives."""
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
    """``overlap_quality``'s distributions, one panel per measure."""
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
    25 % (triangle) and 50 % (square)."""
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
        axis.set_title(f"{boundary}, {measure} (ms; detected - truth)", fontsize=8)
    axes[0].set_yticks(y, labels, fontsize=_FONT)
    return figure


def plot_paired_timing(timing: pd.DataFrame) -> Figure:
    """``paired_timing``'s mean paired differences at 10 % of the peak, in ms:
    row A minus column B."""
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
    row A minus column B."""
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
    """``error_correlations``' mean correlations, method by method."""
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
    doublets."""
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


# The command line


@dataclasses.dataclass(frozen=True)
class Analysis:
    """One analysis: its table, the figure drawn from it, and what each shows.

    Attributes
    ----------
    name : str
        The files' stem: ``<name>.csv`` and ``<name>.png``.
    table : callable
        ``table(tables, matches)``: the analysis's table.
    description : str
        One sentence on what the table shows, for ``summary.md``.
    figure : callable or None
        ``figure(table)``: a matplotlib Figure; None for no figure.
    figure_description : str
        One sentence on what the figure shows.
    """

    name: str
    table: Callable[[RunTables, Matches], pd.DataFrame]
    description: str
    figure: Callable[[pd.DataFrame], Figure] | None = None
    figure_description: str = ""


ANALYSES: tuple[Analysis, ...] = (
    Analysis(
        "failures",
        lambda tables, _matches: failure_counts(tables),
        "Each method's sessions with scores and failures (a missing result, never zero "
        "events), with the first error.",
    ),
    Analysis(
        "point_inventories",
        point_inventories,
        "Recall, precision and false positives per minute of the methods that return "
        "time points, scored by peak containment and never pooled with interval scores.",
    ),
    Analysis(
        "detection_profile",
        detection_profile,
        "Recall per event type against the network truth, per method, pooled over sessions.",
        plot_detection_profile,
        "Recall as a heatmap, method by event type.",
    ),
    Analysis(
        "false_positive_classes",
        false_positive_classes,
        "What each method's false positives (unmatched against its primary expression) "
        "overlap longest: an event type's component, a non-event or nothing.",
        plot_false_positive_classes,
        "Those fractions stacked per method.",
    ),
    Analysis(
        "pairwise_agreement",
        pairwise_agreement,
        "Agreement of every pair of methods against the network truth: Jaccard of their "
        "events, of their true and of their false events, and of the true events found.",
        plot_pairwise_agreement,
        "The four indices as heatmaps, methods in the dendrogram's order.",
    ),
    Analysis(
        "agreement_dendrogram",
        agreement_dendrogram,
        "Methods clustered by average linkage on 1 - Jaccard.",
        plot_agreement_dendrogram,
        "The dendrogram.",
    ),
    Analysis(
        "consensus",
        consensus,
        "How many methods found each true event, by type, and how many methods each "
        "group of overlapping false positives spans.",
        plot_consensus,
        "Both distributions.",
    ),
    Analysis(
        "overlap_quality",
        overlap_quality,
        "IoU, coverage and temporal precision of each method's matched pairs against its "
        "primary expression, with its recall.",
        plot_overlap_quality,
        "Their distributions per method.",
    ),
    Analysis(
        "boundary_errors",
        boundary_errors,
        "Signed and absolute onset and offset errors (detected minus truth) against the "
        "truth at 10, 25 and 50 % of the peak, each median with its pair count and the "
        "method's recall.",
        plot_boundary_errors,
        "Their distributions at 10 % and the medians at 25 and 50 %, in ms.",
    ),
    *(
        Analysis(
            f"paired_timing_{expression}",
            functools.partial(paired_timing, expression=expression),
            f"For methods whose primary expression is {expression}, each pair's error "
            "differences (A minus B) on the true events both found, with a sign-flip test.",
            plot_paired_timing,
            "The mean paired differences at 10 % as heatmaps, in ms.",
        )
        for expression in ("ripple", "burst", "network")
    ),
    Analysis(
        "method_differences",
        method_differences,
        "How every pair of methods' matched events differ in start and end (A minus B), "
        "and how often A's comes first.",
        plot_method_differences,
        "The median differences as heatmaps, in ms.",
    ),
    Analysis(
        "error_correlations",
        error_correlations,
        "Spearman correlation of every pair of methods' signed errors on the network "
        "events both found.",
        plot_error_correlations,
        "The correlations as heatmaps.",
    ),
    Analysis(
        "splits_and_merges",
        splits_and_merges,
        "How often each method splits a true event or merges several, overall and on "
        "ripple doublets.",
        plot_splits_and_merges,
        "Both rates with their intervals.",
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


def _summary(run_name: str, tables: RunTables, files: Sequence[tuple[str, str]]) -> str:
    """``summary.md``: what was analysed, the conventions, each file with
    its sentence, and the failures."""
    main = main_rows(tables.methods)
    counts = failure_counts(tables)
    failed = counts[counts["n_failures"] > 0]
    conditions = ", ".join(tables.sessions["condition_id"].drop_duplicates())
    lines = [
        f"# Benchmark results: {run_name}",
        "",
        (
            f"`analyze.py` on `output/{run_name}/combined/`: {len(tables.sessions)} "
            f"sessions of {conditions}, {len(main)} methods (detectors at their defaults, "
            "every recipe)."
        ),
        (
            "Events are matched one to one to the truth windows at 10 % of the peak (IoU "
            "0), each method against its primary expression unless a file says otherwise. "
            "Times are seconds; a signed error is detected minus truth (negative: early), a "
            "difference between methods A minus B, A named first. Intervals are 95 % "
            "paired-bootstrap intervals over sessions; p-values are two-sided sign-flip "
            "tests over sessions."
        ),
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
        lines.append("No method failed on these sessions.")
    else:
        lines += [
            f"- `{row.method}` ({row.setting}): {row.n_failures} of "
            f"{row.n_sessions + row.n_failures} sessions; {row.error}"
            for row in failed.itertuples()
        ]
    return "\n".join(lines) + "\n"


def analyze_run(
    run_directory: str | os.PathLike[str],
    results_directory: str | os.PathLike[str],
    *,
    workers: int = 1,
    figures: bool = True,
    analyses: Sequence[Analysis] = ANALYSES,
) -> dict[str, float]:
    """Run every analysis on a run and write its results.

    Parameters
    ----------
    run_directory : str or path-like
        ``examples/benchmark/output/<run_name>``; its ``combined/`` is read.
    results_directory : str or path-like
        Rebuilt from scratch (in ``<name>.partial``, renamed into place):
        ``<name>.csv`` per analysis, ``<name>.png`` per figure and
        ``summary.md``.
    workers : int, optional
        Processes for matching the sessions again.
    figures : bool, optional
        Draw the figures (needs matplotlib).
    analyses : sequence of Analysis, optional

    Returns
    -------
    seconds : dict of str to float
        Wall time of loading, matching, each analysis's table and each figure.

    Raises
    ------
    ValueError
        A file would be over ``SIZE_LIMIT``: nothing is written in place.
    """
    root = Path(run_directory)
    seconds = {}
    started = wall_clock.perf_counter()
    tables = load_run(root / "combined")
    seconds["load"] = wall_clock.perf_counter() - started
    started = wall_clock.perf_counter()
    matches = match_run(tables, workers=max(1, min(workers, len(tables.sessions))))
    seconds["match"] = wall_clock.perf_counter() - started
    if figures:
        import matplotlib as mpl

        mpl.use("Agg")
    files = []
    with replace_directory(Path(results_directory)) as partial:
        for analysis in analyses:
            started = wall_clock.perf_counter()
            table = analysis.table(tables, matches)
            seconds[analysis.name] = wall_clock.perf_counter() - started
            text = table.to_csv(index=False, float_format=FLOAT_FORMAT)
            write_result(partial / f"{analysis.name}.csv", text.encode())
            files.append((f"{analysis.name}.csv", analysis.description))
            if figures and analysis.figure is not None:
                started = wall_clock.perf_counter()
                write_result(partial / f"{analysis.name}.png", _png(analysis.figure(table)))
                seconds[f"{analysis.name}.png"] = wall_clock.perf_counter() - started
                files.append((f"{analysis.name}.png", analysis.figure_description))
        summary = _summary(root.name, tables, files)
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
        "--run-directory", help="the run's directory (default: output/<run-name>)"
    )
    parser.add_argument(
        "--results-directory", help="where to write (default: results/<run-name>)"
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

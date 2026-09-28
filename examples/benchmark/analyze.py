"""Analyze a finished benchmark run: what each method finds and misses, and how
methods agree.

Intervals and tests. Every interval is a 95 % percentile interval from
``paired_bootstrap`` over sessions (2000 resamples, seed 0): a resample draws
sessions with replacement, one draw shared by every method, so the methods stay
paired. A difference between two methods carries ``sign_flip_test``'s two-sided
p-value over sessions.

Every results file is at most ``SIZE_LIMIT`` bytes (1 MB): ``write_result``
refuses a larger one before writing anything.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import os
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from conditions import TRUTH_FRACTIONS
from numpy.typing import ArrayLike
from run import TABLES, _concat, load_truth, read_table, truth_window_sets

import ripple_detection as rd
from ripple_detection.evaluate import COMPARISON_COLUMNS

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
        sorted: ``method``, ``setting``, ``primary_expression``, ``role``.
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
        ``method``, ``setting``, ``primary_expression``, ``n_sessions`` (the
        sessions it has scores on, which every analysis pools), ``n_failures``
        (those it has none on) and ``error``, the first failure's message.
    """
    ran = tables.ran.groupby(["method", "setting"]).size().rename("n_sessions")
    failed = tables.failures.groupby(["method", "setting"])
    counts = tables.methods[["method", "setting", "primary_expression"]].join(
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
    """

    windows: pd.DataFrame
    pairs: pd.DataFrame
    overlaps: pd.DataFrame
    false_positives: pd.DataFrame
    comparisons: pd.DataFrame
    consensus: pd.DataFrame
    false_positive_groups: pd.DataFrame


_MATCH_COLUMNS = {
    "windows": WINDOW_COLUMNS,
    "pairs": PAIR_COLUMNS,
    "overlaps": OVERLAP_COLUMNS,
    "false_positives": FALSE_POSITIVE_COLUMNS,
    "comparisons": SESSION_COMPARISON_COLUMNS,
    "consensus": CONSENSUS_COLUMNS,
    "false_positive_groups": GROUP_COLUMNS,
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
    pairs, overlaps, false_positives = [], [], []
    detected = {}
    for method, setting in ran:
        rows = by_method.get((method, setting), events.iloc[:0]).sort_values("event_index")
        bounds, index = _bounds(rows), rows["event_index"].to_numpy()
        detected[method, setting] = bounds
        key = {"session_id": session_id, "method": method, "setting": setting}
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
    match = functools.partial(match_session, primary=primary, levels=levels)
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

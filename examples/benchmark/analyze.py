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
import itertools
import os
from collections.abc import Callable, Collection
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from run import TABLES, load_truth, read_table

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
# The settings of the main analyses: detectors at their defaults, and recipes.
MAIN_SETTINGS = ("default", "literature")
REFERENCE_CONDITION = "reference"
_KEY = ["session_id", "method", "setting"]
# Rows of a large table read at once.
_CHUNK_ROWS = 1_000_000
# Up to this many sessions the sign-flip null is enumerated.
_EXACT_UP_TO = 16


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

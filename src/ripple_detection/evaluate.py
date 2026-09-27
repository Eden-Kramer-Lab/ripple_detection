"""Comparing event inventories, against a truth or against each other.

:func:`match_events` pairs two inventories one-to-one and measures each pair:
how much the events overlap and how far, signed, the detected onset and
offset fall from the reference's. :func:`compare_detectors` summarizes every
pair of methods on one recording, :func:`consensus_counts` counts the methods
that found each true event, and :func:`label_by_overlap` names each event by
the window it overlaps longest.

Every function takes events as an ``(n_events, 2)`` array of
``[start_time, end_time]`` or a DataFrame with those columns, such as a
detector's output; rows need not be sorted, and returned indices are row
positions in the input as given. Every signed quantity is *detected minus
reference*, or *method A minus method B*: negative means earlier.
"""

import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ripple_detection._call_hints import explain_call_errors
from ripple_detection.core import (
    BoolArray,
    FloatArray,
    IntArray,
    _check_non_negative,
    _checked_bounds,
)

EventInventory: TypeAlias = ArrayLike | pd.DataFrame
"""Events as an ``(n_events, 2)`` array of ``[start_time, end_time]``, or a
DataFrame with ``start_time`` and ``end_time`` columns (and optionally
``peak_time``)."""

PAIR_COLUMNS = (
    "reference_index",
    "detected_index",
    "iou",
    "coverage",
    "temporal_precision",
    "onset_error",
    "offset_error",
    "peak_error",
)
"""The columns of :attr:`EventMatching.pairs`, in order."""

COMPARISON_COLUMNS = (
    "method_a",
    "method_b",
    "n_a",
    "n_b",
    "n_matched",
    "jaccard",
    "median_iou",
    "median_onset_difference",
    "onset_difference_iqr",
    "fraction_a_earlier_onset",
    "median_offset_difference",
    "offset_difference_iqr",
    "fraction_a_earlier_offset",
    "jaccard_true",
    "jaccard_false",
    "n_shared_truth",
    "jaccard_truth_ids",
    "onset_error_correlation",
    "offset_error_correlation",
)
"""The columns :func:`compare_detectors` returns, in order."""

_MINIMUM_SHARED_FOR_CORRELATION = 3
"""Shared truth events below which an error correlation is NaN."""


def _peaks(events: EventInventory, n_events: int) -> FloatArray:
    """The ``peak_time`` column as floats, or NaN for an input without one."""
    if isinstance(events, pd.DataFrame) and "peak_time" in events.columns:
        return np.asarray(events["peak_time"], dtype=float)
    return np.full(n_events, np.nan)


def _index(events: EventInventory, n_events: int) -> pd.Index:
    """A DataFrame's own index, so a result assigns back to it by label, or
    the row positions of an array."""
    if isinstance(events, pd.DataFrame):
        return events.index
    return pd.RangeIndex(n_events)


def _ulp(*bounds: FloatArray) -> float:
    """A unit in the last place (ulp) of the largest magnitude among the
    bounds: each is stored to half of it, 1.2e-7 s on a Unix clock."""
    scale = max((float(np.abs(b).max()) for b in bounds if b.size), default=0.0)
    return float(np.spacing(scale))


def _time_rounding(*bounds: FloatArray) -> float:
    """How far a length measured between two of these bounds can round from
    its nominal value, with room: each bound is stored to half an ulp, and
    each subtraction adds about one more."""
    return 8 * _ulp(*bounds)


def _overlapping_pairs(
    first: FloatArray, second: FloatArray
) -> tuple[IntArray, IntArray, FloatArray]:
    """Every pair of a row of `first` and a row of `second` that overlap, as
    the two row positions and the intersection length, ordered by `first`'s
    row and then `second`'s. Touching is not overlap: a difference of two
    floats is positive exactly when the first is larger.

    Only rows of `second` starting before a `first` row ends, and no earlier
    than its start less the longest `second` event (widened by the rounding
    of that length), are candidates, so the work grows with the events and
    their overlaps rather than with ``len(first) * len(second)``."""
    empty = np.empty(0, dtype=int)
    if not (len(first) and len(second)):
        return empty, empty, np.empty(0)
    by_start = np.argsort(second[:, 0], kind="stable")
    starts = second[by_start, 0]
    reach = float((second[:, 1] - second[:, 0]).max()) + _time_rounding(first, second)
    low = np.searchsorted(starts, first[:, 0] - reach, side="left")
    high = np.searchsorted(starts, first[:, 1], side="left")
    counts = np.maximum(high - low, 0)
    first_row = np.repeat(np.arange(len(first)), counts)
    offset = np.arange(len(first_row)) - np.repeat(np.cumsum(counts) - counts, counts)
    second_row = by_start[low[first_row] + offset]
    start = np.maximum(first[first_row, 0], second[second_row, 0])
    end = np.minimum(first[first_row, 1], second[second_row, 1])
    intersection = end - start
    keep = intersection > 0
    first_row, second_row, intersection = first_row[keep], second_row[keep], intersection[keep]
    order = np.lexsort((second_row, first_row))
    return first_row[order], second_row[order], intersection[order]


def _check_mapping(events: object) -> None:
    """Raise ``TypeError`` unless `events` maps method names to inventories: a
    DataFrame iterates over its columns, and would compare them as methods."""
    if not isinstance(events, Mapping):
        msg = (
            "events must map each method's name to its events, such as "
            f"{{'Kay': kay_events, 'HSE': hse_events}}; got {type(events).__name__}."
        )
        raise TypeError(msg)


def _check_minimum_iou(minimum_iou: float) -> None:
    """Raise unless ``0 <= minimum_iou < 1``: at 1 or above no pair, not even
    identical events, could exceed it."""
    _check_non_negative(minimum_iou=minimum_iou)
    if not minimum_iou < 1:
        msg = f"minimum_iou must be below 1, the largest IoU; got {minimum_iou}."
        raise ValueError(msg)


def _fraction(numerator: int, denominator: int) -> float:
    """``numerator / denominator``, NaN for a denominator of 0."""
    return numerator / denominator if denominator else np.nan


def _ratio(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    """``numerator / denominator``, NaN where the denominator is 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def _assign(
    n_reference: int, reference_row: IntArray, detected_row: IntArray, iou: FloatArray
) -> IntArray:
    """The one-to-one assignment with the most pairs and then the largest
    summed IoU, among the eligible pairs given as edges (row positions and
    IoU): the chosen edges' positions, sorted by reference row.

    Solved exactly per connected component of the edges. Each pair weighs
    ``bonus + iou`` with ``bonus`` above the component's largest possible
    summed IoU, so no gain in IoU outweighs one more pair. The maximum pair
    count does not depend on which side is the reference, so the counts are
    symmetric even where summed IoUs tie."""
    if not len(reference_row):
        return np.empty(0, dtype=int)
    n_detected = int(detected_row.max()) + 1
    graph = coo_matrix(
        (np.ones(len(reference_row)), (reference_row, n_reference + detected_row)),
        shape=(n_reference + n_detected, n_reference + n_detected),
    )
    n_components, label = connected_components(graph, directed=False)
    component = label[reference_row]
    n_edges = np.bincount(component, minlength=n_components)
    # a component of one edge is one reference and one detected event
    chosen = [np.flatnonzero(n_edges[component] == 1)]
    by_component = np.argsort(component, kind="stable")
    first_edge = np.concatenate([[0], np.cumsum(n_edges)])
    for c in np.flatnonzero(n_edges > 1):
        edges = by_component[first_edge[c] : first_edge[c + 1]]
        r, r_at = np.unique(reference_row[edges], return_inverse=True)
        d, d_at = np.unique(detected_row[edges], return_inverse=True)
        weight = np.zeros((len(r), len(d)))
        weight[r_at, d_at] = min(len(r), len(d)) + 1.0 + iou[edges]
        edge_at = np.full((len(r), len(d)), -1)
        edge_at[r_at, d_at] = edges
        i, j = linear_sum_assignment(weight, maximize=True)
        chosen.append(edge_at[i, j][edge_at[i, j] >= 0])
    selected = np.concatenate(chosen)
    return selected[np.argsort(reference_row[selected], kind="stable")]


@dataclass(frozen=True, eq=False)
class EventMatching:
    """A one-to-one matching of detected events to reference events, from
    :func:`match_events`.

    Attributes
    ----------
    reference : ndarray, shape (n_reference, 2)
        The reference events' ``[start_time, end_time]``, in input order.
    detected : ndarray, shape (n_detected, 2)
        The detected events' ``[start_time, end_time]``, in input order.
    pairs : pd.DataFrame
        One row per matched pair, sorted by ``reference_index``, with columns

        - ``reference_index``, ``detected_index``: row positions in the
          inputs, for ``.iloc``; a detector's DataFrame is labelled from 1.
        - ``iou``: intersection over union, in (0, 1].
        - ``coverage``: intersection over the reference's length, the
          fraction of the reference event found.
        - ``temporal_precision``: intersection over the detected event's
          length, the fraction of the detected event that is reference.
        - ``onset_error``: detected start minus reference start, in the
          units of the times. Positive: detected late.
        - ``offset_error``: detected end minus reference end. Positive:
          detected late, so the event runs long.
        - ``peak_error``: detected ``peak_time`` minus reference
          ``peak_time`` when both inputs are DataFrames with that column,
          else NaN.
    reference_overlaps : ndarray of int, shape (n_reference,)
        How many detected events overlap each reference event, whether or
        not they are matched.
    detected_overlaps : ndarray of int, shape (n_detected,)
        How many reference events overlap each detected event.

    """

    reference: FloatArray
    detected: FloatArray
    pairs: pd.DataFrame
    reference_overlaps: IntArray
    detected_overlaps: IntArray

    @property
    def recall(self) -> float:
        """Matched pairs over reference events; NaN with no reference events."""
        return _fraction(len(self.pairs), len(self.reference))

    @property
    def precision(self) -> float:
        """Matched pairs over detected events; NaN with no detected events."""
        return _fraction(len(self.pairs), len(self.detected))

    @property
    def f1(self) -> float:
        """``2 n_pairs / (n_reference + n_detected)``, the harmonic mean of
        recall and precision; NaN only when both inventories are empty."""
        return _fraction(2 * len(self.pairs), len(self.reference) + len(self.detected))

    @property
    def unmatched_reference(self) -> IntArray:
        """Row positions of the reference events with no match, ascending."""
        return np.setdiff1d(np.arange(len(self.reference)), self.pairs.reference_index)

    @property
    def unmatched_detected(self) -> IntArray:
        """Row positions of the detected events with no match, ascending."""
        return np.setdiff1d(np.arange(len(self.detected)), self.pairs.detected_index)

    @property
    def split_reference(self) -> IntArray:
        """Row positions of the reference events that two or more detected
        events overlap: split by the detector. Counts any overlap, so a
        fragment below ``minimum_iou`` still splits."""
        return np.flatnonzero(self.reference_overlaps >= 2)

    @property
    def merged_detected(self) -> IntArray:
        """Row positions of the detected events that overlap two or more
        reference events: merged by the detector."""
        return np.flatnonzero(self.detected_overlaps >= 2)

    def boundary_errors(self, reference: EventInventory) -> pd.DataFrame:
        """Onset and offset errors of the same pairs against other bounds for
        the same reference events.

        How errors at another definition of the truth's bounds are measured
        without matching again: match at one, then pass the other.

        Parameters
        ----------
        reference : array_like, shape (n_reference, 2), or pd.DataFrame
            Other ``[start_time, end_time]`` for the reference events, row for
            row as they were matched, such as ``truth_windows`` at another
            ``fraction``.

        Returns
        -------
        errors : pd.DataFrame
            One row per pair, in the order of :attr:`pairs`, with columns
            ``reference_index``, ``detected_index``, ``onset_error`` (detected
            start minus the new reference start) and ``offset_error``
            (detected end minus the new reference end). Positive: detected
            late.

        Raises
        ------
        ValueError
            If `reference` has a different number of rows from the matched
            reference, or a start or end that is not finite or out of order.

        """
        bounds = _checked_bounds(reference, "reference")
        if len(bounds) != len(self.reference):
            msg = (
                f"reference has {len(bounds)} events; the matching was made with "
                f"{len(self.reference)}, and the bounds must be for the same events, "
                "row for row."
            )
            raise ValueError(msg)
        r = self.pairs.reference_index.to_numpy()
        d = self.pairs.detected_index.to_numpy()
        return pd.DataFrame(
            {
                "reference_index": r,
                "detected_index": d,
                "onset_error": self.detected[d, 0] - bounds[r, 0],
                "offset_error": self.detected[d, 1] - bounds[r, 1],
            }
        )


@explain_call_errors
def match_events(
    reference: EventInventory,
    detected: EventInventory,
    *,
    minimum_iou: float = 0.0,
) -> EventMatching:
    """Match detected events to reference events one-to-one.

    Two events overlap when their intersection has positive length: events
    that touch at an endpoint do not. Among the overlapping pairs whose
    intersection over union (IoU) exceeds `minimum_iou`, the matching keeps
    the most pairs, each event in at most one, and among those the largest
    summed IoU, solved exactly (not greedily). The pair count is the same
    with the inventories swapped, so recall and precision exchange and F1 is
    unchanged. Where two assignments tie on both, which one is returned is
    unspecified; the counts and summaries are not.

    Parameters
    ----------
    reference : array_like, shape (n_reference, 2), or pd.DataFrame
        The events to recover, such as the truth or another method's events:
        ``[start_time, end_time]`` per row, or a DataFrame with those columns
        and optionally ``peak_time``. Need not be sorted.
    detected : array_like, shape (n_detected, 2), or pd.DataFrame
        The events to score, in the same form.
    minimum_iou : float, optional
        IoU a pair must exceed to be matched, in [0, 1). Default 0.0: any
        overlap. An IoU within the timestamps' rounding of `minimum_iou`
        counts as equal to it, and is not matched.

    Returns
    -------
    matching : EventMatching
        The pairs with their overlap and signed errors (detected minus
        reference; positive is late), the overlap counts, and recall,
        precision, F1, the unmatched, split and merged events as properties.
        Every index is a row position in the input, for ``.iloc``.

    Raises
    ------
    ValueError
        If either input is not ``(n_events, 2)``, or has a start or end that
        is not finite or an event that ends before it starts (naming the input
        and row), or `minimum_iou` is outside [0, 1).

    Notes
    -----
    Works on the overlapping pairs, not every pair: time and memory grow with
    the events and their overlaps rather than with their product.

    Examples
    --------
    >>> truth = np.array([(0.0, 1.0), (2.0, 3.0)])
    >>> found = np.array([(0.2, 1.1), (5.0, 5.5)])
    >>> matching = match_events(truth, found)
    >>> list(matching.pairs.columns)  # doctest: +NORMALIZE_WHITESPACE
    ['reference_index', 'detected_index', 'iou', 'coverage', 'temporal_precision',
     'onset_error', 'offset_error', 'peak_error']
    >>> bool(matching.pairs.onset_error.iloc[0] > 0)  # found late
    True
    >>> matching.unmatched_reference
    array([1])

    """
    _check_minimum_iou(minimum_iou)
    reference_bounds = _checked_bounds(reference, "reference")
    detected_bounds = _checked_bounds(detected, "detected")
    reference_peaks = _peaks(reference, len(reference_bounds))
    detected_peaks = _peaks(detected, len(detected_bounds))

    reference_row, detected_row, intersection = _overlapping_pairs(
        reference_bounds, detected_bounds
    )
    reference_length = reference_bounds[:, 1] - reference_bounds[:, 0]
    detected_length = detected_bounds[:, 1] - detected_bounds[:, 0]
    # every pair overlaps, so its union is positive
    union = reference_length[reference_row] + detected_length[detected_row] - intersection
    iou = intersection / union
    eligible = np.ones(len(iou), dtype=bool)
    if minimum_iou > 0:
        # IoU's rounding, with u an ulp of the largest bound, each stored to
        # u/2: the intersection rounds by at most 1.5u and the union by 6.5u
        # (two lengths, the intersection and two sums), so IoU moves by at
        # most (1.5u + IoU 6.5u) / union <= 8u / union
        rounding = _time_rounding(reference_bounds, detected_bounds)
        eligible = iou > minimum_iou + rounding * (1.0 / union)

    edges = np.flatnonzero(eligible)[
        _assign(
            len(reference_bounds),
            reference_row[eligible],
            detected_row[eligible],
            iou[eligible],
        )
    ]
    r, d = reference_row[edges], detected_row[edges]
    pairs = pd.DataFrame(
        {
            "reference_index": r,
            "detected_index": d,
            "iou": iou[edges],
            "coverage": _ratio(intersection[edges], reference_length[r]),
            "temporal_precision": _ratio(intersection[edges], detected_length[d]),
            "onset_error": detected_bounds[d, 0] - reference_bounds[r, 0],
            "offset_error": detected_bounds[d, 1] - reference_bounds[r, 1],
            "peak_error": detected_peaks[d] - reference_peaks[r],
        },
        columns=list(PAIR_COLUMNS),
    )
    return EventMatching(
        reference=reference_bounds,
        detected=detected_bounds,
        pairs=pairs,
        reference_overlaps=np.bincount(reference_row, minlength=len(reference_bounds)),
        detected_overlaps=np.bincount(detected_row, minlength=len(detected_bounds)),
    )


def _jaccard(n_a: int, n_b: int, n_matched: int) -> float:
    """Matched events over the events in either inventory; NaN for none."""
    return _fraction(n_matched, n_a + n_b - n_matched)


def _median(values: FloatArray) -> float:
    """The median, NaN for no values."""
    return float(np.median(values)) if len(values) else np.nan


def _signed_summary(kind: str, difference: FloatArray) -> dict[str, float]:
    """Median, interquartile range and fraction negative (A earlier) of the
    signed differences A minus B; NaN for none."""
    if not len(difference):
        return {
            f"median_{kind}_difference": np.nan,
            f"{kind}_difference_iqr": np.nan,
            f"fraction_a_earlier_{kind}": np.nan,
        }
    q25, q50, q75 = np.percentile(difference, [25, 50, 75])
    return {
        f"median_{kind}_difference": float(q50),
        f"{kind}_difference_iqr": float(q75 - q25),
        f"fraction_a_earlier_{kind}": float(np.mean(difference < 0)),
    }


def _tied_ranks(values: FloatArray, tolerance: float) -> FloatArray:
    """Ranks from 1, averaged over each group of sorted values within
    `tolerance` of the group's smallest: errors read off sample times are
    whole samples that round apart by a few ulps, and are ties, not an order.
    Measuring from the smallest, not from each neighbour, keeps a run of
    values closer than `tolerance` in steps from tying as one group."""
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    group = np.zeros(len(values), dtype=int)
    start = 0
    for i in range(1, len(ordered)):
        if ordered[i] - ordered[start] > tolerance:
            start = i
            group[i] = group[i - 1] + 1
        else:
            group[i] = group[i - 1]
    position = np.arange(1.0, len(values) + 1)
    mean_rank = np.bincount(group, weights=position) / np.bincount(group)
    ranks = np.empty(len(values))
    ranks[order] = mean_rank[group]
    return ranks


def _spearman(x: FloatArray, y: FloatArray, tolerance: float) -> float:
    """Spearman's rank correlation, values within `tolerance` of each other
    tied; NaN for fewer than three values, or when either is constant and a
    rank correlation is undefined."""
    if len(x) < _MINIMUM_SHARED_FOR_CORRELATION:
        return np.nan
    x_ranks, y_ranks = _tied_ranks(x, tolerance), _tied_ranks(y, tolerance)
    if np.ptp(x_ranks) == 0 or np.ptp(y_ranks) == 0:
        return np.nan
    return float(np.corrcoef(x_ranks, y_ranks)[0, 1])


def _matched_rows(matching: EventMatching) -> BoolArray:
    """Which detected events of a matching are matched."""
    matched = np.zeros(len(matching.detected), dtype=bool)
    matched[matching.pairs.detected_index.to_numpy()] = True
    return matched


def _truth_columns(a: EventMatching, b: EventMatching, minimum_iou: float) -> dict[str, float]:
    """The columns of :func:`compare_detectors` that need the truth, from each
    method matched to it (the truth the reference of both)."""
    true_a, true_b = _matched_rows(a), _matched_rows(b)
    both_true = match_events(a.detected[true_a], b.detected[true_b], minimum_iou=minimum_iou)
    both_false = match_events(
        a.detected[~true_a], b.detected[~true_b], minimum_iou=minimum_iou
    )
    errors_a = a.pairs.set_index("reference_index")
    errors_b = b.pairs.set_index("reference_index")
    shared = np.intersect1d(errors_a.index, errors_b.index)
    errors_a, errors_b = errors_a.loc[shared], errors_b.loc[shared]
    # an error is two bounds, each within half an ulp u, and a subtraction
    # within u/2, so it is within 1.5u of its nominal value and two errors
    # that are nominally equal are within 3u of each other
    tolerance = 3 * _ulp(a.reference, a.detected, b.detected)
    return {
        "jaccard_true": _jaccard(int(true_a.sum()), int(true_b.sum()), len(both_true.pairs)),
        "jaccard_false": _jaccard(
            int((~true_a).sum()), int((~true_b).sum()), len(both_false.pairs)
        ),
        "n_shared_truth": len(shared),
        "jaccard_truth_ids": _jaccard(len(a.pairs), len(b.pairs), len(shared)),
        "onset_error_correlation": _spearman(
            errors_a.onset_error.to_numpy(), errors_b.onset_error.to_numpy(), tolerance
        ),
        "offset_error_correlation": _spearman(
            errors_a.offset_error.to_numpy(), errors_b.offset_error.to_numpy(), tolerance
        ),
    }


@explain_call_errors
def compare_detectors(
    events: Mapping[str, EventInventory],
    *,
    truth: EventInventory | None = None,
    minimum_iou: float = 0.0,
) -> pd.DataFrame:
    """Pairwise agreement between methods' events on one recording.

    Each pair of methods is matched one-to-one as :func:`match_events` does,
    the first method's events as the reference. With a `truth`, each method
    is also matched to it, which splits the agreement into agreement on true
    events and on false ones, and correlates the two methods' errors on the
    true events both found.

    Parameters
    ----------
    events : mapping of str to array_like of shape (n_events, 2), or pd.DataFrame
        Each method's events, by name.
    truth : array_like, shape (n_true, 2), or pd.DataFrame, optional
        The true events. Without it the truth columns are NaN.
    minimum_iou : float, optional
        IoU a pair must exceed to be matched, in [0, 1), for every matching.
        Default 0.0: any overlap.

    Returns
    -------
    comparison : pd.DataFrame
        One row per unordered pair ``(a, b)``, ``a`` before ``b`` in the
        order of `events`, with columns

        - ``method_a``, ``method_b``; ``n_a``, ``n_b`` their event counts;
          ``n_matched`` the pairs matched.
        - ``jaccard``: ``n_matched / (n_a + n_b - n_matched)``; NaN when both
          are empty.
        - ``median_iou``: over matched pairs.
        - ``median_onset_difference``, ``onset_difference_iqr``: of ``a``'s
          start minus ``b``'s over matched pairs. Negative: ``a`` earlier.
        - ``fraction_a_earlier_onset``: of matched pairs, those where ``a``
          starts strictly before ``b``.
        - ``median_offset_difference``, ``offset_difference_iqr``,
          ``fraction_a_earlier_offset``: the same for the ends.
        - ``jaccard_true``, ``jaccard_false``: ``jaccard`` between the two
          methods' events that matched a true event, and between those that
          matched none: whether their detected intervals agree, which they
          can on different true events.
        - ``n_shared_truth``: true events both methods matched.
        - ``jaccard_truth_ids``: whether they found the same true events,
          ``n_shared_truth`` over the true events either matched; NaN when
          neither matched one.
        - ``onset_error_correlation``, ``offset_error_correlation``:
          Spearman's correlation over the shared true events of the two
          methods' signed errors against the truth, errors within the
          timestamps' rounding of each other tied (as whole samples read off
          the timestamps are); NaN below three shared events or when either
          method's errors are all equal.

        Summaries over no pairs are NaN. The truth columns are present, and
        NaN, without `truth`. Empty, with these columns, for fewer than two
        methods.

    Raises
    ------
    ValueError
        If an inventory is not ``(n_events, 2)`` or has a start or end that is
        not finite or out of order (naming it and the row), or `minimum_iou`
        is outside [0, 1).
    TypeError
        If `events` is not a mapping, such as a single DataFrame.

    Examples
    --------
    >>> truth = np.array([(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)])
    >>> events = {
    ...     "early": np.array([(-0.1, 0.9), (1.9, 2.9), (7.0, 7.2)]),
    ...     "late": np.array([(0.1, 1.1), (2.1, 3.1)]),
    ... }
    >>> comparison = compare_detectors(events, truth=truth)
    >>> list(comparison.columns[:6])
    ['method_a', 'method_b', 'n_a', 'n_b', 'n_matched', 'jaccard']
    >>> bool(comparison.median_onset_difference.iloc[0] < 0)  # early starts first
    True

    """
    _check_mapping(events)
    _check_minimum_iou(minimum_iou)
    names = list(events)
    bounds = {name: _checked_bounds(events[name], f"events[{name!r}]") for name in names}
    to_truth: dict[str, EventMatching] = {}
    if truth is not None:
        truth_bounds = _checked_bounds(truth, "truth")
        to_truth = {
            name: match_events(truth_bounds, bounds[name], minimum_iou=minimum_iou)
            for name in names
        }
    rows = []
    for a, b in itertools.combinations(names, 2):
        matching = match_events(bounds[a], bounds[b], minimum_iou=minimum_iou)
        # errors are b minus a; the differences are a minus b
        onset = -matching.pairs.onset_error.to_numpy()
        offset = -matching.pairs.offset_error.to_numpy()
        row: dict[str, object] = {
            "method_a": a,
            "method_b": b,
            "n_a": len(bounds[a]),
            "n_b": len(bounds[b]),
            "n_matched": len(matching.pairs),
            "jaccard": _jaccard(len(bounds[a]), len(bounds[b]), len(matching.pairs)),
            "median_iou": _median(matching.pairs.iou.to_numpy()),
            **_signed_summary("onset", onset),
            **_signed_summary("offset", offset),
        }
        if to_truth:
            row |= _truth_columns(to_truth[a], to_truth[b], minimum_iou)
        rows.append(row)
    return pd.DataFrame(rows, columns=list(COMPARISON_COLUMNS))


@explain_call_errors
def consensus_counts(
    events: Mapping[str, EventInventory],
    truth: EventInventory,
    *,
    minimum_iou: float = 0.0,
) -> pd.DataFrame:
    """Which methods found each true event, and how many.

    Parameters
    ----------
    events : mapping of str to array_like of shape (n_events, 2), or pd.DataFrame
        Each method's events, by name. No name may be ``"n_methods"``.
    truth : array_like, shape (n_true, 2), or pd.DataFrame
        The true events.
    minimum_iou : float, optional
        IoU a pair must exceed to be matched, in [0, 1), as in
        :func:`match_events`. Default 0.0: any overlap.

    Returns
    -------
    consensus : pd.DataFrame
        One row per true event, on `truth`'s index when it is a DataFrame (so
        the columns assign back to it) and its row positions otherwise: one
        boolean column per method, in the order of `events`, true where that
        method's events matched the true event one-to-one, and ``n_methods``,
        how many did.

    Raises
    ------
    ValueError
        If a method is named ``"n_methods"``, an inventory is not
        ``(n_events, 2)`` or has a start or end that is not finite or out of
        order, or `minimum_iou` is outside [0, 1).
    TypeError
        If `events` is not a mapping, such as a single DataFrame.

    Examples
    --------
    >>> truth = np.array([(0.0, 1.0), (2.0, 3.0)])
    >>> events = {"a": np.array([(0.1, 0.9)]), "b": np.array([(0.0, 1.0), (2.5, 2.6)])}
    >>> consensus = consensus_counts(events, truth)
    >>> list(consensus.columns)
    ['a', 'b', 'n_methods']
    >>> consensus.b.tolist()
    [True, True]

    """
    _check_mapping(events)
    _check_minimum_iou(minimum_iou)
    if "n_methods" in events:
        msg = "No method may be named 'n_methods', the column that counts them."
        raise ValueError(msg)
    truth_bounds = _checked_bounds(truth, "truth")
    found = {}
    for name, inventory in events.items():
        matching = match_events(
            truth_bounds,
            _checked_bounds(inventory, f"events[{name!r}]"),
            minimum_iou=minimum_iou,
        )
        hit = np.zeros(len(truth_bounds), dtype=bool)
        hit[matching.pairs.reference_index.to_numpy()] = True
        found[name] = hit
    consensus = pd.DataFrame(found, index=_index(truth, len(truth_bounds)))
    consensus["n_methods"] = consensus.sum(axis=1).astype(int)
    return consensus


@explain_call_errors
def label_by_overlap(
    events: EventInventory,
    windows: pd.DataFrame,
    *,
    unlabeled: str = "background",
) -> pd.Series:
    """Name each event by the window it overlaps longest.

    Which kind of event a method found: pass windows labelled by type, such
    as ``truth_windows`` output with its ``type`` column renamed ``label``.

    Parameters
    ----------
    events : array_like, shape (n_events, 2), or pd.DataFrame
        The events to label.
    windows : pd.DataFrame
        ``start_time``, ``end_time`` and ``label`` columns, one row per
        window. Windows may overlap each other.
    unlabeled : str, optional
        The label of an event that overlaps no window. Default
        ``"background"``.

    Returns
    -------
    labels : pd.Series
        Named ``label``, one entry per event, on `events`' index when it is a
        DataFrame (so it assigns back to it) and its row positions otherwise:
        the ``label`` of the window the event overlaps longest, the earlier
        window row on a tie (overlaps within the timestamps' rounding of each
        other tie), or `unlabeled`. Touching at an endpoint is not overlap.

    Raises
    ------
    ValueError
        If `windows` has no ``label`` column, or either input is not
        ``(n_events, 2)`` or has a start or end that is not finite or out of
        order.

    Examples
    --------
    >>> windows = pd.DataFrame(
    ...     {"start_time": [0.0, 0.5], "end_time": [1.0, 3.0], "label": ["swr", "emg"]}
    ... )
    >>> label_by_overlap(np.array([(0.2, 0.8), (0.9, 2.0), (5.0, 6.0)]), windows).tolist()
    ['swr', 'emg', 'background']

    """
    if "label" not in windows.columns:
        msg = (
            f"windows needs a 'label' column; it has {list(windows.columns)}. For "
            "truth_windows output, rename 'type' to 'label'."
        )
        raise ValueError(msg)
    bounds = _checked_bounds(events, "events")
    window_bounds = _checked_bounds(windows, "windows")
    labels = np.full(len(bounds), unlabeled, dtype=object)
    event, window, overlap = _overlapping_pairs(bounds, window_bounds)
    longest = np.zeros(len(bounds))
    np.maximum.at(longest, event, overlap)
    tied = overlap >= longest[event] - _time_rounding(bounds, window_bounds)
    # the earliest window row among each event's longest overlaps
    best = np.full(len(bounds), len(window_bounds))
    np.minimum.at(best, event[tied], window[tied])
    has_overlap = longest > 0
    labels[has_overlap] = windows["label"].to_numpy()[best[has_overlap]]
    return pd.Series(labels, index=_index(events, len(bounds)), name="label")

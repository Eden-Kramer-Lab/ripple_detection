"""Which units fire in each event, and the participation criteria built on it."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ripple_detection.core import (
    _NO_TIME_SAMPLES,
    BoolArray,
    FloatArray,
    IntArray,
    _bounds_frame,
    _event_bounds,
    _gap_tolerance,
    _samples_within,
    _warn_at_caller,
    sample_count_within,
)
from ripple_detection.detectors._validation import _check_whole_number, _validate_multiunit


def _selected_units(units: ArrayLike | None, n_units: int) -> IntArray:
    """Column indices of the units that count: all of them, a boolean mask of
    length ``n_units``, or integer indices."""
    if units is None:
        return np.arange(n_units)
    selection = np.asarray(units)
    if selection.dtype == bool:
        if selection.shape != (n_units,):
            msg = (
                f"units as a boolean mask must have one entry per unit, {n_units}; got "
                f"shape {selection.shape}."
            )
            raise ValueError(msg)
        return np.flatnonzero(selection)
    if not np.issubdtype(selection.dtype, np.integer) or selection.ndim != 1:
        msg = "units must be a boolean mask over the units or a 1-D array of unit indices."
        raise ValueError(msg)
    if selection.size and (selection.min() < 0 or selection.max() >= n_units):
        msg = f"units holds indices outside 0 to {n_units - 1}, the units multiunit has."
        raise ValueError(msg)
    return np.unique(selection)


def _warn_if_events_hold_missing(
    is_missing: BoolArray, first: IntArray, last: IntArray
) -> None:
    """Warn, once, with how many events ``[first, last)`` hold a missing sample."""
    missing_before = np.concatenate([[0], np.cumsum(is_missing)])
    n_affected = int(np.count_nonzero(missing_before[last] > missing_before[first]))
    if n_affected:
        _warn_at_caller(
            f"{n_affected} of {len(first)} event(s) hold missing (NaN) multiunit samples, "
            "which count no spike, so their counts may be low."
        )


def count_spikes_in_events(
    event_times: ArrayLike | pd.DataFrame,
    multiunit: ArrayLike,
    time: ArrayLike,
) -> IntArray:
    """Spikes each unit fires inside each event.

    The table the participation criteria of published detectors are read
    from: units active (``(counts > 0).sum(axis=1)``), active units of a
    subset such as place cells (``(counts[:, place_cells] > 0).sum(axis=1)``),
    or total spikes (``counts.sum(axis=1)``). :func:`require_active_units`
    applies the common ones.

    Parameters
    ----------
    event_times : array_like, shape (n_events, 2), or pd.DataFrame
        ``[start_time, end_time]`` per event, or a detector's DataFrame. An
        event holds the samples with ``start_time <= time <= end_time``.
    multiunit : array_like, shape (n_time, n_units)
        Spike counts or indicators per sample, non-negative whole numbers. A
        NaN is a missing sample and counts no spike, with a warning.
    time : array_like, shape (n_time,)
        Sample timestamps in seconds, increasing.

    Returns
    -------
    counts : ndarray of int, shape (n_events, n_units)

    Raises
    ------
    ValueError
        If `multiunit` is not 2-D or holds values that are not spike counts,
        its length differs from `time`'s, or no sample falls within an event.

    Warns
    -----
    UserWarning
        If any event holds a missing (NaN) sample, with the number of such
        events: their counts leave out whatever fired there.

    Examples
    --------
    >>> time = np.arange(10) / 10
    >>> multiunit = np.zeros((10, 3))
    >>> multiunit[[1, 2], 0] = 1
    >>> multiunit[6, 2] = 1
    >>> count_spikes_in_events(np.array([(0.0, 0.3), (0.5, 0.9)]), multiunit, time)
    array([[2, 0, 0],
           [0, 0, 1]])

    """
    spikes = np.asarray(multiunit, dtype=float)
    _validate_multiunit(spikes)
    time = np.asarray(time, dtype=float)
    if len(spikes) != len(time):
        msg = f"multiunit has {len(spikes)} samples and time {len(time)}; they must match."
        raise ValueError(msg)
    events = _event_bounds(event_times)
    first, last = _samples_within(events, time, _NO_TIME_SAMPLES)
    _warn_if_events_hold_missing(np.isnan(spikes).any(axis=1), first, last)
    counts = np.zeros((len(events), spikes.shape[1]), dtype=int)
    for event, (a, b) in enumerate(zip(first, last, strict=True)):
        counts[event] = np.nansum(spikes[a:b], axis=0).astype(int)
    return counts


def require_active_units(
    event_times: ArrayLike | pd.DataFrame,
    multiunit: ArrayLike,
    time: ArrayLike,
    *,
    minimum_active_units: int = 1,
    minimum_active_fraction: float | None = None,
    minimum_spikes: int | None = None,
    units: ArrayLike | None = None,
) -> FloatArray | pd.DataFrame:
    """Keep the events in which enough units fire.

    The participation criteria of published detectors, applied to any event
    inventory: "at least 5 place cells active", "at least 15% of the place
    cells", "at least 5 or 15% of pyramidal cells, whichever is larger" (pass
    both), "at least 3 cells firing at least 5 spikes in total". A unit is
    active in an event when it fires at least one spike there.

    Applied after detection, this runs after a detector's close-event rule,
    so a dropped event may already have suppressed a neighbour; the
    detectors' own ``minimum_active_units`` runs before it.

    Parameters
    ----------
    event_times : array_like, shape (n_events, 2), or pd.DataFrame
        ``[start_time, end_time]`` per event, or a detector's DataFrame,
        returned filtered with every column and its index.
    multiunit : array_like, shape (n_time, n_units)
        Spike counts or indicators per sample, as for the detectors.
    time : array_like, shape (n_time,)
        Sample timestamps in seconds, increasing.
    minimum_active_units : int, optional
        Least number of the selected units active. Default 1.
    minimum_active_fraction : float, optional
        Least fraction, 0 to 1, of the selected units active. Default None.
    minimum_spikes : int, optional
        Least total spikes of the selected units. Default None.
    units : array_like, optional
        The units that count: a boolean mask over the columns of
        `multiunit`, or their indices, such as the place cells. The fraction
        is of these. Default None, every unit.

    Returns
    -------
    kept_events : ndarray, shape (n_kept, 2), or pd.DataFrame
        The events meeting every criterion given, in the input's type and
        order.

    Raises
    ------
    ValueError
        If a criterion is not a whole number or a fraction, asks for more
        units than are selected, or the inputs fail
        :func:`count_spikes_in_events`'s checks.

    Examples
    --------
    >>> time = np.arange(10) / 10
    >>> multiunit = np.zeros((10, 4))
    >>> multiunit[1, [0, 1, 2]] = 1
    >>> multiunit[6, 3] = 1
    >>> events = np.array([(0.0, 0.3), (0.5, 0.9)])
    >>> require_active_units(events, multiunit, time, minimum_active_units=3)
    array([[0. , 0.3]])
    >>> require_active_units(events, multiunit, time, units=[3])
    array([[0.5, 0.9]])

    """
    counts = count_spikes_in_events(event_times, multiunit, time)
    selected = _selected_units(units, counts.shape[1])
    _check_whole_number("minimum_active_units", minimum_active_units, 0)
    if minimum_active_units > len(selected):
        msg = (
            f"minimum_active_units is {minimum_active_units} but {len(selected)} unit(s) "
            "are selected, so no event could be kept."
        )
        raise ValueError(msg)
    if minimum_active_fraction is not None and not 0 <= minimum_active_fraction <= 1:
        msg = (
            f"minimum_active_fraction must be between 0 and 1, got {minimum_active_fraction}."
        )
        raise ValueError(msg)
    if minimum_spikes is not None:
        _check_whole_number("minimum_spikes", minimum_spikes, 0)
    chosen = counts[:, selected]
    n_active = (chosen > 0).sum(axis=1)
    keep = n_active >= minimum_active_units
    if minimum_active_fraction is not None:
        # divide rather than multiply the threshold, so 3 of 20 meets 0.15 exactly
        fraction = n_active / max(len(selected), 1)
        keep &= (fraction >= minimum_active_fraction) | np.isclose(
            fraction, minimum_active_fraction
        )
    if minimum_spikes is not None:
        keep &= chosen.sum(axis=1) >= minimum_spikes
    if isinstance(event_times, pd.DataFrame):
        return event_times.iloc[np.flatnonzero(keep)].copy()
    return _event_bounds(event_times)[keep]


def trim_events_to_spike_windows(
    event_times: ArrayLike | pd.DataFrame,
    multiunit: ArrayLike,
    time: ArrayLike,
    *,
    window: float = 0.02,
    step: float = 0.005,
    minimum_spikes: int = 2,
    units: ArrayLike | None = None,
    minimum_duration: float = 0.0,
) -> FloatArray | pd.DataFrame:
    """Move each event's bounds inward until its edge windows hold enough spikes.

    For decoding rules that need spikes in an event's first and last time
    bins, such as "boundaries adjusted inward to ensure that the first and
    last estimation bins contained a minimum of 2 spikes" with 20 ms bins
    advanced in 5 ms steps (Pfeiffer & Foster 2013). The start moves forward
    by `step` until ``[start, start + window)`` holds `minimum_spikes` spikes
    of the selected units; the end moves back by `step` until
    ``(end - window, end]`` does. An event with no such start or end, or
    whose end window would begin before its start, is dropped.

    Parameters
    ----------
    event_times : array_like, shape (n_events, 2), or pd.DataFrame
        ``[start_time, end_time]`` per event, or a detector's DataFrame.
    multiunit : array_like, shape (n_time, n_units)
        Spike counts or indicators per sample. A NaN counts no spike, with a
        warning when it lies in an event and in a selected unit.
    time : array_like, shape (n_time,)
        Sample timestamps in seconds, increasing.
    window : float, optional
        Length of each edge window in seconds. Default 0.02.
    step : float, optional
        How far a bound moves each time, in seconds. Default 0.005.
    minimum_spikes : int, optional
        Spikes each edge window must hold. Default 2.
    units : array_like, optional
        The units whose spikes count, a boolean mask or indices. Default all.
    minimum_duration : float, optional
        Trimmed events holding fewer samples than this spans are dropped.
        Default 0.0.

    Returns
    -------
    trimmed_events : ndarray, shape (n_kept, 2), or pd.DataFrame
        The trimmed bounds, in input order. For a DataFrame, a DataFrame of
        ``start_time`` and ``end_time`` under the kept events' index: its
        other columns describe the untrimmed event, so they are left out,
        and ``trimmed.join(events.drop(columns=["start_time", "end_time"]))``
        brings back any that still apply.

    Raises
    ------
    ValueError
        If `window` or `step` is not positive, `minimum_spikes` is not a
        whole number of at least 1, or the inputs fail
        :func:`count_spikes_in_events`'s checks.

    Warns
    -----
    UserWarning
        If any event holds a missing (NaN) sample of a selected unit, with
        the number of such events.

    Examples
    --------
    >>> time = np.arange(100) / 100
    >>> multiunit = np.zeros((100, 2))
    >>> multiunit[[30, 32, 60, 61], 0] = 1
    >>> events = np.array([(0.0, 0.99)])
    >>> trim_events_to_spike_windows(events, multiunit, time, window=0.05, step=0.01)
    array([[0.28, 0.64]])

    """
    for name, value in (("window", window), ("step", step)):
        if not 0 < value < np.inf:
            msg = f"{name} must be positive and finite, got {value}."
            raise ValueError(msg)
    _check_whole_number("minimum_spikes", minimum_spikes, 1)
    if minimum_duration < 0:
        msg = f"minimum_duration must be non-negative, got {minimum_duration}."
        raise ValueError(msg)
    count_spikes_in_events(np.empty((0, 2)), multiunit, time)  # validates the inputs
    spikes = np.asarray(multiunit, dtype=float)
    time = np.asarray(time, dtype=float)
    selected = spikes[:, _selected_units(units, spikes.shape[1])]
    pooled = np.nansum(selected, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(pooled)])

    events = _event_bounds(event_times)
    _warn_if_events_hold_missing(
        np.isnan(selected).any(axis=1),
        np.searchsorted(time, events[:, 0], side="left"),
        np.searchsorted(time, events[:, 1], side="right"),
    )
    # a bound reached by stepping rounds to within a few ulps of the largest
    # time, so it is compared with the samples, and the steps counted, within
    # the close-event rule's tolerance for timestamps of that magnitude
    scale = float(max(np.abs(time).max(initial=0.0), np.abs(events).max(initial=0.0)))
    tolerance = _gap_tolerance(step, scale)

    def spikes_between(low: float, high: float, closed_on: str) -> float:
        """Spikes in [low, high) or (low, high], by the cumulative count."""
        if closed_on == "left":
            a = np.searchsorted(time, low - tolerance, "left")
            b = np.searchsorted(time, high - tolerance, "left")
        else:
            a = np.searchsorted(time, low + tolerance, "right")
            b = np.searchsorted(time, high + tolerance, "right")
        return float(cumulative[b] - cumulative[a])

    trimmed, kept = [], []
    for row, (start_time, end_time) in enumerate(events):
        n_steps = int(np.floor((end_time - start_time - window + tolerance) / step))
        if n_steps < 0:
            continue
        # count steps from each bound rather than accumulating them, so the
        # window edges land on the intended times
        starts = start_time + step * np.arange(n_steps + 1)
        ends = end_time - step * np.arange(n_steps + 1)
        start = next(
            (s for s in starts if spikes_between(s, s + window, "left") >= minimum_spikes),
            None,
        )
        end = next(
            (e for e in ends if spikes_between(e - window, e, "right") >= minimum_spikes), None
        )
        if start is None or end is None or end - window < start - tolerance:
            continue
        first = int(np.searchsorted(time, start - tolerance, "left"))
        last = int(np.searchsorted(time, end + tolerance, "right")) - 1
        if not sample_count_within(last - first + 1, time, minimum_duration):
            continue
        trimmed.append((time[first], time[last]))
        kept.append(row)
    bounds = np.asarray(trimmed, dtype=float).reshape(-1, 2)
    if isinstance(event_times, pd.DataFrame):
        return _bounds_frame(bounds, event_times.index[kept])
    return bounds

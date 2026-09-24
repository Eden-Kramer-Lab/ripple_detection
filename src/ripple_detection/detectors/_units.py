"""Which units fire in each event, and the participation criteria built on it."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ripple_detection.core import FloatArray, IntArray, _event_bounds
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
        NaN is a missing sample and counts no spike.
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
    first = np.searchsorted(time, events[:, 0], side="left")
    last = np.searchsorted(time, events[:, 1], side="right")
    if np.any(last == first):
        start_time, end_time = events[np.flatnonzero(last == first)[0]]
        msg = f"No sample of time falls within event [{start_time}, {end_time}]."
        raise ValueError(msg)
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

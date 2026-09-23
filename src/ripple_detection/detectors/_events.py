"""From candidate events to the result: duration ceiling, the shared detection
tail, active-unit counts, and the per-event statistics every detector reports."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.integrate import trapezoid

from ripple_detection.core import (
    BoolArray,
    FloatArray,
    IntArray,
    _is_immobile_at_endpoints,
    exclude_close_events,
    minimum_sample_count,
    nearest_sample_index,
    normalize_signal,
    sample_count_within,
)
from ripple_detection.detectors._blocks import (
    _normalization_mask_over_valid,
    _threshold_blocks,
)


def _exclude_long_events(
    event_times: ArrayLike, time: FloatArray, maximum_duration: float | None
) -> tuple[FloatArray, BoolArray]:
    """Drop events longer than ``maximum_duration``.

    The limit applies to the event as it will be reported, after the bounds
    have been extended past the threshold crossing, because that is the
    duration a published maximum describes. ``minimum_duration`` is the other
    way round: it applies to the run above threshold, the Frank lab
    ``extractevents`` convention.

    Duration is a sample count, not elapsed time: an event is kept when it
    holds at most ``minimum_sample_count(time, maximum_duration)`` samples,
    the same rule ``minimum_duration`` uses. An event whose elapsed time is
    exactly ``maximum_duration`` holds one sample more than that and is
    dropped, so the effective ceiling is one sample short of the number given.
    At 1500 Hz a 0.5 s ceiling admits 750 samples, which span 499.3 ms.

    Applied after close events are excluded or merged, so an over-long event
    still suppresses or absorbs its neighbours before it is itself dropped.

    Parameters
    ----------
    event_times : array_like, shape (n_events, 2)
        ``[start_time, end_time]`` per event.
    time : ndarray, shape (n_time,)
        Sample timestamps in seconds.
    maximum_duration : float or None
        Longest allowed duration in seconds. None keeps every event.

    Returns
    -------
    event_times : ndarray, shape (n_kept, 2)
        The events within the limit.
    keep : ndarray, shape (n_events,)
        Boolean mask into the input, for filtering arrays that run alongside
        the events.

    """
    events = np.asarray(event_times, dtype=float).reshape(-1, 2)
    if maximum_duration is None or len(events) == 0:
        return events, np.ones(len(events), dtype=bool)
    start = nearest_sample_index(time, events[:, 0])
    stop = nearest_sample_index(time, events[:, 1])
    keep = np.asarray(sample_count_within(stop - start + 1, time, 0.0, maximum_duration))
    return events[keep], keep


def _threshold_trace(
    trace: FloatArray,
    time: FloatArray,
    is_valid: BoolArray,
    blocks: list[tuple[int, int]],
    *,
    minimum_duration: float,
    zscore_threshold: float,
    normalization_method: str = "zscore",
    normalization_mask: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Normalize one detection trace over the valid samples and threshold it.

    Within each block, the runs at or above ``zscore_threshold`` that last
    ``minimum_duration``, each extended to the normalization center.

    Returns
    -------
    normalized : ndarray, shape (n_time,)
    candidate_times : ndarray, shape (n_candidates, 2)

    """
    mask = _normalization_mask_over_valid(len(time), is_valid, normalization_mask)
    normalized = normalize_signal(trace, method=normalization_method, normalization_mask=mask)
    candidates = _threshold_blocks(
        normalized, time, blocks, minimum_duration, zscore_threshold
    )
    return normalized, np.asarray(candidates, dtype=float).reshape(-1, 2)


def _finish_events(
    event_times: FloatArray,
    keep: BoolArray,
    time: FloatArray,
    close_event_threshold: float,
    maximum_duration: float | None,
) -> tuple[FloatArray, IntArray]:
    """The last steps every detector but Zugaro, Long and Carey shares, in order.

    First the candidate criteria, given as ``keep`` (the speed rule, and a
    detector's own such as the active-unit count), then the proximity rule,
    then the duration ceiling. Every criterion runs before the proximity rule,
    so an event that fails one cannot suppress its neighbour and then vanish;
    the ceiling runs after it, deliberately (see ``_exclude_long_events``).

    Parameters
    ----------
    event_times : ndarray, shape (n_candidates, 2)
    keep : ndarray of bool, shape (n_candidates,)
    time : ndarray, shape (n_time,)
    close_event_threshold : float
    maximum_duration : float or None

    Returns
    -------
    event_times : ndarray, shape (n_events, 2)
    indices : ndarray of int, shape (n_events,)
        Position of each kept event in the input, for arrays that run
        alongside the candidates.

    """
    indices = np.flatnonzero(keep)
    kept, indices = exclude_close_events(
        event_times[indices], close_event_threshold, included_ripple_inds=indices
    )
    kept, within = _exclude_long_events(kept, time, maximum_duration)
    return kept, indices[within]


def _detect_from_trace(
    trace: FloatArray,
    time: FloatArray,
    speed: FloatArray,
    is_valid: BoolArray,
    blocks: list[tuple[int, int]],
    *,
    minimum_duration: float,
    zscore_threshold: float,
    speed_threshold: float,
    close_event_threshold: float,
    maximum_duration: float | None = None,
    normalization_method: str = "zscore",
    normalization_mask: ArrayLike | None = None,
) -> pd.DataFrame:
    """Threshold one trace, apply the endpoint speed rule and the shared last
    steps, and summarize the events: the whole of Kay and Roumis after their
    trace is formed.

    Parameters
    ----------
    trace : ndarray, shape (n_time,)
        The unnormalized detection trace, NaN outside the valid blocks.
    time, speed : ndarray, shape (n_time,)
    is_valid : ndarray of bool, shape (n_time,)
    blocks : list of (start, stop)
        From ``_valid_blocks``.
    minimum_duration, zscore_threshold, speed_threshold, close_event_threshold,
    maximum_duration, normalization_method, normalization_mask
        As in the public detectors.

    Returns
    -------
    events : pd.DataFrame
        One row per event, indexed by ``event_number``.

    """
    normalized, candidates = _threshold_trace(
        trace,
        time,
        is_valid,
        blocks,
        minimum_duration=minimum_duration,
        zscore_threshold=zscore_threshold,
        normalization_method=normalization_method,
        normalization_mask=normalization_mask,
    )
    keep = _is_immobile_at_endpoints(candidates, speed, time, speed_threshold)
    event_times, _ = _finish_events(
        candidates, keep, time, close_event_threshold, maximum_duration
    )
    return _get_event_stats(event_times, time, normalized, speed, minimum_duration, blocks)


def _count_active_units(multiunit: FloatArray, event_bounds: ArrayLike) -> IntArray:
    """Number of units with at least one spike inside each event.

    Units are counted, not spikes, so a unit bursting hard counts once. The
    interval is closed: a unit whose only spike falls on the event's last
    sample is inside it.

    A loop over events rather than a vectorized cumulative sum, because the
    cumulative form needs a second array the size of ``multiunit`` while this
    reduces one event's slice at a time, and a recording holds far more
    samples than events.

    Parameters
    ----------
    multiunit : ndarray, shape (n_time, n_units)
        Spike counts or indicators per unit.
    event_bounds : array_like, shape (n_events, 2)
        ``[first_sample, last_sample]`` per event, as indices into
        ``multiunit``. Both ends are included.

    Returns
    -------
    n_active_units : ndarray, shape (n_events,)

    """
    bounds = np.asarray(event_bounds, dtype=int).reshape(-1, 2)
    return np.array(
        [int(np.sum(multiunit[start : stop + 1].sum(axis=0) > 0)) for start, stop in bounds],
        dtype=int,
    )


def _max_sustained_zscore(
    time: np.ndarray, data: np.ndarray, minimum_duration: float = 0.015
) -> float:
    """Find the largest value sustained for a minimum duration anywhere in the event.

    The largest threshold at which the event would still be detected: the
    maximum, over every window of ``minimum_sample_count(time, minimum_duration)``
    consecutive samples, of that window's minimum. The sample-count convention
    matches event detection, so an event detected at ``zscore_threshold`` has
    ``max_sustained_zscore >= zscore_threshold`` whenever the statistic is
    computed on the trace that was thresholded (Kay, Roumis, Yu, the HSE
    detector, and Karlsson through the per-channel maximum).
    ``Shvartsman_ripple_detector`` averages over the participating channels,
    so its value can fall below the threshold. This is not the statistic of the Frank
    lab ``extractevents`` routine, which takes a window of the minimum
    duration centered on the peak and reports the lower of its two ends; the
    two differ by a few tenths of a standard deviation on typical events.

    Parameters
    ----------
    time : np.ndarray, shape (n_time,)
    data : np.ndarray, shape (n_time,)
    minimum_duration : float, optional

    Returns
    -------
    max_sustained_zscore : float
        The largest value sustained for ``minimum_duration`` within the event.
        ``nan`` if the event holds fewer samples than the minimum (the sustained
        value is then undefined). Most detectors never produce such an event,
        because their segments meet the minimum by construction.
        ``Long_sharp_wave_ripple_detector`` can: it admits an event on the
        ripple criterion alone, but reports the span of the sharp wave and
        measures this statistic against ``minimum_sharp_wave_duration``.

    """
    if len(data) < 2 and minimum_duration > 0:
        # a single sample has no measurable interval, so no duration is sustained
        return float("nan")
    n_min = minimum_sample_count(time, minimum_duration)
    if len(data) < n_min:
        return float("nan")
    windows = np.lib.stride_tricks.sliding_window_view(np.asarray(data, dtype=float), n_min)
    return float(windows.min(axis=1).max())


def _known_speed_summary(event_speed: FloatArray) -> tuple[float, float, float, float]:
    """Max, min, median and mean of the event's known (finite) speeds; NaN
    for each when no speed inside the event is known."""
    known = event_speed[np.isfinite(event_speed)]
    if known.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    return (
        float(known.max()),
        float(known.min()),
        float(np.median(known)),
        float(known.mean()),
    )


def _get_event_stats(
    event_times: ArrayLike,
    time: ArrayLike,
    zscore_metric: ArrayLike,
    speed: ArrayLike,
    minimum_duration: float,
    blocks: list[tuple[int, int]],
    *,
    participants: ArrayLike | None = None,
    clipped: BoolArray | None = None,
) -> pd.DataFrame:
    """Compute comprehensive statistics for detected events.

    Calculates temporal, z-score, signal, and speed metrics for each event.
    An event's samples are those with ``start_time <= time <= end_time``,
    found by bisection, so the cost does not grow with the recording length.

    Parameters
    ----------
    event_times : array_like, shape (n_events, 2)
        Array of [start_time, end_time] for each event.
    time : array_like, shape (n_time,)
        Time values for each sample, increasing.
    zscore_metric : array_like, shape (n_time,) or (n_time, n_channels)
        Signal the per-event statistics (mean/median/max/min z-score, area,
        total_energy, max_sustained_zscore) are computed from. Its exact meaning depends on
        the caller -- e.g. the consensus trace for Kay, the per-channel maximum for
        Karlsson, or the multiunit firing rate for multiunit_HSE. When participants
        is None, pass a single 1-D trace of shape (n_time,). When participants is
        provided, pass the per-channel signal of shape (n_time, n_channels) so that
        each event's metrics are computed from its participating channels only.
    speed : array_like, shape (n_time,)
        Animal's speed at each time point.
    minimum_duration : float
        Minimum duration for the max_sustained_zscore calculation.
    blocks : list of (start, stop)
        The valid blocks the events were found in (``_valid_blocks``), for
        the ``clipped_start`` and ``clipped_end`` flags.
    participants : array_like of tuple, shape (n_events,), optional
        Channels that participate in each event; z-score metrics are
        averaged over these channels, and ``n_participants`` and
        ``frac_participants`` (of the channels in ``zscore_metric``) are
        reported. Used by Shvartsman_ripple_detector.
    clipped : ndarray of bool, shape (n_events, 2), optional
        Flags to report instead of the ones derived from ``blocks``, for a
        detector whose segmentation knows better whether an event was cut
        off (Zugaro). Default is None.

    Returns
    -------
    event_stats : pd.DataFrame
        One row per event, indexed by ``event_number`` from 1, with columns:
        - start_time, end_time: Event boundaries
        - duration: Event duration (end - start), the elapsed time between the
            first and last sample, one sample interval less than n_samples
            spans; the duration limits are sample counts, so an event of
            exactly the minimum count has a duration below minimum_duration
            by half to one and a half intervals, as the rounding falls (23
            samples, 14.67 ms, at 15 ms and 1500 Hz)
        - n_samples: Number of samples in the event, first to last inclusive;
            the quantity the duration limits test
        - max_sustained_zscore: Largest value sustained for minimum_duration
            (NaN for an event holding fewer samples than the minimum; only
            ``Long_sharp_wave_ripple_detector`` produces one)
        - mean_zscore, median_zscore, max_zscore, min_zscore: Z-score statistics
        - area: Integral of z-score over event duration
        - total_energy: Integral of squared z-score
        - speed_at_start, speed_at_end: Speed at event boundaries, NaN where
            it is unknown
        - max_speed, min_speed, median_speed, mean_speed: Speed statistics
            over the event's samples with a known (not NaN) speed; NaN if none
        - clipped_start, clipped_end: Whether the event's first or last sample
            is the first or last sample of its block, i.e. the event was cut
            off by missing data or the recording edge
        - participants, n_participants, frac_participants: Information on
            which channels exhibited a ripple during the detected event
            (returned if 'participants' input is not None)

    """
    events = np.asarray(event_times, dtype=float).reshape(-1, 2)
    time_arr = np.asarray(time, dtype=float)
    metric = np.asarray(zscore_metric, dtype=float)
    speed_arr = np.asarray(speed, dtype=float)
    if participants is None and metric.ndim != 1:
        msg = (
            f"Without participants, zscore_metric must have shape (n_time,). Got shape "
            f"{metric.shape}."
        )
        raise ValueError(msg)
    if participants is not None and metric.ndim != 2:
        msg = (
            "With participants, zscore_metric must have shape (n_time, n_channels), so "
            f"each event's metrics can come from its participating channels. Got shape "
            f"{metric.shape}."
        )
        raise ValueError(msg)

    participant_channels = (
        None if participants is None else np.asarray(participants, dtype=object)
    )
    first = np.searchsorted(time_arr, events[:, 0], side="left")
    last = np.searchsorted(time_arr, events[:, 1], side="right")
    rows = []
    for index, ((start_time, end_time), a, b) in enumerate(
        zip(events, first, last, strict=True)
    ):
        event_time = time_arr[a:b]
        event_speed = speed_arr[a:b]
        if participant_channels is None:
            z = metric[a:b]
        else:
            channels = np.asarray(participant_channels[index], dtype=int)
            z = metric[a:b][:, channels].mean(axis=1)
        rows.append(
            (
                start_time,
                end_time,
                end_time - start_time,
                b - a,
                _max_sustained_zscore(event_time, z, minimum_duration),
                z.mean(),
                np.median(z),
                z.max(),
                z.min(),
                trapezoid(z, event_time),
                trapezoid(z**2, event_time),
                event_speed[0],
                event_speed[-1],
                *_known_speed_summary(event_speed),
            )
        )

    columns = [
        "start_time",
        "end_time",
        "duration",
        "n_samples",
        "max_sustained_zscore",
        "mean_zscore",
        "median_zscore",
        "max_zscore",
        "min_zscore",
        "area",
        "total_energy",
        "speed_at_start",
        "speed_at_end",
        "max_speed",
        "min_speed",
        "median_speed",
        "mean_speed",
    ]
    values = np.asarray(rows, dtype=float).reshape(-1, len(columns))
    index = pd.Index(np.arange(len(events)) + 1, name="event_number")
    event_stats = pd.DataFrame(dict(zip(columns, values.T, strict=True)), index=index)
    event_stats["n_samples"] = event_stats["n_samples"].astype(int)

    if clipped is None:
        block_starts = np.array([start for start, _ in blocks], dtype=int)
        block_stops = np.array([stop for _, stop in blocks], dtype=int)
        which = np.clip(np.searchsorted(block_starts, first, side="right") - 1, 0, None)
        clipped = np.column_stack([first == block_starts[which], last == block_stops[which]])
    clipped = np.asarray(clipped, dtype=bool).reshape(-1, 2)
    event_stats["clipped_start"] = clipped[:, 0]
    event_stats["clipped_end"] = clipped[:, 1]
    if participant_channels is not None:
        n_participants = np.array(
            [len(channels) for channels in participant_channels], dtype=int
        )
        event_stats["participants"] = participants
        event_stats["n_participants"] = n_participants
        event_stats["frac_participants"] = n_participants / metric.shape[1]
    return event_stats

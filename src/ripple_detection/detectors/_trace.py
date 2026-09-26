"""Detect events on any trace with the package's thresholding and conventions."""

from collections.abc import Sequence
from typing import Literal, get_args

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ripple_detection._call_hints import explain_call_errors
from ripple_detection.core import (
    NORMALIZATION_METHODS,
    SPEED_RULES,
    BoolArray,
    FloatArray,
    IntArray,
    NormalizationMethod,
    SpeedRule,
    _boolean_run_bounds,
    _check_choice,
    _is_immobile,
    _is_immobile_by_rule,
    _merged_bounds,
    _runs_extended_to_mean,
    gaussian_smooth,
    minimum_sample_count,
    nearest_sample_index,
    normalize_signal,
    sample_count_within,
)
from ripple_detection.detectors._blocks import (
    _contiguous_valid_blocks,
    _normalization_mask_over_valid,
    _valid_blocks,
)
from ripple_detection.detectors._events import _finish_events, _get_event_stats
from ripple_detection.detectors._validation import (
    _check_gap,
    _check_positive,
    _check_smoothing_sigma,
    _validate_detector_inputs,
    _validate_duration_limits,
)

TraceNormalizationMethod = NormalizationMethod | Literal["none"]
"""How :func:`detect_events_from_trace` can scale the trace before thresholding."""

TRACE_NORMALIZATION_METHODS: tuple[TraceNormalizationMethod, ...] = (
    *NORMALIZATION_METHODS,
    "none",
)
"""How :func:`detect_events_from_trace` can scale the trace before thresholding."""

TraceSpeedRule = SpeedRule | Literal["restrict"]
"""The speed rules of :func:`exclude_movement`, plus detecting only while slow."""

TRACE_SPEED_RULES: tuple[TraceSpeedRule, ...] = (*SPEED_RULES, "restrict")
"""The speed rules of :func:`exclude_movement`, plus detecting only while slow."""

CloseEventRule = Literal["drop", "merge"]
"""What :func:`detect_events_from_trace` does with events closer than the gap."""

CLOSE_EVENT_RULES: tuple[CloseEventRule, ...] = get_args(CloseEventRule)
"""What :func:`detect_events_from_trace` does with events closer than the gap."""


def _one_trace(trace: ArrayLike) -> FloatArray:
    """The trace as a float column of shape (n_time, 1), or a clear error."""
    values = np.asarray(trace, dtype=float)
    if values.ndim == 2 and values.shape[1] == 1:
        return values
    if values.ndim != 1:
        msg = (
            f"trace must have shape (n_time,), one value per sample; got shape {values.shape}. "
            "Combine channels into one trace first, for example trace = envelopes.mean(axis=1)."
        )
        raise ValueError(msg)
    return values[:, np.newaxis]


def _check_trace_thresholds(
    threshold: FloatArray, levels: tuple[float, ...], bound_search_window: float | None
) -> None:
    """Finite thresholds, at least one bound level, and every level no higher
    than the lowest threshold, so each run at or above the threshold lies in
    a stretch at or above every level."""
    if not np.all(np.isfinite(threshold)):
        msg = f"threshold must be finite, got {threshold if threshold.ndim == 0 else 'a non-finite value'}."
        raise ValueError(msg)
    if not levels:
        msg = "bound_threshold needs at least one level."
        raise ValueError(msg)
    for level in levels:
        if not np.isfinite(level):
            msg = f"bound_threshold must be finite, got {level}."
            raise ValueError(msg)
    lowest = float(np.min(threshold))
    if max(levels) > lowest:
        msg = (
            f"bound_threshold ({max(levels)}) is above threshold ({lowest}). Events end where "
            "the trace falls below bound_threshold, so it must not exceed the threshold "
            "that finds them."
        )
        raise ValueError(msg)
    if len(levels) > 1 and bound_search_window is None:
        msg = (
            "Several bound_threshold levels are fallbacks for a bound not found within "
            "bound_search_window; pass bound_search_window, or a single level."
        )
        raise ValueError(msg)


def _search_bounds(
    segment: FloatArray,
    is_above_threshold: BoolArray,
    n_minimum: int,
    levels: tuple[float, ...],
    n_search: int,
) -> tuple[IntArray, BoolArray]:
    """Bounds sought within ``n_search`` samples of each run's first sample.

    For each run at or above the threshold of ``n_minimum`` samples or more,
    the start is the sample after the last one below the first level within
    ``n_search`` samples before the run's first sample, and the stop the first
    one below it within ``n_search`` after; a side that finds none tries the
    next level, and with none left takes the edge of the search. Runs that
    share an event give one. Returns half-open ``[start, stop)`` bounds and,
    per side, whether the search ended at its edge inside the block."""
    runs = _boolean_run_bounds(is_above_threshold)
    runs = runs[(runs[:, 1] - runs[:, 0]) >= n_minimum]
    n = len(segment)
    found: list[tuple[int, int, bool, bool]] = []
    for anchor in runs[:, 0]:
        low = max(0, anchor - n_search)
        start, capped_start = low, low > 0
        for level in levels:
            below = np.flatnonzero(segment[low:anchor] < level)
            if below.size:
                start, capped_start = low + int(below[-1]) + 1, False
                break
        high = min(n, anchor + n_search + 1)
        stop, capped_end = high, high < n
        for level in levels:
            below = np.flatnonzero(segment[anchor + 1 : high] < level)
            if below.size:
                stop, capped_end = anchor + 1 + int(below[0]), False
                break
        if found and start < found[-1][1]:
            previous = found[-1]
            if stop > previous[1]:
                found[-1] = (previous[0], stop, previous[2], capped_end)
            continue
        found.append((start, stop, capped_start, capped_end))
    table = np.asarray(found, dtype=int).reshape(-1, 4)
    return table[:, :2], table[:, 2:].astype(bool)


def _merge_with_flags(
    events: FloatArray, flags: BoolArray, gap: float
) -> tuple[FloatArray, BoolArray]:
    """``merge_close_events`` within a block, carrying each merged event's
    flags: the start's from its first member, the end's from the member that
    ends it."""
    merged = _merged_bounds(events, gap)
    first = np.searchsorted(events[:, 0], merged[:, 0], side="left")
    after = np.searchsorted(events[:, 0], merged[:, 1], side="right")
    merged_flags = np.zeros((len(merged), 2), dtype=bool)
    for index, (a, b) in enumerate(zip(first, after, strict=True)):
        last = a + int(np.argmax(events[a:b, 1]))
        merged_flags[index] = (flags[a, 0], flags[last, 1])
    return merged, merged_flags


def _block_edges(
    event_times: FloatArray, time: FloatArray, blocks: list[tuple[int, int]]
) -> BoolArray:
    """Whether each event starts on its block's first sample or ends on its last."""
    first = np.searchsorted(time, event_times[:, 0], side="left")
    last = np.searchsorted(time, event_times[:, 1], side="right")
    block_starts = np.array([start for start, _ in blocks], dtype=int)
    block_stops = np.array([stop for _, stop in blocks], dtype=int)
    which = np.clip(np.searchsorted(block_starts, first, side="right") - 1, 0, None)
    return np.column_stack([first == block_starts[which], last == block_stops[which]])


def _restrict_to_slow(
    is_valid: BoolArray,
    time: FloatArray,
    speed: FloatArray,
    speed_threshold: float,
    minimum_duration: float,
) -> tuple[BoolArray, list[tuple[int, int]]]:
    """Valid samples and blocks limited to the slow stretches long enough for
    an event.

    A slow stretch shorter than ``minimum_duration`` is movement with a brief
    dip in speed, not missing data, so it is dropped without the warning a
    short block of missing samples gets. NaN speed is not slow.

    Raises
    ------
    ValueError
        If no sample is slow, or no valid slow stretch is long enough.
    """
    is_slow = _is_immobile(speed, speed_threshold)
    if not np.any(is_slow):
        msg = (
            f"speed_rule='restrict' detects only where speed is at or below "
            f"speed_threshold ({speed_threshold}), and no sample is."
        )
        raise ValueError(msg)
    is_valid = is_valid & is_slow
    n_minimum = minimum_sample_count(time, minimum_duration)
    blocks = [
        (start, stop)
        for start, stop in _contiguous_valid_blocks(is_valid, time)
        if stop - start >= n_minimum
    ]
    if not blocks:
        msg = (
            f"speed_rule='restrict' detects only where speed is at or below "
            f"speed_threshold ({speed_threshold}), and no such stretch of finite samples "
            f"is as long as the {n_minimum} samples an event of minimum_duration "
            f"({minimum_duration} s) needs."
        )
        raise ValueError(msg)
    kept = np.zeros_like(is_valid)
    for start, stop in blocks:
        kept[start:stop] = True
    return kept, blocks


@explain_call_errors
def detect_events_from_trace(
    time: ArrayLike,
    trace: ArrayLike,
    speed: ArrayLike | None,
    sampling_frequency: float,
    *,
    threshold: float | ArrayLike = 2.0,
    bound_threshold: float | Sequence[float] = 0.0,
    bound_search_window: float | None = None,
    smoothing_sigma: float | None = None,
    normalization_method: TraceNormalizationMethod = "zscore",
    normalization_mask: ArrayLike | None = None,
    minimum_duration: float = 0.015,
    minimum_event_duration: float | None = None,
    maximum_duration: float | None = None,
    speed_threshold: float = 4.0,
    speed_rule: TraceSpeedRule = "endpoints",
    close_event_threshold: float = 0.0,
    close_event_rule: CloseEventRule = "drop",
) -> pd.DataFrame:
    """Detect events on a trace you build, with the package's thresholding.

    Published detectors differ mostly in the trace they threshold (the mean
    or the sum of ripple envelopes, an RMS or wavelet power, a population
    rate of chosen cells) and agree on most of what follows. This function
    does what follows: it splits the recording at missing samples, optionally
    smooths, normalizes, finds the runs at or above ``threshold`` that last
    ``minimum_duration``, extends each to where the trace falls below
    ``bound_threshold``, applies the speed rule and the duration and
    close-event rules, and returns the same statistics as the detectors.
    ``Kay_ripple_detector``, ``Roumis_ripple_detector`` and
    ``multiunit_HSE_detector`` run these steps on their own traces.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Sample timestamps in **seconds**, increasing.
    trace : array_like, shape (n_time,) or (n_time, 1)
        The detection trace, one value per sample, for example the mean over
        tetrodes of the ripple-band envelope. NaN or infinity marks a missing
        sample: nothing is smoothed, normalized, thresholded or merged across
        it, and no event spans it.
    speed : array_like, shape (n_time,), or None
        The animal's speed in **cm/s**. NaN is an unknown speed, which splits
        nothing; ``speed_rule`` says how it is treated (``'restrict'`` treats
        it as not slow). None is no speed recorded, unknown everywhere,
        allowed only with ``speed_threshold=np.inf``; the speed columns are
        then NaN.
    sampling_frequency : float
        Sampling rate in Hz.
    threshold : float or array_like of shape (n_time,), optional
        Level the trace must reach, in the units ``normalization_method``
        gives it: standard deviations for ``'zscore'``, scaled MADs for
        ``'median_mad'``, the trace's own units for ``'none'``. A sample
        counts when it is at or above the level. An array gives each sample
        its own level, for a threshold that changes over the recording (per
        session, or as an online rule updated it). Default 2.0.
    bound_threshold : float or sequence of float, optional
        Level at which an event ends, in the same units: each event runs over
        the contiguous samples at or above it that contain a run at or above
        ``threshold``. Default 0.0, the mean (``'zscore'``) or median
        (``'median_mad'``), the rule every z-score detector here uses; papers
        that end events at 0.5, 1 or 2 SD pass that. Must not exceed
        ``threshold`` anywhere; equal to it, events end at the threshold
        crossings. With ``bound_search_window``, a sequence gives fallback
        levels, tried in order on each side that does not find the one
        before, such as ``(0.0, 0.25, 0.5)``.
    bound_search_window : float, optional
        Seconds before and after each run's first sample at or above
        ``threshold`` within which its bounds are sought, as Tirole et al.
        (2022) did: the start is the sample after the last one below the
        level before that sample, the end the sample before the first one
        below it after that sample.
        A side with no such sample at any level ends at the edge of the
        search, and is flagged in ``clipped_start`` or ``clipped_end``. Default
        None: events extend as far as the trace stays at or above the level.
    smoothing_sigma : float, optional
        Standard deviation in **seconds** of a Gaussian applied to the trace
        within each block of valid samples, before normalizing. Default None,
        no smoothing. Smoothing is linear, so smoothing a mean of envelopes
        equals averaging smoothed envelopes.
    normalization_method : {'zscore', 'median_mad', 'none'}, optional
        ``'zscore'`` (default) subtracts the mean and divides by the SD;
        ``'median_mad'`` uses the median and the scaled MAD; ``'none'`` leaves
        the trace as given, for a trace already scaled the way a paper
        specifies (divided by its baseline mean, or by its maximum, say).
    normalization_mask : array_like of bool, shape (n_time,), optional
        Samples the normalization statistics come from, such as immobility
        or sleep. Default None, every valid sample. Not allowed with
        ``normalization_method='none'``, which computes no statistics.
    minimum_duration : float, optional
        Seconds the trace must stay at or above ``threshold``, as a sample
        count (``minimum_sample_count``). Default 0.015. 0.0 requires a
        single sample, for a rule that tests only an event's peak.
    minimum_event_duration : float, optional
        Least duration in **seconds** of the whole event, from bound to
        bound, as an inclusive sample count (``sample_count_within``). Most
        published minimums mean this rather than ``minimum_duration``. It is
        applied with the speed rule, before close events are dropped, so an
        event too short to keep suppresses no neighbour. Default None.
    maximum_duration : float, optional
        Longest duration in **seconds** of the event as reported, applied
        last. Default None, no ceiling.
    speed_threshold : float, optional
        Speed in cm/s at or below which the animal counts as still. Default
        4.0; ``np.inf`` turns the speed rule off.
    speed_rule : {'endpoints', 'all', 'mean', 'median', 'restrict'}, optional
        ``'endpoints'`` (default), ``'all'``, ``'mean'`` and ``'median'``
        test each event as ``exclude_movement`` does. ``'restrict'`` instead
        treats every sample faster than ``speed_threshold``, or of unknown
        speed, as missing: the statistics come from slow samples only, and an
        event is cut where movement starts, which ``clipped_start`` and
        ``clipped_end`` then flag. A slow stretch shorter than
        ``minimum_duration`` is left out without a warning: it is a brief dip
        in speed, not missing data.
    close_event_threshold : float, optional
        Gap in **seconds** below which two events count as close. Default
        0.0, none.
    close_event_rule : {'drop', 'merge'}, optional
        ``'drop'`` (default) keeps the first of close events and drops the
        rest, after the speed rule and ``minimum_event_duration``, as the
        detectors do. ``'merge'`` joins close events into one before those
        rules (``merge_close_events``), so the merged event is what they
        test. Events are never merged across a missing sample.

    Returns
    -------
    events : pd.DataFrame
        One row per event, indexed by ``event_number``, with the columns the
        detectors return (see ``Kay_ripple_detector``). The ``*_zscore``
        columns, ``area`` and ``total_energy`` describe the normalized trace,
        and so are in the trace's own units with ``'none'``. ``clipped_start``
        and ``clipped_end`` flag a bound set by missing data, the recording
        edge, or the edge of ``bound_search_window``.

    Raises
    ------
    ValueError
        If ``speed`` is None while ``speed_threshold`` is finite, the trace
        is not one value per sample, the lengths differ, a
        threshold array is not one value per sample, ``bound_threshold``
        exceeds ``threshold``, fallback levels are given without
        ``bound_search_window``, a choice is not one of
        those listed, a duration or gap is not a plausible number of
        seconds, or ``normalization_mask`` is given with ``'none'``. With
        ``speed_rule='restrict'``, also if no sample is at or below
        ``speed_threshold``, or no slow stretch of finite samples is as long
        as ``minimum_duration``.

    See Also
    --------
    exclude_movement, merge_close_events, require_active_units

    Examples
    --------
    Pfeiffer & Foster (2015): the ripple-band envelope, averaged over
    tetrodes and smoothed with a 12.5 ms Gaussian, above 3 SD with the
    statistics taken while slower than 5 cm/s, bounded at the mean, and kept
    when 50 ms to 2 s long.

    >>> from ripple_detection import filter_ripple_band, get_envelope
    >>> from ripple_detection.simulate import simulate_session, simulate_time
    >>> time = simulate_time(45_000, 1500)
    >>> session = simulate_session(time, [5.0, 10.0, 15.0, 20.0, 25.0], n_channels=4, rng=0)
    >>> envelope = get_envelope(filter_ripple_band(session.lfps, 1500)).mean(axis=1)
    >>> events = detect_events_from_trace(
    ...     time, envelope, session.speed, 1500,
    ...     threshold=3.0, smoothing_sigma=0.0125,
    ...     normalization_mask=session.speed < 5, minimum_duration=0.0,
    ...     minimum_event_duration=0.05, maximum_duration=2.0, speed_threshold=5.0,
    ... )
    >>> "peak_time" in events, bool(len(events))
    (True, True)

    """
    threshold_values = np.asarray(threshold, dtype=float)
    levels = tuple(float(level) for level in np.atleast_1d(np.asarray(bound_threshold, float)))
    _check_trace_thresholds(threshold_values, levels, bound_search_window)
    if bound_search_window is not None:
        _check_positive(bound_search_window=bound_search_window)
        _check_gap(bound_search_window=bound_search_window)
    _check_choice("normalization_method", normalization_method, TRACE_NORMALIZATION_METHODS)
    _check_choice("speed_rule", speed_rule, TRACE_SPEED_RULES)
    _check_choice("close_event_rule", close_event_rule, CLOSE_EVENT_RULES)
    _validate_duration_limits(minimum_duration, maximum_duration)
    if minimum_event_duration is not None:
        _validate_duration_limits(
            minimum_event_duration,
            maximum_duration,
            names=("minimum_event_duration", "maximum_duration"),
        )
    _check_gap(close_event_threshold=close_event_threshold)
    if smoothing_sigma is not None:
        _check_smoothing_sigma(smoothing_sigma=smoothing_sigma)
    if normalization_method == "none" and normalization_mask is not None:
        msg = (
            "normalization_mask selects the samples normalization statistics come from, "
            "but normalization_method='none' computes none. Drop one of the two."
        )
        raise ValueError(msg)

    if speed is None:
        if not np.isposinf(speed_threshold):
            msg = (
                f"speed is None, so no event can be tested against speed_threshold "
                f"({speed_threshold} cm/s). Pass the speed, or speed_threshold=np.inf to "
                "detect without a speed rule."
            )
            raise ValueError(msg)
        speed = np.full(np.shape(time)[:1], np.nan)
    time, values, speed = _validate_detector_inputs(
        time, _one_trace(trace), speed, sampling_frequency, speed_threshold
    )
    if threshold_values.ndim != 0 and threshold_values.shape != time.shape:
        msg = (
            f"threshold as an array must give one level per sample, shape {time.shape}; "
            f"got {threshold_values.shape}."
        )
        raise ValueError(msg)
    is_valid, blocks = _valid_blocks(time, values, minimum_duration=minimum_duration)
    if speed_rule == "restrict":
        is_valid, blocks = _restrict_to_slow(
            is_valid, time, speed, speed_threshold, minimum_duration
        )

    detection_trace = np.full(len(time), np.nan)
    for start, stop in blocks:
        block = values[start:stop, 0]
        if smoothing_sigma is not None:
            block = gaussian_smooth(block, smoothing_sigma, sampling_frequency)
        detection_trace[start:stop] = block
    if normalization_method != "none":
        mask = _normalization_mask_over_valid(len(time), is_valid, normalization_mask)
        detection_trace = normalize_signal(
            detection_trace, method=normalization_method, normalization_mask=mask
        )

    n_minimum = minimum_sample_count(time, minimum_duration)
    n_search = (
        None
        if bound_search_window is None
        else minimum_sample_count(time, bound_search_window)
    )
    candidate_blocks, flag_blocks = [], []
    for start, stop in blocks:
        segment = detection_trace[start:stop]
        level = (
            threshold_values if threshold_values.ndim == 0 else threshold_values[start:stop]
        )
        if n_search is None:
            bounds, _ = _runs_extended_to_mean(
                segment >= levels[0], segment >= level, n_minimum
            )
            flags = np.zeros((len(bounds), 2), dtype=bool)
        else:
            bounds, flags = _search_bounds(
                segment, segment >= level, n_minimum, levels, n_search
            )
        block_events = np.column_stack(
            [time[start + bounds[:, 0]], time[start + bounds[:, 1] - 1]]
        ).reshape(-1, 2)
        if close_event_rule == "merge" and len(block_events):
            block_events, flags = _merge_with_flags(block_events, flags, close_event_threshold)
        candidate_blocks.append(block_events)
        flag_blocks.append(flags)
    candidates = np.concatenate([np.empty((0, 2)), *candidate_blocks])
    capped = np.concatenate([np.empty((0, 2), dtype=bool), *flag_blocks])

    if speed_rule == "restrict":
        keep = np.ones(len(candidates), dtype=bool)
    else:
        keep = _is_immobile_by_rule(candidates, speed, time, speed_threshold, speed_rule)
    if minimum_event_duration is not None and len(candidates):
        n_samples = (
            nearest_sample_index(time, candidates[:, 1])
            - nearest_sample_index(time, candidates[:, 0])
            + 1
        )
        keep &= np.asarray(sample_count_within(n_samples, time, minimum_event_duration))
    gap = close_event_threshold if close_event_rule == "drop" else 0.0
    event_times, kept = _finish_events(candidates, keep, time, gap, maximum_duration)
    clipped = (
        None if n_search is None else _block_edges(event_times, time, blocks) | capped[kept]
    )
    return _get_event_stats(
        event_times, time, detection_trace, speed, minimum_duration, blocks, clipped=clipped
    )

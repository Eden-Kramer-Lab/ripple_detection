"""Detect events on any trace with the package's thresholding and conventions."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ripple_detection._call_hints import explain_call_errors
from ripple_detection.core import (
    SPEED_RULES,
    FloatArray,
    _is_immobile,
    _is_immobile_by_rule,
    _runs_extended_to_mean,
    gaussian_smooth,
    merge_close_events,
    minimum_sample_count,
    nearest_sample_index,
    normalize_signal,
    sample_count_within,
)
from ripple_detection.detectors._blocks import (
    _normalization_mask_over_valid,
    _valid_blocks,
)
from ripple_detection.detectors._events import _finish_events, _get_event_stats
from ripple_detection.detectors._validation import (
    _check_gap,
    _check_smoothing_sigma,
    _validate_detector_inputs,
    _validate_duration_limits,
)

TRACE_NORMALIZATION_METHODS = ("zscore", "median_mad", "none")
"""How :func:`detect_events_from_trace` can scale the trace before thresholding."""

TRACE_SPEED_RULES = (*SPEED_RULES, "restrict")
"""The speed rules of :func:`exclude_movement`, plus detecting only while slow."""

CLOSE_EVENT_RULES = ("drop", "merge")
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


def _check_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        msg = f"{name} must be one of {', '.join(map(repr, choices))}; got {value!r}."
        raise ValueError(msg)


def _check_trace_thresholds(threshold: float, bound_threshold: float) -> None:
    for name, value in (("threshold", threshold), ("bound_threshold", bound_threshold)):
        if not np.isfinite(value):
            msg = f"{name} must be finite, got {value}."
            raise ValueError(msg)
    if bound_threshold > threshold:
        msg = (
            f"bound_threshold ({bound_threshold}) is above threshold ({threshold}). Events "
            "end where the trace falls below bound_threshold, so it must not exceed the "
            "threshold that finds them."
        )
        raise ValueError(msg)


@explain_call_errors
def detect_events_from_trace(
    time: ArrayLike,
    trace: ArrayLike,
    speed: ArrayLike,
    sampling_frequency: float,
    *,
    threshold: float = 2.0,
    bound_threshold: float = 0.0,
    smoothing_sigma: float | None = None,
    normalization_method: str = "zscore",
    normalization_mask: ArrayLike | None = None,
    minimum_duration: float = 0.015,
    minimum_event_duration: float | None = None,
    maximum_duration: float | None = None,
    speed_threshold: float = 4.0,
    speed_rule: str = "endpoints",
    close_event_threshold: float = 0.0,
    close_event_rule: str = "drop",
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
    speed : array_like, shape (n_time,)
        The animal's speed in **cm/s**. NaN is an unknown speed, which splits
        nothing; ``speed_rule`` says how it is treated.
    sampling_frequency : float
        Sampling rate in Hz.
    threshold : float, optional
        Level the trace must reach, in the units ``normalization_method``
        gives it: standard deviations for ``'zscore'``, scaled MADs for
        ``'median_mad'``, the trace's own units for ``'none'``. A sample
        counts when it is at or above the level. Default 2.0.
    bound_threshold : float, optional
        Level at which an event ends, in the same units: each event runs over
        the contiguous samples at or above it that contain a run at or above
        ``threshold``. Default 0.0, the mean (``'zscore'``) or median
        (``'median_mad'``), the rule every z-score detector here uses; papers
        that end events at 0.5, 1 or 2 SD pass that. Must not exceed
        ``threshold``; equal to it, events end at the threshold crossings.
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
        ``clipped_end`` then flag.
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
        and so are in the trace's own units with ``'none'``.

    Raises
    ------
    ValueError
        If the trace is not one value per sample, the lengths differ,
        ``bound_threshold`` exceeds ``threshold``, a choice is not one of
        those listed, a duration or gap is not a plausible number of
        seconds, or ``normalization_mask`` is given with ``'none'``.

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
    _check_trace_thresholds(threshold, bound_threshold)
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

    time, values, speed = _validate_detector_inputs(
        time, _one_trace(trace), speed, sampling_frequency, speed_threshold
    )
    if speed_rule == "restrict":
        is_slow = _is_immobile(speed, speed_threshold)
        if not np.any(is_slow):
            msg = (
                f"speed_rule='restrict' detects only where speed is at or below "
                f"speed_threshold ({speed_threshold}), and no sample is."
            )
            raise ValueError(msg)
        values = values.copy()
        values[~is_slow] = np.nan
    is_valid, blocks = _valid_blocks(time, values, minimum_duration=minimum_duration)

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
    candidate_blocks = []
    for start, stop in blocks:
        segment = detection_trace[start:stop]
        bounds, _ = _runs_extended_to_mean(
            segment >= bound_threshold, segment >= threshold, n_minimum
        )
        block_events = np.column_stack(
            [time[start + bounds[:, 0]], time[start + bounds[:, 1] - 1]]
        ).reshape(-1, 2)
        if close_event_rule == "merge" and len(block_events):
            block_events = merge_close_events(block_events, close_event_threshold)
        candidate_blocks.append(block_events)
    candidates = np.concatenate([np.empty((0, 2)), *candidate_blocks])

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
    event_times, _ = _finish_events(candidates, keep, time, gap, maximum_duration)
    return _get_event_stats(
        event_times, time, detection_trace, speed, minimum_duration, blocks
    )

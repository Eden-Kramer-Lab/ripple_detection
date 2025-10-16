"""Finding sharp-wave ripple events (150-250 Hz) from local field
potentials.
"""

from collections.abc import Generator
from os.path import abspath, dirname, join

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.fftpack import next_fast_len
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.signal import filtfilt, hilbert, remez


def ripple_bandpass_filter(sampling_frequency: float) -> tuple[NDArray, float]:
    """Generate a bandpass filter for the ripple frequency band (150-250 Hz).

    Uses the Remez exchange algorithm to design a finite impulse response (FIR)
    filter with 101 taps and 25 Hz transition bands.

    Parameters
    ----------
    sampling_frequency : float
        Sampling rate of the signal in Hz.

    Returns
    -------
    filter_numerator : ndarray
        Numerator coefficients of the filter.
    filter_denominator : float
        Denominator coefficient (always 1.0 for FIR filters).

    """
    ORDER = 101
    nyquist = 0.5 * sampling_frequency
    TRANSITION_BAND = 25
    RIPPLE_BAND = [150, 250]
    desired = [
        0,
        RIPPLE_BAND[0] - TRANSITION_BAND,
        RIPPLE_BAND[0],
        RIPPLE_BAND[1],
        RIPPLE_BAND[1] + TRANSITION_BAND,
        nyquist,
    ]
    return remez(ORDER, desired, [0, 1, 0], Hz=sampling_frequency), 1.0


def _get_series_start_end_times(series: pd.Series) -> tuple[NDArray, NDArray]:
    """Extracts the start and end times of segments defined by a boolean
    pandas Series.

    Parameters
    ----------
    series : pandas boolean Series (n_time,)
        Consecutive Trues define each segment.

    Returns
    -------
    start_times : ndarray, shape (n_segments,)
        Beginning time of each segment based on the index of the series.
    end_times : ndarray, shape (n_segments,)
        End time of each segment based on the index of the series.

    """
    # Identify starts and ends without using fillna to avoid pandas FutureWarning
    # A start is where current is True AND previous is not True (False or NaN)
    # An end is where current is True AND next is not True (False or NaN)
    shifted_prev = series.shift(1)
    shifted_next = series.shift(-1)

    # Use != True instead of == False to handle NaN properly
    # NaN != True is True, which is what we want for boundaries
    is_start_time = series & (shifted_prev != True)  # noqa: E712
    start_times = np.asarray(series.index[is_start_time])

    is_end_time = series & (shifted_next != True)  # noqa: E712
    end_times = np.asarray(series.index[is_end_time])

    return start_times, end_times


def segment_boolean_series(
    series: pd.Series, minimum_duration: float = 0.015
) -> list[tuple[float, float]]:
    """Extract time segments from a boolean pandas Series.

    Returns a list of tuples where each tuple contains the start and end time
    of a segment. Segments are defined by consecutive True values in the input
    series, where the series index represents time.

    Parameters
    ----------
    series : pd.Series
        Boolean pandas Series with time as index. Consecutive True values
        define each segment.
    minimum_duration : float, optional
        Minimum duration (in same units as index) for a segment to be included.
        Default is 0.015 (15 ms if index is in seconds).

    Returns
    -------
    segments : list of tuple
        List of (start_time, end_time) tuples for each segment that meets
        the minimum duration requirement.

    """
    start_times, end_times = _get_series_start_end_times(series)

    return [
        (start_time, end_time)
        for start_time, end_time in zip(start_times, end_times, strict=False)
        if end_time >= (start_time + minimum_duration)
    ]


def filter_ripple_band(data: ArrayLike, sampling_frequency: float | None = None) -> NDArray:
    """Apply bandpass filter to isolate ripple frequency band (150-250 Hz).

    Uses a pre-computed filter kernel from the Frank lab with 40 dB roll-off,
    10 Hz sidebands, and 1500 Hz sampling frequency. Handles NaN values by
    filtering only non-NaN segments.

    Parameters
    ----------
    data : array_like, shape (n_time,) or (n_time, n_channels)
        Input signal(s) to be filtered. Can be 1D or 2D.
    sampling_frequency : float, optional
        Sampling rate of the input data in Hz. If provided and not equal to
        1500 Hz, a warning is issued since the pre-computed filter is optimized
        for 1500 Hz. For other sampling rates, consider using
        `ripple_bandpass_filter()` to generate a custom filter. Default is None
        (no check performed).

    Returns
    -------
    filtered_data : ndarray, shape (n_time,) or (n_time, n_channels)
        Bandpass filtered signal in the ripple band. NaN values are preserved
        at their original locations.

    Raises
    ------
    ValueError
        If sampling_frequency is too low for the pre-computed filter to work properly.

    Warnings
    --------
    UserWarning
        If sampling_frequency is provided and differs from 1500 Hz.

    See Also
    --------
    ripple_bandpass_filter : Generate custom filter for arbitrary sampling rates.

    """
    import warnings

    EXPECTED_SAMPLING_FREQUENCY = 1500.0
    MINIMUM_SAFE_FREQUENCY = 1200.0  # Pre-computed filter needs ~954 samples minimum

    if sampling_frequency is not None and not np.isclose(
        sampling_frequency, EXPECTED_SAMPLING_FREQUENCY
    ):
        # Check if sampling frequency is too low for pre-computed filter
        if sampling_frequency < MINIMUM_SAFE_FREQUENCY:
            raise ValueError(
                f"Sampling frequency ({sampling_frequency} Hz) is too low for the pre-computed filter.\n"
                f"The pre-computed filter requires at least ~{MINIMUM_SAFE_FREQUENCY} Hz.\n"
                f"\n"
                f"Solution: Generate a custom filter for your sampling frequency:\n"
                f"\n"
                f"  from ripple_detection import ripple_bandpass_filter\n"
                f"  from scipy.signal import filtfilt\n"
                f"  \n"
                f"  filter_num, filter_denom = ripple_bandpass_filter({sampling_frequency})\n"
                f"  filtered_data = filtfilt(filter_num, filter_denom, data, axis=0)\n"
                f"\n"
                f"Or use a higher sampling rate when recording your data."
            )
        else:
            warnings.warn(
                f"The pre-computed ripple filter is optimized for {EXPECTED_SAMPLING_FREQUENCY} Hz sampling.\n"
                f"Your data: {sampling_frequency} Hz. Results may be suboptimal.\n"
                f"For best results, use ripple_bandpass_filter({sampling_frequency}) to generate a custom filter.",
                UserWarning,
                stacklevel=2,
            )

    filter_numerator, filter_denominator = _get_ripplefilter_kernel()

    # Validate data length
    data_array = np.asarray(data)

    # Check if data is multi-dimensional - handle NaN checking appropriately
    if data_array.ndim > 1:
        is_nan = np.any(np.isnan(data_array), axis=-1)
    else:
        is_nan = np.isnan(data_array)

    non_nan_length = np.sum(~is_nan)

    # filtfilt requires data length > 3 * filter_length (for padding)
    min_required_length = 3 * len(filter_numerator)
    if non_nan_length < min_required_length:
        raise ValueError(
            f"Data is too short for the pre-computed filter.\n"
            f"Non-NaN data length: {non_nan_length} samples\n"
            f"Minimum required: {min_required_length} samples (~{min_required_length/sampling_frequency if sampling_frequency else 'N/A':.2f} seconds at {sampling_frequency} Hz)\n"
            f"\n"
            f"Solutions:\n"
            f"  1. Use a longer recording segment\n"
            f"  2. Generate a shorter custom filter:\n"
            f"     from ripple_detection import ripple_bandpass_filter\n"
            f"     from scipy.signal import filtfilt\n"
            f"     filter_num, filter_denom = ripple_bandpass_filter({sampling_frequency})\n"
            f"     filtered_data = filtfilt(filter_num, filter_denom, data, axis=0)"
        )

    filtered_data = np.full_like(data_array, np.nan)
    filtered_data[~is_nan] = filtfilt(
        filter_numerator, filter_denominator, data_array[~is_nan], axis=0
    )
    return filtered_data


def _get_ripplefilter_kernel() -> tuple[NDArray, int]:
    """Load the pre-computed ripple filter kernel from the Frank lab.

    The kernel is a 150-250 Hz bandpass filter with 40 dB roll-off and 10 Hz
    transition sidebands, designed for signals sampled at 1500 Hz.

    Returns
    -------
    filter_numerator : ndarray
        Filter kernel coefficients.
    filter_denominator : int
        Denominator coefficient (always 1 for FIR filters).

    """
    filter_file = join(abspath(dirname(__file__)), "ripplefilter.mat")
    ripplefilter = loadmat(filter_file)
    return ripplefilter["ripplefilter"]["kernel"][0][0].flatten(), 1


def extend_threshold_to_mean(
    is_above_mean: ArrayLike,
    is_above_threshold: ArrayLike,
    time: ArrayLike,
    minimum_duration: float = 0.015,
) -> list[tuple[float, float]]:
    """Extend threshold-crossing segments to where the signal crosses the mean.

    Finds segments where the signal exceeds a threshold for a minimum duration,
    then extends the boundaries of these segments to where the signal crosses
    the mean value.

    Parameters
    ----------
    is_above_mean : array_like, shape (n_time,)
        Boolean array indicating where the signal is above its mean.
    is_above_threshold : array_like, shape (n_time,)
        Boolean array indicating where the signal is above the threshold.
    time : array_like, shape (n_time,)
        Time values corresponding to each sample.
    minimum_duration : float, optional
        Minimum time (in same units as `time`) that signal must remain above
        threshold. Default is 0.015 (15 ms if time is in seconds).

    Returns
    -------
    candidate_ripple_times : list of tuple
        List of (start_time, end_time) tuples for each detected event,
        extended to mean crossings.

    """
    is_above_threshold = pd.Series(is_above_threshold, index=time)
    is_above_mean = pd.Series(is_above_mean, index=time)
    above_mean_segments = segment_boolean_series(
        is_above_mean, minimum_duration=minimum_duration
    )
    above_threshold_segments = segment_boolean_series(
        is_above_threshold, minimum_duration=minimum_duration
    )
    return sorted(_extend_segment(above_threshold_segments, above_mean_segments))


def exclude_movement(
    candidate_ripple_times: ArrayLike,
    speed: ArrayLike,
    time: ArrayLike,
    speed_threshold: float = 4.0,
) -> NDArray | list:
    """Filter out candidate ripples that occur during animal movement.

    Removes events where the animal's speed at either the start or end of the
    event exceeds the specified threshold.

    Parameters
    ----------
    candidate_ripple_times : array_like, shape (n_ripples, 2)
        Array of candidate event times with columns [start_time, end_time].
    speed : array_like, shape (n_time,)
        Animal's speed at each time point.
    time : array_like, shape (n_time,)
        Time values corresponding to speed measurements.
    speed_threshold : float, optional
        Maximum speed (in same units as `speed`) for event to be retained.
        Events with speed > threshold at start or end are excluded.
        Default is 4.0 (cm/s).

    Returns
    -------
    ripple_times : ndarray or list
        Filtered event times where animal speed is below threshold. Returns
        ndarray of shape (n_stationary_ripples, 2), or empty list if no
        events remain.

    """
    candidate_ripple_times = np.array(candidate_ripple_times)
    try:
        speed_at_ripple_start = speed[np.isin(time, candidate_ripple_times[:, 0])]
        speed_at_ripple_end = speed[np.isin(time, candidate_ripple_times[:, 1])]
        is_below_speed_threshold = (speed_at_ripple_start <= speed_threshold) & (
            speed_at_ripple_end <= speed_threshold
        )
        return candidate_ripple_times[is_below_speed_threshold]
    except IndexError:
        return []


def _find_containing_interval(
    interval_candidates: list[tuple[float, float]], target_interval: tuple[float, float]
) -> tuple[float, float]:
    """Find the interval that contains the target interval.

    Identifies which candidate interval contains the target interval by finding
    the candidate with the closest start time that precedes the target start.
    Assumes one candidate interval contains the target (e.g., segments above
    mean contain segments above threshold).

    Parameters
    ----------
    interval_candidates : list of tuple
        List of (start, end) tuples representing candidate intervals.
    target_interval : tuple
        (start, end) tuple representing the target interval to be contained.

    Returns
    -------
    containing_interval : tuple
        The (start, end) tuple from candidates that contains the target.

    """
    candidate_start_times = np.asarray(interval_candidates)[:, 0]
    zero = np.array(0).astype(candidate_start_times.dtype)
    closest_start_ind = np.max((candidate_start_times - target_interval[0] <= zero).nonzero())
    return interval_candidates[closest_start_ind]


def _extend_segment(
    segments_to_extend: list[tuple[float, float]],
    containing_segments: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Extends the boundaries of a segment if it is a subset of one of the
    containing segments.

    Parameters
    ----------
    segments_to_extend : list of 2-element tuples
        Elements are the start and end times
    containing_segments : list of 2-element tuples
        Elements are the start and end times

    Returns
    -------
    extended_segments : list of 2-element tuples

    """
    segments = [
        _find_containing_interval(containing_segments, segment)
        for segment in segments_to_extend
    ]
    return list(set(segments))  # remove duplicate segments


def get_envelope(data: ArrayLike, axis: int = 0) -> NDArray:
    """Extract the instantaneous amplitude (envelope) using Hilbert transform.

    Computes the analytic signal via Hilbert transform and returns its
    magnitude, representing the instantaneous amplitude envelope.

    Parameters
    ----------
    data : array_like
        Input signal. Can be multi-dimensional.
    axis : int, optional
        Axis along which to compute the envelope. Default is 0.

    Returns
    -------
    envelope : ndarray
        Instantaneous amplitude (envelope) of the signal, same shape as input.

    """
    n_samples = data.shape[axis]
    instantaneous_amplitude = np.abs(hilbert(data, N=next_fast_len(n_samples), axis=axis))
    return np.take(instantaneous_amplitude, np.arange(n_samples), axis=axis)


def gaussian_smooth(
    data: ArrayLike,
    sigma: float,
    sampling_frequency: float,
    axis: int = 0,
    truncate: int = 8,
) -> NDArray:
    """Apply 1D Gaussian smoothing to data.

    Convolves the data with a Gaussian kernel. The standard deviation is
    specified in time units (e.g., seconds) and converted to samples using
    the sampling frequency. This is a wrapper around scipy's `gaussian_filter1d`
    with truncation at 8 standard deviations (instead of 4).

    Parameters
    ----------
    data : array_like
        Input data to be smoothed. Can be multi-dimensional.
    sigma : float
        Standard deviation of the Gaussian kernel in time units (e.g., seconds).
    sampling_frequency : float
        Sampling rate in Hz, used to convert sigma from time to samples.
    axis : int, optional
        Axis along which to apply the filter. Default is 0.
    truncate : int, optional
        Number of standard deviations at which to truncate the filter.
        Default is 8 (wider support than scipy's default of 4).

    Returns
    -------
    smoothed_data : ndarray
        Gaussian-smoothed data, same shape as input.

    """
    return gaussian_filter1d(
        data, sigma * sampling_frequency, truncate=truncate, axis=axis, mode="constant"
    )


def threshold_by_zscore(
    zscored_data: ArrayLike,
    time: ArrayLike,
    minimum_duration: float = 0.015,
    zscore_threshold: float = 2,
) -> list[tuple[float, float]]:
    """Find time segments where z-scored data exceeds a threshold.

    Identifies segments where the z-scored signal exceeds the specified
    threshold for at least the minimum duration, then extends these segments
    to where the signal crosses zero (the mean of z-scored data).

    Parameters
    ----------
    zscored_data : array_like, shape (n_time,)
        Z-scored (standardized) input signal.
    time : array_like, shape (n_time,)
        Time values corresponding to each sample.
    minimum_duration : float, optional
        Minimum time that signal must exceed threshold. Default is 0.015
        (15 ms if time is in seconds).
    zscore_threshold : float, optional
        Z-score threshold value. Default is 2 (2 standard deviations).

    Returns
    -------
    candidate_ripple_times : list of tuple
        List of (start_time, end_time) tuples for detected events, extended
        to mean crossings.

    """
    is_above_mean = zscored_data >= 0
    is_above_threshold = zscored_data >= zscore_threshold

    return extend_threshold_to_mean(
        is_above_mean, is_above_threshold, time, minimum_duration=minimum_duration
    )


def merge_overlapping_ranges(
    ranges: list[tuple[float, float]],
) -> Generator[tuple[float, float], None, None]:
    """Merge overlapping and adjacent ranges

    Parameters
    ----------
    ranges : iterable with 2-elements
        Element 1 is the start of the range.
        Element 2 is the end of the range.

    Yields
    -------
    sorted_merged_range : 2-element tuple
        Element 1 is the start of the merged range.
        Element 2 is the end of the merged range.

    >>> list(merge_overlapping_ranges([(5, 7), (3, 5), (-1, 3)]))
    [(-1, 7)]
    >>> list(merge_overlapping_ranges([(5, 6), (3, 4), (1, 2)]))
    [(1, 2), (3, 4), (5, 6)]
    >>> list(merge_overlapping_ranges([]))
    []

    References
    ----------
    .. [1] http://codereview.stackexchange.com/questions/21307/consolidate-
    list-of-ranges-that-overlap

    """
    ranges = iter(sorted(ranges))
    try:
        current_start, current_stop = next(ranges)
    except StopIteration:
        return None
    for start, stop in ranges:
        if start > current_stop:
            # Gap between segments: output current segment and start a new
            # one.
            yield current_start, current_stop
            current_start, current_stop = start, stop
        else:
            # Segments adjacent or overlapping: merge.
            current_stop = max(current_stop, stop)
    yield current_start, current_stop


def exclude_close_events(
    candidate_event_times: ArrayLike, close_event_threshold: float = 1.0
) -> NDArray | list:
    """Remove events that occur too close together in time.

    Filters out successive events that start within `close_event_threshold`
    time units of a previous event's end, keeping only the first event in
    each cluster of closely-spaced events.

    Implementation uses iterative filtering: for each kept event, mark all
    subsequent events that start too soon as excluded.

    Parameters
    ----------
    candidate_event_times : array_like, shape (n_events, 2)
        Array of event times with columns [start_time, end_time].
    close_event_threshold : float, optional
        Minimum time between events. Events starting within this time after
        a previous event ends are excluded. Default is 1.0 (seconds).

    Returns
    -------
    filtered_event_times : ndarray or list
        Filtered event times with shape (n_filtered_events, 2), or empty
        list if no events remain.

    """
    candidate_event_times = np.array(candidate_event_times)

    if candidate_event_times.size == 0:
        return []

    n_events = candidate_event_times.shape[0]
    keep_mask = np.ones(n_events, dtype=bool)

    for i in range(n_events):
        if not keep_mask[i]:
            continue

        end_time = candidate_event_times[i, 1]
        # Mark subsequent events that start too soon as excluded
        later_indices = np.arange(i + 1, n_events)
        is_too_close = candidate_event_times[later_indices, 0] < (
            end_time + close_event_threshold
        )
        keep_mask[later_indices] = keep_mask[later_indices] & ~is_too_close

    filtered_events = candidate_event_times[keep_mask]
    return filtered_events if filtered_events.size > 0 else []


def get_multiunit_population_firing_rate(
    multiunit: ArrayLike, sampling_frequency: float, smoothing_sigma: float = 0.015
) -> NDArray:
    """Calculates the multiunit population firing rate.

    Parameters
    ----------
    multiunit : ndarray, shape (n_time, n_signals)
        Spike indicator matrix. Can be binary (0/1) or spike counts per bin.
    sampling_frequency : float
        Number of samples per second.
    smoothing_sigma : float or np.timedelta
        Amount to smooth the firing rate over time. The default is
        given assuming time is in units of seconds.


    Returns
    -------
    multiunit_population_firing_rate : ndarray, shape (n_time,)

    """
    return gaussian_smooth(
        multiunit.sum(axis=1) * sampling_frequency, smoothing_sigma, sampling_frequency
    )

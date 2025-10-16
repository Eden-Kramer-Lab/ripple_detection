"""High-level detectors for sharp-wave ripple events and multiunit synchrony events."""

from itertools import chain

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import zscore

from ripple_detection.core import (
    exclude_close_events,
    exclude_movement,
    gaussian_smooth,
    get_envelope,
    get_multiunit_population_firing_rate,
    merge_overlapping_ranges,
    threshold_by_zscore,
)


def _preprocess_detector_inputs(
    time: ArrayLike, filtered_lfps: ArrayLike, speed: ArrayLike
) -> tuple[NDArray, NDArray, NDArray]:
    """Remove NaN values from detector inputs.

    Ensures all inputs are aligned by removing any time points where
    LFP data or speed contains NaN values. This preprocessing step is
    shared by all ripple detectors.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Time values for each sample.
    filtered_lfps : array_like, shape (n_time, n_channels)
        Bandpass filtered LFP signals.
    speed : array_like, shape (n_time,)
        Animal's running speed.

    Returns
    -------
    time_clean : ndarray, shape (n_clean_time,)
        Time array with NaN rows removed.
    filtered_lfps_clean : ndarray, shape (n_clean_time, n_channels)
        LFP array with NaN rows removed.
    speed_clean : ndarray, shape (n_clean_time,)
        Speed array with NaN values removed.

    """
    filtered_lfps = np.asarray(filtered_lfps)
    speed = np.asarray(speed)
    time = np.asarray(time)

    not_null = np.all(pd.notnull(filtered_lfps), axis=1) & pd.notnull(speed)

    return time[not_null], filtered_lfps[not_null], speed[not_null]


def get_Kay_ripple_consensus_trace(
    ripple_filtered_lfps: ArrayLike, sampling_frequency: float, smoothing_sigma: float = 0.004
) -> NDArray:
    """Compute Kay consensus trace from multi-channel ripple-filtered LFPs.

    Combines multiple LFP channels into a single consensus trace using the sum
    of squared envelopes, following Kay et al. 2016. The trace is smoothed with
    a Gaussian kernel.

    Parameters
    ----------
    ripple_filtered_lfps : array_like, shape (n_time, n_channels)
        Bandpass filtered LFP signals in the ripple band (150-250 Hz).
    sampling_frequency : float
        Sampling rate in Hz.
    smoothing_sigma : float, optional
        Standard deviation of Gaussian smoothing kernel in seconds.
        Default is 0.004 (4 ms).

    Returns
    -------
    consensus_trace : ndarray, shape (n_time,)
        Combined consensus trace computed as sqrt(sum(envelope^2)).

    References
    ----------
    .. [1] Kay, K., et al. (2016). A hippocampal network for spatial coding
       during immobility and sleep. Nature, 531(7593), 185-190.

    """
    ripple_consensus_trace = np.full_like(ripple_filtered_lfps, np.nan)
    not_null = np.all(pd.notnull(ripple_filtered_lfps), axis=1)

    ripple_consensus_trace[not_null] = get_envelope(np.asarray(ripple_filtered_lfps)[not_null])
    ripple_consensus_trace = np.sum(ripple_consensus_trace**2, axis=1)
    ripple_consensus_trace[not_null] = gaussian_smooth(
        ripple_consensus_trace[not_null], smoothing_sigma, sampling_frequency
    )
    return np.sqrt(ripple_consensus_trace)


def Kay_ripple_detector(
    time: ArrayLike,
    filtered_lfps: ArrayLike,
    speed: ArrayLike,
    sampling_frequency: float,
    speed_threshold: float = 4.0,
    minimum_duration: float = 0.015,
    zscore_threshold: float = 2.0,
    smoothing_sigma: float = 0.004,
    close_ripple_threshold: float = 0.0,
) -> pd.DataFrame:
    """Detect sharp-wave ripple events using multi-channel consensus method.

    Implements the Kay et al. 2016 ripple detection algorithm, which combines
    multiple LFP channels into a consensus trace using sum of squared envelopes.
    Ripples are identified as periods where the z-scored consensus exceeds a
    threshold during immobility.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Time values for each sample.
    filtered_lfps : array_like, shape (n_time, n_channels)
        Bandpass filtered LFP signals in the ripple band (150-250 Hz).
    speed : array_like, shape (n_time,)
        Animal's running speed at each time point.
    sampling_frequency : float
        Sampling rate in Hz.
    speed_threshold : float, optional
        Maximum speed (cm/s) for ripple detection. Ripples during movement
        (speed > threshold) are excluded. Default is 4.0 cm/s.
    minimum_duration : float, optional
        Minimum duration (seconds) that z-score must exceed threshold.
        Default is 0.015 (15 ms).
    zscore_threshold : float, optional
        Number of standard deviations above mean for detection.
        Default is 2.0.
    smoothing_sigma : float, optional
        Standard deviation (seconds) of Gaussian smoothing kernel applied
        to consensus trace. Default is 0.004 (4 ms).
    close_ripple_threshold : float, optional
        Minimum time (seconds) between ripples. Ripples closer than this
        are merged. Default is 0.0 (no merging).

    Returns
    -------
    ripple_times : pd.DataFrame
        DataFrame with one row per detected ripple, containing:
        - start_time, end_time, duration
        - max_thresh: maximum sustained z-score
        - mean_zscore, median_zscore, max_zscore, min_zscore
        - area: integral of z-score
        - total_energy: integral of squared z-score
        - speed metrics: speed_at_start, speed_at_end, max/min/median/mean_speed

    References
    ----------
    .. [1] Kay, K., Sosa, M., Chung, J.E., Karlsson, M.P., Larkin, M.C.,
       and Frank, L.M. (2016). A hippocampal network for spatial coding during
       immobility and sleep. Nature 531, 185-190.

    """
    time, filtered_lfps, speed = _preprocess_detector_inputs(time, filtered_lfps, speed)

    combined_filtered_lfps = get_Kay_ripple_consensus_trace(
        filtered_lfps, sampling_frequency, smoothing_sigma=smoothing_sigma
    )
    combined_filtered_lfps = zscore(combined_filtered_lfps, nan_policy="omit")
    candidate_ripple_times = threshold_by_zscore(
        combined_filtered_lfps, time, minimum_duration, zscore_threshold
    )
    ripple_times = exclude_movement(
        candidate_ripple_times, speed, time, speed_threshold=speed_threshold
    )
    ripple_times = exclude_close_events(ripple_times, close_ripple_threshold)

    return _get_event_stats(
        ripple_times, time, combined_filtered_lfps, speed, minimum_duration
    )


def Karlsson_ripple_detector(
    time: ArrayLike,
    filtered_lfps: ArrayLike,
    speed: ArrayLike,
    sampling_frequency: float,
    speed_threshold: float = 4.0,
    minimum_duration: float = 0.015,
    zscore_threshold: float = 3.0,
    smoothing_sigma: float = 0.004,
    close_ripple_threshold: float = 0.0,
) -> pd.DataFrame:
    """Detect sharp-wave ripples using per-channel detection with merging.

    Implements the Karlsson et al. 2009 algorithm, which detects ripples on
    each LFP channel independently, then merges overlapping events across
    channels. More sensitive to local ripples than consensus methods.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Time values for each sample.
    filtered_lfps : array_like, shape (n_time, n_channels)
        Bandpass filtered LFP signals in the ripple band (150-250 Hz).
    speed : array_like, shape (n_time,)
        Animal's running speed at each time point.
    sampling_frequency : float
        Sampling rate in Hz.
    speed_threshold : float, optional
        Maximum speed (cm/s) for ripple detection. Default is 4.0 cm/s.
    minimum_duration : float, optional
        Minimum duration (seconds) for detection. Default is 0.015 (15 ms).
    zscore_threshold : float, optional
        Z-score threshold for detection. Default is 3.0 (higher than Kay).
    smoothing_sigma : float, optional
        Standard deviation (seconds) of Gaussian smoothing. Default is 0.004 (4 ms).
    close_ripple_threshold : float, optional
        Minimum time (seconds) between ripples. Default is 0.0.

    Returns
    -------
    ripple_times : pd.DataFrame
        DataFrame with detected ripples and comprehensive statistics (see
        Kay_ripple_detector for column descriptions).

    References
    ----------
    .. [1] Karlsson, M.P., and Frank, L.M. (2009). Awake replay of remote
       experiences in the hippocampus. Nature Neuroscience 12, 913-918.

    """
    time, filtered_lfps, speed = _preprocess_detector_inputs(time, filtered_lfps, speed)

    filtered_lfps = get_envelope(filtered_lfps)
    filtered_lfps = gaussian_smooth(
        filtered_lfps, sigma=smoothing_sigma, sampling_frequency=sampling_frequency
    )
    filtered_lfps = zscore(filtered_lfps, nan_policy="omit")
    candidate_ripple_times = [
        threshold_by_zscore(filtered_lfp, time, minimum_duration, zscore_threshold)
        for filtered_lfp in filtered_lfps.T
    ]
    candidate_ripple_times = list(
        merge_overlapping_ranges(chain.from_iterable(candidate_ripple_times))
    )
    ripple_times = exclude_movement(
        candidate_ripple_times, speed, time, speed_threshold=speed_threshold
    )
    ripple_times = exclude_close_events(ripple_times, close_ripple_threshold)

    return _get_event_stats(ripple_times, time, filtered_lfps.mean(axis=1), speed)


def Roumis_ripple_detector(
    time: ArrayLike,
    filtered_lfps: ArrayLike,
    speed: ArrayLike,
    sampling_frequency: float,
    speed_threshold: float = 4.0,
    minimum_duration: float = 0.015,
    zscore_threshold: float = 2.0,
    smoothing_sigma: float = 0.004,
    close_ripple_threshold: float = 0.0,
) -> pd.DataFrame:
    """Detect sharp-wave ripples using averaged square-root envelope method.

    Variant detection method that averages the square-root of squared envelopes
    across channels. Provides a balanced approach between Kay (consensus) and
    Karlsson (per-channel) methods.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Time values for each sample.
    filtered_lfps : array_like, shape (n_time, n_channels)
        Bandpass filtered LFP signals in the ripple band (150-250 Hz).
    speed : array_like, shape (n_time,)
        Animal's running speed at each time point.
    sampling_frequency : float
        Sampling rate in Hz.
    speed_threshold : float, optional
        Maximum speed (cm/s) for ripple detection. Default is 4.0 cm/s.
    minimum_duration : float, optional
        Minimum duration (seconds) for detection. Default is 0.015 (15 ms).
    zscore_threshold : float, optional
        Z-score threshold for detection. Default is 2.0.
    smoothing_sigma : float, optional
        Standard deviation (seconds) of Gaussian smoothing. Default is 0.004 (4 ms).
    close_ripple_threshold : float, optional
        Minimum time (seconds) between ripples. Default is 0.0.

    Returns
    -------
    ripple_times : pd.DataFrame
        DataFrame with detected ripples and comprehensive statistics (see
        Kay_ripple_detector for column descriptions).

    """
    time, filtered_lfps, speed = _preprocess_detector_inputs(time, filtered_lfps, speed)

    filtered_lfps = get_envelope(filtered_lfps) ** 2
    filtered_lfps = gaussian_smooth(
        filtered_lfps, sigma=smoothing_sigma, sampling_frequency=sampling_frequency
    )
    combined_filtered_lfps = np.mean(np.sqrt(filtered_lfps), axis=1)
    combined_filtered_lfps = zscore(combined_filtered_lfps, nan_policy="omit")
    candidate_ripple_times = threshold_by_zscore(
        combined_filtered_lfps, time, minimum_duration, zscore_threshold
    )
    ripple_times = exclude_movement(
        candidate_ripple_times, speed, time, speed_threshold=speed_threshold
    )
    ripple_times = exclude_close_events(ripple_times, close_ripple_threshold)

    return _get_event_stats(
        ripple_times, time, combined_filtered_lfps, speed, minimum_duration
    )


def multiunit_HSE_detector(
    time: ArrayLike,
    multiunit: ArrayLike,
    speed: ArrayLike,
    sampling_frequency: float,
    speed_threshold: float = 4.0,
    minimum_duration: float = 0.015,
    zscore_threshold: float = 2.0,
    smoothing_sigma: float = 0.015,
    close_event_threshold: float = 0.0,
    use_speed_threshold_for_zscore: bool = False,
) -> pd.DataFrame:
    """Detect High Synchrony Events from multiunit spiking activity.

    Identifies periods of elevated population spiking activity during immobility,
    following Davidson et al. 2009. The population firing rate is smoothed and
    z-scored, then thresholded to find synchronous events.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Time values for each sample.
    multiunit : array_like, shape (n_time, n_units)
        Binary spike indicator matrix (1 = spike, 0 = no spike) for each unit.
    speed : array_like, shape (n_time,)
        Animal's running speed at each time point.
    sampling_frequency : float
        Sampling rate in Hz.
    speed_threshold : float, optional
        Maximum speed (cm/s) for event detection. Default is 4.0 cm/s.
    minimum_duration : float, optional
        Minimum duration (seconds) for detection. Default is 0.015 (15 ms).
    zscore_threshold : float, optional
        Z-score threshold for population firing rate. Default is 2.0.
    smoothing_sigma : float, optional
        Standard deviation (seconds) of Gaussian smoothing applied to firing
        rate. Default is 0.015 (15 ms, longer than ripple detectors).
    close_event_threshold : float, optional
        Minimum time (seconds) between events. Default is 0.0.
    use_speed_threshold_for_zscore : bool, optional
        If True, compute z-score statistics (mean/std) using only immobility
        periods (speed < threshold). Default is False (use all time points).

    Returns
    -------
    high_synchrony_events : pd.DataFrame
        DataFrame with detected events and comprehensive statistics (see
        Kay_ripple_detector for column descriptions).

    References
    ----------
    .. [1] Davidson, T.J., Kloosterman, F., and Wilson, M.A. (2009).
       Hippocampal Replay of Extended Experience. Neuron 63, 497–507.

    """
    multiunit = np.asarray(multiunit)
    speed = np.asarray(speed)
    time = np.asarray(time)

    firing_rate = get_multiunit_population_firing_rate(
        multiunit, sampling_frequency, smoothing_sigma
    )

    if use_speed_threshold_for_zscore:
        mean = np.nanmean(firing_rate[speed < speed_threshold])
        std = np.nanstd(firing_rate[speed < speed_threshold])
    else:
        mean = np.nanmean(firing_rate)
        std = np.nanstd(firing_rate)

    firing_rate = (firing_rate - mean) / std
    candidate_high_synchrony_events = threshold_by_zscore(
        firing_rate, time, minimum_duration, zscore_threshold
    )
    high_synchrony_events = exclude_movement(
        candidate_high_synchrony_events, speed, time, speed_threshold=speed_threshold
    )
    high_synchrony_events = exclude_close_events(high_synchrony_events, close_event_threshold)

    return _get_event_stats(high_synchrony_events, time, firing_rate, speed)


def _find_max_thresh(
    time: np.ndarray, data: np.ndarray, minimum_duration: float = 0.015
) -> float:
    """Find the maximum value of a peak that exceeds a
    threshold for a minimum duration.

    Parameters
    ----------
    time : np.ndarray, shape (n_time,)
    data : np.ndarray, shape (n_time,)
    minimum_duration : float, optional

    Returns
    -------
    max_thresh : float
    """
    # Find the peak of the data points
    peak_ind = np.argmax(data)

    # Initialize the search window
    peak_left_ind = peak_ind
    peak_right_ind = peak_ind

    # Expand the window until the time difference exceeds the minimum duration
    while time[peak_right_ind] - time[peak_left_ind] < minimum_duration:
        # Determine the direction to expand
        if peak_right_ind < len(time) - 1 and (
            peak_left_ind == 0 or data[peak_right_ind + 1] > data[peak_left_ind - 1]
        ):
            peak_right_ind += 1
        else:
            peak_left_ind -= 1

    # Return the minimum value between the left and right edges of the window
    return min(data[peak_left_ind], data[peak_right_ind])


def _get_event_stats(
    event_times: ArrayLike,
    time: ArrayLike,
    zscore_metric: ArrayLike,
    speed: ArrayLike,
    minimum_duration: float = 0.015,
) -> pd.DataFrame:
    """Compute comprehensive statistics for detected events.

    Calculates temporal, z-score, signal, and speed metrics for each event.

    Parameters
    ----------
    event_times : array_like, shape (n_events, 2)
        Array of [start_time, end_time] for each event.
    time : array_like, shape (n_time,)
        Time values for each sample.
    zscore_metric : array_like, shape (n_time,)
        Z-scored signal used for detection.
    speed : array_like, shape (n_time,)
        Animal's speed at each time point.
    minimum_duration : float, optional
        Minimum duration for max_thresh calculation. Default is 0.015 (15 ms).

    Returns
    -------
    event_stats : pd.DataFrame
        DataFrame with one row per event and columns:
        - start_time, end_time: Event boundaries
        - duration: Event duration (end - start)
        - max_thresh: Maximum z-score sustained for minimum_duration
        - mean_zscore, median_zscore, max_zscore, min_zscore: Z-score statistics
        - area: Integral of z-score over event duration
        - total_energy: Integral of squared z-score
        - speed_at_start, speed_at_end: Speed at event boundaries
        - max_speed, min_speed, median_speed, mean_speed: Speed statistics

    """
    event_times_arr = np.asarray(event_times)
    time_arr = np.asarray(time)
    zscore_metric_arr = np.asarray(zscore_metric)
    speed_arr = np.asarray(speed)

    index = pd.Index(np.arange(len(event_times_arr)) + 1, name="event_number")
    try:
        speed_at_start = speed_arr[np.isin(time_arr, event_times_arr[:, 0])]
        speed_at_end = speed_arr[np.isin(time_arr, event_times_arr[:, 1])]
    except (IndexError, TypeError):
        speed_at_start = np.full_like(event_times_arr, np.nan)
        speed_at_end = np.full_like(event_times_arr, np.nan)

    mean_zscore = []
    median_zscore = []
    max_zscore = []
    min_zscore = []
    duration = []
    max_speed = []
    min_speed = []
    median_speed = []
    mean_speed = []
    max_thresh = []
    area = []
    total_energy = []

    for start_time, end_time in event_times_arr:
        ind = np.logical_and(time_arr >= start_time, time_arr <= end_time)
        event_zscore = zscore_metric_arr[ind]
        max_thresh.append(
            _find_max_thresh(time_arr[ind], zscore_metric_arr[ind], minimum_duration)
        )
        mean_zscore.append(np.mean(event_zscore))
        median_zscore.append(np.median(event_zscore))
        max_zscore.append(np.max(event_zscore))
        min_zscore.append(np.min(event_zscore))
        area.append(np.trapezoid(event_zscore, time_arr[ind]))
        total_energy.append(np.trapezoid(event_zscore**2, time_arr[ind]))
        duration.append(end_time - start_time)
        max_speed.append(np.max(speed_arr[ind]))
        min_speed.append(np.min(speed_arr[ind]))
        median_speed.append(np.median(speed_arr[ind]))
        mean_speed.append(np.mean(speed_arr[ind]))

    try:
        event_start_times = event_times_arr[:, 0]
        event_end_times = event_times_arr[:, 1]
    except (IndexError, TypeError):
        event_start_times = []
        event_end_times = []

    return pd.DataFrame(
        {
            "start_time": event_start_times,
            "end_time": event_end_times,
            "duration": duration,
            "max_thresh": max_thresh,
            "mean_zscore": mean_zscore,
            "median_zscore": median_zscore,
            "max_zscore": max_zscore,
            "min_zscore": min_zscore,
            "area": area,
            "total_energy": total_energy,
            "speed_at_start": speed_at_start,
            "speed_at_end": speed_at_end,
            "max_speed": max_speed,
            "min_speed": min_speed,
            "median_speed": median_speed,
            "mean_speed": mean_speed,
        },
        index=index,
    )

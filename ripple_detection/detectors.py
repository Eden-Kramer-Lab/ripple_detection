from itertools import chain

import numpy as np
import pandas as pd
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


def get_Kay_ripple_consensus_trace(
    ripple_filtered_lfps, sampling_frequency, smoothing_sigma=0.004
):
    ripple_consensus_trace = np.full_like(ripple_filtered_lfps, np.nan)
    not_null = np.all(pd.notnull(ripple_filtered_lfps), axis=1)

    ripple_consensus_trace[not_null] = get_envelope(
        np.asarray(ripple_filtered_lfps)[not_null]
    )
    ripple_consensus_trace = np.sum(ripple_consensus_trace**2, axis=1)
    ripple_consensus_trace[not_null] = gaussian_smooth(
        ripple_consensus_trace[not_null], smoothing_sigma, sampling_frequency
    )
    return np.sqrt(ripple_consensus_trace)


def Kay_ripple_detector(
    time,
    filtered_lfps,
    speed,
    sampling_frequency,
    speed_threshold=4.0,
    minimum_duration=0.015,
    zscore_threshold=2.0,
    smoothing_sigma=0.004,
    close_ripple_threshold=0.0,
):
    """Find start and end times of sharp wave ripple events (150-250 Hz)
    based on Kay et al. 2016 [1].

    Parameters
    ----------
    time : array_like, shape (n_time,)
    filtered_lfps : array_like, shape (n_time, n_signals)
        Bandpass filtered time series of electric potentials in the ripple band
    speed : array_like, shape (n_time,)
        Running speed of animal
    sampling_frequency : float
        Number of samples per second.
    speed_threshold : float, optional
        Maximum running speed of animal for a ripple
    minimum_duration : float, optional
        Minimum time the z-score has to stay above threshold to be
        considered a ripple. The default is given assuming time is in
        units of seconds.
    zscore_threshold : float, optional
        Number of standard deviations the ripple power must exceed to
        be considered a ripple
    smoothing_sigma : float, optional
        Amount to smooth the time series over time. The default is
        given assuming time is in units of seconds.
    close_ripple_threshold : float, optional
        Exclude ripples that occur within `close_ripple_threshold` of a
        previously detected ripple.

    Returns
    -------
    ripple_times : pandas DataFrame

    References
    ----------
    .. [1] Kay, K., Sosa, M., Chung, J.E., Karlsson, M.P., Larkin, M.C.,
    and Frank, L.M. (2016). A hippocampal network for spatial coding during
    immobility and sleep. Nature 531, 185-190.

    """
    filtered_lfps = np.asarray(filtered_lfps)
    speed = np.asarray(speed)
    time = np.asarray(time)

    not_null = np.all(pd.notnull(filtered_lfps), axis=1) & pd.notnull(speed)
    filtered_lfps, speed, time = (
        filtered_lfps[not_null],
        speed[not_null],
        time[not_null],
    )

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

    return _get_event_stats(ripple_times, time, combined_filtered_lfps, speed, minimum_duration)


def Karlsson_ripple_detector(
    time,
    filtered_lfps,
    speed,
    sampling_frequency,
    speed_threshold=4.0,
    minimum_duration=0.015,
    zscore_threshold=3.0,
    smoothing_sigma=0.004,
    close_ripple_threshold=0.0,
):
    """Find start and end times of sharp wave ripple events (150-250 Hz)
    based on Karlsson et al. 2009 [1].

    Parameters
    ----------
    time : array_like, shpe (n_time,)
    filtered_lfps : array_like, shape (n_time, n_signals)
        Bandpass filtered time series of electric potentials in the ripple band
    speed : array_like, shape (n_time,)
        Running speed of animal
    sampling_frequency : float
        Number of samples per second.
    speed_threshold : float, optional
        Maximum running speed of animal for a ripple
    minimum_duration : float, optional
        Minimum time the z-score has to stay above threshold to be
        considered a ripple. The default is given assuming time is in
        units of seconds.
    zscore_threshold : float, optional
        Number of standard deviations the ripple power must exceed to
        be considered a ripple
    smoothing_sigma : float, optional
        Amount to smooth the time series over time. The default is
        given assuming time is in units of seconds.
    close_ripple_threshold : float, optional
        Exclude ripples that occur within `close_ripple_threshold` of a
        previously detected ripple.

    Returns
    -------
    ripple_times : pandas DataFrame

    References
    ----------
    .. [1] Karlsson, M.P., and Frank, L.M. (2009). Awake replay of remote
    experiences in the hippocampus. Nature Neuroscience 12, 913-918.


    """
    filtered_lfps = np.asarray(filtered_lfps)
    speed = np.asarray(speed)
    time = np.asarray(time)

    not_null = np.all(pd.notnull(filtered_lfps), axis=1) & pd.notnull(speed)
    filtered_lfps, speed, time = (
        filtered_lfps[not_null],
        speed[not_null],
        time[not_null],
    )

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
    time,
    filtered_lfps,
    speed,
    sampling_frequency,
    speed_threshold=4.0,
    minimum_duration=0.015,
    zscore_threshold=2.0,
    smoothing_sigma=0.004,
    close_ripple_threshold=0.0,
):
    """Find start and end times of sharp wave ripple events (150-250 Hz)
    based on [1].

    Parameters
    ----------
    time : array_like, shpe (n_time,)
    filtered_lfps : array_like, shape (n_time, n_signals)
        Bandpass filtered time series of electric potentials in the ripple band
    speed : array_like, shape (n_time,)
        Running speed of animal
    sampling_frequency : float
        Number of samples per second.
    speed_threshold : float, optional
        Maximum running speed of animal for a ripple
    minimum_duration : float, optional
        Minimum time the z-score has to stay above threshold to be
        considered a ripple. The default is given assuming time is in
        units of seconds.
    zscore_threshold : float, optional
        Number of standard deviations the ripple power must exceed to
        be considered a ripple
    smoothing_sigma : float, optional
        Amount to smooth the time series over time. The default is
        given assuming time is in units of seconds.
    close_ripple_threshold : float, optional
        Exclude ripples that occur within `close_ripple_threshold` of a
        previously detected ripple.

    Returns
    -------
    ripple_times : pandas DataFrame

    """
    filtered_lfps = np.asarray(filtered_lfps)
    speed = np.asarray(speed)
    time = np.asarray(time)

    not_null = np.all(pd.notnull(filtered_lfps), axis=1) & pd.notnull(speed)
    filtered_lfps, speed, time = (
        filtered_lfps[not_null],
        speed[not_null],
        time[not_null],
    )

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
    index = pd.Index(np.arange(len(ripple_times)) + 1, name="ripple_number")
    return pd.DataFrame(ripple_times, columns=["start_time", "end_time"], index=index)


def multiunit_HSE_detector(
    time,
    multiunit,
    speed,
    sampling_frequency,
    speed_threshold=4.0,
    minimum_duration=0.015,
    zscore_threshold=2.0,
    smoothing_sigma=0.015,
    close_event_threshold=0.0,
    use_speed_threshold_for_zscore=False,
):
    """Multiunit High Synchrony Event detector. Finds times when the multiunit
    population spiking activity is high relative to the average.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    multiunit : ndarray, shape (n_time, n_signals)
        Binary array of multiunit spike times.
    speed : ndarray, shape (n_time,)
        Running speed of animal
    sampling_frequency : float
        Number of samples per second.
    speed_threshold : float
        Maximum running speed of animal to be counted as an event
    minimum_duration : float
        Minimum time the z-score has to stay above threshold to be
        considered an event.
    zscore_threshold : float
        Number of standard deviations the multiunit population firing rate must
        exceed to be considered an event
    smoothing_sigma : float or np.timedelta
        Amount to smooth the firing rate over time. The default is
        given assuming time is in units of seconds.
    close_event_threshold : float
        Exclude events that occur within `close_event_threshold` of a
        previously detected event.
    use_speed_threshold_for_zscore : bool
        Use speed thresholded multiunit for mean and std for z-score calculation

    Returns
    -------
    high_synchrony_event_times : pandas.DataFrame, shape (n_events, 2)

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
    high_synchrony_events = exclude_close_events(
        high_synchrony_events, close_event_threshold
    )

    return _get_event_stats(high_synchrony_events, time, firing_rate, speed)


def _find_max_thresh(time: np.ndarray, data: np.ndarray, minimum_duration: float=0.015) -> float:
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


def _get_event_stats(event_times, time, zscore_metric, speed, minimum_duration=0.015):
    index = pd.Index(np.arange(len(event_times)) + 1, name="event_number")
    try:
        speed_at_start = speed[np.in1d(time, event_times[:, 0])]
        speed_at_end = speed[np.in1d(time, event_times[:, 1])]
    except (IndexError, TypeError):
        speed_at_start = np.full_like(event_times, np.nan)
        speed_at_end = np.full_like(event_times, np.nan)

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

    for start_time, end_time in event_times:
        ind = np.logical_and(time >= start_time, time <= end_time)
        event_zscore = zscore_metric[ind]
        max_thresh.append(_find_max_thresh(time[ind], zscore_metric[ind], minimum_duration))
        mean_zscore.append(np.mean(event_zscore))
        median_zscore.append(np.median(event_zscore))
        max_zscore.append(np.max(event_zscore))
        min_zscore.append(np.min(event_zscore))
        area.append(np.trapz(event_zscore, time[ind]))
        total_energy.append(np.trapz(event_zscore ** 2, time[ind]))
        duration.append(end_time - start_time)
        max_speed.append(np.max(speed[ind]))
        min_speed.append(np.min(speed[ind]))
        median_speed.append(np.median(speed[ind]))
        mean_speed.append(np.mean(speed[ind]))

    try:
        event_start_times = event_times[:, 0]
        event_end_times = event_times[:, 1]
    except TypeError:
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

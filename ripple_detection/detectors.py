from itertools import chain

import numpy as np
import pandas as pd

from .core import (exclude_movement, gaussian_smooth, get_envelope,
                   merge_overlapping_ranges, filter_ripple_band,
                   threshold_by_zscore)


def Kay_ripple_detector(time, LFPs, speed, sampling_frequency,
                        speed_threshold=4.0, minimum_duration=0.015,
                        zscore_threshold=2.0, smoothing_sigma=0.004):
    '''Find start and end times of sharp wave ripple events (150-250 Hz)
    based on Kay et al. 2016 [1].

    Parameters
    ----------
    time : array_like, shape (n_time,)
    LFPs : array_like, shape (n_time, n_signals)
        Time series of electric potentials
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

    Returns
    -------
    ripple_times : pandas DataFrame

    References
    ----------
    .. [1] Kay, K., Sosa, M., Chung, J.E., Karlsson, M.P., Larkin, M.C.,
    and Frank, L.M. (2016). A hippocampal network for spatial coding during
    immobility and sleep. Nature 531, 185-190.

    '''
    not_null = np.all(pd.notnull(LFPs), axis=1) & pd.notnull(speed)
    LFPs, speed, time = (
        LFPs.copy()[not_null], speed.copy()[not_null], time.copy()[not_null])

    filtered_lfps = np.stack(
        [filter_ripple_band(lfp, sampling_frequency) for lfp in LFPs.T])
    combined_filtered_lfps = np.sum(filtered_lfps ** 2, axis=0)
    combined_filtered_lfps = gaussian_smooth(
        combined_filtered_lfps, smoothing_sigma, sampling_frequency)
    combined_filtered_lfps = np.sqrt(combined_filtered_lfps)
    candidate_ripple_times = threshold_by_zscore(
        combined_filtered_lfps, time, minimum_duration, zscore_threshold)
    ripple_times = exclude_movement(
        candidate_ripple_times, speed, time,
        speed_threshold=speed_threshold)
    index = pd.Index(np.arange(len(ripple_times)) + 1,
                     name='ripple_number')
    return pd.DataFrame(ripple_times, columns=['start_time', 'end_time'],
                        index=index)


def Karlsson_ripple_detector(time, LFPs, speed, sampling_frequency,
                             speed_threshold=4.0, minimum_duration=0.015,
                             zscore_threshold=3.0, smoothing_sigma=0.004):
    '''Find start and end times of sharp wave ripple events (150-250 Hz)
    based on Karlsson et al. 2009 [1].

    Parameters
    ----------
    time : array_like, shpe (n_time,)
    LFPs : array_like, shape (n_time, n_signals)
        Time series of electric potentials
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

    Returns
    -------
    ripple_times : pandas DataFrame

    References
    ----------
    .. [1] Karlsson, M.P., and Frank, L.M. (2009). Awake replay of remote
    experiences in the hippocampus. Nature Neuroscience 12, 913-918.


    '''
    not_null = np.all(pd.notnull(LFPs), axis=1) & pd.notnull(speed)
    LFPs, speed, time = (
        LFPs.copy()[not_null], speed.copy()[not_null], time.copy()[not_null])

    candidate_ripple_times = []
    for lfp in LFPs.T:
        is_nan = np.isnan(lfp)
        filtered_lfp = filter_ripple_band(
            lfp, sampling_frequency=sampling_frequency)
        filtered_lfp = gaussian_smooth(
            get_envelope(filtered_lfp[~is_nan]), sigma=smoothing_sigma,
            sampling_frequency=sampling_frequency)
        lfp_ripple_times = threshold_by_zscore(
            filtered_lfp, time[~is_nan], minimum_duration,
            zscore_threshold)
        candidate_ripple_times.append(lfp_ripple_times)

    candidate_ripple_times = list(merge_overlapping_ranges(
        chain.from_iterable(candidate_ripple_times)))
    ripple_times = exclude_movement(
        candidate_ripple_times, speed, time,
        speed_threshold=speed_threshold)
    index = pd.Index(np.arange(len(ripple_times)) + 1,
                     name='ripple_number')
    return pd.DataFrame(ripple_times, columns=['start_time', 'end_time'],
                        index=index)


def Roumis_ripple_detector(time, LFPs, speed, sampling_frequency,
                           speed_threshold=4.0, minimum_duration=0.015,
                           zscore_threshold=2.0, smoothing_sigma=0.004):
    '''Find start and end times of sharp wave ripple events (150-250 Hz)
    based on [1].

    Parameters
    ----------
    time : array_like, shpe (n_time,)
    LFPs : array_like, shape (n_time, n_signals)
        Time series of electric potentials
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

    Returns
    -------
    ripple_times : pandas DataFrame

    References
    ----------
    [1] https://bitbucket.org/franklab/trodes2ff_shared/src/b156c8d5fef3a2f89e15a678046c52919638162e/extractEventConsensus.m?at=develop&fileviewer=file-view-default

    '''
    not_null = np.all(pd.notnull(LFPs), axis=1) & pd.notnull(speed)
    LFPs, speed, time = (
        LFPs.copy()[not_null], speed.copy()[not_null], time.copy()[not_null])
    filtered_lfps = [filter_ripple_band(lfp, sampling_frequency)
                     for lfp in LFPs.T]
    filtered_lfps = [np.sqrt(gaussian_smooth(
        filtered_lfp ** 2, smoothing_sigma, sampling_frequency))
        for filtered_lfp in filtered_lfps]
    combined_filtered_lfps = np.mean(filtered_lfps, axis=0)
    candidate_ripple_times = threshold_by_zscore(
        combined_filtered_lfps, time, minimum_duration, zscore_threshold)
    ripple_times = exclude_movement(
        candidate_ripple_times, speed, time,
        speed_threshold=speed_threshold)
    index = pd.Index(np.arange(len(ripple_times)) + 1,
                     name='ripple_number')
    return pd.DataFrame(ripple_times, columns=['start_time', 'end_time'],
                        index=index)

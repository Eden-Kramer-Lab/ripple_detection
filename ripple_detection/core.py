'''Finding sharp-wave ripple events (150-250 Hz) from local field
potentials.
'''
from os.path import abspath, dirname, join

import numpy as np
import pandas as pd
from scipy.fftpack import next_fast_len
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.signal import filtfilt, hilbert, remez
from scipy.stats import zscore


def ripple_bandpass_filter(sampling_frequency):
    ORDER = 101
    nyquist = 0.5 * sampling_frequency
    TRANSITION_BAND = 25
    RIPPLE_BAND = [150, 250]
    desired = [0, RIPPLE_BAND[0] - TRANSITION_BAND, RIPPLE_BAND[0],
               RIPPLE_BAND[1], RIPPLE_BAND[1] + TRANSITION_BAND, nyquist]
    return remez(ORDER, desired, [0, 1, 0], Hz=sampling_frequency), 1.0


def _get_series_start_end_times(series):
    '''Extracts the start and end times of segements defined by a boolean
    pandas Series.

    Parameters
    ----------
    series : pandas boolean Series (n_time,)
        Consecutive Trues define each segement.

    Returns
    -------
    start_times : ndarray, shape (n_segments,)
        Beginning time of each segment based on the index of the series.
    end_times : ndarray, shape (n_segments,)
        End time of each segment based on the index of the series.

    '''
    is_start_time = (~series.shift(1).fillna(False)) & series
    start_times = np.asarray(series.index[is_start_time])

    is_end_time = series & (~series.shift(-1).fillna(False))
    end_times = np.asarray(series.index[is_end_time])

    return start_times, end_times


def segment_boolean_series(series, minimum_duration=0.015):
    '''Returns a list of tuples where each tuple contains the start time of
     segement and end time of segment. It takes a boolean pandas series as
     input where the index is time.

     Parameters
     ----------
     series : pandas boolean Series (n_time,)
         Consecutive Trues define each segement.
     minimum_duration : float, optional
         Segments must be at least this duration to be included.

     Returns
     -------
     segments : list of 2-element tuples

     '''
    start_times, end_times = _get_series_start_end_times(series)

    return [(start_time, end_time)
            for start_time, end_time in zip(start_times, end_times)
            if end_time >= (start_time + minimum_duration)]


def filter_ripple_band(data):
    '''Returns a bandpass filtered signal between 150-250 Hz

    Parameters
    ----------
    data : array_like, shape (n_time,)

    Returns
    -------
    filtered_data : array_like, shape (n_time,)

    '''
    filter_numerator, filter_denominator = _get_ripplefilter_kernel()
    is_nan = np.any(np.isnan(data), axis=-1)
    filtered_data = np.full_like(data, np.nan)
    filtered_data[~is_nan] = filtfilt(
        filter_numerator, filter_denominator, data[~is_nan], axis=0)
    return filtered_data


def _get_ripplefilter_kernel():
    '''Returns the pre-computed ripple filter kernel from the Frank lab.
    The kernel is 150-250 Hz bandpass with 40 db roll off and 10 Hz
    sidebands. Sampling frequency is 1500 Hz.
    '''
    filter_file = join(abspath(dirname(__file__)), 'ripplefilter.mat')
    ripplefilter = loadmat(filter_file)
    return ripplefilter['ripplefilter']['kernel'][0][0].flatten(), 1


def extend_threshold_to_mean(is_above_mean, is_above_threshold, time,
                             minimum_duration=0.015):
    '''Extract segments above threshold if they remain above the threshold
    for a minimum amount of time and extend them to the mean.

    Parameters
    ----------
    is_above_mean : ndarray, shape (n_time,)
        Time series indicator function specifying when the
        time series is above the mean
    is_above_threshold : ndarray, shape (n_time,)
        Time series indicator function specifying when the
        time series is above the the threshold.
    time : ndarray, shape (n_time,)

    Returns
    -------
    candidate_ripple_times : list of 2-element tuples
        Each tuple is the start and end time of the candidate ripple.

    '''
    is_above_threshold = pd.Series(is_above_threshold, index=time)
    is_above_mean = pd.Series(is_above_mean, index=time)
    above_mean_segments = segment_boolean_series(
        is_above_mean, minimum_duration=minimum_duration)
    above_threshold_segments = segment_boolean_series(
        is_above_threshold, minimum_duration=minimum_duration)
    return sorted(
        _extend_segment(above_threshold_segments, above_mean_segments))


def exclude_movement(candidate_ripple_times, speed, time,
                     speed_threshold=4.0):
    '''Removes candidate ripples if the animal is moving.

    Parameters
    ----------
    candidate_ripple_times : array_like, shape (n_ripples, 2)
    speed : ndarray, shape (n_time,)
        Speed of animal during recording session.
    time : ndarray, shape (n_time,)
        Time in recording session.
    speed_threshold : float, optional
        Maximum speed for animal to be considered to be moving.

    Returns
    -------
    ripple_times : ndarray, shape (n_ripples, 2)
        Ripple times where the animal is not moving.

    '''
    candidate_ripple_times = np.array(candidate_ripple_times)
    try:
        speed_at_ripple_start = speed[
            np.in1d(time, candidate_ripple_times[:, 0])]
        speed_at_ripple_end = speed[
            np.in1d(time, candidate_ripple_times[:, 1])]
        is_below_speed_threshold = (
            (speed_at_ripple_start <= speed_threshold)
            & (speed_at_ripple_end <= speed_threshold))
        return candidate_ripple_times[is_below_speed_threshold]
    except IndexError:
        return []


def _find_containing_interval(interval_candidates, target_interval):
    '''Returns the interval that contains the target interval out of a list
    of interval candidates.

    This is accomplished by finding the closest start time out of the
    candidate intervals, since we already know that one interval candidate
    contains the target interval (the segements above 0 contain the
    segments above the threshold)
    '''
    candidate_start_times = np.asarray(interval_candidates)[:, 0]
    zero = np.array(0).astype(candidate_start_times.dtype)
    closest_start_ind = np.max(
        (candidate_start_times - target_interval[0] <= zero).nonzero())
    return interval_candidates[closest_start_ind]


def _extend_segment(segments_to_extend, containing_segments):
    '''Extends the boundaries of a segment if it is a subset of one of the
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

    '''
    segments = [_find_containing_interval(containing_segments, segment)
                for segment in segments_to_extend]
    return list(set(segments))  # remove duplicate segments


def get_envelope(data, axis=0):
    '''Extracts the instantaneous amplitude (envelope) of an analytic
    signal using the Hilbert transform'''
    n_samples = data.shape[axis]
    instantaneous_amplitude = np.abs(
        hilbert(data, N=next_fast_len(n_samples), axis=axis))
    return np.take(instantaneous_amplitude, np.arange(n_samples), axis=axis)


def gaussian_smooth(data, sigma, sampling_frequency, axis=0, truncate=8):
    '''1D convolution of the data with a Gaussian.

    The standard deviation of the gaussian is in the units of the sampling
    frequency. The function is just a wrapper around scipy's
    `gaussian_filter1d`, The support is truncated at 8 by default, instead
    of 4 in `gaussian_filter1d`

    Parameters
    ----------
    data : array_like
    sigma : float
    sampling_frequency : int
    axis : int, optional
    truncate : int, optional

    Returns
    -------
    smoothed_data : array_like

    '''
    return gaussian_filter1d(
        data, sigma * sampling_frequency, truncate=truncate, axis=axis,
        mode='constant')


def threshold_by_zscore(data, time, minimum_duration=0.015,
                        zscore_threshold=2):
    '''Standardize the data and determine whether it is above a given
    number.

    Parameters
    ----------
    data : array_like, shape (n_time,)
    zscore_threshold : int, optional

    Returns
    -------
    candidate_ripple_times : pandas Dataframe

    '''
    zscored_data = zscore(data, nan_policy='omit')
    is_above_mean = zscored_data >= 0
    is_above_threshold = zscored_data >= zscore_threshold

    return extend_threshold_to_mean(
        is_above_mean, is_above_threshold, time,
        minimum_duration=minimum_duration)


def merge_overlapping_ranges(ranges):
    '''Merge overlapping and adjacent ranges

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

    '''
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


def exclude_close_events(candidate_event_times, close_event_threshold=1.0):
    '''Excludes successive events that occur within  a `close_event_threshold`
    of a previously occuring event.

    Parameters
    ----------
    candidate_event_times : ndarray or list, shape (n_events, 2)
        Start and end times of possible events
    close_event_threshold : float or np.timedelta


    Returns
    -------
    candidate_event_times : ndarray, shape (n_events - too_close_events, 2)

    '''
    candidate_event_times = np.array(candidate_event_times)
    n_events = candidate_event_times.shape[0]

    new_event_index = np.arange(n_events)
    new_event_times = candidate_event_times.copy()

    for ind, (start_time, end_time) in enumerate(candidate_event_times):
        if np.isin(ind, new_event_index):
            is_too_close = (
                (end_time + close_event_threshold > new_event_times[:, 0])
                & (new_event_index > ind))
            new_event_index = new_event_index[~is_too_close]
            new_event_times = new_event_times[~is_too_close]

    return new_event_times if new_event_times.size > 0 else []


def get_multiunit_population_firing_rate(multiunit, sampling_frequency,
                                         smoothing_sigma=0.015):
    '''Calculates the multiunit population firing rate.

    Parameters
    ----------
    multiunit : ndarray, shape (n_time, n_signals)
        Binary array of multiunit spike times.
    sampling_frequency : float
        Number of samples per second.
    smoothing_sigma : float or np.timedelta
        Amount to smooth the firing rate over time. The default is
        given assuming time is in units of seconds.


    Returns
    -------
    multiunit_population_firing_rate : ndarray, shape (n_time,)

    '''
    return gaussian_smooth(multiunit.mean(axis=1) * sampling_frequency,
                           smoothing_sigma, sampling_frequency)

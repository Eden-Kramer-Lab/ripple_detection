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
from scipy.stats import median_abs_deviation, zscore


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


def _validate_normalization_params(
    method: str,
    normalization_mask: ArrayLike | None,
    normalization_time_range: tuple[float, float] | None,
    time: ArrayLike | None,
) -> None:
    """Validate normalization parameters.

    Parameters
    ----------
    method : str
        Normalization method to validate.
    normalization_mask : array_like or None
        Boolean mask for normalization subset.
    normalization_time_range : tuple or None
        Time range for normalization subset.
    time : array_like or None
        Time array required if time_range is specified.

    Raises
    ------
    ValueError
        If parameters are invalid or incompatible.

    """
    if normalization_mask is not None and normalization_time_range is not None:
        raise ValueError(
            "Cannot specify both 'normalization_mask' and 'normalization_time_range'. "
            "Choose one method for defining the normalization subset."
        )

    if normalization_time_range is not None and time is None:
        raise ValueError(
            "'time' parameter is required when using 'normalization_time_range'. "
            "Provide the time array corresponding to your data samples."
        )

    if method not in ("zscore", "median_mad"):
        raise ValueError(
            f"Invalid normalization method: '{method}'. "
            "Must be either 'zscore' or 'median_mad'."
        )


def _get_normalization_mask(
    data_shape: tuple[int, ...],
    time: ArrayLike | None,
    normalization_mask: ArrayLike | None,
    normalization_time_range: tuple[float, float] | None,
) -> NDArray | None:
    """Determine which data subset to use for computing normalization statistics.

    Parameters
    ----------
    data_shape : tuple
        Shape of the data array.
    time : array_like or None
        Time values for each sample.
    normalization_mask : array_like or None
        Boolean mask specifying samples to use.
    normalization_time_range : tuple or None
        Time range (start, end) for computing statistics.

    Returns
    -------
    mask : ndarray or None
        Boolean mask indicating which samples to use, or None to use all data.

    Raises
    ------
    ValueError
        If mask length doesn't match data, or time range is empty.

    """
    if normalization_mask is not None:
        mask = np.asarray(normalization_mask, dtype=bool)
        if mask.shape[0] != data_shape[0]:
            raise ValueError(
                f"normalization_mask length ({mask.shape[0]}) must match "
                f"data length ({data_shape[0]})."
            )
        return mask
    elif normalization_time_range is not None:
        time_arr = np.asarray(time)
        start_time, end_time = normalization_time_range
        mask = (time_arr >= start_time) & (time_arr <= end_time)
        if not np.any(mask):
            raise ValueError(
                f"normalization_time_range ({start_time}, {end_time}) does not "
                "contain any data points. Check that the time range is valid."
            )
        return mask
    else:
        return None


def _normalize_zscore(data: NDArray, mask: NDArray | None) -> NDArray:
    """Apply z-score normalization (mean/std).

    Parameters
    ----------
    data : ndarray
        Data to normalize.
    mask : ndarray or None
        Boolean mask for subset to compute statistics from, or None for all data.

    Returns
    -------
    normalized : ndarray
        Z-score normalized data.

    """
    if mask is not None:
        # Compute mean and std from subset, apply to all data
        if data.ndim == 1:
            subset = data[mask]
            mean = np.nanmean(subset)
            std = np.nanstd(subset, ddof=0)
            if std == 0 or np.isnan(std):
                # Avoid division by zero
                return np.zeros_like(data)
            normalized = (data - mean) / std
        else:
            # Handle multi-channel data (n_time, n_channels)
            mean = np.nanmean(data[mask], axis=0, keepdims=True)
            std = np.nanstd(data[mask], axis=0, ddof=0, keepdims=True)
            std[std == 0] = 1.0  # Avoid division by zero
            normalized = (data - mean) / std
    else:
        # Use scipy's zscore with nan_policy='omit' for consistency
        normalized = zscore(data, axis=0, nan_policy="omit", ddof=0)

    return normalized


def _normalize_median_mad(data: NDArray, mask: NDArray | None) -> NDArray:
    """Apply median/MAD normalization.

    Parameters
    ----------
    data : ndarray
        Data to normalize.
    mask : ndarray or None
        Boolean mask for subset to compute statistics from, or None for all data.

    Returns
    -------
    normalized : ndarray
        Median/MAD normalized data.

    """
    if mask is not None:
        # Compute median and MAD from subset, apply to all data
        if data.ndim == 1:
            subset = data[mask]
            median = np.nanmedian(subset)
            mad = median_abs_deviation(subset, scale="normal", nan_policy="omit")
            if mad == 0 or np.isnan(mad):
                # Avoid division by zero
                return np.zeros_like(data)
            normalized = (data - median) / mad
        else:
            # Handle multi-channel data (n_time, n_channels)
            median = np.nanmedian(data[mask], axis=0, keepdims=True)
            mad = median_abs_deviation(data[mask], axis=0, scale="normal", nan_policy="omit")
            # Reshape mad for broadcasting
            mad = mad.reshape(1, -1)
            mad[mad == 0] = 1.0  # Avoid division by zero
            normalized = (data - median) / mad
    else:
        # Compute from all data
        if data.ndim == 1:
            median = np.nanmedian(data)
            mad = median_abs_deviation(data, scale="normal", nan_policy="omit")
            if mad == 0 or np.isnan(mad):
                return np.zeros_like(data)
            normalized = (data - median) / mad
        else:
            # Handle multi-channel data
            median = np.nanmedian(data, axis=0, keepdims=True)
            mad = median_abs_deviation(data, axis=0, scale="normal", nan_policy="omit")
            mad = mad.reshape(1, -1)
            mad[mad == 0] = 1.0
            normalized = (data - median) / mad

    return normalized


def normalize_signal(
    data: ArrayLike,
    time: ArrayLike | None = None,
    method: str = "zscore",
    normalization_mask: ArrayLike | None = None,
    normalization_time_range: tuple[float, float] | None = None,
) -> NDArray:
    """Normalize signal using mean/std (z-score) or median/MAD.

    Provides flexible normalization options for ripple detection. The statistics
    (mean/std or median/MAD) can be computed from the entire signal or from a
    custom subset specified by a mask or time range.

    Parameters
    ----------
    data : array_like, shape (n_time,) or (n_time, n_channels)
        Input signal to normalize. Can be 1D or 2D.
    time : array_like, shape (n_time,), optional
        Time values for each sample. Required if `normalization_time_range`
        is specified. Default is None.
    method : {'zscore', 'median_mad'}, optional
        Normalization method:

        - 'zscore': (data - mean) / std
        - 'median_mad': (data - median) / MAD, where MAD is scaled to be
          comparable to standard deviation for normally distributed data

        Default is 'zscore'. Use 'median_mad' for more robust normalization
        when data contains outliers.
    normalization_mask : array_like, shape (n_time,), optional
        Boolean mask specifying which samples to use for computing normalization
        statistics. True indicates samples to include. For example, use
        `speed < speed_threshold` to compute statistics only during immobility.
        Cannot be used with `normalization_time_range`. Default is None (use all data).
    normalization_time_range : tuple of (float, float), optional
        Time range (start_time, end_time) for computing normalization statistics.
        Requires `time` parameter. Cannot be used with `normalization_mask`.
        Default is None (use all data).

    Returns
    -------
    normalized_data : ndarray, shape matches input
        Normalized signal with the same shape as input.

    Raises
    ------
    ValueError
        If both `normalization_mask` and `normalization_time_range` are specified,
        if `normalization_time_range` is used without `time`, or if `method` is
        not recognized.

    Notes
    -----
    The 'median_mad' method uses `scipy.stats.median_abs_deviation` with
    `scale='normal'`, which applies a scaling factor of ~1.4826 to make MAD
    comparable to standard deviation for Gaussian data. This is equivalent to:

    .. math::

        \\text{normalized} = \\frac{x - \\text{median}(x)}{1.4826 \\cdot \\text{MAD}(x)}

    where MAD is the median absolute deviation from the median.

    Both methods use `nan_policy='omit'` to handle NaN values gracefully.

    Examples
    --------
    Basic z-score normalization:

    >>> import numpy as np
    >>> from ripple_detection import normalize_signal
    >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> normalized = normalize_signal(data, method='zscore')

    Robust median/MAD normalization with outliers:

    >>> data_with_outliers = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    >>> normalized = normalize_signal(data_with_outliers, method='median_mad')

    Normalize using only immobility periods:

    >>> time = np.arange(1000) / 1500  # 1500 Hz sampling
    >>> speed = np.random.rand(1000) * 10  # Speed in cm/s
    >>> lfp = np.random.randn(1000)
    >>> immobility_mask = speed < 4.0
    >>> normalized = normalize_signal(lfp, normalization_mask=immobility_mask)

    Normalize using baseline period:

    >>> baseline_range = (0.0, 10.0)  # First 10 seconds
    >>> normalized = normalize_signal(lfp, time=time,
    ...                               normalization_time_range=baseline_range)

    See Also
    --------
    scipy.stats.zscore : Standard z-score normalization
    scipy.stats.median_abs_deviation : Median absolute deviation

    References
    ----------
    .. [1] Leys, C., et al. (2013). Detecting outliers: Do not use standard
       deviation around the mean, use absolute deviation around the median.
       Journal of Experimental Social Psychology, 49(4), 764-766.

    """
    # Validate parameters
    _validate_normalization_params(method, normalization_mask, normalization_time_range, time)

    # Convert to array and determine mask
    data_arr = np.asarray(data, dtype=float)
    mask = _get_normalization_mask(
        data_arr.shape, time, normalization_mask, normalization_time_range
    )

    # Apply normalization
    if method == "zscore":
        return _normalize_zscore(data_arr, mask)
    else:  # method == "median_mad"
        return _normalize_median_mad(data_arr, mask)


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

    Uses vectorized implementation: computes gaps between consecutive events
    and keeps events with sufficient separation from the previous event.

    Parameters
    ----------
    candidate_event_times : array_like, shape (n_events, 2)
        Array of event times with columns [start_time, end_time].
        Must be sorted by start time.
    close_event_threshold : float, optional
        Minimum time between events. Events starting within this time after
        a previous event ends are excluded. Default is 1.0 (seconds).

    Returns
    -------
    filtered_event_times : ndarray or list
        Filtered event times with shape (n_filtered_events, 2), or empty
        list if no events remain.

    Notes
    -----
    This function assumes events are sorted by start time. If the input
    is not sorted, results may be incorrect.

    """
    candidate_event_times = np.array(candidate_event_times)

    if candidate_event_times.size == 0:
        return []

    # For single event, no filtering needed
    if candidate_event_times.shape[0] == 1:
        return candidate_event_times

    # Extract start and end times
    starts = candidate_event_times[:, 0]
    ends = candidate_event_times[:, 1]

    # Compute gaps: time from end of event i to start of event i+1
    gaps = starts[1:] - ends[:-1]

    # Keep first event and any event with sufficient gap from previous
    keep_mask = np.ones(len(candidate_event_times), dtype=bool)
    keep_mask[1:] = gaps >= close_event_threshold

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

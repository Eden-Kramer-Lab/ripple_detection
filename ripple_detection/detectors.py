"""High-level detectors for sharp-wave ripple events and multiunit synchrony events."""

from itertools import chain

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from ripple_detection.core import (
    exclude_close_events,
    exclude_movement,
    gaussian_smooth,
    get_envelope,
    get_multiunit_population_firing_rate,
    merge_overlapping_ranges,
    normalize_signal,
    threshold_by_zscore,
)


def _validate_lfp_dimensions(filtered_lfps: NDArray) -> None:
    """Validate that LFP array is 2D with shape (n_time, n_channels).

    Parameters
    ----------
    filtered_lfps : ndarray
        LFP array to validate.

    Raises
    ------
    ValueError
        If array is not 2D with appropriate shape.

    """
    if filtered_lfps.ndim == 0:
        raise ValueError(
            "filtered_lfps must be a 2D array with shape (n_time, n_channels).\n"
            "Received a scalar value.\n"
            "Expected: A 2D array where each row is a time point and each column is a channel."
        )
    elif filtered_lfps.ndim == 1:
        raise ValueError(
            "filtered_lfps must be a 2D array with shape (n_time, n_channels).\n"
            f"Received a 1D array with shape {filtered_lfps.shape}.\n"
            "If you have a single channel, reshape your data using:\n"
            "  filtered_lfps = filtered_lfps.reshape(-1, 1)"
        )
    elif filtered_lfps.ndim > 2:
        raise ValueError(
            "filtered_lfps must be a 2D array with shape (n_time, n_channels).\n"
            f"Received a {filtered_lfps.ndim}D array with shape {filtered_lfps.shape}.\n"
            "Expected: 2D array with rows as time points and columns as channels."
        )


def _validate_array_lengths(time: NDArray, filtered_lfps: NDArray, speed: NDArray) -> None:
    """Validate that time, LFP, and speed arrays have matching lengths.

    Parameters
    ----------
    time : ndarray
        Time array.
    filtered_lfps : ndarray
        LFP array.
    speed : ndarray
        Speed array.

    Raises
    ------
    ValueError
        If array lengths don't match.

    """
    n_time_samples = len(time)
    n_lfp_samples = len(filtered_lfps)
    n_speed_samples = len(speed)

    if not (n_time_samples == n_lfp_samples == n_speed_samples):
        raise ValueError(
            "Array length mismatch detected. All inputs must have the same length.\n"
            f"  time:         {n_time_samples} samples\n"
            f"  filtered_lfps: {n_lfp_samples} samples\n"
            f"  speed:        {n_speed_samples} samples\n"
            "Ensure your time, LFP, and speed arrays are aligned and have matching lengths."
        )


def _validate_time_units(time: NDArray, sampling_frequency: float, n_samples: int) -> None:
    """Validate that time array is in seconds (not samples).

    Parameters
    ----------
    time : ndarray
        Time array to validate.
    sampling_frequency : float
        Expected sampling frequency in Hz.
    n_samples : int
        Number of samples.

    Raises
    ------
    ValueError
        If time appears to be in samples instead of seconds.

    Warnings
    --------
    UserWarning
        If time step differs significantly from expected.

    """
    import warnings

    if n_samples > 1:
        median_dt = np.median(np.diff(time))
        expected_dt = 1.0 / sampling_frequency

        # Check if time appears to be in samples instead of seconds
        if median_dt > 10 * expected_dt:
            raise ValueError(
                f"Time array appears to be in samples, not seconds.\n"
                f"Median time step: {median_dt:.6f} (expected ~{expected_dt:.6f} for {sampling_frequency} Hz)\n"
                f"\n"
                f"Solution: Convert sample indices to seconds:\n"
                f"  time_seconds = time_samples / {sampling_frequency}"
            )
        # Check if time step is suspiciously different from sampling frequency
        elif not np.isclose(median_dt, expected_dt, rtol=0.2):
            warnings.warn(
                f"Time array step ({median_dt:.6f} s) differs from expected sampling interval "
                f"({expected_dt:.6f} s at {sampling_frequency} Hz).\n"
                f"Verify that:\n"
                f"  1. time is in seconds (not milliseconds or samples)\n"
                f"  2. sampling_frequency ({sampling_frequency} Hz) is correct",
                UserWarning,
                stacklevel=4,
            )


def _validate_speed_units(speed: NDArray, speed_threshold: float) -> None:
    """Validate that speed is in cm/s (not m/s).

    Parameters
    ----------
    speed : ndarray
        Speed array to validate.
    speed_threshold : float
        Speed threshold in cm/s.

    Warnings
    --------
    UserWarning
        If speed values appear to be in m/s instead of cm/s.

    """
    import warnings

    non_nan_speed = speed[pd.notna(speed)]
    if len(non_nan_speed) > 0:
        non_zero_speed = non_nan_speed[non_nan_speed > 0]
        if len(non_zero_speed) > 0:
            median_speed = np.median(non_zero_speed)
            # If median speed is very small and threshold is typical (> 1 cm/s),
            # user likely passed speed in m/s instead of cm/s
            if median_speed < 0.5 and speed_threshold > 1.0:
                warnings.warn(
                    f"Speed values appear very small (median non-zero: {median_speed:.4f}).\n"
                    f"Speed should be in cm/s, not m/s.\n"
                    f"If your speed is in m/s, multiply by 100:\n"
                    f"  speed_cms = speed_ms * 100",
                    UserWarning,
                    stacklevel=4,
                )


def _preprocess_detector_inputs(
    time: ArrayLike,
    filtered_lfps: ArrayLike,
    speed: ArrayLike,
    sampling_frequency: float,
    speed_threshold: float = 4.0,
) -> tuple[NDArray, NDArray, NDArray]:
    """Remove NaN values from detector inputs and validate units.

    Ensures all inputs are aligned by removing any time points where
    LFP data or speed contains NaN values. Also validates that time
    and speed appear to be in the correct units. This preprocessing
    step is shared by all ripple detectors.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Time values for each sample in seconds.
    filtered_lfps : array_like, shape (n_time, n_channels)
        Bandpass filtered LFP signals.
    speed : array_like, shape (n_time,)
        Animal's running speed in cm/s.
    sampling_frequency : float
        Sampling rate in Hz, used to validate time units.
    speed_threshold : float, optional
        Speed threshold in cm/s, used to validate speed units. Default is 4.0.

    Returns
    -------
    time_clean : ndarray, shape (n_clean_time,)
        Time array with NaN rows removed.
    filtered_lfps_clean : ndarray, shape (n_clean_time, n_channels)
        LFP array with NaN rows removed.
    speed_clean : ndarray, shape (n_clean_time,)
        Speed array with NaN values removed.

    Raises
    ------
    ValueError
        If filtered_lfps is not 2D, if array lengths don't match, or if
        time/speed appear to be in incorrect units.

    Warnings
    --------
    UserWarning
        If speed values appear to be in wrong units (m/s instead of cm/s).

    """
    # Convert to arrays
    filtered_lfps = np.asarray(filtered_lfps)
    speed = np.asarray(speed)
    time = np.asarray(time)

    # Run all validations
    _validate_lfp_dimensions(filtered_lfps)
    _validate_array_lengths(time, filtered_lfps, speed)
    _validate_time_units(time, sampling_frequency, len(time))
    _validate_speed_units(speed, speed_threshold)

    # Remove NaN values
    not_null = np.all(pd.notna(filtered_lfps), axis=1) & pd.notna(speed)

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
    not_null = np.all(pd.notna(ripple_filtered_lfps), axis=1)

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
    normalization_method: str = "zscore",
    normalization_mask: ArrayLike | None = None,
    normalization_time_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Detect sharp-wave ripple events using multi-channel consensus method.

    Implements the Kay et al. 2016 ripple detection algorithm, which combines
    multiple LFP channels into a consensus trace using sum of squared envelopes.
    Ripples are identified as periods where the z-scored consensus exceeds a
    threshold during immobility.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Time values for each sample in **seconds**.
    filtered_lfps : array_like, shape (n_time, n_channels)
        LFP signals **already bandpass filtered** to ripple band (150-250 Hz).
        Must be pre-filtered using `filter_ripple_band()` before calling this detector.
    speed : array_like, shape (n_time,)
        Animal's running speed at each time point in **cm/s**.
    sampling_frequency : float
        Sampling rate in Hz.
    speed_threshold : float, optional
        Maximum speed (in cm/s) for ripple detection. Events during movement
        (speed > threshold) are excluded. Default is 4.0 cm/s, which corresponds
        to immobility/slow movement in rodents.

        **Important**: Ensure your speed data is in cm/s. If using m/s, multiply
        by 100. To disable movement exclusion, set to a very large value (e.g., 1e6).
    minimum_duration : float, optional
        Minimum ripple duration in **seconds**. Default is 0.015 (15 milliseconds).
        Typical range: 0.015 - 0.100 s (15-100 ms). Lower values detect shorter
        events but may increase false positives.
    zscore_threshold : float, optional
        Detection sensitivity threshold in standard deviations above mean.
        Default is 2.0. Lower values (e.g., 1.5) detect more events but may
        include false positives. Higher values (e.g., 3.0) are more conservative.
    smoothing_sigma : float, optional
        Standard deviation of Gaussian smoothing kernel in **seconds**.
        Default is 0.004 (4 ms). Rarely needs adjustment; increase for
        noisier data.
    close_ripple_threshold : float, optional
        Minimum time in **seconds** between ripples. Events closer than this
        are merged. Default is 0.0 (no merging). Set to 0.05-0.1 s to merge
        closely-spaced events.
    normalization_method : {'zscore', 'median_mad'}, optional
        Method for normalizing the consensus trace. Default is 'zscore' (mean/std).
        Use 'median_mad' for more robust normalization when data contains outliers.
        The median/MAD method is more resistant to extreme values.
    normalization_mask : array_like, shape (n_time,), optional
        Boolean mask to specify which samples to use for computing normalization
        statistics. For example, use `speed < speed_threshold` to compute
        statistics only during immobility. Cannot be used with
        `normalization_time_range`. Default is None (use all data).
    normalization_time_range : tuple of (float, float), optional
        Time range (start_time, end_time) in seconds for computing normalization
        statistics. Useful for baseline normalization. Cannot be used with
        `normalization_mask`. Default is None (use all data).

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

        Returns empty DataFrame if no ripples detected. If this occurs, try:
        - Lowering zscore_threshold (e.g., from 2.0 to 1.5)
        - Lowering minimum_duration (e.g., from 0.015 to 0.010)
        - Increasing speed_threshold if movement exclusion is too strict
        - Verifying your data contains ripple oscillations (150-250 Hz)

    Examples
    --------
    >>> from ripple_detection import filter_ripple_band, Kay_ripple_detector
    >>> import numpy as np
    >>>
    >>> # Step 1: Prepare your data
    >>> time = np.arange(10000) / 1500  # 10000 samples at 1500 Hz
    >>> raw_lfps = np.random.randn(10000, 4)  # 4 channels of raw LFP
    >>> speed = np.abs(np.random.randn(10000)) * 5  # Speed in cm/s
    >>>
    >>> # Step 2: Filter LFPs to ripple band (REQUIRED)
    >>> filtered_lfps = filter_ripple_band(raw_lfps, sampling_frequency=1500)
    >>>
    >>> # Step 3: Detect ripples
    >>> ripples = Kay_ripple_detector(time, filtered_lfps, speed, sampling_frequency=1500)
    >>> print(f"Detected {len(ripples)} ripple events")

    References
    ----------
    .. [1] Kay, K., Sosa, M., Chung, J.E., Karlsson, M.P., Larkin, M.C.,
       and Frank, L.M. (2016). A hippocampal network for spatial coding during
       immobility and sleep. Nature 531, 185-190.

    """
    time, filtered_lfps, speed = _preprocess_detector_inputs(
        time, filtered_lfps, speed, sampling_frequency, speed_threshold
    )

    combined_filtered_lfps = get_Kay_ripple_consensus_trace(
        filtered_lfps, sampling_frequency, smoothing_sigma=smoothing_sigma
    )
    combined_filtered_lfps = normalize_signal(
        combined_filtered_lfps,
        time=time,
        method=normalization_method,
        normalization_mask=normalization_mask,
        normalization_time_range=normalization_time_range,
    )
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
    normalization_method: str = "zscore",
    normalization_mask: ArrayLike | None = None,
    normalization_time_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Detect sharp-wave ripples using per-channel detection with merging.

    Implements the Karlsson et al. 2009 algorithm, which detects ripples on
    each LFP channel independently, then merges overlapping events across
    channels. More sensitive to local ripples than consensus methods.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Time values for each sample in **seconds**.
    filtered_lfps : array_like, shape (n_time, n_channels)
        LFP signals **already bandpass filtered** to ripple band (150-250 Hz).
        Must be pre-filtered using `filter_ripple_band()` before calling this detector.
    speed : array_like, shape (n_time,)
        Animal's running speed at each time point in **cm/s**.
    sampling_frequency : float
        Sampling rate in Hz.
    speed_threshold : float, optional
        Maximum speed (in cm/s) for ripple detection. Events during movement
        (speed > threshold) are excluded. Default is 4.0 cm/s, which corresponds
        to immobility/slow movement in rodents.

        **Important**: Ensure your speed data is in cm/s. If using m/s, multiply
        by 100. To disable movement exclusion, set to a very large value (e.g., 1e6).
    minimum_duration : float, optional
        Minimum ripple duration in **seconds**. Default is 0.015 (15 milliseconds).
        Typical range: 0.015 - 0.100 s (15-100 ms). Lower values detect shorter
        events but may increase false positives.
    zscore_threshold : float, optional
        Detection sensitivity threshold in standard deviations above mean.
        Default is 3.0 (higher than Kay's 2.0 because per-channel detection
        is more sensitive). Lower values detect more events.
    smoothing_sigma : float, optional
        Standard deviation of Gaussian smoothing kernel in **seconds**.
        Default is 0.004 (4 ms). Rarely needs adjustment; increase for
        noisier data.
    close_ripple_threshold : float, optional
        Minimum time in **seconds** between ripples. Events closer than this
        are merged. Default is 0.0 (no merging). Set to 0.05-0.1 s to merge
        closely-spaced events.
    normalization_method : {'zscore', 'median_mad'}, optional
        Method for normalizing each channel. Default is 'zscore' (mean/std).
        Use 'median_mad' for more robust normalization when data contains outliers.
        The median/MAD method is more resistant to extreme values.
    normalization_mask : array_like, shape (n_time,), optional
        Boolean mask to specify which samples to use for computing normalization
        statistics. For example, use `speed < speed_threshold` to compute
        statistics only during immobility. Cannot be used with
        `normalization_time_range`. Default is None (use all data).
    normalization_time_range : tuple of (float, float), optional
        Time range (start_time, end_time) in seconds for computing normalization
        statistics. Useful for baseline normalization. Cannot be used with
        `normalization_mask`. Default is None (use all data).

    Returns
    -------
    ripple_times : pd.DataFrame
        DataFrame with detected ripples and comprehensive statistics (see
        Kay_ripple_detector for column descriptions).

        Returns empty DataFrame if no ripples detected. If this occurs, try:
        - Lowering zscore_threshold (e.g., from 3.0 to 2.0)
        - Lowering minimum_duration (e.g., from 0.015 to 0.010)
        - Increasing speed_threshold if movement exclusion is too strict
        - Verifying your data contains ripple oscillations (150-250 Hz)

    References
    ----------
    .. [1] Karlsson, M.P., and Frank, L.M. (2009). Awake replay of remote
       experiences in the hippocampus. Nature Neuroscience 12, 913-918.

    """
    time, filtered_lfps, speed = _preprocess_detector_inputs(
        time, filtered_lfps, speed, sampling_frequency, speed_threshold
    )

    filtered_lfps = get_envelope(filtered_lfps)
    filtered_lfps = gaussian_smooth(
        filtered_lfps, sigma=smoothing_sigma, sampling_frequency=sampling_frequency
    )
    filtered_lfps = normalize_signal(
        filtered_lfps,
        time=time,
        method=normalization_method,
        normalization_mask=normalization_mask,
        normalization_time_range=normalization_time_range,
    )
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
    normalization_method: str = "zscore",
    normalization_mask: ArrayLike | None = None,
    normalization_time_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Detect sharp-wave ripples using averaged square-root envelope method.

    Variant detection method that averages the square-root of squared envelopes
    across channels. Provides a balanced approach between Kay (consensus) and
    Karlsson (per-channel) methods.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Time values for each sample in **seconds**.
    filtered_lfps : array_like, shape (n_time, n_channels)
        LFP signals **already bandpass filtered** to ripple band (150-250 Hz).
        Must be pre-filtered using `filter_ripple_band()` before calling this detector.
    speed : array_like, shape (n_time,)
        Animal's running speed at each time point in **cm/s**.
    sampling_frequency : float
        Sampling rate in Hz.
    speed_threshold : float, optional
        Maximum speed (in cm/s) for ripple detection. Events during movement
        (speed > threshold) are excluded. Default is 4.0 cm/s, which corresponds
        to immobility/slow movement in rodents.

        **Important**: Ensure your speed data is in cm/s. If using m/s, multiply
        by 100. To disable movement exclusion, set to a very large value (e.g., 1e6).
    minimum_duration : float, optional
        Minimum ripple duration in **seconds**. Default is 0.015 (15 milliseconds).
        Typical range: 0.015 - 0.100 s (15-100 ms). Lower values detect shorter
        events but may increase false positives.
    zscore_threshold : float, optional
        Detection sensitivity threshold in standard deviations above mean.
        Default is 2.0. Lower values (e.g., 1.5) detect more events but may
        include false positives. Higher values (e.g., 3.0) are more conservative.
    smoothing_sigma : float, optional
        Standard deviation of Gaussian smoothing kernel in **seconds**.
        Default is 0.004 (4 ms). Rarely needs adjustment; increase for
        noisier data.
    close_ripple_threshold : float, optional
        Minimum time in **seconds** between ripples. Events closer than this
        are merged. Default is 0.0 (no merging). Set to 0.05-0.1 s to merge
        closely-spaced events.
    normalization_method : {'zscore', 'median_mad'}, optional
        Method for normalizing the combined trace. Default is 'zscore' (mean/std).
        Use 'median_mad' for more robust normalization when data contains outliers.
        The median/MAD method is more resistant to extreme values.
    normalization_mask : array_like, shape (n_time,), optional
        Boolean mask to specify which samples to use for computing normalization
        statistics. For example, use `speed < speed_threshold` to compute
        statistics only during immobility. Cannot be used with
        `normalization_time_range`. Default is None (use all data).
    normalization_time_range : tuple of (float, float), optional
        Time range (start_time, end_time) in seconds for computing normalization
        statistics. Useful for baseline normalization. Cannot be used with
        `normalization_mask`. Default is None (use all data).

    Returns
    -------
    ripple_times : pd.DataFrame
        DataFrame with detected ripples and comprehensive statistics (see
        Kay_ripple_detector for column descriptions).

        Returns empty DataFrame if no ripples detected. If this occurs, try:
        - Lowering zscore_threshold (e.g., from 2.0 to 1.5)
        - Lowering minimum_duration (e.g., from 0.015 to 0.010)
        - Increasing speed_threshold if movement exclusion is too strict
        - Verifying your data contains ripple oscillations (150-250 Hz)

    """
    time, filtered_lfps, speed = _preprocess_detector_inputs(
        time, filtered_lfps, speed, sampling_frequency, speed_threshold
    )

    filtered_lfps = get_envelope(filtered_lfps) ** 2
    filtered_lfps = gaussian_smooth(
        filtered_lfps, sigma=smoothing_sigma, sampling_frequency=sampling_frequency
    )
    combined_filtered_lfps = np.mean(np.sqrt(filtered_lfps), axis=1)
    combined_filtered_lfps = normalize_signal(
        combined_filtered_lfps,
        time=time,
        method=normalization_method,
        normalization_mask=normalization_mask,
        normalization_time_range=normalization_time_range,
    )
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
    normalization_method: str = "zscore",
    normalization_mask: ArrayLike | None = None,
    normalization_time_range: tuple[float, float] | None = None,
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
        Spike indicator matrix for each unit at each time point.
        Can be either:
        - **Binary** (0 = no spike, 1 = spike) - recommended for consistent results
        - **Spike counts** (0, 1, 2, ...) - also supported, represents number of spikes per bin

        Both formats work, but may produce different sensitivities. For multi-spike
        bins, results are typically more consistent with binary format.
    speed : array_like, shape (n_time,)
        Animal's running speed at each time point.
    sampling_frequency : float
        Sampling rate in Hz.
    speed_threshold : float, optional
        Maximum speed (in cm/s) for event detection. Events during movement
        (speed > threshold) are excluded. Default is 4.0 cm/s, which corresponds
        to immobility/slow movement in rodents.

        **Important**: Ensure your speed data is in cm/s. If using m/s, multiply
        by 100. To disable movement exclusion, set to a very large value (e.g., 1e6).
    minimum_duration : float, optional
        Minimum event duration in **seconds**. Default is 0.015 (15 milliseconds).
        Typical range: 0.015 - 0.100 s (15-100 ms). Lower values detect shorter
        events but may increase false positives.
    zscore_threshold : float, optional
        Detection sensitivity threshold in standard deviations above mean.
        Default is 2.0. Lower values (e.g., 1.5) detect more events but may
        include false positives. Higher values (e.g., 3.0) are more conservative.
    smoothing_sigma : float, optional
        Standard deviation of Gaussian smoothing kernel in **seconds**.
        Default is 0.015 (15 ms, longer than ripple detectors for smoother
        population firing rate estimates).
    close_event_threshold : float, optional
        Minimum time in **seconds** between events. Events closer than this
        are merged. Default is 0.0 (no merging). Set to 0.05-0.1 s to merge
        closely-spaced events.
    use_speed_threshold_for_zscore : bool, optional
        **DEPRECATED**: Use `normalization_mask` instead. If True, compute
        z-score statistics (mean/std) using only immobility periods (speed <
        threshold). Default is False (use all time points). This parameter is
        maintained for backwards compatibility but will be removed in a future
        version.
    normalization_method : {'zscore', 'median_mad'}, optional
        Method for normalizing the firing rate. Default is 'zscore' (mean/std).
        Use 'median_mad' for more robust normalization when data contains outliers.
        The median/MAD method is more resistant to extreme values.
    normalization_mask : array_like, shape (n_time,), optional
        Boolean mask to specify which samples to use for computing normalization
        statistics. For example, use `speed < speed_threshold` to compute
        statistics only during immobility. Cannot be used with
        `normalization_time_range` or `use_speed_threshold_for_zscore`.
        Default is None (use all data).
    normalization_time_range : tuple of (float, float), optional
        Time range (start_time, end_time) in seconds for computing normalization
        statistics. Useful for baseline normalization. Cannot be used with
        `normalization_mask` or `use_speed_threshold_for_zscore`.
        Default is None (use all data).

    Returns
    -------
    high_synchrony_events : pd.DataFrame
        DataFrame with detected events and comprehensive statistics (see
        Kay_ripple_detector for column descriptions).

        Returns empty DataFrame if no events detected. If this occurs, try:
        - Lowering zscore_threshold (e.g., from 2.0 to 1.5)
        - Lowering minimum_duration (e.g., from 0.015 to 0.010)
        - Increasing speed_threshold if movement exclusion is too strict
        - Verifying your multiunit data shows synchronous spiking activity

    References
    ----------
    .. [1] Davidson, T.J., Kloosterman, F., and Wilson, M.A. (2009).
       Hippocampal Replay of Extended Experience. Neuron 63, 497-507.

    """
    multiunit = np.asarray(multiunit)
    speed = np.asarray(speed)
    time = np.asarray(time)

    firing_rate = get_multiunit_population_firing_rate(
        multiunit, sampling_frequency, smoothing_sigma
    )

    # Handle backwards compatibility with use_speed_threshold_for_zscore
    if use_speed_threshold_for_zscore:
        import warnings

        warnings.warn(
            "The 'use_speed_threshold_for_zscore' parameter is deprecated. "
            "Use 'normalization_mask=speed < speed_threshold' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # If old parameter is used, override normalization_mask unless explicitly set
        if normalization_mask is None and normalization_time_range is None:
            normalization_mask = speed < speed_threshold

    firing_rate = normalize_signal(
        firing_rate,
        time=time,
        method=normalization_method,
        normalization_mask=normalization_mask,
        normalization_time_range=normalization_time_range,
    )
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
    try:
        from numpy import trapezoid
    except ImportError:
        # NumPy 1.x
        from numpy import trapz as trapezoid

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
        area.append(trapezoid(event_zscore, time_arr[ind]))
        total_energy.append(trapezoid(event_zscore**2, time_arr[ind]))
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

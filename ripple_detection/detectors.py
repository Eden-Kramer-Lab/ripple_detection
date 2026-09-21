"""High-level detectors for sharp-wave ripple events and multiunit synchrony events."""

from itertools import chain, pairwise

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt

from ripple_detection.core import (
    _get_normalization_mask,
    estimate_noise_threshold,
    exclude_close_events,
    exclude_movement,
    exclude_movement_by_majority,
    gaussian_smooth,
    get_envelope,
    get_multiunit_population_firing_rate,
    merge_overlapping_ranges,
    merge_overlapping_ranges_track_participation,
    normalize_signal,
    normalize_signal_manually,
    threshold_by_zscore,
)

# NumPy 2.x renamed trapz to trapezoid
if hasattr(np, "trapezoid"):
    trapezoid = np.trapezoid
else:
    trapezoid = np.trapz  # type: ignore[attr-defined]  # noqa: NPY201


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
    normalization_mask: ArrayLike | None = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray | None]:
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
    normalization_mask : array_like, shape (n_time,), optional
        Boolean mask over the original time samples. Filtered by the same NaN
        removal so it stays aligned with the cleaned data. Default is None.

    Returns
    -------
    time_clean : ndarray, shape (n_clean_time,)
        Time array with NaN rows removed.
    filtered_lfps_clean : ndarray, shape (n_clean_time, n_channels)
        LFP array with NaN rows removed.
    speed_clean : ndarray, shape (n_clean_time,)
        Speed array with NaN values removed.
    normalization_mask_clean : ndarray or None, shape (n_clean_time,)
        The normalization mask with the same NaN rows removed, or None if no
        mask was provided.

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
    filtered_lfps = np.asarray(filtered_lfps, dtype=float)
    speed = np.asarray(speed)
    time = np.asarray(time)

    # Run all validations
    _validate_lfp_dimensions(filtered_lfps)
    _validate_array_lengths(time, filtered_lfps, speed)
    _validate_time_units(time, sampling_frequency, len(time))
    _validate_speed_units(speed, speed_threshold)

    # Remove NaN values
    not_null = np.all(pd.notna(filtered_lfps), axis=1) & pd.notna(speed)

    # Filter the normalization mask by the same rows so it stays aligned with
    # the cleaned data (otherwise its length no longer matches after NaN removal).
    if normalization_mask is not None:
        normalization_mask = np.asarray(normalization_mask)
        if len(normalization_mask) != len(not_null):
            raise ValueError(
                f"normalization_mask length ({len(normalization_mask)}) must match "
                f"the number of time samples ({len(not_null)})."
            )
        normalization_mask = normalization_mask[not_null]

    return time[not_null], filtered_lfps[not_null], speed[not_null], normalization_mask


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
    # Cast to float so integer input is not truncated and the squared envelope
    # cannot overflow before the square root.
    ripple_filtered_lfps = np.asarray(ripple_filtered_lfps, dtype=float)
    ripple_consensus_trace = np.full_like(ripple_filtered_lfps, np.nan)
    not_null = np.all(pd.notna(ripple_filtered_lfps), axis=1)

    ripple_consensus_trace[not_null] = get_envelope(np.asarray(ripple_filtered_lfps)[not_null])
    ripple_consensus_trace = np.sum(ripple_consensus_trace**2, axis=1)
    ripple_consensus_trace[not_null] = gaussian_smooth(
        ripple_consensus_trace[not_null], smoothing_sigma, sampling_frequency
    )
    return np.sqrt(ripple_consensus_trace)


def _contiguous_valid_blocks(
    is_valid: NDArray, time: NDArray | None, sampling_frequency: float
) -> list[tuple[int, int]]:
    """Split rows into maximal contiguous valid blocks.

    A block ends at an invalid row or, when ``time`` is given, wherever the
    timestamp step exceeds 1.5 sample intervals (a recording gap or the join
    between disjoint intervals).

    Parameters
    ----------
    is_valid : ndarray of bool, shape (n_time,)
        True for rows with finite data in every channel.
    time : ndarray, shape (n_time,), optional
        Sample timestamps in seconds. None declares a regular sample grid.
    sampling_frequency : float
        Nominal sampling rate in Hz.

    Returns
    -------
    blocks : list of (start, stop)
        Half-open row ranges, in order.

    """
    n_time = len(is_valid)
    boundary = np.zeros(n_time + 1, dtype=bool)
    boundary[0] = boundary[-1] = True
    # a block boundary sits between rows i-1 and i where validity changes
    boundary[1:-1] |= is_valid[1:] != is_valid[:-1]
    if time is not None:
        boundary[1:-1] |= np.diff(time) > 1.5 / sampling_frequency
    edges = np.flatnonzero(boundary)
    return [(int(start), int(stop)) for start, stop in pairwise(edges) if is_valid[start]]


def get_Yu_ripple_consensus_trace(
    ripple_filtered_lfps: ArrayLike,
    sampling_frequency: float,
    smoothing_sigma: float = 0.004,
    zscore_per_tetrode: bool = True,
    *,
    time: ArrayLike | None = None,
) -> NDArray:
    """Compute the Yu et al. 2017 consensus trace: median of per-tetrode envelopes.

    Each channel's ripple-band envelope is smoothed with a Gaussian kernel and,
    by default, z-scored over the whole recording; the consensus is the median
    across channels at each sample. A median rather than a sum keeps one
    tetrode from dominating the trace.

    Envelope and smoothing run separately inside each maximal contiguous block
    of valid samples, so missing data never bleeds across a gap; the per-channel
    z-score statistics are pooled over all valid samples. Rows with a non-finite
    value in any channel are returned as NaN.

    Parameters
    ----------
    ripple_filtered_lfps : array_like, shape (n_time, n_channels)
        Bandpass filtered LFP signals in the ripple band (150-250 Hz).
    sampling_frequency : float
        Sampling rate in Hz.
    smoothing_sigma : float, optional
        Standard deviation of the Gaussian smoothing kernel in seconds, applied
        per channel before aggregation. Default is 0.004 (4 ms).
    zscore_per_tetrode : bool, optional
        If True (default), z-score each channel's smoothed envelope (sample
        standard deviation, ``ddof=1``) over all valid samples before taking
        the median, as the original lab implementation does. If False, take
        the median of the raw smoothed envelopes.
    time : array_like, shape (n_time,), optional
        Sample timestamps in seconds. When given, a step larger than 1.5
        sample intervals also ends a block, so disjoint intervals that were
        concatenated are not smoothed across. Default is None, which declares
        a regular sample grid.

    Returns
    -------
    consensus_trace : ndarray, shape (n_time,)
        Median across channels of the smoothed (and z-scored) envelopes, with
        NaN wherever any channel was non-finite.

    Raises
    ------
    ValueError
        If the input is not 2-D, if ``time`` does not match its length, if no
        valid samples exist, or if a channel's standard deviation over the
        valid samples is zero or non-finite when z-scoring is enabled.

    References
    ----------
    .. [1] Yu, J. Y., et al. (2017). Distinct hippocampal-cortical memory
       representations for experiences associated with movement versus
       immobility. eLife, 6, e27621.

    """
    ripple_filtered_lfps = np.asarray(ripple_filtered_lfps, dtype=float)
    _validate_lfp_dimensions(ripple_filtered_lfps)
    n_time = ripple_filtered_lfps.shape[0]
    if time is not None:
        time = np.asarray(time, dtype=float)
        if time.shape != (n_time,):
            raise ValueError(
                f"time has shape {time.shape} but filtered_lfps has {n_time} samples."
            )

    is_valid = np.all(np.isfinite(ripple_filtered_lfps), axis=1)
    if not np.any(is_valid):
        raise ValueError("No sample has finite values in every channel.")

    smoothed = np.full_like(ripple_filtered_lfps, np.nan)
    for start, stop in _contiguous_valid_blocks(is_valid, time, sampling_frequency):
        envelope = get_envelope(ripple_filtered_lfps[start:stop])
        smoothed[start:stop] = gaussian_smooth(envelope, smoothing_sigma, sampling_frequency)

    if zscore_per_tetrode:
        valid_rows = smoothed[is_valid]
        mean = valid_rows.mean(axis=0, keepdims=True)
        std = (
            valid_rows.std(axis=0, ddof=1, keepdims=True)
            if valid_rows.shape[0] > 1
            else np.full((1, smoothed.shape[1]), np.nan)
        )
        bad = ~np.isfinite(std) | (std <= 0)
        if np.any(bad):
            raise ValueError(
                "Cannot z-score channels with zero or undefined standard deviation "
                f"over the valid samples: channel indices {np.flatnonzero(bad).tolist()}."
            )
        smoothed = (smoothed - mean) / std

    consensus_trace = np.full(n_time, np.nan)
    consensus_trace[is_valid] = np.median(smoothed[is_valid], axis=1)
    return consensus_trace


def _boolean_runs(mask: NDArray) -> NDArray:
    """Start (inclusive) and stop (exclusive) indices of each run of True."""
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return changes.reshape(-1, 2)


def _extract_Yu_ripple_events(
    trace: NDArray,
    time: NDArray,
    sampling_frequency: float,
    minimum_duration: float,
    threshold: float,
) -> tuple[NDArray, NDArray, NDArray]:
    """Extract events from one contiguous block of a mean-zero consensus trace.

    A run of consecutive samples at or above ``threshold`` qualifies when it
    holds at least ``round(minimum_duration * sampling_frequency)`` samples;
    each qualifying run is extended to the run of samples strictly above zero
    (the immobility mean) that contains it, and one event is emitted per such
    containing run. This is the sample-count convention of the Frank lab
    ``extractevents`` routine, which the Yu et al. 2017 detector used.

    Parameters
    ----------
    trace : ndarray, shape (n_time,)
        Consensus trace normalized so the immobility mean is zero, for one
        contiguous block with no missing samples.
    time : ndarray, shape (n_time,)
        Native timestamps of the block's samples, in seconds.
    sampling_frequency : float
        Sampling rate in Hz.
    minimum_duration : float
        Minimum time the trace must stay at or above ``threshold``, in
        seconds; converted to a sample count with round-half-up.
    threshold : float
        Detection threshold in the trace's normalized units; must be finite
        and strictly positive.

    Returns
    -------
    event_times : ndarray, shape (n_events, 2)
        ``[start_time, end_time]`` of each event: the native timestamps of the
        first and last samples of the containing above-zero run.
    is_clipped : ndarray of bool, shape (n_events, 2)
        Whether the event's start or end coincides with the block edge, i.e.
        the run was truncated by the end of the available data.
    n_suprathreshold_samples : ndarray of int, shape (n_events,)
        Sample count of the longest qualifying run inside each event.

    Raises
    ------
    ValueError
        If ``threshold`` is not finite or not strictly positive, since the
        threshold-then-mean-crossing rule is undefined otherwise.

    """
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(
            f"threshold must be finite and strictly above the zero mean, got {threshold}."
        )
    trace = np.asarray(trace, dtype=float)
    time = np.asarray(time, dtype=float)
    n_min = max(1, int(np.floor(minimum_duration * sampling_frequency + 0.5)))

    supra_runs = _boolean_runs(trace >= threshold)
    supra_runs = supra_runs[(supra_runs[:, 1] - supra_runs[:, 0]) >= n_min]
    if len(supra_runs) == 0:
        return np.empty((0, 2)), np.empty((0, 2), dtype=bool), np.empty(0, dtype=int)

    above_zero_runs = _boolean_runs(trace > 0)
    # the above-zero run containing each qualifying run's first sample
    containing = np.searchsorted(above_zero_runs[:, 0], supra_runs[:, 0], side="right") - 1
    run_lengths = supra_runs[:, 1] - supra_runs[:, 0]

    event_times = []
    is_clipped = []
    n_suprathreshold = []
    for run_index in np.unique(containing):
        start, stop = above_zero_runs[run_index]
        event_times.append((time[start], time[stop - 1]))
        is_clipped.append((start == 0, stop == len(trace)))
        n_suprathreshold.append(int(run_lengths[containing == run_index].max()))
    return (
        np.asarray(event_times, dtype=float),
        np.asarray(is_clipped, dtype=bool),
        np.asarray(n_suprathreshold, dtype=int),
    )


def Shvartsman_ripple_detector(
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
    manual_normalization: bool = False,
    elec_baselines: ArrayLike | None = None,
    elec_deviations: ArrayLike | None = None,
    participation_threshold: float = 2,
) -> pd.DataFrame:
    """Detect sharp-wave ripples using per-channel detection, only considering
    times when the % of participating channels exceeds a set fraction. Acts
    as a middle ground between Kay method (consensus method) and Karlsson
    method (local ripples) that is less sensitive to random noise
    fluctuations than the Karlsson method.

    Additionally, allows for manual normalization by passing in specific
    inputs for the baselines and deviations for each electrode. For example,
    if you want to normalize across all epochs throughout a day rather than
    within one particular epoch (important for detecting ripples during sleep
    sessions), this method allows that flexibility.

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
        This is the minimum time the signal must stay *above* ``zscore_threshold``
        (per Karlsson et al. 2009); the event is then extended to the surrounding
        mean-crossings, so the reported ``duration`` is typically longer.
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
        are excluded -- the later event is dropped, not merged. Default is 0.0
        (no exclusion). Set to 0.05-0.1 s to drop closely-spaced events.
    normalization_method : {'zscore', 'median_mad'}, optional
        Method for normalizing each channel. Default is 'zscore' (mean/std).
        Use 'median_mad' for more robust normalization when data contains outliers.
        Only used when ``manual_normalization=False``; ignored otherwise.
    normalization_mask : array_like, shape (n_time,), optional
        Boolean mask selecting samples used to compute normalization statistics.
        For example, use `speed < speed_threshold` to compute statistics only
        during immobility. Cannot be used with `normalization_time_range`. Only
        used when ``manual_normalization=False``. Default is None (use all data).
    normalization_time_range : tuple of (float, float), optional
        Time range (start_time, end_time) in seconds for computing normalization
        statistics. Cannot be used with `normalization_mask`. Only used when
        ``manual_normalization=False``. Default is None (use all data).
    manual_normalization : bool, optional
        If True, normalize each channel with the supplied `elec_baselines` and
        `elec_deviations` instead of computing statistics from the data; the
        `normalization_*` parameters above are then ignored. Requires both
        `elec_baselines` and `elec_deviations` (raises ValueError if either is
        missing). Default is False.
    elec_baselines : array_like, shape (n_channels,), optional
        Baseline (center) value per channel. Required when
        ``manual_normalization=True``.
    elec_deviations : array_like, shape (n_channels,), optional
        Deviation (scale) value per channel. Required when
        ``manual_normalization=True``.
    participation_threshold : float, optional
        Participation cutoff for a merged event. If in [0, 1], interpreted as
        the *fraction* of channels that must participate (note 1.0 means all
        channels, not one). If > 1, interpreted as an absolute *number* of
        channels. Default is 2. Each distinct channel with a detected ripple
        anywhere in the merged event counts once, including channels connected
        through a chain of overlapping ripples.

        The denominator for the fraction (and for `frac_participants`) is the
        total number of channels in `filtered_lfps`, including any channel that
        was dropped as degenerate during manual normalization (zero/NaN
        deviation or NaN baseline). A dead channel therefore lowers
        `frac_participants` and makes a fractional threshold of 1.0
        unsatisfiable; drop known-bad channels before calling, or use an
        absolute (> 1) threshold, if that is a concern.

    Returns
    -------
    ripple_times : pd.DataFrame
        DataFrame with detected ripples and comprehensive statistics (see
        Kay_ripple_detector for the shared columns). This detector additionally
        returns ``participants`` (set of every channel whose ripple appears
        anywhere in the event), ``n_participants`` (``len(participants)``), and
        ``frac_participants`` (``n_participants`` / total channels). Participation
        is measured on each channel's full zero-crossing-extended ripple (the same
        extent as the event boundaries). The per-event z-score statistics are
        averaged over the ``participants`` union.

        Returns empty DataFrame if no ripples detected. If this occurs, try:
        - Lowering zscore_threshold (e.g., from 3.0 to 2.0)
        - Lowering minimum_duration (e.g., from 0.015 to 0.010)
        - Increasing speed_threshold if movement exclusion is too strict
        - Verifying your data contains ripple oscillations (150-250 Hz)

    """
    time, filtered_lfps, speed, normalization_mask = _preprocess_detector_inputs(
        time,
        filtered_lfps,
        speed,
        sampling_frequency,
        speed_threshold,
        # normalization_mask is ignored under manual normalization, so don't
        # validate/filter it in that mode (the docstring promises it is unused).
        normalization_mask=None if manual_normalization else normalization_mask,
    )

    filtered_lfps = get_envelope(filtered_lfps)
    filtered_lfps = gaussian_smooth(
        filtered_lfps, sigma=smoothing_sigma, sampling_frequency=sampling_frequency
    )

    if manual_normalization:
        if elec_baselines is None or elec_deviations is None:
            raise ValueError(
                "Must provide elec_baselines and elec_deviations for manual normalization."
            )
        if len(elec_baselines) != len(elec_deviations):
            raise ValueError(
                "Provided elec_baselines and elec_deviations must be the same length."
            )
        if len(elec_baselines) != filtered_lfps.shape[1]:
            raise ValueError(
                "Provided elec_baselines/elec_deviations must have one entry per "
                f"channel (n_channels={filtered_lfps.shape[1]}), got {len(elec_baselines)}."
            )
        # elec_deviations must be std-equivalent: if it was computed as a MAD,
        # multiply by 1.4826 before passing it in.
        filtered_lfps = normalize_signal_manually(
            filtered_lfps,
            elec_baselines,
            elec_deviations,
        )
    else:
        filtered_lfps = normalize_signal(
            filtered_lfps,
            time=time,
            method=normalization_method,
            normalization_mask=normalization_mask,
            normalization_time_range=normalization_time_range,
        )

    # thresholding the normalized ripple times
    candidate_ripple_times = [
        threshold_by_zscore(filtered_lfp, time, minimum_duration, zscore_threshold)
        for filtered_lfp in filtered_lfps.T
    ]

    # Merge each channel's mean-crossing-extended intervals and retain the union
    # of contributing channels, preserving the original participation rule.
    merged_candidates = merge_overlapping_ranges_track_participation(candidate_ripple_times)

    # account for different ways to specify participation threshold (fraction or number of electrodes)
    n_elecs = filtered_lfps.shape[1]
    if participation_threshold < 0:
        raise ValueError("participation_threshold must be non-negative.")
    if participation_threshold <= 1:
        # interpret as a fraction of channels (1.0 means all channels)
        n_elecs_thresh = n_elecs * participation_threshold
    else:
        # interpret as an absolute number of channels
        n_elecs_thresh = participation_threshold

    participation_mask = (
        np.asarray([len(interval[2]) for interval in merged_candidates]) >= n_elecs_thresh
    )
    candidate_ripple_times = merged_candidates[participation_mask, :2]

    candidate_ripple_times, included_ripple_inds = exclude_movement_by_majority(
        candidate_ripple_times, speed, time, speed_threshold=speed_threshold
    )
    ripple_times, included_ripple_inds = exclude_close_events(
        candidate_ripple_times, close_ripple_threshold, included_ripple_inds
    )

    # Keep participant metadata aligned through movement and proximity exclusion.
    participant_sets = merged_candidates[participation_mask, 2]
    participants = participant_sets[included_ripple_inds]
    n_participants = np.array([len(p) for p in participants])
    frac_participants = n_participants / n_elecs

    # get final event stats
    ripple_data = _get_event_stats(
        ripple_times,
        time,
        filtered_lfps,
        speed,
        minimum_duration,
        participants,
        n_participants,
        frac_participants,
    )

    return ripple_data


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
        This is the minimum time the signal must stay *above* ``zscore_threshold``
        (per Karlsson et al. 2009); the event is then extended to the surrounding
        mean-crossings, so the reported ``duration`` is typically longer.
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
        are excluded -- the later event is dropped, not merged. Default is 0.0
        (no exclusion). Set to 0.05-0.1 s to drop closely-spaced events.
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
    time, filtered_lfps, speed, normalization_mask = _preprocess_detector_inputs(
        time,
        filtered_lfps,
        speed,
        sampling_frequency,
        speed_threshold,
        normalization_mask=normalization_mask,
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


def Yu_ripple_detector(
    time: ArrayLike,
    filtered_lfps: ArrayLike,
    speed: ArrayLike,
    sampling_frequency: float,
    speed_threshold: float = 4.0,
    minimum_duration: float = 0.020,
    percentile: float = 99.99,
    smoothing_sigma: float = 0.004,
    close_ripple_threshold: float = 0.0,
    normalization_mask: ArrayLike | None = None,
    normalization_time_range: tuple[float, float] | None = None,
    zscore_per_tetrode: bool = True,
) -> pd.DataFrame:
    """Detect sharp-wave ripples with a data-driven noise threshold (Yu et al. 2017).

    The consensus trace is the median across tetrodes of each tetrode's
    smoothed, z-scored ripple-band envelope (``get_Yu_ripple_consensus_trace``).
    Its values during immobility are taken as noise plus a signal tail; the
    distribution below the mode is mirrored about the mode to estimate the
    noise distribution, and the detection threshold is the ``percentile`` of
    that mirrored distribution (``estimate_noise_threshold``). Events are runs
    of at least ``minimum_duration`` at or above the threshold, extended to
    where the trace returns to the immobility mean.

    Unlike the other detectors in this module, samples with missing data are
    not dropped before processing: the recording is split into contiguous
    valid blocks, and smoothing, thresholding, and event extraction never
    cross a gap. An event truncated by a gap or by the end of the recording is
    kept and flagged in ``clipped_start`` / ``clipped_end``.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Time values for each sample in seconds.
    filtered_lfps : array_like, shape (n_time, n_channels)
        LFP signals already bandpass filtered to the ripple band (150-250 Hz),
        e.g. with ``filter_ripple_band``. NaN marks missing samples.
    speed : array_like, shape (n_time,)
        Animal's running speed in cm/s.
    sampling_frequency : float
        Sampling rate in Hz.
    speed_threshold : float, optional
        Immobility is speed strictly below this value (cm/s); it selects the
        noise sample for the threshold and, at event boundaries, which events
        are kept. Default is 4.0.
    minimum_duration : float, optional
        Minimum time the consensus must stay at or above the threshold, in
        seconds, applied as a sample count (round-half-up). Default is 0.020.
    percentile : float, optional
        Percentile of the mirrored noise distribution used as the threshold.
        Default is 99.99.
    smoothing_sigma : float, optional
        Standard deviation of the per-tetrode Gaussian envelope smoothing, in
        seconds. Default is 0.004 (4 ms).
    close_ripple_threshold : float, optional
        Minimum separation between events in seconds; a later event starting
        within this time of the previous event's end is dropped. Default is
        0.0 (no exclusion).
    normalization_mask : array_like, shape (n_time,), optional
        Boolean mask selecting the noise sample instead of ``speed <
        speed_threshold``. Cannot be combined with ``normalization_time_range``.
    normalization_time_range : tuple of (float, float), optional
        Time range selecting the noise sample instead of the speed rule.
    zscore_per_tetrode : bool, optional
        Z-score each tetrode's smoothed envelope before the median, as the
        original implementation does; the threshold is then estimated on that
        trace and converted to immobility-normalized units. If False, the
        median of raw envelopes is normalized to immobility first and the
        threshold is estimated on the normalized trace, the reading of the
        published text. Default is True.

    Returns
    -------
    ripple_times : pd.DataFrame
        One row per event, indexed by ``event_number``, with the columns of
        the other detectors (``start_time``, ``end_time``, ``duration``,
        ``max_thresh``, z-score and speed statistics) plus ``clipped_start``
        and ``clipped_end`` (event truncated by missing data or the recording
        edge), ``n_suprathreshold_samples`` (longest run at or above the
        threshold), and ``detection_threshold_zscore`` (the threshold in the
        normalized units the statistics are reported in).

    Raises
    ------
    ValueError
        If inputs are malformed, no valid immobility samples exist, the noise
        distribution cannot be resolved (see ``estimate_noise_threshold``), or
        the estimated threshold does not lie above the immobility mean.

    Notes
    -----
    ``max_thresh`` uses the same sample-count duration convention as event
    selection, so an event's ``max_thresh`` is never undefined.

    References
    ----------
    .. [1] Yu, J. Y., et al. (2017). Distinct hippocampal-cortical memory
       representations for experiences associated with movement versus
       immobility. eLife, 6, e27621.

    """
    filtered_lfps = np.asarray(filtered_lfps, dtype=float)
    speed = np.asarray(speed, dtype=float)
    time = np.asarray(time, dtype=float)
    _validate_lfp_dimensions(filtered_lfps)
    _validate_array_lengths(time, filtered_lfps, speed)
    _validate_time_units(time, sampling_frequency, len(time))
    _validate_speed_units(speed, speed_threshold)

    consensus = get_Yu_ripple_consensus_trace(
        filtered_lfps,
        sampling_frequency,
        smoothing_sigma=smoothing_sigma,
        zscore_per_tetrode=zscore_per_tetrode,
        time=time,
    )
    is_valid = np.isfinite(consensus) & np.isfinite(speed)

    noise_mask = _get_normalization_mask(
        consensus.shape, time, normalization_mask, normalization_time_range
    )
    if noise_mask is None:
        noise_mask = speed < speed_threshold
    noise_mask = noise_mask & is_valid
    if not np.any(noise_mask):
        raise ValueError(
            "No valid immobility samples to estimate the noise threshold from "
            f"(speed < {speed_threshold} cm/s with finite LFP in every channel)."
        )

    noise_values = consensus[noise_mask]
    baseline = np.mean(noise_values)
    scale = np.std(noise_values, ddof=0)  # the ddof normalize_signal uses
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            "Immobility consensus has zero or undefined spread; cannot normalize."
        )
    normalized = normalize_signal(consensus, time=time, normalization_mask=noise_mask)
    if zscore_per_tetrode:
        # The original estimates on the median of per-tetrode z-scores, whose
        # units the histogram grid assumes; convert the result to the
        # immobility-normalized units the events are extracted in.
        threshold = estimate_noise_threshold(noise_values, percentile=percentile)
        threshold_zscore = (threshold - baseline) / scale
    else:
        # A raw median of envelopes is not in the grid's units, so follow the
        # paper's text instead: normalize to immobility, then estimate.
        threshold_zscore = estimate_noise_threshold(
            normalized[noise_mask], percentile=percentile
        )
    if not np.isfinite(threshold_zscore) or threshold_zscore <= 0:
        raise ValueError(
            f"Estimated threshold ({threshold_zscore:.4f} SD) does not lie above the "
            "immobility mean; the detection rule is undefined."
        )

    event_times = []
    is_clipped = []
    n_suprathreshold = []
    for start, stop in _contiguous_valid_blocks(is_valid, time, sampling_frequency):
        block_events, block_clipped, block_n = _extract_Yu_ripple_events(
            normalized[start:stop],
            time[start:stop],
            sampling_frequency,
            minimum_duration,
            threshold_zscore,
        )
        event_times.append(block_events)
        is_clipped.append(block_clipped)
        n_suprathreshold.append(block_n)
    event_times = np.concatenate(event_times) if event_times else np.empty((0, 2))
    is_clipped = np.concatenate(is_clipped) if is_clipped else np.empty((0, 2), dtype=bool)
    n_suprathreshold = (
        np.concatenate(n_suprathreshold) if n_suprathreshold else np.empty(0, dtype=int)
    )

    # exclude_movement's rule, kept here so the per-event flags stay aligned
    if len(event_times):
        speed_at_start = speed[np.searchsorted(time, event_times[:, 0])]
        speed_at_end = speed[np.searchsorted(time, event_times[:, 1])]
        keep = (speed_at_start <= speed_threshold) & (speed_at_end <= speed_threshold)
        event_times, is_clipped, n_suprathreshold = (
            event_times[keep],
            is_clipped[keep],
            n_suprathreshold[keep],
        )
    if len(event_times):
        event_times, kept = exclude_close_events(
            event_times,
            close_ripple_threshold,
            included_ripple_inds=np.arange(len(event_times)),
        )
        kept = np.asarray(kept, dtype=int)
        event_times = np.asarray(event_times).reshape(-1, 2)
        is_clipped, n_suprathreshold = is_clipped[kept], n_suprathreshold[kept]

    n_min = max(1, int(np.floor(minimum_duration * sampling_frequency + 0.5)))
    stats_minimum_duration = (n_min - 1) / sampling_frequency
    events = _get_event_stats(
        event_times, time, normalized, speed, minimum_duration=stats_minimum_duration
    )
    events["clipped_start"] = is_clipped[:, 0]
    events["clipped_end"] = is_clipped[:, 1]
    events["n_suprathreshold_samples"] = n_suprathreshold
    events["detection_threshold_zscore"] = threshold_zscore
    return events


def _unit_area_gaussian(sigma_samples: float, n_sd: float) -> NDArray:
    """Unit-area Gaussian kernel truncated at ``n_sd`` standard deviations
    (vandermeerlab ``gausskernel(R, S)`` with ``R = n_sd * S``)."""
    radius = int(np.ceil(n_sd * sigma_samples))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x**2) / (2.0 * sigma_samples**2))
    return kernel / kernel.sum()


def _state_intervals(
    is_in_state: NDArray, time: NDArray, merge_gap: float, minimum_length: float
) -> NDArray:
    """Contiguous runs of a state, merged across gaps shorter than ``merge_gap``
    and dropped when shorter than ``minimum_length`` (vandermeerlab ``TSDtoIV``).
    Returns ``[start_index, stop_index]`` rows, inclusive."""
    bounds = _boolean_runs(is_in_state)
    if len(bounds) == 0:
        return np.empty((0, 2), dtype=int)
    starts, stops = bounds[:, 0], bounds[:, 1] - 1
    if len(starts) > 1:
        gaps = time[starts[1:]] - time[stops[:-1]]
        merge = gaps < merge_gap
        keep_start = np.concatenate([[True], ~merge])
        keep_stop = np.concatenate([~merge, [True]])
        starts, stops = starts[keep_start], stops[keep_stop]
    long_enough = (time[stops] - time[starts]) > minimum_length  # TSDtoIV: strict
    return np.column_stack([starts[long_enough], stops[long_enough]])


def _contained_in_intervals(event_bounds: NDArray, intervals: NDArray) -> NDArray:
    """True for each ``[start, stop]`` event lying inside some interval (vandermeerlab ``restrict``)."""
    if len(event_bounds) == 0:
        return np.empty(0, dtype=bool)
    if len(intervals) == 0:
        return np.zeros(len(event_bounds), dtype=bool)
    inside = (event_bounds[:, [0]] >= intervals[:, 0]) & (
        event_bounds[:, [1]] <= intervals[:, 1]
    )
    return inside.any(axis=1)


def Carey_candidate_detector(
    time: ArrayLike,
    filtered_lfps: ArrayLike,
    multiunit: ArrayLike,
    speed: ArrayLike,
    sampling_frequency: float,
    speed_threshold: float = 4.0,
    edge_threshold: float = 1.0,
    peak_threshold: float = 3.0,
    minimum_duration: float = 0.020,
    minimum_active_units: int = 5,
    ripple_smoothing_sigma: float = 0.010,
    spike_kernel_sigma: float = 0.020,
    spike_cap: float = 2.0,
    baseline_sigma: float = 0.125,
    baseline_cap: float = 4.0,
    theta_lfp: ArrayLike | None = None,
    theta_band: tuple[float, float] = (6.0, 10.0),
    theta_threshold: float = 2.0,
    state_merge_gap: float = 0.050,
    state_minimum_length: float = 0.050,
) -> pd.DataFrame:
    """Detect candidate replay events from ripple power and multiunit activity jointly.

    The candidate-event detector of Carey, Tank & van der Meer 2019 [1]_
    (vandermeerlab ``GenCandidateEvents`` with its Hilbert ripple score
    ``OldWizard`` and multiunit score ``amMUA`` by Elyot Grant and A. Carey)
    [2]_. A ripple score and a multiunit score are combined as their
    **geometric mean**, so an event needs both a ripple and a population
    burst, then z-scored and segmented with two thresholds. Reimplemented
    from the code as read.

    - **Ripple score**: Hilbert envelope of the ripple-band signal, averaged
      across channels, smoothed with a Gaussian (10 ms SD, +/-3 SD), rescaled
      to mean 1.
    - **Multiunit score**: each unit's spike train smoothed with a unit-area
      Gaussian (20 ms SD, +/-5 SD) and capped at the peak that ``spike_cap``
      coincident spikes would give, so no single unit dominates; summed over
      units; a slow baseline (the sum capped at ``baseline_cap`` units'
      worth, smoothed with a 125 ms SD Gaussian) and one unit's cap are
      subtracted; divided by the mean and floored at zero.
    - **Joint score**: ``sqrt(ripple * multiunit)``, rescaled to mean 0.5 and
      z-scored. Note the asymmetry this creates: the multiunit score is
      floored at zero, so a ripple without a population burst cannot be a
      candidate, but the ripple score is an envelope rescaled to mean 1 and
      never zero, so a burst without a ripple can be. The joint score is
      therefore closer to "burst, weighted by ripple power" than to a
      symmetric conjunction. Candidates are runs strictly above ``edge_threshold`` whose
      maximum is strictly above ``peak_threshold``, at least
      ``minimum_duration`` long.
    - **State**: a candidate is kept only if it lies entirely inside a
      low-speed interval (speed below ``speed_threshold``, runs merged across
      gaps under ``state_merge_gap`` and dropped under
      ``state_minimum_length``) and, when ``theta_lfp`` is given, inside a
      low-theta interval (z-scored theta-band envelope below
      ``theta_threshold``, same interval rules), and has at least
      ``minimum_active_units`` units with a spike inside it.

    The original works in samples at 2 kHz (kernel SDs of 40 and 250 samples,
    +/-60-sample ripple smoothing); the defaults here are those values in
    seconds. Its speed limit is 10 pixels/s in tracking units; the default
    here is the package's 4 cm/s. NaN input raises: the scores need
    contiguous data.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Time values for each sample in seconds.
    filtered_lfps : array_like, shape (n_time, n_channels)
        Ripple-band-filtered LFP; the original uses 140-250 Hz on one channel.
    multiunit : array_like, shape (n_time, n_units)
        Spike counts (or indicators) per sample per unit; clusterless marks
        per tetrode work, with the per-unit cap then applying per tetrode.
    speed : array_like, shape (n_time,)
        Animal's running speed in cm/s.
    sampling_frequency : float
        Sampling rate in Hz.
    speed_threshold : float, optional
        Speed below which the animal is considered stopped. Default is 4.0.
    edge_threshold, peak_threshold : float, optional
        Boundary and peak thresholds on the z-scored joint score. Defaults 1
        and 3 (the original's ``DetectorThreshold`` and ``DetectorThreshold2``).
    minimum_duration : float, optional
        Minimum candidate duration in seconds. Default 0.020.
    minimum_active_units : int, optional
        Minimum number of units with a spike inside the candidate. Default 5.
    ripple_smoothing_sigma, spike_kernel_sigma, baseline_sigma : float, optional
        Gaussian standard deviations in seconds. Defaults 0.010, 0.020, 0.125.
    spike_cap, baseline_cap : float, optional
        Per-unit cap in coincident spikes (default 2) and the baseline cap in
        units' worth (default 4).
    theta_lfp : array_like, shape (n_time,), optional
        Raw LFP of a theta channel; when given, candidates during elevated
        theta are excluded. Default None (no theta exclusion).
    theta_band : tuple of (float, float), optional
        Theta pass-band in Hz, Butterworth order 4. Default (6, 10).
    theta_threshold : float, optional
        Theta-envelope z-score at or above which a period is excluded. Default 2.
    state_merge_gap, state_minimum_length : float, optional
        Interval rules for the low-speed and low-theta periods, in seconds.
        Defaults 0.050 and 0.050 (vandermeerlab ``TSDtoIV`` defaults).

    Returns
    -------
    candidate_times : pd.DataFrame
        One row per candidate, indexed by ``event_number``, with the package's
        statistics computed on the z-scored joint score, the speed statistics,
        and ``n_active_units``.

    References
    ----------
    .. [1] Carey, A. A., Tank, D. W., & van der Meer, M. A. A. (2019). Reward
       revaluation biases hippocampal replay content away from the preferred
       outcome. Nature Neuroscience, 22, 1450-1459.
    .. [2] van der Meer lab, ``code-matlab/tasks/Alyssa_Tmaze/GenCandidateEvents.m``,
       ``beta/OldWizard.m``, ``beta/amMUA.m``, ``beta/TSDtoIV2.m``,
       https://github.com/vandermeerlab/vandermeerlab

    """
    filtered_lfps = np.asarray(filtered_lfps, dtype=float)
    multiunit = np.asarray(multiunit, dtype=float)
    speed = np.asarray(speed, dtype=float)
    time = np.asarray(time, dtype=float)
    _validate_lfp_dimensions(filtered_lfps)
    if multiunit.ndim != 2:
        raise ValueError(
            f"multiunit must be a 2D array of shape (n_time, n_units), got shape {multiunit.shape}."
        )
    _validate_array_lengths(time, filtered_lfps, speed)
    if multiunit.shape[0] != len(time):
        raise ValueError(
            f"Array length mismatch: multiunit has {multiunit.shape[0]} samples but time has {len(time)}."
        )
    _validate_time_units(time, sampling_frequency, len(time))
    _validate_speed_units(speed, speed_threshold)
    if (
        np.any(np.isnan(filtered_lfps))
        or np.any(np.isnan(multiunit))
        or np.any(np.isnan(speed))
    ):
        raise ValueError("filtered_lfps, multiunit, and speed must not contain NaN.")
    n_time = len(time)

    # ripple score (OldWizard, 'amplitude', 'wizard' kernel), rescaled to mean 1
    envelope = get_envelope(filtered_lfps).mean(axis=1)
    ripple_score = gaussian_filter1d(
        envelope, ripple_smoothing_sigma * sampling_frequency, truncate=3.0, mode="constant"
    )
    ripple_score = ripple_score / ripple_score.mean()

    # multiunit score (amMUA)
    sigma_samples = spike_kernel_sigma * sampling_frequency
    spike_kernel = _unit_area_gaussian(sigma_samples, 5.0)
    cap = spike_cap / (sigma_samples * np.sqrt(2.0 * np.pi))
    summed = np.zeros(n_time)
    for unit in multiunit.T:
        summed += np.minimum(np.convolve(unit, spike_kernel, mode="same"), cap)
    baseline_kernel = _unit_area_gaussian(baseline_sigma * sampling_frequency, 12.0)
    baseline = np.convolve(
        np.minimum(baseline_cap * cap, summed), baseline_kernel, mode="same"
    )
    mean_summed = summed.mean()
    if mean_summed <= 0:
        raise ValueError("multiunit contains no spikes; cannot form a multiunit score.")
    multiunit_score = np.maximum(0.0, (summed - baseline - cap) / mean_summed)

    joint = np.sqrt(ripple_score * multiunit_score)
    joint = joint * (0.5 / joint.mean()) if joint.mean() > 0 else joint
    zscored = normalize_signal(joint)

    # two-threshold segmentation (TSDtoIV2): runs above the edge, kept if peak above
    bounds = _boolean_runs(zscored > edge_threshold)
    candidates = []
    for start, stop in bounds:
        if zscored[start:stop].max() > peak_threshold:
            candidates.append((start, stop - 1))
    candidates = np.asarray(candidates, dtype=int).reshape(-1, 2)
    if len(candidates):
        duration = time[candidates[:, 1]] - time[candidates[:, 0]]
        candidates = candidates[duration > minimum_duration]  # RemoveIV: strict

    # state restriction: contained in a low-speed (and low-theta) interval
    if len(candidates):
        low_speed = _state_intervals(
            speed < speed_threshold, time, state_merge_gap, state_minimum_length
        )
        candidates = candidates[_contained_in_intervals(candidates, low_speed)]
    if len(candidates) and theta_lfp is not None:
        theta_lfp = np.asarray(theta_lfp, dtype=float)
        if theta_lfp.shape != (n_time,):
            raise ValueError(f"theta_lfp must have shape ({n_time},), got {theta_lfp.shape}.")
        b, a = butter(4, np.asarray(theta_band) / (0.5 * sampling_frequency), btype="bandpass")
        theta_envelope = get_envelope(filtfilt(b, a, theta_lfp))
        low_theta = _state_intervals(
            normalize_signal(theta_envelope) < theta_threshold,
            time,
            state_merge_gap,
            state_minimum_length,
        )
        candidates = candidates[_contained_in_intervals(candidates, low_theta)]

    # minimum number of active units
    n_active = np.array(
        [
            int(np.sum(multiunit[start : stop + 1].sum(axis=0) > 0))
            for start, stop in candidates
        ],
        dtype=int,
    )
    if len(candidates):
        keep = n_active >= minimum_active_units
        candidates, n_active = candidates[keep], n_active[keep]

    event_times = (
        np.column_stack([time[candidates[:, 0]], time[candidates[:, 1]]])
        if len(candidates)
        else np.empty((0, 2))
    )
    events = _get_event_stats(
        event_times, time, zscored, speed, minimum_duration=minimum_duration
    )
    events["n_active_units"] = n_active
    return events


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
        This is the minimum time the signal must stay *above* ``zscore_threshold``
        (per Karlsson et al. 2009); the event is then extended to the surrounding
        mean-crossings, so the reported ``duration`` is typically longer.
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
        are excluded -- the later event is dropped, not merged. Default is 0.0
        (no exclusion). Set to 0.05-0.1 s to drop closely-spaced events.
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
    time, filtered_lfps, speed, normalization_mask = _preprocess_detector_inputs(
        time,
        filtered_lfps,
        speed,
        sampling_frequency,
        speed_threshold,
        normalization_mask=normalization_mask,
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

    return _get_event_stats(
        ripple_times, time, filtered_lfps.mean(axis=1), speed, minimum_duration
    )


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
        This is the minimum time the signal must stay *above* ``zscore_threshold``
        (per Karlsson et al. 2009); the event is then extended to the surrounding
        mean-crossings, so the reported ``duration`` is typically longer.
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
        are excluded -- the later event is dropped, not merged. Default is 0.0
        (no exclusion). Set to 0.05-0.1 s to drop closely-spaced events.
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
    time, filtered_lfps, speed, normalization_mask = _preprocess_detector_inputs(
        time,
        filtered_lfps,
        speed,
        sampling_frequency,
        speed_threshold,
        normalization_mask=normalization_mask,
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
        This is the minimum time the firing rate must stay *above* ``zscore_threshold``;
        the event is then extended to the surrounding mean-crossings, so the reported
        ``duration`` is typically longer.
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
        are excluded -- the later event is dropped, not merged. Default is 0.0
        (no exclusion). Set to 0.05-0.1 s to drop closely-spaced events.
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

    return _get_event_stats(high_synchrony_events, time, firing_rate, speed, minimum_duration)


def _find_max_thresh(
    time: np.ndarray, data: np.ndarray, minimum_duration: float = 0.015
) -> float:
    """Find the largest value sustained around the peak for a minimum duration.

    Starting at the peak, expand a window (toward the higher neighbouring sample)
    until it spans ``minimum_duration``, then return the smaller of the two window
    edges -- the largest value held across the whole window.

    Parameters
    ----------
    time : np.ndarray, shape (n_time,)
    data : np.ndarray, shape (n_time,)
    minimum_duration : float, optional

    Returns
    -------
    max_thresh : float
        The largest value sustained for ``minimum_duration`` around the peak.
        ``nan`` if the event is shorter than ``minimum_duration`` (the sustained
        value is then undefined). Public detectors never produce such events --
        their segments are ``>= minimum_duration`` by construction -- so this
        only affects direct/edge callers.
    """
    # Find the peak of the data points
    peak_ind = np.argmax(data)

    # Initialize the search window
    peak_left_ind = peak_ind
    peak_right_ind = peak_ind

    # Match segment_boolean_series's inclusive duration comparison. Subtracting
    # timestamps can round an accepted boundary below minimum_duration.
    while time[peak_right_ind] < time[peak_left_ind] + minimum_duration:
        can_expand_right = peak_right_ind < len(time) - 1
        can_expand_left = peak_left_ind > 0
        # The window already spans the whole event yet is still shorter than
        # minimum_duration, so a value "sustained for minimum_duration" is
        # undefined. Return nan rather than a misleading endpoint value (and
        # rather than running an index out of bounds).
        if not (can_expand_right or can_expand_left):
            return float("nan")
        # Determine the direction to expand
        if can_expand_right and (
            not can_expand_left or data[peak_right_ind + 1] > data[peak_left_ind - 1]
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
    participants: ArrayLike | None = None,
    n_participants: ArrayLike | None = None,
    frac_participants: ArrayLike | None = None,
) -> pd.DataFrame:
    """Compute comprehensive statistics for detected events.

    Calculates temporal, z-score, signal, and speed metrics for each event.

    Parameters
    ----------
    event_times : array_like, shape (n_events, 2)
        Array of [start_time, end_time] for each event.
    time : array_like, shape (n_time,)
        Time values for each sample.
    zscore_metric : array_like, if participants is None: shape (n_time,); else shape (n_time, n_channels)
        Signal the per-event statistics (mean/median/max/min z-score, area,
        total_energy, max_thresh) are computed from. Its exact meaning depends on
        the caller -- e.g. the consensus trace for Kay, the per-channel mean for
        Karlsson, or the multiunit firing rate for multiunit_HSE. When participants
        is None, pass a single 1-D trace of shape (n_time,). When participants is
        provided, pass the per-channel signal of shape (n_time, n_channels) so that
        each event's metrics are computed from its participating channels only.
    speed : array_like, shape (n_time,)
        Animal's speed at each time point.
    minimum_duration : float, optional
        Minimum duration for max_thresh calculation. Default is 0.015 (15 ms).
    participants: array_like of set, shape (n_events,)
        Set of channels that participate in each event; z-score metrics are
        averaged over these channels. Used by Shvartsman_ripple_detector.
        Optional, default is None.
    n_participants: array_like, shape (n_events,)
        Number of distinct participating channels per event. For
        Shvartsman_ripple_detector this equals ``len(participants[i])``.
        Optional, default is None.
    frac_participants: array_like, shape (n_events,)
        ``n_participants`` divided by the total channel count, per event, as
        supplied by the caller. Used by Shvartsman_ripple_detector. Optional,
        default is None.

    Returns
    -------
    event_stats : pd.DataFrame
        DataFrame with one row per event and columns:
        - start_time, end_time: Event boundaries
        - duration: Event duration (end - start)
        - max_thresh: Maximum z-score sustained for minimum_duration (nan for an
            event shorter than minimum_duration; not produced by the detectors)
        - mean_zscore, median_zscore, max_zscore, min_zscore: Z-score statistics
        - area: Integral of z-score over event duration
        - total_energy: Integral of squared z-score
        - speed_at_start, speed_at_end: Speed at event boundaries
        - max_speed, min_speed, median_speed, mean_speed: Speed statistics
        - participants, n_participants, frac_participants: Information on
            which channels exhibited a ripple during the detected event
            (returned if 'participants' input is not None)

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

    for r, (start_time, end_time) in enumerate(event_times_arr):
        time_mask = np.logical_and(time_arr >= start_time, time_arr <= end_time)

        if participants is None:
            if len(zscore_metric.shape) != 1:
                raise ValueError(
                    "If no participants are listed, the shape of zscore_metric should be (n_time,). "
                    f"Current shape of zscore_metric is {zscore_metric.shape}."
                )

            event_zscore = zscore_metric_arr[time_mask]

        else:
            time_ind = np.where(time_mask)[0]
            elec_ind = np.asarray(list(participants[r]))

            # check that zscore_metric is 2-D
            if len(zscore_metric.shape) != 2:
                raise ValueError(
                    "If participants are listed, the shape of zscore_metric should be (n_time, n_channels) "
                    f"so that relevant metrics can be properly calculated. Current shape of zscore_metric is {zscore_metric.shape}."
                )

            event_zscore = zscore_metric[np.ix_(time_ind, elec_ind)].mean(
                axis=1
            )  # only include the participating electrodes for all of these metrics

        max_thresh.append(
            _find_max_thresh(time_arr[time_mask], event_zscore, minimum_duration)
        )
        mean_zscore.append(np.mean(event_zscore))
        median_zscore.append(np.median(event_zscore))
        max_zscore.append(np.max(event_zscore))
        min_zscore.append(np.min(event_zscore))
        area.append(trapezoid(event_zscore, time_arr[time_mask]))
        total_energy.append(trapezoid(event_zscore**2, time_arr[time_mask]))
        duration.append(end_time - start_time)
        max_speed.append(np.max(speed_arr[time_mask]))
        min_speed.append(np.min(speed_arr[time_mask]))
        median_speed.append(np.median(speed_arr[time_mask]))
        mean_speed.append(np.mean(speed_arr[time_mask]))

    event_start_times: NDArray | list
    event_end_times: NDArray | list
    try:
        event_start_times = event_times_arr[:, 0]
        event_end_times = event_times_arr[:, 1]
    except (IndexError, TypeError):
        event_start_times = []
        event_end_times = []

    event_stats = {
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
    }
    # Shvartsman_ripple_detector passes participation info; the other detectors do not.
    if participants is not None:
        event_stats["participants"] = participants
        event_stats["n_participants"] = n_participants
        event_stats["frac_participants"] = frac_participants

    return pd.DataFrame(event_stats, index=index)

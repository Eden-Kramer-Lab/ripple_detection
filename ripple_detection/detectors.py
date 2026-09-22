"""High-level detectors for sharp-wave ripple events and multiunit synchrony events."""

import warnings
from itertools import chain, pairwise

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.cluster.vq import kmeans2
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.signal import butter, filtfilt

from ripple_detection.core import (
    _boolean_run_bounds,
    _get_normalization_mask,
    _is_immobile_at_endpoints,
    estimate_noise_threshold,
    exclude_close_events,
    exclude_movement,
    exclude_movement_by_majority,
    gaussian_smooth,
    get_envelope,
    get_multiunit_population_firing_rate,
    merge_overlapping_ranges,
    merge_overlapping_ranges_track_participation,
    minimum_sample_count,
    nearest_sample_index,
    normalize_signal,
    normalize_signal_manually,
    sample_count_within,
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


def _validate_time_units(
    time: NDArray, sampling_frequency: float, stacklevel: int = 4
) -> None:
    """Validate that time array is in seconds (not samples).

    Parameters
    ----------
    time : ndarray
        Time array to validate.
    sampling_frequency : float
        Expected sampling frequency in Hz.
    stacklevel : int, optional
        Frames between this function and the caller's line, for the warning.

    Raises
    ------
    ValueError
        If time is not increasing, if its median step is not positive (most
        timestamps repeat), or if it appears to be in samples instead of
        seconds.

    Warnings
    --------
    UserWarning
        If time step differs significantly from expected.

    """
    if len(time) > 1:
        steps = np.diff(time)
        if np.any(steps < 0):
            raise ValueError(
                "time must be increasing. Sort time, and the signals with it, before "
                "detecting: the event and speed lookups assume time order."
            )
        median_dt = np.median(steps)
        expected_dt = 1.0 / sampling_frequency
        if not median_dt > 0:
            raise ValueError(
                f"The median time step is {median_dt}: most timestamps repeat, so no "
                "duration can be measured in samples. Check the time array."
            )

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
                stacklevel=stacklevel,
            )


def _validate_speed_units(speed: NDArray, speed_threshold: float, stacklevel: int = 4) -> None:
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
                    "  speed_cms = speed_ms * 100",
                    UserWarning,
                    stacklevel=stacklevel,
                )


def _validate_detector_inputs(
    time: ArrayLike,
    signal: ArrayLike,
    speed: ArrayLike,
    sampling_frequency: float,
    speed_threshold: float,
    stacklevel: int = 4,
) -> tuple[NDArray, NDArray, NDArray]:
    """Cast the inputs to float arrays and check shapes, lengths and units.

    Parameters
    ----------
    time : array_like, shape (n_time,)
    signal : array_like, shape (n_time, n_channels)
        Ripple-band LFP, raw LFP or spike counts, checked to be 2-D.
    speed : array_like, shape (n_time,)
    sampling_frequency : float
    speed_threshold : float
        Used only to judge whether speed is in cm/s.
    stacklevel : int, optional
        Frames between this function and the caller's line, for warnings.

    Returns
    -------
    time, signal, speed : ndarray
        The inputs as float arrays.

    Raises
    ------
    ValueError
        If the signal is not 2-D, the lengths differ, time is not increasing
        or appears to be in samples.

    """
    signal = np.asarray(signal, dtype=float)
    speed = np.asarray(speed, dtype=float)
    time = np.asarray(time, dtype=float)
    _validate_lfp_dimensions(signal)
    _validate_array_lengths(time, signal, speed)
    _validate_time_units(time, sampling_frequency, stacklevel=stacklevel)
    _validate_speed_units(speed, speed_threshold, stacklevel=stacklevel)
    return time, signal, speed


def _valid_blocks(
    time: NDArray, sampling_frequency: float, *signals: NDArray
) -> tuple[NDArray, list[tuple[int, int]]]:
    """The samples every detector may use, and their contiguous blocks.

    A sample is valid when every channel of every signal is finite. Valid
    samples are split into blocks at every invalid sample and wherever the
    timestamp step exceeds 1.5 times the median step. Every step of every
    detector runs within a block, so nothing is computed across a gap and
    no event spans one.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    sampling_frequency : float
    *signals : ndarray, shape (n_time,) or (n_time, n_channels)
        The LFP, spikes and speed the detector reads.

    Returns
    -------
    is_valid : ndarray of bool, shape (n_time,)
    blocks : list of (start, stop)
        Half-open index ranges of the valid blocks, in order.

    Raises
    ------
    ValueError
        If no sample is valid.

    """
    is_valid = np.ones(len(time), dtype=bool)
    for signal in signals:
        finite = np.isfinite(signal)
        is_valid &= finite.all(axis=1) if finite.ndim == 2 else finite
    if not np.any(is_valid):
        raise ValueError(
            "Every sample has a NaN in a signal or in speed, so there is nothing to "
            "detect on. Check the alignment of the inputs."
        )
    return is_valid, _contiguous_valid_blocks(is_valid, time)


def _drop_short_blocks(
    blocks: list[tuple[int, int]],
    is_valid: NDArray,
    minimum_length: int,
    reason: str,
    stacklevel: int = 3,
) -> list[tuple[int, int]]:
    """Treat blocks shorter than a detector's transform needs as missing.

    Marks their samples invalid in place, warns with their sample ranges, and
    raises if no block remains.
    """
    short = [(start, stop) for start, stop in blocks if stop - start < minimum_length]
    if not short:
        return blocks
    for start, stop in short:
        is_valid[start:stop] = False
    kept = [(start, stop) for start, stop in blocks if stop - start >= minimum_length]
    if not kept:
        raise ValueError(
            f"No block of finite samples is as long as the {minimum_length} samples that "
            f"{reason} needs."
        )
    warnings.warn(
        f"{len(short)} block(s) of finite samples shorter than the {minimum_length} samples "
        f"that {reason} needs are treated as missing (sample ranges "
        f"{short[:5]}{', ...' if len(short) > 5 else ''}).",
        UserWarning,
        stacklevel=stacklevel,
    )
    return kept


def _mask_invalid(signal: NDArray, is_valid: NDArray) -> NDArray:
    """A copy of ``signal`` with NaN at every invalid sample, so a helper that
    splits blocks on its own splits them where the detector does."""
    masked = signal.copy()
    masked[~is_valid] = np.nan
    return masked


def _normalization_mask_over_valid(
    n_time: int, is_valid: NDArray, normalization_mask: ArrayLike | None
) -> NDArray:
    """The samples the normalization statistics come from: the caller's mask,
    if any, restricted to valid samples."""
    mask = _get_normalization_mask((n_time,), normalization_mask)
    mask = is_valid if mask is None else mask & is_valid
    if not np.any(mask):
        raise ValueError(
            "The normalization mask selects no sample that is finite in every signal; "
            "cannot compute normalization statistics."
        )
    return mask


def _threshold_blocks(
    normalized: NDArray,
    time: NDArray,
    blocks: list[tuple[int, int]],
    minimum_duration: float,
    zscore_threshold: float,
) -> list[tuple[float, float]]:
    """``threshold_by_zscore`` within each block; events never span a gap.

    A block with fewer samples than the minimum duration cannot hold an
    event and is skipped, which also keeps the sample count from being
    measured on a block too short to have a median step.
    """
    n_min = minimum_sample_count(time, minimum_duration)
    events: list[tuple[float, float]] = []
    for start, stop in blocks:
        if stop - start >= n_min:
            events.extend(
                threshold_by_zscore(
                    normalized[start:stop],
                    time[start:stop],
                    minimum_duration,
                    zscore_threshold,
                )
            )
    return events


def get_Kay_ripple_consensus_trace(
    ripple_filtered_lfps: ArrayLike,
    sampling_frequency: float,
    smoothing_sigma: float = 0.004,
    *,
    time: ArrayLike | None = None,
) -> NDArray:
    """Compute Kay consensus trace from multi-channel ripple-filtered LFPs.

    Combines multiple LFP channels into a single consensus trace, following
    Kay et al. 2016: ``sqrt(gaussian_smooth(sum(envelope ** 2)))``. The
    smoothing sits between the sum and the square root. The paper's text
    squares the filtered signal itself, not its Hilbert envelope; the two
    traces differ by a constant factor of sqrt(2) once the 4 ms smoothing has
    removed the doubled-frequency term, so the events are the same after the
    z-score. The envelope is what the Frank lab code uses.

    Rows holding a missing value in any channel are excluded, and each
    contiguous run of valid rows is processed on its own, so no envelope or
    smoothing window spans a gap.

    Parameters
    ----------
    ripple_filtered_lfps : array_like, shape (n_time, n_channels)
        Bandpass filtered LFP signals in the ripple band (150-250 Hz).
    sampling_frequency : float
        Sampling rate in Hz.
    smoothing_sigma : float, optional
        Standard deviation of Gaussian smoothing kernel in seconds.
        Default is 0.004 (4 ms).
    time : array_like, shape (n_time,), optional
        Sample timestamps in seconds, used to split at gaps in the timestamps
        as well as at missing samples. Keyword only. Default is None, which
        splits at missing samples only.

    Returns
    -------
    consensus_trace : ndarray, shape (n_time,)
        ``sqrt(gaussian_smooth(sum(envelope ** 2)))`` per sample.

    References
    ----------
    .. [1] Kay, K., Sosa, M., Chung, J. E., Karlsson, M. P., Larkin, M. C., &
       Frank, L. M. (2016). A hippocampal network for spatial coding during
       immobility and sleep. Nature, 531(7593), 185-190.
       doi:10.1038/nature17144

    """
    # Cast to float so integer input is not truncated and the squared envelope
    # cannot overflow before the square root.
    ripple_filtered_lfps = np.asarray(ripple_filtered_lfps, dtype=float)
    ripple_consensus_trace = np.full_like(ripple_filtered_lfps, np.nan)
    not_null = np.all(pd.notna(ripple_filtered_lfps), axis=1)

    time_array = None if time is None else np.asarray(time, dtype=float)
    for start, stop in _contiguous_valid_blocks(not_null, time_array):
        block = ripple_filtered_lfps[start:stop]
        ripple_consensus_trace[start:stop] = get_envelope(block)

    summed_power = np.sum(ripple_consensus_trace**2, axis=1)
    smoothed = np.full(len(summed_power), np.nan)
    for start, stop in _contiguous_valid_blocks(not_null, time_array):
        smoothed[start:stop] = gaussian_smooth(
            summed_power[start:stop], smoothing_sigma, sampling_frequency
        )
    return np.sqrt(smoothed)


def _contiguous_valid_blocks(is_valid: NDArray, time: NDArray | None) -> list[tuple[int, int]]:
    """Split rows into maximal contiguous valid blocks.

    A block ends at an invalid row or, when ``time`` is given, wherever the
    timestamp step exceeds 1.5 times the median step (a recording gap or the
    join between disjoint intervals). The median step is measured from
    ``time`` rather than taken from the nominal sampling rate, so an
    overstated rate cannot turn every sample into its own block.

    Parameters
    ----------
    is_valid : ndarray of bool, shape (n_time,)
        True for rows with finite data in every channel.
    time : ndarray, shape (n_time,), optional
        Sample timestamps in seconds. None declares a regular sample grid.

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
    if time is not None and n_time > 1:
        steps = np.diff(time)
        boundary[1:-1] |= steps > 1.5 * np.median(steps)
    edges = np.flatnonzero(boundary)
    return [(int(start), int(stop)) for start, stop in pairwise(edges) if is_valid[start]]


def _smoothed_envelope(
    filtered_lfps: NDArray,
    blocks: list[tuple[int, int]],
    sampling_frequency: float,
    smoothing_sigma: float,
    square: bool = False,
) -> NDArray:
    """Per-channel envelope, squared if asked, smoothed within each block.

    Neither the Hilbert transform nor the Gaussian kernel spans a gap.
    Samples outside every block are NaN.

    Parameters
    ----------
    filtered_lfps : ndarray, shape (n_time, n_channels)
        Ripple-band LFP.
    blocks : list of (start, stop)
        Half-open index ranges of the valid blocks (``_valid_blocks``).
    sampling_frequency : float
    smoothing_sigma : float
        Gaussian standard deviation in seconds.
    square : bool, optional
        Square the envelope before smoothing. Default is False.

    Returns
    -------
    smoothed : ndarray, shape (n_time, n_channels)

    """
    smoothed = np.full_like(filtered_lfps, np.nan)
    for start, stop in blocks:
        envelope = get_envelope(filtered_lfps[start:stop])
        if square:
            envelope = envelope**2
        smoothed[start:stop] = gaussian_smooth(envelope, smoothing_sigma, sampling_frequency)
    return smoothed


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
    .. [1] Yu, J. Y., Kay, K., Liu, D. F., Grossrubatscher, I., Loback, A.,
       Sosa, M., Chung, J. E., Karlsson, M. P., Larkin, M. C., & Frank, L. M.
       (2017). Distinct hippocampal-cortical memory representations for
       experiences associated with movement versus immobility. eLife, 6,
       e27621. doi:10.7554/eLife.27621

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
    for start, stop in _contiguous_valid_blocks(is_valid, time):
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


def _extract_Yu_ripple_events(
    trace: NDArray,
    time: NDArray,
    minimum_duration: float,
    threshold: float,
) -> tuple[NDArray, NDArray]:
    """Extract events from one contiguous block of a mean-zero consensus trace.

    A run of consecutive samples at or above ``threshold`` qualifies when it
    holds at least ``minimum_sample_count(time, minimum_duration)`` samples.
    Each qualifying run is extended to the run of samples at or above zero,
    the immobility mean, that contains it, as ``threshold_by_zscore`` does. One event is emitted per containing
    run. This is the sample-count convention of the Frank lab
    ``extractevents`` routine, which the Yu et al. 2017 detector used
    (``DFFunctions/extractevents.cpp`` in
    https://github.com/droumis/FFPhy/tree/fce2048/DFFunctions).

    Parameters
    ----------
    trace : ndarray, shape (n_time,)
        Consensus trace normalized so the immobility mean is zero, for one
        contiguous block with no missing samples.
    time : ndarray, shape (n_time,)
        Native timestamps of the block's samples, in seconds.
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
        first and last samples of the containing run at or above zero.
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
    n_min = minimum_sample_count(time, minimum_duration)

    supra_runs = _boolean_run_bounds(trace >= threshold)
    supra_runs = supra_runs[(supra_runs[:, 1] - supra_runs[:, 0]) >= n_min]
    if len(supra_runs) == 0:
        return np.empty((0, 2)), np.empty(0, dtype=int)

    above_zero_runs = _boolean_run_bounds(trace >= 0)
    # the above-zero run containing each qualifying run's first sample
    containing = np.searchsorted(above_zero_runs[:, 0], supra_runs[:, 0], side="right") - 1
    run_lengths = supra_runs[:, 1] - supra_runs[:, 0]

    event_times = []
    n_suprathreshold = []
    for run_index in np.unique(containing):
        start, stop = above_zero_runs[run_index]
        event_times.append((time[start], time[stop - 1]))
        n_suprathreshold.append(int(run_lengths[containing == run_index].max()))
    return np.asarray(event_times, dtype=float), np.asarray(n_suprathreshold, dtype=int)


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
    manual_normalization: bool = False,
    elec_baselines: ArrayLike | None = None,
    elec_deviations: ArrayLike | None = None,
    minimum_participating_channels: int | None = None,
    minimum_participating_fraction: float | None = None,
    maximum_duration: float | None = None,
) -> pd.DataFrame:
    """Detect sharp-wave ripples on each channel, keeping events that enough
    channels share.

    An event is kept when at least ``minimum_participating_channels`` channels
    (default 2), or ``minimum_participating_fraction`` of the channels, detect
    it. This sits between the Kay detector, which builds one consensus trace,
    and the Karlsson detector, which keeps a ripple from any single channel.
    Requiring several channels makes it less sensitive than Karlsson to noise
    on one channel.

    It also accepts a baseline and a deviation per electrode through
    ``manual_normalization``, in place of statistics computed from the data.
    Statistics from a whole recording day rather than one epoch matter for
    sleep sessions; see ``normalize_signal_manually``.

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
        Maximum speed (in cm/s) for ripple detection. An event is kept only if
        at least half of its samples have speed at or below this value
        (``exclude_movement_by_majority``); movement over up to half of an
        event does not exclude it. Default is 4.0 cm/s, which corresponds
        to immobility/slow movement in rodents.

        **Important**: Ensure your speed data is in cm/s. If using m/s, multiply
        by 100. To disable movement exclusion, set to a very large value (e.g., 1e6).
    minimum_duration : float, optional
        Minimum ripple duration in **seconds**. Default is 0.015 (15 milliseconds).
        The signal must stay at or above ``zscore_threshold`` for at least
        ``round(minimum_duration * sampling_frequency)`` consecutive samples
        (per Karlsson & Frank 2009); the event is then extended to the surrounding
        mean-crossings, so the reported ``duration`` is typically longer.
        Typical range: 0.015 - 0.100 s (15-100 ms). Lower values detect shorter
        events but may increase false positives.
    maximum_duration : float, optional
        Longest allowed event duration in **seconds**, applied to the event as
        it is reported rather than to the run above threshold, because that is
        what a published maximum describes. Default is None (no upper limit).
        Published ceilings run from a few hundred milliseconds to a couple of
        seconds.
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
        Only used when ``manual_normalization=False``; supplying it with
        ``manual_normalization=True`` raises ValueError.
    normalization_mask : array_like, shape (n_time,), optional
        Boolean mask selecting samples used to compute normalization statistics.
        For example, use `speed <= speed_threshold` to compute statistics only
        during immobility. Only
        used when ``manual_normalization=False``. Default is None (use all data).
    manual_normalization : bool, optional
        If True, normalize each channel with the supplied `elec_baselines` and
        `elec_deviations` instead of computing statistics from the data. The
        `normalization_*` parameters above must then be left at their defaults;
        supplying one raises ValueError rather than being ignored. Requires both
        `elec_baselines` and `elec_deviations`. Default is False.
    elec_baselines : array_like, shape (n_channels,), optional
        Baseline (center) value per channel. Required when
        ``manual_normalization=True``.
    elec_deviations : array_like, shape (n_channels,), optional
        Deviation (scale) value per channel, on the scale of a standard
        deviation (multiply a MAD by 1.4826 first). Required when
        ``manual_normalization=True``. A zero or NaN entry raises; drop that
        channel before detecting.
    minimum_participating_channels : int, optional
        Number of channels that must detect a ripple in the merged event for
        it to be kept. Default is 2 when neither participation argument is
        given; 0 imposes no criterion. Each distinct channel with a detected
        ripple anywhere in the merged event counts once, including channels
        connected through a chain of overlapping ripples.
    minimum_participating_fraction : float, optional
        The same criterion as a fraction of the channels in `filtered_lfps`,
        in [0, 1]; 1.0 requires every channel. Give one of the two arguments,
        not both. Default is None.

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

    Notes
    -----
    Missing samples: a NaN in any channel of ``filtered_lfps`` or in ``speed``
    marks that sample missing, as does a step in ``time`` larger than 1.5
    sample intervals. The valid samples form contiguous blocks, and every
    step runs within a block, so nothing is computed across a gap and no
    event spans one. An event cut off by a gap or by the recording edge is
    kept and flagged in ``clipped_start`` and ``clipped_end``. Every detector
    in the package follows this rule.
    See the README's "Choosing a detector" table for how the detectors'
    conventions differ.

    References
    ----------
    Unpublished variant contributed by Gabrielle Shvartsman (2026, pull
    request #11); it has no paper of its own. The participation rule requires
    ``minimum_participating_channels`` channels (default 2) to detect the
    ripple, so at the default a single-channel input never produces an event.

    """
    if manual_normalization and (
        normalization_mask is not None or normalization_method != "zscore"
    ):
        raise ValueError(
            "manual_normalization=True uses elec_baselines and elec_deviations, so "
            "normalization_method and normalization_mask must be left at their "
            "defaults. Drop them, or set manual_normalization=False."
        )
    if (
        minimum_participating_channels is not None
        and minimum_participating_fraction is not None
    ):
        raise ValueError(
            "Give minimum_participating_channels or minimum_participating_fraction, not both."
        )
    if minimum_participating_channels is None and minimum_participating_fraction is None:
        minimum_participating_channels = 2
    if minimum_participating_channels is not None and minimum_participating_channels < 0:
        raise ValueError("minimum_participating_channels must be non-negative.")
    if minimum_participating_fraction is not None and not (
        0.0 <= minimum_participating_fraction <= 1.0
    ):
        raise ValueError(
            f"minimum_participating_fraction must lie in [0, 1], got "
            f"{minimum_participating_fraction}."
        )
    _validate_duration_limits(minimum_duration, maximum_duration)
    time, filtered_lfps, speed = _validate_detector_inputs(
        time, filtered_lfps, speed, sampling_frequency, speed_threshold
    )
    is_valid, blocks = _valid_blocks(time, sampling_frequency, filtered_lfps, speed)

    smoothed = _smoothed_envelope(filtered_lfps, blocks, sampling_frequency, smoothing_sigma)
    if manual_normalization:
        if elec_baselines is None or elec_deviations is None:
            raise ValueError(
                "Must provide elec_baselines and elec_deviations for manual normalization."
            )
        normalized = normalize_signal_manually(smoothed, elec_baselines, elec_deviations)
    else:
        mask = _normalization_mask_over_valid(len(time), is_valid, normalization_mask)
        normalized = normalize_signal(
            smoothed, method=normalization_method, normalization_mask=mask
        )

    candidate_ripple_times = [
        _threshold_blocks(channel, time, blocks, minimum_duration, zscore_threshold)
        for channel in normalized.T
    ]
    # Merge each channel's mean-crossing-extended intervals and retain the union
    # of contributing channels, preserving the original participation rule.
    merged_candidates = merge_overlapping_ranges_track_participation(candidate_ripple_times)

    n_elecs = normalized.shape[1]
    # round so that 25 channels at 0.28 ask for 7, not the 7.000000000000001 of
    # floating-point multiplication, which would demand 8
    n_elecs_thresh = (
        minimum_participating_channels
        if minimum_participating_fraction is None
        else round(n_elecs * minimum_participating_fraction, 9)
    )
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
    participants = merged_candidates[participation_mask, 2][included_ripple_inds]
    ripple_times, keep = _exclude_long_events(ripple_times, time, maximum_duration)
    participants = participants[keep]

    n_participants = np.array([len(p) for p in participants], dtype=int)
    return _get_event_stats(
        ripple_times,
        time,
        normalized,
        speed,
        minimum_duration,
        participants,
        n_participants,
        n_participants / n_elecs,
        blocks=blocks,
    )


def _validate_duration_limits(minimum_duration: float, maximum_duration: float | None) -> None:
    """Reject duration limits that leave no admissible event."""
    if maximum_duration is not None and maximum_duration < minimum_duration:
        raise ValueError(
            f"maximum_duration ({maximum_duration}) is below minimum_duration "
            f"({minimum_duration}); no event could satisfy both. Both are in seconds."
        )


def _exclude_long_events(
    event_times: ArrayLike, time: NDArray, maximum_duration: float | None
) -> tuple[NDArray, NDArray]:
    """Drop events longer than ``maximum_duration``.

    The limit applies to the event as it will be reported, after the bounds
    have been extended past the threshold crossing, because that is the
    duration a published maximum describes. ``minimum_duration`` is the other
    way round: it applies to the run above threshold, the Frank lab convention
    the package already follows.

    Duration is a sample count, not elapsed time: an event is kept when it
    holds at most ``round(maximum_duration * sampling_frequency)`` samples,
    the same rule ``minimum_duration`` uses. An event whose elapsed time is
    exactly ``maximum_duration`` holds one sample more than that and is
    dropped, so the effective ceiling is one sample short of the number given.
    At 1500 Hz a 0.5 s ceiling admits 750 samples, which span 499.3 ms.

    Applied after close events are excluded or merged, so an over-long event
    still suppresses or absorbs its neighbours before it is itself dropped.

    Parameters
    ----------
    event_times : array_like, shape (n_events, 2)
        ``[start_time, end_time]`` per event.
    time : ndarray, shape (n_time,)
        Sample timestamps in seconds.
    maximum_duration : float or None
        Longest allowed duration in seconds. None keeps every event.

    Returns
    -------
    event_times : ndarray, shape (n_kept, 2)
        The events within the limit.
    keep : ndarray, shape (n_events,)
        Boolean mask into the input, for filtering arrays that run alongside
        the events.

    """
    events = np.asarray(event_times, dtype=float).reshape(-1, 2)
    if maximum_duration is None or len(events) == 0:
        return events, np.ones(len(events), dtype=bool)
    start = nearest_sample_index(time, events[:, 0])
    stop = nearest_sample_index(time, events[:, 1])
    keep = np.asarray(sample_count_within(stop - start + 1, time, 0.0, maximum_duration))
    return events[keep], keep


def _detect_from_trace(
    trace: NDArray,
    time: NDArray,
    speed: NDArray,
    is_valid: NDArray,
    blocks: list[tuple[int, int]],
    *,
    minimum_duration: float,
    zscore_threshold: float,
    speed_threshold: float,
    close_event_threshold: float,
    maximum_duration: float | None = None,
    normalization_method: str = "zscore",
    normalization_mask: ArrayLike | None = None,
) -> pd.DataFrame:
    """Normalize one detection trace, threshold it, and summarize the events.

    The shared tail of every detector that thresholds a single trace. It
    normalizes the trace over the valid samples. Within each block it takes
    the runs at or above ``zscore_threshold`` that last ``minimum_duration``
    and extends each to the normalization center. It drops events whose first
    or last sample exceeds ``speed_threshold``, then events too close to the
    last retained event, then events longer than ``maximum_duration``. It then
    computes the per-event statistics, flagging events cut off by a block
    edge.

    Parameters
    ----------
    trace : ndarray, shape (n_time,)
        The unnormalized detection trace, NaN outside the valid blocks.
    time : ndarray, shape (n_time,)
        Sample timestamps in seconds.
    speed : ndarray, shape (n_time,)
        Speed in cm/s.
    is_valid : ndarray of bool, shape (n_time,)
    blocks : list of (start, stop)
        From ``_valid_blocks``.
    minimum_duration, zscore_threshold, speed_threshold, close_event_threshold : float
        As in the public detectors.
    maximum_duration : float, optional
        As in the public detectors. Default is None (no upper limit).
    normalization_method, normalization_mask
        Passed to ``normalize_signal``.

    Returns
    -------
    events : pd.DataFrame
        One row per event, indexed by ``event_number``.

    """
    mask = _normalization_mask_over_valid(len(time), is_valid, normalization_mask)
    normalized = normalize_signal(trace, method=normalization_method, normalization_mask=mask)
    candidate_times = _threshold_blocks(
        normalized, time, blocks, minimum_duration, zscore_threshold
    )
    event_times = exclude_movement(
        candidate_times, speed, time, speed_threshold=speed_threshold
    )
    event_times = exclude_close_events(event_times, close_event_threshold)
    event_times, _ = _exclude_long_events(event_times, time, maximum_duration)
    return _get_event_stats(
        event_times, time, normalized, speed, minimum_duration, blocks=blocks
    )


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
    maximum_duration: float | None = None,
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
        Maximum speed (in cm/s) for ripple detection. An event is kept only if
        the speed at its first and last sample is at or below this value
        (``exclude_movement``); speed inside the event is not tested. Apply a
        whole-event rule afterwards if one is needed. Default is 4.0 cm/s, which corresponds
        to immobility/slow movement in rodents.

        **Important**: Ensure your speed data is in cm/s. If using m/s, multiply
        by 100. To disable movement exclusion, set to a very large value (e.g., 1e6).
    minimum_duration : float, optional
        Minimum ripple duration in **seconds**. Default is 0.015 (15 milliseconds).
        The signal must stay at or above ``zscore_threshold`` for at least
        ``round(minimum_duration * sampling_frequency)`` consecutive samples
        (per Karlsson & Frank 2009); the event is then extended to the surrounding
        mean-crossings, so the reported ``duration`` is typically longer.
        Typical range: 0.015 - 0.100 s (15-100 ms). Lower values detect shorter
        events but may increase false positives.
    maximum_duration : float, optional
        Longest allowed event duration in **seconds**, applied to the event as
        it is reported rather than to the run above threshold, because that is
        what a published maximum describes. Default is None (no upper limit).
        Published ceilings run from a few hundred milliseconds to a couple of
        seconds.
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
        statistics. For example, use `speed <= speed_threshold` to compute
        statistics only during immobility. Default is None (use all data).

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
        - clipped_start, clipped_end: whether the event was cut off by missing
          data or the recording edge

        Returns empty DataFrame if no ripples detected. If this occurs, try:
        - Lowering zscore_threshold (e.g., from 2.0 to 1.5)
        - Lowering minimum_duration (e.g., from 0.015 to 0.010)
        - Increasing speed_threshold if movement exclusion is too strict
        - Verifying your data contains ripple oscillations (150-250 Hz)

    Notes
    -----
    Missing samples: a NaN in any channel of ``filtered_lfps`` or in ``speed``
    marks that sample missing, as does a step in ``time`` larger than 1.5
    sample intervals. The valid samples form contiguous blocks, and every
    step runs within a block, so nothing is computed across a gap and no
    event spans one. An event cut off by a gap or by the recording edge is
    kept and flagged in ``clipped_start`` and ``clipped_end``. Every detector
    in the package follows this rule.
    See the README's "Choosing a detector" table for how the detectors'
    conventions differ.

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
    >>> "start_time" in ripples.columns
    True

    References
    ----------
    .. [1] Kay, K., Sosa, M., Chung, J. E., Karlsson, M. P., Larkin, M. C., &
       Frank, L. M. (2016). A hippocampal network for spatial coding during
       immobility and sleep. Nature, 531(7593), 185-190.
       doi:10.1038/nature17144

    """
    _validate_duration_limits(minimum_duration, maximum_duration)
    time, filtered_lfps, speed = _validate_detector_inputs(
        time, filtered_lfps, speed, sampling_frequency, speed_threshold
    )
    is_valid, blocks = _valid_blocks(time, sampling_frequency, filtered_lfps, speed)

    consensus = get_Kay_ripple_consensus_trace(
        _mask_invalid(filtered_lfps, is_valid),
        sampling_frequency,
        smoothing_sigma=smoothing_sigma,
        time=time,
    )
    return _detect_from_trace(
        consensus,
        time,
        speed,
        is_valid,
        blocks,
        minimum_duration=minimum_duration,
        zscore_threshold=zscore_threshold,
        speed_threshold=speed_threshold,
        close_event_threshold=close_ripple_threshold,
        maximum_duration=maximum_duration,
        normalization_method=normalization_method,
        normalization_mask=normalization_mask,
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
    zscore_per_tetrode: bool = True,
    maximum_duration: float | None = None,
) -> pd.DataFrame:
    """Detect sharp-wave ripples with a data-driven noise threshold (Yu et al. 2017).

    The consensus trace is the median across tetrodes of each tetrode's
    smoothed, z-scored ripple-band envelope (``get_Yu_ripple_consensus_trace``).
    Its values during immobility are taken as noise plus a signal tail. The
    part below the mode is mirrored about the mode to estimate the noise
    distribution, and the detection threshold is the ``percentile`` of that
    mirrored distribution (``estimate_noise_threshold``). An event is a run of
    at least ``minimum_duration`` at or above the threshold, extended to where
    the trace returns to the immobility mean.

    Like every detector in the package, it splits the recording into
    contiguous blocks of valid samples, so smoothing, thresholding, and event
    extraction never cross a gap, and an event truncated by a gap or by the
    end of the recording is kept and flagged in ``clipped_start`` and
    ``clipped_end``.

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
        Immobility is speed at or below this value (cm/s), the package's rule
        (the paper says "below 4 cm/s"; the two differ only at exact equality).
        It selects the noise sample for the threshold and, at event
        boundaries, which events are kept. Default is 4.0.
    minimum_duration : float, optional
        Minimum time the consensus must stay at or above the threshold, in
        seconds, applied as a sample count (round-half-up). Default is 0.020.
    maximum_duration : float, optional
        Longest allowed event duration in **seconds**, applied to the event as
        it is reported rather than to the run above threshold, because that is
        what a published maximum describes. Default is None (no upper limit).
        Published ceilings run from a few hundred milliseconds to a couple of
        seconds.
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
        Boolean mask selecting the noise sample instead of ``speed <=
        speed_threshold``.
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
        ``max_thresh``, z-score and speed statistics, ``clipped_start`` and
        ``clipped_end``) plus ``n_suprathreshold_samples`` (longest run at or
        above the threshold) and ``detection_threshold_zscore`` (the threshold
        in the normalized units the statistics are reported in).

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
    .. [1] Yu, J. Y., Kay, K., Liu, D. F., Grossrubatscher, I., Loback, A.,
       Sosa, M., Chung, J. E., Karlsson, M. P., Larkin, M. C., & Frank, L. M.
       (2017). Distinct hippocampal-cortical memory representations for
       experiences associated with movement versus immobility. eLife, 6,
       e27621. doi:10.7554/eLife.27621

    """
    _validate_duration_limits(minimum_duration, maximum_duration)
    time, filtered_lfps, speed = _validate_detector_inputs(
        time, filtered_lfps, speed, sampling_frequency, speed_threshold
    )
    is_valid, blocks = _valid_blocks(time, sampling_frequency, filtered_lfps, speed)

    consensus = get_Yu_ripple_consensus_trace(
        _mask_invalid(filtered_lfps, is_valid),
        sampling_frequency,
        smoothing_sigma=smoothing_sigma,
        zscore_per_tetrode=zscore_per_tetrode,
        time=time,
    )

    if normalization_mask is None:
        normalization_mask = speed <= speed_threshold
    noise_mask = _normalization_mask_over_valid(len(time), is_valid, normalization_mask)
    noise_values = consensus[noise_mask]
    baseline = np.mean(noise_values)
    scale = np.std(noise_values, ddof=0)  # the ddof normalize_signal uses
    normalized = normalize_signal(consensus, normalization_mask=noise_mask)
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

    n_min = minimum_sample_count(time, minimum_duration)
    event_times = [np.empty((0, 2))]
    n_suprathreshold = [np.empty(0, dtype=int)]
    for start, stop in blocks:
        if stop - start < n_min:
            continue
        block_events, block_n = _extract_Yu_ripple_events(
            normalized[start:stop], time[start:stop], minimum_duration, threshold_zscore
        )
        event_times.append(block_events)
        n_suprathreshold.append(block_n)
    event_times = np.concatenate(event_times)
    n_suprathreshold = np.concatenate(n_suprathreshold)

    # the per-event count is filtered alongside the events at each step
    keep = _is_immobile_at_endpoints(event_times, speed, time, speed_threshold)
    event_times, n_suprathreshold = event_times[keep], n_suprathreshold[keep]
    event_times, kept = exclude_close_events(
        event_times, close_ripple_threshold, included_ripple_inds=np.arange(len(event_times))
    )
    n_suprathreshold = n_suprathreshold[kept]
    event_times, keep = _exclude_long_events(event_times, time, maximum_duration)
    n_suprathreshold = n_suprathreshold[keep]

    events = _get_event_stats(
        event_times, time, normalized, speed, minimum_duration=minimum_duration, blocks=blocks
    )
    events["n_suprathreshold_samples"] = n_suprathreshold
    events["detection_threshold_zscore"] = threshold_zscore
    return events


def _zugaro_smoothing_window(sampling_frequency: float) -> int:
    """Moving-average length of the FindRipples power trace: 11 samples at 1250 Hz,
    scaled with the rate and kept odd so the filter is zero-phase."""
    window = round(sampling_frequency / 1250.0 * 11.0)
    return window + 1 if window % 2 == 0 else window


def _two_threshold_events(
    zscored: NDArray,
    time: NDArray,
    low_threshold: float,
    high_threshold: float,
    minimum_inter_ripple_interval: float,
    minimum_duration: float,
    maximum_duration: float | None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Segment a normalized power trace with the FindRipples two-threshold rule.

    Follows the segmentation rule of FMAToolbox ``FindRipples``:

    1. Candidate events are runs strictly above ``low_threshold``. An event
       starts at the last sample *at or below* the threshold before the run and
       ends at the run's last sample, as the original's ``diff``-based
       crossing search does. A run touching the block's first or last sample
       lacks one crossing; the original discards it, this package keeps it
       and flags it ``clipped_start`` or ``clipped_end``.
    2. Consecutive candidates are merged, one neighbor per pass, while the
       gap between them is under ``minimum_inter_ripple_interval`` and the
       merged span is under ``maximum_duration``. This is deliberately not
       :func:`~ripple_detection.core.merge_close_events`: that helper works on
       times with a tolerance at the boundary and an inclusive span cap, while
       FindRipples compares sample indices with strict inequalities. Keeping
       this loop is what keeps the detector faithful to the algorithm it is
       named for.
    3. A candidate is kept only if its maximum is strictly above
       ``high_threshold``.
    4. Candidates whose sample count (first to last sample, inclusive) is
       below ``minimum_duration`` or above ``maximum_duration`` are dropped.
       The package's duration rule (``sample_count_within``: round-half-up
       sample counts, inclusive limits) replaces the original's elapsed-time
       comparison; the two differ by one sample at an exact limit.

    Parameters
    ----------
    zscored : ndarray, shape (n_time,)
        Normalized smoothed power for one contiguous block.
    time : ndarray, shape (n_time,)
        Sample timestamps in seconds.
    low_threshold, high_threshold : float
        Boundary and peak thresholds in standard deviations.
    minimum_inter_ripple_interval, minimum_duration : float
        In seconds.
    maximum_duration : float or None
        In seconds; None removes the ceiling from the merge and from step 4.

    Returns
    -------
    event_times : ndarray, shape (n_events, 2)
        ``[start_time, end_time]`` per event.
    peak_times : ndarray, shape (n_events,)
        Time of the maximum of ``zscored`` within each event.
    clipped : ndarray of bool, shape (n_events, 2)
        Whether the event lacks its rising or its falling crossing, that is,
        its run began on the block's first sample or ended on its last. An
        event whose run began on the second sample has its crossing on the
        first and is not clipped, although it starts there.

    """
    zscored = np.asarray(zscored, dtype=float)
    time = np.asarray(time, dtype=float)
    empty = (np.empty((0, 2)), np.empty(0), np.empty((0, 2), dtype=bool))
    runs = _boolean_run_bounds(zscored > low_threshold)
    if len(runs) == 0:
        return empty
    # an event spans from the last sample at or below the low threshold before
    # the run to the run's last sample; a run that begins on the block's first
    # sample starts there instead, and is clipped
    events = np.column_stack([np.maximum(runs[:, 0] - 1, 0), runs[:, 1] - 1])
    clipped = np.column_stack([runs[:, 0] == 0, runs[:, 1] == len(zscored)])

    while len(events) > 1:
        gap = time[events[1:, 0]] - time[events[:-1, 1]]
        merged_span = time[events[1:, 1]] - time[events[:-1, 0]]
        to_merge = gap < minimum_inter_ripple_interval
        if maximum_duration is not None:
            to_merge &= merged_span < maximum_duration
        if not np.any(to_merge):
            break
        padded = np.concatenate([[False], to_merge])
        run_starts = np.flatnonzero(~padded[:-1] & padded[1:])
        events[run_starts, 1] = events[run_starts + 1, 1]
        clipped[run_starts, 1] = clipped[run_starts + 1, 1]
        events = np.delete(events, run_starts + 1, axis=0)
        clipped = np.delete(clipped, run_starts + 1, axis=0)

    peaks = np.array(
        [start + int(np.argmax(zscored[start : stop + 1])) for start, stop in events],
        dtype=int,
    )
    keep = zscored[peaks] > high_threshold
    n_samples = events[:, 1] - events[:, 0] + 1
    keep &= sample_count_within(n_samples, time, minimum_duration, maximum_duration)
    events, peaks, clipped = events[keep], peaks[keep], clipped[keep]
    return np.column_stack([time[events[:, 0]], time[events[:, 1]]]), time[peaks], clipped


def Zugaro_ripple_detector(
    time: ArrayLike,
    filtered_lfps: ArrayLike,
    speed: ArrayLike,
    sampling_frequency: float,
    speed_threshold: float = 4.0,
    low_threshold: float = 2.0,
    high_threshold: float = 5.0,
    minimum_inter_ripple_interval: float = 0.030,
    minimum_duration: float = 0.020,
    maximum_duration: float | None = 0.100,
    smoothing_window: int | None = None,
    normalization_mask: ArrayLike | None = None,
) -> pd.DataFrame:
    """Detect ripples with the FMAToolbox ``FindRipples`` two-threshold algorithm.

    The algorithm of Hajime Hirase as implemented by Michaël Zugaro in
    FMAToolbox [1]_ (``FindRipples``), carried into buzcode as
    ``bz_FindRipples`` [2]_ and into neurocode [3]_. The ripple-band signal is
    squared, summed across channels, smoothed with a short moving average and
    z-scored. A **low** threshold bounds each event, and an event is kept only
    if its **peak** exceeds a **high** threshold. Neighboring events closer
    than a minimum interval are merged, and events outside a duration range
    are discarded. Csicsvari et al. 1999 [4]_ describe the summed rms-power
    thresholding this rule descends from.

    The same two-threshold rule appears independently in the van der Meer lab's
    ``getSWR`` (vandermeerlab, ``code-matlab/tasks/Replay_Analysis/getSWR.m``):
    140-200 Hz, the Hilbert envelope rather than the squared signal, boundaries
    at 2 SD, merge within 20 ms applied before a 20 ms minimum, peak above
    5 SD. That variant is not implemented separately. It is this detector with
    ``low_threshold=2``, ``high_threshold=5``,
    ``minimum_inter_ripple_interval=0.02``, ``minimum_duration=0.02`` and no
    maximum, apart from using the envelope rather than the squared signal.

    This is a reimplementation from the algorithm, not a transcription. The
    original is GPL-3. Two departures, each documented per parameter below.
    The package's endpoint speed rule is applied. The peak is the maximum of
    the normalized power, not the trough of a single filtered channel.

    Missing samples are handled block-wise, as in every detector here, so
    smoothing and segmentation never cross a gap. A run that touches a gap or
    the record edge lacks one of its two crossings; the original drops it,
    this package keeps it and flags it in ``clipped_start`` or
    ``clipped_end``, so ``events[~(events.clipped_start | events.clipped_end)]``
    reproduces the original rule. A block shorter than the smoothing window is
    treated as missing, with a warning.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Time values for each sample in seconds.
    filtered_lfps : array_like, shape (n_time, n_channels)
        LFP signals already bandpass filtered to the ripple band. FMAToolbox
        documents 100-200 Hz input, buzcode filters 130-200 Hz, neurocode
        80-250 Hz; this package's ``filter_ripple_band`` gives 150-250 Hz.
        Channels are squared and summed, as the original does. NaN marks
        missing samples.
    speed : array_like, shape (n_time,)
        Animal's running speed in cm/s.
    sampling_frequency : float
        Sampling rate in Hz.
    speed_threshold : float, optional
        An event is kept only if the speed at its first and last sample is at
        or below this value (``exclude_movement``). **Not part of the original
        algorithm**, which has no speed criterion; pass ``np.inf`` to disable.
        Default is 4.0.
    low_threshold : float, optional
        Boundary threshold in standard deviations. Default is 2.0 (FMAToolbox
        and buzcode; neurocode's copy uses 0.5).
    high_threshold : float, optional
        Peak threshold in standard deviations. Default is 5.0 (FMAToolbox and
        buzcode; neurocode's copy uses 2.5).
    minimum_inter_ripple_interval : float, optional
        Events separated by less than this are merged, provided the merged
        event stays under ``maximum_duration``. Default is 0.030 s
        (FMAToolbox; neurocode uses 0.050).
    minimum_duration, maximum_duration : float, optional
        Events shorter or longer than these are discarded. Defaults are
        0.020 and 0.100 s (FMAToolbox; neurocode uses 0.025 and 0.500).
        The 100 ms ceiling is FMAToolbox's, and is shorter than the ceilings
        most of the replay literature uses. Raise it for a limit typical of
        that literature rather than of this algorithm's original settings, or
        pass ``maximum_duration=None`` for no ceiling, as the other detectors
        default to.
    smoothing_window : int, optional
        Moving-average length in samples. Default is the original's 11
        samples at 1250 Hz scaled to ``sampling_frequency`` and kept odd. A
        supplied value must be a positive odd integer.
    normalization_mask : array_like, shape (n_time,), optional
        Samples used for the z-score statistics (the original's ``restrict``).

    Returns
    -------
    ripple_times : pd.DataFrame
        One row per event, indexed by ``event_number``, with the columns of
        the other detectors plus ``peak_time``, the time of the maximum
        normalized power within the event.

    References
    ----------
    .. [1] Zugaro, M. FMAToolbox, ``Analyses/FindRipples.m`` (initial algorithm
       by H. Hirase). The repository has no license file; the file headers
       state GPL-3 or later.
       https://github.com/michael-zugaro/FMAToolbox/blob/6bbb3662f7ed1ccf09c5ff4b4d233e27e17c71a6/Analyses/FindRipples.m
    .. [2] Buzsáki lab, buzcode, ``analysis/SharpWaveRipples/bz_FindRipples.m``
       (edited by D. Tingley, 2017), GPL-3.
       https://github.com/buzsakilab/buzcode/blob/0969ddf7f55ccaca8c71969bee4b21f310840047/analysis/SharpWaveRipples/bz_FindRipples.m
    .. [3] AYA lab, neurocode, ``SharpWaveRipples/FindRipples.m`` (no license
       file), doi:10.5281/zenodo.7819979
       https://github.com/ayalab1/neurocode/blob/d166a67ffb73096d8d11b14be6693d96ad63e4ed/SharpWaveRipples/FindRipples.m
    .. [4] Csicsvari, J., Hirase, H., Czurkó, A., Mamiya, A., & Buzsáki, G.
       (1999). Oscillatory coupling of hippocampal pyramidal cells and
       interneurons in the behaving rat. Journal of Neuroscience, 19(1),
       274-287. doi:10.1523/JNEUROSCI.19-01-00274.1999

    """
    _validate_duration_limits(minimum_duration, maximum_duration)
    time, filtered_lfps, speed = _validate_detector_inputs(
        time, filtered_lfps, speed, sampling_frequency, speed_threshold
    )
    window = (
        _zugaro_smoothing_window(sampling_frequency)
        if smoothing_window is None
        else int(smoothing_window)
    )
    if window < 1 or window % 2 == 0:
        raise ValueError(f"smoothing_window must be a positive odd integer, got {window}.")
    is_valid, blocks = _valid_blocks(time, sampling_frequency, filtered_lfps, speed)
    blocks = _drop_short_blocks(blocks, is_valid, window, "the smoothing window")

    kernel = np.ones(window) / window
    power = np.sum(filtered_lfps**2, axis=1)
    smoothed = np.full(len(time), np.nan)
    for start, stop in blocks:
        smoothed[start:stop] = np.convolve(power[start:stop], kernel, mode="same")

    mask = _normalization_mask_over_valid(len(time), is_valid, normalization_mask)
    normalized = normalize_signal(smoothed, normalization_mask=mask)

    n_min = minimum_sample_count(time, minimum_duration)
    event_times = [np.empty((0, 2))]
    peak_times = [np.empty(0)]
    clipped = [np.empty((0, 2), dtype=bool)]
    for start, stop in blocks:
        if stop - start < n_min:
            continue
        block_events, block_peaks, block_clipped = _two_threshold_events(
            normalized[start:stop],
            time[start:stop],
            low_threshold,
            high_threshold,
            minimum_inter_ripple_interval,
            minimum_duration,
            maximum_duration,
        )
        event_times.append(block_events)
        peak_times.append(block_peaks)
        clipped.append(block_clipped)
    event_times = np.concatenate(event_times)
    peak_times = np.concatenate(peak_times)
    clipped_flags = np.concatenate(clipped)

    keep = _is_immobile_at_endpoints(event_times, speed, time, speed_threshold)
    event_times, peak_times, clipped_flags = (
        event_times[keep],
        peak_times[keep],
        clipped_flags[keep],
    )

    # clipped means "lacks a crossing" here, which the segmentation knows and
    # the event's position alone does not: a run beginning on the block's
    # second sample starts on its first yet has its crossing there
    events = _get_event_stats(
        event_times,
        time,
        normalized,
        speed,
        minimum_duration=minimum_duration,
        blocks=blocks,
        clipped=clipped_flags,
    )
    events["peak_time"] = peak_times
    return events


def _gaussian_lowpass_fir(
    cutoff: float, sampling_frequency: float, n_sd: float = 6.0
) -> NDArray:
    """Unit-area Gaussian low-pass kernel with standard deviation
    ``fs / (2 pi cutoff)`` samples, truncated at ``n_sd`` standard deviations
    (Eran Stark's ``makegausslpfir``)."""
    sigma_samples = sampling_frequency / (2.0 * np.pi * cutoff)
    return _unit_area_gaussian(sigma_samples, max(n_sd, 3.0))


def _firfilt(x: NDArray, kernel: NDArray) -> NDArray:
    """Zero-phase FIR filtering along axis 0 with the ends reflected.

    A centered convolution with the signal mirrored at both ends, which is what
    Eran Stark's ``firfilt`` (mirror-pad, causal filter, crop the delay)
    computes for the odd symmetric kernels used here.
    """
    return convolve1d(np.asarray(x, dtype=float), kernel, axis=0, mode="reflect")


def _difference_of_gaussians_band(
    x: NDArray, band: tuple[float, float], sampling_frequency: float
) -> NDArray:
    """Band-pass as the difference of two Gaussian low-passes: low-pass at the
    band's upper edge, minus a low-pass of that at the band's lower edge."""
    low_passed = _firfilt(x, _gaussian_lowpass_fir(band[1], sampling_frequency))
    slow = _firfilt(low_passed, _gaussian_lowpass_fir(band[0], sampling_frequency))
    return low_passed - slow


def _matlab_percentile(values: NDArray, percent: float) -> float:
    """MATLAB ``prctile``: linear interpolation between order statistics placed
    at percentiles 100 (k - 0.5) / n (NumPy's ``hazen`` method)."""
    return float(np.percentile(values, percent, method="hazen"))


def Long_sharp_wave_ripple_detector(
    time: ArrayLike,
    raw_lfps: ArrayLike,
    speed: ArrayLike,
    sampling_frequency: float,
    speed_threshold: float = 4.0,
    sharp_wave_band: tuple[float, float] = (2.0, 50.0),
    ripple_band: tuple[float, float] = (80.0, 250.0),
    sharp_wave_percentile: float = 10.0,
    ripple_power_percentile: float = 50.0,
    window_size: float = 0.040,
    local_window: float = 5.0,
    sharp_wave_thresholds: tuple[float, float] = (0.5, 2.5),
    ripple_thresholds: tuple[float, float] = (0.5, 2.5),
    minimum_separation: float = 0.050,
    minimum_sharp_wave_duration: float = 0.020,
    maximum_sharp_wave_duration: float = 0.500,
    minimum_ripple_duration: float = 0.025,
    random_state: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Detect sharp-wave ripples from a pyramidal-layer and a stratum radiatum channel.

    John D. Long II's two-channel detector (buzcode ``bz_DetectSWR`` [1]_,
    converted to buzcode by Andrea Navas-Olive; filtering routines adapted
    from Eran Stark's ``detect_hfos``; carried into AYA-lab neurocode as
    ``DetectSWR`` [2]_). It needs two channels: one that records the ripple,
    in or just above the CA1 pyramidal layer, and a deeper one that records
    the sharp wave in stratum radiatum. It cannot run on a single layer.

    **Unlike the other detectors, this one takes raw, unfiltered LFP**, shape
    ``(n_time, 2)`` with the ripple channel first, because it filters both
    bands itself. Handing it ripple-band data raises nothing and returns
    nonsense, and no check on the array can tell the two apart for every
    recording, so the caller must know which it holds. The sharp-wave feature is the ripple channel minus the
    radiatum channel after a 2-50 Hz difference-of-Gaussians band-pass; the
    ripple feature is the smoothed rectified 80-250 Hz band of the
    common-average-referenced pair, maximum over the two channels. In each
    non-overlapping 40 ms block the sharp-wave maximum and the ripple maximum
    within +/-20 ms of it form a feature pair; two-cluster k-means on those
    pairs separates sharp-wave ripples (the smaller cluster) from the rest,
    and candidates must exceed the 10th sharp-wave percentile of the SWR
    cluster and the 50th ripple-power percentile of the other. Because that
    sharp-wave cut is taken from the SWR cluster itself, the detector discards
    roughly the weakest tenth of its own events by construction. Each candidate
    is then tested against **local** statistics over +/-5 s: its sharp wave and
    ripple power must each be at least median + 2.5 SD, its boundaries are
    where the sharp wave falls back below median + 0.5 SD, candidates closer
    than 50 ms to the previous candidate are dropped, and duration limits
    apply. Event bounds are the sharp-wave bounds.

    Reimplemented from the algorithm as read. The source states no license.
    Five departures, each documented below. ``random_state`` seeds the
    k-means, where MATLAB's is unseeded. A candidate whose local window holds
    no sample below the boundary threshold is rejected, where the original
    errors. The package's endpoint speed rule is applied afterwards. Missing
    samples (NaN, or a gap in ``time``) split the recording into blocks: the
    filters and candidate windows run within each block, the k-means pools
    the candidates of every block, and a candidate within ``local_window`` of
    a block edge is not evaluated, as the original does at the record edges.
    A block shorter than the sharp-wave low-pass kernel is treated as missing,
    with a warning. ``clipped_start`` and ``clipped_end`` are therefore always
    False for this detector.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Time values for each sample in seconds.
    raw_lfps : array_like, shape (n_time, 2)
        **Raw** LFP: column 0 the ripple (pyramidal-layer) channel, column 1
        the sharp-wave (stratum radiatum) channel. NaN marks missing samples.
    speed : array_like, shape (n_time,)
        Animal's running speed in cm/s.
    sampling_frequency : float
        Sampling rate in Hz.
    speed_threshold : float, optional
        Endpoint speed rule (``exclude_movement``); not part of the original.
        Default is 4.0.
    sharp_wave_band, ripple_band : tuple of (float, float), optional
        Difference-of-Gaussians pass-bands in Hz. The edges are the corner
        frequencies of Gaussian low-passes, not a brick wall: gain is well
        under 1 at the nominal edges and tails off gradually beyond them, as
        in the original. Defaults (2, 50) and
        (80, 250).
    sharp_wave_percentile, ripple_power_percentile : float, optional
        Cluster-derived thresholds: a candidate must exceed this percentile
        of the SWR cluster's sharp-wave feature and this percentile of the
        non-SWR cluster's ripple power. Defaults 10 and 50.
    window_size : float, optional
        Candidate block length in seconds. Default is 0.040, following
        neurocode ``DetectSWR`` (as does ``minimum_separation`` 0.050); buzcode
        ``bz_DetectSWR`` uses 0.200 and 0.100.
    local_window : float, optional
        Half-width in seconds of the window for local statistics. Candidates
        closer than this to either end of their block are not evaluated.
        Default is 5.0.
    sharp_wave_thresholds, ripple_thresholds : tuple of (float, float), optional
        ``(boundary, peak)`` in local standard deviations. Defaults (0.5, 2.5).
    minimum_separation : float, optional
        Minimum time from the previous candidate, in seconds, whether or not
        that candidate was kept. Two quirks of the original are kept: the
        last candidate is exempt from the test, and the first is measured
        from the start of the record. Default 0.050.
    minimum_sharp_wave_duration, maximum_sharp_wave_duration : float, optional
        Sharp-wave duration limits in seconds, applied as inclusive
        round-half-up sample counts (``sample_count_within``). A candidate is
        rejected when both its sharp wave and its ripple are below their
        minimum, or when the sharp wave exceeds the maximum. Defaults 0.020
        and 0.500.
    minimum_ripple_duration : float, optional
        Ripple duration minimum in seconds. Default 0.025. The ripple's sample
        count is the original's, one less than the inclusive count used for
        the sharp wave, so ``ripple_duration`` is one sample shorter than the
        span between its two boundary crossings.
    random_state : int or numpy Generator, optional
        Seed for the k-means initialization. Default None.

    Returns
    -------
    ripple_times : pd.DataFrame
        One row per event, indexed by ``event_number``: ``start_time``,
        ``end_time``, ``peak_time`` (sharp-wave peak), ``duration``, the
        package's z-score statistics computed on the globally z-scored ripple
        power, the speed statistics, and ``sharp_wave_zscore``,
        ``sharp_wave_local_percentile``, ``ripple_power_zscore``,
        ``ripple_power_local_percentile`` (peak values against the local window),
        ``sharp_wave_duration`` and ``ripple_duration`` in seconds.

    References
    ----------
    .. [1] Long, J. D. II. ``bz_DetectSWR.m`` in buzcode, GPL-3. No
       accompanying paper.
       https://github.com/buzsakilab/buzcode/blob/0969ddf7f55ccaca8c71969bee4b21f310840047/analysis/SharpWaveRipples/bz_DetectSWR.m
    .. [2] AYA lab, neurocode, ``SharpWaveRipples/DetectSWR.m`` (no license
       file), doi:10.5281/zenodo.7819979
       https://github.com/ayalab1/neurocode/blob/d166a67ffb73096d8d11b14be6693d96ad63e4ed/SharpWaveRipples/DetectSWR.m

    """
    _validate_duration_limits(minimum_sharp_wave_duration, maximum_sharp_wave_duration)
    lfp = np.asarray(raw_lfps, dtype=float)
    if lfp.ndim != 2 or lfp.shape[1] != 2:
        raise ValueError(
            "raw_lfps must have exactly two channels, shape (n_time, 2): the ripple "
            f"channel first and the sharp-wave channel second; got shape {lfp.shape}."
        )
    time, lfp, speed = _validate_detector_inputs(
        time, lfp, speed, sampling_frequency, speed_threshold
    )
    n_time = len(time)
    is_valid, blocks = _valid_blocks(time, sampling_frequency, lfp, speed)
    slowest_kernel = len(_gaussian_lowpass_fir(sharp_wave_band[0], sampling_frequency))
    blocks = _drop_short_blocks(
        blocks,
        is_valid,
        slowest_kernel,
        f"the {sharp_wave_band[0]} Hz sharp-wave low-pass "
        f"({slowest_kernel / sampling_frequency:.2f} s)",
    )
    rng = np.random.default_rng(random_state)

    # features, within each block
    power_kernel = _gaussian_lowpass_fir(np.mean(ripple_band) / np.pi, sampling_frequency)
    sharp_wave_diff = np.full(n_time, np.nan)
    ripple_power = np.full(n_time, np.nan)
    for start, stop in blocks:
        block_lfp = lfp[start:stop]
        band_lfp = _difference_of_gaussians_band(
            block_lfp, sharp_wave_band, sampling_frequency
        )
        sharp_wave_diff[start:stop] = band_lfp[:, 0] - band_lfp[:, 1]
        referenced = block_lfp - block_lfp.mean(axis=1, keepdims=True)
        ripple = np.abs(
            _difference_of_gaussians_band(referenced, ripple_band, sampling_frequency)
        )
        ripple_power[start:stop] = _firfilt(ripple, power_kernel).max(axis=1)

    # candidate feature pairs, one per non-overlapping window within a block
    window = int(np.floor(window_size * sampling_frequency))
    half_window = window // 2
    bound = int(local_window * sampling_frequency)
    feature_index, sharp_wave_feature, ripple_feature, in_range = [], [], [], []
    for block_start, block_stop in blocks:
        # arange to block_stop inclusive, so a block that is an exact multiple of
        # the window keeps its last complete window; a partial window is dropped
        for w0, w1 in pairwise(np.arange(block_start, block_stop + 1, window)):
            segment = sharp_wave_diff[w0:w1]
            local_arg = int(np.argmax(segment))
            peak = int(w0) + local_arg
            if local_arg in (0, window - 1):
                if peak in (block_start, block_stop - 1):
                    continue
                if int(np.argmax(sharp_wave_diff[peak - 1 : peak + 2])) != 1:
                    continue
            feature_index.append(peak)
            sharp_wave_feature.append(segment[local_arg])
            lo = max(peak - half_window, block_start)
            hi = min(peak + half_window, block_stop - 1)
            ripple_feature.append(ripple_power[lo : hi + 1].max())
            # the local statistics need the whole +/- local_window inside the block
            in_range.append(peak - bound >= block_start and peak + bound <= block_stop - 1)
    feature_index = np.asarray(feature_index, dtype=int)
    sharp_wave_feature = np.asarray(sharp_wave_feature)
    ripple_feature = np.asarray(ripple_feature)
    in_range = np.asarray(in_range, dtype=bool)
    if len(feature_index) < 2:
        raise ValueError("Too few candidate windows to cluster; the recording is too short.")

    features = np.column_stack([sharp_wave_feature, ripple_feature])
    _, labels = kmeans2(features, 2, iter=100, minit="++", seed=rng)
    is_swr_cluster = labels == (0 if np.sum(labels == 0) <= np.sum(labels == 1) else 1)
    if not np.any(is_swr_cluster) or np.all(is_swr_cluster):
        raise ValueError("k-means did not separate the candidate features into two clusters.")
    sharp_wave_cut = _matlab_percentile(
        sharp_wave_feature[is_swr_cluster], sharp_wave_percentile
    )
    ripple_cut = _matlab_percentile(ripple_feature[~is_swr_cluster], ripple_power_percentile)
    is_candidate = (
        is_swr_cluster
        & (sharp_wave_feature > sharp_wave_cut)
        & (ripple_feature > ripple_cut)
        & in_range
    )
    candidate_peaks = feature_index[is_candidate]
    candidate_sharp = sharp_wave_feature[is_candidate]
    candidate_ripple = ripple_feature[is_candidate]
    # time from the previous candidate; the first is measured from the record start
    separation = np.diff(time[candidate_peaks], prepend=time[0])

    sw_boundary, sw_peak = sharp_wave_thresholds
    rp_boundary, rp_peak = ripple_thresholds

    records = []
    n_candidates = len(candidate_peaks)
    for ii, peak in enumerate(candidate_peaks):
        sw_window = sharp_wave_diff[peak - bound : peak + bound + 1]
        sw_median, sw_sd = np.median(sw_window), np.std(sw_window, ddof=1)
        if sw_window[bound] < sw_median + sw_peak * sw_sd:
            continue
        rp_window = ripple_power[peak - bound : peak + bound + 1]
        rp_median, rp_sd = np.median(rp_window), np.std(rp_window, ddof=1)
        if candidate_ripple[ii] < rp_median + rp_peak * rp_sd:
            continue
        if ii < n_candidates - 1 and separation[ii] < minimum_separation:
            continue
        below_before = np.flatnonzero(sw_window[: bound + 1] < sw_median + sw_boundary * sw_sd)
        below_after = np.flatnonzero(sw_window[bound:] < sw_median + sw_boundary * sw_sd)
        if len(below_before) == 0 or len(below_after) == 0:
            continue
        start_offset = bound - below_before[-1]  # samples before the peak
        stop_offset = below_after[0]  # samples after the peak
        sharp_wave_samples = start_offset + stop_offset + 1
        rp_peak_local = (bound - half_window) + int(
            np.argmax(rp_window[bound - half_window : bound + half_window + 1])
        )
        rp_before = np.flatnonzero(
            rp_window[: rp_peak_local + 1] < rp_median + rp_boundary * rp_sd
        )
        rp_after = np.flatnonzero(rp_window[rp_peak_local:] < rp_median + rp_boundary * rp_sd)
        if len(rp_before) == 0 or len(rp_after) == 0:
            continue
        ripple_samples = (rp_peak_local + rp_after[0]) - rp_before[-1]
        ripple_long_enough = sample_count_within(ripple_samples, time, minimum_ripple_duration)
        sharp_wave_long_enough = sample_count_within(
            sharp_wave_samples, time, minimum_sharp_wave_duration
        )
        if not ripple_long_enough and not sharp_wave_long_enough:
            continue
        if (
            not sample_count_within(
                sharp_wave_samples,
                time,
                minimum_sharp_wave_duration,
                maximum_sharp_wave_duration,
            )
            and sharp_wave_long_enough
        ):
            continue
        records.append(
            {
                "peak": peak,
                "start": peak - start_offset,
                "end": peak + stop_offset,
                "sharp_wave_zscore": (candidate_sharp[ii] - sw_median) / sw_sd,
                "sharp_wave_local_percentile": np.mean(sw_window < candidate_sharp[ii]),
                "ripple_power_zscore": (candidate_ripple[ii] - rp_median) / rp_sd,
                "ripple_power_local_percentile": np.mean(rp_window < candidate_ripple[ii]),
                "sharp_wave_duration": sharp_wave_samples / sampling_frequency,
                "ripple_duration": ripple_samples / sampling_frequency,
            }
        )

    detected = pd.DataFrame(records)
    if records:
        event_times = np.column_stack([time[detected["start"]], time[detected["end"]]])
        keep = _is_immobile_at_endpoints(event_times, speed, time, speed_threshold)
        detected = detected[keep].reset_index(drop=True)
        event_times = event_times[keep]
    else:
        event_times = np.empty((0, 2))

    events = _get_event_stats(
        # the reported event spans the sharp wave, so the sustained-value window
        # is measured against the sharp-wave minimum
        event_times,
        time,
        normalize_signal(ripple_power),
        speed,
        minimum_duration=minimum_sharp_wave_duration,
        blocks=blocks,
    )
    events["peak_time"] = (
        time[detected["peak"].to_numpy(dtype=int)] if len(detected) else np.empty(0)
    )
    for column in (
        "sharp_wave_zscore",
        "sharp_wave_local_percentile",
        "ripple_power_zscore",
        "ripple_power_local_percentile",
        "sharp_wave_duration",
        "ripple_duration",
    ):
        events[column] = detected[column].to_numpy() if len(detected) else np.empty(0)
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
    bounds = _boolean_run_bounds(is_in_state)
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
    return np.asarray(inside.any(axis=1), dtype=bool)


def _count_active_units(multiunit: NDArray, event_bounds: ArrayLike) -> NDArray:
    """Number of units with at least one spike inside each event.

    Units are counted, not spikes, so a unit bursting hard counts once. The
    interval is closed: a unit whose only spike falls on the event's last
    sample is inside it.

    A loop over events rather than a vectorized cumulative sum, because the
    cumulative form needs a second array the size of ``multiunit`` while this
    reduces one event's slice at a time, and a recording holds far more
    samples than events.

    Parameters
    ----------
    multiunit : ndarray, shape (n_time, n_units)
        Spike counts or indicators per unit.
    event_bounds : array_like, shape (n_events, 2)
        ``[first_sample, last_sample]`` per event, as indices into
        ``multiunit``. Both ends are included.

    Returns
    -------
    n_active_units : ndarray, shape (n_events,)

    """
    bounds = np.asarray(event_bounds, dtype=int).reshape(-1, 2)
    return np.array(
        [int(np.sum(multiunit[start : stop + 1].sum(axis=0) > 0)) for start, stop in bounds],
        dtype=int,
    )


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
    maximum_duration: float | None = None,
) -> pd.DataFrame:
    """Detect candidate replay events from ripple power and multiunit activity jointly.

    The candidate-event detector of Carey, Tanaka & van der Meer 2019 [1]_
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
    - **Joint score**: ``sqrt(ripple * multiunit)``, z-scored. The original
      first rescales it to mean 0.5, a positive factor the z-score removes,
      so that step is omitted. This combination is asymmetric. The multiunit score is
      floored at zero, so a ripple without a population burst cannot be a
      candidate. The ripple score is an envelope rescaled to mean 1 and is
      never zero, so a burst without a ripple can be. The joint score is
      therefore closer to "burst, weighted by ripple power" than to a
      symmetric conjunction. A candidate is a run strictly above
      ``edge_threshold`` whose maximum is strictly above ``peak_threshold``.
      Its sample count must meet ``minimum_duration`` under the package's
      duration rule.
    - **State**: a candidate is kept only if it lies entirely inside a
      low-speed interval (speed at or below ``speed_threshold``, runs merged across
      gaps under ``state_merge_gap`` and dropped under
      ``state_minimum_length``) and, when ``theta_lfp`` is given, inside a
      low-theta interval (z-scored theta-band envelope below
      ``theta_threshold``, same interval rules), and has at least
      ``minimum_active_units`` units with a spike inside it.

    The original works in samples at 2 kHz (kernel SDs of 40 and 250 samples,
    +/-60-sample ripple smoothing); the defaults here are those values in
    seconds. Its speed limit is 10 pixels/s in tracking units; the default
    here is the package's 4 cm/s. Missing samples (NaN in the LFP, the
    spikes, speed or ``theta_lfp``, or a gap in ``time``) split the recording
    into blocks; the scores, the segmentation and the state intervals run
    within each block, so no candidate spans a gap, and a candidate cut off by
    one is flagged in ``clipped_start`` and ``clipped_end``.

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
        Speed at or below which the animal is considered stopped. Default is 4.0.
    edge_threshold, peak_threshold : float, optional
        Boundary and peak thresholds on the z-scored joint score. Defaults 1
        and 3 (the original's ``DetectorThreshold`` and ``DetectorThreshold2``).
    minimum_duration : float, optional
        Minimum candidate duration in seconds, applied as an inclusive
        round-half-up sample count (``sample_count_within``); the original's
        ``RemoveIV`` compared elapsed time strictly. Default 0.020.
    maximum_duration : float, optional
        Longest allowed event duration in **seconds**, applied to the event as
        it is reported rather than to the run above threshold, because that is
        what a published maximum describes. Default is None (no upper limit).
        Published ceilings run from a few hundred milliseconds to a couple of
        seconds.
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
        Theta pass-band in Hz, Butterworth of total order 4 (order 2 per edge, as MATLAB's `fdesign` 'N' counts it). Default (6, 10).
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
    .. [1] Carey, A. A., Tanaka, Y., & van der Meer, M. A. A. (2019). Reward
       revaluation biases hippocampal replay content away from the preferred
       outcome. Nature Neuroscience, 22(9), 1450-1459.
       doi:10.1038/s41593-019-0464-6
    .. [2] van der Meer lab, ``code-matlab/tasks/Alyssa_Tmaze/GenCandidateEvents.m``
       with ``beta/OldWizard.m``, ``beta/amMUA.m``, and ``beta/TSDtoIV2.m``.
       https://github.com/vandermeerlab/vandermeerlab/blob/82ba3fe29cc3912575b32a0fcdaaa1c4fe097231/code-matlab/tasks/Alyssa_Tmaze/GenCandidateEvents.m

    """
    _validate_duration_limits(minimum_duration, maximum_duration)
    multiunit = np.asarray(multiunit, dtype=float)
    if multiunit.ndim != 2:
        raise ValueError(
            f"multiunit must be a 2D array of shape (n_time, n_units), got shape {multiunit.shape}."
        )
    time, filtered_lfps, speed = _validate_detector_inputs(
        time, filtered_lfps, speed, sampling_frequency, speed_threshold
    )
    n_time = len(time)
    if multiunit.shape[0] != n_time:
        raise ValueError(
            f"Array length mismatch: multiunit has {multiunit.shape[0]} samples but time has {n_time}."
        )
    signals = [filtered_lfps, multiunit, speed]
    theta_filter = None
    if theta_lfp is not None:
        theta_lfp = np.asarray(theta_lfp, dtype=float)
        if theta_lfp.shape != (n_time,):
            raise ValueError(f"theta_lfp must have shape ({n_time},), got {theta_lfp.shape}.")
        signals.append(theta_lfp)
        theta_filter = butter(
            2, np.asarray(theta_band) / (0.5 * sampling_frequency), btype="bandpass"
        )
    is_valid, blocks = _valid_blocks(time, sampling_frequency, *signals)
    if theta_filter is not None:
        # filtfilt needs strictly more samples than its default pad length
        padlen = 3 * max(len(theta_filter[0]), len(theta_filter[1]))
        blocks = _drop_short_blocks(blocks, is_valid, padlen + 1, "the theta filter")

    # ripple score (OldWizard, 'amplitude', 'wizard' kernel), rescaled to mean 1
    ripple_score = np.full(n_time, np.nan)
    for start, stop in blocks:
        envelope = get_envelope(filtered_lfps[start:stop]).mean(axis=1)
        ripple_score[start:stop] = gaussian_filter1d(
            envelope,
            ripple_smoothing_sigma * sampling_frequency,
            truncate=3.0,
            mode="constant",
        )
    ripple_score = ripple_score / np.nanmean(ripple_score)

    # multiunit score (amMUA)
    sigma_samples = spike_kernel_sigma * sampling_frequency
    spike_kernel = _unit_area_gaussian(sigma_samples, 5.0)
    cap = spike_cap / (sigma_samples * np.sqrt(2.0 * np.pi))
    baseline_kernel = _unit_area_gaussian(baseline_sigma * sampling_frequency, 12.0)
    summed = np.full(n_time, np.nan)
    baseline = np.full(n_time, np.nan)
    for start, stop in blocks:
        # convolve1d with zero padding equals np.convolve(..., "same") for these
        # odd symmetric kernels, and also works on a block shorter than the kernel
        block_sum = np.zeros(stop - start)
        for unit in multiunit[start:stop].T:
            block_sum += np.minimum(convolve1d(unit, spike_kernel, mode="constant"), cap)
        summed[start:stop] = block_sum
        baseline[start:stop] = convolve1d(
            np.minimum(baseline_cap * cap, block_sum), baseline_kernel, mode="constant"
        )
    mean_summed = np.nanmean(summed)
    if mean_summed <= 0:
        raise ValueError("multiunit contains no spikes; cannot form a multiunit score.")
    multiunit_score = np.maximum(0.0, (summed - baseline - cap) / mean_summed)
    if not np.any(multiunit_score > 0):
        raise ValueError(
            "The multiunit score never rises above its baseline, so no candidate is "
            "possible: the population never fires more than spike_cap coincident "
            "spikes per unit above its slow rate. Check that multiunit holds spikes at "
            "sampling_frequency, not a rate, and that baseline_cap and spike_cap fit it."
        )

    joint = np.sqrt(ripple_score * multiunit_score)
    zscored = normalize_signal(joint)

    # two-threshold segmentation (TSDtoIV2) within each block: runs above the
    # edge, kept if the peak is above
    candidates = []
    for start, stop in blocks:
        block_z = zscored[start:stop]
        for run_start, run_stop in _boolean_run_bounds(block_z > edge_threshold):
            if block_z[run_start:run_stop].max() > peak_threshold:
                candidates.append((start + run_start, start + run_stop - 1))
    candidates = np.asarray(candidates, dtype=int).reshape(-1, 2)
    if len(candidates):
        n_samples = candidates[:, 1] - candidates[:, 0] + 1
        candidates = candidates[sample_count_within(n_samples, time, minimum_duration)]

    # state restriction: contained in a low-speed (and low-theta) interval
    def _intervals(is_in_state: NDArray) -> NDArray:
        return np.concatenate(
            [np.empty((0, 2), dtype=int)]
            + [
                start
                + _state_intervals(
                    is_in_state[start:stop],
                    time[start:stop],
                    state_merge_gap,
                    state_minimum_length,
                )
                for start, stop in blocks
            ]
        )

    if len(candidates):
        candidates = candidates[
            _contained_in_intervals(candidates, _intervals(speed <= speed_threshold))
        ]
    if len(candidates) and theta_filter is not None:
        theta_envelope = np.full(n_time, np.nan)
        for start, stop in blocks:
            theta_envelope[start:stop] = get_envelope(
                filtfilt(*theta_filter, theta_lfp[start:stop])
            )
        low_theta = _intervals(normalize_signal(theta_envelope) < theta_threshold)
        candidates = candidates[_contained_in_intervals(candidates, low_theta)]

    # minimum number of active units
    n_active = _count_active_units(multiunit, candidates)
    if len(candidates):
        keep = n_active >= minimum_active_units
        candidates, n_active = candidates[keep], n_active[keep]

    event_times = (
        np.column_stack([time[candidates[:, 0]], time[candidates[:, 1]]])
        if len(candidates)
        else np.empty((0, 2))
    )
    event_times, keep = _exclude_long_events(event_times, time, maximum_duration)
    n_active = n_active[keep]
    events = _get_event_stats(
        event_times, time, zscored, speed, minimum_duration=minimum_duration, blocks=blocks
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
    maximum_duration: float | None = None,
) -> pd.DataFrame:
    """Detect sharp-wave ripples using per-channel detection with merging.

    Implements the Karlsson & Frank 2009 algorithm, which detects ripples on
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
        Maximum speed (in cm/s) for ripple detection. An event is kept only if
        the speed at its first and last sample is at or below this value
        (``exclude_movement``); speed inside the event is not tested. Apply a
        whole-event rule afterwards if one is needed. Default is 4.0 cm/s, which corresponds
        to immobility/slow movement in rodents.

        **Important**: Ensure your speed data is in cm/s. If using m/s, multiply
        by 100. To disable movement exclusion, set to a very large value (e.g., 1e6).
    minimum_duration : float, optional
        Minimum ripple duration in **seconds**. Default is 0.015 (15 milliseconds).
        The signal must stay at or above ``zscore_threshold`` for at least
        ``round(minimum_duration * sampling_frequency)`` consecutive samples
        (per Karlsson & Frank 2009); the event is then extended to the surrounding
        mean-crossings, so the reported ``duration`` is typically longer.
        Typical range: 0.015 - 0.100 s (15-100 ms). Lower values detect shorter
        events but may increase false positives.
    maximum_duration : float, optional
        Longest allowed event duration in **seconds**, applied to the event as
        it is reported rather than to the run above threshold, because that is
        what a published maximum describes. Default is None (no upper limit).
        Published ceilings run from a few hundred milliseconds to a couple of
        seconds.
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
        statistics. For example, use `speed <= speed_threshold` to compute
        statistics only during immobility. Default is None (use all data).

    Returns
    -------
    ripple_times : pd.DataFrame
        DataFrame with detected ripples and comprehensive statistics (see
        Kay_ripple_detector for column descriptions). The z-score statistics
        (``max_thresh``, ``mean_zscore``, ``max_zscore``, ...) are computed on
        the elementwise maximum across channels of the per-channel z-scores,
        i.e. the strongest tetrode at each sample, so ``max_thresh`` is at
        least ``zscore_threshold`` for every event.

        Returns empty DataFrame if no ripples detected. If this occurs, try:
        - Lowering zscore_threshold (e.g., from 3.0 to 2.0)
        - Lowering minimum_duration (e.g., from 0.015 to 0.010)
        - Increasing speed_threshold if movement exclusion is too strict
        - Verifying your data contains ripple oscillations (150-250 Hz)

    Notes
    -----
    Missing samples: a NaN in any channel of ``filtered_lfps`` or in ``speed``
    marks that sample missing, as does a step in ``time`` larger than 1.5
    sample intervals. The valid samples form contiguous blocks, and every
    step runs within a block, so nothing is computed across a gap and no
    event spans one. An event cut off by a gap or by the recording edge is
    kept and flagged in ``clipped_start`` and ``clipped_end``. Every detector
    in the package follows this rule.
    See the README's "Choosing a detector" table for how the detectors'
    conventions differ.

    References
    ----------
    .. [1] Karlsson, M. P., & Frank, L. M. (2009). Awake replay of remote
       experiences in the hippocampus. Nature Neuroscience, 12(7), 913-918.
       doi:10.1038/nn.2344

    """
    _validate_duration_limits(minimum_duration, maximum_duration)
    time, filtered_lfps, speed = _validate_detector_inputs(
        time, filtered_lfps, speed, sampling_frequency, speed_threshold
    )
    is_valid, blocks = _valid_blocks(time, sampling_frequency, filtered_lfps, speed)

    smoothed = _smoothed_envelope(filtered_lfps, blocks, sampling_frequency, smoothing_sigma)
    mask = _normalization_mask_over_valid(len(time), is_valid, normalization_mask)
    normalized = normalize_signal(
        smoothed, method=normalization_method, normalization_mask=mask
    )
    candidate_ripple_times = list(
        merge_overlapping_ranges(
            chain.from_iterable(
                _threshold_blocks(channel, time, blocks, minimum_duration, zscore_threshold)
                for channel in normalized.T
            )
        )
    )
    ripple_times = exclude_movement(
        candidate_ripple_times, speed, time, speed_threshold=speed_threshold
    )
    ripple_times = exclude_close_events(ripple_times, close_ripple_threshold)
    ripple_times, _ = _exclude_long_events(ripple_times, time, maximum_duration)

    # statistics on the strongest channel at each sample, so an event that one
    # channel triggered cannot report a sub-threshold max_thresh
    return _get_event_stats(
        ripple_times, time, normalized.max(axis=1), speed, minimum_duration, blocks=blocks
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
    maximum_duration: float | None = None,
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
        Maximum speed (in cm/s) for ripple detection. An event is kept only if
        the speed at its first and last sample is at or below this value
        (``exclude_movement``); speed inside the event is not tested. Apply a
        whole-event rule afterwards if one is needed. Default is 4.0 cm/s, which corresponds
        to immobility/slow movement in rodents.

        **Important**: Ensure your speed data is in cm/s. If using m/s, multiply
        by 100. To disable movement exclusion, set to a very large value (e.g., 1e6).
    minimum_duration : float, optional
        Minimum ripple duration in **seconds**. Default is 0.015 (15 milliseconds).
        The signal must stay at or above ``zscore_threshold`` for at least
        ``round(minimum_duration * sampling_frequency)`` consecutive samples
        (per Karlsson & Frank 2009); the event is then extended to the surrounding
        mean-crossings, so the reported ``duration`` is typically longer.
        Typical range: 0.015 - 0.100 s (15-100 ms). Lower values detect shorter
        events but may increase false positives.
    maximum_duration : float, optional
        Longest allowed event duration in **seconds**, applied to the event as
        it is reported rather than to the run above threshold, because that is
        what a published maximum describes. Default is None (no upper limit).
        Published ceilings run from a few hundred milliseconds to a couple of
        seconds.
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
        statistics. For example, use `speed <= speed_threshold` to compute
        statistics only during immobility. Default is None (use all data).

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

    Notes
    -----
    Missing samples: a NaN in any channel of ``filtered_lfps`` or in ``speed``
    marks that sample missing, as does a step in ``time`` larger than 1.5
    sample intervals. The valid samples form contiguous blocks, and every
    step runs within a block, so nothing is computed across a gap and no
    event spans one. An event cut off by a gap or by the recording edge is
    kept and flagged in ``clipped_start`` and ``clipped_end``. Every detector
    in the package follows this rule.
    See the README's "Choosing a detector" table for how the detectors'
    conventions differ.

    References
    ----------
    Unpublished Frank-lab variant contributed by Demetris Roumis (2017); it has
    no paper of its own. It averages across channels the square root of each
    channel's smoothed squared envelope, then z-scores, between Kay's
    consensus trace and Karlsson's per-channel rule.

    """
    _validate_duration_limits(minimum_duration, maximum_duration)
    time, filtered_lfps, speed = _validate_detector_inputs(
        time, filtered_lfps, speed, sampling_frequency, speed_threshold
    )
    is_valid, blocks = _valid_blocks(time, sampling_frequency, filtered_lfps, speed)

    smoothed_power = _smoothed_envelope(
        filtered_lfps, blocks, sampling_frequency, smoothing_sigma, square=True
    )
    combined = np.mean(np.sqrt(smoothed_power), axis=1)
    return _detect_from_trace(
        combined,
        time,
        speed,
        is_valid,
        blocks,
        minimum_duration=minimum_duration,
        zscore_threshold=zscore_threshold,
        speed_threshold=speed_threshold,
        close_event_threshold=close_ripple_threshold,
        maximum_duration=maximum_duration,
        normalization_method=normalization_method,
        normalization_mask=normalization_mask,
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
    normalization_method: str = "zscore",
    normalization_mask: ArrayLike | None = None,
    maximum_duration: float | None = None,
    minimum_active_units: int = 0,
) -> pd.DataFrame:
    """Detect High Synchrony Events from multiunit spiking activity.

    Identifies periods of elevated population spiking during immobility. The
    population firing rate, summed over units, is smoothed with a Gaussian
    kernel and z-scored. It is then thresholded with the same rules as the LFP
    detectors: at or above ``zscore_threshold`` for ``minimum_duration``, then
    extended to where the rate returns to the mean.

    The 15 ms smoothing kernel follows Davidson et al. 2009 [1]_. The
    selection rule does not. Davidson et al. define a candidate event as a
    period above the mean whose *peak* exceeds 3 s.d. They take the statistics
    from stopped periods only and impose no sustained-duration requirement.
    To approximate that convention, pass ``zscore_threshold=3.0``,
    ``minimum_duration=0.0`` and ``normalization_mask=speed < 5.0``, their
    stopped-period criterion. The defaults here, 2 s.d. held for 15 ms with
    statistics over all samples, are this package's own convention.

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
        Maximum speed (in cm/s) for event detection. An event is kept only if
        the speed at its first and last sample is at or below this value
        (``exclude_movement``); speed inside the event is not tested. Default
        is 4.0 cm/s, which corresponds to immobility/slow movement in rodents.

        **Important**: Ensure your speed data is in cm/s. If using m/s, multiply
        by 100. To disable movement exclusion, set to a very large value (e.g., 1e6).
    minimum_duration : float, optional
        Minimum event duration in **seconds**. Default is 0.015 (15 milliseconds).
        The firing rate must stay at or above ``zscore_threshold`` for at least
        ``round(minimum_duration * sampling_frequency)`` consecutive samples;
        the event is then extended to the surrounding mean-crossings, so the reported
        ``duration`` is typically longer.
        Typical range: 0.015 - 0.100 s (15-100 ms). Lower values detect shorter
        events but may increase false positives.
    maximum_duration : float, optional
        Longest allowed event duration in **seconds**, applied to the event as
        it is reported rather than to the run above threshold, because that is
        what a published maximum describes. Default is None (no upper limit).
        Published ceilings run from a few hundred milliseconds to a couple of
        seconds.
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
    minimum_active_units : int, optional
        Minimum number of units with at least one spike inside an event.
        Events with fewer are dropped. Default is 0, which imposes no
        criterion. Published criteria are most often around five units.
        ``Carey_candidate_detector`` applies the same rule with its original's
        default of 5.
    normalization_method : {'zscore', 'median_mad'}, optional
        Method for normalizing the firing rate. Default is 'zscore' (mean/std).
        Use 'median_mad' for more robust normalization when data contains outliers.
        The median/MAD method is more resistant to extreme values.
    normalization_mask : array_like, shape (n_time,), optional
        Boolean mask to specify which samples to use for computing normalization
        statistics. For example, use `speed <= speed_threshold` to compute
        statistics only during immobility.
        Default is None (use all data).

    Returns
    -------
    high_synchrony_events : pd.DataFrame
        DataFrame with detected events and comprehensive statistics (see
        Kay_ripple_detector for column descriptions), plus
        ``n_active_units``, the number of units with at least one spike
        inside the event.

        Returns empty DataFrame if no events detected. If this occurs, try:
        - Lowering zscore_threshold (e.g., from 2.0 to 1.5)
        - Lowering minimum_duration (e.g., from 0.015 to 0.010)
        - Increasing speed_threshold if movement exclusion is too strict
        - Verifying your multiunit data shows synchronous spiking activity

    Notes
    -----
    Missing samples: a NaN anywhere in ``multiunit`` or ``speed`` marks that
    sample missing, as does a step in ``time`` larger than 1.5 sample
    intervals. The population rate is smoothed within each contiguous block
    of valid samples, no event spans a gap, and an event cut off by one is
    flagged in ``clipped_start`` and ``clipped_end``. A spike count that is
    absent rather than missing should be 0, not NaN.

    The defaults (2 SD, 15 ms smoothing, 15 ms minimum, 4 cm/s) are this
    package's convention. Published multiunit-burst detectors in the same
    lineage use their own values (Davidson et al. 2009 among them), so set them
    explicitly when reproducing a paper.

    References
    ----------
    .. [1] Davidson, T. J., Kloosterman, F., & Wilson, M. A. (2009).
       Hippocampal replay of extended experience. Neuron, 63(4), 497-507.
       doi:10.1016/j.neuron.2009.07.027

    """
    if minimum_active_units < 0:
        raise ValueError(
            f"minimum_active_units must be non-negative, got {minimum_active_units}. "
            "It counts units with at least one spike inside an event; 0 imposes no criterion."
        )
    _validate_duration_limits(minimum_duration, maximum_duration)
    multiunit = np.asarray(multiunit, dtype=float)
    if multiunit.ndim != 2:
        raise ValueError(
            f"multiunit must be a 2D array of shape (n_time, n_units), got shape "
            f"{multiunit.shape}. For a single unit, pass multiunit[:, np.newaxis]."
        )
    time, multiunit, speed = _validate_detector_inputs(
        time, multiunit, speed, sampling_frequency, speed_threshold
    )
    is_valid, blocks = _valid_blocks(time, sampling_frequency, multiunit, speed)

    firing_rate = np.full(len(time), np.nan)
    for start, stop in blocks:
        firing_rate[start:stop] = get_multiunit_population_firing_rate(
            multiunit[start:stop], sampling_frequency, smoothing_sigma
        )

    events = _detect_from_trace(
        firing_rate,
        time,
        speed,
        is_valid,
        blocks,
        minimum_duration=minimum_duration,
        zscore_threshold=zscore_threshold,
        speed_threshold=speed_threshold,
        close_event_threshold=close_event_threshold,
        maximum_duration=maximum_duration,
        normalization_method=normalization_method,
        normalization_mask=normalization_mask,
    )

    start = nearest_sample_index(time, events.start_time.to_numpy())
    stop = nearest_sample_index(time, events.end_time.to_numpy())
    n_active = _count_active_units(multiunit, np.column_stack([start, stop]))
    keep = n_active >= minimum_active_units
    events = events.iloc[np.flatnonzero(keep)].copy()
    events["n_active_units"] = n_active[keep]
    # renumber, so the index is 1..n with no holes as it is for every other
    # detector; Carey filters before _get_event_stats and gets this for free
    events.index = pd.RangeIndex(1, len(events) + 1, name=events.index.name)
    return events


def _find_max_thresh(
    time: np.ndarray, data: np.ndarray, minimum_duration: float = 0.015
) -> float:
    """Find the largest value sustained for a minimum duration anywhere in the event.

    The largest threshold at which the event would still be detected: the
    maximum, over every window of ``minimum_sample_count(time, minimum_duration)``
    consecutive samples, of that window's minimum. The sample-count convention
    matches event detection, so an event detected at ``zscore_threshold`` has
    ``max_thresh >= zscore_threshold``. This is not the statistic of the Frank
    lab ``extractevents`` routine, which takes a window of the minimum
    duration centered on the peak and reports the lower of its two ends; the
    two differ by a few tenths of a standard deviation on typical events.

    Parameters
    ----------
    time : np.ndarray, shape (n_time,)
    data : np.ndarray, shape (n_time,)
    minimum_duration : float, optional

    Returns
    -------
    max_thresh : float
        The largest value sustained for ``minimum_duration`` within the event.
        ``nan`` if the event holds fewer samples than the minimum (the sustained
        value is then undefined). Most detectors never produce such an event,
        because their segments meet the minimum by construction.
        ``Long_sharp_wave_ripple_detector`` can: it admits an event on the
        ripple criterion alone, but reports the span of the sharp wave and
        measures this statistic against ``minimum_sharp_wave_duration``.

    """
    if len(data) < 2 and minimum_duration > 0:
        # a single sample has no measurable interval, so no duration is sustained
        return float("nan")
    n_min = minimum_sample_count(time, minimum_duration)
    if len(data) < n_min:
        return float("nan")
    windows = np.lib.stride_tricks.sliding_window_view(np.asarray(data, dtype=float), n_min)
    return float(windows.min(axis=1).max())


def _get_event_stats(
    event_times: ArrayLike,
    time: ArrayLike,
    zscore_metric: ArrayLike,
    speed: ArrayLike,
    minimum_duration: float = 0.015,
    participants: ArrayLike | None = None,
    n_participants: ArrayLike | None = None,
    frac_participants: ArrayLike | None = None,
    blocks: list[tuple[int, int]] | None = None,
    clipped: NDArray | None = None,
) -> pd.DataFrame:
    """Compute comprehensive statistics for detected events.

    Calculates temporal, z-score, signal, and speed metrics for each event.
    An event's samples are those with ``start_time <= time <= end_time``,
    found by bisection, so the cost does not grow with the recording length.

    Parameters
    ----------
    event_times : array_like, shape (n_events, 2)
        Array of [start_time, end_time] for each event.
    time : array_like, shape (n_time,)
        Time values for each sample, increasing.
    zscore_metric : array_like, shape (n_time,) or (n_time, n_channels)
        Signal the per-event statistics (mean/median/max/min z-score, area,
        total_energy, max_thresh) are computed from. Its exact meaning depends on
        the caller -- e.g. the consensus trace for Kay, the per-channel maximum for
        Karlsson, or the multiunit firing rate for multiunit_HSE. When participants
        is None, pass a single 1-D trace of shape (n_time,). When participants is
        provided, pass the per-channel signal of shape (n_time, n_channels) so that
        each event's metrics are computed from its participating channels only.
    speed : array_like, shape (n_time,)
        Animal's speed at each time point.
    minimum_duration : float, optional
        Minimum duration for max_thresh calculation. Default is 0.015 (15 ms).
    participants : array_like of set, shape (n_events,), optional
        Set of channels that participate in each event; z-score metrics are
        averaged over these channels. Used by Shvartsman_ripple_detector.
    n_participants : array_like, shape (n_events,), optional
        Number of distinct participating channels per event.
    frac_participants : array_like, shape (n_events,), optional
        ``n_participants`` divided by the total channel count, per event.
    blocks : list of (start, stop), optional
        The valid blocks the events were found in (``_valid_blocks``), for
        the ``clipped_start`` and ``clipped_end`` flags. Default is None,
        one block spanning the whole recording.
    clipped : ndarray of bool, shape (n_events, 2), optional
        Flags to report instead of the ones derived from ``blocks``, for a
        detector whose segmentation knows better whether an event was cut
        off (Zugaro). Default is None.

    Returns
    -------
    event_stats : pd.DataFrame
        One row per event, indexed by ``event_number`` from 1, with columns:
        - start_time, end_time: Event boundaries
        - duration: Event duration (end - start)
        - max_thresh: Largest value sustained for minimum_duration (NaN for an
            event holding fewer samples than the minimum; only
            ``Long_sharp_wave_ripple_detector`` produces one)
        - mean_zscore, median_zscore, max_zscore, min_zscore: Z-score statistics
        - area: Integral of z-score over event duration
        - total_energy: Integral of squared z-score
        - speed_at_start, speed_at_end: Speed at event boundaries
        - max_speed, min_speed, median_speed, mean_speed: Speed statistics
        - clipped_start, clipped_end: Whether the event's first or last sample
            is the first or last sample of its block, i.e. the event was cut
            off by missing data or the recording edge
        - participants, n_participants, frac_participants: Information on
            which channels exhibited a ripple during the detected event
            (returned if 'participants' input is not None)

    """
    events = np.asarray(event_times, dtype=float).reshape(-1, 2)
    time_arr = np.asarray(time, dtype=float)
    metric = np.asarray(zscore_metric, dtype=float)
    speed_arr = np.asarray(speed, dtype=float)
    if participants is None and metric.ndim != 1:
        raise ValueError(
            f"Without participants, zscore_metric must have shape (n_time,). Got shape "
            f"{metric.shape}."
        )
    if participants is not None and metric.ndim != 2:
        raise ValueError(
            "With participants, zscore_metric must have shape (n_time, n_channels), so "
            f"each event's metrics can come from its participating channels. Got shape "
            f"{metric.shape}."
        )

    first = np.searchsorted(time_arr, events[:, 0], side="left")
    last = np.searchsorted(time_arr, events[:, 1], side="right")
    rows = []
    for index, ((start_time, end_time), a, b) in enumerate(
        zip(events, first, last, strict=True)
    ):
        event_time = time_arr[a:b]
        event_speed = speed_arr[a:b]
        if participants is None:
            z = metric[a:b]
        else:
            z = metric[a:b][:, list(participants[index])].mean(axis=1)
        rows.append(
            (
                start_time,
                end_time,
                end_time - start_time,
                _find_max_thresh(event_time, z, minimum_duration),
                z.mean(),
                np.median(z),
                z.max(),
                z.min(),
                trapezoid(z, event_time),
                trapezoid(z**2, event_time),
                event_speed[0],
                event_speed[-1],
                event_speed.max(),
                event_speed.min(),
                np.median(event_speed),
                event_speed.mean(),
            )
        )

    columns = [
        "start_time",
        "end_time",
        "duration",
        "max_thresh",
        "mean_zscore",
        "median_zscore",
        "max_zscore",
        "min_zscore",
        "area",
        "total_energy",
        "speed_at_start",
        "speed_at_end",
        "max_speed",
        "min_speed",
        "median_speed",
        "mean_speed",
    ]
    values = np.asarray(rows, dtype=float).reshape(-1, len(columns))
    index = pd.Index(np.arange(len(events)) + 1, name="event_number")
    event_stats = pd.DataFrame(dict(zip(columns, values.T, strict=True)), index=index)

    if clipped is None:
        if blocks is None:
            blocks = [(0, len(time_arr))]
        block_starts = np.array([start for start, _ in blocks], dtype=int)
        block_stops = np.array([stop for _, stop in blocks], dtype=int)
        which = np.clip(np.searchsorted(block_starts, first, side="right") - 1, 0, None)
        clipped = np.column_stack([first == block_starts[which], last == block_stops[which]])
    clipped = np.asarray(clipped, dtype=bool).reshape(-1, 2)
    event_stats["clipped_start"] = clipped[:, 0]
    event_stats["clipped_end"] = clipped[:, 1]
    if participants is not None:
        event_stats["participants"] = participants
        event_stats["n_participants"] = n_participants
        event_stats["frac_participants"] = frac_participants
    return event_stats

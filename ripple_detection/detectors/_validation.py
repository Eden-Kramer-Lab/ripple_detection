"""Input checks shared by the detectors: shapes, lengths, units, duration limits."""

import warnings

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


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


def _validate_duration_limits(minimum_duration: float, maximum_duration: float | None) -> None:
    """Reject duration limits that leave no admissible event."""
    if maximum_duration is not None and maximum_duration < minimum_duration:
        raise ValueError(
            f"maximum_duration ({maximum_duration}) is below minimum_duration "
            f"({minimum_duration}); no event could satisfy both. Both are in seconds."
        )

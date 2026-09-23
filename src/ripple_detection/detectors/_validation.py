"""Input checks shared by the detectors: shapes, lengths, units, duration limits."""

import warnings

import numpy as np
from numpy.typing import ArrayLike

from ripple_detection.core import FloatArray, _check_non_negative


def _validate_lfp_dimensions(filtered_lfps: FloatArray) -> None:
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
    if filtered_lfps.ndim != 2:
        hint = (
            "\nIf you have a single channel, reshape your data using:\n"
            "  filtered_lfps = filtered_lfps.reshape(-1, 1)"
            if filtered_lfps.ndim == 1
            else ""
        )
        msg = (
            "filtered_lfps must be a 2D array with shape (n_time, n_channels).\n"
            f"Received a {filtered_lfps.ndim}D array with shape {filtered_lfps.shape}.{hint}"
        )
        raise ValueError(msg)


def _validate_array_lengths(
    time: FloatArray, filtered_lfps: FloatArray, speed: FloatArray
) -> None:
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
        msg = (
            "Array length mismatch detected. All inputs must have the same length.\n"
            f"  time:         {n_time_samples} samples\n"
            f"  filtered_lfps: {n_lfp_samples} samples\n"
            f"  speed:        {n_speed_samples} samples\n"
            "Ensure your time, LFP, and speed arrays are aligned and have matching lengths."
        )
        raise ValueError(msg)


def _validate_time_units(
    time: FloatArray, sampling_frequency: float, stacklevel: int = 4
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
        If time holds NaN or infinity, is not increasing, has a median step
        that is not positive (most timestamps repeat), appears to be in
        samples instead of seconds, or has a median step more than 10 percent
        away from ``1 / sampling_frequency``.

    Warnings
    --------
    UserWarning
        If the median time step differs from ``1 / sampling_frequency`` by more
        than 2 percent and at most 10.

    """
    if not np.all(np.isfinite(time)):
        msg = (
            f"time holds {np.count_nonzero(~np.isfinite(time))} NaN or infinite value(s). "
            "Every sample needs a timestamp; mark missing data with NaN in the signals."
        )
        raise ValueError(msg)
    if len(time) > 1:
        steps = np.diff(time)
        if np.any(steps < 0):
            msg = (
                "time must be increasing. Sort time, and the signals with it, before "
                "detecting: the event and speed lookups assume time order."
            )
            raise ValueError(msg)
        median_dt = np.median(steps)
        expected_dt = 1.0 / sampling_frequency
        if not median_dt > 0:
            msg = (
                f"The median time step is {median_dt}: most timestamps repeat, so no "
                "duration can be measured in samples. Check the time array."
            )
            raise ValueError(msg)

        # Check if time appears to be in samples instead of seconds
        if median_dt > 10 * expected_dt:
            msg = (
                f"Time array appears to be in samples, not seconds.\n"
                f"Median time step: {median_dt:.6f} (expected ~{expected_dt:.6f} for {sampling_frequency} Hz)\n"
                f"\n"
                f"Solution: Convert sample indices to seconds:\n"
                f"  time_seconds = time_samples / {sampling_frequency}"
            )
            raise ValueError(msg)
        # The nominal rate sets the smoothing widths and windows and the
        # timestamps set the sample counts. Beyond 10 % the two describe
        # different recordings (a stated 300 Hz on 1500 Hz data changed the
        # event count by a quarter), so raise; from 2 % warn, since a nominal
        # rate can differ from an acquisition system's true one by a few percent
        # while clocks drift by far less.
        if not np.isclose(median_dt, expected_dt, rtol=0.10):
            msg = (
                f"The median time step ({median_dt:.6g} s) is "
                f"{median_dt / expected_dt:.3g} times the interval sampling_frequency "
                f"implies ({expected_dt:.6g} s at {sampling_frequency} Hz). Pass the rate "
                "the timestamps were recorded at, and time in seconds."
            )
            raise ValueError(msg)
        if not np.isclose(median_dt, expected_dt, rtol=0.02):
            warnings.warn(
                f"Time array step ({median_dt:.6f} s) differs from expected sampling interval "
                f"({expected_dt:.6f} s at {sampling_frequency} Hz).\n"
                f"Verify that:\n"
                f"  1. time is in seconds (not milliseconds or samples)\n"
                f"  2. sampling_frequency ({sampling_frequency} Hz) is correct",
                UserWarning,
                stacklevel=stacklevel,
            )


def _validate_speed_units(
    speed: FloatArray, speed_threshold: float, stacklevel: int = 4
) -> None:
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
    moving = speed[speed > 0]  # NaN compares False, so it drops out here
    if moving.size == 0 or speed_threshold <= 1.0:
        return
    median_speed = np.median(moving)
    # a median under 0.5 against a typical threshold (> 1 cm/s) is m/s
    if median_speed < 0.5:
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
) -> tuple[FloatArray, FloatArray, FloatArray]:
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
        If ``sampling_frequency`` is not positive and finite,
        ``speed_threshold`` is negative or NaN, the signal is not 2-D, the
        lengths differ, time is not increasing or appears to be in samples, or
        speed is NaN everywhere while the movement criterion is on.

    """
    _check_positive(sampling_frequency=sampling_frequency)
    _check_non_negative(speed_threshold=speed_threshold)
    signal = np.asarray(signal, dtype=float)
    speed = np.asarray(speed, dtype=float)
    time = np.asarray(time, dtype=float)
    _validate_lfp_dimensions(signal)
    _validate_array_lengths(time, signal, speed)
    if speed.size and not np.isposinf(speed_threshold) and not np.any(np.isfinite(speed)):
        msg = (
            "speed is NaN at every sample, so no event can pass the movement criterion. "
            "Pass speed_threshold=np.inf to detect without one."
        )
        raise ValueError(msg)
    _validate_time_units(time, sampling_frequency, stacklevel=stacklevel)
    _validate_speed_units(speed, speed_threshold, stacklevel=stacklevel)
    return time, signal, speed


def _check_finite_non_negative(**values: float) -> None:
    """Raise for a value that is NaN, infinite or negative."""
    for name, value in values.items():
        if not 0 <= value < np.inf:
            msg = f"{name} must be finite and non-negative, got {value}."
            raise ValueError(msg)


def _check_positive(**values: float) -> None:
    """Raise for a value that is not a positive finite number."""
    for name, value in values.items():
        if not 0 < value < np.inf:
            msg = f"{name} must be positive and finite, got {value}."
            raise ValueError(msg)


MAXIMUM_PLAUSIBLE_MINIMUM = 1.0
"""Seconds. No minimum event duration, and no smoothing width, reaches a
second: published minimums run 15-100 ms. A larger value is milliseconds
given where seconds are expected."""

MAXIMUM_PLAUSIBLE_CEILING = 10.0
"""Seconds. Published duration ceilings and merge gaps stay within a couple
of seconds; a larger value is milliseconds given where seconds are expected."""


def _check_seconds(limit: float, consequence: str, **values: float) -> None:
    """Raise for a finite duration above ``limit`` seconds, naming the value
    in milliseconds too: the commonest unit slip is 15 for 15 ms."""
    for name, value in values.items():
        if np.isfinite(value) and value >= limit:
            msg = (
                f"{name} is in seconds; {value} s {consequence}. "
                f"For {value} ms pass {value / 1000}."
            )
            raise ValueError(msg)


def _check_smoothing_sigma(**values: float) -> None:
    """A Gaussian standard deviation in seconds: positive, finite, and under a
    second, since a longer kernel smooths every ripple away and usually means
    milliseconds were given."""
    _check_positive(**values)
    _check_seconds(MAXIMUM_PLAUSIBLE_MINIMUM, "would smooth every ripple away", **values)


def _check_gap(**values: float) -> None:
    """A gap or interval between events in seconds: non-negative, not NaN, and
    not beyond ``MAXIMUM_PLAUSIBLE_CEILING`` (infinity passes)."""
    _check_non_negative(**values)
    for name, value in values.items():
        if np.isfinite(value) and value > MAXIMUM_PLAUSIBLE_CEILING:
            _check_seconds(
                MAXIMUM_PLAUSIBLE_CEILING,
                "is longer than any gap between events",
                **{name: value},
            )


def _check_thresholds(
    low_name: str, low: float, high_name: str, high: float, minimum: float = 0.0
) -> None:
    """Two thresholds, each finite and at or above ``minimum``, the first not
    above the second (a bounds threshold above the peak threshold disables
    the peak test)."""
    for name, value in ((low_name, low), (high_name, high)):
        if not minimum <= value < np.inf:
            msg = f"{name} must be finite and at least {minimum}, got {value}."
            raise ValueError(msg)
    if low > high:
        msg = f"{low_name} ({low}) is above {high_name} ({high}); it must not be."
        raise ValueError(msg)


def _check_whole_number(name: str, value: float, minimum: int) -> None:
    """Raise unless ``value`` is a whole number at or above ``minimum``."""
    if not (np.isfinite(value) and value == int(value) and value >= minimum):
        msg = f"{name} must be a whole number of at least {minimum}, got {value}."
        raise ValueError(msg)


def _check_band(name: str, band: tuple[float, float], sampling_frequency: float) -> None:
    """A frequency band in Hz: two increasing edges between zero and Nyquist."""
    low, high = band
    nyquist = sampling_frequency / 2
    if not 0 < low < high < nyquist:
        msg = (
            f"{name} must be (low, high) with 0 < low < high < {nyquist:g} Hz, the "
            f"Nyquist frequency; got {band}."
        )
        raise ValueError(msg)


def _check_minimum_active_units(minimum_active_units: int, n_units: int) -> None:
    """A whole number of units, no more than there are."""
    _check_whole_number("minimum_active_units", minimum_active_units, 0)
    if minimum_active_units > n_units:
        msg = (
            f"minimum_active_units is {minimum_active_units} but multiunit has {n_units} "
            "unit(s), so no event could be kept."
        )
        raise ValueError(msg)


def _validate_multiunit(multiunit: FloatArray, what: str = "multiunit") -> None:
    """Spike counts or indicators, shape (n_time, n_units): non-negative whole
    numbers where finite.

    A rate in Hz or a baseline-subtracted count passes every shape check and
    changes what the spike cap and the z-score mean, so it is rejected here,
    in the detectors, and not only when a pipeline goes through the registry.
    The values are tested in row chunks, so the test's temporaries stay small
    next to an hour of spikes.
    """
    if multiunit.ndim != 2:
        msg = (
            f"{what} must be a 2D array of shape (n_time, n_units), got shape "
            f"{multiunit.shape}. For a single unit, pass multiunit[:, np.newaxis]."
        )
        raise ValueError(msg)
    for start in range(0, len(multiunit), _MULTIUNIT_CHUNK):
        chunk = multiunit[start : start + _MULTIUNIT_CHUNK]
        finite = chunk[np.isfinite(chunk)]
        if np.any(finite < 0) or np.any(finite != np.round(finite)):
            msg = (
                f"{what}: spike counts or indicators, non-negative whole numbers, but "
                "the array holds other values. Pass counts per sample, not a rate."
            )
            raise ValueError(msg)


_MULTIUNIT_CHUNK = 65_536


def _validate_duration_limits(
    minimum_duration: float,
    maximum_duration: float | None,
    names: tuple[str, str] = ("minimum_duration", "maximum_duration"),
) -> None:
    """Reject duration limits that are not durations, are milliseconds given
    as seconds, or leave no admissible event."""
    minimum_name, maximum_name = names
    _check_finite_non_negative(**{minimum_name: minimum_duration})
    _check_seconds(
        MAXIMUM_PLAUSIBLE_MINIMUM,
        "is longer than any ripple or burst",
        **{minimum_name: minimum_duration},
    )
    if maximum_duration is None:
        return
    if not maximum_duration > 0:
        msg = f"{maximum_name} must be positive or None, got {maximum_duration}."
        raise ValueError(msg)
    if np.isfinite(maximum_duration) and maximum_duration > MAXIMUM_PLAUSIBLE_CEILING:
        _check_seconds(
            MAXIMUM_PLAUSIBLE_CEILING,
            "is longer than any published ceiling",
            **{maximum_name: maximum_duration},
        )
    if maximum_duration < minimum_duration:
        msg = (
            f"{maximum_name} ({maximum_duration}) is below {minimum_name} "
            f"({minimum_duration}); no event could satisfy both. Both are in seconds."
        )
        raise ValueError(msg)

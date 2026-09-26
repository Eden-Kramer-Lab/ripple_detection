"""Input checks shared by the detectors: shapes, lengths, units, duration limits."""

import numpy as np
from numpy.typing import ArrayLike

from ripple_detection.core import (
    FloatArray,
    _check_non_negative,
    _check_number,
    _check_sampling_interval,
    _repeated_timestamps_hint,
    _warn_at_caller,
)


def _validate_lfp_dimensions(filtered_lfps: FloatArray) -> None:
    """Validate that LFP array is 2-D with shape (n_time, n_channels).

    Parameters
    ----------
    filtered_lfps : ndarray
        LFP array to validate.

    Raises
    ------
    ValueError
        If array is not 2-D with appropriate shape.

    """
    if filtered_lfps.ndim != 2:
        hint = (
            "\nIf you have a single channel, reshape your data using:\n"
            "  filtered_lfps = filtered_lfps.reshape(-1, 1)"
            if filtered_lfps.ndim == 1
            else ""
        )
        msg = (
            "filtered_lfps must be a 2-D array with shape (n_time, n_channels).\n"
            f"Received a {filtered_lfps.ndim}-D array with shape {filtered_lfps.shape}.{hint}"
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
        if n_lfp_samples != n_time_samples and filtered_lfps.shape[1] == n_time_samples:
            msg += (
                f"\nThe signal has shape {filtered_lfps.shape}, which looks transposed: "
                "transpose it to (n_time, n_channels), time down the rows (pass .T)."
            )
        if n_speed_samples != n_time_samples and n_lfp_samples == n_time_samples:
            msg += (
                "\nIf speed was sampled on its own clock (position tracking at 30 Hz, say), "
                "interpolate it onto time: speed = np.interp(time, speed_time, speed)."
            )
        raise ValueError(msg)


def _validate_time_units(time: FloatArray, sampling_frequency: float) -> None:
    """Validate that time array is in seconds (not samples).

    Parameters
    ----------
    time : ndarray
        Time array to validate.
    sampling_frequency : float
        Expected sampling frequency in Hz.

    Raises
    ------
    ValueError
        If time holds NaN or infinity, is not increasing, has a median step
        that is not positive (most timestamps repeat), or has a median step
        more than 10 percent away from ``1 / sampling_frequency``; the
        message names time in samples, time in milliseconds, or the rate the
        timestamps imply (``core._check_sampling_interval``).

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
        median_dt = float(np.median(steps))
        if not median_dt > 0:
            msg = (
                f"The median time step is {median_dt}: most timestamps repeat, so no "
                "duration can be measured in samples. Check the time array."
            ) + _repeated_timestamps_hint(time)
            raise ValueError(msg)
        _check_sampling_interval(median_dt, sampling_frequency)


def _validate_speed_units(speed: FloatArray, speed_threshold: float) -> None:
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
    if moving.size == 0 or speed_threshold <= 1.0 or np.isposinf(speed_threshold):
        return  # a threshold at or below 1 is not in cm/s; infinity ignores speed
    median_speed = np.median(moving)
    # a median under 0.5 against a typical threshold (> 1 cm/s) is m/s
    if median_speed < 0.5:
        _warn_at_caller(
            f"Speed values appear very small (median non-zero: {median_speed:.4f}).\n"
            f"Speed should be in cm/s, not m/s.\n"
            f"If your speed is in m/s, multiply by 100:\n"
            "  speed_cms = speed_ms * 100",
        )


UNFILTERED_CUTOFF = 100.0
"""Hz. A ripple-band signal, at any published lower edge (80 Hz and up),
has little power below this; raw LFP has most of its power there."""

_UNFILTERED_POWER_FRACTION = 0.5
"""Share of power below ``UNFILTERED_CUTOFF`` above which a channel is taken
for unfiltered. Measured on simulated sessions at 1000 to 30000 Hz: raw pink
noise holds 0.65-0.86 of its power there, with theta and delta 0.87-0.94, ADC
counts with an offset 0.996; the 150-250 Hz filter's output under 0.001, an
80-250 Hz band's about 0.25, and white noise at 1500 Hz 0.13."""

_SPECTRUM_BUDGET = 2**20
"""Most samples, over every channel, the unfiltered-input check reads."""

_SPECTRUM_SEGMENTS = 32
"""Most stretches of the recording the unfiltered-input check reads."""


def _warn_if_not_ripple_band(
    filtered_lfps: FloatArray, sampling_frequency: float, name: str = "filtered_lfps"
) -> None:
    """Warn when a signal meant to be ripple-band LFP has most of its power
    below ``UNFILTERED_CUTOFF``, as raw LFP and ADC counts do.

    Raw LFP passes every shape check and gives events that follow the slow
    waves rather than the ripples. The spectrum is estimated on at most
    ``_SPECTRUM_SEGMENTS`` stretches of finite rows, spread over the
    recording, each about a quarter of a second long, so the check stays
    cheap on hours of data. No DC is removed: an offset is low-frequency
    power, and a filtered signal has none.

    Parameters
    ----------
    filtered_lfps : ndarray, shape (n_time, n_channels)
    sampling_frequency : float
        In Hz. A rate whose Nyquist frequency is under twice the cutoff
        cannot hold a ripple band, so it is not judged.
    name : str, optional
        The argument's name, for the message.

    Warns
    -----
    UserWarning
        If any channel has more than ``_UNFILTERED_POWER_FRACTION`` of its
        power below ``UNFILTERED_CUTOFF``.

    """
    if sampling_frequency < 4 * UNFILTERED_CUTOFF:
        return
    n_time, n_channels = filtered_lfps.shape
    # a power of two about a quarter second long resolves 4 Hz or finer
    length = min(n_time, int(2 ** np.ceil(np.log2(sampling_frequency / 4))))
    if length < 64 or n_channels == 0:
        return
    missing_before = np.concatenate(
        [[0], np.cumsum(~np.all(np.isfinite(filtered_lfps), axis=1))]
    )
    starts = np.flatnonzero(missing_before[length:] == missing_before[:-length])
    if not starts.size:
        return
    n_segments = int(np.clip(_SPECTRUM_BUDGET // (length * n_channels), 1, _SPECTRUM_SEGMENTS))
    chosen = np.unique(starts[np.linspace(0, starts.size - 1, n_segments).round().astype(int)])
    segments = filtered_lfps[chosen[:, np.newaxis] + np.arange(length)]
    tapered = segments * np.hanning(length)[:, np.newaxis]
    power = (np.abs(np.fft.rfft(tapered, axis=1)) ** 2).sum(axis=0)
    frequencies = np.fft.rfftfreq(length, 1 / sampling_frequency)
    total = power.sum(axis=0)
    low = power[frequencies < UNFILTERED_CUTOFF].sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = low / total  # a silent channel is 0/0, NaN, and never flagged
    flagged = np.flatnonzero(share > _UNFILTERED_POWER_FRACTION)
    if flagged.size:
        shown = ", ".join(str(int(channel)) for channel in flagged[:5])
        _warn_at_caller(
            f"{name} does not look filtered to the ripple band: "
            f"{100 * np.max(share[flagged]):.0f}% of the power of channel(s) "
            f"[{shown}{', ...' if flagged.size > 5 else ''}] lies below "
            f"{UNFILTERED_CUTOFF:g} Hz, where a ripple-band signal has almost none. On raw "
            "LFP or ADC counts the events follow the slow waves, not the ripples. Filter "
            f"first: {name} = filter_ripple_band(lfps, sampling_frequency)."
        )


def _validate_detector_inputs(
    time: ArrayLike,
    signal: ArrayLike,
    speed: ArrayLike,
    sampling_frequency: float,
    speed_threshold: float,
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
    _validate_time_units(time, sampling_frequency)
    _validate_speed_units(speed, speed_threshold)
    return time, signal, speed


def _check_finite_non_negative(**values: float) -> None:
    """Raise for a value that is NaN, infinite or negative."""
    for name, value in values.items():
        if not 0 <= value < np.inf:
            msg = f"{name} must be finite and non-negative, got {value}."
            raise ValueError(msg)


def _check_positive(**values: float) -> None:
    """Raise for a value that is not a positive finite number: ``TypeError``
    for one that is no number at all, such as None."""
    _check_number(**values)
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
    """A smoothing width in seconds, a Gaussian's standard deviation or a
    moving average's length: positive, finite, and under a second, since a
    longer kernel smooths every ripple away and usually means milliseconds
    were given."""
    _check_positive(**values)
    _check_seconds(MAXIMUM_PLAUSIBLE_MINIMUM, "would smooth every ripple away", **values)


def _check_gap(**values: float) -> None:
    """A gap or interval between events in seconds: finite, non-negative, and
    not beyond ``MAXIMUM_PLAUSIBLE_CEILING``. Infinity is rejected: every
    finite spacing falls below it, so it would keep or merge into one event."""
    _check_finite_non_negative(**values)
    for name, value in values.items():
        if value > MAXIMUM_PLAUSIBLE_CEILING:
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
            f"{what} must be a 2-D array of shape (n_time, n_units), got shape "
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
    if not 0 < maximum_duration < np.inf:
        msg = (
            f"{maximum_name} must be positive and finite, or None for no ceiling; "
            f"got {maximum_duration}."
        )
        raise ValueError(msg)
    if maximum_duration > MAXIMUM_PLAUSIBLE_CEILING:
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

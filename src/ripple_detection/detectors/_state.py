"""Brain and behavioural state: a theta/delta ratio and the intervals of a state."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import butter, oaconvolve, sosfiltfilt

from ripple_detection._call_hints import explain_call_errors
from ripple_detection.core import (
    FloatArray,
    _check_choice,
    get_envelope,
    merge_close_events,
)
from ripple_detection.detectors._blocks import _contiguous_valid_blocks, _drop_short_blocks
from ripple_detection.detectors._validation import _check_band, _check_positive

STATE_COMPARISONS = ("<", "<=", ">", ">=")
"""How :func:`state_intervals` compares values with its threshold."""

RATIO_MEASURES = ("amplitude", "power")
"""What :func:`theta_delta_ratio` divides: band envelopes, or their squares."""


def _smooth_slow(values: FloatArray, sigma: float, fs: float) -> FloatArray:
    """A unit-area Gaussian (standard deviation ``sigma`` seconds, +/- 6 SD)
    applied by FFT convolution with zero padding. The kernel of a smoothing of
    seconds has thousands of taps, too many to convolve directly over hours;
    the zero padding lowers both envelopes alike near an edge, which the
    ratio cancels."""
    sigma_samples = sigma * fs
    half = int(np.ceil(6 * sigma_samples))
    offsets = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (offsets / sigma_samples) ** 2)
    smoothed: FloatArray = oaconvolve(values, kernel / kernel.sum(), mode="same")
    return smoothed


def _band_envelope(
    lfp: FloatArray, runs: list[tuple[int, int]], band: tuple[float, float], fs: float
) -> FloatArray:
    """Hilbert envelope of the band-passed LFP, filtered over each run on its own."""
    sos = butter(2, np.asarray(band) / (0.5 * fs), btype="bandpass", output="sos")
    envelope = np.full(lfp.size, np.nan)
    for start, stop in runs:
        envelope[start:stop] = get_envelope(sosfiltfilt(sos, lfp[start:stop]))
    return envelope


@explain_call_errors
def theta_delta_ratio(
    lfp: ArrayLike,
    sampling_frequency: float,
    *,
    time: ArrayLike | None = None,
    theta_band: tuple[float, float] = (6.0, 12.0),
    delta_band: tuple[float, float] = (1.0, 4.0),
    smoothing_sigma: float | None = 1.0,
    measure: str = "amplitude",
) -> FloatArray:
    """The ratio of theta to delta band activity at each sample.

    Many detectors restrict events to non-theta states: slow-wave sleep or
    quiet rest, where theta is low relative to delta. This gives the ratio
    those rules threshold. Each band is band-passed (a Butterworth filter of
    total order 4, applied forward and backward), its Hilbert envelope taken
    and, by default, smoothed; the ratio is theta over delta. With
    ``state_intervals`` it gives the periods of low theta:
    ``state_intervals(ratio, time, 2.0)``, or with the threshold from
    ``two_cluster_threshold(ratio)`` for a k-means split.

    Published definitions differ in the bands, the power estimate (Hilbert
    envelope, spectrogram, multitaper), the smoothing and the threshold, and
    rarely report all four; this computes the Hilbert version, so a
    comparison across papers is of the rule's shape, not its exact numbers.

    Parameters
    ----------
    lfp : array_like, shape (n_time,) or (n_time, 1)
        Raw LFP from one channel, such as the pyramidal layer or a channel
        with strong theta. NaN marks a missing sample.
    sampling_frequency : float
        Sampling rate in Hz.
    time : array_like, shape (n_time,), optional
        Sample timestamps in seconds. When given, a step larger than 1.5
        times the median step splits the recording as a missing sample does.
    theta_band, delta_band : tuple of (float, float), optional
        Pass-bands in Hz. Defaults (6, 12) and (1, 4).
    smoothing_sigma : float, optional
        Standard deviation in **seconds** of a Gaussian applied to each
        band's envelope before dividing, within each run of finite samples.
        Default 1.0; published smoothing runs from about 1 to 10 s. None for
        no smoothing. Near a run's edge both envelopes are averaged over the
        part of the kernel inside the run, which the ratio makes no
        difference to.
    measure : {'amplitude', 'power'}, optional
        Divide the envelopes (default), or their squares, the ratio of band
        powers. The power ratio is the amplitude ratio squared, so a
        threshold of 2 on one is 4 on the other.

    Returns
    -------
    ratio : ndarray, shape (n_time,)
        Theta over delta; NaN at missing samples, in a run too short to
        filter, and where the delta envelope is zero.

    Raises
    ------
    ValueError
        If a band is not ``0 < low < high < Nyquist``, `smoothing_sigma` is
        not positive and finite, `measure` is not one of the two, the LFP has
        more than one channel, or no run of finite samples is long enough to
        filter.

    Warns
    -----
    UserWarning
        When runs of finite samples too short to filter are treated as
        missing.

    Examples
    --------
    >>> fs = 500
    >>> t = np.arange(20 * fs) / fs
    >>> lfp = 3 * np.sin(2 * np.pi * 8 * t) + np.sin(2 * np.pi * 2 * t)
    >>> ratio = theta_delta_ratio(lfp, fs)
    >>> round(float(np.median(ratio)), 1)
    3.0

    """
    _check_positive(sampling_frequency=sampling_frequency)
    _check_band("theta_band", theta_band, sampling_frequency)
    _check_band("delta_band", delta_band, sampling_frequency)
    if smoothing_sigma is not None:
        _check_positive(smoothing_sigma=smoothing_sigma)
    _check_choice("measure", measure, RATIO_MEASURES)
    values = np.asarray(lfp, dtype=float)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        msg = f"lfp must be one channel, shape (n_time,); got shape {values.shape}."
        raise ValueError(msg)
    timestamps = None if time is None else np.asarray(time, dtype=float)
    if timestamps is not None and timestamps.shape != values.shape:
        msg = f"time has shape {timestamps.shape} and lfp {values.shape}; they must match."
        raise ValueError(msg)

    finite = np.isfinite(values)
    if not np.any(finite):
        msg = "lfp holds no finite sample."
        raise ValueError(msg)
    padlen = 3 * (2 * 2 + 1)  # sosfiltfilt's default pad for two second-order sections
    runs = _drop_short_blocks(
        _contiguous_valid_blocks(finite, timestamps), finite, padlen + 1, "the band filters"
    )
    theta = _band_envelope(values, runs, theta_band, sampling_frequency)
    delta = _band_envelope(values, runs, delta_band, sampling_frequency)
    if smoothing_sigma is not None:
        for start, stop in runs:
            theta[start:stop] = _smooth_slow(
                theta[start:stop], smoothing_sigma, sampling_frequency
            )
            delta[start:stop] = _smooth_slow(
                delta[start:stop], smoothing_sigma, sampling_frequency
            )
    if measure == "power":
        theta, delta = theta**2, delta**2
    ratio = np.full(values.size, np.nan)
    np.divide(theta, delta, out=ratio, where=np.isfinite(delta) & (delta > 0))
    return ratio


def state_intervals(
    values: ArrayLike,
    time: ArrayLike,
    threshold: float,
    *,
    comparison: str = "<",
    minimum_duration: float = 0.0,
    merge_gap: float = 0.0,
) -> FloatArray:
    """The intervals during which a signal is below (or above) a threshold.

    For state rules such as "theta/delta ratio below 2", "speed below 1 cm/s
    for at least 5 minutes" or "non-REM epochs longer than 120 s, gaps under
    1 s bridged". The intervals feed :func:`require_overlap` or
    :func:`exclude_overlap` to keep or drop events, or a detector's
    ``normalization_mask`` through ``(time >= start) & (time <= end)``.

    Parameters
    ----------
    values : array_like, shape (n_time,)
        The state signal, such as ``theta_delta_ratio`` or speed. NaN is
        unknown and is never in the state.
    time : array_like, shape (n_time,)
        Sample timestamps in seconds, increasing. A step larger than 1.5
        times the median step ends an interval, as a missing sample does.
    threshold : float
        The level compared with.
    comparison : {'<', '<=', '>', '>='}, optional
        A sample is in the state when ``values comparison threshold``.
        Default ``'<'``.
    minimum_duration : float, optional
        Intervals shorter than this, from first to last sample in seconds,
        are dropped, after merging. Default 0.0.
    merge_gap : float, optional
        Intervals separated by less than this many seconds are joined first.
        Default 0.0, no merging.

    Returns
    -------
    intervals : ndarray, shape (n_intervals, 2)
        ``[start_time, end_time]`` per interval, first and last sample in the
        state, sorted.

    Raises
    ------
    ValueError
        If `values` and `time` differ in shape, `threshold` is not finite,
        `comparison` is not one of the four, or a duration is negative or not
        finite.

    Examples
    --------
    >>> time = np.arange(10.0)
    >>> ratio = np.array([3, 1, 1, 1, 3, 1, 3, 1, 1, 1.0])
    >>> state_intervals(ratio, time, 2.0)
    array([[1., 3.],
           [5., 5.],
           [7., 9.]])
    >>> state_intervals(ratio, time, 2.0, merge_gap=2.5, minimum_duration=3.0)
    array([[1., 9.]])

    """
    signal = np.asarray(values, dtype=float)
    time = np.asarray(time, dtype=float)
    if signal.shape != time.shape or signal.ndim != 1:
        msg = f"values has shape {signal.shape} and time {time.shape}; both must be (n_time,)."
        raise ValueError(msg)
    if not np.isfinite(threshold):
        msg = f"threshold must be finite, got {threshold}."
        raise ValueError(msg)
    _check_choice("comparison", comparison, STATE_COMPARISONS)
    for name, value in (("minimum_duration", minimum_duration), ("merge_gap", merge_gap)):
        if not 0 <= value < np.inf:
            msg = f"{name} must be finite and non-negative, got {value}."
            raise ValueError(msg)
    with np.errstate(invalid="ignore"):
        is_in_state = {
            "<": signal < threshold,
            "<=": signal <= threshold,
            ">": signal > threshold,
            ">=": signal >= threshold,
        }[comparison] & np.isfinite(signal)
    if not np.any(is_in_state):
        return np.empty((0, 2))
    runs = _contiguous_valid_blocks(is_in_state, time)
    intervals: FloatArray = np.array(
        [(time[start], time[stop - 1]) for start, stop in runs], dtype=float
    )
    if merge_gap > 0:
        intervals = merge_close_events(intervals, merge_gap)
    lengths = intervals[:, 1] - intervals[:, 0]
    long_enough = (lengths >= minimum_duration) | np.isclose(lengths, minimum_duration)
    return intervals[long_enough]

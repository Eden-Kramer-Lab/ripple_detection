"""Finding sharp-wave ripple events (150-250 Hz) from local field
potentials.
"""

import functools
import os
import sys
import warnings
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.fftpack import next_fast_len
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert, oaconvolve, remez
from scipy.stats import median_abs_deviation

from ripple_detection._call_hints import NORMALIZE_SIGNAL_WITHOUT_TIME, explain_call_errors

FloatArray = NDArray[np.floating]
"""A NumPy array of floats, the shape stated in each docstring."""

BoolArray = NDArray[np.bool_]
"""A NumPy array of booleans."""

IntArray = NDArray[np.integer]
"""A NumPy array of integers, usually sample indices."""

DEFAULT_RIPPLE_BAND = (150.0, 250.0)
"""Default passband in Hz, the most common choice in the replay literature."""

DEFAULT_TRANSITION_WIDTH = 25.0
"""Default width in Hz of the transition on each side of the passband."""


def ripple_bandpass_filter(
    sampling_frequency: float,
    band: tuple[float, float] | None = None,
    transition_width: float = DEFAULT_TRANSITION_WIDTH,
) -> tuple[FloatArray, float]:
    """Generate a bandpass filter for a ripple frequency band.

    Uses the Remez exchange algorithm to design a finite impulse response (FIR)
    filter. The band defaults to 150-250 Hz with 25 Hz transition bands; both
    are parameters, since published ripple bands vary.

    Parameters
    ----------
    sampling_frequency : float
        Sampling rate of the signal in Hz.
    band : tuple of (float, float), optional
        Passband edges in Hz. Default is None, the 150-250 Hz band
        (``DEFAULT_RIPPLE_BAND``), the most common choice. Published bands
        vary, with lower edges from about 80 to 180 Hz and upper edges from
        200 to 300 Hz.
    transition_width : float, optional
        Width in Hz of the transition on each side of the passband. Default is
        25.0. A narrower transition needs more taps, and therefore a longer
        signal to filter.

    Returns
    -------
    filter_numerator : ndarray, shape (n_taps,)
        Numerator coefficients of the filter. The tap count scales with
        ``sampling_frequency`` so the design holds its specification at every
        rate: at least 101 taps, 155 at 1500 Hz, 3093 at 30 kHz.
    filter_denominator : float
        Denominator coefficient (always 1.0 for FIR filters).

    Notes
    -----
    A 150-250 Hz equiripple FIR with 25 Hz transition bands, designed for
    about 45 dB of stopband attenuation. The measured passband and stopband
    error is about 0.004 (48 dB) at 1500-2000 Hz, rising to 0.009 (41 dB) at
    30 kHz, and smaller at low rates, where the 101-tap minimum dominates.

    At 25 kHz and above the design needs 2500 taps or more, where the
    exchange algorithm no longer reaches its specification: the single-pass
    attenuation settles at about 41 dB (82 dB after the forward-backward
    pass, so still ample), the kernel is about 0.1 s long, and designing it
    takes a few tenths of a second. Decimating such data to 3 kHz or below
    before filtering loses nothing below 300 Hz and gives the specified
    design at a tenth of the cost.

    Examples
    --------
    >>> kernel, denominator = ripple_bandpass_filter(1500)
    >>> len(kernel), denominator
    (155, 1.0)
    >>> len(ripple_bandpass_filter(30_000)[0])  # the tap count grows with the rate
    3093

    """
    STOPBAND_ATTENUATION_DB = 45.0
    MINIMUM_NUMTAPS = 101

    low, high = (float(edge) for edge in (DEFAULT_RIPPLE_BAND if band is None else band))
    nyquist = 0.5 * sampling_frequency
    if transition_width <= 0:
        msg = f"transition_width must be positive, got {transition_width} Hz."
        raise ValueError(msg)
    if low >= high:
        msg = f"band must be (low, high) with low < high, got {band} Hz."
        raise ValueError(msg)
    if low - transition_width <= 0:
        msg = (
            f"band lower edge {low} Hz leaves no room for a {transition_width} Hz "
            "transition above 0 Hz. Raise the edge or narrow the transition."
        )
        raise ValueError(msg)
    if high + transition_width >= nyquist:
        msg = (
            f"band upper edge {high} Hz plus a {transition_width} Hz transition reaches "
            f"the Nyquist frequency {nyquist} Hz of a {sampling_frequency} Hz signal."
        )
        raise ValueError(msg)

    # Kaiser's estimate: the tap count needed for a given attenuation grows as
    # the transition band narrows relative to the sampling rate. A fixed count
    # would meet the specification at one rate only.
    transition = 2.0 * np.pi * transition_width / sampling_frequency
    numtaps = int(np.ceil((STOPBAND_ATTENUATION_DB - 8.0) / (2.285 * transition)))
    numtaps = max(MINIMUM_NUMTAPS, numtaps + 1 - numtaps % 2)
    desired = [
        0,
        low - transition_width,
        low,
        high,
        high + transition_width,
        nyquist,
    ]
    return _remez_bandpass(numtaps, tuple(desired), float(sampling_frequency)).copy(), 1.0


@functools.lru_cache(maxsize=16)
def _remez_bandpass(
    numtaps: int, desired: tuple[float, ...], sampling_frequency: float
) -> FloatArray:
    """``remez`` for a band-pass, cached: at 30 kHz the design takes longer
    than filtering a minute of four channels, and a pipeline filters many
    recordings at one rate. Read-only, since callers share it."""
    kernel = np.asarray(remez(numtaps, list(desired), [0, 1, 0], fs=sampling_frequency))
    kernel.flags.writeable = False
    return kernel


_PACKAGE_DIRECTORY = str(Path(__file__).parent) + os.sep
"""The path the package was imported through, as the code objects record it:
unresolved, so a symlinked install still matches, and ending in a separator,
so a sibling directory with a longer name does not."""


def _warn_at_caller(message: str) -> None:
    """``warnings.warn(message, UserWarning)`` attributed to the first frame
    outside this package: the caller's line, however deep inside the package
    the warning is raised and whatever wraps the public function."""
    frame = sys._getframe(1)
    stacklevel = 2  # the frame that called this function
    while frame.f_back is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIRECTORY):
        frame = frame.f_back
        stacklevel += 1
    warnings.warn(message, UserWarning, stacklevel=stacklevel)


def _generator(seed: int | np.random.Generator | None) -> np.random.Generator:
    """``numpy.random.default_rng(seed)``, refusing the legacy ``RandomState``
    that 1.x's noise functions took: ``default_rng`` accepts one and silently
    draws a different stream from it."""
    given: object = seed  # a caller without a type checker can pass anything
    if isinstance(given, np.random.RandomState):
        msg = (
            "Pass a seed or a numpy.random.Generator, not a RandomState: every random "
            "draw goes through numpy.random.default_rng, which would accept a RandomState "
            "and silently draw a different stream from it."
        )
        raise TypeError(msg)
    return np.random.default_rng(seed)


def minimum_sample_count(time: ArrayLike, minimum_duration: float) -> int:
    """Number of consecutive samples that ``minimum_duration`` spans.

    ``round(minimum_duration * sampling_frequency)`` with round-half-up, the
    convention of the Frank lab ``extractevents`` routine
    (``DFFunctions/extractevents.cpp`` in
    https://github.com/droumis/FFPhy/tree/fce2048/DFFunctions), where the sampling
    interval is the median timestamp step. A duration that is not a whole
    number of samples rounds to the nearest count (22.5 samples -> 23).

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Sample timestamps in seconds. Fewer than two samples give a count
        of 1.
    minimum_duration : float
        Duration in seconds.

    Returns
    -------
    n_samples : int
        At least 1.

    Raises
    ------
    ValueError
        If the median timestamp step is not positive and finite, as when
        most timestamps repeat. A count of 1 would then disable the duration
        criterion.

    Examples
    --------
    >>> import numpy as np
    >>> time = np.arange(1500) / 1500  # 1 s at 1500 Hz
    >>> minimum_sample_count(time, 0.015)  # 22.5 samples rounds half up
    23

    """
    time = np.asarray(time, dtype=float)
    if time.size < 2:
        return 1
    sample_interval = np.median(np.diff(time))
    if not np.isfinite(sample_interval) or sample_interval <= 0:
        msg = (
            f"The median timestamp step is {sample_interval}, so no duration can be "
            "converted to a sample count. Check that time is increasing and in seconds."
        )
        raise ValueError(msg)
    # small tolerance so an exact half-sample product is not lost to round-off
    return max(1, int(np.floor(minimum_duration / sample_interval + 0.5 + 1e-6)))


def sample_count_within(
    n_samples: ArrayLike,
    time: ArrayLike,
    minimum_duration: float,
    maximum_duration: float | None = None,
) -> BoolArray | bool:
    """Whether an event of ``n_samples`` samples meets the package's duration limits.

    Every detector applies this one rule. ``minimum_sample_count`` turns a
    limit in seconds into a sample count, rounding half up from the median
    timestamp step. An event qualifies when its sample count is at least the
    minimum and, where one is given, at most the maximum. Both comparisons are
    inclusive.

    Parameters
    ----------
    n_samples : int or array_like of int
        Number of samples the event spans, first to last inclusive.
    time : array_like, shape (n_time,)
        Sample timestamps in seconds.
    minimum_duration : float
        Shortest allowed duration in seconds.
    maximum_duration : float, optional
        Longest allowed duration in seconds. Default is None (no upper limit).

    Returns
    -------
    qualifies : bool or ndarray of bool
        Same shape as ``n_samples``; a Python bool for a scalar input.

    Examples
    --------
    >>> import numpy as np
    >>> time = np.arange(1500) / 1500
    >>> sample_count_within([22, 23, 24], time, 0.015).tolist()
    [False, True, True]

    """
    counts = np.asarray(n_samples)
    ok = counts >= minimum_sample_count(time, minimum_duration)
    if maximum_duration is not None:
        ok &= counts <= minimum_sample_count(time, maximum_duration)
    return bool(ok) if counts.ndim == 0 else ok


def _boolean_run_bounds(values: ArrayLike) -> IntArray:
    """Start (inclusive) and stop (exclusive) positions of each run of True."""
    padded = np.concatenate([[False], np.asarray(values, dtype=bool), [False]])
    return np.flatnonzero(padded[1:] != padded[:-1]).reshape(-1, 2)


def segment_boolean_series(
    series: pd.Series, minimum_duration: float = 0.015
) -> list[tuple[float, float]]:
    """Extract time segments from a boolean pandas Series.

    Returns a list of tuples where each tuple contains the start and end time
    of a segment. Segments are defined by consecutive True values in the input
    series, where the series index represents time.

    A segment qualifies when it holds at least
    ``minimum_sample_count(series.index, minimum_duration)`` consecutive
    samples, i.e. ``round(minimum_duration * sampling_frequency)``. Counting
    samples rather than subtracting timestamps makes the test exact at every
    sampling rate and time offset, and measures a run that straddles a gap in
    the index by the samples it contains, not by the time it spans.

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
        List of (start_time, end_time) tuples, the timestamps of the first and
        last sample of each segment that meets the minimum sample count.

    """
    if series.isna().any():
        msg = (
            "series contains missing values, which cast to True. Fill or drop "
            "them before segmenting."
        )
        raise ValueError(msg)
    values = series.to_numpy(dtype=bool)
    index = np.asarray(series.index)
    n_min = minimum_sample_count(index, minimum_duration)
    bounds = _boolean_run_bounds(values)
    bounds = bounds[(bounds[:, 1] - bounds[:, 0]) >= n_min]
    return [(index[start], index[stop - 1]) for start, stop in bounds]


@explain_call_errors
def filter_ripple_band(
    data: ArrayLike,
    sampling_frequency: float,
    band: tuple[float, float] | None = None,
    transition_width: float | None = None,
) -> FloatArray:
    """Bandpass filter signal(s) to the ripple band, 150-250 Hz by default.

    At 1500 Hz with the default band and no ``transition_width``, the
    pre-computed 318-tap FIR kernel shipped with the package is used, whether
    the band is left as None or given as ``(150, 250)``. At any other rate,
    for any other band, or when a ``transition_width`` is given, an FIR is
    designed for that rate with ``ripple_bandpass_filter``, so the passband is
    the same in hertz regardless of the sampling rate. The filter is applied
    forward and backward (``filtfilt``) for zero phase distortion.

    A row holding NaN or infinity in any channel is missing. Each contiguous run of
    present rows is filtered on its own, so the filter never sees the step
    between the two sides of a gap. Stitching the sides together instead
    produces a transient at the join that a detector then reads as a
    high-power event spanning the gap. A run with fewer samples than the
    kernel has taps cannot be filtered and is returned as NaN with a warning.

    Parameters
    ----------
    data : array_like, shape (n_time,) or (n_time, n_channels)
        Input signal(s) to be filtered. Can be 1-D or 2-D.
    sampling_frequency : float
        Sampling rate of the input data in Hz.
    band : tuple of (float, float), optional
        Passband edges in Hz. Default is None, the 150-250 Hz band; giving
        ``(150, 250)`` is the same as None.
    transition_width : float, optional
        Width in Hz of the transition on each side of the passband. Default is
        None: a designed filter gets 25 Hz, and at 1500 Hz with the default
        band the shipped kernel, whose transitions are 10 Hz, is used. Giving
        a width always designs a filter with that width, so
        ``transition_width=25.0`` at 1500 Hz is a different filter from the
        default (155 taps and 48 dB against the kernel's 318 taps and 40 dB;
        their outputs differ by up to 0.8 SD).

    Returns
    -------
    filtered_data : ndarray, shape (n_time,) or (n_time, n_channels)
        Bandpass filtered signal in the ripple band. Missing rows, and runs
        too short to filter, are NaN.

    Raises
    ------
    ValueError
        If the sampling rate cannot represent the band, that is, the upper
        edge plus the transition band reaches the Nyquist frequency (from
        ``ripple_bandpass_filter``), or if no run of present rows is long
        enough to filter.

    Warns
    -----
    UserWarning
        If some run of present rows is too short to filter and is returned as
        NaN.

    See Also
    --------
    ripple_bandpass_filter : The filter design used for rates other than 1500 Hz.

    Examples
    --------
    >>> import numpy as np
    >>> from ripple_detection import filter_ripple_band
    >>> lfp = np.random.randn(3000)
    >>> filtered = filter_ripple_band(lfp, sampling_frequency=1500)

    """
    SHIPPED_KERNEL_SAMPLING_FREQUENCY = 1500.0

    default_band = band is None or tuple(float(edge) for edge in band) == DEFAULT_RIPPLE_BAND
    if (
        default_band
        and transition_width is None
        and np.isclose(sampling_frequency, SHIPPED_KERNEL_SAMPLING_FREQUENCY)
    ):
        filter_numerator, filter_denominator = _get_ripplefilter_kernel()
    else:
        filter_numerator, filter_denominator = ripple_bandpass_filter(
            sampling_frequency,
            band=band,
            transition_width=(
                DEFAULT_TRANSITION_WIDTH if transition_width is None else transition_width
            ),
        )

    data_array = np.asarray(data, dtype=float)
    # NaN and infinity alike are missing: one inf inside a run would spread
    # through the whole run under the filter
    finite = np.isfinite(data_array)
    is_present = finite.all(axis=-1) if data_array.ndim > 1 else finite
    # filtfilt reflects padlen samples past each end and needs more samples than
    # that. Its default of 3 x the taps suits IIR filters; an FIR remembers only
    # taps - 1 samples, so that pad length gives the same output and a run
    # needs only as many samples as the kernel.
    padlen = len(filter_numerator) - 1
    min_required_length = len(filter_numerator)
    runs = _boolean_run_bounds(is_present)
    long_enough = (runs[:, 1] - runs[:, 0]) >= min_required_length
    if not np.any(long_enough):
        longest = int((runs[:, 1] - runs[:, 0]).max()) if len(runs) else 0
        msg = (
            f"Signal too short for filtering: the longest run of finite samples holds "
            f"{longest}, but at least {min_required_length} are needed (the filter's tap "
            "count)."
        )
        raise ValueError(msg)
    if not np.all(long_enough):
        short = runs[~long_enough]
        _warn_at_caller(
            f"{len(short)} run(s) of finite samples shorter than the {min_required_length} "
            "samples the filter needs are returned as NaN (sample ranges "
            f"{[(int(a), int(b)) for a, b in short[:5]]}{', ...' if len(short) > 5 else ''}). "
            "Interpolate short gaps before filtering if those samples matter."
        )

    filtered_data = np.full_like(data_array, np.nan)
    kernel = np.asarray(filter_numerator, dtype=float) / filter_denominator
    for start, stop in runs[long_enough]:
        filtered_data[start:stop] = _fir_filtfilt(kernel, data_array[start:stop], padlen)
    return filtered_data


def _fir_filtfilt(kernel: FloatArray, data: FloatArray, padlen: int) -> FloatArray:
    """``scipy.signal.filtfilt(kernel, 1.0, data, axis=0, padlen=padlen)`` for
    an FIR kernel, by FFT convolution.

    filtfilt extends each end by an odd reflection of ``padlen`` samples and
    starts each pass at the steady state for the first value, which for an
    FIR is the same as prepending ``len(kernel) - 1`` copies of it; the two
    passes are then plain convolutions. filtfilt runs them directly, which
    costs the tap count per sample: 3093 taps at 30 kHz. Overlap-add FFT
    convolution gives the same output to rounding (1e-15 of the signal;
    tested against filtfilt) twelve times faster at 30 kHz and three at
    1500 Hz.

    Parameters
    ----------
    kernel : ndarray, shape (n_taps,)
    data : ndarray, shape (n_time,) or (n_time, n_channels)
        At least ``padlen + 1`` samples.
    padlen : int

    Returns
    -------
    filtered : ndarray, same shape as ``data``

    """
    first, last = data[:1], data[-1:]
    extended = np.concatenate(
        [2 * first - data[padlen:0:-1], data, 2 * last - data[-2 : -(padlen + 2) : -1]]
    )
    taps = kernel.reshape((-1,) + (1,) * (data.ndim - 1))
    n_state = len(kernel) - 1

    def one_pass(signal: FloatArray) -> FloatArray:
        primed = np.concatenate([np.repeat(signal[:1], n_state, axis=0), signal])
        return np.asarray(oaconvolve(primed, taps, mode="valid", axes=0))

    forward = one_pass(extended)
    both = one_pass(forward[::-1])[::-1]
    return np.asarray(both[padlen : len(both) - padlen])


def _get_ripplefilter_kernel() -> tuple[FloatArray, float]:
    """Load the pre-computed ripple filter kernel from the Frank lab.

    The kernel is a 150-250 Hz bandpass filter with 40 dB roll-off and 10 Hz
    transition sidebands, designed for signals sampled at 1500 Hz.

    Returns
    -------
    filter_numerator : ndarray
        Filter kernel coefficients.
    filter_denominator : float
        Denominator coefficient (always 1.0 for FIR filters).

    """
    return _load_ripplefilter_kernel().copy(), 1.0


@functools.lru_cache(maxsize=1)
def _load_ripplefilter_kernel() -> FloatArray:
    """The shipped kernel, read from disk once. Read-only, since callers share it."""
    filter_file = Path(__file__).resolve().parent / "ripplefilter.mat"
    kernel = np.asarray(
        loadmat(str(filter_file))["ripplefilter"]["kernel"][0][0].flatten(), dtype=float
    )
    kernel.flags.writeable = False
    return kernel


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
    time = np.asarray(time)
    bounds, _ = _runs_extended_to_mean(
        np.asarray(is_above_mean, dtype=bool),
        np.asarray(is_above_threshold, dtype=bool),
        minimum_sample_count(time, minimum_duration),
    )
    return [(time[start], time[stop - 1]) for start, stop in bounds]


def _runs_extended_to_mean(
    is_above_mean: BoolArray, is_above_threshold: BoolArray, n_min: int
) -> tuple[IntArray, IntArray]:
    """The above-mean runs that contain an above-threshold run of ``n_min``
    samples or more.

    Parameters
    ----------
    is_above_mean, is_above_threshold : ndarray of bool, shape (n_time,)
    n_min : int
        Fewest consecutive above-threshold samples that make a candidate.

    Returns
    -------
    bounds : ndarray of int, shape (n_events, 2)
        Half-open ``[start, stop)`` sample bounds of each containing above-mean
        run, in order, once however many candidates it holds.
    longest : ndarray of int, shape (n_events,)
        Sample count of the longest candidate inside each.

    Raises
    ------
    ValueError
        If a candidate is not inside an above-mean run, which the masks of a
        non-negative threshold on one trace rule out.

    """
    candidates = _boolean_run_bounds(is_above_threshold)
    candidates = candidates[(candidates[:, 1] - candidates[:, 0]) >= n_min]
    if len(candidates) == 0:
        return np.empty((0, 2), dtype=int), np.empty(0, dtype=int)
    above_mean = _boolean_run_bounds(is_above_mean)
    # the above-mean run starting at or before each candidate, by bisection
    containing = np.searchsorted(above_mean[:, 0], candidates[:, 0], side="right") - 1
    if containing[0] < 0:
        msg = (
            f"No candidate interval starts at or before sample {candidates[0, 0]}, so "
            "none can contain the run above threshold."
        )
        raise ValueError(msg)
    outside = candidates[:, 1] > above_mean[containing, 1]
    if np.any(outside):
        run = candidates[np.flatnonzero(outside)[0]]
        msg = f"The run above threshold at samples {tuple(run)} is not inside a run above the mean."
        raise ValueError(msg)
    runs, which = np.unique(containing, return_inverse=True)
    longest = np.zeros(len(runs), dtype=int)
    np.maximum.at(longest, which, candidates[:, 1] - candidates[:, 0])
    return above_mean[runs], longest


def nearest_sample_index(time: ArrayLike, query_times: ArrayLike) -> IntArray:
    """Index of the sample in ``time`` closest to each query time.

    Event bounds come from ``time``, so the match is normally exact. Looking
    the index up per query, rather than testing which samples appear in the
    query set, keeps the result in query order and one entry long per query.
    Repeated, nested, or off-grid query times are therefore handled correctly.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Sample timestamps, increasing.
    query_times : array_like, shape (n_queries,)
        Times to look up.

    Returns
    -------
    index : ndarray, shape (n_queries,)
        Position in ``time`` of the closest sample to each query time.

    Raises
    ------
    ValueError
        If ``time`` is empty.

    """
    time = np.asarray(time, dtype=float)
    query_times = np.asarray(query_times, dtype=float)
    if time.size == 0:
        msg = "time is empty, so no sample can be looked up."
        raise ValueError(msg)
    if time.size == 1:
        return np.zeros(query_times.shape, dtype=int)
    right = np.searchsorted(time, query_times)
    right = np.clip(right, 1, time.size - 1)
    left = right - 1
    closer_to_right = np.abs(time[right] - query_times) < np.abs(query_times - time[left])
    return np.where(closer_to_right, right, left)


def _event_bounds(events: ArrayLike | pd.DataFrame) -> FloatArray:
    """``[start_time, end_time]`` rows from an array or a detector's DataFrame.

    Every helper that takes an event inventory reads it through this, so a
    DataFrame returned by a detector is as valid an input as a bare array,
    and anything else of the wrong shape raises instead of being reshaped
    into nonsense.

    Parameters
    ----------
    events : array_like, shape (n_events, 2), or pd.DataFrame
        A DataFrame needs ``start_time`` and ``end_time`` columns.

    Returns
    -------
    bounds : ndarray, shape (n_events, 2)
        Float; shape ``(0, 2)`` for no events.

    Raises
    ------
    ValueError
        If an array is not ``(n_events, 2)``.

    """
    if isinstance(events, pd.DataFrame):
        bounds = events[["start_time", "end_time"]].to_numpy(dtype=float)
        return np.asarray(bounds, dtype=float).reshape(-1, 2)
    bounds = np.asarray(events, dtype=float)
    if bounds.size == 0:
        return np.empty((0, 2))
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        msg = (
            f"Events must be an array of shape (n_events, 2), [start_time, end_time] per "
            f"row, or a detector's DataFrame; got shape {bounds.shape}."
        )
        raise ValueError(msg)
    return bounds


def _is_immobile(speed: ArrayLike, speed_threshold: float) -> BoolArray:
    """Samples known to be at or below ``speed_threshold``.

    A NaN speed is unknown and so not known to be immobile, unless the
    threshold is infinite, which turns the criterion off.
    """
    speed = np.asarray(speed, dtype=float)
    if np.isposinf(speed_threshold):
        return np.ones(speed.shape, dtype=bool)
    return np.asarray(speed <= speed_threshold, dtype=bool)


def _is_immobile_at_endpoints(
    event_times: FloatArray, speed: ArrayLike, time: ArrayLike, speed_threshold: float
) -> BoolArray:
    """The package's endpoint speed rule: speed at the event's first and last
    sample is at or below ``speed_threshold``; a NaN there fails it (see
    ``_is_immobile``). Returns a bool mask over events."""
    events = _event_bounds(event_times)
    if len(events) == 0:
        return np.zeros(0, dtype=bool)
    immobile = _is_immobile(speed, speed_threshold)
    at_start = immobile[nearest_sample_index(time, events[:, 0])]
    at_end = immobile[nearest_sample_index(time, events[:, 1])]
    return np.asarray(at_start & at_end, dtype=bool)


SPEED_RULES = ("endpoints", "all", "mean", "median")
"""The ways :func:`exclude_movement` can test an event's speed."""


def _samples_within(events: FloatArray, time: ArrayLike) -> tuple[IntArray, IntArray]:
    """Half-open sample ranges ``[first, last)`` of the samples with
    ``start_time <= time <= end_time``, found by bisection.

    Raises
    ------
    ValueError
        If no sample of ``time`` falls within an event.

    """
    time = np.asarray(time, dtype=float)
    first = np.searchsorted(time, events[:, 0], side="left")
    last = np.searchsorted(time, events[:, 1], side="right")
    if np.any(last == first):
        start_time, end_time = events[np.flatnonzero(last == first)[0]]
        msg = (
            f"No speed samples fall within event [{start_time}, {end_time}]; "
            "speed and time do not cover the candidate event."
        )
        raise ValueError(msg)
    return first, last


def _is_immobile_by_rule(
    events: FloatArray,
    speed: ArrayLike,
    time: ArrayLike,
    speed_threshold: float,
    rule: str,
) -> BoolArray:
    """Whether each event passes the speed test ``rule`` (see
    :func:`exclude_movement`); a bool mask over events."""
    if rule not in SPEED_RULES:
        msg = f"rule must be one of {', '.join(map(repr, SPEED_RULES))}; got {rule!r}."
        raise ValueError(msg)
    if rule == "endpoints":
        return _is_immobile_at_endpoints(events, speed, time, speed_threshold)
    if len(events) == 0:
        return np.zeros(0, dtype=bool)
    first, last = _samples_within(events, time)
    if np.isposinf(speed_threshold):
        return np.ones(len(events), dtype=bool)
    speed = np.asarray(speed, dtype=float)
    if rule == "all":
        # a NaN is not known to be at or below the threshold, so it fails too
        not_immobile = np.concatenate([[0], np.cumsum(~_is_immobile(speed, speed_threshold))])
        return np.asarray(not_immobile[last] - not_immobile[first] == 0)
    summarize = np.mean if rule == "mean" else np.median
    keep = np.zeros(len(events), dtype=bool)
    for event, (a, b) in enumerate(zip(first, last, strict=True)):
        known = speed[a:b][np.isfinite(speed[a:b])]
        keep[event] = known.size > 0 and bool(summarize(known) <= speed_threshold)
    return keep


def exclude_movement(
    candidate_ripple_times: ArrayLike | pd.DataFrame,
    speed: ArrayLike,
    time: ArrayLike,
    speed_threshold: float = 4.0,
    rule: str = "endpoints",
) -> FloatArray | pd.DataFrame:
    """Filter out candidate ripples that occur during animal movement.

    By default removes events where the animal's speed at either the start or
    end of the event exceeds the specified threshold; speed inside the event
    is not tested. A NaN speed at either end is unknown, so that event is
    removed too, unless ``speed_threshold`` is ``np.inf``, which keeps every
    event. ``rule`` selects one of the other tests published papers use.

    Parameters
    ----------
    candidate_ripple_times : array_like, shape (n_ripples, 2), or pd.DataFrame
        Candidate event times with columns [start_time, end_time], or a
        detector's DataFrame, which is returned filtered with every column.
    speed : array_like, shape (n_time,)
        Animal's speed at each time point.
    time : array_like, shape (n_time,)
        Time values corresponding to speed measurements.
    speed_threshold : float, optional
        Maximum speed (in same units as `speed`) for event to be retained.
        Default is 4.0 (cm/s).
    rule : {'endpoints', 'all', 'mean', 'median'}, optional
        Which speeds must be at or below `speed_threshold`, over the samples
        with ``start_time <= time <= end_time``:

        - ``'endpoints'`` (default): the first and last sample's, the rule
          every detector here applies. A NaN at either end fails.
        - ``'all'``: every sample's, as in "no speed above 3 cm/s during the
          event". A NaN anywhere fails.
        - ``'mean'``: the mean of the known (not NaN) speeds.
        - ``'median'``: the median of the known speeds, as in "median speed
          below 10 cm/s". :func:`exclude_movement_by_majority` with its
          default of one half is the same test but for how it breaks a tie
          on an even number of samples.

        With ``'mean'`` or ``'median'`` an event with no known speed fails.

    Returns
    -------
    ripple_times : ndarray, shape (n_stationary_ripples, 2), or pd.DataFrame
        The events that pass, in the input's type. Shape ``(0, 2)`` when none
        remain.

    Raises
    ------
    ValueError
        If `rule` is not one of the four, or, for a rule other than
        ``'endpoints'``, no sample of `time` falls within an event.

    Examples
    --------
    >>> time = np.arange(0, 1, 0.1)
    >>> speed = np.array([1, 1, 9, 1, 1, 1, 1, 1, 1, 1.0])
    >>> events = np.array([(0.0, 0.4), (0.5, 0.9)])
    >>> exclude_movement(events, speed, time, rule="endpoints")
    array([[0. , 0.4],
           [0.5, 0.9]])
    >>> exclude_movement(events, speed, time, rule="all")
    array([[0.5, 0.9]])

    """
    events = _event_bounds(candidate_ripple_times)
    keep = _is_immobile_by_rule(events, speed, time, speed_threshold, rule)
    if isinstance(candidate_ripple_times, pd.DataFrame):
        return candidate_ripple_times.iloc[np.flatnonzero(keep)].copy()
    return events[keep]


def _is_immobile_by_majority(
    events: FloatArray,
    speed: ArrayLike,
    time: ArrayLike,
    speed_threshold: float,
    majority_threshold: float,
) -> BoolArray:
    """Whether at least ``majority_threshold`` of each event's samples with a
    known speed are at or below ``speed_threshold``; see
    :func:`exclude_movement_by_majority`."""
    speed = np.asarray(speed, dtype=float)
    # the count of immobile samples is a difference of the cumulative sum at
    # each event's sample bounds
    first, last = _samples_within(events, time)
    is_immobile = _is_immobile(speed, speed_threshold)
    # with the criterion off (an infinite threshold) every sample counts as known
    immobile = np.concatenate([[0], np.cumsum(is_immobile)])
    known = np.concatenate([[0], np.cumsum(np.isfinite(speed) | is_immobile)])
    n_below_threshold = immobile[last] - immobile[first]
    n_known = known[last] - known[first]
    # divide rather than multiply the threshold, so 3 of 10 meets 0.3 exactly
    fraction = np.divide(
        n_below_threshold, n_known, out=np.zeros(len(events)), where=n_known > 0
    )
    return np.asarray((n_known > 0) & (fraction >= majority_threshold))


def exclude_movement_by_majority(
    candidate_ripple_times: ArrayLike | pd.DataFrame,
    speed: ArrayLike,
    time: ArrayLike,
    speed_threshold: float = 4.0,
    majority_threshold: float = 0.5,
) -> FloatArray | pd.DataFrame:
    """Filter out candidate ripples that occur during animal movement.

    Retains an event only if the animal's speed is at or below `speed_threshold`
    for at least `majority_threshold` of the samples within the event whose
    speed is known (not NaN); an event with no known speed is removed, unless
    `speed_threshold` is ``np.inf``, which keeps every event.
    `exclude_movement` instead tests only the event's first and last sample.

    Parameters
    ----------
    candidate_ripple_times : array_like, shape (n_ripples, 2), or pd.DataFrame
        Candidate event times with columns [start_time, end_time], or a
        detector's DataFrame, which is returned filtered with every column.
    speed : array_like, shape (n_time,)
        Animal's speed at each time point.
    time : array_like, shape (n_time,)
        Time values corresponding to speed measurements.
    speed_threshold : float, optional
        Maximum speed (in same units as `speed`) for a sample to count as
        immobile. Default is 4.0 (cm/s).
    majority_threshold : float, optional
        Fraction of within-event samples that must be at or below
        `speed_threshold` for the event to be retained. Default is 0.5.

    Returns
    -------
    ripple_times : ndarray, shape (n_kept, 2), or pd.DataFrame
        The retained events, in the input's type. Shape ``(0, 2)`` when none
        remain.

    Raises
    ------
    ValueError
        If no sample of ``time`` falls within an event.

    """
    events = _event_bounds(candidate_ripple_times)
    keep = _is_immobile_by_majority(events, speed, time, speed_threshold, majority_threshold)
    if isinstance(candidate_ripple_times, pd.DataFrame):
        return candidate_ripple_times.iloc[np.flatnonzero(keep)].copy()
    return events[keep]


def get_envelope(data: ArrayLike, axis: int = 0) -> FloatArray:
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
    data = np.asarray(data, dtype=float)
    n_samples = data.shape[axis]
    instantaneous_amplitude = np.abs(hilbert(data, N=next_fast_len(n_samples), axis=axis))
    return np.take(instantaneous_amplitude, np.arange(n_samples), axis=axis)


def gaussian_smooth(
    data: ArrayLike,
    sigma: float,
    sampling_frequency: float,
    axis: int = 0,
    truncate: float = 8,
) -> FloatArray:
    """Apply 1-D Gaussian smoothing to data.

    Convolves the data with a Gaussian kernel. The standard deviation is
    specified in time units (e.g., seconds) and converted to samples using
    the sampling frequency. This is a wrapper around scipy's `gaussian_filter1d`
    with truncation at 8 standard deviations (instead of 4).

    Near either end of ``axis`` the kernel runs past the data; there it is
    renormalized to unit sum over the samples it covers, so an end is an
    average of the data that exists rather than being pulled toward zero as
    zero padding pulls it. The detectors smooth each block of valid samples as
    its own array, so a trace next to a gap keeps its level, and an event cut
    off by the gap reaches the gap's edge and is flagged there.

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
    truncate : float, optional
        Number of standard deviations at which to truncate the filter.
        Default is 8 (wider support than scipy's default of 4).

    Returns
    -------
    smoothed_data : ndarray
        Gaussian-smoothed data, same shape as input.

    """
    data = np.asarray(data, dtype=float)
    sigma_samples = sigma * sampling_frequency
    smoothed = gaussian_filter1d(
        data, sigma_samples, truncate=truncate, axis=axis, mode="constant"
    )
    weight = gaussian_filter1d(
        np.ones(data.shape[axis]), sigma_samples, truncate=truncate, mode="constant"
    )
    shape = [1] * data.ndim
    shape[axis] = data.shape[axis]
    smoothed = smoothed / weight.reshape(shape)
    return np.asarray(smoothed, dtype=float)


def _get_normalization_mask(
    data_shape: tuple[int, ...], normalization_mask: ArrayLike | None
) -> BoolArray | None:
    """Validate the mask the normalization statistics come from.

    Parameters
    ----------
    data_shape : tuple
        Shape of the data array.
    normalization_mask : array_like or None
        Boolean mask specifying samples to use.

    Returns
    -------
    mask : ndarray or None
        Boolean mask indicating which samples to use, or None to use all data.

    Raises
    ------
    ValueError
        If the mask is not 1-D or not boolean, its length does not match the
        data, or it selects no sample.

    """
    if normalization_mask is None:
        return None
    mask = np.asarray(normalization_mask)
    if mask.ndim != 1:
        msg = (
            f"normalization_mask must be 1-D, shape (n_time,), got shape {mask.shape}. A "
            "2-D mask would pool the statistics of every channel into one."
        )
        raise ValueError(msg)
    if mask.dtype != bool:
        msg = (
            f"normalization_mask must be boolean, got dtype {mask.dtype}. Casting "
            "would make every nonzero value True; pass a comparison such as "
            "speed <= speed_threshold."
        )
        raise ValueError(msg)
    if mask.shape[0] != data_shape[0]:
        msg = (
            f"normalization_mask length ({mask.shape[0]}) must match "
            f"data length ({data_shape[0]})."
        )
        raise ValueError(msg)
    if not np.any(mask):
        msg = "normalization_mask selects no samples; cannot compute normalization statistics."
        raise ValueError(msg)
    return mask


def _normalization_statistics(
    data: FloatArray, mask: BoolArray | None, method: str
) -> tuple[FloatArray, FloatArray]:
    """The center and scale :func:`_normalize` divides by, from ``data[mask]``.

    Parameters
    ----------
    data : ndarray, shape (n_time,) or (n_time, n_channels)
    mask : ndarray of bool, shape (n_time,), or None
        Samples the statistics come from; None uses every sample.
    method : {'zscore', 'median_mad'}

    Returns
    -------
    center, scale : ndarray, shape (1,) or (1, n_channels)
        Mean and standard deviation (``ddof=0``), or median and normal-scaled
        MAD, over the finite samples. A caller that converts a value between
        the raw and the normalized units uses these, so the two agree.

    Raises
    ------
    ValueError
        If the scale of the trace, or of any channel, is zero or undefined
        over the normalization samples. A constant or all-NaN channel has no
        scale, and dividing by a substitute would report its raw values as
        z-scores.

    """
    subset = data if mask is None else data[mask]
    with warnings.catch_warnings():
        # an all-NaN column warns before it is reported as degenerate below
        warnings.simplefilter("ignore", RuntimeWarning)
        if method == "zscore":
            center = np.nanmean(subset, axis=0, keepdims=True)
            scale = np.nanstd(subset, axis=0, ddof=0, keepdims=True)
        else:
            center = np.nanmedian(subset, axis=0, keepdims=True)
            scale = median_abs_deviation(subset, axis=0, scale="normal", nan_policy="omit")
    scale = np.reshape(scale, center.shape)
    degenerate = ~np.isfinite(scale) | (scale <= 0)
    if np.any(degenerate):
        scale_name = "standard deviation" if method == "zscore" else "MAD"
        where = (
            "the trace"
            if data.ndim == 1
            else f"channel(s) {np.flatnonzero(degenerate.ravel()).tolist()}"
        )
        msg = (
            f"Cannot normalize: the {scale_name} of {where} is zero or undefined over "
            "the normalization samples. A constant or all-NaN channel has no scale; "
            "drop it before detecting."
        )
        raise ValueError(msg)
    return np.asarray(center, dtype=float), np.asarray(scale, dtype=float)


def _normalize(data: FloatArray, mask: BoolArray | None, method: str) -> FloatArray:
    """Center and scale ``data`` with :func:`_normalization_statistics` from
    ``data[mask]``; raises as it does for a zero or undefined scale."""
    center, scale = _normalization_statistics(data, mask, method)
    return np.asarray((data - center) / scale, dtype=float)


@explain_call_errors
def normalize_signal(
    data: ArrayLike,
    method: str = "zscore",
    normalization_mask: ArrayLike | None = None,
) -> FloatArray:
    """Normalize signal using mean/std (z-score) or median/MAD.

    The statistics (mean/std or median/MAD) come from the whole signal, or
    from the samples ``normalization_mask`` selects, and are applied to every
    sample.

    Parameters
    ----------
    data : array_like, shape (n_time,) or (n_time, n_channels)
        Input signal to normalize. Can be 1-D or 2-D.
    method : {'zscore', 'median_mad'}, optional
        Normalization method:

        - 'zscore': (data - mean) / std
        - 'median_mad': (data - median) / MAD, where MAD is scaled to be
          comparable to standard deviation for normally distributed data

        Default is 'zscore'. Use 'median_mad' for more robust normalization
        when data contains outliers.
    normalization_mask : array_like of bool, shape (n_time,), optional
        Samples the statistics are computed from. For example,
        ``speed <= speed_threshold`` restricts them to immobility, and
        ``(time >= start) & (time <= end)`` to a baseline period. Default is
        None (use all data).

    Returns
    -------
    normalized_data : ndarray, shape matches input
        Normalized signal with the same shape as input.

    Raises
    ------
    ValueError
        If `method` is not recognized, if the mask is not boolean, has the
        wrong length or selects no samples, or if the scale (standard
        deviation or MAD) of the trace or of any channel is zero or undefined
        over the normalization samples. A constant or all-NaN channel has no
        scale; drop it before detecting.

    Notes
    -----
    The 'median_mad' method uses `scipy.stats.median_abs_deviation` with
    `scale='normal'`, which applies a scaling factor of ~1.4826 to make MAD
    comparable to standard deviation for Gaussian data. This is equivalent to:

    .. math::

        \\text{normalized} = \\frac{x - \\text{median}(x)}{1.4826 \\cdot \\text{MAD}(x)}

    where MAD is the median absolute deviation from the median.

    Both methods ignore NaN samples when computing the statistics. The
    standard deviation is the population one (``ddof=0``); MATLAB's ``std``,
    which the original implementations use, divides by ``n - 1``. On a
    recording of any length the difference is far below 1e-5 of a standard
    deviation.

    The MAD is the median of the absolute deviations from the median, so it is
    zero, or nearly so, whenever more than half the samples tie at the median.
    A sparse trace such as a smoothed spike rate that is zero much of the time
    therefore gets a tiny MAD and enormous "z-scores". Use ``'zscore'`` for
    such traces.

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
    >>> normalized = normalize_signal(lfp, normalization_mask=speed <= 4.0)

    Normalize using a baseline period:

    >>> baseline = (time >= 0.0) & (time <= 0.3)
    >>> normalized = normalize_signal(lfp, normalization_mask=baseline)

    See Also
    --------
    scipy.stats.zscore : Standard z-score normalization
    scipy.stats.median_abs_deviation : Median absolute deviation

    References
    ----------
    .. [1] Leys, C., Ley, C., Klein, O., Bernard, P., & Licata, L. (2013).
       Detecting outliers: Do not use standard deviation around the mean, use
       absolute deviation around the median. Journal of Experimental Social
       Psychology, 49(4), 764-766. doi:10.1016/j.jesp.2013.03.013

    """
    given: object = method  # a caller without a type checker can pass anything
    if not isinstance(given, str):
        msg = (
            f"method must be 'zscore' or 'median_mad', got a {type(method).__name__}. "
            f"{NORMALIZE_SIGNAL_WITHOUT_TIME}"
        )
        raise TypeError(msg)
    if method not in ("zscore", "median_mad"):
        msg = (
            f"Invalid normalization method: '{method}'. "
            "Must be either 'zscore' or 'median_mad'."
        )
        raise ValueError(msg)
    data_arr = np.asarray(data, dtype=float)
    mask = _get_normalization_mask(data_arr.shape, normalization_mask)
    return _normalize(data_arr, mask, method)


def normalize_signal_manually(
    data: ArrayLike,
    channel_baselines: ArrayLike,
    channel_deviations: ArrayLike,
) -> FloatArray:
    """Normalize with supplied baselines and deviations.

    The statistics come from the arguments rather than from ``data``. This
    matters for sleep sessions. A sleep session holds a higher concentration
    of ripples, so its own baseline and deviation are larger. Statistics from
    the whole recording day avoid that bias.

    Parameters
    ----------
    data : array_like, shape (n_time,) or (n_time, n_channels)
        Input signal to normalize. Can be 1-D or 2-D.
    channel_baselines : array_like, shape (n_channels,)
        Baseline (center) value for each channel; a scalar for 1-D data.
    channel_deviations : array_like, shape (n_channels,)
        Deviation (scale) value for each channel; a scalar for 1-D data. Must
        be on the scale of a standard deviation: multiply a MAD by 1.4826
        first.

    Returns
    -------
    normalized_data : ndarray, shape matches input
        ``(data - channel_baselines) / channel_deviations``.

    Raises
    ------
    ValueError
        If the two statistics differ in length or do not have one entry per
        channel, or if any channel's deviation is zero or NaN or its baseline
        is NaN. Such a channel has no scale; drop it before detecting, as
        ``normalize_signal`` also requires.

    """
    data = np.asarray(data, dtype=float)
    baselines = np.atleast_1d(np.asarray(channel_baselines, dtype=float))
    deviations = np.atleast_1d(np.asarray(channel_deviations, dtype=float))
    n_channels = 1 if data.ndim == 1 else data.shape[1]
    if baselines.shape != deviations.shape:
        msg = (
            f"channel_baselines {baselines.shape} and channel_deviations {deviations.shape} "
            "must have the same shape."
        )
        raise ValueError(msg)
    if baselines.shape != (n_channels,):
        msg = (
            "channel_baselines and channel_deviations must have one entry per channel "
            f"(n_channels={n_channels}), got {baselines.size}."
        )
        raise ValueError(msg)
    degenerate = (deviations == 0) | ~np.isfinite(deviations) | ~np.isfinite(baselines)
    if np.any(degenerate):
        msg = (
            "Cannot normalize: channel(s) "
            f"{np.flatnonzero(degenerate).tolist()} have a zero or NaN deviation or a "
            "NaN baseline. Such a channel has no scale; drop it before detecting."
        )
        raise ValueError(msg)
    if data.ndim == 1:
        return np.asarray((data - baselines[0]) / deviations[0], dtype=float)
    return np.asarray((data - baselines) / deviations, dtype=float)


def threshold_by_zscore(
    zscored_data: ArrayLike,
    time: ArrayLike,
    minimum_duration: float = 0.015,
    zscore_threshold: float = 2,
) -> list[tuple[float, float]]:
    """Find time segments where z-scored data exceeds a threshold.

    Identifies segments where the z-scored signal is at or above the
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
    if zscore_threshold < 0:
        msg = (
            f"zscore_threshold must be non-negative, got {zscore_threshold}. The "
            "extension to the crossing point assumes every threshold crossing "
            "lies inside a run above the normalization center."
        )
        raise ValueError(msg)
    zscored = np.asarray(zscored_data, dtype=float)
    return extend_threshold_to_mean(
        zscored >= 0, zscored >= zscore_threshold, time, minimum_duration=minimum_duration
    )


def merge_overlapping_ranges(
    ranges: Iterable[tuple[float, float]],
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
    remaining = iter(sorted(ranges))
    try:
        current_start, current_stop = next(remaining)
    except StopIteration:
        return None
    for start, stop in remaining:
        if start > current_stop:
            # Gap between segments: output current segment and start a new
            # one.
            yield current_start, current_stop
            current_start, current_stop = start, stop
        else:
            # Segments adjacent or overlapping: merge.
            current_stop = max(current_stop, stop)
    yield current_start, current_stop


def merge_overlapping_ranges_track_participation(
    candidate_ripple_times: list[list[tuple[float, float]]],
) -> FloatArray:
    """Merge overlapping/adjacent per-channel ranges, tracking participation.

    Like `merge_overlapping_ranges`, but also records which channels contribute
    to each merged interval. Each channel is counted once across the entire
    merged interval, including chains of overlapping ripples.

    Parameters
    ----------
    candidate_ripple_times : list of length n_channels
        Per-channel lists of (start_time, end_time) tuples.

    Returns
    -------
    merged : ndarray, shape (n_merged, 3), dtype=object
        Each row is ``[start_time, end_time, participating_channels]``, where
        ``participating_channels`` is a set of channel indices.
    """
    all_intervals = []
    for e_idx, intervals in enumerate(candidate_ripple_times):
        for start, end in intervals:
            all_intervals.append((start, end, e_idx))

    all_intervals.sort(key=lambda x: x[0])

    merged: list[list[Any]] = []

    for start, end, e_idx in all_intervals:
        if not merged:
            merged.append([start, end, {e_idx}])
            continue

        last_end = merged[-1][1]
        if start <= last_end:
            merged[-1][1] = max(last_end, end)
            merged[-1][2].add(e_idx)
        else:
            merged.append([start, end, {e_idx}])

    if not merged:
        return np.empty((0, 3), dtype=object)

    return np.asarray(merged, dtype=object)


_GAP_TOLERANCE = 1e-9
"""Relative tolerance for comparing an inter-event gap with a threshold."""


def _is_gap_below(
    gap: FloatArray | float, close_event_threshold: float
) -> BoolArray | np.bool_:
    """Whether an inter-event gap is shorter than the threshold.

    The boundary rule that :func:`exclude_close_events` and
    :func:`merge_close_events` share: a gap equal to the threshold is treated
    as equal rather than as shorter, which binary floating point would
    otherwise decide for it, since 0.15 - 0.1 is 4.999...e-2, just under 0.05.
    The tolerance is relative only, with no absolute term, so the rule stays
    monotonic in the threshold: a default ``atol`` would make a threshold near
    zero merge less than a threshold of zero.

    Parameters
    ----------
    gap : ndarray, shape (n_gaps,), or float
        Time from one event's end to the next event's start.
    close_event_threshold : float
        Separation below which events count as close.

    Returns
    -------
    is_below : ndarray of bool, shape (n_gaps,), or bool

    """
    return (gap < close_event_threshold) & ~np.isclose(
        gap, close_event_threshold, rtol=_GAP_TOLERANCE, atol=0.0
    )


def _check_non_negative(**values: float) -> None:
    """Raise for a value that is NaN or negative. Infinity passes: it is how a
    caller turns the speed criterion off. A gap or ceiling must be finite, which
    ``_check_gap`` and ``_validate_duration_limits`` enforce."""
    for name, value in values.items():
        if not value >= 0:
            msg = f"{name} must be non-negative, got {value}."
            raise ValueError(msg)


CLOSE_EVENT_REFERENCES = ("end", "start")
"""What :func:`exclude_close_events` measures a gap from, in the last kept event."""


def _is_clear_of_close_events(
    events: FloatArray, close_event_threshold: float, measure_from: str = "end"
) -> BoolArray:
    """Which of the sorted ``(n_events, 2)`` events to keep: each is compared
    with the last *retained* event, so a cluster is reduced to its first
    event. Comparing with the immediately preceding candidate instead would
    let a dropped event go on excluding its successors, removing more than
    the first-of-each-cluster rule. The gap runs to each event's start from
    the retained event's end, or from its start with ``measure_from='start'``."""
    if measure_from not in CLOSE_EVENT_REFERENCES:
        msg = (
            f"measure_from must be one of {', '.join(map(repr, CLOSE_EVENT_REFERENCES))}; "
            f"got {measure_from!r}."
        )
        raise ValueError(msg)
    column = 1 if measure_from == "end" else 0
    keep = np.zeros(len(events), dtype=bool)
    if len(events):
        keep[0] = True
        reference = events[0, column]
        for event in range(1, len(events)):
            if not _is_gap_below(events[event, 0] - reference, close_event_threshold):
                keep[event] = True
                reference = events[event, column]
    return keep


def exclude_close_events(
    candidate_event_times: ArrayLike | pd.DataFrame,
    close_event_threshold: float = 1.0,
    measure_from: str = "end",
) -> FloatArray | pd.DataFrame:
    """Remove events that occur too close together in time.

    Filters out successive events that start within `close_event_threshold`
    time units of the last retained event's end, keeping only the first event
    in each cluster of closely-spaced events.

    The Frank lab ``extractevents`` routine instead *merges* events separated
    by less than its minimum separation into one longer event
    (:func:`merge_close_events`). This function drops the later event, so the
    retained events keep their original bounds.

    Parameters
    ----------
    candidate_event_times : array_like, shape (n_events, 2), or pd.DataFrame
        Event times with columns [start_time, end_time], sorted by start
        time, or a detector's DataFrame, which is returned filtered with every
        column.
    close_event_threshold : float, optional
        Minimum time between events. Events starting within this time after
        a previous event ends are excluded. Non-negative. Default is 1.0
        (seconds).
    measure_from : {'end', 'start'}, optional
        Where in the last retained event the gap starts: its end (default),
        or its start, for rules such as "SWRs within 1 s after another SWR
        were excluded" that time the interval from detection. Either way the
        gap ends at the next event's start.

    Returns
    -------
    filtered_event_times : ndarray, shape (n_filtered_events, 2), or pd.DataFrame
        The retained events, in the input's type; shape ``(0, 2)`` when none
        remain.

    Raises
    ------
    ValueError
        If `close_event_threshold` is negative or `measure_from` is not one of
        the two.

    See Also
    --------
    require_isolation : drops every event of a close pair, not just the later.

    Notes
    -----
    This function assumes events are sorted by start time. If the input
    is not sorted, results may be incorrect.

    Examples
    --------
    >>> events = np.array([(0.0, 0.1), (0.5, 0.6), (1.05, 1.1)])
    >>> exclude_close_events(events, 1.0)
    array([[0. , 0.1]])
    >>> exclude_close_events(events, 1.0, measure_from="start")
    array([[0.  , 0.1 ],
           [1.05, 1.1 ]])

    """
    _check_non_negative(close_event_threshold=close_event_threshold)
    events = _event_bounds(candidate_event_times)
    keep = _is_clear_of_close_events(events, close_event_threshold, measure_from)
    if isinstance(candidate_event_times, pd.DataFrame):
        return candidate_event_times.iloc[np.flatnonzero(keep)].copy()
    return events[keep]


def require_isolation(
    event_times: ArrayLike | pd.DataFrame,
    minimum_separation: float,
) -> FloatArray | pd.DataFrame:
    """Keep the events with no other event within a separation on either side.

    Rules such as "only SWRs separated from others by at least 500 ms" drop
    every event of a close pair, where :func:`exclude_close_events` keeps the
    first. An event is kept when the gap from the latest end among the events
    before it, and the gap to the next event's start, are both at least
    `minimum_separation`, within floating-point tolerance.

    Parameters
    ----------
    event_times : array_like, shape (n_events, 2), or pd.DataFrame
        ``[start_time, end_time]`` per event, sorted by start time, or a
        detector's DataFrame, which is returned filtered with every column
        and its index.
    minimum_separation : float
        Least gap, in the units of the event times, to the nearest other
        event on each side. Non-negative; 0 keeps every event that overlaps
        no other.

    Returns
    -------
    isolated_events : ndarray, shape (n_kept, 2), or pd.DataFrame
        The isolated events, in the input's type and order.

    Raises
    ------
    ValueError
        If `minimum_separation` is negative or the events are not sorted by
        start time.

    Examples
    --------
    >>> events = np.array([(0.0, 0.1), (0.3, 0.4), (2.0, 2.1)])
    >>> require_isolation(events, 0.5)
    array([[2. , 2.1]])

    """
    _check_non_negative(minimum_separation=minimum_separation)
    events = _event_bounds(event_times)
    if np.any(np.diff(events[:, 0]) < 0):
        msg = (
            "event_times must be sorted by start time. Sort the events first: "
            "event_times[np.argsort(event_times[:, 0])]."
        )
        raise ValueError(msg)
    keep = np.ones(len(events), dtype=bool)
    if len(events) > 1:
        latest_end_before = np.maximum.accumulate(events[:-1, 1])
        gap_before = events[1:, 0] - latest_end_before
        too_close = np.asarray(_is_gap_below(gap_before, minimum_separation), dtype=bool)
        if minimum_separation == 0:
            too_close = gap_before < 0
        # a close pair loses its second member through the gap before it and
        # its first through the same gap, read as the gap after
        keep[1:] &= ~too_close
        keep[:-1] &= ~too_close
    if isinstance(event_times, pd.DataFrame):
        return event_times.iloc[np.flatnonzero(keep)].copy()
    return events[keep]


MERGE_MEASURES = ("gap", "peak")
"""What :func:`merge_close_events` compares with its threshold."""


def merge_close_events(
    event_times: ArrayLike | pd.DataFrame,
    close_event_threshold: float = 0.0,
    maximum_duration: float | None = None,
    *,
    inclusive: bool = False,
    measure: str = "gap",
) -> FloatArray:
    """Join events separated by less than a gap into one longer event.

    The other convention for closely spaced events is
    :func:`exclude_close_events`, which keeps the first of a cluster and drops
    the rest. This one keeps every event's content: a merged event runs from
    the first start to the last end. Both conventions are used in the
    literature; the Frank lab ``extractevents`` routine merges.

    Merging is repeated until nothing more can be joined, so a chain of events
    each close to the next becomes one event. Events that overlap or nest have
    a gap at or below zero, so they merge whenever the threshold alone decides
    it. With `maximum_duration` set, a merge that would exceed the ceiling does
    not happen, and the result can still hold overlapping events.

    Parameters
    ----------
    event_times : array_like, shape (n_events, 2), or pd.DataFrame
        ``[start_time, end_time]`` per event, sorted by start time, or a
        detector's DataFrame, whose ``start_time`` and ``end_time`` are read.
        Merging changes the bounds, so the result is always an array; the
        other columns of a merged event have no single value.
    close_event_threshold : float, optional
        Events separated by strictly less than this gap are merged. A gap equal
        to the threshold does not merge, within floating-point tolerance,
        unless `inclusive`. Default is 0.0, which merges only events that touch
        or overlap.
    maximum_duration : float, optional
        Ceiling on the merged span. A merge that would produce an event longer
        than this does not happen and both events are kept as they are.
        Default is None (no ceiling).
    inclusive : bool, optional
        Also merge events separated by exactly the threshold, for rules
        written "merged if 40 ms or less apart". Default False.
    measure : {'gap', 'peak'}, optional
        What is compared with the threshold: the gap from one event's end to
        the next one's start (default), or the time between their peaks,
        read from the DataFrame's ``peak_time`` column, for rules such as
        "events whose peaks were less than 70 ms apart were merged". A chain
        is followed peak to peak: after a merge, the next event is measured
        from the peak of the last event merged in. A peak lies inside its
        event, so this merges no more than the gap would at the same
        threshold.

    Returns
    -------
    merged_event_times : ndarray, shape (n_merged_events, 2)
        Merged events, sorted by start time. Shape ``(0, 2)`` when there is
        no input.

    Raises
    ------
    ValueError
        If `close_event_threshold` is negative, the events are not sorted
        by start time, `measure` is not one of the two, or ``measure='peak'``
        is asked of anything but a DataFrame with a ``peak_time`` column.

    Examples
    --------
    >>> events = np.array([(0.0, 0.1), (0.13, 0.2)])
    >>> merge_close_events(events, 0.05)
    array([[0. , 0.2]])
    >>> merge_close_events(np.array([(0.0, 0.1), (0.14, 0.2)]), 0.04, inclusive=True)
    array([[0. , 0.2]])

    """
    _check_non_negative(close_event_threshold=close_event_threshold)
    if measure not in MERGE_MEASURES:
        msg = (
            f"measure must be one of {', '.join(map(repr, MERGE_MEASURES))}; got {measure!r}."
        )
        raise ValueError(msg)
    if measure == "peak" and not (
        isinstance(event_times, pd.DataFrame) and "peak_time" in event_times
    ):
        msg = (
            "measure='peak' reads each event's peak from a peak_time column, so pass a "
            "detector's DataFrame, which has one."
        )
        raise ValueError(msg)
    events = _event_bounds(event_times).copy()
    if events.size == 0:
        return np.empty((0, 2))
    if np.any(np.diff(events[:, 0]) < 0):
        msg = (
            "event_times must be sorted by start time. Sort the events before merging: "
            "event_times[np.argsort(event_times[:, 0])]."
        )
        raise ValueError(msg)
    if measure == "peak":
        assert isinstance(event_times, pd.DataFrame)
        first_peak = event_times["peak_time"].to_numpy(dtype=float).copy()
        last_peak = first_peak.copy()

    while len(events) > 1:
        if measure == "peak":
            gap = first_peak[1:] - last_peak[:-1]
        else:
            gap = events[1:, 0] - events[:-1, 1]
        if close_event_threshold > 0:
            to_merge = np.asarray(_is_gap_below(gap, close_event_threshold), dtype=bool)
            if inclusive:
                to_merge |= np.isclose(
                    gap, close_event_threshold, rtol=_GAP_TOLERANCE, atol=0.0
                )
        else:
            # events that touch merge, which _is_gap_below would exclude
            to_merge = gap <= 0
        if maximum_duration is not None:
            merged_span = np.maximum(events[1:, 1], events[:-1, 1]) - events[:-1, 0]
            to_merge &= (merged_span <= maximum_duration) | np.isclose(
                merged_span, maximum_duration
            )
        if not np.any(to_merge):
            break
        # merge one neighbor per pass, taking the first of each run so a chain
        # collapses left to right rather than skipping a link
        padded = np.concatenate([[False], to_merge])
        run_starts = np.flatnonzero(~padded[:-1] & padded[1:])
        events[run_starts, 1] = np.maximum(events[run_starts, 1], events[run_starts + 1, 1])
        events = np.delete(events, run_starts + 1, axis=0)
        if measure == "peak":
            last_peak[run_starts] = last_peak[run_starts + 1]
            first_peak = np.delete(first_peak, run_starts + 1)
            last_peak = np.delete(last_peak, run_starts + 1)

    return events


def _overlaps(
    event_times: ArrayLike | pd.DataFrame,
    reference_event_times: ArrayLike | pd.DataFrame,
    minimum_overlap: float,
) -> tuple[FloatArray, BoolArray]:
    """The events as bounds, and whether each overlaps the union of the
    references by a positive amount of at least ``minimum_overlap``.

    Bounds must be finite and in order: a NaN or reversed row would silently
    break the sorted arithmetic below for every event after it, which for a
    veto means keeping what it should drop."""
    _check_non_negative(minimum_overlap=minimum_overlap)
    events = _event_bounds(event_times)
    reference = _event_bounds(reference_event_times)
    for name, bounds in (("event_times", events), ("reference_event_times", reference)):
        bad = ~np.isfinite(bounds).all(axis=1) | (bounds[:, 1] < bounds[:, 0])
        if bad.any():
            row = int(np.flatnonzero(bad)[0])
            msg = (
                f"{name} row {row} is {bounds[row].tolist()}: every start and end must be "
                "finite, with the start no later than the end."
            )
            raise ValueError(msg)
    if not (len(events) and len(reference)):
        return events, np.zeros(len(events), dtype=bool)
    reference = reference[np.argsort(reference[:, 0], kind="stable")]
    reference = merge_close_events(reference)
    starts, ends = events[:, 0], events[:, 1]
    ref_start, ref_end = reference[:, 0], reference[:, 1]
    # the reference is disjoint and sorted, so the intervals that can meet
    # an event form one contiguous run; cumulative lengths then give the
    # total overlap without looping over the pairs
    cumulative = np.concatenate([[0.0], np.cumsum(ref_end - ref_start)])
    first = np.searchsorted(ref_end, starts, side="right")
    last = np.searchsorted(ref_start, ends, side="left")
    meets = last > first
    head = np.maximum(0.0, starts - ref_start[np.clip(first, 0, len(reference) - 1)])
    tail = np.maximum(0.0, ref_end[np.clip(last - 1, 0, len(reference) - 1)] - ends)
    overlap = np.where(meets, cumulative[last] - cumulative[first] - head - tail, 0.0)
    return events, np.asarray((overlap > 0) & (overlap >= minimum_overlap))


def require_overlap(
    event_times: ArrayLike | pd.DataFrame,
    reference_event_times: ArrayLike | pd.DataFrame,
    minimum_overlap: float = 0.0,
) -> FloatArray | pd.DataFrame:
    """Keep the events that overlap an event in a second inventory.

    Many studies require a ripple and a population burst together, usually by
    keeping the multiunit bursts that overlap a detected ripple. This composes
    any two detectors into that conjunction::

        bursts = multiunit_HSE_detector(time, multiunit, speed, fs)
        ripples = Kay_ripple_detector(time, lfps, speed, fs)
        both = require_overlap(bursts, ripples)

    The returned events keep their own bounds. This is a filter on
    `event_times`, not an intersection of the two inventories, and it is not
    symmetric: swapping the arguments asks the other question.

    Parameters
    ----------
    event_times : array_like, shape (n_events, 2), or pd.DataFrame
        The events to filter. A DataFrame needs ``start_time`` and
        ``end_time`` columns and is returned as a DataFrame, with every column
        and its index preserved.
    reference_event_times : array_like, shape (n_reference, 2), or pd.DataFrame
        The events to overlap with. Reduced to its union first, so references
        that overlap each other are not counted twice. Need not be sorted.
    minimum_overlap : float, optional
        Least total overlap, in the units of the event times, for an event to
        be kept. Default is 0.0, which requires overlap of positive duration:
        events that merely touch at an endpoint are dropped.

    Returns
    -------
    kept_events : ndarray, shape (n_kept, 2), or pd.DataFrame
        The subset of `event_times` meeting the criterion, in its input order
        and type. :func:`exclude_overlap` returns the rest.

    Raises
    ------
    ValueError
        If `minimum_overlap` is negative, or a start or end in either input is
        not finite or an interval ends before it starts.

    Examples
    --------
    >>> bursts = np.array([(0.0, 0.1), (1.0, 1.1)])
    >>> ripples = np.array([(1.05, 1.5)])
    >>> require_overlap(bursts, ripples)
    array([[1. , 1.1]])

    """
    events, keep = _overlaps(event_times, reference_event_times, minimum_overlap)
    if isinstance(event_times, pd.DataFrame):
        return event_times.iloc[np.flatnonzero(keep)].copy()
    return events[keep]


def exclude_overlap(
    event_times: ArrayLike | pd.DataFrame,
    reference_event_times: ArrayLike | pd.DataFrame,
    minimum_overlap: float = 0.0,
) -> FloatArray | pd.DataFrame:
    """Drop the events that overlap an event in a second inventory.

    The complement of :func:`require_overlap`, for vetoes: drop the events
    that coincide with intervals marked elsewhere, such as artifacts, periods
    of muscle activity, or interictal spikes. This package does not detect
    those; pass their start and end times from whatever marked them. Every
    event is kept by exactly one of the two functions given the same
    arguments.

    A veto that reaches beyond the reference intervals, such as "within
    100 ms of an interictal spike", is overlap with the references widened by
    that much. A zero-length interval has no duration to overlap, so a point
    reference, such as a spike's peak time, vetoes nothing until widened:
    ``exclude_overlap(ripples, peak_times[:, np.newaxis] + [-0.1, 0.1])``.

    Parameters
    ----------
    event_times : array_like, shape (n_events, 2), or pd.DataFrame
        The events to filter. A DataFrame needs ``start_time`` and
        ``end_time`` columns and is returned as a DataFrame, with every column
        and its index preserved.
    reference_event_times : array_like, shape (n_reference, 2), or pd.DataFrame
        The events to avoid. Reduced to its union first, so references that
        overlap each other are not counted twice. Need not be sorted.
    minimum_overlap : float, optional
        Least total overlap, in the units of the event times, for an event to
        be dropped. Default is 0.0, which drops any overlap of positive
        duration: events that merely touch a reference at an endpoint are kept.

    Returns
    -------
    kept_events : ndarray, shape (n_kept, 2), or pd.DataFrame
        The subset of `event_times` that does not meet the criterion, in its
        input order and type.

    Raises
    ------
    ValueError
        If `minimum_overlap` is negative, or a start or end in either input is
        not finite or an interval ends before it starts: such a row would
        switch the veto off, silently, for the events after it.

    Examples
    --------
    >>> ripples = np.array([(0.0, 0.1), (1.0, 1.1)])
    >>> artifacts = np.array([(1.05, 1.5)])
    >>> exclude_overlap(ripples, artifacts)
    array([[0. , 0.1]])

    """
    events, overlaps = _overlaps(event_times, reference_event_times, minimum_overlap)
    keep = ~overlaps
    if isinstance(event_times, pd.DataFrame):
        return event_times.iloc[np.flatnonzero(keep)].copy()
    return events[keep]


def require_trace_peak(
    event_times: ArrayLike | pd.DataFrame,
    trace: ArrayLike,
    time: ArrayLike,
    threshold: float,
) -> FloatArray | pd.DataFrame:
    """Keep the events in which a trace reaches a threshold.

    For rules that confirm one signal's events with another, such as "a
    multiunit burst with a ripple-band z-score of at least 3 inside it".
    Pass the trace already normalized the way the rule states, for example
    ``normalize_signal(get_Kay_ripple_consensus_trace(lfps, fs))``.

    Parameters
    ----------
    event_times : array_like, shape (n_events, 2), or pd.DataFrame
        ``[start_time, end_time]`` per event, or a detector's DataFrame,
        returned filtered with every column and its index. An event holds
        the samples with ``start_time <= time <= end_time``.
    trace : array_like, shape (n_time,)
        The confirming trace. NaN samples are skipped; an event whose samples
        are all NaN is dropped.
    time : array_like, shape (n_time,)
        Sample timestamps, increasing.
    threshold : float
        Level the trace must reach, at or above, somewhere in the event.

    Returns
    -------
    kept_events : ndarray, shape (n_kept, 2), or pd.DataFrame
        The events in which the trace reaches `threshold`, in the input's
        type and order.

    Raises
    ------
    ValueError
        If `trace` and `time` differ in shape, `threshold` is not finite, or
        no sample falls within an event.

    Examples
    --------
    >>> time = np.arange(10) / 10
    >>> ripple_z = np.array([0, 1, 4, 1, 0, 0, 1, 2, 1, 0.0])
    >>> bursts = np.array([(0.0, 0.3), (0.5, 0.9)])
    >>> require_trace_peak(bursts, ripple_z, time, 3.0)
    array([[0. , 0.3]])

    """
    values = np.asarray(trace, dtype=float)
    time = np.asarray(time, dtype=float)
    if values.shape != time.shape:
        msg = f"trace has shape {values.shape} and time {time.shape}; they must match."
        raise ValueError(msg)
    if not np.isfinite(threshold):
        msg = f"threshold must be finite, got {threshold}."
        raise ValueError(msg)
    events = _event_bounds(event_times)
    first = np.searchsorted(time, events[:, 0], side="left")
    last = np.searchsorted(time, events[:, 1], side="right")
    if np.any(last == first):
        start_time, end_time = events[np.flatnonzero(last == first)[0]]
        msg = f"No sample of time falls within event [{start_time}, {end_time}]."
        raise ValueError(msg)
    keep = np.zeros(len(events), dtype=bool)
    for event, (a, b) in enumerate(zip(first, last, strict=True)):
        inside = values[a:b]
        keep[event] = bool(np.any(inside[np.isfinite(inside)] >= threshold))
    if isinstance(event_times, pd.DataFrame):
        return event_times.iloc[np.flatnonzero(keep)].copy()
    return events[keep]


def require_times_inside(
    event_times: ArrayLike | pd.DataFrame,
    times: ArrayLike,
) -> FloatArray | pd.DataFrame:
    """Keep the events that contain at least one of the given times.

    For rules such as "population bursts that contain the peak of at least
    one ripple": ``require_times_inside(bursts, ripples.peak_time)``. A point
    has no duration, so :func:`require_overlap` cannot ask this; here an
    event ``[start, end]`` contains a time ``t`` when ``start <= t <= end``.

    Parameters
    ----------
    event_times : array_like, shape (n_events, 2), or pd.DataFrame
        ``[start_time, end_time]`` per event, or a detector's DataFrame,
        returned filtered with every column and its index.
    times : array_like, shape (n_times,)
        The times to look for, in any order.

    Returns
    -------
    kept_events : ndarray, shape (n_kept, 2), or pd.DataFrame
        The events containing a time, in the input's type and order. The
        rest are ``events.drop(kept.index)`` for a DataFrame.

    Raises
    ------
    ValueError
        If `times` holds NaN or infinity, which no event could contain.

    Examples
    --------
    >>> bursts = np.array([(0.0, 0.3), (0.5, 0.9)])
    >>> require_times_inside(bursts, [0.7, 2.0])
    array([[0.5, 0.9]])

    """
    points = np.sort(np.asarray(times, dtype=float).ravel())
    if not np.all(np.isfinite(points)):
        msg = (
            "times holds NaN or infinity; drop those before asking which events contain them."
        )
        raise ValueError(msg)
    events = _event_bounds(event_times)
    if len(points) == 0:
        keep = np.zeros(len(events), dtype=bool)
    else:
        first_after_start = np.searchsorted(points, events[:, 0], side="left")
        candidate = points[np.clip(first_after_start, 0, len(points) - 1)]
        keep = (first_after_start < len(points)) & (candidate <= events[:, 1])
    if isinstance(event_times, pd.DataFrame):
        return event_times.iloc[np.flatnonzero(keep)].copy()
    return events[keep]


def windows_around_times(
    times: ArrayLike,
    before: float,
    after: float | None = None,
    *,
    merge_overlapping: bool = True,
) -> FloatArray:
    """Fixed windows around times, such as each event's peak.

    For rules that define an event as a fixed window rather than by where a
    trace falls back: "a 100 ms window centered on the peak" is
    ``windows_around_times(events.peak_time, 0.05)``, and "150 ms windows
    centered on every sample above threshold, overlapping windows joined" is
    ``windows_around_times(time[trace >= threshold], 0.075)``.

    Parameters
    ----------
    times : array_like, shape (n_times,)
        Window centers, in any order.
    before : float
        Extent of each window before its time, in the units of `times`.
    after : float, optional
        Extent after it. Default None, the same as `before`.
    merge_overlapping : bool, optional
        Join windows that overlap or touch into one (default). With False,
        one window per time, sorted.

    Returns
    -------
    windows : ndarray, shape (n_windows, 2)
        ``[start, end]`` per window, sorted by start. Windows are not clipped
        to the recording or to its missing samples.

    Raises
    ------
    ValueError
        If `before` or `after` is negative or not finite, or `times` holds NaN
        or infinity.

    Examples
    --------
    >>> windows_around_times([1.0, 1.05, 3.0], 0.05)
    array([[0.95, 1.1 ],
           [2.95, 3.05]])

    """
    after = before if after is None else after
    for name, value in (("before", before), ("after", after)):
        if not 0 <= value < np.inf:
            msg = f"{name} must be finite and non-negative, got {value}."
            raise ValueError(msg)
    centers = np.sort(np.asarray(times, dtype=float).ravel())
    if not np.all(np.isfinite(centers)):
        msg = "times holds NaN or infinity, which has no window."
        raise ValueError(msg)
    windows = np.column_stack([centers - before, centers + after]).reshape(-1, 2)
    if merge_overlapping:
        return merge_close_events(windows)
    return windows


YU_HISTOGRAM_EDGES = np.round(np.arange(-10.0, 50.0 + 0.005, 0.01), 6)
"""Histogram grid of the Yu et al. 2017 noise-threshold estimator.

Bin edges from -10 to 50 in steps of 0.01, in the units of the consensus trace
(the median of per-tetrode z-scored envelopes). Transliterated from
``histbins = -10:0.01:50`` in ``jy_variableripthreshold_corecalculation.m``
(Frank lab, unpublished; not in a public repository). The pipeline that calls
it, ``DFFunctions/AG_extractRipplesJY.m`` in
https://github.com/droumis/FFPhy/tree/fce2048/DFFunctions, is public and shows
the per-tetrode z-score, the median, and the immobility noise sample this
package reproduces.

Read-only, because ``NoiseThresholdDiagnostics.histogram_edges`` hands this
array to the caller, and a change made there in place would move every later
threshold estimate.
"""
YU_HISTOGRAM_EDGES.flags.writeable = False

YU_MODE_SMOOTHING_WINDOW = 11
"""Moving-average window (in bins) used to locate the histogram mode."""

_OUT_OF_GRID_CEILING = 1e-3
"""Largest tolerated fraction of samples outside ``YU_HISTOGRAM_EDGES``."""


def _matlab_smooth(x: ArrayLike, window: int) -> FloatArray:
    """Moving average with MATLAB ``smooth(x, window)`` end handling.

    Interior points average ``window`` neighbors; near either end the window
    shrinks symmetrically (1, 3, 5, ... points) so it never runs off the array.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    half = window // 2
    index = np.arange(n)
    half_width = np.minimum(half, np.minimum(index, n - 1 - index))
    low = index - half_width
    high = index + half_width + 1
    cumulative = np.concatenate([[0.0], np.cumsum(x)])
    return np.asarray((cumulative[high] - cumulative[low]) / (high - low), dtype=float)


def _histc(values: FloatArray, edges: FloatArray) -> IntArray:
    """MATLAB ``histc``: left-closed bins, with a final bin for ``values == edges[-1]``."""
    counts = np.zeros(len(edges), dtype=np.int64)
    counts[:-1], _ = np.histogram(values, bins=edges)
    n_last = int(np.sum(values == edges[-1]))
    counts[-2] -= n_last  # np.histogram closes its last bin on the right
    counts[-1] = n_last
    return counts


@dataclass(frozen=True, eq=False)
class NoiseThresholdDiagnostics:
    """Everything :func:`estimate_noise_threshold` computed on the way to its answer.

    Instances compare by identity: the array fields have no single truth
    value, so a generated ``__eq__`` would raise.

    Attributes
    ----------
    threshold : float
        The estimated threshold, in the units of the input.
    mode : float
        The histogram mode the noise distribution was mirrored about.
    mode_index : int
        Its bin index in ``histogram_edges``.
    mean, min : float
        Mean and minimum of the in-grid samples.
    flank_ratio : float
        Left-flank width over mode-to-mean distance, ``(mode - min) / (mean -
        mode)``; the mirrored distribution can reach past the mean only when
        this exceeds 1. ``inf`` when the mean does not exceed the mode.
    histogram_edges, counts, smoothed_counts : ndarray
        The grid, the raw counts on it, and the counts after the moving
        average that located the mode.
    mirrored_positions, mirrored_counts : ndarray
        The mirrored noise distribution the percentile was read from.
    out_of_grid_fraction : float
        Fraction of samples non-finite or outside the grid.
    n_values, n_in_grid : int
        Sample counts before and after that exclusion.

    """

    threshold: float
    mode: float
    mode_index: int
    mean: float
    min: float
    flank_ratio: float
    histogram_edges: FloatArray
    counts: IntArray
    smoothed_counts: FloatArray
    mirrored_positions: FloatArray
    mirrored_counts: IntArray
    out_of_grid_fraction: float
    n_values: int
    n_in_grid: int


def estimate_noise_threshold(
    values: ArrayLike,
    percentile: float = 99.99,
    histogram_edges: ArrayLike | None = None,
    mode_smoothing_window: int = YU_MODE_SMOOTHING_WINDOW,
) -> float:
    """Estimate a detection threshold from the mirrored noise distribution.

    The threshold rule of Yu et al. 2017. Histogram the consensus envelope
    during immobility on a fixed grid and take the mode. Treat the part below
    the mode as noise alone. Mirror it about the mode to build a symmetric
    noise distribution. Return the value one bin past where that
    distribution's cumulative sum reaches ``percentile``.

    This is a transliteration of the original MATLAB implementation
    (``jy_variableripthreshold_corecalculation.m``). The one deliberate
    departure is the reflection itself: the original computes ``abs(b) + 2m``
    for a left bin ``b`` and mode ``m``, which equals the intended ``2m - b``
    only when ``b <= 0``. This function always reflects as ``2m - b`` and warns
    when the mode is positive, the only case in which the two differ.

    Parameters
    ----------
    values : array_like, shape (n_samples,)
        Consensus envelope samples during immobility, in the units the grid
        assumes (the median of per-tetrode z-scored envelopes). Non-finite
        values and values outside the grid are dropped, as in the original.
    percentile : float, optional
        Percentile of the mirrored noise distribution to use as the threshold.
        Default is 99.99, i.e. a CDF limit of ``1 - 1e-4``.
    histogram_edges : array_like, optional
        Histogram bin edges. Default is ``YU_HISTOGRAM_EDGES``
        (``-10:0.01:50``). Bins are left-closed; the last edge forms its own
        bin, matching MATLAB ``histc``.
    mode_smoothing_window : int, optional
        Moving-average window, in bins, applied to the counts before locating
        the mode (MATLAB ``smooth(counts, 11)``). The mode is the first
        maximum of the smoothed counts. Default is 11.

    Returns
    -------
    threshold : float
        Detection threshold in the units of ``values``.
        :func:`noise_threshold_diagnostics` returns the same estimate with
        everything computed on the way to it.

    Raises
    ------
    ValueError
        If ``histogram_edges`` is not strictly increasing, ``percentile`` is
        not in (0, 100), or ``values`` is empty. And on the data: if more than
        0.1 % of samples fall outside the grid; the mode lies on the first or
        last bin; there are too few samples to resolve the requested
        percentile; or the crossing lands on the last mirrored bin, which
        leaves "one bin past" undefined.

    Warns
    -----
    UserWarning
        If the mode is positive (see Notes).

    Notes
    -----
    The mirrored distribution has an upper bound of ``2m - min(values)``. The
    returned threshold therefore cannot exceed the mode plus the width of the
    left flank, ``2m - min(values)``, whatever the true noise tail does. This
    bound comes from the published method, not from this implementation.

    The threshold therefore lies above the sample mean only when the left
    flank is wider than the distance from the mode to the mean, that is
    ``(m - min) > (mean - m)``. :func:`noise_threshold_diagnostics` reports
    this as ``flank_ratio``. Ripples that are large relative to the in-band
    background inflate the variance and pull the mean above the noise
    ceiling. The ratio then drops below one and ``Yu_ripple_detector`` raises,
    because a threshold below the mean leaves its extension rule undefined.

    References
    ----------
    .. [1] Yu, J. Y., Kay, K., Liu, D. F., Grossrubatscher, I., Loback, A.,
       Sosa, M., Chung, J. E., Karlsson, M. P., Larkin, M. C., & Frank, L. M.
       (2017). Distinct hippocampal-cortical memory representations for
       experiences associated with movement versus immobility. eLife, 6,
       e27621. doi:10.7554/eLife.27621

    """
    return noise_threshold_diagnostics(
        values, percentile, histogram_edges, mode_smoothing_window
    ).threshold


def noise_threshold_diagnostics(
    values: ArrayLike,
    percentile: float = 99.99,
    histogram_edges: ArrayLike | None = None,
    mode_smoothing_window: int = YU_MODE_SMOOTHING_WINDOW,
) -> NoiseThresholdDiagnostics:
    """:func:`estimate_noise_threshold` with everything it computed.

    Same arguments, same estimate, returned as a
    :class:`NoiseThresholdDiagnostics` alongside the histogram, its mode, the
    mirrored distribution and the sample counts, for inspecting why the
    threshold landed where it did. Raises and warns as
    :func:`estimate_noise_threshold` does.

    """
    values = np.asarray(values, dtype=float).ravel()
    edges = (
        YU_HISTOGRAM_EDGES
        if histogram_edges is None
        else np.asarray(histogram_edges, dtype=float).ravel()
    )
    if edges.ndim != 1 or len(edges) < 3 or np.any(np.diff(edges) <= 0):
        msg = "histogram_edges must be a strictly increasing 1-D array."
        raise ValueError(msg)
    if not 0.0 < percentile < 100.0:
        msg = f"percentile must be in (0, 100), got {percentile}."
        raise ValueError(msg)
    if len(values) == 0:
        msg = "values is empty; cannot estimate a noise threshold."
        raise ValueError(msg)

    in_grid = np.isfinite(values) & (values >= edges[0]) & (values <= edges[-1])
    out_of_grid_fraction = 1.0 - in_grid.sum() / len(values)
    if out_of_grid_fraction > _OUT_OF_GRID_CEILING:
        msg = (
            f"{out_of_grid_fraction:.3%} of samples are non-finite or outside the "
            f"histogram grid [{edges[0]}, {edges[-1]}]; the trace is not in the "
            "units the grid assumes."
        )
        raise ValueError(msg)

    counts = _histc(values[in_grid], edges)
    smoothed = _matlab_smooth(counts, mode_smoothing_window)
    mode_index = int(np.argmax(smoothed))  # first maximum, as MATLAB find(..., 1)
    if mode_index in (0, len(edges) - 1):
        msg = (
            "Histogram mode lies on the first or last bin of the grid; the "
            "mirrored distribution is degenerate."
        )
        raise ValueError(msg)
    mode = float(edges[mode_index])
    # The two reflections agree whenever every left-flank bin is non-positive,
    # so the warning is raised only when a bin above zero can contribute.
    if mode > 0 and np.any(edges[:mode_index] > 0):
        _warn_at_caller(
            f"Histogram mode is positive ({mode:.3f}); the original MATLAB "
            "reflection (abs(b) + 2m) would differ from the intended 2m - b "
            "used here."
        )

    left_positions = edges[: mode_index + 1]
    left_counts = counts[: mode_index + 1]
    reflected_positions = np.round(2.0 * mode - edges[:mode_index], 6)
    reflected_counts = counts[:mode_index]
    positions = np.concatenate([left_positions, reflected_positions])
    mirrored_counts = np.concatenate([left_counts, reflected_counts])
    order = np.argsort(positions, kind="stable")
    positions = positions[order]
    mirrored_counts = mirrored_counts[order]

    total = mirrored_counts.sum()
    limit = 1.0 - percentile / 100.0
    if total * limit < 1.0:
        msg = (
            f"Too few samples ({int(in_grid.sum())} in grid, {int(total)} in the "
            f"mirrored histogram) to resolve the {percentile} percentile; need at "
            f"least {int(np.ceil(1.0 / limit))} mirrored counts."
        )
        raise ValueError(msg)
    cdf = np.cumsum(mirrored_counts) / total
    crossing = int(np.argmax(cdf >= percentile / 100.0))
    if crossing + 1 >= len(positions):
        msg = (
            "The CDF crossing lands on the last mirrored bin, so the threshold "
            "(one bin past the crossing) is undefined."
        )
        raise ValueError(msg)
    threshold = float(positions[crossing + 1])

    in_grid_values = values[in_grid]
    mean = float(in_grid_values.mean())
    minimum = float(in_grid_values.min())
    return NoiseThresholdDiagnostics(
        threshold=threshold,
        mode=mode,
        mode_index=mode_index,
        mean=mean,
        min=minimum,
        flank_ratio=(mode - minimum) / (mean - mode) if mean > mode else np.inf,
        histogram_edges=edges,
        counts=counts,
        smoothed_counts=smoothed,
        mirrored_positions=positions,
        mirrored_counts=mirrored_counts,
        out_of_grid_fraction=float(out_of_grid_fraction),
        n_values=len(values),
        n_in_grid=int(in_grid.sum()),
    )


def _unit_area_gaussian(sigma_samples: float, n_sd: float) -> FloatArray:
    """Unit-area Gaussian kernel truncated at ``n_sd`` standard deviations
    (vandermeerlab ``gausskernel(R, S)`` with ``R = n_sd * S``)."""
    radius = int(np.ceil(n_sd * sigma_samples))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x**2) / (2.0 * sigma_samples**2))
    return np.asarray(kernel / kernel.sum(), dtype=float)


def get_multiunit_population_firing_rate(
    multiunit: ArrayLike, sampling_frequency: float, smoothing_sigma: float = 0.015
) -> FloatArray:
    """Calculates the multiunit population firing rate.

    Parameters
    ----------
    multiunit : array_like, shape (n_time, n_signals)
        Spike indicator matrix. Can be binary (0/1) or spike counts per bin.
    sampling_frequency : float
        Number of samples per second.
    smoothing_sigma : float, optional
        Standard deviation of the Gaussian smoothing kernel in seconds.
        Default is 0.015.


    Returns
    -------
    multiunit_population_firing_rate : ndarray, shape (n_time,)

    """
    multiunit = np.asarray(multiunit, dtype=float)
    return gaussian_smooth(
        multiunit.sum(axis=1) * sampling_frequency, smoothing_sigma, sampling_frequency
    )

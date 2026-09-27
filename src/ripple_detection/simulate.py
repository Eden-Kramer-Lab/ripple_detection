"""Simulation tools for generating synthetic LFP data with embedded ripples.

``simulate_LFP`` gives one channel. ``simulate_multichannel_LFP`` gives channels
that share a ripple and part of their noise, ``simulate_sharp_wave_ripple_pair``
the raw two-channel input of the Long detector, ``simulate_multiunit`` spike
trains that burst with the ripples, and ``simulate_session`` all of them at once
with the ground truth, for testing detectors against known events.
``draw_network_events`` draws latent network events of known types,
``draw_non_events`` activity a detector should not report,
``simulate_network_session`` renders both into every detector input, and
``truth_windows`` gives their windows at any fraction of each envelope's peak.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import signal, special

from ripple_detection._call_hints import explain_call_errors
from ripple_detection.core import (
    FloatArray,
    IntArray,
    StrArray,
    _bound_tolerance,
    _check_choice,
    _contiguous_valid_blocks,
    _generator,
    filter_ripple_band,
    ripple_bandpass_filter,
)
from ripple_detection.detectors._validation import _check_whole_number

RIPPLE_FREQUENCY = 200
NoiseType = Literal["white", "pink", "brown"]

EVENT_TYPES = ("swr", "weak_ripple", "burst_only", "ripple_doublet", "sharp_wave_only")
"""The kinds of latent network event ``draw_network_events`` draws."""

NON_EVENT_TYPES = ("spike_leakage", "emg", "fast_gamma", "theta_burst")
"""The kinds of activity a detector should not report, which
``draw_non_events`` draws."""

EXPRESSIONS = ("ripple", "sharp_wave", "burst")
"""How a network event shows: a ripple in the pyramidal-layer LFP, a sharp wave
in the stratum radiatum LFP, a population burst in the spikes."""

UNIT_TYPES = ("place", "pyramidal", "interneuron")
"""Unit labels in ``SimulatedSession.unit_types``; place units are pyramidal
units with place fields."""


def simulate_time(n_samples: int, sampling_frequency: float) -> FloatArray:
    """Generate time array for simulation.

    Parameters
    ----------
    n_samples : int
        Number of samples in the time series.
    sampling_frequency : float
        Sampling rate in Hz.

    Returns
    -------
    time : ndarray, shape (n_samples,)
        Time array in seconds, starting at 0.

    Examples
    --------
    >>> time = simulate_time(3000, 1500)
    >>> time.size, float(time[1])
    (3000, 0.0006666666666666666)

    """
    return np.arange(n_samples) / sampling_frequency


def mean_squared(x: FloatArray) -> float:
    """Calculate the mean squared value of a signal.

    Parameters
    ----------
    x : ndarray
        Input signal.

    Returns
    -------
    ms : float
        Mean of squared absolute values.

    """
    return float((np.abs(x) ** 2.0).mean())


def normalize(y: FloatArray, x: FloatArray | None = None) -> FloatArray:
    """Normalize signal power to match white noise or reference signal.

    Scales the signal `y` to have the same mean squared value as a standard
    normal white noise signal (power = 1) or optionally to match the power
    of a reference signal `x`.

    Parameters
    ----------
    y : ndarray
        Signal to be normalized.
    x : ndarray, optional
        Reference signal. If provided, `y` is normalized to match the power
        of `x`. If None, normalized to unit power (standard normal).
        Default is None.

    Returns
    -------
    normalized_signal : ndarray
        Signal with adjusted power, same shape as `y`.

    Notes
    -----
    The mean power of a Gaussian with mu=0 and sigma=1 is 1.

    If the input signal `y` has zero power (e.g., all zeros), the function
    will return NaN values due to division by zero. This is expected behavior,
    as zero-power signals cannot be meaningfully normalized. In practice, this
    edge case only occurs with artificial test inputs.

    References
    ----------
    Adapted from python-acoustics library.

    """
    reference_power = mean_squared(x) if x is not None else 1.0
    # np.divide, not /, so a zero-power signal gives NaN with a RuntimeWarning, as documented
    return np.asarray(y * np.sqrt(np.divide(reference_power, mean_squared(y))), dtype=float)


@explain_call_errors
def pink(N: int, rng: int | np.random.Generator | None = None) -> FloatArray:
    """Generate pink (1/f) noise.

    Pink noise has equal power in proportionally-wide frequency bands (octaves).
    Power spectral density decreases at 3 dB per octave (1/f spectrum).

    Parameters
    ----------
    N : int
        Number of samples to generate.
    rng : int or numpy.random.Generator, optional
        Seed, or a Generator to draw from. Default is None, a fresh
        unseeded Generator.

    Returns
    -------
    pink_noise : ndarray, shape (N,)
        Pink noise signal normalized to unit power.

    Notes
    -----
    Implementation uses frequency domain method with 1/sqrt(f) scaling.

    References
    ----------
    Adapted from python-acoustics library.

    """
    rng = _generator(rng)
    uneven = N % 2
    X = rng.standard_normal(N // 2 + 1 + uneven) + 1j * rng.standard_normal(
        N // 2 + 1 + uneven
    )
    S = np.sqrt(np.arange(len(X)) + 1.0)  # +1 to avoid divide by zero
    y = (np.fft.irfft(X / S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)


@explain_call_errors
def white(N: int, rng: int | np.random.Generator | None = None) -> FloatArray:
    """Generate white noise.

    White noise has constant power spectral density across all frequencies (flat
    spectrum). Power increases by 3 dB per octave when integrated over octave bands.

    Parameters
    ----------
    N : int
        Number of samples to generate.
    rng : int or numpy.random.Generator, optional
        Seed, or a Generator to draw from. Default is None, a fresh
        unseeded Generator.

    Returns
    -------
    white_noise : ndarray, shape (N,)
        White noise signal from standard normal distribution.

    """
    rng = _generator(rng)
    return rng.standard_normal(N)


@explain_call_errors
def brown(N: int, rng: int | np.random.Generator | None = None) -> FloatArray:
    """Generate brown (Brownian, red) noise.

    Brown noise has power spectral density that decreases at 6 dB per octave
    (1/f² spectrum). Power decreases at 3 dB per octave when integrated over
    octave bands.

    Parameters
    ----------
    N : int
        Number of samples to generate.
    rng : int or numpy.random.Generator, optional
        Seed, or a Generator to draw from. Default is None, a fresh
        unseeded Generator.

    Returns
    -------
    brown_noise : ndarray, shape (N,)
        Brown noise signal normalized to unit power.

    Notes
    -----
    Implementation uses frequency domain method with 1/f scaling.

    References
    ----------
    Adapted from python-acoustics library.

    """
    rng = _generator(rng)
    uneven = N % 2
    X = rng.standard_normal(N // 2 + 1 + uneven) + 1j * rng.standard_normal(
        N // 2 + 1 + uneven
    )
    S = np.arange(len(X)) + 1
    y = np.fft.irfft(X / S).real
    if uneven:
        y = y[:-1]
    return normalize(y)


NOISE_FUNCTION = {
    "white": white,
    "pink": pink,
    "brown": brown,
}


def _draw_per_ripple(
    value: float | tuple[float, float] | Sequence[float] | FloatArray,
    n_ripples: int,
    rng: np.random.Generator,
) -> FloatArray:
    """A scalar repeated per ripple, one uniform draw per ripple from a range,
    or an explicit value per ripple.

    A scalar consumes no randomness. A ``tuple`` is a ``(low, high)`` range
    and draws ``n_ripples`` values from ``rng``. A list or array holds one
    value per ripple and is used as given, so one draw can be shared between
    the functions of this module. The type, not the length, decides, so two
    values for two ripples are never mistaken for a range.
    """
    if isinstance(value, tuple):
        if len(value) != 2:
            msg = f"A range must be a (low, high) tuple, got {value}."
            raise ValueError(msg)
        low, high = (float(bound) for bound in value)
        if not low <= high:
            msg = f"Range must be (low, high) with low <= high, got {value}."
            raise ValueError(msg)
        return rng.uniform(low, high, size=n_ripples)
    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        return np.full(n_ripples, float(values))
    if values.shape != (n_ripples,):
        msg = (
            f"Give a scalar, a (low, high) tuple, or a list or array of one value per "
            f"ripple ({n_ripples}), got {value}."
        )
        raise ValueError(msg)
    return values


@explain_call_errors
def simulate_LFP(
    time: FloatArray,
    ripple_times: float | Sequence[float] | FloatArray,
    ripple_amplitude: float | None = None,
    ripple_duration: float | tuple[float, float] = 0.100,
    noise_type: NoiseType = "pink",
    noise_amplitude: float = 1.3,
    rng: int | np.random.Generator | None = None,
    *,
    ripple_snr: float | None = None,
    ripple_frequency: float | tuple[float, float] = RIPPLE_FREQUENCY,
    sampling_frequency: float | None = None,
) -> FloatArray:
    """Simulate local field potential with embedded ripple oscillations.

    Generates a synthetic LFP signal containing ripple events (sinusoids at
    ``ripple_frequency``) embedded in colored noise. Ripples are amplitude-
    modulated by a Gaussian envelope.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
        Time array in seconds.
    ripple_times : float or array_like of float
        Center time(s) of ripple event(s) in seconds.
    ripple_amplitude : float, optional
        Peak-to-peak amplitude of the ripple oscillation in the signal's units
        (the peak is half this). Default is 2 when ``ripple_snr`` is not
        given. Cannot be combined with ``ripple_snr``.
    ripple_duration : float or (float, float), optional
        Approximate duration in **seconds** of a ripple event, defined as 6
        standard deviations of its Gaussian envelope. A ``(low, high)`` tuple
        draws one duration per ripple uniformly from that range. Default is
        0.100 (100 ms).
    noise_type : {'white', 'pink', 'brown'}, optional
        Type of background noise. Default is 'pink' (1/f), whose ripple-band
        background is closest to recordings. Brown (1/f²) noise, the default
        before 2.0, leaves very little power in the 150-250 Hz band, and the
        fraction falls further as the record lengthens (it is a random walk),
        so ripples of any visible size dominate the band and every detector
        finds every one of them. See Notes.
    noise_amplitude : float, optional
        Amplitude of background noise in the signal's units. Default is 1.3.
    rng : int or numpy.random.Generator, optional
        Seed, or a Generator to draw from, as ``numpy.random.default_rng``
        takes it. The noise is drawn first, then per-ripple
        frequencies, then per-ripple durations; a scalar consumes no
        randomness. So a given seed produces the same noise whatever the
        ripple parameters, but giving a frequency range changes the duration
        draws. Default is None, which draws from the operating system and is
        not reproducible. `np.random.seed` does not control this function;
        pass `rng` to repeat a simulation.
    ripple_snr : float, optional
        Ripple size relative to the **ripple-band** background: the peak
        amplitude of each ripple after ``filter_ripple_band``, divided by the
        standard deviation of the filtered noise. Each ripple's amplitude is
        set from its own filtered peak, so the ratio holds at the band edges
        and for short bursts, where the filter attenuates. Requires
        ``noise_amplitude > 0`` and enough samples for ``filter_ripple_band``
        (as many samples as its kernel has taps). Cannot be combined with
        ``ripple_amplitude``. Default is None.
    ripple_frequency : float or (float, float), optional
        Ripple oscillation frequency in Hz, or a ``(low, high)`` tuple drawn
        uniformly per ripple. Default is 200.
    sampling_frequency : float, optional
        Sampling rate in Hz, used only with ``ripple_snr`` to filter the noise.
        Default is None, which takes it from the median step of ``time``.

    Returns
    -------
    lfp : ndarray, shape (n_time,)
        Simulated LFP signal with embedded ripples.

    Raises
    ------
    ValueError
        If both ``ripple_amplitude`` and ``ripple_snr`` are given, if
        ``ripple_snr`` is given with ``noise_amplitude = 0``, if a range is
        not a ``(low, high)`` tuple with ``low <= high``, if a ripple time lies
        outside ``time``, if a duration is not positive, if a frequency is not
        between zero and the Nyquist frequency, if ``ripple_snr`` is not
        positive, or if ``ripple_amplitude`` or ``noise_amplitude`` is negative
        or NaN. Each of these would otherwise give an all-NaN, empty, or
        aliased ripple with no error.

    Notes
    -----
    ``ripple_snr`` is defined on the filtered signal, not on the z-scored
    envelope or consensus trace a detector thresholds. Those are smoothed,
    which lowers the background's spread more than a ripple's peak, so the
    z-score a detector sees is larger than ``ripple_snr`` by a factor that
    depends on the detector's smoothing and consensus rule and on how much of
    the record the ripples occupy. Measure it for the detector in use rather
    than assuming a fixed mapping.

    The Gaussian envelope has sigma = ripple_duration / 6, so the ripple
    amplitude decays to ~1% at +/-3*sigma from the center.

    Examples
    --------
    >>> time = simulate_time(3000, 1000)  # 3 seconds at 1000 Hz
    >>> lfp = simulate_LFP(time, [1.0, 2.0], rng=0)

    Ripples five times the ripple-band background, varying in frequency and
    duration, on a pink-noise background:

    >>> time = simulate_time(15000, 1500)
    >>> lfp = simulate_LFP(
    ...     time, [2.0, 5.0, 8.0], noise_type='pink', ripple_snr=5,
    ...     ripple_frequency=(150, 250), ripple_duration=(0.04, 0.12), rng=0,
    ... )

    """
    _validate_sizes(ripple_amplitude, ripple_snr, noise_amplitude)
    rng = _generator(rng)
    noise = (noise_amplitude / 2) * NOISE_FUNCTION[noise_type](time.size, rng=rng)
    return noise + _ripple_waveform(
        time,
        _as_ripple_times(ripple_times),
        noise,
        rng,
        ripple_amplitude=ripple_amplitude,
        ripple_snr=ripple_snr,
        ripple_duration=ripple_duration,
        ripple_frequency=ripple_frequency,
        noise_amplitude=noise_amplitude,
        sampling_frequency=sampling_frequency,
    )


def _as_ripple_times(ripple_times: float | Sequence[float] | FloatArray) -> FloatArray:
    """One ripple time or several, as a 1-D float array; NumPy scalars included."""
    return np.atleast_1d(np.asarray(ripple_times, dtype=float))


def _channel_gains(
    channel_gains: Sequence[float] | FloatArray | None, n_channels: int
) -> FloatArray:
    """Each channel's ripple gain, 1 by default, checked against the channel count."""
    gains = (
        np.ones(n_channels)
        if channel_gains is None
        else np.asarray(channel_gains, dtype=float)
    )
    if gains.shape != (n_channels,):
        msg = f"channel_gains must have shape ({n_channels},), got {gains.shape}."
        raise ValueError(msg)
    return gains


def _ripple_waveform(
    time: FloatArray,
    ripple_times: FloatArray,
    reference_noise: FloatArray,
    rng: np.random.Generator,
    *,
    ripple_amplitude: float | None,
    ripple_snr: float | None,
    ripple_duration: float | tuple[float, float] | FloatArray,
    ripple_frequency: float | tuple[float, float] | FloatArray,
    noise_amplitude: float,
    sampling_frequency: float | None,
) -> FloatArray:
    """The ripple bursts alone, zero elsewhere, shape (n_time,).

    Draws the per-ripple frequencies, then durations, from ``rng``. With
    ``ripple_snr`` each burst is sized against the ripple-band spread of
    ``reference_noise``; otherwise its peak is half ``ripple_amplitude``
    (default 2).
    """
    rate = _sampling_rate(time, sampling_frequency)
    band_noise_sd = np.nan
    if ripple_snr is not None:
        if noise_amplitude <= 0:
            msg = "ripple_snr needs a background: noise_amplitude must be > 0."
            raise ValueError(msg)
        band_noise_sd = float(
            filter_ripple_band(reference_noise, sampling_frequency=rate).std()
        )
    frequencies = _draw_per_ripple(ripple_frequency, ripple_times.size, rng)
    durations = _draw_per_ripple(ripple_duration, ripple_times.size, rng)
    _validate_ripples(time, ripple_times, frequencies, durations)
    ripple = np.zeros(time.size)
    _add_ripple_bursts(
        ripple,
        time,
        ripple_times,
        frequencies,
        durations,
        amplitude=2.0 if ripple_amplitude is None else ripple_amplitude,
        ripple_snr=ripple_snr,
        band_noise_sd=band_noise_sd,
        rate=rate,
    )
    return ripple


def _sampling_rate(time: FloatArray, sampling_frequency: float | None) -> float:
    return (
        float(1.0 / np.median(np.diff(time)))
        if sampling_frequency is None
        else float(sampling_frequency)
    )


def _validate_sizes(
    ripple_amplitude: float | None, ripple_snr: float | None, noise_amplitude: float
) -> None:
    """Raise for ripple and noise sizes that would put NaN in the signal or flip
    the ripple, which the detectors would then read as missing samples."""
    if ripple_amplitude is not None and ripple_snr is not None:
        msg = "Give either ripple_amplitude or ripple_snr, not both."
        raise ValueError(msg)
    if ripple_snr is not None and not ripple_snr > 0:
        msg = f"ripple_snr must be positive, got {ripple_snr}."
        raise ValueError(msg)
    if ripple_amplitude is not None and not ripple_amplitude >= 0:
        msg = f"ripple_amplitude must be non-negative, got {ripple_amplitude}."
        raise ValueError(msg)
    if not noise_amplitude >= 0:
        msg = f"noise_amplitude must be non-negative, got {noise_amplitude}."
        raise ValueError(msg)


def _validate_ripples(
    time: FloatArray,
    ripple_times: Sequence[float] | FloatArray,
    frequencies: FloatArray,
    durations: FloatArray,
) -> None:
    """Raise for a ripple that would be all-NaN, empty or aliased without an error."""
    if len(ripple_times) == 0:
        return
    nyquist = 0.5 / np.median(np.diff(time))
    outside = [t for t in ripple_times if not time.min() <= t <= time.max()]
    if outside:
        msg = (
            f"ripple_times {outside} lie outside time "
            f"[{time.min()}, {time.max()}]; the ripple would have no samples."
        )
        raise ValueError(msg)
    if not np.all(durations > 0):
        msg = f"ripple_duration must be positive, got {durations}."
        raise ValueError(msg)
    if not np.all((frequencies > 0) & (frequencies < nyquist)):
        msg = (
            f"ripple_frequency must lie in (0, {nyquist:.1f}) Hz, the Nyquist range of "
            f"time's sampling rate, got {frequencies}."
        )
        raise ValueError(msg)


def _sample_window(time: FloatArray, start: float, end: float) -> slice:
    """The samples in ``[start, end)``; at least one, so a bump narrower than a
    step still lands somewhere."""
    first, last = np.searchsorted(time, [start, end])
    if last <= first:
        last = min(first + 1, time.size)
        first = last - 1
    return slice(int(first), int(last))


def _gaussian_window(
    time: FloatArray, center: float, sigma: float, n_sigma: float
) -> tuple[slice, FloatArray]:
    """The samples within ``n_sigma`` of ``center`` and the unit-peak Gaussian there;
    at least one sample, so a bump narrower than a step still lands somewhere."""
    window = _sample_window(time, center - n_sigma * sigma, center + n_sigma * sigma)
    envelope = np.exp(-((time[window] - center) ** 2) / (2.0 * sigma**2))
    return window, np.asarray(envelope, dtype=float)


def _add_ripple_bursts(
    out: FloatArray,
    time: FloatArray,
    ripple_times: Sequence[float] | FloatArray,
    frequencies: FloatArray,
    durations: FloatArray,
    *,
    amplitude: float,
    ripple_snr: float | None,
    band_noise_sd: float,
    rate: float,
) -> None:
    """Add one Gaussian-windowed sine burst per ripple to ``out`` in place.

    Each burst is evaluated over the samples within 8 sigma of its centre,
    where its envelope is above 1e-14 of the peak, so memory does not grow
    with the ripple count. With ``ripple_snr`` the burst is scaled so that its
    peak after ``filter_ripple_band`` is ``ripple_snr`` times ``band_noise_sd``;
    otherwise its peak is ``amplitude / 2``.
    """
    for ripple_time, frequency, duration in zip(
        ripple_times, frequencies, durations, strict=True
    ):
        window, carrier = _gaussian_window(time, ripple_time, duration / 6, 8.0)
        # phase from the ripple's centre, not the clock, so a ripple looks the
        # same at any time origin; the cosine puts the unit peak at the centre
        burst = np.cos(2 * np.pi * frequency * (time[window] - ripple_time)) * carrier
        if ripple_snr is not None:
            scale = _scale_to_snr(burst, ripple_snr, band_noise_sd, rate)
        else:
            scale = amplitude / 2
        out[window] += scale * burst


def _scale_to_snr(
    burst: FloatArray,
    snr: float,
    band_noise_sd: float,
    rate: float,
    band: tuple[float, float] | None = None,
) -> float:
    """The factor that makes ``burst``'s peak *after the filter* ``snr`` times
    ``band_noise_sd``; the filter's gain depends on frequency and duration."""
    return float(snr * band_noise_sd / _filtered_peak(burst, rate, band))


def _filtered_peak(
    burst: FloatArray, rate: float, band: tuple[float, float] | None = None
) -> float:
    """``burst``'s peak magnitude after ``filter_ripple_band`` (``band`` None:
    the ripple band, the shipped kernel at 1500 Hz). A second of zeros each
    side makes the run long enough for the kernel at any rate, and is what
    the burst is surrounded by in the record."""
    n_pad = int(np.ceil(rate))
    padded = np.zeros(burst.size + 2 * n_pad)
    padded[n_pad : n_pad + burst.size] = burst
    return float(np.abs(filter_ripple_band(padded, sampling_frequency=rate, band=band)).max())


def _correlated_noise(
    n_time: int,
    n_channels: int,
    noise_type: NoiseType,
    noise_amplitude: float,
    shared_noise_fraction: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Channels of unit-power coloured noise, a fraction of whose power is one
    component they all share, scaled as ``simulate_LFP`` scales its noise."""
    if not 0.0 <= shared_noise_fraction <= 1.0:
        msg = f"shared_noise_fraction must lie in [0, 1], got {shared_noise_fraction}."
        raise ValueError(msg)
    shared = NOISE_FUNCTION[noise_type](n_time, rng=rng)
    noise = np.empty((n_time, n_channels))
    for channel in range(n_channels):
        own = NOISE_FUNCTION[noise_type](n_time, rng=rng)
        noise[:, channel] = (noise_amplitude / 2) * (
            np.sqrt(shared_noise_fraction) * shared
            + np.sqrt(1.0 - shared_noise_fraction) * own
        )
    return noise


def _add_common_mode_artifacts(
    out: FloatArray,
    time: FloatArray,
    artifact_times: Sequence[float],
    amplitude: float,
    duration: float,
    rng: np.random.Generator,
) -> None:
    """Add a broadband burst, identical on every channel, at each artifact time."""
    for artifact_time in artifact_times:
        window, envelope = _gaussian_window(time, artifact_time, duration / 6, 4.0)
        burst = amplitude * rng.standard_normal(envelope.size) * envelope
        out[window] += burst[:, np.newaxis]


@explain_call_errors
def simulate_multichannel_LFP(
    time: FloatArray,
    ripple_times: float | Sequence[float] | FloatArray,
    n_channels: int,
    *,
    channel_gains: Sequence[float] | FloatArray | None = None,
    shared_noise_fraction: float = 0.5,
    ripple_amplitude: float | None = None,
    ripple_snr: float | None = None,
    ripple_duration: float | tuple[float, float] | FloatArray = 0.100,
    ripple_frequency: float | tuple[float, float] | FloatArray = RIPPLE_FREQUENCY,
    noise_type: NoiseType = "pink",
    noise_amplitude: float = 1.3,
    artifact_times: Sequence[float] | None = None,
    artifact_amplitude: float | None = None,
    artifact_duration: float = 0.100,
    rng: int | np.random.Generator | None = None,
    sampling_frequency: float | None = None,
) -> FloatArray:
    """Simulate LFP channels that see the same ripples in correlated noise.

    Every channel carries the same ripple waveform, scaled by its gain, as
    channels of one hippocampal array do; each channel's noise is part a
    component shared by all channels, as volume-conducted signal is, and part
    its own. Optional broadband bursts identical on every channel stand in
    for the chewing and muscle artifacts that produce most false positives in
    recordings.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
        Time array in seconds.
    ripple_times : float or array_like of float
        Center time(s) of ripple event(s) in seconds.
    n_channels : int
        Number of channels.
    channel_gains : sequence of float, shape (n_channels,), optional
        Each channel's ripple amplitude relative to ``ripple_amplitude`` or
        ``ripple_snr``. Default None: every channel at 1. A recording's
        channels differ by their distance from the pyramidal layer;
        ``rng.uniform(0.5, 1.0, n_channels)`` is a fair draw.
    shared_noise_fraction : float, optional
        Fraction of each channel's noise power that is the shared component;
        the correlation between two channels' noise. Default 0.5.
    ripple_amplitude, ripple_snr, ripple_duration, ripple_frequency : optional
        As in ``simulate_LFP``, for a channel of gain 1. ``ripple_duration``
        and ``ripple_frequency`` also accept a list or array of one value per
        ripple, so the same draw can be given to ``simulate_multiunit``; a
        ``tuple`` is always a ``(low, high)`` range.
    noise_type : {'white', 'pink', 'brown'}, optional
        Default 'pink'.
    noise_amplitude : float, optional
        Amplitude of each channel's noise, as in ``simulate_LFP``. Default 1.3.
    artifact_times : sequence of float, optional
        Centres of common-mode broadband bursts. Default None: none.
    artifact_amplitude : float, optional
        Standard deviation of an artifact's white noise at its peak, in the
        signal's units. Default None: three times ``noise_amplitude``.
    artifact_duration : float, optional
        Artifact duration in seconds, six standard deviations of its Gaussian
        envelope. Default 0.100.
    rng : int or numpy.random.Generator, optional
        As in ``simulate_LFP``. The shared noise is drawn first, then each
        channel's own noise, then per-ripple frequencies and durations, then
        the artifacts.
    sampling_frequency : float, optional
        As in ``simulate_LFP``.

    Returns
    -------
    lfps : ndarray, shape (n_time, n_channels)

    Raises
    ------
    ValueError
        As ``simulate_LFP`` raises, and if ``channel_gains`` has the wrong
        length or ``shared_noise_fraction`` lies outside [0, 1].

    Examples
    --------
    >>> time = simulate_time(15000, 1500)
    >>> lfps = simulate_multichannel_LFP(
    ...     time, [2.0, 5.0, 8.0], 4, ripple_snr=4, channel_gains=[1.0, 0.8, 0.6, 0.5],
    ...     ripple_frequency=(150, 250), ripple_duration=(0.04, 0.12), rng=0,
    ... )
    >>> lfps.shape
    (15000, 4)

    """
    _validate_sizes(ripple_amplitude, ripple_snr, noise_amplitude)
    if n_channels < 1:
        msg = f"n_channels must be at least 1, got {n_channels}."
        raise ValueError(msg)
    gains = _channel_gains(channel_gains, n_channels)
    rng = _generator(rng)
    lfps = _correlated_noise(
        time.size, n_channels, noise_type, noise_amplitude, shared_noise_fraction, rng
    )
    ripple = _ripple_waveform(
        time,
        _as_ripple_times(ripple_times),
        lfps[:, 0],
        rng,
        ripple_amplitude=ripple_amplitude,
        ripple_snr=ripple_snr,
        ripple_duration=ripple_duration,
        ripple_frequency=ripple_frequency,
        noise_amplitude=noise_amplitude,
        sampling_frequency=sampling_frequency,
    )
    lfps += ripple[:, np.newaxis] * gains
    if artifact_times is not None and len(artifact_times):
        _add_common_mode_artifacts(
            lfps,
            time,
            artifact_times,
            3.0 * noise_amplitude if artifact_amplitude is None else artifact_amplitude,
            artifact_duration,
            rng,
        )
    return lfps


def _add_sharp_waves(
    out: FloatArray,
    time: FloatArray,
    ripple_times: Sequence[float] | FloatArray,
    amplitude: float,
    duration: float,
) -> None:
    """Add a Gaussian deflection of peak ``amplitude`` (signed) at each ripple."""
    for ripple_time in ripple_times:
        window, envelope = _gaussian_window(time, ripple_time, duration / 6, 6.0)
        out[window] += amplitude * envelope


def _add_sharp_wave_pair(
    ripple_channel: FloatArray,
    radiatum_channel: FloatArray,
    time: FloatArray,
    ripple_times: FloatArray,
    amplitude: float,
    duration: float,
    leak: float,
) -> None:
    """The sharp wave under each ripple: negative on the radiatum channel,
    and ``leak`` of it, positive, on the ripple channel. In place."""
    _add_sharp_waves(ripple_channel, time, ripple_times, leak * amplitude, duration)
    _add_sharp_waves(radiatum_channel, time, ripple_times, -amplitude, duration)


@explain_call_errors
def simulate_sharp_wave_ripple_pair(
    time: FloatArray,
    ripple_times: float | Sequence[float] | FloatArray,
    *,
    sharp_wave_amplitude: float = 2.0,
    sharp_wave_duration: float = 0.080,
    sharp_wave_leak: float = 0.3,
    ripple_leak: float = 0.3,
    shared_noise_fraction: float = 0.5,
    ripple_amplitude: float | None = None,
    ripple_snr: float | None = None,
    ripple_duration: float | tuple[float, float] | FloatArray = 0.100,
    ripple_frequency: float | tuple[float, float] | FloatArray = RIPPLE_FREQUENCY,
    noise_type: NoiseType = "pink",
    noise_amplitude: float = 1.3,
    rng: int | np.random.Generator | None = None,
    sampling_frequency: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Simulate the two raw channels ``Long_sharp_wave_ripple_detector`` takes.

    ``raw_lfp`` is a pyramidal-layer channel: the ripple at full amplitude on
    a small positive sharp-wave deflection. ``sharp_wave_lfp`` is a stratum
    radiatum channel: the sharp wave as a negative Gaussian deflection
    centred on each ripple, with a weak copy of the ripple. The two channels'
    noise is correlated as in ``simulate_multichannel_LFP``.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    ripple_times : float or array_like of float
    sharp_wave_amplitude : float, optional
        Peak of the radiatum deflection, in the signal's units. Default 2.0,
        about three standard deviations of the default noise.
    sharp_wave_duration : float, optional
        Six standard deviations of the deflection's Gaussian, in seconds.
        Default 0.080.
    sharp_wave_leak : float, optional
        Fraction of the sharp wave that appears, with positive sign, on the
        pyramidal channel. Default 0.3.
    ripple_leak : float, optional
        Fraction of the ripple that appears on the radiatum channel. Default 0.3.
    shared_noise_fraction, ripple_amplitude, ripple_snr, ripple_duration,
    ripple_frequency, noise_type, noise_amplitude, rng,
    sampling_frequency : optional
        As in ``simulate_multichannel_LFP``.

    Returns
    -------
    raw_lfp : ndarray, shape (n_time,)
        The pyramidal-layer channel, raw.
    sharp_wave_lfp : ndarray, shape (n_time,)
        The stratum radiatum channel, raw.

    Examples
    --------
    >>> time = simulate_time(15_000, 1500)
    >>> raw_lfp, sharp_wave_lfp = simulate_sharp_wave_ripple_pair(time, [5.0], rng=0)
    >>> raw_lfp.shape, sharp_wave_lfp.shape
    ((15000,), (15000,))

    """
    pair = simulate_multichannel_LFP(
        time,
        ripple_times,
        2,
        channel_gains=[1.0, ripple_leak],
        shared_noise_fraction=shared_noise_fraction,
        ripple_amplitude=ripple_amplitude,
        ripple_snr=ripple_snr,
        ripple_duration=ripple_duration,
        ripple_frequency=ripple_frequency,
        noise_type=noise_type,
        noise_amplitude=noise_amplitude,
        rng=rng,
        sampling_frequency=sampling_frequency,
    )
    _add_sharp_wave_pair(
        pair[:, 0],
        pair[:, 1],
        time,
        _as_ripple_times(ripple_times),
        sharp_wave_amplitude,
        sharp_wave_duration,
        sharp_wave_leak,
    )
    return pair[:, 0].copy(), pair[:, 1].copy()


@explain_call_errors
def simulate_multiunit(
    time: FloatArray,
    ripple_times: float | Sequence[float] | FloatArray,
    n_units: int,
    *,
    baseline_rate: float | tuple[float, float] | FloatArray = (0.5, 5.0),
    ripple_rate_gain: float = 8.0,
    participation: float = 0.6,
    ripple_duration: float | tuple[float, float] | FloatArray = 0.100,
    rng: int | np.random.Generator | None = None,
) -> FloatArray:
    """Simulate spike counts per sample for units that burst during ripples.

    Each unit fires as a Poisson process at its baseline rate. In each ripple
    a random subset of units participates; a participating unit's rate is
    multiplied by ``1 + (ripple_rate_gain - 1) * envelope``, where the
    envelope is the ripple's Gaussian (six standard deviations span
    ``ripple_duration``), so its peak rate is ``ripple_rate_gain`` times its
    baseline.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    ripple_times : float or array_like of float
    n_units : int
    baseline_rate : float, (low, high) or array of shape (n_units,), optional
        Baseline rate in spikes per second: one value for every unit, a
        ``(low, high)`` tuple to draw each unit's from, or a list or array of
        one per unit. Default (0.5, 5.0).
    ripple_rate_gain : float, optional
        Peak rate during a ripple relative to baseline. Default 8.0.
    participation : float, optional
        Probability that a unit takes part in a given ripple. Default 0.6.
    ripple_duration : float, (low, high) or array of shape (n_ripples,), optional
        As in ``simulate_LFP``; pass the LFP simulation's draw to couple them.
    rng : int or numpy.random.Generator, optional
        Baseline rates are drawn first, then durations, then participation,
        then the spike counts.

    Returns
    -------
    multiunit : ndarray, shape (n_time, n_units)
        Spike counts per sample, almost all 0 or 1 at ordinary rates.

    """
    if n_units < 1:
        msg = f"n_units must be at least 1, got {n_units}."
        raise ValueError(msg)
    if not 0.0 <= participation <= 1.0:
        msg = f"participation must lie in [0, 1], got {participation}."
        raise ValueError(msg)
    if ripple_rate_gain < 1.0:
        msg = f"ripple_rate_gain must be at least 1, got {ripple_rate_gain}."
        raise ValueError(msg)
    rng = _generator(rng)
    ripple_times = _as_ripple_times(ripple_times)
    n_ripples = ripple_times.size
    rates = _draw_per_ripple(baseline_rate, n_units, rng)
    if not np.all(rates >= 0):
        msg = f"baseline_rate must be non-negative, got {rates}."
        raise ValueError(msg)
    durations = _draw_per_ripple(ripple_duration, n_ripples, rng)
    if not np.all(durations > 0):
        msg = f"ripple_duration must be positive, got {durations}."
        raise ValueError(msg)
    participates = rng.random((n_ripples, n_units)) < participation
    modulation = np.ones((time.size, n_units))
    for ripple_time, duration, units in zip(
        ripple_times, durations, participates, strict=True
    ):
        window, envelope = _gaussian_window(time, ripple_time, duration / 6, 4.0)
        modulation[window, units] += (ripple_rate_gain - 1.0) * envelope[:, np.newaxis]
    step = float(np.median(np.diff(time)))
    return np.asarray(rng.poisson(rates * step * modulation), dtype=float)


def _running_intervals(running_intervals: ArrayLike) -> FloatArray:
    """``(n, 2)`` running bouts, each finite with its start before its end,
    sorted and not overlapping."""
    bouts = np.asarray(running_intervals, dtype=float)
    if bouts.size == 0:
        return np.empty((0, 2))
    if bouts.ndim != 2 or bouts.shape[1] != 2:
        msg = f"running_intervals must be (start, end) pairs, shape (n, 2); got {bouts.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(bouts)) or np.any(bouts[:, 1] <= bouts[:, 0]):
        msg = "Each running interval needs a finite start before its end."
        raise ValueError(msg)
    if np.any(bouts[1:, 0] < bouts[:-1, 1]):
        msg = "running_intervals must be sorted and must not overlap."
        raise ValueError(msg)
    return bouts


@explain_call_errors
def simulate_speed(
    time: FloatArray,
    running_intervals: ArrayLike,
    *,
    peak_speed: float = 30.0,
    still_speed: float = 0.0,
) -> FloatArray:
    """A speed trace of still periods and running bouts.

    Speed is ``still_speed`` outside the bouts and rises and falls smoothly
    within each, as ``sin(pi * phase) ** 2``, to ``peak_speed`` at its middle,
    so each bout begins and ends slow, as a real one does: at the defaults
    the first and last 12% of a bout are under 4 cm/s.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
        Sample timestamps in seconds.
    running_intervals : array_like, shape (n_bouts, 2)
        Start and end of each bout in seconds, sorted, not overlapping.
    peak_speed : float, optional
        Speed at the middle of each bout, cm/s. Default 30.
    still_speed : float, optional
        Speed outside the bouts, cm/s. Default 0.

    Returns
    -------
    speed : ndarray, shape (n_time,)

    Raises
    ------
    ValueError
        If a bout's start is not before its end, the bouts overlap or are
        unsorted, or a speed is negative or not finite.

    Examples
    --------
    >>> time = simulate_time(10000, 1000)
    >>> speed = simulate_speed(time, [(2.0, 4.0)])
    >>> float(speed[1000]), float(speed[3000])
    (0.0, 30.0)

    """
    for name, value in (("peak_speed", peak_speed), ("still_speed", still_speed)):
        if not 0 <= value < np.inf:
            msg = f"{name} must be finite and non-negative, got {value}."
            raise ValueError(msg)
    time = np.asarray(time, dtype=float)
    speed = np.full(time.size, float(still_speed))
    for start, end in _running_intervals(running_intervals):
        inside = (time >= start) & (time <= end)
        phase = (time[inside] - start) / (end - start)
        speed[inside] = still_speed + (peak_speed - still_speed) * np.sin(np.pi * phase) ** 2
    return speed


@explain_call_errors
def simulate_theta_delta(
    time: FloatArray,
    running_intervals: ArrayLike,
    *,
    theta_amplitude: float = 1.0,
    theta_frequency: float = 8.0,
    delta_amplitude: float = 1.0,
    delta_frequency: float = 2.0,
    transition: float = 0.5,
) -> FloatArray:
    """A slow LFP component: theta while running, delta at rest.

    A sine at ``theta_frequency`` inside the running bouts and one at
    ``delta_frequency`` outside them, cross-faded with a raised cosine over
    ``transition`` seconds at each bout's edges, inside the bout. Add it to a
    simulated LFP to give ``theta_delta_ratio`` and the state rules
    something to find; the ripple band is far above both.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
        Sample timestamps in seconds.
    running_intervals : array_like, shape (n_bouts, 2)
        Start and end of each bout in seconds, sorted, not overlapping.
    theta_amplitude, delta_amplitude : float, optional
        Peak amplitude of each rhythm in signal units. Default 1.
    theta_frequency, delta_frequency : float, optional
        In Hz. Defaults 8 and 2.
    transition : float, optional
        Seconds over which the rhythms cross-fade at a bout's edges; the
        bout's first and last ``transition`` seconds, or half the bout if it
        is shorter. Default 0.5.

    Returns
    -------
    signal : ndarray, shape (n_time,)

    Raises
    ------
    ValueError
        If an amplitude is negative, a frequency or ``transition`` is not
        positive, or the intervals are not sorted, non-overlapping bouts.

    Examples
    --------
    >>> time = simulate_time(20000, 1000)
    >>> slow = simulate_theta_delta(time, [(5.0, 15.0)], theta_amplitude=3.0)
    >>> round(float(np.abs(slow[9000:11000]).max()), 1)
    3.0

    """
    for name, value in (
        ("theta_amplitude", theta_amplitude),
        ("delta_amplitude", delta_amplitude),
    ):
        if not 0 <= value < np.inf:
            msg = f"{name} must be finite and non-negative, got {value}."
            raise ValueError(msg)
    for name, value in (
        ("theta_frequency", theta_frequency),
        ("delta_frequency", delta_frequency),
        ("transition", transition),
    ):
        if not 0 < value < np.inf:
            msg = f"{name} must be positive and finite, got {value}."
            raise ValueError(msg)
    time = np.asarray(time, dtype=float)
    running = np.zeros(time.size)
    for start, end in _running_intervals(running_intervals):
        ramp = min(transition, (end - start) / 2)
        inside = (time >= start) & (time <= end)
        edge_distance = np.minimum(time[inside] - start, end - time[inside])
        running[inside] = 0.5 - 0.5 * np.cos(np.pi * np.clip(edge_distance / ramp, 0.0, 1.0))
    theta = theta_amplitude * np.sin(2 * np.pi * theta_frequency * time)
    delta = delta_amplitude * np.sin(2 * np.pi * delta_frequency * time)
    signal: FloatArray = running * theta + (1.0 - running) * delta
    return signal


_EVENT_COLUMNS: dict[str, type | str] = {
    "event_id": "int64",
    "event_type": str,
    "expression": str,
    "component": "int64",
    "center_time": "float64",
    "rise_sigma": "float64",
    "decay_sigma": "float64",
    "envelope_power": "int64",
    "amplitude": "float64",
    "frequency_start": "float64",
    "frequency_end": "float64",
    "participation": "float64",
    "n_participants": "int64",
}
_NON_EVENT_COLUMNS: dict[str, type | str] = {
    "non_event_id": "int64",
    "non_event_type": str,
    "center_time": "float64",
    "rise_sigma": "float64",
    "decay_sigma": "float64",
    "envelope_power": "int64",
    "amplitude": "float64",
    "frequency": "float64",
    "snr_band_low": "float64",
    "snr_band_high": "float64",
    "channel": "int64",
    "n_units": "int64",
    "n_spikes": "int64",
    "isi": "float64",
}
_RIPPLE_CHANNEL_COLUMNS: dict[str, type | str] = {
    "event_id": "int64",
    "component": "int64",
    "channel": "int64",
    "gain": "float64",
    "delay_s": "float64",
}


def _table(columns: dict[str, type | str], values: Mapping[str, ArrayLike]) -> pd.DataFrame:
    """A frame with exactly ``columns``, in order, cast to their dtypes, so an
    empty table and a filled one agree under every pandas version (``str``
    is ``object`` before pandas 3 and the string dtype from it). An integer
    column must hold whole numbers: a cast that would truncate raises."""
    frame = pd.DataFrame({name: np.asarray(values[name]) for name in columns})
    for name, dtype in columns.items():
        if dtype == "int64" and len(frame):
            number = frame[name].to_numpy(dtype=float)
            if not np.all(np.isfinite(number) & (number == np.round(number))):
                msg = f"{name} must hold whole numbers, got {frame[name].tolist()[:5]}."
                raise ValueError(msg)
    return frame.astype(columns)


def _empty(columns: dict[str, type | str]) -> pd.DataFrame:
    """A table with ``columns`` and no rows."""
    return _table(columns, {name: [] for name in columns})


@dataclass(frozen=True, eq=False)
class SimulatedSession:
    """Every signal ``simulate_session`` or ``simulate_network_session``
    produced, with the ground truth.

    Attributes
    ----------
    time : ndarray, shape (n_time,)
    lfps : ndarray, shape (n_time, n_channels)
        Raw multichannel LFP, the ripple channel first; filter it with
        ``filter_ripple_band`` before the ripple-band detectors.
    raw_lfp : ndarray, shape (n_time,)
        The ripple channel, ``lfps[:, 0]``, for ``Long_sharp_wave_ripple_detector``.
    sharp_wave_lfp : ndarray, shape (n_time,)
        The stratum radiatum channel, unfiltered, for the same detector's
        ``sharp_wave_lfp``.
    multiunit : ndarray, shape (n_time, n_units)
        Spike counts per sample.
    speed : ndarray, shape (n_time,)
        Speed in cm/s: 0 throughout (an immobile animal) unless
        ``running_intervals`` was given, then ``simulate_speed``'s bouts,
        rising to ``peak_speed``, and 0 between them.
    ripple_times, ripple_durations, ripple_frequencies : ndarray, shape (n_ripples,)
        Centre, duration (six standard deviations of the envelope) and
        frequency of each ripple, in the order the ripples were given. For a
        network session, one entry per ripple component of ``events``: the
        middle of its span ``[center_time - 3 rise_sigma, center_time + 3
        decay_sigma]``, the span's length and ``frequency_start``, so
        ``ripple_windows`` gives each ripple's span.
    artifact_times : ndarray, shape (n_artifacts,)
    sampling_frequency : float
    events : pandas.DataFrame
        The latent event table, one row per component (ripple, sharp wave,
        burst) of each network event; see ``draw_network_events``. Empty,
        with the same columns and dtypes, for ``simulate_session``.
    non_events : pandas.DataFrame
        One row per non-event, activity a detector should not report; see
        ``draw_non_events``. Empty, with the same columns and dtypes, for
        ``simulate_session`` and when none were rendered.
    unit_types : ndarray of str, shape (n_units,)
        Each unit's type, one of ``UNIT_TYPES``. Empty when the simulator did
        not assign types (``simulate_session``).
    baseline_rates : ndarray, shape (n_units,)
        Each unit's drawn baseline intensity in spikes/s, before event
        modulation; realized rates can be lower under refractory spiking.
        Empty when the simulator did not record them (``simulate_session``).
    running_intervals : ndarray, shape (n_bouts, 2)
        The running bouts, start and end in seconds; shape (0, 2) when the
        animal is still throughout.
    ripple_channels : pandas.DataFrame
        One row per ripple component and pyramidal-layer channel, sorted by
        ``event_id``, ``component``, ``channel``: the ``gain`` the ripple has
        on that channel (the recording-wide channel gain included; 0 where the
        ripple is absent) and its ``delay_s``. The ripple's bounds on a channel
        are its latent bounds plus the delay. Empty for ``simulate_session``.

    Raises
    ------
    ValueError
        If the signals do not share ``time``'s length, the per-ripple arrays
        do not share one length, or ``unit_types`` or ``baseline_rates`` is
        neither empty nor one entry per unit.

    """

    time: FloatArray
    lfps: FloatArray
    raw_lfp: FloatArray
    sharp_wave_lfp: FloatArray
    multiunit: FloatArray
    speed: FloatArray
    ripple_times: FloatArray
    ripple_durations: FloatArray
    ripple_frequencies: FloatArray
    artifact_times: FloatArray
    sampling_frequency: float
    events: pd.DataFrame = field(default_factory=partial(_empty, _EVENT_COLUMNS))
    non_events: pd.DataFrame = field(default_factory=partial(_empty, _NON_EVENT_COLUMNS))
    unit_types: StrArray = field(default_factory=lambda: np.empty(0, dtype="<U11"))
    baseline_rates: FloatArray = field(default_factory=lambda: np.empty(0))
    running_intervals: FloatArray = field(default_factory=lambda: np.empty((0, 2)))
    ripple_channels: pd.DataFrame = field(
        default_factory=partial(_empty, _RIPPLE_CHANNEL_COLUMNS)
    )

    def __post_init__(self) -> None:
        n_time = self.time.shape[0]
        for name in ("lfps", "raw_lfp", "sharp_wave_lfp", "multiunit", "speed"):
            if getattr(self, name).shape[0] != n_time:
                msg = f"{name} has {getattr(self, name).shape[0]} samples; time has {n_time}."
                raise ValueError(msg)
        n_ripples = self.ripple_times.shape
        if not self.ripple_durations.shape == self.ripple_frequencies.shape == n_ripples:
            msg = "ripple_times, ripple_durations and ripple_frequencies differ in length."
            raise ValueError(msg)
        if len(self.unit_types) or len(self.baseline_rates):
            if self.multiunit.ndim != 2:
                msg = "multiunit must be (n_time, n_units) to have unit types or rates."
                raise ValueError(msg)
            n_units = self.multiunit.shape[1]
            for name in ("unit_types", "baseline_rates"):
                length = len(getattr(self, name))
                if length not in (0, n_units):
                    msg = f"{name} has {length} entries; multiunit has {n_units} units."
                    raise ValueError(msg)
        unknown = sorted(set(self.unit_types.tolist()) - set(UNIT_TYPES))
        if unknown:
            msg = f"unit_types has unknown labels {unknown}; use {', '.join(UNIT_TYPES)}."
            raise ValueError(msg)

    @property
    def ripple_windows(self) -> FloatArray:
        """Start and end of each ripple, shape (n_ripples, 2):
        ``ripple_times`` plus or minus half ``ripple_durations``, clipped to
        the recording. For ``simulate_session`` that is where the Gaussian
        envelope is at 1 percent of its peak; for ``simulate_network_session``
        each ripple's latent span at three side scales, without channel
        delays (far below 1 percent at envelope power 4). ``truth_windows``
        gives windows at a chosen fraction of the peak.

        In the order the ripples were given. Windows of ripples closer than
        their durations overlap; each is still one ripple to find.
        """
        half = self.ripple_durations / 2
        return np.column_stack(
            [
                np.maximum(self.ripple_times - half, self.time[0]),
                np.minimum(self.ripple_times + half, self.time[-1]),
            ]
        )


@explain_call_errors
def simulate_session(
    time: FloatArray,
    ripple_times: float | Sequence[float] | FloatArray,
    *,
    n_channels: int = 4,
    n_units: int = 50,
    channel_gains: Sequence[float] | FloatArray | None = None,
    shared_noise_fraction: float = 0.5,
    ripple_snr: float | None = 4.0,
    ripple_amplitude: float | None = None,
    ripple_duration: float | tuple[float, float] = (0.040, 0.120),
    ripple_frequency: float | tuple[float, float] = (150.0, 250.0),
    noise_type: NoiseType = "pink",
    noise_amplitude: float = 1.3,
    sharp_wave_amplitude: float = 2.0,
    sharp_wave_duration: float = 0.080,
    sharp_wave_leak: float = 0.3,
    ripple_leak: float = 0.3,
    baseline_rate: float | tuple[float, float] | FloatArray = (0.5, 5.0),
    ripple_rate_gain: float = 8.0,
    participation: float = 0.6,
    artifact_times: Sequence[float] | None = None,
    artifact_amplitude: float | None = None,
    artifact_duration: float = 0.100,
    running_intervals: ArrayLike | None = None,
    peak_speed: float = 30.0,
    theta_amplitude: float = 0.0,
    delta_amplitude: float = 0.0,
    rng: int | np.random.Generator | None = None,
    sampling_frequency: float | None = None,
) -> SimulatedSession:
    """Simulate every input the detectors take, from one set of ripples.

    One draw of per-ripple durations and frequencies drives the multichannel
    LFP, the sharp-wave pair and the multiunit activity, so the ripple in the
    LFP, the sharp wave under it and the population burst with it are the same
    event. Defaults are pink noise, ripples at four times the ripple-band
    background varying in frequency and duration, four channels with
    half-shared noise, and fifty units.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    ripple_times : float or array_like of float
    n_channels, n_units : int, optional
        Default 4 channels and 50 units. The spike detectors z-score the
        population rate, so their false-positive rate depends on how many
        spikes that rate is built from: on 30 s at the default rates, 20
        units (about 55 spikes/s) gave the HSE detector 22 spurious events
        and Carey 14, 100 units (about 300 spikes/s) gave 2 and 0.
    channel_gains, shared_noise_fraction, ripple_snr, ripple_amplitude,
    ripple_duration, ripple_frequency, noise_type, noise_amplitude,
    artifact_times, artifact_amplitude, artifact_duration : optional
        As in ``simulate_multichannel_LFP``. ``ripple_snr`` defaults to 4.0
        here; give ``ripple_amplitude`` instead to set the size in signal units.
    sharp_wave_amplitude, sharp_wave_duration, sharp_wave_leak, ripple_leak : optional
        As in ``simulate_sharp_wave_ripple_pair``.
    baseline_rate, ripple_rate_gain, participation : optional
        As in ``simulate_multiunit``.
    running_intervals : array_like, shape (n_bouts, 2), optional
        Running bouts, start and end in seconds. The speed is
        ``simulate_speed``'s, with ``peak_speed``. Default None: an immobile
        animal, speed 0 throughout.
    peak_speed : float, optional
        Speed at the middle of each bout, cm/s. Default 30.
    theta_amplitude, delta_amplitude : float, optional
        Amplitudes of ``simulate_theta_delta``'s theta (8 Hz, while running)
        and delta (2 Hz, at rest), added to every channel, the radiatum one
        included, as a field shared across layers: the difference between the
        pyramidal and radiatum channels is unchanged. Default 0, none; about 3
        or more stands out from the default noise.
    rng : int or numpy.random.Generator, optional
        Per-ripple durations and frequencies are drawn first, then the LFP
        (``simulate_multichannel_LFP``'s order, with the radiatum channel
        last), then the multiunit activity.
    sampling_frequency : float, optional
        As in ``simulate_LFP``. Recorded in the result.

    Returns
    -------
    SimulatedSession

    Examples
    --------
    >>> time = simulate_time(30000, 1500)
    >>> session = simulate_session(time, [3.0, 9.0, 15.0], rng=0)
    >>> session.lfps.shape, session.sharp_wave_lfp.shape, session.multiunit.shape
    ((30000, 4), (30000,), (30000, 50))

    """
    if ripple_amplitude is not None:
        ripple_snr = None
    time = np.asarray(time, dtype=float)
    rng = _generator(rng)
    centers = _as_ripple_times(ripple_times)
    frequencies = _draw_per_ripple(ripple_frequency, centers.size, rng)
    durations = _draw_per_ripple(ripple_duration, centers.size, rng)
    gains = _channel_gains(channel_gains, n_channels)
    channels = simulate_multichannel_LFP(
        time,
        centers,
        n_channels + 1,
        channel_gains=np.append(gains, ripple_leak),
        shared_noise_fraction=shared_noise_fraction,
        ripple_amplitude=ripple_amplitude,
        ripple_snr=ripple_snr,
        ripple_duration=durations,
        ripple_frequency=frequencies,
        noise_type=noise_type,
        noise_amplitude=noise_amplitude,
        artifact_times=artifact_times,
        artifact_amplitude=artifact_amplitude,
        artifact_duration=artifact_duration,
        rng=rng,
        sampling_frequency=sampling_frequency,
    )
    lfps, radiatum = channels[:, :n_channels], channels[:, n_channels]
    _add_sharp_wave_pair(
        lfps[:, 0],
        radiatum,
        time,
        centers,
        sharp_wave_amplitude,
        sharp_wave_duration,
        sharp_wave_leak,
    )
    if theta_amplitude > 0 or delta_amplitude > 0:
        # a field shared by every layer, so the pyramidal-minus-radiatum
        # difference, the sharp-wave feature, is unchanged
        slow = simulate_theta_delta(
            time,
            np.empty((0, 2)) if running_intervals is None else running_intervals,
            theta_amplitude=theta_amplitude,
            delta_amplitude=delta_amplitude,
        )
        lfps = lfps + slow[:, np.newaxis]
        radiatum = radiatum + slow
    speed = (
        np.zeros(time.size)
        if running_intervals is None
        else simulate_speed(time, running_intervals, peak_speed=peak_speed)
    )
    multiunit = simulate_multiunit(
        time,
        centers,
        n_units,
        baseline_rate=baseline_rate,
        ripple_rate_gain=ripple_rate_gain,
        participation=participation,
        ripple_duration=durations,
        rng=rng,
    )
    return SimulatedSession(
        time=time,
        lfps=lfps,
        raw_lfp=lfps[:, 0].copy(),
        sharp_wave_lfp=radiatum,
        multiunit=multiunit,
        speed=speed,
        ripple_times=centers,
        ripple_durations=durations,
        ripple_frequencies=frequencies,
        artifact_times=np.asarray(
            [] if artifact_times is None else artifact_times, dtype=float
        ),
        sampling_frequency=_sampling_rate(time, sampling_frequency),
        running_intervals=_bouts(running_intervals),
    )


_REFERENCE_TYPE_PROBABILITIES = {
    "swr": 0.55,
    "weak_ripple": 0.15,
    "burst_only": 0.10,
    "ripple_doublet": 0.10,
    "sharp_wave_only": 0.10,
}
_TYPE_EXPRESSIONS = {
    "swr": ("ripple", "sharp_wave", "burst"),
    "weak_ripple": ("ripple", "sharp_wave", "burst"),
    "burst_only": ("burst",),
    "ripple_doublet": ("ripple", "sharp_wave", "burst"),
    "sharp_wave_only": ("sharp_wave",),
}
_ENVELOPE_POWERS = (2, 4)
_MAX_RIPPLES = 3
_TRIPLET_PROBABILITY = 0.3
# Columns of each event's fixed blocks of variates (see draw_network_events'
# Notes): standard normals, then uniforms.
_N_NORMALS = 1 + 2 * _MAX_RIPPLES + 2 * _MAX_RIPPLES + 2
_N_UNIFORMS = 1 + 4 * _MAX_RIPPLES + _MAX_RIPPLES + 2


def _check_range(
    name: str,
    value: object,
    *,
    lower: float = -np.inf,
    lower_strict: bool = False,
    upper: float = np.inf,
    upper_strict: bool = False,
) -> tuple[float, float]:
    """``value`` as a finite ``(low, high)`` pair with ``low <= high`` inside
    the stated bounds; ``ValueError`` naming ``name`` otherwise."""
    if not isinstance(value, tuple) or len(value) != 2:
        msg = f"{name} must be a (low, high) tuple, got {value!r}."
        raise ValueError(msg)
    low, high = float(value[0]), float(value[1])
    inside = (
        np.isfinite(low)
        and np.isfinite(high)
        and low <= high
        and (low > lower if lower_strict else low >= lower)
        and (high < upper if upper_strict else high <= upper)
    )
    if not inside:
        bounds = (
            f"{'(' if lower_strict else '['}{lower:g}, {upper:g}{')' if upper_strict else ']'}"
        )
        msg = (
            f"{name} must be a finite (low, high) range with low <= high in {bounds}, "
            f"got {value}."
        )
        raise ValueError(msg)
    return low, high


def _check_scalar(
    name: str,
    value: float,
    *,
    lower: float = 0.0,
    lower_strict: bool = False,
    upper: float = np.inf,
) -> float:
    """``value`` as a finite float at or above (or above) ``lower`` and at or
    below ``upper``."""
    number = float(value)
    above = number > lower if lower_strict else number >= lower
    if not (np.isfinite(number) and above and number <= upper):
        relation = ">" if lower_strict else ">="
        limit = f" and <= {upper:g}" if np.isfinite(upper) else ""
        msg = f"{name} must be finite and {relation} {lower:g}{limit}, got {value}."
        raise ValueError(msg)
    return number


def _checked_time(
    time: ArrayLike, sampling_frequency: float | None = None
) -> tuple[FloatArray, float]:
    """``time`` as a 1-D float array of two or more increasing samples with no
    gap (the detectors' rule), and the sampling rate: ``sampling_frequency``,
    which must agree with the median step to the timestamps' rounding, or the
    rate the step gives."""
    time = np.asarray(time, dtype=float)
    if time.ndim != 1 or time.size < 2:
        msg = f"time must be 1-D with at least two samples, got shape {time.shape}."
        raise ValueError(msg)
    if not np.all(np.diff(time) > 0):
        msg = "time must be strictly increasing."
        raise ValueError(msg)
    if len(_contiguous_valid_blocks(np.ones(time.size, dtype=bool), time)) > 1:
        msg = "time must be evenly sampled, without gaps; simulate each block separately."
        raise ValueError(msg)
    rate = _sampling_rate(time, sampling_frequency)
    step = float(np.median(np.diff(time)))
    if abs(1 / rate - step) > max(4 * float(np.spacing(np.abs(time).max())), 1e-6 * step):
        msg = (
            f"sampling_frequency {sampling_frequency} Hz disagrees with time's step, "
            f"{step:.6g} s ({1 / step:.6g} Hz)."
        )
        raise ValueError(msg)
    return time, rate


def _bouts(running_intervals: ArrayLike | None) -> FloatArray:
    """Checked running bouts, ``(0, 2)`` for None."""
    if running_intervals is None:
        return np.empty((0, 2))
    return _running_intervals(running_intervals)


def _rest_intervals(time: FloatArray, bouts: FloatArray) -> FloatArray:
    """(n, 2) stretches of rest: the recording less its first and last second
    and the running ``bouts``."""
    start, end = float(time[0]) + 1.0, float(time[-1]) - 1.0
    rest = []
    cursor = start
    for bout_start, bout_end in bouts:
        if bout_start > cursor:
            rest.append((cursor, min(bout_start, end)))
        cursor = max(cursor, bout_end)
    rest.append((cursor, end))
    intervals = np.asarray(rest, dtype=float)
    return intervals[intervals[:, 1] > intervals[:, 0]]


def _poisson_times(
    intervals: FloatArray, rate: float, rng: np.random.Generator
) -> tuple[FloatArray, IntArray]:
    """A Poisson process of ``rate`` per second on the concatenated
    ``intervals``, mapped back to recording time: the sorted times and the
    index of the interval each lies in. Draws the count, then the positions."""
    lengths = intervals[:, 1] - intervals[:, 0]
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(cumulative[-1])  # the edges' own sum, so every position maps inside
    n = int(rng.poisson(rate * total))
    positions = np.sort(rng.uniform(0.0, total, size=n))
    index = np.clip(np.searchsorted(cumulative, positions, side="right") - 1, 0, None)
    return intervals[index, 0] + positions - cumulative[index], index


@explain_call_errors
def draw_network_events(
    time: ArrayLike,
    *,
    event_rate: float = 0.3,
    type_probabilities: Mapping[str, float] | None = None,
    running_intervals: ArrayLike | None = None,
    ripple_duration: tuple[float, float] = (0.03, 0.15),
    ripple_skew: tuple[float, float] = (0.5, 0.7),
    ripple_frequency: tuple[float, float] = (160.0, 220.0),
    ripple_chirp: tuple[float, float] = (0.0, 30.0),
    ripple_snr: tuple[float, float] = (2.5, 6.0),
    weak_ripple_snr: tuple[float, float] = (1.2, 2.2),
    sharp_wave_duration: tuple[float, float] = (0.04, 0.12),
    sharp_wave_amplitude: tuple[float, float] = (3.0, 8.0),
    sharp_wave_lag: float = 0.01,
    burst_duration_ratio: tuple[float, float] = (1.0, 1.5),
    burst_lag: float = 0.01,
    burst_gain: float = 40.0,
    participation: tuple[float, float] = (0.2, 0.6),
    weak_participation: tuple[float, float] = (0.02, 0.1),
    burst_only_duration: tuple[float, float] = (0.05, 0.3),
    doublet_interval: tuple[float, float] = (0.06, 0.12),
    minimum_separation: float = 0.05,
    strength_correlation: float = 0.0,
    envelope_power: int = 2,
    rng: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Draw latent network events: when they happen, of which type, and the
    envelope, frequency and size of each component.

    A latent network event is expressed as a ripple (pyramidal-layer LFP), a
    sharp wave (stratum radiatum LFP) and a population burst (spikes), in five
    types (``EVENT_TYPES``):

    ==================  ======  ==========  =====  ================================
    Type                Ripple  Sharp wave  Burst  Differences from ``swr``
    ==================  ======  ==========  =====  ================================
    ``swr``             yes     yes         yes    reference
    ``weak_ripple``     weak    half size   weak   ``weak_ripple_snr``,
                                                   ``weak_participation``
    ``burst_only``                          yes    span ``burst_only_duration``
    ``ripple_doublet``  2 or 3  one each    one    the burst spans all ripples
    ``sharp_wave_only``         yes
    ==================  ======  ==========  =====  ================================

    Events occur only at rest: outside ``running_intervals`` and more than a
    second from either end of ``time``. Render the table with
    ``simulate_network_session`` and take its truth windows with
    ``truth_windows``; edit it between the two to fix any value.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Sample timestamps in seconds, increasing and evenly spaced.
    event_rate : float, optional
        Events per second of rest, before events are dropped for being too
        close to the previous one or for not fitting in their stretch of
        rest. Default 0.3, awake immobility (see Notes).
    type_probabilities : mapping of str to float, optional
        Relative frequency of each event type; a type left out never occurs,
        and the values are normalized. Default None: swr 0.55, weak_ripple
        0.15, burst_only 0.10, ripple_doublet 0.10, sharp_wave_only 0.10.
    running_intervals : array_like, shape (n_bouts, 2), optional
        Running bouts, start and end in seconds, sorted and not overlapping.
        Default None: at rest throughout.
    ripple_duration : (float, float), optional
        Range of a ripple's nominal span, ``3 (rise_sigma + decay_sigma)``
        seconds. Its width at half maximum is about 0.39 times the span.
        ``simulate_network_session`` needs each side scale to be at least one
        sample. Default (0.03, 0.15).
    ripple_skew : (float, float), optional
        Range of the fraction of the span after the peak, in (0, 1); above
        0.5 decays more slowly than it rises. Default (0.5, 0.7).
    ripple_frequency : (float, float), optional
        Range of the frequency at the start of the span, Hz, below Nyquist.
        Default (160, 220).
    ripple_chirp : (float, float), optional
        Range of the linear frequency decline over the span, Hz; ``(0, 0)``
        gives constant-frequency ripples. Default (0, 30).
    ripple_snr : (float, float), optional
        Range of a ripple's nominal size: its peak after ``filter_ripple_band``
        over the standard deviation of the filtered stationary background, as
        ``simulate_LFP``'s ``ripple_snr``. For ``swr`` and
        ``ripple_doublet``. Default (2.5, 6.0).
    weak_ripple_snr : (float, float), optional
        The same for ``weak_ripple``. Default (1.2, 2.2).
    sharp_wave_duration : (float, float), optional
        Range of a sharp wave's nominal span, six side scales; symmetric.
        Default (0.04, 0.12).
    sharp_wave_amplitude : (float, float), optional
        Range of the radiatum deflection's peak, in signal units; halved for
        ``weak_ripple``. Default (3, 8).
    sharp_wave_lag : float, optional
        Standard deviation in seconds of a sharp wave's centre about its
        ripple's. Default 0.01.
    burst_duration_ratio : (float, float), optional
        Range of a burst's span relative to its ripple's; the burst takes the
        ripple's skew. For ``swr`` and ``weak_ripple``: a ``ripple_doublet``'s
        burst is symmetric, from its earliest ripple's start to its latest
        ripple's end (three side scales). Default (1.0, 1.5).
    burst_lag : float, optional
        Standard deviation in seconds of a burst's centre about its ripple's;
        not applied to a ``ripple_doublet``'s burst. Default 0.01.
    burst_gain : float, optional
        Peak intensity of a recruited unit relative to its baseline, at least
        1. Default 40.
    participation : (float, float), optional
        Range of the probability that a place unit is recruited by a burst
        (other pyramidal units: half of it), for ``swr``, ``ripple_doublet``
        and ``burst_only``. A latent probability: the fraction of units that
        fire in an event is lower. Default (0.2, 0.6).
    weak_participation : (float, float), optional
        The same for ``weak_ripple``. Default (0.02, 0.1).
    burst_only_duration : (float, float), optional
        Range of a ``burst_only`` burst's nominal span, symmetric. Default
        (0.05, 0.3).
    doublet_interval : (float, float), optional
        Range of the centre-to-centre interval between successive ripples of
        a ``ripple_doublet``, seconds. Default (0.06, 0.12).
    minimum_separation : float, optional
        Seconds required between one event's span (its components' union
        at three side scales) and the next's. Default 0.05.
    strength_correlation : float, optional
        Latent correlation, in [0, 1], between an event's ripple SNR, onset
        frequency, sharp-wave amplitude and participation. Each keeps its
        uniform distribution on its range whatever the value. Default 0:
        independent.
    envelope_power : {2, 4}, optional
        Shape of every envelope, ``exp(-ln 2 (|t| / (sqrt(2 ln 2) sigma))**p)``
        on each side: 2 is a Gaussian with SD sigma; 4 is flatter at the top
        and steeper at the edges, with the same half-maximum width. The spans
        at three and four side scales keep their meaning as nominal extents,
        though at power 4 the envelope is far smaller there. Default 2.
    rng : int or numpy.random.Generator, optional
        Seed, or a Generator to draw from. The draw order is in the Notes.

    Returns
    -------
    events : pandas.DataFrame
        One row per component, sorted by ``event_id``, then ``expression``
        in ``EXPRESSIONS`` order (ripple, sharp wave, burst), then
        ``component``, with a RangeIndex. Columns:

        - ``event_id`` (int): the latent event, numbered from 0 in order of
          its earliest component's centre.
        - ``event_type``, ``expression`` (str): from ``EVENT_TYPES`` and
          ``EXPRESSIONS``.
        - ``component`` (int): 0, or the ripple's (and its sharp wave's)
          place in a ``ripple_doublet``.
        - ``center_time``, ``rise_sigma``, ``decay_sigma`` (float): the
          envelope's peak and side scales, seconds.
        - ``envelope_power`` (int).
        - ``amplitude`` (float): ripple, nominal SNR; sharp wave, radiatum
          peak in signal units (rendered negative); burst, ``burst_gain``.
        - ``frequency_start``, ``frequency_end`` (float): Hz over the
          ripple's span; NaN on other rows.
        - ``participation`` (float): burst rows; NaN on others.
        - ``n_participants`` (int): 0; ``simulate_network_session`` fills
          it in for burst rows.

        Every component's span at four side scales lies inside the rest
        stretch its event began in. With no events the table is empty with
        these columns and dtypes.

    Raises
    ------
    ValueError
        If ``time`` is not 1-D with two or more increasing, evenly spaced
        samples, a rate, lag or separation is negative or not finite,
        ``burst_gain`` is below 1 or not finite, ``type_probabilities`` names
        an unknown type or has a negative or non-finite weight or none
        positive, a range is not a finite
        ``(low, high)`` tuple with ``low <= high`` inside its bounds (positive
        durations, SNRs and intervals, skew in (0, 1), participation in
        [0, 1], frequencies and the frequency after the chirp between 0 and
        Nyquist), ``strength_correlation`` lies outside [0, 1] or
        ``envelope_power`` is not 2 or 4.

    Notes
    -----
    Draw order: the event count (Poisson, ``event_rate`` times the rest
    time), their positions on the concatenated rest time, their types; then
    an (n_events, 15) array of standard normals and after it an (n_events,
    18) array of uniforms, one row per event in time order. Every event draws
    the same block whatever its type or the parameters, and an event too
    close to the previous kept event, or whose span at four side scales
    leaves its stretch of rest, is dropped, not redrawn. So a parameter that
    sets only a size, frequency, participation, the type mix or the strength
    correlation changes only the values it governs. One that moves a span or
    centre (durations, skew, lags, ratios, ``doublet_interval``,
    ``minimum_separation``) can also change which events are dropped, and so
    later events' ``event_id``; ``event_rate``, ``running_intervals`` and
    ``time`` change the draws themselves.

    The normals are the shared strength ``z``; per ripple slot (up to three)
    the residuals for its SNR and onset frequency; per sharp-wave slot its
    lag and the residual for its amplitude; the burst's lag and the residual
    for its participation. A coupled value is ``low + (high - low) *
    ndtr(sqrt(rho) z + sqrt(1 - rho) residual)``. The uniforms are the
    doublet's ripple count (3 with probability 0.3, else 2); per ripple slot
    its span, skew, chirp and the interval from the previous ripple; per
    sharp-wave slot its span; the burst's duration ratio and its
    ``burst_only`` span.

    Reference values, here and in ``simulate_network_session``, and their
    sources. The reference is rat dorsal CA1 at awake rest between running
    bouts. "Assumed" marks a value no source checked supports; values read
    from a figure are approximate. The measured targets the simulator is
    validated against are in the repository's
    ``examples/benchmark/simulator_targets.csv``.

    - Event rate, 0.3 per second of rest: between awake-immobility ripple
      rates of about 0.13-0.22/s (Buzsáki 2015, doi:10.1002/hipo.22488,
      Fig. 3C, read from the figure) and 0.32-0.40 multiunit candidate
      events/s during stops (Davidson, Kloosterman & Wilson 2009,
      doi:10.1016/j.neuron.2009.07.027, Results). Sleep rates are higher,
      0.3-0.5/s (Nguyen et al. 2009, doi:10.3389/neuro.07.011.2009,
      Results). All depend on the detection threshold.
    - Ripple span, 0.03-0.15 s: ripples last 30-150 ms, skewed toward long
      (Buzsáki 2015, "Definition of Pathological Events"), convention
      unstated. The span is nominal, not a threshold-crossing duration: its
      width at half maximum is about 0.39 times it.
    - Onset frequency, 160-220 Hz: ripples of 140-220 Hz (Sullivan et al.
      2011, doi:10.1523/JNEUROSCI.0294-11.2011, abstract); modal per-event
      spectral peaks of 167, 177 and 187 Hz in sleep, quiet waking and
      immobility on a maze (Buzsáki 2015, Fig. 4C caption). "Onset" is
      this model's convention.
    - Chirp, a 0-30 Hz decline, linear over the whole span: recorded ripples
      decelerate (Nguyen et al. 2009, Results; Sullivan et al. 2011,
      Results), but faster and later than this model does. Their frequency
      starts to fall shortly before the envelope's peak, by about 15-20 Hz
      in the median over some 15 ms (Nguyen et al. 2009, Fig. 2C, read from
      the figure), where the model, at the middle of the default ranges (15
      Hz over 0.09 s), falls about 2.5 Hz in any 15 ms. About a quarter of
      ripples rise instead (Nguyen et al. 2009, Discussion); the model omits
      them. Both are limitations.
    - Sharp-wave span, 0.04-0.12 s: sharp waves of 40-100 ms (Buzsáki 2015,
      Introduction), convention unstated; the nominal span is wider than the
      visible deflection, and the upper end, 0.12 s, is assumed.
    - Participation, 0.2-0.6 for place units and half that for other
      pyramidal units: a latent probability, assumed. The observed fraction
      of CA1 pyramidal cells that fire is about 10% in a 50 ms window,
      0-40% by event (Ylinen et al. 1995,
      doi:10.1523/JNEUROSCI.15-01-00030.1995, p. 35), and about 30% in the
      largest events (Csicsvari et al. 2000,
      doi:10.1016/S0896-6273(00)00135-5, Fig. 3C, read from the figure).
    - Doublets: centre-to-centre 0.06-0.12 s, around the 8.8-11.8 ripples/s
      within long replay events (Davidson et al. 2009, Results); two
      ripples or, with probability 0.3, three: assumed.
    - Strength correlation, 0: an assumed, independent control. Sharp-wave
      magnitude correlates with ripple power, r = 0.47 (0.30-0.55 by
      animal; Sullivan et al. 2011, Results), which ``strength_correlation``
      above 0 stands in for (0.6 in the simulator's validation, an assumed
      stress value).
    - Units, 40 place (0.1-0.5 Hz) and 10 other pyramidal (0.5-1.5 Hz):
      within CA1 pyramidal rates, lognormal over 0.001-10 Hz (Mizuseki &
      Buzsáki 2013, doi:10.1016/j.celrep.2013.07.039, Results), with a
      non-theta mean of 1.4 Hz (Csicsvari et al. 1999,
      doi:10.1523/JNEUROSCI.19-01-00274.1999, p. 278).
    - Interneurons, 10 at 8-15 Hz: non-theta means of 8.3 and 14.3 Hz for
      two groups (Csicsvari et al. 1999, p. 278). Their gain of 3 on the
      ripple: interneurons fire about three times their rate outside sharp
      waves at its peak (p. 279). That every interneuron takes part is
      assumed: interneuron types differ, some falling silent (Klausberger et
      al. 2003, doi:10.1038/nature01374, p. 846, under anaesthesia).
    - Assumed: the type mix, the 0.05 s separation, ripple skew, the ripple
      and weak-ripple SNR ranges, sharp-wave amplitudes (3-8, around and
      above the delta amplitude, 4, of ``examples/literature_recipes.py``)
      and lags, the burst gain of 40 (``examples/literature_recipes.py``),
      span ratio and lag, the
      ``burst_only`` span, the weak-ripple values, and the renderer's noise,
      leaks, channel count and theta and delta amplitudes
      (``simulate_session``'s and ``examples/literature_recipes.py``'s).

    Examples
    --------
    >>> time = simulate_time(60 * 1500, 1500)
    >>> events = draw_network_events(time, running_intervals=[(20.0, 35.0)], rng=0)
    >>> list(events.columns)  # doctest: +NORMALIZE_WHITESPACE
    ['event_id', 'event_type', 'expression', 'component', 'center_time',
     'rise_sigma', 'decay_sigma', 'envelope_power', 'amplitude',
     'frequency_start', 'frequency_end', 'participation', 'n_participants']
    >>> bool(events.center_time.between(20.0, 35.0).any())
    False

    """
    time, rate = _checked_time(time)
    nyquist = rate / 2
    event_rate = _check_scalar("event_rate", event_rate)
    probabilities = _type_probabilities(type_probabilities)
    ripple_duration = _check_range(
        "ripple_duration", ripple_duration, lower=0, lower_strict=True
    )
    ripple_skew = _check_range(
        "ripple_skew", ripple_skew, lower=0, lower_strict=True, upper=1, upper_strict=True
    )
    # the renderer sizes a ripple from its samples: each side scale >= a step
    if ripple_duration[0] * min(ripple_skew[0], 1 - ripple_skew[1]) / 3 < 1 / rate:
        msg = (
            f"ripple_duration {ripple_duration} with ripple_skew {ripple_skew} can give a "
            f"side scale under one sample, {1 / rate:g} s; lengthen the shortest ripple."
        )
        raise ValueError(msg)
    ripple_frequency = _check_range(
        "ripple_frequency", ripple_frequency, lower=0, lower_strict=True, upper=nyquist,
        upper_strict=True,
    )  # fmt: skip
    ripple_chirp = _check_range(
        "ripple_chirp", ripple_chirp, lower=0, upper=ripple_frequency[0], upper_strict=True
    )
    ripple_snr = _check_range("ripple_snr", ripple_snr, lower=0, lower_strict=True)
    weak_ripple_snr = _check_range(
        "weak_ripple_snr", weak_ripple_snr, lower=0, lower_strict=True
    )
    sharp_wave_duration = _check_range(
        "sharp_wave_duration", sharp_wave_duration, lower=0, lower_strict=True
    )
    sharp_wave_amplitude = _check_range("sharp_wave_amplitude", sharp_wave_amplitude, lower=0)
    sharp_wave_lag = _check_scalar("sharp_wave_lag", sharp_wave_lag)
    burst_duration_ratio = _check_range(
        "burst_duration_ratio", burst_duration_ratio, lower=0, lower_strict=True
    )
    burst_lag = _check_scalar("burst_lag", burst_lag)
    burst_gain = _check_scalar("burst_gain", burst_gain, lower=1.0)
    participation = _check_range("participation", participation, lower=0, upper=1)
    weak_participation = _check_range(
        "weak_participation", weak_participation, lower=0, upper=1
    )
    burst_only_duration = _check_range(
        "burst_only_duration", burst_only_duration, lower=0, lower_strict=True
    )
    doublet_interval = _check_range(
        "doublet_interval", doublet_interval, lower=0, lower_strict=True
    )
    minimum_separation = _check_scalar("minimum_separation", minimum_separation)
    rho = _check_scalar("strength_correlation", strength_correlation, upper=1.0)
    if envelope_power not in _ENVELOPE_POWERS:
        msg = f"envelope_power must be 2 or 4, got {envelope_power!r}."
        raise ValueError(msg)
    rng = _generator(rng)

    rest = _rest_intervals(time, _bouts(running_intervals))
    event_times, rest_index = _poisson_times(rest, event_rate, rng)
    n_events = event_times.size
    types = rng.choice(len(EVENT_TYPES), size=n_events, p=probabilities)
    normals = rng.standard_normal((n_events, _N_NORMALS))
    uniforms = rng.random((n_events, _N_UNIFORMS))
    # each residual coupled to its event's shared strength (column 0); uniform
    # on [0, 1] for any rho
    coupled = special.ndtr(np.sqrt(rho) * normals[:, :1] + np.sqrt(1.0 - rho) * normals)

    rows: list[dict[str, object]] = []
    last_end, n_kept = -np.inf, 0
    for event in range(n_events):
        event_type = EVENT_TYPES[types[event]]
        z, c, u = normals[event], coupled[event], uniforms[event]
        weak = event_type == "weak_ripple"
        components: list[dict[str, object]] = []
        if event_type in ("swr", "weak_ripple", "ripple_doublet"):
            n_ripples = 1
            if event_type == "ripple_doublet":
                n_ripples = 3 if u[0] < _TRIPLET_PROBABILITY else 2
            center = event_times[event]
            ripples = []
            for j in range(n_ripples):
                if j > 0:
                    center += _scaled(u[4 + 4 * j], doublet_interval)
                span = _scaled(u[1 + 4 * j], ripple_duration)
                skew = _scaled(u[2 + 4 * j], ripple_skew)
                onset = _scaled(c[2 + 2 * j], ripple_frequency)
                rise, decay = span * (1 - skew) / 3, span * skew / 3
                ripples.append((center, rise, decay, span, skew))
                components.append(
                    {
                        "expression": "ripple", "component": j, "center_time": center,
                        "rise_sigma": rise, "decay_sigma": decay,
                        "amplitude": _scaled(
                            c[1 + 2 * j], weak_ripple_snr if weak else ripple_snr
                        ),
                        "frequency_start": onset,
                        "frequency_end": onset - _scaled(u[3 + 4 * j], ripple_chirp),
                    }
                )  # fmt: skip
                sigma = _scaled(u[13 + j], sharp_wave_duration) / 6
                components.append(
                    {
                        "expression": "sharp_wave", "component": j,
                        "center_time": center + sharp_wave_lag * z[7 + 2 * j],
                        "rise_sigma": sigma, "decay_sigma": sigma,
                        "amplitude": _scaled(c[8 + 2 * j], sharp_wave_amplitude)
                        * (0.5 if weak else 1.0),
                    }
                )  # fmt: skip
            if event_type == "ripple_doublet":
                # every ripple's span: an earlier, longer ripple can end last
                start = min(center - 3 * rise for center, rise, *_ in ripples)
                end = max(center + 3 * decay for center, _, decay, *_ in ripples)
                burst_center, burst_rise = (start + end) / 2, (end - start) / 6
                burst_decay = burst_rise
            else:
                ripple_center, _, _, span, skew = ripples[0]
                burst_center = ripple_center + burst_lag * z[13]
                burst_span = span * _scaled(u[16], burst_duration_ratio)
                burst_rise, burst_decay = burst_span * (1 - skew) / 3, burst_span * skew / 3
            burst_participation = weak_participation if weak else participation
        elif event_type == "burst_only":
            burst_center = event_times[event]
            burst_rise = burst_decay = _scaled(u[17], burst_only_duration) / 6
            burst_participation = participation
        else:  # sharp_wave_only
            sigma = _scaled(u[13], sharp_wave_duration) / 6
            components.append(
                {
                    "expression": "sharp_wave", "component": 0,
                    "center_time": event_times[event], "rise_sigma": sigma,
                    "decay_sigma": sigma, "amplitude": _scaled(c[8], sharp_wave_amplitude),
                }
            )  # fmt: skip
        if event_type != "sharp_wave_only":
            components.append(
                {
                    "expression": "burst", "component": 0, "center_time": burst_center,
                    "rise_sigma": burst_rise, "decay_sigma": burst_decay,
                    "amplitude": burst_gain,
                    "participation": _scaled(c[14], burst_participation),
                }
            )  # fmt: skip

        centers = np.array([row["center_time"] for row in components], dtype=float)
        rises = np.array([row["rise_sigma"] for row in components], dtype=float)
        decays = np.array([row["decay_sigma"] for row in components], dtype=float)
        start_3, end_3 = np.min(centers - 3 * rises), np.max(centers + 3 * decays)
        start_4, end_4 = np.min(centers - 4 * rises), np.max(centers + 4 * decays)
        rest_start, rest_end = rest[rest_index[event]]
        if start_3 < last_end + minimum_separation:
            continue
        if start_4 < rest_start or end_4 > rest_end:
            continue
        last_end = end_3
        # kept events' spans are disjoint and in time order, and each centre
        # lies inside its span, so the keep order is the earliest-centre order
        rows.extend(
            {"event_id": n_kept, "event_type": event_type, **row} for row in components
        )
        n_kept += 1

    table = pd.DataFrame(rows, columns=[*_EVENT_COLUMNS]).assign(
        envelope_power=envelope_power, n_participants=0
    )
    return _sorted_events(_table(_EVENT_COLUMNS, table))


def _scaled(u: float, bounds: tuple[float, float]) -> float:
    """``u`` in [0, 1] mapped linearly onto ``(low, high)``."""
    return bounds[0] + (bounds[1] - bounds[0]) * float(u)


def _type_probabilities(type_probabilities: Mapping[str, float] | None) -> FloatArray:
    """The probability of each of ``EVENT_TYPES``, normalized; a type left out has 0."""
    given = _REFERENCE_TYPE_PROBABILITIES if type_probabilities is None else type_probabilities
    unknown = sorted(set(given) - set(EVENT_TYPES))
    if unknown:
        msg = (
            f"type_probabilities has unknown event types {unknown}; "
            f"use {', '.join(map(repr, EVENT_TYPES))}."
        )
        raise ValueError(msg)
    weights = np.array([float(given.get(name, 0.0)) for name in EVENT_TYPES])
    if not (np.all(np.isfinite(weights)) and np.all(weights >= 0) and weights.sum() > 0):
        msg = (
            "type_probabilities must be finite and non-negative with a positive sum, "
            f"got {dict(given)}."
        )
        raise ValueError(msg)
    normalized: FloatArray = weights / weights.sum()
    return normalized


def _sorted_events(events: pd.DataFrame) -> pd.DataFrame:
    """Sorted by ``event_id``, then expression in ``EXPRESSIONS`` order, then
    ``component``, with a RangeIndex."""
    rank = events["expression"].map({name: i for i, name in enumerate(EXPRESSIONS)})
    order = np.lexsort(
        (events["component"].to_numpy(), rank.to_numpy(), events["event_id"].to_numpy())
    )
    return events.iloc[order].reset_index(drop=True)


_REFERENCE_NON_EVENT_RATES = {
    "spike_leakage": 2.0,
    "emg": 1.0,
    "fast_gamma": 2.0,
    "theta_burst": 6.0,
}
# when each kind of non-event occurs, and the uniforms each draws per non-event
_NON_EVENT_STATES = {
    "spike_leakage": "rest",
    "emg": "any",
    "fast_gamma": "any",
    "theta_burst": "running",
}
_NON_EVENT_UNIFORMS = {"spike_leakage": 4, "emg": 1, "fast_gamma": 3, "theta_burst": 2}
_EMG_HIGH_PASS = 100.0  # Hz
# a spike and its after-hyperpolarization, a sample each (about 1 ms at 1500 Hz)
_SPIKE_WAVEFORM = np.array([-1.0, 0.45, 0.2])


def _non_event_rates(rates: Mapping[str, float] | None) -> dict[str, float]:
    """Each of ``NON_EVENT_TYPES``' rate per minute; a type left out has 0."""
    given = _REFERENCE_NON_EVENT_RATES if rates is None else rates
    unknown = sorted(set(given) - set(NON_EVENT_TYPES))
    if unknown:
        msg = (
            f"rates has unknown non-event types {unknown}; "
            f"use {', '.join(map(repr, NON_EVENT_TYPES))}."
        )
        raise ValueError(msg)
    return {
        name: _check_scalar(f"rates[{name!r}]", given.get(name, 0.0))
        for name in NON_EVENT_TYPES
    }


def _check_count_range(name: str, value: object, lower: int) -> tuple[int, int]:
    """``value`` as a ``(low, high)`` pair of whole numbers, ``lower <= low <= high``."""
    low, high = _check_range(name, value, lower=lower)
    if low != np.round(low) or high != np.round(high):
        msg = f"{name} must be a range of whole numbers, got {value}."
        raise ValueError(msg)
    return int(low), int(high)


def _check_sizing_band(
    name: str, band: object, frequencies: tuple[float, float], rate: float
) -> tuple[float, float]:
    """``band`` as ``(low, high)`` Hz with ``0 < low < high`` below Nyquist,
    holding ``frequencies``, and narrow enough for ``filter_ripple_band`` to
    design a filter at ``rate``."""
    low, high = _check_range(
        name, band, lower=0, lower_strict=True, upper=rate / 2, upper_strict=True
    )
    if not low < high:
        msg = f"{name} must have low < high, got {band}."
        raise ValueError(msg)
    if not low <= frequencies[0] <= frequencies[1] <= high:
        msg = f"{name} {band} Hz must contain the burst frequencies, {frequencies} Hz."
        raise ValueError(msg)
    try:
        ripple_bandpass_filter(rate, (low, high))
    except ValueError as error:
        msg = f"{name} {band} Hz cannot be filtered at {rate:g} Hz: {error}"
        raise ValueError(msg) from error
    return low, high


def _whole_draws(u: FloatArray, bounds: tuple[int, int]) -> IntArray:
    """Uniforms ``u`` mapped onto the whole numbers ``low .. high``, each
    equally likely."""
    low, high = bounds
    whole: IntArray = np.minimum(low + np.floor(u * (high - low + 1)), high).astype(np.int64)
    return whole


def _allowed_intervals(time: FloatArray, bouts: FloatArray) -> dict[str, FloatArray]:
    """(n, 2) stretches each state covers, less the recording's first and
    last second: rest, running, and the whole recording (``"any"``)."""
    start, end = float(time[0]) + 1.0, float(time[-1]) - 1.0
    running = np.column_stack([np.maximum(bouts[:, 0], start), np.minimum(bouts[:, 1], end)])
    whole = np.array([[start, end]])
    return {
        "rest": _rest_intervals(time, bouts),
        "running": running[running[:, 1] > running[:, 0]],
        "any": whole[whole[:, 1] > whole[:, 0]],
    }


@explain_call_errors
def draw_non_events(
    time: ArrayLike,
    *,
    rates: Mapping[str, float] | None = None,
    running_intervals: ArrayLike | None = None,
    n_channels: int = 4,
    spike_leakage_units: tuple[int, int] = (1, 3),
    spike_leakage_spikes: tuple[int, int] = (3, 8),
    spike_leakage_isi: tuple[float, float] = (0.003, 0.006),
    spike_leakage_amplitude: float = 2.0,
    emg_duration: tuple[float, float] = (0.05, 0.5),
    emg_amplitude: float = 1.5,
    fast_gamma_frequency: tuple[float, float] = (60.0, 100.0),
    fast_gamma_band: tuple[float, float] = (60.0, 100.0),
    fast_gamma_duration: tuple[float, float] = (0.05, 0.15),
    fast_gamma_snr: tuple[float, float] = (1.5, 4.0),
    theta_burst_units: tuple[int, int] = (5, 15),
    theta_burst_duration: tuple[float, float] = (0.1, 0.3),
    theta_burst_gain: float = 10.0,
    rng: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Draw non-events: activity a detector should not report, each with its
    own truth, so a false positive can be traced to its cause.

    Four kinds (``NON_EVENT_TYPES``), rendered by ``simulate_network_session``
    when given as its ``non_events``:

    =================  ========  ===================================================
    Type               When      Rendered as
    =================  ========  ===================================================
    ``spike_leakage``  rest      ``n_units`` place or other pyramidal units fire
                                 ``n_spikes`` spikes together at intervals of
                                 ``isi``, added to their spike counts, and each
                                 spike leaves a three-sample biphasic waveform,
                                 peak ``amplitude``, on LFP channel ``channel``
    ``emg``            any       white noise high-passed at 100 Hz, its standard
                                 deviation ``amplitude`` at the envelope's peak,
                                 the same on every channel and the radiatum
    ``fast_gamma``     any       a burst at ``frequency``, sized as a ripple is but
                                 in its band (``snr_band_low``, ``snr_band_high``):
                                 its filtered peak is ``amplitude`` times the
                                 filtered stationary noise's standard deviation;
                                 on every channel at the recording-wide gains
    ``theta_burst``    running   ``n_units`` place units' intensity multiplied by
                                 up to ``amplitude``; no LFP
    =================  ========  ===================================================

    Each kind occurs as a Poisson process on the time its state covers, less
    the recording's first and last second. Non-events may overlap network
    events and each other.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Sample timestamps in seconds, increasing and evenly spaced.
    rates : mapping of str to float, optional
        Non-events per minute of the time each kind can occur, before those
        that do not fit are dropped; a kind left out never occurs. Default
        None: spike_leakage 2, emg 1, fast_gamma 2, theta_burst 6.
    running_intervals : array_like, shape (n_bouts, 2), optional
        Running bouts, start and end in seconds, sorted and not overlapping;
        as given to ``draw_network_events``. Default None: at rest
        throughout, so no theta bursts.
    n_channels : int, optional
        Pyramidal-layer channels of the session to render; a leakage burst's
        channel is drawn uniformly from them. Default 4.
    spike_leakage_units : (int, int), optional
        Range of the number of units in a leakage burst, at least 1. Default
        (1, 3).
    spike_leakage_spikes : (int, int), optional
        Range of spikes per unit in a leakage burst, at least 2. Default (3, 8).
    spike_leakage_isi : (float, float), optional
        Range of the interval between a leakage burst's spikes, seconds, at
        least one sample. Default (0.003, 0.006).
    spike_leakage_amplitude : float, optional
        Peak of a leaked spike's waveform in signal units, non-negative.
        Default 2.
    emg_duration : (float, float), optional
        Range of an EMG burst's nominal span, six side scales, symmetric.
        Default (0.05, 0.5).
    emg_amplitude : float, optional
        Standard deviation of an EMG burst at its peak, in signal units,
        non-negative. Default 1.5.
    fast_gamma_frequency : (float, float), optional
        Range of a gamma burst's frequency, Hz, inside ``fast_gamma_band``.
        Default (60, 100).
    fast_gamma_band : (float, float), optional
        The band a gamma burst's SNR is measured in, Hz: the burst and the
        stationary noise are both filtered to it with ``filter_ripple_band``.
        Default (60, 100).
    fast_gamma_duration : (float, float), optional
        Range of a gamma burst's nominal span, symmetric, with side scales of
        at least one sample. Default (0.05, 0.15).
    fast_gamma_snr : (float, float), optional
        Range of a gamma burst's nominal SNR in its band, positive. Default
        (1.5, 4).
    theta_burst_units : (int, int), optional
        Range of the number of place units in a theta burst, at least 1.
        Default (5, 15).
    theta_burst_duration : (float, float), optional
        Range of a theta burst's nominal span, symmetric. Default (0.1, 0.3).
    theta_burst_gain : float, optional
        Peak intensity of a theta burst's units relative to their baseline,
        at least 1. Default 10.
    rng : int or numpy.random.Generator, optional
        Seed, or a Generator to draw from. The draw order is in the Notes.

    Returns
    -------
    non_events : pandas.DataFrame
        One row per non-event, sorted by ``center_time``, with a RangeIndex.
        Columns:

        - ``non_event_id`` (int): numbered from 0 in time order.
        - ``non_event_type`` (str): from ``NON_EVENT_TYPES``.
        - ``center_time``, ``rise_sigma``, ``decay_sigma`` (float): the
          envelope's peak and side scales, seconds. A leakage burst's side
          scales are ``(n_spikes - 1) isi / 6``, so its spikes span three of
          each; ``truth_windows`` reads them, the renderer reads ``n_spikes``
          and ``isi``.
        - ``envelope_power`` (int): 2, a Gaussian.
        - ``amplitude`` (float): the waveform peak, the EMG's peak standard
          deviation, the gamma burst's SNR, or the theta burst's gain.
        - ``frequency``, ``snr_band_low``, ``snr_band_high`` (float): Hz,
          for ``fast_gamma``; NaN otherwise.
        - ``channel`` (int): the leaked spikes' channel; -1 otherwise.
        - ``n_units`` (int): for ``spike_leakage`` and ``theta_burst``; 0
          otherwise. Which units is drawn when the session is rendered.
        - ``n_spikes`` (int), ``isi`` (float, seconds): for
          ``spike_leakage``; 0 and NaN otherwise.

        Every non-event's span at four side scales lies inside the stretch
        of its state it began in. With no non-events the table is empty with
        these columns and dtypes.

    Raises
    ------
    ValueError
        If ``time`` is not 1-D with two or more increasing, evenly spaced
        samples; ``rates`` names an unknown kind or has a negative or
        non-finite rate; a range is not a finite ``(low, high)`` tuple with
        ``low <= high`` inside its bounds (positive durations, SNRs and
        intervals, unit and spike counts whole numbers of at least 1 and 2,
        a leakage interval and gamma side scale of at least one sample, gamma
        frequencies below Nyquist); ``fast_gamma_band`` does not hold
        ``fast_gamma_frequency`` or cannot be filtered at the sampling rate;
        an amplitude is negative or not finite, or ``theta_burst_gain``
        below 1; ``n_channels`` is not a whole number of at least 1; or
        there are EMG bursts and the sampling rate is 200 Hz or less (no
        room above the 100 Hz high-pass).

    See Also
    --------
    draw_network_events : the events a detector should report.
    simulate_network_session : renders both tables.
    truth_windows : each non-event's window at any fraction of its peak.

    Notes
    -----
    Draw order: one draw from ``rng`` seeds a stream per kind, in
    ``NON_EVENT_TYPES`` order. Each stream draws its count (Poisson), the
    positions on the concatenated time its state covers, and then a fixed
    block of uniforms per non-event, in time order: for ``spike_leakage``
    its unit count, spike count, interval and channel; for ``emg`` its span;
    for ``fast_gamma`` its frequency, span and SNR; for ``theta_burst`` its
    unit count and span. A non-event whose span at four side scales leaves
    its stretch is dropped, not redrawn. So one kind's rate or ranges never
    change another kind's non-events, and a parameter that sets only a size
    or frequency changes only the values it governs.

    Reference values and their sources (see ``draw_network_events``' Notes
    for the events'):

    - Leakage bursts: intraburst intervals of CA1 pyramidal cells peak at
      2-6 ms, with a mode of 5.04 +/- 1.00 ms (Mizuseki et al. 2012,
      doi:10.1002/hipo.22002, Results, Figs. 2A-B), hence 3-6 ms. The unit
      and spike counts, the waveform (a spike and its after-hyperpolarization,
      about 1 ms at 1500 Hz; three samples at any rate) and its amplitude are
      assumed.
    - Fast gamma, 60-100 Hz in the reference, an assumed control; nearby
      gamma of 90-140 Hz (Sullivan et al. 2011,
      doi:10.1523/JNEUROSCI.0294-11.2011, abstract and Fig. 1) is a harder
      condition, with ``fast_gamma_frequency`` and ``fast_gamma_band`` both
      (90, 140). Calling gamma a non-event is this benchmark's taxonomy, not
      a physiological classification by frequency.
    - Assumed: every rate, the EMG span, high-pass and amplitude, the gamma
      span and SNR, and the theta bursts' units, gain and span.

    Examples
    --------
    >>> time = simulate_time(60 * 1500, 1500)
    >>> non_events = draw_non_events(time, running_intervals=[(20.0, 35.0)], rng=0)
    >>> list(non_events.columns)  # doctest: +NORMALIZE_WHITESPACE
    ['non_event_id', 'non_event_type', 'center_time', 'rise_sigma',
     'decay_sigma', 'envelope_power', 'amplitude', 'frequency',
     'snr_band_low', 'snr_band_high', 'channel', 'n_units', 'n_spikes', 'isi']
    >>> theta = non_events[non_events.non_event_type == "theta_burst"]
    >>> bool(theta.center_time.between(20.0, 35.0).all())
    True

    """
    time, rate = _checked_time(time)
    nyquist = rate / 2
    per_minute = _non_event_rates(rates)
    _check_whole_number("n_channels", n_channels, 1)
    leakage_units = _check_count_range("spike_leakage_units", spike_leakage_units, 1)
    leakage_spikes = _check_count_range("spike_leakage_spikes", spike_leakage_spikes, 2)
    leakage_isi = _check_range("spike_leakage_isi", spike_leakage_isi, lower=1 / rate)
    leakage_amplitude = _check_scalar("spike_leakage_amplitude", spike_leakage_amplitude)
    emg_span = _check_range("emg_duration", emg_duration, lower=0, lower_strict=True)
    emg_amplitude = _check_scalar("emg_amplitude", emg_amplitude)
    if per_minute["emg"] > 0 and nyquist <= _EMG_HIGH_PASS:
        msg = (
            f"EMG is high-passed at {_EMG_HIGH_PASS:g} Hz, at or above the Nyquist "
            f"frequency, {nyquist:g} Hz; set rates['emg'] to 0."
        )
        raise ValueError(msg)
    gamma_frequency = _check_range(
        "fast_gamma_frequency", fast_gamma_frequency, lower=0, lower_strict=True,
        upper=nyquist, upper_strict=True,
    )  # fmt: skip
    gamma_band = _check_sizing_band("fast_gamma_band", fast_gamma_band, gamma_frequency, rate)
    gamma_span = _check_range("fast_gamma_duration", fast_gamma_duration, lower=6 / rate)
    gamma_snr = _check_range("fast_gamma_snr", fast_gamma_snr, lower=0, lower_strict=True)
    theta_units = _check_count_range("theta_burst_units", theta_burst_units, 1)
    theta_span = _check_range(
        "theta_burst_duration", theta_burst_duration, lower=0, lower_strict=True
    )
    theta_gain = _check_scalar("theta_burst_gain", theta_burst_gain, lower=1.0)
    rng = _generator(rng)

    allowed = _allowed_intervals(time, _bouts(running_intervals))
    seeds = rng.integers(np.iinfo(np.int64).max, size=len(NON_EVENT_TYPES))
    columns: dict[str, list[ArrayLike]] = {name: [] for name in _NON_EVENT_COLUMNS}
    for non_event_type, seed in zip(NON_EVENT_TYPES, seeds, strict=True):
        stream = np.random.default_rng(int(seed))
        intervals = allowed[_NON_EVENT_STATES[non_event_type]]
        centers, index = _poisson_times(intervals, per_minute[non_event_type] / 60, stream)
        u = stream.random((centers.size, _NON_EVENT_UNIFORMS[non_event_type]))
        n = centers.size
        values: dict[str, ArrayLike] = {
            "frequency": np.full(n, np.nan), "snr_band_low": np.full(n, np.nan),
            "snr_band_high": np.full(n, np.nan), "channel": np.full(n, -1),
            "n_units": np.zeros(n), "n_spikes": np.zeros(n), "isi": np.full(n, np.nan),
        }  # fmt: skip
        if non_event_type == "spike_leakage":
            n_spikes = _whole_draws(u[:, 1], leakage_spikes)
            isi = leakage_isi[0] + (leakage_isi[1] - leakage_isi[0]) * u[:, 2]
            sigma = (n_spikes - 1) * isi / 6
            values.update(
                amplitude=np.full(n, leakage_amplitude),
                n_units=_whole_draws(u[:, 0], leakage_units), n_spikes=n_spikes, isi=isi,
                channel=np.floor(u[:, 3] * n_channels),
            )  # fmt: skip
        elif non_event_type == "emg":
            sigma = (emg_span[0] + (emg_span[1] - emg_span[0]) * u[:, 0]) / 6
            values.update(amplitude=np.full(n, emg_amplitude))
        elif non_event_type == "fast_gamma":
            sigma = (gamma_span[0] + (gamma_span[1] - gamma_span[0]) * u[:, 1]) / 6
            values.update(
                frequency=gamma_frequency[0]
                + (gamma_frequency[1] - gamma_frequency[0]) * u[:, 0],
                amplitude=gamma_snr[0] + (gamma_snr[1] - gamma_snr[0]) * u[:, 2],
                snr_band_low=np.full(n, gamma_band[0]),
                snr_band_high=np.full(n, gamma_band[1]),
            )  # fmt: skip
        else:  # theta_burst
            sigma = (theta_span[0] + (theta_span[1] - theta_span[0]) * u[:, 1]) / 6
            values.update(
                amplitude=np.full(n, theta_gain), n_units=_whole_draws(u[:, 0], theta_units)
            )
        stretch = intervals[index]
        keep = (centers - 4 * sigma >= stretch[:, 0]) & (centers + 4 * sigma <= stretch[:, 1])
        values.update(
            non_event_type=np.full(n, non_event_type), center_time=centers,
            rise_sigma=sigma, decay_sigma=sigma, envelope_power=np.full(n, 2),
        )  # fmt: skip
        for name in _NON_EVENT_COLUMNS:
            if name != "non_event_id":
                columns[name].append(np.asarray(values[name])[keep])
    joined = {name: np.concatenate(value) for name, value in columns.items() if value}
    order = np.argsort(joined["center_time"], kind="stable")
    table = {name: value[order] for name, value in joined.items()}
    table["non_event_id"] = np.arange(order.size)
    return _table(_NON_EVENT_COLUMNS, table)


_REFERENCE_UNIT_COUNTS = {"place": 40, "pyramidal": 10, "interneuron": 10}
_REFERENCE_BASELINE_RATES = {
    "place": (0.1, 0.5),
    "pyramidal": (0.5, 1.5),
    "interneuron": (8.0, 15.0),
}
_SPIKE_BLOCK = 8
SPATIAL_PROFILES = ("global", "local")
SPIKE_MODELS = ("poisson", "refractory")
# the renderer's random streams, each seeded from one draw of the caller's rng
_RENDER_STREAMS = (
    "noise",
    "ripple_phases",
    "spatial_profiles",
    "noise_modulation",
    "baseline_rates",
    "participants",
    "non_events",
    "spikes",
)


def _event_envelope(
    time: FloatArray, center: float, rise_sigma: float, decay_sigma: float, power: int
) -> tuple[slice, FloatArray]:
    """The unit-peak envelope ``exp(-ln 2 (|t| / (sqrt(2 ln 2) sigma))**power)``,
    ``sigma`` the rise scale before ``center`` and the decay scale after, over
    the samples within 8 side scales; at least one sample."""
    window = _sample_window(time, center - 8 * rise_sigma, center + 8 * decay_sigma)
    offset = time[window] - center
    sigma = np.where(offset < 0, rise_sigma, decay_sigma)
    scaled = np.abs(offset) / (np.sqrt(2 * np.log(2)) * sigma)
    return window, np.asarray(np.exp(-np.log(2) * scaled**power), dtype=float)


def _chirp_cycles(
    offset: FloatArray, rise_sigma: float, decay_sigma: float, start: float, end: float
) -> FloatArray:
    """Cycles elapsed from ``-3 rise_sigma`` to each ``offset`` from the centre
    under a frequency constant at ``start`` before ``-3 rise_sigma``, linear to
    ``end`` at ``+3 decay_sigma`` and constant after."""
    t0, t1 = -3 * rise_sigma, 3 * decay_sigma
    slope = (end - start) / (t1 - t0)
    inside = np.clip(offset, t0, t1) - t0
    cycles = (
        start * (np.minimum(offset, t0) - t0)
        + start * inside
        + 0.5 * slope * inside**2
        + end * (np.maximum(offset, t1) - t1)
    )
    return np.asarray(cycles, dtype=float)


def _render_ripple(
    time: FloatArray,
    center: float,
    rise_sigma: float,
    decay_sigma: float,
    frequency_start: float,
    frequency_end: float,
    phase: float,
    power: int = 2,
) -> tuple[slice, FloatArray]:
    """A unit-envelope, linearly chirped ripple over ``center`` -8 rise .. +8
    decay: the samples and the waveform there.

    The frequency runs from ``frequency_start`` at -3 rise scales to
    ``frequency_end`` at +3 decay scales, constant outside. The carrier's
    phase is ``phase`` at the centre and is integrated exactly from it, so the
    waveform is the same at any clock origin and a delayed copy is the same
    waveform shifted.
    """
    window, envelope = _event_envelope(time, center, rise_sigma, decay_sigma, power)
    offset = time[window] - center
    cycles = _chirp_cycles(
        offset, rise_sigma, decay_sigma, frequency_start, frequency_end
    ) - _chirp_cycles(np.zeros(1), rise_sigma, decay_sigma, frequency_start, frequency_end)
    return window, np.sin(phase + 2 * np.pi * cycles) * envelope


def _noise_modulation(
    time: FloatArray, log_amplitude: float, period: float, phase: float
) -> FloatArray:
    """``exp(a sin(2 pi (t - t_0) / T + phase))`` scaled to unit RMS; 1 for ``a = 0``.
    Normalized in log space, so a large ``a`` does not overflow."""
    log_gain = log_amplitude * np.sin(2 * np.pi * (time - time[0]) / period + phase)
    log_rms = 0.5 * (special.logsumexp(2 * log_gain) - np.log(time.size))
    return np.asarray(np.exp(log_gain - log_rms), dtype=float)


def _rest_interval_of(
    rest: FloatArray, start: float, end: float
) -> tuple[float, float] | None:
    """The stretch of ``rest`` holding ``[start, end]``, or None."""
    inside = (rest[:, 0] <= start) & (end <= rest[:, 1])
    if not inside.any():
        return None
    first = int(np.flatnonzero(inside)[0])
    return float(rest[first, 0]), float(rest[first, 1])


def _spatial_profile(
    start: float,
    end: float,
    rest: FloatArray,
    gains: FloatArray,
    *,
    local: bool,
    occupancy: float,
    gain_range: tuple[float, float],
    maximum_delay: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    """One ripple's gain and delay on each channel, the recording-wide gains
    included; unselected channels have gain 0 and delay 0. ``start`` and
    ``end`` bound the ripple's span at four side scales.

    ``local`` selects ``max(1, ceil(occupancy n_channels))`` channels, one of
    them the anchor at event gain 1 and delay 0, the rest with a gain from
    ``gain_range`` and a delay uniform on ``[-maximum_delay, maximum_delay]``
    narrowed to the shifts that keep the ripple's span at four side scales in
    its stretch of rest (0 always qualifies). Draws, per ripple, a channel
    permutation, the anchor, then one gain and one delay variate per channel.
    """
    n_channels = gains.size
    if not local:
        return gains.copy(), np.zeros(n_channels)
    order = rng.permutation(n_channels)
    anchor_u = rng.random()
    gain_u, delay_u = rng.random(n_channels), rng.random(n_channels)
    selected = order[: max(1, int(np.ceil(occupancy * n_channels)))]
    # the anchor carries the ripple: a selected channel of positive gain, else
    # the first such channel in the permutation, added to the selection
    carrying = selected[gains[selected] > 0]
    if carrying.size == 0:
        carrying = order[gains[order] > 0][:1]
        selected = np.append(selected, carrying)
    anchor = carrying[int(anchor_u * carrying.size)]
    stretch = _rest_interval_of(rest, start, end)
    if stretch is None:  # not at rest (an edited table): not delayed
        low = high = 0.0
    else:
        low = min(0.0, max(-maximum_delay, stretch[0] - start))
        high = max(0.0, min(maximum_delay, stretch[1] - end))
    event_gains, delays = np.zeros(n_channels), np.zeros(n_channels)
    event_gains[selected] = gain_range[0] + (gain_range[1] - gain_range[0]) * gain_u[selected]
    delays[selected] = low + (high - low) * delay_u[selected]
    event_gains[anchor], delays[anchor] = 1.0, 0.0
    channel_gains = event_gains * gains
    delays[channel_gains == 0] = 0.0  # a channel without the ripple has no delay
    return channel_gains, delays


def _validated_events(events: pd.DataFrame) -> pd.DataFrame:
    """The latent event table with its columns cast, in the given row order
    and with ``n_participants`` 0, or ValueError: the checks that hold
    whatever the recording (vocabularies, finite values, positive scales,
    powers, unique keys, one type per event, components that fit the type,
    sizes)."""
    given = [name for name in _EVENT_COLUMNS if name != "n_participants"]
    missing = [name for name in given if name not in events.columns]
    if missing:
        msg = f"events is missing the columns {missing}; build it with draw_network_events."
        raise ValueError(msg)
    table = _table(
        _EVENT_COLUMNS,
        {**{name: events[name] for name in given}, "n_participants": np.zeros(len(events))},
    )
    for name, vocabulary in (("event_type", EVENT_TYPES), ("expression", EXPRESSIONS)):
        unknown = sorted(set(table[name]) - set(vocabulary))
        if unknown:
            msg = f"events.{name} has unknown values {unknown}; use {', '.join(vocabulary)}."
            raise ValueError(msg)
    scales = table[["center_time", "rise_sigma", "decay_sigma", "amplitude"]].to_numpy()
    if not (np.all(np.isfinite(scales)) and np.all(scales[:, 1:3] > 0)):
        msg = (
            "events needs finite times and amplitudes and positive rise_sigma and decay_sigma."
        )
        raise ValueError(msg)
    if not table.envelope_power.isin(_ENVELOPE_POWERS).all():
        msg = "events.envelope_power must be 2 or 4."
        raise ValueError(msg)
    if table.duplicated(["event_id", "expression", "component"]).any():
        msg = (
            "events has duplicate (event_id, expression, component) rows; give each "
            "latent event its own event_id (tables drawn separately both start at 0)."
        )
        raise ValueError(msg)
    if table[["event_id", "event_type"]].drop_duplicates()["event_id"].duplicated().any():
        msg = "Each event_id must have one event_type; events mixes types under one id."
        raise ValueError(msg)
    _check_components(table)
    if not (table.amplitude[table.expression == "ripple"] > 0).all():
        msg = "A ripple's amplitude, its SNR, must be positive."
        raise ValueError(msg)
    bursts = table[table.expression == "burst"]
    if not (bursts.participation.between(0, 1).all() and (bursts.amplitude >= 1).all()):
        msg = "A burst needs participation in [0, 1] and amplitude (its gain) of at least 1."
        raise ValueError(msg)
    if not (table.amplitude[table.expression == "sharp_wave"] >= 0).all():
        msg = "A sharp wave's amplitude must be non-negative; it is rendered negative."
        raise ValueError(msg)
    return table


def _check_against_recording(table: pd.DataFrame, time: FloatArray, rate: float) -> None:
    """Raise unless every component lies inside the recording and every
    ripple fits its sampling: frequencies below Nyquist, side scales of at
    least one sample."""
    start = table.center_time - 4 * table.rise_sigma
    end = table.center_time + 4 * table.decay_sigma
    if not ((start >= time[0]) & (end <= time[-1])).all():
        msg = (
            "Every component's span at four side scales must lie inside the recording, "
            f"[{time[0]}, {time[-1]}] s; draw the events on the time you render them on."
        )
        raise ValueError(msg)
    nyquist = rate / 2
    ripples = table[table.expression == "ripple"]
    frequencies = ripples[["frequency_start", "frequency_end"]].to_numpy()
    if not np.all((frequencies > 0) & (frequencies < nyquist)):
        msg = f"Ripple frequencies must lie in (0, {nyquist:g}) Hz, the Nyquist range."
        raise ValueError(msg)
    if not (ripples[["rise_sigma", "decay_sigma"]] >= 1 / rate).all().all():
        msg = (
            f"A ripple's rise_sigma and decay_sigma must be at least one sample, {1 / rate:g} "
            "s, for its sampled waveform to hold its shape."
        )
        raise ValueError(msg)


def _check_components(table: pd.DataFrame) -> None:
    """Raise if an event has a component its type does not: a ripple on a
    ``burst_only`` or ``sharp_wave_only`` event, a burst on a
    ``sharp_wave_only`` one, and so on, or a component numbered other than 0
    (from 0 up for a ``ripple_doublet``'s ripples and sharp waves).
    Components may be left out, to render one expression alone; the event
    keeps its type."""
    allowed = table.event_type.map(_TYPE_EXPRESSIONS)
    has = [
        expression in expressions
        for expression, expressions in zip(table.expression, allowed, strict=True)
    ]
    single = (table.event_type != "ripple_doublet") | (table.expression == "burst")
    numbered = (table.component < 0) | (single & (table.component != 0))
    wrong = ~np.asarray(has, dtype=bool) | numbered.to_numpy()
    if wrong.any():
        row = table[wrong].iloc[0]
        msg = (
            f"Event {row.event_id}, a {row.event_type}, has a {row.expression} component "
            f"{row.component}; a {row.event_type} has "
            f"{', '.join(_TYPE_EXPRESSIONS[row.event_type])}, numbered 0 unless a "
            "doublet's ripples and sharp waves."
        )
        raise ValueError(msg)


def _validated_non_events(
    non_events: pd.DataFrame,
    time: FloatArray,
    rate: float,
    n_channels: int,
    unit_types: StrArray,
) -> pd.DataFrame:
    """The non-event table with its columns cast and sorted by
    ``non_event_id``, or ValueError: known kinds, unique ids, finite times
    and amplitudes, positive side scales, power 2, spans inside the
    recording, and each kind's own columns fit to render."""
    missing = [name for name in _NON_EVENT_COLUMNS if name not in non_events.columns]
    if missing:
        msg = f"non_events is missing the columns {missing}; build it with draw_non_events."
        raise ValueError(msg)
    table = _table(_NON_EVENT_COLUMNS, {name: non_events[name] for name in _NON_EVENT_COLUMNS})
    table = table.sort_values("non_event_id", kind="stable").reset_index(drop=True)
    unknown = sorted(set(table.non_event_type) - set(NON_EVENT_TYPES))
    if unknown:
        msg = f"non_events has unknown types {unknown}; use {', '.join(NON_EVENT_TYPES)}."
        raise ValueError(msg)
    if table.non_event_id.duplicated().any():
        msg = "non_events has duplicate non_event_id values."
        raise ValueError(msg)
    scales = table[["center_time", "rise_sigma", "decay_sigma", "amplitude"]].to_numpy()
    if not (np.all(np.isfinite(scales)) and np.all(scales[:, 1:3] > 0)):
        msg = "non_events needs finite times and amplitudes and positive side scales."
        raise ValueError(msg)
    if not (table.envelope_power == 2).all():
        msg = "non_events.envelope_power must be 2."
        raise ValueError(msg)
    start = table.center_time - 4 * table.rise_sigma
    end = table.center_time + 4 * table.decay_sigma
    if not ((start >= time[0]) & (end <= time[-1])).all():
        msg = (
            "Every non-event's span at four side scales must lie inside the recording, "
            f"[{time[0]}, {time[-1]}] s; draw them on the time you render them on."
        )
        raise ValueError(msg)

    kind = table.non_event_type
    leakage, gamma, theta = (
        table[kind == name] for name in ("spike_leakage", "fast_gamma", "theta_burst")
    )
    n_pyramidal = int(np.isin(unit_types, ["place", "pyramidal"]).sum())
    last_sample = (
        (leakage.center_time + (leakage.n_spikes - 1) * leakage.isi / 2 - time[0]) * rate
        + _SPIKE_WAVEFORM.size
        - 1
    )
    fits = (
        leakage.channel.between(0, n_channels - 1)
        & leakage.n_units.between(1, n_pyramidal)
        & (leakage.n_spikes >= 2)
        & (leakage.isi >= 1 / rate)
        & (leakage.amplitude >= 0)
        & np.isclose(leakage.rise_sigma, (leakage.n_spikes - 1) * leakage.isi / 6)
        & (leakage.rise_sigma == leakage.decay_sigma)
        & (np.round(last_sample) < time.size)
    )
    if not fits.all():
        msg = (
            f"A spike_leakage row needs a channel in 0..{n_channels - 1}, 1 to "
            f"{n_pyramidal} units (the place and other pyramidal units), at least 2 "
            f"spikes at intervals of at least one sample, {1 / rate:g} s, a non-negative "
            "amplitude, side scales of (n_spikes - 1) isi / 6, and its spikes inside "
            f"the recording; non-event {leakage.non_event_id[~fits].iloc[0]} has not."
        )
        raise ValueError(msg)
    if not (table.amplitude[kind == "emg"] >= 0).all():
        msg = "An emg row's amplitude, its peak standard deviation, must be non-negative."
        raise ValueError(msg)
    if (kind == "emg").any() and rate / 2 <= _EMG_HIGH_PASS:
        msg = (
            f"EMG is high-passed at {_EMG_HIGH_PASS:g} Hz, at or above the Nyquist "
            f"frequency, {rate / 2:g} Hz."
        )
        raise ValueError(msg)
    fits = (gamma.amplitude > 0) & (gamma[["rise_sigma", "decay_sigma"]] >= 1 / rate).all(
        axis=1
    )
    if not fits.all():
        msg = (
            "A fast_gamma row needs a positive amplitude, its SNR, and side scales of at "
            f"least one sample, {1 / rate:g} s; non-event "
            f"{gamma.non_event_id[~fits].iloc[0]} has not."
        )
        raise ValueError(msg)
    for (low, high), rows in gamma.groupby(["snr_band_low", "snr_band_high"], dropna=False):
        _check_sizing_band(
            f"non-event {rows.non_event_id.iloc[0]}'s (snr_band_low, snr_band_high)",
            (low, high), (rows.frequency.min(), rows.frequency.max()), rate,
        )  # fmt: skip
    n_place = int((unit_types == "place").sum())
    fits = theta.n_units.between(1, n_place) & (theta.amplitude >= 1)
    if not fits.all():
        msg = (
            f"A theta_burst row needs 1 to {n_place} units (the place units) and an "
            f"amplitude, its gain, of at least 1; non-event "
            f"{theta.non_event_id[~fits].iloc[0]} has not."
        )
        raise ValueError(msg)
    return table


def _render_non_events(
    table: pd.DataFrame,
    time: FloatArray,
    rate: float,
    lfps: FloatArray,
    radiatum: FloatArray,
    gains: FloatArray,
    band_noise_sds: Mapping[tuple[float, float], float],
    unit_types: StrArray,
    rng: np.random.Generator,
) -> tuple[list[list[tuple[slice, FloatArray]]], list[tuple[IntArray, IntArray]]]:
    """Add each non-event's LFP to ``lfps`` and ``radiatum`` in place, in
    ``non_event_id`` order, and return the theta bursts' intensity factors
    per unit and the leaked spikes (units, samples).

    Draws, per row: a leakage burst's units, an EMG burst's noise, a gamma
    burst's phase at its centre, a theta burst's units.
    """
    multipliers: list[list[tuple[slice, FloatArray]]] = [[] for _ in unit_types]
    leaked: list[tuple[IntArray, IntArray]] = []
    pyramidal = np.flatnonzero(np.isin(unit_types, ["place", "pyramidal"]))
    place = np.flatnonzero(unit_types == "place")
    if (table.non_event_type == "emg").any():
        sos = signal.butter(4, _EMG_HIGH_PASS, btype="highpass", fs=rate, output="sos")
        # forward and backward: the standard deviation of filtered unit white noise
        emg_sd = float(np.sqrt(np.mean(np.abs(signal.sosfreqz(sos, 4096, fs=rate)[1]) ** 4)))
    for row in table.itertuples():
        if row.non_event_type == "spike_leakage":
            units = rng.choice(pyramidal, size=row.n_units, replace=False)
            offsets = (np.arange(row.n_spikes) - (row.n_spikes - 1) / 2) * row.isi
            samples = np.round((row.center_time + offsets - time[0]) * rate).astype(np.int64)
            leaked.append((units, samples))
            for shift, value in enumerate(_SPIKE_WAVEFORM):
                lfps[samples + shift, row.channel] += row.amplitude * value
            continue
        window, envelope = _event_envelope(
            time, row.center_time, row.rise_sigma, row.decay_sigma, row.envelope_power
        )
        if row.non_event_type == "emg":
            # unpadded: the filter's edge transients fall where the envelope
            # is below 1e-14 of its peak
            noise = signal.sosfiltfilt(sos, rng.standard_normal(envelope.size), padlen=0)
            burst = (row.amplitude / emg_sd) * noise * envelope
            lfps[window] += burst[:, np.newaxis]
            radiatum[window] += burst
        elif row.non_event_type == "fast_gamma":
            band = (row.snr_band_low, row.snr_band_high)
            window, wave = _render_ripple(
                time, row.center_time, row.rise_sigma, row.decay_sigma, row.frequency,
                row.frequency, rng.uniform(0.0, 2 * np.pi), row.envelope_power,
            )  # fmt: skip
            scale = _scale_to_snr(wave, row.amplitude, band_noise_sds[band], rate, band)
            lfps[window] += scale * wave[:, np.newaxis] * gains
        else:  # theta_burst
            factor = 1.0 + (row.amplitude - 1.0) * envelope
            for unit in rng.choice(place, size=row.n_units, replace=False):
                multipliers[unit].append((window, factor))
    return multipliers, leaked


def _unit_layout(
    unit_counts: Mapping[str, int] | None,
    baseline_rate: Mapping[str, tuple[float, float]] | None,
) -> tuple[StrArray, dict[str, tuple[float, float]]]:
    """Each unit's type, in ``UNIT_TYPES`` order, and each type's rate range."""
    counts = _REFERENCE_UNIT_COUNTS if unit_counts is None else dict(unit_counts)
    rates = dict(_REFERENCE_BASELINE_RATES)
    rates.update({} if baseline_rate is None else dict(baseline_rate))
    for name, given in (("unit_counts", counts), ("baseline_rate", rates)):
        unknown = sorted(set(given) - set(UNIT_TYPES))
        if unknown:
            msg = f"{name} has unknown unit types {unknown}; use {', '.join(UNIT_TYPES)}."
            raise ValueError(msg)
    for unit_type, count in counts.items():
        _check_whole_number(f"unit_counts[{unit_type!r}]", count, 0)
    if sum(counts.values()) < 1:
        msg = "unit_counts must give at least one unit."
        raise ValueError(msg)
    ranges = {
        unit_type: _check_range(f"baseline_rate[{unit_type!r}]", rates[unit_type], lower=0)
        for unit_type in UNIT_TYPES
    }
    types = np.array(
        [unit_type for unit_type in UNIT_TYPES for _ in range(int(counts.get(unit_type, 0)))],
        dtype="<U11",
    )
    return types, ranges


def _draw_units(
    time: FloatArray,
    events: pd.DataFrame,
    unit_types: StrArray,
    rate_ranges: Mapping[str, tuple[float, float]],
    *,
    interneuron_gain: float,
    spike_model: str,
    refractory_period: float,
    step: float,
    streams: Mapping[str, np.random.Generator],
    multipliers: Sequence[Sequence[tuple[slice, FloatArray]]] | None = None,
) -> tuple[FloatArray, FloatArray, IntArray]:
    """Baseline rates, spike counts ``(n_time, n_units)`` and the number of
    place and pyramidal participants of each burst row.

    Participants: per burst row in table order, one uniform per unit; a place
    unit takes part below ``participation``, another pyramidal unit below
    half of it. A participant's intensity gains ``(amplitude - 1)`` times the
    burst's envelope; every interneuron gains ``(interneuron_gain - 1)`` times
    the envelope of each event's ripples (their maximum, for a doublet).
    ``multipliers[unit]`` then scales a unit's intensity in each window by
    each factor (a theta burst). Spikes, unit by unit: Poisson counts of the
    intensity times ``step`` (the sampling interval of the resolved rate, not
    the timestamps' spacing, which rounds far from zero), or, for
    ``"refractory"``, at most one per sample, emitted with probability ``1 -
    exp(-intensity step)`` once ``refractory_period`` has passed since the
    unit's last spike.
    """
    n_time, n_units = time.size, unit_types.size
    rates = np.empty(n_units)
    for unit_type in UNIT_TYPES:
        is_type = unit_types == unit_type
        low, high = rate_ranges[unit_type]
        rates[is_type] = streams["baseline_rates"].uniform(low, high, size=is_type.sum())
    place = unit_types == "place"
    pyramidal = unit_types == "pyramidal"
    gains: list[list[tuple[slice, FloatArray]]] = [[] for _ in range(n_units)]
    bursts = events[events.expression == "burst"]
    n_participants = np.zeros(len(bursts), dtype=np.int64)
    for row, burst in enumerate(bursts.itertuples()):
        u = streams["participants"].random(n_units)
        takes_part = (place & (u < burst.participation)) | (
            pyramidal & (u < burst.participation / 2)
        )
        n_participants[row] = takes_part.sum()
        window, envelope = _event_envelope(
            time, burst.center_time, burst.rise_sigma, burst.decay_sigma, burst.envelope_power
        )
        gain = (burst.amplitude - 1.0) * envelope
        for participant in np.flatnonzero(takes_part):
            gains[participant].append((window, gain))
    interneurons = np.flatnonzero(unit_types == "interneuron")
    for _, ripples in events[events.expression == "ripple"].groupby("event_id", sort=True):
        envelopes = [
            _event_envelope(time, r.center_time, r.rise_sigma, r.decay_sigma, r.envelope_power)
            for r in ripples.itertuples()
        ]
        first = min(window.start for window, _ in envelopes)
        last = max(window.stop for window, _ in envelopes)
        union = np.zeros(last - first)
        for window, envelope in envelopes:
            part = slice(window.start - first, window.stop - first)
            union[part] = np.maximum(union[part], envelope)
        interneuron_modulation = (interneuron_gain - 1.0) * union
        for interneuron in interneurons:
            gains[interneuron].append((slice(first, last), interneuron_modulation))
    spikes = np.zeros((n_time, n_units))
    tolerance = _bound_tolerance(time)
    # units are drawn in order into a few rows, then written to the (n_time,
    # n_units) array a block at a time rather than a strided column at a time
    block = np.zeros((min(_SPIKE_BLOCK, n_units), n_time))
    for first_unit in range(0, n_units, _SPIKE_BLOCK):
        units = range(first_unit, min(first_unit + _SPIKE_BLOCK, n_units))
        for row, unit in enumerate(units):
            intensity = np.full(n_time, rates[unit] * step)
            for window, gain in gains[unit]:
                intensity[window] += rates[unit] * step * gain
            for window, factor in [] if multipliers is None else multipliers[unit]:
                intensity[window] *= factor
            if spike_model == "poisson":
                block[row] = streams["spikes"].poisson(intensity)
                continue
            block[row] = 0.0
            uniform = streams["spikes"].random(n_time)
            last_spike = -np.inf
            for sample in np.flatnonzero(uniform < -np.expm1(-intensity)):
                if time[sample] - last_spike >= refractory_period - tolerance:
                    block[row, sample] = 1.0
                    last_spike = time[sample]
        spikes[:, first_unit : first_unit + len(units)] = block[: len(units)].T
    return rates, spikes, n_participants


@explain_call_errors
def simulate_network_session(
    time: ArrayLike,
    events: pd.DataFrame,
    *,
    non_events: pd.DataFrame | None = None,
    n_channels: int = 4,
    unit_counts: Mapping[str, int] | None = None,
    baseline_rate: Mapping[str, tuple[float, float]] | None = None,
    channel_gains: Sequence[float] | FloatArray | None = None,
    spatial_profile: str = "global",
    channel_occupancy: float = 1.0,
    channel_gain_range: tuple[float, float] = (1.0, 1.0),
    channel_delay: float = 0.0,
    shared_noise_fraction: float = 0.5,
    noise_type: NoiseType = "pink",
    noise_amplitude: float = 1.3,
    noise_log_amplitude: float = 0.0,
    noise_modulation_period: float = 60.0,
    sharp_wave_leak: float = 0.3,
    ripple_leak: float = 0.3,
    interneuron_gain: float = 3.0,
    spike_model: str = "poisson",
    refractory_period: float = 0.002,
    running_intervals: ArrayLike | None = None,
    peak_speed: float = 30.0,
    theta_amplitude: float = 4.0,
    delta_amplitude: float = 4.0,
    rng: int | np.random.Generator | None = None,
    sampling_frequency: float | None = None,
) -> SimulatedSession:
    """Render a table of latent network events into every input the
    detectors take, with the truth.

    Each ripple row becomes a chirped oscillation on the pyramidal-layer
    channels (and ``ripple_leak`` of it on the radiatum channel), each sharp
    wave a negative deflection on the radiatum channel (and
    ``sharp_wave_leak`` of it, positive, on channel 0), and each burst a rise
    in the intensity of the place and pyramidal units it recruits; every
    interneuron follows each event's ripples. The noise, slow field and speed
    are ``simulate_session``'s. Non-events from ``draw_non_events``, if
    given, are rendered too. Draw the table with ``draw_network_events``,
    edit it if you like, and take truth windows with ``truth_windows``.

    Parameters
    ----------
    time : array_like, shape (n_time,)
        Sample timestamps in seconds, increasing.
    events : pandas.DataFrame
        A latent event table, as ``draw_network_events`` returns; rendered in
        its sorted order whatever the order given. Components may be left
        out, to render one expression alone; ``event_id`` values are kept as
        given, so a filtered table's ids need not run from 0. Columns beyond
        the table's are not kept.
    non_events : pandas.DataFrame, optional
        A non-event table, as ``draw_non_events`` returns, rendered in
        ``non_event_id`` order: leaked spikes and their waveforms, EMG,
        gamma bursts and theta bursts, as that function describes. The
        leaked spikes are added to the drawn counts, so they neither change
        those counts nor obey ``refractory_period``. Default None: none.
    n_channels : int, optional
        Pyramidal-layer channels; the radiatum channel is separate. Default 4.
    unit_counts : mapping of str to int, optional
        Units of each of ``UNIT_TYPES``; a type left out has none. Default
        None: 40 place, 10 other pyramidal, 10 interneurons, in that order.
    baseline_rate : mapping of str to (float, float), optional
        Range, in spikes/s, of each type's baseline intensity, drawn per unit;
        a type left out keeps its default. Default None: place (0.1, 0.5),
        pyramidal (0.5, 1.5), interneuron (8, 15); see ``draw_network_events``'
        Notes for their sources.
    channel_gains : sequence of float, shape (n_channels,), optional
        Each channel's ripple gain, as in ``simulate_multichannel_LFP``.
        Default None: 1 on every channel.
    spatial_profile : {'global', 'local'}, optional
        ``'global'``: every ripple on every channel at ``channel_gains``, with
        no delay. ``'local'``: each ripple on
        ``max(1, ceil(channel_occupancy * n_channels))`` channels drawn anew,
        one of them the anchor at gain 1 and no delay, the others at a gain
        from ``channel_gain_range`` and a delay uniform on
        ``[-channel_delay, channel_delay]``, narrowed so the delayed ripple's
        span at four side scales stays in its stretch of rest (a ripple
        outside every stretch is not delayed); all times the recording-wide
        ``channel_gains``, the anchor always on a channel of positive gain.
        At the defaults of the next three options, 'local' renders as
        'global'. Default 'global'.
    channel_occupancy : float, optional
        Fraction of channels a local ripple is on, in (0, 1]; used only with
        ``'local'``. Default 1.
    channel_gain_range : (float, float), optional
        Range of a local ripple's non-anchor gains, non-negative; used only
        with ``'local'``. Default (1, 1).
    channel_delay : float, optional
        Largest delay in seconds of a local ripple on a non-anchor channel;
        the whole waveform moves. Used only with ``'local'``. Default 0.
    shared_noise_fraction, noise_type, noise_amplitude : optional
        As in ``simulate_multichannel_LFP``; the radiatum channel's noise is
        drawn with the others. Defaults 0.5, 'pink', 1.3.
    noise_log_amplitude : float, optional
        ``a >= 0`` in the background's slow gain ``exp(a sin(2 pi (t - t0) / T
        + phase))``, scaled to unit RMS over the recording and applied to the
        noise only. Ripples are sized against the stationary noise, so their
        local SNR rises and falls with it. Default 0: stationary.
    noise_modulation_period : float, optional
        ``T`` in seconds, positive; no effect when ``noise_log_amplitude`` is
        0. Default 60.
    sharp_wave_leak, ripple_leak : float, optional
        As in ``simulate_sharp_wave_ripple_pair``, finite and non-negative.
        Default 0.3 each.
    interneuron_gain : float, optional
        Every interneuron's peak intensity during a ripple relative to its
        baseline, at least 1. Default 3.
    spike_model : {'poisson', 'refractory'}, optional
        ``'poisson'``: Poisson counts per sample of the intensity times the
        step, as ``simulate_multiunit`` draws them. ``'refractory'``: at most
        one spike per sample, emitted with probability ``1 - exp(-intensity *
        step)`` once ``refractory_period`` has passed since the unit's last
        spike; the dead time lowers the realized rate below the intensity.
        Default 'poisson'.
    refractory_period : float, optional
        Dead time in seconds, finite and non-negative; used only with
        ``'refractory'``. Default 0.002.
    running_intervals : array_like, shape (n_bouts, 2), optional
        Running bouts, as given to ``draw_network_events``; they set the speed,
        the theta and delta, and the stretches of rest a local delay keeps a
        ripple in. Default None: at rest throughout.
    peak_speed : float, optional
        As in ``simulate_session``. Default 30.
    theta_amplitude, delta_amplitude : float, optional
        As in ``simulate_session``, finite and non-negative, added to every
        channel. Default 4 each.
    rng : int or numpy.random.Generator, optional
        Seed, or a Generator. One draw from it seeds eight independent
        streams, in this order: noise, ripple phases, spatial profiles, noise
        modulation, baseline rates, burst participants, non-events (so
        adding them leaves the others unchanged), spikes; each is used in
        table order. A theta burst changes its units' intensities, and so,
        under Poisson spiking, the counts drawn after them. So a noise-only rendering (an empty
        table) with the same seed has the same noise, and the spike
        model or spatial profile does not change anything drawn from another
        stream.
    sampling_frequency : float, optional
        As in ``simulate_LFP``. Recorded in the result. Give it when the
        timestamps lie far from zero (a Unix time): there they round, the
        median step no longer gives the rate exactly, and the ripples are
        sized with a filter designed for the rate inferred. It must agree
        with the timestamps' step.

    Returns
    -------
    SimulatedSession
        ``lfps`` the pyramidal-layer channels, ``raw_lfp`` channel 0,
        ``sharp_wave_lfp`` the radiatum channel, ``multiunit`` the spike
        counts per sample; ``events`` the table sorted, with
        ``n_participants`` (place and pyramidal units recruited) filled in
        on burst rows; ``ripple_channels`` every ripple's gain and delay on
        every channel; ``unit_types``, ``baseline_rates`` and
        ``running_intervals``; and one ``ripple_times``,
        ``ripple_durations`` and ``ripple_frequencies`` entry per ripple
        row, so ``ripple_windows`` is each ripple's span at three side
        scales.

    Raises
    ------
    ValueError
        If ``time`` is not 1-D with two or more increasing, evenly spaced
        samples, or ``sampling_frequency`` disagrees with its step; ``events``
        lacks a column, has an ``event_id``, ``component``,
        ``envelope_power`` or ``n_participants`` that is not a whole number,
        a component its event type does not have, a duplicate (``event_id``,
        ``expression``, ``component``) row, an ``event_id`` of two types, an
        unknown type or expression, a non-finite time or amplitude, a side
        scale that is not positive, a component whose span at four side
        scales leaves the recording, an envelope power other than 2 or 4, a
        ripple frequency outside (0,
        Nyquist) or outside the ripple band its SNR is measured in, a ripple
        side scale under one sample, a non-positive ripple SNR, a negative
        sharp-wave amplitude, or a burst with participation outside [0, 1] or
        a gain below 1; there are ripples and ``noise_amplitude`` is 0 (an
        SNR needs a background) or every channel gain is 0; ``n_channels`` is
        below 1, ``channel_gains`` has the wrong length or an entry that is
        negative or not finite; a leak or the theta or delta amplitude is
        negative or not finite; ``unit_counts`` names an unknown type or a
        count that is not a non-negative integer, or gives no unit;
        ``baseline_rate`` names an unknown type or a range that is not
        non-negative; ``non_events`` lacks a column, has an unknown type, a
        duplicate ``non_event_id``, a non-finite time or amplitude, a side
        scale that is not positive, an envelope power other than 2, a span
        at four side scales outside the recording, or a row its kind cannot
        render (a leakage burst on a channel that does not exist, with more
        units than the place and pyramidal units, fewer than 2 spikes, an
        interval under a sample, side scales other than ``(n_spikes - 1) isi
        / 6`` or spikes past the end; an EMG burst of negative amplitude, or
        at a rate with no room above 100 Hz; a gamma burst of non-positive
        SNR, side scales under a sample, or a sizing band that is not finite,
        ordered, below Nyquist, holding its frequency and possible to
        filter; a theta burst with more units than the place units or a gain
        below 1), or has gamma bursts while ``noise_amplitude`` or every
        channel gain is 0; ``spatial_profile`` or ``spike_model`` is unknown;
        ``channel_occupancy`` lies outside (0, 1]; ``channel_gain_range`` is
        not a non-negative range; ``channel_delay``, ``noise_log_amplitude``
        or ``refractory_period`` is negative or not finite;
        ``noise_modulation_period`` is not positive; ``interneuron_gain`` is
        below 1; or the running intervals or noise settings are invalid, as
        in ``simulate_session``.

    See Also
    --------
    draw_network_events : draws the table and records where its reference
        values come from.
    draw_non_events : draws the non-events.
    truth_windows : each event's or component's interval at any fraction of
        its envelope's peak.

    Notes
    -----
    A ripple's ``amplitude`` is its nominal SNR: the peak of the unit
    ripple, as rendered, after ``filter_ripple_band`` over the standard
    deviation of the filtered stationary noise on channel 0, before channel
    gains and noise modulation (``simulate_LFP``'s ``ripple_snr``). The
    carrier's phase at the ripple's centre is drawn uniformly.

    Examples
    --------
    >>> time = simulate_time(60 * 1500, 1500)
    >>> events = draw_network_events(time, running_intervals=[(20.0, 35.0)], rng=0)
    >>> session = simulate_network_session(
    ...     time, events, running_intervals=[(20.0, 35.0)], rng=1
    ... )
    >>> session.lfps.shape, session.multiunit.shape
    ((90000, 4), (90000, 60))
    >>> session.unit_types[[0, 40, 50]].tolist()
    ['place', 'pyramidal', 'interneuron']

    """
    time, rate = _checked_time(time, sampling_frequency)
    table = _sorted_events(_validated_events(events))
    _check_against_recording(table, time, rate)
    if n_channels < 1:
        msg = f"n_channels must be at least 1, got {n_channels}."
        raise ValueError(msg)
    gains = _channel_gains(channel_gains, n_channels)
    if not np.all(np.isfinite(gains) & (gains >= 0)):
        msg = f"channel_gains must be finite and non-negative, got {gains}."
        raise ValueError(msg)
    if (table.expression == "ripple").any() and not (gains > 0).any():
        msg = "channel_gains are all 0: no channel would carry the ripples."
        raise ValueError(msg)
    for name, value in (
        ("sharp_wave_leak", sharp_wave_leak),
        ("ripple_leak", ripple_leak),
        ("theta_amplitude", theta_amplitude),
        ("delta_amplitude", delta_amplitude),
    ):
        _check_scalar(name, value)
    unit_types, rate_ranges = _unit_layout(unit_counts, baseline_rate)
    non_event_table = (
        _empty(_NON_EVENT_COLUMNS)
        if non_events is None
        else _validated_non_events(non_events, time, rate, n_channels, unit_types)
    )
    gamma = non_event_table[non_event_table.non_event_type == "fast_gamma"]
    if len(gamma) and not ((gains > 0).any() and noise_amplitude > 0):
        msg = (
            "Fast gamma is sized against the background on the channels that carry it: "
            "noise_amplitude and some channel gain must be > 0."
        )
        raise ValueError(msg)
    _check_choice("spatial_profile", spatial_profile, SPATIAL_PROFILES)
    _check_choice("spike_model", spike_model, SPIKE_MODELS)
    if not 0 < channel_occupancy <= 1:
        msg = f"channel_occupancy must lie in (0, 1], got {channel_occupancy}."
        raise ValueError(msg)
    gain_range = _check_range("channel_gain_range", channel_gain_range, lower=0)
    channel_delay = _check_scalar("channel_delay", channel_delay)
    _validate_sizes(None, None, noise_amplitude)
    noise_log_amplitude = _check_scalar("noise_log_amplitude", noise_log_amplitude)
    noise_modulation_period = _check_scalar(
        "noise_modulation_period", noise_modulation_period, lower_strict=True
    )
    interneuron_gain = _check_scalar("interneuron_gain", interneuron_gain, lower=1.0)
    refractory_period = _check_scalar("refractory_period", refractory_period)
    ripples = table[table.expression == "ripple"]
    if len(ripples) and noise_amplitude <= 0:
        msg = "Ripples are sized against the background: noise_amplitude must be > 0."
        raise ValueError(msg)
    bouts = _bouts(running_intervals)
    seeds = _generator(rng).integers(np.iinfo(np.int64).max, size=len(_RENDER_STREAMS))
    streams = {
        name: np.random.default_rng(int(seed))
        for name, seed in zip(_RENDER_STREAMS, seeds, strict=True)
    }

    stationary = _correlated_noise(
        time.size, n_channels + 1, noise_type, noise_amplitude, shared_noise_fraction,
        streams["noise"],
    )  # fmt: skip
    modulation_phase = streams["noise_modulation"].uniform(0.0, 2 * np.pi)
    channels = (
        stationary
        * _noise_modulation(
            time, noise_log_amplitude, noise_modulation_period, modulation_phase
        )[:, np.newaxis]
    )
    lfps, radiatum = channels[:, :n_channels], channels[:, n_channels]

    band_noise_sd = (
        float(filter_ripple_band(stationary[:, 0], sampling_frequency=rate).std())
        if len(ripples)
        else np.nan
    )
    rest = _rest_intervals(time, bouts)
    ripple_gains = np.zeros((len(ripples), n_channels))
    delays = np.zeros((len(ripples), n_channels))
    for index, ripple in enumerate(ripples.itertuples()):
        phase = streams["ripple_phases"].uniform(0.0, 2 * np.pi)
        render = partial(
            _render_ripple, time, rise_sigma=ripple.rise_sigma,
            decay_sigma=ripple.decay_sigma, frequency_start=ripple.frequency_start,
            frequency_end=ripple.frequency_end, phase=phase, power=ripple.envelope_power,
        )  # fmt: skip
        window, latent = render(ripple.center_time)
        filtered_peak = _filtered_peak(latent, rate)
        # the filter's gain on the ripple: its SNR is measured in the ripple band
        if not filtered_peak / np.abs(latent).max() >= 0.1:
            msg = (
                f"The ripple of event {ripple.event_id} at {ripple.frequency_start:g}-"
                f"{ripple.frequency_end:g} Hz lies outside the ripple band its SNR is "
                "measured in (filter_ripple_band): sizing it would magnify it without bound."
            )
            raise ValueError(msg)
        scale = float(ripple.amplitude * band_noise_sd / filtered_peak)
        radiatum[window] += ripple_leak * scale * latent
        ripple_gains[index], delays[index] = _spatial_profile(
            ripple.center_time - 4 * ripple.rise_sigma,
            ripple.center_time + 4 * ripple.decay_sigma,
            rest, gains, local=spatial_profile == "local", occupancy=channel_occupancy,
            gain_range=gain_range, maximum_delay=channel_delay,
            rng=streams["spatial_profiles"],
        )  # fmt: skip
        for channel in np.flatnonzero(ripple_gains[index]):
            delay = delays[index, channel]
            shifted_window, shifted = (
                (window, latent) if delay == 0 else render(ripple.center_time + delay)
            )
            lfps[shifted_window, channel] += ripple_gains[index, channel] * scale * shifted
    gamma_noise_sds = {
        band: float(
            filter_ripple_band(stationary[:, 0], sampling_frequency=rate, band=band).std()
        )
        for band in sorted(set(zip(gamma.snr_band_low, gamma.snr_band_high, strict=True)))
    }
    del stationary, channels  # the slow field or the views keep what is needed

    for sharp_wave in table[table.expression == "sharp_wave"].itertuples():
        window, envelope = _event_envelope(
            time, sharp_wave.center_time, sharp_wave.rise_sigma, sharp_wave.decay_sigma,
            sharp_wave.envelope_power,
        )  # fmt: skip
        radiatum[window] -= sharp_wave.amplitude * envelope
        lfps[window, 0] += sharp_wave_leak * sharp_wave.amplitude * envelope
    multipliers, leaked = _render_non_events(
        non_event_table, time, rate, lfps, radiatum, gains, gamma_noise_sds, unit_types,
        streams["non_events"],
    )  # fmt: skip

    if theta_amplitude > 0 or delta_amplitude > 0:
        slow = simulate_theta_delta(
            time, bouts, theta_amplitude=theta_amplitude, delta_amplitude=delta_amplitude
        )
        lfps = lfps + slow[:, np.newaxis]
        radiatum = radiatum + slow
    speed = simulate_speed(time, bouts, peak_speed=peak_speed)

    baseline_rates, multiunit, n_participants = _draw_units(
        time, table, unit_types, rate_ranges,
        interneuron_gain=interneuron_gain, spike_model=spike_model,
        refractory_period=refractory_period, step=1 / rate, streams=streams,
        multipliers=multipliers,
    )  # fmt: skip
    for units, samples in leaked:  # after the draw: the drawn counts do not move
        multiunit[np.ix_(samples, units)] += 1.0
    table.loc[table.expression == "burst", "n_participants"] = n_participants
    ripple_channels = _table(
        _RIPPLE_CHANNEL_COLUMNS,
        {
            "event_id": np.repeat(ripples.event_id.to_numpy(), n_channels),
            "component": np.repeat(ripples.component.to_numpy(), n_channels),
            "channel": np.tile(np.arange(n_channels), len(ripples)),
            "gain": ripple_gains.ravel(),
            "delay_s": delays.ravel(),
        },
    )
    return SimulatedSession(
        time=time,
        lfps=lfps,
        raw_lfp=lfps[:, 0].copy(),
        sharp_wave_lfp=radiatum,
        multiunit=multiunit,
        speed=speed,
        ripple_times=(
            ripples.center_time + 1.5 * (ripples.decay_sigma - ripples.rise_sigma)
        ).to_numpy(),
        ripple_durations=(3 * (ripples.rise_sigma + ripples.decay_sigma)).to_numpy(),
        ripple_frequencies=ripples.frequency_start.to_numpy(),
        artifact_times=np.empty(0),
        sampling_frequency=rate,
        events=table,
        non_events=non_event_table,
        unit_types=unit_types,
        baseline_rates=baseline_rates,
        running_intervals=bouts,
        ripple_channels=ripple_channels,
    )


_PEAK_PRIORITY = {"ripple": 0, "burst": 1, "sharp_wave": 2}


@explain_call_errors
def truth_windows(
    table: pd.DataFrame,
    fraction: float = 0.1,
    expression: str | None = None,
) -> pd.DataFrame:
    """Where each simulated event, component or non-event is at or above a
    fraction of its envelope's peak.

    A component's window is ``[center_time - k rise_sigma, center_time + k
    decay_sigma]``, where its latent envelope crosses ``fraction`` of its
    peak: ``k = sqrt(2 ln 2) (ln(1 / fraction) / ln 2) ** (1 / p)`` for
    envelope power ``p``, ``sqrt(-2 ln fraction)`` for a Gaussian. These are
    the latent bounds, anchored to the component's centre; a ripple on a
    delayed channel (``SimulatedSession.ripple_channels``) is moved by its
    delay, and a sampled Hilbert envelope can cross a little elsewhere.

    Parameters
    ----------
    table : pandas.DataFrame
        ``SimulatedSession.events`` (the latent event table) or
        ``SimulatedSession.non_events``.
    fraction : float, optional
        Fraction of the peak, in (0, 1). Default 0.1; higher fractions give
        narrower windows, and at 0.5 each is the width at half maximum.
    expression : {None, 'ripple', 'sharp_wave', 'burst', 'network'}, optional
        For the event table: one row per component of that expression, or,
        for ``'network'``, one row per latent event spanning the union of its
        components' windows. Default None: one row per component of any
        expression. Must be None for the non-event table.

    Returns
    -------
    windows : pandas.DataFrame
        Columns ``id`` (``event_id`` or ``non_event_id``), ``type``
        (``event_type`` or ``non_event_type``), ``start_time``,
        ``end_time`` and ``peak_time`` (the envelope's peak; for a network
        event its ripple's, component 0's for a doublet, else its burst's, else
        its sharp wave's), and, per component of the event table,
        ``expression`` and ``component``. Rows are in the table's order, or
        by ``id`` for ``'network'``, and in the same order at every
        ``fraction``, so windows at two fractions pair up by position.

    Raises
    ------
    ValueError
        If ``fraction`` is not in (0, 1), ``expression`` is not one of the
        choices, or is given for a non-event table, ``table`` is neither
        table, or an envelope power is not 2 or 4.

    Examples
    --------
    >>> time = simulate_time(60 * 1500, 1500)
    >>> events = draw_network_events(time, rng=0)
    >>> ripples = truth_windows(events, 0.1, expression="ripple")
    >>> list(ripples.columns)
    ['id', 'type', 'start_time', 'end_time', 'peak_time', 'expression', 'component']
    >>> half = truth_windows(events, 0.5, expression="ripple")
    >>> narrower = (half.end_time - half.start_time) < (ripples.end_time - ripples.start_time)
    >>> bool(narrower.all())
    True
    >>> list(truth_windows(events, expression="network").columns)
    ['id', 'type', 'start_time', 'end_time', 'peak_time']

    """
    if not 0 < fraction < 1:
        msg = f"fraction must lie in (0, 1), got {fraction}."
        raise ValueError(msg)
    if "event_type" in table.columns:
        id_column, type_column = "event_id", "event_type"
    elif "non_event_type" in table.columns:
        id_column, type_column = "non_event_id", "non_event_type"
    else:
        msg = (
            "table must be a simulated event or non-event table (with event_type or "
            "non_event_type)."
        )
        raise ValueError(msg)
    events = id_column == "event_id"
    if events:
        table = _validated_events(table)
    elif not table["envelope_power"].isin(_ENVELOPE_POWERS).all():
        msg = "table.envelope_power must be 2 or 4."
        raise ValueError(msg)
    if not events and expression is not None:
        msg = "A non-event table has no expressions; leave expression as None."
        raise ValueError(msg)
    rows = table
    if expression is not None:
        _check_choice("expression", expression, (*EXPRESSIONS, "network"))
        if expression != "network":
            rows = table[table["expression"] == expression]
    power = rows["envelope_power"].to_numpy(dtype=float)
    k = np.sqrt(2 * np.log(2)) * (np.log(1 / fraction) / np.log(2)) ** (1 / power)
    center = rows["center_time"].to_numpy(dtype=float)
    windows = pd.DataFrame(
        {
            "id": rows[id_column].to_numpy(dtype=np.int64),
            "type": rows[type_column].to_numpy(),
            "start_time": center - k * rows["rise_sigma"].to_numpy(dtype=float),
            "end_time": center + k * rows["decay_sigma"].to_numpy(dtype=float),
            "peak_time": center,
        }
    )
    if events and expression != "network":
        windows["expression"] = rows["expression"].to_numpy()
        windows["component"] = rows["component"].to_numpy(dtype=np.int64)
    if expression == "network":
        priority = rows["expression"].map(_PEAK_PRIORITY).to_numpy()
        order = np.lexsort((rows["component"].to_numpy(), priority, windows["id"].to_numpy()))
        first = windows.iloc[order].groupby("id", sort=True).first()
        span = windows.groupby("id", sort=True).agg(
            start_time=("start_time", "min"), end_time=("end_time", "max")
        )
        windows = span.assign(type=first["type"], peak_time=first["peak_time"]).reset_index()
        windows = windows[["id", "type", "start_time", "end_time", "peak_time"]]
    labels = {name: str for name in ("type", "expression") if name in windows.columns}
    return windows.astype({"id": "int64", **labels}).reset_index(drop=True)

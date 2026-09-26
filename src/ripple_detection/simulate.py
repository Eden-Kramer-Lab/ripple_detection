"""Simulation tools for generating synthetic LFP data with embedded ripples.

``simulate_LFP`` gives one channel. ``simulate_multichannel_LFP`` gives channels
that share a ripple and part of their noise, ``simulate_sharp_wave_ripple_pair``
the raw two-channel input of the Long detector, ``simulate_multiunit`` spike
trains that burst with the ripples, and ``simulate_session`` all of them at once
with the ground truth, for testing detectors against known events.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import special

from ripple_detection._call_hints import explain_call_errors
from ripple_detection.core import (
    FloatArray,
    IntArray,
    StrArray,
    _generator,
    filter_ripple_band,
)

RIPPLE_FREQUENCY = 200
NoiseType = Literal["white", "pink", "brown"]

EVENT_TYPES = ("swr", "weak_ripple", "burst_only", "ripple_doublet", "sharp_wave_only")
"""The kinds of latent network event ``draw_network_events`` draws."""

NON_EVENT_TYPES = ("spike_leakage", "emg", "fast_gamma", "theta_burst")
"""The kinds of activity a detector should not report."""

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


def _gaussian_window(
    time: FloatArray, center: float, sigma: float, n_sigma: float
) -> tuple[slice, FloatArray]:
    """The samples within ``n_sigma`` of ``center`` and the unit-peak Gaussian there;
    at least one sample, so a bump narrower than a step still lands somewhere."""
    first, last = np.searchsorted(time, [center - n_sigma * sigma, center + n_sigma * sigma])
    if last <= first:
        last = min(first + 1, time.size)
        first = last - 1
    window = slice(int(first), int(last))
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
    ``band_noise_sd``; the filter's gain depends on frequency and duration.

    ``band`` is passed to ``filter_ripple_band``; None filters the ripple band
    (the shipped kernel at 1500 Hz). A second of zeros each side makes the run
    long enough for the kernel at any rate, and is what the burst is
    surrounded by in the record.
    """
    n_pad = int(np.ceil(rate))
    padded = np.zeros(burst.size + 2 * n_pad)
    padded[n_pad : n_pad + burst.size] = burst
    filtered_peak = np.abs(
        filter_ripple_band(padded, sampling_frequency=rate, band=band)
    ).max()
    return float(snr * band_noise_sd / filtered_peak)


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


def _table(columns: dict[str, type | str], values: dict[str, ArrayLike]) -> pd.DataFrame:
    """A frame with exactly ``columns``, in order, cast to their dtypes, so an
    empty table and a filled one agree under every pandas version (``str``
    is ``object`` before pandas 3 and the string dtype from it)."""
    frame = pd.DataFrame({name: np.asarray(values[name]) for name in columns})
    return frame.astype(columns)


def _empty_events() -> pd.DataFrame:
    """The latent event table with no rows."""
    return _table(_EVENT_COLUMNS, {name: [] for name in _EVENT_COLUMNS})


def _empty_non_events() -> pd.DataFrame:
    """The non-event table with no rows."""
    return _table(_NON_EVENT_COLUMNS, {name: [] for name in _NON_EVENT_COLUMNS})


def _empty_ripple_channels() -> pd.DataFrame:
    """The per-channel ripple table with no rows."""
    return _table(_RIPPLE_CHANNEL_COLUMNS, {name: [] for name in _RIPPLE_CHANNEL_COLUMNS})


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
        One row per rendered non-event, activity a detector should not
        report; empty with its columns when there is none.
    unit_types : ndarray of str, shape (n_units,)
        Each unit's type, one of ``UNIT_TYPES``. Empty when the simulator did
        not assign types (``simulate_session``).
    baseline_rates : ndarray, shape (n_units,)
        Each unit's drawn baseline intensity in spikes/s, before event
        modulation; realized rates can be lower under refractory spiking.
        Empty when the simulator did not record them (``simulate_session``).
    running_intervals : ndarray, shape (n_bouts, 2)
        The running bouts, start and end in seconds; ``(0, 2)`` when the
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
    events: pd.DataFrame = field(default_factory=_empty_events)
    non_events: pd.DataFrame = field(default_factory=_empty_non_events)
    unit_types: StrArray = field(default_factory=lambda: np.empty(0, dtype="<U11"))
    baseline_rates: FloatArray = field(default_factory=lambda: np.empty(0))
    running_intervals: FloatArray = field(default_factory=lambda: np.empty((0, 2)))
    ripple_channels: pd.DataFrame = field(default_factory=_empty_ripple_channels)

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
        n_units = self.multiunit.shape[1]
        for name in ("unit_types", "baseline_rates"):
            length = len(getattr(self, name))
            if length not in (0, n_units):
                msg = f"{name} has {length} entries; multiunit has {n_units} units."
                raise ValueError(msg)

    @property
    def ripple_windows(self) -> FloatArray:
        """Start and end of each ripple, shape (n_ripples, 2): the centre plus
        or minus half the duration, where the envelope is at 1 percent of its
        peak, clipped to the recording.

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
        running_intervals=(
            np.empty((0, 2))
            if running_intervals is None
            else _running_intervals(running_intervals)
        ),
    )


_REFERENCE_TYPE_PROBABILITIES = {
    "swr": 0.55,
    "weak_ripple": 0.15,
    "burst_only": 0.10,
    "ripple_doublet": 0.10,
    "sharp_wave_only": 0.10,
}
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
        msg = f"{name} must be a finite (low, high) range with low <= high in {bounds}, got {value}."
        raise ValueError(msg)
    return low, high


def _check_scalar(
    name: str, value: float, *, lower: float = 0.0, lower_strict: bool = False
) -> float:
    """``value`` as a finite float at or above (or above) ``lower``."""
    number = float(value)
    if not (np.isfinite(number) and (number > lower if lower_strict else number >= lower)):
        relation = ">" if lower_strict else ">="
        msg = f"{name} must be finite and {relation} {lower:g}, got {value}."
        raise ValueError(msg)
    return number


def _rest_intervals(time: FloatArray, running_intervals: ArrayLike | None) -> FloatArray:
    """(n, 2) stretches of rest: the recording less its first and last second
    and the running bouts."""
    start, end = float(time[0]) + 1.0, float(time[-1]) - 1.0
    bouts = (
        np.empty((0, 2))
        if running_intervals is None
        else _running_intervals(running_intervals)
    )
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
    total = float(lengths.sum())
    n = int(rng.poisson(rate * total))
    positions = np.sort(rng.uniform(0.0, total, size=n))
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    index = np.clip(np.searchsorted(cumulative, positions, side="right") - 1, 0, None)
    return intervals[index, 0] + positions - cumulative[index], index


@explain_call_errors
def draw_network_events(
    time: ArrayLike,
    *,
    event_rate: float = 0.5,
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
        Sample timestamps in seconds, increasing.
    event_rate : float, optional
        Events per second of rest, before events too close to the previous
        one are dropped. Default 0.5.
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
        Default (0.03, 0.15).
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
        ripple's skew. Default (1.0, 1.5).
    burst_lag : float, optional
        Standard deviation in seconds of a burst's centre about its ripple's.
        Default 0.01.
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
        and steeper at the edges, with the same half-maximum width. Default 2.
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
        If ``time`` is not 1-D with two or more samples, a rate, lag, gain or
        separation is negative or not finite, ``type_probabilities`` names an
        unknown type or has no positive weight, a range is not a finite
        ``(low, high)`` tuple with ``low <= high`` inside its bounds (positive
        durations, SNRs and intervals, skew in (0, 1), participation in
        [0, 1], frequencies and the frequency after the chirp between 0 and
        Nyquist), ``strength_correlation`` lies outside [0, 1] or
        ``envelope_power`` is not 2 or 4.

    Notes
    -----
    Draw order: the event count (Poisson, ``event_rate`` times the rest
    time), their positions on the concatenated rest time, their types; then
    one row per event, in time order, of 15 standard normals and then one of
    18 uniforms. Every event draws the same block whatever its type or the
    parameters, and an event too close to the previous kept event, or whose
    span at four side scales leaves its stretch of rest, is dropped, not
    redrawn. So changing one parameter changes only the values it governs:
    the other events and columns stay where they were.

    The normals are the shared strength ``z``; per ripple slot (up to three)
    the residuals for its SNR and onset frequency; per sharp-wave slot its
    lag and the residual for its amplitude; the burst's lag and the residual
    for its participation. A coupled value is ``low + (high - low) *
    ndtr(sqrt(rho) z + sqrt(1 - rho) residual)``. The uniforms are the
    doublet's ripple count (3 with probability 0.3, else 2); per ripple slot
    its span, skew, chirp and the interval from the previous ripple; per
    sharp-wave slot its span; the burst's duration ratio and its
    ``burst_only`` span.

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
    time = np.asarray(time, dtype=float)
    if time.ndim != 1 or time.size < 2:
        msg = f"time must be 1-D with at least two samples, got shape {time.shape}."
        raise ValueError(msg)
    nyquist = 0.5 / float(np.median(np.diff(time)))
    event_rate = _check_scalar("event_rate", event_rate)
    probabilities = _type_probabilities(type_probabilities)
    ripple_duration = _check_range(
        "ripple_duration", ripple_duration, lower=0, lower_strict=True
    )
    ripple_skew = _check_range(
        "ripple_skew", ripple_skew, lower=0, lower_strict=True, upper=1, upper_strict=True
    )
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
    rho = _check_scalar("strength_correlation", strength_correlation)
    if rho > 1:
        msg = f"strength_correlation must lie in [0, 1], got {strength_correlation}."
        raise ValueError(msg)
    if envelope_power not in (2, 4):
        msg = f"envelope_power must be 2 or 4, got {envelope_power!r}."
        raise ValueError(msg)
    rng = _generator(rng)

    rest = _rest_intervals(time, running_intervals)
    event_times, rest_index = _poisson_times(rest, event_rate, rng)
    n_events = event_times.size
    types = rng.choice(len(EVENT_TYPES), size=n_events, p=probabilities)
    normals = rng.standard_normal((n_events, _N_NORMALS))
    uniforms = rng.random((n_events, _N_UNIFORMS))

    rows: list[tuple[int, str, str, int, float, float, float, float, float, float, float]] = []
    last_end = -np.inf
    for event in range(n_events):
        event_type = EVENT_TYPES[types[event]]
        z, u = normals[event], uniforms[event]

        def strength(
            residual: float, bounds: tuple[float, float], shared: float = z[0]
        ) -> float:
            return _coupled_uniform(shared, residual, rho, bounds)

        weak = event_type == "weak_ripple"
        components = []  # (expression, component, center, rise, decay, amplitude, f0, f1, p)
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
                onset = strength(z[2 + 2 * j], ripple_frequency)
                snr = strength(z[1 + 2 * j], weak_ripple_snr if weak else ripple_snr)
                rise, decay = span * (1 - skew) / 3, span * skew / 3
                ripples.append((center, rise, decay, span, skew))
                components.append(
                    ("ripple", j, center, rise, decay, snr, onset,
                     onset - _scaled(u[3 + 4 * j], ripple_chirp), np.nan)
                )  # fmt: skip
                sharp_wave_sigma = _scaled(u[13 + j], sharp_wave_duration) / 6
                amplitude = strength(z[8 + 2 * j], sharp_wave_amplitude) * (
                    0.5 if weak else 1.0
                )
                components.append(
                    ("sharp_wave", j, center + sharp_wave_lag * z[7 + 2 * j],
                     sharp_wave_sigma, sharp_wave_sigma, amplitude, np.nan, np.nan, np.nan)
                )  # fmt: skip
            if event_type == "ripple_doublet":
                first, last = ripples[0], ripples[-1]
                start, end = first[0] - 3 * first[1], last[0] + 3 * last[2]
                burst_center, burst_rise = (start + end) / 2, (end - start) / 6
                burst_decay = burst_rise
            else:
                ripple_center, _, _, span, skew = ripples[0]
                burst_center = ripple_center + burst_lag * z[13]
                burst_span = span * _scaled(u[16], burst_duration_ratio)
                burst_rise, burst_decay = burst_span * (1 - skew) / 3, burst_span * skew / 3
            components.append(
                ("burst", 0, burst_center, burst_rise, burst_decay, burst_gain, np.nan, np.nan,
                 strength(z[14], weak_participation if weak else participation))
            )  # fmt: skip
        elif event_type == "burst_only":
            sigma = _scaled(u[17], burst_only_duration) / 6
            components.append(
                ("burst", 0, event_times[event], sigma, sigma, burst_gain, np.nan, np.nan,
                 strength(z[14], participation))
            )  # fmt: skip
        else:  # sharp_wave_only
            sigma = _scaled(u[13], sharp_wave_duration) / 6
            components.append(
                ("sharp_wave", 0, event_times[event], sigma, sigma,
                 strength(z[8], sharp_wave_amplitude), np.nan, np.nan, np.nan)
            )  # fmt: skip

        centers = np.array([c[2] for c in components])
        rises = np.array([c[3] for c in components])
        decays = np.array([c[4] for c in components])
        start_3, end_3 = np.min(centers - 3 * rises), np.max(centers + 3 * decays)
        start_4, end_4 = np.min(centers - 4 * rises), np.max(centers + 4 * decays)
        rest_start, rest_end = rest[rest_index[event]]
        if start_3 < last_end + minimum_separation:
            continue
        if start_4 < rest_start or end_4 > rest_end:
            continue
        last_end = end_3
        rows.extend((event, event_type, *component) for component in components)

    if not rows:
        return _empty_events()
    columns = list(zip(*rows, strict=True))
    table = pd.DataFrame(
        {
            "event": np.asarray(columns[0]),
            "event_type": np.asarray(columns[1], dtype=object),
            "expression": np.asarray(columns[2], dtype=object),
            "component": np.asarray(columns[3]),
            "center_time": np.asarray(columns[4], dtype=float),
            "rise_sigma": np.asarray(columns[5], dtype=float),
            "decay_sigma": np.asarray(columns[6], dtype=float),
            "amplitude": np.asarray(columns[7], dtype=float),
            "frequency_start": np.asarray(columns[8], dtype=float),
            "frequency_end": np.asarray(columns[9], dtype=float),
            "participation": np.asarray(columns[10], dtype=float),
        }
    )
    # renumber the kept events in order of their earliest component's centre
    earliest = table.groupby("event", sort=True)["center_time"].min()
    rank = np.argsort(np.argsort(earliest.to_numpy(), kind="stable"), kind="stable")
    table["event_id"] = table["event"].map(pd.Series(rank, index=earliest.index))
    table["envelope_power"] = envelope_power
    table["n_participants"] = 0
    return _sorted_events(
        _table(_EVENT_COLUMNS, {name: table[name] for name in _EVENT_COLUMNS})
    )


def _scaled(u: float, bounds: tuple[float, float]) -> float:
    """``u`` in [0, 1] mapped linearly onto ``(low, high)``."""
    return bounds[0] + (bounds[1] - bounds[0]) * float(u)


def _coupled_uniform(
    shared: float, residual: float, rho: float, bounds: tuple[float, float]
) -> float:
    """A uniform draw on ``bounds`` whose latent normal has correlation ``rho``
    with every other draw sharing ``shared``: ``ndtr(sqrt(rho) shared +
    sqrt(1 - rho) residual)``, uniform for any ``rho``."""
    latent = np.sqrt(rho) * shared + np.sqrt(1.0 - rho) * residual
    return _scaled(float(special.ndtr(latent)), bounds)


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

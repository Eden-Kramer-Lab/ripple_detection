"""Simulation tools for generating synthetic LFP data with embedded ripples.

``simulate_LFP`` gives one channel. ``simulate_multichannel_LFP`` gives channels
that share a ripple and part of their noise, ``simulate_sharp_wave_ripple_pair``
the raw two-channel input of the Long detector, ``simulate_multiunit`` spike
trains that burst with the ripples, and ``simulate_session`` all of them at once
with the ground truth, for testing detectors against known events.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ripple_detection._call_hints import explain_call_errors
from ripple_detection.core import FloatArray, filter_ripple_band

RIPPLE_FREQUENCY = 200
NoiseType = Literal["white", "pink", "brown"]


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


def _generator(seed: int | np.random.Generator | None) -> np.random.Generator:
    """``numpy.random.default_rng(seed)``, refusing the legacy ``RandomState``
    that 1.x's noise functions took: ``default_rng`` accepts one and silently
    draws a different stream from it."""
    given: object = seed  # a caller without a type checker can pass anything
    if isinstance(given, np.random.RandomState):
        msg = (
            "Pass a seed or a numpy.random.Generator, not a RandomState: since 2.0 the "
            "simulators draw through numpy.random.default_rng, so a RandomState would "
            "give a different stream than it did in 1.x."
        )
        raise TypeError(msg)
    return np.random.default_rng(seed)


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
    random_state: int | np.random.Generator | None = None,
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
    random_state : int or numpy.random.Generator, optional
        Seed, or a Generator to draw from, as ``numpy.random.default_rng``
        takes it. The noise is drawn first, then per-ripple
        frequencies, then per-ripple durations; a scalar consumes no
        randomness. So a given seed produces the same noise whatever the
        ripple parameters, but giving a frequency range changes the duration
        draws. Default is None, which draws from the operating system and is
        not reproducible. `np.random.seed` does not control this function;
        pass `random_state` to repeat a simulation.
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
    >>> lfp = simulate_LFP(time, [1.0, 2.0])

    Ripples five times the ripple-band background, varying in frequency and
    duration, on a pink-noise background:

    >>> time = simulate_time(15000, 1500)
    >>> lfp = simulate_LFP(
    ...     time, [2.0, 5.0, 8.0], noise_type='pink', ripple_snr=5,
    ...     ripple_frequency=(150, 250), ripple_duration=(0.04, 0.12), random_state=0,
    ... )

    """
    _validate_sizes(ripple_amplitude, ripple_snr, noise_amplitude)
    rng = _generator(random_state)
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
        burst = np.sin(2 * np.pi * time[window] * frequency) * carrier  # unit peak
        if ripple_snr is not None:
            # scale so that this burst's peak *after the filter* is ripple_snr
            # background SDs; the filter's gain depends on frequency and duration.
            # A second of zeros each side makes the run long enough for the kernel
            # at any rate, and is what the burst is surrounded by in the record.
            n_pad = int(np.ceil(rate))
            padded = np.zeros(burst.size + 2 * n_pad)
            padded[n_pad : n_pad + burst.size] = burst
            filtered_peak = np.abs(filter_ripple_band(padded, sampling_frequency=rate)).max()
            scale = ripple_snr * band_noise_sd / filtered_peak
        else:
            scale = amplitude / 2
        out[window] += scale * burst


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
    random_state: int | np.random.Generator | None = None,
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
    random_state : int or numpy.random.Generator, optional
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
    ...     ripple_frequency=(150, 250), ripple_duration=(0.04, 0.12), random_state=0,
    ... )
    >>> lfps.shape
    (15000, 4)

    """
    _validate_sizes(ripple_amplitude, ripple_snr, noise_amplitude)
    if n_channels < 1:
        msg = f"n_channels must be at least 1, got {n_channels}."
        raise ValueError(msg)
    gains = _channel_gains(channel_gains, n_channels)
    rng = _generator(random_state)
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
    random_state: int | np.random.Generator | None = None,
    sampling_frequency: float | None = None,
) -> FloatArray:
    """Simulate the raw two-channel input of ``Long_sharp_wave_ripple_detector``.

    Column 0 is a pyramidal-layer channel: the ripple at full amplitude on a
    small positive sharp-wave deflection. Column 1 is a stratum radiatum
    channel: the sharp wave as a negative Gaussian deflection centred on each
    ripple, with a weak copy of the ripple. The two channels' noise is
    correlated as in ``simulate_multichannel_LFP``.

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
    ripple_frequency, noise_type, noise_amplitude, random_state,
    sampling_frequency : optional
        As in ``simulate_multichannel_LFP``.

    Returns
    -------
    raw_lfps : ndarray, shape (n_time, 2)
        Raw, unfiltered: the ripple channel, then the sharp-wave channel, the
        order the Long detector takes.

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
        random_state=random_state,
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
    return pair


def simulate_multiunit(
    time: FloatArray,
    ripple_times: float | Sequence[float] | FloatArray,
    n_units: int,
    *,
    baseline_rate: float | tuple[float, float] | FloatArray = (0.5, 5.0),
    ripple_rate_gain: float = 8.0,
    participation: float = 0.6,
    ripple_duration: float | tuple[float, float] | FloatArray = 0.100,
    random_state: int | np.random.Generator | None = None,
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
    random_state : int or numpy.random.Generator, optional
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
    rng = _generator(random_state)
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


@dataclass(frozen=True, eq=False)
class SimulatedSession:
    """Every signal ``simulate_session`` produced, with the ground truth.

    Attributes
    ----------
    time : ndarray, shape (n_time,)
    lfps : ndarray, shape (n_time, n_channels)
        Raw multichannel LFP, the ripple channel first; filter it with
        ``filter_ripple_band`` before the ripple-band detectors.
    raw_lfp_pair : ndarray, shape (n_time, 2)
        The ripple channel and the stratum radiatum channel, unfiltered, for
        ``Long_sharp_wave_ripple_detector``.
    multiunit : ndarray, shape (n_time, n_units)
        Spike counts per sample.
    speed : ndarray, shape (n_time,)
        Zeros: an immobile animal.
    ripple_times, ripple_durations, ripple_frequencies : ndarray, shape (n_ripples,)
        Centre, duration (six standard deviations of the envelope) and
        frequency of each ripple, in the order the ripples were given.
    artifact_times : ndarray, shape (n_artifacts,)
    sampling_frequency : float

    Raises
    ------
    ValueError
        If the signals do not share ``time``'s length or the per-ripple
        arrays do not share one length.

    """

    time: FloatArray
    lfps: FloatArray
    raw_lfp_pair: FloatArray
    multiunit: FloatArray
    speed: FloatArray
    ripple_times: FloatArray
    ripple_durations: FloatArray
    ripple_frequencies: FloatArray
    artifact_times: FloatArray
    sampling_frequency: float

    def __post_init__(self) -> None:
        n_time = self.time.shape[0]
        for name in ("lfps", "raw_lfp_pair", "multiunit", "speed"):
            if getattr(self, name).shape[0] != n_time:
                msg = f"{name} has {getattr(self, name).shape[0]} samples; time has {n_time}."
                raise ValueError(msg)
        n_ripples = self.ripple_times.shape
        if not self.ripple_durations.shape == self.ripple_frequencies.shape == n_ripples:
            msg = "ripple_times, ripple_durations and ripple_frequencies differ in length."
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
    random_state: int | np.random.Generator | None = None,
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
    random_state : int or numpy.random.Generator, optional
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
    >>> session = simulate_session(time, [3.0, 9.0, 15.0], random_state=0)
    >>> session.lfps.shape, session.raw_lfp_pair.shape, session.multiunit.shape
    ((30000, 4), (30000, 2), (30000, 50))

    """
    if ripple_amplitude is not None:
        ripple_snr = None
    time = np.asarray(time, dtype=float)
    rng = _generator(random_state)
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
        random_state=rng,
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
    multiunit = simulate_multiunit(
        time,
        centers,
        n_units,
        baseline_rate=baseline_rate,
        ripple_rate_gain=ripple_rate_gain,
        participation=participation,
        ripple_duration=durations,
        random_state=rng,
    )
    return SimulatedSession(
        time=time,
        lfps=lfps,
        raw_lfp_pair=np.column_stack([lfps[:, 0], radiatum]),
        multiunit=multiunit,
        speed=np.zeros(time.size),
        ripple_times=centers,
        ripple_durations=durations,
        ripple_frequencies=frequencies,
        artifact_times=np.asarray(
            [] if artifact_times is None else artifact_times, dtype=float
        ),
        sampling_frequency=_sampling_rate(time, sampling_frequency),
    )

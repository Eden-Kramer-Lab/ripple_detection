"""Simulation tools for generating synthetic LFP data with embedded ripples."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

RIPPLE_FREQUENCY = 200


def simulate_time(n_samples: int, sampling_frequency: float) -> NDArray:
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

    """
    return np.arange(n_samples) / sampling_frequency


def mean_squared(x: NDArray) -> float:
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
    return (np.abs(x) ** 2.0).mean()


def normalize(y: NDArray, x: NDArray | None = None) -> NDArray:
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
    x = mean_squared(x) if x is not None else 1.0
    return y * np.sqrt(x / mean_squared(y))


def pink(N: int, state: np.random.RandomState | None = None) -> NDArray:
    """Generate pink (1/f) noise.

    Pink noise has equal power in proportionally-wide frequency bands (octaves).
    Power spectral density decreases at 3 dB per octave (1/f spectrum).

    Parameters
    ----------
    N : int
        Number of samples to generate.
    state : np.random.RandomState, optional
        Random number generator state for reproducibility. If None, uses a
        new RandomState. Default is None.

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
    state = np.random.RandomState() if state is None else state
    uneven = N % 2
    X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)) + 1.0)  # +1 to avoid divide by zero
    y = (np.fft.irfft(X / S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)


def white(N: int, state: np.random.RandomState | None = None) -> NDArray:
    """Generate white noise.

    White noise has constant power spectral density across all frequencies (flat
    spectrum). Power increases by 3 dB per octave when integrated over octave bands.

    Parameters
    ----------
    N : int
        Number of samples to generate.
    state : np.random.RandomState, optional
        Random number generator state for reproducibility. If None, uses a
        new RandomState. Default is None.

    Returns
    -------
    white_noise : ndarray, shape (N,)
        White noise signal from standard normal distribution.

    """
    state = np.random.RandomState() if state is None else state
    return state.randn(N)


def brown(N: int, state: np.random.RandomState | None = None) -> NDArray:
    """Generate brown (Brownian, red) noise.

    Brown noise has power spectral density that decreases at 6 dB per octave
    (1/f² spectrum). Power decreases at 3 dB per octave when integrated over
    octave bands.

    Parameters
    ----------
    N : int
        Number of samples to generate.
    state : np.random.RandomState, optional
        Random number generator state for reproducibility. If None, uses a
        new RandomState. Default is None.

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
    state = np.random.RandomState() if state is None else state
    uneven = N % 2
    X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
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
    value: float | Sequence[float] | NDArray, n_ripples: int, state: np.random.RandomState
) -> NDArray:
    """A scalar repeated per ripple, or one uniform draw per ripple from a range.

    A scalar consumes no randomness; a two-element ``(low, high)`` sequence
    draws ``n_ripples`` values from ``state``.
    """
    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        return np.full(n_ripples, float(values))
    if values.shape != (2,):
        raise ValueError(f"A range must have exactly two elements (low, high), got {value}.")
    low, high = values
    if not low <= high:
        raise ValueError(f"Range must be (low, high) with low <= high, got {value}.")
    return state.uniform(low, high, size=n_ripples)


def simulate_LFP(
    time: NDArray,
    ripple_times: float | list[float],
    ripple_amplitude: float | None = None,
    ripple_duration: float | tuple[float, float] = 0.100,
    noise_type: Literal["white", "pink", "brown"] = "brown",
    noise_amplitude: float = 1.3,
    random_state: int | np.random.RandomState | None = None,
    *,
    ripple_snr: float | None = None,
    ripple_frequency: float | tuple[float, float] = RIPPLE_FREQUENCY,
    sampling_frequency: float | None = None,
) -> NDArray:
    """Simulate local field potential with embedded ripple oscillations.

    Generates a synthetic LFP signal containing ripple events (sinusoids at
    ``ripple_frequency``) embedded in colored noise. Ripples are amplitude-
    modulated by a Gaussian envelope.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
        Time array in seconds.
    ripple_times : float or list of float
        Center time(s) of ripple event(s) in seconds.
    ripple_amplitude : float, optional
        Peak-to-peak amplitude of the ripple oscillation in the signal's units
        (the peak is half this). Default is 2 when ``ripple_snr`` is not
        given. Cannot be combined with ``ripple_snr``.
    ripple_duration : float or (float, float), optional
        Approximate duration in **seconds** of a ripple event, defined as 6
        standard deviations of its Gaussian envelope. A ``(low, high)`` pair
        draws one duration per ripple uniformly from that range. Default is
        0.100 (100 ms).
    noise_type : {'white', 'pink', 'brown'}, optional
        Type of background noise. Default is 'brown'. Brown (1/f²) noise
        leaves very little power in the 150-250 Hz band, and the fraction
        falls further as the record lengthens (it is a random walk), so
        ripples of any visible size dominate the band; 'pink' gives a
        ripple-band background closer to recordings. See Notes.
    noise_amplitude : float, optional
        Amplitude of background noise in the signal's units. Default is 1.3.
    random_state : int or np.random.RandomState, optional
        Seed or random state. The noise is drawn first, then per-ripple
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
        (three times its kernel length). Cannot be combined with
        ``ripple_amplitude``. Default is None.
    ripple_frequency : float or (float, float), optional
        Ripple oscillation frequency in Hz, or a ``(low, high)`` range drawn
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
        not a two-element ordered sequence, if a ripple time lies outside
        ``time``, if a duration is not positive, or if a frequency is not
        between zero and the Nyquist frequency. Each of these would otherwise
        give an all-NaN, empty, or aliased ripple with no error.

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
    >>> lfp = simulate_LFP(time, [1.0, 2.0], noise_type='brown')

    Ripples five times the ripple-band background, varying in frequency and
    duration, on a pink-noise background:

    >>> time = simulate_time(15000, 1500)
    >>> lfp = simulate_LFP(
    ...     time, [2.0, 5.0, 8.0], noise_type='pink', ripple_snr=5,
    ...     ripple_frequency=(150, 250), ripple_duration=(0.04, 0.12), random_state=0,
    ... )

    """
    if ripple_amplitude is not None and ripple_snr is not None:
        raise ValueError("Give either ripple_amplitude or ripple_snr, not both.")
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
    noise = (noise_amplitude / 2) * NOISE_FUNCTION[noise_type](time.size, state=random_state)

    if isinstance(ripple_times, (int, float)):
        ripple_times = [ripple_times]
    n_ripples = len(ripple_times)

    if ripple_snr is not None:
        if noise_amplitude <= 0:
            raise ValueError("ripple_snr needs a background: noise_amplitude must be > 0.")
        from ripple_detection.core import filter_ripple_band

        if sampling_frequency is None:
            sampling_frequency = 1.0 / np.median(np.diff(time))
        band_noise_sd = filter_ripple_band(noise, sampling_frequency=sampling_frequency).std()
    elif ripple_amplitude is None:
        ripple_amplitude = 2.0

    frequencies = _draw_per_ripple(ripple_frequency, n_ripples, random_state)
    durations = _draw_per_ripple(ripple_duration, n_ripples, random_state)
    if n_ripples:
        nyquist = 0.5 / np.median(np.diff(time))
        outside = [t for t in ripple_times if not time.min() <= t <= time.max()]
        if outside:
            raise ValueError(
                f"ripple_times {outside} lie outside time "
                f"[{time.min()}, {time.max()}]; the ripple would have no samples."
            )
        if np.any(durations <= 0):
            raise ValueError(f"ripple_duration must be positive, got {ripple_duration}.")
        if np.any(frequencies <= 0) or np.any(frequencies >= nyquist):
            raise ValueError(
                f"ripple_frequency must lie in (0, {nyquist:.1f}) Hz, the Nyquist range of "
                f"time's sampling rate, got {ripple_frequency}."
            )

    signal = []
    for ripple_time, frequency, duration in zip(
        ripple_times, frequencies, durations, strict=True
    ):
        carrier = norm(loc=ripple_time, scale=duration / 6).pdf(time)
        carrier /= carrier.max()
        burst = np.sin(2 * np.pi * time * frequency) * carrier  # unit peak
        if ripple_snr is not None:
            # scale so that this burst's peak *after the filter* is ripple_snr
            # background SDs; the filter's gain depends on frequency and duration
            filtered_peak = np.abs(
                filter_ripple_band(burst, sampling_frequency=sampling_frequency)
            ).max()
            signal.append(ripple_snr * band_noise_sd / filtered_peak * burst)
        else:
            signal.append((ripple_amplitude / 2) * burst)

    return np.sum(signal, axis=0) + noise

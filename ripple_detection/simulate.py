"""Simulation tools for generating synthetic LFP data with embedded ripples."""

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


def simulate_LFP(
    time: NDArray,
    ripple_times: float | list[float],
    ripple_amplitude: float = 2,
    ripple_duration: float = 0.100,
    noise_type: Literal["white", "pink", "brown"] = "brown",
    noise_amplitude: float = 1.3,
) -> NDArray:
    """Simulate local field potential with embedded ripple oscillations.

    Generates a synthetic LFP signal containing ripple events (200 Hz sinusoids)
    embedded in colored noise. Ripples are amplitude-modulated by a Gaussian
    envelope.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
        Time array in seconds.
    ripple_times : float or list of float
        Center time(s) of ripple event(s) in seconds.
    ripple_amplitude : float, optional
        Peak amplitude of ripple oscillation in **arbitrary units**. Default is 2.
        For realistic LFPs, scale to match your recording system (typically µV or mV).
        The ratio to noise_amplitude matters more than absolute values.
    ripple_duration : float, optional
        Approximate duration in **seconds** of ripple event, defined as 6 standard
        deviations of the Gaussian envelope. Default is 0.100 (100 ms).
    noise_type : {'white', 'pink', 'brown'}, optional
        Type of background noise. Default is 'brown' (most realistic for LFP).
    noise_amplitude : float, optional
        Amplitude of background noise in **arbitrary units**. Default is 1.3.
        A ratio of ripple_amplitude/noise_amplitude ≈ 1.5 provides realistic
        signal-to-noise ratio for ripple detection.

    Returns
    -------
    lfp : ndarray, shape (n_time,)
        Simulated LFP signal with embedded ripples.

    Notes
    -----
    Ripple frequency is fixed at 200 Hz (RIPPLE_FREQUENCY constant).
    The Gaussian envelope has sigma = ripple_duration / 6, so the ripple
    amplitude decays to ~1% at +/-3*sigma from the center.

    Examples
    --------
    >>> time = simulate_time(3000, 1000)  # 3 seconds at 1000 Hz
    >>> lfp = simulate_LFP(time, [1.0, 2.0], noise_type='brown')

    """
    noise = (noise_amplitude / 2) * NOISE_FUNCTION[noise_type](time.size)
    ripple_signal = np.sin(2 * np.pi * time * RIPPLE_FREQUENCY)
    signal = []

    if isinstance(ripple_times, (int, float)):
        ripple_times = [ripple_times]

    for ripple_time in ripple_times:
        carrier = norm(loc=ripple_time, scale=ripple_duration / 6).pdf(time)
        carrier /= carrier.max()
        signal.append((ripple_amplitude / 2) * (ripple_signal * carrier))

    return np.sum(signal, axis=0) + noise

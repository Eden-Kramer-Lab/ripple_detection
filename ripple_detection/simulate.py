import numpy as np
from scipy.stats import norm

RIPPLE_FREQUENCY = 200


def simulate_time(n_samples, sampling_frequency):
    return np.arange(n_samples) / sampling_frequency


def mean_squared(x):
    return (np.abs(x) ** 2.0).mean()


def normalize(y, x=None):
    """normalize power in y to a (standard normal) white noise signal.
    Optionally normalize to power in signal `x`.
    #The mean power of a Gaussian with :math:`\\mu=0` and :math:`\\sigma=1` is 1.
    https://github.com/python-acoustics/python-acoustics/tree/master/acoustics
    """
    x = mean_squared(x) if x is not None else 1.0
    return y * np.sqrt(x / mean_squared(y))


def pink(N, state=None):
    """
    Pink noise.

    :param N: Amount of samples.
    :param state: State of PRNG.
    :type state: :class:`np.random.RandomState`

    Pink noise has equal power in bands that are proportionally wide.
    Power density decreases with 3 dB per octave.
    https://github.com/python-acoustics/python-acoustics/tree/master/acoustics
    """
    state = np.random.RandomState() if state is None else state
    uneven = N % 2
    X = (state.randn(N // 2 + 1 + uneven) +
         1j * state.randn(N // 2 + 1 + uneven))
    S = np.sqrt(np.arange(len(X)) + 1.0)  # +1 to avoid divide by zero
    y = (np.fft.irfft(X / S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)


def white(N, state=None):
    """
    White noise.

    :param N: Amount of samples.
    :param state: State of PRNG.
    :type state: :class:`np.random.RandomState`

    White noise has a constant power density. It's narrowband spectrum is therefore flat.
    The power in white noise will increase by a factor of two for each octave band,
    and therefore increases with 3 dB per octave.
    """
    state = np.random.RandomState() if state is None else state
    return state.randn(N)


def brown(N, state=None):
    """
    Brown noise.

    :param N: Amount of samples.
    :param state: State of PRNG.
    :type state: :class:`np.random.RandomState`

    Power decreases with -3 dB per octave.
    Power density decreases with 6 dB per octave.
    """
    state = np.random.RandomState() if state is None else state
    uneven = N % 2
    X = (state.randn(N // 2 + 1 + uneven) + 1j *
         state.randn(N // 2 + 1 + uneven))
    S = np.arange(len(X)) + 1
    y = np.fft.irfft(X / S).real
    if uneven:
        y = y[:-1]
    return normalize(y)


NOISE_FUNCTION = {
    'white': white,
    'pink': pink,
    'brown': brown,
}


def simulate_LFP(time, ripple_times, ripple_amplitude=2,
                 ripple_duration=0.100, noise_type='brown', noise_amplitude=1.3):
    '''Simulate a LFP with a ripple at ripple times
    '''
    noise = (noise_amplitude / 2) * NOISE_FUNCTION[noise_type](time.size)
    ripple_signal = np.sin(2 * np.pi * time * RIPPLE_FREQUENCY)
    signal = []

    try:
        iter(ripple_times)
    except TypeError:
        ripple_times = [ripple_times]

    for ripple_time in ripple_times:
        carrier = norm(loc=ripple_time, scale=ripple_duration / 6).pdf(time)
        carrier /= carrier.max()
        signal.append((ripple_amplitude / 2) * (ripple_signal * carrier))

    return np.sum(signal, axis=0) + noise

"""Synthetic inputs for the detector tests.

Plain functions, not fixtures, because each test asks for different events,
channels and gains. All build at a given sampling rate on a Gaussian-noise
background; ``bursts`` and ``events`` are given in samples.
"""

import numpy as np


def _synthetic_ripple_band(n_time, sampling_frequency, bursts, n_channels=3, seed=0):
    """Ripple-band-like input: Gaussian noise plus 200 Hz bursts of amplitude
    ``gain`` times the noise SD on the given (start_sample, stop_sample, gain)
    intervals, identical on every channel."""
    rng = np.random.default_rng(seed)
    lfps = rng.normal(0.0, 1.0, (n_time, n_channels))
    t = np.arange(n_time) / sampling_frequency
    carrier = np.sin(2 * np.pi * 200.0 * t)
    for start, stop, gain in bursts:
        lfps[start:stop] += gain * carrier[start:stop, np.newaxis]
    return lfps


def _synthetic_two_channel_lfp(
    n_time, sampling_frequency, event_samples, seed=0, sharp_wave=True, ripple=True, gain=1.0
):
    """Raw two-channel LFP: column 0 pyramidal-layer (ripple) channel, column 1
    stratum radiatum (sharp-wave) channel. Each event plants a 200 Hz burst on
    channel 0 and a slow deflection, negative on channel 1 and positive on
    channel 0, both with a Gaussian envelope (sigma 15 ms)."""
    rng = np.random.default_rng(seed)
    lfp = rng.normal(0.0, 1.0, (n_time, 2))
    t = np.arange(n_time) / sampling_frequency
    for center in event_samples:
        envelope = np.exp(-0.5 * ((t - t[center]) / 0.015) ** 2)
        if ripple:
            lfp[:, 0] += gain * 5.0 * envelope * np.sin(2 * np.pi * 200.0 * t)
        if sharp_wave:
            lfp[:, 0] += gain * 3.0 * envelope
            lfp[:, 1] -= gain * 8.0 * envelope
    return lfp


def _synthetic_joint_inputs(
    n_time,
    sampling_frequency,
    events,
    n_units=8,
    seed=0,
    ripple=True,
    spikes=True,
    ripple_gain=20.0,
    rate_gain=8.0,
):
    """Ripple-band LFP (3 channels) plus a multiunit spike matrix. Each event
    adds a 200 Hz burst to the LFP and raises every unit's spike probability
    over a 60 ms window."""
    rng = np.random.default_rng(seed)
    lfps = rng.normal(0.0, 1.0, (n_time, 3))
    base_rate = 0.004  # spikes per sample per unit
    prob = np.full((n_time, n_units), base_rate)
    t = np.arange(n_time) / sampling_frequency
    for center in events:
        window = slice(center - 30, center + 30)
        if ripple:
            lfps[window] += ripple_gain * np.sin(2 * np.pi * 200.0 * t[window])[:, np.newaxis]
        if spikes:
            prob[window] = base_rate * rate_gain
    multiunit = (rng.random((n_time, n_units)) < prob).astype(float)
    return lfps, multiunit

"""Synthetic inputs for the detector and simulator tests.

Plain functions, not fixtures, because each test asks for different events,
channels and gains. The detector inputs build at a given sampling rate on a
Gaussian-noise background; ``bursts`` and ``events`` are given in samples.
The non-event tables are hand-built rows for ``simulate_network_session``.
"""

import numpy as np
import pandas as pd


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


NON_EVENT_COLUMNS = {
    "non_event_id": "int64",
    "non_event_type": "str",
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


def _one_non_event_table(non_event_type, *, non_event_id=0, center_time=5.0, **overrides):
    """One non-event built by hand, centred on ``center_time``: a leakage
    burst of 2 units, 5 spikes 4 ms apart, peak 2, on channel 1; an EMG burst
    of span 0.1 s and peak SD 1.5; a gamma burst of span 0.1 s at 80 Hz and
    SNR 3 in 60-100 Hz; a theta burst of span 0.2 s over 5 units at gain 10.
    ``overrides`` set columns."""
    nan = np.nan
    row = {
        "non_event_id": non_event_id, "non_event_type": non_event_type,
        "center_time": center_time, "envelope_power": 2, "frequency": nan,
        "snr_band_low": nan, "snr_band_high": nan, "channel": -1, "n_units": 0,
        "n_spikes": 0, "isi": nan,
    }  # fmt: skip
    row.update(
        {
            "spike_leakage": {
                "rise_sigma": 4 * 0.004 / 6, "amplitude": 2.0, "channel": 1, "n_units": 2,
                "n_spikes": 5, "isi": 0.004,
            },
            "emg": {"rise_sigma": 0.1 / 6, "amplitude": 1.5},
            "fast_gamma": {
                "rise_sigma": 0.1 / 6, "amplitude": 3.0, "frequency": 80.0,
                "snr_band_low": 60.0, "snr_band_high": 100.0,
            },
            "theta_burst": {"rise_sigma": 0.2 / 6, "amplitude": 10.0, "n_units": 5},
        }[non_event_type]
    )  # fmt: skip
    row["decay_sigma"] = row["rise_sigma"]
    row.update(overrides)
    return pd.DataFrame([row])[list(NON_EVENT_COLUMNS)]


def _non_event_tables(*tables):
    """Hand-built non-events as one table, numbered in order."""
    return pd.concat(
        [table.assign(non_event_id=i) for i, table in enumerate(tables)], ignore_index=True
    )

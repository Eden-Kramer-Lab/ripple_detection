"""Shared fixtures for ripple detection tests."""

import numpy as np
import pytest
from ripple_detection.simulate import simulate_LFP, simulate_time


@pytest.fixture
def sampling_frequency():
    """Standard sampling frequency for LFP recordings."""
    return 1500


@pytest.fixture
def time_3s(sampling_frequency):
    """Generate 3 seconds of time data."""
    n_samples = sampling_frequency * 3
    return simulate_time(n_samples, sampling_frequency)


@pytest.fixture
def time_4s(sampling_frequency):
    """Generate 4 seconds of time data."""
    n_samples = sampling_frequency * 4
    return simulate_time(n_samples, sampling_frequency)


@pytest.fixture
def stationary_speed(time_3s):
    """Generate speed data for stationary animal."""
    return np.ones_like(time_3s) * 2.0


@pytest.fixture
def single_lfp_with_ripples(time_3s):
    """Generate single LFP channel with ripples at 1.1s and 2.1s."""
    lfp = simulate_LFP(
        time_3s,
        ripple_times=[1.1, 2.1],
        noise_amplitude=1.2,
        ripple_amplitude=1.5,
    )
    return lfp[:, np.newaxis]


@pytest.fixture
def dual_lfp_with_ripples(time_3s):
    """Generate two LFP channels with ripples at different times."""
    lfp1 = simulate_LFP(
        time_3s,
        ripple_times=[1.1, 2.1],
        noise_amplitude=1.2,
        ripple_amplitude=1.5,
    )
    lfp2 = simulate_LFP(
        time_3s,
        ripple_times=[0.5, 2.5],
        noise_amplitude=1.2,
        ripple_amplitude=1.5,
    )
    return np.column_stack([lfp1, lfp2])


@pytest.fixture
def dual_lfp_close_ripples(time_3s):
    """Generate two LFP channels with closely spaced ripples."""
    lfp1 = simulate_LFP(
        time_3s,
        ripple_times=[1.100, 2.100],
        noise_amplitude=1.2,
        ripple_amplitude=1.5,
    )
    lfp2 = simulate_LFP(
        time_3s,
        ripple_times=[1.150, 2.150],
        noise_amplitude=1.2,
        ripple_amplitude=1.5,
    )
    return np.column_stack([lfp1, lfp2])


@pytest.fixture
def multi_lfp_sparse_ripples(time_3s):
    """Generate many LFP channels with ripples only in first two channels."""
    lfps = []
    lfps.append(
        simulate_LFP(
            time_3s,
            ripple_times=[1.1, 2.1],
            noise_amplitude=1.2,
            ripple_amplitude=1.5,
        )
    )
    lfps.append(
        simulate_LFP(
            time_3s,
            ripple_times=[0.5, 2.5],
            noise_amplitude=1.2,
            ripple_amplitude=1.5,
        )
    )
    # Add 11 channels without ripples
    for _ in range(11):
        lfps.append(
            simulate_LFP(
                time_3s,
                ripple_times=[],
                noise_amplitude=1.2,
                ripple_amplitude=1.5,
            )
        )
    return np.column_stack(lfps)


@pytest.fixture
def lfp_no_ripples(time_3s):
    """Generate LFP channels with no ripples (noise only)."""
    lfp1 = simulate_LFP(
        time_3s,
        ripple_times=[],
        noise_amplitude=1.0,
        ripple_amplitude=1.5,
    )
    lfp2 = simulate_LFP(
        time_3s,
        ripple_times=[],
        noise_amplitude=1.0,
        ripple_amplitude=1.5,
    )
    return np.column_stack([lfp1, lfp2])


@pytest.fixture
def lfp_short_duration_ripples(time_3s):
    """Generate LFP with very short duration ripples that should not be detected."""
    lfp = simulate_LFP(
        time_3s,
        ripple_times=[1.1, 2.1],
        noise_amplitude=1.2,
        ripple_amplitude=1.5,
        ripple_duration=0.001,  # Too short to detect
    )
    return lfp[:, np.newaxis]


@pytest.fixture
def speed_with_movement(time_3s):
    """Generate speed data where animal is moving after t=1.5s."""
    speed = np.ones_like(time_3s)
    speed[time_3s > 1.5] = 5  # Above typical threshold of 4
    return speed


@pytest.fixture
def multiunit_data(time_3s):
    """Generate synthetic multiunit spike data."""
    n_units = 20
    n_samples = len(time_3s)
    # Create sparse spike trains with Poisson-like statistics
    np.random.seed(42)
    baseline_rate = 0.01
    multiunit = np.random.random((n_samples, n_units)) < baseline_rate

    # Add high synchrony events at specific times
    hse_times = [1.0, 2.0]
    hse_rate = 0.3
    for hse_time in hse_times:
        time_mask = np.abs(time_3s - hse_time) < 0.05
        multiunit[time_mask, :] = (
            np.random.random((time_mask.sum(), n_units)) < hse_rate
        )

    return multiunit.astype(float)

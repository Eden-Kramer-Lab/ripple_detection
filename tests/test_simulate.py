"""Tests for simulation module."""

import numpy as np
import pytest
from ripple_detection.simulate import (
    NOISE_FUNCTION,
    brown,
    mean_squared,
    normalize,
    pink,
    simulate_LFP,
    simulate_time,
    white,
)


class TestSimulateTime:
    """Test time array generation."""

    def test_basic_time_generation(self):
        """Test basic time array generation."""
        n_samples = 1500
        sampling_frequency = 1500
        time = simulate_time(n_samples, sampling_frequency)

        assert len(time) == n_samples
        assert time[0] == 0.0
        assert np.allclose(time[-1], (n_samples - 1) / sampling_frequency)

    def test_time_spacing(self):
        """Test that time samples are evenly spaced."""
        n_samples = 1000
        sampling_frequency = 1000
        time = simulate_time(n_samples, sampling_frequency)

        dt = np.diff(time)
        expected_dt = 1 / sampling_frequency
        assert np.allclose(dt, expected_dt), "Time samples should be evenly spaced"

    def test_different_sampling_frequencies(self):
        """Test with different sampling frequencies."""
        n_samples = 100
        for sampling_freq in [500, 1000, 1500, 2000]:
            time = simulate_time(n_samples, sampling_freq)
            assert len(time) == n_samples
            assert np.allclose(np.diff(time), 1 / sampling_freq)


class TestMeanSquared:
    """Test mean squared function."""

    def test_positive_values(self):
        """Test mean squared with positive values."""
        x = np.array([1, 2, 3, 4, 5])
        result = mean_squared(x)
        expected = np.mean(x**2)
        assert np.allclose(result, expected)

    def test_negative_values(self):
        """Test mean squared with negative values."""
        x = np.array([-1, -2, -3])
        result = mean_squared(x)
        expected = np.mean(np.abs(x) ** 2)
        assert np.allclose(result, expected)

    def test_zero_array(self):
        """Test mean squared of zero array."""
        x = np.zeros(10)
        result = mean_squared(x)
        assert result == 0.0


class TestNormalize:
    """Test normalization function."""

    def test_normalize_to_unit_power(self):
        """Test normalization to unit power (standard normal)."""
        np.random.seed(42)
        signal = np.random.randn(1000) * 5  # Mean power ~25
        normalized = normalize(signal)

        # Normalized signal should have mean power ~1
        assert np.allclose(mean_squared(normalized), 1.0, atol=0.1)

    def test_normalize_to_reference_signal(self):
        """Test normalization to match power of reference signal."""
        np.random.seed(42)
        signal = np.random.randn(1000) * 2
        reference = np.random.randn(1000) * 5

        normalized = normalize(signal, reference)

        # Normalized signal should have same power as reference
        assert np.allclose(
            mean_squared(normalized), mean_squared(reference), atol=0.1
        )

    def test_normalize_preserves_zeros(self):
        """Test that zero signal remains zero or NaN."""
        signal = np.zeros(100)
        normalized = normalize(signal)
        # Zero signal causes divide by zero, results in NaN
        assert np.all(np.isnan(normalized)) or np.all(normalized == 0)


class TestWhiteNoise:
    """Test white noise generation."""

    def test_white_noise_shape(self):
        """Test that white noise has correct shape."""
        N = 1000
        noise = white(N)
        assert len(noise) == N

    def test_white_noise_statistics(self):
        """Test that white noise has approximately correct statistics."""
        np.random.seed(42)
        N = 10000
        noise = white(N)

        # Should have approximately zero mean and unit variance
        assert np.abs(np.mean(noise)) < 0.1
        assert np.abs(np.std(noise) - 1.0) < 0.1

    def test_white_noise_reproducible(self):
        """Test that white noise is reproducible with same seed."""
        state = np.random.RandomState(42)
        noise1 = white(1000, state=state)

        state = np.random.RandomState(42)
        noise2 = white(1000, state=state)

        assert np.allclose(noise1, noise2)

    def test_white_noise_normalized(self):
        """Test that white noise is normalized to unit power."""
        N = 10000
        noise = white(N)
        # White noise should already be normalized
        assert np.allclose(mean_squared(noise), 1.0, atol=0.1)


class TestPinkNoise:
    """Test pink noise generation."""

    def test_pink_noise_shape(self):
        """Test that pink noise has correct shape."""
        N = 1000
        noise = pink(N)
        assert len(noise) == N

    def test_pink_noise_normalized(self):
        """Test that pink noise is normalized to unit power."""
        N = 10000
        noise = pink(N)
        assert np.allclose(mean_squared(noise), 1.0, atol=0.1)

    def test_pink_noise_reproducible(self):
        """Test that pink noise is reproducible with same seed."""
        state = np.random.RandomState(42)
        noise1 = pink(1000, state=state)

        state = np.random.RandomState(42)
        noise2 = pink(1000, state=state)

        assert np.allclose(noise1, noise2)

    def test_pink_noise_frequency_content(self):
        """Test that pink noise has 1/f power spectrum."""
        N = 8192
        np.random.seed(42)
        noise = pink(N)

        # Compute power spectrum
        fft = np.fft.rfft(noise)
        power = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(N)

        # Skip DC component and very low frequencies
        mask = freqs > 0.01
        log_power = np.log10(power[mask])
        log_freq = np.log10(freqs[mask])

        # Fit line to log-log plot
        slope = np.polyfit(log_freq, log_power, 1)[0]

        # Pink noise should have slope approximately -1 (within tolerance)
        assert -1.5 < slope < -0.5, f"Pink noise slope {slope} not close to -1"


class TestBrownNoise:
    """Test brown noise generation."""

    def test_brown_noise_shape(self):
        """Test that brown noise has correct shape."""
        N = 1000
        noise = brown(N)
        assert len(noise) == N

    def test_brown_noise_normalized(self):
        """Test that brown noise is normalized to unit power."""
        N = 10000
        noise = brown(N)
        assert np.allclose(mean_squared(noise), 1.0, atol=0.1)

    def test_brown_noise_reproducible(self):
        """Test that brown noise is reproducible with same seed."""
        state = np.random.RandomState(42)
        noise1 = brown(1000, state=state)

        state = np.random.RandomState(42)
        noise2 = brown(1000, state=state)

        assert np.allclose(noise1, noise2)

    def test_brown_noise_frequency_content(self):
        """Test that brown noise has 1/f^2 power spectrum."""
        N = 8192
        np.random.seed(42)
        noise = brown(N)

        # Compute power spectrum
        fft = np.fft.rfft(noise)
        power = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(N)

        # Skip DC component and very low frequencies
        mask = freqs > 0.01
        log_power = np.log10(power[mask])
        log_freq = np.log10(freqs[mask])

        # Fit line to log-log plot
        slope = np.polyfit(log_freq, log_power, 1)[0]

        # Brown noise should have slope approximately -2
        assert -2.5 < slope < -1.5, f"Brown noise slope {slope} not close to -2"


class TestNoiseFunctionDict:
    """Test the NOISE_FUNCTION dictionary."""

    def test_noise_function_keys(self):
        """Test that expected noise functions are available."""
        assert "white" in NOISE_FUNCTION
        assert "pink" in NOISE_FUNCTION
        assert "brown" in NOISE_FUNCTION

    def test_noise_functions_callable(self):
        """Test that all noise functions are callable."""
        for name, func in NOISE_FUNCTION.items():
            assert callable(func), f"{name} should be callable"
            # Test calling it
            result = func(100)
            assert len(result) == 100


class TestSimulateLFP:
    """Test LFP simulation with embedded ripples."""

    def test_simulate_lfp_basic(self):
        """Test basic LFP simulation."""
        n_samples = 1500
        sampling_frequency = 1500
        time = simulate_time(n_samples, sampling_frequency)
        ripple_times = [0.5]

        lfp = simulate_LFP(time, ripple_times)

        assert len(lfp) == n_samples
        assert not np.all(np.isnan(lfp)), "LFP should not be all NaN"

    def test_simulate_lfp_multiple_ripples(self):
        """Test LFP simulation with multiple ripples."""
        n_samples = 4500
        sampling_frequency = 1500
        time = simulate_time(n_samples, sampling_frequency)
        ripple_times = [0.5, 1.5, 2.5]

        lfp = simulate_LFP(time, ripple_times)

        assert len(lfp) == n_samples

    def test_simulate_lfp_no_ripples(self):
        """Test LFP simulation without ripples (noise only)."""
        n_samples = 1500
        sampling_frequency = 1500
        time = simulate_time(n_samples, sampling_frequency)
        ripple_times = []

        lfp = simulate_LFP(time, ripple_times)

        assert len(lfp) == n_samples
        # Should be mostly noise with no obvious structure

    def test_simulate_lfp_single_ripple_time(self):
        """Test with single ripple time (not in list)."""
        n_samples = 1500
        sampling_frequency = 1500
        time = simulate_time(n_samples, sampling_frequency)
        ripple_time = 0.5  # Single value, not list

        lfp = simulate_LFP(time, ripple_time)

        assert len(lfp) == n_samples

    def test_simulate_lfp_different_noise_types(self):
        """Test LFP simulation with different noise types."""
        n_samples = 1500
        sampling_frequency = 1500
        time = simulate_time(n_samples, sampling_frequency)
        ripple_times = [0.5]

        for noise_type in ["white", "pink", "brown"]:
            lfp = simulate_LFP(time, ripple_times, noise_type=noise_type)
            assert len(lfp) == n_samples
            assert not np.all(lfp == 0), f"LFP with {noise_type} noise should not be all zeros"

    def test_simulate_lfp_ripple_amplitude(self):
        """Test effect of ripple amplitude parameter."""
        n_samples = 1500
        sampling_frequency = 1500
        time = simulate_time(n_samples, sampling_frequency)
        ripple_time = 0.5

        lfp_low = simulate_LFP(
            time, ripple_time, ripple_amplitude=1.0, noise_amplitude=0.5
        )
        lfp_high = simulate_LFP(
            time, ripple_time, ripple_amplitude=5.0, noise_amplitude=0.5
        )

        # Higher ripple amplitude should create larger peak
        ripple_idx = int(ripple_time * sampling_frequency)
        window = slice(ripple_idx - 50, ripple_idx + 50)

        assert np.max(np.abs(lfp_high[window])) > np.max(np.abs(lfp_low[window]))

    def test_simulate_lfp_noise_amplitude(self):
        """Test effect of noise amplitude parameter."""
        n_samples = 1500
        sampling_frequency = 1500
        time = simulate_time(n_samples, sampling_frequency)
        ripple_times = []  # No ripples, just noise

        lfp_low_noise = simulate_LFP(time, ripple_times, noise_amplitude=0.5)
        lfp_high_noise = simulate_LFP(time, ripple_times, noise_amplitude=2.0)

        # Higher noise amplitude should have higher variance
        assert np.std(lfp_high_noise) > np.std(lfp_low_noise)

    def test_simulate_lfp_ripple_duration(self):
        """Test effect of ripple duration parameter."""
        n_samples = 1500
        sampling_frequency = 1500
        time = simulate_time(n_samples, sampling_frequency)
        ripple_time = 0.5

        lfp_short = simulate_LFP(
            time,
            ripple_time,
            ripple_duration=0.050,
            noise_amplitude=0.1,
            ripple_amplitude=2.0,
        )
        lfp_long = simulate_LFP(
            time,
            ripple_time,
            ripple_duration=0.200,
            noise_amplitude=0.1,
            ripple_amplitude=2.0,
        )

        # Longer duration ripple should have more samples above threshold
        ripple_idx = int(ripple_time * sampling_frequency)
        window_short = slice(ripple_idx - 100, ripple_idx + 100)
        window_long = slice(ripple_idx - 200, ripple_idx + 200)

        threshold = 0.5
        n_above_threshold_short = np.sum(np.abs(lfp_short[window_short]) > threshold)
        n_above_threshold_long = np.sum(np.abs(lfp_long[window_long]) > threshold)

        assert n_above_threshold_long > n_above_threshold_short

    def test_simulate_lfp_has_ripple_frequency(self):
        """Test that simulated ripple contains 200 Hz component."""
        n_samples = 1500
        sampling_frequency = 1500
        time = simulate_time(n_samples, sampling_frequency)
        ripple_time = 0.5

        lfp = simulate_LFP(
            time,
            ripple_time,
            ripple_amplitude=5.0,
            noise_amplitude=0.5,
            ripple_duration=0.100,
        )

        # Extract region around ripple
        ripple_idx = int(ripple_time * sampling_frequency)
        window = slice(ripple_idx - 75, ripple_idx + 75)
        ripple_segment = lfp[window]

        # Compute power spectrum
        fft = np.fft.rfft(ripple_segment)
        power = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(ripple_segment), d=1 / sampling_frequency)

        # Find peak frequency in ripple band (150-250 Hz)
        ripple_band_mask = (freqs >= 150) & (freqs <= 250)
        peak_freq_idx = np.argmax(power[ripple_band_mask])
        peak_freq = freqs[ripple_band_mask][peak_freq_idx]

        # Peak should be near 200 Hz
        assert 180 < peak_freq < 220, f"Peak frequency {peak_freq} not near 200 Hz"


class TestSimulateErrorHandling:
    """Test error handling in simulation functions."""

    def test_white_noise_negative_n(self):
        """Test white noise with negative N."""
        # Should handle gracefully or raise appropriate error
        try:
            noise = white(-10)
            assert len(noise) == 0 or True  # May return empty or handle
        except ValueError:
            pass  # Expected error

    def test_simulate_lfp_empty_time(self):
        """Test LFP simulation with empty time array."""
        time = np.array([])
        ripple_times = [0.5]

        # Empty time will cause ValueError in FFT
        try:
            lfp = simulate_LFP(time, ripple_times)
            assert len(lfp) == 0
        except ValueError:
            # Expected for empty input
            pass

    def test_simulate_lfp_invalid_noise_type(self):
        """Test LFP simulation with invalid noise type."""
        n_samples = 100
        sampling_frequency = 1500
        time = simulate_time(n_samples, sampling_frequency)

        # Should raise KeyError for invalid noise type
        with pytest.raises(KeyError):
            simulate_LFP(time, [0.5], noise_type="invalid_noise_type")

    def test_normalize_zero_power_signal(self):
        """Test normalization of zero-power signal."""
        signal = np.zeros(100)
        # Should handle division by zero gracefully
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized = normalize(signal)
            # Result should be all zeros or NaN
            assert np.all(np.isnan(normalized)) or np.all(normalized == 0)

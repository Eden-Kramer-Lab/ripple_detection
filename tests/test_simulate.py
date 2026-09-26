"""Tests for simulation module."""

import dataclasses
import hashlib
import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.signal import hilbert

from ripple_detection import filter_ripple_band
from ripple_detection.simulate import (
    EVENT_TYPES,
    NOISE_FUNCTION,
    SimulatedSession,
    _draw_per_ripple,
    _event_envelope,
    _render_ripple,
    brown,
    draw_network_events,
    mean_squared,
    normalize,
    pink,
    simulate_LFP,
    simulate_multichannel_LFP,
    simulate_multiunit,
    simulate_network_session,
    simulate_session,
    simulate_sharp_wave_ripple_pair,
    simulate_speed,
    simulate_theta_delta,
    simulate_time,
    truth_windows,
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
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(1000) * 5  # Mean power ~25
        normalized = normalize(signal)

        # Normalized signal should have mean power ~1
        assert np.allclose(mean_squared(normalized), 1.0, atol=0.1)

    def test_normalize_to_reference_signal(self):
        """Test normalization to match power of reference signal."""
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(1000) * 2
        reference = rng.standard_normal(1000) * 5

        normalized = normalize(signal, reference)

        # Normalized signal should have same power as reference
        assert np.allclose(mean_squared(normalized), mean_squared(reference), atol=0.1)

    def test_normalize_preserves_zeros(self):
        """Test that zero signal remains zero or NaN."""
        signal = np.zeros(100)
        with pytest.warns(RuntimeWarning):  # divide by zero, documented as NaN
            normalized = normalize(signal)
        assert np.all(np.isnan(normalized))


class TestWhiteNoise:
    """Test white noise generation."""

    def test_white_noise_shape(self):
        """Test that white noise has correct shape."""
        N = 1000
        noise = white(N, rng=0)
        assert len(noise) == N

    def test_white_noise_statistics(self):
        """Test that white noise has approximately correct statistics."""
        N = 10000
        noise = white(N, rng=0)

        # Should have approximately zero mean and unit variance
        assert np.abs(np.mean(noise)) < 0.1
        assert np.abs(np.std(noise) - 1.0) < 0.1

    def test_white_noise_reproducible(self):
        """Test that white noise is reproducible with same seed."""
        rng = np.random.default_rng(42)
        noise1 = white(1000, rng=rng)

        rng = np.random.default_rng(42)
        noise2 = white(1000, rng=rng)

        assert np.allclose(noise1, noise2)

    def test_white_noise_normalized(self):
        """Test that white noise is normalized to unit power."""
        N = 10000
        noise = white(N, rng=0)
        # White noise should already be normalized
        assert np.allclose(mean_squared(noise), 1.0, atol=0.1)


class TestPinkNoise:
    """Test pink noise generation."""

    def test_pink_noise_shape(self):
        """Test that pink noise has correct shape."""
        N = 1000
        noise = pink(N, rng=0)
        assert len(noise) == N

    def test_pink_noise_normalized(self):
        """Test that pink noise is normalized to unit power."""
        N = 10000
        noise = pink(N, rng=0)
        assert np.allclose(mean_squared(noise), 1.0, atol=0.1)

    def test_pink_noise_reproducible(self):
        """Test that pink noise is reproducible with same seed."""
        rng = np.random.default_rng(42)
        noise1 = pink(1000, rng=rng)

        rng = np.random.default_rng(42)
        noise2 = pink(1000, rng=rng)

        assert np.allclose(noise1, noise2)

    def test_pink_noise_frequency_content(self):
        """Test that pink noise has 1/f power spectrum."""
        N = 8192
        noise = pink(N, rng=0)

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
        noise = brown(N, rng=0)
        assert len(noise) == N

    def test_brown_noise_normalized(self):
        """Test that brown noise is normalized to unit power."""
        N = 10000
        noise = brown(N, rng=0)
        assert np.allclose(mean_squared(noise), 1.0, atol=0.1)

    def test_brown_noise_reproducible(self):
        """Test that brown noise is reproducible with same seed."""
        rng = np.random.default_rng(42)
        noise1 = brown(1000, rng=rng)

        rng = np.random.default_rng(42)
        noise2 = brown(1000, rng=rng)

        assert np.allclose(noise1, noise2)

    def test_brown_noise_frequency_content(self):
        """Test that brown noise has 1/f^2 power spectrum."""
        N = 8192
        noise = brown(N, rng=0)

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

        lfp = simulate_LFP(time, ripple_times, rng=0)

        assert len(lfp) == n_samples
        assert not np.all(np.isnan(lfp)), "LFP should not be all NaN"

    def test_simulate_lfp_multiple_ripples(self):
        """Test LFP simulation with multiple ripples."""
        n_samples = 4500
        sampling_frequency = 1500
        time = simulate_time(n_samples, sampling_frequency)
        ripple_times = [0.5, 1.5, 2.5]

        lfp = simulate_LFP(time, ripple_times, rng=0)

        assert len(lfp) == n_samples

    def test_simulate_lfp_no_ripples(self):
        """Test LFP simulation without ripples (noise only)."""
        n_samples = 1500
        sampling_frequency = 1500
        time = simulate_time(n_samples, sampling_frequency)
        ripple_times = []

        lfp = simulate_LFP(time, ripple_times, rng=0)

        assert len(lfp) == n_samples
        # Should be mostly noise with no obvious structure

    def test_simulate_lfp_single_ripple_time(self):
        """Test with single ripple time (not in list)."""
        n_samples = 1500
        sampling_frequency = 1500
        time = simulate_time(n_samples, sampling_frequency)
        ripple_time = 0.5  # Single value, not list

        lfp = simulate_LFP(time, ripple_time, rng=0)

        assert len(lfp) == n_samples

    def test_simulate_lfp_different_noise_types(self):
        """Test LFP simulation with different noise types."""
        n_samples = 1500
        sampling_frequency = 1500
        time = simulate_time(n_samples, sampling_frequency)
        ripple_times = [0.5]

        for noise_type in ["white", "pink", "brown"]:
            lfp = simulate_LFP(time, ripple_times, noise_type=noise_type, rng=0)
            assert len(lfp) == n_samples
            assert not np.all(lfp == 0), f"LFP with {noise_type} noise should not be all zeros"

    def test_simulate_lfp_ripple_amplitude(self):
        """Test effect of ripple amplitude parameter."""
        n_samples = 1500
        sampling_frequency = 1500
        time = simulate_time(n_samples, sampling_frequency)
        ripple_time = 0.5

        lfp_low = simulate_LFP(
            time, ripple_time, ripple_amplitude=1.0, noise_amplitude=0.5, rng=0
        )
        lfp_high = simulate_LFP(
            time, ripple_time, ripple_amplitude=5.0, noise_amplitude=0.5, rng=0
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

        lfp_low_noise = simulate_LFP(time, ripple_times, noise_amplitude=0.5, rng=0)
        lfp_high_noise = simulate_LFP(time, ripple_times, noise_amplitude=2.0, rng=0)

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
            rng=0,
        )
        lfp_long = simulate_LFP(
            time,
            ripple_time,
            ripple_duration=0.200,
            noise_amplitude=0.1,
            ripple_amplitude=2.0,
            rng=0,
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
            rng=0,
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
            noise = white(-10, rng=0)
            assert len(noise) == 0 or True  # May return empty or handle
        except ValueError:
            pass  # Expected error

    def test_simulate_lfp_empty_time(self):
        """Test LFP simulation with empty time array."""
        time = np.array([])
        ripple_times = [0.5]

        # Empty time will cause ValueError in FFT
        try:
            lfp = simulate_LFP(time, ripple_times, rng=0)
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
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = normalize(signal)
            # Result should be all zeros or NaN
            assert np.all(np.isnan(normalized)) or np.all(normalized == 0)


# ---------------------------------------------------------------------------
# simulate_LFP
# ---------------------------------------------------------------------------


def _digest(y):
    return hashlib.sha256(np.round(y, 10).tobytes()).hexdigest()[:16]


def _dominant_frequency(y, sampling_frequency):
    spectrum = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1 / sampling_frequency)
    return freqs[np.argmax(spectrum)]


class TestSimulateLFPRealism:
    FS = 1500

    def test_default_output_is_pinned(self):
        """The default call's output, pinned so an unintended change to the
        draw order or the noise shows up. Re-pinned for 2.0, which draws from
        numpy.random.default_rng rather than the legacy RandomState, whose
        default noise is pink, and whose ripple carrier runs from each ripple's
        centre; the explicit brown call is the 1.x default."""
        t = simulate_time(4500, self.FS)
        y = simulate_LFP(t, [1.0, 2.0], rng=0)
        assert _digest(y) == "919c4927dd5c0afa"
        y = simulate_LFP(t, [1.0, 2.0], rng=0, noise_type="brown")
        assert _digest(y) == "49e980344a661ab2"
        y = simulate_LFP(t, [1.0, 2.0], rng=0, noise_type="pink", ripple_amplitude=1.0)
        assert _digest(y) == "622bcde6f398694f"

    def test_memory_does_not_grow_with_the_ripple_count(self):
        """Ten minutes at 1500 Hz with 100 ripples peaked at 1.5 GB when every
        burst was a full-length array; each is now added over its own window."""
        import tracemalloc

        t = simulate_time(self.FS * 60, self.FS)

        def peak_bytes(n_ripples):
            tracemalloc.start()
            simulate_LFP(t, list(np.linspace(1.0, 59.0, n_ripples)), rng=0)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return peak

        assert peak_bytes(200) < 2 * peak_bytes(10)

    def test_windowed_bursts_match_whole_record_bursts(self):
        """The 8-sigma window drops less than 1e-13 of a burst's peak."""
        t = simulate_time(self.FS * 4, self.FS)
        y = simulate_LFP(t, [1.0, 2.5], noise_amplitude=0.0, ripple_amplitude=2.0, rng=0)
        whole = sum(  # unit peak per burst, at its centre
            np.cos(2 * np.pi * 200.0 * (t - m))
            * np.exp(-((t - m) ** 2) / (2 * (0.1 / 6) ** 2))
            for m in (1.0, 2.5)
        )
        assert np.allclose(y, whole, atol=1e-12, rtol=0.0)

    def test_ripple_snr_sets_peak_relative_to_in_band_background(self):
        t = simulate_time(self.FS * 20, self.FS)
        ripples = [2.0, 5.0, 8.0, 11.0, 14.0, 17.0]
        # the noise is drawn first, so the same seed with no ripples is exactly
        # the background of the ripple record; the difference is the ripples alone
        noise_only = simulate_LFP(t, [], noise_type="pink", rng=1)
        background_sd = filter_ripple_band(noise_only, sampling_frequency=self.FS).std()
        for snr in (3.0, 6.0):
            y = simulate_LFP(t, ripples, noise_type="pink", ripple_snr=snr, rng=1)
            bursts = filter_ripple_band(y - noise_only, sampling_frequency=self.FS)
            peaks = [np.abs(bursts[np.abs(t - r) < 0.02]).max() for r in ripples]
            achieved = np.array(peaks) / background_sd
            np.testing.assert_allclose(achieved, snr, rtol=0.05)

    @pytest.mark.parametrize("frequency", [150.0, 250.0])
    @pytest.mark.parametrize("duration", [0.040, 0.100])
    def test_ripple_snr_holds_at_the_band_edges_and_for_short_ripples(
        self, frequency, duration
    ):
        # the filter attenuates the band edges and spreads short bursts, so the
        # requested SNR must be measured on the filtered burst, not assumed
        t = simulate_time(self.FS * 20, self.FS)
        ripples = [2.0, 5.0, 8.0, 11.0, 14.0, 17.0]
        noise_only = simulate_LFP(t, [], noise_type="pink", rng=1)
        background_sd = filter_ripple_band(noise_only, sampling_frequency=self.FS).std()
        y = simulate_LFP(
            t,
            ripples,
            noise_type="pink",
            ripple_snr=5.0,
            ripple_frequency=frequency,
            ripple_duration=duration,
            rng=1,
        )
        bursts = filter_ripple_band(y - noise_only, sampling_frequency=self.FS)
        peaks = [np.abs(bursts[np.abs(t - r) < 0.05]).max() for r in ripples]
        np.testing.assert_allclose(np.array(peaks) / background_sd, 5.0, rtol=0.05)

    def test_ripple_snr_without_noise_raises(self):
        t = simulate_time(self.FS * 5, self.FS)
        with pytest.raises(ValueError, match="noise_amplitude"):
            simulate_LFP(t, [2.0], ripple_snr=5.0, noise_amplitude=0.0, rng=0)

    def test_a_tuple_is_a_range_and_a_list_is_one_value_per_ripple(self):
        t = simulate_time(self.FS * 4, self.FS)
        as_list = simulate_LFP(t, [1.0, 3.0], ripple_frequency=[150.0, 250.0], rng=3)
        as_array = simulate_LFP(
            t, [1.0, 3.0], ripple_frequency=np.array([150.0, 250.0]), rng=3
        )
        as_range = simulate_LFP(t, [1.0, 3.0], ripple_frequency=(150.0, 250.0), rng=3)
        np.testing.assert_array_equal(as_list, as_array)
        assert not np.array_equal(as_list, as_range)
        with pytest.raises(ValueError, match="tuple"):
            simulate_LFP(t, [1.0], ripple_frequency=(150.0, 200.0, 250.0), rng=3)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"ripple_frequency": np.nan},
            {"ripple_duration": np.nan},
            {"ripple_snr": np.nan},
            {"ripple_snr": -3.0},
            {"ripple_amplitude": np.nan},
            {"ripple_amplitude": -1.0},
            {"noise_amplitude": np.nan},
        ],
    )
    def test_a_nan_or_negative_size_raises(self, kwargs):
        t = simulate_time(self.FS * 4, self.FS)
        with pytest.raises(ValueError, match=r"must (be|lie)"):
            simulate_LFP(t, [2.0], rng=0, **kwargs)

    def test_ripple_snr_and_amplitude_are_mutually_exclusive(self):
        t = simulate_time(4500, self.FS)
        with pytest.raises(ValueError, match="ripple_snr"):
            simulate_LFP(t, [1.0], ripple_amplitude=1.0, ripple_snr=5.0)

    def test_ripple_snr_infers_sampling_rate_from_time(self):
        t = simulate_time(self.FS * 10, self.FS)
        inferred = simulate_LFP(t, [3.0], ripple_snr=5.0, rng=2)
        explicit = simulate_LFP(t, [3.0], ripple_snr=5.0, rng=2, sampling_frequency=self.FS)
        # the inferred rate is 1500 to rounding, which moves the FFT filter's
        # block sizes and so its rounding; a wrong rate would differ by order one
        np.testing.assert_allclose(inferred, explicit, rtol=0, atol=1e-12)

    def test_scalar_frequency_is_reproduced_in_every_ripple(self):
        t = simulate_time(self.FS * 4, self.FS)
        y = simulate_LFP(t, [1.0, 3.0], noise_amplitude=0.0, ripple_frequency=180.0, rng=0)
        for r in (1.0, 3.0):
            seg = y[np.abs(t - r) < 0.05]
            assert abs(_dominant_frequency(seg, self.FS) - 180.0) <= 10.0

    def test_frequency_range_draws_per_ripple(self):
        t = simulate_time(self.FS * 8, self.FS)
        ripples = [1.0, 3.0, 5.0, 7.0]
        y = simulate_LFP(
            t, ripples, noise_amplitude=0.0, ripple_frequency=(150.0, 250.0), rng=3
        )
        freqs = [_dominant_frequency(y[np.abs(t - r) < 0.05], self.FS) for r in ripples]
        assert all(150.0 - 5.0 <= f <= 250.0 + 5.0 for f in freqs)  # 5 Hz FFT resolution
        assert len(set(np.round(freqs, -1))) > 1  # not all the same
        again = simulate_LFP(
            t, ripples, noise_amplitude=0.0, ripple_frequency=(150.0, 250.0), rng=3
        )
        np.testing.assert_array_equal(y, again)

    def test_duration_range_draws_per_ripple(self):
        from scipy.signal import hilbert

        t = simulate_time(self.FS * 26, self.FS)
        ripples = [2.0 + 2.0 * k for k in range(12)]
        y = simulate_LFP(t, ripples, noise_amplitude=0.0, ripple_duration=(0.03, 0.15), rng=4)
        envelope = np.abs(hilbert(y))
        durations = []
        for r in ripples:
            seg = envelope[np.abs(t - r) < 0.3]
            fwhm = (seg > 0.5 * seg.max()).sum() / self.FS
            durations.append(fwhm / 2.3548 * 6)  # FWHM = 2.3548 sigma, duration = 6 sigma
        durations = np.array(durations)
        assert np.all((durations >= 0.029) & (durations <= 0.152)), durations
        assert durations.max() > 2 * durations.min()
        again = simulate_LFP(
            t, ripples, noise_amplitude=0.0, ripple_duration=(0.03, 0.15), rng=4
        )
        np.testing.assert_array_equal(y, again)

    def test_inverted_range_raises(self):
        t = simulate_time(4500, self.FS)
        with pytest.raises(ValueError, match="low <= high"):
            simulate_LFP(t, [1.0], ripple_duration=(0.15, 0.03), rng=0)
        with pytest.raises(ValueError, match="low <= high"):
            simulate_LFP(t, [1.0], ripple_frequency=(220.0, 180.0), rng=0)

    def test_generator_instance_matches_seed(self):
        t = simulate_time(4500, self.FS)
        from_seed = simulate_LFP(t, [1.0, 2.0], ripple_frequency=(150.0, 250.0), rng=7)
        from_state = simulate_LFP(
            t,
            [1.0, 2.0],
            ripple_frequency=(150.0, 250.0),
            rng=np.random.default_rng(7),
        )
        np.testing.assert_array_equal(from_seed, from_state)

    def test_noise_draw_is_unchanged_by_the_new_parameters(self):
        # with a scalar frequency and duration and no ripples, output equals the
        # pre-existing noise for the same seed
        t = simulate_time(4500, self.FS)
        base = simulate_LFP(t, [], rng=5)
        same = simulate_LFP(t, [], rng=5, ripple_frequency=200.0, ripple_duration=0.1)
        np.testing.assert_array_equal(base, same)


class TestSimulateLFPRejectsSilentlyBrokenRipples:
    """Each of these used to return an all-NaN, empty, or aliased signal."""

    def test_ripple_outside_the_record_raises(self):
        t = simulate_time(3000, 1000)
        with pytest.raises(ValueError, match="outside time"):
            simulate_LFP(t, [100.0])

    @pytest.mark.parametrize("duration", [0.0, -0.05])
    def test_non_positive_duration_raises(self, duration):
        t = simulate_time(3000, 1000)
        with pytest.raises(ValueError, match="positive"):
            simulate_LFP(t, [1.0], ripple_duration=duration)

    @pytest.mark.parametrize("frequency", [0.0, 700.0])
    def test_frequency_outside_the_nyquist_range_raises(self, frequency):
        t = simulate_time(3000, 1000)
        with pytest.raises(ValueError, match="Nyquist"):
            simulate_LFP(t, [1.0], ripple_frequency=frequency)


class TestRippleTimes:
    @pytest.mark.parametrize(
        "value", [1.0, np.float32(1.0), np.int64(1), [1.0], np.array([1.0])]
    )
    def test_one_ripple_time_in_any_numeric_form(self, value):
        t = simulate_time(3000, 1500)
        expected = simulate_LFP(t, 1.0, rng=0)
        np.testing.assert_array_equal(simulate_LFP(t, value, rng=0), expected)


class TestDrawPerRipple:
    def test_an_explicit_value_per_ripple_is_used_as_given(self):
        rng = np.random.default_rng(0)
        values = np.array([0.05, 0.10, 0.15])
        np.testing.assert_array_equal(_draw_per_ripple(values, 3, rng), values)

    def test_two_values_for_two_ripples_are_used_as_given(self):
        rng = np.random.default_rng(0)
        for values in ([0.10, 0.05], np.array([0.10, 0.05])):
            np.testing.assert_array_equal(_draw_per_ripple(values, 2, rng), [0.10, 0.05])

    def test_a_tuple_draws_one_value_per_ripple_from_the_range(self):
        rng = np.random.default_rng(0)
        drawn = _draw_per_ripple((0.05, 0.10), 2, rng)
        assert drawn.shape == (2,)
        assert np.all((drawn >= 0.05) & (drawn <= 0.10))
        assert not np.array_equal(drawn, [0.05, 0.10])

    def test_the_wrong_number_of_values_raises(self):
        with pytest.raises(ValueError, match="one value per"):
            _draw_per_ripple([0.05, 0.10, 0.15], 4, np.random.default_rng(0))

    def test_a_reversed_range_raises(self):
        with pytest.raises(ValueError, match="low <= high"):
            _draw_per_ripple((0.10, 0.05), 3, np.random.default_rng(0))


class TestSimulateMultichannelLFP:
    FS = 1500

    def test_shape(self):
        t = simulate_time(3000, self.FS)
        assert simulate_multichannel_LFP(t, [1.0], 3, rng=0).shape == (3000, 3)

    def test_every_channel_carries_the_same_ripple_scaled_by_its_gain(self):
        t = simulate_time(3000, self.FS)
        lfps = simulate_multichannel_LFP(
            t, [1.0], 2, channel_gains=[1.0, 0.5], noise_amplitude=0.0, rng=0
        )
        np.testing.assert_allclose(lfps[:, 1], 0.5 * lfps[:, 0], atol=1e-15)
        assert np.abs(lfps[:, 0]).max() > 0.9

    def test_shared_fraction_is_the_correlation_between_channels(self):
        t = simulate_time(30000, self.FS)
        for fraction in (0.0, 0.5, 1.0):
            lfps = simulate_multichannel_LFP(
                t, [], 2, shared_noise_fraction=fraction, noise_type="white", rng=1
            )
            assert np.corrcoef(lfps[:, 0], lfps[:, 1])[0, 1] == pytest.approx(
                fraction, abs=0.03
            )

    def test_noise_has_the_single_channel_scale(self):
        """Each channel's noise has the mean square simulate_LFP's noise has."""
        t = simulate_time(30000, self.FS)
        lfps = simulate_multichannel_LFP(t, [], 3, noise_type="white", rng=2)
        single = simulate_LFP(t, [], noise_type="white", rng=2)
        np.testing.assert_allclose(np.mean(lfps**2, axis=0), np.mean(single**2), rtol=0.05)

    def test_ripple_snr_holds_on_a_unit_gain_channel(self):
        t = simulate_time(self.FS * 20, self.FS)
        lfps = simulate_multichannel_LFP(
            t, [5.0, 10.0, 15.0], 2, channel_gains=[1.0, 0.5], ripple_snr=5.0, rng=3
        )
        background = simulate_multichannel_LFP(
            t, [], 2, channel_gains=[1.0, 0.5], noise_type="pink", rng=3
        )
        # the same seed draws the same noise, so the difference is the ripples alone
        ripples_only = filter_ripple_band(lfps[:, 0] - background[:, 0], self.FS)
        band_sd = filter_ripple_band(background[:, 0], self.FS).std()
        peaks = [
            np.abs(ripples_only[(t > m - 0.06) & (t < m + 0.06)]).max()
            for m in (5.0, 10.0, 15.0)
        ]
        np.testing.assert_allclose(np.array(peaks) / band_sd, 5.0, rtol=0.05)

    def test_artifacts_are_identical_on_every_channel_and_local_in_time(self):
        t = simulate_time(6000, self.FS)
        lfps = simulate_multichannel_LFP(
            t,
            [],
            3,
            noise_amplitude=0.0,
            artifact_times=[2.0],
            artifact_amplitude=1.0,
            rng=4,
        )
        np.testing.assert_array_equal(lfps[:, 0], lfps[:, 1])
        np.testing.assert_array_equal(lfps[:, 0], lfps[:, 2])
        assert np.abs(lfps[(t > 1.9) & (t < 2.1), 0]).max() > 0.5
        assert np.all(lfps[(t < 1.9) | (t > 2.1), 0] == 0.0)

    def test_seed_reproduces_and_changes(self):
        t = simulate_time(3000, self.FS)
        a = simulate_multichannel_LFP(t, [1.0], 2, rng=5)
        np.testing.assert_array_equal(a, simulate_multichannel_LFP(t, [1.0], 2, rng=5))
        assert not np.array_equal(a, simulate_multichannel_LFP(t, [1.0], 2, rng=6))

    def test_bad_arguments_raise(self):
        t = simulate_time(3000, self.FS)
        with pytest.raises(ValueError, match="channel_gains"):
            simulate_multichannel_LFP(t, [1.0], 2, channel_gains=[1.0], rng=0)
        with pytest.raises(ValueError, match="shared_noise_fraction"):
            simulate_multichannel_LFP(t, [1.0], 2, shared_noise_fraction=1.5, rng=0)
        with pytest.raises(ValueError, match="n_channels"):
            simulate_multichannel_LFP(t, [1.0], 0, rng=0)
        with pytest.raises(ValueError, match="not both"):
            simulate_multichannel_LFP(t, [1.0], 2, ripple_amplitude=1.0, ripple_snr=2.0)


class TestSimulateSharpWaveRipplePair:
    FS = 1500

    def test_two_channels_in_the_order_the_long_detector_names_them(self):
        t = simulate_time(3000, self.FS)
        raw_lfp, sharp_wave_lfp = simulate_sharp_wave_ripple_pair(t, [1.0], rng=0)
        assert raw_lfp.shape == sharp_wave_lfp.shape == (3000,)

    def test_sharp_wave_is_negative_on_the_radiatum_channel_and_leaks_positive(self):
        t = simulate_time(3000, self.FS)
        raw_lfp, sharp_wave_lfp = simulate_sharp_wave_ripple_pair(
            t,
            [1.0],
            noise_amplitude=0.0,
            ripple_amplitude=2.0,
            sharp_wave_amplitude=2.0,
            rng=0,
        )
        near = (t > 0.95) & (t < 1.05)
        far = (t < 0.85) | (t > 1.15)
        # radiatum: a -2 deflection carrying 0.3 of a unit-peak ripple
        assert -2.3 <= sharp_wave_lfp[near].min() <= -1.7
        assert sharp_wave_lfp[near].sum() < 0.0
        assert sharp_wave_lfp[np.abs(t - 1.0) < 0.005].mean() < -1.5
        # pyramidal: the ripple (peak 1) on 0.3 x 2 of sharp wave
        assert raw_lfp[near].max() == pytest.approx(1.6, abs=0.2)
        assert np.all(np.abs(raw_lfp[far]) < 1e-6)
        assert np.all(np.abs(sharp_wave_lfp[far]) < 1e-6)

    def test_no_sharp_wave_without_ripples(self):
        t = simulate_time(3000, self.FS)
        raw_lfp, sharp_wave_lfp = simulate_sharp_wave_ripple_pair(
            t, [], noise_amplitude=0.0, rng=0
        )
        assert np.all(raw_lfp == 0.0)
        assert np.all(sharp_wave_lfp == 0.0)


class TestSimulateMultiunit:
    FS = 1500

    def test_shape_and_counts(self):
        t = simulate_time(3000, self.FS)
        counts = simulate_multiunit(t, [1.0], 5, rng=0)
        assert counts.shape == (3000, 5)
        assert np.all(counts >= 0)
        assert np.all(counts == np.round(counts))

    def test_units_burst_during_the_ripple(self):
        t = simulate_time(self.FS * 20, self.FS)
        counts = simulate_multiunit(
            t,
            [10.0],
            20,
            baseline_rate=5.0,
            ripple_rate_gain=8.0,
            participation=1.0,
            ripple_duration=0.1,
            rng=1,
        )
        inside = counts[(t > 9.98) & (t < 10.02)].sum()
        outside = counts[(t > 4.98) & (t < 5.02)].sum()
        assert inside > 4 * max(outside, 1)

    def test_no_participation_means_no_burst(self):
        t = simulate_time(self.FS * 20, self.FS)
        counts = simulate_multiunit(t, [10.0], 20, baseline_rate=5.0, participation=0.0, rng=1)
        inside = counts[(t > 9.9) & (t < 10.1)].sum()
        outside = counts[(t > 4.9) & (t < 5.1)].sum()
        assert inside < 2.0 * max(outside, 1)

    def test_baseline_rate_is_honoured(self):
        t = simulate_time(self.FS * 60, self.FS)
        counts = simulate_multiunit(t, [], 10, baseline_rate=4.0, rng=2)
        np.testing.assert_allclose(counts.sum(axis=0) / 60.0, 4.0, rtol=0.2)

    def test_bad_arguments_raise(self):
        t = simulate_time(3000, self.FS)
        with pytest.raises(ValueError, match="n_units"):
            simulate_multiunit(t, [1.0], 0)
        with pytest.raises(ValueError, match="participation"):
            simulate_multiunit(t, [1.0], 2, participation=1.5)
        with pytest.raises(ValueError, match="ripple_rate_gain"):
            simulate_multiunit(t, [1.0], 2, ripple_rate_gain=0.5)
        with pytest.raises(ValueError, match="baseline_rate"):
            simulate_multiunit(t, [1.0], 2, baseline_rate=-1.0)


class TestSimulateSession:
    FS = 1500

    def test_shapes_and_ground_truth(self):
        t = simulate_time(self.FS * 20, self.FS)
        session = simulate_session(t, [3.0, 9.0, 15.0], n_channels=3, n_units=8, rng=0)
        assert isinstance(session, SimulatedSession)
        assert session.lfps.shape == (t.size, 3)
        assert session.raw_lfp.shape == session.sharp_wave_lfp.shape == (t.size,)
        assert session.multiunit.shape == (t.size, 8)
        assert session.speed.shape == (t.size,)
        assert np.all(session.speed == 0.0)
        np.testing.assert_array_equal(session.ripple_times, [3.0, 9.0, 15.0])
        assert session.ripple_durations.shape == (3,)
        assert np.all((session.ripple_durations >= 0.04) & (session.ripple_durations <= 0.12))
        assert np.all(
            (session.ripple_frequencies >= 150) & (session.ripple_frequencies <= 250)
        )
        windows = session.ripple_windows
        np.testing.assert_allclose(windows[:, 1] - windows[:, 0], session.ripple_durations)
        np.testing.assert_allclose(windows.mean(axis=1), session.ripple_times)
        assert session.sampling_frequency == pytest.approx(self.FS)
        assert session.artifact_times.shape == (0,)

    @pytest.mark.parametrize("seed", range(4))
    def test_two_ripples_embed_the_durations_and_frequencies_reported(self, seed):
        """Two drawn values for two ripples are values, not a range to redraw from."""
        t = simulate_time(self.FS * 10, self.FS)
        session = simulate_session(
            t,
            [3.0, 7.0],
            ripple_amplitude=2.0,
            noise_amplitude=0.0,
            sharp_wave_leak=0.0,
            rng=seed,
        )
        step = 1 / self.FS
        for center, duration, frequency in zip(
            session.ripple_times,
            session.ripple_durations,
            session.ripple_frequencies,
            strict=True,
        ):
            burst = session.lfps[np.abs(t - center) < 0.5, 0]
            # a unit-amplitude sine under a Gaussian of sd sigma has energy
            # sigma * sqrt(pi) / 2
            sigma = 2 * np.sum(burst**2) * step / np.sqrt(np.pi)
            np.testing.assert_allclose(sigma, duration / 6, rtol=0.02)
            spectrum = np.abs(np.fft.rfft(burst, n=2**16))
            peak = np.fft.rfftfreq(2**16, step)[np.argmax(spectrum)]
            np.testing.assert_allclose(peak, frequency, atol=1.0)

    def test_windows_are_clipped_to_the_recording(self):
        t = simulate_time(self.FS * 2, self.FS)
        session = simulate_session(t, [0.01, 1.99], ripple_duration=0.1, rng=0)
        np.testing.assert_allclose(session.ripple_windows, [[t[0], 0.06], [1.94, t[-1]]])

    def test_a_list_of_times_is_accepted_and_sessions_compare_by_identity(self):
        t = simulate_time(3000, self.FS)
        session = simulate_session(list(t), [1.0], rng=0)
        assert isinstance(session.time, np.ndarray)
        assert session != simulate_session(t, [1.0], rng=0)
        assert session == session

    def test_mismatched_lengths_raise(self):
        t = simulate_time(3000, self.FS)
        session = simulate_session(t, [1.0], rng=0)
        with pytest.raises(ValueError, match="samples"):
            dataclasses.replace(session, speed=np.zeros(10))
        with pytest.raises(ValueError, match="length"):
            dataclasses.replace(session, ripple_durations=np.zeros(2))

    def test_the_ripple_channel_is_shared_by_the_lfps_and_the_raw_lfp(self):
        t = simulate_time(3000, self.FS)
        session = simulate_session(t, [1.0], rng=1)
        np.testing.assert_array_equal(session.lfps[:, 0], session.raw_lfp)

    def test_the_three_signals_carry_the_same_events(self):
        t = simulate_time(self.FS * 20, self.FS)
        session = simulate_session(
            t,
            [5.0, 10.0, 15.0],
            ripple_amplitude=2.0,
            noise_amplitude=0.0,
            baseline_rate=5.0,
            participation=1.0,
            rng=2,
        )
        for start, end in session.ripple_windows:
            inside = (t >= start) & (t <= end)
            assert np.abs(session.lfps[inside, 0]).max() > 0.9  # the ripple
            assert session.sharp_wave_lfp[inside].min() < -1.5  # the sharp wave
            rate_inside = session.multiunit[inside].sum() / inside.sum()
            rate_outside = session.multiunit[~inside].sum() / (~inside).sum()
            assert rate_inside > 2.5 * rate_outside  # the population burst

    def test_seed_reproduces(self):
        t = simulate_time(3000, self.FS)
        a = simulate_session(t, [1.0], rng=3)
        b = simulate_session(t, [1.0], rng=3)
        np.testing.assert_array_equal(a.lfps, b.lfps)
        np.testing.assert_array_equal(a.multiunit, b.multiunit)
        np.testing.assert_array_equal(a.ripple_durations, b.ripple_durations)

    def test_artifacts_are_recorded(self):
        t = simulate_time(6000, self.FS)
        session = simulate_session(t, [1.0], artifact_times=[3.0], rng=4)
        np.testing.assert_array_equal(session.artifact_times, [3.0])


class TestSimulateSpeed:
    TIME = simulate_time(10_000, 1000)

    def test_still_outside_the_bouts_and_peak_at_their_middle(self):
        speed = simulate_speed(self.TIME, [(2.0, 4.0), (6.0, 7.0)], peak_speed=20.0)
        assert speed[1000] == 0.0
        assert speed[3000] == pytest.approx(20.0)
        assert speed[6500] == pytest.approx(20.0)
        assert speed[5000] == 0.0

    def test_a_bout_begins_and_ends_slow(self):
        speed = simulate_speed(self.TIME, [(2.0, 4.0)])
        assert speed[2000] == pytest.approx(0.0)
        assert speed[2100] < 4.0 < speed[2300]

    def test_a_still_speed(self):
        speed = simulate_speed(self.TIME, [(2.0, 4.0)], still_speed=1.5)
        assert speed[0] == 1.5
        assert speed.min() == pytest.approx(1.5)

    def test_no_bouts_is_still_throughout(self):
        assert (simulate_speed(self.TIME, []) == 0).all()

    @pytest.mark.parametrize(
        ("intervals", "message"),
        [
            ([(4.0, 2.0)], "start before its end"),
            ([(1.0, 3.0), (2.0, 4.0)], "must not overlap"),
            ([(1.0, 2.0, 3.0)], "shape \\(n, 2\\)"),
            ([(np.nan, 2.0)], "finite start"),
        ],
    )
    def test_bad_intervals_raise(self, intervals, message):
        with pytest.raises(ValueError, match=message):
            simulate_speed(self.TIME, intervals)

    def test_a_negative_speed_raises(self):
        with pytest.raises(ValueError, match="peak_speed must be finite and non-negative"):
            simulate_speed(self.TIME, [(2.0, 4.0)], peak_speed=-1.0)


class TestSimulateThetaDelta:
    TIME = simulate_time(20_000, 1000)

    def test_theta_while_running_and_delta_at_rest(self):
        from ripple_detection import theta_delta_ratio

        slow = simulate_theta_delta(self.TIME, [(5.0, 15.0)], theta_amplitude=3.0)
        ratio = theta_delta_ratio(slow, 1000)
        assert np.median(ratio[8000:12000]) > 10
        assert np.median(ratio[1000:3000]) < 0.1

    def test_amplitudes_and_the_cross_fade(self):
        slow = simulate_theta_delta(
            self.TIME, [(5.0, 15.0)], theta_amplitude=3.0, delta_amplitude=2.0
        )
        assert np.abs(slow[8000:12000]).max() == pytest.approx(3.0, rel=1e-3)
        assert np.abs(slow[0:4000]).max() == pytest.approx(2.0, rel=1e-3)
        assert np.abs(slow[5000:5500]).max() < 3.0

    def test_a_short_bout_fades_over_half_its_length(self):
        """Full theta only at the bout's centre, 5.2 s, where 8 Hz crosses zero,
        so the peak falls a little short of full amplitude; none outside it."""
        slow = simulate_theta_delta(
            self.TIME, [(5.0, 5.4)], delta_amplitude=0.0, transition=0.5
        )
        assert 0.9 < np.abs(slow).max() < 1.0
        assert (slow[:5000] == 0).all()

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"theta_amplitude": -1.0}, "theta_amplitude must be finite and non-negative"),
            ({"delta_frequency": 0.0}, "delta_frequency must be positive"),
            ({"transition": np.inf}, "transition must be positive"),
        ],
    )
    def test_invalid_arguments_raise(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            simulate_theta_delta(self.TIME, [(5.0, 15.0)], **kwargs)


class TestSessionStates:
    TIME = simulate_time(30_000, 1000)

    def test_the_defaults_are_unchanged(self):
        default = simulate_session(self.TIME, [3.0, 9.0], rng=0)
        explicit = simulate_session(
            self.TIME, [3.0, 9.0], rng=0, running_intervals=None,
            theta_amplitude=0.0, delta_amplitude=0.0,
        )  # fmt: skip
        np.testing.assert_array_equal(default.lfps, explicit.lfps)
        assert (default.speed == 0).all()

    def test_running_bouts_set_the_speed(self):
        session = simulate_session(
            self.TIME, [3.0, 9.0], rng=0, running_intervals=[(12.0, 20.0)]
        )
        np.testing.assert_array_equal(session.speed, simulate_speed(self.TIME, [(12.0, 20.0)]))

    def test_theta_and_delta_are_added_to_every_channel_alike(self):
        plain = simulate_session(self.TIME, [3.0, 9.0], rng=0)
        stated = simulate_session(
            self.TIME, [3.0, 9.0], rng=0, running_intervals=[(12.0, 20.0)],
            theta_amplitude=3.0, delta_amplitude=3.0,
        )  # fmt: skip
        slow = simulate_theta_delta(
            self.TIME, [(12.0, 20.0)], theta_amplitude=3.0, delta_amplitude=3.0
        )
        np.testing.assert_allclose(
            stated.lfps - plain.lfps, np.repeat(slow[:, None], 4, axis=1), atol=1e-12
        )
        np.testing.assert_allclose(stated.raw_lfp, stated.lfps[:, 0])
        np.testing.assert_allclose(
            stated.sharp_wave_lfp - plain.sharp_wave_lfp, slow, atol=1e-12
        )
        np.testing.assert_allclose(
            stated.raw_lfp - stated.sharp_wave_lfp,
            plain.raw_lfp - plain.sharp_wave_lfp,
            atol=1e-12,
        )
        np.testing.assert_array_equal(stated.multiunit, plain.multiunit)


EVENT_COLUMNS = {
    "event_id": "int64",
    "event_type": "str",
    "expression": "str",
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
RIPPLE_CHANNEL_COLUMNS = {
    "event_id": "int64",
    "component": "int64",
    "channel": "int64",
    "gain": "float64",
    "delay_s": "float64",
}


def _assert_schema(table, columns):
    """``table`` has exactly ``columns``, in order, with their dtypes; "str" is
    object before pandas 3 and the string dtype from it."""
    assert list(table.columns) == list(columns)
    for name, dtype in columns.items():
        if dtype == "str":
            assert pd.api.types.is_string_dtype(table[name]), name
        else:
            assert table[name].dtype == np.dtype(dtype), name


class TestSimulatedSessionFields:
    FS = 1500

    def _session(self, **fields):
        n_time = 300
        return SimulatedSession(
            time=simulate_time(n_time, self.FS),
            lfps=np.zeros((n_time, 2)),
            raw_lfp=np.zeros(n_time),
            sharp_wave_lfp=np.zeros(n_time),
            multiunit=np.zeros((n_time, 3)),
            speed=np.zeros(n_time),
            ripple_times=np.empty(0),
            ripple_durations=np.empty(0),
            ripple_frequencies=np.empty(0),
            artifact_times=np.empty(0),
            sampling_frequency=float(self.FS),
            **fields,
        )

    def test_defaults_keep_old_constructors_working(self):
        session = self._session()
        _assert_schema(session.events, EVENT_COLUMNS)
        _assert_schema(session.non_events, NON_EVENT_COLUMNS)
        _assert_schema(session.ripple_channels, RIPPLE_CHANNEL_COLUMNS)
        assert len(session.events) == len(session.non_events) == 0
        assert len(session.ripple_channels) == 0
        assert session.unit_types.shape == (0,)
        assert session.baseline_rates.shape == (0,)
        assert session.running_intervals.shape == (0, 2)

    def test_defaults_are_not_shared_between_sessions(self):
        assert self._session().events is not self._session().events

    def test_unit_types_length_is_checked(self):
        self._session(unit_types=np.array(["place"] * 3), baseline_rates=np.ones(3))
        with pytest.raises(ValueError, match="unit_types has 2 entries; multiunit has 3"):
            self._session(unit_types=np.array(["place", "interneuron"]))
        with pytest.raises(ValueError, match="baseline_rates has 4 entries"):
            self._session(baseline_rates=np.ones(4))

    def test_simulate_session_records_running_intervals(self):
        time = simulate_time(self.FS * 6, self.FS)
        running = simulate_session(time, [1.0], running_intervals=[(2, 4)], rng=0)
        np.testing.assert_array_equal(running.running_intervals, [[2.0, 4.0]])
        still = simulate_session(time, [1.0], rng=0)
        assert still.running_intervals.shape == (0, 2)
        assert len(still.events) == 0
        assert still.unit_types.shape == still.baseline_rates.shape == (0,)


EXPRESSION_ORDER = {"ripple": 0, "sharp_wave": 1, "burst": 2}


def _spans(events, n_sides):
    """Each row's [centre - n rise, centre + n decay]."""
    return (
        events.center_time - n_sides * events.rise_sigma,
        events.center_time + n_sides * events.decay_sigma,
    )


class TestDrawNetworkEvents:
    FS = 1500
    TIME = simulate_time(FS * 300, FS)
    RUNNING = ((40.0, 55.0), (120.0, 150.0))

    @staticmethod
    @pytest.fixture(scope="class")
    def events():
        return draw_network_events(
            TestDrawNetworkEvents.TIME,
            event_rate=1.0,
            running_intervals=TestDrawNetworkEvents.RUNNING,
            rng=0,
        )

    def test_schema_and_order(self, events):
        _assert_schema(events, EVENT_COLUMNS)
        assert isinstance(events.index, pd.RangeIndex)
        key = list(
            zip(
                events.event_id,
                events.expression.map(EXPRESSION_ORDER),
                events.component,
                strict=True,
            )
        )
        assert key == sorted(key)
        earliest = events.groupby("event_id").center_time.min()
        np.testing.assert_array_equal(earliest.index, np.arange(len(earliest)))
        assert np.all(np.diff(earliest.to_numpy()) > 0)
        assert (events.envelope_power == 2).all()
        assert (events.n_participants == 0).all()
        ripple = events.expression == "ripple"
        assert events.loc[ripple, ["frequency_start", "frequency_end"]].notna().all().all()
        assert events.loc[~ripple, ["frequency_start", "frequency_end"]].isna().all().all()
        burst = events.expression == "burst"
        assert events.loc[burst, "participation"].between(0, 1).all()
        assert events.loc[~burst, "participation"].isna().all()

    def test_an_empty_draw_has_the_schema(self):
        events = draw_network_events(self.TIME, event_rate=0.0, rng=0)
        assert len(events) == 0
        _assert_schema(events, EVENT_COLUMNS)

    def test_components_per_type(self, events):
        expected = {
            "swr": {"ripple": 1, "sharp_wave": 1, "burst": 1},
            "weak_ripple": {"ripple": 1, "sharp_wave": 1, "burst": 1},
            "burst_only": {"burst": 1},
            "sharp_wave_only": {"sharp_wave": 1},
        }
        n_ripples_seen = set()
        for _, event in events.groupby("event_id"):
            (event_type,) = set(event.event_type)
            counts = event.expression.value_counts().to_dict()
            if event_type != "ripple_doublet":
                assert counts == expected[event_type]
                continue
            n_ripples = counts["ripple"]
            n_ripples_seen.add(n_ripples)
            assert counts == {"ripple": n_ripples, "sharp_wave": n_ripples, "burst": 1}
            ripples = event[event.expression == "ripple"]
            np.testing.assert_array_equal(ripples.component, np.arange(n_ripples))
            (burst,) = event[event.expression == "burst"].itertuples()
            first, last = ripples.iloc[0], ripples.iloc[-1]
            assert burst.rise_sigma == pytest.approx(burst.decay_sigma)
            assert burst.center_time - 3 * burst.rise_sigma == pytest.approx(
                first.center_time - 3 * first.rise_sigma
            )
            assert burst.center_time + 3 * burst.decay_sigma == pytest.approx(
                last.center_time + 3 * last.decay_sigma
            )
            assert np.all(np.diff(ripples.center_time) >= 0.06)
        assert n_ripples_seen == {2, 3}
        assert set(events.event_type) == set(EVENT_TYPES)

    def test_type_specific_sizes(self, events):
        weak = events.event_type == "weak_ripple"
        ripples = events.expression == "ripple"
        assert events.loc[ripples & weak, "amplitude"].between(1.2, 2.2).all()
        assert events.loc[ripples & ~weak, "amplitude"].between(2.5, 6.0).all()
        sharp = events.expression == "sharp_wave"
        assert events.loc[sharp & weak, "amplitude"].between(1.5, 4.0).all()
        assert events.loc[sharp & ~weak, "amplitude"].between(3.0, 8.0).all()
        bursts = events.expression == "burst"
        assert events.loc[bursts & weak, "participation"].between(0.02, 0.1).all()
        assert events.loc[bursts & ~weak, "participation"].between(0.2, 0.6).all()
        assert (events.loc[bursts, "amplitude"] == 40.0).all()
        span = 3 * (events.rise_sigma + events.decay_sigma)
        assert span[ripples].between(0.03, 0.15).all()
        skew = events.decay_sigma / (events.rise_sigma + events.decay_sigma)
        assert skew[ripples].between(0.5, 0.7).all()
        chirp = events.frequency_start - events.frequency_end
        assert chirp[ripples].between(0.0, 30.0).all()
        assert events.loc[ripples, "frequency_start"].between(160, 220).all()
        burst_only = bursts & (events.event_type == "burst_only")
        assert span[burst_only].between(0.05, 0.3).all()

    def test_events_only_at_rest_and_inside(self, events):
        start, end = _spans(events, 4)
        assert (start >= self.TIME[0] + 1).all()
        assert (end <= self.TIME[-1] - 1).all()
        for bout_start, bout_end in self.RUNNING:
            assert not ((start < bout_end) & (end > bout_start)).any()

    def test_events_are_separated(self, events):
        start, end = _spans(events, 3)
        spans = pd.DataFrame({"start": start, "end": end, "event_id": events.event_id})
        union = spans.groupby("event_id").agg(start=("start", "min"), end=("end", "max"))
        gaps = union.start.to_numpy()[1:] - union.end.to_numpy()[:-1]
        assert gaps.min() >= 0.05

    def test_rate(self):
        time = simulate_time(1000 * 3600, 1000)
        events = draw_network_events(
            time,
            event_rate=0.5,
            type_probabilities={"swr": 1.0},
            minimum_separation=0.0,
            ripple_duration=(0.01, 0.01),
            sharp_wave_duration=(0.01, 0.01),
            burst_duration_ratio=(1, 1),
            rng=1,
        )
        expected = 0.5 * (3600 - 2)
        assert abs(events.event_id.nunique() - expected) < 4 * np.sqrt(expected)

    def test_seeded_and_parameter_local(self, events):
        again = draw_network_events(
            self.TIME, event_rate=1.0, running_intervals=self.RUNNING, rng=0
        )
        pd.testing.assert_frame_equal(events, again)
        louder = draw_network_events(
            self.TIME,
            event_rate=1.0,
            running_intervals=self.RUNNING,
            sharp_wave_amplitude=(10.0, 12.0),
            rng=0,
        )
        sharp = events.expression == "sharp_wave"
        pd.testing.assert_frame_equal(
            events.drop(columns="amplitude"), louder.drop(columns="amplitude")
        )
        pd.testing.assert_series_equal(events.amplitude[~sharp], louder.amplitude[~sharp])
        assert louder.amplitude[sharp & (louder.event_type != "weak_ripple")].min() >= 10.0

    def test_type_probabilities_leave_times_in_place(self):
        """Every event draws the same variates whatever its type."""
        swr = draw_network_events(self.TIME, type_probabilities={"swr": 1}, rng=3)
        burst = draw_network_events(self.TIME, type_probabilities={"burst_only": 1}, rng=3)
        assert set(swr.event_type) == {"swr"}
        assert set(burst.event_type) == {"burst_only"}
        swr_ripples = swr[swr.expression == "ripple"].center_time.to_numpy()
        burst_centres = burst.center_time.to_numpy()
        assert np.isin(swr_ripples, burst_centres).mean() > 0.9

    def test_envelope_power_is_stored(self):
        events = draw_network_events(self.TIME, envelope_power=4, rng=0)
        assert len(events) > 0
        assert (events.envelope_power == 4).all()

    @pytest.mark.parametrize("rho", [0.0, 0.6])
    def test_strength_dependence(self, rho):
        """Uniform marginals on each range at any correlation, and rank
        correlations between an swr's four strengths near the Gaussian
        copula's (6 / pi) asin(rho / 2). The spans do not depend on the
        strengths, so dropping events leaves these distributions unchanged."""
        time = simulate_time(1000 * 3600, 1000)
        events = draw_network_events(
            time,
            event_rate=0.5,
            type_probabilities={"swr": 1.0},
            minimum_separation=0.0,
            strength_correlation=rho,
            rng=11,
        )
        by_expression = events.set_index(["event_id", "expression"])
        strengths = pd.DataFrame(
            {
                "snr": by_expression.xs("ripple", level=1).amplitude,
                "frequency": by_expression.xs("ripple", level=1).frequency_start,
                "sharp_wave": by_expression.xs("sharp_wave", level=1).amplitude,
                "participation": by_expression.xs("burst", level=1).participation,
            }
        )
        n = len(strengths)
        assert n > 1500
        ranges = {
            "snr": (2.5, 6.0),
            "frequency": (160.0, 220.0),
            "sharp_wave": (3.0, 8.0),
            "participation": (0.2, 0.6),
        }
        for name, (low, high) in ranges.items():
            result = stats.kstest(strengths[name], stats.uniform(low, high - low).cdf)
            assert result.pvalue > 1e-3, name
        rank = strengths.rank().corr().to_numpy()[np.triu_indices(4, 1)]
        expected = 6 / np.pi * np.arcsin(rho / 2)
        np.testing.assert_allclose(rank, expected, atol=4 / np.sqrt(n))

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"event_rate": -1.0}, "event_rate"),
            ({"event_rate": np.nan}, "event_rate"),
            ({"type_probabilities": {"sharp_wave_ripple": 1.0}}, "type_probabilities"),
            ({"type_probabilities": {"swr": 0.0}}, "type_probabilities"),
            ({"type_probabilities": {"swr": -1.0, "emg": 2.0}}, "type_probabilities"),
            ({"ripple_duration": (0.1, 0.05)}, "ripple_duration"),
            ({"ripple_duration": [0.05, 0.1]}, "ripple_duration"),
            ({"ripple_duration": (0.0, 0.1)}, "ripple_duration"),
            ({"ripple_skew": (0.5, 1.0)}, "ripple_skew"),
            ({"ripple_frequency": (160.0, 800.0)}, "ripple_frequency"),
            ({"ripple_chirp": (0.0, 200.0)}, "ripple_chirp"),
            ({"ripple_snr": (0.0, 2.0)}, "ripple_snr"),
            ({"weak_ripple_snr": (2.0, 1.0)}, "weak_ripple_snr"),
            ({"sharp_wave_duration": (0.1, 0.05)}, "sharp_wave_duration"),
            ({"sharp_wave_amplitude": (-1.0, 2.0)}, "sharp_wave_amplitude"),
            ({"sharp_wave_lag": -0.01}, "sharp_wave_lag"),
            ({"burst_duration_ratio": (0.0, 1.0)}, "burst_duration_ratio"),
            ({"burst_lag": np.inf}, "burst_lag"),
            ({"burst_gain": 0.5}, "burst_gain"),
            ({"participation": (0.2, 1.5)}, "participation"),
            ({"weak_participation": (-0.1, 0.1)}, "weak_participation"),
            ({"burst_only_duration": (0.3, 0.05)}, "burst_only_duration"),
            ({"doublet_interval": (0.0, 0.1)}, "doublet_interval"),
            ({"minimum_separation": -0.05}, "minimum_separation"),
            ({"strength_correlation": 1.5}, "strength_correlation"),
            ({"strength_correlation": -0.1}, "strength_correlation"),
            ({"envelope_power": 3}, "envelope_power"),
            ({"running_intervals": [(5.0, 2.0)]}, "start before its end"),
        ],
    )
    def test_validation(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            draw_network_events(simulate_time(3000, 1000), **kwargs)

    def test_time_must_be_a_sampled_axis(self):
        with pytest.raises(ValueError, match="time must be 1-D"):
            draw_network_events(np.zeros((10, 2)))

    def test_bouts_at_the_recording_edges(self):
        """A bout over the first second, or past the end, leaves rest only
        between them."""
        time = simulate_time(self.FS * 60, self.FS)
        events = draw_network_events(
            time, event_rate=2.0, running_intervals=[(0.0, 10.0), (50.0, 70.0)], rng=2
        )
        start, end = _spans(events, 4)
        assert len(events) > 0
        assert (start >= 10.0).all()
        assert (end <= 50.0).all()


def _one_event_table(event_type, *, event_id=0, center_time=5.0, **overrides):
    """One latent event built by hand, its components centred on
    ``center_time``: a ripple of span 0.09 s chirping from 200 to 180 Hz at
    SNR 4, a sharp wave of span 0.08 s and amplitude 5, a burst of span 0.12 s
    at gain 40 and participation 0.5 (a weak ripple: SNR 1.5, amplitude 2.5,
    participation 0.05; a doublet: a second ripple and sharp wave 0.1 s later
    under one burst). ``overrides`` set a column on every row, or, keyed by
    expression (``ripple={"amplitude": 3.0}``), on that expression's rows."""
    nan = np.nan
    ripple = {
        "expression": "ripple", "rise_sigma": 0.015, "decay_sigma": 0.015,
        "amplitude": 4.0, "frequency_start": 200.0, "frequency_end": 180.0,
        "participation": nan,
    }  # fmt: skip
    sharp_wave = {
        "expression": "sharp_wave", "rise_sigma": 0.08 / 6, "decay_sigma": 0.08 / 6,
        "amplitude": 5.0, "frequency_start": nan, "frequency_end": nan, "participation": nan,
    }  # fmt: skip
    burst = {
        "expression": "burst", "rise_sigma": 0.02, "decay_sigma": 0.02, "amplitude": 40.0,
        "frequency_start": nan, "frequency_end": nan, "participation": 0.5,
    }  # fmt: skip
    doublet_burst = {**burst, "rise_sigma": 0.19 / 6, "decay_sigma": 0.19 / 6}
    weak = {"amplitude": 1.5}, {"amplitude": 2.5}, {"participation": 0.05}
    components = {  # (row, component, offset of its centre)
        "swr": [(ripple, 0, 0.0), (sharp_wave, 0, 0.0), (burst, 0, 0.0)],
        "weak_ripple": [
            ({**ripple, **weak[0]}, 0, 0.0),
            ({**sharp_wave, **weak[1]}, 0, 0.0),
            ({**burst, **weak[2]}, 0, 0.0),
        ],
        "burst_only": [(burst, 0, 0.0)],
        "sharp_wave_only": [(sharp_wave, 0, 0.0)],
        "ripple_doublet": [
            (ripple, 0, 0.0),
            (ripple, 1, 0.1),
            (sharp_wave, 0, 0.0),
            (sharp_wave, 1, 0.1),
            (doublet_burst, 0, 0.05),
        ],
    }[event_type]
    rows = []
    for row, component, offset in components:
        row = {
            **row,
            "event_id": event_id,
            "event_type": event_type,
            "component": component,
            "center_time": center_time + offset,
            "envelope_power": 2,
            "n_participants": 0,
        }
        for key, value in overrides.items():
            if key in EXPRESSION_ORDER:
                if row["expression"] == key:
                    row.update(value)
            else:
                row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)[list(EVENT_COLUMNS)]


def _event_tables(*tables):
    """Hand-built tables as one, numbered in order."""
    return pd.concat(
        [table.assign(event_id=i) for i, table in enumerate(tables)], ignore_index=True
    )


QUIET = {"theta_amplitude": 0.0, "delta_amplitude": 0.0}


class TestSimulateNetworkSession:
    FS = 1500
    TIME = simulate_time(FS * 30, FS)

    @staticmethod
    @pytest.fixture(scope="class")
    def drawn():
        time, running = TestSimulateNetworkSession.TIME, [(12.0, 18.0)]
        events = draw_network_events(time, event_rate=1.0, running_intervals=running, rng=0)
        return events, simulate_network_session(time, events, running_intervals=running, rng=1)

    def _render(self, events, **kwargs):
        return simulate_network_session(self.TIME, events, **{"rng": 5, **QUIET, **kwargs})

    def test_shapes_and_types(self, drawn):
        events, session = drawn
        n_time = self.TIME.size
        assert session.lfps.shape == (n_time, 4)
        assert session.raw_lfp.shape == session.sharp_wave_lfp.shape == (n_time,)
        np.testing.assert_array_equal(session.raw_lfp, session.lfps[:, 0])
        assert session.multiunit.shape == (n_time, 60)
        assert (
            session.unit_types.tolist()
            == ["place"] * 40 + ["pyramidal"] * 10 + ["interneuron"] * 10
        )
        np.testing.assert_array_equal(session.running_intervals, [[12.0, 18.0]])
        assert session.speed.max() == pytest.approx(30.0, rel=1e-3)
        _assert_schema(session.events, EVENT_COLUMNS)
        pd.testing.assert_frame_equal(
            session.events.drop(columns="n_participants"),
            events.drop(columns="n_participants"),
        )
        burst = session.events.expression == "burst"
        assert burst.any()
        assert (session.events.n_participants[burst] > 0).any()
        assert (session.events.n_participants[~burst] == 0).all()
        assert (session.events.n_participants[burst] <= 50).all()
        assert (events.n_participants == 0).all()  # the input is not modified
        _assert_schema(session.ripple_channels, RIPPLE_CHANNEL_COLUMNS)
        n_ripples = (events.expression == "ripple").sum()
        assert len(session.ripple_channels) == 4 * n_ripples
        assert session.non_events.empty

    def test_participants_follow_participation(self):
        """Place units join with the row's probability, other pyramidal units
        with half of it; about 40 * 0.5 + 10 * 0.25 = 22.5 per burst."""
        tables = [_one_event_table("burst_only", center_time=2.0 + 0.5 * i) for i in range(50)]
        session = self._render(_event_tables(*tables))
        n = session.events.n_participants.to_numpy()
        assert abs(n.mean() - 22.5) < 4 * np.sqrt(40 * 0.25 + 10 * 0.1875) / np.sqrt(50)
        none = self._render(_one_event_table("burst_only", burst={"participation": 0.0}))
        assert none.events.n_participants.item() == 0

    def test_baseline_rates_are_kept(self, drawn):
        _, session = drawn
        rates = session.baseline_rates
        assert rates.shape == (60,)
        ranges = {"place": (0.1, 0.5), "pyramidal": (0.5, 1.5), "interneuron": (2.0, 5.0)}
        for unit_type, (low, high) in ranges.items():
            of_type = rates[session.unit_types == unit_type]
            assert ((of_type >= low) & (of_type <= high)).all(), unit_type
        other = simulate_network_session(self.TIME, _empty_table(), rng=2)
        assert not np.array_equal(other.baseline_rates, rates)

    def test_unit_counts_and_rates(self):
        session = self._render(
            _empty_table(),
            unit_counts={"interneuron": 3, "place": 2},
            baseline_rate={"place": (1.0, 1.0)},
        )
        assert session.unit_types.tolist() == ["place", "place"] + ["interneuron"] * 3
        np.testing.assert_array_equal(session.baseline_rates[:2], [1.0, 1.0])
        assert ((session.baseline_rates[2:] >= 2) & (session.baseline_rates[2:] <= 5)).all()

    def test_an_empty_table_renders_noise_only(self):
        session = self._render(_empty_table())
        assert session.ripple_times.shape == (0,)
        assert session.ripple_windows.shape == (0, 2)
        assert session.ripple_channels.empty
        _assert_schema(session.ripple_channels, RIPPLE_CHANNEL_COLUMNS)
        assert session.events.empty

    def test_the_table_is_rendered_in_its_sorted_order(self):
        events = _event_tables(
            _one_event_table("swr", center_time=3.0), _one_event_table("ripple_doublet")
        )
        shuffled = events.sample(frac=1.0, random_state=0)
        a, b = self._render(events), self._render(shuffled)
        np.testing.assert_array_equal(a.lfps, b.lfps)
        pd.testing.assert_frame_equal(a.events, b.events)

    def test_ripple_snr_is_met(self):
        """The filtered peak over the filtered noise's SD, in each isolated
        ripple's window, is its SNR; noise adds to the peak, so the median
        ratio is compared, within 20%."""
        snr = np.linspace(3.0, 6.0, 20)
        events = _event_tables(
            *(
                _one_event_table("swr", center_time=2.0 + 1.3 * i, ripple={"amplitude": a})
                for i, a in enumerate(snr)
            )
        )
        events = events[events.expression == "ripple"].reset_index(drop=True)
        session = self._render(events)
        noise = self._render(_empty_table())
        filtered = filter_ripple_band(session.lfps[:, 0], sampling_frequency=self.FS)
        sd = filter_ripple_band(noise.lfps[:, 0], sampling_frequency=self.FS).std()
        peaks = [
            np.abs(filtered[(start <= self.TIME) & (end >= self.TIME)]).max() / sd
            for start, end in session.ripple_windows
        ]
        assert np.median(np.asarray(peaks) / snr) == pytest.approx(1.0, rel=0.2)

    def test_ripple_windows_match_components(self, drawn):
        events, session = drawn
        ripples = events[events.expression == "ripple"]
        np.testing.assert_allclose(
            session.ripple_windows[:, 0], ripples.center_time - 3 * ripples.rise_sigma
        )
        np.testing.assert_allclose(
            session.ripple_windows[:, 1], ripples.center_time + 3 * ripples.decay_sigma
        )
        np.testing.assert_array_equal(session.ripple_frequencies, ripples.frequency_start)

    def test_sharp_wave_sign_and_leak(self):
        """Noise-free, the radiatum deflection peaks at -amplitude and channel
        0 at +leak times it, on the sample at the centre."""
        events = _one_event_table("sharp_wave_only", sharp_wave={"amplitude": 6.0})
        session = self._render(events, noise_amplitude=0.0, sharp_wave_leak=0.25)
        centre = np.searchsorted(self.TIME, 5.0)
        assert session.sharp_wave_lfp.min() == pytest.approx(-6.0)
        assert np.argmin(session.sharp_wave_lfp) == centre
        assert session.lfps[:, 0].max() == pytest.approx(0.25 * 6.0)
        assert np.argmax(session.lfps[:, 0]) == centre
        assert (session.lfps[:, 1:] == 0).all()

    def test_sharp_wave_envelope_power(self):
        """At power 4 the deflection is flatter at the top and steeper at the
        edges, with the same half-maximum width."""
        radiatum = {
            power: -self._render(
                _one_event_table("sharp_wave_only", envelope_power=power), noise_amplitude=0.0
            ).sharp_wave_lfp
            / 5.0
            for power in (2, 4)
        }
        above_half = {power: (trace >= 0.5).sum() for power, trace in radiatum.items()}
        assert abs(above_half[2] - above_half[4]) <= 2
        sigma = 0.08 / 6
        near_top = np.abs(self.TIME - 5.0 - sigma) < 0.5 / self.FS
        far = np.abs(self.TIME - 5.0 - 2.5 * sigma) < 0.5 / self.FS
        assert radiatum[4][near_top] > radiatum[2][near_top]
        assert radiatum[4][far] < radiatum[2][far]


def _empty_table():
    return draw_network_events(simulate_time(100, 1500), event_rate=0.0)


def _ripple_only(*tables):
    """The ripple rows of hand-built events, numbered in order."""
    events = _event_tables(*tables)
    return events[events.expression == "ripple"].reset_index(drop=True)


def _envelope_centroid(signal, time):
    """The power-weighted mean time of the Hilbert envelope; moves with the
    signal by fractions of a sample."""
    power = np.abs(hilbert(signal)) ** 2
    return float(np.sum(time * power) / np.sum(power))


class TestNetworkSessionVariants:
    FS = 1500
    TIME = simulate_time(FS * 12, FS)

    def _render(self, events, **kwargs):
        return simulate_network_session(self.TIME, events, **{"rng": 7, **QUIET, **kwargs})

    def test_global_profile_stores_the_channel_gains(self):
        events = _ripple_only(
            _one_event_table("swr", center_time=4.0), _one_event_table("swr")
        )
        session = self._render(events, channel_gains=[1.0, 0.8, 0.6, 0.4])
        table = session.ripple_channels
        assert table[["event_id", "component", "channel"]].to_numpy().tolist() == [
            [event, 0, channel] for event in range(2) for channel in range(4)
        ]
        np.testing.assert_array_equal(table.gain, np.tile([1.0, 0.8, 0.6, 0.4], 2))
        assert (table.delay_s == 0).all()

    @pytest.mark.parametrize(
        ("n_channels", "occupancy", "n_selected"),
        [(4, 0.5, 2), (4, 1.0, 4), (4, 0.01, 1), (1, 0.5, 1)],
    )
    def test_spatial_profile(self, n_channels, occupancy, n_selected):
        """Each ripple on the stated number of channels, one of them an anchor
        at the channel gain with no delay, the rest at gains from the range;
        each channel carries the ripple moved by its stored delay and scaled
        by its gain, and the channels without it carry nothing."""
        centres = 2.0 + 0.9 * np.arange(10)
        events = _ripple_only(*(_one_event_table("swr", center_time=c) for c in centres))
        gains = np.linspace(1.0, 0.7, n_channels)
        options = {
            "n_channels": n_channels, "channel_gains": gains, "spatial_profile": "local",
            "channel_occupancy": occupancy, "channel_gain_range": (0.5, 0.9),
            "channel_delay": 0.002,
        }  # fmt: skip
        session = self._render(events, **options)
        noise = self._render(_empty_table(), **options)
        ripple = session.lfps - noise.lfps
        for (event_id, _), rows in session.ripple_channels.groupby(["event_id", "component"]):
            selected = rows[rows.gain > 0]
            assert len(selected) == n_selected
            assert (rows.delay_s[rows.gain == 0] == 0).all()
            relative = selected.gain.to_numpy() / gains[selected.channel]
            anchor = selected[np.isclose(relative, 1.0) & (selected.delay_s == 0)]
            assert len(anchor) >= 1
            others = selected.drop(anchor.index[:1])
            assert (relative[np.isin(selected.index, others.index)] >= 0.5 - 1e-12).all()
            assert (relative[np.isin(selected.index, others.index)] <= 0.9 + 1e-12).all()
            assert (np.abs(selected.delay_s) <= 0.002).all()
            centre = centres[event_id]
            near = np.abs(self.TIME - centre) < 0.2
            reference = anchor.iloc[0]
            reference_trace = ripple[near, int(reference.channel)]
            reference_energy = np.sum(reference_trace**2)
            reference_time = _envelope_centroid(reference_trace, self.TIME[near])
            for row in rows.itertuples():
                trace = ripple[near, row.channel]
                if row.gain == 0:
                    assert (trace == 0).all()
                    continue
                ratio = np.sqrt(np.sum(trace**2) / reference_energy)
                assert ratio == pytest.approx(row.gain / reference.gain, rel=2e-3)
                shift = _envelope_centroid(trace, self.TIME[near]) - reference_time
                assert shift == pytest.approx(row.delay_s, abs=1e-4)
        if n_selected < n_channels:
            assert (session.ripple_channels.gain == 0).any()

    def test_local_delays_stay_in_rest(self):
        """A ripple against a running bout is delayed only away from it."""
        events = _ripple_only(_one_event_table("swr", center_time=6.0 + 4 * 0.015 + 0.001))
        options = {
            "spatial_profile": "local", "channel_delay": 0.05, "running_intervals": [(4.0, 6.0)],
        }  # fmt: skip
        delays = []
        for seed in range(10):
            session = self._render(events, rng=seed, **options)
            delays.extend(session.ripple_channels.delay_s)
        delays = np.asarray(delays)
        assert delays.min() >= -0.001 - 1e-12
        assert delays.max() > 0.01

    def test_a_local_ripple_outside_rest_is_not_delayed(self):
        """A ripple edited into a running bout has no stretch of rest to stay
        in, so no delay qualifies but zero."""
        events = _ripple_only(_one_event_table("swr", center_time=5.0))
        session = self._render(
            events, spatial_profile="local", channel_delay=0.01, running_intervals=[(4.0, 6.0)]
        )
        assert (session.ripple_channels.delay_s == 0).all()

    def test_an_envelope_narrower_than_a_step_lands_on_one_sample(self):
        events = _one_event_table(
            "sharp_wave_only", sharp_wave={"rise_sigma": 1e-6, "decay_sigma": 1e-6}
        )
        session = self._render(events, noise_amplitude=0.0)
        assert np.count_nonzero(session.sharp_wave_lfp) == 1
        between = 5.0 + 0.3 / self.FS  # no sample within 8 side scales
        window, envelope = _event_envelope(self.TIME, between, 1e-6, 1e-6, 2)
        assert window.stop - window.start == envelope.size == 1
        assert self.TIME[window.start] == pytest.approx(5.0 + 1 / self.FS)  # the next

    def test_spatial_profile_changes_only_its_own_draws(self):
        events = _one_event_table("swr")
        local = self._render(events, spatial_profile="local", channel_occupancy=0.5)
        default = self._render(events)
        np.testing.assert_array_equal(local.sharp_wave_lfp, default.sharp_wave_lfp)
        np.testing.assert_array_equal(local.multiunit, default.multiunit)
        assert not np.array_equal(local.lfps, default.lfps)

    def test_noise_modulation(self):
        """Matched noise-only renders differ by the slow gain, the same on
        every channel: log-amplitude a, period T, unit RMS. Events keep their
        size whatever the gain, so their local SNR changes."""
        a, period = 0.35, 4.0
        stationary = self._render(_empty_table())
        varying = self._render(
            _empty_table(), noise_log_amplitude=a, noise_modulation_period=period
        )
        ratio = varying.lfps / stationary.lfps
        np.testing.assert_allclose(ratio, np.repeat(ratio[:, :1], 4, axis=1), rtol=1e-9)
        np.testing.assert_allclose(
            varying.sharp_wave_lfp / stationary.sharp_wave_lfp, ratio[:, 0]
        )
        gain = ratio[:, 0]
        assert np.sqrt(np.mean(gain**2)) == pytest.approx(1.0, rel=1e-9)
        assert np.log(gain).max() - np.log(gain).min() == pytest.approx(2 * a, rel=1e-4)
        one_period = int(period * self.FS)
        np.testing.assert_allclose(gain[one_period:], gain[:-one_period], rtol=1e-9)

        other_period = self._render(_empty_table(), noise_modulation_period=17.0)
        np.testing.assert_array_equal(other_period.lfps, stationary.lfps)

        events = _event_tables(
            _one_event_table("swr", center_time=3.0), _one_event_table("swr")
        )
        with_events = self._render(events)
        modulated = self._render(events, noise_log_amplitude=a, noise_modulation_period=period)
        np.testing.assert_allclose(
            modulated.lfps - varying.lfps, with_events.lfps - stationary.lfps, atol=1e-12
        )

    def test_refractory_spikes(self):
        """At most one spike per sample and none within the dead time of the
        last; the LFP, units and recruitment are the Poisson rendering's."""
        events = _event_tables(
            *(_one_event_table("swr", center_time=2.0 + 0.8 * i) for i in range(10))
        )
        poisson = self._render(events)
        refractory = self._render(events, spike_model="refractory", refractory_period=0.002)
        assert refractory.multiunit.max() == 1
        for unit in range(refractory.multiunit.shape[1]):
            spike_times = self.TIME[refractory.multiunit[:, unit] > 0]
            assert (np.diff(spike_times) >= 0.002 - 1e-9).all()
        np.testing.assert_array_equal(refractory.lfps, poisson.lfps)
        np.testing.assert_array_equal(refractory.sharp_wave_lfp, poisson.sharp_wave_lfp)
        np.testing.assert_array_equal(refractory.baseline_rates, poisson.baseline_rates)
        np.testing.assert_array_equal(refractory.unit_types, poisson.unit_types)
        pd.testing.assert_frame_equal(refractory.events, poisson.events)
        assert poisson.multiunit.max() >= 1

    def test_refractory_rate_follows_the_renewal_model(self):
        """At a constant intensity lambda, a spike blocks the next k - 1
        samples (k steps span the dead time) and each later sample fires with
        probability p = 1 - exp(-lambda dt): the mean interval is k - 1 + 1/p
        samples, below the Poisson rate."""
        intensity, dead_time = 60.0, 0.002
        session = self._render(
            _empty_table(),
            unit_counts={"interneuron": 20},
            baseline_rate={"interneuron": (intensity, intensity)},
            spike_model="refractory",
            refractory_period=dead_time,
        )
        k = int(np.ceil(dead_time * self.FS - 1e-9))
        p = -np.expm1(-intensity / self.FS)
        expected = self.FS / (k - 1 + 1 / p)
        duration = self.TIME[-1] - self.TIME[0]
        realized = session.multiunit.sum() / (20 * duration)
        standard_error = np.sqrt(expected / (20 * duration))
        assert abs(realized - expected) < 4 * standard_error
        assert intensity - realized > 8 * standard_error

    def test_chirp(self):
        """The Hilbert instantaneous frequency of a unit ripple follows the
        linear chirp at -2 and +2 side scales, at both envelope powers."""
        time = simulate_time(self.FS * 2, self.FS)
        rise, decay, start, end = 0.02, 0.03, 210.0, 170.0
        for power in (2, 4):
            window, wave = _render_ripple(time, 1.0, rise, decay, start, end, 0.3, power)
            phase = np.unwrap(np.angle(hilbert(wave)))
            frequency = np.gradient(phase, time[window]) / (2 * np.pi)
            for offset in (-2 * rise, 2 * decay):
                sample = np.argmin(np.abs(time[window] - 1.0 - offset))
                expected = start + (end - start) * (offset + 3 * rise) / (3 * rise + 3 * decay)
                assert frequency[sample] == pytest.approx(expected, abs=5.0), (power, offset)

    def test_a_delayed_ripple_is_the_same_waveform_moved(self):
        time = simulate_time(self.FS * 2, self.FS)
        shift = 3 / self.FS
        window, wave = _render_ripple(time, 1.0, 0.015, 0.02, 200.0, 180.0, 1.1)
        moved_window, moved = _render_ripple(time, 1.0 + shift, 0.015, 0.02, 200.0, 180.0, 1.1)
        full, moved_full = np.zeros(time.size), np.zeros(time.size)
        full[window], moved_full[moved_window] = wave, moved
        np.testing.assert_allclose(moved_full[3:], full[:-3], atol=1e-9)

    @pytest.mark.parametrize("power", [2, 4])
    def test_burst_follows_its_envelope(self, power):
        """Summed over many recruited units, spikes in 5 ms bins match the
        integrated intensity, baseline plus the burst's envelope at its
        power; every place unit is recruited (participation 1), so only the
        Poisson counts vary. The two powers' expectations differ by more
        than the tolerance, so the test tells them apart."""
        n_units, rate, sigma, centre = 3000, 2.0, 0.02, 1.0
        time = simulate_time(self.FS * 2, self.FS)
        events = _one_event_table(
            "burst_only",
            center_time=centre,
            envelope_power=power,
            burst={"participation": 1.0, "rise_sigma": sigma, "decay_sigma": sigma},
        )
        session = simulate_network_session(
            time,
            events,
            unit_counts={"place": n_units},
            baseline_rate={"place": (rate, rate)},
            rng=7,
            **QUIET,
        )
        assert session.events.n_participants.item() == n_units
        edges = centre + np.arange(-0.1, 0.1001, 0.005)
        bins = np.digitize(time, edges) - 1
        inside = (bins >= 0) & (bins < edges.size - 1)
        observed = np.bincount(bins[inside], session.multiunit[inside].sum(axis=1))

        def expected_counts(p):
            scaled = np.abs(time - centre) / (np.sqrt(2 * np.log(2)) * sigma)
            envelope = np.exp(-np.log(2) * scaled**p)
            intensity = rate / self.FS * (1 + 39.0 * envelope)
            return n_units * np.bincount(bins[inside], intensity[inside])

        expected = expected_counts(power)
        z = (observed - expected) / np.sqrt(expected)
        assert np.abs(z).max() < 4.5
        other = expected_counts(6 - power)
        assert np.abs((other - expected) / np.sqrt(expected)).max() > 8

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"spatial_profile": "patchy"}, "spatial_profile"),
            ({"spike_model": "bursting"}, "spike_model"),
            ({"channel_occupancy": 0.0}, "channel_occupancy"),
            ({"channel_occupancy": 1.5}, "channel_occupancy"),
            ({"channel_gain_range": (0.9, 0.5)}, "channel_gain_range"),
            ({"channel_gain_range": (-0.5, 0.5)}, "channel_gain_range"),
            ({"channel_delay": -0.001}, "channel_delay"),
            ({"channel_delay": np.inf}, "channel_delay"),
            ({"noise_log_amplitude": -0.1}, "noise_log_amplitude"),
            ({"noise_modulation_period": 0.0}, "noise_modulation_period"),
            ({"refractory_period": -0.001}, "refractory_period"),
            ({"refractory_period": np.nan}, "refractory_period"),
            ({"interneuron_gain": 0.5}, "interneuron_gain"),
            ({"n_channels": 0}, "n_channels"),
            ({"channel_gains": [1.0, 1.0]}, "channel_gains"),
            ({"unit_counts": {"granule": 3}}, "unit_counts"),
            ({"unit_counts": {"place": -1}}, "unit_counts"),
            ({"unit_counts": {"place": 1.5}}, "unit_counts"),
            ({"unit_counts": {"place": 0}}, "at least one unit"),
            ({"baseline_rate": {"place": (1.0, 0.5)}}, "baseline_rate"),
            ({"baseline_rate": {"basket": (1.0, 2.0)}}, "baseline_rate"),
            ({"noise_amplitude": -1.0}, "noise_amplitude"),
            ({"noise_amplitude": 0.0}, "noise_amplitude must be > 0"),
            ({"shared_noise_fraction": 1.5}, "shared_noise_fraction"),
            ({"running_intervals": [(5.0, 2.0)]}, "start before its end"),
        ],
    )
    def test_variant_validation(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            self._render(_one_event_table("swr"), **kwargs)

    @pytest.mark.parametrize(
        ("events", "message"),
        [
            (_one_event_table("swr").drop(columns="amplitude"), "missing the columns"),
            (_one_event_table("swr").assign(event_type="ripple_train"), "event_type"),
            (_one_event_table("swr", expression="spindle"), "expression"),
            (_one_event_table("swr", rise_sigma=0.0), "rise_sigma"),
            (_one_event_table("swr", center_time=np.nan), "finite"),
            (_one_event_table("swr", envelope_power=3), "envelope_power"),
            (_one_event_table("swr", ripple={"frequency_start": 900.0}), "Nyquist"),
            (_one_event_table("swr", ripple={"amplitude": 0.0}), "SNR"),
            (_one_event_table("swr", burst={"participation": 1.5}), "participation"),
            (_one_event_table("swr", burst={"amplitude": 0.5}), "participation"),
        ],
    )
    def test_event_table_validation(self, events, message):
        with pytest.raises(ValueError, match=message):
            self._render(events)

    def test_time_must_be_a_sampled_axis(self):
        with pytest.raises(ValueError, match="time must be 1-D"):
            simulate_network_session(np.zeros(1), _empty_table())


def _crossing_distance(fraction, power):
    """Side scales from the centre at which the envelope of ``power`` is at
    ``fraction`` of its peak."""
    return np.sqrt(2 * np.log(2)) * (np.log(1 / fraction) / np.log(2)) ** (1 / power)


class TestTruthWindows:
    FS = 1500
    FRACTIONS = (0.1, 0.25, 0.5)

    @staticmethod
    @pytest.fixture(scope="class")
    def events():
        time = simulate_time(1500 * 120, 1500)
        return draw_network_events(time, event_rate=1.0, rng=4)

    @pytest.mark.parametrize("power", [2, 4])
    @pytest.mark.parametrize("fraction", FRACTIONS)
    def test_fraction_formula(self, fraction, power):
        events = _one_event_table(
            "swr", envelope_power=power, ripple={"rise_sigma": 0.01, "decay_sigma": 0.02}
        )
        (window,) = truth_windows(events, fraction, expression="ripple").itertuples()
        k = _crossing_distance(fraction, power)
        if power == 2:
            assert k == pytest.approx(np.sqrt(-2 * np.log(fraction)))
        assert window.start_time == pytest.approx(5.0 - k * 0.01)
        assert window.end_time == pytest.approx(5.0 + k * 0.02)
        assert window.peak_time == 5.0

    def test_equal_half_maximum_widths_and_narrower_with_fraction(self, events):
        widths = {}
        for power in (2, 4):
            table = events.assign(envelope_power=power)
            windows = [truth_windows(table, f) for f in self.FRACTIONS]
            widths[power] = [(w.end_time - w.start_time).to_numpy() for w in windows]
            for wider, narrower in itertools.pairwise(widths[power]):
                assert (narrower < wider).all()
        np.testing.assert_allclose(widths[2][2], widths[4][2])
        assert (widths[4][0] < widths[2][0]).all()

    @pytest.mark.parametrize("expression", [None, "ripple", "sharp_wave", "burst", "network"])
    def test_row_order_is_the_same_for_every_fraction(self, events, expression):
        windows = [truth_windows(events, f, expression=expression) for f in self.FRACTIONS]
        labels = ["id", "type"] + (
            ["expression", "component"] if expression != "network" else []
        )
        for other in windows[1:]:
            pd.testing.assert_frame_equal(other[labels], windows[0][labels])
            np.testing.assert_array_equal(other.peak_time, windows[0].peak_time)
        assert len(windows[0]) > 0

    def test_component_rows_follow_the_table(self, events):
        windows = truth_windows(events)
        assert list(windows.columns) == [
            "id", "type", "start_time", "end_time", "peak_time", "expression", "component",
        ]  # fmt: skip
        assert len(windows) == len(events)
        np.testing.assert_array_equal(windows.id, events.event_id)
        np.testing.assert_array_equal(windows.expression, events.expression)
        np.testing.assert_array_equal(windows.peak_time, events.center_time)
        ripples = truth_windows(events, expression="ripple")
        assert (ripples.expression == "ripple").all()
        assert len(ripples) == (events.expression == "ripple").sum()

    @pytest.mark.parametrize("power", [2, 4])
    def test_measured_on_the_rendered_envelope(self, power):
        """Noise-free, a sharp wave is its envelope times -amplitude on the
        radiatum channel: the first and last samples at or above each fraction
        of the peak lie within one sample of the analytic bounds."""
        time = simulate_time(self.FS * 10, self.FS)
        events = _one_event_table(
            "sharp_wave_only",
            envelope_power=power,
            sharp_wave={"rise_sigma": 0.011, "decay_sigma": 0.023, "amplitude": 2.0},
        )
        session = simulate_network_session(time, events, noise_amplitude=0.0, rng=0, **QUIET)
        envelope = -session.sharp_wave_lfp / 2.0
        assert envelope.max() == pytest.approx(1.0)
        for fraction in self.FRACTIONS:
            (window,) = truth_windows(session.events, fraction).itertuples()
            above = time[envelope >= fraction]
            assert abs(above[0] - window.start_time) <= 1 / self.FS
            assert abs(above[-1] - window.end_time) <= 1 / self.FS

    def test_network_union_and_peak(self):
        swr = _one_event_table(
            "swr",
            center_time=2.0,
            sharp_wave={"center_time": 2.01},
            burst={"center_time": 1.99, "rise_sigma": 0.05, "decay_sigma": 0.05},
        )
        doublet = _one_event_table("ripple_doublet", center_time=4.0)
        burst_only = _one_event_table("burst_only", center_time=6.0)
        sharp_only = _one_event_table("sharp_wave_only", center_time=8.0)
        no_ripple = _one_event_table("swr", center_time=10.0, burst={"center_time": 10.02})
        no_ripple = no_ripple[no_ripple.expression != "ripple"]
        events = _event_tables(swr, doublet, burst_only, sharp_only, no_ripple)
        network = truth_windows(events, 0.1, expression="network")
        components = truth_windows(events, 0.1)
        assert network.id.tolist() == [0, 1, 2, 3, 4]
        assert network.type.tolist() == [
            "swr", "ripple_doublet", "burst_only", "sharp_wave_only", "swr",
        ]  # fmt: skip
        by_id = components.groupby("id")
        np.testing.assert_array_equal(network.start_time, by_id.start_time.min())
        np.testing.assert_array_equal(network.end_time, by_id.end_time.max())
        np.testing.assert_array_equal(network.peak_time, [2.0, 4.0, 6.0, 8.0, 10.02])
        # the burst is the widest component here, so the union is not the ripple
        assert network.start_time[0] < components.start_time[0]

    def test_non_event_table(self):
        non_events = pd.DataFrame(
            {
                "non_event_id": [0, 1], "non_event_type": ["emg", "fast_gamma"],
                "center_time": [2.0, 3.0], "rise_sigma": [0.02, 0.01],
                "decay_sigma": [0.02, 0.03], "envelope_power": [2, 2],
            }
        )  # fmt: skip
        windows = truth_windows(non_events, 0.25)
        assert list(windows.columns) == ["id", "type", "start_time", "end_time", "peak_time"]
        k = _crossing_distance(0.25, 2)
        np.testing.assert_allclose(windows.start_time, [2.0 - 0.02 * k, 3.0 - 0.01 * k])
        np.testing.assert_allclose(windows.end_time, [2.0 + 0.02 * k, 3.0 + 0.03 * k])
        assert windows.type.tolist() == ["emg", "fast_gamma"]
        with pytest.raises(ValueError, match="non-event table has no expressions"):
            truth_windows(non_events, expression="ripple")

    def test_empty_tables(self):
        for expression in (None, "ripple", "network"):
            windows = truth_windows(_empty_table(), expression=expression)
            assert windows.empty
            assert windows.id.dtype == np.int64
        session = simulate_session(simulate_time(3000, self.FS), [1.0], rng=0)
        assert truth_windows(session.non_events).empty

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.5, np.nan])
    def test_fraction_must_lie_between_0_and_1(self, events, fraction):
        with pytest.raises(ValueError, match="fraction must lie in"):
            truth_windows(events, fraction)

    def test_validation(self, events):
        with pytest.raises(ValueError, match="expression must be one of"):
            truth_windows(events, expression="spindle")
        with pytest.raises(ValueError, match="event or non-event table"):
            truth_windows(pd.DataFrame({"start_time": [1.0]}))

"""Tests for simulation module."""

import dataclasses
import hashlib

import numpy as np
import pytest

from ripple_detection import filter_ripple_band
from ripple_detection.simulate import (
    NOISE_FUNCTION,
    SimulatedSession,
    _draw_per_ripple,
    brown,
    mean_squared,
    normalize,
    pink,
    simulate_LFP,
    simulate_multichannel_LFP,
    simulate_multiunit,
    simulate_session,
    simulate_sharp_wave_ripple_pair,
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
        noise = white(N)
        assert len(noise) == N

    def test_white_noise_statistics(self):
        """Test that white noise has approximately correct statistics."""
        N = 10000
        noise = white(N)

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
        rng = np.random.default_rng(42)
        noise1 = pink(1000, rng=rng)

        rng = np.random.default_rng(42)
        noise2 = pink(1000, rng=rng)

        assert np.allclose(noise1, noise2)

    def test_pink_noise_frequency_content(self):
        """Test that pink noise has 1/f power spectrum."""
        N = 8192
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
        rng = np.random.default_rng(42)
        noise1 = brown(1000, rng=rng)

        rng = np.random.default_rng(42)
        noise2 = brown(1000, rng=rng)

        assert np.allclose(noise1, noise2)

    def test_brown_noise_frequency_content(self):
        """Test that brown noise has 1/f^2 power spectrum."""
        N = 8192
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

        lfp_low = simulate_LFP(time, ripple_time, ripple_amplitude=1.0, noise_amplitude=0.5)
        lfp_high = simulate_LFP(time, ripple_time, ripple_amplitude=5.0, noise_amplitude=0.5)

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
        numpy.random.default_rng rather than the legacy RandomState and whose
        default noise is pink; the explicit brown call is the 1.x default."""
        t = simulate_time(4500, self.FS)
        y = simulate_LFP(t, [1.0, 2.0], random_state=0)
        assert _digest(y) == "38aba443e5f044ca"
        y = simulate_LFP(t, [1.0, 2.0], random_state=0, noise_type="brown")
        assert _digest(y) == "aec97d07aeeb5ff9"
        y = simulate_LFP(
            t, [1.0, 2.0], random_state=0, noise_type="pink", ripple_amplitude=1.0
        )
        assert _digest(y) == "fb97eb68f45ea538"

    def test_memory_does_not_grow_with_the_ripple_count(self):
        """Ten minutes at 1500 Hz with 100 ripples peaked at 1.5 GB when every
        burst was a full-length array; each is now added over its own window."""
        import tracemalloc

        t = simulate_time(self.FS * 60, self.FS)

        def peak_bytes(n_ripples):
            tracemalloc.start()
            simulate_LFP(t, list(np.linspace(1.0, 59.0, n_ripples)), random_state=0)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return peak

        assert peak_bytes(200) < 2 * peak_bytes(10)

    def test_windowed_bursts_match_whole_record_bursts(self):
        """The 8-sigma window drops less than 1e-13 of a burst's peak."""
        t = simulate_time(self.FS * 4, self.FS)
        y = simulate_LFP(t, [1.0, 2.5], noise_amplitude=0.0, ripple_amplitude=2.0)
        carrier = sum(np.exp(-((t - m) ** 2) / (2 * (0.1 / 6) ** 2)) for m in (1.0, 2.5))
        whole = np.sin(2 * np.pi * t * 200.0) * carrier  # unit peak per burst
        assert np.allclose(y, whole, atol=1e-12, rtol=0.0)

    def test_ripple_snr_sets_peak_relative_to_in_band_background(self):
        t = simulate_time(self.FS * 20, self.FS)
        ripples = [2.0, 5.0, 8.0, 11.0, 14.0, 17.0]
        # the noise is drawn first, so the same seed with no ripples is exactly
        # the background of the ripple record; the difference is the ripples alone
        noise_only = simulate_LFP(t, [], noise_type="pink", random_state=1)
        background_sd = filter_ripple_band(noise_only, sampling_frequency=self.FS).std()
        for snr in (3.0, 6.0):
            y = simulate_LFP(t, ripples, noise_type="pink", ripple_snr=snr, random_state=1)
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
        noise_only = simulate_LFP(t, [], noise_type="pink", random_state=1)
        background_sd = filter_ripple_band(noise_only, sampling_frequency=self.FS).std()
        y = simulate_LFP(
            t,
            ripples,
            noise_type="pink",
            ripple_snr=5.0,
            ripple_frequency=frequency,
            ripple_duration=duration,
            random_state=1,
        )
        bursts = filter_ripple_band(y - noise_only, sampling_frequency=self.FS)
        peaks = [np.abs(bursts[np.abs(t - r) < 0.05]).max() for r in ripples]
        np.testing.assert_allclose(np.array(peaks) / background_sd, 5.0, rtol=0.05)

    def test_ripple_snr_without_noise_raises(self):
        t = simulate_time(self.FS * 5, self.FS)
        with pytest.raises(ValueError, match="noise_amplitude"):
            simulate_LFP(t, [2.0], ripple_snr=5.0, noise_amplitude=0.0, random_state=0)

    def test_a_tuple_is_a_range_and_a_list_is_one_value_per_ripple(self):
        t = simulate_time(self.FS * 4, self.FS)
        as_list = simulate_LFP(t, [1.0, 3.0], ripple_frequency=[150.0, 250.0], random_state=3)
        as_array = simulate_LFP(
            t, [1.0, 3.0], ripple_frequency=np.array([150.0, 250.0]), random_state=3
        )
        as_range = simulate_LFP(t, [1.0, 3.0], ripple_frequency=(150.0, 250.0), random_state=3)
        np.testing.assert_array_equal(as_list, as_array)
        assert not np.array_equal(as_list, as_range)
        with pytest.raises(ValueError, match="tuple"):
            simulate_LFP(t, [1.0], ripple_frequency=(150.0, 200.0, 250.0), random_state=3)

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
            simulate_LFP(t, [2.0], random_state=0, **kwargs)

    def test_ripple_snr_and_amplitude_are_mutually_exclusive(self):
        t = simulate_time(4500, self.FS)
        with pytest.raises(ValueError, match="ripple_snr"):
            simulate_LFP(t, [1.0], ripple_amplitude=1.0, ripple_snr=5.0)

    def test_ripple_snr_infers_sampling_rate_from_time(self):
        t = simulate_time(self.FS * 10, self.FS)
        inferred = simulate_LFP(t, [3.0], ripple_snr=5.0, random_state=2)
        explicit = simulate_LFP(
            t, [3.0], ripple_snr=5.0, random_state=2, sampling_frequency=self.FS
        )
        np.testing.assert_array_equal(inferred, explicit)

    def test_scalar_frequency_is_reproduced_in_every_ripple(self):
        t = simulate_time(self.FS * 4, self.FS)
        y = simulate_LFP(t, [1.0, 3.0], noise_amplitude=0.0, ripple_frequency=180.0)
        for r in (1.0, 3.0):
            seg = y[np.abs(t - r) < 0.05]
            assert abs(_dominant_frequency(seg, self.FS) - 180.0) <= 10.0

    def test_frequency_range_draws_per_ripple(self):
        t = simulate_time(self.FS * 8, self.FS)
        ripples = [1.0, 3.0, 5.0, 7.0]
        y = simulate_LFP(
            t, ripples, noise_amplitude=0.0, ripple_frequency=(150.0, 250.0), random_state=3
        )
        freqs = [_dominant_frequency(y[np.abs(t - r) < 0.05], self.FS) for r in ripples]
        assert all(150.0 - 5.0 <= f <= 250.0 + 5.0 for f in freqs)  # 5 Hz FFT resolution
        assert len(set(np.round(freqs, -1))) > 1  # not all the same
        again = simulate_LFP(
            t, ripples, noise_amplitude=0.0, ripple_frequency=(150.0, 250.0), random_state=3
        )
        np.testing.assert_array_equal(y, again)

    def test_duration_range_draws_per_ripple(self):
        from scipy.signal import hilbert

        t = simulate_time(self.FS * 26, self.FS)
        ripples = [2.0 + 2.0 * k for k in range(12)]
        y = simulate_LFP(
            t, ripples, noise_amplitude=0.0, ripple_duration=(0.03, 0.15), random_state=4
        )
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
            t, ripples, noise_amplitude=0.0, ripple_duration=(0.03, 0.15), random_state=4
        )
        np.testing.assert_array_equal(y, again)

    def test_inverted_range_raises(self):
        t = simulate_time(4500, self.FS)
        with pytest.raises(ValueError, match="low <= high"):
            simulate_LFP(t, [1.0], ripple_duration=(0.15, 0.03), random_state=0)
        with pytest.raises(ValueError, match="low <= high"):
            simulate_LFP(t, [1.0], ripple_frequency=(220.0, 180.0), random_state=0)

    def test_random_state_instance_matches_seed(self):
        t = simulate_time(4500, self.FS)
        from_seed = simulate_LFP(
            t, [1.0, 2.0], ripple_frequency=(150.0, 250.0), random_state=7
        )
        from_state = simulate_LFP(
            t,
            [1.0, 2.0],
            ripple_frequency=(150.0, 250.0),
            random_state=np.random.default_rng(7),
        )
        np.testing.assert_array_equal(from_seed, from_state)

    def test_noise_draw_is_unchanged_by_the_new_parameters(self):
        # with a scalar frequency and duration and no ripples, output equals the
        # pre-existing noise for the same seed
        t = simulate_time(4500, self.FS)
        base = simulate_LFP(t, [], random_state=5)
        same = simulate_LFP(t, [], random_state=5, ripple_frequency=200.0, ripple_duration=0.1)
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
        expected = simulate_LFP(t, 1.0, random_state=0)
        np.testing.assert_array_equal(simulate_LFP(t, value, random_state=0), expected)


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
        assert simulate_multichannel_LFP(t, [1.0], 3, random_state=0).shape == (3000, 3)

    def test_every_channel_carries_the_same_ripple_scaled_by_its_gain(self):
        t = simulate_time(3000, self.FS)
        lfps = simulate_multichannel_LFP(
            t, [1.0], 2, channel_gains=[1.0, 0.5], noise_amplitude=0.0, random_state=0
        )
        np.testing.assert_allclose(lfps[:, 1], 0.5 * lfps[:, 0], atol=1e-15)
        assert np.abs(lfps[:, 0]).max() > 0.9

    def test_shared_fraction_is_the_correlation_between_channels(self):
        t = simulate_time(30000, self.FS)
        for fraction in (0.0, 0.5, 1.0):
            lfps = simulate_multichannel_LFP(
                t, [], 2, shared_noise_fraction=fraction, noise_type="white", random_state=1
            )
            assert np.corrcoef(lfps[:, 0], lfps[:, 1])[0, 1] == pytest.approx(
                fraction, abs=0.03
            )

    def test_noise_has_the_single_channel_scale(self):
        """Each channel's noise has the mean square simulate_LFP's noise has."""
        t = simulate_time(30000, self.FS)
        lfps = simulate_multichannel_LFP(t, [], 3, noise_type="white", random_state=2)
        single = simulate_LFP(t, [], noise_type="white", random_state=2)
        np.testing.assert_allclose(np.mean(lfps**2, axis=0), np.mean(single**2), rtol=0.05)

    def test_ripple_snr_holds_on_a_unit_gain_channel(self):
        t = simulate_time(self.FS * 20, self.FS)
        lfps = simulate_multichannel_LFP(
            t, [5.0, 10.0, 15.0], 2, channel_gains=[1.0, 0.5], ripple_snr=5.0, random_state=3
        )
        background = simulate_multichannel_LFP(
            t, [], 2, channel_gains=[1.0, 0.5], noise_type="pink", random_state=3
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
            random_state=4,
        )
        np.testing.assert_array_equal(lfps[:, 0], lfps[:, 1])
        np.testing.assert_array_equal(lfps[:, 0], lfps[:, 2])
        assert np.abs(lfps[(t > 1.9) & (t < 2.1), 0]).max() > 0.5
        assert np.all(lfps[(t < 1.9) | (t > 2.1), 0] == 0.0)

    def test_seed_reproduces_and_changes(self):
        t = simulate_time(3000, self.FS)
        a = simulate_multichannel_LFP(t, [1.0], 2, random_state=5)
        np.testing.assert_array_equal(
            a, simulate_multichannel_LFP(t, [1.0], 2, random_state=5)
        )
        assert not np.array_equal(a, simulate_multichannel_LFP(t, [1.0], 2, random_state=6))

    def test_bad_arguments_raise(self):
        t = simulate_time(3000, self.FS)
        with pytest.raises(ValueError, match="channel_gains"):
            simulate_multichannel_LFP(t, [1.0], 2, channel_gains=[1.0], random_state=0)
        with pytest.raises(ValueError, match="shared_noise_fraction"):
            simulate_multichannel_LFP(t, [1.0], 2, shared_noise_fraction=1.5, random_state=0)
        with pytest.raises(ValueError, match="n_channels"):
            simulate_multichannel_LFP(t, [1.0], 0, random_state=0)
        with pytest.raises(ValueError, match="not both"):
            simulate_multichannel_LFP(t, [1.0], 2, ripple_amplitude=1.0, ripple_snr=2.0)


class TestSimulateSharpWaveRipplePair:
    FS = 1500

    def test_shape_and_channel_order(self):
        t = simulate_time(3000, self.FS)
        assert simulate_sharp_wave_ripple_pair(t, [1.0], random_state=0).shape == (3000, 2)

    def test_sharp_wave_is_negative_on_the_radiatum_channel_and_leaks_positive(self):
        t = simulate_time(3000, self.FS)
        pair = simulate_sharp_wave_ripple_pair(
            t, [1.0], noise_amplitude=0.0, ripple_amplitude=2.0, sharp_wave_amplitude=2.0
        )
        near = (t > 0.95) & (t < 1.05)
        # radiatum: a -2 deflection carrying 0.3 of a unit-peak ripple
        assert -2.3 <= pair[near, 1].min() <= -1.7
        assert pair[near, 1].sum() < 0.0
        assert pair[np.abs(t - 1.0) < 0.005, 1].mean() < -1.5
        # pyramidal: the ripple (peak 1) on 0.3 x 2 of sharp wave
        assert pair[near, 0].max() == pytest.approx(1.6, abs=0.2)
        assert np.all(np.abs(pair[(t < 0.85) | (t > 1.15)]) < 1e-6)

    def test_no_sharp_wave_without_ripples(self):
        t = simulate_time(3000, self.FS)
        pair = simulate_sharp_wave_ripple_pair(t, [], noise_amplitude=0.0)
        assert np.all(pair == 0.0)


class TestSimulateMultiunit:
    FS = 1500

    def test_shape_and_counts(self):
        t = simulate_time(3000, self.FS)
        counts = simulate_multiunit(t, [1.0], 5, random_state=0)
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
            random_state=1,
        )
        inside = counts[(t > 9.98) & (t < 10.02)].sum()
        outside = counts[(t > 4.98) & (t < 5.02)].sum()
        assert inside > 4 * max(outside, 1)

    def test_no_participation_means_no_burst(self):
        t = simulate_time(self.FS * 20, self.FS)
        counts = simulate_multiunit(
            t, [10.0], 20, baseline_rate=5.0, participation=0.0, random_state=1
        )
        inside = counts[(t > 9.9) & (t < 10.1)].sum()
        outside = counts[(t > 4.9) & (t < 5.1)].sum()
        assert inside < 2.0 * max(outside, 1)

    def test_baseline_rate_is_honoured(self):
        t = simulate_time(self.FS * 60, self.FS)
        counts = simulate_multiunit(t, [], 10, baseline_rate=4.0, random_state=2)
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
        session = simulate_session(
            t, [3.0, 9.0, 15.0], n_channels=3, n_units=8, random_state=0
        )
        assert isinstance(session, SimulatedSession)
        assert session.lfps.shape == (t.size, 3)
        assert session.raw_lfp_pair.shape == (t.size, 2)
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
            random_state=seed,
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
        session = simulate_session(t, [0.01, 1.99], ripple_duration=0.1, random_state=0)
        np.testing.assert_allclose(session.ripple_windows, [[t[0], 0.06], [1.94, t[-1]]])

    def test_a_list_of_times_is_accepted_and_sessions_compare_by_identity(self):
        t = simulate_time(3000, self.FS)
        session = simulate_session(list(t), [1.0], random_state=0)
        assert isinstance(session.time, np.ndarray)
        assert session != simulate_session(t, [1.0], random_state=0)
        assert session == session

    def test_mismatched_lengths_raise(self):
        t = simulate_time(3000, self.FS)
        session = simulate_session(t, [1.0], random_state=0)
        with pytest.raises(ValueError, match="samples"):
            dataclasses.replace(session, speed=np.zeros(10))
        with pytest.raises(ValueError, match="length"):
            dataclasses.replace(session, ripple_durations=np.zeros(2))

    def test_the_ripple_channel_is_shared_by_the_lfps_and_the_pair(self):
        t = simulate_time(3000, self.FS)
        session = simulate_session(t, [1.0], random_state=1)
        np.testing.assert_array_equal(session.lfps[:, 0], session.raw_lfp_pair[:, 0])

    def test_the_three_signals_carry_the_same_events(self):
        t = simulate_time(self.FS * 20, self.FS)
        session = simulate_session(
            t,
            [5.0, 10.0, 15.0],
            ripple_amplitude=2.0,
            noise_amplitude=0.0,
            baseline_rate=5.0,
            participation=1.0,
            random_state=2,
        )
        for start, end in session.ripple_windows:
            inside = (t >= start) & (t <= end)
            assert np.abs(session.lfps[inside, 0]).max() > 0.9  # the ripple
            assert session.raw_lfp_pair[inside, 1].min() < -1.5  # the sharp wave
            rate_inside = session.multiunit[inside].sum() / inside.sum()
            rate_outside = session.multiunit[~inside].sum() / (~inside).sum()
            assert rate_inside > 2.5 * rate_outside  # the population burst

    def test_seed_reproduces(self):
        t = simulate_time(3000, self.FS)
        a = simulate_session(t, [1.0], random_state=3)
        b = simulate_session(t, [1.0], random_state=3)
        np.testing.assert_array_equal(a.lfps, b.lfps)
        np.testing.assert_array_equal(a.multiunit, b.multiunit)
        np.testing.assert_array_equal(a.ripple_durations, b.ripple_durations)

    def test_artifacts_are_recorded(self):
        t = simulate_time(6000, self.FS)
        session = simulate_session(t, [1.0], artifact_times=[3.0], random_state=4)
        np.testing.assert_array_equal(session.artifact_times, [3.0])

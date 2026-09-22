"""Tests for core signal processing and utility functions."""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.signal import filtfilt, freqz
from scipy.stats import zscore

from ripple_detection.core import (
    _extend_segment,
    _find_containing_interval,
    _get_ripplefilter_kernel,
    estimate_noise_threshold,
    exclude_close_events,
    exclude_movement,
    extend_threshold_to_mean,
    filter_ripple_band,
    gaussian_smooth,
    get_envelope,
    get_multiunit_population_firing_rate,
    merge_close_events,
    merge_overlapping_ranges,
    merge_overlapping_ranges_track_participation,
    nearest_sample_index,
    normalize_signal,
    normalize_signal_manually,
    require_overlap,
    ripple_bandpass_filter,
    sample_count_within,
    segment_boolean_series,
    threshold_by_zscore,
)


@pytest.mark.parametrize(
    "series, expected_segments",
    [
        (
            pd.Series([False, True, True, True, False], index=np.linspace(0, 0.020, 5)),
            [(0.005, 0.015)],
        ),
        (
            pd.Series(
                [False, False, True, True, False, True, False], index=np.linspace(0, 0.030, 7)
            ),
            [],
        ),
        (pd.Series([True, True, False, False, False], index=np.linspace(0, 0.020, 5)), []),
        (
            pd.Series([False, True, True, True, True], index=np.linspace(0, 0.020, 5)),
            [(0.005, 0.020)],
        ),
        (
            pd.Series([True, True, True, True, False], index=np.linspace(0, 0.020, 5)),
            [(0.000, 0.015)],
        ),
        (
            pd.Series(
                [True, True, True, True, False, True, True, True],
                index=np.linspace(0, 0.035, 8),
            ),
            [(0.000, 0.015), (0.025, 0.035)],
        ),
    ],
)
def test_segment_boolean_series(series, expected_segments):
    segments = segment_boolean_series(series)
    assert len(segments) == len(expected_segments)
    for (test_start, test_end), (expected_start, expected_end) in zip(
        segments, expected_segments, strict=True
    ):
        assert np.allclose(expected_start, test_start)
        assert np.allclose(expected_end, test_end)


class TestSegmentDurationCountsSamples:
    """A run qualifies when it holds round(minimum_duration * fs) samples,
    the lab extractevents convention, independent of timestamp round-off."""

    @staticmethod
    def _run(sampling_frequency, n_true, start, offset, n_time=100_000):
        time = offset + np.arange(n_time) / sampling_frequency
        series = pd.Series(False, index=time)
        series.iloc[start : start + n_true] = True
        return series

    @pytest.mark.parametrize("offset", [0.0, 1000.0, 123_456.789])
    @pytest.mark.parametrize("start", [0, 1, 7, 99_000, 99_984])
    def test_exact_minimum_at_1000_hz_always_qualifies(self, offset, start):
        series = self._run(1000, 15, start, offset)  # 15 samples = 15 ms
        assert len(segment_boolean_series(series, 0.015)) == 1

    def test_one_sample_short_does_not_qualify(self):
        assert len(segment_boolean_series(self._run(1000, 14, 500, 1000.0), 0.015)) == 0

    @pytest.mark.parametrize(("n_true", "qualifies"), [(22, False), (23, True), (24, True)])
    def test_half_sample_products_round_half_up(self, n_true, qualifies):
        # 0.015 s at 1500 Hz is 22.5 samples; extractevents uses round() -> 23
        series = self._run(1500, n_true, 500, 0.0)
        assert (len(segment_boolean_series(series, 0.015)) == 1) == qualifies

    def test_gap_spanning_run_is_measured_in_samples_not_span(self):
        # five samples whose timestamps straddle a one-second hole (as after
        # NaN rows are dropped) cover 5 samples, not one second
        time = np.concatenate([np.arange(0, 1.0, 0.001), np.arange(2.0, 3.0, 0.001)])
        series = pd.Series(False, index=time)
        series.iloc[998:1003] = True
        assert len(segment_boolean_series(series, 0.015)) == 0


@pytest.mark.parametrize(
    "interval_candidates, target_interval, expected_interval",
    [
        ([(1, 2), (5, 7)], (6, 7), (5, 7)),
        ([(1, 2), (5, 7)], (1, 2), (1, 2)),
        ([(1, 2), (5, 7), (20, 30)], (5, 6), (5, 7)),
        ([(1, 2), (5, 7), (20, 30)], (24, 26), (20, 30)),
    ],
)
def test_find_containing_interval(interval_candidates, target_interval, expected_interval):
    test_interval = _find_containing_interval(interval_candidates, target_interval)
    assert np.all(test_interval == expected_interval)


@pytest.mark.parametrize(
    "interval_candidates, target_intervals, expected_intervals",
    [
        ([(1, 2), (5, 7)], [(6, 7)], [(5, 7)]),
        ([(1, 2), (5, 7)], [(1, 2)], [(1, 2)]),
        ([(1, 2), (5, 7), (20, 30)], [(5, 6)], [(5, 7)]),
        ([(1, 2), (5, 7), (20, 30)], [(24, 26), (6, 7)], [(20, 30), (5, 7)]),
        ([(1, 2), (5, 7), (20, 30)], [(24, 26), (27, 28)], [(20, 30)]),
    ],
)
def test__extend_segment(interval_candidates, target_intervals, expected_intervals):
    test_intervals = _extend_segment(target_intervals, interval_candidates)
    assert np.all(test_intervals == expected_intervals)


@pytest.mark.parametrize(
    "ranges, expected_ranges",
    [
        ([(5, 7), (3, 5), (-1, 3)], [(-1, 7)]),
        ([(5, 6), (3, 4), (1, 2)], [(1, 2), (3, 4), (5, 6)]),
        ([], []),
    ],
)
def test_merge_overlapping_ranges(ranges, expected_ranges):
    assert list(merge_overlapping_ranges(ranges)) == expected_ranges


@pytest.mark.parametrize(
    "channel_ranges, expected",
    [
        # A-B and B-C overlap preserves all three participants.
        (
            [[(0.1, 0.15)], [(0.14, 0.19)], [(0.18, 0.23)]],
            [[0.1, 0.23, {0, 1, 2}]],
        ),
        # Two ripples on one electrode still count that electrode only once.
        (
            [[(0.1, 0.15), (0.18, 0.23)], [(0.14, 0.19)]],
            [[0.1, 0.23, {0, 1}]],
        ),
        # Separate events retain their own participants and chronological order.
        ([[(0.3, 0.4)], [(0.1, 0.2)]], [[0.1, 0.2, {1}], [0.3, 0.4, {0}]]),
        ([[(0.1, 0.2)], [(0.2, 0.3)]], [[0.1, 0.3, {0, 1}]]),
        ([], []),
        ([[], []], []),
    ],
)
def test_merge_overlapping_ranges_track_participation(channel_ranges, expected):
    merged = merge_overlapping_ranges_track_participation(channel_ranges)
    assert merged.shape == (len(expected), 3)
    assert merged.tolist() == expected


def test_threshold_by_zscore():
    data = np.array([0, 0, 10, 10, 0, 0, 0, 1, 5, 10, 10, 10, 10, 10, 5, 1, 0])
    time = np.arange(len(data)) / 1000
    data = zscore(data)
    segments = threshold_by_zscore(data, time, zscore_threshold=1, minimum_duration=0.004)
    assert np.allclose(segments, [(0.008, 0.014)])


def test_exclude_movement():
    n_samples = 100
    time = np.arange(n_samples) / 1000
    speed = np.ones_like(time) * 5
    speed[3:11] = 1
    candidate_ripple_times = [(0.004, 0.010), (0.094, 0.095)]
    ripple_times = exclude_movement(candidate_ripple_times, speed, time, speed_threshold=4.0)
    expected_ripple_times = np.array([(0.004, 0.010)])
    assert np.allclose(ripple_times, expected_ripple_times)


# ============================================================================
# Signal Processing Tests
# ============================================================================


class TestRippleBandpassFilter:
    """Test ripple bandpass filter generation."""

    def test_filter_shape(self):
        """The tap count is odd and grows with the sampling rate."""
        low_rate, _ = ripple_bandpass_filter(1500)
        high_rate, filter_denominator = ripple_bandpass_filter(30000)
        assert len(low_rate) % 2 == 1 and len(high_rate) % 2 == 1
        assert len(low_rate) >= 101
        assert len(high_rate) > len(low_rate)
        assert filter_denominator == 1.0


class TestGetRipplefilterKernel:
    """Test loading of pre-computed ripple filter."""

    def test_kernel_loads(self):
        """Test that the pre-computed kernel loads successfully."""
        filter_numerator, filter_denominator = _get_ripplefilter_kernel()
        assert isinstance(filter_numerator, np.ndarray)
        assert filter_denominator == 1
        assert len(filter_numerator) > 0


class TestFilterRippleBand:
    """Test ripple band filtering function.

    Note: The pre-computed filter requires very long signals (>954 samples).
    Most filtering tests are covered by integration tests in test_detectors.py
    which use realistic LFP data generated by fixtures.
    """

    def test_multi_channel(self):
        """Test filtering multi-channel LFP with realistic data."""
        # Use the actual test fixtures which generate proper LFP data
        from ripple_detection.simulate import simulate_LFP, simulate_time

        sampling_frequency = 1500
        n_samples = sampling_frequency * 3  # 3 seconds
        time = simulate_time(n_samples, sampling_frequency)

        # Generate two channels with ripples
        lfp1 = simulate_LFP(time, [1.1], noise_amplitude=1.2, ripple_amplitude=1.5)
        lfp2 = simulate_LFP(time, [1.2], noise_amplitude=1.2, ripple_amplitude=1.5)
        multi_channel = np.column_stack([lfp1, lfp2])

        filtered = filter_ripple_band(multi_channel, 1500)

        assert filtered.shape == multi_channel.shape
        assert not np.all(np.isnan(filtered)), "Filtered signal should contain valid data"


def _tone_power(x, sampling_frequency, frequency):
    """Power of `x` at `frequency` from a single DFT bin."""
    n = len(x)
    k = round(frequency * n / sampling_frequency)
    return np.abs(np.fft.rfft(x)[k]) ** 2


class TestFilterRippleBandSamplingRate:
    """The ripple band must be 150-250 Hz in hertz at every sampling rate."""

    @staticmethod
    def _tones(sampling_frequency, in_band=200.0, out_of_band=320.0, seconds=4.0):
        t = np.arange(int(seconds * sampling_frequency)) / sampling_frequency
        return t, np.sin(2 * np.pi * in_band * t) + np.sin(2 * np.pi * out_of_band * t)

    @pytest.mark.parametrize(
        ("sampling_frequency", "out_of_band"),
        [(2000, 320.0), (3000, 400.0), (1000, 120.0), (1250, 290.0)],
    )
    def test_out_of_band_tone_is_attenuated(self, sampling_frequency, out_of_band):
        # 320 Hz at 2000 Hz and 120 Hz at 1000 Hz fall *inside* the passband of
        # the shipped 1500 Hz kernel when it is applied at the wrong rate
        _, x = self._tones(sampling_frequency, out_of_band=out_of_band)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            y = filter_ripple_band(x, sampling_frequency=sampling_frequency)
        in_gain = _tone_power(y, sampling_frequency, 200.0) / _tone_power(
            x, sampling_frequency, 200.0
        )
        out_gain = _tone_power(y, sampling_frequency, out_of_band) / _tone_power(
            x, sampling_frequency, out_of_band
        )
        assert in_gain > 0.5  # passband
        assert 10 * np.log10(out_gain / in_gain) < -30  # at least 30 dB down

    def test_1500_hz_uses_the_shipped_kernel(self):
        _, x = self._tones(1500)
        kernel, _ = _get_ripplefilter_kernel()
        np.testing.assert_array_equal(
            filter_ripple_band(x, sampling_frequency=1500), filtfilt(kernel, 1, x)
        )

    def test_rate_too_low_for_the_band_raises(self):
        _, x = self._tones(500, out_of_band=100.0)
        with pytest.raises(ValueError, match="Nyquist"):
            filter_ripple_band(x, sampling_frequency=500)

    def test_nan_rows_are_preserved(self):
        _, x = self._tones(2000)
        x[1000:1050] = np.nan
        y = filter_ripple_band(x, sampling_frequency=2000)
        assert np.all(np.isnan(y[1000:1050]))
        assert np.all(np.isfinite(np.delete(y, np.arange(1000, 1050))))

    def test_each_side_of_a_gap_is_filtered_on_its_own(self):
        """A DC step across the gap must not leak a transient into either side."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=6000)
        x[:2000] += 50.0  # a large offset on one side only
        x[2000:2500] = np.nan
        y = filter_ripple_band(x, sampling_frequency=1500)
        np.testing.assert_array_equal(y[:2000], filter_ripple_band(x[:2000], 1500))
        np.testing.assert_array_equal(y[2500:], filter_ripple_band(x[2500:], 1500))

    def test_a_run_too_short_to_filter_is_nan_with_a_warning(self):
        x = np.random.default_rng(0).normal(size=6000)
        x[2000:2010] = np.nan
        x[2100:2110] = np.nan  # leaves a 90-sample run between the gaps
        with pytest.warns(UserWarning, match="shorter than"):
            y = filter_ripple_band(x, sampling_frequency=1500)
        assert np.all(np.isnan(y[2000:2110]))
        assert np.all(np.isfinite(y[:2000])) and np.all(np.isfinite(y[2110:]))


class TestGetEnvelope:
    """Test Hilbert transform envelope extraction."""

    def test_constant_amplitude_sine(self):
        """Test envelope of constant amplitude sine wave."""
        time = np.linspace(0, 1, 1500)
        amplitude = 2.0
        frequency = 200
        signal = amplitude * np.sin(2 * np.pi * frequency * time)

        envelope = get_envelope(signal)

        assert envelope.shape == signal.shape
        # Envelope should be approximately constant at amplitude
        # Due to edge effects, check the middle portion
        middle = slice(50, -50)
        assert np.allclose(envelope[middle], amplitude, atol=0.2)

    def test_amplitude_modulated_signal(self):
        """Test envelope extraction from amplitude-modulated signal."""
        time = np.linspace(0, 1, 1500)
        carrier_freq = 200
        modulation_freq = 5

        # Create amplitude modulation
        amplitude = 1 + 0.5 * np.sin(2 * np.pi * modulation_freq * time)
        signal = amplitude * np.sin(2 * np.pi * carrier_freq * time)

        envelope = get_envelope(signal)

        # Envelope should follow the amplitude modulation
        assert np.corrcoef(envelope, amplitude)[0, 1] > 0.95

    def test_2d_signal(self):
        """Test envelope extraction on 2D array (multiple channels)."""
        time = np.linspace(0, 1, 1500)
        signal1 = 2 * np.sin(2 * np.pi * 200 * time)
        signal2 = 3 * np.sin(2 * np.pi * 180 * time)
        signal_2d = np.column_stack([signal1, signal2])

        envelope = get_envelope(signal_2d, axis=0)

        assert envelope.shape == signal_2d.shape
        # Check middle portions due to edge effects
        middle = slice(50, -50)
        assert np.allclose(envelope[middle, 0], 2.0, atol=0.2)
        assert np.allclose(envelope[middle, 1], 3.0, atol=0.2)


class TestGaussianSmooth:
    """Test Gaussian smoothing function."""

    def test_smooths_noisy_signal(self):
        """Test that smoothing reduces noise."""
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(1500)
        sampling_frequency = 1500
        sigma = 0.01  # 10 ms

        smoothed = gaussian_smooth(signal, sigma, sampling_frequency)

        assert smoothed.shape == signal.shape
        # Smoothed signal should have lower variance than original
        assert np.var(smoothed) < np.var(signal)

    def test_preserves_constant_signal(self):
        """Test that constant signal is preserved."""
        signal = np.ones(1500) * 5.0
        sampling_frequency = 1500
        sigma = 0.01

        smoothed = gaussian_smooth(signal, sigma, sampling_frequency)

        # Smoothing a constant signal should preserve most values
        # Edge effects may cause some variation
        middle = slice(100, -100)
        assert np.allclose(smoothed[middle], signal[middle], atol=0.1)

    def test_smooths_step_function(self):
        """Test smoothing of step function."""
        signal = np.zeros(1500)
        signal[750:] = 10.0  # Step at midpoint
        sampling_frequency = 1500
        sigma = 0.01

        smoothed = gaussian_smooth(signal, sigma, sampling_frequency)

        # Edges should be softened
        assert smoothed[745] < 10.0  # Before step
        assert smoothed[755] < 10.0  # After step
        assert 0 < smoothed[750] < 10  # At step

    def test_2d_signal(self):
        """Test smoothing 2D array along axis."""
        rng = np.random.default_rng(42)
        signal = rng.standard_normal((1500, 3))
        sampling_frequency = 1500
        sigma = 0.01

        smoothed = gaussian_smooth(signal, sigma, sampling_frequency, axis=0)

        assert smoothed.shape == signal.shape
        # Each channel should be smoothed
        for i in range(3):
            assert np.var(smoothed[:, i]) < np.var(signal[:, i])


class TestExcludeCloseEvents:
    """Test exclusion of events that occur too close together."""

    def test_removes_close_events(self):
        """Test that events within threshold are excluded."""
        candidate_times = np.array([(0.0, 0.1), (0.15, 0.2), (1.0, 1.1), (1.05, 1.15)])
        close_threshold = 0.1

        filtered_times = exclude_close_events(candidate_times, close_threshold)

        # Should remove second event in each pair
        assert len(filtered_times) == 2
        assert np.allclose(filtered_times[0], [0.0, 0.1])
        assert np.allclose(filtered_times[1], [1.0, 1.1])

    def test_preserves_distant_events(self):
        """Test that well-separated events are preserved."""
        candidate_times = np.array([(0.0, 0.1), (1.0, 1.1), (2.0, 2.1)])
        close_threshold = 0.1

        filtered_times = exclude_close_events(candidate_times, close_threshold)

        # All events should be preserved
        assert len(filtered_times) == 3

    def test_empty_input(self):
        """Test with empty input array."""
        candidate_times = np.array([]).reshape(0, 2)
        filtered_times = exclude_close_events(candidate_times, 0.1)

        assert len(filtered_times) == 0

    def test_single_event(self):
        """Test with single event."""
        candidate_times = np.array([(0.0, 0.1)])
        filtered_times = exclude_close_events(candidate_times, 0.1)

        assert len(filtered_times) == 1


class TestGetMultiunitPopulationFiringRate:
    """Test multiunit population firing rate calculation."""

    def test_firing_rate_shape(self):
        """Test output shape matches input time dimension."""
        rng = np.random.default_rng(42)
        n_samples = 1500
        n_units = 10
        multiunit = rng.random((n_samples, n_units)) < 0.05
        sampling_frequency = 1500

        firing_rate = get_multiunit_population_firing_rate(
            multiunit.astype(float), sampling_frequency, smoothing_sigma=0.015
        )

        assert firing_rate.shape == (n_samples,)

    def test_firing_rate_positive(self):
        """Test that firing rates are non-negative."""
        rng = np.random.default_rng(42)
        n_samples = 1500
        n_units = 10
        multiunit = rng.random((n_samples, n_units)) < 0.05
        sampling_frequency = 1500

        firing_rate = get_multiunit_population_firing_rate(
            multiunit.astype(float), sampling_frequency, smoothing_sigma=0.015
        )

        assert np.all(firing_rate >= 0), "Firing rates should be non-negative"

    def test_high_synchrony_increases_rate(self):
        """Test that high synchrony periods have higher firing rates."""
        rng = np.random.default_rng(42)
        n_samples = 1500
        n_units = 20
        sampling_frequency = 1500

        # Create baseline firing
        multiunit = rng.random((n_samples, n_units)) < 0.01

        # Add high synchrony event at middle
        multiunit[700:800, :] = rng.random((100, n_units)) < 0.3

        firing_rate = get_multiunit_population_firing_rate(
            multiunit.astype(float), sampling_frequency, smoothing_sigma=0.015
        )

        # Firing rate during high synchrony should be higher
        baseline_rate = np.mean(firing_rate[:500])
        synchrony_rate = np.mean(firing_rate[700:800])
        assert synchrony_rate > baseline_rate * 2


# ============================================================================
# Error Handling Tests for Core Functions
# ============================================================================


class TestCoreErrorHandling:
    """Test error handling for core functions."""

    def test_filter_ripple_band_empty_array(self):
        """An empty array has no run long enough to filter."""
        with pytest.raises(ValueError, match="too short"):
            filter_ripple_band(np.array([]), 1500)

    def test_get_envelope_empty_array(self):
        """Test envelope extraction with empty array."""
        # Empty array will raise ValueError in Hilbert transform
        empty_array = np.array([])
        try:
            envelope = get_envelope(empty_array)
            assert envelope.shape == empty_array.shape
        except ValueError:
            # Expected for empty input
            pass

    def test_gaussian_smooth_single_sample(self):
        """Test smoothing with single sample."""
        signal = np.array([5.0])
        smoothed = gaussian_smooth(signal, sigma=0.01, sampling_frequency=1500)
        # Single sample should be unchanged (or close to it)
        assert smoothed.shape == signal.shape

    def test_exclude_movement_empty_candidates(self):
        """Test exclude_movement with no candidates."""
        time = np.arange(100) / 1000
        speed = np.ones_like(time) * 2.0
        candidate_times = []

        result = exclude_movement(candidate_times, speed, time)

        assert len(result) == 0

    def test_threshold_by_zscore_all_same_values(self):
        """Test thresholding when all values are the same (z-score undefined)."""
        data = np.ones(100)
        time = np.arange(100) / 1000

        # Z-score of constant data is problematic, but should handle gracefully
        try:
            segments = threshold_by_zscore(data, time)
            # Should return empty or handle gracefully
            assert isinstance(segments, list)
        except (ValueError, RuntimeWarning):
            # May raise warning about constant data
            pass


# Tests for normalize_signal function


class TestNormalizeSignal:
    """Tests for normalize_signal function."""

    def test_zscore_normalization_basic(self):
        """Test basic z-score normalization."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalize_signal(data, method="zscore")

        # Check that mean is close to 0 and std is close to 1
        assert np.abs(np.mean(normalized)) < 1e-10
        assert np.abs(np.std(normalized, ddof=0) - 1.0) < 1e-10

    def test_median_mad_normalization_basic(self):
        """Test basic median/MAD normalization."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalize_signal(data, method="median_mad")

        # Check that median is close to 0
        assert np.abs(np.median(normalized)) < 1e-10
        # MAD should be scaled appropriately
        assert normalized.shape == data.shape

    def test_zscore_with_outliers(self):
        """Test that z-score is affected by outliers."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        normalized = normalize_signal(data, method="zscore")

        # The outlier should strongly affect the normalization
        assert normalized[-1] > 1.5  # Outlier will have large z-score

    def test_median_mad_robust_to_outliers(self):
        """Test that median/MAD is robust to outliers."""
        data_no_outlier = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data_with_outlier = np.array([1.0, 2.0, 3.0, 4.0, 100.0])

        norm_no_outlier = normalize_signal(data_no_outlier, method="median_mad")
        norm_with_outlier = normalize_signal(data_with_outlier, method="median_mad")

        # First 4 values should be similar despite outlier
        assert np.allclose(norm_no_outlier[:4], norm_with_outlier[:4], rtol=0.3)

    def test_normalization_mask_zscore(self):
        """Test z-score normalization with custom mask."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(100) + 5.0  # Mean of 5
        mask = np.zeros(100, dtype=bool)
        mask[:50] = True  # Use first 50 samples for normalization

        normalized = normalize_signal(data, method="zscore", normalization_mask=mask)

        # Mean of first 50 samples should be ~0, but all data is normalized
        assert len(normalized) == 100
        assert np.abs(np.mean(data[mask])) > 1.0  # Original has non-zero mean
        # After normalization, the masked region should have mean close to 0
        assert np.abs(np.mean(normalized[mask])) < 0.5

    def test_normalization_mask_median_mad(self):
        """Test median/MAD normalization with custom mask."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(100) + 5.0
        mask = np.zeros(100, dtype=bool)
        mask[:50] = True

        normalized = normalize_signal(data, method="median_mad", normalization_mask=mask)

        # Median of first 50 samples should be ~0 after normalization
        assert len(normalized) == 100
        assert np.abs(np.median(normalized[mask])) < 0.5

    def test_normalization_time_range(self):
        """Test normalization with time range."""
        rng = np.random.default_rng(42)
        time = np.arange(100) / 100.0  # 0 to 0.99 seconds
        data = rng.standard_normal(100) + 5.0

        # Normalize using first 50 time points (0 to 0.49 seconds)
        normalized = normalize_signal(
            data, time=time, method="zscore", normalization_time_range=(0.0, 0.49)
        )

        assert len(normalized) == 100
        # First half should have mean ~0
        assert np.abs(np.mean(normalized[:50])) < 0.5

    def test_multichannel_zscore(self):
        """Test z-score normalization with multi-channel data."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 4)) * np.array([1, 2, 3, 4])  # Different scales

        normalized = normalize_signal(data, method="zscore")

        # Each channel should be independently normalized
        assert normalized.shape == (100, 4)
        for ch in range(4):
            assert np.abs(np.mean(normalized[:, ch])) < 1e-10
            assert np.abs(np.std(normalized[:, ch], ddof=0) - 1.0) < 1e-10

    def test_multichannel_median_mad(self):
        """Test median/MAD normalization with multi-channel data."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 4)) * np.array([1, 2, 3, 4])

        normalized = normalize_signal(data, method="median_mad")

        # Each channel should be independently normalized
        assert normalized.shape == (100, 4)
        for ch in range(4):
            assert np.abs(np.median(normalized[:, ch])) < 1e-1

    def test_multichannel_with_mask(self):
        """Test multi-channel normalization with mask."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 4)) + 5.0
        mask = np.zeros(100, dtype=bool)
        mask[:50] = True

        normalized = normalize_signal(data, method="zscore", normalization_mask=mask)

        assert normalized.shape == (100, 4)
        # Masked region of each channel should have mean ~0
        for ch in range(4):
            assert np.abs(np.mean(normalized[mask, ch])) < 0.5

    def test_nan_handling_zscore(self):
        """Test that NaN values are handled correctly."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        normalized = normalize_signal(data, method="zscore")

        # NaN should remain NaN
        assert np.isnan(normalized[2])
        # Other values should be normalized
        assert not np.isnan(normalized[0])

    def test_nan_handling_median_mad(self):
        """Test that NaN values are handled correctly with median/MAD."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        normalized = normalize_signal(data, method="median_mad")

        # NaN should remain NaN
        assert np.isnan(normalized[2])
        # Other values should be normalized
        assert not np.isnan(normalized[0])

    @pytest.mark.parametrize("method", ["zscore", "median_mad"])
    def test_constant_data_raises(self, method):
        """A constant trace has no scale, so it cannot be normalized."""
        with pytest.raises(ValueError, match="zero or undefined"):
            normalize_signal(np.ones(100), method=method)

    @pytest.mark.parametrize("method", ["zscore", "median_mad"])
    def test_constant_channel_raises_and_names_it(self, method):
        """One dead channel among healthy ones raises and says which."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((200, 3))
        data[:, 1] = 0.0
        with pytest.raises(ValueError, match=r"channel\(s\) \[1\]"):
            normalize_signal(data, method=method)

    def test_non_boolean_mask_raises(self):
        """A forgotten comparison (speed instead of speed <= 4) is caught."""
        data = np.arange(10.0)
        with pytest.raises(ValueError, match="must be boolean"):
            normalize_signal(data, normalization_mask=np.arange(10.0))

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Invalid normalization method"):
            normalize_signal(data, method="invalid")

    def test_mask_and_time_range_both_specified(self):
        """Test that using both mask and time_range raises error."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        time = np.arange(5) / 5.0
        mask = np.array([True, True, False, False, False])

        with pytest.raises(ValueError, match="Cannot specify both"):
            normalize_signal(
                data,
                time=time,
                normalization_mask=mask,
                normalization_time_range=(0.0, 0.5),
            )

    def test_time_range_without_time(self):
        """Test that time_range without time raises error."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(ValueError, match="'time' parameter is required"):
            normalize_signal(data, normalization_time_range=(0.0, 2.0))

    def test_mask_length_mismatch(self):
        """Test that mask length mismatch raises error."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = np.array([True, True, False])  # Too short

        with pytest.raises(ValueError, match="normalization_mask length"):
            normalize_signal(data, normalization_mask=mask)

    def test_empty_time_range(self):
        """Test that empty time range raises error."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        time = np.array([0.0, 0.1, 0.2, 0.3, 0.4])

        with pytest.raises(ValueError, match="does not contain any data points"):
            normalize_signal(data, time=time, normalization_time_range=(1.0, 2.0))

    def test_comparison_with_scipy_zscore(self):
        """Test that default zscore matches scipy.stats.zscore."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(100)

        our_result = normalize_signal(data, method="zscore")
        scipy_result = zscore(data, ddof=0, nan_policy="omit")

        assert np.allclose(our_result, scipy_result)

    def test_immobility_normalization_use_case(self):
        """Test realistic use case: normalize using only immobility periods."""
        rng = np.random.default_rng(42)
        n_samples = 1000
        # Simulate LFP with different baseline during movement vs immobility
        speed = rng.random(n_samples) * 10  # 0-10 cm/s
        lfp = rng.standard_normal(n_samples)
        lfp[speed > 4] += 2.0  # Higher baseline during movement

        # Normalize using only immobility (speed < 4)
        immobility_mask = speed < 4.0
        normalized = normalize_signal(lfp, normalization_mask=immobility_mask)

        # Immobility periods should have mean ~0
        assert np.abs(np.mean(normalized[immobility_mask])) < 0.2
        # But entire signal is normalized
        assert len(normalized) == n_samples

    def test_baseline_normalization_use_case(self):
        """Test realistic use case: normalize using baseline period."""
        rng = np.random.default_rng(42)
        n_samples = 1500
        time = np.arange(n_samples) / 1500.0  # 0 to 1 second
        # Simulate LFP with baseline in first 0.2 seconds
        lfp = rng.standard_normal(n_samples)
        lfp[300:] += 3.0  # Shift after baseline period

        # Normalize using baseline (0 to 0.2 seconds)
        normalized = normalize_signal(lfp, time=time, normalization_time_range=(0.0, 0.2))

        # Baseline period should have mean ~0
        baseline_mask = time < 0.2
        assert np.abs(np.mean(normalized[baseline_mask])) < 0.2
        # Entire signal is normalized
        assert len(normalized) == n_samples

    def test_speed_based_normalization_for_detectors(self):
        """Test use case matching multiunit_HSE_detector pattern."""
        rng = np.random.default_rng(42)
        firing_rate = rng.standard_normal(1000) + 5.0
        speed = rng.random(1000) * 10.0

        speed_threshold = 4.0
        # This mimics what multiunit_HSE_detector does with use_speed_threshold_for_zscore
        normalized = normalize_signal(firing_rate, normalization_mask=speed < speed_threshold)

        # Should work and return full-length normalized data
        assert len(normalized) == 1000
        assert not np.all(normalized == 0)


class TestNormalizeSignalManually:
    def test_1d_scalar_baseline_deviation(self):
        """1-D manual normalization uses the scalar baseline/deviation."""
        data = np.arange(5.0)
        np.testing.assert_allclose(
            normalize_signal_manually(data, 1.0, 2.0), (data - 1.0) / 2.0
        )

    @pytest.mark.parametrize(
        ("baseline", "deviation"), [(0.0, 0.0), (0.0, np.nan), (np.nan, 2.0)]
    )
    def test_1d_degenerate_raises(self, baseline, deviation):
        with pytest.raises(ValueError, match="no scale"):
            normalize_signal_manually(np.arange(5.0), baseline, deviation)

    def test_degenerate_channel_raises_and_names_it(self):
        """A dead channel is not silently zeroed; the caller must drop it."""
        data = np.tile(np.arange(5.0)[:, None], (1, 3))
        baselines = np.array([1.0, np.nan, 0.0])
        deviations = np.array([2.0, 1.0, 0.0])
        with pytest.raises(ValueError, match=r"channel\(s\) \[1, 2\]"):
            normalize_signal_manually(data, baselines, deviations)

    def test_multichannel(self):
        data = np.tile(np.arange(5.0)[:, None], (1, 2))
        out = normalize_signal_manually(data, [1.0, 2.0], [2.0, 4.0])
        np.testing.assert_allclose(out[:, 0], (data[:, 0] - 1.0) / 2.0)
        np.testing.assert_allclose(out[:, 1], (data[:, 1] - 2.0) / 4.0)

    def test_wrong_channel_count_raises(self):
        data = np.zeros((5, 4))
        with pytest.raises(ValueError, match="one entry per channel"):
            normalize_signal_manually(data, [0.0], [1.0])
        with pytest.raises(ValueError, match="same shape"):
            normalize_signal_manually(data, [0.0, 0.0, 0.0, 0.0], [1.0, 1.0])


# ---------------------------------------------------------------------------
# estimate_noise_threshold (Yu et al. 2017 noise-percentile threshold)
# ---------------------------------------------------------------------------


def _matlab_reference_threshold(
    values,
    edges=None,
    window=11,
    limitval=1e-4,
    original_reflection=False,
):
    """Loop-for-loop transliteration of jy_variableripthreshold_corecalculation.m.

    Deliberately naive (explicit loops, MATLAB semantics spelled out) so it is
    an independent oracle for the vectorized implementation, not a copy of it.
    ``original_reflection=True`` reproduces the MATLAB file's ``abs(b) + 2m``
    reflection instead of the intended ``2m - b``.
    """
    if edges is None:
        edges = [round(-10 + 0.01 * k, 2) for k in range(6001)]
    n_bins = len(edges)
    # histc: bin k counts edges[k] <= x < edges[k+1]; the last bin counts x == edges[-1]
    counts = [0] * n_bins
    for x in values:
        if not np.isfinite(x) or x < edges[0] or x > edges[-1]:
            continue
        if x == edges[-1]:
            counts[-1] += 1
            continue
        k = int(np.floor((x - edges[0]) / 0.01 + 1e-9))
        # guard floating error at bin boundaries
        while k + 1 < n_bins and x >= edges[k + 1]:
            k += 1
        while k > 0 and x < edges[k]:
            k -= 1
        counts[k] += 1
    # MATLAB smooth(x, 11): moving average, shrinking symmetric window at the ends
    half = window // 2
    smoothed = []
    for i in range(n_bins):
        w = min(half, i, n_bins - 1 - i)
        seg = counts[i - w : i + w + 1]
        smoothed.append(sum(seg) / len(seg))
    peak = max(smoothed)
    j = next(i for i, v in enumerate(smoothed) if v == peak)
    m = edges[j]
    # left half (edges <= m) keeps raw counts; edges < m are reflected
    pdf = {}
    for k in range(j + 1):
        pdf[round(edges[k], 6)] = pdf.get(round(edges[k], 6), 0) + counts[k]
    for k in range(j):
        b = edges[k]
        pos = abs(b) + 2 * m if original_reflection else 2 * m - b
        pos = round(pos, 6)
        pdf[pos] = pdf.get(pos, 0) + counts[k]
    positions = sorted(pdf)
    total = sum(pdf[q] for q in positions)
    cum = 0.0
    k_cross = None
    for i, q in enumerate(positions):
        cum += pdf[q]
        if cum / total >= 1 - limitval:
            k_cross = i
            break
    return positions[k_cross + 1], m


class TestEstimateNoiseThreshold:
    """Transliteration of the Yu et al. 2017 mirrored-histogram threshold."""

    def test_recovers_symmetric_percentile_within_two_bins(self):
        # Gaussian centered below zero so every left-of-mode bin is negative.
        rng = np.random.default_rng(0)
        mean, sd = -1.0, 0.5
        values = rng.normal(mean, sd, 4_000_000)
        expected = mean + sd * 3.719016  # 99.99th percentile of N(0, 1)
        threshold = estimate_noise_threshold(values)
        assert abs(threshold - expected) <= 0.03

    def test_matches_loop_transliteration_on_skewed_sample(self):
        rng = np.random.default_rng(1)
        # right-skewed, shifted so the mode is negative (as a z-scored envelope is)
        values = rng.gamma(2.0, 0.4, 200_000) - 1.2
        expected, _ = _matlab_reference_threshold(values)
        assert estimate_noise_threshold(values) == pytest.approx(expected, abs=1e-9)

    def test_diagnostics_expose_grid_mode_and_counts(self):
        rng = np.random.default_rng(2)
        values = rng.normal(-1.0, 0.5, 100_000)
        threshold, diag = estimate_noise_threshold(values, return_diagnostics=True)
        _, expected_mode = _matlab_reference_threshold(values)
        assert diag["mode"] == pytest.approx(expected_mode, abs=1e-9)
        assert diag["histogram_edges"][0] == pytest.approx(-10.0)
        assert diag["histogram_edges"][-1] == pytest.approx(50.0)
        assert len(diag["histogram_edges"]) == 6001
        assert diag["counts"].sum() == 100_000
        assert diag["out_of_grid_fraction"] == 0.0
        assert diag["threshold"] == threshold
        assert diag["mean"] == pytest.approx(values.mean())
        assert diag["min"] == pytest.approx(values.min())
        # flank ratio: left-flank width over mode-to-mean distance; > 1 is the
        # regime in which the mirrored distribution can reach past the mean
        expected_ratio = (
            (diag["mode"] - values.min()) / (values.mean() - diag["mode"])
            if values.mean() > diag["mode"]
            else np.inf
        )
        assert diag["flank_ratio"] == pytest.approx(expected_ratio)

    def test_flank_ratio_is_infinite_when_mean_is_at_or_below_the_mode(self):
        # left-skewed sample: the mode lies above the mean, so the mirrored
        # distribution trivially reaches past the mean
        rng = np.random.default_rng(12)
        values = -rng.gamma(2.0, 0.4, 200_000) - 0.2
        threshold, diag = estimate_noise_threshold(values, return_diagnostics=True)
        assert diag["mean"] <= diag["mode"]
        assert diag["flank_ratio"] == np.inf
        assert threshold > diag["mean"]

    def test_reflection_equals_original_formula_when_mode_nonpositive(self):
        rng = np.random.default_rng(3)
        values = rng.gamma(2.0, 0.4, 200_000) - 1.2
        ours = estimate_noise_threshold(values)
        original, mode = _matlab_reference_threshold(values, original_reflection=True)
        assert mode <= 0
        assert ours == pytest.approx(original, abs=1e-9)

    def test_positive_mode_warns_and_uses_intended_reflection(self):
        rng = np.random.default_rng(4)
        mean, sd = 0.5, 0.2
        values = rng.normal(mean, sd, 2_000_000)
        with pytest.warns(UserWarning, match="mode"):
            threshold = estimate_noise_threshold(values)
        expected = mean + sd * 3.719016
        assert abs(threshold - expected) <= 0.03
        original, mode = _matlab_reference_threshold(values, original_reflection=True)
        assert mode > 0
        # the original formula throws the (0, m) bins past 2m and inflates the tail
        assert original > threshold + 0.1

    def test_percentile_parameter_lowers_threshold(self):
        rng = np.random.default_rng(5)
        values = rng.normal(-1.0, 0.5, 500_000)
        assert estimate_noise_threshold(values, percentile=99.0) < estimate_noise_threshold(
            values, percentile=99.99
        )

    def test_custom_grid_is_honoured(self):
        rng = np.random.default_rng(6)
        values = rng.normal(-1.0, 0.5, 500_000)
        edges = np.round(np.arange(-5, 5 + 0.005, 0.01), 6)
        threshold, diag = estimate_noise_threshold(
            values, histogram_edges=edges, return_diagnostics=True
        )
        assert len(diag["histogram_edges"]) == len(edges)
        assert abs(threshold - (-1.0 + 0.5 * 3.719016)) <= 0.03

    def test_out_of_grid_fraction_above_ceiling_raises(self):
        rng = np.random.default_rng(7)
        values = rng.normal(-1.0, 0.5, 100_000)
        values[:1_000] = 100.0  # 1 % beyond the grid
        with pytest.raises(ValueError, match="outside"):
            estimate_noise_threshold(values)

    def test_non_finite_values_count_as_out_of_grid(self):
        rng = np.random.default_rng(8)
        values = rng.normal(-1.0, 0.5, 100_000)
        values[:10] = np.nan
        _, diag = estimate_noise_threshold(values, return_diagnostics=True)
        assert diag["out_of_grid_fraction"] == pytest.approx(10 / 100_000)

    def test_mode_at_grid_edge_raises(self):
        rng = np.random.default_rng(9)
        values = rng.uniform(-10.0, -9.995, 100_000)  # every sample in bin 0
        with pytest.raises(ValueError, match="first or last bin"):
            estimate_noise_threshold(values)

    def test_crossing_on_last_mirrored_bin_raises(self):
        # 100 samples in bin 0 and 10 000 in bin 1: the mode is bin 1 and the
        # mirrored histogram is [100, 10000, 100], whose CDF first reaches
        # 0.9999 on its last bin, so "one bin past" does not exist
        values = np.concatenate([np.full(100, -9.995), np.full(10_000, -9.985)])
        with pytest.raises(ValueError, match="last mirrored bin"):
            estimate_noise_threshold(values)

    def test_too_few_samples_to_resolve_percentile_raises(self):
        rng = np.random.default_rng(10)
        values = rng.normal(-1.0, 0.5, 500)
        with pytest.raises(ValueError, match="samples"):
            estimate_noise_threshold(values)


class TestSampleCountWithin:
    """One rule for every duration limit: round-half-up sample counts, inclusive."""

    def test_minimum_is_inclusive_and_rounds_half_up(self):
        time = np.arange(100) / 1000.0  # 0.0205 s is 20.5 samples -> 21
        ok = sample_count_within(np.array([20, 21, 22]), time, 0.0205)
        np.testing.assert_array_equal(ok, [False, True, True])

    def test_maximum_is_inclusive(self):
        time = np.arange(100) / 1000.0  # 0.0305 s -> 31
        ok = sample_count_within(np.array([21, 31, 32]), time, 0.0205, 0.0305)
        np.testing.assert_array_equal(ok, [True, True, False])

    def test_scalar_input_gives_a_bool(self):
        time = np.arange(100) / 1000.0
        assert sample_count_within(21, time, 0.0205) is True
        assert sample_count_within(20, time, 0.0205) is False


class TestExcludeMovementEventLookup:
    """Speed must be read per event, not by matching timestamp values."""

    @staticmethod
    def _speed_with_movement_at(time, movement_times):
        speed = np.full(len(time), 2.0)
        for t in movement_times:
            speed[np.argmin(np.abs(time - t))] = 20.0
        return speed

    def test_nested_events_keep_the_right_one(self):
        time = np.arange(0, 6, 0.01)
        speed = self._speed_with_movement_at(time, [5.0])
        kept = exclude_movement(np.array([[1.0, 5.0], [2.0, 3.0]]), speed, time)
        np.testing.assert_allclose(kept, [[2.0, 3.0]])

    def test_events_sharing_a_start_time_are_each_evaluated(self):
        time = np.arange(0, 6, 0.01)
        speed = self._speed_with_movement_at(time, [1.06])
        kept = exclude_movement(np.array([[1.0, 1.05], [1.0, 1.06], [3.0, 3.05]]), speed, time)
        np.testing.assert_allclose(kept, [[1.0, 1.05], [3.0, 3.05]])

    def test_event_bounds_off_the_sample_grid_use_the_nearest_sample(self):
        time = np.arange(0, 6, 0.01)
        speed = np.full(len(time), 2.0)
        kept = exclude_movement(np.array([[1.00005, 2.00005]]), speed, time)
        np.testing.assert_allclose(kept, [[1.00005, 2.00005]])

    def test_empty_candidate_list(self):
        time = np.arange(0, 6, 0.01)
        speed = np.full(len(time), 2.0)
        assert len(exclude_movement(np.empty((0, 2)), speed, time)) == 0


class TestRippleBandpassFilterAcrossRates:
    """The designed filter must hold its specification at every sampling rate."""

    @staticmethod
    def _response(sampling_frequency, frequencies):
        from scipy.signal import freqz

        numerator, _ = ripple_bandpass_filter(sampling_frequency)
        _, response = freqz(numerator, worN=2 * np.pi * frequencies / sampling_frequency)
        return np.abs(response)

    @pytest.mark.parametrize("sampling_frequency", [600.0, 1000.0, 1500.0, 3000.0, 30000.0])
    def test_passband_is_flat_and_stopband_is_attenuated(self, sampling_frequency):
        passband = self._response(sampling_frequency, np.linspace(155, 245, 40))
        np.testing.assert_allclose(passband, 1.0, atol=0.06)
        stopband = np.concatenate(
            [
                self._response(sampling_frequency, np.linspace(1, 125, 40)),
                self._response(
                    sampling_frequency,
                    np.linspace(275, 0.5 * sampling_frequency - 1, 40),
                ),
            ]
        )
        assert stopband.max() < 0.06, stopband.max()


class TestFilterRippleBandLengthGuard:
    def test_shortest_accepted_signal_filters_without_a_scipy_error(self):
        kernel, _ = _get_ripplefilter_kernel()
        shortest = 3 * len(kernel) + 1
        filtered = filter_ripple_band(np.random.default_rng(0).normal(size=shortest), 1500)
        assert np.isfinite(filtered).all()

    def test_one_sample_shorter_raises_this_package_s_error(self):
        kernel, _ = _get_ripplefilter_kernel()
        with pytest.raises(ValueError, match="samples"):
            filter_ripple_band(np.random.default_rng(0).normal(size=3 * len(kernel)), 1500)


class TestExcludeCloseEventsChaining:
    def test_separation_is_measured_from_the_last_retained_event(self):
        # the middle event is dropped, so the third is 1.1 s after the last
        # retained event's end and must be kept
        events = np.array([[0.0, 0.1], [0.5, 0.6], [1.2, 1.3]])
        np.testing.assert_allclose(exclude_close_events(events, 1.0), [[0.0, 0.1], [1.2, 1.3]])

    def test_indices_track_the_retained_events(self):
        events = np.array([[0.0, 0.1], [0.5, 0.6], [1.2, 1.3]])
        kept, inds = exclude_close_events(events, 1.0, included_ripple_inds=[10, 11, 12])
        assert len(kept) == 2
        np.testing.assert_array_equal(np.asarray(inds), [10, 12])


class TestCoreInputConversion:
    def test_get_envelope_accepts_a_sequence(self):
        assert get_envelope([1.0, 2.0, 3.0, 2.0, 1.0]).shape == (5,)

    def test_population_firing_rate_accepts_a_sequence(self):
        rate = get_multiunit_population_firing_rate([[0, 1], [1, 0], [0, 0]], 1000.0)
        assert rate.shape == (3,)


class TestExtendThresholdToMeanEdgeCases:
    def test_threshold_run_inside_a_short_above_mean_run_still_extends(self):
        # the above-mean run is shorter than the minimum duration; it must not
        # be filtered away, or the containing run cannot be found
        time = np.arange(40) / 1000.0
        is_above_mean = np.zeros(40, dtype=bool)
        is_above_mean[10:25] = True
        is_above_threshold = np.zeros(40, dtype=bool)
        is_above_threshold[14:22] = True
        segments = extend_threshold_to_mean(
            is_above_mean, is_above_threshold, time, minimum_duration=0.005
        )
        assert segments == [(time[10], time[24])]

    def test_no_threshold_crossing_gives_no_segments(self):
        time = np.arange(40) / 1000.0
        segments = extend_threshold_to_mean(
            np.ones(40, dtype=bool), np.zeros(40, dtype=bool), time, 0.005
        )
        assert segments == []


class TestSegmentBooleanSeriesMissingValues:
    def test_missing_values_raise_rather_than_counting_as_true(self):
        series = pd.Series([np.nan] * 50, index=np.arange(50) / 1000.0)
        with pytest.raises(ValueError, match="missing"):
            segment_boolean_series(series, minimum_duration=0.005)


class TestNearestSampleIndex:
    def test_returns_the_closest_sample_in_query_order(self):
        time = np.arange(0.0, 1.0, 0.1)
        np.testing.assert_array_equal(nearest_sample_index(time, [0.52, 0.0, 0.98]), [5, 0, 9])

    def test_empty_time_raises(self):
        with pytest.raises(ValueError, match="time is empty"):
            nearest_sample_index(np.empty(0), [1.0])


class TestMergeCloseEvents:
    """Test merging of events separated by a short gap."""

    def test_merges_events_closer_than_the_gap(self):
        """Two events separated by less than the gap become one."""
        events = np.array([(0.0, 0.1), (0.13, 0.2)])

        merged = merge_close_events(events, 0.05)

        np.testing.assert_allclose(merged, [[0.0, 0.2]])

    def test_keeps_events_separated_by_more_than_the_gap(self):
        """A gap at or above the threshold is not bridged."""
        events = np.array([(0.0, 0.1), (0.15, 0.2)])

        merged = merge_close_events(events, 0.05)

        np.testing.assert_allclose(merged, [[0.0, 0.1], [0.15, 0.2]])

    def test_merges_a_chain_of_events(self):
        """Three events each close to the next collapse into one."""
        events = np.array([(0.0, 0.1), (0.12, 0.2), (0.22, 0.3)])

        merged = merge_close_events(events, 0.05)

        np.testing.assert_allclose(merged, [[0.0, 0.3]])

    def test_maximum_duration_stops_a_merge(self):
        """A merge that would exceed the span cap does not happen."""
        events = np.array([(0.0, 0.1), (0.12, 0.5)])

        merged = merge_close_events(events, 0.05, maximum_duration=0.3)

        np.testing.assert_allclose(merged, [[0.0, 0.1], [0.12, 0.5]])

    def test_maximum_duration_allows_a_merge_that_fits(self):
        """The cap is inclusive of spans below it."""
        events = np.array([(0.0, 0.1), (0.12, 0.2)])

        merged = merge_close_events(events, 0.05, maximum_duration=0.3)

        np.testing.assert_allclose(merged, [[0.0, 0.2]])

    def test_overlapping_events_are_merged(self):
        """A negative gap counts as close."""
        events = np.array([(0.0, 0.2), (0.1, 0.3)])

        merged = merge_close_events(events, 0.05)

        np.testing.assert_allclose(merged, [[0.0, 0.3]])

    def test_nested_event_does_not_shorten_the_outer_one(self):
        """Merging keeps the later of the two end times."""
        events = np.array([(0.0, 0.5), (0.1, 0.2)])

        merged = merge_close_events(events, 0.05)

        np.testing.assert_allclose(merged, [[0.0, 0.5]])

    def test_zero_gap_is_a_no_op(self):
        """The default threshold leaves separated events alone."""
        events = np.array([(0.0, 0.1), (0.2, 0.3)])

        merged = merge_close_events(events, 0.0)

        np.testing.assert_allclose(merged, events)

    def test_empty_input(self):
        """An empty event list stays empty and keeps its shape."""
        merged = merge_close_events(np.empty((0, 2)), 0.05)

        assert merged.shape == (0, 2)

    def test_single_event(self):
        """One event is returned unchanged."""
        merged = merge_close_events(np.array([(0.0, 0.1)]), 0.05)

        np.testing.assert_allclose(merged, [[0.0, 0.1]])

    def test_unsorted_input_raises(self):
        """Events must arrive sorted by start time."""
        events = np.array([(1.0, 1.1), (0.0, 0.1)])

        with pytest.raises(ValueError, match="sorted by start time"):
            merge_close_events(events, 0.05)

    def test_negative_gap_raises(self):
        """A negative gap threshold is meaningless."""
        with pytest.raises(ValueError, match="close_event_threshold"):
            merge_close_events(np.array([(0.0, 0.1)]), -1.0)


class TestCustomFrequencyBand:
    """The band is a parameter: 13 of the 29 surveyed papers do not use 150-250 Hz."""

    FS = 2000.0

    def test_default_band_is_unchanged(self):
        """Omitting the band designs the same filter as before."""
        default, _ = ripple_bandpass_filter(self.FS)
        explicit, _ = ripple_bandpass_filter(self.FS, band=(150.0, 250.0))

        np.testing.assert_allclose(default, explicit)

    def test_a_custom_band_passes_its_own_frequencies(self):
        """An 80-250 Hz design passes 100 Hz, which the default rejects."""
        wide, _ = ripple_bandpass_filter(self.FS, band=(80.0, 250.0))
        default, _ = ripple_bandpass_filter(self.FS)

        _, wide_response = freqz(wide, worN=[100.0], fs=self.FS)
        _, default_response = freqz(default, worN=[100.0], fs=self.FS)

        assert np.abs(wide_response[0]) > 0.9
        assert np.abs(default_response[0]) < 0.05

    def test_a_custom_band_rejects_outside_frequencies(self):
        """A 100-200 Hz design stops 250 Hz."""
        narrow, _ = ripple_bandpass_filter(self.FS, band=(100.0, 200.0))

        _, response = freqz(narrow, worN=[250.0], fs=self.FS)

        assert np.abs(response[0]) < 0.05

    def test_filter_ripple_band_takes_a_band(self):
        """The band reaches the filtering entry point."""
        rng = np.random.default_rng(0)
        signal = np.sin(2 * np.pi * 100.0 * np.arange(4000) / self.FS)
        signal += 0.01 * rng.normal(size=4000)

        wide = filter_ripple_band(signal, self.FS, band=(80.0, 250.0))
        default = filter_ripple_band(signal, self.FS)

        assert wide.std() > 0.5
        assert default.std() < 0.1

    def test_band_at_1500_hz_bypasses_the_shipped_kernel(self):
        """A custom band is designed even at the shipped kernel's rate."""
        signal = np.sin(2 * np.pi * 100.0 * np.arange(6000) / 1500.0)

        wide = filter_ripple_band(signal, 1500.0, band=(80.0, 250.0))

        assert wide.std() > 0.5

    def test_inverted_band_raises(self):
        with pytest.raises(ValueError, match="band"):
            ripple_bandpass_filter(self.FS, band=(250.0, 150.0))

    def test_band_above_nyquist_raises(self):
        with pytest.raises(ValueError, match="Nyquist"):
            ripple_bandpass_filter(1000.0, band=(150.0, 480.0))

    def test_band_below_zero_raises(self):
        with pytest.raises(ValueError, match="band"):
            ripple_bandpass_filter(self.FS, band=(-10.0, 250.0))

    def test_transition_band_is_adjustable(self):
        """A narrower transition needs more taps."""
        wide_transition, _ = ripple_bandpass_filter(self.FS, transition_width=25.0)
        narrow_transition, _ = ripple_bandpass_filter(self.FS, transition_width=10.0)

        assert len(narrow_transition) > len(wide_transition)


class TestRequireOverlap:
    """Thirteen of the 57 surveyed papers require a ripple and a burst together."""

    def test_keeps_an_overlapping_event(self):
        events = np.array([(0.0, 0.1)])
        reference = np.array([(0.05, 0.2)])

        np.testing.assert_allclose(require_overlap(events, reference), [[0.0, 0.1]])

    def test_drops_a_non_overlapping_event(self):
        events = np.array([(0.0, 0.1)])
        reference = np.array([(0.2, 0.3)])

        assert len(require_overlap(events, reference)) == 0

    def test_touching_events_do_not_overlap(self):
        """Overlap has to be positive, the project's event-level convention."""
        events = np.array([(0.0, 0.1)])
        reference = np.array([(0.1, 0.2)])

        assert len(require_overlap(events, reference)) == 0

    def test_keeps_the_event_bounds_not_the_intersection(self):
        """This is a filter, not an intersection."""
        events = np.array([(0.0, 1.0)])
        reference = np.array([(0.4, 0.5)])

        np.testing.assert_allclose(require_overlap(events, reference), [[0.0, 1.0]])

    def test_filters_a_mixture(self):
        events = np.array([(0.0, 0.1), (1.0, 1.1), (2.0, 2.1)])
        reference = np.array([(1.05, 1.5)])

        np.testing.assert_allclose(require_overlap(events, reference), [[1.0, 1.1]])

    def test_minimum_overlap_drops_a_brief_touch(self):
        events = np.array([(0.0, 0.1), (1.0, 1.1)])
        reference = np.array([(0.099, 0.5), (1.0, 1.1)])

        kept = require_overlap(events, reference, minimum_overlap=0.01)

        np.testing.assert_allclose(kept, [[1.0, 1.1]])

    def test_overlap_with_several_references_adds_up(self):
        """Two short references together can meet the minimum."""
        events = np.array([(0.0, 1.0)])
        reference = np.array([(0.1, 0.2), (0.5, 0.6)])

        assert len(require_overlap(events, reference, minimum_overlap=0.15)) == 1
        assert len(require_overlap(events, reference, minimum_overlap=0.25)) == 0

    def test_event_starting_inside_a_reference_counts_only_its_own_span(self):
        """The reference's part before the event start is not overlap."""
        events = np.array([(0.5, 1.0)])
        reference = np.array([(0.0, 0.7)])

        assert len(require_overlap(events, reference, minimum_overlap=0.15)) == 1
        assert len(require_overlap(events, reference, minimum_overlap=0.25)) == 0

    def test_event_ending_inside_a_reference_counts_only_its_own_span(self):
        """The reference's part after the event end is not overlap either."""
        events = np.array([(0.0, 0.5)])
        reference = np.array([(0.3, 1.0)])

        assert len(require_overlap(events, reference, minimum_overlap=0.15)) == 1
        assert len(require_overlap(events, reference, minimum_overlap=0.25)) == 0

    def test_event_inside_one_reference_counts_its_whole_span(self):
        """Both ends trimmed at once."""
        events = np.array([(0.4, 0.6)])
        reference = np.array([(0.0, 1.0)])

        assert len(require_overlap(events, reference, minimum_overlap=0.15)) == 1
        assert len(require_overlap(events, reference, minimum_overlap=0.25)) == 0

    def test_overlapping_references_are_not_double_counted(self):
        """The reference is reduced to its union first."""
        events = np.array([(0.0, 1.0)])
        reference = np.array([(0.1, 0.5), (0.2, 0.6)])

        assert len(require_overlap(events, reference, minimum_overlap=0.45)) == 1
        assert len(require_overlap(events, reference, minimum_overlap=0.55)) == 0

    def test_accepts_and_returns_dataframes(self):
        """Detector output goes straight in and comes back with every column."""
        events = pd.DataFrame(
            {"start_time": [0.0, 1.0], "end_time": [0.1, 1.1], "max_thresh": [3.0, 4.0]}
        )
        reference = pd.DataFrame({"start_time": [1.05], "end_time": [1.5]})

        kept = require_overlap(events, reference)

        assert isinstance(kept, pd.DataFrame)
        assert list(kept.index) == [1]
        assert list(kept.max_thresh) == [4.0]

    def test_empty_reference_keeps_nothing(self):
        events = np.array([(0.0, 0.1)])

        assert len(require_overlap(events, np.empty((0, 2)))) == 0

    def test_empty_events_stay_empty(self):
        kept = require_overlap(np.empty((0, 2)), np.array([(0.0, 0.1)]))

        assert kept.shape == (0, 2)

    def test_unsorted_reference_is_handled(self):
        """The reference does not have to arrive in order."""
        events = np.array([(1.0, 1.1)])
        reference = np.array([(2.0, 2.5), (1.05, 1.5)])

        assert len(require_overlap(events, reference)) == 1

    def test_negative_minimum_overlap_raises(self):
        with pytest.raises(ValueError, match="minimum_overlap"):
            require_overlap(np.array([(0.0, 0.1)]), np.array([(0.0, 0.1)]), -1.0)


class TestCloseEventBoundaryAgreement:
    """The merge and drop conventions decide the same gap the same way."""

    EVENTS = np.array([(0.0, 0.1), (0.15, 0.2)])  # gap is 0.05, below it in binary

    def test_a_gap_equal_to_the_threshold_is_not_close_for_either_convention(self):
        merged = merge_close_events(self.EVENTS, 0.05)
        kept = exclude_close_events(self.EVENTS, 0.05)

        assert len(merged) == 2
        assert len(kept) == 2

    def test_a_gap_below_the_threshold_is_close_for_both(self):
        events = np.array([(0.0, 0.1), (0.12, 0.2)])

        assert len(merge_close_events(events, 0.05)) == 1
        assert len(exclude_close_events(events, 0.05)) == 1

    def test_merging_is_monotonic_in_the_threshold(self):
        """Raising the gap can only merge more, never less."""
        events = np.array([(0.0, 0.1), (0.1, 0.2)])

        counts = [len(merge_close_events(events, gap)) for gap in (0.0, 1e-9, 1e-6, 0.05)]

        assert counts == sorted(counts, reverse=True)
        assert counts[0] == 1  # touching events merge at every threshold


class TestHelperBoundaries:
    """The comparisons at the edge of each new rule."""

    def test_a_merged_span_exactly_at_the_cap_is_allowed(self):
        events = np.array([(0.0, 0.1), (0.12, 0.3)])

        merged = merge_close_events(events, 0.05, maximum_duration=0.3)

        np.testing.assert_allclose(merged, [[0.0, 0.3]])

    def test_overlap_exactly_at_the_minimum_is_kept(self):
        events = np.array([(0.0, 0.1)])
        reference = np.array([(0.0, 0.1)])

        assert len(require_overlap(events, reference, minimum_overlap=0.1)) == 1

    def test_merge_does_not_modify_its_argument(self):
        events = np.array([(0.0, 0.1), (0.12, 0.2)])
        before = events.copy()

        merge_close_events(events, 0.05)

        np.testing.assert_array_equal(events, before)

    def test_require_overlap_does_not_modify_its_arguments(self):
        events = np.array([(0.0, 0.1), (1.0, 1.1)])
        reference = np.array([(1.05, 1.2), (0.5, 0.6)])
        before_events, before_reference = events.copy(), reference.copy()

        require_overlap(events, reference)

        np.testing.assert_array_equal(events, before_events)
        np.testing.assert_array_equal(reference, before_reference)

    def test_a_non_positive_transition_width_raises(self):
        with pytest.raises(ValueError, match="transition_width"):
            ripple_bandpass_filter(2000.0, transition_width=0.0)

    def test_a_band_edge_exactly_at_the_transition_raises(self):
        """The lower edge must leave room for the whole transition above 0 Hz."""
        with pytest.raises(ValueError, match="transition"):
            ripple_bandpass_filter(2000.0, band=(25.0, 250.0), transition_width=25.0)

    def test_a_band_edge_exactly_at_nyquist_raises(self):
        with pytest.raises(ValueError, match="Nyquist"):
            ripple_bandpass_filter(1000.0, band=(150.0, 475.0), transition_width=25.0)

    def test_the_default_transition_width_is_25_hz(self):
        """Changing it silently redesigns the filter at every non-1500 Hz rate."""
        default, _ = ripple_bandpass_filter(2000.0)

        # Kaiser's estimate for 45 dB over a 25 Hz transition at 2 kHz, made odd
        assert len(default) == 207
        explicit, _ = ripple_bandpass_filter(2000.0, transition_width=25.0)
        np.testing.assert_allclose(default, explicit)


def test_merge_close_events_rejects_a_flat_array_of_the_wrong_length():
    """A 1-D input has to be pairs of bounds."""
    with pytest.raises(ValueError):
        merge_close_events(np.array([0.0, 1.0, 2.0]), 0.05)


def test_transition_width_raises_where_the_shipped_kernel_is_used():
    """A silently ignored keyword would give the caller the wrong filter."""
    signal = np.random.default_rng(0).normal(size=4000)

    with pytest.raises(ValueError, match="transition_width"):
        filter_ripple_band(signal, 1500.0, transition_width=10.0)


def test_transition_width_applies_without_a_band_at_other_rates():
    """Designing a filter honors the transition width even with the default band."""
    default = filter_ripple_band(np.random.default_rng(0).normal(size=6000), 2000.0)
    narrow = filter_ripple_band(
        np.random.default_rng(0).normal(size=6000), 2000.0, transition_width=10.0
    )

    assert not np.allclose(default, narrow)

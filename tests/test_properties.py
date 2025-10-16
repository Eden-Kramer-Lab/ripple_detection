"""Property-based tests using Hypothesis for ripple detection."""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from ripple_detection.core import (
    exclude_close_events,
    filter_ripple_band,
    gaussian_smooth,
    get_envelope,
    merge_overlapping_ranges,
    segment_boolean_series,
    threshold_by_zscore,
)
from ripple_detection.simulate import brown, normalize, pink, simulate_LFP, white


class TestFilterRippleBandProperties:
    """Property-based tests for ripple band filtering."""

    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=2000, max_value=5000),
            elements=st.floats(
                min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
            ),
        )
    )
    @settings(max_examples=20, deadline=2000)
    def test_filter_preserves_shape(self, data):
        """Filtering should preserve the shape of the input."""
        data_2d = data.reshape(-1, 1)
        filtered = filter_ripple_band(data_2d)
        assert filtered.shape == data_2d.shape

    @given(n_samples=st.integers(min_value=2000, max_value=5000))
    @settings(max_examples=20, deadline=2000)
    def test_filter_multichannel_shape(self, n_samples):
        """Filtering should work with multiple channels."""
        data = np.random.randn(n_samples, 3)
        filtered = filter_ripple_band(data)
        assert filtered.shape == data.shape

    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=2000, max_value=5000),
            elements=st.floats(
                min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
            ),
        )
    )
    @settings(max_examples=20, deadline=2000)
    def test_filter_output_finite(self, data):
        """Filtered output should always be finite."""
        data_2d = data.reshape(-1, 1)
        filtered = filter_ripple_band(data_2d)
        assert np.all(np.isfinite(filtered))


class TestGetEnvelopeProperties:
    """Property-based tests for envelope extraction."""

    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1000, max_value=5000),
            elements=st.floats(
                min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
            ),
        )
    )
    @settings(max_examples=20, deadline=1000)
    def test_envelope_always_positive(self, data):
        """Envelope (amplitude) should always be non-negative."""
        envelope = get_envelope(data)
        assert np.all(envelope >= 0)

    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1000, max_value=5000),
            elements=st.floats(
                min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
            ),
        )
    )
    @settings(max_examples=20, deadline=1000)
    def test_envelope_preserves_shape(self, data):
        """Envelope extraction should preserve array shape."""
        envelope = get_envelope(data)
        assert envelope.shape == data.shape

    @given(
        amplitude=st.floats(min_value=0.1, max_value=10.0),
        n_samples=st.integers(min_value=1000, max_value=3000),
    )
    @settings(max_examples=20, deadline=1000)
    def test_envelope_scales_with_amplitude(self, amplitude, n_samples):
        """Envelope should scale proportionally with signal amplitude."""
        time = np.linspace(0, 1, n_samples)
        signal1 = amplitude * np.sin(2 * np.pi * 200 * time)
        signal2 = 2 * amplitude * np.sin(2 * np.pi * 200 * time)

        env1 = get_envelope(signal1)
        env2 = get_envelope(signal2)

        # Ratio should be approximately 2 (within tolerance for edge effects)
        middle = slice(100, -100)
        ratio = np.mean(env2[middle]) / np.mean(env1[middle])
        assert 1.8 < ratio < 2.2


class TestGaussianSmoothProperties:
    """Property-based tests for Gaussian smoothing."""

    @given(
        n_samples=st.integers(min_value=1000, max_value=5000),
        sigma=st.floats(min_value=0.004, max_value=0.05),
    )
    @settings(max_examples=20, deadline=1000)
    def test_smooth_reduces_variance(self, n_samples, sigma):
        """Smoothing should reduce signal variance (for random noisy signals)."""
        # Generate noisy signal with significant variance
        np.random.seed(42)
        data = np.random.randn(n_samples) * 10.0

        smoothed = gaussian_smooth(data, sigma, sampling_frequency=1500)

        # Smoothing should reduce variance
        assert np.var(smoothed) < np.var(data)

    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1000, max_value=5000),
            elements=st.floats(
                min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
            ),
        ),
        sigma=st.floats(min_value=0.001, max_value=0.1),
    )
    @settings(max_examples=20, deadline=1000)
    def test_smooth_preserves_shape(self, data, sigma):
        """Smoothing should preserve array shape."""
        smoothed = gaussian_smooth(data, sigma, sampling_frequency=1500)
        assert smoothed.shape == data.shape

    @given(
        value=st.floats(
            min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        n_samples=st.integers(min_value=1500, max_value=3000),
        sigma=st.floats(min_value=0.004, max_value=0.02),
    )
    @settings(max_examples=20, deadline=1000)
    def test_smooth_preserves_constant(self, value, n_samples, sigma):
        """Smoothing a constant signal should return approximately the same constant."""
        data = np.full(n_samples, value)
        smoothed = gaussian_smooth(data, sigma, sampling_frequency=1500)
        # Check central region to avoid edge effects
        middle = slice(500, -500)
        assert np.allclose(smoothed[middle], value, rtol=1e-3, atol=1e-4)


class TestThresholdByZscoreProperties:
    """Property-based tests for z-score thresholding."""

    @given(
        n_samples=st.integers(min_value=1000, max_value=5000),
        time_above_thresh=st.floats(min_value=0.001, max_value=0.05),
        zscore_threshold=st.floats(min_value=0.5, max_value=3.0),
    )
    @settings(max_examples=20, deadline=1000)
    def test_threshold_output_is_list(self, n_samples, time_above_thresh, zscore_threshold):
        """Thresholding should return a list of tuples."""
        # Generate data with some variance
        np.random.seed(42)
        data = np.random.randn(n_samples) * 2.0

        time = np.linspace(0, n_samples / 1500, n_samples)
        result = threshold_by_zscore(data, time, time_above_thresh, zscore_threshold)
        assert isinstance(result, list)
        # Each element should be a tuple of (start, end)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert item[0] <= item[1]

    @given(n_samples=st.integers(min_value=2000, max_value=5000))
    @settings(max_examples=20, deadline=1000)
    def test_higher_threshold_fewer_detections(self, n_samples):
        """Higher z-score threshold should result in fewer or equal detections."""
        np.random.seed(42)
        data = np.random.randn(n_samples) * 2.0

        time = np.linspace(0, n_samples / 1500, n_samples)
        result_low = threshold_by_zscore(data, time, 0.01, 1.0)
        result_high = threshold_by_zscore(data, time, 0.01, 3.0)

        assert len(result_high) <= len(result_low)


class TestMergeOverlappingRangesProperties:
    """Property-based tests for merging overlapping ranges."""

    @given(
        ranges=st.lists(
            st.tuples(
                st.floats(min_value=0.0, max_value=100.0),
                st.floats(min_value=0.0, max_value=100.0),
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=50, deadline=1000)
    def test_merge_reduces_or_maintains_count(self, ranges):
        """Merging should reduce or maintain the number of ranges."""
        # Filter out invalid ranges and duplicates
        valid_ranges = sorted(
            {(min(s, e), max(s, e)) for s, e in ranges if abs(s - e) > 0.01}
        )

        if len(valid_ranges) == 0:
            return

        merged = list(merge_overlapping_ranges(valid_ranges))

        assert len(merged) <= len(valid_ranges)

    @given(
        ranges=st.lists(
            st.tuples(
                st.floats(min_value=0.0, max_value=100.0),
                st.floats(min_value=0.0, max_value=100.0),
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=50, deadline=1000)
    def test_merged_ranges_sorted(self, ranges):
        """Merged ranges should be sorted by start time."""
        valid_ranges = sorted(
            {(min(s, e), max(s, e)) for s, e in ranges if abs(s - e) > 0.01}
        )

        if len(valid_ranges) == 0:
            return

        merged = list(merge_overlapping_ranges(valid_ranges))

        if len(merged) > 1:
            starts = [m[0] for m in merged]
            assert all(starts[i] <= starts[i + 1] for i in range(len(starts) - 1))

    @given(
        ranges=st.lists(
            st.tuples(
                st.floats(min_value=0.0, max_value=100.0),
                st.floats(min_value=0.0, max_value=100.0),
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=50, deadline=1000)
    def test_merged_ranges_non_overlapping(self, ranges):
        """Merged ranges should not overlap."""
        valid_ranges = sorted(
            {(min(s, e), max(s, e)) for s, e in ranges if abs(s - e) > 0.01}
        )

        if len(valid_ranges) == 0:
            return

        merged = list(merge_overlapping_ranges(valid_ranges))

        if len(merged) > 1:
            for i in range(len(merged) - 1):
                # End of current range should be before start of next
                assert merged[i][1] < merged[i + 1][0]


class TestExcludeCloseEventsProperties:
    """Property-based tests for excluding close events."""

    @given(
        events=st.lists(
            st.tuples(
                st.floats(min_value=0.0, max_value=100.0),
                st.floats(min_value=0.0, max_value=100.0),
            ),
            min_size=0,
            max_size=30,
        ),
        close_event_threshold=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50, deadline=1000)
    def test_exclude_reduces_or_maintains_count(self, events, close_event_threshold):
        """Excluding close events should reduce or maintain event count."""
        valid_events = [(min(s, e), max(s, e)) for s, e in events if s < e]

        if len(valid_events) == 0:
            return

        events_array = np.array(valid_events)
        filtered = exclude_close_events(events_array, close_event_threshold)

        assert len(filtered) <= len(events_array)


class TestSimulationProperties:
    """Property-based tests for simulation functions."""

    @given(n=st.integers(min_value=100, max_value=5000))
    @settings(max_examples=20, deadline=1000)
    def test_white_noise_shape(self, n):
        """White noise should have correct shape."""
        noise = white(n)
        assert noise.shape == (n,)

    @given(n=st.integers(min_value=100, max_value=5000))
    @settings(max_examples=20, deadline=1000)
    def test_pink_noise_shape(self, n):
        """Pink noise should have correct shape."""
        noise = pink(n)
        assert noise.shape == (n,)

    @given(n=st.integers(min_value=100, max_value=5000))
    @settings(max_examples=20, deadline=1000)
    def test_brown_noise_shape(self, n):
        """Brown noise should have correct shape."""
        noise = brown(n)
        assert noise.shape == (n,)

    @given(
        n=st.integers(min_value=100, max_value=2000),
        amplitude=st.floats(min_value=0.1, max_value=10.0),
    )
    @settings(max_examples=20, deadline=1000)
    def test_normalize_scales_power(self, n, amplitude):
        """Normalize should scale signal to match reference power."""
        signal = np.random.randn(n)
        reference = amplitude * np.random.randn(n)

        normalized = normalize(signal, reference)

        # Power should be approximately equal
        power_signal = np.mean(normalized**2)
        power_reference = np.mean(reference**2)

        assert np.isclose(power_signal, power_reference, rtol=0.1)

    @given(
        duration=st.floats(min_value=1.0, max_value=5.0),
        n_ripples=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=20, deadline=2000)
    def test_simulate_lfp_shape(self, duration, n_ripples):
        """Simulated LFP should have correct shape."""
        time = np.arange(0, duration, 1 / 1500)

        # Generate ripple times that don't overlap
        if n_ripples > 0:
            ripple_times = np.linspace(0.5, duration - 0.5, n_ripples)
        else:
            ripple_times = []

        lfp = simulate_LFP(time, ripple_times=ripple_times)

        assert lfp.shape == time.shape

    @given(duration=st.floats(min_value=2.0, max_value=5.0))
    @settings(max_examples=20, deadline=2000)
    def test_simulate_lfp_with_ripple_has_higher_power(self, duration):
        """LFP with ripples should have higher total power than noise-only LFP."""
        ripple_amplitude = 3.0
        noise_amplitude = 0.5

        time = np.arange(0, duration, 1 / 1500)
        ripple_times = [duration / 2]

        lfp_with_ripple = simulate_LFP(
            time,
            ripple_times=ripple_times,
            ripple_amplitude=ripple_amplitude,
            noise_amplitude=noise_amplitude,
        )
        lfp_no_ripple = simulate_LFP(
            time, ripple_times=[], ripple_amplitude=0, noise_amplitude=noise_amplitude
        )

        # Total power with ripple should be higher
        power_with_ripple = np.sum(lfp_with_ripple**2)
        power_no_ripple = np.sum(lfp_no_ripple**2)
        assert power_with_ripple > power_no_ripple


class TestSegmentBooleanSeriesProperties:
    """Property-based tests for boolean series segmentation."""

    @given(
        n_samples=st.integers(min_value=100, max_value=1000),
        true_fraction=st.floats(min_value=0.1, max_value=0.9),
    )
    @settings(max_examples=50, deadline=1000)
    def test_segment_start_before_end(self, n_samples, true_fraction):
        """Start times should always be before or equal to end times."""
        # Create random boolean series
        np.random.seed(42)
        bool_array = np.random.random(n_samples) < true_fraction
        time = np.linspace(0, n_samples / 1500, n_samples)
        series = pd.Series(bool_array, index=time)

        segments = segment_boolean_series(series, minimum_duration=0.001)

        # Each segment should have start <= end
        for start_time, end_time in segments:
            assert start_time <= end_time

    @given(
        n_samples=st.integers(min_value=100, max_value=1000),
        true_fraction=st.floats(min_value=0.1, max_value=0.9),
    )
    @settings(max_examples=50, deadline=1000)
    def test_segment_count_reasonable(self, n_samples, true_fraction):
        """Number of segments should be reasonable given the data."""
        np.random.seed(42)
        bool_array = np.random.random(n_samples) < true_fraction
        time = np.linspace(0, n_samples / 1500, n_samples)
        series = pd.Series(bool_array, index=time)

        segments = segment_boolean_series(series, minimum_duration=0.001)

        # Number of segments should not exceed number of True values
        n_true = np.sum(bool_array)
        assert len(segments) <= n_true + 1

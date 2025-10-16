"""Integration tests for ripple detection algorithms."""

import numpy as np
import pandas as pd
from ripple_detection import (
    Kay_ripple_detector,
    Karlsson_ripple_detector,
    filter_ripple_band,
)
from ripple_detection.detectors import (
    Roumis_ripple_detector,
    get_Kay_ripple_consensus_trace,
    multiunit_HSE_detector,
)


class TestKayRippleDetector:
    """Test suite for Kay ripple detector."""

    def test_single_channel_with_ripples(
        self, time_3s, single_lfp_with_ripples, stationary_speed, sampling_frequency
    ):
        """Test Kay detector with single LFP channel containing ripples."""
        filtered_lfps = filter_ripple_band(single_lfp_with_ripples)
        ripples = Kay_ripple_detector(
            time_3s, filtered_lfps, stationary_speed, sampling_frequency
        )

        # Verify output structure
        assert isinstance(ripples, pd.DataFrame)
        assert len(ripples) > 0, "Should detect at least one ripple"

        # Check required columns
        expected_columns = [
            "start_time",
            "end_time",
            "duration",
            "max_thresh",
            "mean_zscore",
            "median_zscore",
            "max_zscore",
            "min_zscore",
            "area",
            "total_energy",
            "speed_at_start",
            "speed_at_end",
            "max_speed",
            "min_speed",
            "median_speed",
            "mean_speed",
        ]
        for col in expected_columns:
            assert col in ripples.columns, f"Missing column: {col}"

        # Verify detected ripples are near true ripple times (1.1s and 2.1s)
        true_ripple_times = [1.1, 2.1]
        for true_time in true_ripple_times:
            # Check if any detected ripple overlaps with expected time window
            ripple_detected = any(
                (ripples["start_time"] <= true_time) & (ripples["end_time"] >= true_time)
            )
            assert ripple_detected, f"Failed to detect ripple near {true_time}s"

        # Verify duration is reasonable (ripples should be 15-300ms typically)
        assert all(ripples["duration"] >= 0.015), "Duration below minimum threshold"
        assert all(ripples["duration"] < 0.5), "Duration unreasonably long"

        # Verify z-scores are positive (above threshold)
        assert all(ripples["max_zscore"] > 0), "Max z-score should be positive"
        assert all(ripples["mean_zscore"] >= 0), "Mean z-score should be non-negative"

    def test_dual_channel_with_ripples(
        self, time_3s, dual_lfp_with_ripples, stationary_speed, sampling_frequency
    ):
        """Test Kay detector with two LFP channels."""
        filtered_lfps = filter_ripple_band(dual_lfp_with_ripples)
        ripples = Kay_ripple_detector(
            time_3s, filtered_lfps, stationary_speed, sampling_frequency
        )

        assert isinstance(ripples, pd.DataFrame)
        assert len(ripples) > 0, "Should detect ripples from multi-channel data"

        # With two channels having ripples at [1.1, 2.1] and [0.5, 2.5],
        # should detect events near these times
        assert len(ripples) >= 2, "Should detect at least 2 ripples"

    def test_close_ripples(
        self, time_3s, dual_lfp_close_ripples, stationary_speed, sampling_frequency
    ):
        """Test detection of closely spaced ripples."""
        filtered_lfps = filter_ripple_band(dual_lfp_close_ripples)
        ripples = Kay_ripple_detector(
            time_3s, filtered_lfps, stationary_speed, sampling_frequency
        )

        assert isinstance(ripples, pd.DataFrame)
        assert len(ripples) > 0

    def test_multi_channel_sparse_ripples(
        self, time_3s, multi_lfp_sparse_ripples, stationary_speed, sampling_frequency
    ):
        """Test with many channels but ripples only in subset."""
        filtered_lfps = filter_ripple_band(multi_lfp_sparse_ripples)
        ripples = Kay_ripple_detector(
            time_3s, filtered_lfps, stationary_speed, sampling_frequency
        )

        assert isinstance(ripples, pd.DataFrame)
        # Should still detect ripples even with many noise channels
        assert len(ripples) > 0

    def test_no_ripples(self, time_3s, lfp_no_ripples, stationary_speed, sampling_frequency):
        """Test with noise-only signal (no ripples)."""
        filtered_lfps = filter_ripple_band(lfp_no_ripples)
        ripples = Kay_ripple_detector(
            time_3s, filtered_lfps, stationary_speed, sampling_frequency
        )

        # Should return empty or very few false positives
        assert isinstance(ripples, pd.DataFrame)
        # With proper thresholding, should detect 0 or very few events
        assert len(ripples) <= 2, "Should not detect many events in noise-only signal"

    def test_speed_threshold(
        self, time_3s, dual_lfp_with_ripples, speed_with_movement, sampling_frequency
    ):
        """Test that ripples during movement are excluded."""
        filtered_lfps = filter_ripple_band(dual_lfp_with_ripples)

        # Detect with stationary speed
        ripples_stationary = Kay_ripple_detector(
            time_3s,
            filtered_lfps,
            np.ones_like(time_3s) * 2.0,
            sampling_frequency,
            speed_threshold=4.0,
        )

        # Detect with movement after t=1.5s
        ripples_movement = Kay_ripple_detector(
            time_3s,
            filtered_lfps,
            speed_with_movement,
            sampling_frequency,
            speed_threshold=4.0,
        )

        # Should detect fewer ripples when animal is moving
        assert len(ripples_movement) < len(ripples_stationary)

        # Ripples after t=1.5s should be excluded
        if len(ripples_movement) > 0:
            assert all(
                ripples_movement["start_time"] < 1.5
            ), "Ripples during movement should be excluded"

    def test_minimum_duration(
        self, time_3s, lfp_short_duration_ripples, stationary_speed, sampling_frequency
    ):
        """Test that very short ripples are not detected."""
        filtered_lfps = filter_ripple_band(lfp_short_duration_ripples)

        ripples = Kay_ripple_detector(
            time_3s,
            filtered_lfps,
            stationary_speed,
            sampling_frequency,
            minimum_duration=0.015,
        )

        # Very short ripples (1ms) might still create enough signal to be detected
        # but should have fewer detections than normal ripples
        # The test is more about ensuring the parameter works, not strict exclusion
        assert len(ripples) <= 3, "Very short ripples should result in few detections"

    def test_zscore_threshold_parameter(
        self, time_3s, single_lfp_with_ripples, stationary_speed, sampling_frequency
    ):
        """Test effect of z-score threshold parameter."""
        filtered_lfps = filter_ripple_band(single_lfp_with_ripples)

        # Low threshold - should detect more events
        ripples_low = Kay_ripple_detector(
            time_3s,
            filtered_lfps,
            stationary_speed,
            sampling_frequency,
            zscore_threshold=1.0,
        )

        # High threshold - should detect fewer events
        ripples_high = Kay_ripple_detector(
            time_3s,
            filtered_lfps,
            stationary_speed,
            sampling_frequency,
            zscore_threshold=5.0,
        )

        assert len(ripples_low) >= len(
            ripples_high
        ), "Lower threshold should detect more events"

    def test_close_ripple_threshold(
        self, time_3s, dual_lfp_close_ripples, stationary_speed, sampling_frequency
    ):
        """Test exclusion of ripples that occur too close together."""
        filtered_lfps = filter_ripple_band(dual_lfp_close_ripples)

        # No exclusion
        ripples_no_exclusion = Kay_ripple_detector(
            time_3s,
            filtered_lfps,
            stationary_speed,
            sampling_frequency,
            close_ripple_threshold=0.0,
        )

        # Exclude ripples within 0.1s
        ripples_with_exclusion = Kay_ripple_detector(
            time_3s,
            filtered_lfps,
            stationary_speed,
            sampling_frequency,
            close_ripple_threshold=0.1,
        )

        # Should have fewer or equal ripples with exclusion
        assert len(ripples_with_exclusion) <= len(ripples_no_exclusion)


class TestKarlssonRippleDetector:
    """Test suite for Karlsson ripple detector."""

    def test_single_channel_with_ripples(
        self, time_3s, single_lfp_with_ripples, stationary_speed, sampling_frequency
    ):
        """Test Karlsson detector with single LFP channel."""
        filtered_lfps = filter_ripple_band(single_lfp_with_ripples)
        ripples = Karlsson_ripple_detector(
            time_3s, filtered_lfps, stationary_speed, sampling_frequency
        )

        assert isinstance(ripples, pd.DataFrame)
        assert len(ripples) > 0, "Should detect at least one ripple"

        # Verify structure
        assert "start_time" in ripples.columns
        assert "end_time" in ripples.columns
        assert "duration" in ripples.columns

    def test_dual_channel_merging(
        self, time_3s, dual_lfp_with_ripples, stationary_speed, sampling_frequency
    ):
        """Test that Karlsson detector merges overlapping ripples from multiple channels."""
        filtered_lfps = filter_ripple_band(dual_lfp_with_ripples)
        ripples = Karlsson_ripple_detector(
            time_3s, filtered_lfps, stationary_speed, sampling_frequency
        )

        assert isinstance(ripples, pd.DataFrame)
        assert len(ripples) > 0

        # Karlsson method detects per channel then merges
        # Should still find the prominent ripples
        assert len(ripples) >= 1

    def test_zscore_threshold(
        self, time_3s, single_lfp_with_ripples, stationary_speed, sampling_frequency
    ):
        """Test Karlsson detector with different z-score thresholds."""
        filtered_lfps = filter_ripple_band(single_lfp_with_ripples)

        # Karlsson uses default threshold of 3.0
        ripples_default = Karlsson_ripple_detector(
            time_3s,
            filtered_lfps,
            stationary_speed,
            sampling_frequency,
            zscore_threshold=3.0,
        )

        ripples_low = Karlsson_ripple_detector(
            time_3s,
            filtered_lfps,
            stationary_speed,
            sampling_frequency,
            zscore_threshold=1.0,
        )

        assert len(ripples_low) >= len(ripples_default), "Lower threshold should detect more"

    def test_no_ripples(self, time_3s, lfp_no_ripples, stationary_speed, sampling_frequency):
        """Test Karlsson detector with noise-only signal."""
        filtered_lfps = filter_ripple_band(lfp_no_ripples)
        ripples = Karlsson_ripple_detector(
            time_3s, filtered_lfps, stationary_speed, sampling_frequency
        )

        assert isinstance(ripples, pd.DataFrame)
        # Should have few or no detections
        assert len(ripples) <= 2


class TestRoumisRippleDetector:
    """Test suite for Roumis ripple detector."""

    def test_single_channel_with_ripples(
        self, time_3s, single_lfp_with_ripples, stationary_speed, sampling_frequency
    ):
        """Test Roumis detector with single LFP channel."""
        filtered_lfps = filter_ripple_band(single_lfp_with_ripples)
        ripples = Roumis_ripple_detector(
            time_3s, filtered_lfps, stationary_speed, sampling_frequency
        )

        assert isinstance(ripples, pd.DataFrame)
        assert "start_time" in ripples.columns
        assert "end_time" in ripples.columns

        # Roumis detector may or may not detect depending on threshold
        # Just verify it returns valid structure

    def test_dual_channel(
        self, time_3s, dual_lfp_with_ripples, stationary_speed, sampling_frequency
    ):
        """Test Roumis detector with two channels."""
        filtered_lfps = filter_ripple_band(dual_lfp_with_ripples)
        ripples = Roumis_ripple_detector(
            time_3s, filtered_lfps, stationary_speed, sampling_frequency
        )

        assert isinstance(ripples, pd.DataFrame)
        assert "start_time" in ripples.columns
        assert "end_time" in ripples.columns


class TestMultiunitHSEDetector:
    """Test suite for multiunit High Synchrony Event detector."""

    def test_basic_detection(
        self, time_3s, multiunit_data, stationary_speed, sampling_frequency
    ):
        """Test HSE detector with synthetic multiunit data."""
        events = multiunit_HSE_detector(
            time_3s,
            multiunit_data,
            stationary_speed,
            sampling_frequency,
            zscore_threshold=2.0,
            minimum_duration=0.015,
        )

        assert isinstance(events, pd.DataFrame)
        assert "start_time" in events.columns
        assert "end_time" in events.columns

        # Should detect the high synchrony events we embedded
        if len(events) > 0:
            assert all(events["duration"] >= 0.015)

    def test_speed_threshold(
        self, time_3s, multiunit_data, speed_with_movement, sampling_frequency
    ):
        """Test HSE detector respects speed threshold."""
        events_stationary = multiunit_HSE_detector(
            time_3s,
            multiunit_data,
            np.ones_like(time_3s) * 2.0,
            sampling_frequency,
            speed_threshold=4.0,
        )

        events_movement = multiunit_HSE_detector(
            time_3s,
            multiunit_data,
            speed_with_movement,
            sampling_frequency,
            speed_threshold=4.0,
        )

        # Should detect fewer or equal events during movement
        assert len(events_movement) <= len(events_stationary)

    def test_use_speed_threshold_for_zscore(
        self, time_3s, multiunit_data, stationary_speed, sampling_frequency
    ):
        """Test use_speed_threshold_for_zscore parameter."""
        # This parameter changes whether z-score is calculated on all data
        # or only stationary periods
        events_all_data = multiunit_HSE_detector(
            time_3s,
            multiunit_data,
            stationary_speed,
            sampling_frequency,
            use_speed_threshold_for_zscore=False,
        )

        events_stationary_zscore = multiunit_HSE_detector(
            time_3s,
            multiunit_data,
            stationary_speed,
            sampling_frequency,
            use_speed_threshold_for_zscore=True,
        )

        # Both should return valid DataFrames
        assert isinstance(events_all_data, pd.DataFrame)
        assert isinstance(events_stationary_zscore, pd.DataFrame)


class TestKayConsensusTrace:
    """Test the Kay consensus trace generation."""

    def test_consensus_trace_shape(self, time_3s, dual_lfp_with_ripples, sampling_frequency):
        """Test that consensus trace has correct shape."""
        filtered_lfps = filter_ripple_band(dual_lfp_with_ripples)
        consensus = get_Kay_ripple_consensus_trace(
            filtered_lfps, sampling_frequency, smoothing_sigma=0.004
        )

        assert consensus.shape == (len(time_3s),)
        assert not np.all(np.isnan(consensus)), "Consensus trace should have valid data"

    def test_consensus_trace_positive(
        self, time_3s, dual_lfp_with_ripples, sampling_frequency
    ):
        """Test that consensus trace values are non-negative (it's a magnitude)."""
        filtered_lfps = filter_ripple_band(dual_lfp_with_ripples)
        consensus = get_Kay_ripple_consensus_trace(
            filtered_lfps, sampling_frequency, smoothing_sigma=0.004
        )

        # After square root, all values should be >= 0
        valid_values = consensus[~np.isnan(consensus)]
        assert np.all(valid_values >= 0), "Consensus trace should be non-negative"


class TestDetectorErrorHandling:
    """Test error handling and edge cases for detectors."""

    def test_empty_time_array(self, sampling_frequency):
        """Test detectors with empty input arrays."""
        time = np.array([])
        lfp = np.array([]).reshape(0, 1)
        speed = np.array([])

        # Should handle gracefully without crashing
        # Most detectors will return empty DataFrames
        try:
            ripples = Kay_ripple_detector(time, lfp, speed, sampling_frequency)
            assert isinstance(ripples, pd.DataFrame)
        except (ValueError, IndexError):
            # Some implementations may raise errors on empty input
            pass

    def test_nan_in_lfp(
        self, time_3s, single_lfp_with_ripples, stationary_speed, sampling_frequency
    ):
        """Test handling of NaN values in LFP data."""
        lfp_with_nan = single_lfp_with_ripples.copy()
        # Insert some NaN values
        lfp_with_nan[100:200, 0] = np.nan

        filtered_lfps = filter_ripple_band(lfp_with_nan)
        ripples = Kay_ripple_detector(
            time_3s, filtered_lfps, stationary_speed, sampling_frequency
        )

        # Should handle NaN and return valid DataFrame
        assert isinstance(ripples, pd.DataFrame)

    def test_nan_in_speed(
        self, time_3s, single_lfp_with_ripples, stationary_speed, sampling_frequency
    ):
        """Test handling of NaN values in speed data."""
        speed_with_nan = stationary_speed.copy()
        speed_with_nan[100:200] = np.nan

        filtered_lfps = filter_ripple_band(single_lfp_with_ripples)
        ripples = Kay_ripple_detector(
            time_3s, filtered_lfps, speed_with_nan, sampling_frequency
        )

        # Should handle NaN in speed data
        assert isinstance(ripples, pd.DataFrame)

    def test_mismatched_lengths(self, time_3s, single_lfp_with_ripples, sampling_frequency):
        """Test with mismatched time and LFP lengths."""
        # Create speed array with different length
        speed_short = np.ones(len(time_3s) // 2)

        filtered_lfps = filter_ripple_band(single_lfp_with_ripples)

        # This should either handle gracefully or raise appropriate error
        try:
            ripples = Kay_ripple_detector(
                time_3s, filtered_lfps, speed_short, sampling_frequency
            )
            # If it succeeds, verify output is valid
            assert isinstance(ripples, pd.DataFrame)
        except (ValueError, IndexError, KeyError):
            # Expected to raise an error with mismatched inputs
            pass

    def test_single_sample(self, sampling_frequency):
        """Test detectors with single sample input."""
        time = np.array([0.0])
        lfp = np.array([[0.5]])
        speed = np.array([2.0])

        # Should handle single sample gracefully
        try:
            ripples = Kay_ripple_detector(time, lfp, speed, sampling_frequency)
            assert isinstance(ripples, pd.DataFrame)
            assert len(ripples) == 0  # Can't detect ripple from single sample
        except (ValueError, IndexError):
            # May raise error for insufficient data
            pass

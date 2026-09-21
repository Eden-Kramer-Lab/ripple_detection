"""Integration tests for ripple detection algorithms."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import ripple_detection.detectors as detectors_module
from ripple_detection import (
    Karlsson_ripple_detector,
    Kay_ripple_detector,
    Shvartsman_ripple_detector,
    filter_ripple_band,
)
from ripple_detection.core import (
    gaussian_smooth,
    get_envelope,
)
from ripple_detection.detectors import (
    Roumis_ripple_detector,
    _extract_Yu_ripple_events,
    _find_max_thresh,
    get_Kay_ripple_consensus_trace,
    get_Yu_ripple_consensus_trace,
    multiunit_HSE_detector,
)
from ripple_detection.simulate import simulate_LFP


class TestShvartsmanRippleDetector:
    def test_single_channel_with_ripples(
        self, time_3s, single_lfp_with_ripples, stationary_speed, sampling_frequency
    ):
        """Test Shvartsman detector with single LFP channel containing ripples."""
        filtered_lfps = filter_ripple_band(single_lfp_with_ripples)
        ripples = Shvartsman_ripple_detector(
            time_3s, filtered_lfps, stationary_speed, sampling_frequency
        )

        # Verify output structure
        assert isinstance(ripples, pd.DataFrame)

        # Verify empty DataFrame (doesn't exceed 2-channel default participation minimum)
        assert ripples.empty, (
            "Should not detect any ripples because there's a 2-channel participation minimum default"
        )

    def test_single_channel_with_ripples_participation_threshold_0(
        self, time_3s, single_lfp_with_ripples, stationary_speed, sampling_frequency
    ):
        """Test Shvartsman detector with single LFP channel containing ripples, with participation_threshold=0."""
        filtered_lfps = filter_ripple_band(single_lfp_with_ripples)
        ripples = Shvartsman_ripple_detector(
            time_3s,
            filtered_lfps,
            stationary_speed,
            sampling_frequency,
            participation_threshold=0,
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
            "participants",
            "n_participants",
            "frac_participants",
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

        # Verify number of participants
        assert all(ripples["n_participants"] == 1), (
            "Single-channel ripples should have one participant"
        )

    def test_dual_channel_with_ripples(
        self, time_3s, dual_lfp_with_ripples, stationary_speed, sampling_frequency
    ):
        """Test Shvartsman detector with two LFP channels with non-overlapping ripples."""
        filtered_lfps = filter_ripple_band(dual_lfp_with_ripples)
        ripples = Shvartsman_ripple_detector(
            time_3s, filtered_lfps, stationary_speed, sampling_frequency
        )

        # Verify output structure
        assert isinstance(ripples, pd.DataFrame)

        # Verify empty DataFrame (doesn't exceed 2-channel default participation minimum)
        assert ripples.empty, (
            "Should not detect any ripples because they don't co-occur across the two channels"
        )

    def test_dual_channel_with_cooccur_ripples(
        self, time_3s, dual_lfp_with_cooccur_ripples, stationary_speed, sampling_frequency
    ):
        """Test Shvartsman detector with two LFP channels with non-overlapping ripples."""
        filtered_lfps = filter_ripple_band(dual_lfp_with_cooccur_ripples)
        ripples = Shvartsman_ripple_detector(
            time_3s, filtered_lfps, stationary_speed, sampling_frequency
        )

        # Verify output structure
        assert isinstance(ripples, pd.DataFrame)
        assert len(ripples) >= 2, "Should detect at least two ripples"

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
            "participants",
            "n_participants",
            "frac_participants",
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

        # Verify number of participants
        assert all(ripples["n_participants"] == 2), "Each ripple should have two participants"

    def test_close_ripples(
        self, time_3s, dual_lfp_close_ripples, stationary_speed, sampling_frequency
    ):
        """Closely-spaced but offset ripples (ch0 at 1.10s, ch1 at 1.15s) have peaks
        50 ms apart, but their full zero-crossing-extended ripples overlap, so they
        co-occur: both channels participate and the default 2-channel cutoff detects
        them."""
        filtered_lfps = filter_ripple_band(dual_lfp_close_ripples)

        ripples = Shvartsman_ripple_detector(
            time_3s, filtered_lfps, stationary_speed, sampling_frequency
        )
        assert isinstance(ripples, pd.DataFrame)
        assert len(ripples) == 2
        assert all(ripples["n_participants"] == 2)
        assert all(participants == {0, 1} for participants in ripples["participants"])

    def test_multi_channel_sparse_ripples(
        self, time_3s, multi_lfp_sparse_ripples, stationary_speed, sampling_frequency
    ):
        """Test Shvartsman detector with many LFP channels with a subset having non-overlapping ripples."""
        filtered_lfps = filter_ripple_band(multi_lfp_sparse_ripples)
        ripples = Shvartsman_ripple_detector(
            time_3s, filtered_lfps, stationary_speed, sampling_frequency
        )

        # Verify output structure
        assert isinstance(ripples, pd.DataFrame)

        # Verify empty DataFrame (doesn't exceed 2-channel default participation minimum for any single ripple)
        assert ripples.empty, (
            "Should not detect any ripples because they don't co-occur across the sparse channels"
        )

    def test_multi_channel_sparse_cooccur_ripples(
        self, time_3s, multi_lfp_sparse_cooccur_ripples, stationary_speed, sampling_frequency
    ):
        """Test Shvartsman detector with many LFP channels with a subset having co-occurring ripples."""
        filtered_lfps = filter_ripple_band(multi_lfp_sparse_cooccur_ripples)
        ripples = Shvartsman_ripple_detector(
            time_3s, filtered_lfps, stationary_speed, sampling_frequency
        )

        # Verify output structure
        assert isinstance(ripples, pd.DataFrame)
        assert len(ripples) >= 2, "Should detect at least two ripples"

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
            "participants",
            "n_participants",
            "frac_participants",
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

        # Verify number of participants
        assert all(ripples["n_participants"] == 2), "Each ripple should have two participants"
        # ripple channels are indices 0 and 1; the 11 noise channels never participate
        assert all(p == {0, 1} for p in ripples["participants"]), (
            "Participants should be channels 0 and 1"
        )
        assert np.allclose(ripples["frac_participants"], 2 / 13), (
            "frac_participants should be 2/13"
        )
        # Stats are computed over the participating channels {0, 1} only; averaging
        # over all 13 channels would dilute mean_zscore to well below 1.
        assert all(ripples["mean_zscore"] > 1.0), (
            "z-score stats must use participating channels only, not all channels"
        )

    def test_no_ripples(self, time_3s, lfp_no_ripples, stationary_speed, sampling_frequency):
        """Test with noise-only signal (no ripples)."""
        filtered_lfps = filter_ripple_band(lfp_no_ripples)
        ripples = Shvartsman_ripple_detector(
            time_3s, filtered_lfps, stationary_speed, sampling_frequency
        )

        # Should return empty or very few false positives
        assert isinstance(ripples, pd.DataFrame)
        # With proper thresholding, should detect very few events in random noise
        # Allow up to 5 false positives due to stochastic nature of noise
        assert ripples.empty, "Should not detect any events in noise-only signal"

    def test_all_movement_events(
        self,
        time_3s,
        dual_lfp_with_cooccur_ripples,
        speed_with_all_movement,
        sampling_frequency,
    ):
        """Test that if all events occur during movement, all are excluded and proper format is returned."""
        # Detect with movement after t=1.5s
        filtered_lfps = filter_ripple_band(dual_lfp_with_cooccur_ripples)

        ripples_movement = Shvartsman_ripple_detector(
            time_3s,
            filtered_lfps,
            speed_with_all_movement,
            sampling_frequency,
            speed_threshold=4.0,
        )

        # Verify output structure
        assert isinstance(ripples_movement, pd.DataFrame)

        # Verify empty DataFrame
        assert ripples_movement.empty, (
            "Should not detect any ripples because animal is always moving"
        )

    def test_all_but_one_movement_events(
        self, time_3s, dual_lfp_with_cooccur_ripples, speed_with_movement, sampling_frequency
    ):
        """Test that if all but one events occur during movement, all but one are excluded and proper format is returned."""
        # Detect with movement after t=1.5s
        filtered_lfps = filter_ripple_band(dual_lfp_with_cooccur_ripples)

        ripples_movement = Shvartsman_ripple_detector(
            time_3s,
            filtered_lfps,
            speed_with_movement,
            sampling_frequency,
            speed_threshold=4.0,
        )

        # Verify output structure
        assert isinstance(ripples_movement, pd.DataFrame)

        # Verify empty DataFrame
        assert len(ripples_movement) == 1, (
            "Should detect one ripple event that occurs before movement begins"
        )

    def test_speed_threshold(
        self, time_3s, dual_lfp_with_cooccur_ripples, speed_with_movement, sampling_frequency
    ):
        """Test that ripples during movement are excluded."""
        filtered_lfps = filter_ripple_band(dual_lfp_with_cooccur_ripples)

        # Detect with stationary speed
        ripples_stationary = Shvartsman_ripple_detector(
            time_3s,
            filtered_lfps,
            np.ones_like(time_3s) * 2.0,
            sampling_frequency,
            speed_threshold=4.0,
        )

        # Detect with movement after t=1.5s
        ripples_movement = Shvartsman_ripple_detector(
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
            assert all(ripples_movement["start_time"] < 1.5), (
                "Ripples during movement should be excluded"
            )

    def test_minimum_duration(
        self,
        time_3s,
        dual_lfp_with_cooccur_short_ripples,
        stationary_speed,
        sampling_frequency,
    ):
        """Test that very short ripples are not detected."""
        filtered_lfps = filter_ripple_band(dual_lfp_with_cooccur_short_ripples)

        ripples = Shvartsman_ripple_detector(
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
        self, time_3s, multi_lfp_sparse_cooccur_ripples, stationary_speed, sampling_frequency
    ):
        """Test effect of z-score threshold parameter."""
        filtered_lfps = filter_ripple_band(multi_lfp_sparse_cooccur_ripples)

        # Low threshold - should detect more events
        ripples_low = Shvartsman_ripple_detector(
            time_3s,
            filtered_lfps,
            stationary_speed,
            sampling_frequency,
            zscore_threshold=1.0,
        )

        # High threshold - should detect fewer events
        ripples_high = Shvartsman_ripple_detector(
            time_3s,
            filtered_lfps,
            stationary_speed,
            sampling_frequency,
            zscore_threshold=5.0,
        )

        assert len(ripples_low) >= len(ripples_high), (
            "Lower threshold should detect more events"
        )

    def test_close_ripple_threshold(
        self,
        time_3s,
        dual_lfp_with_close_cooccur_ripples,
        stationary_speed,
        sampling_frequency,
    ):
        """Test exclusion of ripples that occur too close together."""
        filtered_lfps = filter_ripple_band(dual_lfp_with_close_cooccur_ripples)

        # No exclusion
        ripples_no_exclusion = Shvartsman_ripple_detector(
            time_3s,
            filtered_lfps,
            stationary_speed,
            sampling_frequency,
            close_ripple_threshold=0.0,
        )

        # Exclude ripples within 0.25s
        ripples_with_exclusion = Shvartsman_ripple_detector(
            time_3s,
            filtered_lfps,
            stationary_speed,
            sampling_frequency,
            close_ripple_threshold=0.25,
        )

        # Should have fewer or equal ripples with exclusion
        assert len(ripples_no_exclusion) == 2
        assert len(ripples_with_exclusion) == 1

    def test_manual_norm_success(
        self, time_3s, multi_lfp_sparse_cooccur_ripples, stationary_speed, sampling_frequency
    ):
        """Manual normalization with valid per-channel baselines/deviations detects ripples."""
        filtered_lfps = filter_ripple_band(multi_lfp_sparse_cooccur_ripples)
        # precompute per-channel baseline/deviation of the smoothed envelope
        # (mirrors supplying day-level stats)
        env = gaussian_smooth(
            get_envelope(filtered_lfps), sigma=0.004, sampling_frequency=sampling_frequency
        )
        ripples = Shvartsman_ripple_detector(
            time_3s,
            filtered_lfps,
            stationary_speed,
            sampling_frequency,
            manual_normalization=True,
            elec_baselines=env.mean(axis=0),
            elec_deviations=env.std(axis=0),
        )
        assert isinstance(ripples, pd.DataFrame)
        assert len(ripples) == 2
        assert all(ripples["n_participants"] == 2)

    def test_manual_norm_no_baseline_inputs(
        self, time_3s, multi_lfp_sparse_cooccur_ripples, stationary_speed, sampling_frequency
    ):
        """Test Shvartsman detector with manual normalization indicated but no baseline values passed in."""
        filtered_lfps = filter_ripple_band(multi_lfp_sparse_cooccur_ripples)
        with pytest.raises(ValueError):
            Shvartsman_ripple_detector(
                time_3s,
                filtered_lfps,
                stationary_speed,
                sampling_frequency,
                manual_normalization=True,
            )

    def test_manual_norm_baseline_deviation_mismatch(
        self, time_3s, multi_lfp_sparse_cooccur_ripples, stationary_speed, sampling_frequency
    ):
        """Test Shvartsman detector with manual normalization indicated but mismatched elec_baselines and elec_deviations lengths."""
        filtered_lfps = filter_ripple_band(multi_lfp_sparse_cooccur_ripples)
        with pytest.raises(ValueError):
            Shvartsman_ripple_detector(
                time_3s,
                filtered_lfps,
                stationary_speed,
                sampling_frequency,
                manual_normalization=True,
                elec_baselines=np.ones(filtered_lfps.shape[1]),
                elec_deviations=np.ones(filtered_lfps.shape[1] - 1),
            )

    def test_manual_norm_lfp_baseline_mismatch(
        self, time_3s, multi_lfp_sparse_cooccur_ripples, stationary_speed, sampling_frequency
    ):
        """Test Shvartsman detector with manual normalization indicated but mismatched elec_baselines and filtered_lfp lengths."""
        filtered_lfps = filter_ripple_band(multi_lfp_sparse_cooccur_ripples)
        with pytest.raises(ValueError):
            Shvartsman_ripple_detector(
                time_3s,
                filtered_lfps,
                stationary_speed,
                sampling_frequency,
                manual_normalization=True,
                elec_baselines=np.ones(filtered_lfps.shape[1] - 1),
                elec_deviations=np.ones(filtered_lfps.shape[1] - 1),
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
        # With proper thresholding, should detect very few events in random noise
        # Allow up to 5 false positives due to stochastic nature of noise
        assert len(ripples) <= 5, "Should not detect many events in noise-only signal"

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
            assert all(ripples_movement["start_time"] < 1.5), (
                "Ripples during movement should be excluded"
            )

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

        assert len(ripples_low) >= len(ripples_high), (
            "Lower threshold should detect more events"
        )

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


def _yu_reference_consensus(filtered_lfps, sampling_frequency, blocks, zscore_per_tetrode):
    """Block-wise reference: envelope and 4 ms smoothing inside each block,
    per-tetrode z-score (ddof=1) pooled over all valid rows, then the median."""
    filtered_lfps = np.asarray(filtered_lfps, dtype=float)
    smoothed = np.full_like(filtered_lfps, np.nan)
    for start, stop in blocks:
        env = get_envelope(filtered_lfps[start:stop])
        smoothed[start:stop] = gaussian_smooth(env, 0.004, sampling_frequency)
    valid = np.all(np.isfinite(smoothed), axis=1)
    if zscore_per_tetrode:
        mean = smoothed[valid].mean(axis=0, keepdims=True)
        std = smoothed[valid].std(axis=0, ddof=1, keepdims=True)
        smoothed = (smoothed - mean) / std
    consensus = np.full(len(smoothed), np.nan)
    consensus[valid] = np.median(smoothed[valid], axis=1)
    return consensus


class TestYuConsensusTrace:
    """Median of per-tetrode smoothed (and z-scored) envelopes, Yu et al. 2017."""

    @pytest.fixture
    def triple_lfp(self, time_3s):
        lfps = [
            simulate_LFP(
                time_3s,
                ripple_times=[1.1],
                noise_amplitude=1.2,
                ripple_amplitude=1.5,
                random_state=s,
            )
            for s in (11, 12, 13)
        ]
        return filter_ripple_band(np.column_stack(lfps))

    def test_shape_and_finite_on_clean_input(self, time_3s, triple_lfp, sampling_frequency):
        consensus = get_Yu_ripple_consensus_trace(triple_lfp, sampling_frequency)
        assert consensus.shape == (len(time_3s),)
        assert np.all(np.isfinite(consensus))

    def test_flag_off_is_median_of_smoothed_envelopes(self, triple_lfp, sampling_frequency):
        expected = _yu_reference_consensus(
            triple_lfp, sampling_frequency, [(0, len(triple_lfp))], zscore_per_tetrode=False
        )
        consensus = get_Yu_ripple_consensus_trace(
            triple_lfp, sampling_frequency, zscore_per_tetrode=False
        )
        np.testing.assert_allclose(consensus, expected, rtol=1e-12, atol=1e-12)

    def test_flag_on_is_median_of_zscored_smoothed_envelopes(
        self, triple_lfp, sampling_frequency
    ):
        expected = _yu_reference_consensus(
            triple_lfp, sampling_frequency, [(0, len(triple_lfp))], zscore_per_tetrode=True
        )
        consensus = get_Yu_ripple_consensus_trace(triple_lfp, sampling_frequency)
        np.testing.assert_allclose(consensus, expected, rtol=1e-12, atol=1e-12)

    def test_per_tetrode_zscore_removes_channel_gain(self, triple_lfp, sampling_frequency):
        scaled = triple_lfp.copy()
        scaled[:, 0] *= 10.0
        with_flag = get_Yu_ripple_consensus_trace(scaled, sampling_frequency)
        reference = get_Yu_ripple_consensus_trace(triple_lfp, sampling_frequency)
        np.testing.assert_allclose(with_flag, reference, rtol=1e-10, atol=1e-10)
        without_flag = get_Yu_ripple_consensus_trace(
            scaled, sampling_frequency, zscore_per_tetrode=False
        )
        assert not np.allclose(without_flag, reference)

    def test_nan_rows_split_blocks_and_stay_nan(self, triple_lfp, sampling_frequency):
        lfps = triple_lfp.copy()
        gap = slice(2000, 2300)
        lfps[gap, 1] = np.nan  # one channel missing makes the whole row invalid
        consensus = get_Yu_ripple_consensus_trace(lfps, sampling_frequency)
        assert np.all(np.isnan(consensus[gap]))
        assert np.all(np.isfinite(consensus[:2000]))
        assert np.all(np.isfinite(consensus[2300:]))
        expected = _yu_reference_consensus(
            lfps, sampling_frequency, [(0, 2000), (2300, len(lfps))], zscore_per_tetrode=True
        )
        np.testing.assert_allclose(consensus, expected, rtol=1e-12, atol=1e-12)

    def test_timestamp_gap_splits_blocks(self, time_3s, triple_lfp, sampling_frequency):
        # rows are contiguous in the array but time jumps by one second at row 2000
        time = time_3s.copy()
        time[2000:] += 1.0
        consensus = get_Yu_ripple_consensus_trace(triple_lfp, sampling_frequency, time=time)
        expected = _yu_reference_consensus(
            triple_lfp,
            sampling_frequency,
            [(0, 2000), (2000, len(triple_lfp))],
            zscore_per_tetrode=True,
        )
        np.testing.assert_allclose(consensus, expected, rtol=1e-12, atol=1e-12)
        contiguous = get_Yu_ripple_consensus_trace(triple_lfp, sampling_frequency)
        assert not np.allclose(consensus, contiguous)

    def test_single_tetrode_ripple_moves_median_less_than_kay_trace(
        self, time_3s, sampling_frequency
    ):
        quiet = [
            simulate_LFP(
                time_3s,
                ripple_times=[],
                noise_amplitude=1.2,
                ripple_amplitude=1.5,
                random_state=s,
            )
            for s in (21, 22, 23, 24)
        ]
        loud = simulate_LFP(
            time_3s,
            ripple_times=[1.5],
            noise_amplitude=1.2,
            ripple_amplitude=6.0,
            random_state=25,
        )
        lfps = filter_ripple_band(np.column_stack([*quiet, loud]))
        yu = get_Yu_ripple_consensus_trace(lfps, sampling_frequency)
        kay = get_Kay_ripple_consensus_trace(lfps, sampling_frequency)
        window = (time_3s > 1.45) & (time_3s < 1.55)
        baseline = time_3s < 1.0

        def peak_z(trace):
            return (trace[window].max() - trace[baseline].mean()) / trace[baseline].std()

        assert peak_z(yu) < 0.5 * peak_z(kay)

    def test_rejects_non_2d_input(self, sampling_frequency):
        with pytest.raises(ValueError, match="2D"):
            get_Yu_ripple_consensus_trace(np.zeros(100), sampling_frequency)

    def test_time_length_mismatch_raises(self, triple_lfp, sampling_frequency):
        with pytest.raises(ValueError, match="time"):
            get_Yu_ripple_consensus_trace(
                triple_lfp, sampling_frequency, time=np.arange(10) / sampling_frequency
            )


class TestExtractYuRippleEvents:
    """Sample-count qualification and mean-crossing extension, Yu et al. 2017."""

    @staticmethod
    def _trace(n_time, above_zero, above_threshold, level=1.0, high=5.0):
        """Build a normalized trace: 0.5 inside above-zero runs, `high` inside
        above-threshold runs, -0.5 elsewhere, and exactly 0 where requested."""
        trace = np.full(n_time, -0.5)
        for start, stop in above_zero:
            trace[start:stop] = 0.5 * level
        for start, stop in above_threshold:
            trace[start:stop] = high
        return trace

    @pytest.mark.parametrize(
        ("sampling_frequency", "n_samples", "qualifies"),
        [
            (1000, 19, False),
            (1000, 20, True),
            (1000, 21, True),
            (1500, 29, False),
            (1500, 30, True),
        ],
    )
    def test_minimum_duration_counts_samples(self, sampling_frequency, n_samples, qualifies):
        n_time = 500
        time = np.arange(n_time) / sampling_frequency
        trace = self._trace(
            n_time, above_zero=[(100, 300)], above_threshold=[(150, 150 + n_samples)]
        )
        events, _, _ = _extract_Yu_ripple_events(trace, time, sampling_frequency, 0.020, 3.0)
        assert (len(events) == 1) == qualifies

    def test_extends_to_containing_above_zero_run(self):
        fs = 1000
        n_time = 500
        time = np.arange(n_time) / fs
        trace = self._trace(n_time, above_zero=[(100, 300)], above_threshold=[(150, 200)])
        events, _, _ = _extract_Yu_ripple_events(trace, time, fs, 0.020, 3.0)
        np.testing.assert_allclose(events, [[time[100], time[299]]])

    def test_zero_sample_ends_a_run(self):
        fs = 1000
        n_time = 500
        time = np.arange(n_time) / fs
        trace = self._trace(n_time, above_zero=[(100, 300)], above_threshold=[(150, 200)])
        trace[250] = 0.0  # equality to the mean ends the run
        events, _, _ = _extract_Yu_ripple_events(trace, time, fs, 0.020, 3.0)
        np.testing.assert_allclose(events, [[time[100], time[249]]])

    def test_two_exceedances_in_one_run_yield_one_event(self):
        fs = 1000
        n_time = 500
        time = np.arange(n_time) / fs
        trace = self._trace(
            n_time, above_zero=[(100, 300)], above_threshold=[(120, 160), (220, 260)]
        )
        events, _, n_supra = _extract_Yu_ripple_events(trace, time, fs, 0.020, 3.0)
        assert len(events) == 1
        assert n_supra[0] == 40  # the longest qualifying run

    def test_sub_minimum_exceedance_does_not_create_event(self):
        fs = 1000
        n_time = 500
        time = np.arange(n_time) / fs
        trace = self._trace(n_time, above_zero=[(100, 300)], above_threshold=[(150, 160)])
        events, _, _ = _extract_Yu_ripple_events(trace, time, fs, 0.020, 3.0)
        assert len(events) == 0

    def test_clipped_at_block_edges_is_flagged(self):
        fs = 1000
        n_time = 200
        time = np.arange(n_time) / fs
        # above zero from the very first sample to the last: clipped both sides
        trace = self._trace(n_time, above_zero=[(0, 200)], above_threshold=[(50, 100)])
        events, clipped, _ = _extract_Yu_ripple_events(trace, time, fs, 0.020, 3.0)
        np.testing.assert_allclose(events, [[time[0], time[-1]]])
        assert clipped.tolist() == [[True, True]]
        trace = self._trace(n_time, above_zero=[(20, 200)], above_threshold=[(50, 100)])
        _, clipped, _ = _extract_Yu_ripple_events(trace, time, fs, 0.020, 3.0)
        assert clipped.tolist() == [[False, True]]

    def test_uses_native_timestamps(self):
        fs = 1000
        n_time = 500
        time = 100.0 + np.arange(n_time) / fs + 1e-4 * np.sin(np.arange(n_time))
        trace = self._trace(n_time, above_zero=[(100, 300)], above_threshold=[(150, 200)])
        events, _, _ = _extract_Yu_ripple_events(trace, time, fs, 0.020, 3.0)
        assert events[0, 0] == time[100]
        assert events[0, 1] == time[299]

    @pytest.mark.parametrize("threshold", [0.0, -1.0, np.nan, np.inf])
    def test_invalid_threshold_raises(self, threshold):
        fs = 1000
        time = np.arange(100) / fs
        with pytest.raises(ValueError, match="threshold"):
            _extract_Yu_ripple_events(np.zeros(100), time, fs, 0.020, threshold)

    def test_empty_result_shapes(self):
        fs = 1000
        time = np.arange(100) / fs
        events, clipped, n_supra = _extract_Yu_ripple_events(
            np.full(100, -0.5), time, fs, 0.020, 3.0
        )
        assert events.shape == (0, 2)
        assert clipped.shape == (0, 2)
        assert n_supra.shape == (0,)


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

    def test_integer_lfp_not_truncated(
        self, time_3s, dual_lfp_with_cooccur_ripples, stationary_speed, sampling_frequency
    ):
        """Integer LFP input is cast to float, not truncated, through the pipeline."""
        lfp_int = np.round(dual_lfp_with_cooccur_ripples * 100).astype(np.int32)

        filtered = filter_ripple_band(lfp_int)
        # Before the fix, filter output kept the integer dtype (truncated values).
        assert np.issubdtype(filtered.dtype, np.floating)

        # Kay consensus stays finite (no int-overflow -> sqrt(negative) -> NaN)
        # and the ripples are still detected.
        ripples = Kay_ripple_detector(time_3s, filtered, stationary_speed, sampling_frequency)
        assert isinstance(ripples, pd.DataFrame)
        assert len(ripples) >= 1

    def test_normalization_mask_with_nan_rows(
        self, time_3s, dual_lfp_with_ripples, stationary_speed, sampling_frequency
    ):
        """normalization_mask stays position-aligned after NaN rows are removed."""
        filtered_lfps = filter_ripple_band(dual_lfp_with_ripples)
        filtered_lfps[100:150, :] = np.nan

        # A mask with genuine False entries (not coinciding with the NaN rows),
        # spanning the original pre-NaN-removal time samples. Before the fix this
        # raised a length-mismatch ValueError once NaN rows were dropped.
        normalization_mask = np.ones(len(time_3s), dtype=bool)
        normalization_mask[1000:1500] = False

        ripples = Kay_ripple_detector(
            time_3s,
            filtered_lfps,
            stationary_speed,
            sampling_frequency,
            normalization_mask=normalization_mask,
        )
        assert isinstance(ripples, pd.DataFrame)

        # Equivalence check: manually dropping the NaN rows and slicing the mask
        # to match must give an identical result. This confirms the internal
        # filtering keeps the mask aligned by position, not merely by length -- an
        # off-by-rows misalignment would shift the normalization window and change
        # the z-scores, so the two runs would diverge.
        # (mirror the detector's own not_null: NaN in the LFP *or* the speed)
        not_null = np.all(~np.isnan(filtered_lfps), axis=1) & ~np.isnan(stationary_speed)
        ripples_manual = Kay_ripple_detector(
            time_3s[not_null],
            filtered_lfps[not_null],
            stationary_speed[not_null],
            sampling_frequency,
            normalization_mask=normalization_mask[not_null],
        )
        pd.testing.assert_frame_equal(ripples, ripples_manual)

    def test_normalization_mask_selects_no_samples(
        self, time_3s, dual_lfp_with_ripples, stationary_speed, sampling_frequency
    ):
        """An all-False normalization_mask raises rather than silently returning
        an empty result from a degenerate normalization."""
        filtered_lfps = filter_ripple_band(dual_lfp_with_ripples)
        normalization_mask = np.zeros(len(time_3s), dtype=bool)

        with pytest.raises(ValueError, match="selects no samples"):
            Kay_ripple_detector(
                time_3s,
                filtered_lfps,
                stationary_speed,
                sampling_frequency,
                normalization_mask=normalization_mask,
            )

    def test_normalization_mask_wrong_length(
        self, time_3s, dual_lfp_with_ripples, stationary_speed, sampling_frequency
    ):
        """A normalization_mask whose length doesn't match the data raises."""
        filtered_lfps = filter_ripple_band(dual_lfp_with_ripples)
        with pytest.raises(ValueError, match="normalization_mask length"):
            Kay_ripple_detector(
                time_3s,
                filtered_lfps,
                stationary_speed,
                sampling_frequency,
                normalization_mask=np.ones(len(time_3s) - 5, dtype=bool),
            )

    def test_manual_normalization_warns_on_degenerate_channel(
        self, time_3s, dual_lfp_with_cooccur_ripples, stationary_speed, sampling_frequency
    ):
        """normalize_signal_manually zeroes a NaN-baseline channel and warns."""
        filtered_lfps = filter_ripple_band(dual_lfp_with_cooccur_ripples)
        n_channels = filtered_lfps.shape[1]
        baselines = np.zeros(n_channels)
        baselines[1] = np.nan  # degenerate channel
        deviations = np.ones(n_channels)

        with pytest.warns(UserWarning, match="Zeroing channel"):
            Shvartsman_ripple_detector(
                time_3s,
                filtered_lfps,
                stationary_speed,
                sampling_frequency,
                manual_normalization=True,
                elec_baselines=baselines,
                elec_deviations=deviations,
            )

    def test_manual_norm_ignores_normalization_mask(
        self, time_3s, dual_lfp_with_cooccur_ripples, stationary_speed, sampling_frequency
    ):
        """normalization_mask is ignored (not validated) under manual normalization."""
        filtered_lfps = filter_ripple_band(dual_lfp_with_cooccur_ripples)
        env = gaussian_smooth(
            get_envelope(filtered_lfps), sigma=0.004, sampling_frequency=sampling_frequency
        )
        # A wrong-length, all-False mask that would raise if it were validated --
        # but the docstring promises it is ignored under manual normalization.
        bad_mask = np.zeros(len(time_3s) - 5, dtype=bool)

        ripples = Shvartsman_ripple_detector(
            time_3s,
            filtered_lfps,
            stationary_speed,
            sampling_frequency,
            manual_normalization=True,
            elec_baselines=env.mean(axis=0),
            elec_deviations=env.std(axis=0),
            normalization_mask=bad_mask,
        )
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


class TestShvartsmanParticipationSemantics:
    """Preserve participation across merged events and subsequent exclusions."""

    @pytest.mark.parametrize("participation_threshold", [3, 1.0])
    def test_chain_of_overlapping_ripples_counts_all_electrodes(
        self, time_3s, stationary_speed, sampling_frequency, participation_threshold
    ):
        """A-B and B-C overlap counts all three electrodes in the merged event."""
        lfps = np.column_stack(
            [
                simulate_LFP(
                    time_3s,
                    [center],
                    noise_amplitude=1.2,
                    ripple_amplitude=1.5,
                    random_state=seed,
                )
                for center, seed in [(1.1, 5), (1.15, 6), (1.2, 7)]
            ]
        )
        filtered = filter_ripple_band(lfps)
        individual = [
            Karlsson_ripple_detector(
                time_3s, filtered[:, [channel]], stationary_speed, sampling_frequency
            )
            for channel in range(3)
        ]
        assert all(len(events) == 1 for events in individual)
        first, middle, last = [events.iloc[0] for events in individual]
        # There is never a three-electrode overlap, but the intervals form one event.
        assert (
            middle["start_time"]
            <= first["end_time"]
            < last["start_time"]
            <= middle["end_time"]
        )

        ripples = Shvartsman_ripple_detector(
            time_3s,
            filtered,
            stationary_speed,
            sampling_frequency,
            participation_threshold=participation_threshold,
        )

        assert len(ripples) == 1
        event = ripples.iloc[0]
        assert event["start_time"] == first["start_time"]
        assert event["end_time"] == last["end_time"]
        assert event["participants"] == {0, 1, 2}
        assert all(type(channel) is int for channel in event["participants"])
        assert event["n_participants"] == len(event["participants"]) == 3
        assert event["frac_participants"] == 1.0

    def test_participant_metadata_stays_aligned_after_exclusion(
        self, time_3s, stationary_speed, sampling_frequency
    ):
        """When movement exclusion drops an event, the survivor keeps its OWN
        participant count/set. Guards the ``participant_sets[included_ripple_inds]``
        bookkeeping, which uniform-participation fixtures cannot exercise."""
        # Channel 0 ripples at 1.1s and 2.1s; channels 1 and 2 only at 1.1s, so the
        # event near 1.1s has 3 participants and the event near 2.1s has just 1.
        ch0 = simulate_LFP(
            time_3s, [1.1, 2.1], noise_amplitude=1.2, ripple_amplitude=1.5, random_state=0
        )
        ch1 = simulate_LFP(
            time_3s, [1.1], noise_amplitude=1.2, ripple_amplitude=1.5, random_state=1
        )
        ch2 = simulate_LFP(
            time_3s, [1.1], noise_amplitude=1.2, ripple_amplitude=1.5, random_state=2
        )
        filtered = filter_ripple_band(np.column_stack([ch0, ch1, ch2]))

        # Sanity: with no movement both events survive with differing participation.
        both = Shvartsman_ripple_detector(
            time_3s, filtered, stationary_speed, sampling_frequency, participation_threshold=0
        )
        assert both["n_participants"].tolist() == [3, 1]

        # A movement burst covering only the first (3-participant) event excludes it.
        speed = np.asarray(stationary_speed, dtype=float).copy()
        speed[(time_3s >= 0.95) & (time_3s <= 1.35)] = 100.0
        ripples = Shvartsman_ripple_detector(
            time_3s,
            filtered,
            speed,
            sampling_frequency,
            participation_threshold=0,
            speed_threshold=4.0,
        )
        # Only the 1-participant event near 2.1s survives, and it must carry its own
        # metadata (a misaligned index would report the excluded event's count of 3).
        assert len(ripples) == 1
        assert ripples["n_participants"].iloc[0] == 1
        assert ripples["participants"].iloc[0] == {0}

    def test_participation_threshold_fraction_means_all_channels(
        self,
        time_3s,
        dual_lfp_with_ripples,
        dual_lfp_with_cooccur_ripples,
        stationary_speed,
        sampling_frequency,
    ):
        """A fractional participation_threshold is a fraction of channels, and 1.0
        means *all* channels (not one)."""
        # The two channels ripple at separate times, so each event has only 1 of 2.
        filtered_sep = filter_ripple_band(dual_lfp_with_ripples)
        # 1.0 requires both channels in one event -> excluded (each has one).
        assert Shvartsman_ripple_detector(
            time_3s,
            filtered_sep,
            stationary_speed,
            sampling_frequency,
            participation_threshold=1.0,
        ).empty
        # 0.5 requires 1 of 2 -> detected (a genuine fraction in (0, 1)).
        assert not Shvartsman_ripple_detector(
            time_3s,
            filtered_sep,
            stationary_speed,
            sampling_frequency,
            participation_threshold=0.5,
        ).empty

        # Co-occurring ripples involve both channels, so 1.0 (all 2) detects them.
        filtered_co = filter_ripple_band(dual_lfp_with_cooccur_ripples)
        assert not Shvartsman_ripple_detector(
            time_3s,
            filtered_co,
            stationary_speed,
            sampling_frequency,
            participation_threshold=1.0,
        ).empty

    def test_degenerate_channel_zeroed_and_counts_in_denominator(
        self, time_3s, dual_lfp_with_cooccur_ripples, stationary_speed, sampling_frequency
    ):
        """A degenerate (NaN-baseline) channel is zeroed so it never participates,
        yet still counts in the frac_participants denominator."""
        filtered_lfps = filter_ripple_band(dual_lfp_with_cooccur_ripples)
        env = gaussian_smooth(
            get_envelope(filtered_lfps), sigma=0.004, sampling_frequency=sampling_frequency
        )
        baselines = env.mean(axis=0).copy()
        deviations = env.std(axis=0).copy()
        baselines[1] = np.nan  # channel 1 degenerate

        with pytest.warns(UserWarning, match="Zeroing channel"):
            ripples = Shvartsman_ripple_detector(
                time_3s,
                filtered_lfps,
                stationary_speed,
                sampling_frequency,
                manual_normalization=True,
                elec_baselines=baselines,
                elec_deviations=deviations,
                participation_threshold=0,
            )

        assert len(ripples) > 0
        # The dead channel never appears among participants...
        assert all(1 not in participants for participants in ripples["participants"])
        assert all(ripples["n_participants"] == 1)
        # ...but the denominator still includes it (1 of 2 channels).
        assert all(ripples["frac_participants"] == 0.5)


class TestFindMaxThresh:
    def test_respects_minimum_duration(self):
        """max_thresh is the largest value sustained for minimum_duration, so a
        longer required duration yields a smaller (or equal) result. Peak at
        index 0 -> only rightward expansion."""
        time = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
        data = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        assert _find_max_thresh(time, data, minimum_duration=0.005) == 8.0
        assert _find_max_thresh(time, data, minimum_duration=0.015) == 6.0
        assert _find_max_thresh(time, data, minimum_duration=0.035) == 2.0

    def test_mid_peak_expands_both_directions(self):
        """A mid-array peak exercises the leftward-expansion branch and the
        neighbour tie-break. From peak 10 at index 2: the window first steps left
        (neighbour 5 > 3), then right (3 > 1), spanning indices 1..3 -> min(5, 3)."""
        time = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
        data = np.array([1.0, 5.0, 10.0, 3.0, 2.0])
        assert _find_max_thresh(time, data, minimum_duration=0.015) == 3.0
        assert _find_max_thresh(time, data, minimum_duration=0.035) == 1.0

    def test_peak_at_last_index_expands_left(self):
        """Peak at the last index forces leftward-only expansion."""
        time = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
        data = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        assert _find_max_thresh(time, data, minimum_duration=0.015) == 6.0
        assert _find_max_thresh(time, data, minimum_duration=0.035) == 2.0

    def test_all_equal_data(self):
        """A flat plateau returns the (shared) value."""
        time = np.array([0.0, 0.01, 0.02, 0.03])
        data = np.array([5.0, 5.0, 5.0, 5.0])
        assert _find_max_thresh(time, data, minimum_duration=0.015) == 5.0

    def test_two_sample_event_long_enough(self):
        """A two-sample event already spanning minimum_duration needs no expansion
        and returns the min of its endpoints."""
        time = np.array([0.0, 0.02])
        data = np.array([10.0, 0.0])
        assert _find_max_thresh(time, data, minimum_duration=0.015) == 0.0

    def test_exact_minimum_duration_does_not_expand_further(self):
        """Rounding at the duration boundary must not include a lower next sample."""
        time = np.arange(200, 222) / 1000
        data = np.arange(22.0, 0.0, -1.0)
        # The window [0.200, 0.220] meets the detector's 20 ms duration rule,
        # although subtracting its endpoints gives 0.01999999999999999.
        assert _find_max_thresh(time, data, minimum_duration=0.02) == 2.0

    def test_short_event_returns_nan(self):
        """An event shorter than minimum_duration cannot sustain the threshold, so
        the value is undefined -> nan (previously ran an index out of bounds)."""
        time = np.array([0.0, 0.001])
        data = np.array([10.0, 0.0])
        assert np.isnan(_find_max_thresh(time, data, minimum_duration=0.015))

    def test_single_sample_event_returns_nan(self):
        """A one-sample event cannot sustain any duration -> nan (no out-of-bounds)."""
        time = np.array([1.0])
        data = np.array([7.0])
        assert np.isnan(_find_max_thresh(time, data, minimum_duration=0.015))


class TestMaxThreshMinimumDuration:
    """Karlsson and multiunit_HSE must honour the caller's minimum_duration for
    max_thresh, not silently fall back to the 15 ms default."""

    @staticmethod
    def _spy_on_find_max_thresh():
        """Patch _find_max_thresh to record the minimum_duration it receives while
        still delegating to the real implementation."""
        real = detectors_module._find_max_thresh
        seen: list[float] = []

        def spy(time, data, minimum_duration=0.015):
            seen.append(minimum_duration)
            return real(time, data, minimum_duration)

        return patch.object(detectors_module, "_find_max_thresh", spy), seen

    def test_karlsson_forwards_minimum_duration_to_max_thresh(
        self, time_3s, dual_lfp_with_cooccur_ripples, stationary_speed, sampling_frequency
    ):
        filtered_lfps = filter_ripple_band(dual_lfp_with_cooccur_ripples)
        patcher, seen = self._spy_on_find_max_thresh()
        with patcher:
            ripples = Karlsson_ripple_detector(
                time_3s,
                filtered_lfps,
                stationary_speed,
                sampling_frequency,
                minimum_duration=0.005,
            )
        assert len(ripples) > 0
        # max_thresh must be computed with the caller's 0.005, not the 0.015 default
        # (this fails if the minimum_duration argument is dropped from the call).
        assert seen and all(md == 0.005 for md in seen)

    def test_hse_forwards_minimum_duration_to_max_thresh(self, time_3s, sampling_frequency):
        multiunit = np.zeros((len(time_3s), 5))
        idx = int(1.0 * sampling_frequency)
        multiunit[idx : idx + 30, :] = 1
        speed = np.ones(len(time_3s)) * 2.0
        patcher, seen = self._spy_on_find_max_thresh()
        with patcher:
            hse = multiunit_HSE_detector(
                time_3s, multiunit, speed, sampling_frequency, minimum_duration=0.005
            )
        assert len(hse) > 0
        assert seen and all(md == 0.005 for md in seen)

    def test_hse_tiny_minimum_duration_is_bounds_safe(self, time_3s, sampling_frequency):
        # A sharp, few-sample synchrony burst detected with a 1 ms minimum used to
        # crash inside max_thresh because it fell back to the 15 ms window.
        multiunit = np.zeros((len(time_3s), 5))
        idx = int(1.0 * sampling_frequency)
        multiunit[idx : idx + 3, :] = 1
        speed = np.ones(len(time_3s)) * 2.0
        hse = multiunit_HSE_detector(
            time_3s, multiunit, speed, sampling_frequency, minimum_duration=0.001
        )
        assert len(hse) > 0
        assert np.all(np.isfinite(hse["max_thresh"].to_numpy()))

    def test_hse_exact_minimum_duration_has_finite_max_thresh(self):
        """A detected event exactly at the duration boundary has a defined threshold."""
        time = np.arange(1000) / 1000
        multiunit = np.zeros((len(time), 5))
        multiunit[200:221] = 1
        hse = multiunit_HSE_detector(
            time,
            multiunit,
            np.zeros(len(time)),
            sampling_frequency=1000,
            minimum_duration=0.02,
            smoothing_sigma=1e-5,  # Preserve the 21-sample plateau.
        )

        assert len(hse) == 1
        np.testing.assert_allclose(hse[["start_time", "end_time"]], [[0.2, 0.22]])
        # For a binary plateau occupying p=21/1000 samples, its z-score is
        # (1-p) / sqrt(p*(1-p)) = sqrt(979/21).
        assert hse["max_thresh"].iloc[0] == pytest.approx(np.sqrt(979 / 21))

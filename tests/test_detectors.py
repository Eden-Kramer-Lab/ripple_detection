"""Integration tests for ripple detection algorithms."""

import contextlib
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from _synthetic import (
    _synthetic_joint_inputs,
    _synthetic_ripple_band,
    _synthetic_two_channel_lfp,
)

from ripple_detection import (
    Carey_candidate_detector,
    Karlsson_ripple_detector,
    Kay_ripple_detector,
    Long_sharp_wave_ripple_detector,
    Shvartsman_ripple_detector,
    Yu_ripple_detector,
    Zugaro_ripple_detector,
    filter_ripple_band,
)
from ripple_detection.core import (
    gaussian_smooth,
    get_envelope,
    minimum_sample_count,
)
from ripple_detection.detectors import (
    Roumis_ripple_detector,
    get_Kay_ripple_consensus_trace,
    get_Yu_ripple_consensus_trace,
    multiunit_HSE_detector,
)
from ripple_detection.detectors import _events as events_module
from ripple_detection.detectors._carey import _contained_in_intervals, _state_intervals
from ripple_detection.detectors._events import (
    _count_active_units,
    _exclude_long_events,
    _get_event_stats,
    _max_sustained_zscore,
)
from ripple_detection.detectors._lfp import _extract_Yu_ripple_events
from ripple_detection.detectors._long import _firfilt
from ripple_detection.detectors._zugaro import _two_threshold_events, _zugaro_smoothing_window
from ripple_detection.simulate import simulate_LFP


class TestShvartsmanRippleDetector:
    def test_single_channel_with_ripples(
        self, time_3s, single_lfp_with_ripples, stationary_speed, sampling_frequency
    ):
        """One channel cannot meet the default two-channel participation minimum,
        so the detector says so rather than returning an empty frame."""
        filtered_lfps = filter_ripple_band(single_lfp_with_ripples, 1500)
        with pytest.raises(ValueError, match="no event could be kept"):
            Shvartsman_ripple_detector(
                time_3s, filtered_lfps, stationary_speed, sampling_frequency
            )

    def test_single_channel_with_ripples_no_participation_criterion(
        self, time_3s, single_lfp_with_ripples, stationary_speed, sampling_frequency
    ):
        """A single channel can only produce events when no participation is required."""
        filtered_lfps = filter_ripple_band(single_lfp_with_ripples, 1500)
        ripples = Shvartsman_ripple_detector(
            time_3s,
            filtered_lfps,
            stationary_speed,
            sampling_frequency,
            minimum_participating_channels=0,
        )

        # Verify output structure
        assert isinstance(ripples, pd.DataFrame)
        assert len(ripples) > 0, "Should detect at least one ripple"

        # Check required columns
        expected_columns = [
            "start_time",
            "end_time",
            "duration",
            "max_sustained_zscore",
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
        filtered_lfps = filter_ripple_band(dual_lfp_with_ripples, 1500)
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
        filtered_lfps = filter_ripple_band(dual_lfp_with_cooccur_ripples, 1500)
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
            "max_sustained_zscore",
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
        filtered_lfps = filter_ripple_band(dual_lfp_close_ripples, 1500)

        ripples = Shvartsman_ripple_detector(
            time_3s, filtered_lfps, stationary_speed, sampling_frequency
        )
        assert isinstance(ripples, pd.DataFrame)
        assert len(ripples) == 2
        assert all(ripples["n_participants"] == 2)
        assert all(participants == (0, 1) for participants in ripples["participants"])

    def test_multi_channel_sparse_ripples(
        self, time_3s, multi_lfp_sparse_ripples, stationary_speed, sampling_frequency
    ):
        """Test Shvartsman detector with many LFP channels with a subset having non-overlapping ripples."""
        filtered_lfps = filter_ripple_band(multi_lfp_sparse_ripples, 1500)
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
        filtered_lfps = filter_ripple_band(multi_lfp_sparse_cooccur_ripples, 1500)
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
            "max_sustained_zscore",
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
        assert all(p == (0, 1) for p in ripples["participants"]), (
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
        filtered_lfps = filter_ripple_band(lfp_no_ripples, 1500)
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
        filtered_lfps = filter_ripple_band(dual_lfp_with_cooccur_ripples, 1500)

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
        filtered_lfps = filter_ripple_band(dual_lfp_with_cooccur_ripples, 1500)

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
        filtered_lfps = filter_ripple_band(dual_lfp_with_cooccur_ripples, 1500)

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
        filtered_lfps = filter_ripple_band(dual_lfp_with_cooccur_short_ripples, 1500)

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
        filtered_lfps = filter_ripple_band(multi_lfp_sparse_cooccur_ripples, 1500)

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
        filtered_lfps = filter_ripple_band(dual_lfp_with_close_cooccur_ripples, 1500)

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
        filtered_lfps = filter_ripple_band(multi_lfp_sparse_cooccur_ripples, 1500)
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
            normalization_method="manual",
            channel_baselines=env.mean(axis=0),
            channel_deviations=env.std(axis=0),
        )
        assert isinstance(ripples, pd.DataFrame)
        assert len(ripples) == 2
        assert all(ripples["n_participants"] == 2)

    def test_manual_norm_no_baseline_inputs(
        self, time_3s, multi_lfp_sparse_cooccur_ripples, stationary_speed, sampling_frequency
    ):
        """Test Shvartsman detector with manual normalization indicated but no baseline values passed in."""
        filtered_lfps = filter_ripple_band(multi_lfp_sparse_cooccur_ripples, 1500)
        with pytest.raises(ValueError, match="needs channel_baselines"):
            Shvartsman_ripple_detector(
                time_3s,
                filtered_lfps,
                stationary_speed,
                sampling_frequency,
                normalization_method="manual",
            )

    def test_manual_norm_baseline_deviation_mismatch(
        self, time_3s, multi_lfp_sparse_cooccur_ripples, stationary_speed, sampling_frequency
    ):
        """Test Shvartsman detector with manual normalization indicated but mismatched channel_baselines and channel_deviations lengths."""
        filtered_lfps = filter_ripple_band(multi_lfp_sparse_cooccur_ripples, 1500)
        with pytest.raises(ValueError, match="same shape"):
            Shvartsman_ripple_detector(
                time_3s,
                filtered_lfps,
                stationary_speed,
                sampling_frequency,
                normalization_method="manual",
                channel_baselines=np.ones(filtered_lfps.shape[1]),
                channel_deviations=np.ones(filtered_lfps.shape[1] - 1),
            )

    def test_manual_norm_lfp_baseline_mismatch(
        self, time_3s, multi_lfp_sparse_cooccur_ripples, stationary_speed, sampling_frequency
    ):
        """Test Shvartsman detector with manual normalization indicated but mismatched channel_baselines and filtered_lfp lengths."""
        filtered_lfps = filter_ripple_band(multi_lfp_sparse_cooccur_ripples, 1500)
        with pytest.raises(ValueError, match="one entry per channel"):
            Shvartsman_ripple_detector(
                time_3s,
                filtered_lfps,
                stationary_speed,
                sampling_frequency,
                normalization_method="manual",
                channel_baselines=np.ones(filtered_lfps.shape[1] - 1),
                channel_deviations=np.ones(filtered_lfps.shape[1] - 1),
            )


class TestKayRippleDetector:
    """Test suite for Kay ripple detector."""

    def test_single_channel_with_ripples(
        self, time_3s, single_lfp_with_ripples, stationary_speed, sampling_frequency
    ):
        """Test Kay detector with single LFP channel containing ripples."""
        filtered_lfps = filter_ripple_band(single_lfp_with_ripples, 1500)
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
            "max_sustained_zscore",
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
        filtered_lfps = filter_ripple_band(dual_lfp_with_ripples, 1500)
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
        filtered_lfps = filter_ripple_band(dual_lfp_close_ripples, 1500)
        ripples = Kay_ripple_detector(
            time_3s, filtered_lfps, stationary_speed, sampling_frequency
        )

        assert isinstance(ripples, pd.DataFrame)
        assert len(ripples) > 0

    def test_multi_channel_sparse_ripples(
        self, time_3s, multi_lfp_sparse_ripples, stationary_speed, sampling_frequency
    ):
        """Test with many channels but ripples only in subset."""
        filtered_lfps = filter_ripple_band(multi_lfp_sparse_ripples, 1500)
        ripples = Kay_ripple_detector(
            time_3s, filtered_lfps, stationary_speed, sampling_frequency
        )

        assert isinstance(ripples, pd.DataFrame)
        # Should still detect ripples even with many noise channels
        assert len(ripples) > 0

    def test_no_ripples(self, time_3s, lfp_no_ripples, stationary_speed, sampling_frequency):
        """Test with noise-only signal (no ripples)."""
        filtered_lfps = filter_ripple_band(lfp_no_ripples, 1500)
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
        filtered_lfps = filter_ripple_band(dual_lfp_with_ripples, 1500)

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
        filtered_lfps = filter_ripple_band(lfp_short_duration_ripples, 1500)

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
        filtered_lfps = filter_ripple_band(single_lfp_with_ripples, 1500)

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
        filtered_lfps = filter_ripple_band(dual_lfp_close_ripples, 1500)

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


class TestShvartsmanNeedsEnoughChannels:
    """The default asks two channels to agree; with one channel no event could
    ever be kept, and returning an empty frame would look like a quiet recording."""

    FS = 1500
    N_TIME = 6000

    def test_one_channel_at_the_default_raises(self, time, stationary):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(2000, 2100, 20.0)])[:, :1]
        with pytest.raises(ValueError, match="no event could be kept"):
            Shvartsman_ripple_detector(time, lfps, stationary, self.FS)

    def test_one_channel_with_a_minimum_of_one_detects(self, time, stationary):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(2000, 2100, 20.0)])[:, :1]
        events = Shvartsman_ripple_detector(
            time, lfps, stationary, self.FS, minimum_participating_channels=1
        )
        assert len(events) == 1

    def test_a_fraction_is_not_checked_against_the_channel_count(self, time, stationary):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(2000, 2100, 20.0)])[:, :1]
        events = Shvartsman_ripple_detector(
            time, lfps, stationary, self.FS, minimum_participating_fraction=1.0
        )
        assert len(events) == 1


class TestKarlssonRippleDetector:
    """Test suite for Karlsson ripple detector."""

    def test_single_channel_with_ripples(
        self, time_3s, single_lfp_with_ripples, stationary_speed, sampling_frequency
    ):
        """Test Karlsson detector with single LFP channel."""
        filtered_lfps = filter_ripple_band(single_lfp_with_ripples, 1500)
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
        filtered_lfps = filter_ripple_band(dual_lfp_with_ripples, 1500)
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
        filtered_lfps = filter_ripple_band(single_lfp_with_ripples, 1500)

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
        filtered_lfps = filter_ripple_band(lfp_no_ripples, 1500)
        ripples = Karlsson_ripple_detector(
            time_3s, filtered_lfps, stationary_speed, sampling_frequency
        )

        assert isinstance(ripples, pd.DataFrame)
        # Should have few or no detections
        assert len(ripples) <= 2


class TestKarlssonEventStatistics:
    """Per-event statistics reflect the strongest channel, so an event that one
    channel triggered at 3 SD cannot report a sub-threshold max_sustained_zscore."""

    def test_max_sustained_zscore_never_below_threshold(self, time_3s, sampling_frequency):
        # ripples on one channel only, four quiet channels
        loud = simulate_LFP(
            time_3s,
            ripple_times=[1.0, 2.0],
            noise_amplitude=1.3,
            ripple_snr=6.0,
            random_state=1,
        )
        quiet = [
            simulate_LFP(time_3s, ripple_times=[], noise_amplitude=1.3, random_state=s)
            for s in (2, 3, 4, 5)
        ]
        lfps = filter_ripple_band(np.column_stack([loud, *quiet]), 1500)
        speed = np.full(len(time_3s), 2.0)
        events = Karlsson_ripple_detector(
            time_3s, lfps, speed, sampling_frequency, zscore_threshold=3.0
        )
        assert len(events) >= 2
        assert np.all(events.max_sustained_zscore >= 3.0)
        assert np.all(events.max_zscore >= 3.0)


class TestRoumisRippleDetector:
    """Test suite for Roumis ripple detector."""

    def test_single_channel_with_ripples(
        self, time_3s, single_lfp_with_ripples, stationary_speed, sampling_frequency
    ):
        """Test Roumis detector with single LFP channel."""
        filtered_lfps = filter_ripple_band(single_lfp_with_ripples, 1500)
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
        filtered_lfps = filter_ripple_band(dual_lfp_with_ripples, 1500)
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

    def test_normalization_mask_restricts_the_baseline(
        self, time_3s, multiunit_data, stationary_speed, sampling_frequency
    ):
        """The replacement for the removed use_speed_threshold_for_zscore."""
        events_all_data = multiunit_HSE_detector(
            time_3s, multiunit_data, stationary_speed, sampling_frequency
        )
        events_immobile_baseline = multiunit_HSE_detector(
            time_3s,
            multiunit_data,
            stationary_speed,
            sampling_frequency,
            normalization_mask=stationary_speed <= 4.0,
        )

        assert isinstance(events_all_data, pd.DataFrame)
        assert isinstance(events_immobile_baseline, pd.DataFrame)

    def test_the_removed_parameter_is_gone(
        self, time_3s, multiunit_data, stationary_speed, sampling_frequency
    ):
        """It warned from 1.7.0 and is removed in 2.0."""
        with pytest.raises(TypeError, match="use_speed_threshold_for_zscore"):
            multiunit_HSE_detector(
                time_3s,
                multiunit_data,
                stationary_speed,
                sampling_frequency,
                use_speed_threshold_for_zscore=True,
            )


class TestMultiunitHSEValidation:
    """multiunit_HSE_detector validates its inputs like the LFP detectors."""

    @pytest.fixture
    def inputs(self):
        fs = 1500
        n = fs * 3
        rng = np.random.default_rng(0)
        multiunit = (rng.random((n, 4)) < 0.02).astype(float)
        return np.arange(n) / fs, multiunit, np.full(n, 2.0), fs

    def test_nan_marks_the_sample_missing_in_spikes_or_speed(self, inputs):
        """A NaN spike count or speed sample is missing data, not an error: the
        burst on the far side of it is still found, and nothing spans it."""
        time, multiunit, speed, fs = inputs
        multiunit[2000:2060] = 1.0  # a burst
        clean = multiunit_HSE_detector(time, multiunit, speed, fs)
        multiunit[100, 1] = np.nan
        speed[300] = np.nan
        events = multiunit_HSE_detector(time, multiunit, speed, fs)
        assert len(events) == len(clean) >= 1
        assert not any((events.start_time <= time[100]) & (events.end_time >= time[100]))

    def test_length_mismatch_raises(self, inputs):
        time, multiunit, speed, fs = inputs
        with pytest.raises(ValueError, match="length"):
            multiunit_HSE_detector(time, multiunit[:-10], speed, fs)

    def test_one_dimensional_multiunit_raises(self, inputs):
        time, multiunit, speed, fs = inputs
        with pytest.raises(ValueError, match="2D"):
            multiunit_HSE_detector(time, multiunit[:, 0], speed, fs)

    def test_time_in_samples_raises(self, inputs):
        time, multiunit, speed, fs = inputs
        with pytest.raises(ValueError, match="samples"):
            multiunit_HSE_detector(np.arange(len(time), dtype=float), multiunit, speed, fs)


class TestKayConsensusTrace:
    """Test the Kay consensus trace generation."""

    def test_consensus_trace_shape(self, time_3s, dual_lfp_with_ripples, sampling_frequency):
        """Test that consensus trace has correct shape."""
        filtered_lfps = filter_ripple_band(dual_lfp_with_ripples, 1500)
        consensus = get_Kay_ripple_consensus_trace(
            filtered_lfps, sampling_frequency, smoothing_sigma=0.004
        )

        assert consensus.shape == (len(time_3s),)
        assert not np.all(np.isnan(consensus)), "Consensus trace should have valid data"

    def test_consensus_trace_positive(
        self, time_3s, dual_lfp_with_ripples, sampling_frequency
    ):
        """Test that consensus trace values are non-negative (it's a magnitude)."""
        filtered_lfps = filter_ripple_band(dual_lfp_with_ripples, 1500)
        consensus = get_Kay_ripple_consensus_trace(
            filtered_lfps, sampling_frequency, smoothing_sigma=0.004
        )

        # After square root, all values should be >= 0
        valid_values = consensus[~np.isnan(consensus)]
        assert np.all(valid_values >= 0), "Consensus trace should be non-negative"


def _yu_reference_consensus(filtered_lfps, sampling_frequency, blocks, zscore_per_channel):
    """Block-wise reference: envelope and 4 ms smoothing inside each block,
    per-tetrode z-score (ddof=1) pooled over all valid rows, then the median."""
    filtered_lfps = np.asarray(filtered_lfps, dtype=float)
    smoothed = np.full_like(filtered_lfps, np.nan)
    for start, stop in blocks:
        env = get_envelope(filtered_lfps[start:stop])
        smoothed[start:stop] = gaussian_smooth(env, 0.004, sampling_frequency)
    valid = np.all(np.isfinite(smoothed), axis=1)
    if zscore_per_channel:
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
        return filter_ripple_band(np.column_stack(lfps), 1500)

    def test_shape_and_finite_on_clean_input(self, time_3s, triple_lfp, sampling_frequency):
        consensus = get_Yu_ripple_consensus_trace(triple_lfp, sampling_frequency)
        assert consensus.shape == (len(time_3s),)
        assert np.all(np.isfinite(consensus))

    def test_flag_off_is_median_of_smoothed_envelopes(self, triple_lfp, sampling_frequency):
        expected = _yu_reference_consensus(
            triple_lfp, sampling_frequency, [(0, len(triple_lfp))], zscore_per_channel=False
        )
        consensus = get_Yu_ripple_consensus_trace(
            triple_lfp, sampling_frequency, zscore_per_channel=False
        )
        np.testing.assert_allclose(consensus, expected, rtol=1e-12, atol=1e-12)

    def test_flag_on_is_median_of_zscored_smoothed_envelopes(
        self, triple_lfp, sampling_frequency
    ):
        expected = _yu_reference_consensus(
            triple_lfp, sampling_frequency, [(0, len(triple_lfp))], zscore_per_channel=True
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
            scaled, sampling_frequency, zscore_per_channel=False
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
            lfps, sampling_frequency, [(0, 2000), (2300, len(lfps))], zscore_per_channel=True
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
            zscore_per_channel=True,
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
        lfps = filter_ripple_band(np.column_stack([*quiet, loud]), 1500)
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
    def test_sample_count_matches_the_package_convention(self):
        # 0.145 s at 1500 Hz is 217.5 samples: the package rounds half up (218),
        # and the Yu extractor must agree rather than lose the half to round-off
        fs, duration = 1500, 0.145
        time = np.arange(3000) / fs
        n_min = minimum_sample_count(time, duration)
        assert n_min == 218
        for n_run, expected in [(n_min - 1, 0), (n_min, 1)]:
            trace = np.zeros(3000)
            trace[1000 : 1000 + n_run] = 5.0
            events, _ = _extract_Yu_ripple_events(trace, time, duration, 2.0)
            assert len(events) == expected, (n_run, len(events))

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
        events, _ = _extract_Yu_ripple_events(trace, time, 0.020, 3.0)
        assert (len(events) == 1) == qualifies

    def test_extends_to_containing_above_zero_run(self):
        fs = 1000
        n_time = 500
        time = np.arange(n_time) / fs
        trace = self._trace(n_time, above_zero=[(100, 300)], above_threshold=[(150, 200)])
        events, _ = _extract_Yu_ripple_events(trace, time, 0.020, 3.0)
        np.testing.assert_allclose(events, [[time[100], time[299]]])

    def test_a_sample_at_the_mean_stays_in_the_run_and_one_below_ends_it(self):
        """The extension rule is the package's: at or above the mean, as in
        threshold_by_zscore."""
        fs = 1000
        n_time = 500
        time = np.arange(n_time) / fs
        trace = self._trace(n_time, above_zero=[(100, 300)], above_threshold=[(150, 200)])
        trace[250] = 0.0
        events, _ = _extract_Yu_ripple_events(trace, time, 0.020, 3.0)
        np.testing.assert_allclose(events, [[time[100], time[299]]])
        trace[250] = -1e-9
        events, _ = _extract_Yu_ripple_events(trace, time, 0.020, 3.0)
        np.testing.assert_allclose(events, [[time[100], time[249]]])

    def test_two_exceedances_in_one_run_yield_one_event(self):
        fs = 1000
        n_time = 500
        time = np.arange(n_time) / fs
        trace = self._trace(
            n_time, above_zero=[(100, 300)], above_threshold=[(120, 160), (220, 260)]
        )
        events, n_supra = _extract_Yu_ripple_events(trace, time, 0.020, 3.0)
        assert len(events) == 1
        assert n_supra[0] == 40  # the longest qualifying run

    def test_sub_minimum_exceedance_does_not_create_event(self):
        fs = 1000
        n_time = 500
        time = np.arange(n_time) / fs
        trace = self._trace(n_time, above_zero=[(100, 300)], above_threshold=[(150, 160)])
        events, _ = _extract_Yu_ripple_events(trace, time, 0.020, 3.0)
        assert len(events) == 0

    def test_a_run_reaching_the_block_edge_is_kept(self):
        fs = 1000
        n_time = 200
        time = np.arange(n_time) / fs
        # above zero from the very first sample to the last
        trace = self._trace(n_time, above_zero=[(0, 200)], above_threshold=[(50, 100)])
        events, _ = _extract_Yu_ripple_events(trace, time, 0.020, 3.0)
        np.testing.assert_allclose(events, [[time[0], time[-1]]])

    def test_uses_native_timestamps(self):
        fs = 1000
        n_time = 500
        time = 100.0 + np.arange(n_time) / fs + 1e-4 * np.sin(np.arange(n_time))
        trace = self._trace(n_time, above_zero=[(100, 300)], above_threshold=[(150, 200)])
        events, _ = _extract_Yu_ripple_events(trace, time, 0.020, 3.0)
        assert events[0, 0] == time[100]
        assert events[0, 1] == time[299]

    @pytest.mark.parametrize("threshold", [0.0, -1.0, np.nan, np.inf])
    def test_invalid_threshold_raises(self, threshold):
        fs = 1000
        time = np.arange(100) / fs
        with pytest.raises(ValueError, match="threshold"):
            _extract_Yu_ripple_events(np.zeros(100), time, 0.020, threshold)

    def test_empty_result_shapes(self):
        fs = 1000
        time = np.arange(100) / fs
        events, n_supra = _extract_Yu_ripple_events(np.full(100, -0.5), time, 0.020, 3.0)
        assert events.shape == (0, 2)
        assert n_supra.shape == (0,)


class TestYuRippleDetector:
    """Yu et al. 2017: median consensus, mirrored-histogram threshold, 20 ms."""

    FS = 1000
    N_TIME = 20_000  # 20 s: enough immobility to resolve the 99.99th percentile

    def test_speed_at_the_threshold_counts_as_immobile(self, time):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(5000, 5060, 20.0)])
        at_threshold = np.full(self.N_TIME, 4.0)
        events = Yu_ripple_detector(time, lfps, at_threshold, self.FS)
        assert len(events) >= 1

    def test_recovers_planted_bursts(self, time, stationary):
        bursts = [(5000, 5060, 20.0), (12000, 12080, 20.0)]
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, bursts)
        events = Yu_ripple_detector(time, lfps, stationary, self.FS)
        assert len(events) == 2
        for (start, stop, _), (_, row) in zip(bursts, events.iterrows(), strict=True):
            assert row.start_time <= time[start] + 0.010
            assert row.end_time >= time[stop - 1] - 0.010
            assert row.end_time - row.start_time < 0.150

    def test_output_columns(self, time, stationary):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(5000, 5060, 20.0)])
        events = Yu_ripple_detector(time, lfps, stationary, self.FS)
        for column in (
            "start_time",
            "end_time",
            "duration",
            "max_sustained_zscore",
            "mean_zscore",
            "max_zscore",
            "speed_at_start",
            "max_speed",
            "clipped_start",
            "clipped_end",
            "n_suprathreshold_samples",
            "detection_threshold_zscore",
        ):
            assert column in events.columns
        assert events.index.name == "event_number"
        assert not events.clipped_start.iloc[0]
        assert not events.clipped_end.iloc[0]
        assert events.n_suprathreshold_samples.iloc[0] >= 20
        assert np.isfinite(events.detection_threshold_zscore.iloc[0])

    def test_spyglass_style_keyword_call_matches_direct_call(self, time, stationary):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(5000, 5060, 20.0)])
        params = {
            "speed_threshold": 4.0,
            "minimum_duration": 0.020,
            "percentile": 99.99,
            "smoothing_sigma": 0.004,
            "zscore_per_channel": True,
        }
        via_dict = Yu_ripple_detector(
            time=time,
            filtered_lfps=lfps,
            speed=stationary,
            sampling_frequency=self.FS,
            **params,
        )
        direct = Yu_ripple_detector(time, lfps, stationary, self.FS)
        pd.testing.assert_frame_equal(via_dict, direct)

    def test_bursts_abutting_a_gap_are_not_joined_across_it(self, time, stationary):
        # Two short bursts on either side of 50 ms of missing LFP. Dropping the
        # missing rows (what the other detectors do) makes them one contiguous
        # burst that qualifies; processing blocks separately must not.
        bursts = [(8008, 8012, 20.0), (8062, 8066, 20.0)]
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, bursts)
        lfps[8012:8062, :] = np.nan
        with_gap = Yu_ripple_detector(time, lfps, stationary, self.FS)
        assert len(with_gap) == 0

        keep = np.ones(self.N_TIME, dtype=bool)
        keep[8012:8062] = False
        dropped = Yu_ripple_detector(
            np.arange(keep.sum()) / self.FS,  # time made contiguous, as row-dropping does
            lfps[keep],
            stationary[keep],
            self.FS,
        )
        assert len(dropped) == 1  # positive control: the join is what creates the event

    def test_burst_reaching_a_gap_is_clipped_and_flagged(self, time, stationary):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(8000, 8040, 20.0)])
        lfps[8030:8100, :] = np.nan  # the burst runs into missing data
        events = Yu_ripple_detector(time, lfps, stationary, self.FS)
        assert len(events) == 1
        assert events.end_time.iloc[0] == time[8029]
        assert bool(events.clipped_end.iloc[0])
        assert not bool(events.clipped_start.iloc[0])

    def test_movement_at_endpoint_excludes_event(self, time, stationary):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(5000, 5060, 20.0)])
        speed = stationary.copy()
        speed[4900:5100] = 10.0  # moving throughout the burst
        events = Yu_ripple_detector(time, lfps, speed, self.FS)
        assert len(events) == 0

    def test_movement_at_the_end_sample_alone_excludes_event(self, time, stationary):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(5000, 5060, 20.0)])
        speed = stationary.copy()
        speed[5030:5200] = 10.0  # still at the start, moving by the end
        assert len(Yu_ripple_detector(time, lfps, stationary, self.FS)) == 1
        assert len(Yu_ripple_detector(time, lfps, speed, self.FS)) == 0

    def test_an_explicit_noise_mask_replaces_the_speed_rule(self, time, stationary):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(5000, 5060, 20.0)])
        mask = np.ones(self.N_TIME, dtype=bool)
        mask[4000:6000] = False  # the burst is not in the noise sample
        events = Yu_ripple_detector(time, lfps, stationary, self.FS, normalization_mask=mask)
        assert len(events) == 1
        assert events.start_time.iloc[0] <= time[5000] <= events.end_time.iloc[0]

    def test_threshold_is_estimated_from_immobility_only(self, time, stationary):
        # a loud, long "artifact" during movement must not raise the threshold
        # enough to hide the immobile burst
        lfps = _synthetic_ripple_band(
            self.N_TIME, self.FS, [(5000, 5060, 20.0), (15000, 17000, 20.0)]
        )
        speed = stationary.copy()
        speed[14000:18000] = 10.0
        events = Yu_ripple_detector(time, lfps, speed, self.FS)
        assert len(events) == 1
        assert abs(events.start_time.iloc[0] - time[5000]) < 0.020

    def test_no_immobility_raises(self, time):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [])
        with pytest.raises(ValueError, match="selects no sample"):
            Yu_ripple_detector(time, lfps, np.full(self.N_TIME, 10.0), self.FS)

    @pytest.mark.parametrize("zscore_per_channel", [True, False])
    def test_both_normalization_readings_detect_without_warnings(
        self, time, stationary, zscore_per_channel
    ):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(5000, 5060, 20.0)])
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # in particular, no positive-mode warning
            events = Yu_ripple_detector(
                time, lfps, stationary, self.FS, zscore_per_channel=zscore_per_channel
            )
        assert len(events) == 1
        assert abs(events.start_time.iloc[0] - time[5000]) < 0.020

    def test_recovers_planted_ripples_in_simulated_lfp(self):
        fs = 1500
        time = np.arange(fs * 20) / fs
        planted = [2.0, 5.0, 8.0, 11.0, 14.0, 17.0]
        # ripples five times the ripple-band background on pink noise
        lfps = filter_ripple_band(
            np.column_stack(
                [
                    simulate_LFP(
                        time,
                        ripple_times=planted,
                        noise_amplitude=1.2,
                        ripple_snr=5.0,
                        random_state=seed,
                    )
                    for seed in (1, 2, 3, 4)
                ]
            ),
            fs,
        )
        events = Yu_ripple_detector(time, lfps, np.full(len(time), 2.0), fs)
        hits = [any((events.start_time <= t) & (events.end_time >= t)) for t in planted]
        assert all(hits)
        assert len(events) <= len(planted) + 2

    def test_pure_noise_yields_few_events(self):
        fs = 1500
        time = np.arange(fs * 20) / fs
        lfps = filter_ripple_band(
            np.column_stack(
                [
                    simulate_LFP(time, ripple_times=[], noise_amplitude=1.2, random_state=seed)
                    for seed in (1, 2, 3, 4)
                ]
            ),
            fs,
        )
        events = Yu_ripple_detector(time, lfps, np.full(len(time), 2.0), fs)
        assert len(events) <= 3
        # the threshold must lie above the immobility mean, which the detector
        # guarantees; how far above depends on the envelope's skew, and 1.8 to
        # 2.5 SD is typical of the mirrored-noise estimate on Gaussian noise
        assert events.detection_threshold_zscore.iloc[0] > 0.0 if len(events) else True

    def test_ripples_dominating_the_variance_raise_explicitly(self):
        # When ripples inflate the immobility SD so far that the mean sits
        # above the noise ceiling, the mirrored distribution cannot exceed the
        # mean and the threshold-then-return-to-mean rule is undefined. The
        # original MATLAB would run anyway; this implementation refuses.
        fs = 1500
        time = np.arange(fs * 20) / fs
        planted = [2.0, 5.0, 8.0, 11.0, 14.0, 17.0]
        # brown noise has almost no ripple-band power, so these ripples are more
        # than a hundred times the background and hold nearly all the variance
        lfps = filter_ripple_band(
            np.column_stack(
                [
                    simulate_LFP(
                        time,
                        ripple_times=planted,
                        noise_type="brown",
                        noise_amplitude=1.2,
                        ripple_amplitude=1.5,
                        random_state=seed,
                    )
                    for seed in (1, 2, 3, 4)
                ]
            ),
            fs,
        )
        with pytest.raises(ValueError, match="above the immobility mean"):
            Yu_ripple_detector(time, lfps, np.full(len(time), 2.0), fs)

    def test_exported_from_package_root(self):
        import ripple_detection

        assert ripple_detection.Yu_ripple_detector is Yu_ripple_detector


class TestTwoThresholdEvents:
    """FMAToolbox FindRipples segmentation on a normalized trace."""

    FS = 1000

    def _trace(self, n_time, runs, base=-0.5):
        """runs: list of (start, stop, level) index ranges set to `level`."""
        z = np.full(n_time, base)
        for start, stop, level in runs:
            z[start:stop] = level
        return z, np.arange(n_time) / self.FS

    def test_start_is_the_sample_before_the_low_crossing_and_end_the_last_above(self):
        z, t = self._trace(500, [(100, 190, 3.0), (140, 160, 6.0)])
        events, peaks, _ = _two_threshold_events(z, t, 2.0, 5.0, 0.030, 0.020, 0.100)
        np.testing.assert_allclose(events, [[t[99], t[189]]])
        assert peaks[0] == t[140]  # first sample of the peak plateau

    def test_peak_must_exceed_high_threshold(self):
        z, t = self._trace(500, [(100, 200, 3.0)])  # above low, never above high
        events, _, _ = _two_threshold_events(z, t, 2.0, 5.0, 0.030, 0.020, 0.100)
        assert len(events) == 0
        z, t = self._trace(500, [(100, 200, 3.0), (150, 151, 5.0)])  # equal to high: strict
        events, _, _ = _two_threshold_events(z, t, 2.0, 5.0, 0.030, 0.020, 0.100)
        assert len(events) == 0

    def test_close_events_merge_when_the_merged_duration_is_under_the_maximum(self):
        # two 30 ms events 20 ms apart: merged span 80 ms < 100 ms maximum -> one event
        z, t = self._trace(500, [(100, 130, 6.0), (150, 180, 6.0)])
        events, _, _ = _two_threshold_events(z, t, 2.0, 5.0, 0.030, 0.020, 0.100)
        assert len(events) == 1
        np.testing.assert_allclose(events, [[t[99], t[179]]])

    def test_close_events_do_not_merge_past_the_maximum_duration(self):
        # 60 ms + 20 ms gap + 60 ms = 140 ms > 100 ms maximum -> stay separate
        z, t = self._trace(500, [(100, 160, 6.0), (180, 240, 6.0)])
        events, _, _ = _two_threshold_events(z, t, 2.0, 5.0, 0.030, 0.020, 0.100)
        assert len(events) == 2

    def test_events_farther_apart_than_the_interval_do_not_merge(self):
        z, t = self._trace(500, [(100, 130, 6.0), (170, 200, 6.0)])  # 40 ms apart
        events, _, _ = _two_threshold_events(z, t, 2.0, 5.0, 0.030, 0.020, 0.100)
        assert len(events) == 2

    def test_duration_limits_are_strict(self):
        # 20 samples above low: start one before -> 20 ms span; not < 20 ms, kept
        z, t = self._trace(500, [(100, 120, 6.0)])
        events, _, _ = _two_threshold_events(z, t, 2.0, 5.0, 0.030, 0.020, 0.100)
        assert len(events) == 1
        z, t = self._trace(500, [(100, 118, 6.0)])  # 18 ms span < 20 ms -> dropped
        events, _, _ = _two_threshold_events(z, t, 2.0, 5.0, 0.030, 0.020, 0.100)
        assert len(events) == 0
        z, t = self._trace(500, [(100, 250, 6.0)])  # 150 ms > 100 ms -> dropped
        events, _, _ = _two_threshold_events(z, t, 2.0, 5.0, 0.030, 0.020, 0.100)
        assert len(events) == 0

    def test_clipped_means_a_missing_crossing_not_a_position(self):
        """A run beginning on the block's second sample has its rising crossing
        on the first, so it is not clipped although the event starts there;
        a run beginning on the first sample is."""
        z, t = self._trace(500, [(1, 50, 6.0), (200, 250, 6.0), (470, 500, 6.0)])
        events, _, clipped = _two_threshold_events(z, t, 2.0, 5.0, 0.0, 0.020, None)
        np.testing.assert_allclose(events[:, 0], [t[0], t[199], t[469]])
        assert clipped.tolist() == [[False, False], [False, False], [False, True]]
        z, t = self._trace(500, [(0, 50, 6.0)])
        _, _, clipped = _two_threshold_events(z, t, 2.0, 5.0, 0.0, 0.020, None)
        assert clipped.tolist() == [[True, False]]

    def test_merged_events_carry_the_flags_of_their_ends(self):
        z, t = self._trace(500, [(0, 40, 6.0), (60, 100, 6.0)])  # 20 ms apart -> merged
        events, _, clipped = _two_threshold_events(z, t, 2.0, 5.0, 0.030, 0.020, None)
        assert len(events) == 1
        assert clipped.tolist() == [[True, False]]

    def test_event_at_the_block_start_or_end_is_kept(self):
        # FindRipples discards an unpaired first or last run; this package keeps
        # it, starting or ending on the block edge, and flags it downstream
        z, t = self._trace(500, [(0, 50, 6.0), (200, 250, 6.0), (470, 500, 6.0)])
        events, _, _ = _two_threshold_events(z, t, 2.0, 5.0, 0.030, 0.020, 0.100)
        np.testing.assert_allclose(events, [[t[0], t[49]], [t[199], t[249]], [t[469], t[499]]])

    def test_duration_limits_are_inclusive_sample_counts(self):
        # an event spans from the sample before the run to the run's last sample;
        # 0.0205 s at 1 kHz rounds half up to 21 samples, 0.0305 s to 31
        for run_length, expected in [(19, 0), (20, 1), (30, 1), (31, 0)]:
            z, t = self._trace(500, [(100, 100 + run_length, 6.0)])
            events, _, _ = _two_threshold_events(z, t, 2.0, 5.0, 0.030, 0.0205, 0.0305)
            assert len(events) == expected, (run_length, len(events))

    def test_empty(self):
        z, t = self._trace(500, [])
        events, peaks, _ = _two_threshold_events(z, t, 2.0, 5.0, 0.030, 0.020, 0.100)
        assert events.shape == (0, 2)
        assert peaks.shape == (0,)


class TestZugaroSmoothingWindow:
    @pytest.mark.parametrize(
        ("fs", "expected"), [(1250, 11), (1500, 13), (1000, 9), (2000, 19), (2500, 23)]
    )
    def test_scales_with_rate_and_stays_odd(self, fs, expected):
        assert _zugaro_smoothing_window(fs) == expected


class TestZugaroRippleDetector:
    FS = 1000
    N_TIME = 20_000

    def test_recovers_planted_bursts(self, time, stationary):
        bursts = [(5000, 5060, 20.0), (12000, 12080, 20.0)]
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, bursts)
        events = Zugaro_ripple_detector(time, lfps, stationary, self.FS)
        assert len(events) == 2
        for (start, stop, _), (_, row) in zip(bursts, events.iterrows(), strict=True):
            assert row.start_time <= time[start] + 0.005
            assert row.end_time >= time[stop - 1] - 0.005
            assert time[start] <= row.peak_time <= time[stop]

    def test_weak_burst_above_low_but_below_high_is_rejected(self, time, stationary):
        # gain 1.0 peaks near 2.7 SD in the normalized power: above the 2 SD
        # boundary threshold, below the 5 SD peak threshold
        weak = _synthetic_ripple_band(self.N_TIME, self.FS, [(5000, 5060, 1.0)])

        def contains_burst(events):
            return bool(
                np.any((events.start_time <= time[5030]) & (events.end_time >= time[5030]))
            )

        assert not contains_burst(Zugaro_ripple_detector(time, weak, stationary, self.FS))
        # a strong burst at the same place is detected under the same defaults
        strong = _synthetic_ripple_band(self.N_TIME, self.FS, [(5000, 5060, 20.0)])
        assert contains_burst(Zugaro_ripple_detector(time, strong, stationary, self.FS))

    def test_output_columns(self, time, stationary):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(5000, 5060, 20.0)])
        events = Zugaro_ripple_detector(time, lfps, stationary, self.FS)
        for column in (
            "start_time",
            "end_time",
            "duration",
            "peak_time",
            "max_sustained_zscore",
            "mean_zscore",
            "max_speed",
        ):
            assert column in events.columns
        assert events.index.name == "event_number"

    def test_channels_are_summed_so_a_duplicated_channel_changes_nothing(
        self, time, stationary
    ):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(5000, 5060, 20.0)], n_channels=1)
        one = Zugaro_ripple_detector(time, lfps, stationary, self.FS)
        two = Zugaro_ripple_detector(time, np.hstack([lfps, lfps]), stationary, self.FS)
        pd.testing.assert_frame_equal(one, two)

    def test_missing_data_is_handled_block_wise(self, time, stationary):
        # one burst well inside the first block is found whole; the burst that
        # runs into the gap ends on its block's last sample and is flagged
        lfps = _synthetic_ripple_band(
            self.N_TIME, self.FS, [(7000, 7040, 20.0), (8000, 8040, 20.0)]
        )
        lfps[8030:8100, :] = np.nan
        events = Zugaro_ripple_detector(time, lfps, stationary, self.FS)
        assert len(events) == 2
        assert time[6990] <= events.start_time.iloc[0] <= time[7010]
        assert events.end_time.iloc[0] <= time[7060]
        assert not events.clipped_start.iloc[0]
        assert not events.clipped_end.iloc[0]
        assert events.end_time.iloc[1] == time[8029]
        assert events.clipped_end.iloc[1]
        assert not events.clipped_start.iloc[1]

    def test_no_finite_sample_raises(self, time, stationary):
        lfps = np.full((self.N_TIME, 2), np.nan)
        with pytest.raises(ValueError, match="nothing to detect"):
            Zugaro_ripple_detector(time, lfps, stationary, self.FS)

    def test_even_smoothing_window_is_rejected(self, time, stationary):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(5000, 5060, 20.0)])
        with pytest.raises(ValueError, match="odd"):
            Zugaro_ripple_detector(time, lfps, stationary, self.FS, smoothing_window=10)
        explicit = Zugaro_ripple_detector(
            time, lfps, stationary, self.FS, smoothing_window=_zugaro_smoothing_window(self.FS)
        )
        default = Zugaro_ripple_detector(time, lfps, stationary, self.FS)
        pd.testing.assert_frame_equal(explicit, default)

    def test_normalization_restricted_to_a_quiet_stretch_raises_the_zscores(
        self, time, stationary
    ):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(5000, 5060, 20.0)])
        whole = Zugaro_ripple_detector(time, lfps, stationary, self.FS)
        mask = np.zeros(self.N_TIME, dtype=bool)
        mask[:4000] = True
        masked = Zugaro_ripple_detector(
            time, lfps, stationary, self.FS, normalization_mask=mask
        )

        def planted(events):
            hit = events[(events.start_time <= time[5030]) & (events.end_time >= time[5030])]
            assert len(hit) == 1
            return hit.iloc[0]

        # the burst is excluded from the normalization stretch, so its z-score rises
        assert planted(masked).max_zscore > planted(whole).max_zscore

    def test_movement_at_endpoint_excludes_event(self, time, stationary):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(5000, 5060, 20.0)])
        speed = stationary.copy()
        speed[4900:5100] = 10.0
        assert len(Zugaro_ripple_detector(time, lfps, speed, self.FS)) == 0

    def test_movement_at_the_end_sample_alone_excludes_event(self, time, stationary):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(5000, 5060, 20.0)])
        speed = stationary.copy()
        speed[5030:5200] = 10.0
        assert len(Zugaro_ripple_detector(time, lfps, stationary, self.FS)) == 1
        assert len(Zugaro_ripple_detector(time, lfps, speed, self.FS)) == 0

    def test_a_peak_threshold_nothing_reaches_gives_an_empty_frame_with_peak_time(
        self, time, stationary
    ):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(5000, 5060, 20.0)])
        events = Zugaro_ripple_detector(time, lfps, stationary, self.FS, high_threshold=50.0)
        assert events.empty
        assert "peak_time" in events.columns

    def test_close_events_merge_when_there_is_no_ceiling(self, time, stationary):
        """Two bursts 20 ms apart, under the 30 ms merge interval, become one
        event whether or not a duration ceiling is in force."""
        lfps = _synthetic_ripple_band(
            self.N_TIME, self.FS, [(5000, 5030, 20.0), (5050, 5080, 20.0)]
        )
        assert (
            len(Zugaro_ripple_detector(time, lfps, stationary, self.FS, maximum_duration=None))
            == 1
        )
        assert len(Zugaro_ripple_detector(time, lfps, stationary, self.FS)) == 1

    def test_no_maximum_duration_keeps_a_long_event(self, time, stationary):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(5000, 5300, 20.0)])
        capped = Zugaro_ripple_detector(time, lfps, stationary, self.FS)
        uncapped = Zugaro_ripple_detector(
            time, lfps, stationary, self.FS, maximum_duration=None
        )
        assert len(capped) == 0
        assert len(uncapped) == 1

    def test_a_block_shorter_than_the_smoothing_window_is_treated_as_missing(
        self, time, stationary
    ):
        """Used to crash with a broadcast error inside np.convolve."""
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(5000, 5060, 20.0)])
        lfps[8000:8100] = np.nan
        lfps[8105:8200] = np.nan  # a five-sample island between two gaps
        with pytest.warns(UserWarning, match="treated as missing"):
            events = Zugaro_ripple_detector(time, lfps, stationary, self.FS)
        assert len(events) == 1

    def test_spyglass_style_keyword_call_matches_direct_call(self, time, stationary):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(5000, 5060, 20.0)])
        params = {
            "speed_threshold": 4.0,
            "low_threshold": 2.0,
            "high_threshold": 5.0,
            "minimum_inter_ripple_interval": 0.030,
            "minimum_duration": 0.020,
            "maximum_duration": 0.100,
        }
        via_dict = Zugaro_ripple_detector(
            time=time,
            filtered_lfps=lfps,
            speed=stationary,
            sampling_frequency=self.FS,
            **params,
        )
        pd.testing.assert_frame_equal(
            via_dict, Zugaro_ripple_detector(time, lfps, stationary, self.FS)
        )

    def test_exported_from_package_root(self):
        import ripple_detection

        assert ripple_detection.Zugaro_ripple_detector is Zugaro_ripple_detector


class TestFirfilt:
    def test_one_dimensional_input_matches_a_single_column(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=300)
        kernel = np.ones(7) / 7
        np.testing.assert_allclose(
            _firfilt(x, kernel), _firfilt(x[:, np.newaxis], kernel)[:, 0]
        )
        assert _firfilt(x, kernel).shape == (300,)


class TestLongSharpWaveRippleDetector:
    FS = 1000
    N_TIME = 40_000  # 40 s; events must sit more than 5 s from either end
    EVENTS = (7000, 11000, 15500, 19000, 23800, 28000, 31500, 34000)

    def test_recovers_planted_sharp_wave_ripples(self, time, stationary):
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, self.EVENTS)
        events = Long_sharp_wave_ripple_detector(
            time, lfp, stationary, self.FS, random_state=0
        )
        hits = [
            any((events.start_time <= time[c]) & (events.end_time >= time[c]))
            for c in self.EVENTS
        ]
        # the sharp-wave cut is the 10th percentile of the SWR cluster itself, so
        # the algorithm discards roughly the weakest tenth of its own events by design
        assert sum(hits) >= int(np.floor(0.9 * len(self.EVENTS))), hits
        assert len(events) <= len(self.EVENTS) + 2

    def test_ripple_without_sharp_wave_is_not_detected(self, time, stationary):
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, self.EVENTS, sharp_wave=False)
        events = Long_sharp_wave_ripple_detector(
            time, lfp, stationary, self.FS, random_state=0
        )
        hits = [
            any((events.start_time <= time[c]) & (events.end_time >= time[c]))
            for c in self.EVENTS
        ]
        assert sum(hits) <= 1

    def test_sharp_wave_without_ripple_is_not_detected(self, time, stationary):
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, self.EVENTS, ripple=False)
        events = Long_sharp_wave_ripple_detector(
            time, lfp, stationary, self.FS, random_state=0
        )
        hits = [
            any((events.start_time <= time[c]) & (events.end_time >= time[c]))
            for c in self.EVENTS
        ]
        assert sum(hits) <= 1

    def test_events_within_the_local_window_of_the_record_ends_are_not_reported(
        self, time, stationary
    ):
        edge_events = [2000, 38000, *self.EVENTS]
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, edge_events)
        events = Long_sharp_wave_ripple_detector(
            time, lfp, stationary, self.FS, random_state=0
        )
        assert np.all(events.peak_time >= 5.0)
        assert np.all(events.peak_time <= time[-1] - 5.0)

    def test_seeded_clustering_is_reproducible(self, time, stationary):
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, self.EVENTS)
        a = Long_sharp_wave_ripple_detector(time, lfp, stationary, self.FS, random_state=3)
        b = Long_sharp_wave_ripple_detector(time, lfp, stationary, self.FS, random_state=3)
        pd.testing.assert_frame_equal(a, b)

    def test_output_columns(self, time, stationary):
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, self.EVENTS)
        events = Long_sharp_wave_ripple_detector(
            time, lfp, stationary, self.FS, random_state=0
        )
        for column in (
            "start_time",
            "end_time",
            "peak_time",
            "duration",
            "sharp_wave_zscore",
            "sharp_wave_local_percentile",
            "ripple_power_zscore",
            "ripple_power_local_percentile",
            "sharp_wave_duration",
            "ripple_duration",
            "speed_at_start",
            "max_speed",
        ):
            assert column in events.columns
        assert events.index.name == "event_number"
        assert np.all(events.sharp_wave_zscore >= 2.5)
        assert np.all(events.ripple_power_zscore >= 2.5)
        assert np.all(
            (events.start_time <= events.peak_time) & (events.peak_time <= events.end_time)
        )

    def test_movement_at_endpoint_excludes_event(self, time, stationary):
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, self.EVENTS)
        speed = stationary.copy()
        speed[6800:7200] = 10.0  # moving through the first event
        events = Long_sharp_wave_ripple_detector(
            time, lfp, stationary, self.FS, random_state=0
        )
        moved = Long_sharp_wave_ripple_detector(time, lfp, speed, self.FS, random_state=0)
        assert any((events.start_time <= time[7000]) & (events.end_time >= time[7000]))
        assert not any((moved.start_time <= time[7000]) & (moved.end_time >= time[7000]))

    def test_record_shorter_than_the_slowest_kernel_raises(self):
        n_time = 500  # the 2 Hz Gaussian low-pass spans 957 samples at 1 kHz
        lfp = _synthetic_two_channel_lfp(n_time, self.FS, ())
        with pytest.raises(ValueError, match=r"as long as the .* samples"):
            Long_sharp_wave_ripple_detector(
                np.arange(n_time) / self.FS, lfp, np.full(n_time, 2.0), self.FS
            )

    def test_minimum_separation_keeps_only_the_last_of_close_candidates(
        self, time, stationary
    ):
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, self.EVENTS)
        events = Long_sharp_wave_ripple_detector(
            time, lfp, stationary, self.FS, minimum_separation=1e6, random_state=0
        )
        assert len(events) <= 1

    def test_no_event_survives_a_tiny_maximum_sharp_wave_duration(self, time, stationary):
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, self.EVENTS)
        events = Long_sharp_wave_ripple_detector(
            time,
            lfp,
            stationary,
            self.FS,
            minimum_sharp_wave_duration=0.001,
            maximum_sharp_wave_duration=0.002,
            random_state=0,
        )
        assert events.empty
        assert "start_time" in events.columns
        assert "sharp_wave_duration" in events.columns

    def test_requires_exactly_two_channels(self, time, stationary):
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, self.EVENTS)
        with pytest.raises(ValueError, match="two"):
            Long_sharp_wave_ripple_detector(time, lfp[:, :1], stationary, self.FS)
        with pytest.raises(ValueError, match="two"):
            Long_sharp_wave_ripple_detector(time, np.hstack([lfp, lfp]), stationary, self.FS)

    def test_nan_splits_the_record_and_events_far_from_it_survive(self, time, stationary):
        """A NaN sample ends a block. Blocks shorter than the sharp-wave kernel
        are treated as missing with a warning; the events on the long side
        are found as before."""
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, self.EVENTS)
        clean = Long_sharp_wave_ripple_detector(time, lfp, stationary, self.FS, random_state=0)
        lfp[100, 0] = np.nan  # leaves a 100-sample block before it
        with pytest.warns(UserWarning, match="treated as missing"):
            events = Long_sharp_wave_ripple_detector(
                time, lfp, stationary, self.FS, random_state=0
            )
        assert len(events) == len(clean) >= 1
        assert not events.clipped_start.any()
        assert not events.clipped_end.any()

    def test_exported_from_package_root(self):
        import ripple_detection

        assert (
            ripple_detection.Long_sharp_wave_ripple_detector is Long_sharp_wave_ripple_detector
        )


class TestCareyStateHelpers:
    """Boundary rules of vandermeerlab TSDtoIV and restrict: gaps merge when
    strictly shorter than merge_gap, intervals survive when strictly longer than
    minimum_length, and containment is closed at both ends."""

    # a quarter-second step keeps every gap and span exact in binary floating point
    TIME = np.arange(300) * 0.25

    def test_gap_equal_to_merge_gap_is_not_merged_and_shorter_is(self):
        state = np.zeros(300, dtype=bool)
        state[0:100] = True
        state[149:300] = True  # time[149] - time[99] is exactly 12.5
        assert len(_state_intervals(state, self.TIME, 12.5, 0.0)) == 2
        assert len(_state_intervals(state, self.TIME, 12.6, 0.0)) == 1

    def test_interval_exactly_minimum_length_is_dropped(self):
        state = np.zeros(300, dtype=bool)
        state[100:151] = True  # spans exactly 12.5
        assert len(_state_intervals(state, self.TIME, 0.0, 12.5)) == 0
        state[151] = True
        assert len(_state_intervals(state, self.TIME, 0.0, 12.5)) == 1

    def test_event_at_an_interval_edge_is_contained(self):
        intervals = np.array([[10, 20]])
        assert _contained_in_intervals(np.array([[10, 20], [10, 15]]), intervals).all()
        assert not _contained_in_intervals(np.array([[9, 15], [15, 21]]), intervals).any()

    def test_state_intervals_with_no_true_sample_is_empty(self):
        time = np.arange(100) / 1000.0
        intervals = _state_intervals(np.zeros(100, dtype=bool), time, 0.05, 0.05)
        assert intervals.shape == (0, 2)

    def test_contained_in_intervals_degenerate_inputs(self):
        assert _contained_in_intervals(
            np.empty((0, 2), dtype=int), np.array([[0, 10]])
        ).shape == (0,)
        result = _contained_in_intervals(
            np.array([[2, 5], [7, 9]]), np.empty((0, 2), dtype=int)
        )
        assert result.shape == (2,)
        assert not result.any()


class TestCareyCandidateDetector:
    FS = 1000
    N_TIME = 20_000
    EVENTS = (3000, 7000, 11000, 15000)

    @staticmethod
    def _hits(events, time, centers):
        return [
            any((events.start_time <= time[c]) & (events.end_time >= time[c])) for c in centers
        ]

    def test_a_population_that_never_bursts_raises(self, time, stationary):
        """A multiunit score that is zero everywhere would z-score to NaN and
        silently return no events; the detector says why instead."""
        rng = np.random.default_rng(0)
        lfps = rng.standard_normal((self.N_TIME, 1))
        multiunit = np.zeros((self.N_TIME, 6))
        multiunit[rng.choice(self.N_TIME, 40, replace=False), 0] = 1.0
        with pytest.raises(ValueError, match="never rises above its baseline"):
            Carey_candidate_detector(time, lfps, multiunit, stationary, self.FS)

    def test_speed_at_the_threshold_counts_as_immobile(self, time):
        lfps, multiunit = _synthetic_joint_inputs(self.N_TIME, self.FS, self.EVENTS)
        at_threshold = np.full(self.N_TIME, 4.0)
        events = Carey_candidate_detector(time, lfps, multiunit, at_threshold, self.FS)
        assert len(events) >= 1

    def test_recovers_events_with_both_ripple_and_burst(self, time, stationary):
        lfps, multiunit = _synthetic_joint_inputs(self.N_TIME, self.FS, self.EVENTS)
        events = Carey_candidate_detector(time, lfps, multiunit, stationary, self.FS)
        assert all(self._hits(events, time, self.EVENTS))
        assert len(events) <= len(self.EVENTS) + 2

    def test_ripple_without_burst_is_not_a_candidate(self, time, stationary):
        lfps, multiunit = _synthetic_joint_inputs(
            self.N_TIME, self.FS, self.EVENTS, spikes=False
        )
        events = Carey_candidate_detector(time, lfps, multiunit, stationary, self.FS)
        assert sum(self._hits(events, time, self.EVENTS)) == 0

    def test_burst_without_ripple_can_still_be_a_candidate(self, time, stationary):
        # The ripple score is the noise envelope rescaled to mean 1, never zero,
        # so the geometric mean does not suppress a burst that lacks a ripple.
        # The multiunit score is floored at zero, so the reverse does not hold
        # (previous test). This asymmetry is the original algorithm's.
        lfps, multiunit = _synthetic_joint_inputs(
            self.N_TIME, self.FS, self.EVENTS, ripple=False
        )
        events = Carey_candidate_detector(time, lfps, multiunit, stationary, self.FS)
        assert sum(self._hits(events, time, self.EVENTS)) >= len(self.EVENTS) - 1

    def test_minimum_active_units_is_enforced(self, time, stationary):
        lfps, multiunit = _synthetic_joint_inputs(self.N_TIME, self.FS, self.EVENTS)
        many = Carey_candidate_detector(
            time, lfps, multiunit, stationary, self.FS, minimum_active_units=5
        )
        too_many = Carey_candidate_detector(
            time, lfps, multiunit, stationary, self.FS, minimum_active_units=9
        )
        assert len(many) >= 1
        assert len(too_many) == 0  # only 8 units exist
        assert np.all(many.n_active_units >= 5)

    def test_event_during_movement_is_excluded(self, time, stationary):
        lfps, multiunit = _synthetic_joint_inputs(self.N_TIME, self.FS, self.EVENTS)
        speed = stationary.copy()
        speed[2800:3200] = 10.0
        events = Carey_candidate_detector(time, lfps, multiunit, speed, self.FS)
        hits = self._hits(events, time, self.EVENTS)
        assert not hits[0]
        assert all(hits[1:])

    def test_theta_exclusion_removes_event_with_strong_theta(self, time, stationary):
        lfps, multiunit = _synthetic_joint_inputs(self.N_TIME, self.FS, self.EVENTS)
        rng = np.random.default_rng(5)
        theta_lfp = rng.normal(0.0, 1.0, self.N_TIME)
        theta_lfp[6000:8000] += 15.0 * np.sin(2 * np.pi * 8.0 * time[6000:8000])
        without = Carey_candidate_detector(time, lfps, multiunit, stationary, self.FS)
        with_theta = Carey_candidate_detector(
            time, lfps, multiunit, stationary, self.FS, theta_lfp=theta_lfp
        )
        assert self._hits(without, time, self.EVENTS)[1]
        assert not self._hits(with_theta, time, self.EVENTS)[1]
        assert all(self._hits(with_theta, time, [self.EVENTS[0], *self.EVENTS[2:]]))

    def test_output_columns(self, time, stationary):
        lfps, multiunit = _synthetic_joint_inputs(self.N_TIME, self.FS, self.EVENTS)
        events = Carey_candidate_detector(time, lfps, multiunit, stationary, self.FS)
        for column in (
            "start_time",
            "end_time",
            "duration",
            "max_zscore",
            "n_active_units",
            "max_speed",
        ):
            assert column in events.columns
        assert events.index.name == "event_number"
        assert np.all(events.max_zscore > 3.0)

    def test_validation(self, time, stationary):
        lfps, multiunit = _synthetic_joint_inputs(self.N_TIME, self.FS, self.EVENTS)
        with pytest.raises(ValueError, match="length"):
            Carey_candidate_detector(time, lfps, multiunit[:-1], stationary, self.FS)
        with pytest.raises(ValueError, match="2D"):
            Carey_candidate_detector(time, lfps, multiunit[:, 0], stationary, self.FS)

    def test_a_theta_run_shorter_than_the_theta_filter_is_treated_as_missing(
        self, time, stationary
    ):
        """The theta filter needs more samples than its pad length; a five-sample
        island of theta between two theta gaps cannot be filtered, so its samples
        are missing for the detector, with a warning."""
        lfps, multiunit = _synthetic_joint_inputs(self.N_TIME, self.FS, self.EVENTS)
        theta = np.random.default_rng(0).standard_normal(self.N_TIME)
        theta[8000:8100] = np.nan
        theta[8105:8200] = np.nan
        with pytest.warns(UserWarning, match="treated as missing"):
            events = Carey_candidate_detector(
                time, lfps, multiunit, stationary, self.FS, theta_lfp=theta
            )
        assert len(events) >= 1

    def test_a_gap_in_another_input_does_not_split_the_theta_filtering(self, time, stationary):
        """A five-sample island in the LFP is a block of its own for the
        detector, but the theta channel is continuous there, so it is filtered
        as one run: no theta run is too short, and nothing is treated as missing."""
        lfps, multiunit = _synthetic_joint_inputs(self.N_TIME, self.FS, self.EVENTS)
        lfps[8000:8100] = np.nan
        lfps[8105:8200] = np.nan
        theta = np.random.default_rng(0).standard_normal(self.N_TIME)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            events = Carey_candidate_detector(
                time, lfps, multiunit, stationary, self.FS, theta_lfp=theta
            )
        assert len(events) >= 1

    def test_a_dropout_in_speed_does_not_restart_the_theta_filter(self, time, stationary):
        """The original filtered the whole theta recording at once. A NaN in speed
        splits the detector's blocks but not the theta filtering, so the theta
        exclusion decides the same events with and without the dropout."""
        lfps, multiunit = _synthetic_joint_inputs(self.N_TIME, self.FS, self.EVENTS)
        rng = np.random.default_rng(3)
        theta = rng.normal(0.0, 1.0, self.N_TIME)
        theta[6000:8000] += 15.0 * np.sin(2 * np.pi * 8.0 * time[6000:8000])
        speed_with_dropout = stationary.copy()
        speed_with_dropout[5900:5910] = np.nan
        without = Carey_candidate_detector(
            time, lfps, multiunit, stationary, self.FS, theta_lfp=theta
        )
        with_dropout = Carey_candidate_detector(
            time, lfps, multiunit, speed_with_dropout, self.FS, theta_lfp=theta
        )
        assert self._hits(without, time, self.EVENTS) == self._hits(
            with_dropout, time, self.EVENTS
        )
        assert not self._hits(with_dropout, time, self.EVENTS)[1]

    def test_nan_marks_the_sample_missing(self, time, stationary):
        """A NaN in the spikes, the LFP or speed ends a block; the candidates
        elsewhere are still found and none spans the missing sample."""
        lfps, multiunit = _synthetic_joint_inputs(self.N_TIME, self.FS, self.EVENTS)
        clean = Carey_candidate_detector(time, lfps, multiunit, stationary, self.FS)
        multiunit[10, 0] = np.nan
        lfps[self.EVENTS[0], :] = np.nan  # in the middle of the first event
        events = Carey_candidate_detector(time, lfps, multiunit, stationary, self.FS)
        assert len(clean) >= 2
        assert not any(
            (events.start_time < time[self.EVENTS[0]])
            & (events.end_time > time[self.EVENTS[0]])
        )
        assert set(self._hits(events, time, self.EVENTS[1:])) == {True}

    def test_multiunit_without_spikes_raises(self, time, stationary):
        lfps, multiunit = _synthetic_joint_inputs(self.N_TIME, self.FS, self.EVENTS)
        with pytest.raises(ValueError, match="no spikes"):
            Carey_candidate_detector(time, lfps, np.zeros_like(multiunit), stationary, self.FS)

    def test_theta_lfp_with_wrong_shape_raises(self, time, stationary):
        lfps, multiunit = _synthetic_joint_inputs(self.N_TIME, self.FS, self.EVENTS)
        with pytest.raises(ValueError, match="theta_lfp must have shape"):
            Carey_candidate_detector(
                time, lfps, multiunit, stationary, self.FS, theta_lfp=np.zeros(self.N_TIME - 1)
            )

    def test_unreachable_peak_threshold_gives_an_empty_table(self, time, stationary):
        lfps, multiunit = _synthetic_joint_inputs(self.N_TIME, self.FS, self.EVENTS)
        events = Carey_candidate_detector(
            time,
            lfps,
            multiunit,
            stationary,
            self.FS,
            high_threshold=1e6,
            theta_lfp=lfps[:, 0],
        )
        assert events.empty
        assert "n_active_units" in events.columns

    def test_exported_from_package_root(self):
        import ripple_detection

        assert ripple_detection.Carey_candidate_detector is Carey_candidate_detector


class TestDetectorErrorHandling:
    """Test error handling and edge cases for detectors."""

    def test_empty_input_raises(self, sampling_frequency):
        with pytest.raises(ValueError, match="nothing to detect"):
            Kay_ripple_detector(
                np.array([]), np.array([]).reshape(0, 1), np.array([]), sampling_frequency
            )

    def test_nan_in_lfp_rows_are_dropped_and_the_ripples_still_found(
        self, time_3s, single_lfp_with_ripples, stationary_speed, sampling_frequency
    ):
        """Rows with NaN are dropped; the planted ripples away from them survive
        and every statistic is finite."""
        lfp_with_nan = single_lfp_with_ripples.copy()
        lfp_with_nan[100:200, 0] = np.nan

        with pytest.warns(UserWarning, match="shorter than"):  # the 100-sample head
            filtered_lfps = filter_ripple_band(lfp_with_nan, 1500)
        ripples = Kay_ripple_detector(
            time_3s, filtered_lfps, stationary_speed, sampling_frequency
        )

        assert len(ripples) >= 2
        assert np.isfinite(ripples.drop(columns=[]).to_numpy(dtype=float)).all()

    def test_nan_speed_inside_a_ripple_splits_it_into_two_clipped_events(
        self, time_3s, stationary_speed, sampling_frequency
    ):
        """Missing speed is missing data: the ripple is cut at the gap, and the
        two halves end and start on the gap's edges, flagged as clipped."""
        speed_with_nan = stationary_speed.copy()
        speed_with_nan[1640:1660] = np.nan  # inside the ripple planted at 1.1 s
        # a long, loud ripple, so each half clears the threshold for 15 ms on its own
        lfp = simulate_LFP(
            time_3s,
            [1.1, 2.1],
            noise_amplitude=1.2,
            ripple_snr=12.0,
            ripple_duration=0.15,
            random_state=0,
        )

        filtered_lfps = filter_ripple_band(lfp[:, np.newaxis], 1500)
        ripples = Kay_ripple_detector(
            time_3s, filtered_lfps, speed_with_nan, sampling_frequency
        )

        before = ripples[ripples.end_time == time_3s[1639]]
        after = ripples[ripples.start_time == time_3s[1660]]
        assert len(before) == 1
        assert len(after) == 1
        assert before.clipped_end.item()
        assert not before.clipped_start.item()
        assert after.clipped_start.item()
        assert not after.clipped_end.item()
        assert not ripples[(ripples.start_time < 1.1) & (ripples.end_time > 1.1)].shape[0]

    def test_integer_lfp_not_truncated(
        self, time_3s, dual_lfp_with_cooccur_ripples, stationary_speed, sampling_frequency
    ):
        """Integer LFP input is cast to float, not truncated, through the pipeline."""
        lfp_int = np.round(dual_lfp_with_cooccur_ripples * 100).astype(np.int32)

        filtered = filter_ripple_band(lfp_int, 1500)
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
        filtered_lfps = filter_ripple_band(dual_lfp_with_ripples, 1500)
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
        filtered_lfps = filter_ripple_band(dual_lfp_with_ripples, 1500)
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
        filtered_lfps = filter_ripple_band(dual_lfp_with_ripples, 1500)
        with pytest.raises(ValueError, match="normalization_mask length"):
            Kay_ripple_detector(
                time_3s,
                filtered_lfps,
                stationary_speed,
                sampling_frequency,
                normalization_mask=np.ones(len(time_3s) - 5, dtype=bool),
            )

    def test_manual_normalization_rejects_a_degenerate_channel(
        self, time_3s, dual_lfp_with_cooccur_ripples, stationary_speed, sampling_frequency
    ):
        """A NaN baseline has no scale, so the detector raises and names the channel."""
        filtered_lfps = filter_ripple_band(dual_lfp_with_cooccur_ripples, 1500)
        n_channels = filtered_lfps.shape[1]
        baselines = np.zeros(n_channels)
        baselines[1] = np.nan  # degenerate channel
        deviations = np.ones(n_channels)

        with pytest.raises(ValueError, match=r"channel\(s\) \[1\]"):
            Shvartsman_ripple_detector(
                time_3s,
                filtered_lfps,
                stationary_speed,
                sampling_frequency,
                normalization_method="manual",
                channel_baselines=baselines,
                channel_deviations=deviations,
            )

    def test_manual_norm_rejects_normalization_mask(
        self, time_3s, dual_lfp_with_cooccur_ripples, stationary_speed, sampling_frequency
    ):
        """normalization_mask cannot be combined with manual normalization."""
        filtered_lfps = filter_ripple_band(dual_lfp_with_cooccur_ripples, 1500)
        env = gaussian_smooth(
            get_envelope(filtered_lfps), sigma=0.004, sampling_frequency=sampling_frequency
        )
        with pytest.raises(ValueError, match="manual"):
            Shvartsman_ripple_detector(
                time_3s,
                filtered_lfps,
                stationary_speed,
                sampling_frequency,
                normalization_method="manual",
                channel_baselines=env.mean(axis=0),
                channel_deviations=env.std(axis=0),
                normalization_mask=np.zeros(len(time_3s), dtype=bool),
            )

    def test_mismatched_lengths(self, time_3s, single_lfp_with_ripples, sampling_frequency):
        filtered_lfps = filter_ripple_band(single_lfp_with_ripples, 1500)
        with pytest.raises(ValueError, match="length mismatch"):
            Kay_ripple_detector(
                time_3s, filtered_lfps, np.ones(len(time_3s) // 2), sampling_frequency
            )

    def test_single_sample_raises(self, sampling_frequency):
        """One sample has no spread to normalize by."""
        with pytest.raises(ValueError, match="zero or undefined"):
            Kay_ripple_detector(np.array([0.0]), np.array([[0.5]]), np.array([2.0]), 1500)


class TestShvartsmanParticipationSemantics:
    """Preserve participation across merged events and subsequent exclusions."""

    @pytest.mark.parametrize(
        "participation",
        [{"minimum_participating_channels": 3}, {"minimum_participating_fraction": 1.0}],
    )
    def test_chain_of_overlapping_ripples_counts_all_electrodes(
        self, time_3s, stationary_speed, sampling_frequency, participation
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
        filtered = filter_ripple_band(lfps, 1500)
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
            **participation,
        )

        assert len(ripples) == 1
        event = ripples.iloc[0]
        assert event["start_time"] == first["start_time"]
        assert event["end_time"] == last["end_time"]
        assert event["participants"] == (0, 1, 2)
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
        filtered = filter_ripple_band(np.column_stack([ch0, ch1, ch2]), 1500)

        # Sanity: with no movement both events survive with differing participation.
        both = Shvartsman_ripple_detector(
            time_3s,
            filtered,
            stationary_speed,
            sampling_frequency,
            minimum_participating_channels=0,
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
            minimum_participating_channels=0,
            speed_threshold=4.0,
        )
        # Only the 1-participant event near 2.1s survives, and it must carry its own
        # metadata (a misaligned index would report the excluded event's count of 3).
        assert len(ripples) == 1
        assert ripples["n_participants"].iloc[0] == 1
        assert ripples["participants"].iloc[0] == (0,)

    def test_participating_fraction_of_one_means_all_channels(
        self,
        time_3s,
        dual_lfp_with_ripples,
        dual_lfp_with_cooccur_ripples,
        stationary_speed,
        sampling_frequency,
    ):
        """minimum_participating_fraction is a fraction of channels, so 1.0 means
        every channel, and the two arguments cannot be combined."""
        # The two channels ripple at separate times, so each event has only 1 of 2.
        filtered_sep = filter_ripple_band(dual_lfp_with_ripples, 1500)
        # 1.0 requires both channels in one event -> excluded (each has one).
        assert Shvartsman_ripple_detector(
            time_3s,
            filtered_sep,
            stationary_speed,
            sampling_frequency,
            minimum_participating_fraction=1.0,
        ).empty
        # 0.5 requires 1 of 2 -> detected (a genuine fraction in (0, 1)).
        assert not Shvartsman_ripple_detector(
            time_3s,
            filtered_sep,
            stationary_speed,
            sampling_frequency,
            minimum_participating_fraction=0.5,
        ).empty

        # Co-occurring ripples involve both channels, so 1.0 (all 2) detects them.
        filtered_co = filter_ripple_band(dual_lfp_with_cooccur_ripples, 1500)
        assert not Shvartsman_ripple_detector(
            time_3s,
            filtered_co,
            stationary_speed,
            sampling_frequency,
            minimum_participating_fraction=1.0,
        ).empty
        with pytest.raises(ValueError, match="not both"):
            Shvartsman_ripple_detector(
                time_3s,
                filtered_co,
                stationary_speed,
                sampling_frequency,
                minimum_participating_channels=1,
                minimum_participating_fraction=0.5,
            )
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            Shvartsman_ripple_detector(
                time_3s,
                filtered_co,
                stationary_speed,
                sampling_frequency,
                minimum_participating_fraction=2.0,
            )

    def test_dead_channel_raises_rather_than_diluting_participation(
        self, time_3s, dual_lfp_with_cooccur_ripples, stationary_speed
    ):
        """A constant channel has no scale. Rather than zero it and let it dilute
        frac_participants, the detector raises and names it."""
        filtered_lfps = filter_ripple_band(dual_lfp_with_cooccur_ripples, 1500)
        filtered_lfps[:, 1] = 0.0

        with pytest.raises(ValueError, match=r"channel\(s\) \[1\]"):
            Shvartsman_ripple_detector(time_3s, filtered_lfps, stationary_speed, 1500)


class TestParticipatingFractionRounding:
    def test_a_fraction_that_multiplies_to_just_over_an_integer_does_not_demand_one_more(
        self, time_3s, stationary_speed, sampling_frequency
    ):
        """25 channels at 0.28 is 7.000000000000001 in floating point; 7
        participating channels must satisfy it."""
        rng = np.random.default_rng(0)
        n_channels = 25
        lfps = np.column_stack(
            [
                simulate_LFP(
                    time_3s, [1.1], noise_amplitude=1.2, ripple_amplitude=1.5, random_state=s
                )
                for s in range(n_channels)
            ]
        )
        filtered = filter_ripple_band(lfps, 1500)
        env = gaussian_smooth(get_envelope(filtered), 0.004, sampling_frequency)
        # only 7 channels carry the ripple loudly enough: silence the others' bursts
        quiet = np.column_stack(
            [
                simulate_LFP(time_3s, [], noise_amplitude=1.2, random_state=100 + s)
                for s in range(n_channels - 7)
            ]
        )
        filtered[:, 7:] = filter_ripple_band(quiet, 1500)
        del env, rng
        seven = Shvartsman_ripple_detector(
            time_3s,
            filtered,
            stationary_speed,
            sampling_frequency,
            minimum_participating_channels=7,
        )
        by_fraction = Shvartsman_ripple_detector(
            time_3s,
            filtered,
            stationary_speed,
            sampling_frequency,
            minimum_participating_fraction=0.28,
        )
        assert len(seven) >= 1
        pd.testing.assert_frame_equal(seven, by_fraction)

    def test_an_empty_result_has_integer_participant_counts(
        self, time_3s, stationary_speed, sampling_frequency
    ):
        lfps = filter_ripple_band(
            np.column_stack(
                [
                    simulate_LFP(time_3s, [], noise_amplitude=1.2, random_state=s)
                    for s in (1, 2)
                ]
            ),
            1500,
        )
        events = Shvartsman_ripple_detector(
            time_3s, lfps, stationary_speed, sampling_frequency, zscore_threshold=50.0
        )
        assert events.empty
        assert events.n_participants.dtype == np.int64


class TestFindMaxThresh:
    def test_peak_in_a_short_excursion_does_not_hide_a_sustained_run(self):
        # the global peak (12.0) is a single sample; the value sustained for the
        # minimum duration anywhere in the event is the 23-sample run at 3.2
        time = np.arange(144) / 1500
        data = np.r_[[0.1] * 40, 12.0, [0.1] * 40, [3.2] * 23, [0.1] * 40]
        assert _max_sustained_zscore(time, data, 0.015) == 3.2

    def test_is_the_largest_threshold_at_which_the_event_still_qualifies(self):
        rng = np.random.default_rng(0)
        time = np.arange(200) / 1000
        data = rng.normal(size=200)
        n_min = minimum_sample_count(time, 0.020)
        brute = max(data[k : k + n_min].min() for k in range(len(data) - n_min + 1))
        assert _max_sustained_zscore(time, data, 0.020) == brute

    """Samples are 10 ms apart unless stated, so a 15 ms minimum is
    round(1.5) = 2 samples and a 35 ms minimum is round(3.5) = 4 samples."""

    def test_respects_minimum_duration(self):
        """max_sustained_zscore is the largest value sustained for minimum_duration, so a
        longer required duration yields a smaller (or equal) result. Peak at
        index 0 -> only rightward expansion."""
        time = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
        data = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        assert _max_sustained_zscore(time, data, minimum_duration=0.005) == 10.0  # 1 sample
        assert _max_sustained_zscore(time, data, minimum_duration=0.015) == 8.0  # 2 samples
        assert _max_sustained_zscore(time, data, minimum_duration=0.035) == 4.0  # 4 samples

    def test_mid_peak_expands_both_directions(self):
        """A mid-array peak exercises the leftward-expansion branch and the
        neighbor tie-break. From peak 10 at index 2 the window first steps left
        (neighbor 5 > 3) -> indices 1..2 -> min(5, 10); for four samples it
        then steps right twice (3 > 1, 2 > 1) -> indices 1..4 -> min(5, 2)."""
        time = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
        data = np.array([1.0, 5.0, 10.0, 3.0, 2.0])
        assert _max_sustained_zscore(time, data, minimum_duration=0.015) == 5.0
        assert _max_sustained_zscore(time, data, minimum_duration=0.035) == 2.0

    def test_peak_at_last_index_expands_left(self):
        """Peak at the last index forces leftward-only expansion."""
        time = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
        data = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        assert _max_sustained_zscore(time, data, minimum_duration=0.015) == 8.0
        assert _max_sustained_zscore(time, data, minimum_duration=0.035) == 4.0

    def test_all_equal_data(self):
        """A flat plateau returns the (shared) value."""
        time = np.array([0.0, 0.01, 0.02, 0.03])
        data = np.array([5.0, 5.0, 5.0, 5.0])
        assert _max_sustained_zscore(time, data, minimum_duration=0.015) == 5.0

    def test_two_sample_event_long_enough(self):
        """With 20 ms samples, one sample already sustains 15 ms
        (round(0.75) = 1), so the peak itself is returned."""
        time = np.array([0.0, 0.02])
        data = np.array([10.0, 0.0])
        assert _max_sustained_zscore(time, data, minimum_duration=0.015) == 10.0

    def test_exact_minimum_duration_does_not_expand_further(self):
        """20 ms at 1000 Hz is exactly 20 samples; the window must not take a
        21st, lower sample because of timestamp round-off."""
        time = np.arange(200, 222) / 1000
        data = np.arange(22.0, 0.0, -1.0)
        assert _max_sustained_zscore(time, data, minimum_duration=0.02) == 3.0

    def test_short_event_returns_nan(self):
        """An event shorter than minimum_duration cannot sustain the threshold, so
        the value is undefined -> nan (previously ran an index out of bounds)."""
        time = np.array([0.0, 0.001])
        data = np.array([10.0, 0.0])
        assert np.isnan(_max_sustained_zscore(time, data, minimum_duration=0.015))

    def test_single_sample_event_returns_nan(self):
        """A one-sample event has no measurable interval, so no positive
        duration is sustained -> nan (no out-of-bounds)."""
        time = np.array([1.0])
        data = np.array([7.0])
        assert np.isnan(_max_sustained_zscore(time, data, minimum_duration=0.015))


class TestSampleCountDurationConvention:
    """Detectors and max_sustained_zscore count samples for the minimum duration."""

    def test_kay_accepts_a_run_of_round_minimum_samples(self):
        fs = 1500
        n = fs * 10
        time = np.arange(n) / fs
        rng = np.random.default_rng(0)
        lfps = rng.normal(0.0, 1.0, (n, 2))
        t = np.arange(n) / fs
        burst = np.sin(2 * np.pi * 200.0 * t)
        # make exactly 23 consecutive samples (round(0.015 * 1500)) loud
        lfps[5000:5023] += 30.0 * burst[5000:5023, np.newaxis]
        events = Kay_ripple_detector(time, lfps, np.full(n, 2.0), fs, minimum_duration=0.015)
        assert len(events) >= 1
        assert any((events.start_time <= time[5000]) & (events.end_time >= time[5022]))

    @pytest.mark.parametrize("offset", [0.0, 1000.0])
    def test_max_sustained_zscore_is_finite_for_an_event_of_exactly_minimum_samples(
        self, offset
    ):
        fs = 1000
        time = offset + np.arange(15) / fs  # 15 samples = 15 ms at 1000 Hz
        data = np.linspace(2.0, 3.0, 15)
        assert np.isfinite(_max_sustained_zscore(time, data, minimum_duration=0.015))

    def test_max_sustained_zscore_is_nan_below_minimum_samples(self):
        fs = 1000
        time = np.arange(14) / fs
        data = np.linspace(2.0, 3.0, 14)
        assert np.isnan(_max_sustained_zscore(time, data, minimum_duration=0.015))


class TestMaxThreshMinimumDuration:
    """Karlsson and multiunit_HSE must honor the caller's minimum_duration for
    max_sustained_zscore, not silently fall back to the 15 ms default."""

    @staticmethod
    def _spy_on_max_sustained_zscore():
        """Patch _max_sustained_zscore to record the minimum_duration it receives while
        still delegating to the real implementation."""
        real = events_module._max_sustained_zscore
        seen: list[float] = []

        def spy(time, data, minimum_duration=0.015):
            seen.append(minimum_duration)
            return real(time, data, minimum_duration)

        return patch.object(events_module, "_max_sustained_zscore", spy), seen

    def test_karlsson_forwards_minimum_duration_to_max_sustained_zscore(
        self, time_3s, dual_lfp_with_cooccur_ripples, stationary_speed, sampling_frequency
    ):
        filtered_lfps = filter_ripple_band(dual_lfp_with_cooccur_ripples, 1500)
        patcher, seen = self._spy_on_max_sustained_zscore()
        with patcher:
            ripples = Karlsson_ripple_detector(
                time_3s,
                filtered_lfps,
                stationary_speed,
                sampling_frequency,
                minimum_duration=0.005,
            )
        assert len(ripples) > 0
        # max_sustained_zscore must be computed with the caller's 0.005, not the 0.015 default
        # (this fails if the minimum_duration argument is dropped from the call).
        assert seen
        assert all(md == 0.005 for md in seen)

    def test_hse_forwards_minimum_duration_to_max_sustained_zscore(
        self, time_3s, sampling_frequency
    ):
        multiunit = np.zeros((len(time_3s), 5))
        idx = int(1.0 * sampling_frequency)
        multiunit[idx : idx + 30, :] = 1
        speed = np.ones(len(time_3s)) * 2.0
        patcher, seen = self._spy_on_max_sustained_zscore()
        with patcher:
            hse = multiunit_HSE_detector(
                time_3s, multiunit, speed, sampling_frequency, minimum_duration=0.005
            )
        assert len(hse) > 0
        assert seen
        assert all(md == 0.005 for md in seen)

    def test_hse_tiny_minimum_duration_is_bounds_safe(self, time_3s, sampling_frequency):
        # A sharp, few-sample synchrony burst detected with a 1 ms minimum used to
        # crash inside max_sustained_zscore because it fell back to the 15 ms window.
        multiunit = np.zeros((len(time_3s), 5))
        idx = int(1.0 * sampling_frequency)
        multiunit[idx : idx + 3, :] = 1
        speed = np.ones(len(time_3s)) * 2.0
        hse = multiunit_HSE_detector(
            time_3s, multiunit, speed, sampling_frequency, minimum_duration=0.001
        )
        assert len(hse) > 0
        assert np.all(np.isfinite(hse["max_sustained_zscore"].to_numpy()))

    def test_hse_exact_minimum_duration_has_finite_max_sustained_zscore(self):
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
        assert hse["max_sustained_zscore"].iloc[0] == pytest.approx(np.sqrt(979 / 21))


class TestNegativeThresholdRejected:
    """The mean-crossing extension assumes threshold runs sit inside above-mean runs."""

    def test_kay_rejects_a_negative_threshold(
        self, time_3s, single_lfp_with_ripples, stationary_speed, sampling_frequency
    ):
        filtered_lfps = filter_ripple_band(single_lfp_with_ripples, 1500)
        with pytest.raises(ValueError, match="non-negative"):
            Kay_ripple_detector(
                time_3s,
                filtered_lfps,
                stationary_speed,
                sampling_frequency,
                zscore_threshold=-1.0,
            )


class TestWarningsPointAtTheCaller:
    """Unit warnings must name the caller's line, not a frame inside the package."""

    @pytest.mark.parametrize("detector", ["Kay", "Zugaro", "Yu"])
    def test_time_step_warning_reports_this_file(self, detector):
        sampling_frequency = 1000.0
        n_time = 20_000
        time_ms = np.arange(n_time) * 0.005  # five times the expected step
        lfps = _synthetic_ripple_band(n_time, sampling_frequency, [(5000, 5060, 20.0)])
        speed = np.full(n_time, 2.0)
        call = {
            "Kay": lambda: Kay_ripple_detector(time_ms, lfps, speed, sampling_frequency),
            "Zugaro": lambda: Zugaro_ripple_detector(time_ms, lfps, speed, sampling_frequency),
            "Yu": lambda: Yu_ripple_detector(time_ms, lfps, speed, sampling_frequency),
        }[detector]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with contextlib.suppress(ValueError):
                call()
        unit_warnings = [w for w in caught if "Time array step" in str(w.message)]
        assert unit_warnings, [str(w.message) for w in caught]
        assert unit_warnings[0].filename == __file__


class TestNormalizationArgumentValidation:
    @pytest.mark.parametrize("detector", ["Yu", "Zugaro"])
    def test_a_non_boolean_mask_is_rejected(self, detector):
        sampling_frequency = 1000
        n_time = 20_000
        time = np.arange(n_time) / sampling_frequency
        lfps = _synthetic_ripple_band(n_time, sampling_frequency, [(5000, 5060, 20.0)])
        speed = np.full(n_time, 2.0)
        detect = Yu_ripple_detector if detector == "Yu" else Zugaro_ripple_detector
        with pytest.raises(ValueError, match="must be boolean"):
            detect(time, lfps, speed, sampling_frequency, normalization_mask=speed)


class TestShvartsmanManualNormalizationValidation:
    def test_normalization_arguments_are_rejected_under_manual_normalization(
        self, time_3s, single_lfp_with_ripples, stationary_speed, sampling_frequency
    ):
        filtered_lfps = filter_ripple_band(single_lfp_with_ripples, 1500)
        with pytest.raises(ValueError, match="manual"):
            Shvartsman_ripple_detector(
                time_3s,
                filtered_lfps,
                stationary_speed,
                sampling_frequency,
                normalization_method="manual",
                channel_baselines=np.array([0.0]),
                channel_deviations=np.array([1.0]),
                normalization_mask=time_3s <= time_3s[100],
            )


class TestRowDropDetectorsSmoothWithinBlocks:
    """After NaN rows are dropped, the envelope and the smoothing stay inside
    each contiguous block, so a ripple just before a gap looks the same
    whether or not the data after the gap is present."""

    FS = 1500

    @pytest.mark.parametrize(
        "detector",
        [
            Kay_ripple_detector,
            Karlsson_ripple_detector,
            Roumis_ripple_detector,
            Shvartsman_ripple_detector,
        ],
    )
    @pytest.mark.parametrize("how", ["nan", "deleted"])
    def test_events_before_a_gap_match_the_block_run_alone(self, detector, how):
        time = np.arange(self.FS * 8) / self.FS
        lfp = _synthetic_ripple_band(
            len(time), self.FS, [(4400, 4500, 20.0), (9000, 9100, 20.0)]
        )
        lfp = np.column_stack([lfp, lfp * 0.8])
        speed = np.full(len(time), 2.0)
        gap = slice(4600, 7000)
        if how == "nan":
            with_gap = lfp.copy()
            with_gap[gap] = np.nan
            gap_time, gap_speed = time, speed
        else:
            keep = np.ones(len(time), dtype=bool)
            keep[gap] = False
            with_gap, gap_time, gap_speed = lfp[keep], time[keep], speed[keep]

        # normalize both over the first block, so only the transform can differ
        whole = detector(
            gap_time, with_gap, gap_speed, self.FS, normalization_mask=gap_time <= time[4599]
        )
        first_block = detector(
            time[:4600],
            lfp[:4600],
            speed[:4600],
            self.FS,
            normalization_mask=np.ones(4600, dtype=bool),
        )

        before_gap = whole[whole.end_time < time[4600]]
        pd.testing.assert_frame_equal(
            before_gap.reset_index(drop=True), first_block.reset_index(drop=True)
        )


class TestKayConsensusTraceMissingSamples:
    def test_smoothing_does_not_cross_a_gap(self):
        sampling_frequency = 1500.0
        n_time = 6000
        time = np.arange(n_time) / sampling_frequency
        rng = np.random.default_rng(0)
        lfps = rng.normal(0.0, 0.1, (n_time, 2))
        lfps[2000:2400] += 10.0  # a strong block right before the gap
        lfps[2400:2600, :] = np.nan
        trace = get_Kay_ripple_consensus_trace(
            lfps, sampling_frequency, smoothing_sigma=0.004, time=time
        )
        assert np.isnan(trace[2400:2600]).all()
        # the samples just after the gap must not carry the pre-gap block
        after_gap = trace[2600:2800]
        quiet = trace[3500:4500]
        assert np.nanmax(after_gap) < 5 * np.nanmedian(quiet)


class TestLongMaxThreshIsFinite:
    """max_sustained_zscore is measured over the reported event, which the sharp wave bounds."""

    FS = 1000
    N_TIME = 40_000
    EVENTS = (7000, 11000, 15500, 19000, 23800, 28000, 31500)

    def test_a_long_ripple_minimum_does_not_make_max_sustained_zscore_nan(self):
        # the event spans the sharp wave, so the sustained-value window must use
        # the sharp-wave minimum; using the ripple minimum can exceed the event
        time = np.arange(self.N_TIME) / self.FS
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, self.EVENTS)
        events = Long_sharp_wave_ripple_detector(
            time,
            lfp,
            np.full(self.N_TIME, 2.0),
            self.FS,
            minimum_ripple_duration=0.400,
            random_state=0,
        )
        assert len(events) >= 1
        assert np.isfinite(events.max_sustained_zscore).all(), (
            events.max_sustained_zscore.to_numpy()
        )


class TestEventStatisticsShapeValidation:
    """The shape checks run on the converted array, so sequences raise ValueError."""

    FS = 1000

    def _inputs(self):
        time = np.arange(1000) / self.FS
        return time, [[0.100, 0.150]], np.full(1000, 2.0)

    def test_two_dimensional_metric_without_participants_raises(self):
        time, events, speed = self._inputs()
        with pytest.raises(ValueError, match=r"must have shape \(n_time,\)"):
            _get_event_stats(events, time, np.zeros((1000, 3)).tolist(), speed, 0.015)

    def test_one_dimensional_metric_with_participants_raises(self):
        time, events, speed = self._inputs()
        with pytest.raises(ValueError, match=r"\(n_time, n_channels\)"):
            _get_event_stats(
                events, time, np.zeros(1000).tolist(), speed, 0.015, participants=[{0}]
            )


LFP_DETECTORS_WITHOUT_A_CEILING = [
    Kay_ripple_detector,
    Karlsson_ripple_detector,
    Roumis_ripple_detector,
    Shvartsman_ripple_detector,
    Yu_ripple_detector,
]


class TestMaximumDuration:
    """A ceiling on event duration, which 25 of the 57 surveyed papers impose.

    Covers the ceiling's boundary too, and the columns that travel with an event.
    """

    FS = 1000
    N_TIME = 20_000  # long enough for the Yu noise threshold
    # the long burst comes first on purpose: if the dropped event were last,
    # a wrong mask on the per-event columns would look the same as a right one
    LONG = (5_000, 5_500, 20.0)  # 500 ms burst
    SHORT = (12_000, 12_060, 20.0)  # 60 ms burst

    @pytest.fixture
    def lfps(self):
        return _synthetic_ripple_band(self.N_TIME, self.FS, [self.LONG, self.SHORT])

    @pytest.mark.parametrize("detector", LFP_DETECTORS_WITHOUT_A_CEILING)
    def test_none_detects_what_the_default_detects(self, detector, time, lfps, stationary):
        """Passing None explicitly changes nothing."""
        default = detector(time, lfps, stationary, self.FS)
        explicit = detector(time, lfps, stationary, self.FS, maximum_duration=None)

        pd.testing.assert_frame_equal(default, explicit)

    @pytest.mark.parametrize("detector", LFP_DETECTORS_WITHOUT_A_CEILING)
    def test_ceiling_drops_the_long_event(self, detector, time, lfps, stationary):
        """Both bursts are found; a ceiling between them keeps only the short one."""
        without = detector(time, lfps, stationary, self.FS)
        assert len(without) == 2

        with_ceiling = detector(time, lfps, stationary, self.FS, maximum_duration=0.2)

        assert len(with_ceiling) == 1
        assert with_ceiling.iloc[0].start_time > 11.0

    @pytest.mark.parametrize("detector", LFP_DETECTORS_WITHOUT_A_CEILING)
    def test_every_kept_event_is_within_the_ceiling(self, detector, time, lfps, stationary):
        """The limit applies to the reported event, not the suprathreshold run."""
        events = detector(time, lfps, stationary, self.FS, maximum_duration=0.2)

        # the limit is a sample count, so the longest kept event spans one
        # sample less than the ceiling in elapsed time
        durations = events.end_time - events.start_time
        assert (durations <= 0.2 - 1 / self.FS + 1e-12).all()

    @pytest.mark.parametrize("detector", LFP_DETECTORS_WITHOUT_A_CEILING)
    def test_ceiling_never_adds_events(self, detector, time, lfps, stationary):
        """Tightening the ceiling can only remove events."""
        loose = detector(time, lfps, stationary, self.FS, maximum_duration=1.0)
        tight = detector(time, lfps, stationary, self.FS, maximum_duration=0.2)

        assert set(tight.start_time) <= set(loose.start_time)

    def test_multiunit_detector_takes_a_ceiling(self, time, stationary):
        """The burst detector gets the same limit."""
        rng = np.random.default_rng(0)
        multiunit = rng.poisson(0.02, (self.N_TIME, 20)).astype(float)
        multiunit[5_000:5_060] += rng.poisson(0.6, (60, 20))
        multiunit[12_000:12_500] += rng.poisson(0.6, (500, 20))

        without = multiunit_HSE_detector(time, multiunit, stationary, self.FS)
        with_ceiling = multiunit_HSE_detector(
            time, multiunit, stationary, self.FS, maximum_duration=0.2
        )

        assert len(with_ceiling) < len(without)
        durations = with_ceiling.end_time - with_ceiling.start_time
        assert (durations <= 0.2 + 1 / self.FS).all()

    def test_a_ceiling_below_the_minimum_raises(self, time, lfps, stationary):
        """The two limits have to leave a usable window."""
        with pytest.raises(ValueError, match="maximum_duration"):
            Kay_ripple_detector(
                time,
                lfps,
                stationary,
                self.FS,
                minimum_duration=0.050,
                maximum_duration=0.010,
            )

    def test_an_event_of_exactly_the_sample_limit_is_kept(self, time):
        """The limit is round(maximum_duration * sampling_frequency) samples."""
        limit_samples = minimum_sample_count(time, 0.2)
        events = np.array([[0.0, (limit_samples - 1) / self.FS]])

        kept, keep = _exclude_long_events(events, time, 0.2)

        assert keep.tolist() == [True]
        assert len(kept) == 1

    def test_one_sample_more_than_the_limit_is_dropped(self, time):
        limit_samples = minimum_sample_count(time, 0.2)
        events = np.array([[0.0, limit_samples / self.FS]])

        _, keep = _exclude_long_events(events, time, 0.2)

        assert keep.tolist() == [False]

    def test_an_event_spanning_exactly_the_ceiling_in_seconds_is_dropped(self, time):
        """Elapsed time and sample count differ by one sample; the rule is samples."""
        events = np.array([[0.0, 0.2]])

        _, keep = _exclude_long_events(events, time, 0.2)

        assert keep.tolist() == [False]

    def test_yu_keeps_the_surviving_events_own_columns(self, time, lfps, stationary):
        """A wrong mask would carry the dropped event's values across."""
        without = Yu_ripple_detector(time, lfps, stationary, self.FS)
        with_ceiling = Yu_ripple_detector(
            time, lfps, stationary, self.FS, maximum_duration=0.2
        )

        assert len(without) == 2
        assert len(with_ceiling) == 1
        assert (
            with_ceiling.iloc[0].n_suprathreshold_samples
            == without.iloc[1].n_suprathreshold_samples
        )
        assert with_ceiling.iloc[0].clipped_start == without.iloc[1].clipped_start

    def test_shvartsman_keeps_the_surviving_events_own_participants(self, time, stationary):
        """participants selects the channels the z-score statistics average over."""
        # the two events must differ in their channels, or carrying the dropped
        # event's participants across would look identical to keeping the right ones
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [self.LONG], n_channels=3)
        short_only = _synthetic_ripple_band(
            self.N_TIME, self.FS, [self.SHORT], n_channels=3, seed=1
        )
        lfps[:, :2] += (
            short_only[:, :2]
            - _synthetic_ripple_band(self.N_TIME, self.FS, [], n_channels=3, seed=1)[:, :2]
        )

        without = Shvartsman_ripple_detector(time, lfps, stationary, self.FS)
        with_ceiling = Shvartsman_ripple_detector(
            time, lfps, stationary, self.FS, maximum_duration=0.2
        )

        assert len(without) == 2
        assert len(with_ceiling) == 1
        assert without.iloc[0].n_participants != without.iloc[1].n_participants
        assert with_ceiling.iloc[0].n_participants == without.iloc[1].n_participants
        assert with_ceiling.iloc[0].mean_zscore == pytest.approx(without.iloc[1].mean_zscore)

    def test_carey_takes_a_ceiling_and_keeps_its_own_unit_count(self, time, stationary):
        """Carey was the one detector whose ceiling no test exercised."""
        rng = np.random.default_rng(0)
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [self.LONG, self.SHORT])
        multiunit = rng.poisson(0.002, (self.N_TIME, 20)).astype(float)
        multiunit[5_000:5_500, :12] += rng.poisson(0.4, (500, 12))
        multiunit[12_000:12_060, :6] += rng.poisson(0.4, (60, 6))

        without = Carey_candidate_detector(
            time, lfps, multiunit, stationary, self.FS, minimum_active_units=3
        )
        with_ceiling = Carey_candidate_detector(
            time,
            lfps,
            multiunit,
            stationary,
            self.FS,
            minimum_active_units=3,
            maximum_duration=0.2,
        )

        assert len(with_ceiling) < len(without)
        assert with_ceiling.iloc[0].n_active_units == without.iloc[-1].n_active_units

    def test_equal_minimum_and_maximum_are_accepted(self, time, lfps, stationary):
        """A single admissible duration is a degenerate window, not an error."""
        events = Kay_ripple_detector(
            time,
            lfps,
            stationary,
            self.FS,
            minimum_duration=0.05,
            maximum_duration=0.05,
        )

        assert len(events) == 0 or (events.end_time - events.start_time).max() < 0.06


class TestMultiunitActiveUnits:
    """Most multiunit papers require a minimum number of units in a burst.

    Covers the count's two edges too: the last sample, and the threshold itself.
    """

    FS = 1000
    N_TIME = 5_000
    N_UNITS = 20

    @pytest.fixture
    def multiunit(self):
        """A three-unit burst at 1.0 s and a ten-unit burst at 3.0 s."""
        multiunit = np.zeros((self.N_TIME, self.N_UNITS))
        multiunit[1_000:1_060, :3] = 4.0
        multiunit[3_000:3_060, :10] = 1.2
        return multiunit

    def test_reports_the_active_unit_count(self, time, multiunit, stationary):
        """Every event carries the number of units that spiked inside it."""
        events = multiunit_HSE_detector(time, multiunit, stationary, self.FS)

        assert "n_active_units" in events
        assert sorted(events.n_active_units) == [3, 10]

    def test_minimum_drops_the_sparse_burst(self, time, multiunit, stationary):
        """A five-unit minimum keeps only the ten-unit burst."""
        events = multiunit_HSE_detector(
            time, multiunit, stationary, self.FS, minimum_active_units=5
        )

        assert len(events) == 1
        assert events.iloc[0].n_active_units == 10
        assert events.iloc[0].start_time > 2.0

    def test_index_is_renumbered_after_filtering(self, time, multiunit, stationary):
        """Every detector returns event_number 1..n with no holes."""
        events = multiunit_HSE_detector(
            time, multiunit, stationary, self.FS, minimum_active_units=5
        )

        assert list(events.index) == list(range(1, len(events) + 1))
        assert events.index.name == "event_number"

    def test_default_drops_nothing(self, time, multiunit, stationary):
        """The default keeps what the detector found before the criterion existed."""
        default = multiunit_HSE_detector(time, multiunit, stationary, self.FS)
        explicit = multiunit_HSE_detector(
            time, multiunit, stationary, self.FS, minimum_active_units=0
        )

        pd.testing.assert_frame_equal(default, explicit)

    def test_counts_units_not_spikes(self, time, stationary):
        """One unit firing many times is one active unit."""
        multiunit = np.zeros((self.N_TIME, self.N_UNITS))
        multiunit[1_000:1_060, 0] = 20.0

        events = multiunit_HSE_detector(time, multiunit, stationary, self.FS)

        assert (events.n_active_units == 1).all()

    def test_empty_result_still_has_the_column(self, time, stationary):
        """A detector that finds nothing returns the column anyway."""
        multiunit = np.zeros((self.N_TIME, self.N_UNITS))
        multiunit[1_000:1_060, 0] = 3.0  # one unit bursts, four short of the minimum

        events = multiunit_HSE_detector(
            time, multiunit, stationary, self.FS, minimum_active_units=5
        )

        assert len(events) == 0
        assert "n_active_units" in events

    def test_negative_minimum_raises(self, time, multiunit, stationary):
        """A negative unit count is not a criterion."""
        with pytest.raises(ValueError, match="minimum_active_units"):
            multiunit_HSE_detector(
                time, multiunit, stationary, self.FS, minimum_active_units=-1
            )

    def test_exactly_the_minimum_number_of_units_is_kept(self, time, stationary):
        multiunit = np.zeros((self.N_TIME, 20))
        multiunit[1_000:1_060, :5] = 3.0

        events = multiunit_HSE_detector(
            time, multiunit, stationary, self.FS, minimum_active_units=5
        )

        assert len(events) == 1
        assert events.iloc[0].n_active_units == 5

    def test_a_unit_spiking_only_on_the_last_sample_counts(self, time, stationary):
        """The event's own final sample is inside it.

        The burst runs to the end of the recording, so the event's last sample
        is the record's last sample and cannot move. Anywhere else, adding a
        spike extends the event past it and the question does not arise.
        """
        multiunit = np.zeros((self.N_TIME, 20))
        multiunit[self.N_TIME - 60 :, :4] = 3.0

        events = multiunit_HSE_detector(time, multiunit, stationary, self.FS)
        assert len(events) == 1
        assert events.iloc[0].end_time == time[-1]

        with_late_unit = multiunit.copy()
        with_late_unit[-1, 7] = 1.0
        late = multiunit_HSE_detector(time, with_late_unit, stationary, self.FS)

        assert late.iloc[0].end_time == time[-1]
        assert late.iloc[0].n_active_units == events.iloc[0].n_active_units + 1


class TestDurationLimitValidation:
    """The two detectors that always had a ceiling validate it like the rest."""

    FS = 1000
    N_TIME = 5_000

    @pytest.fixture
    def lfps(self):
        return _synthetic_ripple_band(self.N_TIME, self.FS, [(1_000, 1_060, 20.0)])

    def test_zugaro_rejects_a_ceiling_below_the_minimum(self, time, lfps, stationary):
        with pytest.raises(ValueError, match="maximum_duration"):
            Zugaro_ripple_detector(
                time,
                lfps,
                stationary,
                self.FS,
                minimum_duration=0.500,
                maximum_duration=0.100,
            )

    def test_long_rejects_a_sharp_wave_ceiling_below_its_minimum(self, time, stationary):
        raw = _synthetic_two_channel_lfp(self.N_TIME, self.FS, [1_000])
        with pytest.raises(ValueError, match="maximum"):
            Long_sharp_wave_ripple_detector(
                time,
                raw,
                stationary,
                self.FS,
                minimum_sharp_wave_duration=0.500,
                maximum_sharp_wave_duration=0.100,
            )


class TestCountActiveUnits:
    """The rule both spike-based detectors share."""

    MULTIUNIT = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    def test_counts_units_not_spikes(self):
        """Unit 0 spikes twice in this window and counts once."""
        assert _count_active_units(self.MULTIUNIT, [[0, 2]]).tolist() == [1]

    def test_both_bounds_are_inside_the_event(self):
        assert _count_active_units(self.MULTIUNIT, [[0, 3]]).tolist() == [2]
        assert _count_active_units(self.MULTIUNIT, [[1, 2]]).tolist() == [1]
        assert _count_active_units(self.MULTIUNIT, [[1, 1]]).tolist() == [0]

    def test_one_count_per_event(self):
        counts = _count_active_units(self.MULTIUNIT, [[0, 0], [3, 3], [0, 3]])

        assert counts.tolist() == [1, 1, 2]

    def test_no_events_gives_an_empty_integer_array(self):
        counts = _count_active_units(self.MULTIUNIT, np.empty((0, 2)))

        assert counts.shape == (0,)
        assert counts.dtype == int


class TestTimeMustBeIncreasing:
    """Every event and speed lookup bisects the timestamps, so unsorted time is
    rejected rather than producing plausible nonsense."""

    FS = 1000
    N_TIME = 5000

    @pytest.fixture
    def swapped_time(self):
        time = np.arange(self.N_TIME) / self.FS
        time[[100, 200]] = time[[200, 100]]
        return time

    @pytest.mark.parametrize(
        "detector",
        [
            Kay_ripple_detector,
            Karlsson_ripple_detector,
            Roumis_ripple_detector,
            Shvartsman_ripple_detector,
            Yu_ripple_detector,
            Zugaro_ripple_detector,
        ],
    )
    def test_ripple_band_detectors_raise(self, detector, swapped_time):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(2000, 2060, 20.0)])
        with pytest.raises(ValueError, match="increasing"):
            detector(swapped_time, lfps, np.full(self.N_TIME, 2.0), self.FS)

    def test_burst_detector_raises(self, swapped_time):
        multiunit = np.zeros((self.N_TIME, 4))
        multiunit[2000:2060] = 1.0
        with pytest.raises(ValueError, match="increasing"):
            multiunit_HSE_detector(swapped_time, multiunit, np.full(self.N_TIME, 2.0), self.FS)

    def test_two_channel_detector_raises(self, swapped_time):
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, [2500])
        with pytest.raises(ValueError, match="increasing"):
            Long_sharp_wave_ripple_detector(
                swapped_time, lfp, np.full(self.N_TIME, 2.0), self.FS
            )

    def test_repeated_timestamps_raise(self):
        time = np.repeat(np.arange(0, self.N_TIME, 3), 3)[: self.N_TIME] / self.FS
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(2000, 2060, 20.0)])
        with pytest.raises(ValueError, match="repeat"):
            Kay_ripple_detector(time, lfps, np.full(self.N_TIME, 2.0), self.FS)


class TestExclusionOrder:
    """Close events are excluded before over-long ones, so an over-long event
    still suppresses its neighbor before it is itself dropped."""

    FS = 1000
    N_TIME = 10_000

    @pytest.mark.parametrize("detector", [Kay_ripple_detector, Karlsson_ripple_detector])
    def test_a_long_event_suppresses_its_neighbor_before_being_dropped(self, detector):
        time = np.arange(self.N_TIME) / self.FS
        lfps = _synthetic_ripple_band(
            self.N_TIME, self.FS, [(5000, 5500, 20.0), (5560, 5600, 20.0)]
        )
        speed = np.full(self.N_TIME, 2.0)
        both = detector(time, lfps, speed, self.FS)
        assert len(both) == 2
        neither = detector(
            time, lfps, speed, self.FS, close_ripple_threshold=0.1, maximum_duration=0.2
        )
        assert len(neither) == 0


class TestCareyMinimumDuration:
    FS = 1000
    N_TIME = 20_000

    def test_a_minimum_longer_than_the_bursts_removes_them(self):
        time = np.arange(self.N_TIME) / self.FS
        lfps, multiunit = _synthetic_joint_inputs(self.N_TIME, self.FS, (5000, 12000))
        speed = np.full(self.N_TIME, 2.0)
        found = Carey_candidate_detector(time, lfps, multiunit, speed, self.FS)
        assert len(found) >= 1
        n_min = int(np.ceil(found.duration.max() * self.FS)) + 5
        none = Carey_candidate_detector(
            time, lfps, multiunit, speed, self.FS, minimum_duration=n_min / self.FS
        )
        assert len(none) == 0


class TestNoEventSpansAGap:
    """The one missing-sample policy: a NaN in any signal or in speed ends a
    block, nothing is computed across the gap, no event spans it, and an event
    cut off by the gap is flagged. Here a gap is cut through the middle of a
    burst, so every detector must return one event ending on the sample before
    the gap and one starting on the sample after it."""

    FS = 1000
    N_TIME = 20_000
    BURST = (5000, 5080)
    GAP = (5035, 5045)

    @staticmethod
    def _check(events, time, gap):
        gap_start, gap_stop = gap
        assert not any(
            (events.start_time < time[gap_start]) & (events.end_time > time[gap_stop - 1])
        ), "an event spans the gap"
        before = events[events.end_time == time[gap_start - 1]]
        after = events[events.start_time == time[gap_stop]]
        assert len(before) == 1
        assert len(after) == 1
        assert before.clipped_end.item()
        assert after.clipped_start.item()
        assert not before.clipped_start.item()
        assert not after.clipped_end.item()
        assert np.isfinite(events.select_dtypes(float).to_numpy()).all()

    @staticmethod
    def _cut(gap, time, *arrays):
        """The gap as concatenated disjoint intervals: the samples are deleted
        and the timestamps jump, with no NaN anywhere."""
        keep = np.ones(len(time), dtype=bool)
        keep[slice(*gap)] = False
        return (time[keep], *(array[keep] for array in arrays))

    @pytest.mark.parametrize(
        "detector",
        [
            Kay_ripple_detector,
            Karlsson_ripple_detector,
            Roumis_ripple_detector,
            Shvartsman_ripple_detector,
            Yu_ripple_detector,
            Zugaro_ripple_detector,
        ],
    )
    @pytest.mark.parametrize("where", ["lfp", "speed", "time"])
    def test_ripple_band_detectors(self, detector, where, time):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(*self.BURST, 20.0)])
        speed = np.full(self.N_TIME, 2.0)
        kwargs = {"maximum_duration": None} if detector is Zugaro_ripple_detector else {}
        if where == "lfp":
            lfps[slice(*self.GAP), 0] = np.nan
            events = detector(time, lfps, speed, self.FS, **kwargs)
        elif where == "speed":
            speed[slice(*self.GAP)] = np.nan
            events = detector(time, lfps, speed, self.FS, **kwargs)
        else:
            events = detector(*self._cut(self.GAP, time, lfps, speed), self.FS, **kwargs)
        self._check(events, time, self.GAP)

    @pytest.mark.parametrize(
        "detector",
        [
            Kay_ripple_detector,
            Karlsson_ripple_detector,
            Roumis_ripple_detector,
            Shvartsman_ripple_detector,
            Yu_ripple_detector,
            Zugaro_ripple_detector,
        ],
    )
    def test_a_single_dropped_sample_is_a_gap(self, detector, time):
        """One missing timestamp is a step of two intervals, past the 1.5 the
        rule allows, so it ends a block like any longer gap."""
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(*self.BURST, 20.0)])
        speed = np.full(self.N_TIME, 2.0)
        kwargs = {"maximum_duration": None} if detector is Zugaro_ripple_detector else {}
        gap = (5040, 5041)
        events = detector(*self._cut(gap, time, lfps, speed), self.FS, **kwargs)
        self._check(events, time, gap)

    @pytest.mark.parametrize("where", ["spikes", "time"])
    def test_burst_detector(self, where, time):
        multiunit = np.zeros((self.N_TIME, 6))
        multiunit[slice(*self.BURST)] = 1.0
        speed = np.full(self.N_TIME, 2.0)
        if where == "spikes":
            multiunit[slice(*self.GAP), 2] = np.nan
            events = multiunit_HSE_detector(time, multiunit, speed, self.FS)
        else:
            events = multiunit_HSE_detector(
                *self._cut(self.GAP, time, multiunit, speed), self.FS
            )
        self._check(events, time, self.GAP)

    @pytest.mark.parametrize("where", ["spikes", "time"])
    def test_joint_detector(self, where, time):
        lfps, multiunit = _synthetic_joint_inputs(self.N_TIME, self.FS, (self.BURST[0] + 40,))
        gap = (self.BURST[0] + 35, self.BURST[0] + 45)
        speed = np.full(self.N_TIME, 2.0)
        if where == "spikes":
            multiunit[slice(*gap), 0] = np.nan
            events = Carey_candidate_detector(
                time, lfps, multiunit, speed, self.FS, minimum_active_units=1
            )
        else:
            events = Carey_candidate_detector(
                *self._cut(gap, time, lfps, multiunit, speed), self.FS, minimum_active_units=1
            )
        assert not any(
            (events.start_time < time[gap[0]]) & (events.end_time > time[gap[1] - 1])
        )
        assert len(events) >= 1

    def test_two_channel_detector_never_evaluates_a_candidate_across_a_gap(self, time):
        centers = (3000, 7000, 11000, 15000)
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, centers)
        lfp[7000:7002, 1] = np.nan  # a gap through the second event
        # candidates within the 5 s local window of a block edge are not evaluated
        events = Long_sharp_wave_ripple_detector(
            time, lfp, np.full(self.N_TIME, 2.0), self.FS, random_state=0
        )
        assert not any((events.start_time < time[7000]) & (events.end_time > time[7001]))
        assert not events.clipped_start.any()
        assert not events.clipped_end.any()


class TestGapRuleUsesTheObservedStep:
    """The block splitter measures the sample step from the timestamps. With a
    nominal rate 1.5 times too high, every step would exceed 1.5 nominal
    intervals and every sample would be its own block, so no event could be
    found; measured from the data, the blocks and the events are unchanged."""

    FS = 1000
    N_TIME = 20_000  # enough immobility for the Yu noise threshold

    @pytest.mark.parametrize(
        "detector",
        [
            Kay_ripple_detector,
            Karlsson_ripple_detector,
            Yu_ripple_detector,
            Zugaro_ripple_detector,
        ],
    )
    def test_an_overstated_rate_warns_but_still_finds_the_ripples(
        self, detector, time, stationary
    ):
        bursts = [(4000, 4060, 20.0), (10000, 10060, 20.0), (16000, 16060, 20.0)]
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, bursts)
        with pytest.warns(UserWarning, match="differs from expected sampling interval"):
            events = detector(time, lfps, stationary, 1.6 * self.FS)
        for start, _, _ in bursts:
            assert any(
                (events.start_time <= time[start + 30]) & (events.end_time >= time[start + 30])
            )


class TestValidationPaths:
    FS = 1000
    N_TIME = 5000

    def test_speed_that_looks_like_metres_per_second_warns(self, time):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(2000, 2060, 20.0)])
        speed = np.full(self.N_TIME, 0.02)  # 2 cm/s written in m/s
        with pytest.warns(UserWarning, match="cm/s, not m/s"):
            Kay_ripple_detector(time, lfps, speed, self.FS)

    def test_a_mask_that_selects_only_missing_samples_raises(self, time, stationary):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(2000, 2060, 20.0)])
        lfps[:1000] = np.nan
        mask = np.zeros(self.N_TIME, dtype=bool)
        mask[:1000] = True
        with pytest.raises(ValueError, match="no sample that is finite"):
            Kay_ripple_detector(time, lfps, stationary, self.FS, normalization_mask=mask)

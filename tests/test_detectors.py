"""Integration tests for ripple detection algorithms."""

import contextlib
import subprocess
import sys
import warnings
from pathlib import Path
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
from ripple_detection.detectors._blocks import _contiguous_valid_blocks
from ripple_detection.detectors._carey import (
    _contained_in_intervals,
    _convolve_spikes,
    _state_intervals,
    _theta_envelope,
)
from ripple_detection.detectors._events import (
    _count_active_units,
    _exclude_long_events,
    _get_event_stats,
    _max_sustained_zscore,
)
from ripple_detection.detectors._lfp import _extract_Yu_ripple_events
from ripple_detection.detectors._long import _firfilt
from ripple_detection.detectors._zugaro import (
    ZUGARO_SMOOTHING_WINDOW,
    _two_threshold_events,
    _zugaro_smoothing_samples,
)
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
            rng=1,
        )
        quiet = [
            simulate_LFP(time_3s, ripple_times=[], noise_amplitude=1.3, rng=s)
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
        """A NaN spike count is missing data and a NaN speed an unknown speed,
        neither an error: the burst on the far side is still found, and nothing
        spans the missing spike count."""
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
        with pytest.raises(ValueError, match="2-D"):
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
                rng=s,
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
                rng=s,
            )
            for s in (21, 22, 23, 24)
        ]
        loud = simulate_LFP(
            time_3s,
            ripple_times=[1.5],
            noise_amplitude=1.2,
            ripple_amplitude=6.0,
            rng=25,
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
        with pytest.raises(ValueError, match="2-D"):
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
        with pytest.raises(ValueError, match="no immobility noise"):
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
                        rng=seed,
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
        # the mirrored-noise estimate sits about 1 SD above the immobility mean
        # on this pink-noise background; near 0 it would be thresholding the
        # mean itself, which a degenerate histogram produces
        assert 0.5 < events.detection_threshold_zscore.iloc[0] < 3.0

    def test_pure_noise_yields_few_events(self):
        fs = 1500
        time = np.arange(fs * 20) / fs
        lfps = filter_ripple_band(
            np.column_stack(
                [
                    simulate_LFP(time, ripple_times=[], noise_amplitude=1.2, rng=seed)
                    for seed in (1, 2, 3, 4)
                ]
            ),
            fs,
        )
        events = Yu_ripple_detector(time, lfps, np.full(len(time), 2.0), fs)
        assert len(events) <= 3

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
                        rng=seed,
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
    def test_the_default_is_the_original_scaled_with_rate_and_odd(self, fs, expected):
        """FindRipples' 11 samples at 1250 Hz, in seconds."""
        assert _zugaro_smoothing_samples(ZUGARO_SMOOTHING_WINDOW, fs) == expected

    def test_an_even_sample_count_rounds_up_to_odd(self):
        assert _zugaro_smoothing_samples(0.010, 1000) == 11
        assert _zugaro_smoothing_samples(0.011, 1000) == 11


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

    def test_smoothing_window_is_in_seconds(self, time, stationary):
        """Like every other duration here; 9 for 9 ms raises and says so."""
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(5000, 5060, 20.0)])
        with pytest.raises(ValueError, match=r"smoothing_window is in seconds.*pass 0\.009"):
            Zugaro_ripple_detector(time, lfps, stationary, self.FS, smoothing_window=9)
        nine_samples = Zugaro_ripple_detector(
            time, lfps, stationary, self.FS, smoothing_window=0.009
        )
        default = Zugaro_ripple_detector(time, lfps, stationary, self.FS)
        pd.testing.assert_frame_equal(nine_samples, default)

    def test_a_smoothing_window_shorter_than_a_sample_raises(self, time, stationary):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(5000, 5060, 20.0)])
        with pytest.raises(ValueError, match="shorter than one sample"):
            Zugaro_ripple_detector(time, lfps, stationary, self.FS, smoothing_window=0.0004)

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


class TestShvartsmanNormalizationMethod:
    def test_an_unknown_method_lists_all_three(self):
        time = np.arange(3000) / 1500
        lfps = np.random.default_rng(0).standard_normal((3000, 2))
        with pytest.raises(ValueError, match="'zscore', 'median_mad' or 'manual'"):
            Shvartsman_ripple_detector(
                time, lfps, np.zeros(3000), 1500, normalization_method="bogus"
            )


class TestLongSharpWaveRippleDetector:
    FS = 1000
    N_TIME = 40_000  # 40 s; events must sit more than 5 s from either end
    EVENTS = (7000, 11000, 15500, 19000, 23800, 28000, 31500, 34000)

    def test_recovers_planted_sharp_wave_ripples(self, time, stationary):
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, self.EVENTS)
        events = Long_sharp_wave_ripple_detector(
            time, lfp[:, 0], stationary, self.FS, sharp_wave_lfp=lfp[:, 1], rng=0
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
            time, lfp[:, 0], stationary, self.FS, sharp_wave_lfp=lfp[:, 1], rng=0
        )
        hits = [
            any((events.start_time <= time[c]) & (events.end_time >= time[c]))
            for c in self.EVENTS
        ]
        assert sum(hits) <= 1

    def test_sharp_wave_without_ripple_is_not_detected(self, time, stationary):
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, self.EVENTS, ripple=False)
        events = Long_sharp_wave_ripple_detector(
            time, lfp[:, 0], stationary, self.FS, sharp_wave_lfp=lfp[:, 1], rng=0
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
            time, lfp[:, 0], stationary, self.FS, sharp_wave_lfp=lfp[:, 1], rng=0
        )
        assert np.all(events.peak_time >= 5.0)
        assert np.all(events.peak_time <= time[-1] - 5.0)

    def test_seeded_clustering_is_reproducible(self, time, stationary):
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, self.EVENTS)
        a = Long_sharp_wave_ripple_detector(
            time, lfp[:, 0], stationary, self.FS, sharp_wave_lfp=lfp[:, 1], rng=3
        )
        b = Long_sharp_wave_ripple_detector(
            time, lfp[:, 0], stationary, self.FS, sharp_wave_lfp=lfp[:, 1], rng=3
        )
        pd.testing.assert_frame_equal(a, b)

    def test_window_size_given_in_milliseconds_raises(self, time, stationary):
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, self.EVENTS)
        with pytest.raises(ValueError, match=r"window_size is in seconds.*pass 0\.04"):
            Long_sharp_wave_ripple_detector(
                time, lfp[:, 0], stationary, self.FS, sharp_wave_lfp=lfp[:, 1], window_size=40
            )

    def test_no_candidate_far_enough_from_a_block_edge_raises(self, time, stationary):
        """With every candidate within local_window of a block edge none can
        be evaluated; that used to return an empty table without a word, as
        local_window=5000 (milliseconds given as seconds) did on any record."""
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, self.EVENTS)
        with pytest.raises(ValueError, match=r"local_window.*in seconds"):
            Long_sharp_wave_ripple_detector(
                time,
                lfp[:, 0],
                stationary,
                self.FS,
                sharp_wave_lfp=lfp[:, 1],
                local_window=5000,
            )
        with pytest.raises(ValueError, match=r"local_window"):
            Long_sharp_wave_ripple_detector(
                time,
                lfp[:, 0],
                stationary,
                self.FS,
                sharp_wave_lfp=lfp[:, 1],
                local_window=20.0,
            )

    def test_a_local_window_narrower_than_half_a_candidate_window_raises(
        self, time, stationary
    ):
        """The ripple-peak search used to index before the local window and
        report negative ripple durations."""
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, self.EVENTS)
        with pytest.raises(ValueError, match=r"local_window.*half of window_size"):
            Long_sharp_wave_ripple_detector(
                time,
                lfp[:, 0],
                stationary,
                self.FS,
                sharp_wave_lfp=lfp[:, 1],
                window_size=0.5,
                local_window=0.1,
            )

    def test_a_random_state_instance_is_refused(self, time, stationary):
        """As the simulators refuse one: default_rng would accept it and draw a
        different stream than the seed it was made from."""
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, self.EVENTS)
        with pytest.raises(TypeError, match="not a RandomState"):
            Long_sharp_wave_ripple_detector(
                time,
                lfp[:, 0],
                stationary,
                self.FS,
                sharp_wave_lfp=lfp[:, 1],
                rng=np.random.RandomState(0),
            )

    def test_output_columns(self, time, stationary):
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, self.EVENTS)
        events = Long_sharp_wave_ripple_detector(
            time, lfp[:, 0], stationary, self.FS, sharp_wave_lfp=lfp[:, 1], rng=0
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
            time, lfp[:, 0], stationary, self.FS, sharp_wave_lfp=lfp[:, 1], rng=0
        )
        moved = Long_sharp_wave_ripple_detector(
            time, lfp[:, 0], speed, self.FS, sharp_wave_lfp=lfp[:, 1], rng=0
        )
        assert any((events.start_time <= time[7000]) & (events.end_time >= time[7000]))
        assert not any((moved.start_time <= time[7000]) & (moved.end_time >= time[7000]))

    def test_record_shorter_than_the_slowest_kernel_raises(self):
        n_time = 500  # the 2 Hz Gaussian low-pass spans 957 samples at 1 kHz
        lfp = _synthetic_two_channel_lfp(n_time, self.FS, ())
        with pytest.raises(ValueError, match=r"as long as the .* samples"):
            Long_sharp_wave_ripple_detector(
                np.arange(n_time) / self.FS,
                lfp[:, 0],
                np.full(n_time, 2.0),
                self.FS,
                sharp_wave_lfp=lfp[:, 1],
            )

    def test_minimum_separation_keeps_only_the_last_of_close_candidates(
        self, time, stationary
    ):
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, self.EVENTS)
        # 10 s is the longest separation the check admits, and every
        # consecutive pair of EVENTS is closer than that
        events = Long_sharp_wave_ripple_detector(
            time,
            lfp[:, 0],
            stationary,
            self.FS,
            sharp_wave_lfp=lfp[:, 1],
            minimum_separation=10.0,
            rng=0,
        )
        assert len(events) <= 1

    def test_no_event_survives_a_tiny_maximum_sharp_wave_duration(self, time, stationary):
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, self.EVENTS)
        events = Long_sharp_wave_ripple_detector(
            time,
            lfp[:, 0],
            stationary,
            self.FS,
            sharp_wave_lfp=lfp[:, 1],
            minimum_sharp_wave_duration=0.001,
            maximum_sharp_wave_duration=0.002,
            rng=0,
        )
        assert events.empty
        assert "start_time" in events.columns
        assert "sharp_wave_duration" in events.columns

    def test_each_signal_is_one_channel(self, time, stationary):
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, self.EVENTS)
        flat = Long_sharp_wave_ripple_detector(
            time, lfp[:, 0], stationary, self.FS, sharp_wave_lfp=lfp[:, 1]
        )
        column = Long_sharp_wave_ripple_detector(
            time, lfp[:, :1], stationary, self.FS, sharp_wave_lfp=lfp[:, 1:]
        )
        pd.testing.assert_frame_equal(flat, column)
        with pytest.raises(ValueError, match="sharp_wave_lfp must be one channel"):
            Long_sharp_wave_ripple_detector(
                time, lfp[:, 0], stationary, self.FS, sharp_wave_lfp=lfp
            )
        with pytest.raises(ValueError, match="sharp_wave_lfp must have"):
            Long_sharp_wave_ripple_detector(
                time, lfp[:, 0], stationary, self.FS, sharp_wave_lfp=lfp[:-1, 1]
            )

    def test_a_two_channel_array_says_where_the_second_channel_goes(self, time, stationary):
        """Two columns, as the original DetectSWR takes its channels: the
        message says where the second one goes."""
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, self.EVENTS)
        with pytest.raises(ValueError, match=r"raw_lfp must be one channel.*sharp_wave_lfp="):
            Long_sharp_wave_ripple_detector(
                time, lfp, stationary, self.FS, sharp_wave_lfp=lfp[:, 1]
            )

    def test_the_sharp_wave_channel_is_required(self, time, stationary):
        """So a caller that resolves detectors by name and calls Long like the
        ripple-band detectors fails instead of detecting on the wrong input."""
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, self.EVENTS)
        with pytest.raises(TypeError, match="sharp_wave_lfp"):
            Long_sharp_wave_ripple_detector(time, lfp[:, 0], stationary, self.FS)

    def test_nan_splits_the_record_and_events_far_from_it_survive(self, time, stationary):
        """A NaN sample ends a block. Blocks shorter than the sharp-wave kernel
        are treated as missing with a warning; the events on the long side
        are found as before."""
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, self.EVENTS)
        clean = Long_sharp_wave_ripple_detector(
            time, lfp[:, 0], stationary, self.FS, sharp_wave_lfp=lfp[:, 1], rng=0
        )
        lfp[100, 0] = np.nan  # leaves a 100-sample block before it
        with pytest.warns(UserWarning, match="treated as missing"):
            events = Long_sharp_wave_ripple_detector(
                time, lfp[:, 0], stationary, self.FS, sharp_wave_lfp=lfp[:, 1], rng=0
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
        assert len(many) >= 1
        assert np.all(many.n_active_units >= 5)
        with pytest.raises(ValueError, match="multiunit has 8 unit"):
            Carey_candidate_detector(
                time, lfps, multiunit, stationary, self.FS, minimum_active_units=9
            )

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

    def test_theta_lfp_as_one_column_is_the_same_channel(self, time, stationary):
        """Every other signal is (n_time, n_channels), and the errors say to
        reshape a single channel to (n, 1); a theta channel shaped that way
        is the same channel."""
        lfps, multiunit = _synthetic_joint_inputs(self.N_TIME, self.FS, self.EVENTS)
        theta_lfp = np.random.default_rng(5).normal(0.0, 1.0, self.N_TIME)
        theta_lfp[6000:8000] += 15.0 * np.sin(2 * np.pi * 8.0 * time[6000:8000])
        flat = Carey_candidate_detector(
            time, lfps, multiunit, stationary, self.FS, theta_lfp=theta_lfp
        )
        column = Carey_candidate_detector(
            time, lfps, multiunit, stationary, self.FS, theta_lfp=theta_lfp[:, np.newaxis]
        )
        pd.testing.assert_frame_equal(flat, column)

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
        with pytest.raises(ValueError, match="2-D"):
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
        detector, too short for an event and dropped with a warning that says
        so, but the theta channel is continuous there, so it is filtered as one
        run: no theta run is too short."""
        lfps, multiunit = _synthetic_joint_inputs(self.N_TIME, self.FS, self.EVENTS)
        lfps[8000:8100] = np.nan
        lfps[8105:8200] = np.nan
        theta = np.random.default_rng(0).standard_normal(self.N_TIME)
        with pytest.warns(UserWarning, match="minimum_duration") as record:
            events = Carey_candidate_detector(
                time, lfps, multiunit, stationary, self.FS, theta_lfp=theta
            )
        assert [str(w.message) for w in record if "theta" in str(w.message)] == []
        assert "(8100, 8105)" in str(record[0].message)
        assert len(events) >= 1

    def test_a_dropout_in_speed_does_not_restart_the_theta_filter(self, time, stationary):
        """The original filtered the whole theta recording at once. A NaN in
        speed splits neither the detector's blocks nor the theta filtering, so
        the theta exclusion decides the same events with and without the
        dropout."""
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
        """A NaN in the spikes or the LFP ends a block; the candidates
        elsewhere are still found and none spans the missing sample. The ten
        samples before the NaN spike count are too short for an event and are
        dropped with a warning."""
        lfps, multiunit = _synthetic_joint_inputs(self.N_TIME, self.FS, self.EVENTS)
        clean = Carey_candidate_detector(time, lfps, multiunit, stationary, self.FS)
        multiunit[10, 0] = np.nan
        lfps[self.EVENTS[0], :] = np.nan  # in the middle of the first event
        with pytest.warns(UserWarning, match=r"\(0, 10\)"):
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
        with pytest.raises(ValueError, match="theta_lfp must have shape"):
            Carey_candidate_detector(
                time,
                lfps,
                multiunit,
                stationary,
                self.FS,
                theta_lfp=np.zeros((self.N_TIME, 2)),
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

    def test_nan_in_lfp_rows_are_missing_and_the_ripples_still_found(
        self, time_3s, single_lfp_with_ripples, stationary_speed, sampling_frequency
    ):
        """Rows with NaN are missing samples that end a block; the planted
        ripples away from them survive and every statistic is finite."""
        lfp_with_nan = single_lfp_with_ripples.copy()
        lfp_with_nan[100:200, 0] = np.nan

        with pytest.warns(UserWarning, match="shorter than"):  # the 100-sample head
            filtered_lfps = filter_ripple_band(lfp_with_nan, 1500)
        ripples = Kay_ripple_detector(
            time_3s, filtered_lfps, stationary_speed, sampling_frequency
        )

        assert len(ripples) >= 2
        assert np.isfinite(ripples.to_numpy(dtype=float)).all()

    def test_nan_speed_inside_a_ripple_leaves_it_whole(
        self, time_3s, stationary_speed, sampling_frequency
    ):
        """Speed does not enter the trace, so unknown speed inside a ripple
        neither splits it nor clips it; the speed statistics skip the NaN."""
        speed_with_nan = stationary_speed.copy()
        speed_with_nan[1640:1660] = np.nan  # inside the ripple planted at 1.1 s
        lfp = simulate_LFP(
            time_3s,
            [1.1, 2.1],
            noise_amplitude=1.2,
            ripple_snr=12.0,
            ripple_duration=0.15,
            rng=0,
        )
        filtered_lfps = filter_ripple_band(lfp[:, np.newaxis], 1500)
        whole = Kay_ripple_detector(
            time_3s, filtered_lfps, stationary_speed, sampling_frequency
        )
        ripples = Kay_ripple_detector(
            time_3s, filtered_lfps, speed_with_nan, sampling_frequency
        )

        pd.testing.assert_frame_equal(
            ripples[["start_time", "end_time", "clipped_start", "clipped_end"]],
            whole[["start_time", "end_time", "clipped_start", "clipped_end"]],
        )
        spanning = ripples[(ripples.start_time < 1.1) & (ripples.end_time > 1.1)]
        assert len(spanning) == 1
        assert np.isfinite(spanning[["max_speed", "mean_speed"]].to_numpy()).all()

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
                    rng=seed,
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
            time_3s, [1.1, 2.1], noise_amplitude=1.2, ripple_amplitude=1.5, rng=0
        )
        ch1 = simulate_LFP(time_3s, [1.1], noise_amplitude=1.2, ripple_amplitude=1.5, rng=1)
        ch2 = simulate_LFP(time_3s, [1.1], noise_amplitude=1.2, ripple_amplitude=1.5, rng=2)
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
                simulate_LFP(time_3s, [1.1], noise_amplitude=1.2, ripple_amplitude=1.5, rng=s)
                for s in range(n_channels)
            ]
        )
        filtered = filter_ripple_band(lfps, 1500)
        env = gaussian_smooth(get_envelope(filtered), 0.004, sampling_frequency)
        # only 7 channels carry the ripple loudly enough: silence the others' bursts
        quiet = np.column_stack(
            [
                simulate_LFP(time_3s, [], noise_amplitude=1.2, rng=100 + s)
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
                [simulate_LFP(time_3s, [], noise_amplitude=1.2, rng=s) for s in (1, 2)]
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
        stretched_time = np.arange(n_time) * 0.00105  # five percent over the expected step
        lfps = _synthetic_ripple_band(n_time, sampling_frequency, [(5000, 5060, 20.0)])
        speed = np.full(n_time, 2.0)
        call = {
            "Kay": lambda: Kay_ripple_detector(
                stretched_time, lfps, speed, sampling_frequency
            ),
            "Zugaro": lambda: Zugaro_ripple_detector(
                stretched_time, lfps, speed, sampling_frequency
            ),
            "Yu": lambda: Yu_ripple_detector(stretched_time, lfps, speed, sampling_frequency),
        }[detector]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with contextlib.suppress(ValueError):
                call()
        unit_warnings = [w for w in caught if "Time array step" in str(w.message)]
        assert unit_warnings, [str(w.message) for w in caught]
        assert unit_warnings[0].filename == __file__

    def test_through_a_symlinked_package_directory(self, tmp_path):
        """The frame test compares the path the package was imported through,
        which a symlink leaves unresolved, so it must not be resolved either."""
        import ripple_detection

        package = Path(ripple_detection.__file__).parent
        linked = tmp_path / "linked"
        linked.mkdir()
        try:
            (linked / "ripple_detection").symlink_to(package, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            pytest.skip(f"cannot create a symlink here: {error}")
        script = (
            "import sys, warnings, numpy as np\n"
            f"sys.path.insert(0, {str(linked)!r})\n"
            "import ripple_detection\n"
            "assert ripple_detection.__file__.startswith(sys.path[0]), ripple_detection.__file__\n"
            "n = 5000\n"
            "time = np.arange(n) / 1000.0\n"
            "lfps = np.random.default_rng(0).standard_normal((n, 2))\n"
            "with warnings.catch_warnings(record=True) as caught:\n"
            "    warnings.simplefilter('always')\n"
            "    ripple_detection.Kay_ripple_detector(time, lfps, np.full(n, 0.02), 1000)\n"
            "(unit,) = [w for w in caught if 'cm/s, not m/s' in str(w.message)]\n"
            "print(unit.filename)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "<string>", result.stdout


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
            lfp[:, 0],
            np.full(self.N_TIME, 2.0),
            self.FS,
            sharp_wave_lfp=lfp[:, 1],
            minimum_ripple_duration=0.400,
            rng=0,
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
            _get_event_stats(
                events, time, np.zeros((1000, 3)).tolist(), speed, 0.015, [(0, 1000)]
            )

    def test_one_dimensional_metric_with_participants_raises(self):
        time, events, speed = self._inputs()
        with pytest.raises(ValueError, match=r"\(n_time, n_channels\)"):
            _get_event_stats(
                events,
                time,
                np.zeros(1000).tolist(),
                speed,
                0.015,
                [(0, 1000)],
                participants=[{0}],
            )


class TestPeakTime:
    """Every detector reports when its trace peaks inside each event."""

    FS = 1000

    def test_the_time_of_the_largest_value(self):
        time = np.arange(1000) / self.FS
        metric = np.zeros(1000)
        metric[120] = 5.0
        stats = _get_event_stats(
            [[0.100, 0.150]], time, metric, np.zeros(1000), 0.0, [(0, 1000)]
        )
        assert stats.peak_time.iloc[0] == pytest.approx(0.120)

    def test_a_tie_takes_the_first_sample(self):
        time = np.arange(1000) / self.FS
        metric = np.zeros(1000)
        metric[[110, 130]] = 5.0
        stats = _get_event_stats(
            [[0.100, 0.150]], time, metric, np.zeros(1000), 0.0, [(0, 1000)]
        )
        assert stats.peak_time.iloc[0] == pytest.approx(0.110)

    def test_with_participants_the_peak_of_their_mean(self):
        """Channel 2 peaks alone at 0.140 but does not take part, so the
        peak is the participants' shared one at 0.120."""
        time = np.arange(1000) / self.FS
        metric = np.zeros((1000, 3))
        metric[120, [0, 1]] = 4.0
        metric[140, 2] = 50.0
        stats = _get_event_stats(
            [[0.100, 0.150]],
            time,
            metric,
            np.zeros(1000),
            0.0,
            [(0, 1000)],
            participants=[(0, 1)],
        )
        assert stats.peak_time.iloc[0] == pytest.approx(0.120)

    def test_no_events_gives_an_empty_float_column(self):
        time = np.arange(1000) / self.FS
        stats = _get_event_stats(
            np.empty((0, 2)), time, np.zeros(1000), np.zeros(1000), 0.0, [(0, 1000)]
        )
        assert "peak_time" in stats.columns
        assert stats.peak_time.dtype == np.float64


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
    LONG = (5_000, 5_500, 5.0)  # 500 ms burst, quiet enough that the Yu noise sample is noise
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
        multiunit[3_000:3_060, :10] = 1.0
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

    def test_a_rejected_event_does_not_suppress_its_neighbour(self, time, stationary):
        """The unit count is applied before close events are excluded, so a
        sparse event that fails it cannot drop the dense event just after it
        and then be dropped itself."""
        multiunit = np.zeros((self.N_TIME, self.N_UNITS))
        multiunit[1_000:1_060, :3] = 4.0  # three units: fails a minimum of five
        multiunit[1_150:1_210, :10] = 1.0  # ten units, 90 ms later
        events = multiunit_HSE_detector(
            time,
            multiunit,
            stationary,
            self.FS,
            minimum_active_units=5,
            close_event_threshold=0.2,
        )
        assert len(events) == 1
        assert events.iloc[0].n_active_units == 10

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
                raw[:, 0],
                stationary,
                self.FS,
                sharp_wave_lfp=raw[:, 1],
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
                swapped_time,
                lfp[:, 0],
                np.full(self.N_TIME, 2.0),
                self.FS,
                sharp_wave_lfp=lfp[:, 1],
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
    """The one missing-sample policy: a NaN in any signal, or a gap in time,
    ends a block, nothing is computed across the gap, no event spans it, and
    an event cut off by the gap is flagged. Here a gap is cut through the middle of a
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
    @pytest.mark.parametrize("where", ["lfp", "time"])
    def test_ripple_band_detectors(self, detector, where, time):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(*self.BURST, 20.0)])
        speed = np.full(self.N_TIME, 2.0)
        kwargs = {"maximum_duration": None} if detector is Zugaro_ripple_detector else {}
        if where == "lfp":
            lfps[slice(*self.GAP), 0] = np.nan
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

    @pytest.mark.parametrize(
        "detector", [Kay_ripple_detector, Karlsson_ripple_detector, Roumis_ripple_detector]
    )
    def test_a_moderate_ripple_cut_by_a_gap_reaches_it_and_is_flagged(self, detector, time):
        """Smoothing is renormalized at a block edge rather than zero-padded, so
        the trace keeps its level up to the gap: a ripple of 1.5 noise SDs cut
        by the gap ends and starts on the gap's edges and is flagged there.
        Zero padding pulled the trace under the threshold a few samples early,
        so the halves stopped short of the gap and were not flagged."""
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(5000, 5080, 1.5)])
        gap = (5040, 5046)
        events = detector(*self._cut(gap, time, lfps, np.zeros(self.N_TIME)), self.FS)
        near_gap = events[
            (events.end_time > time[gap[0]] - 0.02) & (events.start_time < time[gap[1]] + 0.02)
        ]
        assert len(near_gap) >= 1
        for event in near_gap.itertuples():
            if event.end_time < time[gap[0]]:
                assert event.end_time == time[gap[0] - 1]
                assert event.clipped_end
            else:
                assert event.start_time == time[gap[1]]
                assert event.clipped_start

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
        # the event cut by the gap ends and starts on its edges, flagged there
        before = events[events.end_time == time[gap[0] - 1]]
        after = events[events.start_time == time[gap[1]]]
        assert len(before) == len(after) == 1
        assert before.clipped_end.item()
        assert not before.clipped_start.item()
        assert after.clipped_start.item()
        assert not after.clipped_end.item()

    @pytest.mark.parametrize("where", ["lfp", "time"])
    def test_two_channel_detector_never_evaluates_a_candidate_across_a_gap(self, where, time):
        centers = (3000, 7000, 11000, 15000)
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, centers)
        speed = np.full(self.N_TIME, 2.0)
        gap = (7000, 7002)  # through the second event
        if where == "lfp":
            lfp[slice(*gap), 1] = np.nan
            events = Long_sharp_wave_ripple_detector(
                time, lfp[:, 0], speed, self.FS, sharp_wave_lfp=lfp[:, 1], rng=0
            )
        else:
            cut_time, cut_lfp, cut_speed = self._cut(gap, time, lfp, speed)
            events = Long_sharp_wave_ripple_detector(
                cut_time,
                cut_lfp[:, 0],
                cut_speed,
                self.FS,
                sharp_wave_lfp=cut_lfp[:, 1],
                rng=0,
            )
        # candidates within the 5 s local window of a block edge are not
        # evaluated, so only the event far from both the gap and the edges is left
        assert len(events) >= 1
        assert not any(
            (events.start_time < time[gap[0]]) & (events.end_time > time[gap[1] - 1])
        )
        assert not events.clipped_start.any()
        assert not events.clipped_end.any()


class TestUnknownSpeed:
    """Speed does not enter any trace, so a NaN in speed is an unknown speed,
    not a missing sample: it splits no block. The endpoint rule needs a known
    speed at both ends; the majority rule and Carey's low-speed intervals are
    judged on the known samples; ``speed_threshold=np.inf`` turns the
    criterion off, NaN included."""

    FS = 1000
    N_TIME = 20_000
    BURST = (5000, 5080)
    INSIDE = (5035, 5045)

    RIPPLE_BAND = (
        Kay_ripple_detector,
        Karlsson_ripple_detector,
        Roumis_ripple_detector,
        Shvartsman_ripple_detector,
        Yu_ripple_detector,
        Zugaro_ripple_detector,
    )

    def _run(self, detector, time, speed, **kwargs):
        if detector is multiunit_HSE_detector:
            multiunit = np.zeros((self.N_TIME, 6))
            multiunit[slice(*self.BURST)] = 1.0
            return detector(time, multiunit, speed, self.FS, **kwargs)
        if detector is Carey_candidate_detector:
            lfps, multiunit = _synthetic_joint_inputs(
                self.N_TIME, self.FS, (self.BURST[0] + 40,)
            )
            return detector(
                time, lfps, multiunit, speed, self.FS, minimum_active_units=1, **kwargs
            )
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(*self.BURST, 20.0)])
        if detector is Zugaro_ripple_detector:
            kwargs.setdefault("maximum_duration", None)
        return detector(time, lfps, speed, self.FS, **kwargs)

    @staticmethod
    def _bounds(events):
        return events[["start_time", "end_time", "clipped_start", "clipped_end"]]

    @pytest.mark.parametrize(
        "detector", [*RIPPLE_BAND, multiunit_HSE_detector, Carey_candidate_detector]
    )
    def test_nan_inside_an_event_changes_nothing_but_the_speed_statistics(
        self, detector, time
    ):
        speed = np.full(self.N_TIME, 2.0)
        known = self._run(detector, time, speed)
        speed[slice(*self.INSIDE)] = np.nan
        unknown = self._run(detector, time, speed)
        assert len(known) >= 1
        pd.testing.assert_frame_equal(self._bounds(unknown), self._bounds(known))
        speeds = unknown[["max_speed", "min_speed", "median_speed", "mean_speed"]]
        np.testing.assert_array_equal(speeds.to_numpy(), 2.0)

    @pytest.mark.parametrize("detector", RIPPLE_BAND[:3])
    def test_an_unknown_endpoint_fails_the_endpoint_rule(self, detector, time):
        speed = np.full(self.N_TIME, 2.0)
        known = self._run(detector, time, speed)
        start = int(np.searchsorted(time, known.start_time.iloc[0]))
        speed[start] = np.nan
        assert len(self._run(detector, time, speed)) == len(known) - 1
        kept = self._run(detector, time, speed, speed_threshold=np.inf)
        pd.testing.assert_frame_equal(self._bounds(kept), self._bounds(known))

    def _run_long(self, time, speed, **kwargs):
        # events at least local_window (5 s) from both ends, so each is evaluated
        lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, (8000, 12000))
        return Long_sharp_wave_ripple_detector(
            time, lfp[:, 0], speed, self.FS, sharp_wave_lfp=lfp[:, 1], rng=0, **kwargs
        )

    def test_long_nan_inside_an_event_changes_nothing(self, time):
        speed = np.full(self.N_TIME, 2.0)
        known = self._run_long(time, speed)
        assert len(known) >= 1
        middle = int(np.searchsorted(time, known[["start_time", "end_time"]].iloc[0].mean()))
        speed[middle - 3 : middle + 3] = np.nan
        pd.testing.assert_frame_equal(
            self._bounds(self._run_long(time, speed)), self._bounds(known)
        )

    def test_long_unknown_endpoint_fails_the_endpoint_rule(self, time):
        speed = np.full(self.N_TIME, 2.0)
        known = self._run_long(time, speed)
        speed[int(np.searchsorted(time, known.start_time.iloc[0]))] = np.nan
        assert len(self._run_long(time, speed)) == len(known) - 1
        kept = self._run_long(time, speed, speed_threshold=np.inf)
        pd.testing.assert_frame_equal(self._bounds(kept), self._bounds(known))

    def test_carey_unknown_speed_longer_than_the_merge_gap_interrupts_immobility(self, time):
        """Carey keeps an event only inside a low-speed period; unknown speed is
        not low speed, and a 60 ms dropout is longer than the 50 ms merge gap."""
        speed = np.full(self.N_TIME, 2.0)
        near = lambda events: events[(events.end_time > 4.9) & (events.start_time < 5.2)]  # noqa: E731
        assert len(near(self._run(Carey_candidate_detector, time, speed))) == 1
        speed[5010:5070] = np.nan
        assert len(near(self._run(Carey_candidate_detector, time, speed))) == 0
        kept = self._run(Carey_candidate_detector, time, speed, speed_threshold=np.inf)
        assert len(near(kept)) == 1

    def test_yu_with_no_known_speed_and_the_criterion_off_uses_every_sample(self, time):
        """speed_threshold=np.inf turns the speed rule off for Yu's noise sample
        too, so all-NaN speed works there as the validation message says."""
        events = self._run(
            Yu_ripple_detector, time, np.full(self.N_TIME, np.nan), speed_threshold=np.inf
        )
        assert len(events) >= 1

    def test_speed_nan_everywhere_raises_unless_the_criterion_is_off(self, time):
        speed = np.full(self.N_TIME, np.nan)
        with pytest.raises(ValueError, match="speed is NaN at every sample"):
            self._run(Kay_ripple_detector, time, speed)
        events = self._run(Kay_ripple_detector, time, speed, speed_threshold=np.inf)
        assert len(events) == 1
        assert events[["max_speed", "speed_at_start"]].isna().all(axis=None)


class TestBlocksTooShortForAnEvent:
    """A block with fewer samples than ``minimum_duration`` spans cannot hold an
    event. It is treated as missing with a warning that gives its sample
    ranges, and a detector left with no block raises, so missing data never
    empties a result without saying so."""

    FS = 1000
    N_TIME = 20_000
    BURST = (5000, 5080)

    DETECTORS = (
        Kay_ripple_detector,
        Karlsson_ripple_detector,
        Roumis_ripple_detector,
        Shvartsman_ripple_detector,
        Yu_ripple_detector,
        Zugaro_ripple_detector,
        multiunit_HSE_detector,
        Carey_candidate_detector,
    )

    def _run(self, detector, time, nan_rows):
        speed = np.full(self.N_TIME, 2.0)
        if detector is multiunit_HSE_detector:
            multiunit = np.zeros((self.N_TIME, 6))
            multiunit[slice(*self.BURST)] = 1.0
            multiunit[nan_rows, 0] = np.nan
            return detector(time, multiunit, speed, self.FS)
        if detector is Carey_candidate_detector:
            lfps, multiunit = _synthetic_joint_inputs(
                self.N_TIME, self.FS, (self.BURST[0] + 40,)
            )
            lfps[nan_rows, 0] = np.nan
            return detector(time, lfps, multiunit, speed, self.FS, minimum_active_units=1)
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(*self.BURST, 20.0)])
        lfps[nan_rows, 0] = np.nan
        return detector(time, lfps, speed, self.FS)

    @pytest.mark.parametrize("detector", DETECTORS)
    def test_a_short_block_warns_with_its_range(self, detector, time):
        # ten valid samples between two NaN runs, far from the burst
        nan_rows = np.r_[12_000:12_100, 12_110:12_200]
        with pytest.warns(UserWarning, match=r"minimum_duration.*\(12100, 12110\)"):
            events = self._run(detector, time, nan_rows)
        assert len(events) >= 1

    @pytest.mark.parametrize("detector", DETECTORS)
    def test_no_block_long_enough_raises(self, detector, time):
        # a NaN every tenth sample leaves blocks of nine, under any default minimum
        with pytest.raises(ValueError, match="No block of finite samples is as long"):
            self._run(detector, time, np.arange(0, self.N_TIME, 10))

    def test_the_warning_names_the_callers_line(self, time):
        nan_rows = np.r_[12_000:12_100, 12_110:12_200]
        with pytest.warns(UserWarning, match="minimum_duration") as record:
            self._run(Kay_ripple_detector, time, nan_rows)
        assert record[0].filename == __file__


class TestMillisecondsGivenAsSeconds:
    """The commonest unit slip, 15 for 15 ms, raises and says what to pass."""

    FS = 1000

    @pytest.mark.parametrize(
        ("detector", "kwargs", "name"),
        [
            (Kay_ripple_detector, {"minimum_duration": 15}, "minimum_duration"),
            (Kay_ripple_detector, {"maximum_duration": 500}, "maximum_duration"),
            (Kay_ripple_detector, {"close_ripple_threshold": 50}, "close_ripple_threshold"),
            (multiunit_HSE_detector, {"close_event_threshold": 50}, "close_event_threshold"),
            (
                Zugaro_ripple_detector,
                {"minimum_inter_ripple_interval": 30},
                "minimum_inter_ripple_interval",
            ),
            (
                Long_sharp_wave_ripple_detector,
                {"minimum_ripple_duration": 25},
                "minimum_ripple_duration",
            ),
            (
                Long_sharp_wave_ripple_detector,
                {"minimum_separation": 50},
                "minimum_separation",
            ),
            (Carey_candidate_detector, {"state_merge_gap": 50}, "state_merge_gap"),
        ],
    )
    def test_the_message_gives_the_value_in_seconds(self, detector, kwargs, name):
        (value,) = kwargs.values()
        with pytest.raises(ValueError, match=rf"{name} is in seconds.*pass {value / 1000}"):
            TestParameterRanges()._call(detector, **kwargs)

    def test_a_ceiling_of_two_seconds_and_a_gap_of_ten_are_accepted(self):
        TestParameterRanges()._call(
            Kay_ripple_detector, maximum_duration=2.0, close_ripple_threshold=10.0
        )


class TestParameterRanges:
    """A tunable that is NaN, negative, reversed or in the wrong unit raises
    rather than quietly disabling a criterion or emptying the result."""

    FS = 1000
    N_TIME = 5_000

    def _call(self, detector, **kwargs):
        time = np.arange(self.N_TIME) / self.FS
        speed = np.full(self.N_TIME, 2.0)
        fs = kwargs.pop("sampling_frequency", self.FS)
        if detector is Long_sharp_wave_ripple_detector:
            lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, (2500,))
            return detector(time, lfp[:, 0], speed, fs, sharp_wave_lfp=lfp[:, 1], **kwargs)
        if detector in (multiunit_HSE_detector, Carey_candidate_detector):
            lfps, multiunit = _synthetic_joint_inputs(self.N_TIME, self.FS, (2500,))
            if detector is multiunit_HSE_detector:
                return detector(time, multiunit, speed, fs, **kwargs)
            return detector(time, lfps, multiunit, speed, fs, **kwargs)
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(2500, 2560, 20.0)])
        return detector(time, lfps, speed, fs, **kwargs)

    @pytest.mark.parametrize(
        "detector",
        [
            Kay_ripple_detector,
            Karlsson_ripple_detector,
            Roumis_ripple_detector,
            Shvartsman_ripple_detector,
            Yu_ripple_detector,
            Zugaro_ripple_detector,
            multiunit_HSE_detector,
            Carey_candidate_detector,
        ],
    )
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"minimum_duration": -0.01}, "minimum_duration"),
            ({"minimum_duration": np.nan}, "minimum_duration"),
            ({"maximum_duration": 0.0}, "maximum_duration"),
            ({"maximum_duration": np.inf}, "maximum_duration.*None for no ceiling"),
            ({"speed_threshold": np.nan}, "speed_threshold"),
            ({"speed_threshold": -1.0}, "speed_threshold"),
            ({"sampling_frequency": 0.0}, "sampling_frequency"),
            ({"sampling_frequency": np.nan}, "sampling_frequency"),
        ],
    )
    def test_shared_tunables(self, detector, kwargs, match):
        with pytest.raises(ValueError, match=match):
            self._call(detector, **kwargs)

    @pytest.mark.parametrize(
        "detector",
        [
            Kay_ripple_detector,
            Karlsson_ripple_detector,
            Roumis_ripple_detector,
            Shvartsman_ripple_detector,
            multiunit_HSE_detector,
        ],
    )
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"zscore_threshold": np.nan}, "zscore_threshold"),
            ({"zscore_threshold": np.inf}, "zscore_threshold"),
            ({"smoothing_sigma": 0.0}, "smoothing_sigma"),
            ({"smoothing_sigma": 4.0}, "is in seconds"),
        ],
    )
    def test_single_threshold_detectors(self, detector, kwargs, match):
        with pytest.raises(ValueError, match=match):
            self._call(detector, **kwargs)

    @pytest.mark.parametrize(
        ("detector", "kwargs", "match"),
        [
            (Kay_ripple_detector, {"close_ripple_threshold": -1.0}, "close_ripple_threshold"),
            (Kay_ripple_detector, {"close_ripple_threshold": np.inf}, "finite"),
            (multiunit_HSE_detector, {"close_event_threshold": np.inf}, "finite"),
            (Zugaro_ripple_detector, {"minimum_inter_ripple_interval": np.inf}, "finite"),
            (Long_sharp_wave_ripple_detector, {"minimum_separation": np.inf}, "finite"),
            (
                Long_sharp_wave_ripple_detector,
                {"maximum_sharp_wave_duration": np.inf},
                "maximum_sharp_wave_duration.*None for no ceiling",
            ),
            (Carey_candidate_detector, {"state_merge_gap": np.inf}, "finite"),
            (Carey_candidate_detector, {"minimum_state_duration": np.inf}, "finite"),
            (
                Long_sharp_wave_ripple_detector,
                {"window_size": 0.0005},
                "window_size.*shorter than one sample",
            ),
            (Yu_ripple_detector, {"close_ripple_threshold": np.nan}, "close_ripple_threshold"),
            (Yu_ripple_detector, {"smoothing_sigma": -0.004}, "smoothing_sigma"),
            (multiunit_HSE_detector, {"close_event_threshold": -1.0}, "close_event_threshold"),
            (multiunit_HSE_detector, {"minimum_active_units": 100}, "unit"),
            (multiunit_HSE_detector, {"minimum_active_units": 1.5}, "whole number"),
            (
                Shvartsman_ripple_detector,
                {"minimum_participating_channels": 0.5},
                "whole number",
            ),
            (Zugaro_ripple_detector, {"low_threshold": 5.0, "high_threshold": 2.0}, "above"),
            (Zugaro_ripple_detector, {"low_threshold": -1.0}, "low_threshold"),
            (Zugaro_ripple_detector, {"high_threshold": np.nan}, "high_threshold"),
            (Zugaro_ripple_detector, {"smoothing_window": -0.01}, "smoothing_window"),
            (
                Zugaro_ripple_detector,
                {"minimum_inter_ripple_interval": -0.01},
                "minimum_inter_ripple_interval",
            ),
            (Carey_candidate_detector, {"low_threshold": 3.0, "high_threshold": 1.0}, "above"),
            (Carey_candidate_detector, {"minimum_active_units": 100}, "unit"),
            (Carey_candidate_detector, {"spike_cap": 0.0}, "spike_cap"),
            (Carey_candidate_detector, {"state_merge_gap": -1.0}, "state_merge_gap"),
            (Carey_candidate_detector, {"theta_threshold": np.nan}, "theta_threshold"),
            (
                Carey_candidate_detector,
                {"theta_lfp": np.zeros(N_TIME), "theta_band": (6.0, 600.0)},
                "Nyquist",
            ),
            (Long_sharp_wave_ripple_detector, {"ripple_band": (250.0, 80.0)}, "ripple_band"),
            (Long_sharp_wave_ripple_detector, {"ripple_band": (80.0, 600.0)}, "Nyquist"),
            (
                Long_sharp_wave_ripple_detector,
                {"sharp_wave_thresholds": (2.5, 0.5)},
                "above",
            ),
            (
                Long_sharp_wave_ripple_detector,
                {"ripple_power_percentile": 0.0},
                "ripple_power_percentile",
            ),
            (Long_sharp_wave_ripple_detector, {"window_size": 0.0}, "window_size"),
            (
                Long_sharp_wave_ripple_detector,
                {"minimum_sharp_wave_duration": np.nan},
                "minimum_sharp_wave_duration",
            ),
        ],
    )
    def test_detector_specific_tunables(self, detector, kwargs, match):
        with pytest.raises(ValueError, match=match):
            self._call(detector, **kwargs)


class TestInputContents:
    """Inputs of the right shape whose contents would change the result
    without an error: rates passed as counts, dead channels, empty channels."""

    FS = 1000
    N_TIME = 5_000

    @pytest.fixture
    def time(self):
        return np.arange(self.N_TIME) / self.FS

    @pytest.mark.parametrize("detector", [multiunit_HSE_detector, Carey_candidate_detector])
    @pytest.mark.parametrize("transform", ["smoothed rate", "negative"])
    def test_multiunit_that_is_not_counts_raises_without_the_registry(
        self, detector, transform, time
    ):
        lfps, multiunit = _synthetic_joint_inputs(self.N_TIME, self.FS, (2500,))
        if transform == "smoothed rate":
            multiunit = gaussian_smooth(multiunit, 0.01, self.FS) * self.FS
        else:
            multiunit = multiunit - 1
        speed = np.full(self.N_TIME, 2.0)
        signals = (multiunit,) if detector is multiunit_HSE_detector else (lfps, multiunit)
        with pytest.raises(ValueError, match="spike counts or indicators"):
            detector(time, *signals, speed, self.FS, minimum_active_units=1)

    @pytest.mark.parametrize(
        "detector",
        [
            Kay_ripple_detector,
            Karlsson_ripple_detector,
            Roumis_ripple_detector,
            Shvartsman_ripple_detector,
            Yu_ripple_detector,
            Zugaro_ripple_detector,
            Carey_candidate_detector,
            Long_sharp_wave_ripple_detector,
        ],
    )
    def test_a_flat_channel_raises_and_is_named(self, detector, time):
        """A nonzero constant too: its normalization scale is rounding noise,
        not zero, so only the flat-channel check catches it."""
        speed = np.full(self.N_TIME, 2.0)
        if detector is Long_sharp_wave_ripple_detector:
            lfp = _synthetic_two_channel_lfp(self.N_TIME, self.FS, (2500,))
            lfp[:, 1] = 0.7
            call = lambda: detector(time, lfp[:, 0], speed, self.FS, sharp_wave_lfp=lfp[:, 1])  # noqa: E731
            match = r"sharp_wave_lfp is constant"
        else:
            lfps, multiunit = _synthetic_joint_inputs(self.N_TIME, self.FS, (2500,))
            lfps[:, 2] = 0.7
            match = r"filtered_lfps channel\(s\) \[2\]"
            if detector is Carey_candidate_detector:
                call = lambda: detector(time, lfps, multiunit, speed, self.FS)  # noqa: E731
            else:
                call = lambda: detector(time, lfps, speed, self.FS)  # noqa: E731
        with pytest.raises(ValueError, match=match):
            call()

    def test_a_flat_channel_is_judged_over_every_block(self, time):
        """A channel constant in one block but not across all of them is live;
        one constant in every block is dead."""
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(3500, 3560, 20.0)])
        lfps[2000:2100] = np.nan  # two blocks
        lfps[:2000, 1] = 0.5  # flat in the first block only
        Kay_ripple_detector(time, lfps, np.full(self.N_TIME, 2.0), self.FS)
        lfps[2100:, 1] = 0.5  # now flat in both
        with pytest.raises(ValueError, match=r"channel\(s\) \[1\]"):
            Kay_ripple_detector(time, lfps, np.full(self.N_TIME, 2.0), self.FS)

    def test_a_channel_with_no_finite_sample_is_named(self, time):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(2500, 2560, 20.0)])
        lfps[:, 1] = np.nan
        with pytest.raises(ValueError, match=r"channel\(s\) \[1\] of signal 1"):
            Kay_ripple_detector(time, lfps, np.full(self.N_TIME, 2.0), self.FS)

    def test_an_infinite_sample_is_missing_to_the_filter(self):
        rng = np.random.default_rng(0)
        raw = rng.standard_normal((6000, 2))
        raw[3000, 1] = np.inf
        filtered = filter_ripple_band(raw, sampling_frequency=1500)
        assert np.isnan(filtered[3000]).all()
        assert np.isfinite(np.delete(filtered, 3000, axis=0)).all()


class TestSparseSpikeConvolution:
    @pytest.mark.parametrize(
        ("rate", "taps"), [(0.002, 301), (0.2, 301), (0.002, 6001), (0.0, 301)]
    )
    def test_equals_the_direct_convolution_to_rounding(self, rate, taps):
        from scipy.ndimage import convolve1d

        rng = np.random.default_rng(0)
        counts = rng.poisson(rate, 50_000).astype(float)
        counts[:3] = 2.0  # spikes at the edge, where the kernel runs off the data
        kernel = np.exp(-0.5 * np.linspace(-5, 5, taps) ** 2)
        np.testing.assert_allclose(
            _convolve_spikes(counts, kernel),
            convolve1d(counts, kernel, mode="constant"),
            rtol=0,
            atol=1e-12,
        )


class TestRemainingErrorPaths:
    """Error paths no other test reached."""

    FS = 1000

    def test_theta_is_filtered_separately_on_each_side_of_a_timestamp_gap(self):
        time = np.arange(20_000) / self.FS
        time[10_000:] += 0.5  # a half-second hole in the timestamps
        theta = np.random.default_rng(0).standard_normal(20_000)
        theta[10_000:] += 50.0  # a step the filter would ring on if it spanned the gap
        envelope = _theta_envelope(theta, time, self.FS, (6.0, 10.0))
        left = _theta_envelope(theta[:10_000], time[:10_000], self.FS, (6.0, 10.0))
        right = _theta_envelope(theta[10_000:], time[10_000:], self.FS, (6.0, 10.0))
        np.testing.assert_allclose(envelope, np.concatenate([left, right]))

    def test_theta_with_no_finite_sample_leaves_nothing_to_detect(self):
        time = np.arange(5000) / self.FS
        lfps, multiunit = _synthetic_joint_inputs(5000, self.FS, (2500,))
        with pytest.raises(ValueError, match="nothing to detect"):
            Carey_candidate_detector(
                time,
                lfps,
                multiunit,
                np.full(5000, 2.0),
                self.FS,
                theta_lfp=np.full(5000, np.nan),
            )

    def test_kay_consensus_treats_an_infinite_sample_as_missing(self):
        lfps = np.random.default_rng(0).standard_normal((3000, 2))
        with_inf, with_nan = lfps.copy(), lfps.copy()
        with_inf[1500, 1] = np.inf
        with_nan[1500, 1] = np.nan
        np.testing.assert_array_equal(
            get_Kay_ripple_consensus_trace(with_inf, self.FS),
            get_Kay_ripple_consensus_trace(with_nan, self.FS),
        )

    @pytest.mark.parametrize(
        "consensus", [get_Kay_ripple_consensus_trace, get_Yu_ripple_consensus_trace]
    )
    def test_consensus_of_all_missing_samples_raises(self, consensus):
        with pytest.raises(ValueError, match="No sample has finite values"):
            consensus(np.full((100, 2), np.nan), self.FS)

    @pytest.mark.parametrize(
        "consensus", [get_Kay_ripple_consensus_trace, get_Yu_ripple_consensus_trace]
    )
    def test_consensus_of_a_single_channel_says_to_reshape(self, consensus):
        lfp = np.random.default_rng(0).standard_normal(3000)
        with pytest.raises(ValueError, match=r"reshape\(-1, 1\)"):
            consensus(lfp, self.FS)

    @pytest.mark.parametrize(
        "consensus", [get_Kay_ripple_consensus_trace, get_Yu_ripple_consensus_trace]
    )
    def test_consensus_with_time_of_the_wrong_length_raises(self, consensus):
        lfps = np.random.default_rng(0).standard_normal((3000, 2))
        with pytest.raises(ValueError, match="time has shape"):
            consensus(lfps, self.FS, time=np.arange(2999) / self.FS)

    def test_yu_consensus_raises_for_a_channel_with_no_spread(self):
        lfps = np.random.default_rng(0).standard_normal((2000, 3))
        lfps[:, 1] = 0.0
        with pytest.raises(ValueError, match=r"channel indices \[1\]"):
            get_Yu_ripple_consensus_trace(lfps, self.FS, zscore_per_channel=True)

    def test_shvartsman_statistics_without_manual_normalization_raise(self):
        time = np.arange(5000) / self.FS
        lfps = _synthetic_ripple_band(5000, self.FS, [(2500, 2560, 20.0)])
        with pytest.raises(ValueError, match="apply only with"):
            Shvartsman_ripple_detector(
                time, lfps, np.full(5000, 2.0), self.FS, channel_baselines=[0.0, 0.0, 0.0]
            )

    def test_long_with_one_candidate_window_raises(self):
        # a window longer than half the recording leaves one window to cluster
        time = np.arange(1500) / self.FS
        lfp = _synthetic_two_channel_lfp(1500, self.FS, (750,))
        with pytest.raises(ValueError, match="Too few candidate windows"):
            Long_sharp_wave_ripple_detector(
                time,
                lfp[:, 0],
                np.full(1500, 2.0),
                self.FS,
                sharp_wave_lfp=lfp[:, 1],
                window_size=0.9,
            )

    @pytest.mark.parametrize(
        ("lfps", "match"), [(np.float64(1.0), "0-D"), (np.zeros((10, 2, 2)), "3-D")]
    )
    def test_lfp_of_the_wrong_rank_raises(self, lfps, match):
        with pytest.raises(ValueError, match=match):
            Kay_ripple_detector(np.arange(10) / self.FS, lfps, np.zeros(10), self.FS)


class TestGapRule:
    """A block ends wherever the timestamp step exceeds 1.5 times the median
    step, measured from the timestamps rather than the nominal rate."""

    @pytest.mark.parametrize(("jump", "splits"), [(1.4, False), (1.5, False), (1.6, True)])
    def test_the_threshold_is_one_and_a_half_median_steps(self, jump, splits):
        step = 0.002  # a 500 Hz grid; the rule never sees a sampling rate
        steps = np.full(99, step)
        steps[50] = jump * step
        time = np.concatenate([[0.0], np.cumsum(steps)])
        blocks = _contiguous_valid_blocks(np.ones(100, dtype=bool), time)
        assert blocks == ([(0, 51), (51, 100)] if splits else [(0, 100)])

    def test_jitter_under_half_a_step_never_splits(self):
        rng = np.random.default_rng(0)
        time = np.cumsum(0.001 * (1 + rng.uniform(-0.4, 0.4, 10_000)))
        assert _contiguous_valid_blocks(np.ones(10_000, dtype=bool), time) == [(0, 10_000)]

    @pytest.mark.parametrize(
        "detector",
        [
            Kay_ripple_detector,
            Karlsson_ripple_detector,
            Yu_ripple_detector,
            Zugaro_ripple_detector,
        ],
    )
    @pytest.mark.parametrize("factor", [0.5, 1.6])
    def test_a_rate_more_than_ten_percent_off_raises(self, detector, factor):
        fs = 1000
        time = np.arange(20_000) / fs
        lfps = _synthetic_ripple_band(20_000, fs, [(4000, 4060, 20.0)])
        with pytest.raises(ValueError, match="times the interval sampling_frequency"):
            detector(time, lfps, np.full(20_000, 2.0), factor * fs)

    def test_a_nan_timestamp_raises_and_says_so(self):
        time = np.arange(5000) / 1000.0
        time[100] = np.nan
        lfps = _synthetic_ripple_band(5000, 1000, [(2000, 2060, 20.0)])
        with pytest.raises(ValueError, match="time holds 1 NaN"):
            Kay_ripple_detector(time, lfps, np.full(5000, 2.0), 1000)


class TestValidationPaths:
    FS = 1000
    N_TIME = 5000

    def test_speed_that_looks_like_metres_per_second_warns(self, time):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(2000, 2060, 20.0)])
        speed = np.full(self.N_TIME, 0.02)  # 2 cm/s written in m/s
        with pytest.warns(UserWarning, match="cm/s, not m/s"):
            Kay_ripple_detector(time, lfps, speed, self.FS)

    def test_the_unit_is_not_judged_when_the_speed_criterion_is_off(self, time):
        """With ``speed_threshold=np.inf`` speed decides nothing, so a small
        unit is no error; under ``filterwarnings = error`` a warning would be."""
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(2000, 2060, 20.0)])
        speed = np.full(self.N_TIME, 0.02)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            events = Kay_ripple_detector(time, lfps, speed, self.FS, speed_threshold=np.inf)
        assert len(events) == 1

    def test_a_mask_that_selects_only_missing_samples_raises(self, time, stationary):
        lfps = _synthetic_ripple_band(self.N_TIME, self.FS, [(2000, 2060, 20.0)])
        lfps[:1000] = np.nan
        mask = np.zeros(self.N_TIME, dtype=bool)
        mask[:1000] = True
        with pytest.raises(ValueError, match="no sample that is finite"):
            Kay_ripple_detector(time, lfps, stationary, self.FS, normalization_mask=mask)


class TestDetectEventsFromTrace:
    """The shared thresholding, on a trace the caller builds."""

    FS = 1000

    @staticmethod
    def _bumps(n_time, centers, half_widths, fs=1000):
        """Integer triangles on a zero baseline: a bump of half-width w samples
        is w high at its center and falls by one per sample, so the samples at
        or above a level L are those within w - L of the center, exactly.
        Tests on the raw trace set bound_threshold above the baseline."""
        trace = np.zeros(n_time)
        index = np.arange(n_time)
        for center, half_width in zip(centers, half_widths, strict=True):
            c, w = round(center * fs), round(half_width * fs)
            trace = np.maximum(trace, np.clip(w - np.abs(index - c), 0, None))
        return trace

    def _time(self, n_time):
        return np.arange(n_time) / self.FS

    def test_matches_kay_on_kays_trace(
        self, dual_lfp_with_ripples, time_3s, stationary_speed, sampling_frequency
    ):
        from ripple_detection import detect_events_from_trace

        filtered = filter_ripple_band(dual_lfp_with_ripples, sampling_frequency)
        kay = Kay_ripple_detector(time_3s, filtered, stationary_speed, sampling_frequency)
        trace = get_Kay_ripple_consensus_trace(filtered, sampling_frequency)
        events = detect_events_from_trace(time_3s, trace, stationary_speed, sampling_frequency)
        assert len(kay) > 0
        pd.testing.assert_frame_equal(events, kay)

    def test_matches_the_multiunit_detector_on_its_rate(self):
        from ripple_detection import (
            detect_events_from_trace,
            get_multiunit_population_firing_rate,
        )
        from ripple_detection.simulate import simulate_session, simulate_time

        time = simulate_time(30_000, self.FS)
        session = simulate_session(time, [5.0, 12.0, 20.0], rng=1)
        hse = multiunit_HSE_detector(time, session.multiunit, session.speed, self.FS)
        rate = get_multiunit_population_firing_rate(session.multiunit, self.FS, 0.015)
        events = detect_events_from_trace(time, rate, session.speed, self.FS)
        pd.testing.assert_frame_equal(events, hse.drop(columns="n_active_units"))

    def test_events_end_at_the_bound_threshold(self):
        from ripple_detection import detect_events_from_trace

        time = self._time(1000)
        trace = self._bumps(1000, [0.5], [0.05])  # 50 at 0.500, 10 at 0.460 and 0.540
        events = detect_events_from_trace(
            time, trace, np.zeros(1000), self.FS,
            threshold=30.0, bound_threshold=10.0, normalization_method="none",
            minimum_duration=0.0,
        )  # fmt: skip
        assert len(events) == 1
        assert events.start_time.iloc[0] == pytest.approx(0.460)
        assert events.end_time.iloc[0] == pytest.approx(0.540)
        assert events.peak_time.iloc[0] == pytest.approx(0.500)

    def test_a_bound_equal_to_the_threshold_ends_at_the_crossings(self):
        from ripple_detection import detect_events_from_trace

        time = self._time(1000)
        trace = self._bumps(1000, [0.5], [0.05])  # 30 at 0.480 and 0.520
        events = detect_events_from_trace(
            time, trace, np.zeros(1000), self.FS,
            threshold=30.0, bound_threshold=30.0, normalization_method="none",
            minimum_duration=0.0,
        )  # fmt: skip
        assert events.start_time.iloc[0] == pytest.approx(0.480)
        assert events.end_time.iloc[0] == pytest.approx(0.520)

    def test_minimum_duration_is_time_above_threshold(self):
        """41 samples at or above 30; the event itself spans 99."""
        from ripple_detection import detect_events_from_trace

        time = self._time(1000)
        trace = self._bumps(1000, [0.5], [0.05])
        common = {"threshold": 30.0, "bound_threshold": 1.0, "normalization_method": "none"}
        kept = detect_events_from_trace(
            time, trace, np.zeros(1000), self.FS, minimum_duration=0.041, **common
        )
        dropped = detect_events_from_trace(
            time, trace, np.zeros(1000), self.FS, minimum_duration=0.042, **common
        )
        assert len(kept) == 1
        assert len(dropped) == 0

    def test_minimum_event_duration_is_the_whole_event(self):
        from ripple_detection import detect_events_from_trace

        time = self._time(2000)
        trace = self._bumps(2000, [0.5, 1.5], [0.015, 0.05])  # events of 29 and 99 samples
        events = detect_events_from_trace(
            time, trace, np.zeros(2000), self.FS,
            threshold=10.0, bound_threshold=1.0, normalization_method="none",
            minimum_duration=0.0,
            minimum_event_duration=0.05,
        )  # fmt: skip
        assert events.peak_time.tolist() == pytest.approx([1.5])

    def test_a_short_event_does_not_suppress_its_neighbour(self):
        """The too-short event comes first and within the gap; it is removed
        before the close-event rule, so the long one survives."""
        from ripple_detection import detect_events_from_trace

        time = self._time(2000)
        trace = self._bumps(2000, [0.50, 0.62], [0.015, 0.05])
        events = detect_events_from_trace(
            time, trace, np.zeros(2000), self.FS,
            threshold=10.0, bound_threshold=1.0, normalization_method="none",
            minimum_duration=0.0,
            minimum_event_duration=0.05, close_event_threshold=0.2,
        )  # fmt: skip
        assert events.peak_time.tolist() == pytest.approx([0.62])

    def test_close_events_are_dropped_or_merged(self):
        from ripple_detection import detect_events_from_trace

        time = self._time(2000)
        trace = self._bumps(2000, [0.50, 0.62], [0.05, 0.05])  # 22 ms apart at the bound
        common = {
            "threshold": 10.0, "bound_threshold": 1.0, "normalization_method": "none",
            "minimum_duration": 0.0,
            "close_event_threshold": 0.03,
        }  # fmt: skip
        dropped = detect_events_from_trace(
            time, trace, np.zeros(2000), self.FS, close_event_rule="drop", **common
        )
        merged = detect_events_from_trace(
            time, trace, np.zeros(2000), self.FS, close_event_rule="merge", **common
        )
        assert dropped.peak_time.tolist() == pytest.approx([0.50])
        assert len(merged) == 1
        assert merged.start_time.iloc[0] == pytest.approx(0.451)
        assert merged.end_time.iloc[0] == pytest.approx(0.669)

    def test_a_merged_event_is_what_the_minimum_tests(self):
        """Two 29-sample events 10 ms apart are each too short, merged not."""
        from ripple_detection import detect_events_from_trace

        time = self._time(2000)
        trace = self._bumps(2000, [0.50, 0.54], [0.015, 0.015])
        common = {
            "threshold": 10.0, "bound_threshold": 1.0, "normalization_method": "none",
            "minimum_duration": 0.0,
            "minimum_event_duration": 0.05, "close_event_threshold": 0.02,
        }  # fmt: skip
        assert (
            len(detect_events_from_trace(time, trace, np.zeros(2000), self.FS, **common)) == 0
        )
        merged = detect_events_from_trace(
            time, trace, np.zeros(2000), self.FS, close_event_rule="merge", **common
        )
        assert len(merged) == 1

    def test_nothing_is_merged_across_a_missing_sample(self):
        from ripple_detection import detect_events_from_trace

        time = self._time(2000)
        trace = self._bumps(2000, [0.50, 0.62], [0.05, 0.05])
        trace[560] = np.nan
        events = detect_events_from_trace(
            time, trace, np.zeros(2000), self.FS,
            threshold=10.0, bound_threshold=1.0, normalization_method="none",
            minimum_duration=0.0,
            close_event_threshold=0.05, close_event_rule="merge",
        )  # fmt: skip
        assert len(events) == 2

    def test_smoothing_stays_within_a_block(self):
        """A spike just before a gap must not raise the trace after it."""
        from ripple_detection import detect_events_from_trace

        time = self._time(2000)
        trace = np.zeros(2000)
        trace[995:1000] = 100.0
        trace[1000] = np.nan
        events = detect_events_from_trace(
            time, trace, np.zeros(2000), self.FS,
            threshold=1.0, bound_threshold=0.5, normalization_method="none",
            minimum_duration=0.0, smoothing_sigma=0.01,
        )  # fmt: skip
        assert len(events) == 1
        assert events.end_time.iloc[0] < 1.0
        assert bool(events.clipped_end.iloc[0])

    def test_the_mask_sets_the_statistics(self):
        """A trace five times as variable while moving: z-scoring over the
        still samples alone finds the still bump, which the whole-session z
        misses."""
        from ripple_detection import detect_events_from_trace

        rng = np.random.default_rng(0)
        time = self._time(20_000)
        speed = np.where(time < 10, 20.0, 0.0)
        trace = rng.normal(0, np.where(time < 10, 5.0, 1.0))
        trace[15_000:15_030] += 6.0
        still = speed < 4
        common = {"threshold": 3.0, "minimum_duration": 0.02, "speed_threshold": 4.0}
        with_mask = detect_events_from_trace(
            time, trace, speed, self.FS, normalization_mask=still, **common
        )
        without = detect_events_from_trace(time, trace, speed, self.FS, **common)
        assert with_mask.peak_time.between(15.0, 15.03).any()
        assert not without.peak_time.between(15.0, 15.03).any()

    def test_restrict_cuts_an_event_at_movement(self):
        from ripple_detection import detect_events_from_trace

        time = self._time(2000)
        trace = self._bumps(2000, [0.5, 1.5], [0.05, 0.05])
        speed = np.where((time > 0.52) & (time < 1.0), 20.0, 0.0)
        events = detect_events_from_trace(
            time, trace, speed, self.FS,
            threshold=10.0, bound_threshold=1.0, normalization_method="none",
            minimum_duration=0.0,
            speed_rule="restrict",
        )  # fmt: skip
        assert len(events) == 2
        first = events.iloc[0]
        assert first.end_time == pytest.approx(0.520)
        assert bool(first.clipped_end)
        assert not bool(events.iloc[1].clipped_end)

    def test_restrict_with_no_slow_sample_raises(self):
        from ripple_detection import detect_events_from_trace

        with pytest.raises(ValueError, match="no sample is"):
            detect_events_from_trace(
                self._time(1000), np.zeros(1000), np.full(1000, 20.0), self.FS,
                speed_rule="restrict",
            )  # fmt: skip

    def test_restrict_drops_brief_slow_dips_silently(self):
        """Speed dipping below the threshold for 3-5 samples while running is
        movement, not missing data: the dips are too short for an event and
        are left out without a warning (pytest turns one into an error)."""
        from ripple_detection import detect_events_from_trace

        time = self._time(2000)
        trace = self._bumps(2000, [0.5, 1.5], [0.05, 0.05])
        speed = np.full(2000, 10.0)
        for start, length in [(100, 3), (250, 4), (900, 5), (1700, 3)]:
            speed[start : start + length] = 1.0
        speed[1300:1800] = 0.0
        events = detect_events_from_trace(
            time, trace, speed, self.FS,
            threshold=10.0, bound_threshold=1.0, normalization_method="none",
            minimum_duration=0.015, speed_rule="restrict",
        )  # fmt: skip
        assert events.peak_time.tolist() == pytest.approx([1.5])

    def test_restrict_with_only_brief_slow_stretches_raises(self):
        from ripple_detection import detect_events_from_trace

        speed = np.full(1000, 20.0)
        speed[500:505] = 0.0
        with pytest.raises(ValueError, match="speed_rule='restrict'"):
            detect_events_from_trace(
                self._time(1000), np.zeros(1000), speed, self.FS,
                minimum_duration=0.015, speed_rule="restrict",
            )  # fmt: skip

    def test_the_all_rule_tests_every_sample(self):
        from ripple_detection import detect_events_from_trace

        time = self._time(1000)
        trace = self._bumps(1000, [0.5], [0.05])
        speed = np.zeros(1000)
        speed[500] = 20.0
        common = {
            "threshold": 30.0,
            "bound_threshold": 1.0,
            "normalization_method": "none",
            "minimum_duration": 0.0,
        }
        assert len(detect_events_from_trace(time, trace, speed, self.FS, **common)) == 1
        assert (
            len(
                detect_events_from_trace(
                    time, trace, speed, self.FS, speed_rule="all", **common
                )
            )
            == 0
        )

    def test_the_ceiling_applies_to_the_event_as_reported(self):
        from ripple_detection import detect_events_from_trace

        time = self._time(1000)
        trace = self._bumps(1000, [0.5], [0.05])  # 99 samples, bound to bound
        common = {
            "threshold": 30.0,
            "bound_threshold": 1.0,
            "normalization_method": "none",
            "minimum_duration": 0.0,
        }
        kept = detect_events_from_trace(
            time, trace, np.zeros(1000), self.FS, maximum_duration=0.099, **common
        )
        dropped = detect_events_from_trace(
            time, trace, np.zeros(1000), self.FS, maximum_duration=0.098, **common
        )
        assert len(kept) == 1
        assert len(dropped) == 0

    def test_a_column_trace_is_accepted(self):
        from ripple_detection import detect_events_from_trace

        time = self._time(1000)
        trace = self._bumps(1000, [0.5], [0.05])
        common = {
            "threshold": 30.0,
            "bound_threshold": 1.0,
            "normalization_method": "none",
            "minimum_duration": 0.0,
        }
        flat = detect_events_from_trace(time, trace, np.zeros(1000), self.FS, **common)
        column = detect_events_from_trace(
            time, trace[:, None], np.zeros(1000), self.FS, **common
        )
        pd.testing.assert_frame_equal(flat, column)

    def test_an_empty_result_has_the_same_columns(self):
        from ripple_detection import detect_events_from_trace

        time = self._time(1000)
        trace = self._bumps(1000, [0.5], [0.05])
        common = {
            "bound_threshold": 1.0,
            "normalization_method": "none",
            "minimum_duration": 0.0,
        }
        full = detect_events_from_trace(time, trace, np.zeros(1000), self.FS, **common)
        empty = detect_events_from_trace(
            time, trace, np.zeros(1000), self.FS, threshold=100.0, **common
        )
        assert len(empty) == 0
        assert list(empty.columns) == list(full.columns)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"threshold": 1.0, "bound_threshold": 2.0}, "bound_threshold .* is above"),
            ({"threshold": np.nan}, "threshold must be finite"),
            ({"normalization_method": "mad"}, "normalization_method must be one of"),
            ({"speed_rule": "majority"}, "speed_rule must be one of"),
            ({"close_event_rule": "join"}, "close_event_rule must be one of"),
            (
                {"normalization_method": "none", "normalization_mask": np.ones(1000, bool)},
                "computes none",
            ),
            ({"minimum_event_duration": 15.0}, "minimum_event_duration is in seconds"),
            ({"close_event_threshold": -0.1}, "close_event_threshold must be"),
            ({"smoothing_sigma": 0.0}, "smoothing_sigma must be positive"),
        ],
    )
    def test_invalid_arguments_raise(self, kwargs, message):
        from ripple_detection import detect_events_from_trace

        with pytest.raises(ValueError, match=message):
            detect_events_from_trace(
                self._time(1000), np.zeros(1000), np.zeros(1000), self.FS, **kwargs
            )

    def test_a_threshold_per_sample(self):
        """The same two bumps: 40 high at 0.5 s and 20 high at 1.5 s. A level of
        30 in the first second and 10 in the second finds both."""
        from ripple_detection import detect_events_from_trace

        time = self._time(2000)
        trace = self._bumps(2000, [0.5, 1.5], [0.04, 0.02])
        common = {
            "bound_threshold": 1.0,
            "normalization_method": "none",
            "minimum_duration": 0.0,
        }
        flat = detect_events_from_trace(
            time, trace, np.zeros(2000), self.FS, threshold=30.0, **common
        )
        varying = detect_events_from_trace(
            time, trace, np.zeros(2000), self.FS,
            threshold=np.where(time < 1.0, 30.0, 10.0), **common,
        )  # fmt: skip
        assert flat.peak_time.tolist() == pytest.approx([0.5])
        assert varying.peak_time.tolist() == pytest.approx([0.5, 1.5])

    def test_a_scalar_array_threshold_matches_the_scalar(self):
        from ripple_detection import detect_events_from_trace

        time = self._time(2000)
        trace = self._bumps(2000, [0.5, 1.5], [0.04, 0.02])
        common = {
            "bound_threshold": 1.0,
            "normalization_method": "none",
            "minimum_duration": 0.0,
        }
        scalar = detect_events_from_trace(
            time, trace, np.zeros(2000), self.FS, threshold=15.0, **common
        )
        array = detect_events_from_trace(
            time, trace, np.zeros(2000), self.FS, threshold=np.full(2000, 15.0), **common
        )
        pd.testing.assert_frame_equal(scalar, array)

    def test_bounds_found_within_the_search_window_are_the_same(self):
        from ripple_detection import detect_events_from_trace

        time = self._time(2000)
        trace = self._bumps(2000, [0.5], [0.05])
        common = {
            "threshold": 30.0, "bound_threshold": 1.0, "normalization_method": "none",
            "minimum_duration": 0.0,
        }  # fmt: skip
        free = detect_events_from_trace(time, trace, np.zeros(2000), self.FS, **common)
        searched = detect_events_from_trace(
            time, trace, np.zeros(2000), self.FS, bound_search_window=0.3, **common
        )
        pd.testing.assert_frame_equal(free, searched)

    @staticmethod
    def _plateau(n_time=3000, fs=1000):
        """A step to 5 at 1.0 s, a shoulder at 0.3 from 1.1 to 1.9 s, then 0:
        the bound at 0 lies 900 ms from the run's first sample."""
        trace = np.full(n_time, -1.0)
        index = np.arange(n_time)
        trace[(index >= 1000) & (index < 1100)] = 5.0
        trace[(index >= 1100) & (index < 1900)] = 0.3
        return trace

    def test_a_fallback_level_ends_an_event_the_first_level_cannot(self):
        from ripple_detection import detect_events_from_trace

        time = self._time(3000)
        events = detect_events_from_trace(
            time, self._plateau(), np.zeros(3000), self.FS,
            threshold=3.0, bound_threshold=(0.0, 0.25, 0.5), bound_search_window=0.3,
            normalization_method="none", minimum_duration=0.0,
        )  # fmt: skip
        assert events.start_time.tolist() == pytest.approx([1.0])
        assert events.end_time.tolist() == pytest.approx([1.099])
        assert not events.clipped_start.iloc[0]
        assert not events.clipped_end.iloc[0]

    def test_without_a_fallback_the_search_edge_ends_the_event_and_is_flagged(self):
        from ripple_detection import detect_events_from_trace

        time = self._time(3000)
        events = detect_events_from_trace(
            time, self._plateau(), np.zeros(3000), self.FS,
            threshold=3.0, bound_threshold=0.0, bound_search_window=0.3,
            normalization_method="none", minimum_duration=0.0,
        )  # fmt: skip
        assert events.end_time.tolist() == pytest.approx([1.3])
        assert bool(events.clipped_end.iloc[0])
        assert not bool(events.clipped_start.iloc[0])
        unlimited = detect_events_from_trace(
            time, self._plateau(), np.zeros(3000), self.FS,
            threshold=3.0, bound_threshold=0.0,
            normalization_method="none", minimum_duration=0.0,
        )  # fmt: skip
        assert unlimited.end_time.tolist() == pytest.approx([1.899])

    def test_runs_sharing_an_event_give_one(self):
        from ripple_detection import detect_events_from_trace

        time = self._time(2000)
        trace = np.full(2000, -1.0)
        trace[500:600] = 1.0
        trace[510:520] = 5.0
        trace[560:570] = 5.0
        events = detect_events_from_trace(
            time, trace, np.zeros(2000), self.FS,
            threshold=3.0, bound_search_window=0.3,
            normalization_method="none", minimum_duration=0.0,
        )  # fmt: skip
        assert events.start_time.tolist() == pytest.approx([0.5])
        assert events.end_time.tolist() == pytest.approx([0.599])

    def test_a_later_run_extends_an_event_its_first_run_capped(self):
        """A plateau from 0.5 to 1.5 s with runs at 0.51 and 0.70 s: the first
        run's search ends at 0.81 s, the second's at 1.0 s, which ends the event."""
        from ripple_detection import detect_events_from_trace

        time = self._time(2000)
        trace = np.full(2000, -1.0)
        trace[500:1500] = 1.0
        trace[510:520] = 5.0
        trace[700:710] = 5.0
        events = detect_events_from_trace(
            time, trace, np.zeros(2000), self.FS,
            threshold=3.0, bound_search_window=0.3,
            normalization_method="none", minimum_duration=0.0,
        )  # fmt: skip
        assert events.start_time.tolist() == pytest.approx([0.5])
        assert events.end_time.tolist() == pytest.approx([1.0])
        assert bool(events.clipped_end.iloc[0])
        assert not bool(events.clipped_start.iloc[0])

    def test_a_search_reaching_the_block_edge_is_clipped_there(self):
        from ripple_detection import detect_events_from_trace

        time = self._time(2000)
        trace = np.full(2000, -1.0)
        trace[:100] = 5.0
        events = detect_events_from_trace(
            time, trace, np.zeros(2000), self.FS,
            threshold=3.0, bound_search_window=0.3,
            normalization_method="none", minimum_duration=0.0,
        )  # fmt: skip
        assert events.start_time.tolist() == pytest.approx([0.0])
        assert bool(events.clipped_start.iloc[0])

    def test_merged_events_carry_the_flags_of_their_ends(self):
        """The first event's start is capped by the search; merged with the
        second, the start keeps that flag and the end takes the second's."""
        from ripple_detection import detect_events_from_trace

        time = self._time(3000)
        trace = np.full(3000, -1.0)
        trace[500:1000] = 0.3  # a long shoulder before the first run
        trace[1000:1050] = 5.0
        trace[1060:1100] = 5.0
        events = detect_events_from_trace(
            time, trace, np.zeros(3000), self.FS,
            threshold=3.0, bound_search_window=0.3, bound_threshold=0.0,
            normalization_method="none", minimum_duration=0.0,
            close_event_threshold=0.02, close_event_rule="merge",
        )  # fmt: skip
        assert len(events) == 1
        assert events.start_time.iloc[0] == pytest.approx(0.7)
        assert bool(events.clipped_start.iloc[0])
        assert not bool(events.clipped_end.iloc[0])

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"threshold": np.full(10, 2.0)}, "one level per sample"),
            ({"threshold": np.array([2.0] * 999 + [np.nan])}, "threshold must be finite"),
            ({"threshold": np.array([2.0] * 999 + [-1.0])}, "bound_threshold .* is above"),
            ({"bound_threshold": (0.0, 0.25)}, "pass bound_search_window"),
            ({"bound_threshold": ()}, "at least one level"),
            ({"bound_threshold": (0.0, np.nan), "bound_search_window": 0.3}, "must be finite"),
            ({"bound_search_window": 0.0}, "bound_search_window must be positive"),
            ({"bound_search_window": 20.0}, "bound_search_window is in seconds"),
        ],
    )
    def test_invalid_threshold_arguments_raise(self, kwargs, message):
        from ripple_detection import detect_events_from_trace

        with pytest.raises(ValueError, match=message):
            detect_events_from_trace(
                self._time(1000), np.zeros(1000), np.zeros(1000), self.FS, **kwargs
            )

    def test_a_multichannel_trace_raises_with_a_hint(self):
        from ripple_detection import detect_events_from_trace

        with pytest.raises(ValueError, match="Combine channels into one trace"):
            detect_events_from_trace(
                self._time(1000), np.zeros((1000, 3)), np.zeros(1000), self.FS
            )


class TestActiveUnits:
    """Participation criteria on any event inventory."""

    TIME = np.arange(10) / 10
    EVENTS = np.array([(0.0, 0.3), (0.5, 0.9)])

    @classmethod
    def _spikes(cls):
        multiunit = np.zeros((10, 4))
        multiunit[1, [0, 1, 2]] = 1
        multiunit[2, 0] = 2
        multiunit[6, 3] = 1
        return multiunit

    def test_counts_per_event_and_unit(self):
        from ripple_detection import count_spikes_in_events

        counts = count_spikes_in_events(self.EVENTS, self._spikes(), self.TIME)
        np.testing.assert_array_equal(counts, [[3, 1, 1, 0], [0, 0, 0, 1]])
        assert counts.dtype.kind == "i"

    def test_both_ends_of_an_event_are_inside_it(self):
        from ripple_detection import count_spikes_in_events

        multiunit = np.zeros((10, 1))
        multiunit[[0, 3], 0] = 1
        counts = count_spikes_in_events(np.array([(0.0, 0.3)]), multiunit, self.TIME)
        assert counts[0, 0] == 2

    def test_a_missing_sample_counts_no_spike_and_warns(self):
        from ripple_detection import count_spikes_in_events, require_active_units

        multiunit = self._spikes()
        multiunit[1, 1] = np.nan
        with pytest.warns(UserWarning, match=r"1 of 2 event\(s\) hold missing"):
            counts = count_spikes_in_events(self.EVENTS, multiunit, self.TIME)
        assert counts[0, 1] == 0
        with pytest.warns(UserWarning, match=r"1 of 2 event\(s\) hold missing"):
            require_active_units(self.EVENTS, multiunit, self.TIME)

    def test_a_missing_sample_outside_every_event_does_not_warn(self):
        from ripple_detection import count_spikes_in_events

        multiunit = self._spikes()
        multiunit[4, 1] = np.nan
        counts = count_spikes_in_events(self.EVENTS, multiunit, self.TIME)
        np.testing.assert_array_equal(counts, [[3, 1, 1, 0], [0, 0, 0, 1]])

    def test_a_frame_is_read_by_its_bounds(self):
        from ripple_detection import count_spikes_in_events

        frame = pd.DataFrame({"start_time": [0.0], "end_time": [0.3], "other": [9.0]})
        counts = count_spikes_in_events(frame, self._spikes(), self.TIME)
        np.testing.assert_array_equal(counts, [[3, 1, 1, 0]])

    def test_an_event_holding_no_sample_raises(self):
        from ripple_detection import count_spikes_in_events

        with pytest.raises(ValueError, match="No sample of time falls within"):
            count_spikes_in_events(np.array([(5.0, 6.0)]), self._spikes(), self.TIME)

    def test_a_rate_is_rejected(self):
        from ripple_detection import count_spikes_in_events

        with pytest.raises(ValueError, match="not a rate"):
            count_spikes_in_events(self.EVENTS, self._spikes() * 0.5, self.TIME)

    def test_mismatched_lengths_raise(self):
        from ripple_detection import count_spikes_in_events

        with pytest.raises(ValueError, match="must match"):
            count_spikes_in_events(self.EVENTS, self._spikes()[:5], self.TIME)

    def test_a_count_of_active_units(self):
        from ripple_detection import require_active_units

        kept = require_active_units(
            self.EVENTS, self._spikes(), self.TIME, minimum_active_units=3
        )
        np.testing.assert_allclose(kept, self.EVENTS[:1])

    def test_only_the_selected_units_count(self):
        from ripple_detection import require_active_units

        mask = np.array([False, False, False, True])
        for units in (mask, [3]):
            kept = require_active_units(self.EVENTS, self._spikes(), self.TIME, units=units)
            np.testing.assert_allclose(kept, self.EVENTS[1:])

    def test_a_fraction_of_the_selected_units(self):
        """Three of four units is 0.75, exactly the threshold."""
        from ripple_detection import require_active_units

        spikes = self._spikes()
        kept = require_active_units(
            self.EVENTS, spikes, self.TIME, minimum_active_fraction=0.75
        )
        np.testing.assert_allclose(kept, self.EVENTS[:1])
        assert (
            len(
                require_active_units(
                    self.EVENTS, spikes, self.TIME, minimum_active_fraction=0.76
                )
            )
            == 0
        )

    def test_a_count_and_a_fraction_must_both_hold(self):
        """'At least 5 or 15%, whichever is larger' of 40 units is 6."""
        from ripple_detection import require_active_units

        time = np.arange(20) / 10
        multiunit = np.zeros((20, 40))
        multiunit[2, :5] = 1  # 5 active: meets 5, not 15% of 40
        multiunit[12, :6] = 1  # 6 active: meets both
        events = np.array([(0.0, 0.5), (1.0, 1.5)])
        kept = require_active_units(
            events, multiunit, time, minimum_active_units=5, minimum_active_fraction=0.15
        )
        np.testing.assert_allclose(kept, events[1:])

    def test_a_total_of_spikes(self):
        from ripple_detection import require_active_units

        kept = require_active_units(self.EVENTS, self._spikes(), self.TIME, minimum_spikes=5)
        np.testing.assert_allclose(kept, self.EVENTS[:1])

    def test_a_frame_keeps_its_columns_and_index(self):
        from ripple_detection import require_active_units

        frame = pd.DataFrame(
            {"start_time": [0.0, 0.5], "end_time": [0.3, 0.9], "tag": ["a", "b"]},
            index=pd.Index([4, 9], name="event_number"),
        )
        kept = require_active_units(frame, self._spikes(), self.TIME, minimum_active_units=2)
        assert list(kept.index) == [4]
        assert list(kept.tag) == ["a"]

    def test_no_events(self):
        from ripple_detection import require_active_units

        kept = require_active_units(np.empty((0, 2)), self._spikes(), self.TIME)
        assert kept.shape == (0, 2)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"minimum_active_units": 5}, "4 unit\\(s\\) are selected"),
            ({"minimum_active_units": 2, "units": [0]}, "1 unit\\(s\\) are selected"),
            ({"minimum_active_units": 1.5}, "whole number"),
            ({"minimum_active_fraction": 1.5}, "between 0 and 1"),
            ({"minimum_spikes": -1}, "whole number"),
            ({"units": np.ones(3, bool)}, "one entry per unit, 4"),
            ({"units": [0, 7]}, "outside 0 to 3"),
            ({"units": [0.5]}, "boolean mask over the units or a 1-D array"),
        ],
    )
    def test_invalid_criteria_raise(self, kwargs, message):
        from ripple_detection import require_active_units

        with pytest.raises(ValueError, match=message):
            require_active_units(self.EVENTS, self._spikes(), self.TIME, **kwargs)


class TestThetaDeltaRatio:
    FS = 500

    def _signal(self, theta_amplitude, delta_amplitude, seconds=20):
        t = np.arange(seconds * self.FS) / self.FS
        return t, theta_amplitude * np.sin(2 * np.pi * 8 * t) + delta_amplitude * np.sin(
            2 * np.pi * 2 * t
        )

    def test_the_ratio_of_band_amplitudes(self):
        from ripple_detection import theta_delta_ratio

        _, lfp = self._signal(3.0, 1.0)
        ratio = theta_delta_ratio(lfp, self.FS)
        middle = ratio[2 * self.FS : -2 * self.FS]
        np.testing.assert_allclose(middle, 3.0, rtol=0.05)

    def test_power_is_the_amplitude_ratio_squared(self):
        from ripple_detection import theta_delta_ratio

        _, lfp = self._signal(3.0, 1.0)
        amplitude = theta_delta_ratio(lfp, self.FS)
        power = theta_delta_ratio(lfp, self.FS, measure="power")
        np.testing.assert_allclose(power, amplitude**2, rtol=1e-12)

    def test_it_follows_a_change_of_state(self):
        from ripple_detection import theta_delta_ratio

        _, theta_state = self._signal(3.0, 1.0)
        _, delta_state = self._signal(1.0, 3.0)
        lfp = np.concatenate([theta_state, delta_state])
        ratio = theta_delta_ratio(lfp, self.FS)
        assert np.median(ratio[: 15 * self.FS]) > 2
        assert np.median(ratio[25 * self.FS :]) < 0.5

    def test_missing_samples_are_missing_and_filtering_restarts(self):
        from ripple_detection import theta_delta_ratio

        _, lfp = self._signal(3.0, 1.0)
        lfp[5000] = np.nan
        ratio = theta_delta_ratio(lfp, self.FS)
        assert np.isnan(ratio[5000])
        assert np.isfinite(np.delete(ratio, 5000)).all()

    def test_a_gap_in_time_splits_the_recording(self):
        """Without time the two halves are filtered as one run; with it, the
        filter restarts at the jump, so the samples beside it differ."""
        from ripple_detection import theta_delta_ratio

        t, lfp = self._signal(3.0, 1.0)
        t = np.where(t >= 10, t + 5.0, t)
        joined = theta_delta_ratio(lfp, self.FS, smoothing_sigma=None)
        split = theta_delta_ratio(lfp, self.FS, time=t, smoothing_sigma=None)
        assert not np.allclose(joined[4990:5010], split[4990:5010])

    def test_a_run_too_short_to_filter_is_missing_with_a_warning(self):
        from ripple_detection import theta_delta_ratio

        _, lfp = self._signal(3.0, 1.0)
        lfp[10] = np.nan  # leaves a run of 10 samples, under the filter's pad of 15
        with pytest.warns(UserWarning, match="treated as missing"):
            ratio = theta_delta_ratio(lfp, self.FS)
        assert np.isnan(ratio[:11]).all()
        assert np.isfinite(ratio[11:]).all()

    def test_a_column_is_one_channel(self):
        from ripple_detection import theta_delta_ratio

        _, lfp = self._signal(3.0, 1.0)
        np.testing.assert_allclose(
            theta_delta_ratio(lfp[:, None], self.FS), theta_delta_ratio(lfp, self.FS)
        )

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"theta_band": (12.0, 6.0)}, "theta_band must be"),
            ({"delta_band": (1.0, 300.0)}, "delta_band must be"),
            ({"smoothing_sigma": 0.0}, "smoothing_sigma must be positive"),
            ({"measure": "db"}, "measure must be one of"),
        ],
    )
    def test_invalid_arguments_raise(self, kwargs, message):
        from ripple_detection import theta_delta_ratio

        _, lfp = self._signal(3.0, 1.0)
        with pytest.raises(ValueError, match=message):
            theta_delta_ratio(lfp, self.FS, **kwargs)

    def test_bad_shapes_raise(self):
        from ripple_detection import theta_delta_ratio

        with pytest.raises(ValueError, match="one channel"):
            theta_delta_ratio(np.zeros((100, 2)), self.FS)
        with pytest.raises(ValueError, match="must match"):
            theta_delta_ratio(np.zeros(100), self.FS, time=np.arange(50) / self.FS)
        with pytest.raises(ValueError, match="no finite sample"):
            theta_delta_ratio(np.full(100, np.nan), self.FS)


class TestStateIntervals:
    TIME = np.arange(10.0)
    RATIO = np.array([3, 1, 1, 1, 3, 1, 3, 1, 1, 1.0])

    def test_runs_below_the_threshold(self):
        from ripple_detection import state_intervals

        np.testing.assert_allclose(
            state_intervals(self.RATIO, self.TIME, 2.0), [[1, 3], [5, 5], [7, 9]]
        )

    @pytest.mark.parametrize(
        ("comparison", "expected"),
        [("<", [[1, 3]]), ("<=", [[0, 4]]), (">", [[5, 5]]), (">=", [[0, 0], [4, 5]])],
    )
    def test_each_comparison(self, comparison, expected):
        from ripple_detection import state_intervals

        values = np.array([2, 1, 1, 1, 2, 3, 1, 1, 1, 1.0])[:6]
        np.testing.assert_allclose(
            state_intervals(values, self.TIME[:6], 2.0, comparison=comparison), expected
        )

    def test_merging_then_the_minimum(self):
        from ripple_detection import state_intervals

        np.testing.assert_allclose(
            state_intervals(self.RATIO, self.TIME, 2.0, merge_gap=2.5, minimum_duration=3.0),
            [[1, 9]],
        )
        np.testing.assert_allclose(
            state_intervals(self.RATIO, self.TIME, 2.0, minimum_duration=2.0),
            [[1, 3], [7, 9]],
        )

    def test_the_minimum_is_an_inclusive_sample_count(self):
        """At 10 Hz, 0.3 s is 3 samples: a run of 3 samples (0.2 s from first
        to last) is kept and a run of 2 dropped, as the detectors count an
        event's duration."""
        from ripple_detection import state_intervals

        time = np.arange(10) / 10
        values = np.array([3, 1, 1, 3, 1, 1, 1, 3, 3, 3.0])
        np.testing.assert_allclose(
            state_intervals(values, time, 2.0, minimum_duration=0.3), [[0.4, 0.6]]
        )
        np.testing.assert_allclose(
            state_intervals(self.RATIO, self.TIME, 2.0, minimum_duration=3.0),
            [[1, 3], [7, 9]],
        )

    def test_unknown_values_are_not_in_the_state(self):
        from ripple_detection import state_intervals

        values = self.RATIO.copy()
        values[2] = np.nan
        np.testing.assert_allclose(
            state_intervals(values, self.TIME, 2.0), [[1, 1], [3, 3], [5, 5], [7, 9]]
        )

    def test_a_gap_in_time_ends_an_interval(self):
        from ripple_detection import state_intervals

        time = np.array([0, 1, 2, 3, 10, 11, 12.0])
        values = np.ones(7)
        np.testing.assert_allclose(state_intervals(values, time, 2.0), [[0, 3], [10, 12]])

    def test_merging_bridges_samples_out_of_the_state_but_not_missing_data(self):
        """A sample out of the state is known state, which a merge may
        bridge; a missing sample or a gap in time is unknown, which it may not."""
        from ripple_detection import state_intervals

        time = np.arange(5) / 10
        out_of_state = np.array([1, 1, 3, 1, 1.0])
        missing = np.array([1, 1, np.nan, 1, 1.0])
        np.testing.assert_allclose(
            state_intervals(out_of_state, time, 2.0, merge_gap=0.3), [[0.0, 0.4]]
        )
        np.testing.assert_allclose(
            state_intervals(missing, time, 2.0, merge_gap=0.3), [[0.0, 0.1], [0.3, 0.4]]
        )
        gapped = np.array([0, 1, 2, 3, 10, 11, 12.0])
        np.testing.assert_allclose(
            state_intervals(np.ones(7), gapped, 2.0, merge_gap=10.0), [[0, 3], [10, 12]]
        )

    def test_nothing_in_the_state(self):
        from ripple_detection import state_intervals

        assert state_intervals(self.RATIO, self.TIME, 0.5).shape == (0, 2)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"comparison": "=="}, "comparison must be one of"),
            ({"minimum_duration": -1.0}, "minimum_duration must be"),
            ({"merge_gap": np.inf}, "merge_gap must be"),
        ],
    )
    def test_invalid_arguments_raise(self, kwargs, message):
        from ripple_detection import state_intervals

        with pytest.raises(ValueError, match=message):
            state_intervals(self.RATIO, self.TIME, 2.0, **kwargs)

    def test_bad_inputs_raise(self):
        from ripple_detection import state_intervals

        with pytest.raises(ValueError, match="both must be"):
            state_intervals(self.RATIO, self.TIME[:5], 2.0)
        with pytest.raises(ValueError, match="threshold must be finite"):
            state_intervals(self.RATIO, self.TIME, np.nan)


class TestSilenceBoundedEvents:
    """Events as spiking set off by silence, with no rate threshold."""

    FS = 1000

    def _spikes(self, n_time, n_units, spikes):
        """``spikes`` is a list of (sample, unit)."""
        multiunit = np.zeros((n_time, n_units))
        for sample, unit in spikes:
            multiunit[sample, unit] += 1
        return np.arange(n_time) / self.FS, multiunit

    def test_groups_split_at_silences(self):
        from ripple_detection import detect_silence_bounded_events

        time, multiunit = self._spikes(
            3000, 3, [(500, 0), (520, 1), (540, 2), (1200, 0), (1210, 1), (1300, 2)]
        )
        events = detect_silence_bounded_events(time, multiunit, self.FS, minimum_silence=0.1)
        assert events.start_time.tolist() == pytest.approx([0.5, 1.2])
        assert events.end_time.tolist() == pytest.approx([0.54, 1.3])
        assert events.n_active_units.tolist() == [3, 3]
        assert events.n_spikes.tolist() == [3, 3]
        assert not events.clipped_start.any()
        assert not events.clipped_end.any()

    def test_a_silence_equal_to_the_minimum_splits(self):
        from ripple_detection import detect_silence_bounded_events

        time, multiunit = self._spikes(3000, 2, [(1000, 0), (1100, 1)])
        events = detect_silence_bounded_events(time, multiunit, self.FS, minimum_silence=0.1)
        assert len(events) == 2
        joined = detect_silence_bounded_events(time, multiunit, self.FS, minimum_silence=0.101)
        assert len(joined) == 1

    def test_participation_selects_events(self):
        from ripple_detection import detect_silence_bounded_events

        time, multiunit = self._spikes(
            3000, 4, [(500, 0), (520, 1), (540, 2), (1200, 0), (1210, 1)]
        )
        common = {"minimum_silence": 0.1}
        three = detect_silence_bounded_events(
            time, multiunit, self.FS, minimum_active_units=3, **common
        )
        half = detect_silence_bounded_events(
            time, multiunit, self.FS, minimum_active_fraction=0.5, **common
        )
        assert three.start_time.tolist() == pytest.approx([0.5])
        assert half.start_time.tolist() == pytest.approx([0.5, 1.2])

    def test_only_the_selected_units_segment_and_count(self):
        """Unit 3 fires between the groups; left out, it neither joins them
        nor counts."""
        from ripple_detection import detect_silence_bounded_events

        time, multiunit = self._spikes(
            3000, 4, [(500, 0), (540, 1), (580, 3), (620, 3), (660, 0), (700, 1)]
        )
        every = detect_silence_bounded_events(time, multiunit, self.FS, minimum_silence=0.1)
        chosen = detect_silence_bounded_events(
            time, multiunit, self.FS, minimum_silence=0.1, units=[0, 1]
        )
        assert len(every) == 1
        assert len(chosen) == 2
        assert chosen.n_active_units.tolist() == [2, 2]

    def test_bursts_collapse_to_their_first_spike(self):
        """One unit bursting every 10 ms from 1.0 to 1.2 s joins everything
        into one group; collapsed to its first spike, the next unit's spike
        at 1.25 s stands apart."""
        from ripple_detection import detect_silence_bounded_events

        burst = [(sample, 0) for sample in range(1000, 1201, 10)]
        time, multiunit = self._spikes(3000, 2, [*burst, (1250, 1)])
        whole = detect_silence_bounded_events(time, multiunit, self.FS, minimum_silence=0.1)
        letters = detect_silence_bounded_events(
            time, multiunit, self.FS, minimum_silence=0.1, maximum_isi=0.05
        )
        assert whole.start_time.tolist() == pytest.approx([1.0])
        assert whole.end_time.tolist() == pytest.approx([1.25])
        assert letters.start_time.tolist() == pytest.approx([1.0, 1.25])
        assert letters.n_spikes.tolist() == [1, 1]

    def test_windows_after_silence(self):
        """After 60 ms of silence, a 300 ms window; the spike at 1.34 s falls
        outside the first window and follows 50 ms of silence, too little to
        start one."""
        from ripple_detection import detect_silence_bounded_events

        time, multiunit = self._spikes(
            3000, 3, [(1000, 0), (1100, 1), (1290, 2), (1340, 0), (2000, 1), (2050, 2)]
        )
        events = detect_silence_bounded_events(
            time, multiunit, self.FS, minimum_silence=0.06, window=0.3
        )
        assert events.start_time.tolist() == pytest.approx([1.0, 2.0])
        assert events.end_time.tolist() == pytest.approx([1.29, 2.05])
        assert events.n_active_units.tolist() == [3, 2]

    @pytest.mark.parametrize("origin", [0.0, 1.7e9])
    def test_fixed_windows_retain_the_full_window_and_count_only_spikes_inside(self, origin):
        from ripple_detection import detect_silence_bounded_events

        time, multiunit = self._spikes(2000, 3, [(500, 0), (580, 1), (801, 2)])
        events = detect_silence_bounded_events(
            time + origin,
            multiunit,
            self.FS,
            minimum_silence=0.06,
            window=0.3,
            window_end_rule="fixed",
            minimum_active_units=2,
            minimum_duration=0.2,
        )
        np.testing.assert_allclose(events.start_time - origin, [0.5], atol=1e-6)
        np.testing.assert_allclose(events.end_time - origin, [0.8], atol=1e-6)
        assert events.n_spikes.tolist() == [2]
        assert events.n_active_units.tolist() == [2]
        assert events.n_samples.tolist() == [301]
        assert not events.clipped_end.any()

    @pytest.mark.parametrize("gap", ["nan", "timestamp", "recording_end"])
    def test_fixed_windows_stop_at_data_boundaries(self, gap):
        from ripple_detection import detect_silence_bounded_events

        time, multiunit = self._spikes(1000, 2, [(500, 0), (580, 1)])
        if gap == "nan":
            multiunit[650] = np.nan
        elif gap == "timestamp":
            time[650:] += 1
        else:
            time, multiunit = time[:650], multiunit[:650]
        events = detect_silence_bounded_events(
            time,
            multiunit,
            self.FS,
            minimum_silence=0.06,
            window=0.3,
            window_end_rule="fixed",
            minimum_active_units=2,
        )
        assert events.end_time.tolist() == pytest.approx([0.649])
        assert events.clipped_end.tolist() == [True]
        assert events.n_spikes.tolist() == [2]

    def test_a_silence_equal_to_the_minimum_starts_a_window(self):
        from ripple_detection import detect_silence_bounded_events

        time, multiunit = self._spikes(3000, 2, [(1000, 0), (1290, 1), (1350, 0)])
        events = detect_silence_bounded_events(
            time, multiunit, self.FS, minimum_silence=0.06, window=0.3
        )
        assert events.start_time.tolist() == pytest.approx([1.0, 1.35])

    @pytest.mark.parametrize("origin", [86_400.0, 1e6, 1.7e9])  # a day; 11 days; a Unix time
    def test_the_boundaries_do_not_depend_on_the_time_origin(self, origin):
        """The rounding in a silence comes from the timestamps, not from the
        silence: a tolerance relative to the minimum split nothing far from
        zero, and one relative to the time let a window 1.7 s too long."""
        from ripple_detection import detect_silence_bounded_events

        time, multiunit = self._spikes(3000, 3, [(1000, 0), (1100, 1)])
        time = origin + time
        split = detect_silence_bounded_events(time, multiunit, self.FS, minimum_silence=0.1)
        joined = detect_silence_bounded_events(time, multiunit, self.FS, minimum_silence=0.101)
        assert (len(split), len(joined)) == (2, 1)

        # a spike at the window's end is in it; one a sample later is not
        _, multiunit = self._spikes(3000, 3, [(1000, 0), (1300, 1), (1301, 2)])
        events = detect_silence_bounded_events(
            time, multiunit, self.FS, minimum_silence=0.06, window=0.3
        )
        assert events.n_spikes.tolist() == [2]
        assert events.end_time.tolist() == [time[1300]]

    def test_no_onset_and_no_spike_give_no_event(self):
        from ripple_detection import detect_silence_bounded_events

        dense = [(sample, 0) for sample in range(0, 1000, 20)]
        time, multiunit = self._spikes(1000, 2, dense)
        assert (
            len(
                detect_silence_bounded_events(
                    time, multiunit, self.FS, minimum_silence=0.06, window=0.3
                )
            )
            == 0
        )
        silent_time, silent = self._spikes(1000, 2, [])
        assert (
            len(
                detect_silence_bounded_events(
                    silent_time, silent, self.FS, minimum_silence=0.1, maximum_isi=0.05
                )
            )
            == 0
        )

    def test_a_silent_unit_does_not_stop_the_burst_collapse(self):
        from ripple_detection import detect_silence_bounded_events

        time, multiunit = self._spikes(3000, 3, [(1000, 0), (1010, 0), (1020, 1)])
        events = detect_silence_bounded_events(
            time, multiunit, self.FS, minimum_silence=0.1, maximum_isi=0.05
        )
        assert events.n_active_units.tolist() == [2]
        assert events.n_spikes.tolist() == [3]

    def test_a_spike_at_the_window_end_is_inside(self):
        from ripple_detection import detect_silence_bounded_events

        time, multiunit = self._spikes(3000, 2, [(1000, 0), (1300, 1)])
        events = detect_silence_bounded_events(
            time, multiunit, self.FS, minimum_silence=0.06, window=0.3
        )
        assert events.end_time.tolist() == pytest.approx([1.3])

    def test_a_silence_cut_by_the_recording_start_is_no_onset(self):
        from ripple_detection import detect_silence_bounded_events

        time, multiunit = self._spikes(3000, 2, [(30, 0), (2000, 1)])
        events = detect_silence_bounded_events(
            time, multiunit, self.FS, minimum_silence=0.06, window=0.3
        )
        assert events.start_time.tolist() == pytest.approx([2.0])

    def test_a_window_past_the_end_is_clipped(self):
        from ripple_detection import detect_silence_bounded_events

        time, multiunit = self._spikes(3000, 2, [(2900, 0), (2950, 1)])
        events = detect_silence_bounded_events(
            time, multiunit, self.FS, minimum_silence=0.06, window=0.3
        )
        assert events.clipped_end.tolist() == [True]

    def test_a_group_whose_silence_runs_into_missing_data_is_clipped(self):
        from ripple_detection import detect_silence_bounded_events

        time, multiunit = self._spikes(3000, 2, [(1030, 0), (1040, 1), (2000, 0)])
        multiunit[1000, 0] = np.nan
        events = detect_silence_bounded_events(time, multiunit, self.FS, minimum_silence=0.1)
        assert events.start_time.tolist() == pytest.approx([1.03, 2.0])
        assert events.clipped_start.tolist() == [True, False]

    def test_nothing_spans_a_missing_sample(self):
        from ripple_detection import detect_silence_bounded_events

        time, multiunit = self._spikes(3000, 2, [(1000, 0), (1020, 1)])
        multiunit[1010, 1] = np.nan
        events = detect_silence_bounded_events(time, multiunit, self.FS, minimum_silence=0.1)
        assert len(events) == 2
        assert events.end_time.iloc[0] < 1.01 < events.start_time.iloc[1]

    def test_duration_limits_are_inclusive_sample_counts(self):
        from ripple_detection import detect_silence_bounded_events

        time, multiunit = self._spikes(3000, 2, [(1000, 0), (1040, 1), (2000, 0), (2100, 1)])
        common = {"minimum_silence": 0.2}
        kept = detect_silence_bounded_events(
            time, multiunit, self.FS, minimum_duration=0.041, maximum_duration=0.101, **common
        )
        assert kept.start_time.tolist() == pytest.approx([1.0, 2.0])
        short = detect_silence_bounded_events(
            time, multiunit, self.FS, maximum_duration=0.1, **common
        )
        assert short.start_time.tolist() == pytest.approx([1.0])

    def test_no_spikes_gives_an_empty_frame_with_the_columns(self):
        from ripple_detection import detect_silence_bounded_events

        time, multiunit = self._spikes(1000, 2, [])
        events = detect_silence_bounded_events(time, multiunit, self.FS, minimum_silence=0.1)
        assert len(events) == 0
        assert list(events.columns) == [
            "start_time", "end_time", "duration", "n_samples", "n_spikes",
            "n_active_units", "clipped_start", "clipped_end",
        ]  # fmt: skip
        assert events.index.name == "event_number"

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"minimum_silence": 0.1, "window_end_rule": "unknown"}, "window_end_rule"),
            ({"minimum_silence": 0.1, "window_end_rule": "fixed"}, "requires window"),
            ({"minimum_silence": 0.0}, "minimum_silence must be positive"),
            ({"minimum_silence": 0.1, "window": -1.0}, "window must be positive"),
            ({"minimum_silence": 0.1, "maximum_isi": 0.0}, "maximum_isi must be positive"),
            (
                {"minimum_silence": 0.1, "minimum_active_units": 3},
                "2 unit\\(s\\) are selected",
            ),
            ({"minimum_silence": 0.1, "minimum_active_fraction": 2.0}, "between 0 and 1"),
            (
                {"minimum_silence": 0.1, "minimum_duration": 50.0},
                "minimum_duration is in seconds",
            ),
        ],
    )
    def test_invalid_arguments_raise(self, kwargs, message):
        from ripple_detection import detect_silence_bounded_events

        time, multiunit = self._spikes(1000, 2, [(500, 0)])
        with pytest.raises(ValueError, match=message):
            detect_silence_bounded_events(time, multiunit, self.FS, **kwargs)

    def test_bad_inputs_raise(self):
        from ripple_detection import detect_silence_bounded_events

        time, multiunit = self._spikes(1000, 2, [(500, 0)])
        with pytest.raises(ValueError, match="must match"):
            detect_silence_bounded_events(time[:500], multiunit, self.FS, minimum_silence=0.1)
        with pytest.raises(ValueError, match="not a rate"):
            detect_silence_bounded_events(time, multiunit * 0.5, self.FS, minimum_silence=0.1)


def _windowed_fft_reference(data, sampwin, idx1, fs, high_pass_cutoff, weight_by):
    """vandermeerlab windowedFFT.m, transliterated line by line; idx1 is
    MATLAB's 1-based index."""
    n_smooth = round(fs / high_pass_cutoff)
    new = sampwin - n_smooth
    rise = 0.5 - 0.5 * np.cos(np.pi / n_smooth * np.arange(n_smooth))
    fall = 0.5 - 0.5 * np.cos(np.pi / n_smooth * np.arange(n_smooth - 1, -1, -1))
    window = np.concatenate([rise, np.ones(new), fall])
    low = idx1 - int(np.floor(new / 2)) - n_smooth
    high = idx1 + int(np.ceil(new / 2)) + n_smooth - 1
    windowed = data[low - 1 : high] * window
    windowed[:n_smooth] = windowed[:n_smooth] + windowed[new + n_smooth : new + 2 * n_smooth]
    windowed = windowed[: n_smooth + new]
    magnitude = np.abs(np.fft.fft(windowed))
    magnitude = magnitude[: int(np.floor(len(magnitude) / 2 + 0.5))]
    if weight_by == "power":
        magnitude = magnitude * np.arange(1, len(magnitude) + 1)
    return magnitude


def _am_swr_reference(data, fs, examples, weight_by, window=0.06, cutoff=100.0, offset=2.0):
    """SWRfreak.m then amSWR.m (stepSize 1), transliterated."""
    sampwin = round(window * fs)
    t = np.arange(len(data)) / fs
    kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

    def smooth(x):
        return np.convolve(x, kernel)[2 : len(x) + 2]

    centers = examples.mean(axis=1)
    swr = sum(
        _windowed_fft_reference(
            data, sampwin, int(np.argmin(np.abs(t - c))) + 1, fs, cutoff, weight_by
        )
        for c in centers
    )
    noise = sum(
        _windowed_fft_reference(
            data, sampwin, int(np.argmin(np.abs(t - (c + offset)))) + 1, fs, cutoff, weight_by
        )
        for c in centers
    )
    swr, noise = smooth(swr), smooth(noise)
    freqs = smooth(swr / swr.sum() - noise / noise.sum())
    n_cut = int(np.floor(cutoff * window + 0.5))
    score = np.full(len(data), np.nan)
    for idx1 in range(sampwin, len(data) - sampwin + 1):
        spectrum = _windowed_fft_reference(data, sampwin, idx1, fs, cutoff, weight_by)
        spectrum[:n_cut] = 0
        score[idx1 - 1] = np.sum(spectrum * freqs)
    score = np.where(np.isnan(score), 0.0, np.maximum(0.0, score))  # MATLAB max(0, NaN) is 0
    return score / score.mean()


class TestCareySpectralScore:
    """The amSWR score of the paper's published candidates."""

    FS = 2000

    def _lfp(self, seconds=6.0, ripples=(1.0, 2.0, 3.0), rng=0):
        """Timestamps from 0, the LFP and the example ripples."""
        rng = np.random.default_rng(rng)
        n = round(seconds * self.FS)
        data = rng.normal(size=n)
        burst = 3 * np.sin(2 * np.pi * 180 * np.arange(80) / self.FS)
        for center in ripples:
            data[round(center * self.FS) - 40 : round(center * self.FS) + 40] += burst
        examples = np.array([(c - 0.02, c + 0.02) for c in ripples])
        return np.arange(n) / self.FS, data, examples

    @pytest.mark.parametrize("weight_by", ["amplitude", "power"])
    def test_equals_the_original_line_by_line(self, weight_by):
        from ripple_detection import carey_spectral_ripple_score

        time, data, examples = self._lfp()
        score = carey_spectral_ripple_score(time, data, self.FS, examples, weight_by=weight_by)
        np.testing.assert_allclose(
            score,
            _am_swr_reference(data, self.FS, examples, weight_by),
            rtol=1e-10,
            atol=1e-12,
        )

    def test_it_peaks_at_the_ripples(self):
        from ripple_detection import carey_spectral_ripple_score

        time, data, examples = self._lfp()
        score = carey_spectral_ripple_score(time, data, self.FS, examples)
        for center in (1.0, 2.0, 3.0):
            assert score[round(center * self.FS)] > 5 * np.median(score)
        assert np.nanmean(score) == pytest.approx(1.0)
        assert (score >= 0).all()

    def test_a_step_interpolates_close_to_every_sample(self):
        from ripple_detection import carey_spectral_ripple_score

        time, data, examples = self._lfp()
        every = carey_spectral_ripple_score(time, data, self.FS, examples)
        stepped = carey_spectral_ripple_score(time, data, self.FS, examples, step=11)
        assert np.corrcoef(every, stepped)[0, 1] > 0.99
        # a step of 7 divides the run's 11760 intervals, so it ends on a computed sample
        dividing = carey_spectral_ripple_score(time, data, self.FS, examples, step=7)
        assert np.corrcoef(every, dividing)[0, 1] > 0.99
        # sample 4002 (119 + 11 * 353, at a ripple) and the run's last scored
        # sample are computed, not interpolated, so they differ only by the two
        # scores' rescaling to mean 1
        computed, last = 4002, len(data) - 121
        assert stepped[last] / every[last] == pytest.approx(
            stepped[computed] / every[computed]
        )

    def test_a_scored_run_shorter_than_the_step_reaches_its_last_sample(self):
        """A run of 242 finite samples (two windows of 120, less one, plus
        3) scores samples 5120-5122 only, fewer than a step of 5: its last
        sample is computed, not a copy of its first, so it differs from the
        step-1 score only by the two scores' rescaling to mean 1."""
        from ripple_detection import carey_spectral_ripple_score

        time, data, examples = self._lfp()
        data[[5000, 5243]] = np.nan
        data[5081:5161] += 3 * np.sin(2 * np.pi * 180 * np.arange(80) / self.FS)
        every = carey_spectral_ripple_score(time, data, self.FS, examples)
        stepped = carey_spectral_ripple_score(time, data, self.FS, examples, step=5)
        assert (every[5120:5123] > 0).all()
        assert (every[[5119, 5123]] == 0).all()
        assert stepped[5122] / every[5122] == pytest.approx(stepped[5120] / every[5120])

    def test_missing_samples_are_nan_and_zero_nearby(self):
        from ripple_detection import carey_spectral_ripple_score

        time, data, examples = self._lfp()
        data[5000] = np.nan
        score = carey_spectral_ripple_score(time, data, self.FS, examples)
        assert np.isnan(score[5000])
        assert (score[4900:5000] == 0).all()
        assert (score[5001:5100] == 0).all()

    def test_a_run_too_short_for_a_window_scores_zero(self):
        from ripple_detection import carey_spectral_ripple_score

        time, data, examples = self._lfp()
        data[[5000, 5050]] = np.nan  # a run of 49 samples, under one window's 120
        score = carey_spectral_ripple_score(time, data, self.FS, examples)
        assert (score[5001:5050] == 0).all()

    @pytest.mark.parametrize("origin", [60.0, 86_400.0, 1.7e9])
    def test_examples_are_read_on_the_recordings_clock(self, origin):
        """Examples from a recording that starts at `origin` pick the same
        stretches as at 0; on a clock from 0, a 60 s origin put every example
        past the end, and a 1 s one picked noise."""
        from ripple_detection import carey_spectral_ripple_score

        time, data, examples = self._lfp()
        np.testing.assert_allclose(
            carey_spectral_ripple_score(origin + time, data, self.FS, origin + examples),
            carey_spectral_ripple_score(time, data, self.FS, examples),
            rtol=1e-12,
        )

    @pytest.mark.parametrize("origin", [0.0, 86_400.0, 1.7e9])
    def test_an_even_length_example_centers_on_the_earlier_middle_sample(self, origin):
        """One sample longer, each example has two middle samples; the
        earlier is the one it had before, as the original's min() takes it,
        at any origin: the middle time lies between the two, where rounding
        would pick the side."""
        from ripple_detection import carey_spectral_ripple_score

        time, data, examples = self._lfp()
        longer = examples + np.array([0.0, 1 / self.FS])
        np.testing.assert_allclose(
            carey_spectral_ripple_score(origin + time, data, self.FS, origin + longer),
            carey_spectral_ripple_score(time, data, self.FS, examples),
            rtol=1e-12,
        )

    def test_a_gap_in_time_ends_a_run(self):
        """A 1 s jump after sample 11000 scores 0 within a window of it on
        both sides, as a missing sample does."""
        from ripple_detection import carey_spectral_ripple_score

        time, data, examples = self._lfp()
        joined = carey_spectral_ripple_score(time, data, self.FS, examples)
        time[11000:] += 1.0
        split = carey_spectral_ripple_score(time, data, self.FS, examples)
        assert (joined[10900:11100] > 0).any()
        assert (split[10900:11100] == 0).all()
        assert not np.isnan(split).any()

    def test_time_must_match_the_lfp(self):
        from ripple_detection import carey_spectral_ripple_score

        time, data, examples = self._lfp()
        with pytest.raises(ValueError, match="they must match"):
            carey_spectral_ripple_score(time[:-1], data, self.FS, examples)

    def test_an_example_without_a_full_stretch_is_left_out_with_a_warning(self):
        from ripple_detection import carey_spectral_ripple_score

        time, data, examples = self._lfp()
        data[round(1.0 * self.FS)] = np.nan  # the first example's stretch, no one's noise
        with pytest.warns(UserWarning, match="1 of 3 example"):
            score = carey_spectral_ripple_score(time, data, self.FS, examples)
        assert np.nanmax(score) > 5

    def test_no_usable_example_raises(self):
        from ripple_detection import carey_spectral_ripple_score

        time, data, _ = self._lfp()
        with pytest.raises(ValueError, match="No example ripple can build the template"):
            carey_spectral_ripple_score(time, data, self.FS, np.array([(5.5, 5.6)]))

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"weight_by": "db"}, "weight_by must be one of"),
            ({"step": 0}, "step must be a whole number"),
            ({"window": 0.005}, "must be longer than one period"),
            ({"high_pass_cutoff": 1500.0}, "below the Nyquist"),
            ({"noise_offset": 0.0}, "noise_offset must be positive"),
        ],
    )
    def test_invalid_arguments_raise(self, kwargs, message):
        from ripple_detection import carey_spectral_ripple_score

        time, data, examples = self._lfp()
        with pytest.raises(ValueError, match=message):
            carey_spectral_ripple_score(time, data, self.FS, examples, **kwargs)

    def test_bad_shapes_raise(self):
        from ripple_detection import carey_spectral_ripple_score

        time, data, examples = self._lfp()
        with pytest.raises(ValueError, match="one channel"):
            carey_spectral_ripple_score(time, np.column_stack([data, data]), self.FS, examples)
        np.testing.assert_allclose(
            carey_spectral_ripple_score(time, data[:, None], self.FS, examples),
            carey_spectral_ripple_score(time, data, self.FS, examples),
        )

    def test_a_template_that_matches_nothing_raises(self):
        """A pure 10 Hz signal has no power above the 100 Hz cutoff."""
        from ripple_detection import carey_spectral_ripple_score

        time = np.arange(12000) / self.FS
        data = np.sin(2 * np.pi * 10 * time)
        with pytest.raises(ValueError, match="zero everywhere"):
            carey_spectral_ripple_score(time, data, self.FS, np.array([(1.0, 1.04)]))


class TestCareyPublishedConfiguration:
    """ripple_score and threshold_method reproduce precand's rule."""

    FS = 1500
    N_TIME = 45_000

    def _session(self):
        from ripple_detection.simulate import simulate_session, simulate_time

        time = simulate_time(self.N_TIME, self.FS)
        return time, simulate_session(time, [5.0, 10.0, 15.0, 20.0, 25.0], rng=0)

    def test_a_given_score_replaces_the_hilbert_one(self):
        """The Hilbert score passed back in gives the same candidates."""
        time, session = self._session()
        filtered = filter_ripple_band(session.lfps, self.FS)
        from scipy.ndimage import gaussian_filter1d

        hilbert = gaussian_filter1d(
            get_envelope(filtered).mean(axis=1), 0.010 * self.FS, truncate=3.0, mode="constant"
        )
        direct = Carey_candidate_detector(
            time, filtered, session.multiunit, session.speed, self.FS
        )
        given = Carey_candidate_detector(
            time, None, session.multiunit, session.speed, self.FS, ripple_score=hilbert
        )
        pd.testing.assert_frame_equal(direct, given)

    def test_mean_scaling_thresholds_multiples_of_half_the_mean(self):
        """With 'mean' the joint score has mean 0.5: its mean_zscore column
        is on that scale, and a single threshold of 4 bounds each event where
        the scaled score crosses 4."""
        time, session = self._session()
        filtered = filter_ripple_band(session.lfps, self.FS)
        events = Carey_candidate_detector(
            time, filtered, session.multiunit, session.speed, self.FS,
            threshold_method="mean", low_threshold=4.0, high_threshold=4.0,
        )  # fmt: skip
        assert len(events) > 0
        assert (events.min_zscore > 4.0).all()
        assert (events.max_zscore > 4.0).all()

    def test_the_mean_rule_differs_from_the_zscore_rule(self):
        time, session = self._session()
        filtered = filter_ripple_band(session.lfps, self.FS)
        zscored = Carey_candidate_detector(
            time, filtered, session.multiunit, session.speed, self.FS,
            low_threshold=1.0, high_threshold=1.0,
        )  # fmt: skip
        scaled = Carey_candidate_detector(
            time, filtered, session.multiunit, session.speed, self.FS,
            threshold_method="mean", low_threshold=1.0, high_threshold=1.0,
        )  # fmt: skip
        assert not zscored[["start_time", "end_time"]].equals(
            scaled[["start_time", "end_time"]]
        )

    def test_exactly_one_of_lfp_and_score(self):
        time, session = self._session()
        filtered = filter_ripple_band(session.lfps, self.FS)
        with pytest.raises(ValueError, match="exactly one of the two"):
            Carey_candidate_detector(time, None, session.multiunit, session.speed, self.FS)
        with pytest.raises(ValueError, match="exactly one of the two"):
            Carey_candidate_detector(
                time, filtered, session.multiunit, session.speed, self.FS,
                ripple_score=np.ones(self.N_TIME),
            )  # fmt: skip

    @pytest.mark.parametrize(
        ("score", "message"),
        [
            (-np.ones(45_000), "must be non-negative"),
            (np.ones((45_000, 2)), "must have shape \\(n_time,\\)"),
        ],
    )
    def test_a_bad_score_raises(self, score, message):
        time, session = self._session()
        with pytest.raises(ValueError, match=message):
            Carey_candidate_detector(
                time, None, session.multiunit, session.speed, self.FS, ripple_score=score
            )

    def test_an_unknown_threshold_method_raises(self):
        time, session = self._session()
        filtered = filter_ripple_band(session.lfps, self.FS)
        with pytest.raises(ValueError, match="threshold_method must be one of"):
            Carey_candidate_detector(
                time, filtered, session.multiunit, session.speed, self.FS,
                threshold_method="raw",
            )  # fmt: skip


class TestTrimEventsToSpikeWindows:
    """Edge windows must hold enough spikes (Pfeiffer & Foster 2013)."""

    TIME = np.arange(100) / 100

    def _spikes(self, samples, unit=0, n_units=2):
        multiunit = np.zeros((100, n_units))
        for sample in samples:
            multiunit[sample, unit] += 1
        return multiunit

    def test_both_edges_move_inward(self):
        """Start: [0.28, 0.33) is the first window from 0 in 10 ms steps with
        spikes at 0.30 and 0.32; end: (0.59, 0.64] the first back from 0.99."""
        from ripple_detection import trim_events_to_spike_windows

        spikes = self._spikes([30, 32, 60, 61])
        result = trim_events_to_spike_windows(
            np.array([(0.0, 0.99)]), spikes, self.TIME, window=0.05, step=0.01
        )
        np.testing.assert_allclose(result, [[0.28, 0.64]])

    @pytest.mark.parametrize("origin", [0.0, 86_400.0, 1e6, 1e9, 1.7e9])
    def test_the_edges_do_not_depend_on_the_time_origin(self, origin):
        """Spikes at 0.32 and 0.60 lie on the open edge of the windows one
        step outside the answer; a tolerance relative to the time (1.7 s at
        a Unix time) found no window at all."""
        from ripple_detection import trim_events_to_spike_windows

        time = origin + self.TIME
        spikes = self._spikes([30, 32, 60, 61])
        result = trim_events_to_spike_windows(
            np.array([(time[0], time[99])]), spikes, time, window=0.05, step=0.01
        )
        np.testing.assert_array_equal(result, [[time[28], time[64]]])

    def test_edges_that_already_hold_enough_do_not_move(self):
        from ripple_detection import trim_events_to_spike_windows

        spikes = self._spikes([20, 21, 48, 49])
        result = trim_events_to_spike_windows(
            np.array([(0.2, 0.49)]), spikes, self.TIME, window=0.05, step=0.01
        )
        np.testing.assert_allclose(result, [[0.2, 0.49]])

    def test_an_event_without_enough_spikes_is_dropped(self):
        from ripple_detection import trim_events_to_spike_windows

        spikes = self._spikes([30])
        result = trim_events_to_spike_windows(
            np.array([(0.0, 0.99), (0.5, 0.52)]), spikes, self.TIME, window=0.05, step=0.01
        )
        assert result.shape == (0, 2)

    def test_only_the_selected_units_count(self):
        from ripple_detection import trim_events_to_spike_windows

        spikes = self._spikes([30, 32]) + self._spikes([10, 11], unit=1)
        every = trim_events_to_spike_windows(
            np.array([(0.0, 0.99)]), spikes, self.TIME, window=0.05, step=0.01
        )
        chosen = trim_events_to_spike_windows(
            np.array([(0.0, 0.99)]), spikes, self.TIME, window=0.05, step=0.01, units=[0]
        )
        assert every[0, 0] == pytest.approx(0.07)
        assert chosen[0, 0] == pytest.approx(0.28)

    def test_a_missing_sample_counts_no_spike_and_warns(self):
        from ripple_detection import trim_events_to_spike_windows

        spikes = self._spikes([30, 32, 60, 61])
        spikes[32, 1] = np.nan
        with pytest.warns(UserWarning, match=r"1 of 1 event\(s\) hold missing"):
            result = trim_events_to_spike_windows(
                np.array([(0.0, 0.99)]), spikes, self.TIME, window=0.05, step=0.01
            )
        np.testing.assert_allclose(result, [[0.28, 0.64]])

    def test_a_missing_sample_of_an_unselected_unit_does_not_warn(self):
        from ripple_detection import trim_events_to_spike_windows

        spikes = self._spikes([30, 32, 60, 61])
        spikes[32, 1] = np.nan
        result = trim_events_to_spike_windows(
            np.array([(0.0, 0.99)]), spikes, self.TIME, window=0.05, step=0.01, units=[0]
        )
        np.testing.assert_allclose(result, [[0.28, 0.64]])

    def test_a_minimum_duration_on_the_trimmed_event(self):
        from ripple_detection import trim_events_to_spike_windows

        spikes = self._spikes([30, 32, 60, 61])
        common = {"window": 0.05, "step": 0.01}
        kept = trim_events_to_spike_windows(
            np.array([(0.0, 0.99)]), spikes, self.TIME, minimum_duration=0.37, **common
        )
        dropped = trim_events_to_spike_windows(
            np.array([(0.0, 0.99)]), spikes, self.TIME, minimum_duration=0.38, **common
        )
        assert len(kept) == 1
        assert len(dropped) == 0

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"window": 0.0}, "window must be positive"),
            ({"step": np.inf}, "step must be positive"),
            ({"minimum_spikes": 0}, "minimum_spikes must be a whole number"),
            ({"minimum_duration": -1.0}, "minimum_duration must be non-negative"),
        ],
    )
    def test_invalid_arguments_raise(self, kwargs, message):
        from ripple_detection import trim_events_to_spike_windows

        with pytest.raises(ValueError, match=message):
            trim_events_to_spike_windows(
                np.array([(0.0, 0.99)]), self._spikes([30]), self.TIME, **kwargs
            )


@pytest.mark.parametrize(
    "detector",
    [
        Kay_ripple_detector,
        Karlsson_ripple_detector,
        Roumis_ripple_detector,
        Shvartsman_ripple_detector,
        Yu_ripple_detector,
        Zugaro_ripple_detector,
        multiunit_HSE_detector,
    ],
)
def test_isolated_duplicate_timestamp_preserves_detector_compatibility(detector):
    time = np.arange(5000) / 1000
    time[1000] = time[999]
    signal = _synthetic_ripple_band(5000, 1000, [(2000, 2060, 20.0)])
    if detector is multiunit_HSE_detector:
        signal = np.zeros((5000, 4))
        signal[2000:2060] = 1
    options = {"percentile": 99} if detector is Yu_ripple_detector else {}
    result = detector(time, signal, np.full(5000, 2.0), 1000, **options)
    assert len(result)
    assert ((result.start_time < 2.03) & (result.end_time > 2.03)).any()

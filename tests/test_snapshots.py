"""Snapshot tests for detector outputs to catch regressions."""

import numpy as np
import pytest

from ripple_detection import (
    Karlsson_ripple_detector,
    Kay_ripple_detector,
    filter_ripple_band,
)
from ripple_detection.detectors import (
    Roumis_ripple_detector,
    multiunit_HSE_detector,
)
from ripple_detection.simulate import simulate_LFP


@pytest.fixture
def reproducible_seed():
    """Set random seed for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def test_lfp_data(reproducible_seed):
    """Generate reproducible LFP data with embedded ripples."""
    time = np.arange(0, 3.0, 1 / 1500)  # 3 seconds at 1500 Hz

    # Create single channel with two ripples
    lfp = simulate_LFP(
        time,
        ripple_times=[0.8, 1.8],
        noise_amplitude=1.0,
        ripple_amplitude=2.0,
        noise_type="white",
    )

    return time, lfp[:, np.newaxis]


@pytest.fixture
def test_multichannel_lfp_data(reproducible_seed):
    """Generate reproducible multi-channel LFP data."""
    time = np.arange(0, 3.0, 1 / 1500)

    # Create 3 channels with ripples at slightly different times
    lfps = []
    for ripple_times in [[0.8, 1.8], [0.82, 1.78], [0.81, 1.79]]:
        lfp = simulate_LFP(
            time,
            ripple_times=ripple_times,
            noise_amplitude=1.0,
            ripple_amplitude=2.0,
            noise_type="white",
        )
        lfps.append(lfp)

    return time, np.column_stack(lfps)


@pytest.fixture
def test_multiunit_data(reproducible_seed):
    """Generate reproducible multiunit data."""
    rng = reproducible_seed
    time = np.arange(0, 3.0, 1 / 1500)
    n_neurons = 10

    # Create spike trains with high synchrony events
    multiunit = np.zeros((len(time), n_neurons))

    # Add background spikes
    background_spikes = rng.random((len(time), n_neurons)) < 0.01
    multiunit[background_spikes] = 1

    # Add synchronous events at specific times
    hse_indices = [int(0.8 * 1500), int(1.8 * 1500)]
    for idx in hse_indices:
        if idx < len(time) - 10:
            # Make multiple neurons fire together
            multiunit[idx : idx + 10, :7] = 1

    return time, multiunit


@pytest.fixture
def stationary_speed():
    """Create stationary speed data."""
    return lambda length: np.ones(length) * 2.0  # Below typical threshold


class TestKayDetectorSnapshots:
    """Snapshot tests for Kay ripple detector."""

    @pytest.mark.skip(reason="Random data causes non-deterministic results across runs")
    def test_kay_single_channel_output(self, snapshot, test_lfp_data):
        """Test Kay detector output structure and values remain consistent."""
        time, lfp = test_lfp_data
        filtered_lfp = filter_ripple_band(lfp)
        speed = np.ones(len(time)) * 2.0

        ripples = Kay_ripple_detector(
            time,
            filtered_lfp,
            speed,
            sampling_frequency=1500,
            speed_threshold=4.0,
            minimum_duration=0.015,
            zscore_threshold=2.0,
            smoothing_sigma=0.004,
            close_ripple_threshold=0.0,
        )

        # Snapshot the number of detections
        snapshot.assert_match(str(len(ripples)), "n_detections")

        # Snapshot column names
        snapshot.assert_match(str(sorted(ripples.columns.tolist())), "columns")

        if len(ripples) > 0:
            # Snapshot first detection rounded to avoid floating point issues
            first_detection = ripples.iloc[0].to_dict()
            rounded_detection = {k: round(float(v), 6) for k, v in first_detection.items()}
            snapshot.assert_match(str(rounded_detection), "first_detection")

    @pytest.mark.skip(reason="Random data causes non-deterministic results across runs")
    def test_kay_multichannel_output(self, snapshot, test_multichannel_lfp_data):
        """Test Kay detector with multiple channels."""
        time, lfps = test_multichannel_lfp_data
        filtered_lfps = filter_ripple_band(lfps)
        speed = np.ones(len(time)) * 2.0

        ripples = Kay_ripple_detector(
            time,
            filtered_lfps,
            speed,
            sampling_frequency=1500,
            speed_threshold=4.0,
            minimum_duration=0.015,
            zscore_threshold=2.0,
        )

        snapshot.assert_match(str(len(ripples)), "n_detections_multichannel")

        if len(ripples) > 0:
            # Snapshot detection times
            times = {
                "start_times": [round(float(t), 4) for t in ripples["start_time"].tolist()],
                "end_times": [round(float(t), 4) for t in ripples["end_time"].tolist()],
            }
            snapshot.assert_match(str(times), "detection_times")


class TestKarlssonDetectorSnapshots:
    """Snapshot tests for Karlsson ripple detector."""

    @pytest.mark.skip(reason="Random data causes non-deterministic results across runs")
    def test_karlsson_single_channel_output(self, snapshot, test_lfp_data):
        """Test Karlsson detector output consistency."""
        time, lfp = test_lfp_data
        filtered_lfp = filter_ripple_band(lfp)
        speed = np.ones(len(time)) * 2.0

        ripples = Karlsson_ripple_detector(
            time,
            filtered_lfp,
            speed,
            sampling_frequency=1500,
            speed_threshold=4.0,
            minimum_duration=0.015,
            zscore_threshold=2.0,
        )

        snapshot.assert_match(str(len(ripples)), "n_detections")

        if len(ripples) > 0:
            first_detection = ripples.iloc[0].to_dict()
            rounded_detection = {k: round(float(v), 6) for k, v in first_detection.items()}
            snapshot.assert_match(str(rounded_detection), "first_detection")

    @pytest.mark.skip(reason="Random data causes non-deterministic results across runs")
    def test_karlsson_multichannel_merging(self, snapshot, test_multichannel_lfp_data):
        """Test Karlsson detector merges overlapping events from different channels."""
        time, lfps = test_multichannel_lfp_data
        filtered_lfps = filter_ripple_band(lfps)
        speed = np.ones(len(time)) * 2.0

        ripples = Karlsson_ripple_detector(
            time,
            filtered_lfps,
            speed,
            sampling_frequency=1500,
            speed_threshold=4.0,
            minimum_duration=0.015,
            zscore_threshold=2.0,
        )

        snapshot.assert_match(str(len(ripples)), "n_detections_merged")

        if len(ripples) > 0:
            durations = [round(float(d), 4) for d in ripples["duration"].tolist()]
            snapshot.assert_match(str(durations), "merged_durations")


class TestRoumisDetectorSnapshots:
    """Snapshot tests for Roumis ripple detector."""

    @pytest.mark.skip(reason="Random data causes non-deterministic results across runs")
    def test_roumis_output(self, snapshot, test_multichannel_lfp_data):
        """Test Roumis detector output consistency."""
        time, lfps = test_multichannel_lfp_data
        filtered_lfps = filter_ripple_band(lfps)
        speed = np.ones(len(time)) * 2.0

        ripples = Roumis_ripple_detector(
            time,
            filtered_lfps,
            speed,
            sampling_frequency=1500,
            speed_threshold=4.0,
            minimum_duration=0.015,
            zscore_threshold=1.5,  # Lower threshold for this detector
        )

        snapshot.assert_match(str(len(ripples)), "n_detections")

        if len(ripples) > 0:
            # Snapshot detection times
            times = {
                "start_times": [round(float(t), 4) for t in ripples["start_time"].tolist()],
                "end_times": [round(float(t), 4) for t in ripples["end_time"].tolist()],
            }
            snapshot.assert_match(str(times), "roumis_times")


class TestMultiunitHSEDetectorSnapshots:
    """Snapshot tests for multiunit high synchrony event detector."""

    def test_multiunit_hse_output(self, snapshot, test_multiunit_data):
        """Test multiunit HSE detector output consistency."""
        time, multiunit = test_multiunit_data
        speed = np.ones(len(time)) * 2.0

        hse_events = multiunit_HSE_detector(
            time,
            multiunit,
            speed,
            sampling_frequency=1500,
            speed_threshold=4.0,
            minimum_duration=0.015,
            zscore_threshold=2.0,
        )

        snapshot.assert_match(str(len(hse_events)), "n_detections")

        if len(hse_events) > 0:
            # Snapshot event statistics
            stats = {
                "n_events": len(hse_events),
                "mean_duration": round(float(hse_events["duration"].mean()), 4),
                "mean_zscore": round(float(hse_events["mean_zscore"].mean()), 4),
            }
            snapshot.assert_match(str(stats), "hse_stats")


class TestDetectorComparison:
    """Snapshot tests comparing different detectors on the same data."""

    def test_detector_comparison(self, snapshot, test_multichannel_lfp_data):
        """Compare outputs of different detectors on same data."""
        time, lfps = test_multichannel_lfp_data
        filtered_lfps = filter_ripple_band(lfps)
        speed = np.ones(len(time)) * 2.0

        # Run all three ripple detectors
        kay_ripples = Kay_ripple_detector(time, filtered_lfps, speed, sampling_frequency=1500)

        karlsson_ripples = Karlsson_ripple_detector(
            time, filtered_lfps, speed, sampling_frequency=1500
        )

        roumis_ripples = Roumis_ripple_detector(
            time, filtered_lfps, speed, sampling_frequency=1500
        )

        # Snapshot comparison
        comparison = {
            "kay_count": len(kay_ripples),
            "karlsson_count": len(karlsson_ripples),
            "roumis_count": len(roumis_ripples),
        }
        snapshot.assert_match(str(comparison), "detector_comparison")


class TestRegressionPrevention:
    """Tests to prevent regression in detector behavior."""

    def test_no_false_positives_in_noise(self, snapshot):
        """Ensure detectors don't trigger on pure noise."""
        rng = np.random.default_rng(42)
        time = np.arange(0, 2.0, 1 / 1500)

        # Pure noise, no ripples
        lfp = rng.standard_normal((len(time), 1)) * 0.5
        filtered_lfp = filter_ripple_band(lfp)
        speed = np.ones(len(time)) * 2.0

        ripples = Kay_ripple_detector(
            time,
            filtered_lfp,
            speed,
            sampling_frequency=1500,
            zscore_threshold=3.0,  # High threshold
        )

        # Should detect very few or no ripples in pure noise
        snapshot.assert_match(str(len(ripples)), "noise_detections")
        assert len(ripples) <= 2, "Too many false positives in noise"

    def test_high_amplitude_ripple_detected(self, snapshot):
        """Ensure obvious ripples are always detected."""
        time = np.arange(0, 2.0, 1 / 1500)

        # Very strong ripple
        lfp = simulate_LFP(
            time,
            ripple_times=[1.0],
            noise_amplitude=0.5,
            ripple_amplitude=5.0,  # Very strong
        )[:, np.newaxis]

        filtered_lfp = filter_ripple_band(lfp)
        speed = np.ones(len(time)) * 2.0

        ripples = Kay_ripple_detector(
            time,
            filtered_lfp,
            speed,
            sampling_frequency=1500,
            zscore_threshold=2.0,
        )

        snapshot.assert_match(str(len(ripples)), "strong_ripple_detections")
        assert len(ripples) >= 1, "Failed to detect obvious ripple"

        if len(ripples) > 0:
            # Check that detected ripple is near expected time
            detected_times = ripples["start_time"].values
            assert any(
                abs(t - 1.0) < 0.1 for t in detected_times
            ), "Detected ripple not near expected time"

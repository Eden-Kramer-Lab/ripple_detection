"""End-to-end tests: simulated sessions with known ripples through the public API.

Every detector is driven the way a pipeline drives it, by name from the registry,
on the output of ``simulate_session``, and judged against the ground truth.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from ripple_detection import (
    DETECTORS,
    MULTIUNIT,
    RAW_LFP,
    RIPPLE_BAND_LFP,
    Karlsson_ripple_detector,
    Kay_ripple_detector,
    Long_sharp_wave_ripple_detector,
    filter_ripple_band,
    get_detector,
    simulate_LFP,
    simulate_session,
    simulate_time,
)

FS = 1500.0
# at least 6 s from either edge: the Long detector does not evaluate a candidate
# within local_window (5 s) of a block edge, as the original does at record edges
RIPPLES = [6.0, 10.0, 14.0, 18.0, 22.0, 24.0]
LFP_DETECTORS = [
    "Kay_ripple_detector",
    "Karlsson_ripple_detector",
    "Roumis_ripple_detector",
    "Shvartsman_ripple_detector",
    "Yu_ripple_detector",
    "Zugaro_ripple_detector",
]
ALL_DETECTORS = list(DETECTORS)

# The parameter set Spyglass stores for its "default" RippleParameters entry.
SPYGLASS_DEFAULT = {
    "speed_threshold": 4.0,
    "minimum_duration": 0.015,
    "zscore_threshold": 2.0,
    "smoothing_sigma": 0.004,
    "close_ripple_threshold": 0.0,
}
ACCEPT_SPYGLASS_DEFAULT = {
    "Kay_ripple_detector",
    "Karlsson_ripple_detector",
    "Roumis_ripple_detector",
    "Shvartsman_ripple_detector",
}

# Parameters under which no event can exist, for the empty-result schema: a
# minimum just under the second the unit check allows, far beyond any ripple.
EMPTY_PARAMS = {
    "Kay_ripple_detector": {"minimum_duration": 0.9},
    "Karlsson_ripple_detector": {"minimum_duration": 0.9},
    "Roumis_ripple_detector": {"minimum_duration": 0.9},
    "Shvartsman_ripple_detector": {"minimum_duration": 0.9},
    "Yu_ripple_detector": {"minimum_duration": 0.9},
    "Zugaro_ripple_detector": {"minimum_duration": 0.9, "maximum_duration": None},
    "Long_sharp_wave_ripple_detector": {
        "minimum_sharp_wave_duration": 0.45,
        "minimum_ripple_duration": 0.9,
    },
    "Carey_candidate_detector": {"minimum_duration": 0.9},
    "multiunit_HSE_detector": {"minimum_duration": 0.9},
}

# False positives a detector at its defaults is allowed on this 30 s session.
# Measured on seeds 0-7 of this session: Kay 0-4, Roumis 0-5; each budget is
# one above the worst seed. Yu's data-driven threshold falls to about 1.1 SD
# when channels share half their noise (see TestYuUnderCorrelatedNoise) and
# gave 0-17, so its budget is wide.
FALSE_POSITIVE_BUDGET = {
    "Kay_ripple_detector": 5,
    "Roumis_ripple_detector": 6,
    "Karlsson_ripple_detector": 4,
    "Shvartsman_ripple_detector": 2,
    "Yu_ripple_detector": 20,
    "Zugaro_ripple_detector": 4,
    "Long_sharp_wave_ripple_detector": 4,
    "Carey_candidate_detector": 3,
    "multiunit_HSE_detector": 6,
}
# How far an event's bounds may sit from the ripple envelope's 1 percent points:
# the envelope-based detectors extend each event to where the trace returns to
# its mean, which lands within a few tens of milliseconds of those points.
# Zugaro's bounds are where the squared power crosses 2 SD, later. Long marks
# the sharp wave, Carey and HSE the population burst, which the ripple window
# does not bound, so they are not held to it.
BOUNDARY_TOLERANCE_MS = dict.fromkeys(LFP_DETECTORS, 40.0) | {"Zugaro_ripple_detector": 50.0}

BASE_COLUMNS = [
    "start_time", "end_time", "duration", "n_samples", "max_sustained_zscore",
    "mean_zscore", "median_zscore", "max_zscore", "min_zscore", "area", "total_energy",
    "speed_at_start", "speed_at_end", "max_speed", "min_speed", "median_speed",
    "mean_speed", "clipped_start", "clipped_end", "peak_time",
]  # fmt: skip


@pytest.fixture(scope="module")
def session():
    time = simulate_time(int(30 * FS), FS)
    return simulate_session(
        time,
        RIPPLES,
        n_channels=4,
        n_units=100,
        channel_gains=[1.0, 0.9, 0.7, 0.6],
        ripple_snr=6.0,
        rng=0,
    )


@pytest.fixture(scope="module")
def filtered(session):
    return filter_ripple_band(session.lfps, sampling_frequency=FS)


@pytest.fixture(scope="module")
def loud_session():
    """One loud, isolated ripple in the middle of 20 s."""
    time = simulate_time(int(20 * FS), FS)
    return simulate_session(
        time,
        [10.0],
        n_channels=4,
        n_units=100,
        ripple_snr=10.0,
        ripple_duration=0.08,
        ripple_frequency=200.0,
        rng=3,
    )


def signals_for(name, session, filtered):
    """The positional signals the registry says the detector takes, in order."""
    by_kind = {
        RIPPLE_BAND_LFP: filtered,
        MULTIUNIT: session.multiunit,
        RAW_LFP: session.raw_lfp,
    }
    return tuple(by_kind[kind] for kind in get_detector(name).inputs)


def keyword_signals_for(name, session):
    """The signals the detector requires by name, from the session fields
    named after them."""
    return {
        signal: getattr(session, signal)
        for signal in get_detector(name).required_keyword_inputs
    }


def run(name, session, filtered, **params):
    spec = get_detector(name)
    return spec.detector(
        session.time,
        *signals_for(name, session, filtered),
        session.speed,
        FS,
        **keyword_signals_for(name, session),
        **params,
    )


def overlaps(events, window):
    start, end = window
    return (events.start_time <= end) & (events.end_time >= start)


def recall(events, windows):
    return np.mean([overlaps(events, w).any() for w in windows])


def n_false_positives(events, windows):
    matched = np.zeros(len(events), dtype=bool)
    for w in windows:
        matched |= overlaps(events, w).to_numpy()
    return int((~matched).sum())


class TestRegistryPath:
    """name -> spec -> check_parameters -> check_inputs -> call, for every detector."""

    @pytest.mark.parametrize("name", ALL_DETECTORS)
    def test_defaults_check_and_run(self, name, session, filtered):
        spec = get_detector(name)
        spec.check_parameters(spec.parameters)
        spec.check_inputs(
            *signals_for(name, session, filtered), **keyword_signals_for(name, session)
        )
        events = run(name, session, filtered, **spec.parameters)
        assert isinstance(events, pd.DataFrame)
        assert events.index.name == "event_number"
        assert {"start_time", "end_time"} <= set(events.columns)

    @pytest.mark.parametrize("name", ALL_DETECTORS)
    def test_the_spyglass_default_parameter_set(self, name, session, filtered):
        """Recorded: which detectors accept the dict Spyglass stores for Kay."""
        spec = get_detector(name)
        if name in ACCEPT_SPYGLASS_DEFAULT:
            spec.check_parameters(SPYGLASS_DEFAULT)
            events = run(name, session, filtered, **SPYGLASS_DEFAULT)
            assert recall(events, session.ripple_windows) >= 0.8
        else:
            with pytest.raises(ValueError, match="does not take"):
                spec.check_parameters(SPYGLASS_DEFAULT)


class TestGroundTruthRecovery:
    """Six ripples six times the ripple-band background, 30 s, four channels,
    a hundred units."""

    @pytest.mark.parametrize("name", ALL_DETECTORS)
    def test_recall_and_false_positives(self, name, session, filtered):
        events = run(name, session, filtered)
        assert recall(events, session.ripple_windows) >= 5 / 6
        assert n_false_positives(events, session.ripple_windows) <= FALSE_POSITIVE_BUDGET[name]

    @pytest.mark.parametrize("name", LFP_DETECTORS)
    def test_event_boundaries_are_near_the_true_ones(self, name, session, filtered):
        events = run(name, session, filtered)
        tolerance = BOUNDARY_TOLERANCE_MS[name] / 1000
        for start, end in session.ripple_windows:
            hits = events[overlaps(events, (start, end))]
            if len(hits) == 0:
                continue
            assert abs(hits.start_time.min() - start) <= tolerance, name
            assert abs(hits.end_time.max() - end) <= tolerance, name


class TestWholeChainAtOtherRates:
    """simulate -> filter_ripple_band (designed filter) -> detector, away from 1500 Hz."""

    @pytest.mark.parametrize("fs", [1000.0, 2000.0])
    @pytest.mark.parametrize(
        "name", ["Kay_ripple_detector", "Karlsson_ripple_detector", "Zugaro_ripple_detector"]
    )
    def test_ripples_are_recovered(self, name, fs):
        time = simulate_time(int(20 * fs), fs)
        sess = simulate_session(
            time,
            [3.0, 8.0, 13.0, 17.0],
            n_channels=3,
            n_units=5,
            ripple_snr=6.0,
            rng=1,
        )
        filt = filter_ripple_band(sess.lfps, sampling_frequency=fs)
        events = get_detector(name).detector(sess.time, filt, sess.speed, fs)
        assert recall(events, sess.ripple_windows) == 1.0

    def test_kay_at_30_khz(self):
        fs = 30000.0
        time = simulate_time(int(8 * fs), fs)
        sess = simulate_session(
            time, [2.0, 4.0, 6.0], n_channels=2, n_units=3, ripple_snr=6.0, rng=2
        )
        filt = filter_ripple_band(sess.lfps, sampling_frequency=fs)
        events = Kay_ripple_detector(sess.time, filt, sess.speed, fs)
        assert recall(events, sess.ripple_windows) == 1.0


class TestCrossDetectorAgreement:
    """One loud, isolated ripple: every detector finds it, the ripple-band
    detectors and Long find exactly one event there, and the ripple-band
    detectors place its start within 25 ms of one another."""

    def test_every_detector_finds_the_ripple(self, loud_session):
        filt = filter_ripple_band(loud_session.lfps, sampling_frequency=FS)
        window = loud_session.ripple_windows[0]
        starts = {}
        for name in ALL_DETECTORS:
            events = run(name, loud_session, filt)
            hits = events[overlaps(events, window)]
            if name in ("Carey_candidate_detector", "multiunit_HSE_detector"):
                assert len(hits) >= 1, name  # a population burst may split
            else:
                assert len(hits) == 1, name
            starts[name] = hits.start_time.min()
        lfp_starts = np.array([starts[name] for name in LFP_DETECTORS])
        assert lfp_starts.max() - lfp_starts.min() <= 0.025


class TestYuUnderCorrelatedNoise:
    """Yu's threshold is read off the mirrored left flank of the immobility
    histogram. Channels that share noise give the median of their z-scored
    envelopes a heavier right tail, which pulls the estimate down and lets
    noise through; the same channels with independent noise do not. Pinned so
    a change in the estimator shows up here."""

    def test_shared_noise_lowers_the_threshold_and_admits_noise(self):
        time = simulate_time(int(30 * FS), FS)
        thresholds, false_positives = {}, {}
        for shared in (0.0, 0.5):
            sess = simulate_session(
                time,
                RIPPLES,
                n_channels=4,
                n_units=5,
                channel_gains=[1.0, 0.9, 0.7, 0.6],
                shared_noise_fraction=shared,
                ripple_snr=6.0,
                rng=0,
            )
            filt = filter_ripple_band(sess.lfps, sampling_frequency=FS)
            events = run("Yu_ripple_detector", sess, filt)
            assert recall(events, sess.ripple_windows) == 1.0
            thresholds[shared] = events.detection_threshold_zscore.iloc[0]
            false_positives[shared] = n_false_positives(events, sess.ripple_windows)
        assert thresholds[0.5] < thresholds[0.0] - 0.5
        assert false_positives[0.0] <= 1
        assert false_positives[0.5] >= 5


class TestOutputContract:
    @pytest.mark.parametrize("name", ALL_DETECTORS)
    def test_columns_dtypes_index_and_order(self, name, session, filtered):
        events = run(name, session, filtered)
        assert len(events) > 0
        assert set(BASE_COLUMNS) <= set(events.columns)
        assert events.index.name == "event_number"
        np.testing.assert_array_equal(events.index, np.arange(1, len(events) + 1))
        assert events.n_samples.dtype.kind == "i"
        assert events.clipped_start.dtype == bool
        assert events.clipped_end.dtype == bool
        for column in BASE_COLUMNS:
            if column not in ("n_samples", "clipped_start", "clipped_end"):
                assert events[column].dtype == np.float64, column
        assert events.start_time.is_monotonic_increasing
        assert (events.end_time >= events.start_time).all()
        assert events.start_time.le(events.peak_time).all()
        assert events.peak_time.le(events.end_time).all()

    @pytest.mark.parametrize("name", ALL_DETECTORS)
    def test_an_empty_result_has_the_same_schema(self, name, session, filtered):
        full = run(name, session, filtered)
        empty = run(name, session, filtered, **EMPTY_PARAMS[name])
        assert len(empty) == 0
        assert list(empty.columns) == list(full.columns)
        assert empty.index.name == "event_number"
        assert dict(empty.dtypes) == dict(full.dtypes)


class TestInvariances:
    @pytest.mark.parametrize("name", LFP_DETECTORS)
    def test_a_time_offset_shifts_the_events_and_nothing_else(self, name, session, filtered):
        base = run(name, session, filtered)
        offset = 12345.678
        shifted = get_detector(name).detector(
            session.time + offset, filtered, session.speed, FS
        )
        np.testing.assert_allclose(shifted.start_time - offset, base.start_time, atol=1e-6)
        np.testing.assert_allclose(shifted.end_time - offset, base.end_time, atol=1e-6)
        np.testing.assert_allclose(shifted.max_zscore, base.max_zscore)

    @pytest.mark.parametrize("name", LFP_DETECTORS)
    def test_scaling_the_signal_changes_nothing(self, name, session, filtered):
        base = run(name, session, filtered)
        scaled = get_detector(name).detector(
            session.time, 1000.0 * filtered, session.speed, FS
        )
        np.testing.assert_allclose(scaled.start_time, base.start_time)
        np.testing.assert_allclose(scaled.end_time, base.end_time)
        np.testing.assert_allclose(scaled.max_zscore, base.max_zscore, rtol=1e-9)

    @pytest.mark.parametrize("name", LFP_DETECTORS)
    def test_channel_order_does_not_matter(self, name, session, filtered):
        base = run(name, session, filtered)
        reversed_channels = get_detector(name).detector(
            session.time, filtered[:, ::-1], session.speed, FS
        )
        np.testing.assert_allclose(reversed_channels.start_time, base.start_time, atol=1e-9)
        np.testing.assert_allclose(reversed_channels.end_time, base.end_time, atol=1e-9)

    @pytest.mark.parametrize("detector", [Kay_ripple_detector, Karlsson_ripple_detector])
    def test_a_higher_threshold_finds_a_subset(self, detector, session, filtered):
        low = detector(session.time, filtered, session.speed, FS, zscore_threshold=2.0)
        high = detector(session.time, filtered, session.speed, FS, zscore_threshold=3.0)
        assert len(high) <= len(low)
        for _, event in high.iterrows():
            assert overlaps(low, (event.start_time, event.end_time)).any()


class TestRecordingEdges:
    def test_a_ripple_cut_by_the_recording_end_is_flagged(self):
        time = simulate_time(int(6 * FS), FS)
        lfp = simulate_LFP(time, [1.0, time[-1] - 0.01], ripple_snr=8.0, rng=4)
        filt = filter_ripple_band(lfp[:, np.newaxis], sampling_frequency=FS)
        events = Kay_ripple_detector(time, filt, np.zeros_like(time), FS)
        last = events.iloc[-1]
        assert last.end_time == time[-1]
        assert last.clipped_end
        assert not last.clipped_start
        assert not events.iloc[0].clipped_start

    def test_a_ripple_cut_by_the_recording_start_is_flagged(self):
        time = simulate_time(int(6 * FS), FS)
        lfp = simulate_LFP(time, [0.01, 4.0], ripple_snr=8.0, rng=5)
        filt = filter_ripple_band(lfp[:, np.newaxis], sampling_frequency=FS)
        events = Kay_ripple_detector(time, filt, np.zeros_like(time), FS)
        first = events.iloc[0]
        assert first.start_time == time[0]
        assert first.clipped_start
        assert not first.clipped_end


class TestLongSeeding:
    def test_none_runs_and_an_int_equals_its_generator(self, session):
        args = (session.time, session.raw_lfp, session.speed, FS)
        signal = {"sharp_wave_lfp": session.sharp_wave_lfp}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Long_sharp_wave_ripple_detector(*args, **signal, rng=None)
        from_int = Long_sharp_wave_ripple_detector(*args, **signal, rng=7)
        from_generator = Long_sharp_wave_ripple_detector(
            *args, **signal, rng=np.random.default_rng(7)
        )
        pd.testing.assert_frame_equal(from_int, from_generator)

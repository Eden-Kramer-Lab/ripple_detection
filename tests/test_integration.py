"""End-to-end tests: simulated sessions with known ripples through the public API.

Every detector is driven the way a pipeline drives it, by name from the registry,
on the output of ``simulate_session``, and judged against the ground truth.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from _synthetic import _non_event_tables, _one_non_event_table

import ripple_detection as rd
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


ORIGINS = [86_400.0, 1.7e9]  # a day into a recording; a Unix time
RUNNING = [(2.0, 4.0), (13.5, 14.5), (26.0, 28.0)]  # the second covers a ripple


def assert_times_shifted(shifted, base, origin):
    """Times computed on timestamps moved by `origin` are `base` moved by it,
    to within the timestamps' own rounding (a few ulps of `origin`, under a
    thousandth of a sample): the same samples, not merely nearby ones."""
    shifted, base = np.asarray(shifted, dtype=float), np.asarray(base, dtype=float)
    assert shifted.shape == base.shape
    np.testing.assert_allclose(shifted - origin, base, rtol=0, atol=8 * np.spacing(origin))


# integrals over the timestamps, which carry each step's rounding
TIME_INTEGRALS = {"area", "total_energy"}


def assert_frame_shifted(shifted, base, origin):
    """A detector's DataFrame moved by `origin`: its ``*_time`` columns as in
    `assert_times_shifted`, its durations to the same rounding, integrals over
    time to the relative rounding of one step, and every other column as
    computed at 0."""
    assert list(shifted.columns) == list(base.columns)
    assert len(shifted) == len(base)
    for column in base:
        if column.endswith("_time"):
            assert_times_shifted(shifted[column], base[column], origin)
        elif column == "duration":
            np.testing.assert_allclose(
                shifted[column], base[column], rtol=0, atol=8 * np.spacing(origin)
            )
        elif base[column].dtype == object:  # Shvartsman's participant channels
            assert shifted[column].tolist() == base[column].tolist(), column
        elif column in TIME_INTEGRALS:
            np.testing.assert_allclose(
                shifted[column], base[column], rtol=np.spacing(origin) * FS, err_msg=column
            )
        else:
            np.testing.assert_allclose(
                shifted[column], base[column], rtol=1e-9, err_msg=column
            )


@pytest.fixture(scope="module")
def moving_session():
    """The session's ripples with running bouts, theta and delta, so speed and
    state rules have something to decide."""
    time = simulate_time(int(30 * FS), FS)
    return simulate_session(
        time,
        RIPPLES,
        n_channels=4,
        n_units=100,
        ripple_snr=6.0,
        running_intervals=RUNNING,
        theta_amplitude=1.0,
        delta_amplitude=1.0,
        rng=0,
    )


@pytest.fixture(scope="module")
def base(moving_session):
    """The moving session's ripple-band LFP, Kay events and their bounds."""
    filtered = filter_ripple_band(moving_session.lfps, sampling_frequency=FS)
    kay = Kay_ripple_detector(moving_session.time, filtered, moving_session.speed, FS)
    events = kay[["start_time", "end_time"]].to_numpy()
    assert len(events) >= 4
    return filtered, kay, events


class TestTimeOrigin:
    """Every public function that reads timestamps gives, on a clock that
    starts a day or a Unix time in, what it gives from 0, moved by the
    origin. Far from zero a timestamp rounds to a few ulps of its magnitude
    (2.4e-7 s at 1.7e9), so a tolerance relative to a time is too wide there
    and an absolute one too narrow; thresholds below are measured from the
    data, so the comparisons land exactly on their boundaries."""

    @pytest.mark.parametrize("origin", ORIGINS)
    @pytest.mark.parametrize("name", ALL_DETECTORS)
    def test_detectors(self, name, origin, moving_session, base):
        filtered, _, _ = base
        spec = get_detector(name)
        signals = signals_for(name, moving_session, filtered)
        keywords = keyword_signals_for(name, moving_session)
        at_zero = spec.detector(
            moving_session.time, *signals, moving_session.speed, FS, **keywords
        )
        shifted = spec.detector(
            moving_session.time + origin, *signals, moving_session.speed, FS, **keywords
        )
        assert len(at_zero) > 0
        assert_frame_shifted(shifted, at_zero, origin)

    @pytest.mark.parametrize("origin", ORIGINS)
    def test_event_rules_on_measured_boundaries(self, origin, moving_session, base):
        _, kay, events = base
        time, shifted = moving_session.time, moving_session.time + origin
        moved = events + origin

        gaps = events[1:, 0] - events[:-1, 1]
        gap = float(gaps.min())
        for kwargs in ({}, {"inclusive": True}):
            assert_times_shifted(
                rd.merge_close_events(moved, gap, **kwargs),
                rd.merge_close_events(events, gap, **kwargs),
                origin,
            )
        assert_times_shifted(
            rd.require_isolation(moved, gap), rd.require_isolation(events, gap), origin
        )

        peaks = kay.peak_time.to_numpy()
        peak_gap = float(np.diff(peaks).min())
        moved_frame = kay.assign(
            start_time=kay.start_time + origin,
            end_time=kay.end_time + origin,
            peak_time=kay.peak_time + origin,
        )
        assert_times_shifted(
            rd.merge_close_events(moved_frame, peak_gap, measure="peak"),
            rd.merge_close_events(kay, peak_gap, measure="peak"),
            origin,
        )

        # references three samples later: each event overlaps its own by its
        # length less three samples, which is the minimum asked of the first
        index = np.searchsorted(time, events)
        reference = time[np.minimum(index + 3, len(time) - 1)]
        overlap = float(events[0, 1] - reference[0, 0])
        for rule in (rd.require_overlap, rd.exclude_overlap):
            assert_times_shifted(
                rule(moved, reference + origin, overlap),
                rule(events, reference, overlap),
                origin,
            )
        assert_times_shifted(
            rd.require_times_inside(moved, peaks + origin),
            rd.require_times_inside(events, peaks),
            origin,
        )

        # every other event as an interval, read off each clock's own samples,
        # which round apart from the moved events by a few ulps
        own = index[::2]
        kept = rd.require_inside(events, time[own])
        assert len(kept) == len(own)
        assert_times_shifted(rd.require_inside(moved, shifted[own]), kept, origin)
        np.testing.assert_array_equal(
            rd.intervals_to_mask(shifted, shifted[own]), rd.intervals_to_mask(time, time[own])
        )
        later = np.minimum(index + 3, len(time) - 1)[::2]
        assert_times_shifted(
            rd.intersect_intervals(shifted[own], shifted[later]),
            rd.intersect_intervals(time[own], time[later]),
            origin,
        )

        speed = moving_session.speed
        for rule in ("endpoints", "all", "mean", "median"):
            assert_times_shifted(
                rd.exclude_movement(moved, speed, shifted, 4.0, rule),
                rd.exclude_movement(events, speed, time, 4.0, rule),
                origin,
            )
        assert_times_shifted(
            rd.exclude_movement_by_majority(moved, speed, shifted),
            rd.exclude_movement_by_majority(events, speed, time),
            origin,
        )

    @pytest.mark.parametrize("origin", ORIGINS)
    def test_spike_and_trace_rules(self, origin, moving_session, base):
        filtered, _, events = base
        time, shifted = moving_session.time, moving_session.time + origin
        moved = events + origin
        multiunit = moving_session.multiunit
        np.testing.assert_array_equal(
            rd.count_spikes_in_events(moved, multiunit, shifted),
            rd.count_spikes_in_events(events, multiunit, time),
        )
        assert_times_shifted(
            rd.require_active_units(moved, multiunit, shifted, minimum_active_units=10),
            rd.require_active_units(events, multiunit, time, minimum_active_units=10),
            origin,
        )
        assert_times_shifted(
            rd.trim_events_to_spike_windows(moved, multiunit, shifted),
            rd.trim_events_to_spike_windows(events, multiunit, time),
            origin,
        )

        trace = rd.get_Kay_ripple_consensus_trace(filtered, FS, time=time)
        np.testing.assert_allclose(
            rd.get_Kay_ripple_consensus_trace(filtered, FS, time=shifted), trace, rtol=1e-12
        )
        np.testing.assert_allclose(
            rd.get_Yu_ripple_consensus_trace(filtered, FS, time=shifted),
            rd.get_Yu_ripple_consensus_trace(filtered, FS, time=time),
            rtol=1e-12,
        )
        level = float(np.median(trace))
        for sides in ("both", "start", "end"):
            assert_times_shifted(
                rd.trim_events_to_trace(moved, trace, shifted, level, sides=sides),
                rd.trim_events_to_trace(events, trace, time, level, sides=sides),
                origin,
            )
        assert_times_shifted(
            rd.require_trace_peak(moved, trace, shifted, 3 * level),
            rd.require_trace_peak(events, trace, time, 3 * level),
            origin,
        )
        zscored = (trace - trace.mean()) / trace.std()
        assert_times_shifted(
            rd.threshold_by_zscore(zscored, shifted),
            rd.threshold_by_zscore(zscored, time),
            origin,
        )
        assert_frame_shifted(
            rd.detect_events_from_trace(shifted, trace, moving_session.speed, FS),
            rd.detect_events_from_trace(time, trace, moving_session.speed, FS),
            origin,
        )

    @pytest.mark.parametrize("origin", ORIGINS)
    def test_spiking_state_and_score(self, origin, moving_session, base):
        _, _, events = base
        time, shifted = moving_session.time, moving_session.time + origin
        few_units = {"units": np.arange(5)}
        for kwargs in ({"minimum_silence": 0.1}, {"minimum_silence": 0.1, "window": 0.2}):
            at_zero = rd.detect_silence_bounded_events(
                time, moving_session.multiunit, FS, **kwargs, **few_units
            )
            assert len(at_zero) > 0
            assert_frame_shifted(
                rd.detect_silence_bounded_events(
                    shifted, moving_session.multiunit, FS, **kwargs, **few_units
                ),
                at_zero,
                origin,
            )

        ratio = rd.theta_delta_ratio(moving_session.raw_lfp, FS, time=time)
        np.testing.assert_allclose(
            rd.theta_delta_ratio(moving_session.raw_lfp, FS, time=shifted), ratio, rtol=1e-12
        )
        still = rd.state_intervals(moving_session.speed, time, 1.0)
        assert len(still) >= 3
        length = float(np.diff(still, axis=1).min())
        gap = float((still[1:, 0] - still[:-1, 1]).min())
        for kwargs in ({"minimum_duration": length}, {"merge_gap": gap}):
            assert_times_shifted(
                rd.state_intervals(moving_session.speed, shifted, 1.0, **kwargs),
                rd.state_intervals(moving_session.speed, time, 1.0, **kwargs),
                origin,
            )

        examples = events[:3]
        np.testing.assert_allclose(
            rd.carey_spectral_ripple_score(
                shifted, moving_session.raw_lfp, FS, examples + origin
            ),
            rd.carey_spectral_ripple_score(time, moving_session.raw_lfp, FS, examples),
            rtol=1e-12,
        )

    @pytest.mark.parametrize("origin", ORIGINS)
    def test_sample_counts(self, origin, moving_session):
        time = moving_session.time
        for duration in (0.015, 0.1, 1 / 3):
            assert rd.minimum_sample_count(time + origin, duration) == rd.minimum_sample_count(
                time, duration
            )
        n_samples = np.arange(40)
        np.testing.assert_array_equal(
            rd.sample_count_within(n_samples, time + origin, 0.01, 0.02),
            rd.sample_count_within(n_samples, time, 0.01, 0.02),
        )

    @pytest.mark.parametrize("origin", ORIGINS)
    def test_simulated_speed_state_and_spikes(self, origin):
        """Speed follows the running bouts and spikes the ripples, not the
        clock; each agrees to its slope times the timestamps' rounding. Theta
        and delta are rhythms on the clock, so they agree at origins that are
        whole cycles of both, as these are."""
        time = simulate_time(int(30 * FS), FS)
        running, ripples = np.asarray(RUNNING), np.asarray(RIPPLES)
        for simulate in (rd.simulate_speed, rd.simulate_theta_delta):
            at_zero = simulate(time, running)
            atol = np.abs(np.gradient(at_zero, time)).max() * 8 * np.spacing(origin)
            np.testing.assert_allclose(
                simulate(time + origin, running + origin), at_zero, rtol=0, atol=atol
            )
        np.testing.assert_array_equal(
            rd.simulate_multiunit(time + origin, ripples + origin, 100, rng=0),
            rd.simulate_multiunit(time, ripples, 100, rng=0),
        )

    @pytest.mark.parametrize("origin", ORIGINS)
    def test_simulated_ripples(self, origin):
        time = simulate_time(int(30 * FS), FS)
        ripples = np.asarray(RIPPLES)
        at_zero = rd.simulate_LFP(time, ripples, ripple_frequency=(150.0, 250.0), rng=0)
        shifted = rd.simulate_LFP(
            time + origin, ripples + origin, ripple_frequency=(150.0, 250.0), rng=0
        )
        atol = np.abs(np.gradient(at_zero, time)).max() * 8 * np.spacing(origin)
        np.testing.assert_allclose(shifted, at_zero, rtol=0, atol=atol)

    @pytest.mark.parametrize("origin", ORIGINS)
    def test_simulated_network_events(self, origin):
        """Latent events follow the running bouts, not the clock: the table
        and its truth windows move by the origin, and the rendered session
        agrees to its slope times the timestamps' rounding (theta and delta
        are whole cycles at these origins); the spikes are the same."""
        time = simulate_time(int(30 * FS), FS)
        running = np.asarray(RUNNING)
        at_zero = rd.draw_network_events(
            time, event_rate=1.0, running_intervals=running, rng=0
        )
        shifted = rd.draw_network_events(
            time + origin, event_rate=1.0, running_intervals=running + origin, rng=0
        )
        assert len(at_zero) > 0
        assert_times_shifted(shifted.center_time, at_zero.center_time, origin)
        pd.testing.assert_frame_equal(
            shifted.drop(columns="center_time"), at_zero.drop(columns="center_time")
        )
        for expression in (None, "network"):
            windows = rd.truth_windows(at_zero, 0.25, expression=expression)
            moved = rd.truth_windows(shifted, 0.25, expression=expression)
            for column in ("start_time", "end_time", "peak_time"):
                assert_times_shifted(moved[column], windows[column], origin)
            labels = [c for c in windows if not c.endswith("_time")]
            pd.testing.assert_frame_equal(moved[labels], windows[labels])

        # the rate is given: far from zero the median timestamp step no longer
        # gives 1500 Hz exactly, and the SNR sizing filters at the rate
        session = rd.simulate_network_session(
            time, at_zero, running_intervals=running, rng=1, sampling_frequency=FS
        )
        moved_session = rd.simulate_network_session(
            time + origin, shifted, running_intervals=running + origin, rng=1,
            sampling_frequency=FS,
        )  # fmt: skip
        for name in ("lfps", "sharp_wave_lfp", "speed"):
            signal = getattr(session, name)
            atol = np.abs(np.gradient(signal, time, axis=0)).max() * 8 * np.spacing(origin)
            np.testing.assert_allclose(getattr(moved_session, name), signal, rtol=0, atol=atol)
        np.testing.assert_array_equal(moved_session.multiunit, session.multiunit)
        pd.testing.assert_frame_equal(moved_session.ripple_channels, session.ripple_channels)

    @pytest.mark.parametrize("origin", ORIGINS)
    def test_simulated_non_events(self, origin):
        """Non-events follow the running bouts, not the clock: the table and
        its truth windows move by the origin; rendered, the LFP agrees to its
        slope times the timestamps' rounding and the spikes, leaked ones
        included, are the same. (A leaked spike within the timestamps'
        rounding of where its sample changes can move by one; none of this
        draw's spikes is.)"""
        time = simulate_time(int(30 * FS), FS)
        running = np.asarray(RUNNING)
        rates = {"spike_leakage": 20.0, "emg": 10.0, "fast_gamma": 20.0, "theta_burst": 60.0}
        at_zero = rd.draw_non_events(time, rates=rates, running_intervals=running, rng=0)
        shifted = rd.draw_non_events(
            time + origin, rates=rates, running_intervals=running + origin, rng=0
        )
        assert set(at_zero.non_event_type) == set(rd.NON_EVENT_TYPES)
        assert_times_shifted(shifted.center_time, at_zero.center_time, origin)
        pd.testing.assert_frame_equal(
            shifted.drop(columns="center_time"), at_zero.drop(columns="center_time")
        )
        windows = rd.truth_windows(at_zero, 0.25)
        moved = rd.truth_windows(shifted, 0.25)
        for column in ("start_time", "end_time", "peak_time"):
            assert_times_shifted(moved[column], windows[column], origin)

        events = rd.draw_network_events(time, event_rate=0.0)
        session = rd.simulate_network_session(
            time, events, non_events=at_zero, running_intervals=running, rng=1,
            sampling_frequency=FS,
        )  # fmt: skip
        moved_session = rd.simulate_network_session(
            time + origin, events, non_events=shifted, running_intervals=running + origin,
            rng=1, sampling_frequency=FS,
        )  # fmt: skip
        for name in ("lfps", "sharp_wave_lfp"):
            signal = getattr(session, name)
            atol = np.abs(np.gradient(signal, time, axis=0)).max() * 8 * np.spacing(origin)
            np.testing.assert_allclose(getattr(moved_session, name), signal, rtol=0, atol=atol)
        np.testing.assert_array_equal(moved_session.multiunit, session.multiunit)

    def test_draws_check_the_rate_given(self):
        """Far from zero the timestamps' step gives 1500.1 Hz, a Nyquist
        frequency of 750.05 Hz: a frequency of 750.03 Hz draws on it, but not
        at the rate given, which the renderer then uses. A rate the step
        disagrees with raises."""
        time = simulate_time(int(30 * FS), FS) + 1.7e9
        rd.draw_network_events(time, ripple_frequency=(160.0, 750.03), rng=0)
        with pytest.raises(ValueError, match="ripple_frequency"):
            rd.draw_network_events(
                time, ripple_frequency=(160.0, 750.03), rng=0, sampling_frequency=FS
            )
        high = {"fast_gamma_frequency": (60.0, 750.03), "fast_gamma_band": (60.0, 100.0)}
        with pytest.raises(ValueError, match="fast_gamma_frequency"):
            rd.draw_non_events(time, rng=0, sampling_frequency=FS, **high)
        for draw in (rd.draw_network_events, rd.draw_non_events):
            with pytest.raises(ValueError, match="disagrees with time's step"):
                draw(time, rng=0, sampling_frequency=1000.0)
            # the rate only enters the checks: the table is the same
            pd.testing.assert_frame_equal(
                draw(time, rng=0, sampling_frequency=FS), draw(time, rng=0)
            )

    @pytest.mark.parametrize("origin", ORIGINS)
    def test_leaked_spikes_halfway_between_samples(self, origin):
        """Leaked spikes that fall halfway between samples, and one just short
        of halfway, take the nearest sample, a tie the later one, at any
        origin: three 3 ms apart about 3 s (the first and last halfway), two a
        sample apart about 4 s (both halfway, not collapsed onto one), two
        two samples apart about 5 s plus half a sample (a centre the clock
        rounds far from zero), and three a sample apart 0.45 of a sample past
        2 s."""
        time = simulate_time(int(6 * FS), FS)
        bursts = [
            (3.0, 3, 0.003),
            (4.0, 2, 1 / FS),
            (5.0 + 0.5 / FS, 2, 2 / FS),
            (2.0 + 0.45 / FS, 3, 1 / FS),
        ]  # (centre, spikes, interval)
        rows = _non_event_tables(
            *(
                _one_non_event_table(
                    "spike_leakage", center_time=center, n_spikes=n_spikes, isi=isi,
                    rise_sigma=(n_spikes - 1) * isi / 6, decay_sigma=(n_spikes - 1) * isi / 6,
                    channel=0, n_units=1,
                )
                for center, n_spikes, isi in bursts
            )
        )  # fmt: skip
        events = rd.draw_network_events(time, event_rate=0.0)
        options = {"rng": 1, "sampling_frequency": FS, "noise_amplitude": 0.0}
        at_zero = rd.simulate_network_session(time, events, non_events=rows, **options)
        moved = rows.assign(center_time=rows.center_time + origin)
        shifted = rd.simulate_network_session(
            time + origin, events, non_events=moved, **options
        )
        plain = rd.simulate_network_session(time, events, **options)
        leaked = np.flatnonzero((at_zero.multiunit - plain.multiunit).sum(axis=1))
        np.testing.assert_array_equal(
            leaked, [2999, 3000, 3001, 4496, 4500, 4505, 6000, 6001, 7500, 7502]
        )
        np.testing.assert_array_equal(shifted.multiunit, at_zero.multiunit)

    @pytest.mark.parametrize("origin", ORIGINS)
    @pytest.mark.parametrize("spike_model", ["poisson", "refractory"])
    def test_simulated_network_spikes_at_a_given_rate(self, origin, spike_model):
        """With the rate given, spikes come from it, not from the timestamps'
        spacing, which rounds far from zero: the same counts at any origin."""
        time = simulate_time(int(60 * FS), FS)
        events = rd.draw_network_events(time, event_rate=0.0)
        options = {
            "unit_counts": {"interneuron": 10}, "spike_model": spike_model, "rng": 1,
            "sampling_frequency": FS,
        }  # fmt: skip
        at_zero = rd.simulate_network_session(time, events, **options)
        shifted = rd.simulate_network_session(time + origin, events, **options)
        assert at_zero.multiunit.sum() > 0
        np.testing.assert_array_equal(shifted.multiunit, at_zero.multiunit)

    @pytest.mark.parametrize("origin", ORIGINS)
    def test_evaluation(self, origin, moving_session, base):
        """Matching, comparison, consensus and labels on a moved clock: the
        same pairs, errors to the timestamps' rounding, overlap ratios to
        that rounding over the shortest length, and a minimum IoU and a tie
        between two windows' overlaps measured from the data, so they land
        exactly on their boundaries."""
        _, kay, events = base
        time, shifted = moving_session.time, moving_session.time + origin
        windows = moving_session.ripple_windows
        moved_kay = kay.assign(
            start_time=kay.start_time + origin,
            end_time=kay.end_time + origin,
            peak_time=kay.peak_time + origin,
        )
        # every event two samples later at the start and one earlier at the
        # end, read off each clock's own samples
        index = np.searchsorted(time, events)
        inner = np.clip(index + np.array([2, -1]), 0, len(time) - 1)
        shortest = min(np.diff(windows).min(), np.diff(events).min())
        time_atol = 8 * np.spacing(origin)
        ratio_atol = 4 * time_atol / shortest

        def assert_matchings_shifted(moved, at_zero):
            pd.testing.assert_frame_equal(
                moved.pairs[["reference_index", "detected_index"]],
                at_zero.pairs[["reference_index", "detected_index"]],
            )
            for column in ("iou", "coverage", "temporal_precision"):
                np.testing.assert_allclose(
                    moved.pairs[column], at_zero.pairs[column], rtol=0, atol=ratio_atol
                )
            for column in ("onset_error", "offset_error", "peak_error"):
                np.testing.assert_allclose(
                    moved.pairs[column], at_zero.pairs[column], rtol=0, atol=time_atol
                )
            np.testing.assert_array_equal(moved.reference_overlaps, at_zero.reference_overlaps)
            np.testing.assert_array_equal(moved.detected_overlaps, at_zero.detected_overlaps)

        at_zero = rd.match_events(windows, kay)
        moved = rd.match_events(windows + origin, moved_kay)
        assert len(at_zero.pairs) >= 4
        assert_matchings_shifted(moved, at_zero)
        narrower = windows + np.array([0.005, -0.005])
        np.testing.assert_allclose(
            moved.boundary_errors(narrower + origin),
            at_zero.boundary_errors(narrower),
            rtol=0,
            atol=time_atol,
        )

        # an IoU the data gives, which must be exceeded: that pair drops at
        # every origin, the pairs above it stay
        minimum_iou = float(np.median(at_zero.pairs.iou))
        strict = rd.match_events(windows, kay, minimum_iou=minimum_iou)
        assert 0 < len(strict.pairs) < len(at_zero.pairs)
        assert_matchings_shifted(
            rd.match_events(windows + origin, moved_kay, minimum_iou=minimum_iou), strict
        )

        methods = {"kay": events, "inner": time[inner]}
        moved_methods = {"kay": events + origin, "inner": shifted[inner]}
        comparison = rd.compare_detectors(methods, truth=windows)
        moved_comparison = rd.compare_detectors(moved_methods, truth=windows + origin)
        counts = ["method_a", "method_b", "n_a", "n_b", "n_matched", "n_shared_truth"]
        pd.testing.assert_frame_equal(moved_comparison[counts], comparison[counts])
        assert comparison.onset_error_correlation.notna().all()
        for column in comparison.columns.drop(counts):
            atol = time_atol if "difference" in column else ratio_atol
            np.testing.assert_allclose(
                moved_comparison[column], comparison[column], rtol=0, atol=atol, err_msg=column
            )
        pd.testing.assert_frame_equal(
            rd.consensus_counts(moved_methods, windows + origin),
            rd.consensus_counts(methods, windows),
        )

        # per event, two abutting windows, from its start to its middle sample
        # and on as far again, and an event reaching as far past each end, so
        # it overlaps both by as many samples: a tie, to the earlier window
        middle = (index[:, 0] + index[:, 1]) // 2
        labels = np.array(["early", "late"] * len(events))

        def labelled(clock):
            first = np.column_stack([clock[index[:, 0]], clock[middle]])
            second = np.column_stack([clock[middle], clock[2 * middle - index[:, 0]]])
            bounds = np.stack([first, second], axis=1).reshape(-1, 2)
            return pd.DataFrame(bounds, columns=["start_time", "end_time"]).assign(
                label=labels
            )

        def straddling(clock):
            return np.column_stack(
                [clock[2 * index[:, 0] - middle], clock[2 * middle - index[:, 0]]]
            )

        at_zero_labels = rd.label_by_overlap(straddling(time), labelled(time))
        assert (at_zero_labels == "early").all()
        pd.testing.assert_series_equal(
            rd.label_by_overlap(straddling(shifted), labelled(shifted)), at_zero_labels
        )


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

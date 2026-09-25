"""End-to-end tests: simulated sessions with known ripples through the public API.

Every detector is driven the way a pipeline drives it, by name from the registry,
on the output of ``simulate_session``, and judged against the ground truth.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

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

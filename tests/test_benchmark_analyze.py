"""The benchmark's analyses (examples/benchmark/analyze.py): the paired bootstrap
and sign-flip test, the held-out split and the results size limit; loading, matching
and every analysis on a run written in the runner's schema, with hand-chosen truth
and events, one of its two sessions at a Unix clock origin; the command's files.
No test draws a figure."""

import dataclasses
import functools
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from _synthetic import _non_event_tables, _one_non_event_table

import ripple_detection as rd

UNIX_ORIGIN = 1_700_000_000.0
# A component's window at 10 % of its peak spans its centre plus or minus
# _SPAN_SIGMAS side scales.
_SPAN_SIGMAS = np.sqrt(-2 * np.log(0.1))
KAY = ("Kay_ripple_detector", "default")
KAY_SWEEP = ("Kay_ripple_detector", "3.0")
# a population-burst recipe and a ripple recipe
MALLORY = ("recipe:mallory_2025", "literature")
KARLSSON = ("recipe:karlsson_2009", "literature")


@pytest.fixture(scope="module")
def analyze(benchmark_import):
    return benchmark_import("analyze")


@pytest.fixture(scope="module")
def run(benchmark_import):
    return benchmark_import("run")


def _event_table(run, components, origin=0.0):
    """Latent events built by hand from ``(event_id, event_type, expression,
    component, centre, half_span)`` rows, the half-span that of the
    component's window at 10 % of its peak, centres from ``origin``."""
    rows = [
        {
            "event_id": event_id, "event_type": event_type, "expression": expression,
            "component": component, "center_time": origin + centre,
            "rise_sigma": half / _SPAN_SIGMAS, "decay_sigma": half / _SPAN_SIGMAS,
            "envelope_power": 2, "amplitude": 1.0,
            "frequency_start": 200.0 if expression == "ripple" else np.nan,
            "frequency_end": 200.0 if expression == "ripple" else np.nan,
            "participation": 0.5 if expression == "burst" else np.nan, "n_participants": 0,
        }
        for event_id, event_type, expression, component, centre, half in components
    ]  # fmt: skip
    return pd.DataFrame(rows).astype(run._TABLE_DTYPES["events"])


def _windows(events, expression="ripple"):
    """The truth windows of ``events`` at 10 % of the peak, shape (n, 2)."""
    return rd.truth_windows(events, 0.1, expression)[["start_time", "end_time"]].to_numpy()


def _one_session(events, detected):
    """A 10 s session of ``events``, an EMG burst at 9 s and ``detected``."""
    return {
        "events": events,
        "non_events": _non_event_tables(_one_non_event_table("emg", center_time=9.0)),
        "duration": 10.0,
        "detected": detected,
    }


def _session_row(session_id, condition_id, replicate, spec, event_time):
    """A ``sessions.csv`` row for a hand-built session."""
    types = spec["events"].drop_duplicates("event_id")["event_type"]
    kinds = spec["non_events"]["non_event_type"]
    return {
        "session_id": session_id,
        "condition_id": condition_id,
        "replicate": replicate,
        "seed": replicate,
        "duration_s": spec["duration"],
        "rest_s": spec["duration"],
        "event_time_s": event_time,
        **{f"n_events_{kind}": int((types == kind).sum()) for kind in rd.EVENT_TYPES},
        **{f"n_non_events_{kind}": int((kinds == kind).sum()) for kind in rd.NON_EVENT_TYPES},
        "simulate_s": 0.0,
        "detect_s": 0.0,
    }


def _write_run(run, root, sessions, condition_id="reference"):
    """A finished run of one condition in the runner's schema, combined.

    Each session is a dict: its ``events`` and ``non_events`` tables,
    ``duration`` in seconds, ``detected``, each (method, setting)'s
    ``[start, end]`` rows, None for a call that failed, and optionally
    ``spikes`` (``time``, ``multiunit``, ``unit_types``), from which the
    runner's own counters count the active units of every event and truth
    window. Scores come from the runner's own ``score_events``."""
    outputs = []
    for replicate, spec in enumerate(sessions):
        session_id = f"{condition_id}/{replicate}"
        windows = {
            expression: tuple(np.asarray(f[["start_time", "end_time"]]) for f in frames)
            for expression, frames in run.truth_window_sets(spec["events"]).items()
        }
        event_time = run._interval_union(windows["network"][0])
        minutes_outside = (spec["duration"] - event_time) / 60
        records, events, metrics, failures = [], [], [], []
        for (method, setting), bounds in spec["detected"].items():
            key = {"session_id": session_id, "method": method, "setting": setting}
            records.append(
                {"session_id": session_id, **run.method_records([(method, setting)])[0]}
            )
            if bounds is None:
                failures.append({**key, "error": "ValueError: made to fail"})
                continue
            detected = pd.DataFrame(
                np.reshape(bounds, (-1, 2)), columns=["start_time", "end_time"]
            )
            active = (0, 0)
            if "spikes" in spec:
                active = run.active_counts(np.reshape(bounds, (-1, 2)), spec["spikes"])
            events.append(
                detected.assign(
                    **key,
                    event_index=np.arange(len(detected)),
                    peak_time=detected.mean(axis=1),
                    n_active_units=active[0],
                    n_active_principal=active[1],
                )[list(run.EVENT_COLUMNS)]
            )
            scores = run.score_events(windows, detected, minutes_outside)
            metrics.append(scores.assign(**key)[list(run.METRIC_COLUMNS)])
        outputs.append(
            run.SessionOutput(
                sessions=pd.DataFrame(
                    [_session_row(session_id, condition_id, replicate, spec, event_time)]
                ),
                truth=run._truth(
                    SimpleNamespace(events=spec["events"], non_events=spec["non_events"]),
                    session_id,
                ),
                truth_counts=(
                    run._truth_counts(windows, spec["spikes"], session_id)
                    if "spikes" in spec
                    else pd.DataFrame(columns=list(run.TRUTH_COUNT_COLUMNS))
                ),
                ripple_channels=pd.DataFrame(columns=list(run.RIPPLE_CHANNEL_COLUMNS)),
                units=pd.DataFrame(
                    {
                        "session_id": session_id,
                        "unit": np.arange(len(spec["spikes"].unit_types)),
                        "unit_type": spec["spikes"].unit_types,
                        "baseline_rate": 0.0,
                    }
                    if "spikes" in spec
                    else {},
                    columns=list(run.UNIT_COLUMNS),
                ),
                methods=pd.DataFrame(records, columns=list(run.METHOD_COLUMNS)),
                events=run._concat(events, run.EVENT_COLUMNS),
                metrics=run._concat(metrics, run.METRIC_COLUMNS),
                failures=pd.DataFrame(failures, columns=list(run.FAILURE_COLUMNS)),
                warnings=pd.DataFrame(columns=list(run.WARNING_COLUMNS)),
                results={},
                runtimes={},
            )
        )
    run.write_condition(root / "conditions" / condition_id, outputs)
    factor, level = condition_id.split("=") if "=" in condition_id else ("reference",) * 2
    listed = pd.DataFrame(
        [{"condition_id": condition_id, "factor": factor, "level": level, "params": "{}"}]
    )
    path = root / "conditions.csv"
    if path.exists():
        listed = pd.concat([run.read_table(path), listed], ignore_index=True)
    listed.to_csv(path, index=False)
    return run.combine(root)


def _tiny_session(run, origin, *, mallory_fails=False):
    """One of each event type, a spike-leakage burst and an EMG burst, and
    three methods' events, all from ``origin``: Kay finds three ripples (the
    doublet's two as one event) and has three false positives, over the
    burst-only event, the leakage and nothing; its sweep point finds one
    ripple; Mallory finds three bursts and has two false positives, over the
    EMG and overlapping Kay's last (or fails)."""
    events = _event_table(
        run,
        [
            (0, "swr", "ripple", 0, 2.0, 0.05),
            (0, "swr", "sharp_wave", 0, 2.0, 0.04),
            (0, "swr", "burst", 0, 2.0, 0.06),
            (1, "weak_ripple", "ripple", 0, 4.0, 0.05),
            (1, "weak_ripple", "sharp_wave", 0, 4.0, 0.04),
            (1, "weak_ripple", "burst", 0, 4.0, 0.06),
            (2, "burst_only", "burst", 0, 6.0, 0.1),
            (3, "ripple_doublet", "ripple", 0, 8.0, 0.04),
            (3, "ripple_doublet", "ripple", 1, 8.1, 0.04),
            (3, "ripple_doublet", "sharp_wave", 0, 8.0, 0.03),
            (3, "ripple_doublet", "sharp_wave", 1, 8.1, 0.03),
            (3, "ripple_doublet", "burst", 0, 8.05, 0.1),
            (4, "sharp_wave_only", "sharp_wave", 0, 10.0, 0.04),
        ],
        origin,
    )
    non_events = _non_event_tables(
        _one_non_event_table("spike_leakage", center_time=origin + 12.0),
        _one_non_event_table("emg", center_time=origin + 14.0),
    )
    kay = [
        (1.95, 2.05),
        (3.96, 4.04),
        (5.95, 6.05),
        (7.97, 8.13),
        (11.999, 12.001),
        (16.0, 16.1),
    ]
    mallory = [(1.94, 2.06), (5.9, 6.1), (7.95, 8.15), (13.99, 14.01), (16.05, 16.2)]
    return {
        "events": events,
        "non_events": non_events,
        "duration": 20.0,
        "detected": {
            KAY: origin + np.array(kay),
            KAY_SWEEP: origin + np.array([(1.95, 2.05)]),
            MALLORY: None if mallory_fails else origin + np.array(mallory),
        },
    }


@pytest.fixture(scope="module")
def tiny_run(run, tmp_path_factory):
    """Two sessions of the reference condition, the second at a Unix clock
    origin, where Mallory fails."""
    sessions = [_tiny_session(run, 0.0), _tiny_session(run, UNIX_ORIGIN, mallory_fails=True)]
    return _write_run(run, tmp_path_factory.mktemp("run"), sessions)


@pytest.fixture(scope="module")
def tiny_tables(analyze, tiny_run):
    return analyze.load_run(tiny_run)


def _design_bootstrap(frame, statistic, *, key, n_resamples, seed=0, level=0.95):
    """The paired bootstrap written as plainly as possible: one concatenated
    frame per resample, each drawn value's rows relabelled with its draw."""
    rng = np.random.default_rng(seed)
    values = frame[key].unique()
    groups = dict(tuple(frame.groupby(key)))
    draws = []
    for _ in range(n_resamples):
        pick = rng.choice(values, size=values.size, replace=True)
        resampled = [
            groups[v].assign(
                session_id=groups[v].session_id.astype(str) + f"#{k}",
                replicate=groups[v].replicate.astype(str) + f"#{k}",
            )
            for k, v in enumerate(pick)
        ]
        draws.append(statistic(pd.concat(resampled, ignore_index=True)))
    draws = pd.DataFrame(draws)
    alpha = (1 - level) / 2
    return pd.DataFrame(
        {
            "estimate": statistic(frame),
            "low": draws.quantile(alpha),
            "high": draws.quantile(1 - alpha),
        }
    )


def _per_session(values_a, shift):
    """Two methods on each session, the second ``shift`` above the first."""
    n = len(values_a)
    return pd.DataFrame(
        {
            "session_id": [f"reference/{k}" for k in range(n)] * 2,
            "replicate": list(range(n)) * 2,
            "method": ["a"] * n + ["b"] * n,
            "value": [*values_a, *(v + shift for v in values_a)],
        }
    )


def _mean_by_method(frame):
    return frame.groupby("method")["value"].mean()


def test_paired_bootstrap_by_hand(analyze):
    frame = _per_session([1.0, 4.0, 2.0, 8.0, 5.0, 3.0], shift=10.0)
    got = analyze.paired_bootstrap(frame, _mean_by_method, key="session_id", n_resamples=400)
    # the estimate is the full sample's statistic, inside its interval
    assert got.estimate.tolist() == [23 / 6, 23 / 6 + 10]
    assert (got.low < got.estimate).all()
    assert (got.estimate < got.high).all()
    # every method gets the same draws: b's interval is a's moved by the shift
    np.testing.assert_allclose(
        got.loc["b", ["low", "high"]], got.loc["a", ["low", "high"]] + 10
    )
    # and the draws are the plain algorithm's, value for value
    pd.testing.assert_frame_equal(
        got, _design_bootstrap(frame, _mean_by_method, key="session_id", n_resamples=400)
    )


def test_paired_bootstrap_counts_a_session_drawn_twice_twice(analyze):
    frame = _per_session([1.0, 2.0, 3.0], shift=0.0)
    # sessions per resample, grouped by the relabelled id: always all three draws
    sizes = analyze.paired_bootstrap(
        frame,
        lambda f: pd.Series(
            {"sessions": f.session_id.nunique(), "replicates": f.replicate.nunique()}
        ),
        key="session_id",
        n_resamples=50,
    )
    assert sizes.low.tolist() == [3, 3]
    assert sizes.high.tolist() == [3, 3]


def _mean_difference(frame):
    by_condition = frame.groupby("condition_id")["value"].mean()
    return pd.Series({"difference": by_condition["b"] - by_condition["a"]})


def test_paired_bootstrap_keeps_replicates_paired(analyze):
    values = [3, 9, 4, 1, 7, 6, 2, 8]
    n = len(values)
    frame = pd.DataFrame(
        {
            "session_id": [f"a/{k}" for k in range(n)] + [f"b/{k}" for k in range(n)],
            "condition_id": ["a"] * n + ["b"] * n,
            "replicate": list(range(n)) * 2,
            "value": [*values, *(v + 1 for v in values)],
        }
    )
    paired = analyze.paired_bootstrap(
        frame, _mean_difference, key="replicate", n_resamples=300
    )
    assert paired.loc["difference"].tolist() == [1.0, 1.0, 1.0]
    # drawing sessions instead breaks the pairs, and the interval opens
    unpaired = analyze.paired_bootstrap(
        frame, _mean_difference, key="session_id", n_resamples=300
    )
    assert unpaired.loc["difference", "low"] < 1 < unpaired.loc["difference", "high"]


def test_paired_bootstrap_refuses_a_statistic_indexed_by_its_draws(analyze):
    frame = _per_session([1.0, 2.0, 3.0], shift=0.0)
    # indexed by session: a resample's relabelled ids are not the estimate's
    by_session = functools.partial(
        analyze.paired_bootstrap,
        frame,
        lambda f: f.groupby("session_id")["value"].mean(),
        n_resamples=5,
    )
    with pytest.raises(ValueError, match="index is not the estimate's"):
        by_session(key="session_id")
    with pytest.raises(ValueError, match="key must be 'session_id' or 'replicate'"):
        analyze.paired_bootstrap(frame, _mean_by_method, key="method", n_resamples=5)


@pytest.mark.parametrize(
    ("differences", "p_value"),
    [
        ([1, 1, 1, 1], 2 / 16),
        ([1, -1], 1.0),
        # 16 sessions: every flip, the two of one sign as large as observed
        (np.ones(16), 2 / 2**16),
        # 17 and more: random flips, (k + 1) / (n + 1), never 0 nor the exact
        # 2 / 2 ** 17 (no flip of 999 matches all the signs)
        (np.ones(17), 1 / 1000),
        (np.ones(20), 1 / 1000),
    ],
)
def test_sign_flip_p_values(analyze, differences, p_value):
    assert analyze.sign_flip_test(differences, n_resamples=999) == p_value


def test_sign_flip_needs_finite_pairs(analyze):
    for differences in ([np.nan, 0.0], [np.nan, np.nan], [np.inf, 1.0]):
        with pytest.raises(ValueError, match="finite paired differences"):
            analyze.sign_flip_test(differences)
    assert np.isnan(analyze.sign_flip_test([]))


def test_bootstrap_p_by_hand(analyze):
    draws = np.array(
        [
            [-1.0, np.nan, 0.0, 1.0, np.nan],
            [1.0, -1.0, 1.0, 2.0, np.nan],
            [2.0, 1.0, 2.0, 3.0, np.nan],
            [3.0, 2.0, 3.0, 4.0, np.nan],
            [4.0, 3.0, 4.0, 5.0, np.nan],
        ]
    )
    p, n_draws = analyze.bootstrap_p(draws)
    # 1 of 5 at or below 0; 1 of the 4 defined; a draw at 0 counts on both
    # sides; none at or below 0; no draw defined
    np.testing.assert_array_equal(p[:4], [2 / 5, 2 / 4, 2 / 5, 0.0])
    assert np.isnan(p[4])
    assert n_draws.tolist() == [5, 4, 5, 5, 0]
    # capped at 1: every draw at 0 is on both sides
    assert analyze.bootstrap_p(np.zeros((4, 1)))[0].tolist() == [1.0]


def test_bootstrap_p_below_005_when_the_interval_excludes_0(analyze):
    """The p-value and the 95 % percentile interval come from the same draws,
    so p < 0.05 exactly when the interval excludes 0, but within one draw of
    p = 0.05, where the interpolated bound can fall on either side."""
    rng = np.random.default_rng(4)
    draws = rng.normal(size=(2000, 1)) + np.linspace(-4, 4, 401)
    draws[rng.random(draws.shape) < 0.01] = np.nan
    p, n_draws = analyze.bootstrap_p(draws)
    low, high = analyze.percentile_intervals(draws)
    excludes = (low > 0) | (high < 0)
    boundary = np.abs(p - 0.05) <= 2 / n_draws
    assert boundary.sum() <= 3
    np.testing.assert_array_equal((p < 0.05)[~boundary], excludes[~boundary])
    assert excludes.any()
    assert (~excludes).any()
    # at the boundary: 50 of 2000 draws at or below 0, the 2.5 % quantile
    # interpolated between the 50th (0) and the 51st (1)
    edge = np.concatenate([np.zeros(50), np.ones(1950)])[:, np.newaxis]
    assert analyze.bootstrap_p(edge)[0].tolist() == [0.05]
    assert analyze.percentile_intervals(edge)[0][0] > 0


def test_choose_setting_boundary_and_ties(analyze):
    # a setting exactly at the target is allowed
    assert analyze.choose_setting([0.5, 0.8], [0.5, 1.0], 1.0) == 1
    # of equal recalls, the first in threshold order
    assert analyze.choose_setting([0.8, 0.8, 0.6], [1.0, 0.5, 0.2], 1.0) == 0
    assert analyze.choose_setting([0.8, 0.9], [1.5, 2.0], 1.0) is None


def test_held_out_membership_is_the_same_in_every_condition(analyze):
    # by replicate id alone: the odd ones, of the reference's 20 replicates
    # and of another condition's 10
    assert {k for k in range(20) if analyze.is_held_out(k)} == set(range(1, 20, 2))
    assert {k for k in range(10) if analyze.is_held_out(k)} == {1, 3, 5, 7, 9}


def test_results_size_limit(analyze, tmp_path):
    path = tmp_path / "table.csv"
    with pytest.raises(ValueError, match="over the 1,000,000-byte limit"):
        analyze.write_result(path, b"x" * (analyze.SIZE_LIMIT + 1))
    assert not path.exists()
    analyze.write_result(path, b"x" * analyze.SIZE_LIMIT)
    assert path.stat().st_size == analyze.SIZE_LIMIT


def _pairs(frame, columns=("method", "setting")):
    return set(frame[list(columns)].itertuples(index=False, name=None))


def test_main_analyses_include_recipes(analyze, run, tiny_run, tiny_tables):
    main = {KAY, MALLORY}
    assert _pairs(tiny_tables.methods) == main
    for table in (tiny_tables.events, tiny_tables.ran):
        assert _pairs(table) == main
    # the selection itself, on every row the run wrote
    written = run.read_table(tiny_run / "methods.csv")
    assert _pairs(written) == {KAY, KAY_SWEEP, MALLORY}
    assert _pairs(analyze.main_rows(written)) == main
    everything = analyze.load_run(tiny_run, settings=None)
    assert _pairs(everything.methods) == {KAY, KAY_SWEEP, MALLORY}


def test_a_missing_result_is_a_failure(analyze, tiny_tables):
    assert tiny_tables.failures.to_dict("records") == [
        {
            "session_id": "reference/1",
            "method": MALLORY[0],
            "setting": MALLORY[1],
            "error": "ValueError: made to fail",
        }
    ]
    counts = analyze.failure_counts(tiny_tables).set_index("method")
    assert counts.loc[KAY[0], ["n_sessions", "n_failures", "error"]].tolist() == [2, 0, ""]
    assert counts.loc[MALLORY[0], ["n_sessions", "n_failures"]].tolist() == [1, 1]


def test_scores_lost_without_a_failure_row_are_a_failure(analyze, run, tmp_path):
    combined = _write_run(run, tmp_path, [_tiny_session(run, 0.0)])
    # one call's scores lost, and nothing recorded: still a failure, not zero events
    metrics = run.read_table(combined / "metrics.csv.gz")
    metrics[metrics.method != MALLORY[0]].to_csv(combined / "metrics.csv.gz", index=False)
    failures = analyze.load_run(combined).failures
    assert failures[["method", "error"]].to_dict("records") == [
        {"method": MALLORY[0], "error": ""}
    ]


def test_loading_reads_back_what_the_run_wrote(analyze, run, tiny_run, tiny_tables):
    expected = run.load_truth(tiny_run / "truth.csv.gz")
    assert list(tiny_tables.truth) == ["reference/0", "reference/1"]
    for session_id, (events, non_events) in tiny_tables.truth.items():
        pd.testing.assert_frame_equal(events, expected[session_id][0])
        pd.testing.assert_frame_equal(non_events, expected[session_id][1])
    # read in chunks, the events equal the whole table's main rows, and what
    # ran the scores' main rows
    events = analyze.main_rows(run.read_table(tiny_run / "events.csv.gz"))
    pd.testing.assert_frame_equal(tiny_tables.events, events.reset_index(drop=True))
    metrics = analyze.main_rows(run.read_table(tiny_run / "metrics.csv.gz"))
    ran = metrics[["session_id", "method", "setting"]].drop_duplicates()
    pd.testing.assert_frame_equal(tiny_tables.ran, ran.reset_index(drop=True))
    assert tiny_tables.events.end_time.max() > UNIX_ORIGIN


def test_the_reference_analyses_take_reference_main_tables(
    analyze, tiny_tables, two_condition_run
):
    assert (tiny_tables.conditions, tiny_tables.settings) == (
        ("reference",),
        ("default", "literature"),
    )
    everything = analyze.load_run(two_condition_run, conditions=None, settings=None)
    assert everything.conditions == ("reference", "spike_model=refractory")
    assert everything.settings is None
    # the analyses of the reference refuse any other selection
    for tables in (
        everything,
        analyze.load_run(two_condition_run, settings=None),
        analyze.load_run(two_condition_run, conditions=["spike_model=refractory"]),
    ):
        with pytest.raises(ValueError, match="the reference condition's main settings"):
            analyze.Inputs(tables, None, None, None)
    analyze.Inputs(tiny_tables, None, None, None)


def test_loading_an_unknown_condition_raises(analyze, tiny_run):
    with pytest.raises(ValueError, match=r"no session of the conditions \['ripple_snr=low'\]"):
        analyze.load_run(tiny_run, conditions=["reference", "ripple_snr=low"])


@pytest.fixture(scope="module")
def tiny_matches(analyze, tiny_tables):
    return analyze.match_run(tiny_tables)


def test_matching_again_gives_the_runner_scores(analyze, run, tiny_run, tiny_matches):
    pairs = tiny_matches.pairs
    reference = tiny_matches.windows.groupby(["session_id", "expression"]).size()
    metrics = analyze.main_rows(run.read_table(tiny_run / "metrics.csv.gz"))
    metrics = metrics[metrics.minimum_iou == 0]
    for row in metrics.itertuples():
        found = pairs[
            (pairs.session_id == row.session_id)
            & (pairs.method == row.method)
            & (pairs.expression == row.expression)
        ]
        assert len(found) == row.n_matched
        assert reference[row.session_id, row.expression] == row.n_reference
        for column in analyze.ERROR_COLUMNS:
            median = found[column].median() if len(found) else np.nan
            assert median == pytest.approx(getattr(row, f"median_{column}"), nan_ok=True)
    assert len(metrics) == 3 * 4  # Kay twice, Mallory once; four expressions


def test_false_positive_labels(tiny_matches):
    labels = tiny_matches.false_positives.groupby(["session_id", "method"])["label"]
    assert labels.apply(sorted).to_dict() == {
        ("reference/0", KAY[0]): ["background", "burst_only:burst", "spike_leakage"],
        ("reference/0", MALLORY[0]): ["background", "emg"],
        ("reference/1", KAY[0]): ["background", "burst_only:burst", "spike_leakage"],
    }


def test_splits_and_merges_count_the_doublet(tiny_matches):
    kay = tiny_matches.overlaps[tiny_matches.overlaps.method == KAY[0]]
    counts = kay.groupby("subset")[["n_truth", "n_split", "n_detected", "n_merged"]].sum()
    # the doublet's two ripples found as one event, in each session
    assert counts.loc["all"].tolist() == [8, 0, 12, 2]
    assert counts.loc["ripple_doublet"].tolist() == [4, 0, 2, 2]


def test_an_event_over_one_doublet_ripple_is_not_merged(analyze, run):
    session = _tiny_session(run, 0.0)
    # one event over the doublet's first ripple alone, one over both
    events = pd.DataFrame(
        {
            "method": KAY[0],
            "setting": KAY[1],
            "event_index": [0, 1],
            "start_time": [7.97, 7.97],
            "end_time": [8.03, 8.13],
        }
    )
    for bounds, merged in (([0], 0), ([1], 1)):
        matches = analyze.match_session(
            "reference/0",
            events.iloc[bounds],
            (session["events"], session["non_events"]),
            [KAY],
            {KAY: "ripple"},
        )
        doublet = matches.overlaps.set_index("subset").loc["ripple_doublet"]
        assert [doublet.n_truth, doublet.n_detected, doublet.n_merged] == [2, 1, merged]


def test_touching_false_positives_are_separate_groups(analyze):
    bounds = np.array([[0.0, 1.0], [1.0, 2.0], [3.0, 4.0], [3.5, 5.0], [4.0, 4.5]])
    # sharing a bound is not overlapping; overlapping by any length joins
    assert analyze._connected_groups(bounds).tolist() == [0, 1, 2, 2, 2]


def test_matching_in_parallel_matches_in_order(analyze, tiny_tables, tiny_matches):
    parallel = analyze.match_run(tiny_tables, workers=2)
    for name in analyze._MATCH_COLUMNS:
        pd.testing.assert_frame_equal(getattr(parallel, name), getattr(tiny_matches, name))


# Few resamples: two sessions have at most three distinct draws.
FEW = 50


def _by(table, *columns):
    return table.set_index(list(columns))


def test_profile_and_consensus_on_tiny_run(analyze, tiny_tables, tiny_matches):
    profile = _by(
        analyze.detection_profile(tiny_tables, tiny_matches, n_resamples=FEW),
        "method",
        "event_type",
    )
    kinds = list(rd.EVENT_TYPES)
    # Kay, both sessions: every type but the sharp wave alone, the burst-only
    # event by an event over its burst
    kay = profile.loc[KAY[0]].loc[kinds]
    assert kay.n_true.tolist() == [2, 2, 2, 2, 2]
    assert kay.n_found.tolist() == [2, 2, 2, 2, 0]
    assert kay.recall.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]
    assert (kay.recall_low == kay.recall).all()
    assert (kay.recall_high == kay.recall).all()
    # Mallory, the first session only: it failed on the second
    mallory = profile.loc[MALLORY[0]].loc[kinds]
    assert mallory.n_true.tolist() == [1, 1, 1, 1, 1]
    assert mallory.recall.tolist() == [1.0, 0.0, 1.0, 1.0, 0.0]
    assert mallory[["n_sessions", "n_failures"]].drop_duplicates().to_numpy().tolist() == [
        [1, 1]
    ]
    assert (kay[["n_sessions", "n_failures"]].to_numpy() == [2, 0]).all()

    table = analyze.consensus(tiny_tables, tiny_matches)
    counts = {
        (row.kind, row.event_type, row.n_methods): row.count for row in table.itertuples()
    }
    assert counts == {
        ("true_event", "swr", 1): 1,
        ("true_event", "swr", 2): 1,
        ("true_event", "weak_ripple", 1): 2,
        ("true_event", "burst_only", 1): 1,
        ("true_event", "burst_only", 2): 1,
        ("true_event", "ripple_doublet", 1): 1,
        ("true_event", "ripple_doublet", 2): 1,
        ("true_event", "sharp_wave_only", 0): 2,
        # Kay's three false positives in each session, Mallory's over the EMG,
        # and the two overlapping ones as one group
        ("false_positive_group", "all", 1): 6,
        ("false_positive_group", "all", 2): 1,
    }
    assert list(table.event_type.drop_duplicates()) == [*kinds, "all"]
    assert table.fraction.tolist()[-2:] == [6 / 7, 1 / 7]
    assert table[
        ["n_methods_compared", "n_failed_calls"]
    ].drop_duplicates().to_numpy().tolist() == [[2, 1]]


def test_false_positive_classes_on_tiny_run(analyze, tiny_tables, tiny_matches):
    classes = analyze.false_positive_classes(tiny_tables, tiny_matches, n_resamples=FEW)
    # every label for every method, zeros included
    assert len(classes) == 2 * classes.label.nunique()
    assert classes.label.iloc[-1] == "background"
    shown = _by(classes[classes.n_events > 0], "method", "label")
    assert shown.fraction.to_dict() == {
        (KAY[0], "burst_only:burst"): 1 / 3,
        (KAY[0], "spike_leakage"): 1 / 3,
        (KAY[0], "background"): 1 / 3,
        (MALLORY[0], "emg"): 1 / 2,
        (MALLORY[0], "background"): 1 / 2,
    }
    assert shown.n_unmatched.to_dict()[KAY[0], "background"] == 6


def test_splits_and_merges_on_tiny_run(analyze, tiny_tables, tiny_matches):
    rates = _by(
        analyze.splits_and_merges(tiny_tables, tiny_matches, n_resamples=FEW),
        "method",
        "subset",
    )
    assert rates.loc[(KAY[0], "all"), ["split_rate", "merge_rate"]].tolist() == [0.0, 1 / 6]
    assert rates.loc[(KAY[0], "ripple_doublet"), ["n_detected", "merge_rate"]].tolist() == [
        2,
        1,
    ]
    assert rates.loc[
        (MALLORY[0], "all"), ["n_truth", "n_detected", "n_failures"]
    ].tolist() == [
        4,
        5,
        1,
    ]


def test_agreement_and_differences_on_tiny_run(analyze, tiny_tables, tiny_matches):
    agreement = analyze.pairwise_agreement(tiny_tables, tiny_matches, n_resamples=FEW)
    row = agreement.iloc[0]
    assert (row.method_a, row.method_b, row.truth_expression) == (
        KAY[0],
        MALLORY[0],
        "network",
    )
    # the one session both ran: 4 of 6 and 5 events matched; 3 of the 4 network
    # events either found were found by both
    assert row.jaccard == pytest.approx(4 / 7)
    assert row.jaccard_truth_ids == pytest.approx(3 / 4)
    assert [row.n_sessions, row.n_failures_a, row.n_failures_b] == [1, 0, 1]
    differences = analyze.method_differences(tiny_tables, tiny_matches, n_resamples=FEW)
    # Kay's starts minus Mallory's over the matched events: 0.01, 0.05, 0.02, -0.05
    assert differences.median_onset_difference.iloc[0] == pytest.approx(0.015)
    assert differences.median_onset_difference_p.iloc[0] == 1.0  # one session
    dendrogram = analyze.agreement_dendrogram(tiny_tables, tiny_matches)
    assert dendrogram.method.tolist() == [KAY[0], MALLORY[0], ""]
    assert dendrogram.distance.iloc[-1] == pytest.approx(3 / 7)


def test_pair_tables_count_the_sessions_each_value_pools(analyze, tiny_tables, tiny_matches):
    # both sessions compare Kay and Mallory; the second's correlation and
    # onset difference are undefined
    network = tiny_matches.comparisons[tiny_matches.comparisons.truth_expression == "network"]
    network = pd.concat([network.iloc[:1]] * 2, ignore_index=True).assign(
        session_id=["reference/0", "reference/1"],
        onset_error_correlation=[0.5, np.nan],
        offset_error_correlation=[0.25, -0.25],
        median_onset_difference=[0.01, np.nan],
    )
    matches = dataclasses.replace(tiny_matches, comparisons=network)
    correlations = analyze.error_correlations(tiny_tables, matches, n_resamples=FEW).iloc[0]
    assert correlations.n_sessions == 2
    assert (
        correlations.onset_error_correlation,
        correlations.onset_error_correlation_n_sessions,
    ) == (0.5, 1)
    assert correlations.offset_error_correlation_n_sessions == 2
    differences = analyze.method_differences(tiny_tables, matches, n_resamples=FEW).iloc[0]
    assert differences.median_onset_difference_n_sessions == 1
    # the test runs on the one finite session and says the other was dropped
    assert (
        differences.median_onset_difference_p,
        differences.median_onset_difference_n_dropped,
    ) == (1.0, 1)
    assert differences.median_offset_difference_n_dropped == 0


def test_overlap_quality_on_tiny_run(analyze, tiny_tables, tiny_matches):
    quality = _by(
        analyze.overlap_quality(tiny_tables, tiny_matches, n_resamples=FEW),
        "method",
        "measure",
    )
    # Kay's IoUs, in each session: 1, 0.8 and 7/17 (the doublet's ripple)
    assert quality.loc[(KAY[0], "iou"), "n_pairs"] == 6
    assert quality.loc[(KAY[0], "iou"), "median"] == pytest.approx(0.8, abs=1e-5)
    assert quality.loc[(KAY[0], "iou"), "q05"] == pytest.approx(7 / 17, abs=1e-5)
    assert quality.loc[(KAY[0], "iou"), "recall"] == 3 / 4
    assert quality.loc[(MALLORY[0], "coverage"), ["n_pairs", "median"]].tolist() == [3, 1.0]


@pytest.fixture(scope="module")
def timing_run(run, tmp_path_factory):
    """One session of six ripples: Kay finds the first three at their bounds;
    Karlsson finds all six, the last three 20 ms late at both ends."""
    events = _event_table(run, [(k, "swr", "ripple", 0, 1.0 + k, 0.05) for k in range(6)])
    windows = _windows(events)
    late = windows + np.array([[0.0], [0.0], [0.0], [0.02], [0.02], [0.02]])
    session = _one_session(events, {KAY: windows[:3], KARLSSON: late})
    return _write_run(run, tmp_path_factory.mktemp("timing"), [session])


CAREY = ("Carey_candidate_detector", "default")


@pytest.fixture(scope="module")
def network_run(run, tmp_path_factory):
    """Three swr events (ripple and burst about one centre); Carey's events
    are their network windows, starting 30 ms early and 10 and 20 ms late."""
    components = []
    for k in range(3):
        components += [
            (k, "swr", "ripple", 0, 1.0 + k, 0.05),
            (k, "swr", "burst", 0, 1.0 + k, 0.06),
        ]
    events = _event_table(run, components)
    network = _windows(events, "network")
    onsets = np.array([-0.03, 0.01, 0.02])
    session = _one_session(events, {CAREY: network + np.column_stack([onsets, np.zeros(3)])})
    return _write_run(run, tmp_path_factory.mktemp("network"), [session]), events


def test_boundary_errors_of_a_network_method(analyze, network_run):
    root, events = network_run
    tables = analyze.load_run(root)
    errors = analyze.boundary_errors(tables, analyze.match_run(tables), n_resamples=FEW)
    carey = errors[errors.method == CAREY[0]]
    # its primary expression, and the ripple and the burst windows as well
    assert carey.expression.drop_duplicates().tolist() == ["network", "ripple", "burst"]
    onset = _by(
        carey[(carey.boundary == "onset") & (carey.expression == "network")],
        "fraction",
        "measure",
    )
    assert onset.loc[(0.1, "signed"), "median"] == pytest.approx(0.01)
    # absolute: the median of 30, 10 and 20 ms
    assert onset.loc[(0.1, "absolute"), "median"] == pytest.approx(0.02)
    # at 25 % of the peak each error is against that fraction's window
    starts = {
        fraction: rd.truth_windows(events, fraction, "network").start_time.to_numpy()
        for fraction in (0.1, 0.25)
    }
    later = np.median(starts[0.1] + np.array([-0.03, 0.01, 0.02]) - starts[0.25])
    assert onset.loc[(0.25, "signed"), "median"] == pytest.approx(later)
    assert (carey.n_pairs == 3).all()


def test_paired_timing_uses_shared_truth_only(analyze, timing_run):
    tables = analyze.load_run(timing_run)
    matches = analyze.match_run(tables)
    errors = analyze.boundary_errors(tables, matches, n_resamples=FEW)
    onset = _by(
        errors[
            (errors.fraction == 0.1)
            & (errors.boundary == "onset")
            & (errors.measure == "signed")
        ],
        "method",
    )
    assert onset.loc[KAY[0], ["n_pairs", "median", "recall"]].tolist() == [3, 0.0, 0.5]
    assert onset.loc[KARLSSON[0], ["n_pairs", "recall"]].tolist() == [6, 1.0]
    assert onset.loc[KARLSSON[0], "median"] == pytest.approx(0.01, abs=1e-12)

    timing = analyze.paired_timing(tables, matches, "ripple", n_resamples=FEW)
    assert timing.fraction.tolist() == [0.1, 0.25, 0.5]
    row = timing.iloc[0]
    assert (row.method_a, row.method_b) == (KAY[0], KARLSSON[0])
    assert [row.n_shared, row.n_sessions, row.n_sessions_without] == [3, 1, 0]
    assert row.jaccard_truth_ids == 0.5
    for stem in ("onset_signed", "onset_absolute", "offset_signed", "offset_absolute"):
        assert row[f"{stem}_pooled"] == 0.0
        assert row[f"{stem}_estimate"] == 0.0
        assert row[f"{stem}_p"] == 1.0
    # no pair shares a burst primary expression here
    assert analyze.paired_timing(tables, matches, "burst", n_resamples=FEW).empty


@pytest.fixture(scope="module")
def correlated_run(run, tmp_path_factory):
    """Four swr events whose bursts start 30 to 55 ms before their ripples
    and end 30 to 5 ms after them. Against the ripple windows Kay's onset
    errors rise (0 to 3 ms) where Karlsson's fall, and both methods' offset
    errors rise alike; against the network windows the bursts' lead, the
    same for both, makes their onset errors rise together."""
    lead = [0.0, 0.01, 0.02, 0.025]
    components = []
    for k, shift in enumerate(lead):
        components += [
            (k, "swr", "ripple", 0, 1.0 + k, 0.05),
            (k, "swr", "burst", 0, 1.0 + k - shift, 0.08),
        ]
    events = _event_table(run, components)
    windows = _windows(events)
    rising = np.array([0.0, 0.001, 0.002, 0.003])
    kay = windows + np.column_stack([rising, rising])
    karlsson = windows + np.column_stack([rising[::-1], rising])
    session = _one_session(events, {KAY: kay, KARLSSON: karlsson})
    return _write_run(run, tmp_path_factory.mktemp("correlated"), [session])


def test_error_correlations_use_the_shared_primary_expression(analyze, correlated_run):
    tables = analyze.load_run(correlated_run)
    matches = analyze.match_run(tables)
    row = analyze.error_correlations(tables, matches, n_resamples=FEW).iloc[0]
    assert (row.method_a, row.method_b, row.truth_expression) == (
        KAY[0],
        KARLSSON[0],
        "ripple",
    )
    # onset and offset each read from their own errors
    assert (row.onset_error_correlation, row.offset_error_correlation) == (-1.0, 1.0)
    # beside them, the same against the network truth
    assert (row.network_onset_error_correlation, row.network_offset_error_correlation) == (
        1.0,
        1.0,
    )


def _shifted_session(run, onsets, *, found=None):
    """Ripples 1 s apart, one per entry of ``onsets``: Kay's and Karlsson's
    onset errors on each, (kay, karlsson) in seconds, their offsets exact;
    ``found`` gives each method's ripples (default every one)."""
    events = _event_table(
        run, [(k, "swr", "ripple", 0, 1.0 + k, 0.05) for k in range(len(onsets))]
    )
    windows = _windows(events)
    shifts = np.array(onsets, dtype=float).reshape(-1, 2)
    kay, karlsson = found or (range(len(onsets)), range(len(onsets)))
    return _one_session(
        events,
        {
            KAY: (windows + np.column_stack([shifts[:, 0], np.zeros(len(shifts))]))[list(kay)],
            KARLSSON: (windows + np.column_stack([shifts[:, 1], np.zeros(len(shifts))]))[
                list(karlsson)
            ],
        },
    )


@pytest.fixture(scope="module")
def offset_timing_run(run, tmp_path_factory):
    """Kay starts 5 ms early and Karlsson 10 ms on three ripples of the first
    session, 5 and 20 ms on the second's one; in the third both run but find
    different ripples."""
    sessions = [
        _shifted_session(run, [(-0.005, -0.010)] * 3),
        _shifted_session(run, [(-0.005, -0.020)]),
        _shifted_session(run, [(0.0, 0.0)] * 2, found=([0], [1])),
    ]
    return _write_run(run, tmp_path_factory.mktemp("offsets"), sessions)


def test_paired_timing_differences_by_hand(analyze, offset_timing_run):
    tables = analyze.load_run(offset_timing_run)
    row = analyze.paired_timing(
        tables, analyze.match_run(tables), "ripple", n_resamples=FEW
    ).iloc[0]
    assert (row.method_a, row.method_b) == (KAY[0], KARLSSON[0])
    assert [row.n_shared, row.n_sessions, row.n_sessions_without] == [4, 2, 1]
    # A minus B, signed: Kay starts 5 ms later on three events, 15 ms on one
    assert row.onset_signed_pooled == pytest.approx(0.005)
    # the estimate is the mean of the two sessions' medians, not the pooled median
    assert row.onset_signed_estimate == pytest.approx(0.010)
    # absolute: Kay is closer to the truth, so |A| - |B| is negative
    assert row.onset_absolute_pooled == pytest.approx(-0.005)
    assert row.onset_absolute_estimate == pytest.approx(-0.010)
    # two sessions of one sign: half of the four sign flips are as large
    assert row.onset_signed_p == 0.5
    assert row.offset_signed_estimate == row.offset_absolute_estimate == 0.0


@pytest.fixture(scope="module")
def difference_run(run, tmp_path_factory):
    """Five sessions of one ripple: Kay starts 4 ms after Karlsson in four,
    4 ms before it in the fifth."""
    onsets = [(0.004, 0.0)] * 4 + [(-0.004, 0.0)]
    sessions = [_shifted_session(run, [onset]) for onset in onsets]
    return _write_run(run, tmp_path_factory.mktemp("differences"), sessions)


def test_method_differences_test_the_signed_session_medians(analyze, difference_run):
    tables = analyze.load_run(difference_run)
    row = analyze.method_differences(tables, analyze.match_run(tables), n_resamples=FEW).iloc[
        0
    ]
    assert (row.method_a, row.method_b, row.n_sessions) == (KAY[0], KARLSSON[0], 5)
    # A minus B: positive, Kay later, on the mean of 4, 4, 4, 4 and -4 ms
    assert row.median_onset_difference == pytest.approx(0.0024)
    assert row.fraction_a_earlier_onset == pytest.approx(0.2)
    # of the 32 sign flips, 12 give a mean at least as large: the flips of
    # all five (2) and those leaving four against one (10)
    assert row.median_onset_difference_p == pytest.approx(12 / 32)
    assert row.median_offset_difference == 0.0


def test_average_linkage_is_not_single_linkage(analyze):
    jaccard = pd.Series({("a", "b"): 0.8, ("a", "c"): 0.4, ("b", "c"): 0.1})
    tree = analyze.agreement_linkage(["a", "b", "c"], jaccard)
    # a and b join at 0.2; c joins them at the mean of 0.6 and 0.9, where
    # single linkage would give 0.6 and complete 0.9
    assert tree[:, 2].tolist() == pytest.approx([0.2, 0.75])
    assert tree[:, 3].tolist() == [2, 3]


DAVIDSON = ("recipe:davidson_2009_ripples", "literature")
LEE = ("recipe:lee_2002", "literature")


def test_point_inventories_come_from_the_catalog(analyze):
    from ripple_detection.literature_methods import list_methods

    outputs = list_methods().set_index("name")["output"]
    expected = {
        f"recipe:{config.config_id}"
        for config in analyze.RECIPES
        if outputs[config.method] == "ripple peaks"
    }
    assert analyze.point_methods() == expected
    assert {DAVIDSON[0], "recipe:wu_2014_ripples"} <= expected
    assert analyze.scoring_rule(DAVIDSON[0]) == "peak_containment"
    # single-sample events are intervals all the same
    assert analyze.scoring_rule(LEE[0]) == "interval"
    assert analyze.scoring_rule(KAY[0]) == "interval"


@pytest.mark.parametrize("origin", [0.0, UNIX_ORIGIN])
def test_match_peaks_by_hand(analyze, origin):
    windows = origin + np.array([[1.0, 1.1], [1.05, 1.2], [3.0, 3.1], [5.0, 5.1]])
    peaks = origin + np.array([1.08, 1.09, 3.1, 4.0, 5.1 + 1e-5])
    pairs = analyze.match_peaks(windows, peaks)
    # the first two windows share both points, one each; the third's end is
    # closed; a point outside every window, and one past the last's end, match none
    assert pairs.tolist() == [[0, 0], [1, 1], [2, 2]]
    # a point within the timestamps' rounding of a bound is on it
    on_bound = np.nextafter(np.nextafter(windows[3, 1], np.inf), np.inf)
    assert analyze.match_peaks(windows[3:], [on_bound]).tolist() == [[0, 0]]
    assert analyze.match_peaks(np.empty((0, 2)), peaks).shape == (0, 2)
    assert analyze.match_peaks(windows, []).shape == (0, 2)


def _largest_matching(windows, peaks):
    """The most window-point pairs, by trying every assignment."""
    if not len(windows):
        return 0
    (start, end), rest = windows[0], windows[1:]
    best = _largest_matching(rest, peaks)
    for position, peak in enumerate(peaks):
        if start <= peak <= end:
            others = peaks[:position] + peaks[position + 1 :]
            best = max(best, 1 + _largest_matching(rest, others))
    return best


def test_match_peaks_is_the_largest_matching(analyze):
    rng = np.random.default_rng(0)
    for _ in range(300):
        starts = rng.integers(0, 8, size=rng.integers(0, 6)).astype(float)
        windows = np.column_stack([starts, starts + rng.integers(0, 4, size=starts.size)])
        peaks = rng.integers(0, 10, size=rng.integers(0, 6)).astype(float)
        pairs = analyze.match_peaks(windows, peaks)
        assert len(pairs) == _largest_matching(windows.tolist(), peaks.tolist())
        # one to one, each point inside its window
        assert len(set(pairs[:, 0])) == len(set(pairs[:, 1])) == len(pairs)
        for row, position in pairs:
            assert windows[row, 0] <= peaks[position] <= windows[row, 1]


def _point_session(run, origin):
    """The tiny session with Kay, Davidson's ripple peaks and Lee's
    single-sample events, all from ``origin``: Davidson's points lie on the
    first ripple's peak, on the second's end (closed), twice inside the
    doublet's second ripple, and on nothing."""
    session = _tiny_session(run, origin)
    windows = rd.truth_windows(session["events"], 0.1, "ripple")
    second_end = windows["end_time"].iloc[1]
    peaks = np.array([origin + 2.0, second_end, origin + 8.1, origin + 8.12, origin + 12.0])
    session["detected"] = {
        KAY: session["detected"][KAY],
        DAVIDSON: np.column_stack([peaks, peaks]),
        LEE: origin + np.array([(6.0, 6.0), (2.0, 2.0)]),
    }
    return session


@pytest.fixture(scope="module")
def point_run(run, tmp_path_factory):
    sessions = [_point_session(run, 0.0), _point_session(run, UNIX_ORIGIN)]
    return _write_run(run, tmp_path_factory.mktemp("points"), sessions)


@pytest.fixture(scope="module")
def point_matched(analyze, point_run):
    """The point run's tables and their matches."""
    tables = analyze.load_run(point_run)
    return tables, analyze.match_run(tables)


def test_point_inventories_are_scored_apart(analyze, point_matched):
    tables, matches = point_matched
    assert tables.methods.set_index("method")["scoring"].to_dict() == {
        DAVIDSON[0]: "peak_containment",
        KAY[0]: "interval",
        LEE[0]: "interval",
    }
    table = analyze.point_inventories(tables, matches, n_resamples=FEW)
    row = table.iloc[0]
    assert len(table) == 1
    assert (row.method, row.scoring) == (DAVIDSON[0], "peak_containment")
    # in each session: four ripple windows, five points, three matched
    assert [row.n_reference, row.n_detected, row.n_matched] == [8, 10, 6]
    assert [row.recall, row.precision] == [0.75, 0.6]
    minutes = ((20.0 - tables.sessions.event_time_s) / 60).sum()
    assert row.false_positives_per_minute == pytest.approx(4 / minutes)
    assert row.recall_low == row.recall_high == 0.75
    # an interval rule never sees Davidson; Lee's single samples it does
    interval_tables = [
        analyze.detection_profile(tables, matches, n_resamples=FEW),
        analyze.false_positive_classes(tables, matches, n_resamples=FEW),
        analyze.overlap_quality(tables, matches, n_resamples=FEW),
        analyze.boundary_errors(tables, matches, n_resamples=FEW),
        analyze.splits_and_merges(tables, matches, n_resamples=FEW),
        matches.pairs,
        matches.false_positives,
    ]
    for frame in interval_tables:
        assert DAVIDSON[0] not in set(frame.method)
    assert LEE[0] in set(interval_tables[0].method)
    agreement = analyze.pairwise_agreement(tables, matches, n_resamples=FEW)
    assert set(agreement.method_a) | set(agreement.method_b) == {KAY[0], LEE[0]}
    assert set(analyze.agreement_dendrogram(tables, matches).method) == {KAY[0], LEE[0], ""}
    assert analyze.consensus(tables, matches).n_methods_compared.unique().tolist() == [2]
    counts = analyze.failure_counts(tables).set_index("method")
    assert counts.loc[DAVIDSON[0], "scoring"] == "peak_containment"


@pytest.fixture(scope="module")
def failing_run(run, tmp_path_factory):
    """The tiny sessions with Kay and Mallory running and Karlsson's recipe
    and Davidson's ripple peaks failing on every session."""
    sessions = [_tiny_session(run, 0.0), _tiny_session(run, UNIX_ORIGIN)]
    for session in sessions:
        session["detected"] = {
            KAY: session["detected"][KAY],
            MALLORY: session["detected"][MALLORY],
            KARLSSON: None,
            DAVIDSON: None,
        }
    return _write_run(run, tmp_path_factory.mktemp("failing"), sessions)


def test_a_method_that_never_ran_keeps_its_rows(analyze, failing_run):
    tables = analyze.load_run(failing_run)
    matches = analyze.match_run(tables, levels=(0.0, 0.2, 0.5))
    quick = {"n_resamples": FEW}
    bouts = {"reference/0": np.array([[10.0, 15.0]])}
    bouts["reference/1"] = bouts["reference/0"] + UNIX_ORIGIN
    per_method = {
        "detection_profile": analyze.detection_profile(tables, matches, **quick),
        "false_positive_classes": analyze.false_positive_classes(tables, matches, **quick),
        "splits_and_merges": analyze.splits_and_merges(tables, matches, **quick),
        "overlap_quality": analyze.overlap_quality(tables, matches, **quick),
        "boundary_errors": analyze.boundary_errors(tables, matches, **quick),
        "participation_bias": analyze.participation_bias(tables, matches, **quick),
        "boundary_effect": analyze.boundary_effect(tables, matches, **quick),
        "matching_sensitivity": analyze.matching_sensitivity(tables, matches, **quick),
        "rates_by_state": analyze.rates_by_state(tables, bouts=bouts, **quick),
        "point_inventories": analyze.point_inventories(tables, matches, **quick),
    }
    measured = {
        "detection_profile": "recall",
        "false_positive_classes": "fraction",
        "splits_and_merges": "split_rate",
        "overlap_quality": "median",
        "boundary_errors": "median",
        "participation_bias": "ratio_of_means",
        "boundary_effect": "mean_difference",
        "matching_sensitivity": "recall",
        "rates_by_state": "rate",
        "point_inventories": "recall",
    }
    for name, table in per_method.items():
        failed = (
            DAVIDSON[0] if name in ("rates_by_state", "point_inventories") else KARLSSON[0]
        )
        rows = table[table.method == failed]
        # every row a method that ran has, each counting its failures
        ran = table[table.method == KAY[0]] if name != "point_inventories" else rows
        assert len(rows) == len(ran) > 0, name
        assert (rows.n_failures == 2).all(), name
        assert (rows.n_sessions == 0).all(), name
        assert rows[measured[name]].isna().all(), name
    pairs = {
        "pairwise_agreement": analyze.pairwise_agreement(tables, matches, **quick),
        "method_differences": analyze.method_differences(tables, matches, **quick),
        "error_correlations": analyze.error_correlations(tables, matches, **quick),
        "paired_timing_ripple": analyze.paired_timing(tables, matches, "ripple", **quick),
    }
    for name, table in pairs.items():
        rows = table[(table.method_a == KAY[0]) & (table.method_b == KARLSSON[0])]
        assert len(rows) > 0, name
        assert (rows.n_failures_b == 2).all(), name
        assert (rows.n_failures_a == 0).all(), name
    agreement = pairs["pairwise_agreement"].set_index(["method_a", "method_b"])
    assert agreement.loc[(KAY[0], KARLSSON[0]), "n_sessions"] == 0
    assert np.isnan(agreement.loc[(KAY[0], KARLSSON[0]), "jaccard"])
    timing = pairs["paired_timing_ripple"]
    assert timing.fraction.tolist() == [0.1, 0.25, 0.5]
    assert timing.onset_signed_estimate.isna().all()


def test_other_levels_match_the_primary_expression_and_the_network(analyze, failing_run):
    tables = analyze.load_run(failing_run)
    pairs = analyze.match_run(tables, levels=(0.0, 0.2)).pairs
    found = {
        level: set(zip(rows.method, rows.expression, strict=True))
        for level, rows in pairs.groupby("minimum_iou")
    }
    # every expression at IoU 0, Kay's events over sharp waves included
    assert (KAY[0], "sharp_wave") in found[0.0]
    # past it, what the analyses read there: each primary expression and the network
    assert found[0.2] == {
        (KAY[0], "ripple"),
        (KAY[0], "network"),
        (MALLORY[0], "burst"),
        (MALLORY[0], "network"),
    }


def test_resample_weights_are_the_bootstrap_draws(analyze):
    rng = np.random.default_rng(1)
    frame = _per_session(rng.integers(0, 50, size=7).tolist(), shift=3)
    sums = analyze.paired_bootstrap(
        frame, lambda f: f.groupby("method")["value"].sum(), key="session_id", n_resamples=200
    )
    weights = analyze.resample_weights(7, n_resamples=200)
    assert (weights.sum(axis=1) == 7).all()
    per_session = frame.pivot_table(index="session_id", columns="method", values="value")
    # the sessions in the order paired_bootstrap lists them
    per_session = per_session.loc[frame.session_id.unique()]
    low, high = analyze.percentile_intervals(weights @ per_session.to_numpy())
    assert low.tolist() == sums.low.tolist()
    assert high.tolist() == sums.high.tolist()


def _grouped_rows(rows_per_session, whole):
    """Seven sessions, listed out of order, of three methods, with
    ``rows_per_session`` rows of each: four value columns, whole numbers or
    not, a tenth of the values missing."""
    rng = np.random.default_rng(4)
    n = 7 * 3 * rows_per_session
    values = rng.integers(1, 9, size=(n, 4)) if whole else rng.normal(1.0, 3.0, (n, 4))
    values = values.astype(float)
    values[rng.random((n, 4)) < 0.1] = np.nan
    sessions = [f"reference/{k}" for k in (3, 0, 5, 1, 6, 2, 4)]
    return pd.DataFrame(
        values, columns=["matched_sum", "matched_n", "all_sum", "all_n"]
    ).assign(
        session_id=np.repeat(sessions, 3 * rows_per_session),
        method=np.tile(["x", "y", "z"], 7 * rows_per_session),
    )


@pytest.mark.parametrize(
    ("factory", "arguments", "names", "rows_per_session", "whole"),
    [
        (
            "_ratio_of_sums",
            (("matched_sum", "matched_n"), ("all_sum", "matched_n")),
            2,
            1,
            False,
        ),
        ("_means", ("matched_sum", "all_n"), 2, 1, False),
        # a session's several rows of a group sum exactly when whole
        ("_means", ("matched_sum", "all_n"), 2, 3, True),
        ("_medians", ("matched_sum", "all_sum"), 2, 3, False),
        ("_ratio_of_means", (), 1, 2, True),
    ],
)
def test_grouped_intervals_are_the_paired_bootstrap(
    analyze, factory, arguments, names, rows_per_session, whole
):
    frame = _grouped_rows(rows_per_session, whole)
    statistic = getattr(analyze, factory)(*arguments)
    names = ["first", "second"][:names]
    got = analyze.grouped_intervals(frame, ["method"], statistic, names, n_resamples=FEW)
    # the plain algorithm: the statistic of each resample's concatenated rows
    codes = frame.groupby("method").ngroup().to_numpy()
    expected = analyze.paired_bootstrap(
        frame.assign(replicate=0, group=codes),
        lambda f: pd.Series(statistic(f.group.to_numpy(), f, 3).ravel()),
        key="session_id",
        n_resamples=FEW,
    )
    for position, name in enumerate(names):
        block = expected.iloc[3 * position : 3 * position + 3]
        np.testing.assert_array_equal(got[name], block.estimate)
        np.testing.assert_array_equal(got[f"{name}_low"], block.low)
        np.testing.assert_array_equal(got[f"{name}_high"], block.high)


def test_a_ratio_over_nothing_is_missing(analyze):
    frame = pd.DataFrame({"found": [1.0, 0.0, 2.0], "n": [0.0, 0.0, 4.0]})
    codes = np.arange(3)
    # one found of none is missing, not infinite, as none of none is
    np.testing.assert_array_equal(
        analyze._ratio_of_sums(("found", "n"))(codes, frame, 3), [[np.nan, np.nan, 0.5]]
    )
    np.testing.assert_array_equal(
        analyze._means("found")(codes, frame.assign(found=[np.nan, 1.0, 3.0]), 3),
        [[np.nan, 1.0, 3.0]],
    )


def test_weighted_medians_count_each_value(analyze):
    rng = np.random.default_rng(2)
    values = rng.normal(size=60).round(1)
    groups = rng.integers(0, 5, size=60)
    medians = analyze.WeightedMedians(values, groups, 6)
    for _ in range(20):
        counts = rng.integers(0, 3, size=60)
        expected = [
            np.median(np.repeat(values[groups == g], counts[groups == g]))
            if counts[groups == g].sum()
            else np.nan
            for g in range(6)
        ]
        np.testing.assert_allclose(medians(counts), expected, rtol=0, atol=1e-12)
    assert np.isnan(analyze.WeightedMedians([], [], 2)(np.array([]))).all()


def test_scores_of_every_condition_match_the_runner(analyze, run, tiny_run, tiny_matches):
    scores = analyze.load_scores(tiny_run.parent)
    metrics = run.read_table(tiny_run / "metrics.csv.gz")
    primary = {KAY[0]: "ripple", MALLORY[0]: "burst"}
    expected = metrics[metrics.expression == metrics.method.map(primary)]
    key = ["session_id", "method", "setting", "minimum_iou"]
    got = scores.counts.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        got,
        expected[list(analyze.COUNT_COLUMNS)].sort_values(key).reset_index(drop=True),
        check_dtype=False,
    )
    assert scores.failures.to_dict("records") == [
        {"session_id": "reference/1", "method": MALLORY[0], "setting": MALLORY[1]}
    ]
    # the errors at IoU 0 of the main settings are the pairs matched again
    errors = scores.errors.astype(dict.fromkeys(("session_id", "method", "setting"), str))
    pairs = tiny_matches.pairs
    for method, expression in primary.items():
        mine = analyze.main_rows(errors[(errors.method == method) & (errors.minimum_iou == 0)])
        theirs = pairs[(pairs.method == method) & (pairs.expression == expression)]
        assert sorted(mine.onset_error) == sorted(theirs.onset_error_10)
    # every setting of the reference at every level, for the operating curves
    sweep = errors[errors.setting == KAY_SWEEP[1]]
    assert sorted(sweep.minimum_iou.unique()) == [0.0, 0.2, 0.5]
    assert scores.sessions.minutes.tolist() == pytest.approx(
        ((20.0 - scores.sessions.event_time_s) / 60).tolist()
    )


def test_scores_count_point_inventories_by_containment(analyze, point_run):
    scores = analyze.load_scores(point_run.parent)
    points = scores.counts[scores.counts.method == DAVIDSON[0]]
    assert points.minimum_iou.isna().all()
    assert points[["n_reference", "n_detected", "n_matched"]].sum().tolist() == [8, 10, 6]
    assert DAVIDSON[0] not in set(scores.errors.method.astype(str))
    assert DAVIDSON[0] not in set(scores.participation.method)


def test_appendix_scores_every_method_against_every_expression(analyze, point_matched):
    tables, matches = point_matched
    table = analyze.appendix_expressions(tables, matches, n_resamples=FEW)
    assert table.expression.drop_duplicates().tolist() == [
        "network",
        "ripple",
        "sharp_wave",
        "burst",
    ]
    kay = _by(table[table.method == KAY[0]], "expression")
    # per session: four ripple windows, six events, three matched (the
    # doublet's two ripples by one event); IoUs 1, 0.8 and 7/17
    ripple = kay.loc["ripple"]
    assert [ripple.n_reference, ripple.n_detected, ripple.n_matched] == [8, 12, 6]
    assert [ripple.recall, ripple.precision] == [0.75, 0.5]
    minutes = ((20.0 - tables.sessions.event_time_s) / 60).sum()
    assert ripple.false_positives_per_minute == pytest.approx(6 / minutes)
    assert (ripple.n_pairs, ripple.median_iou) == (6, pytest.approx(0.8, abs=1e-5))
    assert ripple.recall_low == ripple.recall_high == 0.75
    assert kay.primary.to_dict() == {
        "network": False,
        "ripple": True,
        "sharp_wave": False,
        "burst": False,
    }
    assert (kay.scoring == "interval").all()
    # Davidson's points by containment against every expression: of the five
    # sharp-wave windows, the first ripple's peak and the doublet's second
    # ripple's hold one each; no bound or overlap measure
    davidson = _by(table[table.method == DAVIDSON[0]], "expression")
    assert (davidson.scoring == "peak_containment").all()
    assert davidson.loc["sharp_wave", ["n_reference", "n_matched", "recall"]].tolist() == [
        10,
        4,
        0.4,
    ]
    assert davidson.loc["ripple", "recall"] == 0.75
    assert davidson[["n_pairs", "median_iou", "median_abs_onset_error"]].isna().all().all()


def test_appendix_curves_read_every_expression(analyze, run, tiny_run):
    scores = analyze.load_scores(tiny_run.parent)
    metrics = run.read_table(tiny_run / "metrics.csv.gz")
    for expression in ("network", "sharp_wave"):
        curves = analyze.expression_curves(scores, expression)
        assert (curves.expression == expression).all()
        mine = metrics[(metrics.expression == expression)]
        expected = mine.groupby(["method", "setting", "minimum_iou"])[
            ["n_matched", "n_reference"]
        ].sum()
        got = curves.set_index(["method", "setting", "minimum_iou"])
        for key, row in expected.iterrows():
            assert got.loc[key, "recall"] == pytest.approx(row.n_matched / row.n_reference)
        assert set(got.index) == set(expected.index)
        assert curves.set_index("setting").kind.to_dict() == {
            "3.0": "sweep",
            "default": "default",
            "literature": "recipe",
        }


def _design_at_fp_rate(curve, target, floor, columns):
    """``at_fp_rate`` as the benchmark's design writes it, in pandas."""
    x = np.log(np.maximum(curve.fp_rate.to_numpy(float), floor))
    ranked = curve.assign(_x=x, _order=np.arange(len(curve)))
    ranked = ranked.sort_values(["_x", "recall", "_order"], ascending=[True, False, True])
    points = ranked.drop_duplicates("_x", keep="first")
    xs, t = points._x.to_numpy(), np.log(target)
    if not (xs[0] <= t <= xs[-1]):
        return pd.Series(np.nan, index=list(columns))
    return pd.Series({c: float(np.interp(t, xs, points[c].to_numpy(float))) for c in columns})


def test_at_fp_rate_interpolates_in_log_rate(analyze):
    curve = pd.DataFrame(
        {
            "fp_rate": [8.0, 2.0, 0.5, 0.0],
            "recall": [0.9, 0.6, 0.4, 0.2],
            "onset": [0.04, 0.02, 0.0, -0.02],
        }
    )
    columns = ["recall", "onset"]
    at = functools.partial(analyze.at_fp_rate, curve, floor=0.25, columns=columns)
    # halfway between 0.5 and 2 per minute in log rate, and between 2 and 8
    assert at(1.0).tolist() == pytest.approx([0.5, 0.01])
    assert at(4.0).tolist() == pytest.approx([0.75, 0.03])
    # a rate of 0 counts as the floor, and below it or past the end is NaN
    assert at(0.25).tolist() == pytest.approx([0.2, -0.02])
    assert at(0.2).isna().all()
    assert at(9.0).isna().all()


def test_at_fp_rate_keeps_one_setting(analyze):
    curve = pd.DataFrame(
        {
            "fp_rate": [2.0, 1.0, 1.0, 0.5],
            "recall": [0.9, 0.6, 0.8, 0.5],
            "onset": [0.03, 0.02, -0.01, 0.0],
        }
    )
    found = analyze.at_fp_rate(curve, 1.0, 0.1, ["recall", "onset"])
    # the better setting at that rate, whole: its own onset, not the other's
    assert found.tolist() == pytest.approx([0.8, -0.01])


def test_read_off_kinds_say_where_a_value_is_interpolated(analyze):
    fp_rate, recall = [8.0, 2.0, 2.0, 0.5, 0.0], [0.9, 0.6, 0.7, 0.4, 0.2]
    targets = [0.2, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 9.0]
    kinds = analyze.read_off_kinds(fp_rate, recall, targets, 0.25)
    # at a setting's rate (0 floored to 0.25, a repeated rate once), between
    # two, and outside the curve
    assert kinds.tolist() == [
        "",
        "tested",
        "tested",
        "interpolated",
        "tested",
        "interpolated",
        "tested",
        "",
    ]
    assert analyze.read_off_kinds([], [], [1.0], 0.25).tolist() == [""]


def test_at_fp_rate_is_the_design(analyze):
    rng = np.random.default_rng(3)
    for _ in range(200):
        n = int(rng.integers(1, 7))
        curve = pd.DataFrame(
            {
                "fp_rate": rng.choice([0.0, 0.0, 0.5, 1.0, 2.0, 4.0], size=n),
                "recall": rng.choice([0.2, 0.5, 0.5, 0.9], size=n),
                "onset": rng.normal(size=n),
            }
        )
        for target in (0.1, 0.5, 1.5, 4.0):
            np.testing.assert_array_equal(
                analyze.at_fp_rate(curve, target, 0.2, ["recall", "onset"]),
                _design_at_fp_rate(curve, target, 0.2, ["recall", "onset"]),
            )


def _hand_scores(analyze, counts, errors=(), minutes=10.0):
    """ConditionScores built by hand: ``counts`` and ``errors`` are dicts
    with ``condition_id`` and ``replicate`` in place of a session, every
    session ``minutes`` long outside the network windows, every method's
    primary expression ripple but Mallory's (burst)."""
    counts = pd.DataFrame(list(counts)).assign(
        session_id=lambda f: f.condition_id + "/" + f.replicate.astype(str)
    )
    counts = (
        counts.fillna({"minimum_iou": 0.0})
        if "minimum_iou" in counts
        else counts.assign(minimum_iou=0.0)
    )
    sessions = counts[["session_id", "condition_id", "replicate"]].drop_duplicates()
    sessions = sessions.assign(
        duration_s=600.0 + 60 * minutes, rest_s=600.0, event_time_s=600.0, minutes=minutes
    ).reset_index(drop=True)
    methods = counts[["method", "setting"]].drop_duplicates().reset_index(drop=True)
    methods["primary_expression"] = np.where(methods.method == MALLORY[0], "burst", "ripple")
    methods["scoring"] = methods.method.map(analyze.scoring_rule)
    cells = {c: [part.split("=") for part in c.split(",")] for c in sessions.condition_id}
    conditions = pd.DataFrame(
        [
            {
                "condition_id": c,
                "factor": ",".join(part[0] for part in parts) if "=" in c else "reference",
                "level": ",".join(part[-1] for part in parts),
            }
            for c, parts in cells.items()
        ]
    )
    errors = pd.DataFrame(
        list(errors), columns=[*analyze.ERROR_ROW_COLUMNS, "condition_id", "replicate"]
    )
    errors["session_id"] = errors.condition_id + "/" + errors.replicate.astype(str)
    errors = errors.fillna({"minimum_iou": 0.0})
    return analyze.ConditionScores(
        sessions=sessions,
        conditions=conditions,
        methods=methods,
        counts=counts[list(analyze.COUNT_COLUMNS)],
        errors=errors[list(analyze.ERROR_ROW_COLUMNS)],
        participation=pd.DataFrame(columns=list(analyze.PARTICIPATION_COLUMNS)),
        failures=pd.DataFrame(columns=["session_id", "method", "setting"]),
    )


def _kay(condition, replicate, setting, matched, detected, reference=10, method=KAY[0]):
    return {
        "condition_id": condition, "replicate": replicate, "method": method,
        "setting": setting, "n_reference": reference, "n_detected": detected,
        "n_matched": matched,
    }  # fmt: skip


def _error(condition, replicate, setting, onset, method=KAY[0], offset=0.0, level=0.0):
    return {
        "condition_id": condition, "replicate": replicate, "method": method,
        "setting": setting, "minimum_iou": level, "onset_error": onset,
        "offset_error": offset,
    }  # fmt: skip


def test_operating_curves_and_points_by_hand(analyze):
    # two sessions; per session 10 truth windows and 10 minutes outside them
    counts, errors = [], []
    for replicate in (0, 1):
        for setting, matched, detected in (("2.0", 8, 28), ("3.0", 6, 11), ("4.0", 3, 3)):
            counts.append(_kay("reference", replicate, setting, matched, detected))
        errors += [_error("reference", replicate, "3.0", onset) for onset in (-0.01, 0.01)]
        errors += [_error("reference", replicate, "2.0", -0.03)]
    scores = _hand_scores(analyze, counts, errors)
    curves = analyze.operating_curves(scores)
    at_zero = curves[curves.minimum_iou == 0].set_index("setting")
    assert at_zero.recall.to_dict() == {"2.0": 0.8, "3.0": 0.6, "4.0": 0.3}
    assert at_zero.false_positives_per_minute.to_dict() == {"2.0": 2.0, "3.0": 0.5, "4.0": 0.0}
    assert at_zero.loc["3.0", ["median_onset_error", "median_abs_onset_error"]].tolist() == [
        0.0,
        0.01,
    ]
    assert (at_zero.kind == "sweep").all()
    assert at_zero.threshold.tolist() == [2.0, 3.0, 4.0]
    points = analyze.operating_points(scores, n_resamples=FEW)
    points = points[points.minimum_iou == 0].set_index("fp_target")
    curve = at_zero.rename(columns={"false_positives_per_minute": "fp_rate"})
    for target in (0.5, 1.0, 2.0):
        expected = analyze.at_fp_rate(
            curve, target, 0.5 / 20, ["recall", "median_onset_error", "median_offset_error"]
        )
        assert points.loc[target, list(analyze.AT_TARGET)].tolist() == pytest.approx(
            expected.tolist()
        )
    # the two sessions are alike: every resample reads the same values off
    assert points.loc[1.0, "recall_low"] == pytest.approx(points.loc[1.0, "recall"])
    # 5 per minute is past the curve's end: missing, not its last point
    assert points.loc[5.0, ["recall", "recall_low", "recall_high"]].isna().all()
    assert points.attained.tolist() == [1.0, 1.0, 1.0, 0.0]
    # 0.5 and 2 per minute are two settings' rates; 1 lies between them
    assert points.read_off.tolist() == ["tested", "interpolated", "tested", ""]


def test_operating_curves_label_defaults_and_recipes(analyze):
    counts = []
    for replicate in (0, 1):
        counts += _curve("reference", replicate, KAY[0], ((8, 28), (6, 11)))
        counts += [
            _kay("reference", replicate, "default", 6, 11),
            _kay("reference", replicate, KARLSSON[1], 5, 9, method=KARLSSON[0]),
        ]
    curves = analyze.operating_curves(_hand_scores(analyze, counts))
    at_zero = curves[curves.minimum_iou == 0]
    assert at_zero[["method", "setting", "kind"]].to_numpy().tolist() == [
        [KAY[0], "2.0", "sweep"],
        [KAY[0], "3.0", "sweep"],
        [KAY[0], "default", "default"],
        [KARLSSON[0], "literature", "recipe"],
    ]
    assert at_zero.threshold.isna().tolist() == [False, False, True, True]


def test_an_unreached_target_keeps_no_interval(analyze):
    # the first session's curve reaches 4 false positives a minute, the
    # second's 0.5, the two pooled 2.25: 3 per minute is past the pooled
    # curve, though resamples of the first session alone reach it
    counts = [
        *_curve("reference", 0, KAY[0], ((8, 48), (6, 8))),
        *_curve("reference", 1, KAY[0], ((8, 13), (6, 9))),
    ]
    points = analyze.operating_points(
        _hand_scores(analyze, counts), targets=(3.0,), n_resamples=FEW
    )
    row = points[points.minimum_iou == 0].iloc[0]
    assert 0 < row.attained < 1
    assert np.isnan(row.recall)
    assert np.isnan(row.recall_low)
    assert np.isnan(row.recall_high)


def test_held_out_threshold_reports_held_out_replicates(analyze):
    counts, errors = [], []
    for replicate in range(10):
        if analyze.is_held_out(replicate):
            counts += [
                _kay("x=y", replicate, "2.0", 10, 11),
                _kay("x=y", replicate, "3.0", 4, 10),
            ]
            errors += [_error("x=y", replicate, "3.0", 0.01, offset=0.02)]
        else:
            counts += [
                _kay("x=y", replicate, "2.0", 9, 30),
                _kay("x=y", replicate, "3.0", 8, 12),
            ]
            errors += [_error("x=y", replicate, "3.0", -0.02)]
    scores = _hand_scores(analyze, counts, errors)
    table = analyze.held_out_thresholds(scores, condition="x=y", n_resamples=FEW)
    row = table.set_index("fp_target").loc[1.0]
    # chosen on the even replicates: 3.0 keeps 0.4 false positives per minute
    # there, 2.0 has 2.1; the odd ones would have chosen 2.0
    assert row.setting == "3.0"
    assert [row.calibration_recall, row.calibration_fp_rate] == pytest.approx([0.8, 0.4])
    # reported on the odd replicates alone
    assert [row.recall, row.false_positives_per_minute] == pytest.approx([0.4, 0.6])
    assert [row.median_onset_error, row.median_offset_error] == [0.01, 0.02]
    assert row.recall_low == row.recall_high == pytest.approx(0.4)
    assert (row.n_calibration_sessions, row.n_held_out_sessions) == (5, 5)
    assert row.held_out_replicates == "1 3 5 7 9"
    # no setting keeps 0.2 per minute on the calibration replicates: nothing chosen
    none = analyze.held_out_thresholds(
        scores, condition="x=y", targets=(0.2,), n_resamples=FEW
    ).iloc[0]
    assert none.setting == ""
    assert (
        none[["calibration_recall", "recall", "recall_low", "median_onset_error"]].isna().all()
    )


def _snr_run(analyze, shift=-2, extra_reference=2):
    """Kay on four replicates of the reference and of each ripple SNR level,
    the low level finding ``-shift`` fewer of 10 windows in every replicate
    and the high one as many more, with onsets 5 ms later and earlier; the
    reference holds ``extra_reference`` more replicates, shared by no other."""
    matched = [5, 7, 6, 4, 8, 6][: 4 + extra_reference]
    counts, errors = [], []
    for condition, step, onset in (
        ("reference", 0, 0.0),
        ("ripple_snr=low", shift, 0.005),
        ("ripple_snr=high", -shift, -0.005),
    ):
        for replicate, found in enumerate(
            matched if condition == "reference" else matched[:4]
        ):
            counts.append(_kay(condition, replicate, "default", found + step, 12))
            errors += [
                _error(condition, replicate, "default", replicate / 100 + onset + delta)
                for delta in (-0.001, 0.001)
            ]
    return _hand_scores(analyze, counts, errors)


def test_robustness_pairs_conditions_by_replicate(analyze):
    table = analyze.robustness(_snr_run(analyze), n_resamples=FEW)
    assert table.factor.unique().tolist() == ["ripple_snr"]
    recall = table[table.measure == "recall"].set_index("level")
    assert recall.index.tolist() == ["low", "reference", "high"]
    # pooled over the four shared replicates, the reference's two others left out
    assert recall.n_replicates.unique().tolist() == [4]
    assert recall.value.tolist() == pytest.approx([0.35, 0.55, 0.75])
    # every replicate moves by the same amount, so paired resamples do too
    assert recall.change.tolist() == pytest.approx([-0.2, 0.0, 0.2])
    for level, change in (("low", -0.2), ("high", 0.2)):
        row = recall.loc[level]
        assert [row.change_low, row.change_high] == pytest.approx([change, change])
        assert (row.change_p, row.n_draws, row.n_defined, row.n_defined_reference) == (
            0.0,
            FEW,
            4,
            4,
        )
    # while the values themselves vary with the replicates drawn
    assert recall.loc["low", "value_low"] < 0.35 < recall.loc["low", "value_high"]
    onset = table[table.measure == "median_onset_error"].set_index("level")
    assert onset.change.tolist() == pytest.approx([0.005, 0.0, -0.005])
    assert onset.loc["low", "change_low"] == pytest.approx(0.005)
    listed = analyze.recall_changes(table)
    assert listed[["factor", "method", "lowest", "highest"]].to_numpy().tolist() == [
        ["ripple_snr", KAY[0], "low", "high"]
    ]
    assert listed.span.tolist() == pytest.approx([0.4])
    assert analyze.recall_changes(table, threshold=0.5).empty


def test_conditions_are_found_by_their_listed_factor_and_level(analyze):
    scores = _snr_run(analyze)
    # an id that is not "<factor>=<level>": conditions.csv says what it is
    renamed = {"condition_id": {"ripple_snr=low": "snr_low"}}
    scores = dataclasses.replace(
        scores,
        sessions=scores.sessions.replace(renamed),
        conditions=scores.conditions.replace(renamed),
    )
    table = analyze.robustness(scores, measures=("recall",), n_resamples=FEW)
    assert table.set_index("level").condition_id.to_dict() == {
        "low": "snr_low",
        "reference": "reference",
        "high": "ripple_snr=high",
    }


def _failed(scores, sessions, method, settings):
    """``scores`` with ``method``'s rows at ``settings`` on ``sessions`` removed
    and recorded as failures."""
    counts, errors = scores.counts, scores.errors
    drop = (
        counts.session_id.isin(sessions)
        & (counts.method == method)
        & counts.setting.isin(settings)
    )
    lost = (
        errors.session_id.isin(sessions)
        & (errors.method == method)
        & errors.setting.isin(settings)
    )
    return dataclasses.replace(
        scores,
        counts=counts[~drop].reset_index(drop=True),
        errors=errors[~lost].reset_index(drop=True),
        failures=pd.concat(
            [scores.failures, counts.loc[drop, ["session_id", "method", "setting"]]],
            ignore_index=True,
        ),
    )


def test_paired_changes_use_only_replicates_run_in_every_condition(analyze):
    # Kay finds 2 of 10 windows on replicates 0 and 1 and 9 on 2 and 3, in
    # both conditions alike, but fails on 0 and 1 under low SNR; Roumis runs
    # everywhere
    counts = []
    for condition in ("reference", "ripple_snr=low"):
        for replicate, found in enumerate((2, 2, 9, 9)):
            counts.append(_kay(condition, replicate, "default", found, 12))
            counts.append(_kay(condition, replicate, "default", 5, 12, method="Roumis"))
    scores = _failed(
        _hand_scores(analyze, counts),
        ["ripple_snr=low/0", "ripple_snr=low/1"],
        KAY[0],
        ["default"],
    )
    changes = analyze.paired_changes(
        scores,
        ["reference", "ripple_snr=low"],
        "reference",
        measures=("recall",),
        n_resamples=FEW,
    ).set_index(["method", "condition_id"])
    kay = changes.loc[(KAY[0], "ripple_snr=low")]
    # both values over replicates 2 and 3, never the reference's easy and hard
    # replicates against the other condition's easy ones alone
    assert kay.value == pytest.approx(0.9)
    assert changes.loc[(KAY[0], "reference"), "value"] == pytest.approx(0.9)
    assert [kay.change, kay.change_low, kay.change_high] == pytest.approx([0.0, 0.0, 0.0])
    assert (kay.n_replicates, kay.n_dropped, kay.n_failures) == (2, 2, 2)
    # replicates 2 and 3, each with a recall of its own on both sides, 9 true
    # events found on each
    assert (kay.n_defined, kay.n_defined_reference) == (2, 2)
    assert (kay.n_events, kay.n_events_reference) == (18, 18)
    roumis = changes.loc[("Roumis", "ripple_snr=low")]
    assert (roumis.n_replicates, roumis.n_dropped, roumis.n_failures) == (4, 0, 0)


def _two_curves(condition, replicate, method=KAY[0]):
    """Replicates 0 and 1 find 8 and 6 of 10 windows at settings 2.0 and 3.0,
    the others 4 and 2, at 2 and 0.5 false positives a minute everywhere."""
    points = ((8, 28), (6, 11)) if replicate < 2 else ((4, 24), (2, 7))
    return _curve(condition, replicate, method, points)


def test_sweep_recalls_use_only_replicates_with_the_whole_sweep(analyze):
    # the same curves in the reference and under refractory spiking, but
    # Kay's setting 3.0 fails on replicates 0 and 1 there
    counts = []
    for condition in ("reference", "spike_model=refractory"):
        for replicate in range(4):
            counts += _two_curves(condition, replicate)
            counts += [_kay(condition, replicate, "default", 6, 11)]
    alternative = "spike_model=refractory"
    scores = _failed(
        _hand_scores(analyze, counts),
        [f"{alternative}/0", f"{alternative}/1"],
        KAY[0],
        ["3.0"],
    )
    changes, _ = analyze.model_sensitivity(scores, n_resamples=FEW)
    at_one = changes[
        (changes.alternative == alternative)
        & (changes.measure == "recall_at_fp")
        & (changes.fp_target == 1.0)
    ].iloc[0]
    # halfway (in log rate) between 0.4 and 0.2, both curves over replicates 2 and 3
    assert [at_one.reference_value, at_one.value, at_one.change] == pytest.approx(
        [0.3, 0.3, 0.0]
    )
    assert (at_one.status, at_one.n_replicates, at_one.n_dropped) == ("compared", 2, 2)


def test_a_change_is_tested_on_the_resamples_of_its_interval(analyze):
    """Replicate 0 has 2 truth windows, both found in the reference and
    neither under low SNR (its own change -1); replicates 1 to 3 have 20, 10
    found in the reference and 12 under low SNR (+0.1 each). The pooled
    change of a resample with counts w is sum(w d) / sum(w n), d the found
    windows' change and n the windows: its p-value is from those resamples,
    the ones its interval comes from."""
    counts = []
    for replicate, (windows, found, low) in enumerate([(2, 2, 0)] + [(20, 10, 12)] * 3):
        counts.append(_kay("reference", replicate, "default", found, 30, reference=windows))
        counts.append(_kay("ripple_snr=low", replicate, "default", low, 30, reference=windows))
    scores = _hand_scores(analyze, counts)
    changes = analyze.paired_changes(
        scores,
        ["reference", "ripple_snr=low"],
        "reference",
        measures=("recall",),
        n_resamples=FEW,
    ).set_index("condition_id")
    row = changes.loc["ripple_snr=low"]
    assert row.change == pytest.approx(4 / 62)
    weights = analyze.resample_weights(4, n_resamples=FEW)
    draws = weights @ np.array([-2.0, 2.0, 2.0, 2.0]) / (weights @ np.array([2.0, 20, 20, 20]))
    p, n_draws = analyze.bootstrap_p(draws[:, np.newaxis])
    low, high = analyze.percentile_intervals(draws[:, np.newaxis])
    assert [row.change_low, row.change_high] == pytest.approx([low[0], high[0]])
    assert row.change_p == pytest.approx(p[0])
    assert row.n_draws == n_draws[0] == FEW
    assert (row.change_p < 0.05) == (row.change_low > 0 or row.change_high < 0)
    # what the change rests on: every replicate defined on both sides, 36
    # true events found here and 32 in the reference
    assert (row.n_defined, row.n_defined_reference) == (4, 4)
    assert (row.n_events, row.n_events_reference) == (36, 32)
    # the reference's own change is 0 in every resample
    assert changes.loc["reference", "change_p"] == 1.0


def test_the_recall_change_at_a_target_is_tested_on_the_resamples_of_its_interval(analyze):
    """Kay's recall at 1 false positive a minute is 0.7 in every reference
    session and 0.8, 0.6, 0.7 and 0.5 under refractory spiking, at the same
    false-positive rates, so a resample's change is its weighted mean of
    +0.1, -0.1, 0 and -0.2."""
    counts = []
    alternative = "spike_model=refractory"
    for replicate, (first, second) in enumerate([(9, 7), (7, 5), (8, 6), (6, 4)]):
        counts += _curve("reference", replicate, KAY[0], ((8, 28), (6, 11)))
        counts += _curve(
            alternative, replicate, KAY[0], ((first, first + 20), (second, second + 5))
        )
        for condition in ("reference", alternative):
            counts.append(_kay(condition, replicate, "default", 6, 11))
    changes, _ = analyze.model_sensitivity(_hand_scores(analyze, counts), n_resamples=FEW)
    row = changes[
        (changes.alternative == alternative)
        & (changes.measure == "recall_at_fp")
        & (changes.fp_target == 1.0)
    ].iloc[0]
    assert row.change == pytest.approx(-0.05)
    draws = analyze.resample_weights(4, n_resamples=FEW) @ np.array([0.1, -0.1, 0.0, -0.2]) / 4
    p, n_draws = analyze.bootstrap_p(draws[:, np.newaxis])
    assert row.change_p == pytest.approx(p[0])
    assert row.n_draws == n_draws[0] == FEW
    # every replicate's own curve reaches the target on both sides; the
    # events at the setting nearest it (2.0: 2 and 0.5 per minute are as
    # near) are 30 true events found there and 32 in the reference
    assert (row.n_defined, row.n_defined_reference) == (4, 4)
    assert (row.n_events, row.n_events_reference) == (30, 32)


def _unreached_curve(condition, replicate, method):
    """A sweep whose pooled curve stops at 1.125 false positives a minute (0
    and 1.5 per minute per session at setting 3.0), while a resample drawing
    replicate 0 twice or more comes down to 1 per minute."""
    low = (6, 6) if replicate == 0 else (6, 21)
    return _curve(condition, replicate, method, ((8, 58), low))


def test_a_target_the_pooled_curve_misses_has_no_test(analyze):
    """Some resamples reach 1 false positive a minute where the pooled curve
    does not: the missing estimate has no p-value and no draws counted."""
    counts = []
    for replicate in range(4):
        counts += _curve("reference", replicate, KAY[0], ((8, 28), (6, 11)))
        counts += _unreached_curve("reference", replicate, ROUMIS)
    scores = _hand_scores(analyze, counts)
    found = analyze._sweep_recalls(
        scores, ["reference"], [0, 1, 2, 3], [KAY[0], ROUMIS], [1.0], FEW
    )
    assert np.isnan(found.estimate[0, 1, 0])
    assert np.isfinite(found.draws[:, 0, 1, 0]).any()
    table = analyze.operating_differences(scores, targets=(1.0,), n_resamples=FEW)
    row = table.iloc[0]
    assert np.isnan(row.difference)
    assert np.isnan(row.difference_p)
    assert row.n_draws == 0
    # nor under an alternative model
    for replicate in range(4):
        counts += _curve("spike_model=refractory", replicate, KAY[0], ((8, 28), (6, 11)))
        counts += _unreached_curve("spike_model=refractory", replicate, ROUMIS)
    changes, _ = analyze.model_sensitivity(_hand_scores(analyze, counts), n_resamples=FEW)
    missed = changes[
        (changes.alternative == "spike_model=refractory")
        & (changes.method == ROUMIS)
        & (changes.fp_target == 1.0)
    ].iloc[0]
    assert np.isnan(missed.change)
    assert np.isnan(missed.change_p)
    assert missed.n_draws == 0


def test_operating_points_pool_the_sessions_with_the_whole_sweep(analyze):
    counts = [row for replicate in range(4) for row in _two_curves("reference", replicate)]
    scores = _failed(_hand_scores(analyze, counts), ["reference/0"], KAY[0], ["3.0"])
    points = analyze.operating_points(scores, n_resamples=FEW)
    row = points[(points.minimum_iou == 0) & (points.fp_target == 1.0)].iloc[0]
    # replicates 1, 2 and 3: recall 0.8 / 3 + 0.4 * 2 / 3 at 2 per minute, 0.6 / 3
    # + 0.2 * 2 / 3 at 0.5, halfway between in log rate
    assert row.recall == pytest.approx((1.6 / 3 + 1.0 / 3) / 2)
    assert (row.n_sessions, row.n_dropped, row.n_failures) == (3, 1, 1)
    curves = analyze.operating_curves(scores)
    sweep = curves[(curves.minimum_iou == 0)].set_index("setting")
    assert sweep.n_sessions.to_dict() == {"2.0": 3, "3.0": 3}
    assert sweep.n_dropped.to_dict() == {"2.0": 1, "3.0": 0}
    assert sweep.recall.to_dict() == pytest.approx({"2.0": 1.6 / 3, "3.0": 1.0 / 3})


def test_recall_changes_are_listed_past_the_threshold_only(analyze, tiny_tables):
    table = pd.DataFrame(
        {
            "factor": "ripple_snr",
            "level": ["low", "reference", "high"],
            "method": KAY[0],
            "setting": KAY[1],
            "measure": "recall",
            "value": [0.25, 0.5, 0.75],
        }
    )
    # a span of exactly the threshold is not more than it
    assert analyze.recall_changes(table, threshold=0.5).empty
    assert analyze.recall_changes(table, threshold=0.49).span.tolist() == [0.5]
    points = table.assign(method=DAVIDSON[0], setting=DAVIDSON[1], scoring="peak_containment")
    both = pd.concat([table.assign(scoring="interval"), points], ignore_index=True)
    summary = analyze._summary("x", tiny_tables, [], {"robustness_recall": both}, None)
    assert f"- `ripple_snr`: `{KAY[0]}` (default) 0.250 at low to 0.750 at high\n" in summary
    # a point method's recall is by peak containment, and says so
    assert (
        f"- `ripple_snr`: `{DAVIDSON[0]}` (literature) 0.250 at low to 0.750 at high, by "
        "peak containment\n" in summary
    )


def _participation_run(analyze):
    """Kay's default on four replicates of the reference and of refractory
    spiking: onsets 10 ms early in both, offsets 20 ms late in the reference
    and 30 ms under refractory spiking; 11 events a session, half of the
    principal units active in each in the reference and 0.3 under
    refractory spiking."""
    counts, errors, participation = [], [], []
    for condition, offset, active in (
        ("reference", 0.02, 0.5),
        ("spike_model=refractory", 0.03, 0.3),
    ):
        for replicate in range(4):
            counts.append(_kay(condition, replicate, "default", 6, 11))
            errors.append(_error(condition, replicate, "default", -0.01, offset=offset))
            participation.append(
                {
                    "session_id": f"{condition}/{replicate}",
                    "method": KAY[0],
                    "setting": "default",
                    "n_events": 11,
                    "principal_fraction": 11 * active,
                }
            )
    scores = _hand_scores(analyze, counts, errors)
    return dataclasses.replace(scores, participation=pd.DataFrame(participation))


def test_model_sensitivity_of_participation_and_errors(analyze):
    changes, _ = analyze.model_sensitivity(_participation_run(analyze), n_resamples=FEW)
    mine = changes[
        (changes.alternative == "spike_model=refractory") & (changes.method == KAY[0])
    ].set_index("measure")
    # the mean fraction of principal units active per event, not its sum
    assert [
        mine.loc["participation", "reference_value"],
        mine.loc["participation", "value"],
    ] == pytest.approx([0.5, 0.3])
    assert mine.loc["participation", "change"] == pytest.approx(-0.2)
    # offsets from the offsets, onsets from the onsets
    assert mine.loc["median_offset_error", "change"] == pytest.approx(0.01)
    assert mine.loc["median_onset_error", "change"] == pytest.approx(0.0)


def test_participation_is_the_principal_fraction_of_each_event(analyze):
    events = pd.DataFrame(
        {
            "session_id": "reference/0",
            "method": [KAY[0], KAY[0], KAY[0], DAVIDSON[0]],
            "setting": ["default", "default", "3.0", "literature"],
            "event_index": [0, 1, 0, 0],
            "n_active_principal": [3, 1, 2, 3],
        }
    )
    ran = pd.DataFrame(
        {
            "session_id": "reference/0",
            "method": [KAY[0], KARLSSON[0], KAY[0], DAVIDSON[0]],
            "setting": ["default", "literature", "3.0", "literature"],
        }
    )
    # three principal units (place and pyramidal) and an interneuron
    units = pd.DataFrame(
        {
            "session_id": "reference/0",
            "unit": range(4),
            "unit_type": ["place", "pyramidal", "interneuron", "pyramidal"],
        }
    )
    table = analyze._participation(events, ran, units)
    # main interval settings only; Karlsson ran and found nothing
    assert table[["method", "n_events"]].to_numpy().tolist() == [[KAY[0], 2], [KARLSSON[0], 0]]
    assert table.principal_fraction.tolist() == pytest.approx([3 / 3 + 1 / 3, 0.0])


def test_robustness_crossed_cells(analyze):
    counts = []
    for condition, found in (
        ("reference", 6),
        ("ripple_snr=low", 4),
        ("participation=high", 7),
        ("ripple_snr=low,participation=high", 3),
    ):
        counts += [_kay(condition, replicate, "default", found, 12) for replicate in range(3)]
    table = analyze.robustness_crossed(_hand_scores(analyze, counts), n_resamples=FEW)
    recall = table[table.measure == "recall"]
    assert recall[["factors", "level_1", "level_2"]].to_numpy().tolist() == [
        ["ripple_snr,participation", "low", "reference"],
        ["ripple_snr,participation", "low", "high"],
        ["ripple_snr,participation", "reference", "reference"],
        ["ripple_snr,participation", "reference", "high"],
    ]
    assert recall.change.tolist() == pytest.approx([-0.2, -0.3, 0.0, 0.1])


def _spiking_session(run, origin):
    """The tiny session's events, recruited cells (10 in the swr's burst, 2,
    6 and 4 in the others'), and three units: a recruited pyramidal cell that
    stays silent, an interneuron firing at the swr's peak and a place cell
    firing inside its network window but past its ripple's. Kay's events
    are the swr ripple's window and the doublet's first; Karlsson's is the
    swr ripple's, stretched to the place cell's spike."""
    session = _tiny_session(run, origin)
    events = session["events"]
    recruited = {0: 10, 1: 2, 2: 6, 3: 4}
    burst = events.expression == "burst"
    events.loc[burst, "n_participants"] = events.loc[burst, "event_id"].map(recruited)
    time = origin + np.arange(0, 20.0, 0.001)
    multiunit = np.zeros((len(time), 3))
    multiunit[np.searchsorted(time, origin + 2.0), 1] = 1
    multiunit[np.searchsorted(time, origin + 2.055), 2] = 1
    ripples = _windows(events)
    session["spikes"] = SimpleNamespace(
        time=time,
        multiunit=multiunit,
        unit_types=np.array(["pyramidal", "interneuron", "place"]),
    )
    session["detected"] = {
        KAY: ripples[[0, 2]],
        KARLSSON: np.array([[ripples[0, 0], origin + 2.056]]),
    }
    return session


@pytest.fixture(scope="module")
def spiking_matched(analyze, run, tmp_path_factory):
    """The spiking sessions' tables, one at a Unix clock origin, and their
    matches."""
    sessions = [_spiking_session(run, 0.0), _spiking_session(run, UNIX_ORIGIN)]
    tables = analyze.load_run(_write_run(run, tmp_path_factory.mktemp("spiking"), sessions))
    return tables, analyze.match_run(tables)


def test_boundary_effect_is_zero_for_equal_bounds(analyze, spiking_matched):
    tables, matches = spiking_matched
    effect = _by(
        analyze.boundary_effect(tables, matches, n_resamples=FEW), "method", "selection"
    )
    # Kay's bounds are the truth's: the same units either way, although the
    # recruited pyramidal cell was silent, the interneuron fired and the
    # network window is wider
    for selection in ("all", "principal"):
        row = effect.loc[(KAY[0], selection)]
        assert (row.n_pairs, row.mean_difference, row.median_difference) == (4, 0.0, 0.0)
    assert effect.loc[(KAY[0], "all"), ["mean_detected", "mean_truth"]].tolist() == [0.5, 0.5]
    # Karlsson's longer event also holds the place cell's spike
    for selection in ("all", "principal"):
        row = effect.loc[(KARLSSON[0], selection)]
        assert (row.n_pairs, row.mean_difference) == (2, 1.0)
        assert row.mean_difference_low == row.mean_difference_high == 1.0


def test_participation_bias_by_hand(analyze, spiking_matched):
    tables, matches = spiking_matched
    bias = _by(analyze.participation_bias(tables, matches, n_resamples=FEW), "method")
    # every session: bursts of 10, 2, 6 and 4 recruited cells, mean 5.5; Kay
    # finds the swr (10) and the doublet (4), Karlsson the swr
    kay = bias.loc[KAY[0]]
    assert [kay.n_matched_events, kay.n_events] == [4, 8]
    assert [kay.mean_matched, kay.mean_all] == [7.0, 5.5]
    assert kay.ratio_of_means == pytest.approx(7 / 5.5)
    assert kay.ks_statistic == pytest.approx(0.25)
    assert bias.loc[KARLSSON[0], "ratio_of_means"] == pytest.approx(10 / 5.5)


def test_rates_by_state_by_hand(analyze, tiny_tables):
    bouts = {session: np.array([[10.0, 15.0]]) for session in tiny_tables.truth}
    bouts["reference/1"] = bouts["reference/1"] + UNIX_ORIGIN
    rates = _by(
        analyze.rates_by_state(tiny_tables, bouts=bouts, n_resamples=FEW), "method", "state"
    )
    # Kay: one event (12 s) in the 5 s bout, five in the other 15 s, per session
    kay = rates.loc[KAY[0]]
    assert kay.n_events.tolist() == [10, 2]
    assert kay.rate.tolist() == pytest.approx([20.0, 12.0])
    # five network events per 15 s of rest, no theta burst in the bout
    assert kay.true_rate.tolist() == pytest.approx([20.0, 0.0])
    # Mallory, on its one session: the EMG burst (14 s) in the bout
    assert rates.loc[MALLORY[0], "n_events"].tolist() == [4, 1]
    assert rates.loc[MALLORY[0], "minutes"].tolist() == pytest.approx([0.25, 5 / 60])


def test_an_event_at_a_bout_end_is_running(analyze, tiny_tables):
    kay = tiny_tables.events[tiny_tables.events.method == KAY[0]]
    # Kay's last two events, in each session: one peaks where a bout starts,
    # the other where it ends
    bouts = {
        session: np.array(
            [[peaks.iloc[-2], peaks.iloc[-2] + 1.0], [peaks.iloc[-1] - 1.0, peaks.iloc[-1]]]
        )
        for session, peaks in kay.groupby("session_id").peak_time
    }
    assert bouts["reference/1"][0, 0] > UNIX_ORIGIN
    rates = _by(
        analyze.rates_by_state(tiny_tables, bouts=bouts, n_resamples=FEW), "method", "state"
    )
    assert rates.loc[KAY[0], "n_events"].tolist() == [8, 4]


def test_session_bouts_are_the_simulated_schedule(analyze, benchmark_import):
    conditions = benchmark_import("conditions")
    rng = np.random.default_rng(conditions.stage_seeds(3)[0])
    expected = conditions.running_schedule(200.0, rng)
    rest = 200.0 - np.diff(expected, axis=1).sum()
    sessions = pd.DataFrame(
        {"session_id": ["x/3"], "replicate": [3], "duration_s": [200.0], "rest_s": [rest]}
    )
    assert len(expected)
    assert np.array_equal(analyze.session_bouts(sessions)["x/3"], expected)
    # rest to the timestamps' rounding: a few units in the last place, not 1 us
    ulp = np.spacing(200.0)
    assert len(analyze.session_bouts(sessions.assign(rest_s=rest + 4 * ulp)))
    for wrong in (rest - 1, rest + 1e-7):
        with pytest.raises(ValueError, match="x/3: the running schedule drawn again"):
            analyze.session_bouts(sessions.assign(rest_s=wrong))


@pytest.fixture(scope="module")
def sliver_run(run, tmp_path_factory):
    """Two ripples; Kay finds the first at its bounds and the second by a
    sliver, Karlsson the first alone."""
    events = _event_table(run, [(k, "swr", "ripple", 0, 2.0 + 2 * k, 0.05) for k in range(2)])
    windows = _windows(events)
    sliver = [windows[1, 1] - 0.005, windows[1, 1] + 0.1]
    session = _one_session(
        events, {KAY: np.array([windows[0], sliver]), KARLSSON: windows[:1]}
    )
    return _write_run(run, tmp_path_factory.mktemp("sliver"), [session])


@pytest.fixture(scope="module")
def mixed_error_run(run, tmp_path_factory):
    """Three ripples Kay finds starting 20 ms early and 5 and 10 ms late, and
    one false positive."""
    events = _event_table(run, [(k, "swr", "ripple", 0, 1.0 + k, 0.05) for k in range(3)])
    windows = _windows(events)
    onsets = np.array([-0.02, 0.005, 0.01])
    kay = np.vstack([windows + np.column_stack([onsets, np.zeros(3)]), [[6.0, 6.1]]])
    session = _one_session(events, {KAY: kay})
    return _write_run(run, tmp_path_factory.mktemp("mixed"), [session])


def test_matching_sensitivity_by_hand(analyze, mixed_error_run):
    tables = analyze.load_run(mixed_error_run)
    matches = analyze.match_run(tables, levels=(0.0, 0.2, 0.5))
    points = pd.DataFrame(
        {
            "method": KAY[0],
            "minimum_iou": 0.0,
            "fp_target": analyze.FP_TARGETS,
            "recall": [0.1, 0.2, 0.3, 0.4],
        }
    )
    table = analyze.matching_sensitivity(tables, matches, points, n_resamples=FEW)
    row = table[table.minimum_iou == 0].iloc[0]
    # three of three found, three of four events true: F1 is 2 * 3 / (3 + 4)
    assert [row.recall, row.precision, row.f1] == pytest.approx([1.0, 0.75, 6 / 7])
    # the median of |-20|, 5 and 10 ms, not of the signed errors
    assert row.median_abs_onset_error == pytest.approx(0.01)
    assert row.median_abs_offset_error == pytest.approx(0.0, abs=1e-12)
    # each detector's recall at the target rates, from operating_points
    assert [row[f"recall_at_{target:g}"] for target in analyze.FP_TARGETS] == [
        0.1,
        0.2,
        0.3,
        0.4,
    ]
    assert np.isnan(table[table.minimum_iou == 0.5].iloc[0]["recall_at_1"])


def test_matching_sensitivity_levels(analyze, sliver_run):
    tables = analyze.load_run(sliver_run)
    matches = analyze.match_run(tables, levels=(0.0, 0.2, 0.5))
    table = analyze.matching_sensitivity(tables, matches, n_resamples=FEW)
    kay = _by(table[table.method == KAY[0]], "minimum_iou")
    assert kay.index.tolist() == [0.0, 0.2, 0.5]
    # the sliver counts at IoU 0 only
    assert kay.recall.tolist() == [1.0, 0.5, 0.5]
    assert kay.n_matched.tolist() == [2, 1, 1]
    assert kay.precision.tolist() == [1.0, 0.5, 0.5]
    assert kay.f1.tolist() == pytest.approx([1.0, 0.5, 0.5])
    # IoU 0.025 (5 of 200 ms) and 1: the IoU distribution behind the headline
    assert kay.loc[0.0, "median_iou"] == pytest.approx((0.025 + 1) / 2)
    assert kay.loc[0.2, ["iou_q25", "median_iou"]].tolist() == pytest.approx([1.0, 1.0])
    assert kay.recall_swr.tolist() == [1.0, 0.5, 0.5]
    # Karlsson ties Kay once the sliver is gone
    assert table.set_index(["method", "minimum_iou"])["rank"].to_dict() == {
        (KARLSSON[0], 0.0): 2,
        (KARLSSON[0], 0.2): 1,
        (KARLSSON[0], 0.5): 1,
        (KAY[0], 0.0): 1,
        (KAY[0], 0.2): 1,
        (KAY[0], 0.5): 1,
    }
    summary = analyze._summary(
        "sliver", tables, [], {"matching_sensitivity": table}, pd.DataFrame()
    )
    assert f"- `{KARLSSON[0]}` (ripple): 2, 1, 1\n" in summary
    changes = analyze.order_changes(table)
    assert changes.to_dict("records") == [
        {
            "method": KARLSSON[0],
            "setting": KARLSSON[1],
            "primary_expression": "ripple",
            "rank_0": 2,
            "rank_0.2": 1,
            "rank_0.5": 1,
        }
    ]


SWEPT_KARLSSON = "Karlsson_ripple_detector"


def _curve(condition, replicate, method, points):
    """A two-setting sweep of 10 truth windows over 10 minutes per session:
    ``points`` gives (matched, detected) at settings 2.0 and 3.0."""
    return [
        _kay(condition, replicate, setting, matched, detected, method=method)
        for setting, (matched, detected) in zip(("2.0", "3.0"), points, strict=True)
    ]


def _model_run(analyze, *, skip=()):
    """Kay and Karlsson in the reference and every alternative model but
    ``skip``, four replicates each. Kay's recall at 1 per minute is 0.7,
    Karlsson's 0.6, everywhere but under refractory spiking, where Kay's
    falls to 0.4; no curve reaches 5 per minute."""
    kay, karlsson = ((8, 28), (6, 11)), ((7, 27), (5, 10))
    counts = []
    for condition in ["reference", *(c for _, _, c in analyze.MODEL_ALTERNATIVES)]:
        if condition in skip:
            continue
        own = ((5, 25), (3, 8)) if condition == "spike_model=refractory" else kay
        for replicate in range(4):
            counts += _curve(condition, replicate, KAY[0], own)
            counts += _curve(condition, replicate, SWEPT_KARLSSON, karlsson)
            counts += [
                _kay(condition, replicate, "default", 6, 11),
                _kay(condition, replicate, "default", 5, 10, method=SWEPT_KARLSSON),
            ]
    return _hand_scores(analyze, counts)


def test_model_sensitivity_includes_all_variants(analyze):
    alternatives = [c for _, _, c in analyze.MODEL_ALTERNATIVES]
    assert len(alternatives) == 6
    scores = _model_run(analyze, skip=("envelope_power=quartic",))
    changes, orders = analyze.model_sensitivity(scores, n_resamples=FEW)
    assert list(dict.fromkeys(changes.alternative)) == alternatives
    # an alternative the run lacks is reported, every value missing: the
    # reference's own results cannot stand in for it
    missing = changes[changes.alternative == "envelope_power=quartic"]
    assert (missing.status == "not run").all()
    assert missing[["value", "change", "change_low"]].isna().all().all()
    assert "envelope_power=quartic" not in set(orders.alternative)
    at_fp = changes[changes.measure == "recall_at_fp"].set_index(
        ["alternative", "method", "fp_target"]
    )
    refractory = at_fp.loc[("spike_model=refractory", KAY[0], 1.0)]
    assert [refractory.reference_value, refractory.value] == pytest.approx([0.7, 0.4])
    assert [refractory.change, refractory.change_low, refractory.change_high] == pytest.approx(
        [-0.3, -0.3, -0.3]
    )
    assert (refractory.change_p, refractory.n_draws) == (0.0, FEW)
    assert at_fp.loc[("noise_modulation=varying", KAY[0], 1.0), "change"] == 0
    # 5 per minute is past every curve: missing, not the end of the curve
    unreachable = at_fp.xs(5.0, level="fp_target")
    assert (unreachable.status == "unattainable").all()
    assert unreachable[["change", "change_low", "change_p"]].isna().all().all()
    # Kay ahead of Karlsson in the reference, behind under refractory spiking
    order = orders.set_index(["alternative", "fp_target"])
    flipped = order.loc[("spike_model=refractory", 1.0)]
    assert (flipped.method_a, flipped.method_b) == (SWEPT_KARLSSON, KAY[0])
    assert [flipped.reference_difference, flipped.alternative_difference] == pytest.approx(
        [-0.1, 0.2]
    )
    assert flipped.supported
    assert flipped.reversed
    assert flipped.p_reversed == 1.0
    kept = order.loc[("noise_modulation=varying", 1.0)]
    assert kept.supported
    assert not kept.reversed
    assert np.isnan(order.loc[("noise_modulation=varying", 5.0), "reference_difference"])
    # the main settings' measures, paired the same way
    recall = changes[(changes.measure == "recall") & (changes.method == KAY[0])]
    assert recall.change.dropna().tolist() == [0.0] * 5
    nothing_moved = analyze.validation_changes(_checks().iloc[:0])
    lines = analyze.model_sensitivity_statements(changes, orders, nothing_moved)
    assert lines[0].startswith("- `strength_correlation=coupled` (validation: no target")
    assert "not in this run" in lines[-1]
    # a report never read is not a report where nothing moved
    unread = analyze.model_sensitivity_statements(changes, orders)
    assert unread[0].startswith(
        "- `strength_correlation=coupled` (validation: the report was not read, so what "
        "this alternative changes in it is unknown)"
    )
    refractory_lines = [
        line for line in lines if "refractory" in line or "reversed at" in line
    ]
    # reversed at 0.5, 1 and 2 per minute; 5 is out of reach for both detectors
    assert "of 3 reference orders" in refractory_lines[0]
    assert (
        "0 keep that support, 0 lose it (0 of them with the point estimates reversed), "
        "3 reverse and 0 cannot be compared here" in refractory_lines[0]
    )
    assert "2 detector targets are out of reach" in refractory_lines[0]
    assert refractory_lines[2].startswith("  - reversed at 1/min: `Karlsson_ripple_detector`")
    kept = next(line for line in lines if line.startswith("- `noise_modulation=varying`"))
    assert (
        "3 keep that support, 0 lose it (0 of them with the point estimates reversed), "
        "0 reverse and 0 cannot" in kept
    )


def test_model_sensitivity_keeps_failures_apart_from_unreachable_targets(analyze):
    scores = _model_run(analyze)
    counts = scores.counts
    # under coupled strengths both sweeps' second setting has 1 false positive
    # a minute, so 0.5 per minute is out of reach there alone
    shifted = (counts.session_id.str.startswith("strength_correlation=coupled")) & (
        counts.setting == "3.0"
    )
    counts = counts.assign(
        n_detected=np.where(shifted, counts.n_matched + 10, counts.n_detected)
    )
    scores = dataclasses.replace(scores, counts=counts)
    # under refractory spiking Kay's whole sweep fails, and Karlsson's default
    refractory = [f"spike_model=refractory/{replicate}" for replicate in range(4)]
    scores = _failed(scores, refractory, KAY[0], ["2.0", "3.0"])
    scores = _failed(scores, refractory, SWEPT_KARLSSON, ["default"])
    changes, orders = analyze.model_sensitivity(scores, n_resamples=FEW)
    mine = changes[changes.alternative == "spike_model=refractory"].set_index(
        ["method", "measure", "fp_target"]
    )
    kay = mine.loc[KAY[0]].loc["recall_at_fp"]
    # missing because Kay failed, never "unattainable", at every target
    assert kay.status.unique().tolist() == ["failed"]
    assert kay[
        ["n_failures", "n_replicates", "n_dropped"]
    ].drop_duplicates().to_numpy().tolist() == [[8, 0, 4]]
    karlsson = mine.loc[SWEPT_KARLSSON]
    assert karlsson.loc[("recall_at_fp", 5.0), ["status", "n_failures"]].tolist() == [
        "unattainable",
        0,
    ]
    recall = karlsson.xs("recall", level=0)
    assert recall[["status", "n_failures"]].to_numpy().tolist() == [["failed", 4]]
    assert (mine.loc[KAY[0]].xs("recall", level=0).n_failures == 0).all()
    order = orders.set_index(["alternative", "fp_target"])
    failed = order.loc[("spike_model=refractory", 1.0)]
    assert (failed.status, failed.n_failures_a, failed.n_failures_b) == ("failed", 0, 8)
    assert order.loc[("strength_correlation=coupled", 0.5), "status"] == "unattainable"
    assert order.loc[("strength_correlation=coupled", 1.0), "status"] == "compared"
    lines = analyze.model_sensitivity_statements(changes, orders)
    coupled = next(line for line in lines if "`strength_correlation=coupled`" in line)
    assert (
        "of 3 reference orders of detectors by recall at a common false-positive rate that "
        "their intervals support, 2 keep that support, 0 lose it (0 of them with the point "
        "estimates reversed), 0 reverse and 1 cannot be compared here (0 for a failure, 1 "
        "out of reach) and 0 more are untested" in coupled
    )
    # Kay failed on every replicate there, so no order has a reference
    # difference over replicates both ran on: untested, at every target
    spiking = next(line for line in lines if "`spike_model=refractory`" in line)
    assert "of 0 reference orders" in spiking
    assert "and 4 more are untested because a detector failed" in spiking
    assert "0 of 1 main settings' recall compared moves" in spiking
    assert "(1 failed)" in spiking
    assert "1 detector targets are out of reach in one condition and 4 missing" in spiking


# Per target, A minus B in the reference and in the alternative, each as its
# estimate and the range its resamples span evenly: a supported order the
# alternative reverses with an interval excluding 0, one whose point estimate
# alone reverses, an unsupported one the alternative orders the other way, one
# that keeps its support, one whose alternative point estimate lies below 0 but
# its interval above (the order survives, not reversed), and an unsupported one
# whose point estimates alone have opposite signs.
_HAND_ORDERS = (
    ((0.1, 0.05, 0.15), (-0.1, -0.15, -0.05)),
    ((0.1, 0.05, 0.15), (-0.02, -0.1, 0.05)),
    ((0.02, -0.05, 0.1), (-0.1, -0.15, -0.05)),
    ((0.1, 0.05, 0.15), (0.08, 0.03, 0.12)),
    ((0.1, 0.05, 0.15), (-0.01, 0.021, 0.049)),
    ((0.02, -0.05, 0.1), (-0.02, -0.1, 0.05)),
)
_HAND_TARGETS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)


def _hand_orders(analyze, alternative="spike_model=refractory"):
    """``_orders`` of Karlsson (A) and Kay (B) over ``_HAND_ORDERS``: B's
    recall 0.5 everywhere, A's 0.5 plus the difference, 41 resamples."""
    n_resamples = 41
    estimate = np.full((2, 2, len(_HAND_TARGETS)), 0.5)
    draws = np.full((n_resamples, 2, 2, len(_HAND_TARGETS)), 0.5)
    for t, conditions in enumerate(_HAND_ORDERS):
        for c, (value, low, high) in enumerate(conditions):
            estimate[c, 0, t] += value
            draws[:, c, 0, t] += np.linspace(low, high, n_resamples)
    found = analyze.SweepRecalls(
        estimate=estimate,
        draws=draws,
        defined=np.full((2, 2, len(_HAND_TARGETS)), 4),
        events=np.full((2, 2, len(_HAND_TARGETS)), 20.0),
        replicates=[{0, 1, 2, 3}, {0, 1, 2, 3}],
        nearest=np.full((2, 2, len(_HAND_TARGETS)), "2.0", dtype=object),
        read_off=np.full((2, 2, len(_HAND_TARGETS)), "interpolated", dtype=object),
    )
    detectors = [SWEPT_KARLSSON, KAY[0]]
    return analyze._orders(
        alternative,
        detectors,
        pd.Series("ripple", index=detectors),
        _HAND_TARGETS,
        {(0, 1): found},
        4,
        dict.fromkeys(detectors, 0),
    )


def test_a_reversal_needs_the_alternative_interval_to_exclude_zero(analyze):
    orders = _hand_orders(analyze)
    assert list(orders.columns) == list(analyze.ORDER_COLUMNS)
    orders = orders.set_index("fp_target")
    assert orders.supported.tolist() == [True, True, False, True, True, False]
    # a supported order whose alternative interval lies on the other side of 0
    assert orders.reversed.tolist() == [True, False, False, False, False, False]
    # opposite point estimates, the alternative's interval holding 0,
    # supported or not
    assert orders.point_reversed.tolist() == [False, True, False, False, False, True]
    assert orders.loc[1.0, "alternative_low"] < 0 < orders.loc[1.0, "alternative_high"]


def test_statements_count_point_reversals_as_lost_support(analyze):
    alternative = "spike_model=refractory"
    orders = _hand_orders(analyze, alternative)
    changes = pd.DataFrame(
        [
            {
                "alternative": alternative,
                "status": "compared",
                "method": KAY[0],
                "measure": "recall",
                "change": 0.0,
                "change_low": -0.1,
                "change_high": 0.1,
            }
        ]
    )
    lines = analyze.model_sensitivity_statements(changes, orders)
    mine = [line for line in lines if alternative in line or line.startswith("  - ")]
    assert (
        "of 4 reference orders of detectors by recall at a common false-positive rate that "
        "their intervals support, 2 keep that support, 1 lose it (1 of them with the point "
        "estimates reversed), 1 reverse and 0 cannot be compared here" in mine[0]
    )
    assert mine[1].startswith(
        f"  - reversed at 0.5/min: `{SWEPT_KARLSSON}` minus `{KAY[0]}` +0.100"
    )
    assert mine[2].startswith(
        f"  - point estimate reversed at 1/min: `{SWEPT_KARLSSON}` minus `{KAY[0]}` +0.100"
    )
    assert "the order loses its support" in mine[2]
    assert len(mine) == 3
    # a candidate trend of the supported reversal alone
    trends = analyze.candidate_trends({"model_sensitivity_orders": orders})
    reversals = trends[trends.kind == "model_order_reversal"]
    assert reversals.statement.str.startswith("At 0.5 false positives").tolist() == [True]


def _checks():
    """A validation report's checks: the refractory model moves the rate by
    2 % and the width by less than 1 %."""
    return pd.DataFrame(
        {
            "check": ["rate", "rate", "rate", "width", "width", "noise"],
            "kind": ["target"] * 5 + ["rendering"],
            "condition_id": [
                "reference",
                "spike_model=refractory",
                "noise_modulation=varying",
                "reference",
                "spike_model=refractory",
                "spike_model=refractory",
            ],
            "statistic": ["mean_hz"] * 3 + ["median_ms"] * 2 + ["z"],
            "observed": [11.81, 11.57, 11.81, 44.0, 44.2, 3.0],
        }
    )


def _report(root, checks, *, spec_hash=None, checks_hash=None, listed=True):
    """A run directory whose ``run_spec.json`` names a validation report
    written beside it (by absolute path), with the hashes given or the files'
    own; its spec lists ``checks.csv`` unless not ``listed``."""
    import hashlib
    import json

    report = root / "validation"
    report.mkdir(parents=True)
    checks.to_csv(report / "checks.csv", index=False)
    digest = hashlib.sha256((report / "checks.csv").read_bytes()).hexdigest()
    spec = {"artifacts": {"checks.csv": checks_hash or digest} if listed else {}}
    (report / "spec.json").write_text(json.dumps(spec))
    identity = {
        "path": str(report / "spec.json"),
        "sha256": spec_hash or hashlib.sha256((report / "spec.json").read_bytes()).hexdigest(),
    }
    run_directory = root / "run"
    run_directory.mkdir()
    (run_directory / "run_spec.json").write_text(json.dumps({"validation_report": identity}))
    return run_directory


def test_the_validation_report_is_read_from_the_run_spec(analyze, tmp_path):
    changed, problem = analyze._validation(_report(tmp_path / "good", _checks()))
    assert problem == ""
    assert changed[["alternative", "check"]].to_numpy().tolist() == [
        ["spike_model=refractory", "rate"]
    ]
    # every way a report can be missing or not the one the run used says so
    cases = {
        "no run_spec.json": tmp_path / "nowhere",
        "differs from the one the run was checked against": _report(
            tmp_path / "spec", _checks(), spec_hash="0" * 64
        ),
        "checks.csv does not match its recorded hash": _report(
            tmp_path / "checks", _checks(), checks_hash="0" * 64
        ),
    }
    missing = _report(tmp_path / "missing", _checks())
    (missing.parent / "validation" / "checks.csv").unlink()
    cases["checks.csv is missing"] = missing
    cases["does not list checks.csv"] = _report(tmp_path / "unlisted", _checks(), listed=False)
    for reason, run_directory in cases.items():
        changed, problem = analyze._validation(run_directory)
        assert changed is None
        assert reason in problem


def test_the_command_reads_the_run_s_validation_report(analyze, two_condition_run, tmp_path):
    import shutil

    report = _report(tmp_path / "report", _checks())
    run_directory = tmp_path / "run"
    shutil.copytree(two_condition_run.parent, run_directory)
    shutil.copy(report / "run_spec.json", run_directory / "run_spec.json")
    model = [
        analysis
        for analysis in analyze.ANALYSES
        if analysis.name in ("model_sensitivity", "model_sensitivity_orders")
    ]
    results = tmp_path / "results"
    analyze.analyze_run(run_directory, results, figures=False, analyses=model, n_resamples=FEW)
    summary = (results / "summary.md").read_text()
    assert "- `spike_model=refractory` (validation: rate 11.81 to 11.57)" in summary
    assert "The validation report was not read" not in summary


def test_validation_changes_list_moved_statistics(analyze):
    changed = analyze.validation_changes(_checks())
    assert changed.to_dict("records") == [
        {
            "alternative": "spike_model=refractory",
            "check": "rate",
            "statistic": "mean_hz",
            "reference": 11.81,
            "observed": 11.57,
        }
    ]
    scores = _model_run(analyze)
    changes, orders = analyze.model_sensitivity(scores, n_resamples=FEW)
    lines = analyze.model_sensitivity_statements(changes, orders, changed)
    assert any("(validation: rate 11.81 to 11.57)" in line for line in lines)


ROUMIS = "Roumis_ripple_detector"


def _operating_run(analyze):
    """Four reference replicates: Kay's recall at 1 false positive a minute
    is 0.7, Karlsson's 0.6, and Roumis's curve (3 and 5 per minute) never
    comes down to it."""
    counts = []
    for replicate in range(4):
        counts += _curve("reference", replicate, KAY[0], ((8, 28), (6, 11)))
        counts += _curve("reference", replicate, SWEPT_KARLSSON, ((7, 27), (5, 10)))
        counts += _curve("reference", replicate, ROUMIS, ((8, 58), (6, 36)))
    return _hand_scores(analyze, counts)


def test_operating_differences_are_paired_by_session(analyze):
    table = analyze.operating_differences(_operating_run(analyze), n_resamples=FEW)
    at_one = table[table.fp_target == 1.0].set_index(["method_a", "method_b"])
    # Karlsson first by name: its recall minus Kay's
    row = at_one.loc[(SWEPT_KARLSSON, KAY[0])]
    assert [row.recall_a, row.recall_b] == pytest.approx([0.6, 0.7])
    # every session's curve gives the same difference, so every resample does
    assert [row.difference, row.difference_low, row.difference_high] == pytest.approx(
        [-0.1, -0.1, -0.1]
    )
    # the p-value from those same resamples: none at or above 0
    assert (row.difference_p, row.n_draws, row.n_replicates) == (0.0, FEW, 4)
    assert (row.read_off_a, row.read_off_b) == ("interpolated", "interpolated")
    unreached = at_one.loc[(KAY[0], ROUMIS)]
    assert (unreached.reached_a, unreached.reached_b) == (True, False)
    assert (unreached.read_off_a, unreached.read_off_b) == ("interpolated", "")
    # 5 per minute is Roumis's first setting's rate, past Kay's curve
    at_five = table[table.fp_target == 5.0].set_index(["method_a", "method_b"])
    assert at_five.loc[(KAY[0], ROUMIS), ["read_off_a", "read_off_b"]].tolist() == [
        "",
        "tested",
    ]
    assert np.isnan(unreached.difference)
    assert np.isnan(unreached.difference_p)
    assert unreached.n_draws == 0


def test_operating_differences_test_the_resamples_of_their_interval(analyze):
    """Kay's recall at 1 false positive a minute is 0.7 in every session,
    Karlsson's 0.8, 0.6, 0.7 and 0.5, at the same false-positive rates, so a
    resample's pooled difference is its weighted mean of +0.1, -0.1, 0 and
    -0.2: the p-value is from those same resamples."""
    counts = []
    for replicate, (first, second) in enumerate([(9, 7), (7, 5), (8, 6), (6, 4)]):
        counts += _curve("reference", replicate, KAY[0], ((8, 28), (6, 11)))
        counts += _curve(
            "reference", replicate, SWEPT_KARLSSON, ((first, first + 20), (second, second + 5))
        )
    table = analyze.operating_differences(_hand_scores(analyze, counts), n_resamples=FEW)
    row = table[table.fp_target == 1.0].iloc[0]
    assert row.difference == pytest.approx(-0.05)
    draws = analyze.resample_weights(4, n_resamples=FEW) @ np.array([0.1, -0.1, 0.0, -0.2]) / 4
    p, n_draws = analyze.bootstrap_p(draws[:, np.newaxis])
    assert row.difference_p == pytest.approx(p[0])
    assert row.n_draws == n_draws[0] == FEW
    assert 0 < row.difference_p < 1
    low, high = analyze.percentile_intervals(draws[:, np.newaxis])
    assert [row.difference_low, row.difference_high] == pytest.approx([low[0], high[0]])


def test_operating_order_trend_is_the_paired_difference(analyze):
    scores = _operating_run(analyze)
    trends = analyze.candidate_trends(
        {
            "operating_points": analyze.operating_points(scores, n_resamples=FEW),
            "operating_differences": analyze.operating_differences(scores, n_resamples=FEW),
        }
    )
    row = trends[trends.kind == "operating_order"].iloc[0]
    # the leader is the detector with the best recall, the difference paired
    assert row.method == KAY[0]
    assert [row.value, row.low, row.high, row.p] == pytest.approx([0.1, 0.1, 0.1, 0.0])
    assert (
        f"{KAY[0]} minus {SWEPT_KARLSSON} +0.100 (+0.100, +0.100), bootstrap p < 2/{FEW} "
        f"(approximate, from the interval's {FEW} resamples) over 4 sessions, paired."
    ) in row.statement
    # a detector whose curve does not reach the target is named, not dropped
    assert f"{ROUMIS} does not reach 1 false positive a minute" in row.statement
    # the settings to look at: each curve's nearest to 1 per minute (2 and 0.5
    # are as near; the first in threshold order)
    assert [row.spot_method_a, row.spot_setting_a] == [KAY[0], "2.0"]
    assert [row.spot_method_b, row.spot_setting_b] == [SWEPT_KARLSSON, "2.0"]
    assert row.scoring == "interval"


def test_candidate_trends_carry_their_evidence(analyze):
    robustness = analyze.robustness(_snr_run(analyze), n_resamples=FEW)
    changes, orders = analyze.model_sensitivity(
        _model_run(analyze, skip=("envelope_power=quartic",)), n_resamples=FEW
    )
    ranks = pd.DataFrame(
        {
            "method": [KAY[0]] * 3,
            "setting": ["default"] * 3,
            "primary_expression": ["ripple"] * 3,
            "minimum_iou": [0.0, 0.2, 0.5],
            "rank": [1, 2, 5],
        }
    )
    recall = robustness[robustness.measure == "recall"]
    # the same rows for a point method
    points = recall.assign(method=DAVIDSON[0], setting=DAVIDSON[1], scoring="peak_containment")
    trends = analyze.candidate_trends(
        {
            "robustness_recall": pd.concat([recall, points], ignore_index=True),
            "model_sensitivity": changes,
            "model_sensitivity_orders": orders,
            "matching_sensitivity": ranks,
        }
    )
    assert trends[trends.kind == "matching_rank"].statement.tolist() == [
        (
            f"{KAY[0]}'s rank by recall among ripple methods moves from 1 at IoU 0 to 5 at "
            "0.5 (descriptive: ranks carry no interval or test)."
        )
    ]
    robust = trends[(trends.kind == "robustness") & (trends.method == KAY[0])]
    # the two levels whose change excludes 0, the reference level never
    assert robust.condition_id.tolist() == ["ripple_snr=low", "ripple_snr=high"]
    assert robust.value.tolist() == pytest.approx([-0.2, 0.2])
    assert robust.spot_selection.tolist() == ["missed", "found"]
    assert robust.statement.iloc[0] == (
        f"{KAY[0]}'s recall against ripple changes by -0.200 (-0.200, -0.200) from the "
        "reference to ripple_snr=low."
    )
    # one method and its setting per spot column
    assert robust[
        ["spot_method_a", "spot_setting_a", "spot_method_b", "spot_setting_b"]
    ].drop_duplicates().to_numpy().tolist() == [[KAY[0], "default", "", ""]]
    assert (robust.scoring == "interval").all()
    point = trends[(trends.kind == "robustness") & (trends.method == DAVIDSON[0])].iloc[0]
    assert point.scoring == "peak_containment"
    assert point.spot_setting_a == "literature"
    assert (
        f"{DAVIDSON[0]}'s recall against ripple by peak containment changes" in point.statement
    )
    reversals = trends[trends.kind == "model_order_reversal"]
    assert set(reversals.condition_id) == {"spike_model=refractory"}
    assert set(reversals.spot_method_a) == {SWEPT_KARLSSON}
    assert set(reversals.spot_method_b) == {KAY[0]}
    # at 0.5 per minute both detectors' 3.0 setting is exactly there
    half = reversals[reversals.statement.str.startswith("At 0.5 ")].iloc[0]
    assert [half.spot_setting_a, half.spot_setting_b] == ["3.0", "3.0"]
    assert list(trends.columns) == list(analyze.TREND_COLUMNS)
    assert "spot_event_type" not in trends.columns


def test_a_spot_check_knows_which_methods_failed(benchmark_import, tiny_run):
    spot_check = benchmark_import("spot_check")
    # Mallory failed on the second session only: its lane says so there
    assert spot_check.session_failures(tiny_run.parent, "reference/1") == {MALLORY}
    assert spot_check.session_failures(tiny_run.parent, "reference/0") == set()


def _trend_rows(method, **columns):
    return {"method": method, "setting": "default", **columns}


def test_every_kind_of_candidate_trend(analyze):
    bias = pd.DataFrame(
        [
            _trend_rows(
                "a", ratio_of_means=1.2, ratio_of_means_low=1.1, ratio_of_means_high=1.3
            ),
            # its interval holds 1: no trend
            _trend_rows(
                "b", ratio_of_means=1.05, ratio_of_means_low=0.9, ratio_of_means_high=1.2
            ),
        ]
    )
    effect = pd.DataFrame(
        [
            # all units moved, principal ones not: no trend about principal units
            _trend_rows(
                "a",
                selection="all",
                mean_difference=2.0,
                mean_difference_low=1.0,
                mean_difference_high=3.0,
            ),
            _trend_rows(
                "a",
                selection="principal",
                mean_difference=0.5,
                mean_difference_low=-0.5,
                mean_difference_high=1.5,
            ),
            _trend_rows(
                "b",
                selection="principal",
                mean_difference=-1.0,
                mean_difference_low=-2.0,
                mean_difference_high=-0.5,
            ),
        ]
    )
    model = pd.DataFrame(
        [
            _trend_rows(
                method,
                alternative="spike_model=refractory",
                status="compared",
                measure="recall",
                scoring="interval",
                change=change,
                change_low=change - 0.05,
                change_high=change + 0.05,
                change_p=0.01,
            )
            for method, change in (("a", -0.2), ("b", 0.2))
        ]
    )
    # thirteen large changes in one condition and one small in another: the
    # small one is still among the twelve kept, one per condition first
    robust = pd.DataFrame(
        [
            _trend_rows(
                f"m{k}",
                level="low",
                condition_id=condition,
                primary_expression="ripple",
                scoring="interval",
                change=change,
                change_low=change - 0.01,
                change_high=change + 0.01,
                change_p=0.01,
            )
            for k, (condition, change) in enumerate(
                [("ripple_snr=low", 0.5)] * 13 + [("participation=low", 0.05)]
            )
        ]
    )
    trends = analyze.candidate_trends(
        {
            "participation_bias": bias,
            "boundary_effect": effect,
            "model_sensitivity": model,
            "robustness_recall": robust,
        }
    )
    kinds = trends.groupby("kind").method.apply(list).to_dict()
    assert kinds["participation_bias"] == ["a"]
    assert kinds["boundary_effect"] == ["b"]
    assert "principal units" in trends[trends.kind == "boundary_effect"].statement.iloc[0]
    changed = trends[trends.kind == "model_change"].set_index("method")
    # a fall is looked at in the events missed, a rise in those found
    assert changed.spot_selection.to_dict() == {"b": "found", "a": "missed"}
    robustness = trends[trends.kind == "robustness"]
    assert len(robustness) == 12
    assert "participation=low" in set(robustness.condition_id)


def test_summary_accounts_for_point_and_single_sample_methods(analyze, point_matched):
    summary = analyze._summary("points", point_matched[0], [], {}, None)
    listed = f"Point inventories (`{DAVIDSON[0]}`; the catalog's output " + '"ripple peaks")'
    assert listed in summary
    assert (
        f"Interval methods whose events can be one sample long (`{LEE[0]}`) keep the "
        "interval rule" in summary
    )


def test_select_events_for_a_spot_check(analyze, tiny_tables, point_matched):
    select = functools.partial(analyze.select_from, tiny_tables)
    # Kay's one event over the doublet matches one of its two ripples
    missed = select(*KAY, "missed")
    assert missed.label.tolist() == ["ripple_doublet", "ripple_doublet"]
    assert missed.session_id.tolist() == ["reference/0", "reference/1"]
    assert missed.start_time.iloc[1] > UNIX_ORIGIN
    assert len(select(*KAY, "found")) == 6
    assert len(select(*KAY, "found", event_type="ripple_doublet")) == 2
    false = select(*KAY, "false_positive")
    assert false.label.tolist() == ["burst_only:burst", "spike_leakage", "background"] * 2
    assert (
        select(*KAY, "false_positive", event_type="burst_only").label.tolist()
        == ["burst_only:burst"] * 2
    )
    # Mallory, on the session it ran: the weak ripple's burst
    assert select(*MALLORY, "missed")[["session_id", "label"]].to_numpy().tolist() == [
        ["reference/0", "weak_ripple"]
    ]
    # a point method by containment
    assert len(analyze.select_from(point_matched[0], *DAVIDSON, "false_positive")) == 4
    with pytest.raises(ValueError, match="selection must be one of"):
        select(*KAY, "early")
    with pytest.raises(ValueError, match=r"no Kay_ripple_detector \(8\.0\)"):
        select(KAY[0], "8.0", "missed")


def _hand_groups(analyze, tiny_tables):
    """Hand tables for grouping: two sessions, the second at a Unix clock
    origin, and methods whose events agree or differ as their names say."""
    origin = UNIX_ORIGIN
    two = [(1.0, 2.0), (3.0, 4.0)]
    one = [(origin + 1.0, origin + 2.0)]
    shifted = [(origin + 1.0, np.nextafter(origin + 2.0, np.inf))]
    # method: (primary expression, scoring, events on s0, events on s1; None failed)
    spec = {
        "a": ("ripple", "interval", two, one),
        "b_same_as_a": ("ripple", "interval", two[::-1], one),
        "c_one_ulp_off": ("ripple", "interval", two, shifted),
        "d_failed_on_s1": ("ripple", "interval", two, None),
        "e_same_as_d": ("ripple", "interval", two, None),
        "f_burst": ("burst", "interval", two, one),
        "g_none": ("ripple", "interval", [], []),
        "h_none": ("ripple", "interval", [], []),
        "i_points": ("ripple", "peak_containment", two, one),
    }
    sessions = ["s0", "s1"]
    ran, events = [], []
    for method, (_, _, *outputs) in spec.items():
        for session_id, bounds in zip(sessions, outputs, strict=True):
            if bounds is None:
                continue
            ran.append((session_id, method, "default"))
            events += [(session_id, method, "default", *pair) for pair in bounds]
    return dataclasses.replace(
        tiny_tables,
        sessions=pd.DataFrame({"session_id": sessions}),
        methods=pd.DataFrame(
            [
                (method, "default", primary, scoring)
                for method, (primary, scoring, *_) in spec.items()
            ],
            columns=["method", "setting", "primary_expression", "scoring"],
        ),
        ran=pd.DataFrame(ran, columns=["session_id", "method", "setting"]),
        events=pd.DataFrame(
            events, columns=["session_id", "method", "setting", "start_time", "end_time"]
        ),
    )


def test_identical_groups_by_hand(analyze, tiny_tables):
    groups = analyze.identical_groups(_hand_groups(analyze, tiny_tables))
    assert groups.group.tolist() == [
        "a",
        "a",
        "c_one_ulp_off",
        "d_failed_on_s1",
        "d_failed_on_s1",
        "f_burst",
        "g_none",
        "g_none",
        "i_points",
    ]
    by = groups.set_index("method")
    assert by.loc["b_same_as_a", "members"] == "a b_same_as_a"
    assert by.loc["e_same_as_d", "members"] == "d_failed_on_s1 e_same_as_d"
    assert by.n_members.tolist() == [2, 2, 1, 2, 2, 1, 2, 2, 1]


def test_a_group_lists_its_stand_ins(analyze):
    assert analyze.method_stand_ins(KAY[0]) == ()
    assert analyze.method_stand_ins("recipe:liu_2019_awake") == (
        "pyramidal",
        "behavior_intervals",
    )
    stand_ins = analyze._group_stand_ins
    assert stand_ins([KAY[0], "recipe:gillespie_2021"]) == ""
    assert stand_ins(["recipe:grosmark_2016", "recipe:yang_2024"]) == (
        "pyramidal sleep_intervals behavior_intervals external_ripples"
    )
    # members that differ are listed one by one
    assert stand_ins(["recipe:liu_2019", "recipe:liu_2019_awake"]) == (
        "recipe:liu_2019 (pyramidal sleep_intervals); "
        "recipe:liu_2019_awake (pyramidal behavior_intervals)"
    )
    assert stand_ins([KAY[0], MALLORY[0]]) == (f"{KAY[0]} (none); {MALLORY[0]} (pyramidal)")


@pytest.fixture(scope="module")
def tiny_levels(analyze, tiny_tables):
    """The tiny run matched at every minimum IoU, and running bouts: one ends
    where Kay's first false positive's time is (its peak, here its bounds'
    midpoint), one where its last's is, and one starts 0.5 ms past its middle
    one's, each session's from its own clock origin."""
    matches = analyze.match_run(tiny_tables, levels=(0.0, 0.2, 0.5))
    unmatched = matches.false_positives
    bouts = {}
    for session_id, origin in (("reference/0", 0.0), ("reference/1", UNIX_ORIGIN)):
        kay = unmatched[(unmatched.session_id == session_id) & (unmatched.method == KAY[0])]
        times = ((kay.start_time + kay.end_time) / 2).sort_values().to_numpy()
        bouts[session_id] = np.array(
            [
                [origin + 5.9, times[0]],
                [times[1] + 0.0005, origin + 13.0],
                [origin + 15.0, times[2]],
            ]
        )
    return matches, bouts


def test_false_positive_rates_place_detections_by_their_time(
    analyze, tiny_tables, tiny_levels
):
    matches, bouts = tiny_levels
    rates = analyze.false_positive_rates(tiny_tables, matches, bouts=bouts, n_resamples=FEW)
    rates = rates.set_index("method")
    # Kay, per session: the false positive over the burst-only event and the
    # last one peak where a bout ends (closed: running); the leakage one
    # starts before the second bout, its peak 0.5 ms before it (rest)
    kay = rates.loc[KAY[0]]
    assert [kay.n_unmatched, kay.n_unmatched_rest, kay.n_unmatched_running] == [6, 2, 4]
    minutes = (20.0 - tiny_tables.sessions.event_time_s.to_numpy()) / 60
    # the bouts last 2.1495 s, 0.1 s of it inside the burst-only event's window
    running = 2 * 2.0495 / 60
    assert kay.minutes == pytest.approx(minutes.sum())
    assert kay.running_minutes == pytest.approx(running, abs=1e-6)
    assert kay.rest_minutes == pytest.approx(minutes.sum() - running, abs=1e-6)
    assert kay.unmatched_per_minute == pytest.approx(6 / minutes.sum())
    assert kay.unmatched_rest_per_rest_minute == pytest.approx(2 / kay.rest_minutes)
    # Mallory, on the session it ran: the EMG's and the last, both at rest
    mallory = rates.loc[MALLORY[0]]
    assert [mallory.n_unmatched_rest, mallory.n_unmatched_running] == [2, 0]
    assert mallory.minutes == pytest.approx(minutes[0])
    assert (mallory.n_sessions, mallory.n_failures) == (1, 1)
    # the overall rate is the appendix's against the primary expression,
    # interval too: the same counts, minutes and resamples
    appendix = analyze.appendix_expressions(tiny_tables, matches, n_resamples=FEW)
    primary = appendix[appendix.primary].set_index("method")
    for method in (KAY[0], MALLORY[0]):
        found, expected = rates.loc[method], primary.loc[method]
        assert found.n_unmatched == expected.n_detected - expected.n_matched
        for part in ("", "_low", "_high"):
            assert (
                found[f"unmatched_per_minute{part}"]
                == (expected[f"false_positives_per_minute{part}"])
            )


def test_false_positive_rates_place_by_peak_else_midpoint(analyze, tiny_tables, tiny_levels):
    """As rates_by_state places events: by the peak where there is one."""
    matches, bouts = tiny_levels
    events = tiny_tables.events.copy()
    kay = events.method == KAY[0]
    # the leakage false positive (11.999 to 12.001 s) peaks at its end, inside
    # the bout that starts 0.5 ms past its midpoint; the last has no peak, so
    # its midpoint, on a bout's end, places it
    leakage = kay & np.isclose(events.start_time % 100, 11.999)
    last = kay & np.isclose(events.start_time % 100, 16.0)
    assert leakage.sum() == last.sum() == 2
    events.loc[leakage, "peak_time"] = events.loc[leakage, "end_time"]
    events.loc[last, "peak_time"] = np.nan
    moved = dataclasses.replace(tiny_tables, events=events)
    rates = analyze.false_positive_rates(moved, matches, bouts=bouts, n_resamples=FEW)
    kay_rates = rates.set_index("method").loc[KAY[0]]
    assert [kay_rates.n_unmatched_rest, kay_rates.n_unmatched_running] == [0, 6]


def test_false_positive_rates_by_default_draw_the_bouts_again(analyze, tiny_tables):
    # the tiny run's sessions are too short for a running bout: all rest
    rates = analyze.false_positive_rates(
        tiny_tables, analyze.match_run(tiny_tables), n_resamples=FEW
    )
    assert (rates.n_unmatched_running == 0).all()
    assert rates.rest_minutes.tolist() == rates.minutes.tolist()
    wrong = dataclasses.replace(tiny_tables, sessions=tiny_tables.sessions.assign(rest_s=19.0))
    with pytest.raises(ValueError, match="the running schedule drawn again"):
        analyze.false_positive_rates(wrong, analyze.match_run(tiny_tables))


@pytest.fixture(scope="module")
def tiny_compact(analyze, tiny_tables, tiny_levels):
    matches, bouts = tiny_levels
    sensitivity = analyze.matching_sensitivity(tiny_tables, matches, n_resamples=FEW)
    errors = analyze.boundary_errors(tiny_tables, matches, n_resamples=FEW)
    compact = analyze.compact_comparison(
        tiny_tables, matches, sensitivity, errors, bouts=bouts, n_resamples=FEW
    )
    return compact, sensitivity, errors


def test_what_was_not_computed_is_refused(analyze, tiny_tables, tiny_levels, tiny_compact):
    # matched at IoU 0 alone, as match_run matches by default
    with pytest.raises(ValueError, match=r"not formed at minimum IoU \[0\.2, 0\.5\]"):
        analyze.matching_sensitivity(tiny_tables, analyze.match_run(tiny_tables))
    matches, bouts = tiny_levels
    _, sensitivity, errors = tiny_compact
    without = errors[~np.isclose(errors.fraction, 0.5)]
    with pytest.raises(ValueError, match=r"no errors at \[50\] % of the peak"):
        analyze.compact_comparison(
            tiny_tables, matches, sensitivity, without, bouts=bouts, n_resamples=FEW
        )


def test_compact_comparison_takes_its_numbers_from_their_tables(analyze, tiny_compact):
    compact, sensitivity, errors = tiny_compact
    assert list(compact.columns) == list(analyze.COMPACT_COLUMNS)
    assert compact[
        ["primary_expression", "method", "members", "n_members"]
    ].to_numpy().tolist() == [
        ["ripple", KAY[0], KAY[0], 1],
        ["burst", MALLORY[0], MALLORY[0], 1],
    ]
    assert compact.stand_in_inputs.tolist() == ["", "pyramidal"]
    assert compact[["n_sessions", "n_failures"]].to_numpy().tolist() == [[2, 0], [1, 1]]
    for row in compact.to_dict("records"):
        method = row["method"]
        for level in (0.0, 0.5):
            source = sensitivity[
                (sensitivity.method == method) & (sensitivity.minimum_iou == level)
            ].iloc[0]
            for name, parts in (
                ("n_matched", ("",)),
                ("recall", ("", "_low", "_high")),
                ("precision", ("", "_low", "_high")),
            ):
                for part in parts:
                    column = f"{name}_iou{level:g}{part}"
                    assert row[column] == source[f"{name}{part}"], (method, column)
        at_zero = sensitivity[(sensitivity.method == method) & (sensitivity.minimum_iou == 0)]
        assert row["n_reference"] == at_zero.n_reference.iloc[0]
        assert row["n_detected"] == at_zero.n_detected.iloc[0]
        signed = errors[
            (errors.method == method)
            & (errors.expression == row["primary_expression"])
            & (errors.measure == "signed")
        ]
        for fraction, percent in ((0.1, 10), (0.5, 50)):
            for boundary in ("onset", "offset"):
                source = signed[
                    np.isclose(signed.fraction, fraction) & (signed.boundary == boundary)
                ].iloc[0]
                name = f"{boundary}_error_{percent}"
                assert [row[name], row[f"{name}_low"], row[f"{name}_high"]] == [
                    source["median"],
                    source.median_low,
                    source.median_high,
                ]
                assert row["n_pairs"] == source.n_pairs
    # Kay, per session: three of four ripples at IoU 0; its event over the
    # doublet overlaps each ripple by IoU 0.41, below 0.5
    kay = compact.iloc[0]
    assert [kay.n_reference, kay.n_matched_iou0, kay["n_matched_iou0.5"]] == [8, 6, 4]


def test_compact_points_are_the_point_inventories(analyze, point_matched):
    tables, matches = point_matched
    points = analyze.point_inventories(tables, matches, n_resamples=FEW)
    compact = analyze.compact_points(tables, points)
    assert list(compact.columns) == list(analyze.COMPACT_POINT_COLUMNS)
    row, source = compact.iloc[0], points.iloc[0]
    assert len(compact) == 1
    assert (row.method, row.members, row.stand_in_inputs) == (DAVIDSON[0], DAVIDSON[0], "")
    assert [row.n_reference, row.n_detected, row.n_matched, row.n_unmatched] == [8, 10, 6, 4]
    for name in ("recall", "precision"):
        for part in ("", "_low", "_high"):
            assert row[f"{name}{part}"] == source[f"{name}{part}"]
    for part in ("", "_low", "_high"):
        assert (
            row[f"unmatched_per_minute{part}"] == source[f"false_positives_per_minute{part}"]
        )


def test_compact_held_out_sets_measured_beside_interpolated(analyze):
    scores = _operating_run(analyze)
    thresholds = analyze.held_out_thresholds(scores, n_resamples=FEW)
    points = analyze.operating_points(scores, n_resamples=FEW)
    table = analyze.compact_held_out(thresholds, points)
    assert list(table.columns) == list(analyze.COMPACT_HELD_OUT_COLUMNS)
    assert len(table) == 2 * len(thresholds)
    rows = table.set_index(["method", "fp_target", "source"])
    held = rows.loc[(KAY[0], 1.0, "held_out")]
    chosen = thresholds[(thresholds.method == KAY[0]) & (thresholds.fp_target == 1.0)].iloc[0]
    assert (held.kind, held.setting) == ("measured", chosen.setting)
    assert [held.recall, held.unmatched_per_minute, held.n_sessions] == [
        chosen.recall,
        chosen.false_positives_per_minute,
        chosen.n_held_out_sessions,
    ]
    read = rows.loc[(KAY[0], 1.0, "operating_point")]
    at_one = points[(points.method == KAY[0]) & (points.minimum_iou == 0)]
    at_one = at_one[at_one.fp_target == 1.0].iloc[0]
    assert (read.kind, read.setting) == ("interpolated", "")
    assert [read.recall, read.recall_low, read.recall_high] == [
        at_one.recall,
        at_one.recall_low,
        at_one.recall_high,
    ]
    assert np.isnan(read.unmatched_per_minute)
    # Roumis's curve never comes down to 1 per minute; 5 is a setting's rate
    assert rows.loc[(ROUMIS, 1.0, "held_out"), "kind"] == ""
    assert rows.loc[(ROUMIS, 1.0, "operating_point"), "kind"] == ""
    assert rows.loc[(ROUMIS, 5.0, "operating_point"), "kind"] == "tested"
    assert rows.loc[(ROUMIS, 5.0, "held_out"), "kind"] == "measured"


def test_compact_page_formats_every_number(analyze, tiny_tables, tiny_compact):
    compact, _, _ = tiny_compact
    results = {
        f"compact_{target}": compact[compact.primary_expression == target]
        for target in analyze.COMPACT_TARGETS
    }
    page = analyze.compact_page("tiny", tiny_tables, results)
    kay = compact.iloc[0]
    assert page.startswith("# Compact comparison: tiny\n")
    assert "reference condition's 2 simulated sessions (20 s each)" in page
    assert (
        f"| `{KAY[0]}` | 1 | {kay.recall_iou0:.2f} / {kay['recall_iou0.5']:.2f} "
        f"| {kay.precision_iou0:.2f} / {kay['precision_iou0.5']:.2f} "
        f"| {kay.unmatched_per_minute:#.3g} | {kay.unmatched_rest_per_rest_minute:#.3g} "
        f"| {1000 * kay.onset_error_10:+.1f} / {1000 * kay.onset_error_50:+.1f} "
        f"| {1000 * kay.offset_error_10:+.1f} / {1000 * kay.offset_error_50:+.1f} |"
    ) in page
    assert f"| `{MALLORY[0]}` (1 failed) | 1 |" in page
    assert f"- `{MALLORY[0]}`: pyramidal" in page
    # no method is headlined against the network in the tiny run
    network = page.split("## Network", 1)[1].split("## Definitions", 1)[0]
    assert "0 methods in 0 groups." in network
    assert network.count("- none") == 2
    for phrase in (
        "post hoc, not predeclared",
        "its peak, else the midpoint of its bounds",
        "`interpolated`",
        "this simulator's reference sessions",
        "`n_unmatched_running`",
    ):
        assert phrase in page
    # formatted numbers only, never a float's full repr
    assert not re.search(r"\.\d{5,}", page)


@pytest.fixture(scope="module")
def two_condition_run(run, tmp_path_factory):
    """The tiny run's sessions in the reference (Mallory failing on the
    second) and, with Kay's sweep point finding nothing, under refractory
    spiking."""
    root = tmp_path_factory.mktemp("two")
    _write_run(
        run,
        root,
        [_tiny_session(run, 0.0), _tiny_session(run, UNIX_ORIGIN, mallory_fails=True)],
    )
    sessions = [_tiny_session(run, 0.0), _tiny_session(run, UNIX_ORIGIN)]
    for session in sessions:
        session["detected"][KAY_SWEEP] = np.empty((0, 2))
    return _write_run(run, root, sessions, condition_id="spike_model=refractory")


def test_the_command_writes_every_table_and_the_summary(analyze, two_condition_run, tmp_path):
    results = tmp_path / "results" / "tiny"
    seconds = analyze.analyze_run(
        two_condition_run.parent, results, figures=False, n_resamples=FEW
    )
    names = [analysis.name for analysis in analyze.ANALYSES]
    assert list(seconds) == ["load", "match", "scores", *names]
    assert sorted(path.name for path in results.iterdir()) == sorted(
        [
            *(f"{name}.csv" for name in names),
            "candidate_trends.csv",
            "compact.md",
            "summary.md",
        ]
    )
    summary = (results / "summary.md").read_text()
    for analysis in analyze.ANALYSES:
        assert f"- `{analysis.name}.csv`: {analysis.description}" in summary
    assert "- `compact.md`: The compact tables" in summary
    # the compact tables hold the numbers of the tables written beside them
    compact = pd.read_csv(results / "compact_ripple.csv", keep_default_na=False)
    sensitivity = pd.read_csv(results / "matching_sensitivity.csv")
    kay = sensitivity[sensitivity.method == KAY[0]].set_index("minimum_iou")
    assert compact.loc[0, ["recall_iou0", "recall_iou0.5"]].tolist() == [
        kay.loc[0.0, "recall"],
        kay.loc[0.5, "recall"],
    ]
    assert "## Ripple" in (results / "compact.md").read_text()
    assert "2 sessions of reference, 2 methods" in summary
    # the run directory read, wherever it is
    assert f"on `{two_condition_run.resolve().as_posix()}/`: 2 sessions" in summary
    for heading in (
        "## Recall changing by more than 0.1 across a factor",
        "## Order changes with the minimum IoU",
        "## Model sensitivity",
        "## Held-out thresholds",
        "## Trends and spot checks",
    ):
        assert heading in summary
    assert "Point inventories (none in this run;" in summary
    assert (
        "Across every condition, 1 calls failed (sweeps included), by method, setting and "
        f"condition:\n\n- `{MALLORY[0]}` (literature), `reference`: 1 sessions" in summary
    )
    # the run has no validation report: the summary says so, never "nothing moved"
    assert "- `spike_model=refractory` (validation: the report was not read" in summary
    assert "The validation report was not read (" in summary
    assert "- `envelope_power=quartic`: not in this run" in summary
    # the tables across conditions hold the second condition
    robust = pd.read_csv(results / "robustness_recall.csv")
    assert robust[["factor", "level"]].drop_duplicates().to_numpy().tolist() == [
        ["spike_model", "reference"],
        ["spike_model", "refractory"],
    ]
    model = pd.read_csv(results / "model_sensitivity.csv", keep_default_na=False)
    assert set(model.status) == {"compared", "not run", "unattainable"}
    trends = pd.read_csv(results / "candidate_trends.csv")
    assert list(trends.columns) == list(analyze.TREND_COLUMNS)
    assert (
        f"- `{MALLORY[0]}` (literature): 1 of 2 sessions; ValueError: made to fail" in summary
    )
    # the tables read back as the functions give them
    profile = pd.read_csv(results / "detection_profile.csv")
    assert profile.recall.tolist()[:5] == [1.0, 1.0, 1.0, 1.0, 0.0]
    failures = pd.read_csv(results / "failures.csv", keep_default_na=False)
    assert failures[["method", "n_sessions", "n_failures"]].to_numpy().tolist() == [
        [KAY[0], 2, 0],
        [MALLORY[0], 1, 1],
    ]


def test_the_command_keeps_the_hand_written_spot_checks(analyze, tiny_run, tmp_path):
    results = tmp_path / "results"
    (results / "spot_checks").mkdir(parents=True)
    (results / "spot_checks" / "kay_missed_weak.png").write_bytes(b"figure")
    (results / "trends.md").write_text("# Trends\n\nKay misses weak ripples.\n")
    (results / "stale.csv").write_text("an earlier analysis\n")
    # attribution.py's results, another command's, in a nested directory
    (results / "attribution" / "nested").mkdir(parents=True)
    (results / "attribution" / "spikes_sobol.csv").write_text("y,factor\n")
    (results / "attribution" / "nested" / "figure.png").write_bytes(b"bars")
    failures = next(a for a in analyze.ANALYSES if a.name == "failures")
    analyze.analyze_run(tiny_run.parent, results, figures=False, analyses=[failures])
    # the tables are rebuilt; what was written by hand or by another command is
    # carried over
    assert not (results / "stale.csv").exists()
    assert (results / "spot_checks" / "kay_missed_weak.png").read_bytes() == b"figure"
    assert (results / "trends.md").read_text() == "# Trends\n\nKay misses weak ripples.\n"
    assert (results / "attribution" / "spikes_sobol.csv").read_text() == "y,factor\n"
    assert (results / "attribution" / "nested" / "figure.png").read_bytes() == b"bars"
    summary = (results / "summary.md").read_text()
    assert "[trends.md](trends.md)" in summary
    assert "carried over" in summary
    assert "`attribution/`" in summary
    # without them, the summary says there are none yet
    fresh = tmp_path / "fresh"
    analyze.analyze_run(tiny_run.parent, fresh, figures=False, analyses=[failures])
    assert not (fresh / "spot_checks").exists()
    assert not (fresh / "attribution").exists()
    assert "No `trends.md` has been written yet" in (fresh / "summary.md").read_text()


def test_a_rebuild_keeps_what_is_written_while_it_runs(analyze, tiny_run, tmp_path):
    """attribution.py's results, or an edit to trends.md, written into the
    results while a rebuild runs, are in the rebuilt directory."""
    results = tmp_path / "results"
    (results / "attribution").mkdir(parents=True)
    (results / "attribution" / "spikes_oat.csv").write_text("before\n")
    (results / "trends.md").write_text("# Trends\n")

    def table(inputs):
        (results / "attribution" / "lfp_sobol.csv").write_text("during\n")
        (results / "attribution" / "spikes_oat.csv").write_text("rewritten\n")
        (results / "trends.md").write_text("# Trends\n\nEdited during the rebuild.\n")
        return pd.DataFrame({"x": [1]})

    analyze.analyze_run(
        tiny_run.parent, results, figures=False, analyses=[analyze.Analysis("x", table, "A.")]
    )
    assert (results / "x.csv").exists()
    assert (results / "attribution" / "lfp_sobol.csv").read_text() == "during\n"
    assert (results / "attribution" / "spikes_oat.csv").read_text() == "rewritten\n"
    assert (results / "trends.md").read_text() == "# Trends\n\nEdited during the rebuild.\n"


def test_the_analysis_registry_is_checked(analyze, tiny_run, tmp_path):
    def table(inputs):
        return pd.DataFrame()

    def figure(table):
        raise AssertionError  # never drawn here

    with pytest.raises(ValueError, match="a figure needs its description"):
        analyze.Analysis("x", table, "A table.", figure)
    with pytest.raises(ValueError, match="a figure description without a figure"):
        analyze.Analysis("x", table, "A table.", None, "A figure.")
    for reserved in ("load", "match", "scores", "candidate_trends"):
        with pytest.raises(ValueError, match=f"'{reserved}' is reserved"):
            analyze.Analysis(reserved, table, "A table.")
    twice = [analyze.Analysis("x", table, "A table.")] * 2
    results = tmp_path / "results"
    with pytest.raises(ValueError, match=r"The analysis names repeat: \['x'\]"):
        analyze.analyze_run(tiny_run.parent, results, figures=False, analyses=twice)
    assert not results.exists()
    # the tables the trends and the summary read are ones the command writes
    names = {analysis.name for analysis in analyze.ANALYSES}
    read = {
        analyze.ROBUSTNESS_RECALL,
        analyze.MODEL_CHANGES,
        analyze.MODEL_ORDERS,
        analyze.MATCHING,
        analyze.PARTICIPATION_BIAS,
        analyze.BOUNDARY_EFFECT,
        analyze.OPERATING_POINTS,
        analyze.OPERATING_DIFFERENCES,
    }
    assert read <= names
    assert len(names) == len(analyze.ANALYSES)


def test_a_file_over_the_limit_stops_the_command(analyze, tiny_run, tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "earlier.csv").write_text("kept\n")
    huge = analyze.Analysis(
        "huge", lambda inputs: pd.DataFrame({"x": np.arange(300_000)}), "Too big."
    )
    with pytest.raises(ValueError, match=r"huge\.csv would be .* over the 1,000,000-byte"):
        analyze.analyze_run(tiny_run.parent, results, figures=False, analyses=[huge])
    # the results in place are left as they were
    assert [path.name for path in results.iterdir()] == ["earlier.csv"]


def test_the_command_refuses_no_workers(analyze, capsys):
    with pytest.raises(SystemExit):
        analyze.main(["--run-name", "v1", "--workers", "0"])
    assert "--workers must be at least 1" in capsys.readouterr().err

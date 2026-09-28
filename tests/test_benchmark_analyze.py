"""The benchmark's analyses (examples/benchmark/analyze.py): the paired bootstrap
and sign-flip test, the held-out split and the results size limit; loading a run
written in the runner's schema, with hand-chosen truth and events, one of its two
sessions at a Unix clock origin."""

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
    ``duration`` in seconds, and ``detected``, each (method, setting)'s
    ``[start, end]`` rows, None for a call that failed. Scores come from the
    runner's own ``score_events``."""
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
            events.append(
                detected.assign(
                    **key,
                    event_index=np.arange(len(detected)),
                    peak_time=detected.mean(axis=1),
                    n_active_units=0,
                    n_active_principal=0,
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
                truth_counts=pd.DataFrame(columns=list(run.TRUTH_COUNT_COLUMNS)),
                ripple_channels=pd.DataFrame(columns=list(run.RIPPLE_CHANNEL_COLUMNS)),
                units=pd.DataFrame(columns=list(run.UNIT_COLUMNS)),
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
    listed = {"condition_id": condition_id, "factor": "reference", "level": "reference"}
    pd.DataFrame([{**listed, "params": "{}"}]).to_csv(root / "conditions.csv", index=False)
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


def test_sign_flip_exact(analyze):
    assert analyze.sign_flip_test([1, 1, 1, 1]) == 2 / 16
    assert analyze.sign_flip_test([1, -1]) == 1.0
    # 17 sessions and more: random flips, never 0 (no flip of 999 matches all 20)
    assert analyze.sign_flip_test(np.ones(20), n_resamples=999) == 1 / 1000


def test_sign_flip_needs_finite_pairs(analyze):
    for differences in ([np.nan, 0.0], [np.nan, np.nan], [np.inf, 1.0]):
        with pytest.raises(ValueError, match="finite paired differences"):
            analyze.sign_flip_test(differences)
    assert np.isnan(analyze.sign_flip_test([]))


def test_held_out_membership_is_the_same_in_every_condition(analyze):
    reference, other = range(20), range(10)
    held_out = {k for k in reference if analyze.is_held_out(k)}
    assert {k for k in other if analyze.is_held_out(k)} == {1, 3, 5, 7, 9}
    assert all(analyze.is_held_out(k) == (k in held_out) for k in other)
    assert len(held_out) == 10
    assert sum(analyze.is_held_out(k) for k in other) == 5


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
    for table in (tiny_tables.metrics, tiny_tables.events, tiny_tables.ran):
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
    # read in chunks, events and scores equal the whole tables' main rows
    for name, loaded in (
        ("events.csv.gz", tiny_tables.events),
        ("metrics.csv.gz", tiny_tables.metrics),
    ):
        whole = analyze.main_rows(run.read_table(tiny_run / name)).reset_index(drop=True)
        pd.testing.assert_frame_equal(loaded, whole)
    assert tiny_tables.events.end_time.max() > UNIX_ORIGIN


def test_loading_an_unknown_condition_raises(analyze, tiny_run):
    with pytest.raises(ValueError, match=r"no session of the conditions \['ripple_snr=low'\]"):
        analyze.load_run(tiny_run, conditions=["reference", "ripple_snr=low"])

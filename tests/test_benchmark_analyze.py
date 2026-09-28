"""The benchmark's analyses (examples/benchmark/analyze.py): the paired bootstrap
and sign-flip test, the held-out split and the results size limit; loading, matching
and every analysis on a run written in the runner's schema, with hand-chosen truth
and events, one of its two sessions at a Unix clock origin; the command's files.
No test draws a figure."""

import dataclasses
import functools
import inspect
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


@pytest.fixture(scope="module")
def tiny_matches(analyze, tiny_tables):
    return analyze.match_run(tiny_tables)


def test_matching_again_gives_the_runner_scores(analyze, tiny_tables, tiny_matches):
    pairs = tiny_matches.pairs
    reference = tiny_matches.windows.groupby(["session_id", "expression"]).size()
    metrics = tiny_tables.metrics[tiny_tables.metrics.minimum_iou == 0]
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
    windows = rd.truth_windows(events, 0.1, "ripple")[["start_time", "end_time"]].to_numpy()
    late = windows + np.array([[0.0], [0.0], [0.0], [0.02], [0.02], [0.02]])
    session = {
        "events": events,
        "non_events": _non_event_tables(_one_non_event_table("emg", center_time=9.0)),
        "duration": 10.0,
        "detected": {KAY: windows[:3], KARLSSON: late},
    }
    return _write_run(run, tmp_path_factory.mktemp("timing"), [session])


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


def _quick(analysis):
    """``analysis`` with few resamples, where its table takes any."""
    if "n_resamples" not in inspect.signature(analysis.table).parameters:
        return analysis
    return dataclasses.replace(
        analysis, table=functools.partial(analysis.table, n_resamples=FEW)
    )


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


def test_point_inventories_are_scored_apart(analyze, point_run):
    tables = analyze.load_run(point_run)
    matches = analyze.match_run(tables)
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


def _kay(condition, replicate, setting, matched, detected, reference=10, **extra):
    return {
        "condition_id": condition, "replicate": replicate, "method": KAY[0],
        "setting": setting, "n_reference": reference, "n_detected": detected,
        "n_matched": matched, **extra,
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
        assert (row.change_p, row.n_paired) == (2 / 16, 4)
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
            counts.append({**_kay(condition, replicate, "default", 5, 12), "method": "Roumis"})
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
    assert (kay.n_replicates, kay.n_dropped, kay.n_paired, kay.n_failures) == (2, 2, 2, 2)
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
    ripples = rd.truth_windows(events, 0.1, "ripple")[["start_time", "end_time"]].to_numpy()
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
def spiking_run(run, tmp_path_factory):
    sessions = [_spiking_session(run, 0.0), _spiking_session(run, UNIX_ORIGIN)]
    return _write_run(run, tmp_path_factory.mktemp("spiking"), sessions)


def test_boundary_effect_is_zero_for_equal_bounds(analyze, spiking_run):
    tables = analyze.load_run(spiking_run)
    matches = analyze.match_run(tables)
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


def test_participation_bias_by_hand(analyze, spiking_run):
    tables = analyze.load_run(spiking_run)
    matches = analyze.match_run(tables)
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
    with pytest.raises(ValueError, match="x/3: the running schedule drawn again"):
        analyze.session_bouts(sessions.assign(rest_s=rest - 1))


@pytest.fixture(scope="module")
def sliver_run(run, tmp_path_factory):
    """Two ripples; Kay finds the first at its bounds and the second by a
    sliver, Karlsson the first alone."""
    events = _event_table(run, [(k, "swr", "ripple", 0, 2.0 + 2 * k, 0.05) for k in range(2)])
    windows = rd.truth_windows(events, 0.1, "ripple")[["start_time", "end_time"]].to_numpy()
    sliver = [windows[1, 1] - 0.005, windows[1, 1] + 0.1]
    session = {
        "events": events,
        "non_events": _non_event_tables(_one_non_event_table("emg", center_time=9.0)),
        "duration": 10.0,
        "detected": {KAY: np.array([windows[0], sliver]), KARLSSON: windows[:1]},
    }
    return _write_run(run, tmp_path_factory.mktemp("sliver"), [session])


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
        {**_kay(condition, replicate, setting, matched, detected), "method": method}
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
                {**_kay(condition, replicate, "default", 6, 11)},
                {**_kay(condition, replicate, "default", 5, 10), "method": SWEPT_KARLSSON},
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
    assert (refractory.change_p, refractory.n_paired) == (2 / 16, 4)
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
    lines = analyze.model_sensitivity_statements(changes, orders)
    assert lines[0].startswith("- `strength_correlation=coupled` (validation: no target")
    assert "not in this run" in lines[-1]
    refractory_lines = [
        line for line in lines if "refractory" in line or "reversed at" in line
    ]
    # reversed at 0.5, 1 and 2 per minute; 5 is out of reach for both detectors
    assert "of 3 reference orders" in refractory_lines[0]
    assert "0 keep that support, 0 lose it and 3 reverse" in refractory_lines[0]
    assert "2 detector targets are out of reach" in refractory_lines[0]
    assert refractory_lines[2].startswith("  - reversed at 1/min: `Karlsson_ripple_detector`")
    kept = next(line for line in lines if line.startswith("- `noise_modulation=varying`"))
    assert "3 keep that support, 0 lose it and 0 reverse" in kept


def test_validation_changes_list_moved_statistics(analyze):
    checks = pd.DataFrame(
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
    changed = analyze.validation_changes(checks)
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
    trends = analyze.candidate_trends(
        {
            "robustness_recall": robustness[robustness.measure == "recall"],
            "model_sensitivity": changes,
            "model_sensitivity_orders": orders,
            "matching_sensitivity": ranks,
        }
    )
    assert trends[trends.kind == "matching_rank"].statement.tolist() == [
        f"{KAY[0]}'s rank by recall among ripple methods moves from 1 at IoU 0 to 5 at 0.5."
    ]
    robust = trends[trends.kind == "robustness"]
    # the two levels whose change excludes 0, the reference level never
    assert robust.condition_id.tolist() == ["ripple_snr=low", "ripple_snr=high"]
    assert robust.value.tolist() == pytest.approx([-0.2, 0.2])
    assert robust.spot_selection.tolist() == ["missed", "found"]
    assert robust.statement.iloc[0] == (
        f"{KAY[0]}'s recall against ripple changes by -0.200 (-0.200, -0.200) from the "
        "reference to ripple_snr=low."
    )
    reversals = trends[trends.kind == "model_order_reversal"]
    assert set(reversals.condition_id) == {"spike_model=refractory"}
    assert set(reversals.spot_methods) == {f"{SWEPT_KARLSSON} {KAY[0]}"}
    assert list(trends.columns) == list(analyze.TREND_COLUMNS)


def test_select_events_for_a_spot_check(analyze, tiny_tables, point_run):
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
    points = analyze.load_run(point_run)
    assert len(analyze.select_from(points, *DAVIDSON, "false_positive")) == 4
    with pytest.raises(ValueError, match="selection must be one of"):
        select(*KAY, "early")
    with pytest.raises(ValueError, match=r"no Kay_ripple_detector \(8\.0\)"):
        select(KAY[0], "8.0", "missed")


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
    analyses = [_quick(analysis) for analysis in analyze.ANALYSES]
    seconds = analyze.analyze_run(
        two_condition_run.parent, results, figures=False, analyses=analyses
    )
    names = [analysis.name for analysis in analyze.ANALYSES]
    assert list(seconds) == ["load", "match", "scores", *names]
    assert sorted(path.name for path in results.iterdir()) == sorted(
        [*(f"{name}.csv" for name in names), "candidate_trends.csv", "summary.md"]
    )
    summary = (results / "summary.md").read_text()
    for analysis in analyze.ANALYSES:
        assert f"- `{analysis.name}.csv`: {analysis.description}" in summary
    assert "2 sessions of reference, 2 methods" in summary
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
    assert "- `spike_model=refractory` (validation: no target statistic" in summary
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

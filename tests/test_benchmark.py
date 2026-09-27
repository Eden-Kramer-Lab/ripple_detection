"""The benchmark runner (examples/benchmark/run.py): what one session writes, its
scores against ``match_events``, exact round trips of results and truth, failures
recorded rather than raised, and the command line's resume and combine rules, on
short simulated sessions with a few methods."""

import dataclasses
import json
import shutil
import subprocess
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ripple_detection as rd

UNIX_ORIGIN = 1_700_000_000.0
SHORT = {"session.duration_s": 60.0}
# Two detectors, a sweep point of one, and two recipes: Gridchyn 2020 (adaptive,
# with diagnostics and a pre-rest baseline) and Karlsson 2009.
METHODS = (
    ("Kay_ripple_detector", "default"),
    ("Kay_ripple_detector", "3.0"),
    ("Shvartsman_ripple_detector", "default"),
    ("recipe:gridchyn_2020", "literature"),
    ("recipe:karlsson_2009", "literature"),
)
LEVELS = (0.0, 0.2, 0.5)
EXPRESSIONS = ("ripple", "sharp_wave", "burst", "network")

# The output contract's columns, table by table.
SESSION_COLUMNS = [
    "session_id",
    "condition_id",
    "replicate",
    "seed",
    "duration_s",
    "rest_s",
    "event_time_s",
    *(f"n_events_{kind}" for kind in rd.EVENT_TYPES),
    *(f"n_non_events_{kind}" for kind in rd.NON_EVENT_TYPES),
    "simulate_s",
    "detect_s",
]
TRUTH_COLUMNS = [
    "session_id",
    "table",
    "id",
    "type",
    "expression",
    "component",
    "center_time",
    "rise_sigma",
    "decay_sigma",
    "envelope_power",
    "amplitude",
    "frequency_start",
    "frequency_end",
    "participation",
    "n_participants",
    "frequency",
    "snr_band_low",
    "snr_band_high",
    "channel",
    "n_units",
    "n_spikes",
    "isi",
]
TRUTH_COUNT_COLUMNS = [
    "session_id",
    "expression",
    "row",
    "n_active_units",
    "n_active_principal",
]
RIPPLE_CHANNEL_COLUMNS = ["session_id", "event_id", "component", "channel", "gain", "delay_s"]
UNIT_COLUMNS = ["session_id", "unit", "unit_type", "baseline_rate"]
METHOD_COLUMNS = [
    "session_id",
    "method",
    "setting",
    "doi",
    "role",
    "inventory",
    "stage",
    "primary_expression",
    "resolved_options",
    "input_policy",
    "assumptions",
    "interpretation",
]
EVENT_COLUMNS = [
    "session_id",
    "method",
    "setting",
    "event_index",
    "start_time",
    "end_time",
    "peak_time",
    "n_active_units",
    "n_active_principal",
]
METRIC_COLUMNS = [
    "session_id",
    "method",
    "setting",
    "expression",
    "minimum_iou",
    "n_reference",
    "n_detected",
    "n_matched",
    "recall",
    "precision",
    "f1",
    "false_positives_per_minute",
    "median_iou",
    "median_coverage",
    "median_temporal_precision",
    "median_onset_error_10",
    "median_offset_error_10",
    "median_onset_error_25",
    "median_offset_error_25",
    "median_onset_error_50",
    "median_offset_error_50",
    "median_abs_onset_error_10",
    "median_abs_offset_error_10",
    "median_abs_onset_error_25",
    "median_abs_offset_error_25",
    "median_abs_onset_error_50",
    "median_abs_offset_error_50",
    "n_split",
    "n_merged",
]
FAILURE_COLUMNS = ["session_id", "method", "setting", "error"]
WARNING_COLUMNS = ["session_id", "method", "setting", "category", "message"]
CONDITION_COLUMNS = ["condition_id", "factor", "level", "params"]
MANIFEST_KEYS = {
    "run_name",
    "git_commit",
    "package_version",
    "numpy_version",
    "scipy_version",
    "command",
    "started",
    "finished",
    "n_workers",
}

SWEEP = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0)
THRESHOLD_SWEEPS = {
    "Kay_ripple_detector": ("zscore_threshold", SWEEP),
    "Karlsson_ripple_detector": ("zscore_threshold", SWEEP),
    "Roumis_ripple_detector": ("zscore_threshold", SWEEP),
    "Shvartsman_ripple_detector": ("zscore_threshold", SWEEP),
    "multiunit_HSE_detector": ("zscore_threshold", SWEEP),
    "Zugaro_ripple_detector": ("high_threshold", (2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)),
    "Carey_candidate_detector": ("high_threshold", (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)),
    "Yu_ripple_detector": ("percentile", (99.0, 99.5, 99.9, 99.95, 99.99, 99.995, 99.999)),
    "Long_sharp_wave_ripple_detector": (
        "peak_thresholds",
        (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0),
    ),
}
DETECTOR_EXPRESSION = {
    "Kay_ripple_detector": "ripple",
    "Karlsson_ripple_detector": "ripple",
    "Roumis_ripple_detector": "ripple",
    "Shvartsman_ripple_detector": "ripple",
    "Yu_ripple_detector": "ripple",
    "Zugaro_ripple_detector": "ripple",
    "Long_sharp_wave_ripple_detector": "ripple",
    "Carey_candidate_detector": "network",
    "multiunit_HSE_detector": "burst",
}
# Recipes whose recordings take spike counts: population, participation,
# baseline and example-ripple inputs.
COUNTED_RECIPES = (
    "mallory_2025",
    "krause_2022",
    "gridchyn_2020",
    "carey_2019",
    "davidson_2009",
)

# For the command line: three conditions, one replicate of 30 s, one method, and
# the commit the runs record, whatever state the checkout running the tests is in.
RUN_CONDITIONS = ["reference", "ripple_snr=high", "emg_rate=3"]
RUN_METHODS = (("Kay_ripple_detector", "default"),)
COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(scope="module")
def run(benchmark_import):
    return benchmark_import("run")


@pytest.fixture(scope="module")
def conditions_module(benchmark_import):
    return benchmark_import("conditions")


@pytest.fixture(scope="module")
def validation(benchmark_import):
    return benchmark_import("validate_simulator")


@pytest.fixture(scope="module")
def recipe_configs(benchmark_import):
    return benchmark_import("recipe_configs")


@pytest.fixture(scope="module")
def reference(conditions_module):
    return conditions_module.conditions()[0]


@pytest.fixture(scope="module")
def session(conditions_module, reference):
    """Replicate 0 of the reference, one minute: one running bout between rests."""
    return conditions_module.simulate_condition(reference, 0, SHORT)


@pytest.fixture(scope="module")
def output(run, reference):
    return run.run_session(reference, 0, methods=METHODS, overrides=SHORT)


@pytest.fixture(scope="module")
def written(run, output, tmp_path_factory):
    directory = tmp_path_factory.mktemp("condition") / "reference"
    run.write_condition(directory, [output])
    return directory


def _median(values):
    """The median, NaN for no values (NumPy would warn)."""
    values = np.asarray(values, dtype=float)
    return np.median(values) if len(values) else np.nan


def _union_seconds(bounds):
    """The union's length, merged one interval at a time."""
    total, end = 0.0, -np.inf
    for start, stop in sorted(map(tuple, bounds)):
        if start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def _call(run, session, method, setting):
    """The result of the runner's call of ``method`` at ``setting``."""
    prepare = next(c for c in run.method_calls(session) if c[:2] == (method, setting))[2]
    return prepare()()


def test_the_sweeps_and_expressions_are_the_contracts(run):
    assert run.MATCH_IOU_LEVELS == LEVELS
    assert run.TRUTH_FRACTIONS == (0.1, 0.25, 0.5)
    assert run.THRESHOLD_SWEEPS == THRESHOLD_SWEEPS
    assert run.DETECTOR_EXPRESSION == DETECTOR_EXPRESSION
    assert set(run.THRESHOLD_SWEEPS) == set(rd.DETECTORS)


def test_every_method_and_setting_has_one_record_and_one_call(run, recipe_configs, session):
    records = run.method_records()
    pairs = [(record["method"], record["setting"]) for record in records]
    assert len(pairs) == len(set(pairs))
    expected = [
        (name, setting)
        for name in rd.DETECTORS
        for setting in ("default", *(repr(float(v)) for v in THRESHOLD_SWEEPS[name][1]))
    ] + [(f"recipe:{config.config_id}", "literature") for config in recipe_configs.RECIPES]
    assert pairs == expected
    assert [call[:2] for call in run.method_calls(session)] == pairs
    recipes = {f"recipe:{config.config_id}": config for config in recipe_configs.RECIPES}
    for record in records:
        if record["method"] in recipes:
            assert record == recipe_configs.method_record(recipes[record["method"]])
        else:
            assert record["primary_expression"] == DETECTOR_EXPRESSION[record["method"]]
            assert record["stage"] == "detection"
            assert record["doi"] == record["role"] == record["interpretation"] == ""
    # Long's raw input is channel 0, which under a local profile need not carry a ripple
    long = next(r for r in records if r["method"] == "Long_sharp_wave_ripple_detector")
    policy = json.loads(long["input_policy"])
    assert policy["positional"]["raw_lfp"] == "channel 0 of session.lfps, unfiltered"
    np.testing.assert_array_equal(session.raw_lfp, session.lfps[:, 0])


@pytest.mark.parametrize(
    ("method", "setting", "expected"),
    [
        ("Kay_ripple_detector", "default", {}),
        ("Kay_ripple_detector", "2.5", {"zscore_threshold": 2.5}),
        ("Zugaro_ripple_detector", "4.0", {"high_threshold": 4.0, "low_threshold": 2.0}),
        ("Carey_candidate_detector", "1.5", {"high_threshold": 1.5, "low_threshold": 1.0}),
        ("Yu_ripple_detector", "99.95", {"percentile": 99.95}),
        (
            "Long_sharp_wave_ripple_detector",
            "2.5",
            {"sharp_wave_thresholds": [0.5, 2.5], "ripple_thresholds": [0.5, 2.5]},
        ),
    ],
)
def test_a_setting_resolves_every_tunable(run, session, method, setting, expected):
    record = next(
        r for r in run.method_records() if (r["method"], r["setting"]) == (method, setting)
    )
    options = json.loads(record["resolved_options"])
    defaults = json.loads(json.dumps(rd.get_detector(method).parameters))
    assert options == {**defaults, **expected}
    assert "peak_thresholds" not in options
    # the call runs with them, and its result records them
    assert _call(run, session, method, setting).attrs == {
        "method": method,
        "options": options,
        "ripple_detection_version": rd.__version__,
    }


@pytest.mark.parametrize(
    ("method", "setting", "options"),
    [
        ("Kay_ripple_detector", "1.5", {"zscore_threshold": 1.5}),
        ("Zugaro_ripple_detector", "2.5", {"high_threshold": 2.5}),
        ("Yu_ripple_detector", "99.0", {"percentile": 99.0}),
        (
            "Long_sharp_wave_ripple_detector",
            "1.5",
            {"sharp_wave_thresholds": (0.5, 1.5), "ripple_thresholds": (0.5, 1.5)},
        ),
    ],
)
def test_sweep_values_and_speed_reach_the_detectors(run, session, method, setting, options):
    """A sweep point's events are the detector's, called directly on the
    session's signals and speed with the swept value."""
    detector = getattr(rd, method)
    fs = session.sampling_frequency
    if method == "Long_sharp_wave_ripple_detector":
        signals, keywords = (session.raw_lfp,), {"sharp_wave_lfp": session.sharp_wave_lfp}
    else:
        signals, keywords = (rd.filter_ripple_band(session.lfps, fs),), {}

    def direct(speed, **given):
        return detector(session.time, *signals, speed, fs, **keywords, **given)

    expected = direct(session.speed, **options)
    pd.testing.assert_frame_equal(
        _call(run, session, method, setting), expected, check_exact=True
    )
    # at this point the value and the speed both change the events
    assert not direct(session.speed).equals(expected)
    assert not direct(np.zeros_like(session.speed), **options).equals(expected)


def test_an_unknown_method_raises(run):
    with pytest.raises(ValueError, match="runs no"):
        run.method_records([("Kay_ripple_detector", "9.0")])


def test_run_session_schema(run, output, written):
    expected = {
        "sessions": SESSION_COLUMNS,
        "truth": TRUTH_COLUMNS,
        "truth_counts": TRUTH_COUNT_COLUMNS,
        "ripple_channels": RIPPLE_CHANNEL_COLUMNS,
        "units": UNIT_COLUMNS,
        "methods": METHOD_COLUMNS,
        "events": EVENT_COLUMNS,
        "metrics": METRIC_COLUMNS,
        "failures": FAILURE_COLUMNS,
        "warnings": WARNING_COLUMNS,
    }
    for name, columns in expected.items():
        assert list(getattr(output, name).columns) == columns, name
        suffix = ".csv" if name in ("methods", "failures", "warnings") else ".csv.gz"
        assert list(run.read_table(written / f"{name}{suffix}").columns) == columns, name
    kay = run.read_table(written / "results" / "Kay_ripple_detector__default.csv.gz")
    result = output.results["Kay_ripple_detector", "default"]
    assert list(kay.columns) == ["event_number", "session_id", *result.columns]
    assert sorted(p.name for p in (written / "results").iterdir()) == sorted(
        f"{method.replace(':', '--')}__{setting}{suffix}"
        for method, setting in METHODS
        for suffix in (".csv.gz", ".json")
    )
    assert output.failures.empty
    assert sorted(output.results) == sorted(METHODS)
    assert list(zip(output.methods.method, output.methods.setting, strict=True)) == list(
        METHODS
    )
    metrics = output.metrics
    assert len(metrics) == len(METHODS) * len(EXPRESSIONS) * len(LEVELS)
    for _, rows in metrics.groupby(["method", "setting", "expression"]):
        assert tuple(rows.minimum_iou) == LEVELS
    assert set(metrics.expression) == set(EXPRESSIONS)
    row = output.sessions.iloc[0]
    assert (row.session_id, row.condition_id, row.replicate, row.seed) == (
        "reference/0",
        "reference",
        0,
        20260924,
    )
    assert row.duration_s == 60.0
    assert set(output.truth_counts.expression) == set(EXPRESSIONS)


def test_the_session_row_counts_the_truth(run, output, session):
    row = output.sessions.iloc[0]
    events = session.events.drop_duplicates("event_id")
    for kind in rd.EVENT_TYPES:
        assert row[f"n_events_{kind}"] == (events.event_type == kind).sum()
    for kind in rd.NON_EVENT_TYPES:
        assert row[f"n_non_events_{kind}"] == (session.non_events.non_event_type == kind).sum()
    bouts = session.running_intervals
    assert row.rest_s == pytest.approx(60.0 - np.sum(bouts[:, 1] - bouts[:, 0]))
    # two bouts: rest is what their lengths leave, not the span between them
    two = dataclasses.replace(session, running_intervals=np.array([[5.0, 10.0], [20.0, 32.0]]))
    assert run.evaluate_session(two, "s/0", ()).sessions.rest_s.iloc[0] == pytest.approx(43.0)
    network = rd.truth_windows(session.events, 0.1, "network")
    assert row.event_time_s == pytest.approx(
        _union_seconds(network[["start_time", "end_time"]].to_numpy())
    )
    assert len(output.units) == session.multiunit.shape[1]
    np.testing.assert_array_equal(output.units.baseline_rate, session.baseline_rates)


def test_the_union_length_of_windows_in_any_order(run):
    bounds = np.array([[0.1, 0.4], [-0.3, -0.2], [0.3, 0.5], [0.5, 0.6], [0.2, 0.25]])
    # [-0.3, -0.2] and [0.1, 0.6]: a window before the clock's zero counts too
    assert run._interval_union(bounds) == pytest.approx(0.6)
    assert run._interval_union(bounds[:0]) == 0.0


def test_read_table_keeps_text_as_text(run, written, tmp_path):
    methods = run.read_table(written / "methods.csv")
    kay = methods[methods.method == "Kay_ripple_detector"]
    assert list(kay.setting) == ["default", "3.0"]
    assert (kay[["doi", "role", "inventory", "interpretation"]] == "").all().all()
    # a table whose settings are all swept values, and levels that look like numbers
    metrics = run.read_table(written / "metrics.csv.gz")
    swept = metrics[metrics.setting == "3.0"]
    run._write_table(swept, tmp_path / "swept.csv.gz")
    assert pd.read_csv(tmp_path / "swept.csv.gz").setting.dtype == float
    read = run.read_table(tmp_path / "swept.csv.gz")
    assert (read.setting == "3.0").all()
    pd.testing.assert_frame_equal(read, swept.reset_index(drop=True), check_exact=True)
    assert read.median_iou.dtype == float
    listed = pd.DataFrame(
        {
            "condition_id": ["n_units=30"],
            "factor": ["n_units"],
            "level": ["30"],
            "params": ["{}"],
        }
    )
    run._write_table(listed, tmp_path / "conditions.csv")
    assert run.read_table(tmp_path / "conditions.csv").level.tolist() == ["30"]


def test_metrics_agree_with_match_events(run, written, session):
    events = run.read_table(written / "events.csv.gz")
    metrics = run.read_table(written / "metrics.csv.gz")
    network = rd.truth_windows(session.events, 0.1, "network")
    minutes = (60.0 - _union_seconds(network[["start_time", "end_time"]].to_numpy())) / 60
    method = ("Kay_ripple_detector", "default")
    detected = events[(events.method == method[0]) & (events.setting == method[1])]
    assert len(detected) > 0
    for expression in EXPRESSIONS:
        truth = [rd.truth_windows(session.events, f, expression) for f in (0.1, 0.25, 0.5)]
        for level in LEVELS:
            matching = rd.match_events(truth[0], detected, minimum_iou=level)
            row = metrics[
                (metrics.method == method[0])
                & (metrics.setting == method[1])
                & (metrics.expression == expression)
                & (metrics.minimum_iou == level)
            ].iloc[0]
            assert (row.n_reference, row.n_detected, row.n_matched) == (
                len(truth[0]),
                len(detected),
                len(matching.pairs),
            )
            np.testing.assert_equal(
                [row.recall, row.precision, row.f1],
                [matching.recall, matching.precision, matching.f1],
            )
            fp = len(matching.unmatched_detected) / minutes
            assert row.false_positives_per_minute == pytest.approx(fp, rel=1e-12)
            for column in ("iou", "coverage", "temporal_precision"):
                np.testing.assert_equal(
                    row[f"median_{column}"], _median(matching.pairs[column])
                )
            for percent, windows in zip((10, 25, 50), truth, strict=True):
                errors = matching.boundary_errors(windows)
                for kind in ("onset", "offset"):
                    signed = errors[f"{kind}_error"].to_numpy()
                    np.testing.assert_equal(
                        row[f"median_{kind}_error_{percent}"], _median(signed)
                    )
                    np.testing.assert_equal(
                        row[f"median_abs_{kind}_error_{percent}"], _median(np.abs(signed))
                    )
            assert row.n_split == len(matching.split_reference)
            assert row.n_merged == len(matching.merged_detected)
    # the higher levels turn matches away, so the levels are not decoration
    kay = metrics[(metrics.method == method[0]) & (metrics.setting == method[1])]
    matched = kay.groupby("minimum_iou").n_matched.sum()
    assert matched[0.0] > matched[0.5]


def test_a_sliver_matches_at_zero_iou_only(run, session):
    windows = run.truth_window_sets(session.events)
    ripple = windows["ripple"][0].iloc[0]
    length = ripple.end_time - ripple.start_time
    # overlapping the first ripple by a hundredth of it, from well before
    sliver = pd.DataFrame(
        {
            "start_time": [ripple.start_time - 5 * length],
            "end_time": [ripple.start_time + length / 100],
            "peak_time": [ripple.start_time],
        }
    )
    scores = run.score_events(windows, sliver, 1.0).set_index(["expression", "minimum_iou"])
    assert scores.loc[("ripple", 0.0), "n_matched"] == 1
    assert scores.loc[("ripple", 0.2), "n_matched"] == 0
    assert scores.loc[("ripple", 0.5), "n_matched"] == 0
    assert scores.loc[("ripple", 0.2), "false_positives_per_minute"] == 1.0
    assert scores.loc[("ripple", 0.0), "median_iou"] < 0.2


def test_event_summaries_count_the_units_in_each_event(run, output, session):
    principal = np.isin(session.unit_types, ("place", "pyramidal"))
    for (method, setting), result in output.results.items():
        rows = output.events[
            (output.events.method == method) & (output.events.setting == setting)
        ]
        np.testing.assert_array_equal(rows.event_index, np.arange(len(result)))
        np.testing.assert_array_equal(rows.start_time, result.start_time)
        np.testing.assert_array_equal(rows.peak_time, result.peak_time)
        counts = rd.count_spikes_in_events(result, session.multiunit, session.time)
        np.testing.assert_array_equal(rows.n_active_units, (counts > 0).sum(axis=1))
        np.testing.assert_array_equal(
            rows.n_active_principal, (counts[:, principal] > 0).sum(axis=1)
        )
    for expression in EXPRESSIONS:
        truth = rd.truth_windows(session.events, 0.1, expression)
        rows = output.truth_counts[output.truth_counts.expression == expression]
        counts = rd.count_spikes_in_events(truth, session.multiunit, session.time)
        np.testing.assert_array_equal(rows.row, np.arange(len(truth)))
        np.testing.assert_array_equal(rows.n_active_units, (counts > 0).sum(axis=1))


def test_counts_and_scores_are_the_same_at_a_unix_clock_origin(run, session):
    shifted = dataclasses.replace(
        session,
        time=session.time + UNIX_ORIGIN,
        running_intervals=session.running_intervals + UNIX_ORIGIN,
        events=session.events.assign(center_time=session.events.center_time + UNIX_ORIGIN),
    )
    windows = run.truth_window_sets(session.events)
    moved = run.truth_window_sets(shifted.events)
    truth = windows["network"][0]
    bounds = truth[["start_time", "end_time"]].to_numpy()
    moved_bounds = moved["network"][0][["start_time", "end_time"]].to_numpy()
    for got, expected in zip(
        run.active_counts(moved_bounds, shifted),
        run.active_counts(bounds, session),
        strict=True,
    ):
        np.testing.assert_array_equal(got, expected)
    detected = truth[["start_time", "end_time", "peak_time"]] + 0.003
    scores = run.score_events(windows, detected, 1.0)
    moved_scores = run.score_events(moved, detected + UNIX_ORIGIN, 1.0)
    counts = ["n_reference", "n_detected", "n_matched", "n_split", "n_merged"]
    pd.testing.assert_frame_equal(moved_scores[counts], scores[counts])
    # Timestamps near 1.7e9 s carry about 2.4e-7 s of rounding each.
    tolerance = 16 * np.spacing(UNIX_ORIGIN + 60.0)
    errors = [column for column in scores if "error" in column]
    np.testing.assert_allclose(moved_scores[errors], scores[errors], rtol=0, atol=tolerance)


def test_results_round_trip(run, output, written, tmp_path):
    assert {"threshold_updates", "diagnostics"} <= set(
        output.results["recipe:gridchyn_2020", "literature"].attrs
    )
    for (method, setting), result in output.results.items():
        path = written / "results" / f"{method.replace(':', '--')}__{setting}.csv.gz"
        loaded = run.load_results(path)
        assert list(loaded) == ["reference/0"]
        pd.testing.assert_frame_equal(loaded["reference/0"], result, check_exact=True)
        assert loaded["reference/0"].attrs == result.attrs
    # a Unix clock origin, and a second session without events, come back as they went
    later = {key: result.iloc[:0].copy() for key, result in output.results.items()}
    for key, result in later.items():
        result.attrs = output.results[key].attrs
    moved = {
        key: result.assign(
            start_time=result.start_time + UNIX_ORIGIN, end_time=result.end_time + UNIX_ORIGIN
        )
        for key, result in output.results.items()
    }
    for key, result in moved.items():
        result.attrs = output.results[key].attrs
    first = dataclasses.replace(output, results=moved)
    second = dataclasses.replace(
        output, sessions=output.sessions.assign(session_id="reference/1"), results=later
    )
    run.write_condition(tmp_path / "two", [first, second])
    for key in output.results:
        path = tmp_path / "two" / "results" / f"{run.result_stem(*key)}.csv.gz"
        loaded = run.load_results(path)
        assert list(loaded) == ["reference/0", "reference/1"]
        pd.testing.assert_frame_equal(loaded["reference/0"], moved[key], check_exact=True)
        pd.testing.assert_frame_equal(loaded["reference/1"], later[key], check_exact=True)
        assert loaded["reference/1"].attrs == later[key].attrs


def test_model_metadata_round_trip(run, conditions_module, tmp_path):
    quartic = next(
        c for c in conditions_module.conditions() if c.condition_id == "envelope_power=quartic"
    )
    # local ripples and many fast-gamma non-events too, in 30 s
    overrides = {
        "session.duration_s": 30.0,
        "render.spatial_profile": "local",
        "render.channel_occupancy": 0.5,
        "render.channel_gain_range": (0.5, 1.0),
        "render.channel_delay": 0.002,
        "non_events.rates.fast_gamma": 30.0,
    }
    session = conditions_module.simulate_condition(quartic, 0, overrides)
    output = run.run_session(quartic, 0, methods=(), overrides=overrides)
    assert output.results == {}
    assert output.metrics.empty
    assert output.methods.empty
    run.write_condition(tmp_path / "quartic", [output])
    events, non_events = run.load_truth(tmp_path / "quartic" / "truth.csv.gz")[
        "envelope_power=quartic/0"
    ]
    pd.testing.assert_frame_equal(events, session.events, check_exact=True)
    pd.testing.assert_frame_equal(non_events, session.non_events, check_exact=True)
    assert (events.envelope_power == 4).all()
    gamma = non_events[non_events.non_event_type == "fast_gamma"]
    assert len(gamma)
    assert gamma[["snr_band_low", "snr_band_high"]].notna().all().all()
    for fraction in (0.1, 0.25, 0.5):
        for expression in EXPRESSIONS:
            pd.testing.assert_frame_equal(
                rd.truth_windows(events, fraction, expression),
                rd.truth_windows(session.events, fraction, expression),
                check_exact=True,
            )
        pd.testing.assert_frame_equal(
            rd.truth_windows(non_events, fraction),
            rd.truth_windows(session.non_events, fraction),
            check_exact=True,
        )
    channels = run.read_table(tmp_path / "quartic" / "ripple_channels.csv.gz")
    pd.testing.assert_frame_equal(
        channels.drop(columns="session_id"), session.ripple_channels, check_exact=True
    )
    assert (channels.gain == 0).any()
    assert (channels.delay_s != 0).any()


def _raises(error):
    """A stub method call's preparation: the call raises ``error``."""

    def call():
        raise error

    return lambda: call


def test_failures_are_recorded_not_raised(run, session, monkeypatch):
    long_message = "x" * 500
    real = run.method_calls

    def calls(given):
        kay = next(c for c in real(given) if c[:2] == ("Kay_ripple_detector", "default"))
        return [
            ("stub_value", "default", _raises(ValueError("bad value"))),
            kay,
            ("stub_index", "default", _raises(IndexError("index 3 is out of bounds"))),
            ("stub_long", "default", _raises(ValueError(long_message))),
        ]

    monkeypatch.setattr(run, "method_calls", calls)
    output = run.evaluate_session(session, "reference/0")
    failures = output.failures.set_index("method").error
    assert failures["stub_value"] == "ValueError: bad value"
    assert failures["stub_index"] == "IndexError: index 3 is out of bounds"
    assert failures["stub_long"] == f"ValueError: {long_message}"[:200]
    for frame in (output.events, output.metrics):
        assert set(frame.method) == {"Kay_ripple_detector"}
    assert list(output.results) == [("Kay_ripple_detector", "default")]
    assert set(output.runtimes) == {(name, "default") for name in failures.index} | {
        ("Kay_ripple_detector", "default")
    }


def _returns(frame):
    """A stub method call's preparation: the call returns ``frame``."""
    return lambda: lambda: frame


def test_a_sub_sample_event_is_kept_with_no_active_units(run, session, monkeypatch):
    # some methods' bounds lie up to half a sample off the grid: an event
    # between two samples holds none, and so no spike
    step = 1 / session.sampling_frequency
    between = session.time[100] + np.array([0.25, 0.75]) * step
    found = pd.DataFrame(
        {"start_time": [between[0], 1.0], "end_time": [between[1], 1.2], "peak_time": np.nan}
    )
    monkeypatch.setattr(
        run, "method_calls", lambda given: [("stub", "default", _returns(found))]
    )
    output = run.evaluate_session(session, "reference/0")
    assert output.failures.empty
    assert list(output.results) == [("stub", "default")]
    counts = rd.count_spikes_in_events(found.iloc[1:], session.multiunit, session.time)
    np.testing.assert_array_equal(output.events.n_active_units, [0, (counts > 0).sum()])
    assert len(output.metrics) == len(EXPRESSIONS) * len(LEVELS)


def test_a_scoring_error_raises_rather_than_failing_the_method(run, session, monkeypatch):
    def broken(*args):
        msg = "a bug in the benchmark"
        raise RuntimeError(msg)

    monkeypatch.setattr(run, "score_events", broken)
    with pytest.raises(RuntimeError, match="a bug in the benchmark"):
        run.evaluate_session(session, "reference/0", [("Kay_ripple_detector", "default")])


def test_warnings_are_recorded_not_raised(run, session, monkeypatch):
    long_message = "y" * 500
    found = pd.DataFrame({"start_time": [1.0], "end_time": [1.2], "peak_time": [1.1]})

    def warns():
        for _ in range(2):
            warnings.warn("twice from one line", UserWarning, stacklevel=1)
        warnings.warn(long_message, RuntimeWarning, stacklevel=1)
        return found

    def warns_then_fails():
        warnings.warn("before failing", DeprecationWarning, stacklevel=1)
        msg = "bad value"
        raise ValueError(msg)

    def prepares_with_a_warning():
        warnings.warn("while preparing", UserWarning, stacklevel=1)
        return lambda: found

    monkeypatch.setattr(
        run,
        "method_calls",
        lambda given: [
            ("stub_warns", "default", lambda: warns),
            ("stub_fails", "3.0", lambda: warns_then_fails),
            ("stub_prepares", "default", prepares_with_a_warning),
        ],
    )
    output = run.evaluate_session(session, "reference/0")
    assert list(output.results) == [("stub_warns", "default"), ("stub_prepares", "default")]
    assert list(output.failures.method) == ["stub_fails"]
    key = {"session_id": "reference/0", "method": "stub_warns", "setting": "default"}
    assert output.warnings.to_dict("records") == [
        {**key, "category": "UserWarning", "message": "twice from one line"},
        {**key, "category": "UserWarning", "message": "twice from one line"},
        {**key, "category": "RuntimeWarning", "message": long_message[:200]},
        {
            **key,
            "method": "stub_fails",
            "setting": "3.0",
            "category": "DeprecationWarning",
            "message": "before failing",
        },
        # a warning building a call's inputs is the call's
        {
            **key,
            "method": "stub_prepares",
            "category": "UserWarning",
            "message": "while preparing",
        },
    ]


GRIDCHYN = (("recipe:gridchyn_2020", "literature"),)


def test_gridchyn_without_a_valid_baseline_fails_explicitly(run, session, conditions_module):
    # running throughout: no rest to stand in for the pre-rest baseline, which
    # the policy cannot build (every benchmark session starts and ends at rest)
    no_rest = dataclasses.replace(
        session, running_intervals=np.array([[session.time[0], session.time[-1]]])
    )
    with pytest.raises(ValueError, match="The session has no rest"):
        run.evaluate_session(no_rest, "s/0", GRIDCHYN)
    # a baseline without a spike, which the method rejects
    silent = dataclasses.replace(session, multiunit=np.zeros_like(session.multiunit))
    failed = run.evaluate_session(silent, "s/0", GRIDCHYN).failures
    assert len(failed) == 1
    assert failed.error.iloc[0].startswith("ValueError: Pre-rest must have a positive")
    # no running bout at all is rest throughout, a valid baseline
    reference = conditions_module.conditions()[0]
    still = conditions_module.simulate_condition(reference, 0, {"session.duration_s": 30.0})
    assert len(still.running_intervals) == 0
    output = run.evaluate_session(still, "s/0", GRIDCHYN)
    assert output.failures.empty
    assert list(output.results) == list(GRIDCHYN)


def _broken(*args):
    msg = "a bug in the benchmark's input code"
    raise RuntimeError(msg)


@pytest.mark.parametrize("step", ["recording input", "behavior_intervals", "integer counts"])
def test_an_input_error_raises_rather_than_failing_the_method(
    run, recipe_configs, session, monkeypatch, step
):
    """The inputs a recipe runs on are the benchmark's own: an error building
    them is a bug to fix, never the method's failure."""
    if step == "recording input":
        method = GRIDCHYN[0]
        policy = recipe_configs._POLICY["multiunit"]
        broken = dataclasses.replace(policy, get=_broken)
        monkeypatch.setitem(recipe_configs._POLICY, "multiunit", broken)
    elif step == "behavior_intervals":
        method = ("recipe:yang_2024", "literature")
        policy = recipe_configs._POLICY["behavior_intervals"]
        broken = dataclasses.replace(policy, get=_broken)
        monkeypatch.setitem(recipe_configs._POLICY, "behavior_intervals", broken)
    else:
        method = GRIDCHYN[0]
        monkeypatch.setattr(run, "_integer_counts", _broken)
    with pytest.raises(RuntimeError, match="input code"):
        run.evaluate_session(session, "reference/0", [method])


def test_a_recipe_that_raises_is_a_failure(run, session, monkeypatch):
    monkeypatch.setattr(run, "run_recipe", _broken)
    output = run.evaluate_session(session, "reference/0", GRIDCHYN)
    assert output.failures.to_dict("records") == [
        {
            "session_id": "reference/0",
            "method": GRIDCHYN[0][0],
            "setting": "literature",
            "error": "RuntimeError: a bug in the benchmark's input code",
        }
    ]
    assert output.results == {}


def test_integer_counts_give_the_float_results(run, recipe_configs, session):
    counted = run._integer_counts(session)
    assert counted.multiunit.dtype == np.int16
    np.testing.assert_array_equal(counted.multiunit, session.multiunit)
    configs = {config.config_id: config for config in recipe_configs.RECIPES}
    for name in COUNTED_RECIPES:
        config = configs[name]
        assert "multiunit" in recipe_configs.policy_inputs(config), name
        results = [
            recipe_configs.run_recipe(
                config,
                recipe_configs.make_recording(given, config),
                recipe_configs.behavior_intervals(given, config),
            )
            for given in (session, counted)
        ]
        pd.testing.assert_frame_equal(results[1], results[0], check_exact=True)
        assert results[1].attrs == results[0].attrs, name
    # counts that are not whole numbers stay as they are
    fractional = dataclasses.replace(session, multiunit=session.multiunit + 0.5)
    assert run._integer_counts(fractional) is fractional


# The command line


def _fake_report(calls):
    def require(path, resolved):
        calls.append((path, resolved))
        return {
            "path": str(path),
            "sha256": "0" * 64,
            "simulation_fingerprint": "1" * 64,
            "target_table_hash": "2" * 64,
        }

    return require


@pytest.fixture
def cli(run, tmp_path, monkeypatch):
    """The runner writing under ``tmp_path``, its preflight recording its
    arguments, and ``run_session`` recording each session it runs."""
    reports, sessions = [], []
    monkeypatch.setattr(run, "OUTPUT", tmp_path)
    monkeypatch.setattr(run, "_require_report", _fake_report(reports))
    monkeypatch.setattr(run, "_git_commit", lambda: COMMIT)
    real = run.run_session

    def counted(condition, replicate, methods=None, overrides=None):
        sessions.append(condition.condition_id)
        return real(condition, replicate, methods, overrides)

    monkeypatch.setattr(run, "run_session", counted)

    def start(name="v1", *, resume=False, **options):
        arguments = {
            "validation_report": "report/spec.json",
            "condition_ids": RUN_CONDITIONS,
            "replicates": 1,
            "duration": 30.0,
            "workers": 1,
            "methods": RUN_METHODS,
            **options,
        }
        return run.run_benchmark(name, resume=resume, **arguments)

    return start, reports, sessions


@pytest.fixture(scope="module")
def finished_run(run, tmp_path_factory):
    """A finished run of the three conditions, to copy."""
    root = tmp_path_factory.mktemp("runs")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(run, "OUTPUT", root)
        patch.setattr(run, "_require_report", _fake_report([]))
        patch.setattr(run, "_git_commit", lambda: COMMIT)
        run.run_benchmark(
            "v1",
            validation_report="report/spec.json",
            condition_ids=RUN_CONDITIONS,
            replicates=1,
            duration=30.0,
            workers=1,
            methods=RUN_METHODS,
        )
    return root / "v1"


def _copy(finished_run, tmp_path):
    return Path(shutil.copytree(finished_run, tmp_path / "v1"))


def _snapshot(directory):
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_a_run_writes_its_files_and_combines_them(run, cli, conditions_module, capsys):
    start, reports, sessions = cli
    root = start()
    assert sessions == RUN_CONDITIONS
    # the preflight saw every selected condition's parameters, overrides applied
    ((_, resolved),) = reports
    by_id = {c.condition_id: c for c in conditions_module.conditions()}
    overrides = {"session.duration_s": 30.0}
    assert resolved == {
        i: conditions_module.resolve(by_id[i], overrides) for i in RUN_CONDITIONS
    }
    manifest = json.loads((root / "manifest.json").read_text())
    assert set(manifest) == MANIFEST_KEYS
    assert manifest["finished"] is not None
    spec = json.loads((root / "run_spec.json").read_text())
    assert manifest["git_commit"] == spec["git_commit"] == COMMIT
    assert spec["validation_report"] == {
        "path": "report/spec.json",
        "sha256": "0" * 64,
        "simulation_fingerprint": "1" * 64,
        "target_table_hash": "2" * 64,
    }
    assert spec["replicates"] == dict.fromkeys(RUN_CONDITIONS, 1)
    assert spec["seeds"] == {condition_id: [20260924] for condition_id in RUN_CONDITIONS}
    assert list(spec["methods"]) == ["Kay_ripple_detector default"]
    listed = run.read_table(root / "conditions.csv")
    assert list(listed.columns) == CONDITION_COLUMNS
    assert list(listed.condition_id) == RUN_CONDITIONS
    for condition_id, params in zip(listed.condition_id, listed.params, strict=True):
        assert params == conditions_module.resolved_json(by_id[condition_id], overrides)
    for condition_id in RUN_CONDITIONS:
        assert run.condition_is_finished(root / "conditions" / condition_id)
    combined = root / "combined"
    for name in ("sessions.csv.gz", "metrics.csv.gz", "events.csv.gz", "methods.csv"):
        table = run.read_table(combined / name)
        assert sorted(set(table.session_id)) == sorted(f"{i}/0" for i in RUN_CONDITIONS)
    for condition_id in RUN_CONDITIONS:
        assert (
            combined / "results" / condition_id / "Kay_ripple_detector__default.json"
        ).exists()
    assert json.loads((combined / "manifest.json").read_text()) == {
        "included": sorted(RUN_CONDITIONS),
        "missing": [],
    }
    assert not capsys.readouterr().out


def test_done_json_verifies_every_file(run, finished_run, tmp_path):
    root = _copy(finished_run, tmp_path) / "conditions" / "reference"
    assert run.condition_is_finished(root)
    done = json.loads((root / "done.json").read_text())["files"]
    assert done["sessions.csv.gz"]["rows"] == 1
    assert done["results/Kay_ripple_detector__default.json"]["rows"] is None
    (root / "stray.txt").write_text("")
    assert not run.condition_is_finished(root)
    (root / "stray.txt").unlink()
    with (root / "failures.csv").open("a") as file:
        file.write("\n")
    assert not run.condition_is_finished(root)
    (root / "done.json").unlink()
    assert not run.condition_is_finished(root)


def test_resume_skips_finished_conditions(run, cli, finished_run, tmp_path):
    start, reports, sessions = cli
    root = _copy(finished_run, tmp_path)
    before = _snapshot(root / "conditions")
    start(resume=True)
    assert sessions == []
    assert len(reports) == 1  # the preflight still runs
    assert _snapshot(root / "conditions") == before


@pytest.mark.parametrize("damage", ["partial", "tampered"])
def test_resume_reruns_interrupted_conditions(
    run, cli, finished_run, tmp_path, damage, capsys
):
    start, _, sessions = cli
    root = _copy(finished_run, tmp_path)
    conditions = root / "conditions"
    if damage == "partial":
        (conditions / "emg_rate=3").rename(conditions / "emg_rate=3.partial")
        deleted = f"Deleting {conditions / 'emg_rate=3.partial'}: interrupted while written"
    else:
        with (conditions / "emg_rate=3" / "events.csv.gz").open("ab") as file:
            file.write(b"\0")
        deleted = f"Deleting {conditions / 'emg_rate=3'}: its files do not match done.json"
    start(resume=True)
    assert sessions == ["emg_rate=3"]
    assert capsys.readouterr().out.splitlines() == [deleted]
    assert not (conditions / "emg_rate=3.partial").exists()
    assert run.condition_is_finished(conditions / "emg_rate=3")
    # the same files again, but for the wall-clock times in sessions.csv.gz
    rerun = _snapshot(conditions / "emg_rate=3")
    first = _snapshot(finished_run / "conditions" / "emg_rate=3")
    assert set(rerun) == set(first)
    for name in set(rerun) - {"sessions.csv.gz", "done.json"}:
        assert rerun[name] == first[name], name


def test_resume_after_two_finished_and_one_interrupted(run, cli, finished_run, tmp_path):
    start, _, sessions = cli
    root = _copy(finished_run, tmp_path)
    conditions = root / "conditions"
    # C was being written when the run stopped: no done.json, an unfinished table
    partial = conditions / "emg_rate=3.partial"
    (conditions / "emg_rate=3").rename(partial)
    (partial / "done.json").unlink()
    (partial / "metrics.csv.gz").unlink()
    shutil.rmtree(root / "combined")
    finished = {name: _snapshot(conditions / name) for name in RUN_CONDITIONS[:2]}
    start(resume=True)
    assert sessions == ["emg_rate=3"]
    for name, files in finished.items():
        assert _snapshot(conditions / name) == files
        assert run.condition_is_finished(conditions / name)
    assert run.condition_is_finished(conditions / "emg_rate=3")
    combined = run.read_table(root / "combined" / "sessions.csv.gz")
    assert list(combined.condition_id) == sorted(RUN_CONDITIONS)
    assert sorted(p.name for p in (root / "combined" / "results").iterdir()) == sorted(
        RUN_CONDITIONS
    )


def test_the_manifest_records_the_workers_used(cli, finished_run, tmp_path):
    start, _, _ = cli
    # four requested, one session to run: it runs in this process
    root = start("one", condition_ids=["reference"], workers=4)
    assert json.loads((root / "manifest.json").read_text())["n_workers"] == 1
    # resuming records the workers the resumed run used
    root = _copy(finished_run, tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    (root / "manifest.json").write_text(json.dumps({**manifest, "n_workers": 7}))
    (root / "conditions" / "emg_rate=3").rename(root / "conditions" / "emg_rate=3.partial")
    start(resume=True, workers=4)
    resumed = json.loads((root / "manifest.json").read_text())
    assert resumed == {**manifest, "n_workers": 1, "finished": resumed["finished"]}


def test_combine_is_derived(run, cli, finished_run, tmp_path):
    start, reports, sessions = cli
    root = _copy(finished_run, tmp_path)
    combined = _snapshot(root / "combined")
    shutil.rmtree(root / "combined")
    run.main(["--run-name", "v1", "--combine"])
    assert _snapshot(root / "combined") == combined
    assert reports == []  # combining needs no report
    # resume never reads combined/: damaging it reruns nothing, and it is rebuilt
    (root / "combined" / "sessions.csv.gz").write_bytes(b"damaged")
    (root / "combined" / "metrics.csv.gz").unlink()
    start(resume=True)
    assert sessions == []
    assert _snapshot(root / "combined") == combined


def test_combine_names_the_conditions_it_leaves_out(run, cli, finished_run, tmp_path, capsys):
    root = _copy(finished_run, tmp_path)
    conditions = root / "conditions"
    # one condition interrupted, one whose files no longer match done.json
    (conditions / "emg_rate=3").rename(conditions / "emg_rate=3.partial")
    with (conditions / "ripple_snr=high" / "events.csv.gz").open("ab") as file:
        file.write(b"\0")
    # and a combine that stopped half way
    (root / "combined.partial").mkdir()
    (root / "combined.partial" / "stray.csv").write_text("")
    run.main(["--run-name", "v1", "--combine"])
    printed = capsys.readouterr().out
    assert "emg_rate=3, ripple_snr=high" in printed
    combined = root / "combined"
    assert json.loads((combined / "manifest.json").read_text()) == {
        "included": ["reference"],
        "missing": ["emg_rate=3", "ripple_snr=high"],
    }
    assert list(run.read_table(combined / "sessions.csv.gz").session_id) == ["reference/0"]
    assert sorted(p.name for p in (combined / "results").iterdir()) == ["reference"]
    assert not (root / "combined.partial").exists()
    assert not (combined / "stray.csv").exists()


@pytest.mark.parametrize(
    ("change", "keys"),
    [
        ({"duration": 45.0}, "conditions.reference.session.duration_s"),
        ({"replicates": 2}, "replicates.reference"),
        (
            {"methods": (*RUN_METHODS, ("Kay_ripple_detector", "3.0"))},
            "methods.Kay_ripple_detector 3.0",
        ),
    ],
)
def test_resume_rejects_a_changed_specification(cli, finished_run, tmp_path, change, keys):
    start, _, sessions = cli
    root = _copy(finished_run, tmp_path)
    (root / "conditions" / "emg_rate=3").rename(root / "conditions" / "emg_rate=3.partial")
    before = _snapshot(root)
    with pytest.raises(SystemExit, match="differs at") as raised:
        start(resume=True, **change)
    assert keys in str(raised.value)
    assert sessions == []
    assert _snapshot(root) == before


def test_a_report_that_cannot_back_the_run_stops_it_first(
    run, validation, cli, tmp_path, monkeypatch
):
    start, _, sessions = cli

    def not_ready(path, resolved):
        msg = "status is not_ready: widths out of range"
        raise validation.ReportNotReady(msg)

    monkeypatch.setattr(run, "_require_report", not_ready)
    with pytest.raises(SystemExit, match="not_ready: widths"):
        start()
    assert sessions == []
    assert list(tmp_path.iterdir()) == []

    # any other error is a bug, raised as it is
    def broken(path, resolved):
        msg = "a bug in the preflight"
        raise ValueError(msg)

    monkeypatch.setattr(run, "_require_report", broken)
    with pytest.raises(ValueError, match="a bug in the preflight"):
        start()
    assert list(tmp_path.iterdir()) == []


def test_the_real_preflight_stops_the_run_with_its_own_message(run, tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OUTPUT", tmp_path / "runs")
    missing = tmp_path / "missing" / "spec.json"
    arguments = {"condition_ids": ["reference"], "replicates": 1, "methods": RUN_METHODS}
    with pytest.raises(SystemExit) as raised:
        run.run_benchmark("v1", validation_report=missing, **arguments)
    assert str(raised.value) == (
        f"No simulator validation report at {missing}: run validate_simulator.py first."
    )
    stale = tmp_path / "stale" / "spec.json"
    stale.parent.mkdir()
    stale.write_text(json.dumps({"status": "not_ready", "reasons": ["widths"]}))
    with pytest.raises(SystemExit) as raised:
        run.run_benchmark("v1", validation_report=stale, **arguments)
    message = str(raised.value)
    assert message.count("cannot back this run") == 1
    assert "its status is 'not_ready', not 'ready' (widths)" in message
    assert not (tmp_path / "runs").exists()


def _git(directory, *arguments):
    found = subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "commit.gpgsign=false",
            *arguments,
        ],
        cwd=directory,
        capture_output=True,
        text=True,
        check=True,
    )
    return found.stdout.strip()


def test_the_git_commit_is_flagged_dirty(run, tmp_path):
    """Changes under src/ or examples/benchmark/, untracked files included,
    make the commit ``<sha>-dirty``; changes elsewhere do not."""
    assert run._git_commit(tmp_path) == "unknown"
    repository = tmp_path / "repository"
    for name in ("src/a.py", "examples/benchmark/b.py", "docs/c.md"):
        (repository / name).parent.mkdir(parents=True, exist_ok=True)
        (repository / name).write_text("0\n")
    _git(repository, "init", "-q")
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "first")
    sha = _git(repository, "rev-parse", "HEAD")
    # the commit is the repository's, whichever of its directories asks
    here = repository / "examples" / "benchmark"
    assert run._git_commit(here) == sha
    (repository / "docs" / "c.md").write_text("1\n")
    assert run._git_commit(here) == sha
    (repository / "src" / "a.py").write_text("1\n")
    assert run._git_commit(here) == f"{sha}-dirty"
    _git(repository, "checkout", "-q", "--", "src")
    assert run._git_commit(here) == sha
    (here / "new.py").write_text("")
    assert run._git_commit(here) == f"{sha}-dirty"


def test_an_unknown_or_dirty_commit_stops_the_run(
    run, cli, finished_run, tmp_path, monkeypatch
):
    start, _, sessions = cli
    monkeypatch.setattr(run, "_git_commit", lambda: "unknown")
    with pytest.raises(SystemExit, match="git commit is unknown"):
        start()
    assert list(tmp_path.iterdir()) == []
    root = _copy(finished_run, tmp_path)
    (root / "conditions" / "emg_rate=3").rename(root / "conditions" / "emg_rate=3.partial")
    before = _snapshot(root)
    with pytest.raises(SystemExit, match="git commit is unknown"):
        start(resume=True)
    # a dirty tree may start a run, which records it, but never resume one
    monkeypatch.setattr(run, "_git_commit", lambda: f"{COMMIT}-dirty")
    with pytest.raises(SystemExit, match="committed, clean code"):
        start(resume=True)
    assert _snapshot(root) == before
    assert sessions == []
    start("dirty")
    spec = json.loads((tmp_path / "dirty" / "run_spec.json").read_text())
    assert spec["git_commit"] == f"{COMMIT}-dirty"


def test_the_command_line_checks_its_arguments(run, cli, finished_run, tmp_path):
    start, _, sessions = cli
    with pytest.raises(SystemExit, match="Unknown conditions"):
        start(condition_ids=["reference", "ripple_snr=huge"])
    _copy(finished_run, tmp_path)
    with pytest.raises(SystemExit, match="exists; pass --resume"):
        start()
    with pytest.raises(SystemExit, match="No run to resume"):
        start("other", resume=True)
    for argv in (
        ["--run-name", "v2"],
        [
            "--run-name",
            "v2",
            "--smoke",
            "--conditions",
            "reference",
            "--validation-report",
            "r",
        ],
        ["--run-name", "v1", "--combine", "--resume"],
    ):
        with pytest.raises(SystemExit):
            run.main(argv)
    assert sessions == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["v1"]


def test_the_command_line_takes_a_crossed_cell(run, conditions_module, monkeypatch):
    chosen = []
    monkeypatch.setattr(
        run, "run_benchmark", lambda name, **options: chosen.append(options["condition_ids"])
    )
    for text in ("ripple_snr=low,participation=low", "reference,participation=low"):
        run.main(["--run-name", "x", "--conditions", text, "--validation-report", "r"])
    run.main(["--run-name", "x", "--validation-report", "r"])
    assert chosen == [
        ["ripple_snr=low,participation=low"],
        ["reference", "participation=low"],
        [c.condition_id for c in conditions_module.conditions()],
    ]
    with pytest.raises(SystemExit):
        run.main(
            ["--run-name", "x", "--conditions", "reference,nope", "--validation-report", "r"]
        )
    assert len(chosen) == 3


def test_smoke_prints_its_measurements(cli, capsys, tmp_path):
    start, _, sessions = cli
    # the validation report's cost per session, as validate_simulator records it
    report = tmp_path / "report" / "spec.json"
    report.parent.mkdir()
    measured = [(10.0, 2**30), (12.0, 1.5 * 2**30)]
    report.write_text(
        json.dumps(
            {
                "sessions": [
                    {
                        "condition_id": "reference",
                        "replicate": 10000 + i,
                        "seconds": s,
                        "peak_rss_bytes": b,
                    }
                    for i, (s, b) in enumerate(measured)
                ]
            }
        )
    )
    start(
        "smoke",
        smoke=True,
        condition_ids=None,
        replicates=None,
        workers=4,
        validation_report=report,
    )
    assert sessions == ["reference"]
    printed = capsys.readouterr().out
    # 860 sessions of 11 s: 2.63 CPU hours
    for text in (
        "Kay_ripple_detector default:",
        "Simulate:",
        "Peak resident memory:",
        "events.csv.gz: ",
        "results/:",
        "Full grid at this duration: 440 sessions",
        "on 4 workers",
        (
            "Validation of the full grid at this duration: 860 sessions (43 conditions x 20 "
            "replicates), 11.0 s and at most 1.50 GiB each (the report's 2 sessions): 2.6 "
            "CPU hours, 0.7 h on 4 workers"
        ),
        "Validation and benchmark together:",
        "Decision rules:",
        "keep duration_s",
        "workers = min(requested 4",
    ):
        assert text in printed, text


def test_a_worker_failure_cancels_the_queued_sessions(run, cli, monkeypatch):
    """The first session to fail stops the run: queued sessions never start.
    Threads stand in for the processes, so the stub sessions are seen."""
    start, _, _ = cli
    release = threading.Event()
    started = []

    def run_one(condition, replicate, methods, overrides):
        started.append((condition.condition_id, replicate))
        if len(started) == 1:
            msg = "the first session failed"
            raise RuntimeError(msg)
        # the others hold their worker until the test ends; the timeout only
        # bounds a run that waits for every queued session
        release.wait(timeout=2.0)

    monkeypatch.setattr(run, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(run, "_run_one", run_one)
    try:
        with pytest.raises(RuntimeError, match="the first session failed"):
            start("failing", replicates=3, workers=2)
        # the failed session, the one running beside it and at most one the
        # freed worker took before the queue was cancelled; not all nine
        assert len(started) <= 3
    finally:
        release.set()


def test_replicates_are_written_in_order_with_their_seeds(run, cli):
    """Two replicates on two workers, whichever finishes first."""
    start, _, _ = cli
    root = start("two", condition_ids=["reference"], replicates=2, workers=2)
    condition = root / "conditions" / "reference"
    rows = run.read_table(condition / "sessions.csv.gz")
    assert list(rows.session_id) == ["reference/0", "reference/1"]
    assert list(rows.replicate) == [0, 1]
    assert list(rows.seed) == [20260924, 20260925]
    spec = json.loads((root / "run_spec.json").read_text())
    assert spec["replicates"] == {"reference": 2}
    assert spec["seeds"] == {"reference": [20260924, 20260925]}
    assert json.loads((root / "manifest.json").read_text())["n_workers"] == 2
    # each replicate is its own draw, in its own rows of every table
    units = run.read_table(condition / "units.csv.gz")
    first, second = (rows.baseline_rate.to_numpy() for _, rows in units.groupby("session_id"))
    assert not np.array_equal(first, second)


def test_workers_write_what_one_process_writes(run, cli, finished_run):
    start, _, _ = cli
    root = start("pooled", workers=2)
    for condition_id in RUN_CONDITIONS:
        expected = _snapshot(finished_run / "conditions" / condition_id)
        got = _snapshot(root / "conditions" / condition_id)
        # sessions.csv holds wall-clock times; everything else is identical
        assert set(got) == set(expected)
        for name in set(got) - {"sessions.csv.gz", "done.json"}:
            assert got[name] == expected[name], name

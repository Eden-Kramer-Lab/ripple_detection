"""The benchmark's attribution (examples/benchmark/attribution.py): the Sobol and
Shapley estimators on known cases; templates compiled and run from public
primitives, and verified against each configuration's public call on short
reference sessions, a gap and a Unix clock origin; the factor spaces and
reference configurations; memoized evaluation on a bounded, read-only session
context; reference sessions simulated again from a run's saved parameters and
checked against what it saved. No test draws a figure."""

import dataclasses
import gzip
import json
import math
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
from scipy.stats import qmc

import ripple_detection as rd

SHORT = 30.0  # seconds: the halved sessions of the run the templates are checked on
LONG = 60.0
N_REPLICATES = 5
FS = 1500.0


@pytest.fixture(scope="module")
def attribution(benchmark_import):
    return benchmark_import("attribution")


@pytest.fixture(scope="module")
def run(benchmark_import):
    return benchmark_import("run")


@pytest.fixture(scope="module")
def analyze(benchmark_import):
    return benchmark_import("analyze")


@pytest.fixture(scope="module")
def conditions(benchmark_import):
    return benchmark_import("conditions")


@pytest.fixture(scope="module")
def recipes(benchmark_import):
    return benchmark_import("recipe_configs").RECIPES


def _write_run(run, conditions, root, duration, replicates):
    """A run of the reference condition at ``duration`` seconds, written by
    the runner's own functions, with no method run: what attribution reads."""
    reference = conditions.conditions()[0]
    overrides = {"session.duration_s": duration}
    outputs = [
        run.run_session(reference, replicate, methods=(), overrides=overrides)
        for replicate in range(replicates)
    ]
    run.write_condition(root / "conditions" / "reference", outputs)
    row = {
        "condition_id": "reference",
        "factor": "reference",
        "level": "reference",
        "params": conditions.resolved_json(reference, overrides),
    }
    run._write_table(pd.DataFrame([row]), root / "conditions.csv")
    return root


@pytest.fixture(scope="module")
def short_run(run, conditions, tmp_path_factory):
    """Five 30 s reference sessions: a run whose duration_s was halved."""
    return _write_run(run, conditions, tmp_path_factory.mktemp("short"), SHORT, N_REPLICATES)


@pytest.fixture(scope="module")
def long_run(run, conditions, tmp_path_factory):
    return _write_run(run, conditions, tmp_path_factory.mktemp("long"), LONG, 1)


@pytest.fixture(scope="module")
def contexts(attribution, short_run):
    """The short run's first two sessions, then a gap and a Unix clock origin."""
    parameters = attribution.reference_parameters(short_run)
    found = list(attribution.reference_contexts(short_run, range(2)))
    found += [
        attribution.SessionContext(session, f"edge/{name}")
        for name, session in attribution.edge_sessions(parameters, SHORT).items()
    ]
    return found


@pytest.fixture(scope="module")
def long_context(attribution, long_run):
    """The 60 s run's session, which has a running bout."""
    return next(iter(attribution.reference_contexts(long_run, [0])))


@pytest.fixture(scope="module")
def verified(attribution, recipes, contexts):
    return attribution.verify_all(recipes, contexts)


# Sobol indices and Shapley values


def _ishigami(x, a=7.0, b=0.1):
    return np.sin(x[:, 0]) + a * np.sin(x[:, 1]) ** 2 + b * x[:, 2] ** 4 * np.sin(x[:, 0])


def test_sobol_on_ishigami(attribution):
    n, d = 2**13, 3
    sample = qmc.Sobol(d=2 * d, scramble=True, seed=0).random(n) * 2 * np.pi - np.pi
    a, b = sample[:, :d], sample[:, d:]
    ab = []
    for i in range(d):
        mixed = a.copy()
        mixed[:, i] = b[:, i]
        ab.append(_ishigami(mixed))
    first, total = attribution.sobol_indices(_ishigami(a), _ishigami(b), np.array(ab))
    np.testing.assert_allclose(first, [0.314, 0.442, 0.0], atol=0.03)
    np.testing.assert_allclose(total, [0.558, 0.442, 0.244], atol=0.03)
    table = attribution.sobol_intervals(
        _ishigami(a), _ishigami(b), np.array(ab), n_resamples=200
    )
    np.testing.assert_array_equal(table["first"], first)
    assert (table["first_low"] <= table["first"]).all()
    assert (table["total"] <= table["total_high"]).all()
    assert (table["total_high"] - table["total_low"] > 0).all()
    assert (table["finite_rows"] == n).all()
    assert (table[["first_finite_draws", "total_finite_draws"]] == 200).all().all()


def test_sobol_intervals_with_missing_outputs(attribution):
    """An index with a missing output has no estimate and so no interval; the
    finite rows and draws behind each are counted."""
    rng = np.random.default_rng(1)
    n = 64
    y_a, y_b, y_ab = rng.normal(size=n), rng.normal(size=n), rng.normal(size=(3, n))
    y_ab[1, 5] = np.nan
    table = attribution.sobol_intervals(y_a, y_b, y_ab, n_resamples=300)
    assert table.loc[1, ["first", "first_low", "first_high", "total"]].isna().all()
    assert table.loc[1, ["total_low", "total_high"]].isna().all()
    assert (
        table.loc[[0, 2], ["first_low", "first_high", "total_low", "total_high"]]
        .notna()
        .all()
        .all()
    )
    assert table["finite_rows"].tolist() == [n, n - 1, n]
    draws = table["first_finite_draws"].tolist()
    assert draws[0] == draws[2] == 300
    # the draws leaving row 5 out: about (1 - 1/n)^n of them
    assert 0 < draws[1] < 300
    assert draws == table["total_finite_draws"].tolist()
    # a missing output of A: every index is missing
    y_a[0] = np.nan
    table = attribution.sobol_intervals(y_a, y_b, y_ab, n_resamples=300)
    assert table[["first", "first_low", "total", "total_high"]].isna().all().all()
    assert table["finite_rows"].tolist() == [n - 1, n - 2, n - 1]


def _toy(subset):
    """A 10-factor toy: |S|^1.5, plus 2 when x0 and x1 are both in S."""
    return len(subset) ** 1.5 + 2.0 * ({"x0", "x1"} <= subset)


def test_shapley_additive_and_efficiency(attribution):
    weights = {"a": 0.5, "b": -1.25, "c": 3.0}
    phi, error = attribution.shapley(lambda s: sum(weights[i] for i in s), weights)
    assert phi == pytest.approx(weights, abs=1e-12)
    assert set(error.values()) == {0.0}

    def interacting(subset):
        return len(subset) ** 2 + 5.0 * ("a" in subset and "c" in subset)

    phi, _ = attribution.shapley(interacting, "abcd")
    assert sum(phi.values()) == pytest.approx(interacting(set("abcd")) - 0, abs=1e-9)
    # a and c share the interaction; b and d only the size
    assert phi["a"] == pytest.approx(phi["c"])
    assert phi["a"] - phi["b"] == pytest.approx(2.5)

    factors = [f"x{i}" for i in range(10)]
    exact, _ = attribution.shapley(_toy, factors, exact_up_to=10)
    assert sum(exact.values()) == pytest.approx(_toy(set(factors)), abs=1e-9)
    sampled, error = attribution.shapley(_toy, factors, n_permutations=4000)
    assert max(abs(sampled[f] - exact[f]) for f in factors) < 0.05
    assert all(value > 0 for value in error.values())


@pytest.mark.parametrize("n_factors", [3, 10])
def test_shapley_asks_only_the_subsets_listed(attribution, n_factors):
    factors = [f"x{i}" for i in range(n_factors)]
    listed = attribution.shapley_subsets(factors, n_permutations=16)
    assert len(listed) == len(set(listed))
    asked = set()

    def value(subset):
        asked.add(subset)
        return _toy(subset)

    attribution.shapley(value, factors, n_permutations=16)
    assert asked == set(listed)
    if n_factors <= attribution.SHAPLEY_EXACT_UP_TO:
        assert len(listed) == 2**n_factors


# Templates and pipelines


def test_every_configuration_has_a_template_or_a_reason(attribution, recipes):
    ids = [config.config_id for config in recipes]
    assert set(attribution.TEMPLATES).isdisjoint(attribution.FIXED_POINTS)
    assert set(attribution.TEMPLATES) | set(attribution.FIXED_POINTS) == set(ids)
    assert all(reason for reason in attribution.FIXED_POINTS.values())
    assert all(source for _, source in attribution.TEMPLATES.values())
    by_id = {config.config_id: config for config in recipes}
    with pytest.raises(ValueError, match="custom peak merging"):
        attribution.template_of(by_id["mallory_2025"])
    assert attribution.template_of(by_id["bendor_2012"]).merge_gap == 0.05
    stranger = dataclasses.replace(by_id["bendor_2012"], config_id="bendor_2012.other")
    with pytest.raises(KeyError, match="neither"):
        attribution.template_of(stranger)


def test_compile_orders_the_steps(attribution):
    template = attribution.TEMPLATES["yang_2024"][0]
    pipeline = attribution.compile(template)
    assert [step.operation for step in pipeline.steps] == [
        "active_units",
        "inside",
        "contains_time",
    ]
    assert pipeline.core.signal[0].parameters == (
        ("units", "pyramidal"),
        ("smoothing_sigma", 0.015),
    )
    assert pipeline.core.speed_threshold == np.inf
    restricted = attribution.compile(attribution.TEMPLATES["drieu_2018"][0])
    assert restricted.core.restrict_to == "rest"
    # a restriction is a level of the state with no post step
    assert attribution.RESTRICTIONS == {"restrict:rest": "rest"}
    assert all(attribution.STATES[level] is None for level in attribution.RESTRICTIONS)
    assert restricted.steps == ()
    squared = dataclasses.replace(attribution.TEMPLATES["pfeiffer_2015"][0], trace="squared")
    compiled = attribution.compile(squared)
    assert [step.operation for step in compiled.core.signal] == ["mean_envelope", "square"]
    assert compiled.core.smoothing_sigma == 0.0125
    assert hash(compiled) == hash(attribution.compile(squared))
    for field, value in (("speed", "fast"), ("state", "asleep"), ("trace", "log")):
        with pytest.raises(ValueError, match=r"not a level|trace must"):
            attribution.compile(dataclasses.replace(squared, **{field: value}))


def test_compile_makes_a_hashable_pipeline(attribution):
    template = attribution.TEMPLATES["pfeiffer_2015"][0]
    listed = dataclasses.replace(template, band=[150.0, 250.0])
    assert listed.band == (150.0, 250.0)
    assert attribution.compile(listed) == attribution.compile(template)
    unhashable = dataclasses.replace(template, maximum_duration={"seconds": 2.0})
    with pytest.raises(TypeError, match=r"LfpTemplate\(.*'seconds': 2.0.*is not hashable"):
        attribution.compile(unhashable)


def test_a_spike_core_takes_no_lfp_signal_steps(attribution, long_context):
    pipeline = attribution.compile(attribution.TEMPLATES["igata_2021"][0])
    core = pipeline.core
    for changed, message in (
        (
            dataclasses.replace(core, signal=(*core.signal, attribution.Step("square"))),
            "no further",
        ),
        (dataclasses.replace(core, smoothing_sigma=0.01), "smoothing"),
    ):
        with pytest.raises(ValueError, match=message):
            attribution.run_pipeline(
                attribution.Pipeline(changed, pipeline.steps), long_context
            )


# How many templates stand for their methods on the two 30 s sessions and the
# edge cases: update deliberately when a configuration or a template changes.
# davidson_2009 needs running within 30 s, and a session under 35 s has no bout.
IN_SPACE_SHORT = {"spikes": 14, "lfp": 4}


def test_in_space_recipes(attribution, recipes, verified):
    for family, count in IN_SPACE_SHORT.items():
        rows = verified[verified["family"] == family]
        assert int(rows["in_space"].sum()) == count
    in_space = verified[verified["in_space"]]
    # positive controls: every template verified found events to compare
    assert (in_space["n_events"] > 0).all()
    assert set(in_space["config_id"]) <= set(attribution.TEMPLATES)
    out = verified[(verified["family"] != "") & ~verified["in_space"]]
    assert out["config_id"].tolist() == ["davidson_2009"]
    assert out["reason"].iloc[0].startswith("no event")
    fixed = verified[verified["family"] == ""]
    assert set(fixed["config_id"]) == set(attribution.FIXED_POINTS)
    assert not fixed["in_space"].any()
    assert (fixed["reason"] == fixed["config_id"].map(attribution.FIXED_POINTS)).all()


def test_a_session_with_running_verifies_davidson(attribution, recipes, long_context):
    config = next(c for c in recipes if c.config_id == "davidson_2009")
    assert len(long_context.session.running_intervals)
    row = attribution.verify_all([config], [long_context]).iloc[0]
    assert row["in_space"]
    assert row["n_events"] > 0


def test_a_wrong_template_is_not_in_space(attribution, recipes, contexts, monkeypatch):
    config = next(c for c in recipes if c.config_id == "igata_2021")
    assert attribution.in_space(config, contexts)
    template, source = attribution.TEMPLATES["igata_2021"]
    for change in ({"threshold": 2.5}, {"minimum_active_units": 15}, {"units": "place"}):
        monkeypatch.setitem(
            attribution.TEMPLATES,
            "igata_2021",
            (dataclasses.replace(template, **change), source),
        )
        row = attribution.verify_all([config], contexts).iloc[0]
        assert not row["in_space"]
        assert row["reason"].startswith("events differ on reference/0")


@pytest.mark.parametrize(
    ("config_id", "change"),
    [
        ("yang_2024", {"smoothing_sigma": 0.015 * 1.1}),
        ("pfeiffer_2015", {"smoothing_sigma": 0.0125 * 1.1}),
        ("krause_2022_hse", {"bound_fraction": 0.5}),
        ("ambrose_2016", {"channels": 3}),
    ],
)
def test_a_wrong_template_with_the_same_counts_is_not_in_space(
    attribution, recipes, contexts, monkeypatch, config_id, change
):
    """The bounds are compared, not only how many events there are."""
    config = next(c for c in recipes if c.config_id == config_id)
    template, source = attribution.TEMPLATES[config_id]
    changed = dataclasses.replace(template, **change)
    for context in contexts:
        right = context.events(attribution.compile(template))
        assert len(context.events(attribution.compile(changed))) == len(right)
    monkeypatch.setitem(attribution.TEMPLATES, config_id, (changed, source))
    row = attribution.verify_all([config], contexts).iloc[0]
    assert not row["in_space"]
    assert row["reason"].startswith("events differ")


def test_perturbations(attribution, recipes):
    """Every value a template sets, moved: continuous ones 10 % either way,
    integers by one, categorical ones to each other level of the space;
    values at a step's absence are left alone."""
    lfp = {factor.name: factor for factor in attribution.factor_space(recipes, "lfp")}
    assert attribution.perturbations(attribution.TEMPLATES["pfeiffer_2015"][0], lfp) == [
        ("smoothing_sigma", 0.0125 * 0.9),
        ("smoothing_sigma", 0.0125 * 1.1),
        ("normalization_period", "session"),
        ("threshold", 3.0 * 0.9),
        ("threshold", 3.0 * 1.1),
        ("minimum_event_duration", 0.05 * 0.9),
        ("minimum_event_duration", 0.05 * 1.1),
        ("maximum_duration", None),
        ("speed", "restrict<5"),
    ]
    spikes = {factor.name: factor for factor in attribution.factor_space(recipes, "spikes")}
    found = attribution.perturbations(attribution.TEMPLATES["chenani_2019"][0], spikes)
    assert [value for name, value in found if name == "minimum_active_units"] == [4, 6]
    assert [value for name, value in found if name == "bound_fraction"] == [0.0, 1.0, 0.5]
    assert [value for name, value in found if name == "units"] == ["pyramidal", "all"]
    assert {name for name, _ in found} == {
        "units", "smoothing_sigma", "threshold", "bound_fraction",
        "minimum_active_units", "state",
    }  # fmt: skip


def test_sensitivity(attribution, recipes, contexts):
    table = attribution.sensitivity(recipes, contexts, "lfp")
    space = {factor.name: factor for factor in attribution.factor_space(recipes, "lfp")}
    ids = attribution.in_space_ids("lfp")
    expected = [
        (config_id, name, attribution._as_text(value))
        for config_id in ids
        for name, value in attribution.perturbations(
            attribution.TEMPLATES[config_id][0], space
        )
    ]
    found = zip(table["config_id"], table["factor"], table["perturbed"], strict=True)
    assert list(found) == expected

    def row(config_id, factor, perturbed=None):
        chosen = (table["config_id"] == config_id) & (table["factor"] == factor)
        if perturbed is not None:
            chosen &= table["perturbed"] == perturbed
        return table[chosen]

    # told apart on the first session where the events differ, bound for bound
    template = attribution.TEMPLATES["pfeiffer_2015"][0]
    for name, value in (("smoothing_sigma", 0.0125 * 1.1), ("maximum_duration", None)):
        changed = attribution.compile(dataclasses.replace(template, **{name: value}))
        right = attribution.compile(template)
        differs = [
            context.label
            for context in contexts
            if not np.array_equal(context.events(changed), context.events(right))
        ]
        found = row("pfeiffer_2015", name, attribution._as_text(value))
        assert found["told_apart"].tolist() == [bool(differs)]
        assert found["session"].tolist() == [differs[0] if differs else ""]
    # no 30 s session has a ripple past 2 s: the maximum is not exercised
    assert row("pfeiffer_2015", "maximum_duration")["exercised"].tolist() == [False]
    exercised = table.groupby(["config_id", "factor"])["told_apart"].transform("any")
    assert (table["exercised"] == exercised).all()
    assert row("pfeiffer_2015", "smoothing_sigma")["exercised"].tolist() == [True, True]


def test_equal_empty_results_verify_nothing(attribution, recipes, contexts, monkeypatch):
    config = next(c for c in recipes if c.config_id == "bendor_2012")
    template, source = attribution.TEMPLATES["bendor_2012"]
    # neither finds an event: a rate never 1000 SD up, and no public call
    unreachable = dataclasses.replace(template, threshold=1000.0)
    monkeypatch.setitem(attribution.TEMPLATES, "bendor_2012", (unreachable, source))
    monkeypatch.setattr(attribution, "recipe_events", lambda config, session: np.empty((0, 2)))
    row = attribution.verify_all([config], contexts).iloc[0]
    assert not row["in_space"]
    assert row["reason"].startswith("no event")


def test_edge_sessions(attribution, recipes, short_run, contexts):
    parameters = attribution.reference_parameters(short_run)
    edges = attribution.edge_sessions(parameters, SHORT)
    gap, moved = edges["gap"], edges["unix_origin"]
    start, end = attribution.gap_interval(gap)
    assert end - start == pytest.approx(attribution.GAP_WIDTH)
    # the gap lies inside a sharp-wave ripple's truth window, at rest
    windows = rd.truth_windows(gap.events, 0.1, "network")
    inside = (windows["start_time"] < start) & (windows["end_time"] > end)
    assert (windows.loc[inside, "type"] == "swr").any()
    no_swr = dataclasses.replace(gap, events=gap.events[gap.events["event_type"] != "swr"])
    with pytest.raises(ValueError, match="no sharp-wave ripple at rest"):
        attribution.gap_interval(no_swr)
    missing = (gap.time >= start) & (gap.time < end)
    assert missing.any()
    assert np.isnan(gap.lfps[missing]).all()
    assert np.isfinite(gap.lfps[~missing]).all()
    assert np.isnan(gap.multiunit[missing]).all()
    assert np.isnan(gap.sharp_wave_lfp[missing]).all()
    # events of each family are cut at the gap, and their templates still verify
    before = gap.time[np.flatnonzero(missing)[0] - 1]
    after = gap.time[np.flatnonzero(missing)[-1] + 1]
    reach = 2 * attribution.BIN_WIDTH
    for family in attribution.FAMILIES:
        cut = []
        for config in recipes:
            if config.config_id not in attribution.in_space_ids(family):
                continue
            events = attribution.recipe_events(config, gap)
            ends = (events[:, 1] <= before) & (events[:, 1] > before - reach)
            starts = (events[:, 0] >= after) & (events[:, 0] < after + reach)
            if (ends | starts).any():
                cut.append(config.config_id)
        assert cut, family
    assert moved.time[0] == attribution.UNIX_ORIGIN
    np.testing.assert_array_equal(
        moved.running_intervals - attribution.UNIX_ORIGIN, gap.running_intervals
    )
    # the events a template finds move with the clock, bound for bound
    pipeline = attribution.compile(attribution.TEMPLATES["pfeiffer_2015"][0])
    plain = attribution.SessionContext(
        dataclasses.replace(moved, time=gap.time, running_intervals=gap.running_intervals),
        "plain",
    )
    shifted = contexts[-1].events(pipeline)
    assert len(shifted)
    np.testing.assert_allclose(
        shifted - attribution.UNIX_ORIGIN, plain.events(pipeline), rtol=0, atol=1e-6
    )
    # nothing spans the gap
    blocked = contexts[-2].events(attribution.compile(attribution.TEMPLATES["igata_2021"][0]))
    spans = (blocked[:, 0] < end) & (blocked[:, 1] >= start)
    assert not spans.any()


# Factor spaces and reference configurations

SPIKE_SPACE = [
    ("units", "categorical", ("pyramidal", "all", "place")),
    ("smoothing_sigma", "continuous", (0.005, 0.08)),
    ("normalization_period", "categorical", ("rest", "session", "speed<5", "speed<4")),
    ("threshold", "continuous", (2.0, 4.0)),
    ("bound_fraction", "categorical", (0.0, 1.0, 1 / 3, 0.5)),
    ("minimum_event_duration", "continuous", (0.0, 0.1)),
    ("maximum_duration", "categorical", (0.5, 2.0, 0.8, None, 0.75)),
    ("merge_gap", "continuous", (0.0, 0.05)),
    (
        "speed",
        "categorical",
        ("none", "all<=3", "restrict<5", "endpoints<5", "all<=5", "endpoints<4"),
    ),
    ("minimum_active_units", "integer", (0, 1, 2, 3, 4, 5)),
    ("state", "categorical", ("inside:rest", "none", "restrict:rest", "overlap:running_30s")),
    (
        "coincidence",
        "categorical",
        (
            "peak_inside:external_ripples",
            "overlap:long_swrs",
            "none",
            "overlap:muessig_2019_ripples",
        ),
    ),
]
LFP_SPACE = [
    ("channels", "categorical", (3, None)),
    ("normalization_period", "categorical", ("speed<5", "session")),
    ("threshold", "continuous", (2.0, 3.0)),
    ("minimum_event_duration", "continuous", (0.0, 0.05)),
    ("maximum_duration", "categorical", (2.0, None)),
    ("speed", "categorical", ("endpoints<=5", "restrict<5")),
]


@pytest.mark.parametrize(("family", "space"), [("spikes", SPIKE_SPACE), ("lfp", LFP_SPACE)])
def test_factor_space_is_pinned(attribution, recipes, family, space):
    found = attribution.factor_space(recipes, family)
    assert [(f.name, f.kind, f.levels) for f in found] == space
    assert attribution.factor_space(recipes[:1], family) == ()
    with pytest.raises(ValueError, match="family must be"):
        attribution.factor_space(recipes, "sharp_wave")


# Each template's bound in SD as its method sets it; the rest bound at the mean.
METHOD_BOUNDS = {
    "farooq_2019_neuron": 2.0,
    "chenani_2019": 1.0,
    "muessig_2019": 3.0,
    "bendor_2012": 2.0,
}


def test_bounds_are_fractions_of_the_threshold(attribution, recipes):
    for config_id, (template, _) in attribution.TEMPLATES.items():
        core = attribution.compile(template).core
        # bit for bit the method's bound, so the verification can hold
        assert core.bound_threshold == METHOD_BOUNDS.get(config_id, 0.0), config_id
        assert core.bound_threshold == template.bound_fraction * template.threshold
    # so every configuration a Sobol or Shapley analysis asks for bounds at or
    # below its threshold
    for family in attribution.FAMILIES:
        reference = attribution.reference_template(family)
        a, b, ab = attribution.sobol_design(
            attribution.factor_space(recipes, family), reference, 64
        )
        pairs = [
            (attribution.TEMPLATES[config_id][0], reference)
            for config_id in attribution.in_space_ids(family)
        ]
        subsets = [
            dataclasses.replace(first, **{name: getattr(second, name) for name in subset})
            for first, second in pairs
            for subset in attribution.shapley_subsets(attribution._differing(first, second))
        ]
        for template in [*a, *b, *(t for column in ab for t in column), *subsets]:
            core = attribution.compile(template).core
            assert core.bound_threshold <= core.threshold


def test_factor_values(attribution):
    continuous = attribution.Factor("threshold", "continuous", (2.0, 4.0))
    assert continuous.value(0.0) == 2.0
    assert continuous.value(0.25) == 2.5
    assert continuous.points() == (2.0, 2.5, 3.0, 3.5, 4.0)
    categorical = attribution.Factor("speed", "categorical", ("a", "b", "c"))
    values = [categorical.value(u) for u in (0.0, 0.33, 0.34, 0.99, 1.0)]
    assert values == ["a", "a", "b", "c", "c"]
    assert categorical.points() == ("a", "b", "c")


def _spike_template(attribution, **values):
    base = {
        "units": "all", "smoothing_sigma": 0.01, "normalization_period": "session",
        "threshold": 2.0, "bound_fraction": 0.0, "minimum_event_duration": 0.05,
        "maximum_duration": None, "merge_gap": 0.0, "speed": "none",
        "minimum_active_units": 0, "state": "none", "coincidence": "none",
    }  # fmt: skip
    return attribution.SpikeTemplate(**{**base, **values})


def test_reference_template(attribution):
    templates = [
        _spike_template(attribution),
        _spike_template(
            attribution, units="place", smoothing_sigma=0.02, normalization_period="speed<5",
            threshold=3.0, bound_fraction=0.5, minimum_event_duration=0.1,
            maximum_duration=0.5, merge_gap=0.05, speed="all<=3", minimum_active_units=3,
            state="inside:rest",
        ),
        _spike_template(
            attribution, units="place", smoothing_sigma=0.04, normalization_period="rest",
            threshold=5.0, bound_fraction=0.5, minimum_event_duration=0.0,
            speed="restrict<5", minimum_active_units=4, coincidence="overlap:long_swrs",
        ),
    ]  # fmt: skip
    reference = attribution.reference_template("spikes", templates)
    assert reference == _spike_template(
        attribution,
        units="place",  # the mode
        smoothing_sigma=0.02,  # medians
        threshold=3.0,
        bound_fraction=0.5,
        minimum_event_duration=0.05,
        minimum_active_units=3,
        # three-way ties go to the first template's value
        normalization_period="session",
        speed="none",
    )
    # an integer median is rounded down
    pair = [templates[0], _spike_template(attribution, minimum_active_units=5)]
    assert attribution.reference_template("spikes", pair).minimum_active_units == 2
    lfp = attribution.reference_template("lfp")
    assert (lfp.normalization_period, lfp.minimum_event_duration) == ("speed<5", 0.025)
    with pytest.raises(ValueError, match="no template"):
        attribution.reference_template("lfp", [])


def test_identical_templates_count_once(attribution, recipes, monkeypatch):
    templates = attribution.TEMPLATES
    assert templates["grosmark_2016"][0] == templates["yang_2024"][0]
    written = attribution.in_space_ids("spikes")
    assert "grosmark_2016" in written
    distinct = attribution.distinct_ids("spikes")
    assert distinct == tuple(i for i in written if i != "grosmark_2016")
    # the reference takes each template once: counted twice, the second
    # template would win the mode and move the median
    first = _spike_template(attribution)
    second = _spike_template(attribution, units="place", threshold=3.0)
    reference = attribution.reference_template("spikes", [first, second, second])
    assert (reference.units, reference.threshold) == ("all", 2.5)
    assert attribution.reference_template("spikes") == attribution.reference_template(
        "spikes", [templates[i][0] for i in distinct]
    )
    # the stop rule counts templates, not configurations: eight written, seven distinct
    kept = [c for c in recipes if c.config_id in written[:7] or c.config_id == "grosmark_2016"]
    monkeypatch.setattr(attribution, "RECIPES", tuple(kept))
    assert len(attribution.in_space_ids("spikes")) == 8
    with pytest.raises(SystemExit, match="7 represented methods"):
        attribution.main(["--run-name", "x", "--family", "spikes", "--analysis", "sobol"])


def test_the_lowest_agreement_pairs_are_decomposed(attribution, monkeypatch):
    ids = attribution.distinct_ids("spikes")
    names = {attribution.compile(attribution.TEMPLATES[i][0]): i for i in ids}
    assert len(names) == len(ids)
    agreement = {
        ("liu_2023", "chenani_2019"): 0.1,
        ("igata_2021", "bendor_2012"): 0.2,
        ("farooq_2019_neuron", "silva_2015"): 0.3,
    }

    def evaluate_many(pipelines, references, run_directory, *, workers=1):
        return [
            {
                name: [agreement.get((names[p], names[r]), 0.5)] * attribution.K
                for name in attribution.Y_NAMES
            }
            for p, r in zip(pipelines, references, strict=True)
        ]

    monkeypatch.setattr(attribution, "evaluate_many", evaluate_many)
    monkeypatch.setattr(attribution, "N_LOWEST_PAIRS", 2)
    pairs = attribution.shapley_pair_list("spikes", "unused")
    # each distinct template against the reference, then the two lowest pairs
    assert pairs == [
        *((config_id, "reference") for config_id in ids),
        ("liu_2023", "chenani_2019"),
        ("igata_2021", "bendor_2012"),
    ]


def test_sobol_design(attribution):
    factors = (
        attribution.Factor("threshold", "continuous", (2.0, 4.0)),
        attribution.Factor("units", "categorical", ("all", "place")),
    )
    reference = _spike_template(attribution)
    a, b, ab = attribution.sobol_design(factors, reference, 8)
    assert len(a) == len(b) == 8
    assert len(ab) == 2
    sample = qmc.Sobol(d=4, scramble=True, seed=0).random(8)
    assert [t.threshold for t in a] == pytest.approx(2.0 + 2.0 * sample[:, 0])
    assert [t.threshold for t in b] == pytest.approx(2.0 + 2.0 * sample[:, 2])
    for i, name in enumerate(("threshold", "units")):
        other = "units" if name == "threshold" else "threshold"
        assert [getattr(t, name) for t in ab[i]] == [getattr(t, name) for t in b]
        assert [getattr(t, other) for t in ab[i]] == [getattr(t, other) for t in a]
    assert {t.smoothing_sigma for t in a} == {reference.smoothing_sigma}


# Sessions, outputs and memoization


def test_evaluate_config_is_memoized(attribution, long_context):
    context = long_context
    context.release()
    reference = attribution.compile(attribution.reference_template("spikes"))
    pipeline = attribution.compile(attribution.TEMPLATES["igata_2021"][0])
    before = context.n_runs
    first = attribution.evaluate_config(pipeline, [context], reference)
    assert context.n_runs == before + 2  # the pipeline's detection and the reference's
    again = attribution.evaluate_config(pipeline, [context], reference)
    assert again == first
    assert context.n_runs == before + 2
    attribution.evaluate_config(reference, [context], reference)
    assert context.n_runs == before + 2
    other = attribution.compile(attribution.TEMPLATES["chenani_2019"][0])
    attribution.evaluate_config(other, [context], reference)
    assert context.n_runs == before + 3
    context.release()
    assert attribution.evaluate_config(pipeline, [context], reference) == first
    assert context.n_runs == before + 5


def test_the_context_is_read_only_and_bounded(attribution, long_context):
    recording = long_context.recording
    for values in (recording.multiunit, recording.session.lfps, recording.speed,
                   recording.place_cells, long_context.rest):  # fmt: skip
        with pytest.raises(ValueError, match="read-only"):
            values[0] = 0
    pipeline = attribution.compile(attribution.TEMPLATES["pfeiffer_2015"][0])
    with pytest.raises(ValueError, match="read-only"):
        long_context.events(pipeline)[0] = 0.0
    trace = long_context.population("all", 0.015)
    cached = [trace.time, trace.data, trace.speed, trace.first_sample, trace.last_sample]
    for values in (values for values in cached if values is not None):
        with pytest.raises(ValueError, match="read-only"):
            values[0] = 0.0
    calls = []
    cache = attribution._Cache(2)
    for key in ("a", "b", "a", "c", "b"):
        cache.get(key, lambda key=key: calls.append(key) or key)
    # "b" was least recently used when "c" came in, so it is computed again
    assert calls == ["a", "b", "c", "b"]
    assert len(cache) == 2


def test_session_outputs(attribution, long_context):
    context = long_context
    events = context.events(attribution.compile(attribution.TEMPLATES["igata_2021"][0]))
    reference = context.events(attribution.compile(attribution.TEMPLATES["chenani_2019"][0]))
    found = attribution.session_outputs(events, reference, context, "spikes")
    truth = rd.truth_windows(context.session.events, 0.1, "burst")[["start_time", "end_time"]]
    matching = rd.match_events(truth, events)
    assert found["f1"] == matching.f1
    network = rd.truth_windows(context.session.events, 0.1, "network")
    assert found["f1_network"] == rd.match_events(network, events).f1
    assert found["events_per_minute"] == len(events) / (LONG / 60)
    later = rd.truth_windows(context.session.events, 0.25, "burst")
    onset = matching.boundary_errors(later)["onset_error"]
    assert found["onset_error_25"] == np.median(onset)
    shared = len(rd.match_events(events, reference).pairs)
    assert found["jaccard_reference"] == shared / (len(events) + len(reference) - shared)
    lfp = attribution.session_outputs(events, reference, context, "lfp")
    ripples = rd.truth_windows(context.session.events, 0.1, "ripple")
    assert lfp["f1"] == rd.match_events(ripples, events).f1
    empty = np.empty((0, 2))
    assert attribution.jaccard(empty, empty) == 1.0
    assert attribution.jaccard(events, empty) == 0.0
    nothing = attribution.session_outputs(empty, empty, context, "spikes")
    assert np.isnan(nothing["onset_error_25"])
    assert nothing["events_per_minute"] == 0


# Reference sessions simulated again


def test_a_halved_run_regenerates_its_halved_sessions(attribution, conditions, short_run):
    parameters = attribution.reference_parameters(short_run)
    assert parameters["session"]["duration_s"] == SHORT
    reference = conditions.conditions()[0]
    assert parameters == conditions.resolve(reference, {"session.duration_s": SHORT})
    session = attribution.reference_session(short_run, 3)
    assert len(session.time) == SHORT * FS
    expected = conditions.simulate_condition(reference, 3, {"session.duration_s": SHORT})
    np.testing.assert_array_equal(session.lfps, expected.lfps)
    np.testing.assert_array_equal(session.multiunit, expected.multiunit)
    labels = [c.label for c in attribution.reference_contexts(short_run, [1, 4])]
    assert labels == ["reference/1", "reference/4"]


def _rewrite(path, change):
    """Rewrite a gzipped table with one value changed, the rest as written."""
    frame = pd.read_csv(path, float_precision="round_trip", dtype=str, keep_default_na=False)
    with gzip.open(path, "wt", newline="") as handle:
        change(frame).to_csv(handle, index=False)


def _set_first(column, value, table=None):
    """Set ``column`` of reference/0's first row (of ``table``, in the truth)."""

    def change(frame):
        rows = frame["session_id"] == "reference/0"
        if table is not None:
            rows &= frame["table"] == table
        frame.loc[frame.index[rows][0], column] = value
        return frame

    return change


@pytest.mark.parametrize(
    ("table", "change", "names"),
    [
        ("truth.csv.gz", _set_first("amplitude", "1.5", "event"), "its events"),
        ("truth.csv.gz", _set_first("n_spikes", "999", "non_event"), "its non-events"),
        ("sessions.csv.gz", _set_first("seed", "7"), "seed 7"),
        ("sessions.csv.gz", _set_first("duration_s", "60.0"), "duration 60.0 s"),
        ("ripple_channels.csv.gz", _set_first("gain", "0.5"), "its ripple channels"),
    ],
)
def test_a_session_unlike_the_runs_raises(
    attribution, short_run, tmp_path, table, change, names
):
    copy = tmp_path / "run"
    shutil.copytree(short_run, copy)
    _rewrite(copy / "conditions" / "reference" / table, change)
    with pytest.raises(ValueError, match=f"reference/0 is not the run's: .*{names}"):
        attribution.reference_session(copy, 0)


def test_a_run_without_the_session_or_reference_raises(attribution, run, short_run, tmp_path):
    with pytest.raises(ValueError, match="holds no session reference/7"):
        attribution.reference_session(short_run, 7)
    copy = tmp_path / "run"
    shutil.copytree(short_run, copy)
    table = run.read_table(copy / "conditions.csv").assign(condition_id="emg_rate=0")
    run._write_table(table, copy / "conditions.csv")
    with pytest.raises(ValueError, match="0 reference rows"):
        attribution.reference_parameters(copy)


def test_check_report(attribution, short_run, tmp_path, monkeypatch):
    identity = {
        "path": "examples/benchmark/validation/v9/spec.json",
        "sha256": "a" * 64,
        "simulation_fingerprint": "b" * 64,
        "target_table_hash": "c" * 64,
    }
    asked = []

    def ready(path, resolved):
        asked.append((path, resolved))
        return {**identity, "path": str(path), "sessions": []}

    monkeypatch.setattr(attribution, "require_ready_report", ready)
    parameters = attribution.reference_parameters(short_run)
    copy = tmp_path / "run"
    copy.mkdir()
    (copy / "run_spec.json").write_text(json.dumps({"validation_report": identity}))
    attribution.check_report(copy, parameters)
    assert asked == [(attribution.REPOSITORY / identity["path"], {"reference": parameters})]
    stale = {**identity, "simulation_fingerprint": "d" * 64, "sha256": "e" * 64}
    (copy / "run_spec.json").write_text(json.dumps({"validation_report": stale}))
    with pytest.raises(ValueError, match="at: sha256, simulation_fingerprint"):
        attribution.check_report(copy, parameters)


# The analyses, on the five 30 s sessions


def test_evaluate_many_on_two_workers(attribution, short_run):
    pipelines = [
        attribution.compile(attribution.TEMPLATES[name][0])
        for name in ("pfeiffer_2015", "berners_lee_2021", "ambrose_2016")
    ]
    references = pipelines[::-1]
    one = attribution.evaluate_many(pipelines, references, short_run, chunk_size=2)
    two = attribution.evaluate_many(pipelines, references, short_run, workers=2, chunk_size=2)
    assert one == two
    assert all(len(values) == attribution.K for found in one for values in found.values())
    contexts = list(attribution.reference_contexts(short_run))
    for pipeline, reference, found in zip(pipelines, references, one, strict=True):
        means = attribution.evaluate_config(pipeline, contexts, reference)
        assert means == {name: attribution._mean(found[name]) for name in attribution.Y_NAMES}
    assert attribution._WORKER_CONTEXT == {}


def test_a_failing_configuration_is_named(attribution, short_run):
    good = attribution.compile(attribution.TEMPLATES["pfeiffer_2015"][0])
    bad = dataclasses.replace(good, steps=(attribution.Step("bogus"),))
    pipelines = [good, good, bad]
    named = (
        r"(?s)configuration 2 \(the chunk from 2\) on reference/0"
        r".*ValueError: Unknown post step 'bogus'.*Step\(operation='bogus'"
    )
    with pytest.raises(RuntimeError, match=named):
        attribution.evaluate_many(pipelines, [good] * 3, short_run, chunk_size=2)
    # the session this process held is released, failure or not
    assert attribution._WORKER_CONTEXT == {}


def test_a_failing_chunk_cancels_the_queued_ones(attribution, monkeypatch):
    """Threads stand in for the processes, so the stub chunks are seen."""
    release = threading.Event()
    started = []

    def evaluate_chunk(run_directory, replicate, start, pipelines, references):
        started.append((replicate, start))
        if len(started) == 1:
            msg = "the first chunk failed"
            raise RuntimeError(msg)
        # the others hold their worker until the test ends; the timeout only
        # bounds a run that waits for every queued chunk
        release.wait(timeout=2.0)
        return [{}] * len(pipelines)

    monkeypatch.setattr(attribution, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(attribution, "_evaluate_chunk", evaluate_chunk)
    try:
        with pytest.raises(RuntimeError, match="the first chunk failed"):
            attribution.evaluate_many(
                [None] * 4, [None] * 4, "unused", workers=2, chunk_size=1
            )
        # the failed chunk, the one beside it and at most one the freed worker
        # took before the queue was cancelled; not all twenty
        assert len(started) <= 3
    finally:
        release.set()


def test_one_at_a_time(attribution, analyze, recipes, short_run):
    rows, changes = attribution.one_at_a_time("lfp", short_run)
    space = attribution.factor_space(recipes, "lfp")
    n_configurations = 1 + sum(len(factor.points()) for factor in space)
    assert len(rows) == n_configurations * len(attribution.Y_NAMES)
    assert len(changes) == (n_configurations - 1) * len(attribution.Y_NAMES)
    assert rows[[f"replicate_{k}" for k in range(attribution.K)]].notna().all().all()
    reference = attribution.reference_template("lfp")
    # a factor at its reference level changes nothing, exactly
    at_reference = changes[
        (changes["factor"] == "normalization_period") & (changes["level"] == "speed<5")
    ]
    assert (at_reference[["change", "low", "high"]].to_numpy() == 0).all()
    assert reference.normalization_period == "speed<5"
    # the interval is the paired bootstrap's over the sessions, draw for draw
    values = np.array([0.1, np.nan, 0.3, -0.2, 0.5])
    frame = pd.DataFrame(
        {"session_id": [f"s{k}" for k in range(5)], "replicate": range(5), "change": values}
    )
    expected = analyze.paired_bootstrap(
        frame,
        lambda sample: pd.Series({"change": np.nanmean(sample["change"])}),
        key="session_id",
    ).loc["change"]
    found = attribution.session_interval(values)
    assert found == pytest.approx(
        {
            "change": expected["estimate"],
            "low": expected["low"],
            "high": expected["high"],
            "n_sessions": 4,
        }
    )
    # one session gives a change but no interval
    alone = attribution.session_interval([np.nan, 0.1, np.nan, np.nan, np.nan])
    assert alone["change"] == 0.1
    assert alone["n_sessions"] == 1
    assert np.isnan([alone["low"], alone["high"]]).all()
    assert "n_sessions" in changes.columns
    jaccard = changes[(changes["y"] == "jaccard_reference")]
    assert (jaccard["value"] <= 1).all()
    both = rows[rows["y"] == "f1"].set_index(["factor", "level"])["value"]
    lowered = changes[(changes["factor"] == "threshold") & (changes["y"] == "f1")].iloc[0]
    assert lowered["change"] == pytest.approx(
        both["threshold", lowered["level"]] - both["reference", ""]
    )


def test_sobol(attribution, recipes, short_run):
    rows, indices = attribution.sobol("lfp", short_run, n=4)
    d = len(attribution.factor_space(recipes, "lfp"))
    assert len(rows) == 4 * (d + 2) * len(attribution.Y_NAMES)
    assert rows.groupby("matrix").size().to_dict() == {
        "A": 4 * len(attribution.Y_NAMES),
        "AB": 4 * d * len(attribution.Y_NAMES),
        "B": 4 * len(attribution.Y_NAMES),
    }
    assert len(indices) == d * len(attribution.Y_NAMES)
    assert list(indices.columns[:2]) == ["y", "factor"]
    # the number of rows is written with both tables
    assert (rows["sobol_n"] == 4).all()
    assert (indices["sobol_n"] == 4).all()
    f1 = rows[rows["y"] == "f1"]
    y = f1["value"].to_numpy()
    first, total = attribution.sobol_indices(y[:4], y[4:8], y[8:].reshape(d, 4))
    selected = indices[indices["y"] == "f1"]
    np.testing.assert_array_equal(selected["first"], first)
    np.testing.assert_array_equal(selected["total"], total)


def test_shapley_pairs(attribution, short_run):
    rows, values = attribution.shapley_pairs("lfp", short_run)
    ids = attribution.in_space_ids("lfp")
    pairs = list(dict.fromkeys(values["pair"]))
    # every configuration against the reference, then the lowest-agreement pairs
    assert pairs[: len(ids)] == [f"{config_id}|reference" for config_id in ids]
    assert len(pairs) == len(ids) + math.comb(len(ids), 2)
    for (_, y), table in values.groupby(["pair", "y"]):
        # efficiency: the values add up to the pair's whole difference
        assert table["phi"].sum() == pytest.approx(
            table["v_all"].iloc[0] - table["v_empty"].iloc[0], abs=1e-9
        )
        assert (table["error"] == 0).all()
        if y == "jaccard_reference":
            assert table["v_all"].iloc[0] == 1.0
    a = pairs[0].split("|")[0]
    contexts = list(attribution.reference_contexts(short_run))
    template = attribution.TEMPLATES[a][0]
    reference = attribution.compile(attribution.reference_template("lfp"))
    expected = attribution.evaluate_config(attribution.compile(template), contexts, reference)
    empty = rows[(rows["pair"] == pairs[0]) & (rows["subset"] == "")]
    assert dict(zip(empty["y"], empty["value"], strict=True)) == pytest.approx(expected)


def test_fixed_point_outputs(attribution, recipes, contexts, monkeypatch, capsys):
    chosen = [
        c for c in recipes if c.config_id in ("gupta_2010", "mallory_2025", "igata_2021")
    ]
    original = attribution.run_recipe

    def fails_at_the_unix_origin(config, recording, intervals):
        if config.config_id == "mallory_2025" and recording.time[0] > 0:
            message = "made to fail"
            raise ValueError(message)
        return original(config, recording, intervals)

    monkeypatch.setattr(attribution, "run_recipe", fails_at_the_unix_origin)
    # a failure the run recorded is kept as data, and printed
    recorded = {("mallory_2025", "edge/unix_origin")}
    table = attribution.fixed_point_outputs(chosen, contexts, recorded=recorded)
    assert "mallory_2025 failed on edge/unix_origin" in capsys.readouterr().err
    assert table[["config_id", "family"]].to_numpy().tolist() == [
        ["mallory_2025", "spikes"],
        ["mallory_2025", "lfp"],
        ["gupta_2010", "spikes"],
        ["gupta_2010", "lfp"],
    ]
    failed = table[table["config_id"] == "mallory_2025"]
    assert (failed["error"] == "ValueError: made to fail").all()
    assert failed[list(attribution.Y_NAMES)].isna().all().all()
    gupta = table[table["config_id"] == "gupta_2010"].set_index("family")
    assert gupta.loc["spikes", "events_per_minute"] == gupta.loc["lfp", "events_per_minute"]
    assert gupta.loc["spikes", "f1"] != gupta.loc["lfp", "f1"]
    assert (gupta["reason"] == attribution.FIXED_POINTS["gupta_2010"]).all()
    # one the run did not record stops the analysis
    unrecorded = "mallory_2025 raised on edge/unix_origin.*the run recorded no such failure"
    with pytest.raises(RuntimeError, match=unrecorded):
        attribution.fixed_point_outputs(chosen, contexts)

    # and an error building a call's inputs is never a method's failure
    def broken_inputs(session, config):
        message = "no intervals"
        raise KeyError(message)

    monkeypatch.setattr(attribution, "behavior_intervals", broken_inputs)
    with pytest.raises(KeyError, match="no intervals"):
        attribution.fixed_point_outputs(chosen, contexts, recorded=recorded)


def test_recorded_failures(attribution, run, short_run, tmp_path):
    copy = tmp_path / "run"
    shutil.copytree(short_run, copy)
    assert attribution.recorded_failures(copy) == set()
    failures = pd.DataFrame(
        {
            "session_id": ["reference/1", "reference/2", "reference/3"],
            "method": ["recipe:mallory_2025", "Kay_ripple_detector", "recipe:gupta_2010"],
            "setting": ["literature", "default", "literature"],
            "error": ["ValueError: a", "ValueError: b", "ValueError: c"],
        }
    )
    run._write_table(failures, copy / "conditions" / "reference" / "failures.csv")
    assert attribution.recorded_failures(copy) == {
        ("mallory_2025", "reference/1"),
        ("gupta_2010", "reference/3"),
    }


def test_smoke(attribution, recipes, short_run, monkeypatch):
    checked = []
    monkeypatch.setattr(
        attribution,
        "check_report",
        lambda directory, parameters: checked.append((directory, parameters)),
    )
    report = attribution.smoke("lfp", short_run, workers=4)
    # the run's validation report is checked first, as before every analysis
    assert checked == [(short_run, attribution.reference_parameters(short_run))]
    d = len(attribution.factor_space(recipes, "lfp"))
    assert report["d"] == d
    assert report["n_in_space"] == 4
    assert report["configurations_timed"] == attribution.SMOKE_CONFIGURATIONS
    assert report["configurations"]["sobol_128"] == 128 * (d + 2)
    hours = report["hours"]["sobol_256"]
    per = report["seconds_per_configuration"]
    assert hours == pytest.approx(256 * (d + 2) * attribution.K * per / 4 / 3600)


def test_the_command_line(attribution, recipes, short_run, tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        attribution.main(["--run-name", "x", "--family", "lfp", "--workers", "0"])
    assert "--workers must be at least 1" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        attribution.main(["--run-name", "x", "--family", "sharp_wave"])
    copy = tmp_path / "run"
    shutil.copytree(short_run, copy)
    results = tmp_path / "results"
    names = ("gupta_2010", "karlsson_2009", "igata_2021", *attribution.in_space_ids("lfp"))
    kept = [c for c in recipes if c.config_id in names]
    monkeypatch.setattr(attribution, "RECIPES", tuple(kept))
    monkeypatch.setattr(attribution, "check_report", lambda directory, parameters: None)
    arguments = ["--run-name", "x", "--family", "lfp", "--workers", "1"]
    arguments += ["--run-directory", str(copy), "--results-directory", str(results)]
    attribution.main([*arguments, "--analysis", "oat"])
    written = sorted(path.name for path in results.iterdir())
    assert written == [
        "lfp_factor_space.csv",
        "lfp_fixed_points.csv",
        "lfp_in_space.csv",
        "lfp_oat.csv",
        "lfp_reference.csv",
        "lfp_sensitivity.csv",
    ]
    in_space = pd.read_csv(results / "lfp_in_space.csv")
    # the fixed points and the family's templates, not the other family's
    listed = [c.config_id for c in kept if c.config_id != "igata_2021"]
    assert in_space["config_id"].tolist() == listed
    assert in_space["in_space"].sum() == 4
    # below the minimum, every table says so, the override not given
    assert (in_space["caveat"] == "rests on 4 methods, below the design's 8").all()
    assert attribution.family_caveat(8, below_minimum=True) == ""
    # every fixed point, scored for the family
    fixed = pd.read_csv(results / "lfp_fixed_points.csv")
    assert fixed[["config_id", "family"]].to_numpy().tolist() == [
        [c.config_id, "lfp"] for c in kept if c.config_id in attribution.FIXED_POINTS
    ]
    assert fixed[list(attribution.Y_NAMES[:3])].notna().all().all()
    perturbed = pd.read_csv(results / "lfp_sensitivity.csv")
    assert set(perturbed["config_id"]) == set(attribution.in_space_ids("lfp"))
    assert "template values no perturbation changes" in capsys.readouterr().err
    rows = pd.read_csv(copy / "attribution" / "lfp_oat.csv.gz")
    assert set(rows["y"]) == set(attribution.Y_NAMES)
    # four represented LFP methods: no Sobol or Shapley analysis; all runs the
    # one-at-a-time analysis first, and a named one is refused at once
    refused = "4 represented methods, fewer than 8: no Sobol or Shapley"
    (results / "lfp_oat.csv").unlink()
    with pytest.raises(SystemExit, match=refused):
        attribution.main([*arguments, "--analysis", "all"])
    assert (results / "lfp_oat.csv").exists()
    assert not (results / "lfp_sobol.csv").exists()
    monkeypatch.setattr(attribution, "verify_family", None)
    for analysis in ("sobol", "shapley"):
        with pytest.raises(SystemExit, match=refused):
            attribution.main([*arguments, "--analysis", analysis])


def test_the_override_runs_a_family_below_the_minimum_labelled(
    attribution, recipes, short_run, tmp_path, monkeypatch, capsys
):
    copy = tmp_path / "run"
    shutil.copytree(short_run, copy)
    results = tmp_path / "results"
    kept = [
        c for c in recipes if c.config_id in ("gupta_2010", *attribution.in_space_ids("lfp"))
    ]
    monkeypatch.setattr(attribution, "RECIPES", tuple(kept))
    monkeypatch.setattr(attribution, "check_report", lambda directory, parameters: None)
    sobol = attribution.sobol
    asked = []

    def small_sobol(family, run_directory, *, n, workers):
        asked.append(n)
        return sobol(family, run_directory, n=4, workers=workers)

    monkeypatch.setattr(attribution, "sobol", small_sobol)
    drawn = []
    for name in ("plot_sobol", "plot_shapley"):
        monkeypatch.setattr(
            attribution, name, lambda *arguments, name=name, **options: (name, options)
        )
    monkeypatch.setattr(attribution, "_save_figure", lambda path, figure: drawn.append(figure))
    arguments = ["--run-name", "x", "--family", "lfp", "--workers", "1"]
    arguments += ["--run-directory", str(copy), "--results-directory", str(results)]
    attribution.main([*arguments, "--analysis", "all", "--below-minimum", "--sobol-n", "128"])
    assert asked == [128]
    caveat = "rests on 4 methods, below the design's 8; the maintainer chose to run it"
    assert caveat in capsys.readouterr().err
    written = sorted(path.name for path in results.iterdir())
    assert [name for name in written if name.endswith(("_sobol.csv", "_shapley.csv"))] == [
        "lfp_shapley.csv",
        "lfp_sobol.csv",
    ]
    tables = [*results.iterdir(), *(copy / "attribution").iterdir()]
    assert len(tables) == len(written) + 3
    for path in tables:
        assert (pd.read_csv(path)["caveat"] == caveat).all(), path.name
    assert {name for name, _ in drawn} == {"plot_sobol", "plot_shapley"}
    assert all(options == {"caveat": caveat} for _, options in drawn)

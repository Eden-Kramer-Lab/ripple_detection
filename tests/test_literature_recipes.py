"""The packaged default inventories run through the simulation demonstration."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ripple_detection import load_literature_parameters

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "literature_recipes.py"


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("literature_recipes", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["literature_recipes"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def recipes():
    from ripple_detection import literature_methods

    return literature_methods


@pytest.fixture(scope="module")
def recording(example):
    return example.make_recording(duration=45.0)


@pytest.fixture(scope="module")
def results(example, recording):
    return example.run_all(recording)


def test_every_surveyed_paper_has_a_recipe_or_a_reason(recipes):
    rows = {entry.row for entry in recipes.RECIPES} | set(recipes.NOT_REPRODUCED)
    assert rows == set(range(len(load_literature_parameters())))


def test_each_recipe_names_its_paper_as_the_survey_does(recipes):
    survey = load_literature_parameters()
    for entry in recipes.RECIPES:
        author = survey.loc[entry.row, "First Author"]
        assert entry.paper.startswith(author), (entry.row, entry.paper)
        assert str(survey.loc[entry.row, "Year"]) in entry.paper, (entry.row, entry.paper)
        assert entry.note, entry.paper


def test_every_recipe_runs(recipes, results):
    defaults = results.loc[results.configuration == "default"]
    assert set(defaults.method) == {entry.run.__name__ for entry in recipes.RECIPES}
    assert len(defaults) == len(recipes.RECIPES)
    assert not results.duplicated(["method", "configuration"]).any()
    assert set(
        results.loc[
            results.configuration != "default", ["method", "configuration"]
        ].itertuples(index=False, name=None)
    ) == {
        ("olafsdottir_2015", "bayesian_candidates"),
        ("olafsdottir_2017", "trajectory"),
    }
    assert (results.n_events >= 0).all()


def test_the_demonstration_records_warnings_instead_of_hiding_them(results):
    # The column records any warning a method raises; none does on the demo.
    assert (results.warnings.fillna("") == "").all()


@pytest.mark.parametrize(
    ("name", "configuration", "options", "baseline"),
    [
        ("kaefer_2020", "default", {}, [[0, 2]]),
        ("olafsdottir_2015", "default", {"minimum_active_units": 0}, [[0, 12]]),
        ("olafsdottir_2015", "bayesian_candidates", {"minimum_active_units": 7}, [[0, 12]]),
        ("olafsdottir_2017", "default", {"analysis": "arm"}, [[0, 12]]),
        ("olafsdottir_2017", "trajectory", {"analysis": "trajectory"}, [[0, 12]]),
    ],
)
def test_demo_rows_reproduce_from_recorded_options(
    recipes, example, recording, results, name, configuration, options, baseline
):
    from dataclasses import replace

    row = results.loc[
        (results.method == name) & (results.configuration == configuration)
    ].iloc[0]
    assert json.loads(row.options) == options
    assert json.loads(row.supplied_baseline_intervals) == baseline
    configured = replace(recording, baseline_intervals=np.array(baseline))
    events = recipes.run_method(row.method, configured, **json.loads(row.options))
    for column, value in example.score(events, recording.session.ripple_windows).items():
        assert row[column] == value
    assert row.doi == events.attrs["doi"]
    assert row.role == events.attrs["role"]
    # Kaefer's configuration must not change the baseline for subsequent calls.
    np.testing.assert_array_equal(recording.baseline_intervals, [[0, 12]])


@pytest.mark.parametrize("name", ["olafsdottir_2015", "olafsdottir_2017"])
def test_demo_retains_both_broad_and_filtered_inventories(results, name):
    rows = results.loc[results.method == name]
    broad = rows.loc[rows.configuration == "default"].iloc[0]
    filtered = rows.loc[rows.configuration != "default"].iloc[0]
    assert 0 < filtered.n_events < broad.n_events


def test_every_recipe_returns_events_inside_the_recording(recipes, recording):
    for entry in recipes.RECIPES:
        events = recipes.bounds(entry.run(recording))
        assert events.shape[1] == 2, entry.paper
        assert np.all(events[:, 0] <= events[:, 1]), entry.paper
        assert np.all(events >= recording.time[0]), entry.paper
        assert np.all(events <= recording.time[-1]), entry.paper


# Recipes that find no simulated ripple on the demonstration, and why.
NO_RECALL_ON_THE_DEMONSTRATION = {
    # A third of the 40 place cells never fire within one silence-bounded
    # group here: 8 groups reach a fifth of them, none a third.
    "foster_2006": "one third of the probe cells",
}


def test_every_recipe_finds_the_simulated_ripples(results):
    """Not a claim about the papers: a recipe that finds nothing on data full
    of ripples is more likely broken than strict, unless its criterion is
    known to exceed what the demonstration's cells supply."""
    missed = set(results.loc[results.recall == 0, "method"])
    assert missed == set(NO_RECALL_ON_THE_DEMONSTRATION)


def test_pfeiffer_recipe_retains_data_on_both_sides_of_missing_lfp(recipes, example):
    rec = example.make_recording(duration=45.0)
    rec.session.lfps[32_000] = np.nan
    envelope = rec.envelope((150.0, 250.0))
    assert np.isfinite(envelope[:32_000]).all()
    assert np.isnan(envelope[32_000]).all()
    assert np.isfinite(envelope[32_001:]).all()
    events = recipes.pfeiffer_2015(rec)
    assert (events.end_time < rec.time[32_000]).any()
    assert (events.start_time > rec.time[32_000]).any()
    assert not (
        (events.start_time <= rec.time[32_000]) & (events.end_time >= rec.time[32_000])
    ).any()


@pytest.mark.parametrize("gap", ["nan", "timestamp"])
def test_bush_does_not_merge_across_missing_data(recipes, monkeypatch, gap):
    time = np.arange(1000) / 1000
    trace = np.zeros(1000)
    trace[100:160] = trace[170:230] = 1
    trace[125:135] = trace[195:205] = 10
    if gap == "nan":
        trace[160:170] = np.nan
    else:
        time[165:] += 0.001
    spikes = np.zeros((1000, 10))
    spikes[130] = spikes[200] = 1
    session = SimpleNamespace(
        time=time, sampling_frequency=1000, speed=np.zeros(1000), multiunit=spikes
    )
    rec = recipes.Recording(session, np.ones(10, dtype=bool), np.ones(10, dtype=bool))
    monkeypatch.setattr(rec, "rate", lambda *args: trace)
    events = recipes.bounds(recipes.bush_2022(rec))
    np.testing.assert_allclose(events, [[time[100], time[159]], [time[170], time[229]]])


def test_diba_recipe_keeps_the_fixed_window(recipes):
    time = np.arange(2000) / 1000
    spikes = np.zeros((len(time), 5))
    spikes[[500, 520, 540, 560, 580], np.arange(5)] = 1
    session = SimpleNamespace(
        time=time, sampling_frequency=1000, speed=np.zeros(len(time)), multiunit=spikes
    )
    rec = recipes.Recording(session, np.ones(5, dtype=bool), np.ones(5, dtype=bool))
    np.testing.assert_allclose(
        recipes.bounds(recipes.diba_2007(rec, behavior_intervals=[[0.0, 2.0]])), [[0.5, 0.8]]
    )


def test_kaefer_fft_label_detects_demonstration_ripples(results):
    result = results.loc[results.row == 18].iloc[0]
    assert result.n_events > 0
    assert result.recall > 0


def test_the_implementation_guide_lists_every_method(recipes):
    import re

    guide = Path(__file__).resolve().parents[1] / "docs" / "literature" / "implementation.md"
    table = guide.read_text().split("## Per-paper entry points")[1]
    listed = set(re.findall(r"`([a-z0-9_]+)`", table))
    assert listed == set(recipes.list_methods().name)


def test_the_measured_data_walkthrough_runs(tmp_path):
    """examples/measured_walkthrough.py, from loaders to saved provenance."""
    path = SCRIPT.with_name("measured_walkthrough.py")
    spec = importlib.util.spec_from_file_location("measured_walkthrough", path)
    assert spec is not None
    assert spec.loader is not None
    walkthrough = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(walkthrough)
    results = walkthrough.main(tmp_path)
    assert all(len(events) for events in results.values()), {
        name: len(events) for name, events in results.items()
    }
    assert results["shin_2019_candidates"].attrs["options"]["stage"] == "decoding_candidates"
    chenani = results["chenani_2019"]
    assert chenani.attrs["behavior_intervals"] is not None
    for name in results:
        assert (tmp_path / f"{name}.csv").exists()
        assert (tmp_path / f"{name}.json").exists()


def test_walkthrough_bins_only_spikes_within_recorded_samples():
    """Spikes before the recording, after its last sample's period, or inside a
    timestamp gap belong to no LFP sample and must not be counted."""
    path = SCRIPT.with_name("measured_walkthrough.py")
    spec = importlib.util.spec_from_file_location("measured_walkthrough_binning", path)
    assert spec is not None
    assert spec.loader is not None
    walkthrough = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(walkthrough)
    origin = 1_700_000_000.0
    time = origin + np.r_[np.arange(1000), np.arange(2000, 3000)] / 1000  # 1 s gap
    inside = origin + np.array([0.0, 0.5005, 0.999, 2.0, 2.9994])
    outside = origin + np.array([-0.5, 1.5, 3.5, 100.0])
    counts = walkthrough.spike_counts(time, [np.sort(np.r_[inside, outside])])
    assert counts.sum() == len(inside)
    assert counts[-1, 0] == 1  # only the spike at 2.9994 s, not those after the end
    assert counts[999, 0] == 1  # only 0.999 s, not the spike in the gap at 1.5 s

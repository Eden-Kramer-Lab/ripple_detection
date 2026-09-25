"""The packaged default inventories run through the simulation demonstration."""

import importlib.util
import sys
import unicodedata
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ripple_detection import load_literature_parameters

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "literature_recipes.py"


def _ascii(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


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
        author = _ascii(survey.loc[entry.row, "First Author"])
        assert entry.paper.startswith(author), (entry.row, entry.paper)
        assert str(survey.loc[entry.row, "Year"]) in entry.paper, (entry.row, entry.paper)
        assert entry.note, entry.paper


def test_every_recipe_runs(recipes, results):
    assert len(results) == len(recipes.RECIPES)
    assert (results.n_events >= 0).all()


def test_every_recipe_returns_events_inside_the_recording(recipes, recording):
    for entry in recipes.RECIPES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            events = recipes.bounds(entry.run(recording))
        assert events.shape[1] == 2, entry.paper
        assert np.all(events[:, 0] <= events[:, 1]), entry.paper
        assert np.all(events >= recording.time[0]), entry.paper
        assert np.all(events <= recording.time[-1]), entry.paper


def test_most_recipes_find_the_simulated_ripples(results):
    """Not a claim about the papers: a recipe that finds nothing on data full
    of ripples is more likely broken than strict."""
    assert (results.recall > 0).mean() > 0.8


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
    np.testing.assert_allclose(recipes.bounds(recipes.diba_2007(rec)), [[0.5, 0.8]])


def test_kaefer_fft_label_detects_demonstration_ripples(results):
    result = results.loc[results.row == 18].iloc[0]
    assert result.n_events > 0
    assert result.recall > 0

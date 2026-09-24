"""The per-paper recipes in examples/literature_recipes.py run and cover the survey."""

import importlib.util
import sys
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pytest

from ripple_detection import load_literature_parameters

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "literature_recipes.py"


def _ascii(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


@pytest.fixture(scope="module")
def recipes():
    spec = importlib.util.spec_from_file_location("literature_recipes", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["literature_recipes"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def recording(recipes):
    return recipes.make_recording(duration=45.0)


@pytest.fixture(scope="module")
def results(recipes, recording):
    return recipes.run_all(recording)


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

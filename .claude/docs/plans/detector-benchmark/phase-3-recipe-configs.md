# Phase 3 — Recipes as declarative configs, reproducing today's events exactly

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#recipe-executor)

Rewrite the 57 literature recipes from functions into data: `RecipeConfig`s run by one executor
over a registry of components, in `examples/benchmark/recipe_configs.py`. The runner (phase 4)
needs recipes it can call uniformly, and attribution (phase 6) needs recipes it can take apart
and recombine. **Every recipe must return exactly the events it returns today.**

**Inputs to read first:**

- [examples/literature_recipes.py](../../../../examples/literature_recipes.py) — all of it: `Recording` (59-150), `make_recording` (153-178), helpers (181-220), `Recipe`/`recipe` (226-243), the 57 recipe functions (247-1168), `NOT_REPRODUCED` (1171), `score`/`run_all`/`main` (1185-1223).
- [tests/test_literature_recipes.py](../../../../tests/test_literature_recipes.py) — how the script is imported (`importlib`, `sys.modules`) and what is asserted.
- Phase 2's `score` in `examples/literature_recipes.py` (this phase runs after phase 2; both edit the file).
- [tests/test_snapshots.py](../../../../tests/test_snapshots.py) and `tests/snapshots/` — pytest-snapshot usage (`snapshot.assert_match`, `--snapshot-update`).
- `docs/literature/README.md` "Recipes" section — describes the script; updated here.

**Contracts referenced:**

- [Recipe config](shared-contracts.md#recipe-config) — defined here; hashability and determinism must not be weakened (phase 6 caches by value).
- [Primary expression](shared-contracts.md#primary-expression) — every config carries one; the rule is tested here.

**Designs referenced:** [recipe executor](designs.md#recipe-executor), [recipe classification](designs.md#recipe-classification).

## Tasks

Three commits, in this order, so exactness is checked where it can be (in-process, both
implementations present) and guarded afterwards by a snapshot that survives CI's other platforms
and dependency floors:

- **Commit 1: rounded snapshot from today's functions.** Add
  `tests/test_literature_recipes.py::test_recipe_events_match_snapshot`: on the module's 45 s
  recording, for every recipe, the event count and the bounds rounded to 6 decimals (the rounding
  `tests/test_snapshots.py:127-129` uses), as JSON keyed by `f"{row:02d}:{paper}"`, compared with
  `snapshot.assert_match`. Generate with `--snapshot-update`, check it has 57 keys and events where
  the results CSV shows them, and commit the test and snapshot alone.
- **Commit 2: configs beside the functions.** Create `examples/benchmark/recipe_configs.py` per
  [designs.md#recipe-executor](designs.md#recipe-executor): the config types, `step()`,
  `PlusSamples`, `bounds`; `Recording` copied from `literature_recipes.py:59-150` with per-instance
  caches and the additions in the design; the registries and `CORES`; `run_pipeline`; `RECIPES`,
  one `RecipeConfig` per recipe in row order, `primary_expression` by the contract's rule, `note`
  per the design's notes rule. Follow the classification table; move a recipe to a less specific
  core only when exactness requires it, with a one-line comment saying why. Shared sub-rules become
  `Pipeline` or signal constants (`PFEIFFER_2015_SWRS`, `TIROLE`, `MICHON`, `FAROOQ`,
  `POPULATION_WITH_RIPPLE_PEAK`, `KARLSSON_RULE`, `ZUGARO_RIPPLE_PEAKS`), not functions. Add
  `test_configs_reproduce_the_functions`: for every recipe, `bounds(run_pipeline(config.pipeline,
  rec))` equals `bounds(function(rec))` **exactly** (`np.array_equal`) on the module's 45 s
  recording. Also run the same comparison once by hand on `make_recording(90.0)` and record the
  result in the PR (the suite has no slow-test marker, and 90 s would slow every run). Iterate
  until both pass.
- **Commit 3: remove the functions.** Rewrite `examples/literature_recipes.py` on top of the
  configs: keep the module docstring (updated to say where the configs live),
  `SAMPLING_FREQUENCY`, `RUNNING_INTERVALS`, `RIPPLE_TIMES`, `OUTPUT`, `make_recording` (now
  passing `running_intervals`), `NOT_REPRODUCED`, `score` (phase 2's), `run_all`, `main`; import
  `Recording`, `RECIPES`, `run_pipeline`, `bounds` from `recipe_configs` after
  `sys.path.insert(0, str(Path(__file__).with_name("benchmark")))`, so `recipes.bounds` still
  resolves. **Delete** the 57 recipe functions, `Recipe`, `recipe`, and the helpers
  (`within_duration`, `within_intervals`, `only_in`, `zugaro_ripple_peaks`, `_spiking_filter`,
  `_michon`, `_pfeiffer_2015_swrs`, `_tirole`, `_farooq`, `_population_with_ripple_peak`,
  `_karlsson_rule`), and `test_configs_reproduce_the_functions` with them. `run_all` calls
  `run_pipeline(entry.pipeline, rec)`. In `tests/test_literature_recipes.py`, the loader fixture
  also puts `examples/benchmark` on `sys.path` (module-scoped, removed afterwards) and
  `test_every_recipe_returns_events_inside_the_recording` (`:60-68`) calls
  `run_pipeline(entry.pipeline, recording)` instead of `entry.run(recording)`. The snapshot passes
  **without** `--snapshot-update`; `uv run python examples/literature_recipes.py` leaves
  `examples/literature_recipes_results.csv` unchanged (`git diff --exit-code`).
- Create `examples/benchmark/README.md` with a first section, "Recipe configs": what a config is,
  the four cores, `PlusSamples`, how to add a recipe, the exactness rule. Phases 4-6 extend this
  file.
- Lint and format `examples/benchmark/` with `uv run ruff check` / `uv run ruff format` by hand
  (the pre-commit hooks cover only `src/` and `tests/`); type hints throughout, though mypy does
  not run on examples.
- **Docs:** `docs/literature/README.md` "Recipes" section: configs live in
  `examples/benchmark/recipe_configs.py`, run by `examples/literature_recipes.py`. CLAUDE.md:
  mention `examples/benchmark/recipe_configs.py` beside the recipes command under Development
  Commands → Testing. `tests/CLAUDE.md` item 11: the snapshot and the configs. No CHANGELOG entry
  (examples only, no user-facing API).

## Deliberately not in this phase

- Changing any recipe's behavior, parameters or notes (beyond inlining helper docstrings) — even
  known deviations. A recipe fix is a separate commit after this phase, with its own snapshot
  update and a stated reason.
- The CSV corrections to the literature survey — on hold by the maintainer; untouched.
- Attribution templates and factor spaces — phase 6. No "canonical form" fields on the configs.
- Running recipes on network sessions — phase 4. `Recording.from_session` gets a unit test on a
  hand-built `SimulatedSession`, nothing more.
- Moving configs into `src/` (overview decision 10).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_recipe_events_match_snapshot` | Committed from today's functions (commit 1); passes unchanged after commits 2 and 3. |
| `test_configs_reproduce_the_functions` (commit 2 only) | Exact bounds equality, config against function, for every recipe on the 45 s recording (the 90 s check is manual, in the PR). |
| `test_every_surveyed_paper_has_a_recipe_or_a_reason`, `test_each_recipe_names_its_paper_as_the_survey_does`, `test_every_recipe_runs` and the rest of today's file | Pass unchanged in meaning. |
| `test_primary_expression_follows_trigger` | Each config's `primary_expression` equals the contract's rule applied to its `trigger`. |
| `test_notes_are_self_contained` | No `note` contains "See _". |
| `test_configs_are_hashable_and_kinds_exist` | `hash(config)` works for every recipe; every `Step.kind` found anywhere in a config is a key of the registry for its slot. |
| `test_partner_pipelines_run_once` | Running Yang 2024 and Grosmark 2016 on one `Recording` runs the shared Zugaro partner once (count via a wrapped registry entry). |
| `test_caches_are_per_recording` | Two `Recording`s do not share cached traces, and a discarded one is garbage-collected (`weakref`). |
| `test_plus_samples_resolves` | `PlusSamples(0.04, 1)` resolves to `0.04 + 1 / fs` for the recording's rate. |
| `test_recording_from_session_groups` | A `SimulatedSession` with `unit_types` `["place", "pyramidal", "interneuron"]` gives `place_cells == [T, F, F]`, `pyramidal == [T, T, F]`; `running_intervals` passed through. |
| `test_classification_is_recorded` | The count of each core type equals the classification table in the design, or the test's expected counts were updated in the same commit as a documented move. |
| CI | Green, including the dependency-floors job. If the floors job disagrees with the rounded snapshot and the difference is floating point (an event moving by one sample, a bound in the sixth decimal), snapshot event counts plus bounds rounded to 3 decimals instead, and say so in the PR; any other difference is a bug. |
| manual | `examples/literature_recipes_results.csv` unchanged after rerunning the script (90 s). |

## Fixtures

The existing module-scoped 45 s recording in `tests/test_literature_recipes.py`; the new snapshot
file under `tests/snapshots/`. Nothing else.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind): no recipe function or helper remains in `literature_recipes.py`.
- User-facing documentation listed as tasks is updated, not deferred.
- The three commits are in order, the snapshot file is byte-identical across them, and the exact in-process comparison passed before the functions were deleted.

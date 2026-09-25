# Phase 6 — Attribution: which rule components explain recipes' disagreement

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#attribution)

Recipes differ in many components at once (trace, smoothing, normalization period, threshold,
bounds, duration limits, merging, speed rule, cell-count rule, state restriction, coincidence).
This phase measures how much each component matters: one factor at a time from a survey-median
reference, variance-based Sobol indices over the space the literature spans, and Shapley
decompositions of the difference between particular recipes.

**Inputs to read first:**

- Phase 3: `examples/benchmark/recipe_configs.py` (`RecipeConfig`, `run_recipe`,
  `make_recording`, `RECIPES`) and the installed public primitives.
- Phase 4: `examples/benchmark/conditions.py` (`simulate_condition`, `conditions`) — attribution
  uses the reference condition's sessions, re-simulated by seed.
- Phase 2: `match_events`.
- Phase 5: `paired_bootstrap` in `examples/benchmark/analyze.py`.

**Contracts referenced:**

- [Recipe config](shared-contracts.md#recipe-config) — method configurations call the package; only independently verified experimental templates are decomposable.
- [Primary expression](shared-contracts.md#primary-expression) — for context only: `Y` uses one expression per family ([designs.md#attribution](designs.md#attribution), "Outputs `Y`"), not each recipe's primary expression.
- [Conditions](shared-contracts.md#conditions) — the reference sessions are `simulate_condition(reference, k)`, `k = 0..4`.
- [Benchmark outputs](shared-contracts.md#benchmark-outputs) — attribution writes under `output/<run_name>/attribution/` and `results/<run_name>/attribution/`.

**Designs referenced:** [attribution](designs.md#attribution).

## Tasks

- `examples/benchmark/attribution.py`:
  - Define the experimental `Step`, `ThresholdCore`, `Pipeline` types and `run_pipeline`
    using public package primitives, as in the design. Keep the named-method reference
    path as `run_recipe`; experimental templates never replace it.
  - Families (`spikes`, `lfp`) from each verified template's signal source; template dataclasses
    `SpikeTemplate` and `LfpTemplate` with the design's factors, and `compile(template) -> Pipeline`.
  - `factor_space(recipes, family)` by the design's range-and-levels rule; `template_of(config)`;
    `in_space(config)` (event equality on the `K` reference sessions, per the design);
    `reference_template(family)` (median/mode). Apply the design's stop rule: fewer than 8 in-space
    recipes in a family → report to the maintainer before running Sobol or Shapley on it.
  - `evaluate_config(pipeline, recordings) -> dict[str, float]`: the `Y`s in the design averaged
    over the `K = 5` reference sessions, memoized by `(pipeline, replicate)`; a bounded analysis context per
    session. Do not add persistent caches to mutable public recordings; any local cache
    requires explicit immutable inputs, release and invalidation tests.
  - `one_at_a_time(family)`, `sobol(family, n=256)`, `shapley_pairs(family)` per the design;
    `sobol_indices` and `shapley` as pure functions (tested on known cases).
  - A CLI: `uv run python examples/benchmark/attribution.py --run-name NAME --family spikes|lfp
    [--analysis oat|sobol|shapley|all] [--workers N] [--smoke]`, parallel over configurations with
    `ProcessPoolExecutor`, writing `output/<run_name>/attribution/<family>_<analysis>.csv.gz` (one
    row per configuration × `Y`) and summaries plus figures (bars of first-order and total indices
    with intervals; Shapley waterfalls per pair, with Monte Carlo standard errors; one-at-a-time ΔY
    tables) to `results/<run_name>/attribution/`. Plotting functions import matplotlib inside
    themselves; tests do not call them (dependency-floors job).
- **Smoke test first**: `--smoke` evaluates 20 configurations on one session, reports seconds per
  configuration, and extrapolates each analysis's cost (`N (d + 2) K` for Sobol; `2^|D| K` or
  `128 |D| K` per Shapley pair). If Sobol at `N = 256` extrapolates past 4 hours at the available
  workers, use `N = 128` and say so in the summary; if `N = 128` also exceeds 4 hours, stop and
  report the measurement to the maintainer instead of running it.
- Run all three analyses for both families on the phase-4 run's reference sessions (in `tmux`, by
  hand); record commands and wall times in `examples/benchmark/README.md`.
- Fixed points: list all configured methods without a verified template and the reason,
  with their public-call `Y`s on the same sessions, so no method silently drops out.
- `examples/benchmark/README.md`: an "Attribution" section: the families and factors, the reference
  configurations (printed from `reference_template`), how to read first-order vs total indices and
  Shapley values, and the limits (factor independence assumed by Sobol while real recipes co-vary;
  fixed points not decomposed).
- `tests/CLAUDE.md`: an item for `tests/test_benchmark_attribution.py`.

## Deliberately not in this phase

- Decomposing detector-based, silence-bounded or custom recipes: their internals are not expressed
  as templates (overview risk on non-decomposable recipes).
- Attribution across conditions: reference condition only. Repeating it at other conditions is a
  follow-up if the robustness analysis (phase 5) shows rankings that flip.
- Shapley over all recipe pairs (the design's cost argument).
- Changing recipe configs to make more of them fit a template. The package method and configured options remain the reference.
- Surrogate models, SALib or any new dependency.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/test_benchmark_attribution.py::test_sobol_on_ishigami` | Ishigami function (a = 7, b = 0.1) with `N = 2^13`: first-order ≈ (0.314, 0.442, 0.0), total ≈ (0.558, 0.442, 0.244), within 0.03. |
| `test_shapley_additive_and_efficiency` | For `v(S) = Σ_{i∈S} w_i`, `φ_i = w_i` exactly; for a non-additive `v`, `Σ φ = v(D) − v(∅)` to 1e-9 (exact path); on the 10-factor toy in the design, the Monte Carlo path with `n_permutations=4000` is within 0.05 of exact and its standard errors are positive. |
| `test_in_space_recipes` | On two short reference sessions, every recipe `in_space` reports gives the same events from `compile(template_of(config))` as from `run_recipe(config, recording)`; positive controls and boundary/gap cases also pass; the in-space count per family is pinned (updated deliberately when recipes change). Use 30 s sessions to keep the suite in seconds; the full `K = 5` 600 s check is `in_space` itself at run time. |
| `test_factor_space_is_pinned` | `factor_space(RECIPES, family)` equals a literal written in the test. |
| `test_reference_template` | Median/mode rule on a hand-built list of three templates. |
| `test_evaluate_config_is_memoized` | A repeated configuration does not re-run detection (call count). |
| manual | Smoke-test timings and the chosen `N` recorded; fixed-point table present in the results. |

## Fixtures

Hand-built template lists and toy value functions inline; one 60 s reference session for the
memoization test, re-simulated by seed.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
- Every recipe appears in the results, decomposed or as a fixed point.

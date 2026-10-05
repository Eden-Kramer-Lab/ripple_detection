# Phase 2 — `ripple_detection.evaluate`: matching, overlap, signed boundaries, agreement

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#matching)

A public module for comparing event inventories, against a truth or against each other: one-to-one
matching by IoU, per-pair overlap and signed onset/offset errors, splits and merges, pairwise
agreement between methods with signed differences and error correlation, consensus per true
event, and labelling of events by what they overlap. Users can run it on their own detections
(two detectors on one recording, or detections against hand labels).

**Inputs to read first:**

- [src/ripple_detection/core.py:797-834](../../../../src/ripple_detection/core.py) — `_event_bounds`, the input contract every function reads through.
- [src/ripple_detection/core.py:2107-2171](../../../../src/ripple_detection/core.py) — `_overlaps`: the finite/ordered bounds check at 2129-2138 to extract, and the "touching is not overlap" rule.
- [src/ripple_detection/_call_hints.py:143](../../../../src/ripple_detection/_call_hints.py) — `explain_call_errors`.
- [examples/simulation_study.py:79-105](../../../../examples/simulation_study.py) — `score`, replaced here; `main` and the notebook `examples/simulation_study.ipynb` that read its CSV.
- [examples/literature_recipes.py:87-96](../../../../examples/literature_recipes.py) — the recipes' `score`, also any-overlap, replaced here.
- [README.md:294-352](../../../../README.md) — "How the detectors compare on simulated data", whose table and text change with the scoring.
- Phase 1a's `truth_windows` and `simulate_network_session` (for the integration test only).

**Contracts referenced:**

- [Event inventory input](shared-contracts.md#event-inventory-input) — accepted by every function here.
- [Matching and pair metrics](shared-contracts.md#matching-and-pair-metrics) — implemented here; one-to-one optimal matching and the sign convention must not be weakened.
- [Detector comparison table](shared-contracts.md#detector-comparison-table) — implemented here.

**Designs referenced:** [matching](designs.md#matching), [agreement statistics](designs.md#agreement-statistics).

## Tasks

- **Behavior-preserving extraction:** move the bounds check at `core.py:2129-2138` into
  `core._check_bounds(name: str, bounds: FloatArray) -> None` and call it from `_overlaps`. Suite
  green before and after; separate commit.
- Create `src/ripple_detection/evaluate.py` with `EventInventory = ArrayLike | pd.DataFrame`,
  `EventMatching` (frozen dataclass; properties computed from `pairs` and the overlap counts),
  `match_events`, `compare_detectors`, `consensus_counts`, `label_by_overlap`, per the designs.
  Every public function wrapped with `explain_call_errors`; NumPy docstrings with shapes, the sign
  convention stated in each docstring that returns a signed quantity, and a doctest example that
  prints column names or booleans, not counts. mypy strict clean.
- Export `match_events`, `EventMatching`, `compare_detectors`, `consensus_counts` and
  `label_by_overlap` from `ripple_detection` (`EventInventory` stays in `ripple_detection.evaluate`);
  update `tests/test_public_api.py`.
- **Replace `examples/simulation_study.py:79-105` `score`** with a function built on
  `match_events`: `recall`, `precision`, `f1` one-to-one; `onset_ms`/`offset_ms` the medians of the
  pairs' errors; add `n_split` and `n_merged`. Delete the old body. Rerun
  `uv run python examples/simulation_study.py` (about two minutes), commit the new
  `simulation_study_results.csv`, re-execute `examples/simulation_study.ipynb` in place, and read
  its narrative for claims the new numbers contradict.
- **Replace `examples/literature_recipes.py:87-96` `score`** the same way (`recall` one-to-one,
  `false_positives` = unmatched detected); rerun the script and commit the new results CSV. One
  scoring rule across the examples.
- **Docs:**
  - README "How the detectors compare on simulated data" (`README.md:294`): the hit rule becomes
    "matched one-to-one to a ripple window it overlaps"; regenerate the table from the new CSV;
    edit any bullet whose number moved. The PR description shows the old and new tables side by
    side.
  - README: a subsection "Evaluating detections" under Examples, after "Combining two detectors"
    (`README.md:394`): matching against truth, comparing two detectors with signed differences.
  - CHANGELOG `[2.0.0]` Added: the module; Changed: the examples' scoring is one-to-one.
  - `llms.txt`: a line under "Building blocks for published variants".
  - CLAUDE.md "Core Module Structure": six modules, item for `evaluate.py`.
  - `tests/CLAUDE.md`: an item for `tests/test_evaluate.py`.

## Deliberately not in this phase

- The benchmark runner and any analysis over many sessions — phases 4 and 5.
- Bootstrap, permutation tests, operating curves — `examples/benchmark/analyze.py` in phase 5, not
  the public module (they are benchmark statistics, not something a user needs per recording).
- Greedy or many-to-one matching modes. One rule, one-to-one optimal; `reference_overlaps` and
  `detected_overlaps` already give any-overlap counts for users who want them.
- Clustering helpers — phase 5 calls `scipy.cluster.hierarchy` directly.
- Changing `require_overlap`, `exclude_overlap` or `_overlaps` beyond the check extraction.

## Validation slice

| Test | Asserts |
| --- | --- |
| existing suite | Green before and after the `_check_bounds` extraction. |
| CI | Green, including the dependency-floors job (SciPy 1.10: `linear_sum_assignment(maximize=True)`, `csr_array`, `bmat` must work there). |
| `TestMatchEvents::test_identical_inventories` | Every event matched to itself, IoU 1, errors 0. |
| `TestMatchEvents::test_empty_inputs` | Empty reference or detected: no pairs; recall/precision NaN where the denominator is 0; `f1` NaN only when both empty. |
| `TestMatchEvents::test_pair_metrics_by_hand` | Reference `[[0, 10]]`, detected `[[2, 12]]`: IoU 8/12, coverage 0.8, temporal precision 0.8, onset +2, offset +2. |
| `TestMatchEvents::test_touching_is_not_overlap` | `[[0, 1]]` vs `[[1, 2]]`: no pair, overlap counts 0. |
| `TestMatchEvents::test_optimal_not_greedy` | Reference `[[0, 4], [4.5, 10]]`, detected `[[0, 10], [5, 6]]`: two pairs (d0–r0, d1–r1), not the greedy single pair d0–r1. |
| `TestMatchEvents::test_matching_is_symmetric` | `[[0, 1], [2, 4]]` against `[[0, 4], [2.5, 3]]` gives two pairs both ways; for random inventories with tied IoUs, swapping gives the same pair count, summed IoU, F1 and Jaccard. |
| `TestMatchEvents::test_split_and_merge` | One reference overlapped by three detected: `split_reference == [0]`, one pair (the largest IoU); one detected over two references: `merged_detected == [0]`, one pair. |
| `TestMatchEvents::test_minimum_iou` | A pair at IoU 0.2 is kept at `minimum_iou=0.1`, dropped at 0.3, and still counted in the overlap counts. |
| `TestMatchEvents::test_indices_refer_to_input_rows` | Shuffled inputs give pairs whose indices point at the right rows of the inputs as given. |
| `TestMatchEvents::test_peak_error_from_dataframes` | DataFrames with `peak_time` give `peak_error`; arrays give NaN. |
| `TestMatchEvents::test_boundary_errors_at_other_bounds` | `boundary_errors(narrower)` equals detected minus the narrower bounds for the same pairs. |
| `TestMatchEvents::test_invalid_bounds_raise` | NaN or reversed rows raise `ValueError` naming the input and row. |
| `TestMatchEventsProperties` (hypothesis) | `n_pairs <= min(n_reference, n_detected)`; indices unique on each side; `len(match(a, b).pairs) == len(match(b, a).pairs)`; IoU in (0, 1]; every pair overlaps. |
| `TestCompareDetectors::test_signed_differences_by_hand` | Two hand-built inventories: `jaccard`, `median_onset_difference` (a − b), IQR, `fraction_a_earlier_onset` equal hand values. |
| `TestCompareDetectors::test_with_truth` | `jaccard_true`, `jaccard_false`, `n_shared_truth`, and the error correlations equal hand values (Spearman on ≥ 3 shared events; NaN below 3). |
| `TestCompareDetectors::test_pair_order_and_columns` | One row per unordered pair in mapping order; the contract's columns in order, also for empty inventories; the truth columns present and NaN without `truth`. |
| `TestConsensusCounts::test_by_hand` | Boolean columns and `n_methods` per truth event. |
| `TestLabelByOverlap::test_longest_overlap_wins_and_ties` | Longest overlap's label; ties to the earlier window row; no overlap → `unlabeled`. |
| `TestEvaluateOnSimulatedSession::test_kay_against_ripple_truth` | On a 60 s `simulate_network_session` (only `swr`, SNR (4, 6)), Kay's recall against `truth_windows(..., "ripple")` is above 0.8 and every pair's errors are finite: an end-to-end check that the pieces fit, not a benchmark result (no claim about the sign of the errors). |
| doctests | Each public function's example runs. |

## Fixtures

Hand-built `(n, 2)` arrays and small DataFrames inline in `tests/test_evaluate.py`; a
module-scoped fixture for the one simulated session. Hypothesis strategies generate sorted,
finite, ordered bounds (`tests/test_properties.py` has the house style).

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind): both examples' old `score` bodies are gone.
- User-facing documentation listed as tasks is updated, not deferred; the README table matches the committed CSV.

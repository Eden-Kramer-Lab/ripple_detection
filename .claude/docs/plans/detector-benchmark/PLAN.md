# Detector Benchmark Implementation Plan

**Status:** Phase 1a done (2026-09-26, commits a326acf..94124f9 on `detector-benchmark`, independently reviewed). Its reference-value changes (event rate 0.3/s, interneurons 8-15 Hz) are in [designs.md#parameter-sources](designs.md#parameter-sources). Phase 1b done (2026-09-26, commits 9e089e1 onward, independently reviewed for correctness and test quality; findings fixed). Its implementation choices are in [designs.md#non-events](designs.md#non-events). Phase 2 done (2026-09-27, commits 8763f06..a26bb76, reviewed for spec and correctness, test quality and silent failures; findings fixed). Its departures from the matching sketch are in [designs.md#matching](designs.md#matching). Decision 14 (2026-09-27, [overview.md](overview.md)) adds scoring safeguards to phases 4 and 5: matching at predeclared IoU levels, absolute and paired boundary errors, held-out thresholds. Phase 3 done (2026-09-27, commits df8f0ed onward, reviewed for correctness, test quality, silent failures, comments and types; findings fixed, then simplified with outputs bit-identical). Its input policy, exclusions and primary expressions are in [designs.md#recipe-classification](designs.md#recipe-classification). The maintainer kept the event-rate levels 0.15/0.3/0.6. Phase 4 done (2026-09-28, commits 0fe2d28 onward, reviewed for correctness, test quality, silent failures, comments and types; findings fixed; simplified with outputs bit-identical). The maintainer chose two recorded reference revisions (ripple span, burst gain), 20 validation replicates and an uncommitted measurements.csv; the ready report is `examples/benchmark/validation/v1`, and the full run `v1` (43 conditions, 440 sessions) is in `examples/benchmark/output/v1`. Details in [designs.md#runner](designs.md#runner). Phase 5 done (2026-09-28, commits 952b8ef onward, reviewed for correctness, test quality, silent failures, comments and types; findings fixed; simplified with every result CSV byte-identical). Point inventories are scored by peak containment; the stated trends and the brown-noise confound are in `examples/benchmark/results/v1/trends.md`; details and the simulator follow-up in [designs.md#bootstrap-and-permutation-tests](designs.md#bootstrap-and-permutation-tests). Phase 6 done (2026-09-28, commits 2537821 onward, reviewed for correctness, test quality, silent failures, comments and types; findings fixed; simplified with outputs identical): spikes 14 distinct in-space templates, lfp 4 (run below the minimum by the maintainer's decision), the bound a fraction of the threshold; results in `examples/benchmark/results/v1/attribution`, details in [designs.md#attribution](designs.md#attribution). A compact comparison of run v1 followed (2026-09-30, commits 7584790 onward, reviewed for correctness, test quality, silent failures and comments; findings fixed; simplified with results/v1 byte-identical): `results/v1/compact.md` and its tables, details in [designs.md#bootstrap-and-permutation-tests](designs.md#bootstrap-and-permutation-tests). Event-free false positives followed (2026-09-30, commits 97c3d04 onward, after an external review; reviewed for correctness, test quality, silent failures and comments; findings fixed; simplified with results/v1 byte-identical): every false-positive rate counts only unmatched detections outside every network window, per IoU level; a curve wholly within a budget reads its best tested setting, a lower bound that supports a comparison one way only. Details in [designs.md#operating-curves](designs.md#operating-curves). Next: phase 7.

A systematic evaluation of how well ripple_detection's nine detectors and the literature recipes
(`ripple_detection.literature_methods`) capture events, on simulated sessions whose truth is known: do
they find the same events or different ones, which kinds of event each finds, how much matched
events overlap, and how accurately (signed) each places onsets and offsets against the truth and
against each other; plus operating curves, robustness, which rule components explain their
disagreement, and the effect on event rates and cell participation. The simulator and the
evaluation functions ship in the package for users; the benchmark runner and analyses live in
`examples/benchmark/`. Work happens on branch `detector-benchmark` (PR #26), based on `master`, which
has the `literature-methods` work (PR #25).

## Reading order

For agent invocation, **load only the slice you need**:

0. **Starting or resuming?** Check Status above, then open the first phase not yet done. The order between phases is in [overview.md#rollout-strategy](overview.md#rollout-strategy). The decisions there are settled; ask before reopening one.
1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, contracts/designs it depends on, tasks, validation slice, and fixtures.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need a per-component design?** [designs.md](designs.md).
4. **Need broader scope / decisions / risks / dependency policy?** [overview.md](overview.md).
5. **Implementing or assessing simulator validity?** [simulator-validation.md](simulator-validation.md),
   including rendered-measurement targets, six model alternatives and the pre-benchmark report.

## Files

- [overview.md](overview.md) — every decision made while designing the benchmark, integration points, goals and non-goals, dependency policy, risks
- [shared-contracts.md](shared-contracts.md) — the truth table, non-event table, event-table input, matching result, recipe config, and benchmark output schemas
- [designs.md](designs.md) — generator algorithms, matching, agreement statistics, public-method adapter, Sobol and Shapley estimators, bootstrap
- [simulator-validation.md](simulator-validation.md) — source/measurement conventions, structural sensitivity conditions and validation report requirements
- Phases (each ships as a separable PR):
  - [phase-1a-network-events.md](phase-1a-network-events.md) — latent-event simulator: event types and per-expression truth, in `simulate.py`
  - [phase-1b-non-events.md](phase-1b-non-events.md) — spike-waveform leakage, EMG artifacts, fast-gamma and theta-state bursts
  - [phase-2-evaluation.md](phase-2-evaluation.md) — `ripple_detection.evaluate`: matching, overlap, signed boundary errors, agreement, consensus
  - [phase-3-recipe-configs.md](phase-3-recipe-configs.md) — benchmark call configurations using the installed paper methods
  - [phase-4-runner.md](phase-4-runner.md) — simulator validation report, then benchmark conditions, threshold sweeps, recipes and event-level outputs
  - [phase-5-analyses.md](phase-5-analyses.md) — agreement, boundaries, operating curves, robustness, rates and participation
  - [phase-6-attribution.md](phase-6-attribution.md) — one-component-at-a-time, Sobol and Shapley attribution of disagreement
  - [phase-7-reference-recordings.md](phase-7-reference-recordings.md) — real recordings with released events: verify inputs, run the package method, match and explain differences

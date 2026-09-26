# Detector Benchmark Implementation Plan

**Status:** Not started.

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

## Files

- [overview.md](overview.md) — every decision made while designing the benchmark, integration points, goals and non-goals, dependency policy, risks
- [shared-contracts.md](shared-contracts.md) — the truth table, non-event table, event-table input, matching result, recipe config, and benchmark output schemas
- [designs.md](designs.md) — generator algorithms, matching, agreement statistics, public-method adapter, Sobol and Shapley estimators, bootstrap
- Phases (each ships as a separable PR):
  - [phase-1a-network-events.md](phase-1a-network-events.md) — latent-event simulator: event types and per-expression truth, in `simulate.py`
  - [phase-1b-non-events.md](phase-1b-non-events.md) — spike-waveform leakage, EMG artifacts, fast-gamma and theta-state bursts
  - [phase-2-evaluation.md](phase-2-evaluation.md) — `ripple_detection.evaluate`: matching, overlap, signed boundary errors, agreement, consensus
  - [phase-3-recipe-configs.md](phase-3-recipe-configs.md) — benchmark call configurations using the installed paper methods
  - [phase-4-runner.md](phase-4-runner.md) — benchmark runner: conditions, threshold sweeps, recipes, event-level outputs
  - [phase-5-analyses.md](phase-5-analyses.md) — agreement, boundaries, operating curves, robustness, rates and participation
  - [phase-6-attribution.md](phase-6-attribution.md) — one-component-at-a-time, Sobol and Shapley attribution of disagreement
  - [phase-7-reference-recordings.md](phase-7-reference-recordings.md) — real recordings with released events: verify inputs, run the package method, match and explain differences

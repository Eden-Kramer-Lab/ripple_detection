# Mallory 2025 — The time course and organization of hippocampal replay

## Sources and verification status

Updated September 25, 2026 after reading the user-supplied published supplement,
`science.ads4760_sm.pdf` (37 pages; 6,996,578 bytes). Its cover identifies Science
387, 541 (2025), DOI 10.1126/science.ads4760. Methods pp. 5–8 and Fig. S2 pp. 16–17
were checked directly. This closes the earlier supplement retrieval gap.

- Published main text: matched Zotero attachment V86BPFP4; positional errors on p. 541.
- Published supplement: user-supplied local Downloads copy. SHA-256:
  `11de2fe56bfcd149e24dd81d2b02c3d29156c70f3f46c26bb1cbe1050f2bea59`.
- [Paper code v1.0.0](https://github.com/caitlinmallory/TimeCourseOrganizationOfHippocampalReplay/tree/128513a656bdacfc11a0e0dfbcaf20e2c4515520),
  commit `128513a656bdacfc11a0e0dfbcaf20e2c4515520`, also archived as
  [Zenodo 14237298](https://doi.org/10.5281/zenodo.14237298).
- Earlier comparison source: bioRxiv v1, DOI 10.1101/2024.07.18.604185,
  under the title “Self-avoidance dominates the selection of hippocampal replay.”
  The published supplement now supplies the authoritative text citations below.

Trigger: MUA/SDE candidates followed by decoding criteria on the linear track;
decoding over the entire session in the open field. SWRs provide a control candidate
source. The row combines protocols; its scalar columns do not define one detector.

## Published Methods

| Quantity | Published value and scope | Supplement page |
|---|---|---|
| Spike density | All cells, 1 ms nonoverlapping spike-count bins | 5 |
| Ripple signal | One selected pyramidal-layer tetrode, 150–250 Hz, Hilbert envelope | 5 |
| SDE/SWR smoothing and detection | Gaussian SD 12.5 ms; z-score during speed <5 cm/s; peaks >3 SD; extend to the mean | 5 |
| Track replay | SDE candidates; >66% posterior in one directional map; weighted correlation >0.6; maximum jump <40% of track; coverage >20%; ≥10 cells | 6 |
| Track controls | SWR candidates; increased correlation threshold 0.7; independent rank-order method | 6–7; Fig. S2, 16–17 |
| Rank-order control | Median spike times versus place-field-peak order, separate direction templates; Spearman p<0.05; ≥10 participating cells in a >3-SD SDE | 7 |
| Replay decoding | Track 20 ms; arena 80 ms; text describes overlap by 5 ms, code specifies a 5 ms step | 6–7 |
| Place fields | Movement >5 cm/s; 2 cm track / 2×2 cm arena bins; Gaussian SD 2 cm track / 8 cm arena | 5 |
| Behavioral decoding | Nonoverlapping 400 ms windows; discard sessions with mean error >10 cm | 6 |
| Arena subsequences | Speed <5 cm/s; spread <0.0048L cm; COM jump threshold 0.4√L cm, where L is the number of spatial bins | 7 |
| Arena merging/duration | Merge gaps <20 cm and <50 ms; final duration >50 ms | 7 |
| Arena overlap/range | Primary analysis has no SDE/SWR-overlap requirement or minimum spatial range; Fig. S2 supplies controls | 8 |

The published main text (p. 541) reports mean positional errors of 2.9 cm on the
track and 3.4 cm in the arena. These use the behavioral 400 ms windows, not the
20/80 ms replay windows.

For the rank-order control, positive/negative correlations label forward/reverse
replay. If both direction templates pass, retain the larger absolute correlation.
The released `spearman_median.m` computes median spike times and calls MATLAB
`corr(..., 'type', 'Spearman')`. No shuffle count is specified for event detection.
The 5000 shuffles on supplement pp. 8–9 test replay-pair timing/content statistics;
they must not populate the per-event `Shuffles (#)` field.

## Released implementation and text/code differences

SDE/ripple candidates:

- `load_candidateEventTimes.m` uses Gaussian SD 0.0125 s, speed ≤5 cm/s,
  segment-wise baseline normalization and NaN masking of movement/artifacts.
- `find_candidate_events_2.m` sets peak 3 SD, bounds 0 SD, 70 ms peak separation,
  minimum length 0 and maximum infinity. Candidate peaks sharing a start are
  collapsed; nearby peaks are merged. This is distinct from replay-subsequence
  merging at 50 ms.
- `load_spikeDensity_pyramidal_only.m` selects excitatory cells. The published
  text says all cells; these source descriptions remain different.
- Ripple artifact rejection calls the unprovided `remove_lfp_artifacts_cm` with a
  session threshold and ±0.2 s padding, plus a manual bad-LFP list. The original
  threshold implementation cannot be established from this release.

Track replay:

- The manuscript block of `do_combine_linear_track_replay_events.m` selects
  `spike_filtered` / event choice 6, ≥10 cells, coverage 0.2, correlation 0.6 and
  posterior-map difference 0.33. The latter corresponds to >66.5% in one map;
  the text rounds this to >66%.
- `filter_candidate_events.m` keeps bins with jump <40% of track and posterior
  peak >5/n_bins, merges gaps <50 ms with jump <40%, removes segments <30 ms,
  then keeps the longest segment. `load_replay_criterion.m` sets a 50 ms replay
  minimum. Neither duration rule is an initial SDE duration limit.

Arena replay:

- **The earlier 50/100 ms discrepancy is resolved at the configuration level.**
  `load_AnalysisInformation_cm.m:22–25` contains 100 ms defaults, but
  `load_replayEvents_cm.m:55–61` overrides replay and sequence duration to 50 ms.
  `Open_Field/do_combine_open_field_replay_events.m:40,77`, named by the README
  for the paper figures, independently requires duration ≥50 ms. Text says >50 ms;
  the inclusive code boundary remains a small difference. The 100 ms default is
  not evidence of the final replay cutoff.
- `load_replayEvents_cm.m:56–58` uses spread `0.0012*maze_size^2` and jump
  `0.2*maze_size` cm. For square arenas with 2 cm bins, these equal the published
  `0.0048L` and `0.4√L` thresholds. It also uses `0.2*maze_size` cm for the
  **between-sequence spatial merge**, whereas the published text specifies 20 cm.
  `compute_allSequences_NaNseparated_merge.m` confirms the spatial threshold is
  divided by bin size before comparing bin-coordinate distances; these merge
  rules coincide only when maze size is 100 cm.
- `compute_filtering_binDecoding_cm.m:15` computes `abs(diff(x(:,2)))`, the
  first COM coordinate's change. Supplement p. 8 defines the jump as the
  two-dimensional Euclidean distance. The merging helper does use both spatial
  coordinates; it does not resolve the separate within-sequence jump difference.
- The arena figure-combination script excludes a session flag described as
  decoding worse than 5 cm, whereas the published session-error cutoff is 10 cm.
  It consumes stored tables; the original flag-generation/runtime provenance is
  not established here, so this does not replace the published 10 cm criterion.

Spatial smoothing:

- Track `compute_rateMap.m` uses a nominal one-bin (2 cm) Gaussian through
  `filtfilt`, so 2 cm describes each pass, not the combined kernel.
- Arena configuration sets smoothing 8 and bin size 2; `compute_rateMap.m`
  passes 8/2 to `setUp_gaussFilt.m`. That helper uses the argument as the
  covariance in `mvnpdf`, yielding nominal SD √4 bins ×2 cm =4 cm. Published
  supplement p. 5 explicitly states SD 8 cm. The CSV preserves both values.

These are inspections of the pinned released implementation, not execution of
its MATLAB pipeline or proof of every historical session override. Resolving the
remaining text/code differences needs original configurations or author clarification.

## Inherited from

Open-field decoding follows Widloski & Foster 2022; the published Methods explicitly
scale spread/jump thresholds to arena size and use >50 ms duration. Track criteria
follow the Ambrose/Pfeiffer/Foster lineage. The 3-SD, 12.5-ms candidate rules are
restated directly in the published supplement.

## Current CSV disposition

Seven fields were refined after receiving the supplement: detection notes, decoding
method, spatial smoothing, trajectory, decoding notes, shuffle method and significance.
The row now includes the rank-order control and distinguishes published versus code
smoothing. Existing 3-SD peaks, 12.5-ms smoothing, 150–250 Hz ripple band, 20/80-ms
replay windows and 400-ms behavioral windows are directly confirmed.

All 35 cell citations were updated in the [field-status index](../parameter_verification_2026-09-25.csv).
The [correction ledger](../parameter_corrections_2026-09-25.csv) and
[follow-up history](../parameter_recheck_changes_2026-09-25.csv) preserve the baseline
and current values. No other paper's CSV row changed in this supplement follow-up.

## Package mapping

Tier: B for the linear-track SDE candidate step (the decoding criteria that follow are D); D for open-field replay.     Needs radiatum: n   Needs theta: n   Needs sleep scoring: n
Recipe (SDE candidates, code-exact):
```python
import numpy as np
from ripple_detection import (get_multiunit_population_firing_rate, normalize_signal,
    threshold_by_zscore)
# spikes: (n_time, n_pyramidal_units) counts at 1 kHz; still = speed <= 5 (per session segment)
rate = get_multiunit_population_firing_rate(spikes, 1000.0, smoothing_sigma=0.0125)
z = normalize_signal(rate, normalization_mask=still)
z[~still] = np.nan                                  # code NaNs moving samples: events end at movement
sde = np.array(threshold_by_zscore(z, time, minimum_duration=0.0, zscore_threshold=3))
# code merge: peaks < 70 ms apart -> one event (earlier start, later end); user code on peak times
# then: decode 20 ms / 5 ms bins inside each SDE, keep longest smooth segment, apply
# >66% one-map posterior, |wc|>0.6, max jump<40%, coverage>20%, >=10 cells  -> out of scope (decoder)
```
Tier-A approximation for the SDE step: `multiunit_HSE_detector(time, spikes, speed, 1000, speed_threshold=5, minimum_duration=0.0, zscore_threshold=3, smoothing_sigma=0.0125, normalization_mask=speed <= 5, minimum_active_units=10)`. The 10-cell rule in the paper applies to the replay sub-interval, not the SDE, so it is only a loose proxy.
Ripple events for the fig. S2 variant (B): a single pyramidal-layer channel, `filter_ripple_band(lfp, fs, band=(150, 250))` → `get_envelope` → `gaussian_smooth(0.0125)` → `normalize_signal(mask=still & ~artifact)` → set moving/artifact samples to NaN → `threshold_by_zscore(z, time, 0.0, 3)`. Tier-A approximation: `Kay_ripple_detector` with one channel, `smoothing_sigma=0.0125, zscore_threshold=3, minimum_duration=0.0, speed_threshold=5, normalization_mask=still`. With a single channel, Kay's trace is sqrt(smoothed squared envelope). That is close to the code's smoothed envelope but not identical.
Remaining deviations:

- Peak-separation merge (peaks <= 70 ms apart merged, earlier start / later end) versus the package's gap-based `merge_close_events` or drop-later `exclude_close_events`. Close, not identical. Exact reproduction needs peak times from user code.
- Movement: code-exact via NaN masking (B). The endpoint speed rule (A) drops events touching movement instead of truncating them.
- Units: the published supplement says all cells; the code uses excitatory cells.
- Kernel: the code's `gausswin` window is 5σ wide (±2.5σ), versus the package's Gaussian truncated at 8σ. Minor.
- Artifact rejection for ripples (per-session threshold with ±0.2 s padding, plus a manual bad_lfp list) is user-supplied as NaN or a normalization mask.
- The replay criteria (posterior-map fraction, weighted correlation, jump, coverage, cell count, longest smooth sub-segment) and all open-field detection need a decoder, which is out of scope.
- Published Methods are now checked; the text/code differences above remain explicit.
Smallest package addition (if C): n/a. Optional convenience: a `peak_time` column on the threshold detectors plus a peak-separation merge would make the 70 ms merge direct.

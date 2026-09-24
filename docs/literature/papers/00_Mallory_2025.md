# Mallory 2025 — The time course and organization of hippocampal replay
Source:
- (a) Published Science main text (9 pp): Zotero full-text cache of the published PDF. It contains no Materials and Methods, which are in the online supplement.
- (b) Methods: bioRxiv preprint v1 "Self-avoidance dominates the selection of hippocampal replay" (Mallory, Widloski, Foster; doi 10.1101/2024.07.18.604185; PMC11275714), rendered in a browser and saved to a local copy
- (c) Code: Zenodo 10.5281/zenodo.14237298 (caitlinmallory/TimeCourseOrganizationOfHippocampalReplay v1.0.0, 2024-11-28; this is the code the Science paper cites as ref. 80), in a local copy
- The Science supplement (science.ads4760_sm.pdf) could not be retrieved (HTTP 403 / Cloudflare), so the published methods text itself is unverified. Everything below comes from (b) and (c) and should be checked against the Science supplement.
title verified: yes for (a) (Mallory, Widloski, Foster; Science 387:541–548). (b) is the preprint of the same study under an earlier title.
Trigger: MUA (linear track: spike-density events as candidates, then decoding criteria) + decoding (open field: decoded posterior only)

## Method as implemented
Published main text (a):
- "Decoding on a fine timescale revealed abundant replay during reward-associated immobility (materials and methods)."
- "We obtained similar results when requiring replays to coincide with a sharp-wave ripple or applying alternative replay detection methods (fig. S2)."
- "Although the overall rates of spike density events, sharp-wave ripples, and replays were similar between MEC active or inactive trials (fig. S5)". So SDEs and SWRs are detected as event types in the published study.

Preprint Methods (b), "Spike density and sharp-wave ripple amplitude":
- "Population spike density was computed by totaling the number of spikes from all cells in 1 ms non-overlapping time bins. The LFP from a selected tetrode in the pyramidal layer with visually identified sharp-wave ripples was band-pass filtered between 150 and 250 Hz, and the amplitude was computed as the envelope of the Hilbert transform. Spike density and ripple amplitude were both smoothed with a Gaussian kernel (12.5 ms standard deviation [SD]). Periods in which the rat's speed was below 5 cm/s were z-scored. Peaks in the z-scored spike density or ripple amplitude traces exceeding 3 SD above the mean were identified as spike density or ripple events. The start and end time of each event was defined as the times on either side of the peak at which the z-scored signal crossed the mean."

Preprint Methods (b), "Replay detection – Linear Track" (DETECTION = SDE candidate + decoding criteria):
- "The posterior probability over position and movement direction was computed for each candidate replay event (spike density events). Replay events were required to satisfy the following criteria: >66% of the posterior probability located in one of the directional maps, weighted correlation >0.6, max. jump distance<40% of the track, spatial coverage > 20% of the track and at least 10 cells participating ... We verified our results using SWRs as candidates". Decoding window: "linear track: 20 ms overlapping by 5 ms".

Preprint Methods (b), "Replay detection – Open Field" (DETECTION = decoding only):
- "To identify replay, we decoded the posterior probability over position over the entire session using 80 ms time windows overlapping by 5 ms ... A subsequence was defined as a set of contiguous bins meeting the following criteria: rat speed <5 cm/s, posterior spread (m<0.0048L ...) and posterior COM jump size ... Neighboring subsequences were merged if they were separated by less than 20cm and 50ms. The final sequence was required to have a duration greater than 50ms."
- "Except in [fig. S2], we did not require events to overlap with SWRs or SDEs or impose a minimum spatial coverage threshold. These relatively permissive criteria were selected to avoid detection heuristics that may inadvertently bias the content of replay discovered."

Code (c) confirms the linear track uses SDE candidates and gives the exact detection:
- `do_combine_linear_track_replay_events.m` (block marked "% MANUSCRIPT:"): `candidate_events_to_plot = 'spike_filtered'` gives `event_choice = 6`, the SDE-derived events. It also sets `num_cells_participating_thr = 10`, `override_coverage_thr = 0.2`, `override_weighted_r_thr = 0.6`, `override_posterior_diff_thr = 0.33` (a 0.33 left/right difference is equivalent to > 66.5% in one map), `override_sde_thr = -inf`, `override_ripple_thr = -inf`. README: "Fig S2: as in fig 2, but using ripples instead of SDEs as candidate events".
- `load_spikeDensity_pyramidal_only.m`: `restrict_to_excitatory_cells = 1`; 1 ms bins (`spikeDensityStepSize = 1e-3` in `load_AnalysisInformation_cm.m`). The code uses excitatory cells only, whereas the preprint says "all cells".
- `load_candidateEventTimes.m`, per session segment:
  - Smoothing: `smoothing_sigma = 0.0125` for both spike density and ripple amplitude.
  - Baseline: mean and SD over samples with speed <= `speedThr` (5 cm/s) and, for ripples, not in artifact samples (`bad_inds = unique([artifact_inds; moving_inds])`).
  - Masking: moving (and artifact) samples are set to NaN before peak finding, so events cannot extend into movement.
  - Ripple LFP: a single tetrode (`Experiment_Information.ripple_refs`), band `SWRFreqRange = [150,250]` (comment: "John used 120-170"). Artifacts are removed by `remove_lfp_artifacts_cm(LFP, t, Experiment_Information.artifact_threshold, 0.2, 0.2, ...)`, called with a per-session threshold and ±0.2 s padding (the function body is not in the repo, so the kind of threshold is not visible), plus a hand-made `bad_lfp` list.
- `find_candidate_events_2.m` (used for both SDEs and ripples):
  - `events_lo_std_cutoff = 0; events_hi_std_cutoff = 3; events_min_peak_separation = 0.07; events_min_length = 0.0; events_max_length = inf;`
  - `findpeaks(...,'minpeakheight',3)`, with bounds walked outward while z > 0.
  - Peaks sharing a start are collapsed.
  - Peaks within 70 ms (peak to peak) are merged, taking the earlier start and the larger peak. The later event's end is kept.
  - No minimum or maximum duration.
- `filter_candidate_events.m` (linear track, within each SDE):
  - Decoding bins are kept where the jump is < 40% of the track and the peak posterior is > 5/n_bins.
  - Segments are merged across gaps < 50 ms and jumps < 40% of the track, and segments < 30 ms are removed.
  - Only the longest segment is kept, so the replay is a sub-interval of the SDE.
- `load_replay_criterion.m`: `duration_thr = 0.05; max_jump_distance_thr = 0.4; weighted_r_thr = 0.6; coverage_thr_percent_of_track = 0.20`.
- Open field: the decoder "filtering" method (Analysis_Information: `sequence_posteriorSpreadThr = 10`, `sequence_jumpThr = 20`, `sequence_deltThr = 0.05`, `sequence_deltxThr = 20`, `sequence_spikeDensityThr = -Inf`).

Behavioural windows (analysis): replays were analysed within stopping periods at reward, "the first time the rat entered within 5 cm of the filled reward well at a speed of 3 cm/s or less" (preprint), with rates over 0–10 s. Brain state, radiatum: not used.

## Inherited from
- Open-field replay detection follows Widloski & Foster 2022 (manifest 09; code comments call it "John Widloski's filtering method"). The preprint differs in the numbers: spread threshold scaled to arena size, duration > 50 ms instead of > 100 ms.
- The linear-track criteria cite Ambrose 2016 for the posterior-difference threshold (code comment `posterior_in_map_thr = 0.33; %Ambrose: 0.33`).
- The SDE/ripple rules (12.5 ms smoothing, 3 SD, bounds at mean) match the Pfeiffer & Foster 2013 / Ambrose 2016 lineage, restated rather than cited.

## Code
Zenodo 10.5281/zenodo.14237298. GitHub: https://github.com/caitlinmallory/TimeCourseOrganizationOfHippocampalReplay/tree/v1.0.0.

## Survey CSV discrepancies
- Detection "Decoding": for the linear track (the source of the CSV's own notes and its Min. Cells = 10), candidates are spike-density events: "The posterior probability over position and movement direction was computed for each candidate replay event (spike density events)" (preprint; code `event_choice = 6`). It should read MUA/SDE + decoding criteria for the linear track and decoding for the open field. This is the most important discrepancy.
- MUA Z-score Thresh.: CSV N/A. The value is 3 ("Peaks ... exceeding 3 SD above the mean were identified as spike density ... events").
- MUA smooth: CSV N/A. The value is 12.5 ms ("smoothed with a Gaussian kernel (12.5 ms standard deviation [SD])").
- Min. Duration 50 ms and Combine 50 ms: these are open-field decoding parameters ("merged if they were separated by less than 20cm and 50ms ... duration greater than 50ms"). The linear-track SDE detection has no duration limit (`events_min_length = 0.0; events_max_length = inf`). The linear replay has `duration_thr = 0.05` in the code. The SDE merge rule is peak-to-peak < 70 ms (code only).
- Min. Cells 10: linear track only ("at least 10 cells participating").
- Detection Notes (linear-track criteria) match the preprint text.
- Caveat: all of this rests on the preprint and the code, not the published supplement.

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
- Units: the preprint says all cells, the code uses excitatory cells.
- Kernel: the code's `gausswin` window is 5σ wide (±2.5σ), versus the package's Gaussian truncated at 8σ. Minor.
- Artifact rejection for ripples (per-session threshold with ±0.2 s padding, plus a manual bad_lfp list) is user-supplied as NaN or a normalization mask.
- The replay criteria (posterior-map fraction, weighted correlation, jump, coverage, cell count, longest smooth sub-segment) and all open-field detection need a decoder, which is out of scope.
- Published methods are unverified (Science supplement not retrieved).
Smallest package addition (if C): n/a. Optional convenience: a `peak_time` column on the threshold detectors plus a peak-separation merge would make the 70 ms merge direct.

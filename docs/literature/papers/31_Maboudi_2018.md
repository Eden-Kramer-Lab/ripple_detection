# Maboudi 2018 — Uncovering temporal structure in hippocampal output patterns
Source: the extracted text (pdftotext of the Dropbox PDF); title verified: yes
Trigger: MUA (population burst events from pooled single + multi-units; no LFP criterion)

## Method as implemented
Materials and methods, "Population burst events" (p. 16), linear-track data (Diba & Buzsáki 2007 recordings):
- Signal: "a SDF was calculated by counting the total number of spikes across all recorded single and multi-units in non-overlapping 1 ms time bins."
- Smoothing: "The SDF was then smoothed using a Gaussian kernel (20 ms standard deviation, 60 ms half-width)."
- Normalization/threshold: "Candidate events were identified as time windows with a peak SDF of at least three standard deviations above the mean calculated over all the session." (Whole session, including running.)
- Bounds: "The boundaries of each event were set to time points of crossing the mean, preceding and following the peak."
- Speed (detection criterion, mean over the event): "Events during which animals were moving (average movement speed of >5 cm/s) were excluded from all further analyses".
- Inclusion for analysis (applied to every analysis, so effectively part of the event set): "we then binned each PBE into 20 ms (non-sliding) time bins. Spikes from putative interneurons (mean firing rate when moving >10 Hz) were excluded, as were events with duration less than four time bins or with fewer than four active pyramidal cells." (≥80 ms; ≥4 active pyramidal cells, counted after dropping interneuron spikes.)
- Results (p. 3): "We did not add any other restrictions on behavior, LFP, or the participation of place cells." SWRs are described as accompanying most PBEs but are not a criterion. "While we identified active behavior using a speed criterion, we found similar results when we instead used a theta-state detection approach (not shown)."
- Open-field data: "we used the previously reported criteria (Pfeiffer and Foster, 2013) for identifying PBEs prior to binning (10 ms standard deviation kernel, minimum of 10% of units active, duration between 50 ms and 2000 ms)."
- W-maze (Karlsson) and post-task sleep datasets: PBE criteria and sleep scoring not stated separately.
- ANALYSIS, not detection: HMM on 20 ms bins; Bayesian replay with column-cycle shuffles.

## Inherited from
Linear track: nothing deferred (fully specified). Open field: Pfeiffer & Foster 2013 (manifest 45_Pfeiffer_2013; the paper restates its criteria; note Pfeiffer's text there says the stopped-period histogram uses speed <5 cm/s and a 10 ms SD kernel).

## Code
https://github.com/kemerelab/UncoveringTemporalStructureHippocampus (archived at elifesciences-publications); nelpy https://github.com/nelpy. The released events were checked below; the detector code is absent.

### Code search, September 2026

- **Own repository, no detection code.** [kemerelab/UncoveringTemporalStructureHippocampus @f86b7dc](https://github.com/kemerelab/UncoveringTemporalStructureHippocampus/tree/f86b7dc) (2019-02-07) ships one session's detected events (`data/fig1.nel`, session 16-40-19).
  - Its 457 candidates all peak at >= 3 SD of the stored trace (lowest 3.003); the shortest is 80 ms.
  - All 277 final PBEs have a mean speed of at most 4.44 cm/s.
  - Rerunning the detection on the stored trace (above the mean, peak >= 3 SD, 80-750 ms) reproduces 452 of the 457.
  - The open-field analysis (`Figure6.ipynb`) uses Pfeiffer's event times (`BradRippleStartEndTimes`) rather than detecting its own.
- **Possibilities, inferred from the released data, not proposed:**
  - Smoothing of 10 ms, not the paper's 20 ms. The spectrum of the stored trace falls off between 20 and 60 Hz as a 10 ms SD Gaussian does, not as a 20 ms kernel cut at 60 ms. 10 ms is nelpy's `get_mua` default (`sigma = 0.01 # 10 ms standard deviation`, [nelpy/nelpy @1255f57](https://github.com/nelpy/nelpy/tree/1255f57) utils.py line 260, May 2018). One session only.
  - A maximum near 750 ms (nelpy's default; the longest event is 0.696 s), not stated in the paper.
  - 3 of the 277 final PBEs are 60 ms (3 bins), under the stated minimum of 4 bins.

## Survey CSV discrepancies
- MUA Z-score Thresh. = #N/A; paper: 3 SD ("peak SDF of at least three standard deviations above the mean").
- MUA smooth (ms) = #N/A; paper: 20 ms SD Gaussian.
- Min. Cells (#) = #N/A; paper: ≥4 active pyramidal cells.
- Min. Duration (ms) = #N/A; paper: ≥4 × 20 ms bins = 80 ms (applied after binning).
- Speed 5: agrees (mean speed over the event).

## Package mapping
Tier: A (detection); the ≥4-active-pyramidal-cell inclusion needs a few lines of user code     Needs radiatum: n   Needs theta: n   Needs sleep scoring: n (linear track; the sleep dataset's scoring is not described)
Recipe (1 ms bins; `units` = all single + multi-units, (n_time, n_units)):
```python
pbe = multiunit_HSE_detector(
    time, units, speed, 1000,
    speed_threshold=np.inf,          # the paper's rule is on mean speed, applied next
    zscore_threshold=3.0, minimum_duration=0.0,
    smoothing_sigma=0.020,           # statistics over the whole session (default mask)
)
pbe = pbe[(pbe.mean_speed <= 5.0) & (pbe.duration >= 0.080)]
n_pyr = [(pyr[(time >= a) & (time <= b)].sum(axis=0) > 0).sum()
         for a, b in zip(pbe.start_time, pbe.end_time)]
pbe = pbe[np.array(n_pyr) >= 4]
```
Remaining deviations:
- Kernel truncated at 8 SD in the package vs "60 ms half-width" (3 SD) in the paper.
- "At least three SD" matches the package's at-or-above rule; one sample suffices with `minimum_duration=0.0`.
- Duration "less than four time bins" is a bin count after 20 ms binning; the recipe uses a continuous 80 ms.
- Active-pyramidal count is user code (the detector's `minimum_active_units` counts every column of `units`, including multi-units and interneurons).
Smallest package addition (if C): n/a. (Optional convenience: let `minimum_active_units` count a column subset, e.g. a boolean `unit_mask`.)

# Yamamoto 2017 — Direct Medial Entorhinal Cortex Input to Hippocampal CA1 Is Crucial for Extended Quiet Awake Replay
Source: the extracted text (pdftotext of the Zotero PDF, including STAR Methods and Supplemental Figures); title verified: yes (Yamamoto & Tonegawa, Neuron 96:217–227.e1–e4)
Trigger: SWR+MUA

## Method as implemented

Detection (candidate ripple / replay events):
- "In order to determine candidate events, we used both ripple band power (140-200 Hz) and multi-unit activities. We first identified periods in which ripple band LFP power of selected recording channel exceeded the 3 SD level of the baseline." (STAR Methods, "Candidate Ripple and Replay Event Detection", p. e3).
- MUA: "For the linear probes, we first identified positional range of recording sites along the probes and summed spike counts within the region of interest at 10 ms time bins as spiking activities. For the tetrodes, we first performed standard spike sorting based on spike amplitudes and binned them into 10 ms bins. Sum was then computed across tetrodes that are located in the putative pyramidal cell layer." (same section). With tetrodes, the MUA is the sum of sorted spikes (Figs 4–5). With silicone probes, it is detected spikes on the region-of-interest sites (Figs 1–3, 6).
- MUA threshold and bounds: "We then identified instantaneous multi-unit activity peak firing rates that exceeded the threshold (i.e., 3 SD of the baseline level) and then searched a time-window around the instantaneous multi-unit activity peak until the power reached the cut-off (1 SD) level at both ends. We defined the extracted time window as candidate events and assign identification numbers that was similar algorithm to the previous report (Davidson et al., 2009)." (same section).
- Ambiguities (not stated):
  - (i) How the ripple-power criterion and the MUA criterion combine. It is probably an AND (an MUA peak within a ripple-power period), but that is not said.
  - (ii) Which trace sets the 1 SD bounds. The text says "the power", which suggests ripple power, but the window is searched "around the instantaneous multi-unit activity peak".
  - (iii) What "baseline" is: which period or state the mean and SD come from.
  - (iv) How "power" is computed (squared filtered signal, Hilbert, or band-integrated), whether it is smoothed, and the filter type.
  - (v) Which channel is the "selected recording channel".
- Smoothing: none stated for either trace in detection. The figure legends mention Gaussian smoothing (σ = 20 ms) only for cross-correlation trend lines.
- Minimum and maximum duration: not stated. The typical "single ripple duration (~80 ms)" is reported.
- Merging for detection: not stated.
- Speed: not stated for detection. "Velocity filters were applied (2 cm/s) to extract valid run segments from the electrophysiological data." (STAR Methods, "Behavior Position Tracking", p. e2). That is for run and place-field data.
- Brain state (all results are split by state): "The 15 s epochs that have movement of less than 2 cm and delta/theta power ratio that is greater than 5 SD were classified as slow-wave sleep ... Similar algorithm is used to define quiet awake but with different thresholds (more than 2 cm movement over 15 s and delta/theta power ratio of 2 SD to 5 SD period." (STAR Methods, "Sleep Classification", p. e3). Delta is 1–4 Hz and theta is 6–12 Hz, from AR (modified covariance) PSD estimates ("LFP Power Density Estimation"). Supp. Fig. S1C shows the delta/theta ratio smoothed with σ = 10 s and σ = 60 s, with 2 SD and 5 SD thresholds.
- Ripple-burst classification (analysis on detected events): "Singlet ripples were defined as ripple events that are temporally separated by 200 ms or more to adjacent ripple events. For the doublets and triplets (ripple bursts), the temporal lags were set to less than 200 ms but greater than 70 ms ... In the event that adjacent ripple closer than 70 ms, these were categorized as single ripple." (STAR Methods, "Ripple Burst Analysis", p. e3). The lags are between ripple peaks.
- Cell participation: none stated for candidates. The replay analysis requires continuous posterior segments of ≥ 4 bins (fragmentation index), which is analysis.

## Inherited from
"similar algorithm to the previous report (Davidson et al., 2009)". Davidson (manifest 50_Davidson_2009; findings file exists) uses MUA from all > 100 µV spikes in 1 ms bins with σ = 15 ms, peak ≥ 3 SD, bounds at the mean, and statistics over STOP (< 5 cm/s). Yamamoto differs:
- 10 ms bins and no stated smoothing.
- Bounds at 1 SD instead of the mean.
- An added ripple-power > 3 SD requirement (140–200 Hz).
- "Baseline" instead of STOP.
Following Davidson does not resolve the ambiguities, because Yamamoto departs from it exactly where the ambiguities are. Sleep classification cites Haggerty & Ji 2014 for the delta/theta ratio. I did not follow it, because the thresholds are stated here.

## Code
No code link in the paper.

### Code search, September 2026

- Searched GitHub (code, repositories, and the lab's and authors' accounts), Zenodo, Figshare and CRCNS/DANDI: no released code, same-lab code or event files found. No relevant Tonegawa-lab or first-author repositories, or code for the ripple doublet/triplet rule, were found in those searches.

## Survey CSV discrepancies
Row 34:
- SWR Low Band 100: the detection band is 140–200 Hz ("ripple band power (140-200 Hz)"; the Supp. Fig. S1A legend says "Ripple band (140-200 Hz)"). 100–200 Hz is the band integrated from the PSD for ripple power in the state analysis ("The power of delta, theta and ripple were obtained by integrating the PSD estimates for 1-4Hz, 6-12 and 100-200Hz").
- Animal Speed 2: not a detection criterion. 2 cm/s is the velocity filter for run segments. The sleep classification uses movement < 2 cm per 15 s epoch, which is a displacement, not a speed.
- Combine Events Thresh. 70: not a detection merge. 70 ms is the ripple-burst classification rule (adjacent ripple peaks < 70 ms apart count as one ripple, 70–200 ms makes a burst). One could read it as a merge for burst counting only.
- Matches: MUA z 3, SWR z 3, "extend to 1 STD", SWR electrodes 1 ("selected recording channel"), smoothing #N/A (not stated), durations #N/A.
- Not recorded: the QAW vs SWS classification (delta/theta ratio in SD units plus head movement per 15 s epoch), and the fact that "baseline" is undefined.
Count: 3 discrepancies (band 100→140 Hz; speed not a detection criterion; the 70 ms is a burst-classification rule).

## Package mapping
Tier: B (approximate; D for exact reproduction because of the ambiguities listed above)     Needs radiatum: n   Needs theta: y (delta/theta ratio for state)   Needs sleep scoring: y (QAW vs SWS; head movement plus delta/theta)
Recipe (one reading: ripple power > 3 SD AND MUA peak > 3 SD, bounds where MUA falls to 1 SD):
```python
import numpy as np, pandas as pd
from ripple_detection import (filter_ripple_band, get_envelope, normalize_signal,
                              segment_boolean_series, require_overlap)
from ripple_detection.core import extend_threshold_to_mean

filtered = filter_ripple_band(lfp_selected_channel[:, None], fs, band=(140.0, 200.0))
power = get_envelope(filtered)[:, 0] ** 2                    # "power": measure not stated
z_r = normalize_signal(power, normalization_mask=baseline)   # "baseline": not defined
ripple_periods = np.asarray(segment_boolean_series(pd.Series(z_r > 3, index=time), minimum_duration=0.0))

mua = counts_10ms.sum(axis=1).astype(float)                  # pyramidal-layer tetrodes / probe ROI sites
z_m = normalize_signal(mua, normalization_mask=baseline_bins)
cand = np.asarray(extend_threshold_to_mean(z_m > 1, z_m > 3, bin_time, minimum_duration=0.0))
cand = require_overlap(cand, ripple_periods)
# alternative reading (bounds on ripple power): bring z_r to the 10 ms grid, then
# extend_threshold_to_mean(z_r_binned > 1, (z_m > 3) & (z_r_binned > 3), bin_time, 0.0)
qaw_events = require_overlap(cand, qaw_intervals)            # user-computed state (15 s epochs)
```
Smoke-tested on simulated LFP + multiunit (it runs; 20 candidates on 23 simulated ripples).
Remaining deviations:
- The whole recipe rests on interpretive choices: the AND rule, which trace sets the bounds, the baseline period, the power measure (Hilbert² vs squared signal vs other), no smoothing, and FIR remez vs the unstated filter. None can be checked against the paper.
- The two signals are on different time bases (LFP at fs, MUA in 10 ms bins). The user must align them. The package detectors assume one sampling grid.
- No existing detector combines a ripple-power gate with an MUA trace and 1 SD bounds. Carey's detector combines envelope and MUA, but as a geometric-mean score, which is a different rule.
- The package has no state classification. The delta/theta ratio (AR PSD, SD-unit thresholds, 10 s or 60 s smoothing) and the 15 s head-movement epochs are user code.
- Ripple-burst labelling (lags < 70 ms, 70–200 ms, ≥ 200 ms between peaks) is user code on the event peaks. The package reports peak_time only for Zugaro and Long.
Smallest package addition (if C): not needed for an approximate B. For A, add a `bounds_zscore` option to `multiunit_HSE_detector` plus a documented "require ripple-power overlap" recipe (require_overlap already exists). The paper's ambiguities would remain.

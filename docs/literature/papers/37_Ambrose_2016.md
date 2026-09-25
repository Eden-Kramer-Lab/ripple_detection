# Ambrose 2016 — Reverse Replay of Hippocampal Place Cells Is Uniquely Modulated by Changing Reward
Source: the extracted text (pdftotext of Dropbox/Papers/Neuron-/2016/Neuron-2016-Ambrose et al-...pdf, 14 pp, main text only); the PMC author manuscript (PMC6013068, efetch XML) has the same SWR paragraph word for word. title verified: yes (Ambrose, Pfeiffer, Foster; Neuron 91, 1124–1136)
Trigger: SWR

## Method as implemented
Detection (all from Experimental Procedures, "Sharp-Wave Ripple Detection", p. 1134, unless noted):
- Channels: "One channel was selected from each of four to seven tetrodes for LFP analysis." Tetrodes were in CA1 ("40-tetrode microdrive targeting area CA1", p. 1134). How the channel was chosen: not stated. Layer: not stated beyond CA1.
- Recording: "LFP was digitally filtered between 0.1 and 500 Hz and recorded at 3,255 Hz." (p. 1134)
- Filter / envelope / smoothing / combining: "LFP was band-pass filtered between 150 and 250 Hz and the smoothed (Gaussian kernel, SD = 12.5 ms) Hilbert envelope of this signal was averaged across all channels." That is, each channel's envelope is smoothed, then the channels are averaged. Filter type and order: not stated.
- Threshold: "Peaks in the envelope exceeding 3 SD above the mean were identified as SWRs." The mean and SD are computed over: not stated.
- Bounds: "The start and end time of each SWR event is defined as the time at which the signal crossed the mean."
- Minimum or maximum duration: not stated. Merging or dropping close events: not stated.
- Behavioural restriction: "Stopping periods were defined as times in which the rat's velocity was less than 5 cm/s and his position was within 10 cm of the well location. All analysis was restricted to stopping periods." (p. 1134). Results: "during stopping periods at the track ends, we identified SWRs as peaks in ripple power (150–250 Hz) in the LFP" (p. 1125). So detection is restricted to stopping periods. Whether stopping periods also define the z-score baseline is not stated.
- Brain state, cell participation, artifact rule, sharp wave: none stated.

Analysis (not detection):
- "SWRs were used as candidate replay events." Replay = "Candidate events whose posterior probability's weighted correlation ... exceeded 0.6", using 20 ms windows overlapping by 10 ms (p. 1134). In a robustness check, candidates also had to pass a Monte Carlo shuffle (1,500 shuffles, p < 0.05). Direction was assigned by > 66.5% of the posterior in one map.

## Inherited from
Nothing is deferred for SWR detection; the paragraph stands alone. Pfeiffer & Foster 2013 (manifest 45) is cited only for the drive and for decoding. Its LFP paragraph is nearly identical and fills in the unstated baseline: "smoothed (Gaussian kernel, s.d. = 12.5 ms). This processed signal was averaged across all tetrodes and ripple events were identified as local peaks with an amplitude greater than 3 s.d. above the mean, using only periods when the rat's velocity was less than 5 cm s-1. The start and end boundaries for each event were defined as the point when the signal crossed the mean." (the extracted text, Methods, "Local field potential analysis"). Treat this as the lab convention (an inference), not as Ambrose's stated method.

## Code
None linked.

### Code search, September 2026

- **Later, a co-author's.** [Brad-E-Pfeiffer/DeepSuperficialSWRs @9f6ab57](https://github.com/Brad-E-Pfeiffer/DeepSuperficialSWRs/tree/9f6ab57) (2023).
  - It re-analyses linear-track sessions of rats Janni, Harpy and Ettin recorded in 2009-10 under "Reward/BigReward/Reward" and "Reward/NoReward/Reward" (`DEEP_VS_SUPERFICIAL_RIPPLE_PARTICIPATION_ANALYSIS.m` lines 17-26), Ambrose's reward-change design.
  - It sets `Ripple_Minimum_Duration=0.05` and `Ripple_Maximum_Duration=0.5` (lines 85-86), applied in `DSRP_FIND_RIPPLE_EVENTS.m` (line 289).
  - One electrode per tetrode carries LFP (line 93).
  - That these are the paper's recordings is inferred; the code does not say so.
- **Possibility, not proposed:** a 50-500 ms limit was applied, which would make the durations `Not reported` rather than `#N/A`.
- **Other code from the lineage disagrees:**
  - [Brad-Pfeiffer/MouseDevelopmentalAnalysisCode @950fc28](https://github.com/Brad-Pfeiffer/MouseDevelopmentalAnalysisCode/tree/950fc28) (`KJ_BEHAVIOR_FIND_RIPPLE_EVENTS.m` line 332) also uses 50-500 ms.
  - [Brad-E-Pfeiffer/ThetaForwardReverseCode @bc714a2](https://github.com/Brad-E-Pfeiffer/ThetaForwardReverseCode/tree/bc714a2) (`IRFS_FIND_RIPPLE_EVENTS.m` line 206, Wang, Foster & Pfeiffer 2020) uses 50 ms to 1 s.
  - The Foster lab's [caitlinmallory/TimeCourseOrganizationOfHippocampalReplay @128513a](https://github.com/caitlinmallory/TimeCourseOrganizationOfHippocampalReplay/tree/128513a) (`find_candidate_events_2.m` lines 13-14) uses no limits.

## Survey CSV discrepancies
- SWR electrodes: CSV ">1". Paper: "One channel was selected from each of four to seven tetrodes", so 4–7.
- Min. Duration: CSV 50 ms. Paper: not stated in the main text or the author manuscript.
- Max Duration: CSV 500 ms. Paper: not stated. The supplement (7 figures, 4 tables, no supplemental methods listed) was not retrieved (Cell/Europe PMC blocked), so a table there cannot be ruled out.
- Animal speed 5: consistent, but the paper's restriction is the stopping period, which also requires being within 10 cm of the well.
- All other detection fields (SWR 3 SD, smoothing 12.5 ms, 150–250 Hz, Combine N/A) match.

## Package mapping
Tier: B     Needs radiatum: n   Needs theta: n   Needs sleep scoring: n
Recipe:
```python
import numpy as np, pandas as pd
from ripple_detection import (filter_ripple_band, get_envelope, gaussian_smooth,
    normalize_signal, threshold_by_zscore, segment_boolean_series, require_overlap)
filt = filter_ripple_band(lfp, fs, band=(150, 250))          # (n_time, 4..7), one channel per tetrode
trace = gaussian_smooth(get_envelope(filt), 0.0125, fs).mean(axis=1)   # mean of smoothed envelopes
stopping = (speed < 5) & (distance_to_well < 10)
z = normalize_signal(trace, normalization_mask=stopping)     # baseline = stopping (inferred, see above)
events = np.array(threshold_by_zscore(z, time, minimum_duration=0.0, zscore_threshold=3))
stop_iv = np.array(segment_boolean_series(pd.Series(stopping, index=time), minimum_duration=0.0))
events = require_overlap(events, stop_iv)
```
Tier-A approximation: `Roumis_ripple_detector(time, filt, speed, fs, speed_threshold=5, minimum_duration=0.0, zscore_threshold=3, smoothing_sigma=0.0125, normalization_mask=stopping)`.
Remaining deviations:
- Roumis averages sqrt(Gaussian-smoothed squared envelope). The paper averages the Gaussian-smoothed envelope. These differ, and the first is always at least as large as the second. The B recipe above is exact.
- Package threshold is ">= 3" and the extension runs to ">= 0". The paper says "exceeding" and "crossed the mean". This matters only for samples exactly at the threshold.
- Stopping restriction: require_overlap keeps events that overlap a stopping interval, and the endpoint speed rule tests only the first and last samples. The paper restricts to stopping periods but does not say how events at a stopping-period edge were handled.
- The within-10-cm-of-well criterion is user code (position data).
- Filter design (FIR remez, zero-phase) versus the paper's unstated filter.
- The normalization period is unstated in the paper. Using stopping periods is an inference from Pfeiffer & Foster 2013.
- The paper's LFP is sampled at 3,255 Hz. filter_ripple_band designs a remez FIR for any rate.
Smallest package addition (if C): n/a

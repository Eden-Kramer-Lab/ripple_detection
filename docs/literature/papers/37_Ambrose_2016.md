# Ambrose 2016 — Reverse Replay of Hippocampal Place Cells Is Uniquely Modulated by Changing Reward
Source: the extracted text (pdftotext of Dropbox/Papers/Neuron-/2016/Neuron-2016-Ambrose et al-...pdf, 14 pp, main text only); the PMC author manuscript (PMC6013068, efetch XML) has the same SWR paragraph word for word. title matched (Ambrose, Pfeiffer, Foster; Neuron 91, 1124–1136)
Trigger: SWR

[Paper](https://doi.org/10.1016/j.neuron.2016.07.047) · [Field evidence](../evidence.csv) · [Source catalog](../sources.md)

Additional sources: [ambrose-supplement](../sources.md#ambrose-supplement), [deep-superficial-code](../sources.md#deep-superficial-code), [mouse-development-code](../sources.md#mouse-development-code), [theta-code](../sources.md#theta-code).

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

Nothing is deferred for SWR detection; the paragraph stands alone. Pfeiffer & Foster 2013 ([paper note](45_Pfeiffer_2013.md)) is cited only for the drive and for decoding. Its LFP paragraph is nearly identical and fills in the unstated baseline: "smoothed (Gaussian kernel, s.d. = 12.5 ms). This processed signal was averaged across all tetrodes and ripple events were identified as local peaks with an amplitude greater than 3 s.d. above the mean, using only periods when the rat's velocity was less than 5 cm s-1. The start and end boundaries for each event were defined as the point when the signal crossed the mean." (the extracted text, Methods, "Local field potential analysis"). Treat this as the lab convention (an inference), not as Ambrose's stated method.

## Code

None linked.

### Related code

- **Later, a co-author's.** [Brad-E-Pfeiffer/DeepSuperficialSWRs @9f6ab57](https://github.com/Brad-E-Pfeiffer/DeepSuperficialSWRs/tree/9f6ab57) (2023).
  - It re-analyses linear-track sessions of rats Janni, Harpy and Ettin recorded in 2009-10 under "Reward/BigReward/Reward" and "Reward/NoReward/Reward" (`DEEP_VS_SUPERFICIAL_RIPPLE_PARTICIPATION_ANALYSIS.m` lines 17-26), Ambrose's reward-change design.
  - It sets `Ripple_Minimum_Duration=0.05` and `Ripple_Maximum_Duration=0.5` (lines 85-86), applied in `DSRP_FIND_RIPPLE_EVENTS.m` (line 289).
  - One electrode per tetrode carries LFP (line 93).
  - That these are the paper's recordings is inferred; the code does not say so.

- The later 50–500 ms configuration does not establish Ambrose’s duration limits; the published limits remain `Not reported`.

- **Other code from the lineage disagrees:**
  - [Brad-Pfeiffer/MouseDevelopmentalAnalysisCode @950fc28](https://github.com/Brad-Pfeiffer/MouseDevelopmentalAnalysisCode/tree/950fc28) (`KJ_BEHAVIOR_FIND_RIPPLE_EVENTS.m` line 332) also uses 50-500 ms.
  - [Brad-E-Pfeiffer/ThetaForwardReverseCode @bc714a2](https://github.com/Brad-E-Pfeiffer/ThetaForwardReverseCode/tree/bc714a2) (`IRFS_FIND_RIPPLE_EVENTS.m` line 206, Wang, Foster & Pfeiffer 2020) uses 50 ms to 1 s.
  - The Foster lab's [caitlinmallory/TimeCourseOrganizationOfHippocampalReplay @128513a](https://github.com/caitlinmallory/TimeCourseOrganizationOfHippocampalReplay/tree/128513a) (`find_candidate_events_2.m` lines 13-14) uses no limits.

## Analysis and interpretation

Primary replay selection requires weighted correlation >0.6. Verification uses a
stricter 0.7 threshold or 1500 posterior shuffles with p<0.05 (main Methods and
supplementary Tables S2–S3, pp. 10–11).

Supplementary Table S1 (p. 9) gives:

| Scope | Replays | SWRs | Replay fraction |
|---|---:|---:|---:|
| Experiment 1 | 738 | 3312 | 22.3% |
| Experiment 2 | 409 | 2636 | 15.5% |
| Pooled | 1147 | 5948 | 19.3% |

Behavioral reconstruction uses nonoverlapping 200 ms windows (main Methods,
PDF p. 12). Table S1 reports 74–95% of bidirectional bins and 74–96% of directional
bins with error <10 cm. These are accuracy fractions, not a pooled mean error
in centimeters.

## Uncertainties

Detector-duration limits and a pooled reconstruction error in centimeters are not reported in the main text or complete supplement. Later lab code cannot establish the historical duration settings. Ambrose’s normalization period is unstated; the stopped-period lab convention is an inference.

## Package mapping

Packaged primary method: `ambrose_2016` in [literature_methods.py](../../../src/ripple_detection/literature_methods.py). Its docstring records implementation choices and assumptions. Simulation checks establish that it runs; they do not establish equivalence to the authors’ original event set.

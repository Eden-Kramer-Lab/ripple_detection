# Grosmark 2016 — Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences
Source: the extracted text (main text + Supplementary Materials, from the Dropbox PDF); title verified: yes
Also consulted: CRCNS hc-11 data description (https://crcns.org/files/data/hc-11/crcns_hc-11_data_description.pdf, a local copy); Grosmark et al. 2012 Neuron (ref. 31) via Wayback copy of PMC3608095 (a local copy).
Trigger: SWR+MUA (population synchrony event that contains an LFP ripple peak)

## Method as implemented
Supplementary Materials, "Ripple Event Detection" (p. 4):
- MUA trace: "the combined spiking of all recorded CA1 pyramidal cells were binned in 1ms bins and convolved with a 15 ms Gaussian kernel (5)." Whether 15 ms is the SD or the width: not stated (ref. 5, Pfeiffer & Foster 2013, uses a 10 ms SD, so the citation does not settle it).
- Normalization/threshold: "a trigger rate was defined as being 3 standard deviations above the mean of all 1 ms bins within NREM epochs of both PRE and POST epochs combined. Putative population synchrony events were detected when the smoothed firing rate vector crossed the trigger rate."
- Bounds: "the time points at which the convolved firing rate vector returned to the mean of all within-NREM firing rate bins."
- LFP: "Independently, sharp wave-ripple events were detected from the pyramidal layer LFP." Band, envelope, threshold, channel count: NOT STATED. Fig. S4 legend shows "the ripple-frequency (150-250Hz) filtered LFP" (display only).
- Conjunction: "Population synchrony events that did not contain at least one LFP-detected ripple (as assessed by the time-stamp of its peak ripple power) were discarded (50.6% of population events met this criteria)."
- Inclusion: events that "2) lasted between 50 to 500 ms, and 3) occurred during non-theta or 'off-line' states (quite waking [immobility] or NREM) and 4) in which at least five distinct pyramidal cells each fired at least one spike were termed 'Ripple events'". Duration applies to the PBE (mean-to-mean) bounds. No speed number.
- Sleep scoring (Supplementary, p. 3): "Sleep scoring was performed using hippocampal LFP (theta/delta ratio), accelerometer (movement), and E.M.G. data as previously described (31)." hc-11 description: states "scored based on pyramidal-layer LFP wavelet characteristics (particularly the theta (5-10 Hz) to delta (1 to 4 Hz) ratio) as well movement information (as derived from EMG, accelerometer or position tracking). All state scoring was performed using TheStateEditor" (a GUI; manual/semi-manual).
- ANALYSIS, not detection (Bayesian replay): "Only ripple events with durations of at least 100 ms and in which at least 10% or 5 (whichever was greater) of place cells fired were considered for this analysis"; 20 ms bins.

## Inherited from
- Sleep scoring → Grosmark et al. 2012 (ref. 31): "REM and non-REM episodes were identified offline using the ratio of the power in theta band (5–11 Hz) to delta band (1–4 Hz) of LFP"; recordings "while the behavior of the rat and LFPs from several channels were monitored by the experimenter".
- LFP ripple detection: not deferred to anything. Context only (not cited for it): Grosmark 2012 detected ripples "during nontheta periods from the band-pass filtered (120–250 Hz) trace by defining periods during which ripple power is continuously greater than mean 2 SD, and peak of power in the periods was greater than mean 3 SD". The hc-11 dataset lists "Grosmark, A.D., Long J. and Buzsáki, G." as authors; "Long J." is presumably John D. Long II, author of bz_DetectSWR (radiatum-dependent) — identity not confirmed. Whether his detector produced the 2016 ripples is unknown — do not treat as evidence.

## Code
None linked in the paper. Data: CRCNS hc-11.

### Code search, September 2026

- The data hold no event files: CRCNS hc-11 (the description, v0.8 p. 4, lists spikes, position and epochs) and DANDI 000044 v0.250624.0426 (two NWB files read: epochs, states, LFP and units only).
- **Possibility, from lab code of the time, not the paper's.** [buzsakilab/buzcoderough @f3486d9](https://github.com/buzsakilab/buzcoderough/tree/f3486d9), `LFP/EventDetection/detect_swr/detect_swr.m` (J. Long, 2015):
  - its example session is a Grosmark rat's (`buddy140_060813_reo`, line 54);
  - it defaults to 80-250 Hz and 0.5 / 2.5 SD on a sharp-wave channel (lines 139-151).
  - If it made the 2016 ripples, the CSV's 150-250 Hz, 3 SD, 15 ms and one electrode would all be wrong; nothing shows that it did.

## Survey CSV discrepancies
- SWR Z-score Thresh. = 3: not stated for the LFP ripple; 3 SD is the MUA trigger.
- SWR smooth (ms) = 15: not stated; 15 ms is the MUA Gaussian.
- SWR Low/High Band = 150/250: only the Fig. S4 display filter; the detection band is not stated.
- SWR electrodes = 1: not stated ("the pyramidal layer LFP").
- Detection Notes "during NREM (manually scored)": events come from NREM and quiet waking (including MAZE immobility); NREM defines the normalization statistics.
- MUA z 3, MUA smooth 15, Min cells 5, 50–500 ms: agree.

## Package mapping
Tier: A (population part); the LFP ripple part is D as written (method not reported)     Needs radiatum: unknown (LFP method unstated; the PBE part does not)   Needs theta: y (non-theta states, via state scoring)   Needs sleep scoring: y (NREM mask for statistics; off-line state restriction)
Recipe (1 ms bins; `pyr` = CA1 pyramidal spike counts, (n_time, n_cells)):
```python
pbe = multiunit_HSE_detector(
    time, pyr, speed, 1000,
    speed_threshold=np.inf,            # state-based, no speed number
    zscore_threshold=3.0, minimum_duration=0.0,
    smoothing_sigma=0.015,             # if "15 ms kernel" is the SD
    normalization_mask=is_nrem,        # NREM of PRE+POST
    maximum_duration=0.5, minimum_active_units=5,
)
pbe = pbe[pbe.duration >= 0.05]
half = 0.5 / fs_lfp
pbe = require_overlap(pbe, ripple_peak_times[:, None] + [-half, half])   # contains a ripple peak
pbe = exclude_overlap(pbe, theta_or_active_wake_intervals)             # user state scoring
```
Remaining deviations:
- LFP ripple inventory: detector unknown; any choice (e.g. `Kay_ripple_detector`, `Zugaro_ripple_detector`) is a substitution.
- "Crossed the trigger rate" = any sample above 3 SD; package uses at-or-above with `minimum_duration=0.0` (one sample). Equivalent up to the tie.
- Gaussian SD vs width ambiguity (15 ms).
- "Contains the ripple's peak time" needs the point widened by half a sample for `require_overlap` (positive-overlap rule).
- Duration bounds inclusive in the package vs "between 50 to 500 ms" (strictness unstated).
- `exclude_overlap` drops a PBE touching any theta interval; the paper's rule ("occurred during") is unstated at the boundary.
Smallest package addition (if C): n/a for the population part. For the LFP part, the paper would need to state its method.

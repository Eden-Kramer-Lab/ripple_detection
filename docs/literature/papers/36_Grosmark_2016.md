# Grosmark 2016 — Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences
Source: the extracted text (main text + Supplementary Materials, from the Dropbox PDF); title matched
Also consulted: CRCNS hc-11 data description (https://crcns.org/files/data/hc-11/crcns_hc-11_data_description.pdf, a local copy); Grosmark et al. 2012 Neuron (ref. 31) via Wayback copy of PMC3608095 (a local copy).
Trigger: SWR+MUA (population synchrony event that contains an LFP ripple peak)

[Paper](https://doi.org/10.1126/science.aad1935) · [Field evidence](../evidence.csv) · [Source catalog](../sources.md)

Additional sources: [buzcoderough](../sources.md#buzcoderough), [grosmark-data](../sources.md#grosmark-data).

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

### Related code

- CRCNS hc-11 describes spikes, position and epochs. In DANDI 000044 v0.250624.0426, HDF5 metadata from all eight NWB files has only epochs under intervals, an empty analysis group, and behavior/ecephys processing. No SWR/replay interval table was found; raw arrays were not exhaustively downloaded.

- **Possibility, from lab code of the time, not the paper's.** [buzsakilab/buzcoderough @f3486d9](https://github.com/buzsakilab/buzcoderough/tree/f3486d9), `LFP/EventDetection/detect_swr/detect_swr.m` (J. Long, 2015):
  - its example session is a Grosmark rat's (`buddy140_060813_reo`, line 54);
  - it defaults to 80-250 Hz and 0.5 / 2.5 SD on a sharp-wave channel (lines 139-151).
  - Nothing connects this implementation to the original 2016 events; the independent LFP settings remain unresolved.

## Analysis and interpretation

The duration columns record the ripple-event rule, 50-500 ms. The at-least-100 ms criterion selects events for Bayesian replay analysis and is kept in the detection notes.

## Uncertainties

The original independent LFP detector and its threshold, band, channel count and caller remain unresolved. CRCNS/DANDI metadata and all eight inspected NWB interval catalogs do not supply a documented original SWR pipeline. Later buzcoderough defaults cannot fill these fields.

## Package mapping

Executable example: `grosmark_2016` in [literature_recipes.py](../../../examples/literature_recipes.py). Its docstring records implementation choices and assumptions. Simulation checks establish that it runs; they do not establish equivalence to the authors’ original event set.

# Muessig 2019 — Coordinated Emergence of Hippocampal Replay and Theta Sequences during Post-natal Development
Source: the extracted text (pdftotext of the Dropbox PDF; Current Biology 29, 834–840.e1–e4, 2019; Muessig, Lasek, Varsavsky, Cacucci, Wills). Supplemental figures (Fig. S1, which characterizes rest/SWR/MUA events) are not in the text and were not consulted. Title verified: yes
Trigger: SWR+MUA (MUA bursts kept only if they overlap an SWR)

## Method as implemented
All from STAR Methods, "Detection of slow-wave sleep, sharp-wave ripples and multi-unit activity bursts", p. e2–e3, txt lines 833–851, unless noted.

- **State gate ("rest"; restricts all analyses):** "The brain states slow-wave sleep (SWS), rapid-eye movement sleep (REM) and awake movement were defined following [22]. A multitaper power spectral density estimate of the hippocampal local field potential (LFP) was derived for 1.6 s windows, overlapping by 0.8 s (MATLAB function ‘pmtm’). From this, power in the delta and theta bands were calculated in each window. As theta frequency changes during development [18], theta and delta peak frequencies were calculated for each session, defined as the peak frequency of the fast Fourier transform of the LFP, in the bands 5-11Hz (theta) and 1.5-4Hz (delta). Mean running speed for each 1.6 s bin was also estimated. In the absence of EMG recordings, we could not unequivocally discriminate between slow wave sleep and quiet immmobility, we therefore restricted all analyses to epochs termed ‘rest’. Rest was defined as epochs with running speed < 2.5cm/s, and theta/delta power ratio < 2 and waking movement as theta/delta power ratio > 2 and speed > 2.5cm/s."
  - Ratio: theta power / delta power per 1.6 s window (0.8 s step, multitaper), each band centered on a per-session peak frequency found within 5–11 Hz and 1.5–4 Hz. The width of the band around each peak is not stated. Threshold 2 (absolute), AND speed < 2.5 cm/s.
  - [22] = O'Neill et al. 2008 Nat Neurosci. Paywalled and not in Dropbox; not read.
  - Role: "restricted all analyses to epochs termed ‘rest’". The events analyzed lie in rest (a detection-time or analysis-time restriction; the text does not separate them). The detection statistics are not said to be computed over rest only (see the SWR threshold below).
- **SWR detection:** "Sharp-wave ripples were detected by first filtering the LFP in the band 100-250Hz. The instantaneous power of the filtered LFP was then estimated by calculating the root mean square over 7ms intervals (MATLAB function ‘envelope’ with option ‘rms’). From all LFPs across tetrodes in the CA1 layer, the LFP whose power estimate had the highest standard deviation was then used to define ripple events, as 100ms windows around the peak power, whenever the power was greater than the 99th percentile of all powers in the trial (approximately equal to 4 standard deviations above the mean)."
  - Filter type: not stated. Envelope: moving RMS over 7 ms. One channel: the CA1 tetrode whose RMS power has the highest SD. Threshold: the 99th percentile over the whole trial (not z-scored; "≈ 4 SD" is the authors' approximation). Bounds: a **fixed 100 ms window** around the power peak (centered, inferred from "around"). No min/max duration beyond the fixed window.
- **MUA bursts:** "Multi-unit activity (MUA) bursts were defined by binning all spikes from CS cells into 1ms bins and smoothing the resulting binned spike train with a Gaussian kernel (s.d. 10ms). MUA events were then defined as crossing of a threshold defined as 3 standard deviations above the mean of the smoothed spike train, with a duration from 100-750ms."
  - Signal: pooled sorted complex-spike (putative pyramidal) cells (CS criteria and TINT manual isolation, txt 785–791). Bounds: "crossing of a threshold", read literally as the above-3 SD period. Extension to the mean is not mentioned. Duration 100–750 ms applies to the MUA event. The mean/SD period is not stated ("the smoothed spike train", presumably the trial).
- **Conjunction:** "Only MUA bursts which temporally overlapped (even in part) with SWR events were included in the replay analysis." Stated as a replay-analysis inclusion rule. In effect it is the event definition: Results say "we first defined ensemble spiking events as bursts of multi-unit activity (MUA) that coincided with SWRs" (txt ~206–208). Decoding spans the MUA burst: "spanning the duration of the MUA burst" (txt 867). So the event bounds are the MUA bounds.
- **Awake (RUN) events:** "SWR/MUA joint events during the RUN trial were defined using the same criteria as those in sleep trials. Only data from non-locomotory epochs during RUN were included in further analyses, these were defined using the same criteria as rest during sleep trials with the exception that the limit for running speed was set to < 1cm/s." (txt 884–886).
- Cell participation: none per event. Session inclusion: "Replay analysis was only applied to CS cell ensembles in which > 25 CS cells fired > 75 spikes during RUN" (txt 881; a session-level criterion).
- Merging / close events / artifact rejection: not stated.

Analysis-only:
- The reactivation (pairwise co-firing) analysis uses "all spikes occurring in rest windows, during SWS epochs" (txt 854). Separate from replay events.

## Inherited from
Sleep/rest-state definitions from O'Neill et al. 2008 (ref 22), not accessible (Nature Neuroscience paywall; not in Dropbox). Muessig restates the operative rule (speed < 2.5 cm/s, theta/delta < 2), so the unread citation mainly leaves the band widths around the per-session peak frequencies open. SWR and MUA rules are stated in full and not attributed.

## Code
None linked in the text.

## Survey CSV discrepancies
Row 29.
1. **SWR Z-score Thresh. 4:** the paper's threshold is the **99th percentile** of RMS power over the trial; "approximately equal to 4 standard deviations" is the authors' gloss. The CSV value is a paraphrase, not the rule.
2. **SWR smooth "#N/A":** the power is a moving RMS over **7 ms** windows, effectively a 7 ms smoothing window. It could be recorded as 7 (RMS window).
3. The SWR bounds are a fixed 100 ms window around the peak; no CSV field captures this. The notes could say "SWR = fixed 100 ms window around RMS peak > 99th percentile; single highest-SD tetrode".
Consistent: Detection "SWR, MUA"; MUA 3 SD; speed 2.5 (rest; RUN uses < 1 cm/s); MUA smooth 10 ms; SWR electrodes 1; band 100–250 Hz; min 100 / max 750 ms (the MUA burst); min cells "#N/A"; notes "theta/delta ratio".

## Package mapping
Tier: B     Needs radiatum: n   Needs theta: y (theta/delta power ratio, per-session peak frequencies)   Needs sleep scoring: y (rest = speed < 2.5 cm/s & theta/delta < 2 per 1.6 s window; user code)

Recipe (`cs_counts`: (n_time, n_CS_cells) at 1 kHz; `lfps`: (n_time, n_tetrodes) CA1 LFP):
```python
import numpy as np, pandas as pd
from ripple_detection import (filter_ripple_band, get_multiunit_population_firing_rate, normalize_signal,
                              segment_boolean_series, require_overlap, exclude_overlap, multiunit_HSE_detector)
from ripple_detection.core import extend_threshold_to_mean

# rest mask (user code): pmtm-like PSD in 1.6 s windows / 0.8 s step; theta & delta power around
# per-session peaks (5-11 Hz, 1.5-4 Hz); rest = (theta/delta < 2) & (window speed < 2.5)

# SWR (user code around package filters): moving RMS over 7 ms, best channel, 99th percentile, 100 ms window
filt = filter_ripple_band(lfps, fs, band=(100, 250))
w = round(0.007 * fs)
rms = np.sqrt(np.apply_along_axis(lambda x: np.convolve(x**2, np.ones(w) / w, "same"), 0, filt))
p = rms[:, np.argmax(rms.std(axis=0))]
runs = segment_boolean_series(pd.Series(p > np.percentile(p, 99), index=time), minimum_duration=0.0)
peaks = [time[(time >= s) & (time <= e)][np.argmax(p[(time >= s) & (time <= e)])] for s, e in runs]
swr = np.column_stack([np.subtract(peaks, 0.05), np.add(peaks, 0.05)])

# MUA bursts bounded at the 3 SD crossings, 100-750 ms
z = normalize_signal(get_multiunit_population_firing_rate(cs_counts, 1000, smoothing_sigma=0.010))
mua = np.array(extend_threshold_to_mean(z >= 3, z >= 3, time, minimum_duration=0.100)).reshape(-1, 2)
mua = mua[mua[:, 1] - mua[:, 0] <= 0.750]
events = require_overlap(mua, swr)              # "overlapped (even in part)"
events = exclude_overlap(events, non_rest_intervals)   # restrict to rest (or require containment)
# one-call MUA alternative (bounds extended to the mean):
# multiunit_HSE_detector(time, cs_counts, speed, 1000, speed_threshold=np.inf, zscore_threshold=3.0,
#                        smoothing_sigma=0.010, minimum_duration=0.100, maximum_duration=0.750)
```
(Smoke-tested on simulated data; runs.)

Remaining deviations:
- **SWR envelope:** moving RMS (7 ms) is not a package option. The package's detectors use a Gaussian-smoothed Hilbert envelope, so the RMS must be user-computed (known gap).
- **SWR threshold:** a percentile of raw power over the trial, not a z-score. No detector offers a percentile threshold. `estimate_noise_threshold` is a different, Yu-style rule.
- **SWR bounds:** a fixed ±50 ms window around the power peak, with no extension or threshold bounds. No detector option exists (known gap). Merging of overlapping windows from nearby supra-threshold runs is not stated.
- **Channel selection:** the single tetrode with the highest SD of power. User code; the package's multi-channel detectors combine channels instead.
- **MUA bounds:** "crossing of a threshold" read as the above-3 SD period (the recipe does this with `extend_threshold_to_mean`). `multiunit_HSE_detector` would extend to the mean and then apply the 750 ms ceiling to the wider event. `>=` vs "crossing" (strict) differ at the threshold.
- The MUA mean/SD period is not stated. The rest restriction is stated only as "all analyses", not as the normalization period.
- `require_overlap` with `minimum_overlap=0` requires positive-duration overlap. Events that only touch at an endpoint are dropped; "even in part" suggests any overlap, so this is equivalent except for exact touches.
- Theta/delta rest scoring with per-session peak frequencies and multitaper windows is user code. The band width around each peak is not stated.
Smallest package addition (if C): n/a (B). Two options would move the SWR half to A: an RMS-window envelope, and a "fixed window around peak" event-bound mode with a percentile threshold. The package brief lists both as gaps.

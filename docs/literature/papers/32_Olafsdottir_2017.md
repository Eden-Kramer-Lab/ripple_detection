# Ólafsdóttir 2017 — Task Demands Predict a Dynamic Switch in the Content of Awake Hippocampal Replay
Source: the extracted text (pdftotext of the Zotero PDF, including STAR Methods); title verified: yes (Ólafsdóttir, Carpenter & Barry, Neuron 96:925)
Trigger: MUA (the pooled rate of sorted CA1 place cells). A ripple is used only in a control analysis.

## Method as implemented

Detection. Quotes are from STAR Methods, p. e2, unless noted:
- Lineage: "Putative reactivation events were identified based on the activity of hippocampal place cells using a similar method to Pfeiffer and Foster (2013) and Ólafsdóttir et al. (2016)."
- Units (p. e2): "Hippocampal cells were classified as place cells if they exhibited firing greater than its mean rate for 20 contiguous bins and if the peak firing rate was > 1 Hz. Interneurons, identified by narrow waveforms and high firing rates, were excluded". The bins are 2 cm, so 20 bins is 40 cm. Spikes were manually sorted in Tint.
- Signal and smoothing: "multi-unit (MU) activity from CA1 place cells were binned into 1ms temporal bins and smoothed with a Guassian kernel (s = 5ms)." (sic) <!-- codespell:ignore -->
- Threshold: "Periods when the MU activity exceeded the mean rate by 3 standard deviations were identified as candidate reactivation events."
- Bounds: "The start and end points of each candidate event were determined as the time when the MU activity fell back to the mean."
- Minimum duration: "Events less than 40ms long were rejected." This applies to the whole event.
- Speed and location: "Further, events were excluded if the animals' movement speed during the event exceeded 3cm/s or if the animals were located away from the two corners (total number of events = 4425)."
  - "Speed during the event exceeded 3 cm/s" is most naturally read as any sample above 3. It could also mean the mean; the paper does not say.
  - Main text (p. 926): "they were limited to periods when the animals' speed remained below 3 cm/s."
  - Position is sampled at 50 Hz (LED tracking).
- Participation. The arm-reactivation analysis has none. The replay-trajectory analysis adds one: "Event detection for the replay trajectory analysis was identical to that for the arm reactivation analysis except we included an additional cell activity criteria for selecting events. Namely, at least 15% of the place cell ensemble or more than 5 place cells, whichever was higher, needed to be active during an event for it be included for analyses."
  - The denominator is the recorded place-cell ensemble. The count is max(15%, > 5), where "more than 5" means ≥ 6.
- Normalization period: not stated.
- Merging or maximum duration for MU events: not stated.
- Brain state: none for detection (awake task, corner stops).

Control analyses only (not detection). STAR Methods, "Control analyses" and "Local field potential analysis", p. e4:
- Ripple, optional: "Fourth, we limited replay trajectory events to those which overlapped with a detected ripple (150-250Hz) event".
  - The ripple detector: "the LFP was first down-sampled to 1.2kHz and then band-pass filtered between 150 and 250Hz ... An instantaneous measure of power was found by taking the squared complex modulus of the signal at each time point. ... For ripple event detection, we identified periods where the ripple power exceeded 2.5std above the mean. The start and the end of a ripple event was marked by the point when the power crossed the mean. Events lasting less than 40ms or more than 500ms were excluded and events separated by less than 40ms were joined together."
  - The channel is not stated ("LFP from CA1"). Power smoothing for detection is not stated.
- Theta, optional: "Third, we limited the reactivation events to those whose log(theta/delta) ratio was at least one standard deviation below the mean log(theta/delta) ratio measured during movement (> 10cm/s)." Theta is 6–12 Hz and delta 2–4 Hz, both as squared Hilbert modulus.
- Other controls: speed-matched subsampling; excluding trajectories shorter than 2 m.
- Engaged versus disengaged labels (time since corner arrival or before departure) are analysis only.

## Inherited from
- Ólafsdóttir 2016 (in the manifest as 39_Olafsdottir_2016). This paper is the same rule plus a 3 cm/s speed rule, the corner restriction and the max(15%, > 5) participation.
- Pfeiffer & Foster 2013 (in the manifest as 45_Pfeiffer_2013). Followed one hop: all clustered units, 10 ms Gaussian, peak > mean + 3 SD with statistics from times < 5 cm/s, bounds at the mean, ≥ 10% of units, 50–2000 ms.
- No detail needed for the tier is left open.

## Code
None linked.

## Survey CSV discrepancies
Row 32.
- SWR Z-score Thresh.: CSV "2.5". This is the threshold of a separate ripple detector used only in a control (keeping trajectory events that overlap a ripple). It is not part of candidate detection.
- SWR Low/High Band: CSV "#N/A". The control ripple detector uses 150–250 Hz. Either fill these in as control-only, or leave all SWR fields N/A.
- Max Duration (ms): CSV "500". This applies only to the control ripple events. The MU candidate events have no stated maximum.
- Combine Events Thresh. (ms): CSV "40". This applies only to the control ripple events ("events separated by less than 40ms were joined together"). No merging is stated for MU events.
- Min. Cells (#): CSV "5". The paper requires "at least 15% of the place cell ensemble or more than 5 place cells, whichever was higher". That is max(15%, ≥ 6), and only in the replay-trajectory analysis; the arm-reactivation analysis has none.
- Matching fields: MUA 3 SD, speed 3 cm/s, MUA smooth 5 ms, min duration 40 ms.

## Package mapping
Tier: A (detection; the corner restriction needs user-computed intervals, and the ripple and theta controls are B)     Needs radiatum: n   Needs theta: n for detection (y for the log theta/delta control)   Needs sleep scoring: n
Recipe:
```python
import numpy as np
from ripple_detection import multiunit_HSE_detector
from ripple_detection.core import exclude_movement_by_majority, require_overlap

fs = 1000.0
n_pc = place_spikes.shape[1]
ev = multiunit_HSE_detector(
    time, place_spikes, speed, fs,
    zscore_threshold=3.0, minimum_duration=0.0, smoothing_sigma=0.005,
    speed_threshold=np.inf,
    minimum_active_units=max(int(np.ceil(0.15 * n_pc)), 6))  # trajectory analysis; 0 for arm reactivation
ev = ev[ev.duration >= 0.040]
ev = exclude_movement_by_majority(ev, speed, time, speed_threshold=3.0, majority_threshold=1.0)  # no sample > 3
ev = require_overlap(ev, corner_intervals)   # user-computed intervals when the animal is at either corner

# control only: ripple overlap (B)
from ripple_detection import filter_ripple_band, get_envelope, normalize_signal
from ripple_detection.core import extend_threshold_to_mean, merge_close_events
pz = normalize_signal(get_envelope(filter_ripple_band(ca1_lfp, 1200, band=(150, 250))) ** 2)  # power
rip = np.array(extend_threshold_to_mean(pz >= 0, pz >= 2.5, lfp_time, 0.0))
d = rip[:, 1] - rip[:, 0]
rip = merge_close_events(rip[(d >= 0.040) & (d <= 0.500)], 0.040)  # order of exclude vs join not stated
ev_ctrl = require_overlap(ev, rip)
```
Remaining deviations:
- Speed rule: `majority_threshold=1.0` implements "no sample above 3 cm/s". If the authors meant mean speed, that needs user code. Their speed is at 50 Hz.
- Threshold: "exceeded" 3 SD (strict) versus the package's ≥. The normalization period is not stated.
- "More than 5 place cells" is read as ≥ 6. Rounding of 15% is not stated.
- Control ripple: the package detectors threshold the envelope (Karlsson) or sqrt of summed squared envelopes (Kay), never z-scored power. So it needs user code on `get_envelope(...)**2`. Also, "2.5std above the mean" is strict versus the package's ≥, the channel is not stated, and the order of the 40–500 ms exclusion versus the 40 ms join is not stated.
- Theta control: the log(theta/delta) ratio needs user code (the package has no theta/delta).
Smallest package addition (if C): n/a.

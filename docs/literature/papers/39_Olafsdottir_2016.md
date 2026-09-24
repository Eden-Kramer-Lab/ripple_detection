# Ólafsdóttir 2016 — Coordinated grid and place cell replay during rest
Source: the extracted text (pdftotext of the Dropbox PDF, including the Online Methods and supplementary legends); title verified: yes (Ólafsdóttir, Carpenter & Barry, Nat Neurosci 19:792)
Trigger: MUA (the pooled spike rate of sorted CA1 place cells; no ripple)

## Method as implemented

Detection. Quotes are from Online Methods, "Data analysis":
- Scope: "We identified replay events from the rest session on the basis of the activity of hippocampal place cells using a similar method to Pfeiffer and Foster7 and Ólafsdóttir et al.9"
- Units: "Hippocampal cells were classified as place cells if their firing field's peak firing rate exceeded 1 Hz and was at least 20 cm long. Interneurons, identified by narrow waveforms and high firing rates, were excluded from all analyses."
- Signal and smoothing: "multi-unit (MU) activity from hippocampal place cells only were binned into 1 ms temporal bins and smoothed with a Gaussian kernel (σ = 5 ms)." This is sorted place cells pooled across CA1 tetrodes. The glyph "S = 5 ms" in the text is σ.
- Threshold: "We identified periods when the MU activity exceeded the mean rate by 3 s.d. as putative replay events".
- Bounds: "and determined the start and end points of each putative replay event as the time when the MU activity fell back to the mean."
- Duration and participation: "Events less than 40 ms long or which included activity from less than 15% of the recorded place cell ensemble were rejected (4,382 events included in total)."
  - The minimum duration applies to the whole event, start to end.
  - The participation denominator is all recorded place cells in the session, the same cells that make up the MU trace.
- Normalization period: not stated. The mean and SD are presumably from the rest session, since events are found "from the rest session", but that is an inference.
- Speed and brain state: none stated. Rest took place in an enclosure for 1.5 h ("rats were placed in the rest enclosure for an hour and a half"). No immobility, theta or sleep criterion.
- Merging or maximum duration: not stated.
- Ripple: not used. There are no occurrences of "ripple" in the paper, Online Methods or supplementary legends.

Later analysis restrictions (not detection):
- Decoding uses 10 ms bins. Line-fit replay scoring has "robust replay events exhibiting clear, straight trajectories (each P < 0.2 versus their own shuffle)" (main text, p. 792).

## Inherited from
- Ólafsdóttir 2015 (ref 9, in the manifest as 41_Olafsdottir_2015). It takes the 15% participation idea from there, but the 2015 rule is silence-bounded, not a rate threshold.
- Pfeiffer & Foster 2013 (ref 7, in the manifest as 45_Pfeiffer_2013). Followed one hop: "A histogram (1-ms bins) of all clustered units for times when the rat's velocity was less than 5 cm s−1 was smoothed (Gaussian kernel, standard deviation of 10 ms). Population events were defined as peaks in the smoothed histogram greater than the mean + 3 standard deviations. Start and end boundaries ... where the smoothed histogram crossed the mean ... fewer than 10% of the clustered units participated or with boundaries less than 50 ms or greater than 2,000 ms apart were excluded".
- The 2016 paper changes these to place cells only, 5 ms, 40 ms, 15%, no maximum and no speed rule. The 2016 text is complete for the tier decision.

## Code
None linked.

## Survey CSV discrepancies
Row 39. No discrepancies. The row matches: MUA of place cells, 1 ms bins, Gaussian 5 ms, peak > 3 SD, bounds at the mean, ≥ 40 ms, ≥ 15% of the place-cell ensemble, speed N/A, SWR N/A, max duration N/A, combine N/A.

## Package mapping
Tier: A     Needs radiatum: n   Needs theta: n   Needs sleep scoring: n
Recipe:
```python
import numpy as np
from ripple_detection import multiunit_HSE_detector

fs = 1000.0
# place_spikes: (n_time, n_place_cells) 1 ms counts, REST session only
n_pc = place_spikes.shape[1]
ev = multiunit_HSE_detector(
    time, place_spikes, speed, fs,
    zscore_threshold=3.0, minimum_duration=0.0,          # peak > 3 SD, bounds at the mean
    smoothing_sigma=0.005,
    speed_threshold=np.inf,                               # no speed criterion stated
    minimum_active_units=int(np.ceil(0.15 * n_pc)))      # >= 15% of the place-cell ensemble
ev = ev[ev.duration >= 0.040]                             # "events less than 40 ms long ... rejected"
```
Remaining deviations:
- Threshold inequality: "exceeded ... by 3 s.d." is strict (>); the package uses ≥.
- Normalization period is not stated. Passing only rest-session data (or `normalization_mask` for rest) is an assumption.
- Gaussian kernel truncation is not stated (the package truncates the Gaussian at a fixed number of SDs).
- Rounding of 15% to a cell count (ceil assumed) is not stated.
- The duration minimum is imposed by filtering `duration`, because the detector's `minimum_duration` would apply to the above-threshold run. `duration` is the time from first to last sample of the event.
Smallest package addition (if C): n/a.

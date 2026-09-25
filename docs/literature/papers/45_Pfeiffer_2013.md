# Pfeiffer 2013 — Hippocampal place-cell sequences depict future paths to remembered goals
Source: the extracted text (pdftotext of the Dropbox PDF, including the online Methods); title verified: yes (Pfeiffer & Foster, Nature 497:74–79)
Trigger: MUA (the population rate of sorted, clustered units, not threshold-crossing multiunit)

## Method as implemented

Detection (candidate "population events"). All quotes are from Methods, "Sequential event analysis", unless noted:
- Signal: sorted single units. "A histogram (1-ms bins) of all clustered units for times when the rat's velocity was less than 5 cm s−1 was smoothed (Gaussian kernel, standard deviation of 10 ms)." The glyphs in the text are "5 cm s21" and "mean 1 3 standard deviations", pdftotext renderings of cm s−1 and mean + 3.
- Unit set: "all clustered units". The cluster section says "Clustered units that may correspond to putative inhibitory neurons were excluded on the basis of spike width and mean firing rate". Whether that exclusion also applies to this histogram is not stated. Inference: probably yes.
- 40 tetrodes in CA1, 20 per hemisphere (Methods Summary). Spikes from all tetrodes are pooled into one histogram.
- Threshold: "Population events were defined as peaks in the smoothed histogram greater than the mean + 3 standard deviations." The mean and SD are those of the histogram built "for times when the rat's velocity was less than 5 cm s−1". Inference from the sentence structure: the statistics come from immobility, and detection runs only in immobility.
- Bounds: "Start and end boundaries for each population event were defined as the points where the smoothed histogram crossed the mean."
- Inward trim: "To prevent estimation artefacts, the time window boundaries for each candidate event were adjusted inward (if necessary) to ensure that the first and last estimation bins contained a minimum of 2 spikes." The estimation bins are 20 ms windows advanced in 5 ms steps (same section).
- Participation and duration: "Candidate events in which fewer than 10% of the clustered units participated or with boundaries less than 50 ms or greater than 2,000 ms apart were excluded from analysis." Both limits apply to the event bounds (which may be the trimmed ones; the order is not stated). "Participated" is not defined further.
- Merging of close events: not stated.
- Brain state: task sessions only. No theta or sleep criterion.
- Results summary: "We identified candidate events as brief increases in population spiking activity during periods of immobility while the rat performed the task" (p. 75).

Later analysis restrictions (not detection):
- Trajectory events: "each candidate replay event was truncated to the longest sequence of time frames with peak posterior probability less than 20 cm from that of the previous frame. Candidate events with fewer than 10 steps in the final sequence or a start-to-end distance less than 40 cm were eliminated" (Methods). This is defined by decoding.
- A separate SWR detection is used only for LFP analyses (SWR-triggered spectrograms): "For each tetrode, one representative electrode was selected ... band-pass filtered between 150 and 250 Hz, and the absolute value of the Hilbert transform of this filtered signal was then smoothed (Gaussian kernel, s.d. = 12.5 ms). This processed signal was averaged across all tetrodes and ripple events were identified as local peaks with an amplitude greater than 3 s.d. above the mean, using only periods when the rat's velocity was less than 5 cm s−1. The start and end boundaries for each event were defined as the point when the signal crossed the mean." (Methods, "Local field potential analysis"). Channels are combined by averaging the smoothed per-tetrode Hilbert amplitudes. This is the same rule Pfeiffer 2015 uses for detection.

## Inherited from
Nothing for the event rule, which is fully specified. Decoding is "as previously described23" (Davidson et al. 2009) and tetrode placement "as previously described22"; neither affects detection.

## Code
No code or data link in the paper.

### Code search, September 2026

- **Later.** The Pfeiffer lab's population-event finder ([Brad-Pfeiffer/MouseDevelopmentalAnalysisCode @950fc28](https://github.com/Brad-Pfeiffer/MouseDevelopmentalAnalysisCode/tree/950fc28), `KJ_FIND_POPULATION_EVENTS.m`) has no participation criterion, so it says nothing about the 10%.

## Survey CSV discrepancies
- Min. Cells (#): CSV "#N/A". The paper says "Candidate events in which fewer than 10% of the clustered units participated ... were excluded". The value should be "10% of clustered units", a fraction rather than a count.
- All other detection fields match (3 SD, 5 cm/s, 10 ms, 50 ms and 2000 ms; SWR fields N/A because the SWR detection is analysis-only).
- Detection Notes "MUA" hides that the signal is sorted clustered units in 1 ms bins, not unsorted multiunit.

## Package mapping
Tier: B (A without the inward spike-count trim)     Needs radiatum: n   Needs theta: n   Needs sleep scoring: n
Recipe:
```python
import numpy as np
from ripple_detection import multiunit_HSE_detector
# units: (n_time, n_units) 1 ms spike counts of clustered (excitatory) units
fs = 1000
stop = speed < 5.0
n_units = units.shape[1]
cand = multiunit_HSE_detector(
    time, units, speed, fs,
    smoothing_sigma=0.010, zscore_threshold=3.0, minimum_duration=0.0,  # peak > 3 SD, bounds at mean
    normalization_mask=stop, speed_threshold=5.0,
    maximum_duration=2.0,
    minimum_active_units=int(np.ceil(0.10 * n_units)),                  # ">= 10% of clustered units"
)
# user code (B): move each start later / end earlier until the first / last 20 ms window holds >= 2 spikes
cand = trim_to_two_spikes(cand, spike_times, window=0.020, step=0.005)  # ~10 lines, not in the package
cand = cand[(cand.duration >= 0.050) & (cand.duration <= 2.0)]
```
Alternative for the "histogram only for times when velocity < 5" reading: set `units[speed >= 5] = np.nan`. Each immobility period then becomes its own block, so events cannot extend into movement, smoothing stays inside the period, and the z-score is computed over immobile samples only. Pass `speed_threshold=np.inf`. Smoke-tested (the HSE part) on simulate_session data.
Remaining deviations:
- Speed rule: the package's endpoint test (≤ 5 at first and last sample) versus the paper's histogram restricted to times < 5 cm/s. The NaN-masking alternative is closer but smooths within each immobility block, and the paper does not say how it handled the edges.
- The inward trim to ≥2 spikes in the first and last 20 ms estimation window is not in the package (user code).
- Order of the trim and the 10% or duration exclusion is not stated. HSE counts n_active_units on the untrimmed bounds.
- Participation: "participated" is read as ≥1 spike in the event (the HSE definition). The paper does not define it, and does not say whether the 10% is of all clustered units or of excitatory units.
- The minimum duration applies to the extended event, so it is imposed by filtering `duration`. The HSE `minimum_duration` would instead apply to the above-threshold run, which is why it is set to 0.
- Threshold inequality: "greater than" 3 SD (strict) versus the package's ≥.
Smallest package addition (if C): n/a. A boundary-trimming helper (trim event bounds until k spikes fall in the edge window) would make this A.

## Independent parameter recheck — September 25, 2026

The current CSV supersedes the historical discrepancy list below. Independent checks and
source limitations are indexed in [the source recheck](../source_recheck.md) and
[the complete field-status ledger](../parameter_verification_2026-09-25.csv).

- **Sig. Thresh.**: `Criteria-based events; all p<0.02 under both shuffles in verification`. Pfeiffer 2013 Methods, PDF p. 8: 5000 cell-ID and place-field-shift shuffles verify the selected trajectory events.

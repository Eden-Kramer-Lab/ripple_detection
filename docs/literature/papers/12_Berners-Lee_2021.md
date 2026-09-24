# Berners-Lee 2021 — Prefrontal Cortical Neurons Are Selective for Non-Local Hippocampal Representations during Replay and Behavior
Source: the extracted text (pdftotext of Dropbox J Neurosci 2021 PDF); title verified: yes (Berners-Lee, Wu, Foster; J Neurosci 41(27):5894–5908). Note: this PDF's font renders ">" as "." (".2 SD" = ">2 SD", ".5 cm/s" = ">5 cm/s"). The same substitution appears in the place-field text ("moving .5 cm/s"), which confirms the reading.
Trigger: SWR

## Method as implemented
Detection (Materials and Methods, "Candidate event and replay analysis", pp. 5896–5897):
- Channels: "For each recording session, we identified the three tetrodes from which the most HP neurons were isolated." All are in dorsal CA1 ("Tetrodes were gradually moved into the CA1 pyramidal cell layer", p. 5895). Which wire of each tetrode was used: not stated.
- Filter / envelope / smoothing: "The LFP from these tetrodes was bandpass filtered between 150 and 250 Hz, and the absolute value of the Hilbert transform of this filtered signal was then smoothed (Gaussian kernel, SD = 12.5 ms)." Filter type: not stated.
- Combining: "To examine SWRs, these processed signals were averaged across all three tetrodes" (mean of the smoothed envelopes).
- Threshold and speed: "SWRs were identified as local peaks with an amplitude >2 SD above the mean, excluding periods when the rat's speed was >5 cm/s." The SD is taken over: not stated. The sentence can be read as excluding fast periods from detection, from the baseline, or both.
- Bounds: "The start and end boundaries for each event were defined as the point when the signal crossed the mean."
- Duration (applies to the mean-to-mean event): "SWRs shorter than 50 ms or longer than 2 s were excluded from further analysis."
- Merging close events: not stated. Brain state, cell participation, artifact rule: not stated.
- Results: "In each session, we identified SWRs while the rat was paused on the track (mean = 1985.3, range: 696–2909, total: 21,838" (p. 5903–5904).

Analysis (not detection):
- A Bayesian decoder (Davidson et al. 2009; 20 ms bins, 5 ms step) was applied to the candidate events. "Arm-replays" were defined from MAP-function subregions (MAP > 4× chance for >= 50 ms, arm coverage > 50%, weighted correlation > 0.3 or, stricter, > 0.6 with max jump < 0.4).

## Inherited from
"Data from four of the 11 sessions analyzed here were also used in a previous study (Wu and Foster, 2014). The recording and preprocessing methods in this paper are identical to that study and are re-stated here." (p. 5895). The SWR procedure is fully restated. Following the citation one hop: Wu & Foster 2014 (manifest 43, the extracted text) did NOT use SWRs as candidates. Its candidates were place-cell spike-density events: "smoothed spike density function ... (10 ms time bins; Gaussian filter SD = 15 ms). Candidate events were defined as epochs of spikes during which spike densities were above the mean of the function, and contained peaks above 2 SDs over the mean. Only candidate events that occurred when a rat's speed was <5 cm/s were considered." Its separate ripple detection used 13–15 tetrodes, envelope averaged then smoothed (SD 8 ms), peaks > 2.5 SD, "both calculated across all stopping periods". So the "identical" claim covers recording and preprocessing only. Berners-Lee 2021's SWR candidate definition is its own and is authoritative here.

## Code
None linked in the paper.

## Survey CSV discrepancies
No discrepancies. SWR 2 SD, speed 5, smoothing 12.5 ms, 3 electrodes, 150–250 Hz, min 50 ms, max 2000 ms and combine N/A all match the paper. (The survey's "3" electrodes is really "3 tetrodes with the most isolated HP neurons".)

## Package mapping
Tier: B     Needs radiatum: n   Needs theta: n   Needs sleep scoring: n
Recipe:
```python
import numpy as np
from ripple_detection import (filter_ripple_band, get_envelope, gaussian_smooth,
    normalize_signal, threshold_by_zscore, exclude_movement)
filt = filter_ripple_band(lfp_3tet, fs, band=(150, 250))            # (n_time, 3)
trace = gaussian_smooth(get_envelope(filt), 0.0125, fs).mean(axis=1)
still = speed <= 5
z = normalize_signal(trace, normalization_mask=still)               # baseline period not stated (inference)
events = np.array(threshold_by_zscore(z, time, minimum_duration=0.0, zscore_threshold=2))
events = exclude_movement(events, speed, time, speed_threshold=5)   # or mask z where speed > 5
dur = events[:, 1] - events[:, 0]
events = events[(dur >= 0.050) & (dur <= 2.0)]
```
Tier-A approximation: `Roumis_ripple_detector(time, filt, speed, fs, speed_threshold=5, minimum_duration=0.0, zscore_threshold=2, smoothing_sigma=0.0125, normalization_mask=speed <= 5, maximum_duration=2.0)`, then keep `duration >= 0.05`.
Remaining deviations:
- Roumis combines as mean of sqrt(smoothed squared envelope). The paper uses the mean of the smoothed envelope. The B recipe above is exact.
- Minimum duration: the paper's 50 ms applies to the mean-to-mean event, so it is applied afterwards on `duration`, not as the detector's `minimum_duration`, which in the package applies to the above-threshold run.
- "Excluding periods when speed > 5 cm/s" is unspecified. Options: the endpoint rule (exclude_movement), masking the trace so events end at movement, or excluding fast periods from the baseline. The paper does not say which, or whether more than one applies.
- ">= 2" in the package versus ">2 SD" in the paper, and ">= 0" versus "crossed the mean".
- Filter design is unstated in the paper; the package uses a remez FIR, zero-phase.
Smallest package addition (if C): n/a

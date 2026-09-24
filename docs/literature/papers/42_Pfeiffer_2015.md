# Pfeiffer 2015 — Autoassociative dynamics in the generation of sequences of hippocampal place cells
Source: the extracted text (pdftotext of the Dropbox PDF: the report plus Supplementary Materials; the first column of page 1 belongs to the preceding bumblebee article); title verified: yes (Pfeiffer & Foster, Science 349:180–183)
Trigger: SWR

## Method as implemented

Detection (SWR events). Quotes are from Supplementary Materials, Materials and Methods, "Local Field Potential Analysis", SM p. 3:
- Channels: pyramidal layer, one wire per tetrode, all tetrodes that carried excitatory units. "For each tetrode, one representative electrode was selected and the LFP signal was analyzed. Only tetrodes on which excitatory hippocampal cells were recorded were used, thus all LFP signals were recorded in the pyramidal layer." The number of tetrodes used is not stated. The drive has 40 tetrodes, 20 per hemisphere (SM, Materials and Methods, p. 2). LFP was recorded at 3,255.6 Hz, 0.1–500 Hz.
- Filter and envelope: "the LFP was band-pass filtered between 150 and 250 Hz, and the absolute value of the Hilbert transform of this filtered signal was then smoothed (Gaussian kernel, SD = 12.5 ms)." The filter type and order are not stated.
- Channel combination: "This processed signal was averaged across all tetrodes". This is the mean over tetrodes of the Gaussian-smoothed Hilbert amplitude. Amplitude is not squared.
- Threshold and normalization: "ripple events were identified as local peaks with an amplitude greater than 3 SD above the mean, excluding periods when the rat's velocity was greater than 5 cm/sec." Whether the speed exclusion applies to the mean and SD, to the detection, or to both is not stated. The companion 2013 wording, "using only periods when the rat's velocity was less than 5 cm s−1", suggests both (inference).
- Bounds: "The start and end boundaries for each event were defined as the point when the signal crossed the mean."
- Duration: "SWRs shorter than 50 ms or longer than 2 s were excluded from further analysis." This applies to the mean-crossing-bounded event.
- Merging of close events: not stated. Several local peaks inside one supra-mean epoch share the same bounds, so they are effectively one event (inference).
- No MUA, cell-count, theta or sleep criterion for detection. Recording was during task behaviour (open field and linear track).

Later analysis restrictions (not detection):
- Trajectory events: "each candidate replay event was truncated to the longest sequence of time frames with a weighted mean posterior probability less than 50 cm from that of the previous frame. Candidate events with fewer than 10 steps in the final sequence or a start-to-end distance less than 80 cm were eliminated from future analysis" (SM, "Trajectory Event Analysis"). The decoding window is 20 ms advanced in 5 ms steps.
- Rats were included only if they had at least 80 simultaneously recorded place units (SM, "Cluster Analysis"). This is a dataset criterion.

## Inherited from
Decoding is "as previously described (10)", which is Pfeiffer & Foster 2013. The SWR rule is written out in full here. It is the same rule Pfeiffer 2013 used only for its LFP analyses (quoted in 45_Pfeiffer_2013.md), with the 50 ms–2 s limits added. The rule descends from Davidson et al. 2009's ripple-amplitude trace (mean amplitude across sites, Gaussian SD 12.5 ms), with 3 SD in place of 2.5 SD and bounds at the mean.

## Code
No code link for this paper. The "supporting scripts are available from Dryad Digital Repository: doi:10.5061/dryad.gf774" line on page 1 belongs to the preceding bumblebee article (Kerr et al.), not to Pfeiffer & Foster.

## Survey CSV discrepancies
No discrepancies (SWR, 3 SD, 5 cm/s, 12.5 ms, 150–250 Hz, 50 ms and 2000 ms all match). The "SWR electrodes (#)" value ">1" is vague rather than wrong. The paper says one electrode per tetrode, on every tetrode that carried excitatory cells, and averages across them. The count is not stated.

## Package mapping
Tier: B (A-level approximation: Roumis_ripple_detector)     Needs radiatum: n   Needs theta: n   Needs sleep scoring: n
Recipe (exact trace, B):
```python
import numpy as np
from ripple_detection import (filter_ripple_band, get_envelope, gaussian_smooth,
                              normalize_signal, threshold_by_zscore, exclude_movement)
# lfps: (n_time, n_tetrodes), one pyramidal-layer wire per tetrode
filtered = filter_ripple_band(lfps, fs, band=(150.0, 250.0))
trace = gaussian_smooth(get_envelope(filtered), 0.0125, fs).mean(axis=1)   # mean of smoothed amplitudes
z = normalize_signal(trace, normalization_mask=speed <= 5.0)
events = np.asarray(threshold_by_zscore(z, time, minimum_duration=0.0, zscore_threshold=3.0))
events = exclude_movement(events, speed, time, speed_threshold=5.0)       # one reading of "excluding periods"
duration = events[:, 1] - events[:, 0]
swrs = events[(duration >= 0.050) & (duration <= 2.0)]
```
Approximation with a detector (A):
```python
ev = Roumis_ripple_detector(time, filtered, speed, fs, smoothing_sigma=0.0125, zscore_threshold=3.0,
                            minimum_duration=0.0, normalization_mask=speed <= 5.0,
                            speed_threshold=5.0, maximum_duration=2.0)
ev = ev[ev.duration >= 0.050]
```
Both recipes were smoke-tested on simulate_session data at 1 kHz, and both recovered all simulated ripples.
Remaining deviations:
- Roumis combines channels differently. It computes mean(sqrt(Gaussian-smoothed envelope²)), a Gaussian-weighted RMS amplitude, per channel (checked in `_lfp.py`, `_smoothed_envelope(..., square=True)` then `np.mean(np.sqrt(...))`). The paper uses mean(Gaussian-smoothed envelope). Hence B for the exact trace.
- Speed: "excluding periods when velocity > 5 cm/s" is ambiguous. The recipe applies it to the normalization (mask) and to the event endpoints. An alternative reading (NaN-out moving samples so they are excluded outright) is also expressible.
- Filter: the package uses an equiripple FIR (remez) with zero-phase filtfilt. The paper's filter design is not stated.
- "Local peaks greater than 3 SD" (strict) versus the package's ≥, and bounds at z ≥ 0 versus crossing the mean. Multiple peaks in one supra-mean epoch become one event in the package. In the paper they would share identical bounds.
- The minimum duration applies to the extended event, so it is applied as a filter on duration after detection, with the detector's minimum_duration set to 0.
Smallest package addition (if C): n/a. A `square=False` / "mean amplitude" combination option on Roumis (or a detector that thresholds a user-supplied trace with the shared tail) would make this A.

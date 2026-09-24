# Kudrimoti 1999 — Reactivation of Hippocampal Cell Assemblies: Effects of Behavioral State, Experience, and EEG Dynamics
Source: the extracted text (pdftotext of the Dropbox PDF); title verified: yes
Trigger: SWR (used to split SWS into ripple vs inter-ripple segments; the main reactivation measure is not event-gated)

## Method as implemented
- Signal (Methods, "Recording protocol", p. 4091–4092): continuous EEG from "one channel from each of 12 tetrodes ... bandpass filtered between 1 and 100 Hz or between 1 Hz and 3 kHz, and then sampled at 200 Hz or 1 kHz"; "The 1 kHz–sampled data were used to identify 100–200 Hz 'ripples' in the EEG during SWS".
- Channel choice (EEG analysis, p. 4092): "The experimenter selected two raw EEG traces with the largest overall theta rhythm amplitudes, sampled at 200 Hz and 1 kHz, respectively"; ripples from "the 1 kHz–sampled EEG traces ... recorded near the pyramidal layer". One channel (inferred from "two raw EEG traces", one per sampling rate).
- Filter and rule (p. 4092): "The traces were digitally bandpass filtered between 100 and 300 Hz, and the times of the ripples were identified by an algorithm that detected a ripple when the amplitude of the filtered EEG crossed a set threshold and remained above the threshold for at least 25 msec."
  - Amplitude measure (rectified, envelope, RMS): not stated.
  - Threshold value and units: not stated ("a set threshold"; no SD).
  - Bounds: "start and end times" are reported, rule not stated (threshold crossings is the natural reading; inference).
  - Maximum duration, merging: not stated.
- Session selection (Results, p. 4094): "only those data sets with robust ripple activity in the EEG record (i.e., those with optimal positioning of the EEG electrodes) were considered for analysis of activity during ripples".
- Sleep scoring (Methods, "Sleep scoring"): online "combination of behavioral and EEG criteria. The experimenter observed the rat's behavioral state on the television monitor, while writing down the EEG state four to five times per minute and concurrently listening to an audio monitor for ripples"; states LIA, SWS ("the rat clearly sleeping; but no theta rhythm in the EEG; onset often associated with obvious neocortical spindle activity"), REM ("theta in the hippocampal EEG; the rat sleeping"). Offline: the two EEG traces were "visually inspected ... from the start to the end of both prebehavior and postbehavior sleep. The sleep-scoring notes were matched with the EEG trace ... to identify periods of REM activity, LIA and SWS states, and awake theta".
- Speed: none (nest rest/sleep).
- ANALYSIS, not detection: firing-rate correlations in 100 ms bins over whole SWS epochs (EV); for the ripple analysis, "an inter-ripple interval midway between the two ripples bounding it and of duration equal to that of the preceding ripple was selected" (duration-matched comparison).

## Inherited from
Nothing deferred for the detector itself (the citations O'Keefe 1976, Buzsáki et al. 1992, Ylinen et al. 1995 are for the phenomenon).

## Code
None linked.

## Survey CSV discrepancies
No discrepancies. Minor: "SWR Z-score Thresh." = "Not reported" is right, but the threshold is an amplitude threshold, not a z-score; the paper also writes "100–200 Hz ripples" in one place while filtering 100–300 Hz (CSV uses 100–300, the filter band — correct). "Min. Duration 25" applies to the time above threshold, not to a separately bounded event.

## Package mapping
Tier: B (structure reproducible; the threshold value is not reported, so the paper's exact inventory cannot be reproduced)     Needs radiatum: n   Needs theta: y (only through the manual sleep scoring)   Needs sleep scoring: y (manual, online behavior + visual EEG)
Recipe:
```python
from ripple_detection.core import extend_threshold_to_mean
env = get_envelope(filter_ripple_band(lfp_1ch, 1000, band=(100, 300)))[:, 0]
above = env >= threshold                        # unreported; absolute amplitude
events = extend_threshold_to_mean(above, above, time, minimum_duration=0.025)  # bounds = the crossings
events = require_overlap(np.asarray(events), sws_epochs)   # user-supplied SWS intervals
```
(Alternative: `Karlsson_ripple_detector` on the one channel with `minimum_duration=0.025`, `speed_threshold=np.inf`, but it thresholds a z-score and extends bounds to the mean, both unlike the paper.)
Remaining deviations:
- Threshold value unknown; any z-score or absolute value is a guess.
- Hilbert envelope vs unspecified "amplitude of the filtered EEG" (could be rectified signal).
- Bound rule unstated; the recipe takes the threshold crossings.
- Sleep states from manual scoring must be supplied.
Smallest package addition (if C): n/a.

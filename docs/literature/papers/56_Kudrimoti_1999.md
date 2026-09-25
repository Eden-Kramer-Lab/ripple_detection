# Kudrimoti 1999 — Reactivation of Hippocampal Cell Assemblies: Effects of Behavioral State, Experience, and EEG Dynamics
Source: the extracted text (pdftotext of the Dropbox PDF); title matched
Trigger: SWR (used to split SWS into ripple vs inter-ripple segments; the main reactivation measure is not event-gated)

[Paper](https://doi.org/10.1523/JNEUROSCI.19-10-04090.1999) · [Field evidence](../evidence.csv) · [Source catalog](../sources.md)

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

## Uncertainties

The single ripple-channel count is inferred from the selected EEG traces, not explicitly stated. The absolute ripple threshold is not reported. Any executable recipe must assume a threshold and sleep-state rule; it cannot be an exact reconstruction.

## Package mapping

Executable example: `kudrimoti_1999` in [literature_recipes.py](../../../examples/literature_recipes.py). Its docstring records implementation choices and assumptions. Simulation checks establish that it runs; they do not establish equivalence to the authors’ original event set.

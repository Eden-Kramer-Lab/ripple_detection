# Pfeiffer 2013 — Hippocampal place-cell sequences depict future paths to remembered goals
Source: the extracted text (pdftotext of the Dropbox PDF, including the online Methods); title matched (Pfeiffer & Foster, Nature 497:74–79)
Trigger: MUA (the population rate of sorted, clustered units, not threshold-crossing multiunit)

[Paper](https://doi.org/10.1038/nature12112) · [Field evidence](../evidence.csv) · [Source catalog](../sources.md)

Additional sources: [mouse-development-code](../sources.md#mouse-development-code), [theta-code](../sources.md#theta-code).

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

### Related code

- **Later.** The Pfeiffer lab's population-event finder ([Brad-Pfeiffer/MouseDevelopmentalAnalysisCode @950fc28](https://github.com/Brad-Pfeiffer/MouseDevelopmentalAnalysisCode/tree/950fc28), `KJ_FIND_POPULATION_EVENTS.m`) has no participation criterion, so it says nothing about the 10%.

## Analysis and interpretation

Exclude events:
steps (<10)
start-to-end distance (<40 cm)

Trajectory events are selected by criteria; 5000 cell-identity and place-field-shift shuffles verify the selected events, all p<0.02 under both shuffles (Methods, PDF p. 8).

## Uncertainties

Related later lab code cannot establish the original population-event caller or its participation options.

## Package mapping

Packaged primary method: `pfeiffer_2013` in [literature_methods.py](../../../src/ripple_detection/literature_methods.py). Its docstring records implementation choices and assumptions. Simulation checks establish that it runs; they do not establish equivalence to the authors’ original event set.

Additional inventories in the same module: `pfeiffer_2013_ripples`. See their docstrings for required settings and output stages.

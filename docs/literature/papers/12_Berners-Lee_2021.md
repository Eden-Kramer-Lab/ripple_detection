# Berners-Lee 2021 — Prefrontal Cortical Neurons Are Selective for Non-Local Hippocampal Representations during Replay and Behavior
Source: the extracted text (pdftotext of Dropbox J Neurosci 2021 PDF); title matched (Berners-Lee, Wu, Foster; J Neurosci 41(27):5894–5908). Note: this PDF's font renders ">" as "." (".2 SD" = ">2 SD", ".5 cm/s" = ">5 cm/s"). The same substitution appears in the place-field text ("moving .5 cm/s"), which confirms the reading.
Trigger: SWR

[Paper](https://doi.org/10.1523/JNEUROSCI.1158-20.2021) · [Field evidence](../evidence.csv) · [Source catalog](../sources.md)

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

"Data from four of the 11 sessions analyzed here were also used in a previous study (Wu and Foster, 2014). The recording and preprocessing methods in this paper are identical to that study and are re-stated here." (p. 5895). The SWR procedure is fully restated. Following the citation one hop: Wu & Foster 2014 ([paper note](43_Wu_2014.md), the extracted text) did NOT use SWRs as candidates. Its candidates were place-cell spike-density events: "smoothed spike density function ... (10 ms time bins; Gaussian filter SD = 15 ms). Candidate events were defined as epochs of spikes during which spike densities were above the mean of the function, and contained peaks above 2 SDs over the mean. Only candidate events that occurred when a rat's speed was <5 cm/s were considered." Its separate ripple detection used 13–15 tetrodes, envelope averaged then smoothed (SD 8 ms), peaks > 2.5 SD, "both calculated across all stopping periods". So the "identical" claim covers recording and preprocessing only. Berners-Lee 2021's SWR candidate definition is its own and is authoritative here.

## Code

None linked in the paper.

## Analysis and interpretation

demarcate events based on:
length of subregion in time (>50 ms)
arm coverage (>50% of the arm)
 weighted correlation (abs. value > .3)

## Reproduction limits

The method-specific qualifications above apply. The executable recipe documents its assumptions about unstated parameters, state scoring and simulation stand-ins.

## Package mapping

Executable example: `berners_lee_2021` in [literature_recipes.py](../../../examples/literature_recipes.py). Its docstring records implementation choices and assumptions. Simulation checks establish that it runs; they do not establish equivalence to the authors’ original event set.

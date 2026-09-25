# Yamamoto 2017 — Direct Medial Entorhinal Cortex Input to Hippocampal CA1 Is Crucial for Extended Quiet Awake Replay
Source: the extracted text (pdftotext of the Zotero PDF, including STAR Methods and Supplemental Figures); title matched (Yamamoto & Tonegawa, Neuron 96:217–227.e1–e4)
Trigger: SWR+MUA

[Paper](https://doi.org/10.1016/j.neuron.2017.09.017) · [Field evidence](../evidence.csv) · [Source catalog](../sources.md)

## Method as implemented

Detection (candidate ripple / replay events):

- "In order to determine candidate events, we used both ripple band power (140-200 Hz) and multi-unit activities. We first identified periods in which ripple band LFP power of selected recording channel exceeded the 3 SD level of the baseline." (STAR Methods, "Candidate Ripple and Replay Event Detection", p. e3).

- MUA: "For the linear probes, we first identified positional range of recording sites along the probes and summed spike counts within the region of interest at 10 ms time bins as spiking activities. For the tetrodes, we first performed standard spike sorting based on spike amplitudes and binned them into 10 ms bins. Sum was then computed across tetrodes that are located in the putative pyramidal cell layer." (same section). With tetrodes, the MUA is the sum of sorted spikes (Figs 4–5). With silicone probes, it is detected spikes on the region-of-interest sites (Figs 1–3, 6).

- MUA threshold and bounds: "We then identified instantaneous multi-unit activity peak firing rates that exceeded the threshold (i.e., 3 SD of the baseline level) and then searched a time-window around the instantaneous multi-unit activity peak until the power reached the cut-off (1 SD) level at both ends. We defined the extracted time window as candidate events and assign identification numbers that was similar algorithm to the previous report (Davidson et al., 2009)." (same section).

- Ambiguities (not stated):
  - (i) How the ripple-power criterion and the MUA criterion combine. It is probably an AND (an MUA peak within a ripple-power period), but that is not said.
  - (ii) Which trace sets the 1 SD bounds. The text says "the power", which suggests ripple power, but the window is searched "around the instantaneous multi-unit activity peak".
  - (iii) What "baseline" is: which period or state the mean and SD come from.
  - (iv) How "power" is computed (squared filtered signal, Hilbert, or band-integrated), whether it is smoothed, and the filter type.
  - (v) Which channel is the "selected recording channel".

- Smoothing: none stated for either trace in detection. The figure legends mention Gaussian smoothing (σ = 20 ms) only for cross-correlation trend lines.

- Minimum and maximum duration: not stated. The typical "single ripple duration (~80 ms)" is reported.

- Merging for detection: not stated.

- Speed: not stated for detection. "Velocity filters were applied (2 cm/s) to extract valid run segments from the electrophysiological data." (STAR Methods, "Behavior Position Tracking", p. e2). That is for run and place-field data.

- Brain state (all results are split by state): "The 15 s epochs that have movement of less than 2 cm and delta/theta power ratio that is greater than 5 SD were classified as slow-wave sleep ... Similar algorithm is used to define quiet awake but with different thresholds (more than 2 cm movement over 15 s and delta/theta power ratio of 2 SD to 5 SD period." (STAR Methods, "Sleep Classification", p. e3). Delta is 1–4 Hz and theta is 6–12 Hz, from AR (modified covariance) PSD estimates ("LFP Power Density Estimation"). Supp. Fig. S1C shows the delta/theta ratio smoothed with σ = 10 s and σ = 60 s, with 2 SD and 5 SD thresholds.

- Ripple-burst classification (analysis on detected events): "Singlet ripples were defined as ripple events that are temporally separated by 200 ms or more to adjacent ripple events. For the doublets and triplets (ripple bursts), the temporal lags were set to less than 200 ms but greater than 70 ms ... In the event that adjacent ripple closer than 70 ms, these were categorized as single ripple." (STAR Methods, "Ripple Burst Analysis", p. e3). The lags are between ripple peaks.

- Cell participation: none stated for candidates. The replay analysis requires continuous posterior segments of ≥ 4 bins (fragmentation index), which is analysis.

## Inherited from

"similar algorithm to the previous report (Davidson et al., 2009)". Davidson (manifest 50_Davidson_2009; findings file exists) uses MUA from all > 100 µV spikes in 1 ms bins with σ = 15 ms, peak ≥ 3 SD, bounds at the mean, and statistics over STOP (< 5 cm/s). Yamamoto differs:

- 10 ms bins and no stated smoothing.

- Bounds at 1 SD instead of the mean.

- An added ripple-power > 3 SD requirement (140–200 Hz).

- "Baseline" instead of STOP.
Following Davidson does not resolve the ambiguities, because Yamamoto departs from it exactly where the ambiguities are. Sleep classification cites Haggerty & Ji 2014 for the delta/theta ratio. I did not follow it, because the thresholds are stated here.

## Code

No code link in the paper.

### Related code

- Searched GitHub (code, repositories, and the lab's and authors' accounts), Zenodo, Figshare and CRCNS/DANDI: no original detector caller established in these searches. No relevant Tonegawa-lab or first-author repositories, or code for the ripple doublet/triplet rule, were found in those searches.

## Analysis and interpretation

Methods (PDF p. 16) specify 10 ms replay bins and 2000 posterior-column shuffles, but no numerical replay p-value cutoff or bin step.

## Uncertainties

The numerical replay p-value cutoff and decoding step are not reported. The combination of ripple and MUA traces and the trace that determines candidate boundaries are under-specified.

## Package mapping

Executable example: `yamamoto_2017` in [literature_recipes.py](../../../examples/literature_recipes.py). Its docstring records implementation choices and assumptions. Simulation checks establish that it runs; they do not establish equivalence to the authors’ original event set.

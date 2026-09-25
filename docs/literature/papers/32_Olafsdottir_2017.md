# Ólafsdóttir 2017 — Task Demands Predict a Dynamic Switch in the Content of Awake Hippocampal Replay
Source: the extracted text (pdftotext of the Zotero PDF, including STAR Methods); title matched (Ólafsdóttir, Carpenter & Barry, Neuron 96:925)
Trigger: MUA (the pooled rate of sorted CA1 place cells). A ripple is used only in a control analysis.

[Paper](https://doi.org/10.1016/j.neuron.2017.09.035) · [Field evidence](../evidence.csv) · [Source catalog](../sources.md)

Additional sources: [shipley-code](../sources.md#shipley-code), [barry-python](../sources.md#barry-python).

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

- The paper states its candidate-event rule directly.

## Code

None linked.

### Related code

- **Later.** The [Barry-lab Shipley repository @d75b85b](https://github.com/Barry-lab/Publication_Shipley-et-al.-Disrupted-hippocampal-replay-in-an-Alzheimer-s-mouse-model/tree/d75b85b), `runReplayAnalysisNew.m` lines 181-185, builds the MUA from place cells only and passes it to `detectMUA`; the cell criterion is applied at decoding. This is 2025 code descended from Bush's detector, so the evidence is weak.

- [Barry-lab/PythonSpkAnalysis @d51c15a](https://github.com/Barry-lab/PythonSpkAnalysis/tree/d51c15a) ports the lab's old `detect_ripples` (5 SD peak, 0.5 SD bounds, 50 ms boxcar), which does not match this paper's control rule (> 2.5 SD, 40-500 ms).

## Reproduction limits

The method-specific qualifications above apply. The executable recipe documents its assumptions about unstated parameters, state scoring and simulation stand-ins.

## Package mapping

Executable example: `olafsdottir_2017` in [literature_recipes.py](../../../examples/literature_recipes.py). Its docstring records implementation choices and assumptions. Simulation checks establish that it runs; they do not establish equivalence to the authors’ original event set.

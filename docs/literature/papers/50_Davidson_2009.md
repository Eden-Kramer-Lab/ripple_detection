# Davidson 2009 — Hippocampal Replay of Extended Experience
Source: the extracted text (pdftotext of the Dropbox PDF, including the Supplemental Data); title matched (Davidson, Kloosterman & Wilson, Neuron 63:497–507)
Trigger: MUA

[Paper](https://doi.org/10.1016/j.neuron.2009.07.027) · [Field evidence](../evidence.csv) · [Source catalog](../sources.md)

## Method as implemented

Detection (candidate replay events, "CAND"):

- Signal: every threshold-crossing spike, sorted or not (true multiunit, not sorted units). "A smoothed histogram (1 ms bins; Gaussian kernel, SD = 15 ms) was constructed of multiunit activity (MUA) including all spikes with a peak amplitude greater than 100 mV on any channel, whether or not they are part of an isolated cluster." (Experimental Procedures, "Candidate Replay Events", p. 505). The extracted text reads "100 mV". The PDF's own text layer also says "mV", so this is presumably a μ→m glyph substitution and the value is 100 μV (inference).

- Channels: all electrodes pooled. Results: "periods during STOP with elevated multiunit activity across all electrodes" (p. 497). The Fig. 1D and Fig. S4 legends plot MUA as "average spike rate per tetrode, including unclustered spikes". Averaging rather than summing only rescales the trace, so the z-score is the same. 9–18 tetrodes or octrodes in CA1 (p. 504).

- Bins and smoothing: 1 ms bins, Gaussian SD 15 ms (quoted above).

- Normalization over STOP only: "Mean and standard deviation of MUA during STOP was calculated" (p. 505). STOP is "linearized speed is <5 cm/s", after "Linearized velocity was smoothed with a Gaussian kernel (SD = 0.25 s)" (Experimental Procedures, "Electrophysiology and Behavior", p. 505).

- Threshold and bounds: "candidate replay events were defined as epochs during which MUA was higher than the mean and peak rate was at least three standard deviations above the mean" (p. 505). So the peak must be ≥ 3 SD and the bounds are the crossings of the mean.

- Minimum and maximum duration for detection: not stated. The observed range was "event durations ranged from 40 to 1018 ms" (Results, p. 497).

- Merging of close events: not stated.

- Speed or behaviour: events lie in STOP. "Candidate replay events ('CAND') were identified as periods during STOP with elevated multiunit activity" (Results, p. 497). How STOP is applied (whole event, endpoints or peak) is not stated.

- Cell or unit participation: not stated (no minimum).

- Brain state: no sleep scoring. Proxy (see the analysis restrictions below): "Only candidate events within 30 s of RUN were analyzed to exclude possible sleep periods" (p. 505). RUN is smoothed linear speed >15 cm/s (p. 505).

Later analysis restrictions (not detection):

- Decoding only of events ≥100 ms: "We next applied the decoding algorithm to nonoverlapping 20 ms time windows in all candidate events lasting at least 100 ms" (Results, p. 498).

- The 30-s-of-RUN rule is worded as an analysis restriction ("were analyzed"). Its purpose is to exclude sleep.

- A separate ripple detection is used only for ripple-rate and ripple-triggered analyses: "band-pass filtering the local field potential (LFP) signal between 150 and 250 Hz, then taking the absolute value of the Hilbert-transformed signal ... The mean ripple amplitude across all recording sites was then smoothed (Gaussian kernel, SD = 12.5 ms) to give a single continuous measure of ripple activity. Individual ripples were detected as local peaks in this signal with an amplitude larger than 2.5 SD above the mean (mean and SD measured during STOP epochs)" ("Ripple Detection and Ripple-Triggered Analyses", p. 505). This combines channels as the mean of the per-site Hilbert amplitudes, then applies Gaussian smoothing. Each local peak is one ripple, with no bounds ("allows for the detection of closely spaced ripples"). The procedure is described as "a variation of Skaggs' (Skaggs et al., 2007) ripple-detection procedure".

## Inherited from

Nothing for the candidate-event (MUA) rule, which is fully specified in the paper. The ripple analysis is "a variation of Skaggs' (Skaggs et al., 2007)" procedure. Its parameters are given in full, so I did not follow that citation.

## Code

No code or data link in the paper.

## Reproduction limits

The method-specific qualifications above apply. The executable recipe documents its assumptions about unstated parameters, state scoring and simulation stand-ins.

## Package mapping

Packaged primary method: `davidson_2009` in [literature_methods.py](../../../src/ripple_detection/literature_methods.py). Its docstring records implementation choices and assumptions. Simulation checks establish that it runs; they do not establish equivalence to the authors’ original event set.

Additional inventories in the same module: `davidson_2009_ripples`. See their docstrings for required settings and output stages.

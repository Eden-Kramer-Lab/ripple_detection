# Ólafsdóttir 2016 — Coordinated grid and place cell replay during rest
Source: the extracted text (pdftotext of the Dropbox PDF, including the Online Methods and supplementary legends); title matched (Ólafsdóttir, Carpenter & Barry, Nat Neurosci 19:792)
Trigger: MUA (the pooled spike rate of sorted CA1 place cells; no ripple)

[Paper](https://doi.org/10.1038/nn.4291) · [Field evidence](../evidence.csv) · [Source catalog](../sources.md)

Additional sources: [olafsdottir-software](../sources.md#olafsdottir-software), [barry-data](../sources.md#barry-data).

## Method as implemented

Detection. Quotes are from Online Methods, "Data analysis":

- Scope: "We identified replay events from the rest session on the basis of the activity of hippocampal place cells using a similar method to Pfeiffer and Foster7 and Ólafsdóttir et al.9"

- Units: "Hippocampal cells were classified as place cells if their firing field's peak firing rate exceeded 1 Hz and was at least 20 cm long. Interneurons, identified by narrow waveforms and high firing rates, were excluded from all analyses."

- Signal and smoothing: "multi-unit (MU) activity from hippocampal place cells only were binned into 1 ms temporal bins and smoothed with a Gaussian kernel (σ = 5 ms)." This is sorted place cells pooled across CA1 tetrodes. The glyph "S = 5 ms" in the text is σ.

- Threshold: "We identified periods when the MU activity exceeded the mean rate by 3 s.d. as putative replay events".

- Bounds: "and determined the start and end points of each putative replay event as the time when the MU activity fell back to the mean."

- Duration and participation: "Events less than 40 ms long or which included activity from less than 15% of the recorded place cell ensemble were rejected (4,382 events included in total)."
  - The minimum duration applies to the whole event, start to end.
  - The participation denominator is all recorded place cells in the session, the same cells that make up the MU trace.

- Normalization period: not stated. The mean and SD are presumably from the rest session, since events are found "from the rest session", but that is an inference.

- Speed and brain state: none stated. Rest took place in an enclosure for 1.5 h ("rats were placed in the rest enclosure for an hour and a half"). No immobility, theta or sleep criterion.

- Merging or maximum duration: not stated.

- Ripple: not used. There are no occurrences of "ripple" in the paper, Online Methods or supplementary legends.

Later analysis restrictions (not detection):

- Decoding uses 10 ms bins. Line-fit replay scoring has "robust replay events exhibiting clear, straight trajectories (each P < 0.2 versus their own shuffle)" (main text, p. 792).

## Inherited from

- Ólafsdóttir 2015 (ref 9, [paper note](41_Olafsdottir_2015.md)). It takes the 15% participation idea from there, but the 2015 rule is silence-bounded, not a rate threshold.

- Pfeiffer & Foster 2013 (ref 7, [paper note](45_Pfeiffer_2013.md)). Followed one hop: "A histogram (1-ms bins) of all clustered units for times when the rat's velocity was less than 5 cm s−1 was smoothed (Gaussian kernel, standard deviation of 10 ms). Population events were defined as peaks in the smoothed histogram greater than the mean + 3 standard deviations. Start and end boundaries ... where the smoothed histogram crossed the mean ... fewer than 10% of the clustered units participated or with boundaries less than 50 ms or greater than 2,000 ms apart were excluded".

- The 2016 paper changes these to place cells only, 5 ms, 40 ms, 15%, no maximum and no speed rule.

## Code

The [publisher’s supplementary software](../sources.md#olafsdottir-software) contains four MATLAB posterior/line-fitting helpers, including `decode_calcPosterior` and `lineTraj_decode`. Temporal/spatial bin sizes are arguments. No detector or shuffle driver is supplied.

## Analysis and interpretation

Exhaustive line-fit search over velocity and intercept

Methods: 10 ms replay bins, 100 independent place-field rotations and p<0.2 for the main analysis; strongest replay subset uses p<0.025. Mean behavioral reconstruction error is 20 cm with 500 ms bins. Fraction 41.7% is derived from 1826/4382 putative events. Nonoverlapping steps are inferred from binning; released fitting functions take bin size as an argument and do not supply a detector caller.

## Uncertainties

Nonoverlapping 10 ms steps are inferred; the published 10 ms bin width is explicit. Released fitting helpers accept bin size as an argument and do not supply the original detector/shuffle caller.

## Package mapping

Packaged primary method: `olafsdottir_2016` in [literature_methods.py](../../../src/ripple_detection/literature_methods.py). Its docstring records implementation choices and assumptions. Simulation checks establish that it runs; they do not establish equivalence to the authors’ original event set.

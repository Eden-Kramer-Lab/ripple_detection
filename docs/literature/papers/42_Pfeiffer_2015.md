# Pfeiffer 2015 — Autoassociative dynamics in the generation of sequences of hippocampal place cells
Source: the extracted text (pdftotext of the Dropbox PDF: the report plus Supplementary Materials; the first column of page 1 belongs to the preceding bumblebee article); title matched (Pfeiffer & Foster, Science 349:180–183)
Trigger: SWR

[Paper](https://doi.org/10.1126/science.aaa9633) · [Field evidence](../evidence.csv) · [Source catalog](../sources.md)

Additional sources: [deep-superficial-code](../sources.md#deep-superficial-code), [mouse-development-code](../sources.md#mouse-development-code), [theta-code](../sources.md#theta-code).

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

## Analysis and interpretation

Exclude events:
steps (<10)
start-to-end distance (<80 cm)

## Uncertainties

Related later lab code cannot prove historical per-session settings. Text/code comparisons need an original caller before a conflicting later default can replace the published value.

## Package mapping

Packaged primary method: `pfeiffer_2015` in [literature_methods.py](../../../src/ripple_detection/literature_methods.py). Its docstring records implementation choices and assumptions. Simulation checks establish that it runs; they do not establish equivalence to the authors’ original event set.

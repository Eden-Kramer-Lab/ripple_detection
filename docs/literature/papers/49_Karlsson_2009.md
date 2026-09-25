# Karlsson 2009 — Awake replay of remote experiences in the hippocampus
Source: the extracted text (pdftotext of the Dropbox PDF, main text + Online Methods + Supplementary Information); title matched (Karlsson MP & Frank LM, Nat Neurosci 12:913–918, doi:10.1038/nn.2344)
Trigger: SWR

[Paper](https://doi.org/10.1038/nn.2344) · [Field evidence](../evidence.csv) · [Source catalog](../sources.md)

Additional sources: [ffphy](../sources.md#ffphy).

## Method as implemented

DETECTION (Online Methods, first page, right column):

- Signal and channels: "SWRs were identified on the basis of peaks in the LFP recorded from one channel from each tetrode in the CA3 and CA1 cell layers." Note CA3 as well as CA1 tetrodes; sorted units are not used for detection. LFP sampling rate not stated in this paper (Carr 2012, same rats/system, says 1.5 kHz).

- Filter / envelope / smoothing: "The raw LFP data were bandpass-filtered between 150–250 Hz, and the SWR envelope was determined using a Hilbert transform. The envelope was smoothed with a Gaussian (4-ms s.d.)." Filter type not stated.

- Threshold, minimum duration, combination: "We initially identified SWR events as sets of times when the smoothed envelope stayed above 3 s.d. of the mean for at least 15 ms on at least one tetrode." So: per-tetrode z-scoring, per-tetrode threshold, ANY tetrode triggers. The 15 ms applies to the run above 3 s.d. (the "threshold crossing event"), not to the whole event.

- Normalization period: "3 s.d. of the mean": the period over which mean and s.d. are computed is not stated.

- Bounds: "We defined the entire SWR as including times immediately before and after that threshold crossing event during which the envelope exceeded the mean."

- Merging across channels: "Overlapping SWRs were combined across tetrodes, so many events extended beyond the SWR seen on a single tetrode."

- Maximum duration: not stated. Merging/dropping of close (non-overlapping) events: not stated.

- Speed: main text p. 914: "To avoid confusing replay events and sequential firing during movement-related phase precession, we examined SWRs that occurred when rats were moving less than 2 cm s−1." Supplementary Information: "we used a 2 cm / second maximum velocity cutoff for all main analyses to exclude periods associated with movement". Which sample(s) of the event the speed is tested at: not stated. The phrasing ("we examined", "for all main analyses") reads as an analysis restriction rather than part of the detector; the paper does not say which.

- Brain state: none for detection.

ANALYSIS restrictions (not detection):

- Candidate replay events (main text p. 914; Online Methods): "Candidate replay events were defined as SWRs during which at least five place cells from the replayed environment fired at least one spike each." Place cell = peak rate >= 3 Hz outside SWRs.

- Awake vs quiescent split by immobility time (main text p. 914): awake = "times when rats had been immobile no more than 5 s"; quiescent = "immobile for 5 s or more".

- Theta/delta ratio (Online Methods; Supplementary): theta 6–12 Hz and delta 0.5–4 Hz Hilbert envelopes, 1-s s.d. Gaussian, ratio on the CA3 tetrode with highest variance; used only to describe the state at SWR times (Supplementary Fig. 15), not to select SWRs.

- Supplementary "Lack of remote replay outside of SWRs" control uses a 2 s.d. SWR threshold to exclude spikes; not the main detector.

## Inherited from

- Online Methods: "A distinct set of analyses of the data used in this study and the associated methods have been presented previously [20]" = Karlsson & Frank 2008, J Neurosci 28:14271 (Dropbox PDF, converted to a local copy). It gives the same detector verbatim, adding the kernel width: "The envelope was smoothed with a Gaussian with a SD of 4 ms and a width of 32 ms. SWRs were defined as contiguous periods when the smoothed SWR envelope stayed above 3 SDs of the mean for at least 15 ms on at least one tetrode." (p. 14273–14274). A 32 ms width at 4 ms SD is a truncation at +/-4 SD.

- The package's own `Karlsson_ripple_detector` cites this paper as its source (src/ripple_detection/detectors/_lfp.py, docstring references).

## Code

No code link in the paper (analysis in custom Matlab). A later Frank-lab code snapshot was checked below.

### Related code

- **Lab convention.** [droumis/FFPhy @fce2048](https://github.com/droumis/FFPhy/tree/fce2048), `Functions/getripples.m` lines 29 and 116: with no cell filter it uses every tetrode with cells (CA1 and CA3) and counts an event on any one. It cannot be dated to 2009, and no code by Karlsson was found. CRCNS hc-6 lists no ripple files.

## Uncertainties

FFPhy is supporting lab code; it cannot establish the exact 2009 per-session options.

## Package mapping

Packaged primary method: `karlsson_2009` in [literature_methods.py](../../../src/ripple_detection/literature_methods.py). Its docstring records implementation choices and assumptions. Simulation checks establish that it runs; they do not establish equivalence to the authors’ original event set.

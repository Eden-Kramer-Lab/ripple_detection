# Wikenheiser 2013 — The balance of forward and backward hippocampal sequences shifts across behavioral states
Source: the extracted text (pdftotext of the Dropbox PDF, the NIH author manuscript, PMC3774294); title matched (Wikenheiser & Redish, Hippocampus 23:22–29). Page numbers below are the manuscript's.
Trigger: SWR (ripple-power threshold crossings, each expanded to a fixed 150 ms window)

[Paper](https://doi.org/10.1002/hipo.22049) · [Field evidence](../evidence.csv) · [Source catalog](../sources.md)

Additional sources: [jackson-2006](../sources.md#jackson-2006).

## Method as implemented

Detection (Methods, "Bayesian decoding of ripple events and theta cycles", p. 4):

- Signal, filter, power: "To detect ripple events, the LFP recorded from the pyramidal cell layer was band-pass filtered at ripple frequency (140–220 hz) and the power in this band was estimated via the Hilbert transform." Number of channels and whether channels were combined: not stated (the singular "the LFP" suggests one channel; inference). Whether "power" is the Hilbert amplitude or its square: not stated. Smoothing: not stated. Filter type: not stated. 12-tetrode hyperdrives, CA1 (p. 3).

- Threshold and window: "As in previous studies (e.g. Karlsson and Frank 2009), candidate ripple events were defined as 150 ms windows centered on times when ripple power exceeded a threshold (one standard deviation the baseline value)." The word "above" appears to be missing; the "baseline" (period for mean and SD) is not defined. Whether the window is centred on each supra-threshold sample, on each upward crossing, or on the peak is not stated.

- Rationale: "We chose a lower ripple threshold than some previous studies to err on the side of decoding many putative events and used a bootstrapping procedure ... to assess the significance of each event's sequence content."

- Merging: "Overlapping events were concatenated."

- Cells: "Only events in which at least 3 neurons fired a total of at least 5 spikes were included for analysis."

- Sleep (pre- and post-run rest): "To isolate times of sleep during pre- and post- periods, only events surrounded by at least 30 s of little or no motion (movement speed < 2 cm/s) and mean theta-delta ratio < 0 were included." Epoch definition: "Data were restricted to times when the theta-delta ratio was below 0 and movement speed was < 2 cm/s" (p. 3).

- Awake (run-LIA): "To ensure that theta sequences were not misclassified as ripple events within the run-LIA epoch, candidate events were included only if the rat's average speed during the event was < 2 cm/s and the average theta-delta ratio during the event was < 0." Epoch: "The run-LIA epoch was defined as times when the z-scored ratio of theta to delta (2–4 Hz) oscillatory power (Csicsvari et al., 1999; Jackson et al., 2006) fell below 0. The run-theta epoch consisted of times when the z-scored theta-delta ratio exceeded 0.5" (p. 3). Theta band "(6–10 Hz)" (Abstract).

- Minimum and maximum duration: none beyond the fixed 150 ms window (merged windows can be longer).

- Result: "Of 24,240 candidate ripples (across all sessions), 6796 (28%) were deemed significant" (p. 5).

Later analysis restrictions (not detection): decoding in 10 ms windows, "only time steps containing at least one spike were decoded"; significance by bootstrap on the cumulative sum of decoded-position differences. Theta-cycle sequences (fissure LFP; theta-delta ratio > 0.5 SD; ≥ 3 cells with ≥ 5 spikes) are a separate analysis, not SWR events.

## Inherited from

- "As in previous studies (e.g. Karlsson and Frank 2009)" is attached to the ripple-power thresholding. Karlsson & Frank 2009 used a 3 SD threshold on a 4 ms-smoothed Hilbert envelope with bounds at the mean (see [paper note](49_Karlsson_2009.md)); it has no 150 ms window. So only the general approach is inherited; the window and the 1 SD threshold are this paper's.

- Theta-delta ratio: Jackson, Johnson & Redish 2006 (J Neurosci 26:12415; PMC6674885, primary Methods inspected): theta (6–10 Hz) and delta (2–4 Hz) Hilbert amplitudes "averaged across traces", then non-theta periods from the log-transformed theta/delta ratio relative to the session mean. Jackson 2006 detected SWRs differently (100–250 Hz, log amplitude > 2.5σ, crossings < 20 ms removed, events concatenated if < 100 ms apart); Wikenheiser does not cite it for ripple detection.

- Preprocessing cites Jackson et al. (2006) and Wikenheiser and Redish (2011). The [Jackson Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC6674885/) supply the direct source; they do not establish Wikenheiser-specific runtime settings.

## Code

No code link.

### Related code

- Searched GitHub (code, repositories, and the lab's and authors' accounts), Zenodo, Figshare and CRCNS/DANDI: no original detector caller established in these searches. The Redish lab's public code is MClust; `awikenheiser` and `kkeus/Wikenheiser-Lab-Group` hold Open Ephys and tsd tools, with no ripple windowing.

## Analysis and interpretation

Only time steps with at least 1 spike included

## Reproduction limits

The method-specific qualifications above apply. The executable recipe documents its assumptions about unstated parameters, state scoring and simulation stand-ins.

## Package mapping

Executable example: `wikenheiser_2013` in [literature_recipes.py](../../../examples/literature_recipes.py). Its docstring records implementation choices and assumptions. Simulation checks establish that it runs; they do not establish equivalence to the authors’ original event set.

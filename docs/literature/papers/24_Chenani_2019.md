# Chenani 2019 — Hippocampal CA1 replay becomes less prominent but more rigid without inputs from medial entorhinal cortex
Source: the extracted text (pdftotext of the Dropbox PDF, Nature Communications 10:1341). The layout is two-column, so I read the columns separately. Title verified: yes
Trigger: MUA (place-cell population bursts of sorted units)

[Paper](https://doi.org/10.1038/s41467-019-09280-0) · [Field evidence](../evidence.csv) · [Source catalog](../sources.md)

Additional sources: [chenani-code](../sources.md#chenani-code).

## Method as implemented

Detection criteria (place-cell bursts, the events used for sequence and replay analysis):

- Units: sorted place cells only, identified by a two-way ANOVA on position × running direction. "Place-selective firing rates along with the direction of each run (leftward/rightward) were then put under a two-way ANOVA test in order to find spatially modulated and directional selective units. Those were identified as place cells." (Methods "Place cell identification and burst rates", p. 11)

- Rate and smoothing: "Place cell rates were calculated as the total number of spikes of all place cells per time bin of 1 ms convolved with a Gaussian kernel (σ = 30 ms)." (p. 11)

- Two-threshold rule: "A pace [sic] cell burst was defined as the time span while the place cell rate remained at least one standard deviation (SD) above its average during a period when the peak firing rate reached at least three SD." (p. 11). The period for the average and SD is not stated ("its average").

- Participation: "Only place cell bursts with five or more active cells were considered for the analysis." (p. 11). Main text: "place cell bursts (consisting of at least five place cells during periods when the population activity exceeded three standard deviations over the mean) during PRE, RUN, and POST epochs" (Results, p. 4).

- Periods, stated for RATES (and, by the definition of rest periods, for which RUN bursts count; the latter is inferred): "Burst rates were calculated as number of place cell bursts divided by immobility time in an epoch (PRE, RUN, and POST). For RUN epochs only the time that the animal spent in the reward zone was taken into account. Reward zones were defined individually for each session based on visual judgment of the spatial distribution of running speeds." (p. 11). Also "Rest periods during PRE and POST epochs were defined as periods while the animal was placed in a Plexiglas enclosure in a familiar room. Rest periods during RUN epochs were defined when the animal was at the reward site at either end of the track." (General data organization, p. 11)

- No speed threshold, duration limit, merging, theta or sleep-stage criterion is stated for bursts.

- Epoch-level analysis restriction: "Epochs with 20 or less place cell bursts were excluded from further analysis." (p. 11)

Secondary HFE detection (Fig. 5, SWR vs fast-gamma bursts), a separate event type:

- Channels: "In each session the least noisy and most stable channels were selected for further analysis. Successively, the selected LFP signal were whitened using a second order autoregressive (AR, 2) model" (LFP analysis and HFEs, p. 11). How many channels, and how they were combined, is not stated. The LFP is "one channel of each tetrode ... 1–450 Hz" (p. 10).

- "Candidate events were detected using a threshold on the absolute value of Hilbert transform (smoothed using a Gaussian kernel with σ = 12 ms) of the band passed (100–250 Hz) LFP. Peaks higher than three SD were recorded as candidate events with event duration defined as times where the absolute value of the Hilbert transform remained above one SD." (p. 11)

- Then classification: "Power spectra of the whitened signals of all candidate HFE events were determined using multitaper method ... Power vectors were projected to PC space and clustering was performed on the first two PCs using different clustering algorithms ... (MiniBatchKMeans, SpectralClustering, Ward, Birch) ... we accept the most stable partitioning" (p. 11), into SWR (150–200 Hz peak) and FGB clusters (Results, p. 7).

## Inherited from

Nothing is deferred for the burst detection. The SSI (rank-order) method cites refs 10 and 13 (sequence analysis, not detection).

## Code

https://github.com/cleibold/ReactivationCode ("The custom MATLAB core routines for sequence analysis", p. 11). I opened it: it contains only rank-order / SSI scripts (testsession.m, rankseq.m, checktempseq.m, ...). There is NO burst or HFE detection code, so nothing further can be resolved from it.

## Reproduction limits

The method-specific qualifications above apply. The executable recipe documents its assumptions about unstated parameters, state scoring and simulation stand-ins.

## Package mapping

Executable example: `chenani_2019` in [literature_recipes.py](../../../examples/literature_recipes.py). Its docstring records implementation choices and assumptions. Simulation checks establish that it runs; they do not establish equivalence to the authors’ original event set.

# Bhattarai 2020 — Distinct effects of reward and navigation history on hippocampal forward and reverse replays

Sources: published main text from the matched Dropbox/Zotero PDF; published SI
Appendix `pnas.1912533117.sapp.pdf`, supplied locally by the user and independently
read on September 25, 2026. The 22-page supplement cover gives the exact title,
authors (Bhattarai, Lee and Jung) and DOI 10.1073/pnas.1912533117. Methods pp. 2–7
were read, and the image of Table S1 on p. 21 was inspected directly.

The supplement fingerprint is recorded in the source catalog.

Trigger: SWR+MUA (silence-bounded place-cell bursts that must coincide with a detected SWR; the SWRs themselves require ≥ 5 place cells)

[Paper](https://doi.org/10.1073/pnas.1912533117) · [Field evidence](../evidence.csv) · [Source catalog](../sources.md)

Additional sources: [bhattarai-supplement](../sources.md#bhattarai-supplement), [bhattarai-data](../sources.md#bhattarai-data), [jackson-2006](../sources.md#jackson-2006).

## Method as implemented

SWR detection (SI Appendix, Materials and Methods, "SWR events", SI p. 4):

- "LFP signals obtained from one channel each from two tetrodes carrying well-isolated unit signals were filtered between 100 and 250 Hz. Instantaneous LFP power was calculated at each time point (sampling frequency, 2000 Hz), smoothed with a boxcar filter (width, 50 ms) and averaged." How "instantaneous power" was computed (squared Hilbert amplitude, squared signal) is not stated; the averaging is across the two channels (inference from the sentence). 15 tetrodes in CA1 on one hemisphere per animal (SI p. 3).

- Threshold and bounds: "Points of LFP power exceeding three SD above the mean were identified as candidate SWR events. The beginning and end of each candidate SWR event were determined at one SD above the mean." The period for mean and SD is not stated (inference: whole session).

- Duration and merging: "Candidate SWR events with ≤ 20 ms durations were excluded and two events with a ≤ 100 ms inter-ripple interval were merged (2)." Ref. 2 = Jackson, Johnson & Redish 2006. Whether the interval is end-to-start or peak-to-peak is not stated.

- Cells: "to further eliminate false positive events, we included those candidate SWR events coincident with spikes of at least five different place cells (from any block of a given session) in the analysis (3)." Ref. 3 = Grosmark & Buzsáki 2016.

- Maximum duration: not stated. Speed: not stated for SWR detection. Theta: a theta-power measure exists ("filtered between 4 and 10 Hz. Instantaneous theta power was calculated at each time point and z-scored", SI p. 4) but is not used as a detection criterion in the text I found.

Replay candidates (SI "Replay events", SI p. 4; main text p. 692):

- "A candidate replay event consisted of five or more place cells from a given block (determined using forced-choice and correct free-choice trials) activated together within a 300-ms window with at least one coincident SWR event and preceded by a > 60 ms period of silence (4)." Ref. 4 = Diba & Buzsáki 2007. Main text: "We identified a candidate replay event as a set of spikes emitted by at least 5 different place cells from a given block preceded by a >60-ms silent period and with at least one coincident SWR".

- Whose silence (block place cells, all place cells, all units) is not stated. How the event end is set is not stated: the reported "mean (±SD) duration was 151.9 ± 61.5 ms" (main text) implies the event is not the full 300 ms window (inference: first to last spike within it).

- Speed: none stated for detection. Tuning curves exclude "periods of immobility (speed < 4 cm/s)" (SI "Bayesian decoding"), an encoding-model criterion, not detection. The main text notes "The majority of SWRs were observed in the reward zone" (p. 692).

- 2,573 candidates for the current trajectory; significance by circular-shift shuffle of decoded positions ("P < 0.05 (n = 1342; 52.2% of candidate replay events"), analysis.

Later analysis restrictions: replay significance uses 1000 circular time shifts of decoded positions and linear-regression R² (p<0.05). Weighted correlation measures replay strength/direction. Replay bins are 20 ms with 10 ms steps; 50 ms bins are for behavioral reconstruction. Session sets with median absolute reconstruction error >20 cm are excluded (SI pp. 5–6).

## Inherited from

- Diba & Buzsáki 2007 (ref. 4; [Diba 2007](51_Diba_2007.md), Supplementary Methods): "When the animal's speed was ≤ 10 cm s–1 ... pre-play/replay events were detected by searching for silent periods ≥ 60 ms. If ≥ 5 or ≥ 30 percent of place-cells (whichever was greater) from a template fired within the next 300 ms, an event was recorded." Bhattarai keeps the 60 ms silence, 300 ms window and 5 cells, drops the 30% alternative and does not state Diba's speed ≤ 10 cm/s, and adds the coincident-SWR requirement.

- Jackson et al. 2006 (ref. 2, for the ≤ 100 ms merge; [primary Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC6674885/), primary Methods inspected): "Threshold crossings <20 ms were removed; the remaining events were concatenated if <100 ms apart." Bhattarai uses the same numbers with ≤.

- Grosmark & Buzsáki 2016 (ref. 3, for the ≥ 5 cells; [Grosmark 2016](36_Grosmark_2016.md)): events "in which at least five distinct pyramidal cells each fired at least one spike".

## Code

No code link. Data: "The raw data is deposited in Figshare with identifier doi.org/10.6084/m9.figshare.10032866.v2" (main text).

### Related code

- **Own data.** [Figshare 10.6084/m9.figshare.10032866.v2](https://doi.org/10.6084/m9.figshare.10032866.v2), DataSet.zip: behaviour, events (trial timing only, per MetaData.docx), one tetrode's LFP, spikes and video tracking. No deposited detector code or documented SWR/replay event table was established from the archive inventory and metadata. The public-code searches did not establish an original detector repository; this is a bounded search result.

## Analysis and interpretation

Published SI Appendix pp. 5-6: replay decoding uses 20 ms windows with a 10 ms step; behavioral reconstruction uses 50 ms windows. Templates use 2 cm spatial bins, Gaussian SD 6 cm and exclude speed <4 cm/s. Fit decoded posterior-mode positions against time-bin number, then compare R^2 with 1000 circular shifts of the decoded-position sequence; p is the fraction of shuffled R^2 values exceeding the observed value, significant at p<0.05. Weighted correlation measures strength/direction and includes only posterior bins >0.01. Exclude session sets with median absolute reconstruction error >20 cm (3 of 28); analyzed-session error is 6.56 +/- 0.46 cm (mean +/- SEM). Table S1 (p. 21) confirms 1342 replays / 2573 candidates = 52.2% rounded.

The 25 session-set errors in supplementary Table S1 (p. 21) average 6.5636 cm. The sample-SD SEM from the rounded entries is 0.4664 cm and the population-SD SEM is 0.4570 cm. The published 0.46 cm SEM does not specify SD normalization. Candidate and replay counts sum independently to 2573 and 1342.

## Uncertainties

The published SI does not define the instantaneous-power calculation, SWR normalization period, exact inter-ripple interval, silence population or replay event-end rule. The public archive and code searches do not establish the original detector implementation.

## Package mapping

Packaged primary method: `bhattarai_2020` in [literature_methods.py](../../../src/ripple_detection/literature_methods.py). Its docstring records implementation choices and assumptions. Simulation checks establish that it runs; they do not establish equivalence to the authors’ original event set.

Additional inventories in the same module: `bhattarai_2020_ripples`. See their docstrings for required settings and output stages.

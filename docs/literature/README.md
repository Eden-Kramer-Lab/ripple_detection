# Literature parameters and evidence

The [packaged CSV](../../src/ripple_detection/data/literature_detection_parameters.csv)
is the authoritative parameter table for the 57 surveyed papers. Load it with
`ripple_detection.load_literature_parameters()`. It retains the existing 35-column
schema and paper order.

- [Public dataset catalog](datasets.md) defines the separate packaged
  `literature_datasets.csv`: links, paper/dataset relationships, available
  inputs and event-annotation inspection status. Load it with
  `ripple_detection.load_literature_datasets()` and join by paper DOI.
- [evidence.csv](evidence.csv) records one status and source location for every
  `(doi, column)` pair. It contains no duplicate parameter values.
- [sources.md](sources.md) catalogs primary sources, pinned code versions, archive
  inspection scopes and selected artifact fingerprints.
- [Paper notes](#papers) explain methods, inherited definitions, text/code
  differences, interpretation and remaining uncertainties, with one note per paper.
- [Packaged methods](../../src/ripple_detection/literature_methods.py) own executable
  interpretations and required inputs; the [simulation script](../../examples/literature_recipes.py)
  supplies demonstration data. Simulated checks do not establish historical replication.
- [Implementation guide](implementation.md) explains measured inputs, available
  detector inventories, remaining limitations and verification.

These files describe the current state. Git records corrections and earlier audit
snapshots; separate change ledgers are unnecessary. Source inspection is current
through September 25, 2026. No original author pipeline was rerun.

## Reading a paper's record

Start with its note in the [paper index](#papers), then look up the paper's DOI in
the packaged CSV. Match that DOI and the exact column name in `evidence.csv` to
find the supporting citation and verification status. The DOI identifies the
paper: two Farooq 2019 papers have different journals and DOIs. Numeric filename
prefixes retain the survey's original zero-based order; they are not citation numbers.

For example, Bhattarai 2020 has 20 ms replay windows with 10 ms steps, while its
behavioral reconstruction uses 50 ms windows. Those belong to `Time bin (ms)`,
`Time bin step (ms)` and `Reconst. Error bin (ms)`, respectively. The paper note
explains the distinction, and the evidence index points to the supplement.

The loader is convenient for numeric summaries. To inspect the original text of
every cell, read the CSV with `pandas.read_csv(..., dtype=str, keep_default_na=False)`;
this preserves ranges, qualified values and the literal missing-value markers.

In citations, `PDF` or `combined PDF` page numbers count from the beginning of the
inspected file. Supplement/SI references identify the supplemental document;
printed journal pages are also used in the notes. Section names, figure/table
numbers and quoted passages help locate the evidence across different copies.

Common abbreviations: **LFP**, local field potential; **SWR/SPW-R**, sharp-wave
ripple; **MUA/MU**, multiunit activity; **PBE**, population burst event; **HSE**,
high-synchrony event; **SDE/SDF**, spike-density event/function; **SWS**, slow-wave
sleep; **NREM/REM**, non-rapid/rapid-eye-movement sleep; **SD**, standard deviation;
**SEM**, standard error of the mean; **COM**, center of mass. Event names describe
the authors' definitions and do not guarantee equivalent detection rules.

## Interpretation

A row may combine a primary detector, candidate-inclusion criteria and a control or
secondary detector. Read `Detection Notes` before treating a row as a configuration.
Control values remain in their columns with their role identified. Place-field
construction cutoffs, behavioral reconstruction windows and decoded-content criteria
are distinguished from candidate detection.

A paper's own released implementation sets a value when the relevant execution path
is established; differences from its text remain explicit. Related or later lab code
cannot establish original settings. Unknown callers and session overrides remain
limitations even when a published number is verified.

`#N/A` means no applicable criterion in the summarized method; `Not reported` means
no numerical value was established in the consulted relevant description. Ranges,
fractions, inequalities and assumptions stay as text. The loader converts these to
missing in numeric columns, so scalar summaries do not include them.

Gaussian SD, moving-average width, RMS window and spectral window are different
quantities. Likewise, event-boundary gaps, pooled-spike gaps and peak-to-peak intervals
are distinct. Electrode counts describe sampled/combined channels, not the minimum
number that must cross threshold. Strict inequalities and analysis-specific scopes
are explained in the paper notes.

`Time bin` and `Time bin step` describe the replay or sequence analysis;
`Reconst. Error bin` describes the behavioral reconstruction used to assess error.
`Position bin` and `Place field smooth` describe spatial templates. `% Decoded`
must be read with `% Decoded denom.` and the condition-specific notes. Shuffle
counts and significance thresholds may describe per-event selection, control
analyses or aggregate event-count tests; the notes distinguish those uses.
Despite the legacy column names, a threshold need not be a z-score: raw,
adaptive and multiplicative thresholds are labeled explicitly in their entries.

## Evidence statuses

| Status | Meaning |
|---|---|
| `checked_paper` | Supported by the primary text or figure at the cited location. |
| `checked_code` | Supported by an inspected released implementation or stored configuration; original execution may still have provenance limits. |
| `checked_metadata` | Bibliographic or species metadata checked against the matched source. |
| `derived` | Arithmetic, unit conversion or explicitly approximate figure reading. |
| `inferred` | Requires the stated assumption. |
| `unresolved` | The specific value or required original source is not established. |
| `not_reported` | No value established in the consulted relevant Methods. |
| `not_applicable` | No corresponding criterion/quantity, or an empty contextual field. |
| `reviewed_context` | Narrative reviewed against sources; numerical claims inherit their field statuses. |

To find open numerical questions, filter `evidence.csv` by `unresolved` or `inferred`
and follow `paper_note`. Additional uncertainties about callers, state definitions
and parameters outside the CSV live in each paper note. A negative code search is
bounded evidence, not proof that no code exists.

## Maintaining the survey

Edit parameter values only in the packaged CSV. Review the corresponding DOI/column
citations in `evidence.csv`, update the paper note's interpretation or uncertainties,
and update `sources.md` when source versions or inspection scope change. Keep recipe
assumptions distinct from reported parameters. Historical changes belong in Git.
Dataset links and availability summaries belong in the separate dataset CSV;
its [maintenance rules](datasets.md#maintenance) distinguish unverified contents
from a scoped finding that event annotations were not found.

Run the literature and recipe checks after changes:

```sh
pytest tests/test_literature.py tests/test_literature_methods.py tests/test_literature_recipes.py -q --no-cov
```

The checks cover table loading, README statistics, evidence coverage and links,
fingerprint formatting, and execution/coverage of the example recipes. They do
not re-read publications or verify the scientific meaning of a changed value.

## Papers

| Paper | Journal | Note |
|---|---|---|
| Mallory 2025 | Science | [Methods and evidence](papers/00_Mallory_2025.md) |
| Widloski 2025 | Nature Communications | [Methods and evidence](papers/01_Widloski_2025.md) |
| Yang 2024 | Science | [Methods and evidence](papers/02_Yang_2024.md) |
| Huelin Gorriz 2023 | Nature Communications | [Methods and evidence](papers/03_HuelinGorriz_2023.md) |
| Harvey 2023 | Neuron | [Methods and evidence](papers/04_Harvey_2023.md) |
| Liu 2023 | Science | [Methods and evidence](papers/05_Liu_2023.md) |
| Tirole 2022 | eLife | [Methods and evidence](papers/06_Tirole_2022.md) |
| Bush 2022 | Current Biology | [Methods and evidence](papers/07_Bush_2022.md) |
| Berners-Lee 2022 | Neuron | [Methods and evidence](papers/08_Berners-Lee_2022.md) |
| Widloski 2022 | Neuron | [Methods and evidence](papers/09_Widloski_2022.md) |
| Krause 2022 | Neuron | [Methods and evidence](papers/10_Krause_2022.md) |
| Mou 2022 | Neuron | [Methods and evidence](papers/11_Mou_2022.md) |
| Berners-Lee 2021 | Journal of Neuroscience | [Methods and evidence](papers/12_Berners-Lee_2021.md) |
| Denovellis 2021 | eLife | [Methods and evidence](papers/13_Denovellis_2021.md) |
| Gillespie 2021 | Neuron | [Methods and evidence](papers/14_Gillespie_2021.md) |
| Michon 2021 | Current Biology | [Methods and evidence](papers/15_Michon_2021.md) |
| Igata 2021 | PNAS | [Methods and evidence](papers/16_Igata_2021.md) |
| Gridchyn 2020 | Neuron | [Methods and evidence](papers/17_Gridchyn_2020.md) |
| Kaefer 2020 | Neuron | [Methods and evidence](papers/18_Kaefer_2020.md) |
| Bhattarai 2020 | PNAS | [Methods and evidence](papers/19_Bhattarai_2020.md) |
| Stella 2019 | Neuron | [Methods and evidence](papers/20_Stella_2019.md) |
| Xu 2019 | Neuron | [Methods and evidence](papers/21_Xu_2019.md) |
| Farooq 2019 | Neuron | [Methods and evidence](papers/22_Farooq_2019.md) |
| Farooq 2019 | Science | [Methods and evidence](papers/23_Farooq_2019.md) |
| Chenani 2019 | Nature Communications | [Methods and evidence](papers/24_Chenani_2019.md) |
| Michon 2019 | Current Biology | [Methods and evidence](papers/25_Michon_2019.md) |
| Liu 2019 | Hippocampus | [Methods and evidence](papers/26_Liu_2019.md) |
| Shin 2019 | Neuron | [Methods and evidence](papers/27_Shin_2019.md) |
| Carey 2019 | Nature Neuroscience | [Methods and evidence](papers/28_Carey_2019.md) |
| Muessig 2019 | Current Biology | [Methods and evidence](papers/29_Muessig_2019.md) |
| Drieu 2018 | Science | [Methods and evidence](papers/30_Drieu_2018.md) |
| Maboudi 2018 | eLife | [Methods and evidence](papers/31_Maboudi_2018.md) |
| Ólafsdóttir 2017 | Neuron | [Methods and evidence](papers/32_Olafsdottir_2017.md) |
| Wu 2017 | Nature Neuroscience | [Methods and evidence](papers/33_Wu_2017.md) |
| Yamamoto 2017 | Neuron | [Methods and evidence](papers/34_Yamamoto_2017.md) |
| Tang 2017 | Journal of Neuroscience | [Methods and evidence](papers/35_Tang_2017.md) |
| Grosmark 2016 | Science | [Methods and evidence](papers/36_Grosmark_2016.md) |
| Ambrose 2016 | Neuron | [Methods and evidence](papers/37_Ambrose_2016.md) |
| Jadhav 2016 | Neuron | [Methods and evidence](papers/38_Jadhav_2016.md) |
| Ólafsdóttir 2016 | Nature Neuroscience | [Methods and evidence](papers/39_Olafsdottir_2016.md) |
| Silva 2015 | Nature Neuroscience | [Methods and evidence](papers/40_Silva_2015.md) |
| Ólafsdóttir 2015 | eLife | [Methods and evidence](papers/41_Olafsdottir_2015.md) |
| Pfeiffer 2015 | Science | [Methods and evidence](papers/42_Pfeiffer_2015.md) |
| Wu 2014 | Journal of Neuroscience | [Methods and evidence](papers/43_Wu_2014.md) |
| Wikenheiser 2013 | Hippocampus | [Methods and evidence](papers/44_Wikenheiser_2013.md) |
| Pfeiffer 2013 | Nature | [Methods and evidence](papers/45_Pfeiffer_2013.md) |
| Carr 2012 | Neuron | [Methods and evidence](papers/46_Carr_2012.md) |
| Bendor 2012 | Nature Neuroscience | [Methods and evidence](papers/47_Bendor_2012.md) |
| Gupta 2010 | Neuron | [Methods and evidence](papers/48_Gupta_2010.md) |
| Karlsson 2009 | Nature Neuroscience | [Methods and evidence](papers/49_Karlsson_2009.md) |
| Davidson 2009 | Neuron | [Methods and evidence](papers/50_Davidson_2009.md) |
| Diba 2007 | Nature Neuroscience | [Methods and evidence](papers/51_Diba_2007.md) |
| Ji 2007 | Nature Neuroscience | [Methods and evidence](papers/52_Ji_2007.md) |
| Foster 2006 | Nature | [Methods and evidence](papers/53_Foster_2006.md) |
| Lee 2002 | Neuron | [Methods and evidence](papers/54_Lee_2002.md) |
| Nádasdy 1999 | Journal of Neuroscience | [Methods and evidence](papers/55_Nadasdy_1999.md) |
| Kudrimoti 1999 | Journal of Neuroscience | [Methods and evidence](papers/56_Kudrimoti_1999.md) |

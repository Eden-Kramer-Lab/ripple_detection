# Implementing the surveyed detection methods

The package supports ripple detection on users' recordings and executable
implementations of published detection methods. General detectors and paper-specific
workflows belong in the installed package; examples demonstrate how to supply data
and choose a method. Required inputs, analysis stages and unresolved source choices
must be explicit so users can interpret the resulting events.

Method comparisons should call these same implementations, with the method name,
resolved options and analysis stage recorded alongside each event inventory.
Benchmark conditions, parameter sweeps and reporting can live outside the package.
Future replay methods can consume candidate events and the required neural and
behavioral data, with decoding, sequence scoring and significance testing as
separate steps. Ripple detection alone does not establish replay.

The installed [literature_methods module](../../src/ripple_detection/literature_methods.py)
provides **86 executable inventories**: 57 demonstration defaults and 29 additional
ripple, HFE, MUA or protocol functions, covering 56 of the 57 surveyed papers.
Widloski 2022 defines events by decoding and has no candidate detector to port.
For Widloski 2025 and Kaefer 2020, the runnable defaults provide secondary ripple
labels; for Gupta 2010, only the SWR gate is implemented. These roles appear in
`list_methods()` and result metadata, and their missing sequence definitions are
listed in `NOT_REPRODUCED`. The demonstration groups runnable inventories
regardless of their role in the paper.
Several functions expose further named interpretations through keyword arguments.
An executable inventory is not a claim of historical replication.

The [CSV, evidence and paper notes](README.md) remain the authority for reported
values and their provenance. The module owns executable interpretations; this
page owns usage and implementation limits. A CSV row can combine multiple
protocols or analysis stages, so it is **not** automatically converted into one
detector configuration. Git retains the previous assessment and corrections.

## Use measured recordings

```python
from ripple_detection.literature_methods import Recording, list_methods, run_method

# Raw LFP and spike counts share time (seconds); speed is cm/s.
# Select the intended channels and sorted populations before calling.
recording = Recording.from_arrays(
    time, sampling_frequency,
    lfps=selected_lfps,
    multiunit=spike_counts,
    speed=speed,
    pyramidal=pyramidal_mask,
    place_cells=place_cell_mask,
    sleep_intervals=nrem_intervals,
    baseline_intervals=baseline_intervals,
    artifact_intervals=artifact_intervals,
)
methods = list_methods()
events = run_method("pfeiffer_2013_ripples", recording)
print(events.attrs["doi"], events.attrs["output"])
```

`list_methods()` lists names, DOI, output role, signature-required options, demonstration grouping and
interpretation. `role` describes scientific use; `inventory` distinguishes default
from additional demonstration entries, independently of that role. Read the function's docstring and its paper note before choosing
an entry: a burst candidate, a ripple control and a decoded replay are different
objects. Functions with conditional measured-data requirements describe them in
their docstrings; the signature-only option list cannot express those conditions.

`Recording.from_arrays` copies inputs, validates selections, and masks artifact
intervals. NaN speed remains unknown; omitted speed makes every method whose
result depends on speed raise instead. Missing cell selections are empty;
methods needing those cells raise. Measured data never acquire synthetic sleep
labels or synthetic templates. Pass actual `templates` for Ólafsdóttir 2015,
`example_ripples` for Carey, `external_ripples` for Yang/Grosmark's unresolved
historical ripple gate, and `reference_lfp` for reference-subtracted methods.
For external ripples, three columns give start/end/peak; two columns explicitly
use the midpoint as the peak. Supply true peaks when the rule requires them.
Yang/Grosmark also require eligible quiet-waking/NREM `behavior_intervals`,
separate from the NREM normalization baseline. Chenani (reward zones),
Ólafsdóttir 2015 (rest) and 2017 (corners), Diba (track-end reward areas) and
Foster (facing-direction epochs) require their eligible `behavior_intervals` too. Transforms are computed afresh
on each call, so a recording can be released after use and changed arrays do
not retain stale filtered results. Simulation fallbacks are limited to explicit
`SimulatedSession` inputs.

Optional `behavior_intervals` retain wholly contained events in both `run_method`
and direct named method calls. The dispatcher does not change normalization.
For example, these intervals can express caller-curated
reward zones, rest sessions, track ends or facing-direction restrictions.
The explicit Farooq Science awake-frame method also masks its trace to these
intervals before estimating normalization statistics; Liu awake frames restrict
their spike input to these intervals. These are method-specific uses.
`baseline_intervals` are consumed only where a method explicitly requests a
caller-selected baseline, including Krause's text HSE interpretation. Whole-session
normalization and a paper's specified stopping or sleep baseline retain their
own epochs. Mallory uses stopped samples in each supplied recording segment;
Wikenheiser defaults to whole-session normalization and requires
`normalization="baseline"` to use supplied baseline intervals. Its original
baseline epoch is unspecified. The shared population and ripple helpers do not
override these choices.

Spikes are dense counts on the supplied timestamp grid. `population_trace` bins
these observations onto the method's native grid **before** smoothing. It cannot
recover spike-time precision absent from the input. Sparse spike-time input,
independent LFP/spike clocks and streaming acquisition are not yet supported.
Single-channel rules use the first selected LFP channel. Automatic channel
ranking, cell classification and tetrode membership must be supplied upstream.

`run_method` and direct named methods always return a DataFrame with start/end times and elapsed duration,
retaining available peak, channel, trigger and clipping columns. Its `attrs`
record the method, DOI, output role, interpretation, resolved options and input
sampling rate. Gridchyn also records threshold updates. Array-returning internal
compositions do not retain every diagnostic column of the underlying detector.
Preserve attrs explicitly if exporting to a format such as CSV that drops them.

The [simulation script](../../examples/literature_recipes.py) supplies synthetic
cell groups, templates, reference and a baseline spanning the initial 0–12 s
rest epoch, including its simulated events (clipped for shorter recordings).
Kaefer alone uses the initial 0–2 s baseline to exercise its FFT detection path;
its published baseline epoch is unspecified. These simulation choices are not
historical settings or a controlled method comparison. Its state fallbacks are explicitly
shortened for demonstration. Its results CSV records `method`, `configuration`, DOI,
role, resolved `options` and `supplied_baseline_intervals` beside each result. The
last two fields are JSON; a supplied baseline does not imply that the method uses
it instead of its own normalization epoch.

The demo produces 59 configurations: the 57 default inventories and two additional
Ólafsdóttir analysis settings. The 2015 `bayesian_candidates` row selects
`minimum_active_units=7`; its default retains the broader per-template inventory.
The 2017 `trajectory` row uses `analysis="trajectory"`; its default remains the
arm-reactivation inventory. These rows apply candidate-selection rules only;
decoding and replay significance are not implemented by selecting them.
`false_positives` counts detected events with no overlap with a simulated ripple.
For population-event inventories this alone does not establish a detection error.
Other inventories are exercised with their required settings in the tests.

## Implemented distinctions

- Filtering, Hilbert envelopes and recipe transforms operate inside valid blocks.
  Missing values and timestamp gaps remain boundaries through subsequent merges.
  Fixed-window silence events are separately selectable from last-spike events;
  Diba uses the fixed policy.
- Native population grids replace the sliding-count substitutes for Mou,
  Wu 2014/2017, Ji, Michon, Krause and the shared population recipes. Berners-Lee
  and Maboudi use their documented finite kernels. Sample timestamps remain
  quantized to the caller's input clock. Native-grid events are reported at the
  first and last recorded samples of their first and last bins: closed bounds
  holding exactly the samples those bins counted, so participation counts see no
  spike from a neighboring bin. Bins holding no sample (bins narrower than the
  sample spacing, or uneven timestamps) are skipped, so bounds are always
  recorded timestamps. Duration limits count bins (n bins last n bin
  widths) and close-event gaps are measured edge to edge; later duration rules on
  these bounds count samples inclusively. Interval restrictions (curated sleep
  or rest) keep only bins whose counted samples all lie inside an interval, and
  containment in behavior intervals allows the clock's rounding error. Tirole
  applies its released rules to bin times and reports the bins' samples.
- Tirole uses the finite forward/backward kernel, threshold-anchor grouping,
  inclusive crossing samples, fallback bounds, sampled-speed rule and a
  reconstructed ripple preprocessing path. Polyphase LFP resampling and modern
  missing-data behavior can differ from the original acquisition pipeline.
- Mallory retains the larger peak for subsequent peak-distance merges, including
  the released tie rule. MUA and ripple candidates are separate; decoded-replay
  cell restrictions are downstream. Supply one normalization segment per call.
- Gridchyn uses trailing counts, refractory triggers and feedback updates. The
  implementation follows the published rate rule, using the actual number of
  triggers per update interval. It does not reproduce the apparent extra-count
  offset in the C++ release, or hardware timing. Offline bounds accompany the
  causal trigger times. Its feedback clock counts observed samples and pauses
  across data gaps, an explicit offline policy.
- Kaefer uses reference subtraction, FFT chunks and the reported stride; FFT
  taper/anchor details are explicit reconstruction choices. Nádasdy uses summed
  RMS instead of a Hilbert-envelope substitute.
- Denovellis follows the historical filter and squared-filtered-LFP trace.
  Harvey's default code path uses DetectSWR with a radiatum channel; the
  no-radiatum FindRipples branch and published difference-of-Gaussians path
  are separately named. `stage="decoding_candidates"` applies the released
  >=80 ms, >=5 place cells and <50% empty 20 ms bin gates. Its complete
  multi-dataset curation and EMG veto pipeline is not reproduced.
- Krause preserves the released trimming convention: bins end strictly before
  the SWR end, and the trimmed end retains the unbinned remainder. Moving the
  SWR end can therefore change the trimmed bounds with unchanged spikes.
- All 18 previously omitted secondary ripple/HFE branches have entry points.
  Davidson/Wu retain distinct local peaks; Ji merges low-threshold excursions
  before its high-peak gate; Lee/Foster use rectified-LFP rules. Igata and Chenani
  return per-channel candidates where aggregation/classification is unresolved.
- Detection and decoding-candidate participation are selectable stages for
  Harvey, Mou, Shin, Grosmark, Jadhav, Carr, Wu 2017, Drieu and Ji. Their default
  `stage="detection"` returns the initial inventory; `"decoding_candidates"`
  applies the documented analysis filters. Ólafsdóttir retains its explicit
  arm/trajectory and minimum-cell options.
- Muessig uses caller-supplied eligible rest intervals for whole-event containment.
  The published state rule uses mean speed and theta/delta power in 1.6 s windows
  stepped by 0.8 s: <2.5 cm/s for rest trials and <1 cm/s for RUN. Curated intervals
  must implement the chosen trial's criteria; the package does not rescore them.
  `sample_speed_veto=True` adds the previous, stricter veto on native-grid speed
  samples. That extra veto is optional and is not specified by the paper.
- Speed limits keep each source's inequality: a stated "less than" (Karlsson,
  Carr, Jadhav, Tang, Gillespie, Davidson, Wu 2014, Silva, Pfeiffer 2013,
  Ambrose) is strict; Shin's "<=4 cm/s" and limits whose inequality is unstated
  are inclusive, as each docstring says.
- Farooq and Grosmark interpret the ambiguous 15 ms Gaussian width as SD.
  Gupta defaults to Jackson's log of mean Hilbert amplitude, an inheritance
  inference that can be disabled with `log_amplitude=False`.
- Pfeiffer 2013 secondary ripples restrict both detection and normalization to
  speed <5 cm/s. Bush's secondary FIR uses forward/backward filtering (effective
  order 800), an explicit choice where the paper only specifies order 400.
  Stella uses symmetric odd-length wavelet support centered on zero.
- Additional population and protocol inventories include Widloski's burst labels,
  Krause/Denovellis/Gillespie controls, Maboudi open field, separate Bhattarai and
  Muessig ripples, Farooq Science awake frames and Liu awake/ripple-associated
  frames. Muessig RUN and Wikenheiser run-LIA are selectable method options.

## Remaining choices and verification

Several functions require settings that were not established by the source audit:
RMS windows/bounds, wavelet settings, power definitions, AR coefficients, and
normalization epochs. Those requirements make a selected interpretation runnable;
they do not turn a caller's choice into a published value. Huelin Gorriz's
published-cap and related-code options both remain reconstructions because its
original extractor is absent. Mou exposes both plausible scaling interpretations.

Some methods still use shared equiripple filters, generic Gaussian boundary
handling and the package's inclusive sample-count convention where exact source
behavior is unknown or has not been ported. The paper notes and function
docstrings identify important known differences. This package does not claim
sample-for-sample equivalence for every listed method.

Work beyond these implementations includes Igata's under-specified population
classification, Chenani's multitaper/PCA stable-partition selection, automatic
sleep/EMG/manual artifact curation, hardware closed-loop pipelines, and replay
scoring/decoding. For Widloski, Kaefer and Gupta, the implemented labels or gates
must not be mistaken for their decoded sequence definitions.

Validation includes all default inventories on simulation, all added inventory
functions with explicit inputs, and distinguishing cases for native bins, finite
kernels, retained-peak merging, threshold bounds, adaptive updates, reference
subtraction, fixed windows, templates and artifact gaps with positive controls.
Regression cases cover duration/bin boundaries at Unix timestamps, baseline
selection, stage filters, direct-call parity, input errors and recording lifetime.
The shared detector suite also remains in use. Simulated recall is a smoke check, not scientific validation.

**Outstanding verification:** no complete author pipeline has been executed on
matching original recordings. Historical equivalence would require comparing
intermediate traces, normalization statistics, candidate boundaries and final
inventories against versioned reference outputs. Event counts alone would not
establish equivalence. The published-value evidence gaps in the paper notes remain
unchanged by adding these implementations.

## Per-paper entry points

This table is navigation, not a second parameter table. Use `list_methods()` for
the executable inventory and function docstrings for required inputs/options.
The linked note contains the source evidence and qualifications for each paper.

| Paper | Available entry points |
|---|---|
| [Mallory 2025](papers/00_Mallory_2025.md) | `mallory_2025`, `mallory_2025_ripples` |
| [Widloski 2025](papers/01_Widloski_2025.md) | `widloski_2025`, `widloski_2025_bursts` |
| [Yang 2024](papers/02_Yang_2024.md) | `yang_2024` |
| [Huelin Gorriz 2023](papers/03_HuelinGorriz_2023.md) | `huelin_gorriz_2023` |
| [Harvey 2023](papers/04_Harvey_2023.md) | `harvey_2023_code`, `harvey_2023_text`, `harvey_2023_no_radiatum` |
| [Liu 2023](papers/05_Liu_2023.md) | `liu_2023` |
| [Tirole 2022](papers/06_Tirole_2022.md) | `tirole_2022` |
| [Bush 2022](papers/07_Bush_2022.md) | `bush_2022`, `bush_2022_ripples` |
| [Berners-Lee 2022](papers/08_Berners-Lee_2022.md) | `berners_lee_2022` |
| [Widloski 2022](papers/09_Widloski_2022.md) | No candidate detector: events defined by decoding |
| [Krause 2022](papers/10_Krause_2022.md) | `krause_2022`, `krause_2022_hse` |
| [Mou 2022](papers/11_Mou_2022.md) | `mou_2022` |
| [Berners-Lee 2021](papers/12_Berners-Lee_2021.md) | `berners_lee_2021` |
| [Denovellis 2021](papers/13_Denovellis_2021.md) | `denovellis_2021`, `denovellis_2021_mua` |
| [Gillespie 2021](papers/14_Gillespie_2021.md) | `gillespie_2021`, `gillespie_2021_mua` |
| [Michon 2021](papers/15_Michon_2021.md) | `michon_2021` |
| [Igata 2021](papers/16_Igata_2021.md) | `igata_2021`, `igata_2021_ripples` |
| [Gridchyn 2020](papers/17_Gridchyn_2020.md) | `gridchyn_2020`, `gridchyn_2020_ripples` |
| [Kaefer 2020](papers/18_Kaefer_2020.md) | `kaefer_2020` |
| [Bhattarai 2020](papers/19_Bhattarai_2020.md) | `bhattarai_2020`, `bhattarai_2020_ripples` |
| [Stella 2019](papers/20_Stella_2019.md) | `stella_2019` |
| [Xu 2019](papers/21_Xu_2019.md) | `xu_2019`, `xu_2019_ripples` |
| [Farooq 2019](papers/22_Farooq_2019.md) | `farooq_2019_neuron`, `farooq_2019_neuron_ripples` |
| [Farooq 2019](papers/23_Farooq_2019.md) | `farooq_2019_science`, `farooq_2019_science_ripples`, `farooq_2019_science_awake` |
| [Chenani 2019](papers/24_Chenani_2019.md) | `chenani_2019`, `chenani_2019_hfe` |
| [Michon 2019](papers/25_Michon_2019.md) | `michon_2019` |
| [Liu 2019](papers/26_Liu_2019.md) | `liu_2019`, `liu_2019_ripples`, `liu_2019_awake`, `liu_2019_ripple_frames` |
| [Shin 2019](papers/27_Shin_2019.md) | `shin_2019` |
| [Carey 2019](papers/28_Carey_2019.md) | `carey_2019` |
| [Muessig 2019](papers/29_Muessig_2019.md) | `muessig_2019`, `muessig_2019_ripples` |
| [Drieu 2018](papers/30_Drieu_2018.md) | `drieu_2018`, `drieu_2018_ripples` |
| [Maboudi 2018](papers/31_Maboudi_2018.md) | `maboudi_2018`, `maboudi_2018_open_field` |
| [Ólafsdóttir 2017](papers/32_Olafsdottir_2017.md) | `olafsdottir_2017`, `olafsdottir_2017_ripples` |
| [Wu 2017](papers/33_Wu_2017.md) | `wu_2017` |
| [Yamamoto 2017](papers/34_Yamamoto_2017.md) | `yamamoto_2017` |
| [Tang 2017](papers/35_Tang_2017.md) | `tang_2017` |
| [Grosmark 2016](papers/36_Grosmark_2016.md) | `grosmark_2016` |
| [Ambrose 2016](papers/37_Ambrose_2016.md) | `ambrose_2016` |
| [Jadhav 2016](papers/38_Jadhav_2016.md) | `jadhav_2016` |
| [Ólafsdóttir 2016](papers/39_Olafsdottir_2016.md) | `olafsdottir_2016` |
| [Silva 2015](papers/40_Silva_2015.md) | `silva_2015` |
| [Ólafsdóttir 2015](papers/41_Olafsdottir_2015.md) | `olafsdottir_2015` |
| [Pfeiffer 2015](papers/42_Pfeiffer_2015.md) | `pfeiffer_2015` |
| [Wu 2014](papers/43_Wu_2014.md) | `wu_2014`, `wu_2014_ripples` |
| [Wikenheiser 2013](papers/44_Wikenheiser_2013.md) | `wikenheiser_2013` |
| [Pfeiffer 2013](papers/45_Pfeiffer_2013.md) | `pfeiffer_2013`, `pfeiffer_2013_ripples` |
| [Carr 2012](papers/46_Carr_2012.md) | `carr_2012` |
| [Bendor 2012](papers/47_Bendor_2012.md) | `bendor_2012` |
| [Gupta 2010](papers/48_Gupta_2010.md) | `gupta_2010` |
| [Karlsson 2009](papers/49_Karlsson_2009.md) | `karlsson_2009` |
| [Davidson 2009](papers/50_Davidson_2009.md) | `davidson_2009`, `davidson_2009_ripples` |
| [Diba 2007](papers/51_Diba_2007.md) | `diba_2007`, `diba_2007_ripples` |
| [Ji 2007](papers/52_Ji_2007.md) | `ji_2007`, `ji_2007_ripples` |
| [Foster 2006](papers/53_Foster_2006.md) | `foster_2006`, `foster_2006_ripples` |
| [Lee 2002](papers/54_Lee_2002.md) | `lee_2002`, `lee_2002_ripples` |
| [Nádasdy 1999](papers/55_Nadasdy_1999.md) | `nadasdy_1999` |
| [Kudrimoti 1999](papers/56_Kudrimoti_1999.md) | `kudrimoti_1999` |

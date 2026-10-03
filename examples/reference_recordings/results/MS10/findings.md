# MS10: released `bz_FindRipples` events against the package

Session `PetersenP/MS10/Peter_MS10_170307_154746_concat` (Buzsáki lab databank). The released
events are from the Wayback capture `20231129113529` of `*.ripples.events.mat` (11305 B,
complete, sha256 `d28e4f4d…`): 429 events, channel 46, thresholds [2 5] SD, durations
[50 150] ms, passband [120 180] Hz, 1250 Hz, `restrict` empty, `stdev` 202112.11. The LFP is
DANDI 000059 v0.250624.0444, asset `8941eed3`, `/processing/ecephys/LFP/LFP`
(23613750 × 64 int16).

The released events are another pipeline's output, not ground truth. Every number below comes
from the files in this folder: `inputs.json`, `stage_check.json`, `comparison.csv`,
`attribution.json` and `runtime.json`.

## Result

| Pair (reference vs detected) | n ref | n det | matched | recall | precision | median IoU | identical bounds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| released vs transcription | 429 | 429 | 429 | 1.000 | 1.000 | 1.000 | 429 |
| released vs package | 429 | 549 | 429 | 1.000 | 0.781 | 1.000 | 429 |
| transcription vs package | 429 | 549 | 429 | 1.000 | 0.781 | 1.000 | 429 |

The numbers are the same at minimum IoU 0, 0.2 and 0.5. "Identical bounds" means both bounds
are within half a sample. The remaining differences, at most 3.6e-12 s, come from MATLAB
building timestamps with `0:1/1250:…` where this run divides the sample index by 1250. There
are no splits or merges.

All 429 released events are recovered with their exact bounds. The package also finds 120
events the source does not release. All 120 come from one post-processing rule, explained
below.

## Inputs (inputs.json)

- **Rates:** the `.xml`, the NWB (`starting_time` 0, `rate` 1250) and the stored `frequency`
  all give 1250 Hz.
- **Length:** the `.lfp` head capture declares 3,022,560,000 bytes = 23,613,750 × 64 × 2, which
  equals the DANDI row count. The `.xml` and DANDI both have 64 channels.
- **Head match:** the first 512 rows of all 64 DANDI columns equal the `.lfp` head exactly. The
  column map is the identity, no column is ambiguous, and the stored channel 46 is DANDI
  column 46.
- **Channel:** `session.mat`'s `Ripple` tag is 47. CellExplorer tags are 1-based, and buzcode
  stores the channel 0-based (`bz_GetLFP` loads `channels+1`), so 46 + 1 = 47 agrees. The tag's
  electrode group 3 is `sessionInfo`'s group 3 (0-based channels 37-47), which contains 46.
- **Gaps:** none are possible. The LFP is int16, so it holds no NaN, and the series has a rate
  rather than timestamps. The session concatenates 4 epochs (boundaries 2807.424, 3486.636 and
  5497.416 s, end 18891 s). No released event spans a boundary. The source treated the
  recording as one continuous signal, and so does this run.
- **Events:** all 429 lie within samples 5953 to 23,602,582 of 23,613,750, on the 1250 Hz grid
  to within 3.7e-9 samples (tolerance 3.3e-8). Durations run from 19.2 to 148.8 ms.
- **Speed:** the source has no speed rule, so the package runs with `speed_threshold=np.inf`.

## Code version (inputs.json, `source_code`)

The MAT header dates the file Thu Oct 15 16:03:31 2020. The version of `buzsakilab/buzcode`
current then (eec82c13, 2020-04-14) cannot have written it: it saves `timestamps`/`detectorinfo`
and has `EMGThresh`, `minDuration` and `plotType`. No version on any branch of that repository
combines a `passband` parameter with the stored `times`/`detectorName`/`detectorParams` layout.

The match is `petersenpeter/buzcode` `analysis/lfp/bz_FindRipples.m` at **bc3fc91** (committed
2021-01-14). Its change from 5a0b450 adds the `passband` parameter, makes `saveMat` default to
true and saves `.ripples.events.mat`. That gives exactly the stored parameter set (`basepath,
channel, durations, frequency, passband, restrict, saveMat, show, stdev, thresholds`) and file
name. The file was therefore written by that change before it was committed.

What that version does:

- **Filter:** `bz_FilterLFP` calls `bz_Filter` with its defaults: **cheby2, order 4, 20 dB,
  Nyquist 625**, run through `filtfilt`. It does not use `buzsakilab/buzcode`'s butter order 3.
- **Smoothing:** `Filter0`, an 11-sample centred moving average, zero-padded at both ends.
- **Normalization:** `unity` over all samples, with `std` using N-1.
- **Bounds:** crossings of > 2 SD. `start` is the last sample at or below the threshold and
  `stop` the last above. Incomplete runs at the edges are dropped.
- **Merge:** events merge while the next start minus the current stop is below 62.5 samples,
  with **no cap**.
- **Peak test:** an event is kept if its maximum exceeds 5 SD.
- **Peak:** the trough of the filtered signal.
- **Duration:** events longer than 0.150 s are dropped. There is no minimum, no noise channel
  and no EMG rule.

## Stage check (stage_check.json)

| Quantity | Value |
| --- | --- |
| stored `stdev` | 202112.1139641 |
| recomputed (cheby2, transfer-function form, MATLAB padding) | 202112.1139650 |
| relative difference | 4.5e-12 |
| same design as second-order sections | relative difference -5.0e-11 |
| butter order 3 (`buzsakilab/buzcode`'s filter) instead | **+40.2%** |
| stored `peakNormedPower`, recomputed within each released event | 429 of 429 within 1e-6 relative (maximum 9.9e-12) |

The filter and normalization match the source before any threshold. The code version
matters: the other lineage's filter would change the SD by 40%.

## Differences, by stage

**1. 120 package-only events come from the merge rule (post-processing).** The source merges
runs with no cap and then drops events over 150 ms. The package, like FMAToolbox's current
`FindRipples` (6bbb366, lines 173-188: "unless this would yield too long a ripple"), merges only
while the merged span stays under `maximum_duration`. The numbers in `attribution.json` show
this rule is the whole difference:

- Without the duration ceiling, the transcription has 48 events of 152.0-912.8 ms. The source
  merged these and then dropped them.
- All 120 package-only events lie inside those 48 (`rows_not_inside_one` is empty). The 48 hold
  1, 2, 3, 4, 5 and 7 package events in 5, 26, 10, 4, 2 and 1 cases.
- The package-only events last 22.4-149.6 ms, and each has a peak of at least 5.30 SD. Each is
  a fragment of a long chain that passes the tests on its own.
- Running the package with `maximum_duration=None` removes the merge cap. Dropping its events
  over 188 samples afterwards (the package's own ceiling at 150 ms) gives 477 events before the
  ceiling (429 + 48) and **429 after, identical to the released events** (recall 1, precision
  1, 429 identical bounds).

So the package's thresholds, bounds, merge criterion, peak test and ceiling reproduce this
source exactly; only the cap differs. This is not a package bug: the package implements the
FMAToolbox rule its docstring cites. But every buzcode version read for this task (both forks,
2017-2020) merges without a cap. The docstring names buzcode as a carrier without saying this,
so a buzcode user gets 28% more events here (549 against 429). That is reported as a
documentation concern; nothing in the package is changed.

The examples are in `released_vs_package_unmatched_detected.png`: four of the 120, spread over
the session. The grey lane is the source's merged event before the ceiling, and the orange
fragments are the package's.

**2. The peak definitions differ (no effect on events).** The released peak is the trough of
the filtered signal; the package's is the maximum normalized power (the second departure its
docstring lists). Over the 429 matched pairs, 235 peaks are within half a sample of each other.
The package-minus-released quantiles are 5%: -3.2 ms, 25%: 0, median: 0, 75%: +0.8 ms and
95%: +3.2 ms, with extremes of -24.8 and +47.2 ms. The transcription, which uses the trough,
reproduces every released peak exactly.

**3. Nothing else differs.** No package event is clipped. The package's N in place of N-1
changes the SD by a factor of 1 - 2.1e-8 and moves no bound. `minimum_duration=0` matches the
source's lack of a minimum: the shortest released event, 19.2 ms, is kept by both.

## Data note (not a difference)

520 samples of channel 46 (2.2e-5 of the recording) sit at the int16 limits, and 94 of the
429 released events contain at least one such sample. The plotted examples show raw
deflections of tens of thousands of counts. This version of the source vetoes nothing, so
these large discharges are in the released inventory and in both reproductions alike. Anyone
using these events as ripples should look at them.

## Settings and assumptions

These are listed in full in `inputs.json`'s `assumptions`:

- The scipy design and `filtfilt` padding are assumed equal to MATLAB's. They were not run in
  MATLAB; the stage check's 4.5e-12 supports the assumption.
- `minimum_duration=0`, because the source has none and the detector accepts 0.
- `smoothing_window` stays at its default, checked to be 11 samples at 1250 Hz.
- `normalization_mask=None`, because the stored `restrict` is empty.
- Zero speed with `speed_threshold=np.inf`.
- The merge cap is left as the package implements it.

Nothing was tuned toward agreement.

## Cost (runtime.json, this machine)

**Stream.** A 10-minute slice read 10 HDF5 chunks: 90.2 MB in 52-54 s, with a peak RSS of
0.49-0.52 GB. That extrapolates to 24 min for the whole column's 289 chunks, whose exact
stored size is 2.55 GB.

The whole read took 76 s for 2.55 GB, with a peak RSS of 2.44 GB. remfile grows its requests
on long sequential reads, so the slice overstated the time by about 19 times. Its cache of up
to 1 GB put the memory above the slice's.

**Cache.** The column is kept in `<cache>/sessions/MS10/lfp_column46.npy` (47 MB), so reruns
do not stream again.

**Other steps.** Stage 1.5-1.8 s (1.6 GB), detect 1.4-1.8 s (2.0-2.1 GB: package 1.0 s,
transcription segmentation 0.03 s), compare under 0.1 s, explain 1.9 s.

## Reproduce

```bash
uv run --with remfile --with h5py python examples/reference_recordings/databank.py MS10
```

The command runs all the steps. `--steps stream --measure-minutes 10` gives the slice
measurement.

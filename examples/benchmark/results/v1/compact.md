# Compact comparison: v1

The reference condition's 20 simulated sessions (600 s each) and the main methods, each detector at its defaults and every recipe, each against its primary expression, one row per group of methods with identical detections. Estimates only: each number's 95 % interval and counts are in `compact_<target>.csv`. Held-out and interpolated operating values are in `compact_held_out.csv`, the point inventories, scored by peak containment, in `compact_points.csv`.

## Ripple

35 methods in 29 groups.

| Method | n | Recall, IoU 0 / 0.5 | Precision, IoU 0 / 0.5 | FP / min | FP at rest / rest min | Unmatched / session min | Onset ms, 10 / 50 % | Offset ms, 10 / 50 % |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `Karlsson_ripple_detector` | 4 | 0.72 / 0.64 | 0.84 / 0.74 | 1.44 | 1.98 | 1.42 | -8.3 / -24.1 | +4.8 / +28.6 |
| `Kay_ripple_detector` | 2 | 0.83 / 0.78 | 0.78 / 0.73 | 2.42 | 3.23 | 2.39 | +0.4 / -14.6 | -3.2 / +18.5 |
| `Long_sharp_wave_ripple_detector` | 1 | 0.68 / 0.32 | 0.96 / 0.45 | 0.00 | 0.00 | 0.295 | +11.9 / -3.9 | -28.2 / -4.8 |
| `Roumis_ripple_detector` | 1 | 0.83 / 0.78 | 0.78 / 0.73 | 2.39 | 3.17 | 2.37 | +0.1 / -14.7 | -3.0 / +18.9 |
| `Shvartsman_ripple_detector` | 1 | 0.66 / 0.59 | 0.91 / 0.81 | 0.660 | 0.830 | 0.650 | -9.7 / -25.8 | +6.4 / +30.1 |
| `Yu_ripple_detector` | 1 | 0.85 / 0.80 | 0.53 / 0.50 | 7.72 | 10.4 | 7.62 | +1.6 / -13.1 | -4.8 / +17.2 |
| `Zugaro_ripple_detector` | 1 | 0.67 / 0.49 | 0.87 / 0.63 | 1.05 | 1.35 | 1.04 | +14.3 / -1.6 | -23.1 / +1.2 |
| `recipe:ambrose_2016` | 2 | 0.73 / 0.52 | 0.91 / 0.64 | 0.777 | 0.967 | 0.765 | -16.7 / -32.6 | +14.3 / +38.7 |
| `recipe:berners_lee_2021` | 1 | 0.79 / 0.56 | 0.61 / 0.43 | 5.28 | 6.90 | 5.18 | -15.9 / -32.2 | +13.4 / +37.8 |
| `recipe:bhattarai_2020_ripples` | 1 | 0.45 / 0.41 | 0.98 / 0.90 | 0.0665 | 0.0837 | 0.0700 | +1.3 / -16.4 | -9.3 / +16.8 |
| `recipe:denovellis_2021` | 1 | 0.83 / 0.78 | 0.79 / 0.74 | 2.28 | 3.05 | 2.25 | +0.8 / -13.7 | -3.8 / +18.2 |
| `recipe:foster_2006_ripples` | 2 | 0.08 / 0.00 | 0.44 / 0.01 | 1.03 | 0.967 | 1.01 | +33.2 / +13.2 | -46.3 / -15.5 |
| `recipe:gupta_2010` | 1 | 0.81 / 0.68 | 0.10 / 0.08 | 77.7 | 77.2 | 76.8 | -0.3 / -14.6 | -5.5 / +15.7 |
| `recipe:harvey_2023_code` | 1 | 0.56 / 0.25 | 0.99 / 0.45 | 0.00 | 0.00 | 0.0450 | +13.0 / -3.5 | -30.4 / -5.5 |
| `recipe:harvey_2023_no_radiatum` | 1 | 0.60 / 0.38 | 0.55 / 0.34 | 5.13 | 5.29 | 5.07 | +1.0 / -15.7 | -4.5 / +19.9 |
| `recipe:harvey_2023_text` | 1 | 0.60 / 0.39 | 0.89 / 0.59 | 0.695 | 0.00 | 0.725 | +14.2 / -1.5 | -22.9 / +0.3 |
| `recipe:igata_2021_ripples` | 1 | 0.87 / 0.82 | 0.18 / 0.17 | 20.7 | 20.4 | 39.0 | +1.0 / -14.7 | -2.6 / +20.5 |
| `recipe:jadhav_2016` | 1 | 0.59 / 0.52 | 0.83 / 0.74 | 1.26 | 1.71 | 1.24 | -8.6 / -25.0 | +4.7 / +28.9 |
| `recipe:ji_2007_ripples` | 1 | 0.00 / 0.00 | 0.07 / 0.01 | 0.675 | 0.601 | 0.665 | -32.2 / -45.4 | -11.5 / +5.1 |
| `recipe:kaefer_2020` | 1 | 0.01 / 0.00 | 0.21 / 0.05 | 0.399 | 0.411 | 0.390 | -100.8 / -121.8 | +67.2 / +95.9 |
| `recipe:karlsson_2009` | 1 | 0.72 / 0.64 | 0.84 / 0.75 | 1.41 | 1.98 | 1.38 | -8.3 / -24.1 | +4.8 / +28.6 |
| `recipe:kudrimoti_1999` | 1 | 0.02 / 0.00 | 0.84 / 0.05 | 0.0460 | 0.0685 | 0.0450 | +31.7 / +11.5 | -45.2 / -15.4 |
| `recipe:mallory_2025_ripples` | 1 | 0.69 / 0.50 | 0.83 / 0.60 | 1.42 | 1.83 | 1.40 | -16.0 / -32.4 | +12.2 / +35.2 |
| `recipe:muessig_2019_ripples` | 1 | 0.88 / 0.72 | 0.04 / 0.03 | 164 | 163 | 225 | -7.2 / -26.6 | +0.8 / +22.9 |
| `recipe:nadasdy_1999` | 1 | 0.47 / 0.46 | 0.87 / 0.86 | 0.695 | 1.04 | 0.685 | +4.8 / -10.6 | -8.6 / +14.1 |
| `recipe:pfeiffer_2015` | 1 | 0.73 / 0.52 | 0.91 / 0.65 | 0.762 | 0.967 | 0.750 | -16.7 / -32.6 | +14.3 / +38.7 |
| `recipe:stella_2019` | 1 | 0.66 / 0.55 | 0.70 / 0.58 | 2.93 | 4.37 | 2.92 | +11.1 / -4.0 | -18.9 / +3.8 |
| `recipe:widloski_2025` | 1 | 0.51 / 0.00 | 0.85 / 0.00 | 0.925 | 0.761 | 0.910 | -174.7 / -192.5 | +169.8 / +198.1 |
| `recipe:wikenheiser_2013` | 1 | 0.83 / 0.04 | 0.09 / 0.00 | 83.5 | 124 | 82.0 | -150.7 / -167.3 | +136.8 / +164.0 |

Rows standing for methods with identical detections:

- `Karlsson_ripple_detector`: also `recipe:carr_2012`, `recipe:shin_2019`, `recipe:tang_2017`
- `Kay_ripple_detector`: also `recipe:gillespie_2021`
- `recipe:ambrose_2016`: also `recipe:pfeiffer_2013_ripples`
- `recipe:foster_2006_ripples`: also `recipe:lee_2002_ripples`

Inputs served by a stand-in:

- `recipe:bhattarai_2020_ripples`: place_cells
- `recipe:foster_2006_ripples`: sleep_intervals
- `recipe:harvey_2023_code`: pyramidal
- `recipe:harvey_2023_no_radiatum`: pyramidal
- `recipe:harvey_2023_text`: baseline_intervals
- `recipe:ji_2007_ripples`: baseline_intervals
- `recipe:kaefer_2020`: reference_lfp baseline_intervals
- `recipe:kudrimoti_1999`: sleep_intervals
- `recipe:nadasdy_1999`: baseline_intervals sleep_intervals
- `recipe:stella_2019`: sleep_intervals
- `recipe:wikenheiser_2013`: sleep_intervals

## Burst

36 methods in 33 groups.

| Method | n | Recall, IoU 0 / 0.5 | Precision, IoU 0 / 0.5 | FP / min | FP at rest / rest min | Unmatched / session min | Onset ms, 10 / 50 % | Offset ms, 10 / 50 % |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `multiunit_HSE_detector` | 1 | 0.93 / 0.71 | 0.34 / 0.26 | 18.2 | 24.3 | 17.9 | -21.1 / -43.0 | +19.1 / +49.6 |
| `recipe:bendor_2012` | 1 | 0.67 / 0.61 | 0.94 / 0.86 | 0.435 | 0.624 | 0.435 | +10.7 / -11.8 | -18.1 / +13.3 |
| `recipe:berners_lee_2022` | 1 | 0.67 / 0.58 | 0.83 / 0.73 | 1.39 | 1.89 | 1.36 | -14.3 / -39.2 | +12.0 / +45.6 |
| `recipe:bush_2022` | 1 | 0.63 / 0.59 | 0.88 / 0.83 | 0.849 | 1.04 | 0.830 | +1.2 / -22.0 | -3.9 / +26.9 |
| `recipe:chenani_2019` | 1 | 0.67 / 0.60 | 0.90 / 0.80 | 0.741 | 1.10 | 0.725 | -19.8 / -43.1 | +15.4 / +45.9 |
| `recipe:davidson_2009` | 1 | 0.79 / 0.63 | 0.76 / 0.61 | 2.42 | 3.25 | 2.40 | -20.5 / -42.9 | +18.0 / +49.1 |
| `recipe:denovellis_2021_mua` | 1 | 0.93 / 0.71 | 0.34 / 0.26 | 18.2 | 24.3 | 17.9 | -21.1 / -43.0 | +19.2 / +49.6 |
| `recipe:diba_2007` | 1 | 0.25 / 0.07 | 0.99 / 0.28 | 0.0256 | 0.0381 | 0.0250 | -70.6 / -95.9 | +79.1 / +118.9 |
| `recipe:drieu_2018` | 1 | 0.77 / 0.71 | 0.48 / 0.44 | 8.20 | 12.2 | 8.11 | -5.1 / -27.3 | +2.4 / +30.3 |
| `recipe:farooq_2019_neuron` | 1 | 0.28 / 0.28 | 1.00 / 0.99 | 0.00511 | 0.00761 | 0.00500 | +10.9 / -19.9 | -16.4 / +23.4 |
| `recipe:farooq_2019_science` | 2 | 0.28 / 0.28 | 1.00 / 0.99 | 0.00 | 0.00 | 0.00 | +10.9 / -19.9 | -16.4 / +23.4 |
| `recipe:foster_2006` | 1 | 0.15 / 0.14 | 1.00 / 0.89 | 0.00 | 0.00 | 0.00 | -2.9 / -38.3 | +5.5 / +46.8 |
| `recipe:gillespie_2021_mua` | 1 | 0.82 / 0.65 | 0.77 / 0.61 | 2.43 | 3.30 | 2.40 | -20.6 / -42.9 | +18.2 / +49.3 |
| `recipe:gridchyn_2020` | 1 | 0.90 / 0.76 | 0.17 / 0.14 | 43.2 | 43.2 | 42.7 | +11.6 / -9.0 | +0.0 / +27.3 |
| `recipe:igata_2021` | 1 | 0.94 / 0.72 | 0.25 / 0.19 | 28.5 | 28.3 | 28.0 | -20.6 / -42.4 | +18.0 / +48.6 |
| `recipe:ji_2007` | 1 | 0.72 / 0.56 | 0.91 / 0.71 | 0.721 | 1.07 | 0.705 | +14.6 / -9.3 | -22.5 / +9.7 |
| `recipe:krause_2022_hse` | 1 | 0.82 / 0.54 | 0.81 / 0.53 | 1.94 | 2.60 | 1.91 | -32.0 / -55.6 | +29.6 / +61.3 |
| `recipe:lee_2002` | 1 | 0.94 / 0.33 | 0.06 / 0.02 | 137 | 205 | 135 | -40.2 / -63.2 | +27.3 / +58.8 |
| `recipe:liu_2019` | 2 | 0.85 / 0.12 | 0.13 / 0.02 | 55.3 | 82.4 | 54.2 | -146.4 / -170.8 | +133.4 / +162.1 |
| `recipe:maboudi_2018` | 1 | 0.79 / 0.52 | 0.84 / 0.55 | 1.51 | 2.07 | 1.48 | -32.5 / -56.6 | +30.5 / +61.7 |
| `recipe:maboudi_2018_open_field` | 2 | 0.74 / 0.69 | 0.73 / 0.68 | 2.76 | 3.66 | 2.71 | -2.2 / -23.6 | -0.7 / +27.8 |
| `recipe:mallory_2025` | 1 | 0.76 / 0.65 | 0.65 / 0.55 | 4.12 | 5.58 | 4.03 | -15.2 / -39.5 | +14.4 / +44.4 |
| `recipe:mou_2022` | 1 | 0.86 / 0.40 | 0.53 / 0.24 | 7.63 | 7.79 | 7.49 | -50.6 / -76.1 | +50.1 / +82.2 |
| `recipe:olafsdottir_2015` | 1 | 0.67 / 0.55 | 0.63 / 0.52 | 3.87 | 5.75 | 3.78 | +4.5 / -20.0 | -4.7 / +26.5 |
| `recipe:olafsdottir_2015.bayesian_candidates` | 1 | 0.60 / 0.51 | 0.79 / 0.67 | 1.61 | 2.39 | 1.57 | +3.3 / -22.0 | -4.1 / +28.8 |
| `recipe:olafsdottir_2016` | 1 | 0.56 / 0.48 | 0.92 / 0.79 | 0.332 | 0.289 | 0.480 | +9.8 / -13.3 | -16.4 / +15.5 |
| `recipe:olafsdottir_2017` | 1 | 0.74 / 0.57 | 0.47 / 0.37 | 7.60 | 11.3 | 8.15 | +11.7 / -10.3 | -17.5 / +12.6 |
| `recipe:olafsdottir_2017.trajectory` | 1 | 0.56 / 0.48 | 0.94 / 0.81 | 0.194 | 0.289 | 0.345 | +9.8 / -13.3 | -16.4 / +15.5 |
| `recipe:silva_2015` | 1 | 0.60 / 0.54 | 0.83 / 0.75 | 1.24 | 1.62 | 1.21 | -12.0 / -38.1 | +10.1 / +43.4 |
| `recipe:widloski_2025_bursts` | 1 | 0.52 / 0.02 | 0.98 / 0.04 | 0.133 | 0.198 | 0.130 | -169.1 / -198.3 | +155.8 / +193.7 |
| `recipe:wu_2014` | 1 | 0.85 / 0.68 | 0.26 / 0.20 | 24.6 | 32.3 | 24.2 | -17.2 / -40.1 | +15.6 / +44.9 |
| `recipe:wu_2017` | 1 | 0.54 / 0.43 | 0.88 / 0.71 | 0.583 | 0.731 | 0.725 | +11.0 / -11.0 | -19.0 / +12.7 |
| `recipe:xu_2019` | 1 | 0.75 / 0.60 | 0.68 / 0.54 | 3.53 | 3.78 | 3.45 | -7.4 / -32.4 | +20.4 / +50.6 |

Rows standing for methods with identical detections:

- `recipe:farooq_2019_science`: also `recipe:farooq_2019_science_awake`
- `recipe:liu_2019`: also `recipe:liu_2019_awake`
- `recipe:maboudi_2018_open_field`: also `recipe:pfeiffer_2013`

Inputs served by a stand-in:

- `recipe:bush_2022`: pyramidal
- `recipe:chenani_2019`: place_cells behavior_intervals
- `recipe:diba_2007`: place_cells behavior_intervals
- `recipe:drieu_2018`: place_cells sleep_intervals
- `recipe:farooq_2019_neuron`: pyramidal sleep_intervals
- `recipe:farooq_2019_science`: recipe:farooq_2019_science (pyramidal place_cells sleep_intervals); recipe:farooq_2019_science_awake (pyramidal place_cells behavior_intervals)
- `recipe:foster_2006`: place_cells behavior_intervals
- `recipe:gridchyn_2020`: baseline_intervals
- `recipe:ji_2007`: sleep_intervals
- `recipe:lee_2002`: place_cells sleep_intervals
- `recipe:liu_2019`: recipe:liu_2019 (pyramidal sleep_intervals); recipe:liu_2019_awake (pyramidal behavior_intervals)
- `recipe:maboudi_2018`: pyramidal
- `recipe:maboudi_2018_open_field`: pyramidal
- `recipe:mallory_2025`: pyramidal
- `recipe:olafsdottir_2015`: templates behavior_intervals
- `recipe:olafsdottir_2015.bayesian_candidates`: templates behavior_intervals
- `recipe:olafsdottir_2016`: place_cells
- `recipe:olafsdottir_2017`: place_cells behavior_intervals
- `recipe:olafsdottir_2017.trajectory`: place_cells behavior_intervals
- `recipe:silva_2015`: pyramidal
- `recipe:wu_2014`: place_cells
- `recipe:xu_2019`: pyramidal

## Network

13 methods in 10 groups.

| Method | n | Recall, IoU 0 / 0.5 | Precision, IoU 0 / 0.5 | FP / min | FP at rest / rest min | Unmatched / session min | Onset ms, 10 / 50 % | Offset ms, 10 / 50 % |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `Carey_candidate_detector` | 1 | 0.82 / 0.78 | 0.65 / 0.63 | 4.78 | 6.37 | 4.72 | +1.5 / -18.4 | -4.4 / +22.6 |
| `recipe:bhattarai_2020` | 1 | 0.36 / 0.12 | 0.98 / 0.32 | 0.0869 | 0.114 | 0.0850 | -83.4 / -105.2 | +2.5 / +36.5 |
| `recipe:carey_2019` | 1 | 0.83 / 0.71 | 0.43 / 0.37 | 12.2 | 16.4 | 12.2 | +12.6 / -7.9 | -17.1 / +10.5 |
| `recipe:grosmark_2016` | 2 | 0.41 / 0.36 | 0.99 / 0.86 | 0.0358 | 0.0533 | 0.0350 | -13.6 / -34.6 | +15.1 / +45.3 |
| `recipe:huelin_gorriz_2023` | 2 | 0.50 / 0.43 | 0.99 / 0.85 | 0.0716 | 0.107 | 0.0700 | -15.0 / -36.7 | +16.4 / +47.8 |
| `recipe:krause_2022` | 1 | 0.62 / 0.50 | 0.92 / 0.75 | 0.511 | 0.662 | 0.625 | +13.3 / -6.3 | -17.0 / +10.7 |
| `recipe:liu_2023` | 1 | 0.41 / 0.39 | 1.00 / 0.93 | 0.0102 | 0.0152 | 0.0100 | -7.0 / -28.9 | +6.4 / +38.8 |
| `recipe:michon_2019` | 2 | 0.25 / 0.24 | 1.00 / 0.97 | 0.00 | 0.00 | 0.00 | -2.1 / -24.0 | +1.6 / +32.2 |
| `recipe:muessig_2019` | 1 | 0.10 / 0.09 | 1.00 / 0.89 | 0.00 | 0.00 | 0.00 | +25.5 / -6.2 | -31.0 / +9.0 |
| `recipe:yamamoto_2017` | 1 | 0.56 / 0.20 | 0.78 / 0.29 | 0.424 | 0.525 | 1.69 | +29.2 / +8.1 | -36.9 / -7.7 |

Rows standing for methods with identical detections:

- `recipe:grosmark_2016`: also `recipe:yang_2024`
- `recipe:huelin_gorriz_2023`: also `recipe:tirole_2022`
- `recipe:michon_2019`: also `recipe:michon_2021`

Inputs served by a stand-in:

- `recipe:bhattarai_2020`: place_cells
- `recipe:carey_2019`: example_ripples
- `recipe:grosmark_2016`: pyramidal sleep_intervals behavior_intervals external_ripples
- `recipe:huelin_gorriz_2023`: place_cells
- `recipe:krause_2022`: place_cells
- `recipe:liu_2023`: pyramidal
- `recipe:muessig_2019`: pyramidal sleep_intervals

## Definitions

- **Target and truth windows.** A method is scored against its primary expression (`methods.csv`), and its table is that expression's: the ripple, the burst, or the network event, a latent event's ripple, sharp-wave and burst components joined. A truth window is where a component's envelope is at or above 10 % of its peak (errors are measured at 50 % too).
- **Matching.** One to one. IoU 0, any overlap, is the predeclared primary matching. IoU >= 0.5 is reported beside it as a second headline; it was added after run v1's results had been seen, so it is post hoc, not predeclared. Recall is the truth windows matched over all of them, precision the detections matched over all, each pooled over the sessions a method ran on.
- **FP / min.** A false positive is a detection matching no truth window of its primary expression at IoU 0 whose time, its peak, else the midpoint of its bounds, lies outside every network window at 10 % (closed: a time on a window's start or end is inside). The false positives, at rest or running, over the minutes outside every network window (`minutes`): numerator and denominator cover the same time. This is the false-positive rate of every other table in these results. It is not a rate at rest.
- **FP at rest / rest min.** The false positives at rest over the minutes of rest outside every network window (`rest_minutes`). A detection is at rest or running by its time, as `rates_by_state` places events: its peak, else the midpoint of its bounds, against the session's running bouts as closed intervals on the timestamps (a time on a bout's start or end is running), not by overlap. The bouts are drawn again from each session's saved seed and checked against the run's saved rest time. False positives while running are counted in the CSV (`n_false_positives_running`, over `running_minutes`), with no rate; the two add up to `n_false_positives`.
- **Unmatched / session min.** Every unmatched detection, false positive or not, over the sessions' whole minutes (`session_minutes`): the unmatched burden. Those whose times lie inside a network window (`n_unmatched_in_events` in the CSV), such as a ripple method's detection of a sharp-wave-only event, are no false positives for any rate. Precision counts every unmatched detection.
- **Onset and offset.** The median signed error, detected minus truth (negative: early), in ms, over the pairs matched at IoU 0 against the truth windows at 10 % of the peak, each pair's error measured at the 10 % and at the 50 % bounds of its truth event. A method that finds only the easy events can time them better: read each beside its recall.
- **Held-out and interpolated.** `compact_held_out.csv` sets, per detector and target false-positive rate, the recall and rate measured on the odd (held-out) replicates at the tested setting chosen on the even ones (`measured`) beside `operating_points`' recall read off the curve pooled over the sessions every setting ran on (`n_sessions`): `interpolated` between two tested settings' rates in log rate, so no setting was run there, `tested` where the target is one setting's rate, which happens only when a pooled rate equals the target exactly, or `within budget` where every setting's rate is below the target, so the best recall of any setting is read (Long's detector, with no false positive at any setting): the recall at the best tested setting, a lower bound on the recall at that budget, since settings with more false positives were not tested. Only a held-out value is a number a threshold recommendation may quote; its setting is the best recall among those within the target on the even replicates.
- **Identical groups.** Methods sharing a primary expression whose detections are identical on every session, start, end and event times (peak, else midpoint) equal exactly and the same failures, are one row, named by the first in the methods' order; `n` counts its members, and `members` in the CSV lists them.
- **Stand-ins.** The inputs the benchmark serves a recipe in place of something the simulator lacks (the input policy's stand-ins: rest for sleep, baseline or eligible epochs, units selected by the simulator's labels, one template, a zero reference, an external or example ripple inventory). `stand_in_inputs` in the CSV lists a group's when its members share them, else each member's.
- **Intervals.** 95 % paired-bootstrap intervals over sessions (2000 resamples, seed 0), in the CSVs, not here.
- **Scope.** These numbers describe this simulator's reference sessions and its taxonomy of events and non-events only, not recorded data, and no method's own published performance.

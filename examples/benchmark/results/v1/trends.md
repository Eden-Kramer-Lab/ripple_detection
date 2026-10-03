# Trends stated for run v1

Each trend below comes from `candidate_trends.csv` and was stated only after its underlying events
were looked at: the figure named beside it is in `spot_checks/`, six events drawn from sessions
simulated again from the run's saved parameters. Intervals are 95% paired bootstrap intervals; a
change is a condition minus the reference over the replicates both share. Every statement is about
this simulator, under its stated assumptions and the benchmark's input policy
(`examples/benchmark/README.md`), not about recordings.

## 1. Ripples confined to some channels reverse the ripple detectors' order

Under the local spatial profile (a ripple on half the channels, with channel-specific gains and
delays), Karlsson's detector, which thresholds each channel, overtakes the detectors that pool
channels first: at 1 false positive a minute, Karlsson minus Roumis in recall goes from -0.114 in
the reference to +0.189 (reversed in every resample), and Karlsson also overtakes Kay at 1, 2 and
5 false positives a minute and Yu at 5. The pooled-trace recipes lose most: Nádasdy 1999 -0.440
(-0.468, -0.417), Zugaro -0.344 (-0.382, -0.309), Harvey 2023 (text) -0.338 (-0.368, -0.311).
Spot checks `01_model_order_reversal_Karlsson_ripple_detector_spatial_profile-local.png` and
`02_model_change_recipe-nadasdy_1999_spatial_profile-local.png`: the missed ripples are often
absent from channel 0 and present on others, which is what a pooled or single-channel trace
misses and a per-channel rule keeps.

## 2. Carey 2019's candidates depend on the number of units and on the slow field

`recipe:carey_2019`'s recall against the network falls by -0.826 (-0.844, -0.806) at 120 units,
where it returns no event in any of the 10 sessions (no call failed), and by -0.447 (-0.474,
-0.420) with no theta or delta field. Spot checks `07_robustness_recipe-carey_2019_n_units-120.png`
and `08_robustness_recipe-carey_2019_slow_amplitude-0.png` show clear ripples, sharp waves and
spike bursts with an empty Carey lane. At 120 units the mechanism is in the detector's multiunit
score: its slow baseline is capped at four units' worth of coincident spikes, so with 120 units
(20 interneurons at 8-15 Hz) the population sits above the cap nearly always, the score is high
everywhere, and the joint score rescaled to mean 0.5 never reaches its threshold of 4. Without a
slow field, a likely cause (not verified here) is the method's low-theta state requirement, a
theta/delta ratio with no delta to anchor it.

## 3. Long-event recipes look best at any overlap and worst at IoU 0.5

`recipe:lee_2002` ranks first by recall among burst methods at IoU 0 and 28th at IoU 0.5;
`recipe:liu_2019` and `recipe:liu_2019_awake` move from 8th to 33rd, `recipe:wikenheiser_2013` from
4th to 29th among ripple methods (ranks are descriptive; they carry no interval). These recipes
emit many long events: in the reference, `lee_2002` about 1,400 events a session and `liu_2019`
events covering about 56% of all rest time, so almost any true event overlaps one. The same bounds
add +8.13 (+7.81, +8.45) principal units counted active per true event for `liu_2019` and +8.04
(+7.78, +8.31) for `widloski_2025`. Spot checks `03_matching_rank_recipe-lee_2002_reference.png`
and `05_boundary_effect_recipe-liu_2019_reference.png` show single events spanning a true event and
far beyond it.

## 4. Methods that require many active cells find the events that recruit many cells

The true events found by `recipe:foster_2006`, `recipe:muessig_2019` and `recipe:diba_2007` recruit
1.49 (1.45, 1.53), 1.46 (1.42, 1.51) and 1.41 (1.38, 1.45) times as many cells on average as all
true events. Spot check `04_participation_bias_recipe-foster_2006_reference.png`: the events
Foster 2006 misses have visibly sparser spiking. This is a selection effect of cell-count rules,
reported beside, never subtracted from, observed participation.

## 5. No supported order between Roumis and Kay at 1 false positive a minute

Against ripple truth at 1 false positive a minute, among the curves read between tested
settings, Roumis has the highest recall, 0.786 (0.766, 0.805), and Kay the next, 0.784 (0.762,
0.802); their paired difference is +0.002 (-0.002, +0.008) over 20 sessions, bootstrap p = 0.25
(approximate, from the interval's 2000 resamples), so the data do not order them. Yu's curve
does not reach 1 false positive a minute. Long has no false positive at any threshold tested,
so it is read within budget, at its best tested setting: 0.766 (0.754, 0.778), a lower bound on
its recall at that rate, since no setting with more false positives was tested. The data
therefore do not place it below Roumis or Kay. Spot check
`06_operating_order_Roumis_ripple_detector_reference.png`: the two detectors give the same
events on the events drawn.

## Not a detector trend: `noise_type=brown`

Every LFP ripple detector's recall collapses under brown noise (Kay -0.82, `wu_2014_ripples` by
peak containment -0.894 (-0.911, -0.878)), but the condition is confounded. Ripples are sized
against the ripple-band noise, which brown noise makes about 25 times smaller (band SD 0.0058 against
0.144 for pink), while EMG and spike-leakage artifacts keep their absolute amplitudes: inside them
the band SD is 0.134, about ten times the ripples'. 96% of Kay's events under brown noise lie on
spike leakage (60%) or EMG (36%), against 9% in the reference. Spot check
`00_robustness_recipe-wu_2014_ripples_noise_type-brown.png` shows the missed ripples, visible once
each window is normalised. No statement here is drawn from this condition; sizing the artifacts
against the noise, as the gamma bursts already are, is recorded as a simulator follow-up.

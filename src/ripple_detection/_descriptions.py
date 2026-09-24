"""What each detector's inputs, tunables and output columns mean, in units.

The source of ``DetectorSpec.describe``. A test holds it to the code: every
keyword parameter of every registered detector has an entry here, no entry
is left without a parameter, and the columns listed are the columns each
detector returns, in order.
"""

POSITIONAL = {
    "time": ("(n_time,)", "s", "Sample timestamps, increasing."),
    "speed": (
        "(n_time,)",
        "cm/s",
        (
            "The animal's speed at each sample. NaN is an unknown speed, which splits "
            "nothing; speed_threshold says how the detector treats it."
        ),
    ),
    "sampling_frequency": ("scalar", "Hz", "The rate the samples were recorded at."),
}
"""The arguments every detector takes around its signals: shape, unit, meaning."""

SIGNALS = {
    "ripple_band_lfp": (
        "(n_time, n_channels)",
        "signal units",
        (
            "LFP filtered to the ripple band, e.g. with filter_ripple_band. NaN marks a "
            "missing sample."
        ),
    ),
    "raw_lfp": (
        "(n_time,) or (n_time, 1)",
        "signal units",
        "Raw, unfiltered LFP from one channel. NaN marks a missing sample.",
    ),
    "multiunit": (
        "(n_time, n_units)",
        "spikes per sample",
        (
            "Spike counts or indicators per unit, non-negative whole numbers, not a rate. "
            "NaN marks a missing sample."
        ),
    ),
}
"""Each signal kind: shape, unit, meaning."""

SIGNAL_ROLES = {
    ("Long_sharp_wave_ripple_detector", "raw_lfp"): (
        "The pyramidal-layer channel, which records the ripple; raw, unfiltered. NaN "
        "marks a missing sample."
    ),
    ("Long_sharp_wave_ripple_detector", "sharp_wave_lfp"): (
        "The stratum radiatum channel, which records the sharp wave; raw, unfiltered. "
        "Required."
    ),
    ("Carey_candidate_detector", "theta_lfp"): (
        "Optional theta channel, raw; when given, an event must lie inside a period of "
        "low theta power."
    ),
}
"""What a signal is to one detector, where its kind alone does not say: every
signal passed by name, and a raw channel whose place matters."""

PARAMETERS = {
    "speed_threshold": (
        "cm/s",
        (
            "Immobility is speed at or below this. An event is kept when its first and "
            "last samples are immobile, which an unknown (NaN) speed is not; np.inf "
            "turns the criterion off."
        ),
    ),
    "minimum_duration": (
        "s",
        (
            "Shortest run at or above the threshold that makes an event, before the event "
            "is extended to the mean; applied as a round-half-up sample count."
        ),
    ),
    "maximum_duration": (
        "s",
        (
            "Longest event as reported, or None for no ceiling; a sample count, so the "
            "ceiling is one sample short in elapsed time."
        ),
    ),
    "zscore_threshold": ("SD", "Detection threshold on the normalized detection trace."),
    "smoothing_sigma": (
        "s",
        "Standard deviation of the Gaussian that smooths each channel's envelope.",
    ),
    "close_ripple_threshold": (
        "s",
        (
            "An event starting within this of the previous kept event's end is dropped; 0 "
            "keeps every event."
        ),
    ),
    "close_event_threshold": (
        "s",
        (
            "An event starting within this of the previous kept event's end is dropped; 0 "
            "keeps every event."
        ),
    ),
    "ripple_score": (
        "",
        (
            "Array of shape (n_time,): a non-negative ripple score used in place of the "
            "Hilbert score of the LFP, e.g. carey_spectral_ripple_score; pass the LFP as "
            "None with it. None forms the score from the LFP."
        ),
    ),
    "threshold_method": (
        "",
        (
            "'zscore' thresholds the z-scored joint score; 'mean' rescales it to mean 0.5, "
            "as the original precand step did, so 4 is eight times the mean."
        ),
    ),
    "normalization_method": (
        "",
        (
            "'zscore' (mean and SD) or 'median_mad' (median and scaled MAD, robust to "
            "large events)."
        ),
    ),
    "normalization_mask": (
        "",
        (
            "Boolean array of shape (n_time,) selecting the samples the normalization "
            "statistics come from, e.g. (time >= start) & (time <= end); None for every "
            "valid sample."
        ),
    ),
    "channel_baselines": (
        "signal units",
        "With normalization_method='manual': each channel's baseline, shape (n_channels,).",
    ),
    "channel_deviations": (
        "signal units",
        "With normalization_method='manual': each channel's scale, shape (n_channels,).",
    ),
    "minimum_participating_channels": (
        "channels",
        (
            "Fewest channels that must detect an event; 2 when neither this nor "
            "minimum_participating_fraction is given."
        ),
    ),
    "minimum_participating_fraction": (
        "fraction",
        "Fraction of the channels that must detect an event, in place of a count.",
    ),
    "percentile": (
        "percentile",
        (
            "Percentile of the mirrored immobility-noise distribution taken as the threshold, "
            "estimated at each call."
        ),
    ),
    "zscore_per_channel": (
        "",
        (
            "Z-score each channel's smoothed envelope before the median across channels, as "
            "the original does."
        ),
    ),
    "low_threshold": ("SD", "Bounds: an event is a run strictly above this."),
    "high_threshold": ("SD", "The event's peak must be strictly above this."),
    "minimum_inter_ripple_interval": (
        "s",
        (
            "Events closer than this are merged, when the merged event stays within "
            "maximum_duration."
        ),
    ),
    "smoothing_window": (
        "s",
        (
            "Length of the moving average, rounded to an odd number of samples; the "
            "default is the original's 11 samples at 1250 Hz."
        ),
    ),
    "sharp_wave_band": (
        "Hz",
        "(low, high) difference-of-Gaussians band of the sharp-wave feature.",
    ),
    "ripple_band": (
        "Hz",
        "(low, high) difference-of-Gaussians band of the ripple-power feature.",
    ),
    "sharp_wave_percentile": (
        "percentile",
        (
            "A candidate must exceed this percentile of the sharp-wave-ripple cluster's "
            "sharp-wave feature."
        ),
    ),
    "ripple_power_percentile": (
        "percentile",
        "A candidate must exceed this percentile of the other cluster's ripple power.",
    ),
    "window_size": ("s", "Length of the non-overlapping candidate windows."),
    "local_window": (
        "s",
        (
            "Half-width of the window for the local statistics; a candidate this close to "
            "a block edge is not evaluated."
        ),
    ),
    "sharp_wave_thresholds": (
        "SD",
        "(boundary, peak) in local standard deviations of the sharp-wave feature.",
    ),
    "ripple_thresholds": (
        "SD",
        "(boundary, peak) in local standard deviations of the ripple power.",
    ),
    "minimum_separation": (
        "s",
        (
            "A candidate closer than this to the previous candidate, kept or not, is "
            "dropped; the last candidate is exempt, and the first is measured from the "
            "start of the record, as in the original."
        ),
    ),
    "minimum_sharp_wave_duration": (
        "s",
        "Shortest sharp wave; a candidate needs this or minimum_ripple_duration.",
    ),
    "maximum_sharp_wave_duration": ("s", "Longest sharp wave; a longer one is dropped."),
    "minimum_ripple_duration": (
        "s",
        "Shortest ripple; a candidate needs this or minimum_sharp_wave_duration.",
    ),
    "rng": (
        "",
        (
            "Seed or numpy.random.Generator for the k-means; None for the original's "
            "unseeded run."
        ),
    ),
    "minimum_active_units": (
        "units",
        "Fewest units with at least one spike inside an event; 0 imposes no criterion.",
    ),
    "ripple_smoothing_sigma": (
        "s",
        "Standard deviation of the Gaussian that smooths the ripple score.",
    ),
    "spike_smoothing_sigma": ("s", "Standard deviation of each unit's spike kernel."),
    "spike_cap": ("spikes", "Cap on each unit's smoothed contribution, in coincident spikes."),
    "baseline_smoothing_sigma": (
        "s",
        "Standard deviation of the slow baseline subtracted from the multiunit score.",
    ),
    "baseline_cap": (
        "units",
        "Cap on the summed score before its baseline is taken, in units' worth.",
    ),
    "theta_band": ("Hz", "(low, high) theta band of theta_lfp."),
    "theta_threshold": (
        "SD",
        "Theta-envelope z-score at or above which a period is excluded.",
    ),
    "state_merge_gap": (
        "s",
        "Low-speed (and low-theta) periods closer than this are joined.",
    ),
    "minimum_state_duration": (
        "s",
        "Low-speed (and low-theta) periods no longer than this are dropped.",
    ),
}
"""Each tunable: unit ("" when it has none) and meaning, where it means the
same in every detector that takes it."""

OVERRIDES = {
    ("Shvartsman_ripple_detector", "speed_threshold"): (
        "cm/s",
        (
            "Immobility is speed at or below this; an event is kept when at least half of "
            "its samples with known speed are immobile. np.inf turns the criterion off."
        ),
    ),
    ("Carey_candidate_detector", "speed_threshold"): (
        "cm/s",
        (
            "Immobility is speed at or below this; the whole event must lie inside a "
            "low-speed period, which an unknown (NaN) speed interrupts unless "
            "state_merge_gap bridges it. np.inf turns the criterion off."
        ),
    ),
    ("Yu_ripple_detector", "speed_threshold"): (
        "cm/s",
        (
            "Immobility is speed at or below this. It selects the noise sample the "
            "threshold is estimated from, and an event is kept when its first and last "
            "samples are immobile."
        ),
    ),
    ("Zugaro_ripple_detector", "minimum_duration"): (
        "s",
        "Shortest event as reported; a round-half-up sample count.",
    ),
    ("Carey_candidate_detector", "minimum_duration"): (
        "s",
        "Shortest event as reported; a round-half-up sample count.",
    ),
    ("Yu_ripple_detector", "minimum_duration"): (
        "s",
        (
            "Shortest run at or above the estimated threshold that makes an event, before "
            "the event is extended to the immobility mean; a round-half-up sample count."
        ),
    ),
    ("Shvartsman_ripple_detector", "normalization_method"): (
        "",
        (
            "'zscore', 'median_mad', or 'manual' with channel_baselines and "
            "channel_deviations from elsewhere, such as a whole recording day."
        ),
    ),
    ("Yu_ripple_detector", "normalization_mask"): (
        "",
        (
            "Boolean array of shape (n_time,) selecting the noise sample the threshold is "
            "estimated from and the trace normalized to; None for the immobile samples."
        ),
    ),
    ("multiunit_HSE_detector", "smoothing_sigma"): (
        "s",
        "Standard deviation of the Gaussian that smooths the population rate.",
    ),
    ("Kay_ripple_detector", "smoothing_sigma"): (
        "s",
        (
            "Standard deviation of the Gaussian that smooths the sum of the channels' "
            "squared envelopes."
        ),
    ),
    ("Roumis_ripple_detector", "smoothing_sigma"): (
        "s",
        "Standard deviation of the Gaussian that smooths each channel's squared envelope.",
    ),
    ("Carey_candidate_detector", "low_threshold"): (
        "SD, or multiples of half the mean",
        (
            "Bounds on the joint score, z-scored or scaled as threshold_method says: an "
            "event is a run strictly above this."
        ),
    ),
    ("Carey_candidate_detector", "high_threshold"): (
        "SD, or multiples of half the mean",
        (
            "The event's peak on the joint score, scaled as threshold_method says, must be "
            "strictly above this."
        ),
    ),
}
"""Where a tunable means something particular in one detector."""

_DESCRIPTIVE = (
    "Largest value of the detection trace sustained for minimum_duration. "
    "Descriptive only: {why}, so it can fall below the detection threshold."
)

COLUMN_OVERRIDES = {
    ("Shvartsman_ripple_detector", "max_sustained_zscore"): _DESCRIPTIVE.format(
        why="it is taken on the mean over the participating channels"
    ),
    ("Zugaro_ripple_detector", "max_sustained_zscore"): _DESCRIPTIVE.format(
        why="events come from a two-threshold rule"
    ),
    ("Carey_candidate_detector", "max_sustained_zscore"): _DESCRIPTIVE.format(
        why="events come from a two-threshold rule"
    ),
    ("Long_sharp_wave_ripple_detector", "max_sustained_zscore"): _DESCRIPTIVE.format(
        why="events come from clustering, and it is NaN for an event shorter than "
        "minimum_sharp_wave_duration"
    ),
    ("Long_sharp_wave_ripple_detector", "peak_time"): (
        "s. Time of the sharp-wave peak, not of the ripple power the other statistics "
        "describe."
    ),
}
"""Where a column means something particular in one detector."""

COLUMNS = {
    "start_time": "s. Time of the event's first sample.",
    "end_time": "s. Time of the event's last sample.",
    "duration": "s. end_time - start_time, one sample interval less than n_samples spans.",
    "n_samples": "Samples in the event, first to last inclusive; what the duration limits test.",
    "max_sustained_zscore": (
        "Largest z-score sustained for minimum_duration: the highest threshold "
        "that would still find the event. Named max_thresh before 2.0, which "
        "approximated it."
    ),
    "mean_zscore": "Mean of the detection trace over the event.",
    "median_zscore": "Median of the detection trace over the event.",
    "max_zscore": "Maximum of the detection trace over the event.",
    "min_zscore": "Minimum of the detection trace over the event.",
    "area": "SD s. Integral of the detection trace over the event.",
    "total_energy": "SD^2 s. Integral of the squared detection trace over the event.",
    "speed_at_start": "cm/s. Speed at the first sample; NaN when unknown.",
    "speed_at_end": "cm/s. Speed at the last sample; NaN when unknown.",
    "max_speed": "cm/s. Over the samples with known speed; NaN if none.",
    "min_speed": "cm/s. Over the samples with known speed; NaN if none.",
    "median_speed": "cm/s. Over the samples with known speed; NaN if none.",
    "mean_speed": "cm/s. Over the samples with known speed; NaN if none.",
    "clipped_start": "The event was cut off at its start by missing data or the recording edge.",
    "clipped_end": "The event was cut off at its end by missing data or the recording edge.",
    "peak_time": (
        "s. Time of the largest value of the detection trace in the event; the first "
        "such sample on a tie."
    ),
}
"""The columns every detector returns, in order; the index is event_number."""

EXTRA_COLUMNS = {
    "Shvartsman_ripple_detector": {
        "participants": "Sorted tuple of the channels that detected the event.",
        "n_participants": "Number of participating channels.",
        "frac_participants": "n_participants over the number of channels.",
    },
    "Yu_ripple_detector": {
        "n_suprathreshold_samples": "Longest run at or above the threshold inside the event.",
        "detection_threshold_zscore": "SD. The threshold estimated for this call.",
    },
    "Long_sharp_wave_ripple_detector": {
        "sharp_wave_zscore": "Local SD. Peak sharp-wave feature against its local window.",
        "sharp_wave_local_percentile": (
            "Fraction (0-1) of the local window below the peak sharp wave."
        ),
        "ripple_power_zscore": "Local SD. Peak ripple power against its local window.",
        "ripple_power_local_percentile": (
            "Fraction (0-1) of the local window below the peak ripple power."
        ),
        "sharp_wave_duration": (
            "s. Samples of the sharp wave, first to last inclusive, over the rate: one "
            "sample interval more than duration, which is end_time - start_time."
        ),
        "ripple_duration": "s. Elapsed time between the ripple's two boundary crossings.",
    },
    "Carey_candidate_detector": {
        "n_active_units": "Units with at least one spike inside the event.",
    },
    "multiunit_HSE_detector": {
        "n_active_units": "Units with at least one spike inside the event.",
    },
}
"""The columns a detector adds after the shared ones."""

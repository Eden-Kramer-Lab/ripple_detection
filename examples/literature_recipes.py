"""One recipe per surveyed paper: its event rule written with this package.

Each function below reproduces, as closely as the package allows, the event
detection a paper in the literature survey (``load_literature_parameters()``)
describes. Parameter values live in the packaged CSV; field evidence and source
versions are indexed by ``docs/literature/README.md``. Paper notes under
``docs/literature/papers/`` explain interpretation and uncertainties. Each recipe's
docstring names the rule and says where it departs from the paper. Papers
whose events are defined by decoding have no recipe, only the detection they
use to label events where they have one.

Run on a simulated session (running bouts with theta, rest with delta, ripples
at rest with bursting units and a radiatum sharp wave), every recipe is
scored against the known ripples: how many events it finds, the fraction of
ripples it overlaps and the events that overlap none. Simulated data say
whether a recipe runs and behaves, not whether it matches the paper's events.

Run with ``uv run python examples/literature_recipes.py``; it writes
``literature_recipes_results.csv`` beside this file.

Conventions in the docstrings: "assumed" marks a value or choice the paper
does not report; "scaled" marks a state epoch of minutes shortened to fit the
simulated session. Stand-ins used throughout: ``place_cells`` (units 0-39)
for sorted place cells, a template's cells or a probe sequence;
``pyramidal`` (units 0-49) for sorted pyramidal or principal cells; all 60
units for multiunit or unsorted spikes; per-sample counts for spike counts in
bins; channel 0 for a paper's "best" or single channel; and speed plus a
theta/delta ratio for sleep or rest scoring, which the papers did from EMG,
video or by hand. Where a paper gives no normalization period, the recipe
normalizes over the whole session (assumed).
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import butter, fftconvolve, sosfiltfilt

import ripple_detection as rd

OUTPUT = Path(__file__).with_name("literature_recipes_results.csv")

SAMPLING_FREQUENCY = 1500.0
RUNNING_INTERVALS = [(12.0, 24.0), (40.0, 52.0), (70.0, 80.0)]
RIPPLE_TIMES = [
    3.0, 5.5, 8.0, 10.0,
    27.0, 30.0, 33.0, 36.0,
    55.0, 58.0, 61.0, 64.0, 67.0,
    83.0, 86.0, 88.0,
]  # fmt: skip


@dataclass(eq=False)
class Recording:
    """A simulated session with the unit groups and cached traces the recipes use."""

    session: rd.SimulatedSession
    place_cells: np.ndarray
    pyramidal: np.ndarray

    @property
    def time(self) -> np.ndarray:
        return self.session.time

    @property
    def fs(self) -> float:
        return self.session.sampling_frequency

    @property
    def speed(self) -> np.ndarray:
        return self.session.speed

    @property
    def multiunit(self) -> np.ndarray:
        return self.session.multiunit

    @functools.cache  # noqa: B019 - one recording per run
    def filtered(self, band: tuple[float, float]) -> np.ndarray:
        return rd.filter_ripple_band(self.session.lfps, self.fs, band=band)

    @functools.cache  # noqa: B019
    def envelope(self, band: tuple[float, float]) -> np.ndarray:
        return rd.get_envelope(self.filtered(band))

    def mean_envelope(
        self, band: tuple[float, float], channels: int | None = None
    ) -> np.ndarray:
        envelope = self.envelope(band)
        return envelope[:, :channels].mean(axis=1) if channels else envelope.mean(axis=1)

    def rate(self, units: np.ndarray | None, sigma: float) -> np.ndarray:
        spikes = self.multiunit if units is None else self.multiunit[:, units]
        return rd.get_multiunit_population_firing_rate(spikes, self.fs, sigma)

    def counts(self, units: np.ndarray | None = None) -> np.ndarray:
        spikes = self.multiunit if units is None else self.multiunit[:, units]
        return spikes.sum(axis=1)

    @functools.cache  # noqa: B019
    def ratio(
        self,
        theta: tuple[float, float] = (6.0, 12.0),
        delta: tuple[float, float] = (1.0, 4.0),
        smoothing_sigma: float = 1.0,
        measure: str = "amplitude",
    ) -> np.ndarray:
        return rd.theta_delta_ratio(
            self.session.raw_lfp, self.fs, theta_band=theta, delta_band=delta,
            smoothing_sigma=smoothing_sigma, measure=measure,
        )  # fmt: skip

    def intervals_to_mask(self, intervals: np.ndarray) -> np.ndarray:
        mask = np.zeros(self.time.size, dtype=bool)
        for start, end in np.asarray(intervals).reshape(-1, 2):
            mask |= (self.time >= start) & (self.time <= end)
        return mask

    def mask_to_intervals(self, mask: np.ndarray) -> np.ndarray:
        return rd.state_intervals(mask.astype(float), self.time, 0.5, comparison=">")

    def sleep(
        self,
        speed_below: float,
        ratio_below: float,
        stillness: float = 0.0,
        theta: tuple[float, float] = (6.0, 12.0),
        delta: tuple[float, float] = (1.0, 4.0),
        smoothing_sigma: float = 1.0,
        measure: str = "amplitude",
    ) -> np.ndarray:
        """Intervals of stillness lasting ``stillness`` seconds that are also of
        low theta/delta: the stand-in for sleep or quiet rest."""
        still = rd.state_intervals(
            self.speed, self.time, speed_below, minimum_duration=stillness
        )
        ratio = self.ratio(theta, delta, smoothing_sigma, measure)
        low_theta = rd.state_intervals(ratio, self.time, ratio_below)
        return self.mask_to_intervals(
            self.intervals_to_mask(still) & self.intervals_to_mask(low_theta)
        )

    def awake(self, sleep: np.ndarray) -> np.ndarray:
        """The complement of ``sleep``."""
        return self.mask_to_intervals(~self.intervals_to_mask(sleep))


def make_recording(duration: float = 90.0, rng: int = 0) -> Recording:
    """The simulated session every recipe runs on."""
    time = rd.simulate_time(int(duration * SAMPLING_FREQUENCY), SAMPLING_FREQUENCY)
    ripples = [t for t in RIPPLE_TIMES if t < duration - 1]
    running = [(a, min(b, duration - 5)) for a, b in RUNNING_INTERVALS if a < duration - 6]
    # place cells sparse at rest and strongly recruited in ripples, as in
    # replay; other pyramidal cells a little busier; interneurons busiest
    baseline_rate = np.concatenate(
        [np.linspace(0.1, 0.5, 40), np.linspace(0.5, 1.5, 10), np.linspace(2.0, 5.0, 10)]
    )
    session = rd.simulate_session(
        time,
        ripples,
        n_channels=4,
        n_units=60,
        baseline_rate=baseline_rate,
        ripple_rate_gain=40.0,
        ripple_duration=(0.05, 0.2),
        sharp_wave_amplitude=6.0,  # the largest deflection at rest in radiatum, above delta
        running_intervals=running,
        theta_amplitude=4.0,
        delta_amplitude=4.0,
        rng=rng,
    )
    units = np.arange(60)
    return Recording(session, place_cells=units < 40, pyramidal=units < 50)


def bounds(events: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(events, pd.DataFrame):
        return events[["start_time", "end_time"]].to_numpy(dtype=float).reshape(-1, 2)
    return np.asarray(events, dtype=float).reshape(-1, 2)


def within_duration(events: np.ndarray, low: float = 0.0, high: float = np.inf) -> np.ndarray:
    """Events whose elapsed duration is from ``low`` to ``high`` seconds."""
    events = bounds(events)
    duration = events[:, 1] - events[:, 0]
    return events[(duration >= low - 1e-9) & (duration <= high + 1e-9)]


def within_intervals(events: pd.DataFrame | np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """Events lying entirely inside one of the intervals."""
    events, intervals = bounds(events), np.asarray(intervals, dtype=float).reshape(-1, 2)
    if len(intervals) == 0:
        return events[:0]
    which = np.searchsorted(intervals[:, 0], events[:, 0], side="right") - 1
    inside = (which >= 0) & (events[:, 1] <= intervals[np.clip(which, 0, None), 1])
    return events[inside]


def only_in(rec: Recording, values: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """``values`` (a trace or the spikes) missing outside the intervals, so
    detection runs inside them only."""
    mask = rec.intervals_to_mask(intervals)
    return np.where(mask if values.ndim == 1 else mask[:, None], values, np.nan)


def zugaro_ripple_peaks(rec: Recording, band: tuple[float, float]) -> pd.DataFrame:
    """bz_FindRipples-like ripples on one pyramidal channel, for the
    Buzsaki-lineage papers that require a ripple peak but do not describe
    their ripple detector (assumed: Huszar et al. 2022's 5 SD peak and 2 SD
    bounds, 20-200 ms; its noise-channel veto is not reproduced)."""
    return rd.Zugaro_ripple_detector(
        rec.time, rec.filtered(band)[:, :1], rec.speed, rec.fs,
        low_threshold=2.0, high_threshold=5.0, maximum_duration=0.2,
        speed_threshold=np.inf,
    )  # fmt: skip


# --------------------------------------------------------------------------- recipes


@dataclass
class Recipe:
    row: int
    paper: str
    trigger: str
    run: Callable[[Recording], pd.DataFrame | np.ndarray]
    note: str


RECIPES: list[Recipe] = []


def recipe(row: int, paper: str, trigger: str):
    def register(function):
        RECIPES.append(Recipe(row, paper, trigger, function, (function.__doc__ or "").strip()))
        return function

    return register


@recipe(0, "Mallory 2025", "MUA")
def mallory_2025(rec):
    """Linear-track candidates: excitatory-cell density, 12.5 ms Gaussian,
    z-scored over speed <= 5 with moving samples masked, peak > 3 SD, bounds at
    the mean, events whose peaks are <= 70 ms apart merged (after a merge the
    code measures from the larger peak, the package from the last), >= 10 cells
    (a replay-stage criterion, applied to the whole event as a proxy). Read
    from the preprint and the code; the decoding criteria are not reproduced."""
    events = rd.detect_events_from_trace(
        rec.time, rec.rate(rec.pyramidal, 0.0125), rec.speed, rec.fs,
        threshold=3.0, minimum_duration=0.0, speed_rule="restrict", speed_threshold=5.0,
    )  # fmt: skip
    merged = (
        rd.merge_close_events(events, 0.07, measure="peak", inclusive=True)
        if len(events)
        else bounds(events)
    )
    return rd.require_active_units(
        merged, rec.multiunit, rec.time, minimum_active_units=10, units=rec.pyramidal
    )


@recipe(1, "Widloski 2025", "decoding (ripple label)")
def widloski_2025(rec):
    """Replays are defined by decoding (not reproduced). This is the ripple
    label: 100-220 Hz, one channel per tetrode, envelope smoothed with an
    80 ms Gaussian and averaged, z-scored over stopping (speed < 5), peak > 2 SD
    for >= 15 ms, bounds at the mean, merged < 50 ms (the text; the code merges
    none)."""
    return rd.detect_events_from_trace(
        rec.time, rec.mean_envelope((100.0, 220.0)), rec.speed, rec.fs,
        threshold=2.0, smoothing_sigma=0.08, normalization_mask=rec.speed < 5,
        minimum_duration=0.015, close_event_threshold=0.05, close_event_rule="merge",
        speed_threshold=np.inf,
    )  # fmt: skip


def _population_with_ripple_peak(rec, sleep):
    """Grosmark 2016 and Yang 2024 (the text is shared): pyramidal rate, 15 ms
    Gaussian, 3 SD with the statistics over NREM, bounds at the mean,
    50-500 ms, >= 5 cells, in quiet waking or NREM (a stand-in: stillness with
    low theta/delta), and a ripple peak inside (ripple detector assumed, see
    zugaro_ripple_peaks). Follows the text: Yang's code z-scores the whole
    recording and requires the ripple's start inside."""
    events = rd.detect_events_from_trace(
        rec.time, rec.rate(rec.pyramidal, 0.015), rec.speed, rec.fs,
        threshold=3.0, normalization_mask=rec.intervals_to_mask(sleep), minimum_duration=0.0,
        minimum_event_duration=0.05, maximum_duration=0.5, speed_threshold=np.inf,
    )  # fmt: skip
    events = rd.exclude_overlap(events, rec.awake(sleep))
    events = rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_units=5, units=rec.pyramidal
    )
    return rd.require_times_inside(events, zugaro_ripple_peaks(rec, (130.0, 200.0)).peak_time)


@recipe(2, "Yang 2024", "SWR+MUA")
def yang_2024(rec):
    """See _population_with_ripple_peak; the state stand-in is speed < 4 and
    theta/delta < 1 (assumed; the paper used SleepScoreMaster with manual
    curation)."""
    return _population_with_ripple_peak(rec, rec.sleep(4.0, 1.0))


def _tirole(rec):
    """Approximation to Tirole 2022's released pipeline: all spikes (for sorted
    plus unsorted), nominal Gaussian 10 ms SD applied forward and back. This
    example uses the untruncated approximation (about 14.14 ms); the finite
    41-point released kernel has effective SD about 12.58 ms, and the text
    says 5 ms. Then z >= 3, bounds at z < 0 sought
    within 300 ms and relaxed to 0.25 then 0.5; events >= 100 ms, then merged
    < 50 ms, median speed <= 5, >= 5 place cells, and the ripple-band amplitude
    (channel 0 for the best channel, 125-300 Hz Hilbert amplitude, 15 ms moving
    average; the text says 100 ms) z >= 3 inside. The text's 750 ms maximum is
    not applied, as the code does not apply it."""
    events = rd.detect_events_from_trace(
        rec.time, rec.rate(None, 0.010 * np.sqrt(2)), rec.speed, rec.fs,
        threshold=3.0, bound_threshold=(0.0, 0.25, 0.5), bound_search_window=0.3,
        minimum_duration=0.0, minimum_event_duration=0.1, speed_threshold=np.inf,
    )  # fmt: skip
    merged = rd.merge_close_events(events, 0.05) if len(events) else bounds(events)
    merged = rd.exclude_movement(merged, rec.speed, rec.time, 5.0, rule="median")
    merged = rd.require_active_units(
        merged, rec.multiunit, rec.time, minimum_active_units=5, units=rec.place_cells
    )
    amplitude = uniform_filter1d(rec.envelope((125.0, 300.0))[:, 0], round(0.015 * rec.fs))
    return rd.require_trace_peak(merged, rd.normalize_signal(amplitude), rec.time, 3.0)


@recipe(3, "Huelin Gorriz 2023", "SWR+MUA")
def huelin_gorriz_2023(rec):
    """Related Tirole pipeline as an approximation; see _tirole. Huelin Gorriz's
    release omits the called extract_replay_events function, so its identity
    with Tirole's extractor is unproven. This example omits the published
    750 ms cap; the survey retains that published limit."""
    return _tirole(rec)


def _spiking_filter(rec, ripples):
    """eventSpikingTreshold: keep an event when the z-scored pyramidal count
    (10 ms windows), averaged over +/- 10 ms of its peak, exceeds 0.5."""
    count = uniform_filter1d(rec.counts(rec.pyramidal), round(0.01 * rec.fs))
    near_peak = uniform_filter1d(rd.normalize_signal(count), round(0.02 * rec.fs))
    if len(ripples) == 0:
        return ripples
    keep = near_peak[rd.core.nearest_sample_index(rec.time, ripples.peak_time)] > 0.5
    return ripples[keep]


@recipe(4, "Harvey 2023 (code)", "SWR")
def harvey_2023_code(rec):
    """The code's path for data without a radiatum channel: bz_FindRipples on
    one channel (channel 0 for the highest-ripple-power channel), thresholds 1
    and 3, 100-250 Hz, 20-300 ms, events 50 ms apart merged, then the spiking
    filter (see _spiking_filter). The EMG-from-LFP veto is not reproduced."""
    ripples = rd.Zugaro_ripple_detector(
        rec.time, rec.filtered((100.0, 250.0))[:, :1], rec.speed, rec.fs,
        low_threshold=1.0, high_threshold=3.0, minimum_inter_ripple_interval=0.05,
        minimum_duration=0.02, maximum_duration=0.3, speed_threshold=np.inf,
    )  # fmt: skip
    return _spiking_filter(rec, ripples)


@recipe(4, "Harvey 2023 (text)", "SWR (needs radiatum)")
def harvey_2023_text(rec):
    """As written: 80-250 Hz power (rectified, low-passed at 55 Hz; a
    Butterworth band-pass stands in for the difference of Gaussians), events
    over 4 SD extended to 1 SD, >= 15 ms, kept when a radiatum sharp wave
    (5-40 Hz, negative (assumed), > 2.5 SD, 20-400 ms) overlaps. The 4 SD
    clipping of the power estimate is not reproduced. The period the SDs come
    from is not stated: stillness (< 4 cm/s, assumed) stands in for the lab's
    NREM baseline in Oliva et al. 2016 and Fernandez-Ruiz et al. 2019."""
    sos = butter(2, np.array([80.0, 250.0]) / (rec.fs / 2), btype="bandpass", output="sos")
    low = butter(2, 55.0 / (rec.fs / 2), output="sos")
    power = sosfiltfilt(low, np.abs(sosfiltfilt(sos, rec.session.raw_lfp)))
    ripples = rd.detect_events_from_trace(
        rec.time, power, rec.speed, rec.fs,
        threshold=4.0, bound_threshold=1.0, normalization_mask=rec.speed < 4,
        minimum_duration=0.0, minimum_event_duration=0.015, speed_threshold=np.inf,
    )  # fmt: skip
    band = butter(2, np.array([5.0, 40.0]) / (rec.fs / 2), btype="bandpass", output="sos")
    sharp_waves = rd.detect_events_from_trace(
        rec.time, -sosfiltfilt(band, rec.session.sharp_wave_lfp), rec.speed, rec.fs,
        threshold=2.5, bound_threshold=2.5, normalization_mask=rec.speed < 4,
        minimum_duration=0.0, minimum_event_duration=0.02, maximum_duration=0.4,
        speed_threshold=np.inf,
    )  # fmt: skip
    return rd.require_overlap(ripples, sharp_waves)


@recipe(5, "Liu 2023", "SWR+MUA (needs radiatum)")
def liu_2023(rec):
    """DetectSWR at neurocode defaults on the pyramidal and radiatum channels
    (the text's 1 SD bounds and 15-400 ms limits are not applied; manual
    curation is not reproduced); the candidates are pyramidal-cell bursts
    (10 ms Gaussian, assumed to be its SD; > 2 SD, bounds at the mean,
    100-500 ms) overlapping an SWR."""
    swrs = rd.Long_sharp_wave_ripple_detector(
        rec.time, rec.session.raw_lfp, rec.speed, rec.fs,
        sharp_wave_lfp=rec.session.sharp_wave_lfp, speed_threshold=np.inf,
    )  # fmt: skip
    bursts = rd.detect_events_from_trace(
        rec.time, rec.rate(rec.pyramidal, 0.010), rec.speed, rec.fs,
        threshold=2.0, minimum_duration=0.0, minimum_event_duration=0.1,
        maximum_duration=0.5, speed_threshold=np.inf,
    )  # fmt: skip
    return rd.require_overlap(bursts, swrs)


@recipe(6, "Tirole 2022", "SWR+MUA")
def tirole_2022(rec):
    """See _tirole."""
    return _tirole(rec)


@recipe(7, "Bush 2022", "MUA")
def bush_2022(rec):
    """Pyramidal cells, 5 ms Gaussian, peak z >= 3, bounds at z >= 0; merged
    when <= 40 ms apart, events <= 40 ms dropped, then >= 5 or 15% of
    pyramidal cells (whichever is larger), median speed <= 10, <= 0.5 s."""
    events = rd.detect_events_from_trace(
        rec.time, rec.rate(rec.pyramidal, 0.005), rec.speed, rec.fs,
        threshold=3.0, minimum_duration=0.0, speed_threshold=np.inf,
    )  # fmt: skip
    merged = (
        rd.merge_close_events(events, 0.04, inclusive=True) if len(events) else bounds(events)
    )
    merged = within_duration(merged, low=0.04 + 1 / rec.fs)
    merged = rd.require_active_units(
        merged, rec.multiunit, rec.time,
        minimum_active_units=5, minimum_active_fraction=0.15, units=rec.pyramidal,
    )  # fmt: skip
    merged = rd.exclude_movement(merged, rec.speed, rec.time, 10.0, rule="median")
    return within_duration(merged, high=0.5)


@recipe(8, "Berners-Lee 2022", "MUA")
def berners_lee_2022(rec):
    """Pyramidal-cell density, 10 ms Gaussian, >= 3 SD, bounds at the mean,
    100-500 ms; statistics over speed < 5 and events cut at movement (the code)."""
    return rd.detect_events_from_trace(
        rec.time, rec.rate(rec.pyramidal, 0.010), rec.speed, rec.fs,
        threshold=3.0, minimum_duration=0.0, minimum_event_duration=0.1,
        maximum_duration=0.5, speed_rule="restrict", speed_threshold=5.0,
    )  # fmt: skip


def _pfeiffer_2015_swrs(rec, threshold=3.0, channels=None, maximum_duration=2.0):
    """Mean over tetrodes of the 150-250 Hz envelope, 12.5 ms Gaussian, above
    ``threshold`` SD with statistics over speed < 5 (one reading of
    "excluding periods of movement"), speed <= 5 at both ends, bounds at the
    mean, 50 ms to ``maximum_duration``."""
    return rd.detect_events_from_trace(
        rec.time, rec.mean_envelope((150.0, 250.0), channels), rec.speed, rec.fs,
        threshold=threshold, smoothing_sigma=0.0125, normalization_mask=rec.speed < 5,
        minimum_duration=0.0, minimum_event_duration=0.05, maximum_duration=maximum_duration,
        speed_threshold=5.0,
    )  # fmt: skip


@recipe(10, "Krause 2022", "SWR")
def krause_2022(rec):
    """Pfeiffer & Foster 2015 SWRs, each trimmed to its population burst:
    place-cell rate per cell (12 ms boxcar, as the code's 4 x 3 ms bins) above
    2 spikes/s, first to last crossing, >= 30 ms."""
    swrs = _pfeiffer_2015_swrs(rec)
    per_cell = rec.counts(rec.place_cells) * rec.fs / rec.place_cells.sum()
    rate = uniform_filter1d(per_cell, round(0.012 * rec.fs))
    return rd.trim_events_to_trace(swrs, rate, rec.time, 2.0, minimum_duration=0.03)


@recipe(11, "Mou 2022", "MUA")
def mou_2022(rec):
    """All spikes, 20 ms Gaussian (2 x 10 ms bins), scaled to 0-1 over the
    session, peak > 0.35, bounds at 0.15, merged < 30 ms; >= 4 active template
    cells (an analysis criterion; place cells stand in for a template)."""
    smoothed = rd.gaussian_smooth(rec.counts(), 0.02, rec.fs)
    scaled = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
    events = rd.detect_events_from_trace(
        rec.time, scaled, rec.speed, rec.fs,
        threshold=0.35, bound_threshold=0.15, normalization_method="none",
        minimum_duration=0.0, close_event_threshold=0.03, close_event_rule="merge",
        speed_threshold=np.inf,
    )  # fmt: skip
    return rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_units=4, units=rec.place_cells
    )


@recipe(12, "Berners-Lee 2021", "SWR")
def berners_lee_2021(rec):
    """Pfeiffer & Foster 2015's rule at 2 SD on 3 tetrodes (the first three
    stand in for the three with the most pyramidal cells)."""
    return _pfeiffer_2015_swrs(rec, threshold=2.0, channels=3)


@recipe(13, "Denovellis 2021", "SWR")
def denovellis_2021(rec):
    """Kay_ripple_detector at its defaults (this package). The analysis keeps
    SWRs with spikes on >= 2 tetrodes, which the simulation has no tetrodes for."""
    return rd.Kay_ripple_detector(rec.time, rec.filtered((150.0, 250.0)), rec.speed, rec.fs)


@recipe(14, "Gillespie 2021", "SWR")
def gillespie_2021(rec):
    """Kay consensus trace, 2 SD for 15 ms, speed < 4."""
    return rd.Kay_ripple_detector(rec.time, rec.filtered((150.0, 250.0)), rec.speed, rec.fs)


def _michon(rec):
    """Michon 2019 / 2021 (Kloosterman lab): ripple envelope (140-225 Hz, mean
    of 3 tetrodes) and all-spike count (per sample, for 5 ms bins), each
    smoothed with a 15 ms Gaussian and then a 3 s moving median subtracted
    (the text's order; the lab's code detrends first); ripples peak > 8 SD,
    bounds 0.5 SD, merged < 20 ms, >= 40 ms; bursts peak > 4 SD, bounds
    0.5 SD, >= 80 ms; candidates are bursts overlapping a ripple, with speed
    < 5 at both ends (which samples the paper tests is not stated)."""
    window = round(3.0 * rec.fs) | 1

    def detrended(trace):
        smoothed = rd.gaussian_smooth(trace, 0.015, rec.fs)
        return smoothed - median_filter(smoothed, size=window, mode="nearest")

    common = {
        "bound_threshold": 0.5, "minimum_duration": 0.0, "close_event_threshold": 0.02,
        "close_event_rule": "merge", "speed_threshold": np.inf,
    }  # fmt: skip
    ripples = rd.detect_events_from_trace(
        rec.time, detrended(rec.mean_envelope((140.0, 225.0), channels=3)), rec.speed, rec.fs,
        threshold=8.0, minimum_event_duration=0.04, **common,
    )  # fmt: skip
    bursts = rd.detect_events_from_trace(
        rec.time, detrended(rec.counts()), rec.speed, rec.fs,
        threshold=4.0, minimum_event_duration=0.08, **common,
    )  # fmt: skip
    return rd.exclude_movement(rd.require_overlap(bursts, ripples), rec.speed, rec.time, 5.0)


@recipe(15, "Michon 2021", "SWR+MUA")
def michon_2021(rec):
    """See _michon."""
    return _michon(rec)


@recipe(16, "Igata 2021 (candidates)", "SWR+MUA")
def igata_2021(rec):
    """The candidate stage only: the rate of all recorded neurons (15 ms
    Gaussian) z-scored over stopping (< 5 cm/s), > 2 SD, bounds at the mean,
    > 4 active neurons, 50 ms-2 s; no speed rule on events (none stated). The
    GMM split on rate and ripple power that follows is under-specified."""
    events = rd.detect_events_from_trace(
        rec.time, rec.rate(None, 0.015), rec.speed, rec.fs,
        threshold=2.0, normalization_mask=rec.speed < 5, minimum_duration=0.0,
        minimum_event_duration=0.05, maximum_duration=2.0, speed_threshold=np.inf,
    )  # fmt: skip
    return rd.require_active_units(events, rec.multiunit, rec.time, minimum_active_units=5)


@recipe(17, "Gridchyn 2020 (fixed threshold)", "MUA")
def gridchyn_2020(rec):
    """All spikes counted in a 20 ms window (centred; the online code's window
    trails), above 3.5 x their mean over the rest before the first run (for
    the pre-rest session), bounds at that mean, 150 ms refractory (onset to
    onset), onset moved to the first spike. The online per-minute adjustment
    of the multiplier is not run."""
    count = uniform_filter1d(rec.counts(), round(0.02 * rec.fs))
    baseline = count[rec.time < RUNNING_INTERVALS[0][0]].mean()
    events = rd.detect_events_from_trace(
        rec.time, count / baseline, rec.speed, rec.fs,
        threshold=3.5, bound_threshold=1.0, normalization_method="none",
        minimum_duration=0.0, speed_threshold=np.inf,
    )  # fmt: skip
    events = rd.exclude_close_events(events, 0.15, measure_from="start")
    return rd.trim_events_to_trace(events, rec.counts(), rec.time, 1.0, sides="start")


@recipe(18, "Kaefer 2020", "decoding (SWR label)")
def kaefer_2020(rec):
    """Replay is defined by decoding (not reproduced). This is the secondary
    SWR detector: ripple-band RMS over 240 ms (sliding per sample, for 20 ms
    steps; no reference-channel subtraction), mean over tetrodes, 5 SD peak,
    1.5 SD bounds."""
    power = uniform_filter1d(rec.filtered((150.0, 250.0)) ** 2, round(0.24 * rec.fs), axis=0)
    return rd.detect_events_from_trace(
        rec.time, np.sqrt(power).mean(axis=1), rec.speed, rec.fs,
        threshold=5.0, bound_threshold=1.5, minimum_duration=0.0, speed_threshold=np.inf,
    )  # fmt: skip


@recipe(19, "Bhattarai 2020", "SWR+MUA")
def bhattarai_2020(rec):
    """SWRs: 100-250 Hz power (the squared filtered signal, assumed for
    "instantaneous power") in a 50 ms boxcar on 2 channels, peak > 3 SD, bounds
    1 SD, longer than 20 ms, merged <= 100 ms, >= 5 place cells. Replays: > 60 ms
    of place-cell silence (whose silence counts is not stated), then >= 5 place
    cells within 300 ms, coinciding with an SWR."""
    power = uniform_filter1d(
        rec.filtered((100.0, 250.0))[:, :2] ** 2, round(0.05 * rec.fs), axis=0
    ).mean(axis=1)
    swrs = rd.detect_events_from_trace(
        rec.time, power, rec.speed, rec.fs,
        threshold=3.0, bound_threshold=1.0, minimum_duration=0.0, speed_threshold=np.inf,
    )  # fmt: skip
    swrs = within_duration(swrs, low=0.02 + 1 / rec.fs)
    swrs = rd.merge_close_events(swrs, 0.1, inclusive=True) if len(swrs) else swrs
    swrs = rd.require_active_units(
        swrs, rec.multiunit, rec.time, minimum_active_units=5, units=rec.place_cells
    )
    replays = rd.detect_silence_bounded_events(
        rec.time, rec.multiunit, rec.fs,
        minimum_silence=0.06 + 1 / rec.fs, window=0.3, units=rec.place_cells,
        minimum_active_units=5,
    )  # fmt: skip
    return rd.require_overlap(replays, swrs)


@recipe(20, "Stella 2019", "SWR")
def stella_2019(rec):
    """REM periods removed first, by a theta/delta power ratio (the paper's
    multitaper estimate, bands and cutoff are not given; 6-12 / 1-4 Hz and 2
    assumed); then Morlet-wavelet power in 150-250 Hz on each electrode (6
    frequencies, 7-cycle wavelets, assumed), z-scored over the remaining
    periods, maximum across electrodes, peak > 5 SD, bounds at 2 SD. The whole
    session stands in for the post-exploration sleep box."""
    rem = rec.intervals_to_mask(
        rd.state_intervals(rec.ratio(measure="power"), rec.time, 2.0, comparison=">")
    )
    frequencies = np.linspace(150.0, 250.0, 6)
    raw = rec.session.lfps
    power = np.zeros(raw.shape)
    for frequency in frequencies:
        sigma = 7 / (2 * np.pi * frequency)
        t = np.arange(-4 * sigma, 4 * sigma, 1 / rec.fs)
        wavelet = np.exp(2j * np.pi * frequency * t) * np.exp(-(t**2) / (2 * sigma**2))
        wavelet /= np.abs(wavelet).sum()
        power += np.abs(fftconvolve(raw, wavelet[:, None], mode="same", axes=0)) ** 2
    rms = np.where(rem[:, None], np.nan, np.sqrt(power / len(frequencies)))
    return rd.detect_events_from_trace(
        rec.time, rd.normalize_signal(rms).max(axis=1), rec.speed, rec.fs,
        threshold=5.0, bound_threshold=2.0, normalization_method="none",
        minimum_duration=0.0, speed_threshold=np.inf,
    )  # fmt: skip


@recipe(21, "Xu 2019", "MUA")
def xu_2019(rec):
    """Pyramidal cells, 15 ms Gaussian, peak > 3 SD, bounds at the mean,
    75-750 ms, >= 4 cells, >= 5 spikes, >= 10% of cells, onset at the first
    spike. Normalization period not stated (assumed the whole session)."""
    events = rd.detect_events_from_trace(
        rec.time, rec.rate(rec.pyramidal, 0.015), rec.speed, rec.fs,
        threshold=3.0, minimum_duration=0.0, minimum_event_duration=0.075,
        maximum_duration=0.75, speed_threshold=np.inf,
    )  # fmt: skip
    events = rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_units=4,
        minimum_active_fraction=0.1, minimum_spikes=5, units=rec.pyramidal,
    )  # fmt: skip
    return rd.trim_events_to_trace(
        events, rec.counts(rec.pyramidal), rec.time, 1.0, sides="start"
    )


def _farooq(rec, sleep, units):
    """Farooq 2019 (both papers): pyramidal-cell rate, 15 ms Gaussian, the
    periods above 2 SD (bounds at 2 SD), 100-800 ms, >= 5 cells, inside SWS."""
    events = rd.detect_events_from_trace(
        rec.time, only_in(rec, rec.rate(rec.pyramidal, 0.015), sleep), rec.speed, rec.fs,
        threshold=2.0, bound_threshold=2.0, minimum_duration=0.0,
        minimum_event_duration=0.1, maximum_duration=0.8, speed_threshold=np.inf,
    )  # fmt: skip
    return rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_units=5, units=units
    )


@recipe(22, "Farooq 2019 (Neuron)", "MUA")
def farooq_2019_neuron(rec):
    """SWS: speed < 1 cm/s for >= 5 s (scaled from 5 min) and theta/delta
    (6-12 / 1-4 Hz Hilbert amplitude, 5 s Gaussian) < 2; >= 5 neurons (the
    Methods; the Results say place cells). Awake-rest frames are not
    reproduced."""
    return _farooq(rec, rec.sleep(1.0, 2.0, stillness=5.0, smoothing_sigma=5.0), rec.pyramidal)


@recipe(23, "Farooq 2019 (Science)", "MUA")
def farooq_2019_science(rec):
    """SWS: speed < 2 cm/s and theta/delta (4-10 / 1-3 Hz, 10 s Gaussian) below
    its mean; >= 5 place-responsive cells. The manual review and the awake
    frames (< 1 cm/s) are not reproduced."""
    ratio = rec.ratio((4.0, 10.0), (1.0, 3.0), smoothing_sigma=10.0)
    still = rd.state_intervals(rec.speed, rec.time, 2.0)
    low = rd.state_intervals(ratio, rec.time, float(np.nanmean(ratio)))
    sleep = rec.mask_to_intervals(rec.intervals_to_mask(still) & rec.intervals_to_mask(low))
    return _farooq(rec, sleep, rec.place_cells)


@recipe(24, "Chenani 2019", "MUA")
def chenani_2019(rec):
    """Place-cell rate, 30 ms Gaussian, peak >= 3 SD, bounds >= 1 SD, >= 5
    active cells. The reward zones, chosen by eye, are not reproduced."""
    events = rd.detect_events_from_trace(
        rec.time, rec.rate(rec.place_cells, 0.030), rec.speed, rec.fs,
        threshold=3.0, bound_threshold=1.0, minimum_duration=0.0, speed_threshold=np.inf,
    )  # fmt: skip
    return rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_units=5, units=rec.place_cells
    )


@recipe(25, "Michon 2019", "SWR+MUA")
def michon_2019(rec):
    """See _michon."""
    return _michon(rec)


@recipe(26, "Liu 2019", "MUA")
def liu_2019(rec):
    """Pyramidal spikes inside SWS (speed < 1 cm/s and theta/delta < 2, 5 s
    Gaussian), split at >= 100 ms of silence, >= 4 cells, 80 ms-1.2 s. The
    awake-rest frames (< 2 cm/s) are not reproduced."""
    sleep = rec.sleep(1.0, 2.0, smoothing_sigma=5.0)
    return rd.detect_silence_bounded_events(
        rec.time, only_in(rec, rec.multiunit, sleep), rec.fs,
        minimum_silence=0.1, units=rec.pyramidal, minimum_active_units=4,
        minimum_duration=0.08, maximum_duration=1.2,
    )  # fmt: skip


def _karlsson_rule(rec, speed_threshold):
    """Karlsson & Frank 2009: each tetrode's 4 ms-smoothed envelope, 3 SD for
    >= 15 ms on any tetrode, bounds at the mean, overlapping events combined."""
    return rd.Karlsson_ripple_detector(
        rec.time,
        rec.filtered((150.0, 250.0)),
        rec.speed,
        rec.fs,
        speed_threshold=speed_threshold,
    )


@recipe(27, "Shin 2019", "SWR")
def shin_2019(rec):
    """The Karlsson rule at <= 4 cm/s; for the analyses, whole events >= 50 ms
    with >= 5 place cells."""
    events = within_duration(_karlsson_rule(rec, 4.0), low=0.05)
    return rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_units=5, units=rec.place_cells
    )


@recipe(28, "Carey 2019", "SWR+MUA")
def carey_2019(rec):
    """The published candidates' rule: the amSWR spectral score (template
    from example ripples; the paper picked them by hand, here the five
    largest Kay events), joint score rescaled to mean 0.5 and thresholded at 4,
    inside low-speed and low-theta intervals, >= 20 ms, >= 5 units."""
    kay = rd.Kay_ripple_detector(rec.time, rec.filtered((150.0, 250.0)), rec.speed, rec.fs)
    examples = kay.nlargest(5, "max_zscore")
    score = rd.carey_spectral_ripple_score(rec.time, rec.session.raw_lfp, rec.fs, examples)
    return rd.Carey_candidate_detector(
        rec.time, None, rec.multiunit, rec.speed, rec.fs,
        ripple_score=score, threshold_method="mean", low_threshold=4.0, high_threshold=4.0,
        theta_lfp=rec.session.raw_lfp,
    )  # fmt: skip


@recipe(29, "Muessig 2019", "SWR+MUA")
def muessig_2019(rec):
    """Bursts of pyramidal ("CS") cells, 10 ms Gaussian, the period above 3 SD
    (bounds at 3 SD, as "crossing of a threshold" reads), 100-750 ms,
    overlapping an SWR: 7 ms moving RMS of 100-250 Hz on the channel whose RMS
    varies most, above its 99th percentile, a 100 ms window at the peak; rest
    only (speed < 2.5 and a theta/delta power ratio < 2; the paper's bands are
    per-session peaks and its estimate multitaper). RUN-trial events are not
    reproduced."""
    rms = np.sqrt(
        uniform_filter1d(rec.filtered((100.0, 250.0)) ** 2, round(0.007 * rec.fs), axis=0)
    )
    rms = rms[:, int(np.argmax(rms.std(axis=0)))]
    level = float(np.percentile(rms, 99))
    peaks = rd.detect_events_from_trace(
        rec.time, rms, rec.speed, rec.fs,
        threshold=level, bound_threshold=level, normalization_method="none",
        minimum_duration=0.0, speed_threshold=np.inf,
    )  # fmt: skip
    swr_windows = rd.windows_around_times(peaks.peak_time, 0.05)
    bursts = rd.detect_events_from_trace(
        rec.time, rec.rate(rec.pyramidal, 0.010), rec.speed, rec.fs,
        threshold=3.0, bound_threshold=3.0, minimum_duration=0.0,
        minimum_event_duration=0.1, maximum_duration=0.75, speed_threshold=np.inf,
    )  # fmt: skip
    rest = rec.sleep(2.5, 2.0, measure="power")
    return rd.require_overlap(rd.require_overlap(bursts, swr_windows), rest)


@recipe(30, "Drieu 2018", "MUA")
def drieu_2018(rec):
    """Place-cell rate, 10 ms Gaussian, peak > 3 SD, bounds at the mean,
    <= 500 ms, inside SWS found by k-means (two clusters, assumed) on a
    theta/delta power ratio (6-10 / 1-4 Hz; Hilbert power for the paper's
    spectrogram, clustered over the whole session rather than sleep sessions),
    epochs longer than 2 s (scaled from 120 s) with gaps < 1 s bridged."""
    ratio = rec.ratio((6.0, 10.0), (1.0, 4.0), measure="power")
    sleep = rd.state_intervals(
        ratio, rec.time, rd.two_cluster_threshold(ratio), comparison="<=",
        merge_gap=1.0, minimum_duration=2.0,
    )  # fmt: skip
    return rd.detect_events_from_trace(
        rec.time, only_in(rec, rec.rate(rec.place_cells, 0.010), sleep), rec.speed, rec.fs,
        threshold=3.0, minimum_duration=0.0, maximum_duration=0.5, speed_threshold=np.inf,
    )  # fmt: skip


@recipe(31, "Maboudi 2018", "MUA")
def maboudi_2018(rec):
    """All units, 20 ms Gaussian, peak >= 3 SD over the session, bounds at
    the mean, mean speed <= 5, >= 80 ms, >= 4 active pyramidal cells."""
    events = rd.detect_events_from_trace(
        rec.time, rec.rate(None, 0.020), rec.speed, rec.fs,
        threshold=3.0, minimum_duration=0.0, minimum_event_duration=0.08,
        speed_rule="mean", speed_threshold=5.0,
    )  # fmt: skip
    return rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_units=4, units=rec.pyramidal
    )


@recipe(32, "Olafsdottir 2017", "MUA")
def olafsdottir_2017(rec):
    """Place cells, 5 ms Gaussian, > 3 SD, bounds at the mean, >= 40 ms, no
    speed above 3 cm/s in the event, and (for the trajectory analysis) >= 15%
    of place cells or more than 5. The corner restriction is not reproduced."""
    events = rd.detect_events_from_trace(
        rec.time, rec.rate(rec.place_cells, 0.005), rec.speed, rec.fs,
        threshold=3.0, minimum_duration=0.0, minimum_event_duration=0.04,
        speed_rule="all", speed_threshold=3.0,
    )  # fmt: skip
    return rd.require_active_units(
        events, rec.multiunit, rec.time,
        minimum_active_units=6, minimum_active_fraction=0.15, units=rec.place_cells,
    )  # fmt: skip


@recipe(33, "Wu 2017", "MUA")
def wu_2017(rec):
    """All spikes in 10 ms bins (a sliding 10 ms boxcar here), no smoothing,
    peak >= 4 SD, bounds at the mean, 50-400 ms."""
    count = uniform_filter1d(rec.counts(), round(0.01 * rec.fs))
    return rd.detect_events_from_trace(
        rec.time, count, rec.speed, rec.fs,
        threshold=4.0, minimum_duration=0.0, minimum_event_duration=0.05,
        maximum_duration=0.4, speed_threshold=np.inf,
    )  # fmt: skip


@recipe(34, "Yamamoto 2017 (one reading)", "SWR+MUA")
def yamamoto_2017(rec):
    """One reading of an ambiguous rule: summed spikes in 10 ms bins (a
    sliding 10 ms boxcar here), peak > 3 SD, bounds at 1 SD, kept when
    overlapping a period of 140-200 Hz power above 3 SD on one channel. The
    paper does not say how the two combine or which trace sets the bounds."""
    ripples = rd.detect_events_from_trace(
        rec.time, rec.envelope((140.0, 200.0))[:, 0] ** 2, rec.speed, rec.fs,
        threshold=3.0, bound_threshold=3.0, minimum_duration=0.0, speed_threshold=np.inf,
    )  # fmt: skip
    count = uniform_filter1d(rec.counts(), round(0.01 * rec.fs))
    bursts = rd.detect_events_from_trace(
        rec.time, count, rec.speed, rec.fs,
        threshold=3.0, bound_threshold=1.0, minimum_duration=0.0, speed_threshold=np.inf,
    )  # fmt: skip
    return rd.require_overlap(bursts, ripples)


@recipe(35, "Tang 2017", "SWR")
def tang_2017(rec):
    """The Karlsson rule at < 4 cm/s (smoothing and minimum inherited)."""
    return _karlsson_rule(rec, 4.0)


@recipe(36, "Grosmark 2016", "SWR+MUA")
def grosmark_2016(rec):
    """See _population_with_ripple_peak; NREM stand-in speed < 4 and
    theta/delta < 1 (assumed; the paper scored states by hand). The decoding
    stage's >= 100 ms and >= 5 or 10% of place cells are applied too."""
    events = _population_with_ripple_peak(rec, rec.sleep(4.0, 1.0))
    events = within_duration(events, low=0.1)
    return rd.require_active_units(
        events, rec.multiunit, rec.time,
        minimum_active_units=5, minimum_active_fraction=0.1, units=rec.place_cells,
    )  # fmt: skip


@recipe(37, "Ambrose 2016", "SWR")
def ambrose_2016(rec):
    """Pfeiffer & Foster 2015's trace on 4 tetrodes, > 3 SD, detected only
    while stopped (< 5 cm/s, stated in the paper). Statistics also come from
    stopping periods (the lab's convention, inferred); no duration limits are
    reported in the main Methods or supplement. The proximity to the well is
    not reproduced."""
    return rd.detect_events_from_trace(
        rec.time, rec.mean_envelope((150.0, 250.0)), rec.speed, rec.fs,
        threshold=3.0, smoothing_sigma=0.0125, minimum_duration=0.0,
        speed_rule="restrict", speed_threshold=5.0,
    )  # fmt: skip


@recipe(38, "Jadhav 2016", "SWR")
def jadhav_2016(rec):
    """The Karlsson rule at < 4 cm/s; SWRs within 1 s after the previous
    one's start dropped; >= 4 CA1 cells (all units) for candidates."""
    events = rd.exclude_close_events(_karlsson_rule(rec, 4.0), 1.0, measure_from="start")
    return rd.require_active_units(events, rec.multiunit, rec.time, minimum_active_units=4)


@recipe(39, "Olafsdottir 2016", "MUA")
def olafsdottir_2016(rec):
    """Place cells, 5 ms Gaussian, > 3 SD, bounds at the mean, >= 40 ms,
    >= 15% of the place cells; no speed rule. There is no separate rest
    session here: detection and statistics span the whole simulated session."""
    events = rd.detect_events_from_trace(
        rec.time, rec.rate(rec.place_cells, 0.005), rec.speed, rec.fs,
        threshold=3.0, minimum_duration=0.0, minimum_event_duration=0.04,
        speed_threshold=np.inf,
    )  # fmt: skip
    return rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_fraction=0.15, units=rec.place_cells
    )


@recipe(40, "Silva 2015", "MUA")
def silva_2015(rec):
    """Sorted units without interneurons (pyramidal; the Results say all
    recorded units, the Fig. 1c legend place cells), 10 ms Gaussian, > 3 SD,
    bounds at the mean, only while < 5 cm/s, 100-500 ms."""
    return rd.detect_events_from_trace(
        rec.time, rec.rate(rec.pyramidal, 0.010), rec.speed, rec.fs,
        threshold=3.0, minimum_duration=0.0, minimum_event_duration=0.1,
        maximum_duration=0.5, speed_rule="restrict", speed_threshold=5.0,
    )  # fmt: skip


@recipe(41, "Olafsdottir 2015", "MUA")
def olafsdottir_2015(rec):
    """Per template (two halves of the place cells stand in): >= 15% of its
    cells within <= 300 ms, bounded by >= 50 ms of silence (reading (a));
    >= 7 of its cells active for decoding. The rest periods are not separated:
    the whole session is searched."""
    found = []
    for template in (np.arange(60) < 20, (np.arange(60) >= 20) & (np.arange(60) < 40)):
        events = rd.detect_silence_bounded_events(
            rec.time, rec.multiunit, rec.fs,
            minimum_silence=0.05, units=template, minimum_active_fraction=0.15,
            maximum_duration=0.3,
        )  # fmt: skip
        found.append(
            bounds(
                rd.require_active_units(
                    events, rec.multiunit, rec.time, minimum_active_units=7, units=template
                )
            )
        )
    events = np.concatenate(found)
    return events[np.argsort(events[:, 0], kind="stable")]


@recipe(42, "Pfeiffer 2015", "SWR")
def pfeiffer_2015(rec):
    """See _pfeiffer_2015_swrs."""
    return _pfeiffer_2015_swrs(rec)


@recipe(43, "Wu 2014", "MUA")
def wu_2014(rec):
    """Place-cell density (per sample, for 10 ms bins), 15 ms Gaussian, > 2 SD
    over the session, bounds at the mean, speed < 5 at both ends (assumed; the
    paper does not say which samples). The reward-area restriction is not
    reproduced."""
    return rd.detect_events_from_trace(
        rec.time, rec.rate(rec.place_cells, 0.015), rec.speed, rec.fs,
        threshold=2.0, minimum_duration=0.0, speed_threshold=5.0,
    )  # fmt: skip


@recipe(44, "Wikenheiser 2013", "SWR")
def wikenheiser_2013(rec):
    """Rest branch only: 140-220 Hz power (mean of the tetrodes, assumed; the
    paper says "the LFP") above 1 SD of the session (assumed), 150 ms windows
    centred on every such sample, overlapping windows joined and clipped to
    the recording, >= 3 cells and >= 5 spikes, lying inside rest: >= 2 s
    (scaled from 30 s) below 2 cm/s with a z-scored theta (6-10 Hz) to delta
    (2-4 Hz) power ratio below 0. The run-LIA branch is not reproduced."""
    power = rd.normalize_signal(rec.mean_envelope((140.0, 220.0)) ** 2)
    windows = np.clip(
        rd.windows_around_times(rec.time[power >= 1.0], 0.075), *rec.time[[0, -1]]
    )
    events = rd.require_active_units(
        windows, rec.multiunit, rec.time, minimum_active_units=3, minimum_spikes=5
    )
    ratio = rd.normalize_signal(rec.ratio((6.0, 10.0), (2.0, 4.0), measure="power"))
    still = rd.state_intervals(rec.speed, rec.time, 2.0, minimum_duration=2.0)
    low = rd.state_intervals(ratio, rec.time, 0.0)
    rest = rec.mask_to_intervals(rec.intervals_to_mask(still) & rec.intervals_to_mask(low))
    return within_intervals(events, rest)


@recipe(45, "Pfeiffer 2013", "MUA")
def pfeiffer_2013(rec):
    """Clustered pyramidal units' histogram (interneurons excluded, inferred)
    only while < 5 cm/s, 10 ms Gaussian, > 3 SD, bounds at the mean; bounds
    moved inward until the first and last 20 ms windows (5 ms steps) hold 2
    spikes; then (order assumed) >= 10% of units and 50 ms-2 s."""
    events = rd.detect_events_from_trace(
        rec.time, rec.rate(rec.pyramidal, 0.010), rec.speed, rec.fs,
        threshold=3.0, minimum_duration=0.0, speed_rule="restrict", speed_threshold=5.0,
    )  # fmt: skip
    events = rd.trim_events_to_spike_windows(
        events, rec.multiunit, rec.time, units=rec.pyramidal
    )
    events = rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_fraction=0.1, units=rec.pyramidal
    )
    return within_duration(events, 0.05, 2.0)


@recipe(46, "Carr 2012", "SWR")
def carr_2012(rec):
    """The Karlsson rule on CA1 at < 4 cm/s; >= 5 place cells for candidates."""
    return rd.require_active_units(
        _karlsson_rule(rec, 4.0), rec.multiunit, rec.time,
        minimum_active_units=5, units=rec.place_cells,
    )  # fmt: skip


@recipe(47, "Bendor 2012", "MUA")
def bendor_2012(rec):
    """Davidson's multiunit signal (all spikes, 15 ms Gaussian), peak z >= 4,
    bounds z >= 2, merged < 50 ms, >= 50 ms; z over the whole session
    (assumed). The NREM, REM and awake labelling of events is not reproduced."""
    return rd.detect_events_from_trace(
        rec.time, rec.rate(None, 0.015), rec.speed, rec.fs,
        threshold=4.0, bound_threshold=2.0, minimum_duration=0.0,
        close_event_threshold=0.05, close_event_rule="merge", minimum_event_duration=0.05,
        speed_threshold=np.inf,
    )  # fmt: skip


@recipe(48, "Gupta 2010", "sequence (SWR gate)")
def gupta_2010(rec):
    """Events are windows grown by a spike-order score (not reproduced). This
    is the SWR gate: the 180-220 Hz Hilbert amplitude averaged over tetrodes
    (Jackson et al. 2006's measure, assumed) above 2 SD. The >= 3 active cells
    and the pause at a reward site are not reproduced."""
    return rd.detect_events_from_trace(
        rec.time, rec.mean_envelope((180.0, 220.0)), rec.speed, rec.fs,
        threshold=2.0, minimum_duration=0.0, speed_threshold=np.inf,
    )  # fmt: skip


@recipe(49, "Karlsson 2009", "SWR")
def karlsson_2009(rec):
    """The Karlsson rule at < 2 cm/s (CA1 and CA3 tetrodes)."""
    return _karlsson_rule(rec, 2.0)


@recipe(50, "Davidson 2009", "MUA")
def davidson_2009(rec):
    """All spikes, 15 ms Gaussian, peak >= 3 SD over stopping (< 5 cm/s),
    bounds at the mean, speed < 5 at both ends; within 30 s of running (RUN:
    speed > 15 cm/s), which keeps every event on this 90 s session."""
    events = rd.detect_events_from_trace(
        rec.time, rec.rate(None, 0.015), rec.speed, rec.fs,
        threshold=3.0, normalization_mask=rec.speed < 5, minimum_duration=0.0,
        speed_threshold=5.0,
    )  # fmt: skip
    running = rd.state_intervals(rec.speed, rec.time, 15.0, comparison=">")
    return rd.require_overlap(events, running + np.array([-30.0, 30.0]))


@recipe(51, "Diba 2007", "MUA")
def diba_2007(rec):
    """>= 60 ms of silence (of the template's cells, assumed), then >= 5 and
    >= 30% of the template's cells (whichever is greater) in the next 300 ms,
    speed <= 10 at both ends (assumed). All place cells stand in for a
    template (the paper's averaged 9); the track-end reward areas are not
    reproduced."""
    events = rd.detect_silence_bounded_events(
        rec.time, rec.multiunit, rec.fs,
        minimum_silence=0.06, window=0.3, units=rec.place_cells,
        minimum_active_units=5, minimum_active_fraction=0.3,
    )  # fmt: skip
    return rd.exclude_movement(events, rec.speed, rec.time, 10.0)


@recipe(52, "Ji 2007", "MUA")
def ji_2007(rec):
    """Pooled counts per 10 ms (per sample, scaled), 30 ms Gaussian, above T,
    the first minimum of their histogram over SWS (100 bins, 3-bin smoothing,
    assumed), bounded at T, frames with gaps < 80 ms joined (one of the
    per-animal 70-90 ms). SWS stand-in: speed < 4 and theta/delta < 1
    (assumed; the paper used EMG, ripples, theta and cortical delta)."""
    sleep = rec.sleep(4.0, 1.0)
    count = rd.gaussian_smooth(rec.counts(), 0.03, rec.fs) * rec.fs * 0.01
    level = rd.histogram_minimum_threshold(
        count[rec.intervals_to_mask(sleep)], bins=100, smoothing_window=3
    )
    return rd.detect_events_from_trace(
        rec.time, only_in(rec, count, sleep), rec.speed, rec.fs,
        threshold=level, bound_threshold=level, normalization_method="none",
        minimum_duration=0.0, close_event_threshold=0.08, close_event_rule="merge",
        speed_threshold=np.inf,
    )  # fmt: skip


@recipe(53, "Foster 2006", "MUA")
def foster_2006(rec):
    """Probe cells' spikes during stopping (< 5 cm/s, assumed) pooled and split
    at gaps of more than 50 ms, >= 1/3 of the cells, <= 500 ms. All place cells
    stand in for a probe sequence; the facing-direction rule is not
    reproduced."""
    stopped = rec.mask_to_intervals(rec.speed < 5)
    return rd.detect_silence_bounded_events(
        rec.time, only_in(rec, rec.multiunit, stopped), rec.fs,
        minimum_silence=0.05 + 1 / rec.fs, units=rec.place_cells,
        minimum_active_fraction=1 / 3, maximum_duration=0.5,
    )  # fmt: skip


@recipe(54, "Lee 2002", "MUA")
def lee_2002(rec):
    """Template cells' spikes in SWS (stand-in: speed < 4 and theta/delta < 1;
    the paper used a theta/total power ratio and video), each cell's bursts
    (ISI < 50 ms) collapsed to their first spike, split where letters are more
    than 100 ms apart. All place cells stand in for one direction's template."""
    sleep = rec.sleep(4.0, 1.0)
    return rd.detect_silence_bounded_events(
        rec.time, only_in(rec, rec.multiunit, sleep), rec.fs,
        minimum_silence=0.1 + 1 / rec.fs, maximum_isi=0.05, units=rec.place_cells,
    )  # fmt: skip


@recipe(55, "Nadasdy 1999", "SWR")
def nadasdy_1999(rec):
    """150-250 Hz power summed over electrodes (Roumis's trace, 4 ms
    smoothing, assumed for the unstated RMS window), 7 SD over the whole
    session (assumed for "background"), no minimum duration, during sleep
    (stand-in: speed < 4 (assumed) and a 5-10 / 2-4 Hz theta/delta ratio below 1
    (assumed))."""
    events = rd.Roumis_ripple_detector(
        rec.time, rec.filtered((150.0, 250.0)), rec.speed, rec.fs,
        zscore_threshold=7.0, minimum_duration=0.0, speed_threshold=np.inf,
    )  # fmt: skip
    return rd.require_overlap(events, rec.sleep(4.0, 1.0, theta=(5.0, 10.0), delta=(2.0, 4.0)))


@recipe(56, "Kudrimoti 1999", "SWR")
def kudrimoti_1999(rec):
    """One channel, 100-300 Hz amplitude above a threshold for >= 25 ms,
    bounded at it, during SWS (stand-in: speed < 4 and theta/delta < 1). The
    threshold is not reported: 3 SD of the envelope over SWS is assumed."""
    sleep = rec.sleep(4.0, 1.0)
    envelope = rec.envelope((100.0, 300.0))[:, 0]
    in_sleep = envelope[rec.intervals_to_mask(sleep)]
    level = float(in_sleep.mean() + 3 * in_sleep.std())
    return rd.detect_events_from_trace(
        rec.time, only_in(rec, envelope, sleep), rec.speed, rec.fs,
        threshold=level, bound_threshold=level, normalization_method="none",
        minimum_duration=0.025, speed_threshold=np.inf,
    )  # fmt: skip


NOT_REPRODUCED = {
    9: (
        "Widloski 2022",
        (
            "events are defined by decoding; ripple amplitude and spike density are "
            "reference traces only"
        ),
    ),
}


# --------------------------------------------------------------------------- run


def score(events: pd.DataFrame | np.ndarray, ripple_windows: np.ndarray) -> dict[str, float]:
    found = bounds(events)
    n_ripples = len(ripple_windows)
    return {
        "n_events": len(found),
        "recall": len(rd.require_overlap(ripple_windows, found)) / n_ripples
        if len(found)
        else 0.0,
        "false_positives": len(rd.exclude_overlap(found, ripple_windows)),
    }


def run_all(rec: Recording) -> pd.DataFrame:
    rows = []
    for entry in sorted(RECIPES, key=lambda entry: entry.row):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            events = entry.run(rec)
        rows.append(
            {
                "row": entry.row,
                "paper": entry.paper,
                "trigger": entry.trigger,
                **score(events, rec.session.ripple_windows),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    rec = make_recording()
    results = run_all(rec)
    results.to_csv(OUTPUT, index=False)
    with pd.option_context("display.width", 120, "display.max_rows", 100):
        print(results.to_string(index=False))
    print(f"\n{len(results)} recipes; not reproduced: {NOT_REPRODUCED}")


if __name__ == "__main__":
    main()

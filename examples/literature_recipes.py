"""One recipe per surveyed paper: its event rule written with this package.

Each function below reproduces, as closely as the package allows, the event
detection a paper in the literature survey (``load_literature_parameters()``)
describes. The parameters and their sources are in that paper's notes under
``docs/literature/papers/``, named by the survey's 0-based row; each recipe's
docstring names the row and says where it departs from the paper. Papers whose
events are defined by decoding have no recipe, only the detection they use to
label events where they have one.

Run on a simulated session (running bouts with theta, rest with delta, ripples
at rest with bursting units and a radiatum sharp wave), every recipe is
scored against the known ripples: how many it finds, the fraction of ripples
it overlaps and the events that overlap none. Simulated data say whether a
recipe runs and behaves, not whether it matches the paper's events.

Run with ``uv run python examples/literature_recipes.py``; it writes
``literature_recipes_results.csv`` beside this file.

Parameters a paper does not report are marked "assumed". The simulated
session is minutes long, so rules on state epochs of minutes are scaled down;
those are marked "scaled".
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
        self, theta: tuple[float, float] = (6.0, 12.0), delta: tuple[float, float] = (1.0, 4.0)
    ) -> np.ndarray:
        return rd.theta_delta_ratio(
            self.session.raw_lfp, self.fs, theta_band=theta, delta_band=delta
        )

    def intervals_to_mask(self, intervals: np.ndarray) -> np.ndarray:
        mask = np.zeros(self.time.size, dtype=bool)
        for start, end in np.asarray(intervals).reshape(-1, 2):
            mask |= (self.time >= start) & (self.time <= end)
        return mask

    def sleep(
        self,
        speed_below: float,
        ratio_below: float,
        minimum_duration: float,
        theta: tuple[float, float] = (6.0, 12.0),
        delta: tuple[float, float] = (1.0, 4.0),
    ) -> np.ndarray:
        """Intervals of stillness and low theta/delta, both lasting ``minimum_duration``."""
        still = rd.state_intervals(
            self.speed, self.time, speed_below, minimum_duration=minimum_duration
        )
        low_theta = rd.state_intervals(self.ratio(theta, delta), self.time, ratio_below)
        both = self.intervals_to_mask(still) & self.intervals_to_mask(low_theta)
        return rd.state_intervals(
            both.astype(float),
            self.time,
            0.5,
            comparison=">",
            minimum_duration=minimum_duration,
        )


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
    events = bounds(events)
    duration = events[:, 1] - events[:, 0]
    return events[(duration >= low - 1e-9) & (duration <= high + 1e-9)]


def only_in(rec: Recording, trace: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """The trace with the samples outside the intervals missing, so detection
    runs inside them only."""
    return np.where(rec.intervals_to_mask(intervals), trace, np.nan)


def zugaro_ripple_peaks(rec: Recording, band: tuple[float, float]) -> pd.DataFrame:
    """bz_FindRipples-like ripples, for the Buzsaki-lineage papers that require
    a ripple peak but do not describe their ripple detector (assumed: Huszar
    et al. 2022's 5 SD peak and 2 SD bounds, 20-200 ms)."""
    return rd.Zugaro_ripple_detector(
        rec.time, rec.filtered(band), rec.speed, rec.fs,
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
    the mean, peaks <= 70 ms apart merged, >= 10 cells. Read from the preprint
    and the code; the decoding criteria that follow are not reproduced."""
    events = rd.detect_events_from_trace(
        rec.time, rec.rate(rec.pyramidal, 0.0125), rec.speed, rec.fs,
        threshold=3.0, minimum_duration=0.0, speed_rule="restrict", speed_threshold=5.0,
    )  # fmt: skip
    merged = (
        rd.merge_close_events(events, 0.07, measure="peak") if len(events) else bounds(events)
    )
    return rd.require_active_units(
        merged, rec.multiunit, rec.time, minimum_active_units=10, units=rec.pyramidal
    )


@recipe(1, "Widloski 2025", "decoding (ripple label)")
def widloski_2025(rec):
    """Replays are defined by decoding (not reproduced). This is the ripple
    label: 100-220 Hz, one channel per tetrode, envelope smoothed with an
    80 ms Gaussian and averaged, z-scored over speed < 5, peak > 2 SD for
    >= 15 ms, bounds at the mean, merged < 50 ms (the text; the code merges
    none)."""
    return rd.detect_events_from_trace(
        rec.time, rec.mean_envelope((100.0, 220.0)), rec.speed, rec.fs,
        threshold=2.0, smoothing_sigma=0.08, normalization_mask=rec.speed < 5,
        minimum_duration=0.015, close_event_threshold=0.05, close_event_rule="merge",
        speed_threshold=5.0,
    )  # fmt: skip


def _population_with_ripple_peak(rec, sleep):
    """Grosmark 2016 and Yang 2024: pyramidal rate, 15 ms Gaussian, 3 SD with
    the statistics over NREM, bounds at the mean, 50-500 ms, >= 5 cells, and a
    ripple peak inside (ripple detector assumed, see zugaro_ripple_peaks)."""
    events = rd.detect_events_from_trace(
        rec.time, rec.rate(rec.pyramidal, 0.015), rec.speed, rec.fs,
        threshold=3.0, normalization_mask=rec.intervals_to_mask(sleep), minimum_duration=0.0,
        minimum_event_duration=0.05, maximum_duration=0.5, speed_threshold=np.inf,
    )  # fmt: skip
    events = rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_units=5, units=rec.pyramidal
    )
    return rd.require_times_inside(events, zugaro_ripple_peaks(rec, (130.0, 200.0)).peak_time)


@recipe(2, "Yang 2024", "SWR+MUA")
def yang_2024(rec):
    """As Grosmark 2016 (the text is shared); events in quiet waking or NREM,
    here stillness with low theta/delta (scaled: >= 2 s)."""
    return _population_with_ripple_peak(rec, rec.sleep(4.0, 1.0, 2.0))


def _tirole(rec):
    """Tirole 2022 / Huelin Gorriz 2023 (shared code): all spikes, Gaussian
    10 ms SD applied forward and back (about 14 ms), z >= 3, bounds at z < 0
    sought within 300 ms, relaxed to 0.25 then 0.5; events >= 100 ms, then
    merged < 50 ms, median speed <= 5, >= 5 place cells, and ripple-band power
    (best channel, 125-300 Hz, 15 ms moving average, as the code) z >= 3 inside."""
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
    power = uniform_filter1d(rec.filtered((125.0, 300.0))[:, 0] ** 2, round(0.015 * rec.fs))
    return rd.require_trace_peak(merged, rd.normalize_signal(power), rec.time, 3.0)


@recipe(3, "Huelin Gorriz 2023", "SWR+MUA")
def huelin_gorriz_2023(rec):
    """Tirole 2022's pipeline, which the paper's code calls unchanged."""
    return _tirole(rec)


@recipe(4, "Harvey 2023 (code)", "SWR")
def harvey_2023_code(rec):
    """The code's path for data without a radiatum channel: bz_FindRipples,
    thresholds 1 and 3, 100-250 Hz, 20-300 ms, events 50 ms apart merged."""
    return rd.Zugaro_ripple_detector(
        rec.time, rec.filtered((100.0, 250.0)), rec.speed, rec.fs,
        low_threshold=1.0, high_threshold=3.0, minimum_inter_ripple_interval=0.05,
        minimum_duration=0.02, maximum_duration=0.3, speed_threshold=np.inf,
    )  # fmt: skip


@recipe(4, "Harvey 2023 (text)", "SWR (needs radiatum)")
def harvey_2023_text(rec):
    """As written: 80-250 Hz power (rectified, low-passed at 55 Hz; a
    Butterworth band-pass stands in for the difference of Gaussians), events
    over 4 SD extended to 1 SD, >= 15 ms, kept when a radiatum sharp wave
    (5-40 Hz, negative, > 2.5 SD, 20-400 ms) overlaps. The period the SDs come
    from is not stated: stillness is assumed, as the lab's NREM baseline in
    Oliva et al. 2016 and Fernandez-Ruiz et al. 2019."""
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
    """DetectSWR on the pyramidal and radiatum channels; the candidates are
    pyramidal-cell bursts (10 ms Gaussian, > 2 SD, bounds at the mean,
    100-500 ms) overlapping an SWR. Manual curation is not reproduced."""
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
    ``threshold`` SD with statistics over speed < 5, bounds at the mean,
    50 ms to ``maximum_duration``."""
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
    session, peak > 0.35, bounds at 0.15, merged < 30 ms, >= 4 cells
    (candidates). Per-sample counts stand in for 10 ms bins."""
    smoothed = rd.gaussian_smooth(rec.counts(), 0.02, rec.fs)
    scaled = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
    events = rd.detect_events_from_trace(
        rec.time, scaled, rec.speed, rec.fs,
        threshold=0.35, bound_threshold=0.15, normalization_method="none",
        minimum_duration=0.0, close_event_threshold=0.03, close_event_rule="merge",
        speed_threshold=np.inf,
    )  # fmt: skip
    return rd.require_active_units(events, rec.multiunit, rec.time, minimum_active_units=4)


@recipe(12, "Berners-Lee 2021", "SWR")
def berners_lee_2021(rec):
    """Pfeiffer & Foster 2015's rule on 3 tetrodes at 2 SD."""
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
    """Michon 2019 / 2021 (Kloosterman lab): ripple envelope (140-225 Hz,
    mean of the tetrodes) and all-spike count, each 15 ms Gaussian and minus a
    3 s moving median; ripples peak > 8 SD, bounds 0.5 SD, merged < 20 ms,
    >= 40 ms; bursts peak > 4 SD, bounds 0.5 SD, >= 80 ms; candidates are
    bursts overlapping a ripple during immobility (< 5 cm/s)."""
    window = round(3.0 * rec.fs) | 1

    def detrended(trace):
        smoothed = rd.gaussian_smooth(trace, 0.015, rec.fs)
        return smoothed - median_filter(smoothed, size=window, mode="nearest")

    common = {
        "bound_threshold": 0.5, "minimum_duration": 0.0, "close_event_threshold": 0.02,
        "close_event_rule": "merge", "speed_threshold": np.inf,
    }  # fmt: skip
    ripples = rd.detect_events_from_trace(
        rec.time, detrended(rec.mean_envelope((140.0, 225.0))), rec.speed, rec.fs,
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
    """The candidate stage only: pyramidal rate (15 ms Gaussian) z-scored over
    stopping (< 5 cm/s), > 2 SD, bounds at the mean, > 4 cells, 50 ms-2 s. The
    GMM split on rate and ripple power that follows is under-specified."""
    events = rd.detect_events_from_trace(
        rec.time, rec.rate(rec.pyramidal, 0.015), rec.speed, rec.fs,
        threshold=2.0, normalization_mask=rec.speed < 5, minimum_duration=0.0,
        minimum_event_duration=0.05, maximum_duration=2.0, speed_threshold=5.0,
    )  # fmt: skip
    return rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_units=5, units=rec.pyramidal
    )


@recipe(17, "Gridchyn 2020 (fixed threshold)", "MUA")
def gridchyn_2020(rec):
    """All spikes counted in a 20 ms window, above 3.5 x their mean over the
    rest before the first run, bounds at that mean, 150 ms refractory (onset to
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
    SWR detector: ripple-band RMS over 240 ms, mean over tetrodes, 5 SD peak,
    1.5 SD bounds."""
    power = uniform_filter1d(rec.filtered((150.0, 250.0)) ** 2, round(0.24 * rec.fs), axis=0)
    return rd.detect_events_from_trace(
        rec.time, np.sqrt(power).mean(axis=1), rec.speed, rec.fs,
        threshold=5.0, bound_threshold=1.5, minimum_duration=0.0, speed_threshold=np.inf,
    )  # fmt: skip


@recipe(19, "Bhattarai 2020", "SWR+MUA")
def bhattarai_2020(rec):
    """SWRs: 100-250 Hz power in a 50 ms boxcar on 2 channels, peak > 3 SD,
    bounds 1 SD, > 20 ms, merged <= 100 ms, >= 5 place cells. Replays: > 60 ms
    of place-cell silence, then >= 5 place cells within 300 ms, coinciding
    with an SWR."""
    power = uniform_filter1d(
        rec.filtered((100.0, 250.0))[:, :2] ** 2, round(0.05 * rec.fs), axis=0
    ).mean(axis=1)
    swrs = rd.detect_events_from_trace(
        rec.time, power, rec.speed, rec.fs,
        threshold=3.0, bound_threshold=1.0, minimum_duration=0.0,
        minimum_event_duration=0.02 + 1 / rec.fs, speed_threshold=np.inf,
    )  # fmt: skip
    swrs = rd.merge_close_events(swrs, 0.1, inclusive=True) if len(swrs) else bounds(swrs)
    swrs = rd.require_active_units(
        swrs, rec.multiunit, rec.time, minimum_active_units=5, units=rec.place_cells
    )
    replays = rd.detect_silence_bounded_events(
        rec.time, rec.multiunit, rec.fs,
        minimum_silence=0.06, window=0.3, units=rec.place_cells, minimum_active_units=5,
    )  # fmt: skip
    return rd.require_overlap(replays, swrs)


@recipe(20, "Stella 2019", "SWR")
def stella_2019(rec):
    """Morlet-wavelet power in 150-250 Hz on each electrode, z-scored,
    maximum across electrodes, peak > 5 SD, bounds at 2 SD; REM excluded by a
    theta/delta ratio (cutoff not stated; assumed 1.5 here)."""
    frequencies = np.linspace(150.0, 250.0, 6)
    raw = rec.session.lfps
    power = np.zeros(raw.shape)
    for frequency in frequencies:
        sigma = 7 / (2 * np.pi * frequency)  # a 7-cycle wavelet
        t = np.arange(-4 * sigma, 4 * sigma, 1 / rec.fs)
        wavelet = np.exp(2j * np.pi * frequency * t) * np.exp(-(t**2) / (2 * sigma**2))
        wavelet /= np.abs(wavelet).sum()
        power += np.abs(fftconvolve(raw, wavelet[:, None], mode="same", axes=0)) ** 2
    zscored = rd.normalize_signal(np.sqrt(power / len(frequencies)))
    events = rd.detect_events_from_trace(
        rec.time, zscored.max(axis=1), rec.speed, rec.fs,
        threshold=5.0, bound_threshold=2.0, normalization_method="none",
        minimum_duration=0.0, speed_threshold=np.inf,
    )  # fmt: skip
    rem = rd.state_intervals(rec.ratio(), rec.time, 1.5, comparison=">")
    return rd.exclude_overlap(events, rem)


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
    trace = only_in(rec, rec.rate(rec.pyramidal, 0.015), sleep)
    events = rd.detect_events_from_trace(
        rec.time, trace, rec.speed, rec.fs,
        threshold=2.0, bound_threshold=2.0, minimum_duration=0.0,
        minimum_event_duration=0.1, maximum_duration=0.8, speed_threshold=np.inf,
    )  # fmt: skip
    return rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_units=5, units=units
    )


@recipe(22, "Farooq 2019 (Neuron)", "MUA")
def farooq_2019_neuron(rec):
    """SWS: speed < 1 cm/s (scaled: for >= 5 s, not 5 min) and theta/delta
    (6-12 / 1-4 Hz Hilbert) < 2; >= 5 neurons."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return _farooq(rec, rec.sleep(1.0, 2.0, 5.0), rec.pyramidal)


@recipe(23, "Farooq 2019 (Science)", "MUA")
def farooq_2019_science(rec):
    """SWS: speed < 2 cm/s and theta/delta (4-10 / 1-3 Hz) below its mean; >=
    5 place-responsive cells. The manual review is not reproduced."""
    ratio = rec.ratio((4.0, 10.0), (1.0, 3.0))
    still = rd.state_intervals(rec.speed, rec.time, 2.0)
    low = rd.state_intervals(ratio, rec.time, float(np.nanmean(ratio)))
    both = rec.intervals_to_mask(still) & rec.intervals_to_mask(low)
    sleep = rd.state_intervals(both.astype(float), rec.time, 0.5, comparison=">")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
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
    """Pyramidal spikes split at >= 100 ms of silence, >= 4 cells, 80 ms-1.2 s,
    inside SWS (speed < 1 cm/s and theta/delta < 2)."""
    events = rd.detect_silence_bounded_events(
        rec.time, rec.multiunit, rec.fs,
        minimum_silence=0.1, units=rec.pyramidal, minimum_active_units=4,
        minimum_duration=0.08, maximum_duration=1.2,
    )  # fmt: skip
    return rd.require_overlap(events, rec.sleep(1.0, 2.0, 2.0))


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
    score = rd.carey_spectral_ripple_score(rec.session.raw_lfp, rec.fs, examples)
    return rd.Carey_candidate_detector(
        rec.time, None, rec.multiunit, rec.speed, rec.fs,
        ripple_score=score, threshold_method="mean", low_threshold=4.0, high_threshold=4.0,
        theta_lfp=rec.session.raw_lfp,
    )  # fmt: skip


@recipe(29, "Muessig 2019", "SWR+MUA")
def muessig_2019(rec):
    """Bursts of pyramidal ("CS") cells, 10 ms Gaussian, crossing 3 SD,
    100-750 ms, overlapping an SWR: 7 ms moving RMS of 100-250 Hz on one
    tetrode above its 99th percentile, a 100 ms window at the peak; rest only
    (speed < 2.5 and theta/delta < 2)."""
    rms = np.sqrt(
        uniform_filter1d(rec.filtered((100.0, 250.0))[:, 0] ** 2, round(0.007 * rec.fs))
    )
    level = float(np.percentile(rms, 99))
    peaks = rd.detect_events_from_trace(
        rec.time, rms, rec.speed, rec.fs,
        threshold=level, bound_threshold=level, normalization_method="none",
        minimum_duration=0.0, speed_threshold=np.inf,
    )  # fmt: skip
    swr_windows = rd.windows_around_times(peaks.peak_time, 0.05)
    bursts = rd.detect_events_from_trace(
        rec.time, rec.rate(rec.pyramidal, 0.010), rec.speed, rec.fs,
        threshold=3.0, minimum_duration=0.0, minimum_event_duration=0.1,
        maximum_duration=0.75, speed_threshold=np.inf,
    )  # fmt: skip
    return rd.require_overlap(
        rd.require_overlap(bursts, swr_windows), rec.sleep(2.5, 2.0, 1.0)
    )


@recipe(30, "Drieu 2018", "MUA")
def drieu_2018(rec):
    """Place-cell rate, 10 ms Gaussian, peak > 3 SD, bounds at the mean,
    <= 500 ms, inside SWS found by k-means on theta/delta (6-10 / 1-4 Hz),
    epochs (scaled: > 2 s, not 120 s) with gaps < 1 s bridged."""
    ratio = rec.ratio((6.0, 10.0), (1.0, 4.0))
    sleep = rd.state_intervals(
        ratio, rec.time, rd.two_cluster_threshold(ratio), merge_gap=1.0, minimum_duration=2.0
    )
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
    speed above 3 cm/s in the event, >= 15% of place cells or more than 5.
    The corner restriction is not reproduced."""
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
    """All spikes in 10 ms bins (a 10 ms boxcar here), no smoothing, peak
    >= 4 SD, bounds at the mean, 50-400 ms."""
    count = uniform_filter1d(rec.counts(), round(0.01 * rec.fs))
    return rd.detect_events_from_trace(
        rec.time, count, rec.speed, rec.fs,
        threshold=4.0, minimum_duration=0.0, minimum_event_duration=0.05,
        maximum_duration=0.4, speed_threshold=np.inf,
    )  # fmt: skip


@recipe(34, "Yamamoto 2017 (one reading)", "SWR+MUA")
def yamamoto_2017(rec):
    """One reading of an ambiguous rule: summed spikes in 10 ms bins, peak
    > 3 SD, bounds at 1 SD, kept when inside a period of 140-200 Hz power
    above 3 SD on one channel. The paper does not say how the two combine or
    which trace sets the bounds."""
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
    """See _population_with_ripple_peak; NREM as stillness with low
    theta/delta (scaled: >= 2 s). The decoding stage's >= 100 ms and >= 5 or
    10% of place cells are applied too."""
    events = _population_with_ripple_peak(rec, rec.sleep(4.0, 1.0, 2.0))
    events = within_duration(events, low=0.1)
    return rd.require_active_units(
        events, rec.multiunit, rec.time,
        minimum_active_units=5, minimum_active_fraction=0.1, units=rec.place_cells,
    )  # fmt: skip


@recipe(37, "Ambrose 2016", "SWR")
def ambrose_2016(rec):
    """Pfeiffer & Foster 2015's trace on 4 tetrodes, > 3 SD during stopping;
    no duration limits (the survey's 50 and 500 ms are in no source). The
    proximity to the well is not reproduced."""
    return rd.detect_events_from_trace(
        rec.time, rec.mean_envelope((150.0, 250.0)), rec.speed, rec.fs,
        threshold=3.0, smoothing_sigma=0.0125, normalization_mask=rec.speed < 5,
        minimum_duration=0.0, speed_threshold=5.0,
    )  # fmt: skip


@recipe(38, "Jadhav 2016", "SWR")
def jadhav_2016(rec):
    """The Karlsson rule at < 4 cm/s; SWRs within 1 s after the previous
    one's start dropped; >= 4 CA1 cells for candidates."""
    events = rd.exclude_close_events(_karlsson_rule(rec, 4.0), 1.0, measure_from="start")
    return rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_units=4, units=rec.pyramidal
    )


@recipe(39, "Olafsdottir 2016", "MUA")
def olafsdottir_2016(rec):
    """Place cells, 5 ms Gaussian, > 3 SD, bounds at the mean, >= 40 ms,
    >= 15% of the place cells; no speed rule."""
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
    """Sorted units' density, 10 ms Gaussian, > 3 SD, bounds at the mean,
    only while < 5 cm/s, 100-500 ms."""
    return rd.detect_events_from_trace(
        rec.time, rec.rate(rec.place_cells, 0.010), rec.speed, rec.fs,
        threshold=3.0, minimum_duration=0.0, minimum_event_duration=0.1,
        maximum_duration=0.5, speed_rule="restrict", speed_threshold=5.0,
    )  # fmt: skip


@recipe(41, "Olafsdottir 2015", "MUA")
def olafsdottir_2015(rec):
    """Per template (here two halves of the place cells): >= 15% of its cells
    within <= 300 ms, bounded by >= 50 ms of silence; >= 7 active cells for
    decoding."""
    templates = [np.arange(60) < 20, (np.arange(60) >= 20) & (np.arange(60) < 40)]
    found = [
        bounds(
            rd.detect_silence_bounded_events(
                rec.time, rec.multiunit, rec.fs,
                minimum_silence=0.05, units=template, minimum_active_fraction=0.15,
                maximum_duration=0.3,
            )
        )
        for template in templates
    ]  # fmt: skip
    events = np.concatenate(found)
    events = events[np.argsort(events[:, 0], kind="stable")]
    return rd.require_active_units(events, rec.multiunit, rec.time, minimum_active_units=7)


@recipe(42, "Pfeiffer 2015", "SWR")
def pfeiffer_2015(rec):
    """See _pfeiffer_2015_swrs."""
    return _pfeiffer_2015_swrs(rec)


@recipe(43, "Wu 2014", "MUA")
def wu_2014(rec):
    """Place-cell density, 15 ms Gaussian, > 2 SD over the session, bounds at
    the mean, speed < 5. The reward-area restriction is not reproduced."""
    return rd.detect_events_from_trace(
        rec.time, rec.rate(rec.place_cells, 0.015), rec.speed, rec.fs,
        threshold=2.0, minimum_duration=0.0, speed_threshold=5.0,
    )  # fmt: skip


@recipe(44, "Wikenheiser 2013", "SWR")
def wikenheiser_2013(rec):
    """140-220 Hz power (mean of the tetrodes) above 1 SD; 150 ms windows
    centred on every such sample, overlapping windows joined; >= 3 cells and
    >= 5 spikes; at rest (scaled: >= 2 s below 2 cm/s) with low theta/delta."""
    power = rd.normalize_signal(rec.mean_envelope((140.0, 220.0)) ** 2)
    windows = rd.windows_around_times(rec.time[power >= 1.0], 0.075)
    windows = windows[(windows[:, 0] >= rec.time[0]) & (windows[:, 1] <= rec.time[-1])]
    events = rd.require_active_units(
        windows, rec.multiunit, rec.time, minimum_active_units=3, minimum_spikes=5
    )
    return rd.require_overlap(events, rec.sleep(2.0, 1.0, 2.0))


@recipe(45, "Pfeiffer 2013", "MUA")
def pfeiffer_2013(rec):
    """Clustered units' histogram only while < 5 cm/s, 10 ms Gaussian, > 3 SD,
    bounds at the mean; bounds moved inward until the first and last 20 ms
    windows (5 ms steps) hold 2 spikes; >= 10% of units; 50 ms-2 s."""
    events = rd.detect_events_from_trace(
        rec.time, rec.rate(None, 0.010), rec.speed, rec.fs,
        threshold=3.0, minimum_duration=0.0, speed_rule="restrict", speed_threshold=5.0,
    )  # fmt: skip
    events = rd.trim_events_to_spike_windows(events, rec.multiunit, rec.time)
    events = rd.require_active_units(
        events, rec.multiunit, rec.time, minimum_active_fraction=0.1
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
    bounds z >= 2, merged < 50 ms, >= 50 ms."""
    return rd.detect_events_from_trace(
        rec.time, rec.rate(None, 0.015), rec.speed, rec.fs,
        threshold=4.0, bound_threshold=2.0, minimum_duration=0.0,
        close_event_threshold=0.05, close_event_rule="merge", minimum_event_duration=0.05,
        speed_threshold=np.inf,
    )  # fmt: skip


@recipe(48, "Gupta 2010", "sequence (SWR gate)")
def gupta_2010(rec):
    """Events are windows grown by a spike-order score (not reproduced). This
    is the SWR gate: 180-220 Hz power, mean over tetrodes, above 2 SD."""
    return rd.detect_events_from_trace(
        rec.time, rec.mean_envelope((180.0, 220.0)) ** 2, rec.speed, rec.fs,
        threshold=2.0, minimum_duration=0.0, speed_threshold=np.inf,
    )  # fmt: skip


@recipe(49, "Karlsson 2009", "SWR")
def karlsson_2009(rec):
    """The Karlsson rule at < 2 cm/s (CA1 and CA3 tetrodes)."""
    return _karlsson_rule(rec, 2.0)


@recipe(50, "Davidson 2009", "MUA")
def davidson_2009(rec):
    """All spikes, 15 ms Gaussian, peak >= 3 SD over stopping (< 5 cm/s),
    bounds at the mean, during stopping; within 30 s of running."""
    events = rd.detect_events_from_trace(
        rec.time, rec.rate(None, 0.015), rec.speed, rec.fs,
        threshold=3.0, normalization_mask=rec.speed < 5, minimum_duration=0.0,
        speed_threshold=5.0,
    )  # fmt: skip
    near_running = np.asarray(RUNNING_INTERVALS) + np.array([-30.0, 30.0])
    return rd.require_overlap(events, near_running)


@recipe(51, "Diba 2007", "MUA")
def diba_2007(rec):
    """>= 60 ms of silence, then >= 5 (or > 30%) of the template's cells in
    the next 300 ms (the union of the two), at <= 10 cm/s."""
    common = {"minimum_silence": 0.06, "window": 0.3, "units": rec.place_cells}
    by_count = bounds(
        rd.detect_silence_bounded_events(
            rec.time, rec.multiunit, rec.fs, minimum_active_units=5, **common
        )
    )
    by_fraction = bounds(
        rd.detect_silence_bounded_events(
            rec.time, rec.multiunit, rec.fs, minimum_active_fraction=0.3, **common
        )
    )
    events = np.unique(np.concatenate([by_count, by_fraction]), axis=0)
    return rd.exclude_movement(events, rec.speed, rec.time, 10.0)


@recipe(52, "Ji 2007", "MUA")
def ji_2007(rec):
    """Pooled counts per 10 ms, 30 ms Gaussian, above T, the first minimum of
    their histogram, bounded at T, frames with gaps < 80 ms joined; SWS only."""
    count = rd.gaussian_smooth(rec.counts(), 0.03, rec.fs) * rec.fs * 0.01
    level = rd.histogram_minimum_threshold(count, bins=100, smoothing_window=3)
    return rd.detect_events_from_trace(
        rec.time, only_in(rec, count, rec.sleep(4.0, 1.0, 2.0)), rec.speed, rec.fs,
        threshold=level, bound_threshold=level, normalization_method="none",
        minimum_duration=0.0, close_event_threshold=0.08, close_event_rule="merge",
        speed_threshold=np.inf,
    )  # fmt: skip


@recipe(53, "Foster 2006", "MUA")
def foster_2006(rec):
    """Probe cells' pooled spikes split at gaps > 50 ms, >= 1/3 of the cells,
    <= 500 ms, while stopped (assumed < 5 cm/s)."""
    events = rd.detect_silence_bounded_events(
        rec.time, rec.multiunit, rec.fs,
        minimum_silence=0.05, units=rec.place_cells, minimum_active_fraction=1 / 3,
        maximum_duration=0.5,
    )  # fmt: skip
    return rd.exclude_movement(events, rec.speed, rec.time, 5.0)


@recipe(54, "Lee 2002", "MUA")
def lee_2002(rec):
    """Template cells' bursts (ISI < 50 ms) collapsed to their first spike,
    split where letters are > 100 ms apart; SWS only."""
    events = rd.detect_silence_bounded_events(
        rec.time, rec.multiunit, rec.fs,
        minimum_silence=0.1, maximum_isi=0.05, units=rec.place_cells,
    )  # fmt: skip
    return rd.require_overlap(events, rec.sleep(4.0, 1.0, 2.0))


@recipe(55, "Nadasdy 1999", "SWR")
def nadasdy_1999(rec):
    """150-250 Hz power summed over electrodes, 7 SD (Roumis's RMS-like
    trace), during sleep."""
    events = rd.Roumis_ripple_detector(
        rec.time, rec.filtered((150.0, 250.0)), rec.speed, rec.fs,
        zscore_threshold=7.0, speed_threshold=np.inf,
    )  # fmt: skip
    return rd.require_overlap(events, rec.sleep(4.0, 1.0, 2.0))


@recipe(56, "Kudrimoti 1999", "SWR")
def kudrimoti_1999(rec):
    """One channel, 100-300 Hz amplitude above a threshold for >= 25 ms,
    bounded at it. The threshold is not reported: 3 SD of the envelope is
    assumed."""
    envelope = rec.envelope((100.0, 300.0))[:, 0]
    level = float(envelope.mean() + 3 * envelope.std())
    return rd.detect_events_from_trace(
        rec.time, envelope, rec.speed, rec.fs,
        threshold=level, bound_threshold=level, normalization_method="none",
        minimum_duration=0.025, speed_threshold=np.inf,
    )  # fmt: skip


NOT_REPRODUCED = {
    9: ("Widloski 2022", "events are defined by decoding; ripples are reference traces only"),
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

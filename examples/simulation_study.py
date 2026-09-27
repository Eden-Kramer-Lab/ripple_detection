"""Compare the detectors on simulated sessions with known ripples.

Runs every detector at its defaults on ``simulate_session`` output over a grid of
ripple size (``ripple_snr``), channel count (4, 16, 32) and seed, plus:

- ripple-free sessions, for the false-positive rate;
- sessions where only a quarter or half of 32 channels carry the ripple;
- threshold sweeps for Kay, Karlsson and Zugaro, so the detectors can be compared
  at a matched false-positive rate rather than only at their defaults;
- sessions with a sparse population of 20 units, for the spike detectors;
- sessions with common-mode artifacts.

Writes one row per detector and condition to ``simulation_study_results.csv`` beside
this file; ``simulation_study.ipynb`` reads that file and plots it.

Run with ``uv run python examples/simulation_study.py``; about a quarter of an hour.
"""

from __future__ import annotations

import functools
import itertools
import time as wall_clock
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import ripple_detection as rd

SAMPLING_FREQUENCY = 1500.0
DURATION_S = 120.0
N_RIPPLES = 40
N_UNITS = 100
SPARSE_UNITS = 20
SNR_LEVELS = (1.5, 2.0, 3.0, 4.0, 6.0)
CHANNEL_COUNTS = (4, 16, 32)
CARRIER_FRACTIONS = (0.25, 0.5)  # of 32 channels; 1.0 is the main grid
SWEEP_CHANNELS = 16
THRESHOLD_SWEEPS = {
    "Kay_ripple_detector": ("zscore_threshold", (2.0, 2.5, 3.0, 3.5, 4.0, 5.0)),
    "Karlsson_ripple_detector": ("zscore_threshold", (2.0, 2.5, 3.0, 3.5, 4.0, 5.0)),
    "Zugaro_ripple_detector": ("high_threshold", (3.0, 4.0, 5.0, 6.0, 8.0)),
}
SEEDS = (0, 1, 2)
ARTIFACT_SNR = 3.0
N_ARTIFACTS = 20
OUTPUT = Path(__file__).with_name("simulation_study_results.csv")

LFP_DETECTORS = (
    "Kay_ripple_detector",
    "Karlsson_ripple_detector",
    "Roumis_ripple_detector",
    "Shvartsman_ripple_detector",
    "Yu_ripple_detector",
    "Zugaro_ripple_detector",
)


def event_times(rng: np.random.Generator, n_events: int) -> np.ndarray:
    """Event centres spread over the session, jittered, at least 0.4 of the mean
    spacing apart."""
    spacing = (DURATION_S - 4.0) / n_events
    return 2.0 + spacing * (np.arange(n_events) + 0.5 + rng.uniform(-0.3, 0.3, n_events))


def channel_gains(
    rng: np.random.Generator, n_channels: int, fraction: float = 1.0
) -> np.ndarray:
    """Channel 0 at gain 1 (the ``ripple_snr`` reference), the other carriers drawn
    between 0.5 and 1, and the rest silent; ``fraction`` of the channels carry."""
    gains = np.concatenate([[1.0], rng.uniform(0.5, 1.0, n_channels - 1)])
    n_carriers = max(1, round(fraction * n_channels))
    gains[n_carriers:] = 0.0
    return gains


def score(events: pd.DataFrame, windows: np.ndarray) -> dict[str, float]:
    """``events`` matched one-to-one to the true ``windows``: recall, precision and F1,
    the pairs' median onset and offset errors (event minus window, in ms, so negative
    is early), the ripples overlapped by two or more events (split) and the events
    overlapping two or more ripples (merged)."""
    matching = rd.match_events(windows, events)
    return {
        "recall": matching.recall,
        "precision": matching.precision,
        "f1": matching.f1,
        "onset_ms": 1000 * matching.pairs.onset_error.median(),  # NaN with no pairs
        "offset_ms": 1000 * matching.pairs.offset_error.median(),
        "n_split": len(matching.split_reference),
        "n_merged": len(matching.merged_detected),
    }


def call(detector, args: tuple, **params) -> tuple[pd.DataFrame, str, float]:
    """Run a detector; an error is recorded, not raised."""
    started = wall_clock.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            events = detector(*args, **params)
        error = ""
    except ValueError as exc:
        events, error = pd.DataFrame(columns=["start_time", "end_time"]), str(exc)[:120]
    return events, error, wall_clock.perf_counter() - started


def detector_calls(session: rd.SimulatedSession, filtered: np.ndarray) -> dict[str, tuple]:
    fs = session.sampling_frequency
    calls: dict[str, tuple] = {
        name: (rd.DETECTORS[name].detector, (session.time, filtered, session.speed, fs))
        for name in LFP_DETECTORS
    }
    calls["Long_sharp_wave_ripple_detector"] = (
        functools.partial(
            rd.Long_sharp_wave_ripple_detector, sharp_wave_lfp=session.sharp_wave_lfp
        ),
        (session.time, session.raw_lfp, session.speed, fs),
    )
    calls["Carey_candidate_detector"] = (
        rd.Carey_candidate_detector,
        (session.time, filtered, session.multiunit, session.speed, fs),
    )
    calls["multiunit_HSE_detector"] = (
        rd.multiunit_HSE_detector,
        (session.time, session.multiunit, session.speed, fs),
    )
    return calls


def run_defaults(session: rd.SimulatedSession) -> list[dict[str, object]]:
    """Every detector at its defaults on one session."""
    filtered = rd.filter_ripple_band(
        session.lfps, sampling_frequency=session.sampling_frequency
    )
    rows = []
    for name, (detector, args) in detector_calls(session, filtered).items():
        events, error, runtime = call(detector, args)
        rows.append(
            {
                "detector": name,
                "runtime_s": runtime,
                "n_events": len(events),
                "error": error,
                **score(events, session.ripple_windows),
            }
        )
    return rows


def run_sweeps(session: rd.SimulatedSession) -> list[dict[str, object]]:
    """The threshold sweeps on one session."""
    filtered = rd.filter_ripple_band(
        session.lfps, sampling_frequency=session.sampling_frequency
    )
    calls = detector_calls(session, filtered)
    rows = []
    for name, (parameter, values) in THRESHOLD_SWEEPS.items():
        detector, args = calls[name]
        for value in values:
            events, error, runtime = call(detector, args, **{parameter: value})
            rows.append(
                {
                    "detector": name,
                    "threshold_parameter": parameter,
                    "threshold_value": value,
                    "runtime_s": runtime,
                    "n_events": len(events),
                    "error": error,
                    **score(events, session.ripple_windows),
                }
            )
    return rows


def main() -> None:
    time = rd.simulate_time(int(DURATION_S * SAMPLING_FREQUENCY), SAMPLING_FREQUENCY)
    rows: list[dict[str, object]] = []

    def record(condition: dict[str, object], session_rows: list[dict[str, object]]) -> None:
        rows.extend({**condition, **row} for row in session_rows)

    for snr, n_channels, seed in itertools.product(SNR_LEVELS, CHANNEL_COUNTS, SEEDS):
        rng = np.random.default_rng(seed)
        session = rd.simulate_session(
            time,
            list(event_times(rng, N_RIPPLES)),
            n_channels=n_channels,
            n_units=N_UNITS,
            channel_gains=channel_gains(rng, n_channels),
            ripple_snr=snr,
            rng=seed,
        )
        condition = {
            "condition": "ripples",
            "ripple_snr": snr,
            "n_channels": n_channels,
            "carrier_fraction": 1.0,
            "seed": seed,
        }
        record(condition, run_defaults(session))
        if n_channels == SWEEP_CHANNELS and snr in (3.0, 4.0):
            record({**condition, "condition": "threshold sweep"}, run_sweeps(session))
        print(f"ripples  snr={snr:<4} channels={n_channels:<3} seed={seed}", flush=True)

    for n_channels, seed in itertools.product(CHANNEL_COUNTS, SEEDS):
        rng = np.random.default_rng(100 + seed)
        session = rd.simulate_session(
            time,
            [],
            n_channels=n_channels,
            n_units=N_UNITS,
            channel_gains=channel_gains(rng, n_channels),
            rng=100 + seed,
        )
        condition = {
            "condition": "noise only",
            "ripple_snr": np.nan,
            "n_channels": n_channels,
            "carrier_fraction": np.nan,
            "seed": seed,
        }
        record(condition, run_defaults(session))
        if n_channels == SWEEP_CHANNELS:
            record({**condition, "condition": "threshold sweep"}, run_sweeps(session))
        print(f"noise    channels={n_channels:<3} seed={seed}", flush=True)

    for fraction, snr, seed in itertools.product(CARRIER_FRACTIONS, (3.0, 6.0), SEEDS):
        rng = np.random.default_rng(400 + seed)
        n_channels = max(CHANNEL_COUNTS)
        session = rd.simulate_session(
            time,
            list(event_times(rng, N_RIPPLES)),
            n_channels=n_channels,
            n_units=N_UNITS,
            channel_gains=channel_gains(rng, n_channels, fraction),
            ripple_snr=snr,
            rng=400 + seed,
        )
        record(
            {
                "condition": "partial carriers",
                "ripple_snr": snr,
                "n_channels": n_channels,
                "carrier_fraction": fraction,
                "seed": seed,
            },
            run_defaults(session),
        )
        print(f"carriers fraction={fraction} snr={snr:<4} seed={seed}", flush=True)

    for seed in SEEDS:
        rng = np.random.default_rng(300 + seed)
        session = rd.simulate_session(
            time,
            list(event_times(rng, N_RIPPLES)),
            n_channels=SWEEP_CHANNELS,
            n_units=SPARSE_UNITS,
            channel_gains=channel_gains(rng, SWEEP_CHANNELS),
            ripple_snr=4.0,
            rng=300 + seed,
        )
        record(
            {
                "condition": "sparse population",
                "ripple_snr": 4.0,
                "n_channels": SWEEP_CHANNELS,
                "carrier_fraction": 1.0,
                "seed": seed,
            },
            run_defaults(session),
        )
        print(f"sparse   seed={seed}", flush=True)

    for seed in SEEDS:
        rng = np.random.default_rng(200 + seed)
        centres = event_times(rng, N_RIPPLES + N_ARTIFACTS)
        which = rng.permutation(len(centres)) < N_RIPPLES
        session = rd.simulate_session(
            time,
            list(centres[which]),
            n_channels=SWEEP_CHANNELS,
            n_units=N_UNITS,
            channel_gains=channel_gains(rng, SWEEP_CHANNELS),
            ripple_snr=ARTIFACT_SNR,
            artifact_times=list(centres[~which]),
            rng=200 + seed,
        )
        record(
            {
                "condition": "artifacts",
                "ripple_snr": ARTIFACT_SNR,
                "n_channels": SWEEP_CHANNELS,
                "carrier_fraction": 1.0,
                "seed": seed,
            },
            run_defaults(session),
        )
        print(f"artifact seed={seed}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT, index=False)
    print(f"wrote {len(results)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()

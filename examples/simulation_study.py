"""Compare the detectors on simulated sessions with known ripples.

Runs every detector at its defaults on ``simulate_session`` output over a grid of
ripple size (``ripple_snr``), channel count and seed, plus ripple-free sessions for
the false-positive rate, sessions with a sparse population of 20 units for the spike
detectors, and sessions with common-mode artifacts, and writes one row per detector
and condition to ``simulation_study_results.csv`` beside this file.
``simulation_study.ipynb`` reads that file and plots it.

Run with ``uv run python examples/simulation_study.py``; a few minutes on a laptop.
"""

from __future__ import annotations

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
CHANNEL_COUNTS = (1, 4, 8)
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


def event_times(rng: np.random.Generator, n_events: int, minimum_gap: float) -> np.ndarray:
    """Event centres, uniform over the session and at least ``minimum_gap`` apart."""
    while True:
        centres = np.sort(rng.uniform(2.0, DURATION_S - 2.0, n_events))
        if n_events < 2 or np.all(np.diff(centres) >= minimum_gap):
            return centres


def score(events: pd.DataFrame, windows: np.ndarray) -> dict[str, float]:
    """Recall, precision and boundary errors of ``events`` against true ``windows``."""
    n_true = len(windows)
    if len(events) == 0:
        return {
            "recall": 0.0 if n_true else np.nan,
            "precision": np.nan,
            "onset_ms": np.nan,
            "offset_ms": np.nan,
        }
    starts, ends = events.start_time.to_numpy(), events.end_time.to_numpy()
    matched_true = np.zeros(n_true, dtype=bool)
    matched_event = np.zeros(len(events), dtype=bool)
    onsets, offsets = [], []
    for i, (true_start, true_end) in enumerate(windows):
        overlaps = (starts <= true_end) & (ends >= true_start)
        if overlaps.any():
            matched_true[i] = True
            matched_event |= overlaps
            onsets.append((starts[overlaps].min() - true_start) * 1000)
            offsets.append((ends[overlaps].max() - true_end) * 1000)
    return {
        "recall": matched_true.mean() if n_true else np.nan,
        "precision": matched_event.mean(),
        "onset_ms": float(np.median(onsets)) if onsets else np.nan,
        "offset_ms": float(np.median(offsets)) if offsets else np.nan,
    }


def run_detectors(session: rd.SimulatedSession) -> list[dict[str, object]]:
    """Every detector at its defaults on one session; errors are recorded, not raised."""
    fs = session.sampling_frequency
    filtered = rd.filter_ripple_band(session.lfps, sampling_frequency=fs)
    calls: dict[str, tuple] = {
        name: (rd.DETECTORS[name].detector, (session.time, filtered, session.speed, fs))
        for name in LFP_DETECTORS
    }
    calls["Long_sharp_wave_ripple_detector"] = (
        rd.Long_sharp_wave_ripple_detector,
        (session.time, session.raw_lfp_pair, session.speed, fs),
    )
    calls["Carey_candidate_detector"] = (
        rd.Carey_candidate_detector,
        (session.time, filtered, session.multiunit, session.speed, fs),
    )
    calls["multiunit_HSE_detector"] = (
        rd.multiunit_HSE_detector,
        (session.time, session.multiunit, session.speed, fs),
    )
    rows = []
    for name, (detector, args) in calls.items():
        started = wall_clock.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                events = detector(*args)
            error = ""
        except ValueError as exc:
            events, error = pd.DataFrame(columns=["start_time", "end_time"]), str(exc)[:120]
        rows.append(
            {
                "detector": name,
                "runtime_s": wall_clock.perf_counter() - started,
                "n_events": len(events),
                "error": error,
                **score(events, session.ripple_windows),
            }
        )
    return rows


def main() -> None:
    time = rd.simulate_time(int(DURATION_S * SAMPLING_FREQUENCY), SAMPLING_FREQUENCY)
    rows: list[dict[str, object]] = []

    def record(condition: dict[str, object], session: rd.SimulatedSession) -> None:
        rows.extend({**condition, **row} for row in run_detectors(session))

    for snr, n_channels, seed in itertools.product(SNR_LEVELS, CHANNEL_COUNTS, SEEDS):
        rng = np.random.default_rng(seed)
        centres = event_times(rng, N_RIPPLES, 0.5)
        gains = np.concatenate([[1.0], rng.uniform(0.5, 1.0, n_channels - 1)])
        session = rd.simulate_session(
            time,
            list(centres),
            n_channels=n_channels,
            n_units=N_UNITS,
            channel_gains=gains,
            ripple_snr=snr,
            random_state=seed,
        )
        record(
            {
                "condition": "ripples",
                "ripple_snr": snr,
                "n_channels": n_channels,
                "seed": seed,
            },
            session,
        )
        print(f"ripples  snr={snr:<4} channels={n_channels} seed={seed}", flush=True)

    for n_channels, seed in itertools.product(CHANNEL_COUNTS, SEEDS):
        rng = np.random.default_rng(100 + seed)
        gains = np.concatenate([[1.0], rng.uniform(0.5, 1.0, n_channels - 1)])
        session = rd.simulate_session(
            time,
            [],
            n_channels=n_channels,
            n_units=N_UNITS,
            channel_gains=gains,
            random_state=100 + seed,
        )
        record(
            {
                "condition": "noise only",
                "ripple_snr": np.nan,
                "n_channels": n_channels,
                "seed": seed,
            },
            session,
        )
        print(f"noise    channels={n_channels} seed={seed}", flush=True)

    for seed in SEEDS:
        rng = np.random.default_rng(300 + seed)
        centres = event_times(rng, N_RIPPLES, 0.5)
        gains = np.concatenate([[1.0], rng.uniform(0.5, 1.0, 3)])
        session = rd.simulate_session(
            time,
            list(centres),
            n_channels=4,
            n_units=SPARSE_UNITS,
            channel_gains=gains,
            ripple_snr=4.0,
            random_state=300 + seed,
        )
        record(
            {
                "condition": "sparse population",
                "ripple_snr": 4.0,
                "n_channels": 4,
                "seed": seed,
            },
            session,
        )
        print(f"sparse   seed={seed}", flush=True)

    for seed in SEEDS:
        rng = np.random.default_rng(200 + seed)
        centres = event_times(rng, N_RIPPLES + N_ARTIFACTS, 0.5)
        which = rng.permutation(len(centres)) < N_RIPPLES
        gains = np.concatenate([[1.0], rng.uniform(0.5, 1.0, 3)])
        session = rd.simulate_session(
            time,
            list(centres[which]),
            n_channels=4,
            n_units=N_UNITS,
            channel_gains=gains,
            ripple_snr=ARTIFACT_SNR,
            artifact_times=list(centres[~which]),
            random_state=200 + seed,
        )
        record(
            {
                "condition": "artifacts",
                "ripple_snr": ARTIFACT_SNR,
                "n_channels": 4,
                "seed": seed,
            },
            session,
        )
        print(f"artifact seed={seed}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT, index=False)
    print(f"wrote {len(results)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()

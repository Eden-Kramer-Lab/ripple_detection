"""Run published detection methods on your own recording, step by step.

A walkthrough of ``ripple_detection.literature_methods`` with measured data:

1. Prepare inputs: raw LFP on one clock, spike times binned to counts,
   speed resampled onto the LFP timestamps, channels and cells selected.
2. Mind the memory: every signal is held as float64.
3. Curate intervals: a normalization baseline, sleep scoring, and the
   behavioral epochs a particular method needs.
4. Discover methods and check a call before running it.
5. Choose an analysis stage, run several methods, read their diagnostics.
6. Save the events with their provenance.

The "recording" is simulated so the script runs anywhere in a few seconds; the
``load_*`` functions stand in for your own loaders and return what those
would: arrays on your acquisition system's clock. Run it with

    uv run python examples/measured_walkthrough.py [output_directory]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import ripple_detection as rd
from ripple_detection.literature_methods import (
    Recording,
    check_method,
    list_methods,
    load_events,
    run_method,
    save_events,
)

# --------------------------------------------------------------------------
# Stand-ins for your loaders. A 60 s session: rest (0-20 s), running on a track
# (20-40 s), rest again (40-60 s), on a Unix-time acquisition clock.

CLOCK_ORIGIN = 1_700_000_000.0  # seconds; any clock works if every stream shares it
LFP_RATE = 1500.0  # Hz
DURATION = 60.0  # s
_SESSION = rd.simulate_session(
    np.arange(int(DURATION * LFP_RATE)) / LFP_RATE,
    [3.0, 7.0, 11.0, 15.0, 44.0, 48.0, 52.0, 56.0],
    n_channels=4,
    n_units=40,
    baseline_rate=np.r_[np.full(30, 0.5), np.full(10, 3.0)],
    ripple_rate_gain=40.0,
    ripple_snr=8.0,
    running_intervals=[(20.0, 40.0)],
    theta_amplitude=4.0,
    delta_amplitude=4.0,
    rng=7,
)


def load_lfp() -> tuple[np.ndarray, np.ndarray]:
    """Timestamps (s) and raw broadband LFP, shape (n_time, n_channels), int16."""
    time = CLOCK_ORIGIN + _SESSION.time
    # Channel 2 is the pyramidal-layer channel with the largest ripples here.
    raw = _SESSION.lfps[:, [1, 2, 0, 3]] * 100
    return time, raw.astype(np.int16)


def load_spike_times() -> list[np.ndarray]:
    """One array of spike times (s) per sorted unit."""
    rng = np.random.default_rng(0)
    samples, units = np.nonzero(_SESSION.multiunit)
    counts = _SESSION.multiunit[samples, units].astype(int)
    samples, units = np.repeat(samples, counts), np.repeat(units, counts)
    times = CLOCK_ORIGIN + _SESSION.time[samples] + rng.uniform(0, 1 / LFP_RATE, len(samples))
    return [np.sort(times[units == unit]) for unit in range(_SESSION.multiunit.shape[1])]


def load_position_speed() -> tuple[np.ndarray, np.ndarray]:
    """Video frame times (s) and speed (cm/s) at 30 Hz; the camera starts 0.5 s late."""
    frames = np.arange(0.5, DURATION, 1 / 30)
    return CLOCK_ORIGIN + frames, np.interp(frames, _SESSION.time, _SESSION.speed)


# --------------------------------------------------------------------------
# 1. Prepare inputs.


def spike_counts(time: np.ndarray, spike_times: list[np.ndarray]) -> np.ndarray:
    """Count each unit's spikes per LFP sample, shape (n_time, n_units).

    A spike belongs to the sample at or before it, and only while it falls
    within that sample's period: up to the next timestamp, or one median step
    across a gap (a step over 1.5 median steps, the detectors' rule) and after
    the last sample. Spikes before the first sample, after the last sample's
    period, or inside a gap (spike sorting often covers more than the selected
    LFP) are not counted.
    """
    step = np.diff(time)
    sample_period = np.median(step)
    period_end = np.append(
        np.where(step <= 1.5 * sample_period, time[1:], time[:-1] + sample_period),
        time[-1] + sample_period,
    )
    counts = np.zeros((len(time), len(spike_times)), dtype=np.uint8)
    for unit, times in enumerate(spike_times):
        sample = np.searchsorted(time, times, side="right") - 1
        recorded = sample >= 0
        recorded[recorded] = times[recorded] < period_end[sample[recorded]]
        np.add.at(counts[:, unit], sample[recorded], 1)
    return counts


def prepare_recording() -> Recording:
    """Everything on the LFP's timestamps, as Recording.from_arrays expects."""
    time, raw_lfp = load_lfp()

    # Raw, not ripple-filtered, LFP: each method applies its own band and
    # filter (100-250, 150-250, 80-250 Hz...). Filtered input would be filtered
    # twice. Select channels before building the recording: methods that use
    # one channel take the FIRST, so put the pyramidal-layer channel first.
    lfps = raw_lfp[:, [1, 0, 2]]  # three CA1 pyramidal-layer channels, best first

    # Spike counts, not rates: one integer count per unit per LFP sample. A
    # spike belongs to the sample at or before it; finer spike timing than the
    # LFP's sampling is not kept.
    counts = spike_counts(time, load_spike_times())

    # Speed on the LFP timestamps, NaN where the camera saw nothing: unknown
    # speed never passes a speed rule, and without speed at all, methods with
    # a speed rule refuse to run instead of guessing.
    frame_time, frame_speed = load_position_speed()
    speed = np.interp(time, frame_time, frame_speed, left=np.nan, right=np.nan)

    # Cell selections are yours: boolean masks (or unit indices) over the
    # columns of counts. Place cells come from your place-field analysis;
    # "pyramidal" excludes putative interneurons. Methods name which they use.
    unit_rates = counts.sum(axis=0) / DURATION
    pyramidal = unit_rates < 2.0  # stand-in for your cell classification
    place_cells = pyramidal & (np.arange(counts.shape[1]) < 25)  # stand-in

    # 3. Curated intervals, sorted [start, end] pairs in seconds on the same clock.
    # baseline_intervals: the epoch a method normalizes over when it asks for a
    # caller-selected baseline (here the pre-task rest).
    # sleep_intervals: your sleep scoring (NREM/SWS), for methods restricted to
    # sleep; nothing is inferred from speed for measured data. The two rest
    # epochs stand in for scored NREM here.
    # behavior_intervals are not part of the recording: see main().
    rest = np.array([[0.0, 20.0], [40.0, 60.0]]) + CLOCK_ORIGIN
    return Recording.from_arrays(
        time,
        LFP_RATE,
        lfps=lfps,
        multiunit=counts,
        speed=speed,
        pyramidal=pyramidal,
        place_cells=place_cells,
        baseline_intervals=rest[:1],
        sleep_intervals=rest,
        # artifact_intervals=... marks every signal missing there
    )


# --------------------------------------------------------------------------
# 2. Memory: the recording holds float64 copies, so NaN can mark missing
# samples. Here: 90,000 samples x (3 channels + 40 units) x 8 bytes = 31 MB.
# One hour at 1500 Hz with 100 units is 4.32 GB of counts alone
# (5.4 million samples x 100 x 8 bytes), whatever the input's type.


def main(output_directory: Path) -> dict[str, pd.DataFrame]:
    recording = prepare_recording()
    n_bytes = recording.multiunit.nbytes + recording.session.lfps.nbytes
    print(f"Recording holds {n_bytes / 1e6:.0f} MB of float64 signals.")

    # 4. Discover methods. Each row says what the method needs: signals (and
    # which LFP channels), cells, intervals and what they must mean, options
    # without a published value, a fixed rate, stages and the bin grid.
    catalog = list_methods().set_index("name")
    print(catalog.loc[catalog.paper.str.startswith("Pfeiffer"), ["output", "role"]])
    print(
        "shin_2019 needs:",
        catalog.loc["shin_2019", "signals"],
        catalog.loc["shin_2019", "cells"],
    )

    # Check before running: every missing input at once, nothing run.
    for problem in check_method("yang_2024", recording):
        print("yang_2024 lacks", problem)

    # behavior_intervals belong to one call: their meaning differs per method.
    # Chenani 2019 needs the reward zones: here, the track end where the animal
    # stopped after running (a stand-in for your curated zones).
    reward_zones = np.array([[40.0, 50.0]]) + CLOCK_ORIGIN
    assert check_method("chenani_2019", recording, behavior_intervals=reward_zones) == []

    # 5. Choose a stage: "detection" (the default) is the initial inventory;
    # "decoding_candidates" applies the paper's filters before decoding.
    results = {
        "karlsson_2009": run_method("karlsson_2009", recording),
        "pfeiffer_2013": run_method("pfeiffer_2013", recording),
        "shin_2019_candidates": run_method(
            "shin_2019", recording, stage="decoding_candidates"
        ),
        "ji_2007": run_method("ji_2007", recording),
        "chenani_2019": run_method("chenani_2019", recording, behavior_intervals=reward_zones),
    }

    # Every result starts with the same columns, numbered from 1; attrs say
    # what ran. Read the diagnostics before changing anything: where were the
    # signals valid, what did the intervals cover, and which step lost events?
    for name, events in results.items():
        diagnostics = events.attrs["diagnostics"]
        steps = ", ".join(f"{d['step']}: {d['events']}" for d in diagnostics["detections"])
        print(
            f"{name}: {len(events)} events ({events.attrs['role']}, "
            f"grid {events.attrs['grid']['bin_width'] or 'input samples'}); "
            f"speed known {diagnostics['signals']['speed']['valid_fraction']:.0%}; "
            f"detections {steps}; "
            f"{diagnostics['events_before_behavior_intervals']} before behavior_intervals"
        )

    # 6. Save the events with their provenance: CSV plus a JSON sidecar
    # (package version, DOI, resolved options, inputs, grid, diagnostics).
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, events in results.items():
        save_events(events, output_directory / f"{name}.csv")
    restored = load_events(output_directory / "karlsson_2009.csv")
    assert restored.attrs["doi"] == results["karlsson_2009"].attrs["doi"]
    print(f"Saved {len(results)} inventories to {output_directory}")
    return results


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("walkthrough_events"))

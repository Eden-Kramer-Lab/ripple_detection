"""Run the packaged literature methods on a simulated session.

For measured inputs use ripple_detection.literature_methods.Recording.from_arrays.
Simulation overlap measures exercise the code, not agreement with historical events.
"""

import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import ripple_detection as rd
from ripple_detection import literature_methods as methods

OUTPUT = Path(__file__).with_name("literature_recipes_results.csv")

SAMPLING_FREQUENCY = 1500.0

RUNNING_INTERVALS = [(12.0, 24.0), (40.0, 52.0), (70.0, 80.0)]

RIPPLE_TIMES = [
    3.0,
    5.5,
    8.0,
    10.0,
    27.0,
    30.0,
    33.0,
    36.0,
    55.0,
    58.0,
    61.0,
    64.0,
    67.0,
    83.0,
    86.0,
    88.0,
]


def make_recording(duration: float = 90.0, rng: int = 0) -> methods.Recording:
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
        ripple_snr=8.0,  # also exercises the 240 ms FFT-window ripple label
        ripple_duration=(0.05, 0.2),
        sharp_wave_amplitude=6.0,  # the largest deflection at rest in radiatum, above delta
        running_intervals=running,
        theta_amplitude=4.0,
        delta_amplitude=4.0,
        rng=rng,
    )
    units = np.arange(60)
    return methods.Recording(
        session,
        place_cells=units < 40,
        pyramidal=units < 50,
        # Initial rest epoch, including its simulated events. Used only by methods
        # that explicitly request a baseline; no quiet-only selection by truth.
        baseline_intervals=np.array([[time[0], min(RUNNING_INTERVALS[0][0], time[-1])]]),
        reference_lfp=np.zeros_like(time),
        templates=(units < 20, (units >= 20) & (units < 40)),
    )


def score(events: pd.DataFrame | np.ndarray, ripple_windows: np.ndarray) -> dict[str, float]:
    found = methods.bounds(events)
    n_ripples = len(ripple_windows)
    return {
        "n_events": len(found),
        "recall": len(rd.require_overlap(ripple_windows, found)) / n_ripples
        if len(found)
        else 0.0,
        "false_positives": len(rd.exclude_overlap(found, ripple_windows)),
    }


def run_all(rec: methods.Recording) -> pd.DataFrame:
    rows = []
    for entry in sorted(methods.RECIPES, key=lambda entry: entry.row):
        # Kaefer's baseline epoch is unspecified. This FFT demonstration uses
        # the initial two seconds, which exercise its positive detection path.
        # Gridchyn and the other explicit-baseline methods keep the initial rest
        # epoch. Neither choice overrides methods with their own normalization.
        method_rec = rec
        if entry.run.__name__ == "kaefer_2020":
            method_rec = replace(
                rec,
                baseline_intervals=np.array(
                    [[rec.time[0], min(rec.time[0] + 2.0, rec.time[-1])]]
                ),
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            events = entry.run(method_rec)
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
    print(f"\n{len(results)} recipes; not reproduced: {methods.NOT_REPRODUCED}")


if __name__ == "__main__":
    main()

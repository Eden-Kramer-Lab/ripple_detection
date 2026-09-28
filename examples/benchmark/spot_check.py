"""Look at single events of a benchmark run before trusting its aggregates.

Re-simulates one session of a finished run from its saved parameters and seed, and plots,
for each event type, six of its true events: the ripple-band and radiatum signals,
the spikes, the truth windows of each expression at every fraction of
``TRUTH_FRACTIONS``, and the events Kay, Karlsson, the HSE detector and two recipes
found there, read from the run's ``events.csv.gz``. Misaligned windows, events in
the wrong units or events missing where the signal plainly holds one show up here
first. One PNG per event type goes to ``<run directory>/spot_check/``.

Usage, from the repository root::

    uv run python examples/benchmark/spot_check.py --run-name smoke
        [--session reference/0] [--per-type 6]
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from conditions import TRUTH_FRACTIONS, parameters_from_json, session_seed, simulate_parameters
from run import EXPRESSIONS, OUTPUT, read_table, truth_window_sets

import ripple_detection as rd

if TYPE_CHECKING:
    from matplotlib.figure import SubFigure

# The methods drawn beside the truth: three detectors at their defaults, a
# ripple recipe and a population-burst recipe.
METHODS = (
    ("Kay_ripple_detector", "default"),
    ("Karlsson_ripple_detector", "default"),
    ("multiunit_HSE_detector", "default"),
    ("recipe:jadhav_2016", "literature"),
    ("recipe:mallory_2025", "literature"),
)
# Seconds shown on each side of an event's network window.
_MARGIN = 0.15
_COLORS = {
    "ripple": "#0072B2",
    "sharp_wave": "#D55E00",
    "burst": "#009E73",
    "network": "#555555",
}


def load_session(
    run_directory: Path, session_id: str
) -> tuple[rd.SimulatedSession, pd.DataFrame]:
    """A run's session, simulated again, and the events the run found in it.

    Parameters
    ----------
    run_directory : pathlib.Path
        ``examples/benchmark/output/<run_name>``.
    session_id : str
        ``"{condition_id}/{replicate}"``.

    Returns
    -------
    session : SimulatedSession
    events : pandas.DataFrame
        The session's rows of the condition's ``events.csv.gz``.

    Raises
    ------
    ValueError
        The saved parameters are not a full set (``parameters_from_json``),
        or the run's seed for the session is not the replicate's, so the
        re-simulated session would not be the one the methods saw.
    """
    condition_id, replicate = session_id.rsplit("/", 1)
    listed = read_table(run_directory / "conditions.csv").set_index("condition_id")
    parameters = parameters_from_json(listed.loc[condition_id, "params"])
    directory = run_directory / "conditions" / condition_id
    sessions = read_table(directory / "sessions.csv.gz").set_index("session_id")
    if int(sessions.loc[session_id, "seed"]) != session_seed(int(replicate)):
        msg = (
            f"{session_id}: the run's seed is not replicate {replicate}'s, so the "
            "session cannot be simulated again."
        )
        raise ValueError(msg)
    events = read_table(directory / "events.csv.gz")
    return (
        simulate_parameters(parameters, int(replicate)),
        events[events.session_id == session_id],
    )


def chosen_events(events: pd.DataFrame, event_type: str, per_type: int) -> list[int]:
    """Up to ``per_type`` event ids of ``event_type``, spread over the session.

    Parameters
    ----------
    events : pandas.DataFrame
        A latent event table.
    event_type : str
    per_type : int

    Returns
    -------
    ids : list of int
        Evenly spaced through the session's events of that type, in order.
    """
    ids = np.unique(events.loc[events.event_type == event_type, "event_id"])
    if not len(ids):
        return []
    picks = np.unique(np.linspace(0, len(ids) - 1, min(per_type, len(ids))).round())
    return [int(ids[int(pick)]) for pick in picks]


def _draw_event(
    figure: SubFigure,
    session: rd.SimulatedSession,
    filtered: np.ndarray[Any, Any],
    windows: Mapping[str, Sequence[pd.DataFrame]],
    found: pd.DataFrame,
    event_id: int,
) -> None:
    """One event: signals, spikes, then the truth and each method's events."""
    network = windows["network"][0].set_index("id").loc[event_id]
    start, end = network.start_time - _MARGIN, network.end_time + _MARGIN
    shown = (session.time >= start) & (session.time <= end)
    time = session.time[shown]
    signal, spikes, lanes = figure.subplots(
        3, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1.5, 2]}
    )
    ripple = filtered[shown, 0] / np.std(filtered[:, 0])
    radiatum = session.sharp_wave_lfp[shown] / np.std(session.sharp_wave_lfp)
    signal.plot(time, ripple + 4, color=_COLORS["ripple"], linewidth=0.6)
    signal.plot(time, radiatum, color=_COLORS["sharp_wave"], linewidth=0.6)
    signal.set_yticks([0, 4], ["radiatum (SD)", "ripple band (SD)"])
    signal.set_title(f"event {event_id}, network peak {network.peak_time:.3f} s", fontsize=8)
    order = np.argsort(session.unit_types, kind="stable")
    rows, units = np.nonzero(session.multiunit[shown][:, order])
    spikes.scatter(time[rows], units, s=1, color="black", marker="|")
    spikes.set_yticks([])
    spikes.set_ylabel("units", fontsize=7)

    labels = []
    for lane, expression in enumerate(EXPRESSIONS):
        for rank, truth in enumerate(windows[expression]):
            for row in truth[truth.id == event_id].itertuples():
                lanes.plot(
                    [row.start_time, row.end_time],
                    [lane, lane],
                    color=_COLORS[expression],
                    linewidth=2 + 2 * rank,
                    alpha=0.3 + 0.3 * rank,
                    solid_capstyle="butt",
                )
        labels.append(f"truth {expression}")
    for lane, (method, setting) in enumerate(METHODS, start=len(EXPRESSIONS)):
        rows_found = found[(found.method == method) & (found.setting == setting)]
        near = rows_found[(rows_found.end_time >= start) & (rows_found.start_time <= end)]
        for row in near.itertuples():
            lanes.plot(
                [row.start_time, row.end_time], [lane, lane], color="black", linewidth=3
            )
            lanes.plot([row.peak_time], [lane], marker="|", color="white", markersize=6)
        labels.append(method.removeprefix("recipe:").removesuffix("_ripple_detector"))
    lanes.set_yticks(range(len(labels)), labels, fontsize=6)
    lanes.set_ylim(len(labels) - 0.5, -0.5)
    lanes.set_xlim(start, end)
    lanes.set_xlabel("time (s)", fontsize=7)
    for axes in (signal, spikes, lanes):
        axes.tick_params(labelsize=6)


def plot_session(run_directory: Path, session_id: str, per_type: int = 6) -> list[Path]:
    """Write one PNG per event type of a run's session.

    Parameters
    ----------
    run_directory : pathlib.Path
    session_id : str
    per_type : int, optional
        Events per type.

    Returns
    -------
    paths : list of pathlib.Path
        The PNGs, in ``run_directory / "spot_check"``; none for a type the
        session lacks.
    """
    import matplotlib.pyplot as plt

    session, found = load_session(run_directory, session_id)
    filtered = rd.filter_ripple_band(session.lfps, session.sampling_frequency)
    windows = truth_window_sets(session.events)
    directory = run_directory / "spot_check"
    directory.mkdir(exist_ok=True)
    paths = []
    for event_type in rd.EVENT_TYPES:
        ids = chosen_events(session.events, event_type, per_type)
        if not ids:
            continue
        n_rows = (len(ids) + 1) // 2
        figure = plt.figure(figsize=(12, 4.5 * n_rows), layout="constrained")
        figure.suptitle(
            f"{session_id}: {event_type}; truth at fractions "
            f"{', '.join(map(str, TRUTH_FRACTIONS))} (thin to thick)",
            fontsize=10,
        )
        blocks = figure.subfigures(n_rows, 2, squeeze=False).ravel()
        for block, event_id in zip(blocks, ids, strict=False):
            _draw_event(block, session, filtered, windows, found, event_id)
        path = directory / f"{session_id.replace('/', '_')}_{event_type}.png"
        figure.savefig(path, dpi=110)
        plt.close(figure)
        paths.append(path)
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    """The command line; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--session", default="reference/0", help="{condition_id}/{replicate}")
    parser.add_argument("--per-type", type=int, default=6)
    args = parser.parse_args(argv)
    for path in plot_session(OUTPUT / args.run_name, args.session, args.per_type):
        print(path)


if __name__ == "__main__":
    main()

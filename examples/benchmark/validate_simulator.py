"""Validate the benchmark's network simulator before any detector runs on it.

Simulates validation replicates of the benchmark's conditions (indices 10000 up,
through ``conditions.session_seed``, apart from the benchmark's own replicates),
measures the rendered sessions and compares the measurements with the targets
in ``simulator_targets.csv``. It uses the public simulator and signal helpers
only: no detector, recipe or literature method is imported or called.

    uv run python examples/benchmark/validate_simulator.py --validation-id v1
        [--conditions all|ID,ID] [--replicates 20] [--duration S] [--workers N]
        [--output-root DIR] [--no-figures]

writes ``<output-root>/<validation-id>/`` (by default
``examples/benchmark/validation/v1/``):

- ``spec.json``: the resolved simulation parameters of every validated
  condition, the reference's recorded revisions (``REFERENCE_REVISIONS``), the
  replicates and their seeds, package versions,
  ``simulation_fingerprint``, ``target_table_hash``, ``status`` (``"ready"`` or
  ``"not_ready"``) with the reasons, and the SHA-256 of every other file here.
- ``measurements.csv``: one row per condition, replicate, group and measured
  quantity (``condition_id, replicate, group, quantity, statistic, value, n``).
- ``checks.csv``: one row per check and condition (``check, kind,
  condition_id, applies, evidence_status, statistic, observed, lower, upper,
  n, passed, note``). ``kind`` is ``"target"`` (a row of the target table,
  pooled over the replicates) or ``"rendering"`` (the simulator does what its
  contract says); ``applies`` says whether the row gates readiness.
- ``report.md`` and small PNGs.

The report is ready when every rendering check passes in every condition it
applies to (``rendering_check_applies``: a channel-profile check needs more
than one channel, the gamma check gamma bursts, the refractory check that
spike model), with something measured and no replicate's value NaN, and every
target whose evidence is ``supported`` passes in the conditions its
``conditions`` column names. Targets that are ``assumed``, or whose state differs from the
reference's (``conditions`` "none"), and every condition the column does not
name, are reported without gating. ``require_ready_report`` is the check a
benchmark run makes before its first detector call.

Measured sessions are rendered again with the same rendering seed and options:
with an empty event table and no non-events (the matched noise-only
rendering: the renderer's fixed random streams make its noise identical), and
with groups of ripples, sharp waves or gamma bursts whose windows do not
overlap (isolated components, so a doublet's ripples or a long ripple's
neighbour are measured apart). A rendering minus the noise-only rendering is
the noise-free signal. The measurement conventions are listed in
``MEASUREMENT_CHOICES``, the rendering checks in ``RENDERING_CHECKS``, and both
are repeated in the report.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import resource
import shutil
import sys
import time as clock
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import scipy
from conditions import (
    REFERENCE_REVISIONS,
    Condition,
    ReferenceRevision,
    resolve,
    select_conditions,
    session_seed,
    simulate_condition,
)
from numpy.typing import ArrayLike
from scipy import signal, stats

import ripple_detection as rd
from ripple_detection.core import BoolArray, FloatArray, IntArray, filter_ripple_band

HERE = Path(__file__).resolve().parent
TARGETS = HERE / "simulator_targets.csv"
OUTPUT_ROOT = HERE / "validation"
CONDITIONS_SOURCE = HERE / "conditions.py"
PACKAGE = Path(rd.__file__).resolve().parent

FIRST_REPLICATE = 10000
# The predeclared replicates per condition; fewer cannot back a run.
DEFAULT_REPLICATES = 20
TRUTH_FRACTIONS = (0.1, 0.25, 0.5)

# Measurement conventions, in seconds and hertz.
REST_EDGE = 1.0  # rest leaves out the recording's first and last second, as the draws do
RMS_BAND = (100.0, 250.0)
RMS_WINDOW = 0.017
RMS_THRESHOLD_SD = 2.0
PATEL_PEAK_SD = 5.0
PATEL_MINIMUM_DURATION = 0.020
FREQUENCY_OFFSET = 0.0075
FFT_RESOLUTION = 0.25
GAIN_WINDOW = 0.005
PARTICIPATION_WINDOW = 0.025
COUNT_BIN = 0.010
POWER_WINDOW = 10.0
LOCAL_SNR_WINDOW = 10.0
EXAMPLE_HALF_WIDTH = 0.25
ISOLATION_MARGIN = 0.06  # beyond eight side scales: the Hilbert window's padding and more
PSD_MAXIMUM = 400.0
PSD_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (6.0, 10.0),
    "gamma": (60.0, 100.0),
    "fast_gamma": (90.0, 140.0),
    "ripple": (150.0, 250.0),
}

# Rendering-check tolerances, set from the renderer's contract and numerical
# precision, not from measured sessions.
SIZING_SPREAD = 1e-6  # relative: every ripple is sized against one noise SD
SNR_TOLERANCE = 0.05  # relative: a band SD of noise-only data stands for the stationary SD
# relative residual of a channel against the anchor waveform shifted and scaled by its
# stored delay and gain; a delay 0.1 sample off leaves about 8% at 200 Hz
PROFILE_TOLERANCE = 0.01
MODULATION_TOLERANCE = 0.05  # log amplitude
RATE_Z = 4.0  # standard errors of a pooled spike count
MINIMUM_MODULATION_PERIODS = 2.0
ROUNDING = 1e-9  # relative: a statistic on a bound, to rounding, lies within it

# The target table's labels of the six alternative models, and their condition ids.
MODEL_LABELS = {
    "coupled": "strength_correlation=coupled",
    "local": "spatial_profile=local",
    "varying": "noise_modulation=varying",
    "nearby": "fast_gamma_band=nearby",
    "refractory": "spike_model=refractory",
    "quartic": "envelope_power=quartic",
}

# Each target quantity: the sampled quantity it pools and the groups (event
# types) it pools over, None for all.
TARGET_SAMPLES: dict[str, tuple[str, tuple[str, ...] | None]] = {
    "ripple_event_rate": ("ripple_event_rate", None),
    "ripple_peak_frequency": ("ripple_peak_frequency", None),
    "ripple_frequency_decline": ("ripple_frequency_decline", None),
    "ripple_duration": ("ripple_duration", ("swr",)),
    "ripple_duration_sleep": ("ripple_duration_sleep", None),
    "sharp_wave_duration": ("sharp_wave_duration", None),
    "pyramidal_baseline_rate": ("pyramidal_baseline_rate", None),
    "interneuron_baseline_rate": ("interneuron_baseline_rate", None),
    "interneuron_ripple_gain": ("interneuron_ripple_gain", None),
    "pyramidal_ripple_gain": ("pyramidal_ripple_gain", None),
    "observed_participation": ("observed_participation", None),
    "observed_participation_largest": ("observed_participation", None),
    "sharp_wave_ripple_power_correlation": (
        "sharp_wave_ripple_power",
        ("swr", "ripple_doublet"),
    ),
    "sharp_wave_ripple_power_correlation_control": (
        "sharp_wave_ripple_power",
        ("swr", "ripple_doublet"),
    ),
    "sharp_wave_frequency_relation": (
        "sharp_wave_ripple_frequency",
        ("swr", "ripple_doublet"),
    ),
    "doublet_spacing": ("doublet_spacing", None),
}

# How each measurement reads the target table's `measurement` text where the
# text leaves a choice; the report lists these.
MEASUREMENT_CHOICES = (
    (
        "Rest is the recording less its first and last second and the running bouts; rates "
        "over rest count its samples, the event rate its intervals' lengths."
    ),
    (
        "The noise-only rendering has no events and no non-events: the background is the "
        "noise and the slow field. The noise-free signal is a rendering minus it."
    ),
    (
        "Isolated components: ripples, sharp waves and gamma bursts are rendered in groups "
        "whose windows (eight side scales, the local delay and 60 ms more each side) do not "
        "overlap, so a doublet's ripples, or a long ripple and its neighbour, are measured "
        "apart. Leaving rows out redraws the carrier phases and local spatial draws of the "
        "rows after the first left out, so each isolated ripple is measured against its own "
        "rendering's ripple_channels; frequencies and widths do not depend on the phase."
    ),
    (
        "Ripple peak frequency: the largest |FFT| of the isolated ripple on its anchor "
        "channel (gain largest, delay 0) over its eight-scale window, zero-padded to "
        f"{FFT_RESOLUTION} Hz resolution."
    ),
    (
        "Ripple frequency decline: the Hilbert phase derivative of the isolated anchor "
        f"waveform at the samples nearest the latent centre minus and plus "
        f"{FREQUENCY_OFFSET * 1e3:g} ms (the modulation envelope's peak)."
    ),
    (
        f"Ripple duration: channel 0 of the session filtered {RMS_BAND[0]:g}-{RMS_BAND[1]:g} "
        f"Hz with filter_ripple_band, its RMS in a centred window of round("
        f"{RMS_WINDOW} fs) samples, and the run strictly above the mean plus "
        f"{RMS_THRESHOLD_SD:g} SD of the noise-only rendering's RMS (whole session) that "
        "holds the sample nearest the ripple's centre; its length is last minus first "
        "sample time (closed bounds). A ripple whose RMS is not above the threshold at "
        "its centre counts as never crossing."
    ),
    (
        f"Sleep ripple duration (Patel et al.): the same run, kept when at least "
        f"{PATEL_MINIMUM_DURATION * 1e3:g} ms long and reaching the mean plus "
        f"{PATEL_PEAK_SD:g} SD; over every ripple component."
    ),
    (
        "Sharp-wave width: the run of the isolated radiatum deflection at or above 10% of "
        "its sampled peak that holds the peak, last minus first sample time."
    ),
    (
        "Firing rates, ripple gains and participation count the session's spikes, leaked "
        "spikes included. Ripple-gain windows are the samples within "
        f"{GAIN_WINDOW * 1e3:g} ms of each ripple centre of swr and ripple_doublet events; "
        "the baseline is rest outside every event's network window at 10% of peak."
    ),
    (
        "Observed participation: place and other pyramidal units with a spike within "
        f"{PARTICIPATION_WINDOW * 1e3:g} ms of the first ripple's centre (a 50 ms window); "
        "latent recruitment (n_participants) is reported separately and never used for it."
    ),
    (
        "Correlations and doublet spacing read the rendered session's event table: the "
        "latent amplitudes, SNRs and centres the renderer drew the signals from."
    ),
    (
        f"Background: Welch PSD of noise-only channel 0 (1 s segments); ripple-band power "
        f"in consecutive {POWER_WINDOW:g} s windows; the noise modulation's log amplitude "
        "from a least-squares sinusoid at the configured period through the windows' log "
        "power, divided by 2 and by the window's averaging gain sinc(W/T)."
    ),
    (
        "SNR: nominal is the table's amplitude; anchor SNR the filtered peak of the "
        "isolated anchor waveform over the ripple-band SD of noise-only channel 0; "
        "recording-wide SNR the mean over channels of each channel's filtered peak over "
        f"that channel's noise SD; event-local SNR uses the SD within "
        f"{LOCAL_SNR_WINDOW / 2:g} s of the ripple's centre."
    ),
    (
        "Spike counts: variance over mean of each unit's counts in 10 ms bins wholly at "
        "rest; inter-spike intervals and population silent gaps (between consecutive "
        "spikes of any unit of the population) within each stretch of rest."
    ),
)


class ReportNotReady(ValueError):
    """Every reason a report cannot back a run, in one message."""


# ---------------------------------------------------------------------------
# Fingerprints


# Package modules whose code changes no simulated value: error messages only.
_UNFINGERPRINTED = ("ripple_detection._call_hints",)
_PACKAGE_NAME = "ripple_detection"


def _module_path(package: Path, module: str) -> Path:
    """The source file of ``module``, a ``ripple_detection`` module."""
    parts = module.split(".")[1:]
    path = package.joinpath(*parts)
    return path / "__init__.py" if path.is_dir() else path.with_suffix(".py")


def _package_imports(tree: ast.Module) -> dict[str, tuple[str, str]]:
    """Names a module imports from the package: local name -> (module, name)."""
    imports: dict[str, tuple[str, str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(_PACKAGE_NAME):
            for alias in node.names:
                imports[alias.asname or alias.name] = (str(node.module), alias.name)
    return imports


def _definitions(tree: ast.Module, text: str) -> dict[str, tuple[str, ast.AST]]:
    """A module's top-level definitions: name -> (source with decorators, node)."""
    lines = text.splitlines(keepends=True)
    found: dict[str, tuple[str, ast.AST]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
            first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            first = node.lineno
        else:
            continue
        source = "".join(lines[first - 1 : node.end_lineno])
        for name in names:
            found[name] = (source, node)
    return found


def _package_aliases(tree: ast.Module) -> set[str]:
    """Local names bound to the package itself (``import ripple_detection as rd``)."""
    return {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == _PACKAGE_NAME
    }


def _used_names(node: ast.AST, aliases: set[str]) -> tuple[set[str], set[str]]:
    """Names ``node`` reads, and attributes it reads from a package alias."""
    names, attributes = set(), set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif (
            isinstance(child, ast.Attribute)
            and isinstance(child.value, ast.Name)
            and child.value.id in aliases
        ):
            attributes.add(child.attr)
    return names, attributes


def _simulation_sources(package: Path, conditions_source: Path) -> list[tuple[str, bytes]]:
    """The code a benchmark session's values depend on, as (label, bytes).

    ``conditions.py`` and ``simulate.py`` whole, the shipped filter kernel, and,
    from every other package module, the top-level definitions they use and
    those definitions use in turn (following imports within the package, except
    the modules in ``_UNFINGERPRINTED``). Detector code a session does not reach
    is left out, so editing it changes no fingerprint.
    """
    sources = [
        ("conditions.py", conditions_source.read_bytes()),
        (f"{_PACKAGE_NAME}/simulate.py", (package / "simulate.py").read_bytes()),
        (f"{_PACKAGE_NAME}/ripplefilter.mat", (package / "ripplefilter.mat").read_bytes()),
    ]
    parsed: dict[str, tuple[ast.Module, str]] = {}

    def parse(module: str) -> tuple[ast.Module, str]:
        if module not in parsed:
            text = _module_path(package, module).read_text()
            parsed[module] = (ast.parse(text), text)
        return parsed[module]

    pending: list[tuple[str, str]] = []
    whole = {
        "conditions": conditions_source,
        f"{_PACKAGE_NAME}.simulate": package / "simulate.py",
    }
    init_imports = _package_imports(parse(_PACKAGE_NAME)[0])
    for path in whole.values():
        tree = ast.parse(path.read_text())
        pending.extend(_package_imports(tree).values())
        _, attributes = _used_names(tree, _package_aliases(tree))
        pending.extend(
            init_imports[name] for name in sorted(attributes) if name in init_imports
        )
    included: dict[tuple[str, str], str] = {}
    while pending:
        module, name = pending.pop()
        if module in whole.keys() | set(_UNFINGERPRINTED) or (module, name) in included:
            continue
        tree, text = parse(module)
        definitions = _definitions(tree, text)
        imports = _package_imports(tree)
        if name in imports:
            pending.append(imports[name])
            included[module, name] = f"from {imports[name][0]} import {imports[name][1]}"
            continue
        if name not in definitions:  # a module attribute set another way, or a builtin
            continue
        source, node = definitions[name]
        included[module, name] = source
        names, _ = _used_names(node, _package_aliases(tree))
        pending.extend(
            (module, used)
            for used in sorted(names - {name})
            if used in definitions or used in imports
        )
    sources.extend(
        (f"{module}:{name}", included[module, name].encode())
        for module, name in sorted(included)
    )
    return sources


def simulation_fingerprint() -> str:
    """SHA-256 of the simulation code and the signal helpers it uses.

    Covers ``conditions.py``, ``ripple_detection/simulate.py``, the shipped
    filter kernel and the package functions and constants those two reach
    (``filter_ripple_band`` and its helpers, for instance), but no detector a
    session does not call, so a change confined to detectors leaves it as it is.

    Returns
    -------
    fingerprint : str
        64 hexadecimal digits.
    """
    digest = hashlib.sha256()
    for label, content in _simulation_sources(PACKAGE, CONDITIONS_SOURCE):
        digest.update(label.encode() + b"\0" + hashlib.sha256(content).digest())
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_table_hash(path: Path = TARGETS) -> str:
    """SHA-256 of the target table's bytes.

    Parameters
    ----------
    path : pathlib.Path, optional
        Default ``simulator_targets.csv`` beside this script.

    Returns
    -------
    digest : str
    """
    return _file_hash(Path(path))


# ---------------------------------------------------------------------------
# Measurement helpers


def _time_tolerance(time: FloatArray) -> float:
    """How far a bound may round from a timestamp: a few ulps of the largest
    timestamp, at least a nanosecond."""
    return max(1e-9, 4 * float(np.spacing(np.max(np.abs(time)))))


def interval_union(intervals: FloatArray) -> FloatArray:
    """The union of closed intervals, sorted and disjoint.

    Parameters
    ----------
    intervals : ndarray, shape (n_intervals, 2)

    Returns
    -------
    union : ndarray, shape (n_union, 2)
        Touching or overlapping intervals merged.
    """
    bounds = np.asarray(intervals, dtype=float).reshape(-1, 2)
    if not len(bounds):
        return bounds
    bounds = bounds[np.argsort(bounds[:, 0], kind="stable")]
    running_end = np.maximum.accumulate(bounds[:, 1])
    starts = np.flatnonzero(np.r_[True, bounds[1:, 0] > running_end[:-1]])
    return np.column_stack([bounds[starts, 0], np.maximum.reduceat(bounds[:, 1], starts)])


def interval_mask(time: FloatArray, intervals: FloatArray) -> BoolArray:
    """Samples inside any interval, bounds included to the timestamps' rounding.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    intervals : ndarray, shape (n_intervals, 2)
        In any order, overlapping or not.

    Returns
    -------
    mask : ndarray of bool, shape (n_time,)
    """
    union = interval_union(intervals)
    if not len(union):
        return np.zeros(time.shape, dtype=bool)
    tolerance = _time_tolerance(time)
    which = np.searchsorted(union[:, 0], time + tolerance, side="right") - 1
    inside = (which >= 0) & (time <= union[np.clip(which, 0, None), 1] + tolerance)
    return np.asarray(inside, dtype=bool)


def rest_intervals(time: FloatArray, bouts: FloatArray, edge: float = REST_EDGE) -> FloatArray:
    """The recording less ``edge`` seconds at each end and the running bouts.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    bouts : ndarray, shape (n_bouts, 2)
        Running bouts, sorted and disjoint, in the timestamps' clock.
    edge : float, optional

    Returns
    -------
    rest : ndarray, shape (n_rest, 2)
    """
    start, end = float(time[0]) + edge, float(time[-1]) - edge
    edges = np.r_[start, np.clip(np.asarray(bouts, dtype=float).ravel(), start, end), end]
    rest = edges.reshape(-1, 2)
    return np.asarray(rest[rest[:, 1] > rest[:, 0]], dtype=float)


def run_around(
    values: FloatArray, index: int, level: float, *, strict: bool = False
) -> tuple[int, int] | None:
    """The run of samples at or above ``level`` (above, if ``strict``) that
    holds ``index``: its first and last sample, or None if ``index`` is not in
    one.

    Parameters
    ----------
    values : ndarray, shape (n_time,)
    index : int
    level : float
    strict : bool, optional

    Returns
    -------
    run : (int, int) or None
    """
    above = values > level if strict else values >= level
    if not above[index]:
        return None
    before = np.flatnonzero(~above[:index])
    after = np.flatnonzero(~above[index:])
    first = int(before[-1]) + 1 if before.size else 0
    last = index + int(after[0]) - 1 if after.size else values.size - 1
    return first, last


def envelope_widths(
    time: FloatArray, envelope: FloatArray, fractions: Sequence[float]
) -> FloatArray:
    """Full widths of an envelope at fractions of its sampled peak.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    envelope : ndarray, shape (n_time,)
        Non-negative, one peak of interest (its largest sample).
    fractions : sequence of float
        In (0, 1].

    Returns
    -------
    widths : ndarray, shape (n_fractions,)
        Seconds from the first to the last sample of the run at or above each
        fraction of the peak that holds the peak (closed bounds).
    """
    peak = int(np.argmax(envelope))
    widths = np.empty(len(fractions))
    for i, fraction in enumerate(fractions):
        run = run_around(envelope, peak, fraction * envelope[peak])
        assert run is not None  # the peak is at or above any fraction of itself
        widths[i] = time[run[1]] - time[run[0]]
    return widths


def moving_rms(values: FloatArray, n_samples: int) -> FloatArray:
    """Root mean square over a centred window of ``n_samples``.

    Parameters
    ----------
    values : ndarray, shape (n_time,)
    n_samples : int

    Returns
    -------
    rms : ndarray, shape (n_time,)
        numpy's ``"same"`` convolution alignment; windows shortened at the ends
        are still divided by ``n_samples``.
    """
    power = np.convolve(values**2, np.full(n_samples, 1.0 / n_samples), mode="same")
    return np.sqrt(np.maximum(power, 0.0))


def spike_train(multiunit: FloatArray) -> tuple[IntArray, IntArray, FloatArray]:
    """The nonzero counts of a ``(n_time, n_units)`` array: samples, units and
    counts, sorted by sample then unit."""
    samples, units = np.nonzero(multiunit)
    return samples, units, multiunit[samples, units]


def observed_participation(
    time: FloatArray,
    spike_samples: IntArray,
    spike_units: IntArray,
    units: IntArray,
    centers: FloatArray,
    half_width: float,
) -> FloatArray:
    """Fraction of ``units`` that spike within ``half_width`` of each centre.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    spike_samples, spike_units : ndarray of int, shape (n_spikes,)
        The sample and unit of each spike (or each nonzero count).
    units : ndarray of int, shape (n_candidates,)
        The units that count.
    centers : ndarray, shape (n_centers,)
        Seconds.
    half_width : float
        Seconds; a spike at ``|t - centre| <= half_width`` counts.

    Returns
    -------
    fraction : ndarray, shape (n_centers,)
        NaN when ``units`` is empty.
    """
    if not len(units):
        return np.full(len(centers), np.nan)
    keep = np.isin(spike_units, units)
    times, owners = time[spike_samples[keep]], spike_units[keep]
    order = np.argsort(times, kind="stable")
    times, owners = times[order], owners[order]
    tolerance = _time_tolerance(time)
    first = np.searchsorted(times, centers - half_width - tolerance, side="left")
    last = np.searchsorted(times, centers + half_width + tolerance, side="right")
    counts = [np.unique(owners[a:b]).size for a, b in zip(first, last, strict=True)]
    return np.asarray(counts, dtype=float) / len(units)


def silent_gaps(
    time: FloatArray, spike_samples: IntArray, intervals: FloatArray
) -> FloatArray:
    """Intervals between consecutive population spikes inside each interval.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    spike_samples : ndarray of int, shape (n_spikes,)
        Samples holding a spike of any unit of the population.
    intervals : ndarray, shape (n_intervals, 2)
        Sorted and disjoint; a gap spanning two of them is left out.

    Returns
    -------
    gaps : ndarray, shape (n_gaps,)
        Seconds between consecutive distinct spike samples of one interval.
    """
    times = np.unique(time[spike_samples])
    union = interval_union(intervals)
    if not len(union):
        return np.empty(0)
    tolerance = _time_tolerance(time)
    which = np.searchsorted(union[:, 0], times + tolerance, side="right") - 1
    inside = (which >= 0) & (times <= union[np.clip(which, 0, None), 1] + tolerance)
    times, which = times[inside], which[inside]
    same = which[1:] == which[:-1]
    return np.asarray(np.diff(times)[same], dtype=float)


def fft_peak_frequency(waveform: FloatArray, sampling_frequency: float) -> float:
    """Frequency of the largest ``|FFT|`` of ``waveform``, zero-padded to
    ``FFT_RESOLUTION``."""
    n_fft = max(waveform.size, int(np.ceil(sampling_frequency / FFT_RESOLUTION)))
    spectrum = np.abs(np.fft.rfft(waveform, n=n_fft))
    return float(np.fft.rfftfreq(n_fft, 1 / sampling_frequency)[np.argmax(spectrum)])


def fractional_shift(waveform: FloatArray, shift: float) -> FloatArray:
    """``waveform`` delayed by ``shift`` samples (any real number), by a phase
    ramp on its FFT: exact for a band-limited waveform with room to move
    before either end, which the FFT would wrap around.

    Parameters
    ----------
    waveform : ndarray, shape (n_time,)
    shift : float
        Samples; positive delays.

    Returns
    -------
    shifted : ndarray, shape (n_time,)
    """
    frequency = np.fft.rfftfreq(waveform.size)
    spectrum = np.fft.rfft(waveform) * np.exp(-2j * np.pi * frequency * shift)
    return np.asarray(np.fft.irfft(spectrum, n=waveform.size), dtype=float)


def instantaneous_frequency(time: FloatArray, waveform: FloatArray) -> FloatArray:
    """Hilbert phase derivative in hertz, same shape as ``waveform``."""
    phase = np.unwrap(np.angle(signal.hilbert(waveform)))
    return np.asarray(np.gradient(phase, time) / (2 * np.pi), dtype=float)


def threshold_distance(fraction: float, power: int) -> float:
    """Side scales from the centre to ``fraction`` of an envelope's peak."""
    return float(np.sqrt(2 * np.log(2)) * (np.log(1 / fraction) / np.log(2)) ** (1 / power))


def refractory_rate(rate: FloatArray, step: float, refractory_period: float) -> FloatArray:
    """Expected realized rate of the refractory spike model at a constant
    intensity: a spike with probability ``p = 1 - exp(-rate step)`` per sample,
    none in the ``m`` samples after a spike closer than ``refractory_period``,
    so ``p / (step (1 + m p))``."""
    blocked = max(int(np.ceil(refractory_period / step * (1 - 1e-9))) - 1, 0)
    p = -np.expm1(-np.asarray(rate, dtype=float) * step)
    return np.asarray(p / (step * (1 + blocked * p)), dtype=float)


# ---------------------------------------------------------------------------
# One session


@dataclass
class SessionResult:
    """What one validation session keeps: summaries, never full signals.

    ``samples`` has one row per measured item (``quantity, group, x, y``),
    pooled over replicates for the target checks; ``measurements`` the
    session's summary rows; ``checks`` its rendering checks (``check,
    observed, n, note``); ``traces`` small arrays for the figures.
    """

    condition_id: str
    replicate: int
    samples: pd.DataFrame
    measurements: pd.DataFrame
    checks: pd.DataFrame
    traces: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    peak_rss_bytes: int = 0


@dataclass(frozen=True)
class _NoiseOnly:
    """The matched noise-only rendering's signals, without its spikes.

    Attributes
    ----------
    lfps : ndarray, shape (n_time, n_channels)
    sharp_wave_lfp : ndarray, shape (n_time,)
    """

    lfps: FloatArray
    sharp_wave_lfp: FloatArray


class _Render(Protocol):
    """A session's rendering of another table, with its seed and options."""

    def __call__(
        self, table: pd.DataFrame, non_events: pd.DataFrame | None = None
    ) -> rd.SimulatedSession: ...


def _render_seed(replicate: int) -> int:
    """The rendering stage's seed of ``replicate``, as ``simulate_condition``
    derives it (the fourth of four seeds drawn from ``session_seed``)."""
    seeds = np.random.default_rng(session_seed(replicate)).integers(
        np.iinfo(np.int64).max, size=4
    )
    return int(seeds[3])


def _peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage if sys.platform == "darwin" else usage * 1024)


class _Collector:
    """Accumulates a session's sample, measurement and check rows."""

    def __init__(self) -> None:
        self.samples: list[tuple[str, str, float, float]] = []
        self.measurements: list[tuple[str, str, str, float, int]] = []
        self.checks: list[tuple[str, float, int, str]] = []

    def sample(self, quantity: str, group: str, x: ArrayLike, y: ArrayLike = np.nan) -> None:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.broadcast_to(np.asarray(y, dtype=float), x.shape)
        self.samples.extend((quantity, group, a, b) for a, b in zip(x, y, strict=True))

    def measure(
        self,
        quantity: str,
        group: str,
        statistic: str,
        value: float | np.floating[Any],
        n: int,
    ) -> None:
        self.measurements.append((quantity, group, statistic, float(value), int(n)))

    def summarize(self, quantity: str, group: str, values: Iterable[float]) -> None:
        values = np.asarray(list(values), dtype=float)
        finite = values[np.isfinite(values)]
        self.measure(quantity, group, "n_missing", values.size - finite.size, values.size)
        for statistic, function in (
            ("median", np.median),
            ("mean", np.mean),
            ("p05", partial(np.percentile, q=5)),
            ("p95", partial(np.percentile, q=95)),
        ):
            value = float(function(finite)) if finite.size else np.nan
            self.measure(quantity, group, statistic, value, finite.size)

    def check(self, name: str, observed: float, n: int, note: str = "") -> None:
        self.checks.append((name, float(observed), int(n), note))


def _nearest(time: FloatArray, when: float) -> int:
    """The sample nearest ``when`` (the earlier on a tie)."""
    after = int(np.clip(np.searchsorted(time, when), 1, time.size - 1))
    return after - 1 if when - time[after - 1] <= time[after] - when else after


def _window(time: FloatArray, start: float, end: float, pad: int = 0) -> slice:
    """Samples in ``[start, end)`` widened by ``pad`` each side, at least one."""
    first, last = np.searchsorted(time, [start, end])
    first, last = (
        max(int(first) - pad, 0),
        min(max(int(last), int(first) + 1) + pad, time.size),
    )
    return slice(first, last)


def _padded_filtered_peak(
    waveform: FloatArray, rate: float, band: tuple[float, float] | None = None
) -> float:
    """Peak of ``|filter_ripple_band|`` of ``waveform`` with a second of zeros
    each side, as the renderer sizes a burst."""
    n_pad = int(np.ceil(rate))
    padded = np.zeros(waveform.size + 2 * n_pad)
    padded[n_pad : n_pad + waveform.size] = waveform
    return float(np.abs(filter_ripple_band(padded, sampling_frequency=rate, band=band)).max())


def isolated_groups(rows: pd.DataFrame, margin: float) -> list[pd.DataFrame]:
    """Rows split into groups within which no two rows' windows overlap.

    Parameters
    ----------
    rows : pandas.DataFrame
        Event or non-event rows (``center_time``, ``rise_sigma``,
        ``decay_sigma``).
    margin : float
        Seconds added to each side of a row's window of eight side scales.

    Returns
    -------
    groups : list of pandas.DataFrame
        Each in the rows' order; together, every row once. Greedy by window
        start, so as few groups as the windows' largest overlap.
    """
    starts = (rows.center_time - 8 * rows.rise_sigma - margin).to_numpy()
    ends = (rows.center_time + 8 * rows.decay_sigma + margin).to_numpy()
    last_end: list[float] = []
    group = np.empty(len(rows), dtype=int)
    for i in np.argsort(starts, kind="stable"):
        free = [g for g, end in enumerate(last_end) if end < starts[i]]
        group[i] = free[0] if free else len(last_end)
        if free:
            last_end[group[i]] = ends[i]
        else:
            last_end.append(ends[i])
    return [rows[group == g] for g in range(len(last_end))]


def measure_session(
    condition: Condition,
    replicate: int,
    overrides: Mapping[str, Any] | None = None,
    keep_traces: bool = False,
) -> SessionResult:
    """Simulate and measure one validation session.

    Parameters
    ----------
    condition : Condition
    replicate : int
        A validation replicate index (``FIRST_REPLICATE`` up).
    overrides : mapping, optional
        As in ``conditions.resolve``.
    keep_traces : bool, optional
        Keep the PSD, windowed band power and example snippets for figures.

    Returns
    -------
    SessionResult
    """
    started = clock.perf_counter()
    parameters = resolve(condition, overrides)
    session = simulate_condition(condition, replicate, overrides)
    time = session.time
    rate = float(session.sampling_frequency)
    seed = _render_seed(replicate)
    bouts = session.running_intervals

    def rendered(
        table: pd.DataFrame, non_events: pd.DataFrame | None = None
    ) -> rd.SimulatedSession:
        return rd.simulate_network_session(
            time,
            table,
            non_events=non_events,
            running_intervals=bouts,
            rng=np.random.default_rng(seed),
            sampling_frequency=rate,
            **parameters["render"],
        )

    out = _Collector()
    events, non_events = session.events, session.non_events
    rest = rest_intervals(time, session.running_intervals)
    rest_mask = interval_mask(time, rest)
    network = rd.truth_windows(events, 0.1, "network")
    baseline_mask = rest_mask & ~interval_mask(
        time, network[["start_time", "end_time"]].to_numpy()
    )
    samples, units, counts = spike_train(session.multiunit)
    unit_types, baseline_rates = session.unit_types, session.baseline_rates
    lfps, radiatum = session.lfps, session.sharp_wave_lfp
    ripple_channels = session.ripple_channels
    snippets = _examples(session) if keep_traces else {}
    del session  # the spike array is the largest; keep only what is measured

    rendering = rendered(events.iloc[:0])
    noise = _NoiseOnly(rendering.lfps, rendering.sharp_wave_lfp)
    del rendering  # its spike array is as large as the session's
    _check_noise_matched(out, time, lfps, radiatum, noise, events, non_events, parameters)
    band_sds = filter_ripple_band(noise.lfps, sampling_frequency=rate).std(axis=0)
    filtered_noise = filter_ripple_band(noise.lfps[:, 0], sampling_frequency=rate)
    traces = _background(out, time, noise, filtered_noise, lfps, parameters, keep_traces)

    _event_table_measures(out, events, rest)
    _ripple_measures(out, time, events, noise, rendered, band_sds, filtered_noise, parameters)
    _spatial_draws(out, ripple_channels, parameters)
    _sharp_wave_measures(out, time, events, noise, rendered)
    _gamma_measures(out, time, non_events, noise, rendered, events, parameters)
    _rms_durations(out, time, events, lfps[:, 0], noise.lfps[:, 0], rate)
    del noise, lfps, radiatum
    _spike_measures(
        out,
        time,
        events,
        non_events,
        samples,
        units,
        counts,
        unit_types,
        baseline_rates,
        rest,
        rest_mask,
        baseline_mask,
        parameters,
    )
    _model_metadata(out, events, non_events, parameters)
    return SessionResult(
        condition_id=condition.condition_id,
        replicate=replicate,
        samples=pd.DataFrame(out.samples, columns=["quantity", "group", "x", "y"]),
        measurements=pd.DataFrame(
            out.measurements, columns=["quantity", "group", "statistic", "value", "n"]
        ),
        checks=pd.DataFrame(out.checks, columns=["check", "observed", "n", "note"]),
        traces={**traces, "examples": snippets} if keep_traces else {},
        seconds=clock.perf_counter() - started,
        peak_rss_bytes=_peak_rss_bytes(),
    )


def _examples(session: rd.SimulatedSession) -> dict[str, dict[str, Any]]:
    """Channel 0, the radiatum channel and the spikes around the first event of
    each type and the first non-event of each kind."""
    time = session.time
    firsts = [
        (str(kind), float(rows.center_time.iloc[0]))
        for table, column in (
            (session.events, "event_type"),
            (session.non_events, "non_event_type"),
        )
        for kind, rows in table.groupby(column, sort=False)
    ]
    examples = {}
    for kind, center in firsts:
        window = _window(time, center - EXAMPLE_HALF_WIDTH, center + EXAMPLE_HALF_WIDTH)
        spikes = np.nonzero(session.multiunit[window])
        examples[kind] = {
            "time": time[window] - center,
            "lfp": session.lfps[window, 0].copy(),
            "radiatum": session.sharp_wave_lfp[window].copy(),
            "spike_time": time[window][spikes[0]] - center,
            "spike_unit": spikes[1],
            "unit_types": session.unit_types,
        }
    return examples


def _check_noise_matched(
    out: _Collector,
    time: FloatArray,
    lfps: FloatArray,
    radiatum: FloatArray,
    noise: _NoiseOnly,
    events: pd.DataFrame,
    non_events: pd.DataFrame,
    parameters: Mapping[str, Mapping[str, Any]],
) -> None:
    """The noise-only rendering equals the session outside every rendered
    component (eight side scales, the local delay and a few samples)."""
    reach = (
        float(parameters["render"]["channel_delay"])
        + 4 / parameters["session"]["sampling_frequency"]
    )
    spans = [
        np.column_stack(
            [
                table.center_time - 8 * table.rise_sigma - reach,
                table.center_time + 8 * table.decay_sigma + reach,
            ]
        )
        for table in (events, non_events)
    ]
    outside = ~interval_mask(time, np.concatenate(spans))
    difference = max(
        float(np.max(np.abs(lfps[outside] - noise.lfps[outside]), initial=0.0)),
        float(np.max(np.abs(radiatum[outside] - noise.sharp_wave_lfp[outside]), initial=0.0)),
    )
    out.check("noise_only_matched", difference, int(outside.sum()))


def _background(
    out: _Collector,
    time: FloatArray,
    noise: _NoiseOnly,
    filtered_noise: FloatArray,
    lfps: FloatArray,
    parameters: Mapping[str, Mapping[str, Any]],
    keep_traces: bool,
) -> dict[str, Any]:
    """PSD, windowed ripple-band power, the modulation fit and coherence."""
    rate = float(parameters["session"]["sampling_frequency"])
    render = parameters["render"]
    n_segment = round(rate)
    frequency, psd = signal.welch(noise.lfps[:, 0], fs=rate, nperseg=n_segment)
    for band, (low, high) in PSD_BANDS.items():
        inside = (frequency >= low) & (frequency <= high)
        out.measure("background_band_power", band, "mean_psd", psd[inside].mean(), 1)
    n_window = round(POWER_WINDOW * rate)
    n_windows = time.size // n_window
    power = (filtered_noise[: n_windows * n_window] ** 2).reshape(n_windows, n_window)
    window_power = power.mean(axis=1)
    centers = time[: n_windows * n_window].reshape(n_windows, n_window).mean(axis=1)
    out.summarize("windowed_ripple_band_power", "noise_only", window_power)
    out.measure(
        "windowed_ripple_band_power",
        "noise_only",
        "cv",
        window_power.std() / window_power.mean() if n_windows else np.nan,
        n_windows,
    )
    period = float(render["noise_modulation_period"])
    duration = time.size / rate  # seconds the samples cover
    fitted = np.nan
    if duration >= MINIMUM_MODULATION_PERIODS * period and n_windows >= 6:
        phase = 2 * np.pi * (centers - time[0]) / period
        design = np.column_stack([np.ones(n_windows), np.sin(phase), np.cos(phase)])
        coefficients = np.linalg.lstsq(design, np.log(window_power), rcond=None)[0]
        fitted = float(np.hypot(*coefficients[1:]) / (2 * np.sinc(POWER_WINDOW / period)))
    out.measure("noise_modulation", "noise_only", "log_amplitude", fitted, n_windows)
    note = (
        ""
        if np.isfinite(fitted)
        else (
            f"not measurable: needs {MINIMUM_MODULATION_PERIODS:g} modulation periods "
            f"({MINIMUM_MODULATION_PERIODS * period:g} s) and 6 windows"
        )
    )
    out.check(
        "noise_modulation_amplitude",
        abs(fitted - float(render["noise_log_amplitude"])),
        n_windows,
        note,
    )
    if lfps.shape[1] > 1:
        for label, data in (("noise_only", noise.lfps), ("session", lfps)):
            f, coherence = signal.coherence(data[:, 0], data[:, 1], fs=rate, nperseg=n_segment)
            inside = (f >= 150.0) & (f <= 250.0)
            out.measure(
                "channel_coherence", label, "ripple_band_mean", coherence[inside].mean(), 1
            )
    if not keep_traces:
        return {}
    keep = frequency <= PSD_MAXIMUM
    return {
        "psd": (frequency[keep], psd[keep]),
        "window_power": (centers - time[0], window_power),
    }


def _event_table_measures(out: _Collector, events: pd.DataFrame, rest: FloatArray) -> None:
    """Realized rates and type proportions, event-strength pairs and doublet
    spacing, from the rendered session's event table."""
    rest_seconds = float(np.sum(rest[:, 1] - rest[:, 0]))
    types = events.groupby("event_id", sort=True).event_type.first()
    with_ripple = events.loc[events.expression == "ripple", "event_id"].nunique()
    out.sample("ripple_event_rate", "all", [with_ripple], [rest_seconds])
    out.measure("event_rate", "all", "per_rest_s", len(types) / rest_seconds, len(types))
    out.measure(
        "event_rate", "with_ripple", "per_rest_s", with_ripple / rest_seconds, with_ripple
    )
    if events.empty:
        return
    for event_type in rd.EVENT_TYPES:
        count = int((types == event_type).sum())
        out.measure(
            "event_type_proportion",
            event_type,
            "fraction",
            count / len(types) if len(types) else np.nan,
            count,
        )
    first = events[events.component == 0].pivot_table(
        index="event_id",
        columns="expression",
        values=["amplitude", "frequency_start"],
        aggfunc="first",
    )
    for event_type in ("swr", "weak_ripple", "ripple_doublet"):
        ids = types.index[types == event_type]
        if not len(ids):
            continue
        sharp_wave = first.loc[ids, ("amplitude", "sharp_wave")].to_numpy()
        snr = first.loc[ids, ("amplitude", "ripple")].to_numpy()
        onset = first.loc[ids, ("frequency_start", "ripple")].to_numpy()
        out.sample("sharp_wave_ripple_power", event_type, sharp_wave, snr**2)
        out.sample("sharp_wave_ripple_frequency", event_type, sharp_wave, onset)
    ripples = events[events.expression == "ripple"]
    doublets = ripples[ripples.event_type == "ripple_doublet"]
    spacing = doublets.groupby("event_id", sort=True).center_time.diff().dropna()
    out.sample("doublet_spacing", "ripple_doublet", spacing.to_numpy() * 1e3)
    for expression, rows in events.groupby("expression", sort=False):
        power = rows.envelope_power.to_numpy()
        spans = (rows.rise_sigma + rows.decay_sigma).to_numpy()
        for label, widths in (
            ("nominal", 3 * spans),
            ("half_maximum", [threshold_distance(0.5, p) for p in power] * spans),
            ("ten_percent", [threshold_distance(0.1, p) for p in power] * spans),
        ):
            out.summarize(
                f"latent_width_ms_{label}", str(expression), np.asarray(widths) * 1e3
            )


def _anchor(channels: pd.DataFrame) -> int:
    """The anchor channel of one ripple: delay 0 and the largest gain."""
    candidates = channels[channels.delay_s == 0]
    return int(candidates.channel.iloc[int(np.argmax(candidates.gain.to_numpy()))])


def _ripple_measures(
    out: _Collector,
    time: FloatArray,
    events: pd.DataFrame,
    noise: _NoiseOnly,
    rendered: _Render,
    band_sds: FloatArray,
    filtered_noise: FloatArray,
    parameters: Mapping[str, Mapping[str, Any]],
) -> None:
    """Frequencies, widths, SNRs and the spatial rendering of isolated ripples."""
    rate = float(parameters["session"]["sampling_frequency"])
    step = 1.0 / rate
    maximum_delay = float(parameters["render"]["channel_delay"])
    reach = int(np.ceil(maximum_delay * rate)) + 2
    local_half = round(LOCAL_SNR_WINDOW / 2 * rate)
    sizing, nominal_ratio, residuals, energy_mismatch = [], [], [], 0
    margin = ISOLATION_MARGIN + maximum_delay + reach * step
    for table in isolated_groups(events[events.expression == "ripple"], margin):
        isolated = rendered(table)
        difference = isolated.lfps - noise.lfps
        channels_of = dict(tuple(isolated.ripple_channels.groupby(["event_id", "component"])))
        truth = {
            fraction: rd.truth_windows(table, fraction, "ripple")
            for fraction in TRUTH_FRACTIONS
        }
        del isolated
        for position, row in enumerate(table.itertuples()):
            channels = channels_of[row.event_id, row.component]
            anchor = _anchor(channels)
            window = _window(
                time,
                row.center_time - 8 * row.rise_sigma,
                row.center_time + 8 * row.decay_sigma,
            )
            waveform = difference[window, anchor]
            group = str(row.event_type)
            out.sample("ripple_peak_frequency", group, [fft_peak_frequency(waveform, rate)])
            padded = _window(
                time,
                row.center_time - 8 * row.rise_sigma,
                row.center_time + 8 * row.decay_sigma,
                pad=int(0.05 * rate),
            )
            analytic_time = time[padded]
            frequency = instantaneous_frequency(analytic_time, difference[padded, anchor])
            before = _nearest(analytic_time, row.center_time - FREQUENCY_OFFSET)
            after = _nearest(analytic_time, row.center_time + FREQUENCY_OFFSET)
            out.sample(
                "ripple_frequency_decline", group, [frequency[before] - frequency[after]]
            )
            envelope = np.abs(signal.hilbert(difference[padded, anchor]))
            peak = int(np.argmax(envelope))
            out.sample(
                "ripple_envelope_peak_offset_ms",
                group,
                [(analytic_time[peak] - row.center_time) * 1e3],
            )
            for fraction in TRUTH_FRACTIONS:
                run = run_around(envelope, peak, fraction * envelope[peak])
                assert run is not None
                bounds = truth[fraction].iloc[position]
                out.sample(
                    f"ripple_hilbert_start_error_ms_{fraction:g}",
                    group,
                    [(analytic_time[run[0]] - bounds.start_time) * 1e3],
                )
                out.sample(
                    f"ripple_hilbert_end_error_ms_{fraction:g}",
                    group,
                    [(analytic_time[run[1]] - bounds.end_time) * 1e3],
                )
                if fraction in (0.1, 0.5):
                    out.sample(
                        f"ripple_hilbert_width_ms_{fraction:g}",
                        group,
                        [(analytic_time[run[1]] - analytic_time[run[0]]) * 1e3],
                    )
            gain = float(channels.gain[channels.channel == anchor].iloc[0])
            filtered_peak = _padded_filtered_peak(waveform, rate)
            sizing.append(filtered_peak / (row.amplitude * gain))
            anchor_snr = filtered_peak / band_sds[0]
            nominal_ratio.append(anchor_snr / (row.amplitude * gain))
            out.sample("ripple_snr_nominal", group, [row.amplitude])
            out.sample("ripple_snr_anchor", group, [anchor_snr])
            center = _nearest(time, row.center_time)
            local = filtered_noise[max(center - local_half, 0) : center + local_half]
            out.sample("ripple_snr_event_local", group, [filtered_peak / local.std()])
            wide = _window(
                time,
                row.center_time - 8 * row.rise_sigma,
                row.center_time + 8 * row.decay_sigma,
                pad=reach,
            )
            per_channel = [
                _padded_filtered_peak(difference[wide, c], rate) / band_sds[c]
                for c in range(difference.shape[1])
            ]
            out.sample("ripple_snr_recording_wide", group, [np.mean(per_channel)])
            reference = difference[wide, anchor]
            for channel in channels.itertuples():
                if channel.channel == anchor:
                    continue
                trace = difference[wide, channel.channel]
                if channel.gain == 0:
                    energy_mismatch += bool(np.any(trace != 0))
                    continue
                expected = (channel.gain / gain) * fractional_shift(
                    reference, channel.delay_s * rate
                )
                residuals.append(
                    float(np.linalg.norm(trace - expected) / np.linalg.norm(trace))
                )
        del difference
    sizing_array = np.asarray(sizing)
    spread = (
        float((sizing_array.max() - sizing_array.min()) / np.median(sizing_array))
        if sizing_array.size
        else 0.0
    )
    out.check("ripple_sizing", spread, sizing_array.size, "" if sizing else "no ripples")
    out.check(
        "ripple_nominal_snr",
        abs(float(np.median(nominal_ratio)) - 1) if nominal_ratio else 0.0,
        len(nominal_ratio),
        "" if nominal_ratio else "no ripples",
    )
    notes = []
    if energy_mismatch:
        notes.append(f"{energy_mismatch} zero-gain channel(s) carry a ripple")
    if not residuals:
        notes.append("no channel besides the anchor carries a ripple")
    out.check(
        "channel_profile_rendering",
        max(residuals, default=0.0) + energy_mismatch,
        len(residuals),
        "; ".join(notes),
    )


def _spatial_draws(
    out: _Collector, ripple_channels: pd.DataFrame, parameters: Mapping[str, Mapping[str, Any]]
) -> None:
    """Occupancy and delays of the session's ripples, and whether each
    follows its spatial profile's rules."""
    render = parameters["render"]
    n_channels = int(render["n_channels"])
    violations, n_ripples = 0, 0
    occupancy, delays = [], []
    for _, channels in ripple_channels.groupby(["event_id", "component"], sort=True):
        n_ripples += 1
        carrying = channels[channels.gain > 0]
        occupancy.append(len(carrying) / n_channels)
        anchor = _anchor(channels)
        others = carrying[carrying.channel != anchor]
        delays.extend(np.abs(others.delay_s.to_numpy()) * 1e3)
        if render["spatial_profile"] == "global":
            violations += int((channels.gain != 1.0).any() or (channels.delay_s != 0).any())
            continue
        low, high = render["channel_gain_range"]
        expected = max(1, int(np.ceil(render["channel_occupancy"] * n_channels)))
        violations += int(
            len(carrying) != expected
            or float(channels.gain[channels.channel == anchor].iloc[0]) != 1.0
            or not others.gain.between(low, high).all()
            or (others.delay_s.abs() > render["channel_delay"]).any()
            or (channels.delay_s[channels.gain == 0] != 0).any()
        )
    out.summarize("channel_occupancy", "ripple", occupancy)
    out.summarize("channel_delay_ms", "non_anchor", delays)
    out.check("spatial_profile_draws", violations, n_ripples)


def _sharp_wave_measures(
    out: _Collector,
    time: FloatArray,
    events: pd.DataFrame,
    noise: _NoiseOnly,
    rendered: _Render,
) -> None:
    """Widths of isolated sharp waves and their crossings against the truth
    windows."""
    step = float(np.median(np.diff(time)))
    worst, n_crossings = 0.0, 0
    sharp_waves = events[events.expression == "sharp_wave"]
    for table in isolated_groups(sharp_waves, ISOLATION_MARGIN):
        deflection = noise.sharp_wave_lfp - rendered(table).sharp_wave_lfp
        truth = {
            fraction: rd.truth_windows(table, fraction, "sharp_wave")
            for fraction in TRUTH_FRACTIONS
        }
        for position, row in enumerate(table.itertuples()):
            window = _window(
                time,
                row.center_time - 8 * row.rise_sigma,
                row.center_time + 8 * row.decay_sigma,
            )
            local = deflection[window]
            local_time = time[window]
            widths = envelope_widths(local_time, local, (0.5, 0.1))
            out.sample("sharp_wave_duration", str(row.event_type), [widths[1] * 1e3])
            out.sample("sharp_wave_width_ms_0.5", str(row.event_type), [widths[0] * 1e3])
            center = _nearest(local_time, row.center_time)
            for fraction in TRUTH_FRACTIONS:
                run = run_around(local, center, fraction * row.amplitude)
                bounds = truth[fraction].iloc[position]
                if run is None:
                    worst = np.inf
                    continue
                n_crossings += 2
                worst = max(
                    worst,
                    abs(local_time[run[0]] - bounds.start_time) / step,
                    abs(local_time[run[1]] - bounds.end_time) / step,
                )
    out.check(
        "sharp_wave_truth_crossings",
        worst,
        n_crossings,
        "" if n_crossings else "no sharp waves",
    )


def _gamma_measures(
    out: _Collector,
    time: FloatArray,
    non_events: pd.DataFrame,
    noise: _NoiseOnly,
    rendered: _Render,
    events: pd.DataFrame,
    parameters: Mapping[str, Mapping[str, Any]],
) -> None:
    """Gamma bursts' SNR in their stored band, from a rendering of the
    non-events alone."""
    rate = float(parameters["session"]["sampling_frequency"])
    gamma = non_events[non_events.non_event_type == "fast_gamma"]
    if not len(gamma):
        out.check("gamma_sizing", 0.0, 0, "no gamma bursts")
        return
    gains = parameters["render"]["channel_gains"]
    gain = 1.0 if gains is None else float(gains[0])
    band_sd: dict[tuple[float, float], float] = {}
    ratios = []
    for table in isolated_groups(gamma, ISOLATION_MARGIN):
        difference = rendered(events.iloc[:0], table).lfps[:, 0] - noise.lfps[:, 0]
        for row in table.itertuples():
            band = (float(row.snr_band_low), float(row.snr_band_high))
            if band not in band_sd:
                band_sd[band] = float(
                    filter_ripple_band(
                        noise.lfps[:, 0], sampling_frequency=rate, band=band
                    ).std()
                )
            window = _window(
                time,
                row.center_time - 8 * row.rise_sigma,
                row.center_time + 8 * row.decay_sigma,
            )
            snr = _padded_filtered_peak(difference[window], rate, band) / band_sd[band]
            ratios.append(snr / (row.amplitude * gain))
            out.sample("gamma_snr_ratio", "fast_gamma", [ratios[-1]])
    if gain == 0:
        out.check("gamma_sizing", 0.0, 0, "channel 0 carries no gamma")
        return
    out.check("gamma_sizing", abs(float(np.median(ratios)) - 1), len(ratios))


def _rms_durations(
    out: _Collector,
    time: FloatArray,
    events: pd.DataFrame,
    channel: FloatArray,
    noise_channel: FloatArray,
    rate: float,
) -> None:
    """Ripple durations by the RMS convention and by Patel et al.'s."""
    n_samples = max(round(RMS_WINDOW * rate), 1)
    rms = moving_rms(
        filter_ripple_band(channel, sampling_frequency=rate, band=RMS_BAND), n_samples
    )
    noise_rms = moving_rms(
        filter_ripple_band(noise_channel, sampling_frequency=rate, band=RMS_BAND), n_samples
    )
    mean, sd = float(noise_rms.mean()), float(noise_rms.std())
    low, high = mean + RMS_THRESHOLD_SD * sd, mean + PATEL_PEAK_SD * sd
    crossed: dict[str, list[bool]] = {}
    for row in events[events.expression == "ripple"].itertuples():
        run = run_around(rms, _nearest(time, row.center_time), low, strict=True)
        duration = np.nan if run is None else float(time[run[1]] - time[run[0]])
        group = str(row.event_type)
        if group != "ripple_doublet":
            out.sample("ripple_duration", group, [duration * 1e3])
            crossed.setdefault(group, []).append(run is not None)
        keep = (
            run is not None
            and duration >= PATEL_MINIMUM_DURATION - _time_tolerance(time)
            and float(rms[run[0] : run[1] + 1].max()) >= high
        )
        out.sample("ripple_duration_sleep", group, [duration * 1e3 if keep else np.nan])
    for group, flags in crossed.items():
        out.measure(
            "ripple_duration",
            group,
            "fraction_never_crossing",
            1 - float(np.mean(flags)),
            len(flags),
        )


def _spike_measures(
    out: _Collector,
    time: FloatArray,
    events: pd.DataFrame,
    non_events: pd.DataFrame,
    samples: IntArray,
    units: IntArray,
    counts: FloatArray,
    unit_types: Any,
    baseline_rates: FloatArray,
    rest: FloatArray,
    rest_mask: BoolArray,
    baseline_mask: BoolArray,
    parameters: Mapping[str, Mapping[str, Any]],
) -> None:
    """Firing rates, ripple gains, participation, count variability,
    inter-spike intervals, silent gaps and the spike model's realized rate."""
    rate = float(parameters["session"]["sampling_frequency"])
    step = 1.0 / rate
    render = parameters["render"]
    n_units = unit_types.size
    principal = np.flatnonzero(np.isin(unit_types, ["place", "pyramidal"]))
    interneurons = np.flatnonzero(unit_types == "interneuron")
    populations = {"pyramidal": principal, "interneuron": interneurons}

    def counts_in(mask: BoolArray) -> FloatArray:
        at = mask[samples]
        return np.bincount(units[at], weights=counts[at], minlength=n_units)

    rest_counts = counts_in(rest_mask)
    rest_seconds = rest_mask.sum() * step
    rest_rates = rest_counts / rest_seconds
    out.sample("pyramidal_baseline_rate", "pyramidal", rest_rates[principal])
    out.sample("interneuron_baseline_rate", "interneuron", rest_rates[interneurons])
    ripples = events[events.expression == "ripple"]
    strong = ripples[ripples.event_type.isin(["swr", "ripple_doublet"])].center_time.to_numpy()
    window_mask = interval_mask(
        time, np.column_stack([strong - GAIN_WINDOW, strong + GAIN_WINDOW])
    )
    window_counts, baseline_counts = counts_in(window_mask), counts_in(baseline_mask)
    for label, members in populations.items():
        window_seconds = window_mask.sum() * step * members.size
        baseline_seconds = baseline_mask.sum() * step * members.size
        quantity = f"{label}_ripple_gain"
        out.sample(quantity, "window", [window_counts[members].sum()], [window_seconds])
        out.sample(quantity, "baseline", [baseline_counts[members].sum()], [baseline_seconds])
    for unit_type in rd.UNIT_TYPES:
        members = np.flatnonzero(unit_types == unit_type)
        if not members.size:
            continue
        out.measure(
            "baseline_rate_drawn",
            unit_type,
            "mean_hz",
            baseline_rates[members].mean(),
            members.size,
        )
        out.measure(
            "firing_rate_rest_outside_events",
            unit_type,
            "mean_hz",
            baseline_counts[members].sum() / (baseline_mask.sum() * step * members.size),
            members.size,
        )
        out.measure(
            "firing_rate_ripple_windows",
            unit_type,
            "mean_hz",
            window_counts[members].sum() / (window_mask.sum() * step * members.size)
            if window_mask.any()
            else np.nan,
            members.size,
        )

    first_ripples = ripples[ripples.component == 0]
    fractions = observed_participation(
        time,
        samples,
        units,
        principal,
        first_ripples.center_time.to_numpy(),
        PARTICIPATION_WINDOW,
    )
    bursts = events[events.expression == "burst"].set_index("event_id")
    for event_type, rows in first_ripples.assign(fraction=fractions).groupby(
        "event_type", sort=False
    ):
        out.sample("observed_participation", str(event_type), rows.fraction.to_numpy())
        latent = bursts.loc[rows.event_id, "n_participants"].to_numpy() / max(
            principal.size, 1
        )
        out.summarize("latent_recruitment", str(event_type), latent)

    n_bin = max(round(COUNT_BIN * rate), 1)
    n_bins = time.size // n_bin
    whole_bins = rest_mask[: n_bins * n_bin].reshape(n_bins, n_bin).all(axis=1)
    binned = np.zeros((n_bins, n_units))
    inside = samples < n_bins * n_bin
    np.add.at(binned, (samples[inside] // n_bin, units[inside]), counts[inside])
    binned = binned[whole_bins]
    means = binned.mean(axis=0)
    fano = np.where(means > 0, binned.var(axis=0) / np.where(means > 0, means, 1), np.nan)
    for unit_type in rd.UNIT_TYPES:
        members = np.flatnonzero(unit_types == unit_type)
        if members.size:
            out.summarize("count_fano_10ms", unit_type, fano[members])

    at_rest = rest_mask[samples]
    for label, members in (*populations.items(), ("all", np.arange(n_units))):
        chosen = at_rest & np.isin(units, members)
        gaps = silent_gaps(time, samples[chosen], rest)
        out.summarize("population_silent_gap_ms", label, gaps * 1e3)
        out.measure(
            "population_silent_gap_ms",
            label,
            "max",
            gaps.max() * 1e3 if gaps.size else np.nan,
            gaps.size,
        )
        intervals = _unit_intervals(time, samples[chosen], units[chosen], rest)
        out.summarize("inter_spike_interval_ms", label, intervals * 1e3)
        out.measure(
            "inter_spike_interval_ms",
            label,
            "fraction_below_2ms",
            float(np.mean(intervals < 0.002)) if intervals.size else np.nan,
            intervals.size,
        )
    out.measure("spikes_per_sample", "all", "max", counts.max(initial=0.0), counts.size)

    # interneurons fire at their baseline intensity outside every ripple's
    # eight-scale window, at rest (theta bursts and leaks recruit no interneuron)
    spans = np.column_stack(
        [
            ripples.center_time - 8 * ripples.rise_sigma - 2 * step,
            ripples.center_time + 8 * ripples.decay_sigma + 2 * step,
        ]
    )
    quiet = rest_mask & ~interval_mask(time, spans)
    observed = float(counts_in(quiet)[interneurons].sum())
    rates = baseline_rates[interneurons]
    if render["spike_model"] == "refractory":
        rates = refractory_rate(rates, step, float(render["refractory_period"]))
    expected = float(rates.sum() * quiet.sum() * step)
    out.sample("interneuron_rate_realization", "interneuron", [observed], [expected])

    if render["spike_model"] == "refractory":
        _check_refractory(out, time, non_events, samples, units, counts, render)


def _unit_intervals(
    time: FloatArray, samples: IntArray, units: IntArray, rest: FloatArray
) -> FloatArray:
    """Each unit's intervals between consecutive spikes within one stretch of rest."""
    order = np.lexsort((samples, units))
    samples, units = samples[order], units[order]
    stretch = np.searchsorted(rest[:, 0], time[samples], side="right")
    same = (units[1:] == units[:-1]) & (stretch[1:] == stretch[:-1])
    return np.asarray(np.diff(time[samples])[same], dtype=float)


def _check_refractory(
    out: _Collector,
    time: FloatArray,
    non_events: pd.DataFrame,
    samples: IntArray,
    units: IntArray,
    counts: FloatArray,
    render: Mapping[str, Any],
) -> None:
    """No endogenous spikes closer than the refractory period, nor two in a
    sample; spikes at samples near a leakage burst are left out, since leaked
    spikes are added after the draw and need not obey it."""
    step = float(np.median(np.diff(time)))
    leaks = non_events[non_events.non_event_type == "spike_leakage"]
    half = (leaks.n_spikes - 1) / 2 * leaks.isi
    near_leak = interval_mask(
        time,
        np.column_stack(
            [leaks.center_time - half - 2 * step, leaks.center_time + half + 2 * step]
        ),
    )
    keep = ~near_leak[samples]
    samples, units, counts = samples[keep], units[keep], counts[keep]
    order = np.lexsort((samples, units))
    samples, units, counts = samples[order], units[order], counts[order]
    same = units[1:] == units[:-1]
    close = np.diff(time[samples])[same] < float(
        render["refractory_period"]
    ) - _time_tolerance(time)
    violations = int(close.sum() + (counts > 1).sum())
    out.check("refractory_spiking", violations, samples.size)


def _model_metadata(
    out: _Collector,
    events: pd.DataFrame,
    non_events: pd.DataFrame,
    parameters: Mapping[str, Mapping[str, Any]],
) -> None:
    """Rows carry the configured envelope powers and gamma sizing bands."""
    power = int(parameters["events"]["envelope_power"])
    band = tuple(parameters["non_events"]["fast_gamma_band"])
    gamma = non_events[non_events.non_event_type == "fast_gamma"]
    other = non_events[non_events.non_event_type != "fast_gamma"]
    wrong = (
        int((events.envelope_power != power).sum())
        + int((non_events.envelope_power != 2).sum())
        + int(((gamma.snr_band_low != band[0]) | (gamma.snr_band_high != band[1])).sum())
        + int((other.snr_band_low.notna() | other.snr_band_high.notna()).sum())
    )
    out.check("model_metadata", wrong, len(events) + len(non_events))


# ---------------------------------------------------------------------------
# Targets and checks

# Rendering checks: statistic, bounds, and where they apply (a scope of
# rendering_check_applies: "all", "multichannel", "fast_gamma" or "refractory").
RENDERING_CHECKS: dict[str, tuple[str, float, float, str]] = {
    "noise_only_matched": ("max_abs_difference", 0.0, 0.0, "all"),
    "sharp_wave_truth_crossings": ("max_error_samples", 0.0, 1.0, "all"),
    "ripple_sizing": ("relative_spread", 0.0, SIZING_SPREAD, "all"),
    "ripple_nominal_snr": ("relative_error", 0.0, SNR_TOLERANCE, "all"),
    "spatial_profile_draws": ("violations", 0.0, 0.0, "all"),
    "channel_profile_rendering": (
        "max_relative_residual",
        0.0,
        PROFILE_TOLERANCE,
        "multichannel",
    ),
    "gamma_sizing": ("relative_error", 0.0, SNR_TOLERANCE, "fast_gamma"),
    "noise_modulation_amplitude": ("absolute_error", 0.0, MODULATION_TOLERANCE, "all"),
    "model_metadata": ("violations", 0.0, 0.0, "all"),
    "interneuron_rate_realization": ("z", -RATE_Z, RATE_Z, "all"),
    "refractory_spiking": ("violations", 0.0, 0.0, "refractory"),
}

# What each rendering check establishes; the report lists these.
RENDERING_DESCRIPTIONS = {
    "noise_only_matched": (
        "The session equals its noise-only rendering outside every component's eight "
        "side scales (plus the local delay and 4 samples): the matched noise is identical."
    ),
    "sharp_wave_truth_crossings": (
        "Each isolated sharp wave crosses 10%, 25% and 50% of its latent amplitude within "
        "a sample of its truth_windows bounds."
    ),
    "ripple_sizing": (
        "Every ripple's filtered peak on its anchor channel over its SNR and anchor gain "
        "is one constant (the band noise SD it was sized against)."
    ),
    "ripple_nominal_snr": (
        "The median anchor SNR against noise-only channel 0's ripple-band SD over the "
        "nominal SNR times the anchor gain is 1."
    ),
    "spatial_profile_draws": (
        "Every ripple's stored gains and delays follow its profile: global, every channel "
        "at gain 1 and no delay; local, the configured channel count, the anchor at gain "
        "1 and no delay, the others within the gain range and delay limit, zero gain "
        "with zero delay."
    ),
    "channel_profile_rendering": (
        "Every channel that carries a ripple holds the anchor waveform shifted by the "
        "stored delay and scaled by the stored gain ratio; zero-gain channels hold none."
    ),
    "gamma_sizing": (
        "The median gamma-burst SNR, in its stored band against noise-only channel 0's "
        "SD in that band, over the drawn SNR is 1."
    ),
    "noise_modulation_amplitude": (
        "The log amplitude of the background's slow gain, fitted to the noise-only "
        "ripple-band power, equals noise_log_amplitude (0 when stationary)."
    ),
    "model_metadata": (
        "Every event row stores the configured envelope power, every non-event power 2, "
        "gamma rows the configured sizing band and other rows none."
    ),
    "interneuron_rate_realization": (
        "Interneuron spikes at rest outside every ripple's eight side scales, pooled over "
        "replicates, against the count their baseline rates give under the spike model "
        "(for 'refractory', p / (step (1 + m p)) with m blocked samples): z of the "
        "difference."
    ),
    "refractory_spiking": (
        "No two drawn spikes of a unit closer than the refractory period, nor two in a "
        "sample (spikes near a leakage burst left out: leaked spikes are added after "
        "the draw)."
    ),
}


def rendering_check_applies(
    scope: str, parameters: Mapping[str, Mapping[str, Any]]
) -> tuple[bool, str]:
    """Whether a rendering check of ``scope`` applies to a condition.

    Parameters
    ----------
    scope : str
        ``"all"``; ``"multichannel"`` (more than one channel, so a channel
        besides a ripple's anchor to compare with it); ``"fast_gamma"`` (gamma
        bursts are drawn and channel 0 carries them); ``"refractory"`` (the
        refractory spike model).
    parameters : mapping
        The condition's resolved parameters (``conditions.resolve``).

    Returns
    -------
    applies : bool
    why_not : str
        Empty when it applies.
    """
    render = parameters["render"]
    if scope == "multichannel" and int(render["n_channels"]) < 2:
        return False, "not applicable: one channel, so none besides a ripple's anchor"
    if scope == "fast_gamma":
        gains = render["channel_gains"]
        if float(parameters["non_events"]["rates"]["fast_gamma"]) == 0:
            return False, "not applicable: no gamma bursts are drawn"
        if gains is not None and float(gains[0]) == 0:
            return False, "not applicable: channel 0 carries no gamma"
    if scope == "refractory" and render["spike_model"] != "refractory":
        return False, "not applicable: the spike model has no refractory period"
    return True, ""


def load_targets(path: Path = TARGETS) -> pd.DataFrame:
    """The target table, as committed."""
    return pd.read_csv(path)


def target_conditions(text: str) -> tuple[str, ...]:
    """The condition ids a target's ``conditions`` text names (``"none (...)"``
    names none)."""
    if text.strip().startswith("none"):
        return ()
    labels = [label.strip() for label in text.split(";")]
    unknown = [label for label in labels if label != "reference" and label not in MODEL_LABELS]
    if unknown:
        msg = f"Unknown condition labels {unknown} in the target table."
        raise ValueError(msg)
    return tuple(
        "reference" if label == "reference" else MODEL_LABELS[label] for label in labels
    )


def target_table_problems(targets: pd.DataFrame) -> list[str]:
    """What stops the target table from backing a report: unresolved evidence,
    quantities with no measurement, statistics with no pooling rule, unknown
    condition labels."""
    problems = []
    for row in targets.itertuples():
        if row.evidence_status not in ("supported", "assumed"):
            problems.append(
                f"target {row.quantity}: evidence status {row.evidence_status!r} is unresolved"
            )
        if row.quantity not in TARGET_SAMPLES:
            problems.append(f"target {row.quantity}: no measurement is defined for it")
        if _pooling(str(row.target_statistic)) is None:
            problems.append(
                f"target {row.quantity}: no pooling rule for statistic "
                f"{row.target_statistic!r}"
            )
        try:
            target_conditions(str(row.conditions))
        except ValueError as error:
            problems.append(f"target {row.quantity}: {error}")
    return problems


def _pooling(statistic: str) -> Any:
    """The pooling rule for a target statistic, or None."""
    if statistic == "rate_per_s":
        return lambda s: (s.x.sum() / s.y.sum(), int(s.x.sum()))
    if statistic == "ratio":

        def ratio(s: pd.DataFrame) -> tuple[float, int]:
            window, base = s[s.group == "window"], s[s.group == "baseline"]
            value = (window.x.sum() / window.y.sum()) / (base.x.sum() / base.y.sum())
            return float(value), int(window.x.sum())

        return ratio
    if statistic in ("pearson_r", "spearman_r"):
        function = stats.pearsonr if statistic == "pearson_r" else stats.spearmanr

        def correlation(s: pd.DataFrame) -> tuple[float, int]:
            pairs = s[np.isfinite(s.x) & np.isfinite(s.y)]
            if len(pairs) < 3:
                return np.nan, len(pairs)
            return float(function(pairs.x, pairs.y)[0]), len(pairs)

        return correlation
    reducers: dict[str, Callable[[FloatArray], Any]] = {
        "median": np.median,
        "mean": np.mean,
        "p95": partial(np.percentile, q=95),
    }
    prefix = statistic.split("_")[0]
    if prefix in reducers:
        reduce = reducers[prefix]

        def reduced(s: pd.DataFrame) -> tuple[float, int]:
            values = s.x[np.isfinite(s.x)].to_numpy()
            return (float(reduce(values)) if values.size else np.nan), values.size

        return reduced
    return None


def pooled_target(targets_row: Any, samples: pd.DataFrame) -> tuple[float, int]:
    """A target's statistic over one condition's samples, pooled over replicates."""
    quantity, groups = TARGET_SAMPLES[targets_row.quantity]
    chosen = samples[samples.quantity == quantity]
    if groups is not None and targets_row.target_statistic != "ratio":
        chosen = chosen[chosen.group.isin(groups)]
    value, n = _pooling(str(targets_row.target_statistic))(chosen)
    return float(value), int(n)


def build_checks(
    results: Sequence[SessionResult],
    selected: Sequence[Condition],
    parameters: Mapping[str, Mapping[str, Mapping[str, Any]]],
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """Every check for every validated condition, target rows pooled over the
    replicates.

    Every rendering check has a row in every condition, ``applies`` saying
    whether it gates the condition (``rendering_check_applies``). Its observed
    value is the largest over the replicates, NaN if any replicate's is NaN,
    and it fails when nothing was measured (``n`` 0).
    """
    rows = []
    for condition in selected:
        cid = condition.condition_id
        mine = [r for r in results if r.condition_id == cid]
        samples = pd.concat([r.samples for r in mine], ignore_index=True)
        for target in targets.itertuples():
            applies_to = target_conditions(str(target.conditions))
            observed, n = pooled_target(target, samples)
            gated = cid in applies_to and target.evidence_status == "supported"
            rows.append(
                {
                    "check": target.quantity,
                    "kind": "target",
                    "condition_id": cid,
                    "applies": gated,
                    "evidence_status": target.evidence_status,
                    "statistic": target.target_statistic,
                    "observed": observed,
                    "lower": float(target.lower),
                    "upper": float(target.upper),
                    "n": n,
                    "note": "" if cid in applies_to else "reported, not gated here",
                }
            )
        session_checks = pd.concat([r.checks for r in mine], ignore_index=True)
        for name, (statistic, lower, upper, scope) in RENDERING_CHECKS.items():
            applies, why_not = rendering_check_applies(scope, parameters[cid])
            if name == "interneuron_rate_realization":
                pooled = samples[samples.quantity == name]
                expected = float(pooled.y.sum())
                observed = (
                    (float(pooled.x.sum()) - expected) / np.sqrt(expected)
                    if expected
                    else np.nan
                )
                n, note = int(pooled.x.sum()), ""
            else:
                these = session_checks[session_checks.check == name]
                # a NaN in any replicate is the check's value, never skipped
                observed = (
                    float(these.observed.max())
                    if len(these) and these.observed.notna().all()
                    else np.nan
                )
                n = int(these.n.sum())
                note = "; ".join(sorted(set(these.note) - {""}))
            if not applies:
                note = why_not
            elif n == 0:
                note = "; ".join(filter(None, ["nothing measured", note]))
            rows.append(
                {
                    "check": name,
                    "kind": "rendering",
                    "condition_id": cid,
                    "applies": applies,
                    "evidence_status": "mathematical",
                    "statistic": statistic,
                    "observed": observed,
                    "lower": lower,
                    "upper": upper,
                    "n": n,
                    "note": note,
                }
            )
    checks = pd.DataFrame(rows)
    # bounds hold to floating-point rounding: a median of 60 samples at 1500 Hz,
    # from timestamp differences, is 39.999999999999996 ms
    slack = ROUNDING * np.maximum(1.0, np.maximum(checks.lower.abs(), checks.upper.abs()))
    measured = (checks.kind != "rendering") | (checks.n > 0)
    checks["passed"] = (
        (checks.observed >= checks.lower - slack)
        & (checks.observed <= checks.upper + slack)
        & measured
    )
    return checks


def readiness(checks: pd.DataFrame, targets: pd.DataFrame) -> list[str]:
    """Why the report is not ready: every gated check that fails, and the
    target table's problems. Empty when ready."""
    reasons = target_table_problems(targets)
    failed = checks[checks.applies & ~checks.passed]
    for row in failed.itertuples():
        reasons.append(
            f"{row.kind} check {row.check} fails in {row.condition_id}: {row.statistic} "
            f"{row.observed:.4g} outside [{row.lower:g}, {row.upper:g}]"
            + (f" ({row.note})" if row.note else "")
        )
    return reasons


def build_measurements(results: Sequence[SessionResult]) -> pd.DataFrame:
    """Every session's summary rows, plus per-group summaries of its samples."""
    frames = []
    for result in results:
        collector = _Collector()
        for (quantity, group), rows in result.samples.groupby(
            ["quantity", "group"], sort=True
        ):
            if rows.y.notna().any():
                pairs = rows[np.isfinite(rows.x) & np.isfinite(rows.y)]
                if quantity.startswith("sharp_wave_ripple") and len(pairs) >= 3:
                    collector.measure(
                        quantity,
                        group,
                        "pearson_r",
                        stats.pearsonr(pairs.x, pairs.y)[0],
                        len(pairs),
                    )
                    collector.measure(
                        quantity,
                        group,
                        "spearman_r",
                        stats.spearmanr(pairs.x, pairs.y)[0],
                        len(pairs),
                    )
                else:
                    collector.measure(quantity, group, "x_sum", rows.x.sum(), len(rows))
                    collector.measure(quantity, group, "y_sum", rows.y.sum(), len(rows))
                continue
            collector.summarize(quantity, group, rows.x)
        summary = pd.DataFrame(
            collector.measurements, columns=["quantity", "group", "statistic", "value", "n"]
        )
        frame = pd.concat([result.measurements, summary], ignore_index=True)
        frames.append(
            frame.assign(condition_id=result.condition_id, replicate=result.replicate)
        )
    columns = ["condition_id", "replicate", "group", "quantity", "statistic", "value", "n"]
    return pd.concat(frames, ignore_index=True)[columns]


# ---------------------------------------------------------------------------
# Report and specification


def _canonical(value: Any) -> Any:
    """JSON types: tuples as lists, keys sorted."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flat: dict[str, Any] = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}{key}."))
        return flat
    return {prefix.rstrip("."): value}


def revision_records(revisions: Sequence[ReferenceRevision]) -> list[dict[str, Any]]:
    """Each revision of a ``REFERENCE`` value as a dict of JSON types, the
    form ``spec.json`` holds."""
    return [_canonical(asdict(revision)) for revision in revisions]


def _revision_lines(revisions: Sequence[ReferenceRevision]) -> list[str]:
    """The report's table of ``REFERENCE`` revisions."""
    if not revisions:
        return ["None: every reference value is as first set."]
    lines = [
        "| parameter | previous | revised | reason | evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| `{r['key']}` | {json.dumps(r['previous'])} | {json.dumps(r['revised'])} | "
        f"{r['reason']} | {r['evidence']} |"
        for r in revision_records(revisions)
    ]
    return lines


def _versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "ripple_detection": rd.__version__,
    }


def validate(
    validation_id: str,
    selected: Sequence[Condition],
    n_replicates: int = DEFAULT_REPLICATES,
    duration: float | None = None,
    workers: int = 1,
    output_root: Path = OUTPUT_ROOT,
    figures: bool = True,
) -> Path:
    """Simulate, measure and write a validation report.

    Parameters
    ----------
    validation_id : str
        The report's directory name under ``output_root``.
    selected : sequence of Condition
    n_replicates : int, optional
        Replicates ``FIRST_REPLICATE`` up.
    duration : float, optional
        ``session.duration_s`` override, seconds.
    workers : int, optional
        Processes; 1 measures in this process.
    output_root : pathlib.Path, optional
    figures : bool, optional
        Draw the PNGs (needs matplotlib).

    Returns
    -------
    spec_path : pathlib.Path
        The written ``spec.json``. The directory is replaced whole.
    """
    if n_replicates < 1:
        msg = f"n_replicates must be at least 1, got {n_replicates}."
        raise ValueError(msg)
    if not selected:
        msg = "No condition selected."
        raise ValueError(msg)
    started = clock.perf_counter()
    overrides = {} if duration is None else {"session.duration_s": float(duration)}
    replicates = list(range(FIRST_REPLICATE, FIRST_REPLICATE + n_replicates))
    tasks = [
        (condition, replicate, overrides, replicate == replicates[0])
        for condition in selected
        for replicate in replicates
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_measure_task, tasks))
    else:
        results = [_measure_task(task) for task in tasks]
    parameters = {c.condition_id: resolve(c, overrides) for c in selected}
    return write_report(
        output_root / validation_id,
        validation_id,
        results,
        selected,
        parameters,
        replicates,
        overrides,
        figures,
        clock.perf_counter() - started,
    )


def _measure_task(task: tuple[Condition, int, Mapping[str, Any], bool]) -> SessionResult:
    condition, replicate, overrides, keep_traces = task
    return measure_session(condition, replicate, overrides, keep_traces)


def write_report(
    directory: Path,
    validation_id: str,
    results: Sequence[SessionResult],
    selected: Sequence[Condition],
    parameters: Mapping[str, Mapping[str, Mapping[str, Any]]],
    replicates: Sequence[int],
    overrides: Mapping[str, Any],
    figures: bool,
    runtime: float,
) -> Path:
    """Write the report's files into ``directory`` (replacing it) and return
    ``spec.json``'s path."""
    targets = load_targets()
    checks = build_checks(results, selected, parameters, targets)
    measurements = build_measurements(results)
    reasons = readiness(checks, targets)
    partial_directory = directory.with_name(directory.name + ".partial")
    shutil.rmtree(partial_directory, ignore_errors=True)
    partial_directory.mkdir(parents=True)
    measurements.to_csv(partial_directory / "measurements.csv", index=False)
    checks.to_csv(partial_directory / "checks.csv", index=False)
    pictures = _figures(partial_directory, results, checks, selected) if figures else []
    (partial_directory / "report.md").write_text(
        _report_text(
            validation_id, checks, reasons, selected, replicates, overrides, pictures, results
        )
    )
    artifacts = {path.name: _file_hash(path) for path in sorted(partial_directory.iterdir())}
    spec = {
        "validation_id": validation_id,
        "status": "not_ready" if reasons else "ready",
        "reasons": reasons,
        "simulation_fingerprint": simulation_fingerprint(),
        "target_table_hash": target_table_hash(),
        "first_replicate": replicates[0],
        "replicates": list(replicates),
        "seeds": {str(r): session_seed(r) for r in replicates},
        "overrides": dict(overrides),
        "conditions": {cid: _canonical(value) for cid, value in parameters.items()},
        "reference_revisions": revision_records(REFERENCE_REVISIONS),
        "versions": _versions(),
        "validator_hash": _file_hash(Path(__file__)),
        "runtime_s": runtime,
        "sessions": [
            {
                "condition_id": r.condition_id,
                "replicate": r.replicate,
                "seconds": r.seconds,
                "peak_rss_bytes": r.peak_rss_bytes,
            }
            for r in results
        ],
        "artifacts": artifacts,
    }
    (partial_directory / "spec.json").write_text(json.dumps(spec, indent=2, allow_nan=False))
    shutil.rmtree(directory, ignore_errors=True)
    partial_directory.rename(directory)
    return directory / "spec.json"


def require_ready_report(
    spec_path: str | Path, resolved: Mapping[str, Mapping[str, Any]]
) -> str:
    """Check that a validation report can back a run of ``resolved``.

    Parameters
    ----------
    spec_path : str or pathlib.Path
        The report's ``spec.json``.
    resolved : mapping of str to mapping
        ``condition_id`` to its resolved parameters (``conditions.resolve``).

    Returns
    -------
    digest : str
        SHA-256 of ``spec.json``, for the run's specification.

    Raises
    ------
    ReportNotReady
        Listing every problem: no readable ``spec.json``; a status other than
        ready; fewer replicates than ``DEFAULT_REPLICATES``; a simulation
        fingerprint or target-table hash other than the current one; an
        artifact missing or changed; a condition of
        ``resolved`` the report does not cover, or covers with other
        parameters.
    """
    path = Path(spec_path)
    try:
        spec = json.loads(path.read_text())
    except FileNotFoundError:
        msg = f"No simulator validation report at {path}: run validate_simulator.py first."
        raise ReportNotReady(msg) from None
    except (OSError, json.JSONDecodeError) as error:
        msg = f"The simulator validation report {path} cannot be read: {error}"
        raise ReportNotReady(msg) from None
    problems = []
    if spec.get("status") != "ready":
        stated = "; ".join(spec.get("reasons") or []) or "no reason recorded"
        problems.append(f"its status is {spec.get('status')!r}, not 'ready' ({stated})")
    n_replicates = len(spec.get("replicates") or [])
    if n_replicates < DEFAULT_REPLICATES:
        problems.append(
            f"it has {n_replicates} replicates, fewer than the {DEFAULT_REPLICATES} "
            "predeclared (DEFAULT_REPLICATES)"
        )
    if spec.get("simulation_fingerprint") != simulation_fingerprint():
        problems.append(
            "the simulation code has changed since it was made "
            "(simulation_fingerprint differs)"
        )
    if spec.get("target_table_hash") != target_table_hash():
        problems.append(
            "simulator_targets.csv has changed since it was made (target_table_hash differs)"
        )
    artifacts = spec.get("artifacts") or {}
    if not artifacts:
        problems.append("it records no artifact hashes")
    for name, digest in sorted(artifacts.items()):
        artifact = path.parent / name
        if not artifact.is_file():
            problems.append(f"artifact {name} is missing")
        elif _file_hash(artifact) != digest:
            problems.append(f"artifact {name} does not match its recorded hash")
    covered = spec.get("conditions") or {}
    for condition_id, parameters in resolved.items():
        if condition_id not in covered:
            problems.append(f"condition {condition_id} is not covered")
            continue
        wanted, have = _flatten(_canonical(dict(parameters))), _flatten(covered[condition_id])
        differing = sorted(
            k for k in wanted.keys() | have.keys() if wanted.get(k) != have.get(k)
        )
        if differing:
            problems.append(
                f"condition {condition_id} was validated with other settings: "
                f"{', '.join(differing)}"
            )
    if problems:
        msg = (
            f"The simulator validation report {path} cannot back this run:\n- "
            + "\n- ".join(problems)
        )
        raise ReportNotReady(msg)
    return _file_hash(path)


def _format(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.4g}"


def _report_text(
    validation_id: str,
    checks: pd.DataFrame,
    reasons: Sequence[str],
    selected: Sequence[Condition],
    replicates: Sequence[int],
    overrides: Mapping[str, Any],
    pictures: Sequence[str],
    results: Sequence[SessionResult],
) -> str:
    """The report's Markdown."""
    status = "not ready" if reasons else "ready"
    lines = [
        f"# Simulator validation `{validation_id}`",
        "",
        (
            f"Status: **{status}**. Conditions: {len(selected)}; replicates "
            f"{replicates[0]}-{replicates[-1]}; overrides: {dict(overrides) or 'none'}; "
            f"simulation fingerprint `{simulation_fingerprint()[:16]}`; target table "
            f"`{target_table_hash()[:16]}`."
        ),
        "",
        (
            "Readiness means the rendering checks pass and every source-based target passes "
            "in the conditions it applies to. It does not certify biological realism: "
            "assumed properties stay assumed, and the stress levels of the grid are "
            "reported, not gated."
        ),
        "",
    ]
    if reasons:
        lines += ["## Why it is not ready", "", *(f"- {reason}" for reason in reasons), ""]
    lines += [
        "## Targets",
        "",
        (
            "Observed statistic pooled over the replicates; bounds include the allowance "
            "stated in the target table. Gated rows are marked `*`."
        ),
        "",
    ]
    targets = checks[checks.kind == "target"]
    shown = [
        c.condition_id
        for c in selected
        if c.condition_id == "reference" or c.condition_id in MODEL_LABELS.values()
    ]
    header = "| target | evidence | statistic | bounds | " + " | ".join(shown) + " |"
    lines += [header, "|" + " --- |" * (4 + len(shown))]
    for quantity, rows in targets.groupby("check", sort=False):
        first = rows.iloc[0]
        cells = []
        for cid in shown:
            row = rows[rows.condition_id == cid].iloc[0]
            mark = ("*" if row.applies else "") + ("" if row.passed else " (out)")
            cells.append(f"{_format(row.observed)}{mark}")
        lines.append(
            f"| {quantity} | {first.evidence_status} | {first.statistic} | "
            f"[{first.lower:g}, {first.upper:g}] | " + " | ".join(cells) + " |"
        )
    rendering = checks[checks.kind == "rendering"]
    lines += [
        "",
        "## Rendering checks",
        "",
        "| check | statistic | bounds | passing, of the conditions it applies to | worst |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, rows in rendering.groupby("check", sort=False):
        applicable = rows[rows.applies]
        worst = applicable.observed.abs().max(skipna=False) if len(applicable) else np.nan
        lines.append(
            f"| {name} | {rows.statistic.iloc[0]} | [{rows.lower.iloc[0]:g}, "
            f"{rows.upper.iloc[0]:g}] | {int(applicable.passed.sum())}/{len(applicable)} | "
            f"{_format(worst)} |"
        )
    lines += ["", *(f"- `{name}`: {text}" for name, text in RENDERING_DESCRIPTIONS.items())]
    others = [c.condition_id for c in selected if c.condition_id not in shown]
    if others:
        lines += [
            "",
            "## Stress levels (reported, not gated)",
            "",
            "| condition | " + " | ".join(targets.check.unique()) + " |",
            "|" + " --- |" * (1 + targets.check.nunique()),
        ]
        for cid in others:
            rows = targets[targets.condition_id == cid]
            lines.append(f"| {cid} | " + " | ".join(_format(v) for v in rows.observed) + " |")
    lines += [
        "",
        "## Measurement choices",
        "",
        *(f"- {choice}" for choice in MEASUREMENT_CHOICES),
    ]
    lines += [
        "",
        "## Limitations",
        "",
        (
            "- Every property whose target is `assumed`, and every simulator parameter the "
            "draw_network_events Notes mark as assumed, remains an assumption."
        ),
        (
            "- The reference draws event strengths independently: its lack of coupling "
            "between sharp-wave and ripple magnitude is an assumed control, not physiology."
        ),
        (
            "- Ripples chirp linearly and only downward; recorded ripples fall faster around "
            "the peak and a quarter rise (ripple_frequency_decline is reported, not gated)."
        ),
        (
            "- Correlations, doublet spacing and the sharp-wave width check what the rendered "
            "event table carries more than the rendering itself."
        ),
        (
            "- A Hilbert envelope of a sampled carrier differs from the modulation envelope; "
            "the ripple_hilbert_* measurements report by how much."
        ),
        "",
        "## Parameter revisions",
        "",
        *_revision_lines(REFERENCE_REVISIONS),
        "",
    ]
    total = sum(r.seconds for r in results)
    lines += [f"Session time: {total:.1f} s over {len(results)} sessions.", ""]
    lines += [f"![{name}]({name})" for name in pictures]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Figures (matplotlib is imported only here)


def _figures(
    directory: Path,
    results: Sequence[SessionResult],
    checks: pd.DataFrame,
    selected: Sequence[Condition],
) -> list[str]:
    """Draw the report's PNGs into ``directory``; return their names."""
    import matplotlib as mpl

    mpl.use("Agg")
    shown = [
        c.condition_id
        for c in selected
        if c.condition_id == "reference" or c.condition_id in MODEL_LABELS.values()
    ]
    names = [
        _plot_targets(directory, checks, shown),
        _plot_distributions(directory, results, shown),
        _plot_background(directory, results, shown),
    ]
    names += _plot_examples(directory, results, shown)
    return names


def _plot_targets(directory: Path, checks: pd.DataFrame, shown: Sequence[str]) -> str:
    import matplotlib.pyplot as plt

    targets = checks[(checks.kind == "target") & checks.condition_id.isin(shown)]
    quantities = list(targets.check.unique())
    n_columns = 4
    n_rows = int(np.ceil(len(quantities) / n_columns))
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(12, 2.5 * n_rows), squeeze=False)
    for axis, quantity in zip(axes.flat, quantities, strict=False):
        rows = targets[targets.check == quantity].set_index("condition_id").loc[list(shown)]
        axis.axhspan(rows.lower.iloc[0], rows.upper.iloc[0], color="0.85")
        colours = ["tab:blue" if a else "0.5" for a in rows.applies]
        axis.scatter(range(len(rows)), rows.observed, c=colours, s=18, zorder=3)
        axis.set_xticks(
            range(len(rows)), [c.split("=")[-1] for c in rows.index], rotation=60, fontsize=7
        )
        axis.set_title(f"{quantity}\n({rows.evidence_status.iloc[0]})", fontsize=7)
        axis.tick_params(labelsize=7)
    for axis in axes.flat[len(quantities) :]:
        axis.set_visible(False)
    figure.tight_layout()
    name = "targets.png"
    figure.savefig(directory / name, dpi=80)
    plt.close(figure)
    return name


def _plot_distributions(
    directory: Path, results: Sequence[SessionResult], shown: Sequence[str]
) -> str:
    import matplotlib.pyplot as plt

    samples = pd.concat(
        [r.samples.assign(condition_id=r.condition_id) for r in results], ignore_index=True
    )
    quantities = [
        "ripple_peak_frequency",
        "ripple_duration",
        "sharp_wave_duration",
        "observed_participation",
    ]
    figure, axes = plt.subplots(1, len(quantities) + 1, figsize=(15, 2.8))
    for axis, quantity in zip(axes, quantities, strict=False):
        for cid in shown:
            values = samples[(samples.condition_id == cid) & (samples.quantity == quantity)].x
            values = values[np.isfinite(values)]
            if len(values):
                axis.hist(values, bins=30, histtype="step", density=True, label=cid)
        axis.set_title(quantity, fontsize=8)
        axis.tick_params(labelsize=7)
    axes[0].legend(fontsize=5)
    joint = axes[-1]
    for cid in ("reference", "strength_correlation=coupled"):
        pairs = samples[
            (samples.condition_id == cid) & (samples.quantity == "sharp_wave_ripple_power")
        ]
        if len(pairs):
            joint.scatter(pairs.x, pairs.y, s=4, alpha=0.5, label=cid)
    joint.set_xlabel("sharp-wave amplitude", fontsize=7)
    joint.set_ylabel("ripple SNR squared", fontsize=7)
    joint.legend(fontsize=5)
    joint.tick_params(labelsize=7)
    figure.tight_layout()
    name = "distributions.png"
    figure.savefig(directory / name, dpi=80)
    plt.close(figure)
    return name


def _plot_background(
    directory: Path, results: Sequence[SessionResult], shown: Sequence[str]
) -> str:
    import matplotlib.pyplot as plt

    figure, (psd_axis, power_axis) = plt.subplots(1, 2, figsize=(10, 3))
    for result in results:
        if result.condition_id not in shown or "psd" not in result.traces:
            continue
        frequency, psd = result.traces["psd"]
        psd_axis.loglog(frequency[1:], psd[1:], lw=0.8, label=result.condition_id)
        centers, power = result.traces["window_power"]
        power_axis.plot(centers, power, lw=0.8, label=result.condition_id)
    psd_axis.set_xlabel("Hz", fontsize=7)
    psd_axis.set_title("noise-only channel 0 PSD", fontsize=8)
    power_axis.set_xlabel("s", fontsize=7)
    power_axis.set_title(f"ripple-band power, {POWER_WINDOW:g} s windows", fontsize=8)
    power_axis.legend(fontsize=5)
    for axis in (psd_axis, power_axis):
        axis.tick_params(labelsize=7)
    figure.tight_layout()
    name = "background.png"
    figure.savefig(directory / name, dpi=80)
    plt.close(figure)
    return name


def _plot_examples(
    directory: Path, results: Sequence[SessionResult], shown: Sequence[str]
) -> list[str]:
    import matplotlib.pyplot as plt

    names = []
    for result in results:
        examples = result.traces.get("examples") or {}
        if result.condition_id not in shown or not examples:
            continue
        kinds = [k for k in (*rd.EVENT_TYPES, *rd.NON_EVENT_TYPES) if k in examples]
        figure, axes = plt.subplots(
            3, len(kinds), figsize=(2.2 * len(kinds), 5), squeeze=False
        )
        for column, kind in enumerate(kinds):
            example = examples[kind]
            axes[0, column].plot(example["time"], example["lfp"], lw=0.5)
            axes[1, column].plot(example["time"], example["radiatum"], lw=0.5)
            colours = np.where(
                example["unit_types"][example["spike_unit"]] == "interneuron", "tab:red", "k"
            )
            axes[2, column].scatter(
                example["spike_time"], example["spike_unit"], s=1, c=colours
            )
            axes[0, column].set_title(kind, fontsize=7)
            for axis in axes[:, column]:
                axis.tick_params(labelsize=6)
        axes[0, 0].set_ylabel("channel 0", fontsize=7)
        axes[1, 0].set_ylabel("radiatum", fontsize=7)
        axes[2, 0].set_ylabel("unit", fontsize=7)
        figure.suptitle(f"{result.condition_id}, replicate {result.replicate}", fontsize=8)
        figure.tight_layout()
        name = f"examples_{result.condition_id.replace('=', '-')}.png"
        figure.savefig(directory / name, dpi=70)
        plt.close(figure)
        names.append(name)
    return names


# ---------------------------------------------------------------------------
# Command line


def main(argv: Sequence[str] | None = None) -> int:
    """Run the validation from the command line; 0 when the report is ready."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--validation-id", required=True)
    parser.add_argument(
        "--conditions",
        default="all",
        help="'all' or ids, comma-separated (a crossed cell's id holds its own comma)",
    )
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--duration", type=float, default=None, help="seconds per session")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--no-figures", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        selected = select_conditions(arguments.conditions)
    except ValueError as error:
        parser.error(str(error))
    spec_path = validate(
        arguments.validation_id,
        selected,
        n_replicates=arguments.replicates,
        duration=arguments.duration,
        workers=arguments.workers,
        output_root=arguments.output_root,
        figures=not arguments.no_figures,
    )
    spec = json.loads(spec_path.read_text())
    print(f"{spec_path}: {spec['status']}")
    for reason in spec["reasons"]:
        print(f"- {reason}")
    return 0 if spec["status"] == "ready" else 1


if __name__ == "__main__":
    sys.exit(main())

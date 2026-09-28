"""Attribute the literature methods' disagreement to the components of their rules.

Methods differ in many components at once: the signal they threshold, its
smoothing, the period normalizing it, the threshold and bound, duration limits,
merging, the speed rule, a cell count, a state and a coincident partner event.
This module expresses methods as experimental compositions of those components,
built from the package's public primitives, and measures how much each matters:
one factor at a time from a reference configuration, Sobol indices over the
space the represented methods span, and Shapley decompositions of the
difference between two configurations.

Usage, from the repository root (see README.md, "Attribution")::

    uv run python examples/benchmark/attribution.py --run-name NAME
        --family spikes|lfp [--analysis oat|sobol|shapley|all] [--workers N]
        [--smoke] [--below-minimum] [--run-directory PATH]
        [--results-directory PATH]

It reads a finished run's reference condition (``conditions.csv``'s saved
parameters and ``conditions/reference/``), simulates its first ``K`` sessions
again and checks each against what the run saved before anything runs.

Templates. A template (``SpikeTemplate``, ``LfpTemplate``) is one flat, frozen
set of factor values; ``compile`` turns it into a ``Pipeline`` (a
``ThresholdCore`` and post steps, each a ``Step``) and ``run_pipeline`` runs
that on a session's ``SessionContext``. ``TEMPLATES`` maps a configuration of
``recipe_configs.RECIPES`` to the template written after reading its method's
body, with the source of each value; ``FIXED_POINTS`` gives every other
configuration and why it has none. A template stands for its method only when
``in_space`` finds identical events from both on the sessions checked:
positive controls (some event must be found), a gap of missing samples, a Unix
clock origin and the ``K`` reference sessions. The configuration's public call
(``recipe_configs.run_recipe``) stays the only source of its events in every
other analysis; a template never replaces it.

Families. ``spikes`` (a population rate on 1 ms bins) and ``lfp`` (the mean
ripple-band envelope of the first channels, or its square), each analysed
separately, since their factors differ. ``factor_space`` gives each factor's
range (continuous: the least and largest value among the family's represented
methods) or levels (categorical: the distinct values, in configuration order;
integer: every whole number between the least and largest); a factor with one
value is not varied. ``reference_template`` sets each factor to its median
(integers rounded down) or mode (ties to the first in configuration order).

Outputs ``Y``, each the mean over the ``K`` sessions (``Y_NAMES``): ``f1``,
against the family's expression (``burst`` for spikes, ``ripple`` for the LFP)
at 10 % of the peak, any overlap; ``f1_network``; ``events_per_minute``;
``onset_error_25``, the median signed onset error of the matched pairs against
the windows at 25 % (mean over the sessions with a pair); ``jaccard_reference``,
matched events over the union against the reference configuration's (1 when
both are empty).

Analyses (``--analysis``): ``oat`` (``one_at_a_time``), ``sobol`` (``sobol``, at
``SOBOL_N`` rows) and ``shapley`` (``shapley_pairs``). A family with fewer than
``MINIMUM_IN_SPACE`` represented methods (identical templates once) runs no Sobol
or Shapley analysis: the command stops and says so, unless ``--below-minimum``
is given, when it runs them and every output of the family carries
``family_caveat``'s label (a ``caveat`` column, and the figures' titles).
``--smoke`` evaluates ``SMOKE_CONFIGURATIONS`` configurations on one session and
prints the cost of each analysis, writing nothing.

Outputs: ``output/<run_name>/attribution/<family>_<analysis>.csv.gz`` (one row
per configuration and ``Y``: its factors, the ``Y``, the mean and each
session's value) and ``results/<run_name>/attribution/``:
``<family>_in_space.csv`` (the family's templates, every one verified, since the
command stops otherwise, and every fixed point with its reason),
``<family>_fixed_points.csv`` (every configuration without a template, with its
public-call ``Y``s against the family's expression and reference),
``<family>_factor_space.csv``, ``<family>_reference.csv``,
``<family>_oat.csv`` (each level's change in each ``Y``, with a 95 % paired
bootstrap interval over the sessions), ``<family>_sobol.csv`` (first-order and
total indices with 95 % bootstrap intervals over the rows) and
``<family>_shapley.csv`` (each pair's Shapley values and Monte Carlo standard
errors), with figures of the last two.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import os
import sys
import time as wall_clock
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from analyze import resample_weights, write_result
from conditions import parameters_from_json, session_seed, simulate_parameters
from numpy.typing import ArrayLike
from recipe_configs import (
    _POLICY,
    RECIPES,
    RecipeConfig,
    behavior_intervals,
    external_ripples,
    make_recording,
    rest_intervals,
    run_recipe,
)
from run import (
    _REPORT_IDENTITY,
    OUTPUT,
    RIPPLE_CHANNEL_COLUMNS,
    _integer_counts,
    _write_table,
    load_truth,
    read_table,
)
from scipy.stats import qmc
from validate_simulator import peak_rss_bytes, require_ready_report

import ripple_detection as rd
from ripple_detection.core import FloatArray, nearest_sample_index
from ripple_detection.literature_methods import (
    PopulationTrace,
    Recording,
    bounds,
    population_trace,
)

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent.parent
RESULTS = HERE / "results"

Params = tuple[tuple[str, Any], ...]

FAMILIES = ("spikes", "lfp")
# The expression each family's F1 is scored against.
FAMILY_EXPRESSION = {"spikes": "burst", "lfp": "ripple"}
REFERENCE_CONDITION = "reference"
# The reference sessions every Y averages: replicates 0 to K - 1.
K = 5
# The spikes family's population grid, in seconds.
BIN_WIDTH = 0.001
Y_NAMES = ("f1", "f1_network", "events_per_minute", "onset_error_25", "jaccard_reference")
# Fewer represented methods than this and a family runs no Sobol or Shapley.
MINIMUM_IN_SPACE = 8
SOBOL_N = 256
SOBOL_SEED = 0
N_SOBOL_RESAMPLES = 1000
OAT_POINTS = 5
SHAPLEY_EXACT_UP_TO = 8
SHAPLEY_PERMUTATIONS = 128
N_LOWEST_PAIRS = 10
SMOKE_CONFIGURATIONS = 20
# The short session the edge cases are cut from, and where its gap lies.
EDGE_DURATION = 60.0
GAP = (20.0, 21.0)
UNIX_ORIGIN = 1_700_000_000.0
# Pipelines' events kept per session, and traces (large arrays) per session.
EVENT_CACHE_SIZE = 4096
TRACE_CACHE_SIZE = 8


# Pipelines


@dataclasses.dataclass(frozen=True)
class Step:
    """One named operation of a pipeline with its parameters.

    Attributes
    ----------
    operation : str
        A signal operation (``"rate"``, ``"mean_envelope"``, ``"square"``) in a
        core's ``signal``, or a post step (``"active_units"``, ``"inside"``,
        ``"overlap"``, ``"contains_time"``) in a pipeline's ``steps``.
    parameters : tuple of (str, object) pairs
        Keyword arguments of the operation, each value hashable.
    """

    operation: str
    parameters: Params = ()


@dataclasses.dataclass(frozen=True)
class ThresholdCore:
    """The thresholding at the heart of a pipeline, ``detect_events_from_trace``'s.

    Attributes
    ----------
    signal : tuple of Step
        The trace: ``rate`` (units, smoothing_sigma: a ``population_trace`` on
        ``BIN_WIDTH`` bins), or ``mean_envelope`` (band, channels:
        ``Recording.mean_envelope``) followed by ``square`` or not.
    restrict_to : str or None
        Intervals (``"rest"``) outside which the trace is missing, so detection
        and its statistics are inside them only; None, none.
    normalization_period : str
        The samples the z-score statistics come from (``NORMALIZATION``).
    smoothing_sigma : float
        Gaussian SD in seconds applied by the detection (the LFP trace); 0.0, none.
    threshold, bound_threshold : float
        In SD.
    minimum_event_duration : float
        Seconds, the whole event; 0.0, none.
    maximum_duration : float or None
        Seconds; None, none.
    speed_rule : str
    speed_threshold : float
        cm/s; ``np.inf`` turns the speed rule off.
    close_event_threshold : float
        Events closer than this many seconds are merged; 0.0, none.
    """

    signal: tuple[Step, ...]
    restrict_to: str | None
    normalization_period: str
    smoothing_sigma: float
    threshold: float
    bound_threshold: float
    minimum_event_duration: float
    maximum_duration: float | None
    speed_rule: str
    speed_threshold: float
    close_event_threshold: float


@dataclasses.dataclass(frozen=True)
class Pipeline:
    """A threshold core and the post steps applied to its events, in order.

    Attributes
    ----------
    core : ThresholdCore
    steps : tuple of Step
    """

    core: ThresholdCore
    steps: tuple[Step, ...] = ()


# Templates


@dataclasses.dataclass(frozen=True)
class SpikeTemplate:
    """A configuration of the ``spikes`` family: one field per factor.

    Attributes
    ----------
    units : {"all", "place", "pyramidal"}
        The units the population rate pools, and whose activity
        ``minimum_active_units`` counts.
    smoothing_sigma : float
        Gaussian SD of the rate, in seconds, on 1 ms bins.
    normalization_period : str
        A key of ``NORMALIZATION``.
    threshold : float
        In SD.
    bound_fraction : float
        The bound as a fraction of the threshold: events extend to where the
        trace falls below ``bound_fraction * threshold`` SD; 0.0, the mean.
        Between 0 and 1, so every combination of factors bounds at or below
        its threshold.
    minimum_event_duration : float
        Seconds; 0.0, none.
    maximum_duration : float or None
        Seconds; None, none.
    merge_gap : float
        Seconds between bin edges below which events merge; 0.0, none.
    speed : str
        A key of ``SPEED``: the speed rule and its threshold together.
    minimum_active_units : int
        0, none.
    state : str
        A key of ``STATES``.
    coincidence : str
        A key of ``COINCIDENCES``.
    """

    units: str
    smoothing_sigma: float
    normalization_period: str
    threshold: float
    bound_fraction: float
    minimum_event_duration: float
    maximum_duration: float | None
    merge_gap: float
    speed: str
    minimum_active_units: int
    state: str
    coincidence: str

    def __post_init__(self) -> None:
        _lists_to_tuples(self)


@dataclasses.dataclass(frozen=True)
class LfpTemplate:
    """A configuration of the ``lfp`` family: one field per factor.

    Attributes
    ----------
    band : pair of float
        Hz.
    channels : int or None
        The first this many channels' envelopes are averaged; None, every one.
    trace : {"amplitude", "squared"}
        The mean envelope, or its square.
    smoothing_sigma : float
        Gaussian SD in seconds; 0.0, none.
    normalization_period, threshold, bound_fraction, minimum_event_duration, \
maximum_duration, merge_gap, speed, state, coincidence
        As ``SpikeTemplate``'s.
    """

    band: tuple[float, float]
    channels: int | None
    trace: str
    smoothing_sigma: float
    normalization_period: str
    threshold: float
    bound_fraction: float
    minimum_event_duration: float
    maximum_duration: float | None
    merge_gap: float
    speed: str
    state: str
    coincidence: str

    def __post_init__(self) -> None:
        _lists_to_tuples(self)


def _lists_to_tuples(template: SpikeTemplate | LfpTemplate) -> None:
    """A template's list values (a band read from JSON) as tuples, so its
    pipeline hashes."""
    for field in dataclasses.fields(template):
        value = getattr(template, field.name)
        if isinstance(value, list):
            object.__setattr__(template, field.name, tuple(value))


Template = SpikeTemplate | LfpTemplate
TemplateT = TypeVar("TemplateT", SpikeTemplate, LfpTemplate)

# Each factor's kind: "continuous", "integer" or "categorical".
FACTOR_KINDS: dict[str, str] = {
    "units": "categorical",
    "band": "categorical",
    "channels": "categorical",
    "trace": "categorical",
    "smoothing_sigma": "continuous",
    "normalization_period": "categorical",
    "threshold": "continuous",
    "bound_fraction": "categorical",
    "minimum_event_duration": "continuous",
    "maximum_duration": "categorical",
    "merge_gap": "continuous",
    "speed": "categorical",
    "minimum_active_units": "integer",
    "state": "categorical",
    "coincidence": "categorical",
}

_NO_SPEED_RULE = ("endpoints", np.inf)
# Speed levels: (speed_rule, speed_threshold). "<" is the next float below.
SPEED: dict[str, tuple[str, float]] = {
    "none": _NO_SPEED_RULE,
    "endpoints<=5": ("endpoints", 5.0),
    "endpoints<5": ("endpoints", float(np.nextafter(5.0, -np.inf))),
    "endpoints<4": ("endpoints", float(np.nextafter(4.0, -np.inf))),
    "all<=3": ("all", 3.0),
    "all<=5": ("all", 5.0),
    "restrict<5": ("restrict", float(np.nextafter(5.0, -np.inf))),
}
# The samples each normalization period takes its statistics from; the
# statistics are always over valid (finite) samples of the trace.
NORMALIZATION = ("session", "speed<5", "speed<4", "rest")
# State levels: a trace restriction (RESTRICTIONS) or a post step.
STATES: dict[str, Step | None] = {
    "none": None,
    "restrict:rest": None,
    "inside:rest": Step("inside", (("intervals", "rest"),)),
    "overlap:running_30s": Step("overlap", (("partner", "running_30s"),)),
}
# The state levels that restrict the trace, to the intervals of the partner named.
RESTRICTIONS: dict[str, str] = {"restrict:rest": "rest"}
# Coincidence levels: a whole post step with its partner.
COINCIDENCES: dict[str, Step | None] = {
    "none": None,
    "overlap:long_swrs": Step("overlap", (("partner", "long_swrs"),)),
    "overlap:muessig_2019_ripples": Step("overlap", (("partner", "muessig_2019_ripples"),)),
    "peak_inside:external_ripples": Step(
        "contains_time", (("partner", "external_ripple_peaks"),)
    ),
}


def family_of(template: Template) -> str:
    """``"spikes"`` or ``"lfp"``: the family of a template's signal.

    Parameters
    ----------
    template : SpikeTemplate or LfpTemplate

    Returns
    -------
    family : str
    """
    return "spikes" if isinstance(template, SpikeTemplate) else "lfp"


def pipeline_family(pipeline: Pipeline) -> str:
    """``"spikes"`` for a population-rate core, else ``"lfp"``.

    Parameters
    ----------
    pipeline : Pipeline

    Returns
    -------
    family : str
    """
    return "spikes" if pipeline.core.signal[0].operation == "rate" else "lfp"


def compile(template: Template) -> Pipeline:
    """The pipeline a template stands for, in one fixed order.

    The core's signal (a 1 ms population rate of the template's units with its
    smoothing, or the mean envelope and its square), the state's restriction
    if the state is one, normalization, threshold, bound, duration limits,
    the speed rule and merging; then the post steps: the active-unit count when
    above 0, the state when it is an overlap or containment step, and the
    coincidence step.

    Parameters
    ----------
    template : SpikeTemplate or LfpTemplate

    Returns
    -------
    pipeline : Pipeline

    Raises
    ------
    ValueError
        A level the tables (``SPEED``, ``NORMALIZATION``, ``STATES``,
        ``COINCIDENCES``) do not define, or a trace other than ``"amplitude"``
        and ``"squared"``.
    TypeError
        A factor value is not hashable, so the pipeline's events could not be
        cached; the message names the template.
    """
    for table, value in (
        (SPEED, template.speed),
        (NORMALIZATION, template.normalization_period),
        (STATES, template.state),
        (COINCIDENCES, template.coincidence),
    ):
        if value not in table:
            msg = f"{value!r} is not a level this module defines; see {sorted(table)}."
            raise ValueError(msg)
    if isinstance(template, SpikeTemplate):
        signal: tuple[Step, ...] = (
            Step(
                "rate",
                (("units", template.units), ("smoothing_sigma", template.smoothing_sigma)),
            ),
        )
        smoothing = 0.0
    else:
        signal = (
            Step("mean_envelope", (("band", template.band), ("channels", template.channels))),
        )
        if template.trace == "squared":
            signal += (Step("square"),)
        elif template.trace != "amplitude":
            msg = f"trace must be 'amplitude' or 'squared'; got {template.trace!r}."
            raise ValueError(msg)
        smoothing = template.smoothing_sigma
    speed_rule, speed_threshold = SPEED[template.speed]
    core = ThresholdCore(
        signal=signal,
        restrict_to=RESTRICTIONS.get(template.state),
        normalization_period=template.normalization_period,
        smoothing_sigma=smoothing,
        threshold=template.threshold,
        bound_threshold=template.bound_fraction * template.threshold,
        minimum_event_duration=template.minimum_event_duration,
        maximum_duration=template.maximum_duration,
        speed_rule=speed_rule,
        speed_threshold=speed_threshold,
        close_event_threshold=template.merge_gap,
    )
    steps = []
    if isinstance(template, SpikeTemplate) and template.minimum_active_units > 0:
        steps.append(
            Step(
                "active_units",
                (("units", template.units), ("minimum", template.minimum_active_units)),
            )
        )
    post = (STATES[template.state], COINCIDENCES[template.coincidence])
    steps.extend(step for step in post if step is not None)
    pipeline = Pipeline(core, tuple(steps))
    try:
        hash(pipeline)
    except TypeError as error:
        msg = f"{template} is not hashable, so its events cannot be cached: {error}."
        raise TypeError(msg) from error
    return pipeline


# Sessions


class _Cache:
    """A least-recently-used mapping holding at most ``size`` entries."""

    def __init__(self, size: int) -> None:
        self.size = size
        self._entries: OrderedDict[Hashable, Any] = OrderedDict()

    def get(self, key: Hashable, compute: Callable[[], Any]) -> Any:
        if key in self._entries:
            self._entries.move_to_end(key)
            return self._entries[key]
        value = compute()
        self._entries[key] = value
        if len(self._entries) > self.size:
            self._entries.popitem(last=False)
        return value

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()


def _counted(session: rd.SimulatedSession) -> rd.SimulatedSession:
    """The runner's integer copy of the spike counts where it holds them exactly
    (``run._integer_counts``); counts with missing (NaN) samples stay float."""
    with np.errstate(invalid="ignore"):
        return _integer_counts(session)


def _read_only(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    values.flags.writeable = False
    return values


class SessionContext:
    """One session's inputs for running pipelines, and what they have computed.

    The context owns its recording (``Recording.from_arrays`` of the session's
    LFPs, radiatum channel, speed and spike counts, with the input policy's
    place and pyramidal selections), whose arrays it makes read-only, so
    nothing cached from them can go stale. Traces are cached by the immutable
    parameters that make them (at most ``TRACE_CACHE_SIZE``), a pipeline's
    events by the pipeline (at most ``EVENT_CACHE_SIZE``), a partner's events by
    its name; ``release`` empties every cache. Nothing is attached to a
    recording the package or a caller holds.

    Parameters
    ----------
    session : SimulatedSession
        A network session with unit types and running bouts.
    label : str
        Names the session in messages, such as ``"reference/0"``.

    Attributes
    ----------
    session : SimulatedSession
        With its spike counts as int16 when that holds them exactly.
    label : str
    recording : Recording
    rest : ndarray, shape (n_intervals, 2)
        ``recipe_configs.rest_intervals``: outside every running bout.
    minutes : float
        The session's length.
    windows : dict of str to dict of float to ndarray
        By expression and fraction (0.1, 0.25), the truth windows' bounds.
    n_runs : int
        How many pipelines this context has run, a cache miss each.
    """

    def __init__(self, session: rd.SimulatedSession, label: str) -> None:
        self.session = _counted(session)
        self.label = label
        self.recording = Recording.from_arrays(
            session.time,
            session.sampling_frequency,
            lfps=session.lfps,
            multiunit=self.session.multiunit,
            speed=session.speed,
            sharp_wave_lfp=session.sharp_wave_lfp,
            place_cells=_POLICY["place_cells"].get(session),
            pyramidal=_POLICY["pyramidal"].get(session),
        )
        signals = self.recording.session
        for values in (
            signals.time,
            signals.lfps,
            signals.raw_lfp,
            signals.sharp_wave_lfp,
            signals.multiunit,
            self.recording.place_cells,
            self.recording.pyramidal,
        ):
            _read_only(values)
        if signals.speed is not None:
            _read_only(signals.speed)
        self.rest = _read_only(rest_intervals(session))
        self.minutes = len(session.time) / session.sampling_frequency / 60
        self.windows = {
            expression: {
                fraction: _read_only(
                    bounds(rd.truth_windows(session.events, fraction, expression))
                )
                for fraction in (0.1, 0.25)
            }
            for expression in ("network", *FAMILY_EXPRESSION.values())
        }
        self.n_runs = 0
        self._traces = _Cache(TRACE_CACHE_SIZE)
        self._events = _Cache(EVENT_CACHE_SIZE)
        self._partners: dict[str, FloatArray] = {}

    def release(self) -> None:
        """Empty every cache: traces, events and partners."""
        self._traces.clear()
        self._events.clear()
        self._partners.clear()

    def units(self, name: str) -> np.ndarray[Any, Any] | None:
        """The unit selection ``name`` names: None for ``"all"``."""
        if name == "all":
            return None
        if name in ("place", "pyramidal"):
            selection: np.ndarray[Any, Any] = getattr(
                self.recording, "place_cells" if name == "place" else "pyramidal"
            )
            return selection
        msg = f"units must be 'all', 'place' or 'pyramidal'; got {name!r}."
        raise ValueError(msg)

    def population(self, units: str, smoothing_sigma: float) -> PopulationTrace:
        """``population_trace`` of ``units`` on ``BIN_WIDTH`` bins, smoothed, its
        arrays read-only."""

        def compute() -> PopulationTrace:
            trace = population_trace(
                self.recording,
                bin_width=BIN_WIDTH,
                units=self.units(units),
                smoothing_sigma=smoothing_sigma,
            )
            for values in (
                trace.time,
                trace.data,
                trace.speed,
                trace.first_sample,
                trace.last_sample,
            ):
                if values is not None:
                    _read_only(values)
            return trace

        trace: PopulationTrace = self._traces.get(("rate", units, smoothing_sigma), compute)
        return trace

    def mean_envelope(self, band: tuple[float, float], channels: int | None) -> FloatArray:
        """``Recording.mean_envelope(band, channels)``, read-only."""
        return self._traces.get(  # type: ignore[no-any-return]
            ("mean_envelope", band, channels),
            lambda: _read_only(self.recording.mean_envelope(band, channels)),
        )

    def normalization_mask(self, period: str) -> np.ndarray[Any, Any] | None:
        """The input samples ``period`` takes statistics from; None, every one."""
        if period == "session":
            return None
        if period == "rest":
            return self.recording.intervals_to_mask(self.rest)
        limits = {"speed<5": 5.0, "speed<4": 4.0}
        if period in limits:
            return np.asarray(self.recording.speed < limits[period], dtype=bool)
        msg = f"normalization_period must be one of {NORMALIZATION}; got {period!r}."
        raise ValueError(msg)

    def partner(self, name: str) -> FloatArray:
        """A post step's partner: intervals, or times for ``contains_time``."""
        if name not in self._partners:
            self._partners[name] = _read_only(PARTNERS[name](self))
        return self._partners[name]

    def events(self, pipeline: Pipeline) -> FloatArray:
        """``run_pipeline(pipeline, self)``, run once while it stays cached."""

        def run() -> FloatArray:
            self.n_runs += 1
            return _read_only(run_pipeline(pipeline, self))

        return self._events.get(pipeline, run)  # type: ignore[no-any-return]


def _long_swrs(context: SessionContext) -> FloatArray:
    """liu_2023's SWRs: ``Long_sharp_wave_ripple_detector`` at its defaults on
    the first channel and the radiatum, speed unknown (the method's recording
    has none) and no speed rule."""
    recording = context.recording
    return bounds(
        rd.Long_sharp_wave_ripple_detector(
            recording.time,
            recording.session.raw_lfp,
            np.full(len(recording.time), np.nan),
            recording.fs,
            sharp_wave_lfp=recording.session.sharp_wave_lfp,
            speed_threshold=np.inf,
        )
    )


def _configured_events(config_id: str) -> Callable[[SessionContext], FloatArray]:
    """A configured method's public-call events on the context's session."""

    def events(context: SessionContext) -> FloatArray:
        config = _config(config_id)
        return recipe_events(config, context.session)

    return events


def _running_30s(context: SessionContext) -> FloatArray:
    """Running (speed above 15 cm/s) widened by 30 s on each side, davidson_2009's
    "within 30 s of running"."""
    recording = context.recording
    running = rd.state_intervals(recording.speed, recording.time, 15.0, comparison=">")
    return np.asarray(running + np.array([-30.0, 30.0]), dtype=float)


# Post steps' partners by name.
PARTNERS: dict[str, Callable[[SessionContext], FloatArray]] = {
    "rest": lambda context: context.rest,
    "running_30s": _running_30s,
    "long_swrs": _long_swrs,
    "muessig_2019_ripples": _configured_events("muessig_2019_ripples"),
    "external_ripple_peaks": lambda context: external_ripples(context.session)[:, 2],
}


def _spike_events(core: ThresholdCore, context: SessionContext) -> FloatArray:
    if len(core.signal) > 1:
        msg = (
            "A population-rate core takes no further signal step; got "
            f"{[step.operation for step in core.signal[1:]]}."
        )
        raise ValueError(msg)
    if core.smoothing_sigma != 0:
        msg = (
            "A population-rate core is smoothed by its rate step, not by the core's "
            f"smoothing; got smoothing_sigma={core.smoothing_sigma}."
        )
        raise ValueError(msg)
    parameters = dict(core.signal[0].parameters)
    trace = context.population(parameters["units"], parameters["smoothing_sigma"])
    if core.restrict_to is not None:
        inside = trace.bins_inside(context.partner(core.restrict_to))
        trace = dataclasses.replace(trace, data=np.where(inside, trace.data, np.nan))
    options = _detection_options(core)
    mask = context.normalization_mask(core.normalization_period)
    if mask is not None:
        options["normalization_mask"] = mask[
            nearest_sample_index(context.recording.time, trace.time)
        ]
    return bounds(trace.detect(**options))


def _lfp_events(core: ThresholdCore, context: SessionContext) -> FloatArray:
    parameters = dict(core.signal[0].parameters)
    values = context.mean_envelope(parameters["band"], parameters["channels"])
    for step in core.signal[1:]:
        if step.operation != "square":
            msg = f"Unknown signal operation {step.operation!r}."
            raise ValueError(msg)
        values = values**2
    recording = context.recording
    if core.restrict_to is not None:
        inside = recording.intervals_to_mask(context.partner(core.restrict_to))
        values = np.where(inside, values, np.nan)
    options = _detection_options(core)
    if core.smoothing_sigma > 0:
        options["smoothing_sigma"] = core.smoothing_sigma
    mask = context.normalization_mask(core.normalization_period)
    if mask is not None:
        options["normalization_mask"] = mask
    return bounds(
        rd.detect_events_from_trace(
            recording.time, values, recording.speed, recording.fs, **options
        )
    )


def _detection_options(core: ThresholdCore) -> dict[str, Any]:
    """``detect_events_from_trace`` keywords of a core, its signal aside: one
    sample above the threshold is enough (every represented method's rule)."""
    options: dict[str, Any] = {
        "threshold": core.threshold,
        "bound_threshold": core.bound_threshold,
        "minimum_duration": 0.0,
        "maximum_duration": core.maximum_duration,
        "speed_rule": core.speed_rule,
        "speed_threshold": core.speed_threshold,
    }
    if core.minimum_event_duration > 0:
        options["minimum_event_duration"] = core.minimum_event_duration
    if core.close_event_threshold > 0:
        options["close_event_threshold"] = core.close_event_threshold
        options["close_event_rule"] = "merge"
    return options


def _post_step(step: Step, events: FloatArray, context: SessionContext) -> FloatArray:
    parameters = dict(step.parameters)
    recording = context.recording
    if step.operation == "active_units":
        kept = rd.require_active_units(
            events,
            recording.multiunit,
            recording.time,
            minimum_active_units=parameters["minimum"],
            units=context.units(parameters["units"]),
        )
    elif step.operation == "inside":
        kept = rd.require_inside(events, context.partner(parameters["intervals"]))
    elif step.operation == "overlap":
        kept = rd.require_overlap(events, context.partner(parameters["partner"]))
    elif step.operation == "contains_time":
        kept = rd.require_times_inside(events, context.partner(parameters["partner"]))
    else:
        msg = f"Unknown post step {step.operation!r}."
        raise ValueError(msg)
    return bounds(kept)


def run_pipeline(pipeline: Pipeline, context: SessionContext) -> FloatArray:
    """The events of a pipeline on one session.

    Parameters
    ----------
    pipeline : Pipeline
    context : SessionContext

    Returns
    -------
    events : ndarray, shape (n_events, 2)
        Closed ``[start_time, end_time]`` bounds on recorded timestamps, in
        time order.

    Raises
    ------
    ValueError
        An operation this module does not define, or what the package's
        functions raise.
    """
    core = pipeline.core
    operation = core.signal[0].operation
    if operation == "rate":
        events = _spike_events(core, context)
    elif operation == "mean_envelope":
        events = _lfp_events(core, context)
    else:
        msg = f"Unknown signal operation {operation!r}."
        raise ValueError(msg)
    for step in pipeline.steps:
        events = _post_step(step, events, context)
    return events


# Which configurations have a template


def _spikes(**values: Any) -> SpikeTemplate:
    """A spike template, a factor the method does not use at its "none" value."""
    defaults: dict[str, Any] = {
        "normalization_period": "session",
        "bound_fraction": 0.0,
        "minimum_event_duration": 0.0,
        "maximum_duration": None,
        "merge_gap": 0.0,
        "speed": "none",
        "minimum_active_units": 0,
        "state": "none",
        "coincidence": "none",
    }
    return SpikeTemplate(**{**defaults, **values})


def _lfp(**values: Any) -> LfpTemplate:
    """An LFP template, a factor the method does not use at its "none" value."""
    defaults: dict[str, Any] = {
        "band": (150.0, 250.0),
        "channels": None,
        "trace": "amplitude",
        "smoothing_sigma": 0.0,
        "normalization_period": "session",
        "bound_fraction": 0.0,
        "minimum_event_duration": 0.0,
        "maximum_duration": None,
        "merge_gap": 0.0,
        "speed": "none",
        "state": "none",
        "coincidence": "none",
    }
    return LfpTemplate(**{**defaults, **values})


_REST_SLEEP = (
    "sleep_intervals are the input policy's rest (the samples outside the running "
    "bouts), so rec.sleep returns rest"
)

# Every template, by configuration, each written after reading the method's
# body in ripple_detection.literature_methods: the source of each value, and
# the benchmark's input policy where it supplies one. All population methods
# use _detect_population or _detect_population_in on 1 ms bins with
# minimum_duration=0.0; every value not listed is the method's absence of that
# step.
TEMPLATES: dict[str, tuple[Template, str]] = {
    "yang_2024": (
        _spikes(
            units="pyramidal",
            smoothing_sigma=0.015,
            normalization_period="rest",
            threshold=3.0,
            minimum_event_duration=0.05,
            maximum_duration=0.5,
            minimum_active_units=5,
            state="inside:rest",
            coincidence="peak_inside:external_ripples",
        ),
        (
            "_population_with_ripple_peak: pyramidal, 15 ms, statistics over sleep, 3 SD, "
            "50-500 ms, >= 5 pyramidal cells, an external ripple peak inside, inside the "
            f"eligible behavior_intervals. {_REST_SLEEP}; behavior_intervals are rest; the "
            "external ripples are the policy's Zugaro stand-in, peaks in its third column"
        ),
    ),
    "liu_2023": (
        _spikes(
            units="pyramidal",
            smoothing_sigma=0.010,
            threshold=2.0,
            minimum_event_duration=0.1,
            maximum_duration=0.5,
            coincidence="overlap:long_swrs",
        ),
        (
            "liu_2023: pyramidal bursts, 10 ms, 2 SD, 100-500 ms, overlapping a "
            "Long_sharp_wave_ripple_detector SWR at its defaults (speed unknown, no rule)"
        ),
    ),
    "igata_2021": (
        _spikes(
            units="all",
            smoothing_sigma=0.015,
            normalization_period="speed<5",
            threshold=2.0,
            minimum_event_duration=0.05,
            maximum_duration=2.0,
            minimum_active_units=5,
        ),
        (
            "igata_2021: every unit, 15 ms, statistics over speed < 5, 2 SD, 50 ms-2 s, "
            ">= 5 active units of every unit, no speed rule"
        ),
    ),
    "farooq_2019_neuron": (
        _spikes(
            units="pyramidal",
            smoothing_sigma=0.015,
            threshold=2.0,
            bound_fraction=1.0,
            minimum_event_duration=0.1,
            maximum_duration=0.8,
            minimum_active_units=5,
            state="restrict:rest",
        ),
        (
            "_farooq: _detect_population_in(sleep), the smoothed trace missing outside "
            "sleep bins; pyramidal, 15 ms, 2 SD with bounds at 2 SD, 100-800 ms, >= 5 "
            f"pyramidal cells. {_REST_SLEEP}"
        ),
    ),
    "chenani_2019": (
        _spikes(
            units="place",
            smoothing_sigma=0.030,
            threshold=3.0,
            bound_fraction=1 / 3,
            minimum_active_units=5,
            state="inside:rest",
        ),
        (
            "chenani_2019: place cells, 30 ms, 3 SD with bounds at 1 SD, >= 5 place cells; "
            "run_method keeps events inside behavior_intervals, the policy's rest"
        ),
    ),
    "muessig_2019": (
        _spikes(
            units="pyramidal",
            smoothing_sigma=0.010,
            threshold=3.0,
            bound_fraction=1.0,
            minimum_event_duration=0.1,
            maximum_duration=0.75,
            state="inside:rest",
            coincidence="overlap:muessig_2019_ripples",
        ),
        (
            "muessig_2019 (trial rest, no sample speed veto: speed rule off): pyramidal, "
            "10 ms, 3 SD with bounds at 3 SD, 100-750 ms, overlapping muessig_2019_ripples "
            f"(its configured public call), inside rec.sleep. {_REST_SLEEP}"
        ),
    ),
    "drieu_2018": (
        _spikes(
            units="place",
            smoothing_sigma=0.010,
            threshold=3.0,
            maximum_duration=0.5,
            state="restrict:rest",
        ),
        (
            "_drieu_events with sleep_intervals supplied: _detect_population_in(sleep), "
            f"place cells, 10 ms, 3 SD, at most 500 ms; detection stage. {_REST_SLEEP}"
        ),
    ),
    "olafsdottir_2017": (
        _spikes(
            units="place",
            smoothing_sigma=0.005,
            threshold=3.0,
            minimum_event_duration=0.04,
            speed="all<=3",
            state="inside:rest",
        ),
        (
            "olafsdottir_2017 (analysis arm): place cells, 5 ms, 3 SD, >= 40 ms, every "
            "event speed <= 3 (speed_rule all); run_method keeps events inside "
            "behavior_intervals, the policy's rest"
        ),
    ),
    "grosmark_2016": (
        _spikes(
            units="pyramidal",
            smoothing_sigma=0.015,
            normalization_period="rest",
            threshold=3.0,
            minimum_event_duration=0.05,
            maximum_duration=0.5,
            minimum_active_units=5,
            state="inside:rest",
            coincidence="peak_inside:external_ripples",
        ),
        ("grosmark_2016 (detection stage) is _population_with_ripple_peak, as yang_2024"),
    ),
    "silva_2015": (
        _spikes(
            units="pyramidal",
            smoothing_sigma=0.010,
            threshold=3.0,
            minimum_event_duration=0.1,
            maximum_duration=0.5,
            speed="restrict<5",
        ),
        (
            "silva_2015: pyramidal, 10 ms, 3 SD, 100-500 ms, detected only while speed < 5 "
            "(speed_rule restrict)"
        ),
    ),
    "bendor_2012": (
        _spikes(
            units="all",
            smoothing_sigma=0.015,
            threshold=4.0,
            bound_fraction=0.5,
            minimum_event_duration=0.05,
            merge_gap=0.05,
        ),
        (
            "bendor_2012: every unit, 15 ms, 4 SD with bounds at 2 SD, merged < 50 ms "
            "apart, >= 50 ms"
        ),
    ),
    "davidson_2009": (
        _spikes(
            units="all",
            smoothing_sigma=0.015,
            normalization_period="speed<5",
            threshold=3.0,
            speed="endpoints<5",
            state="overlap:running_30s",
        ),
        (
            "davidson_2009: every unit, 15 ms, statistics over speed < 5, 3 SD, speed < 5 "
            "at both ends, overlapping running (> 15 cm/s) widened by 30 s"
        ),
    ),
    "widloski_2025_bursts": (
        _spikes(
            units="all",
            smoothing_sigma=0.08,
            normalization_period="speed<5",
            threshold=3.0,
            minimum_event_duration=0.05,
        ),
        (
            "widloski_2025_bursts: every unit, 80 ms, statistics over speed < 5, 3 SD, "
            ">= 50 ms, no speed rule"
        ),
    ),
    "krause_2022_hse": (
        _spikes(units="all", smoothing_sigma=0.02, threshold=3.0, speed="all<=5"),
        (
            "krause_2022_hse (interpretation text, no baseline_intervals: statistics over "
            "the whole recording): every unit, 20 ms, 3 SD, every event speed <= 5"
        ),
    ),
    "gillespie_2021_mua": (
        _spikes(
            units="all",
            smoothing_sigma=0.015,
            normalization_period="speed<4",
            threshold=3.0,
            speed="endpoints<4",
        ),
        (
            "gillespie_2021_mua: every unit, 15 ms, statistics over speed < 4, 3 SD, speed "
            "< 4 at both ends"
        ),
    ),
    "pfeiffer_2015": (
        _lfp(
            smoothing_sigma=0.0125,
            normalization_period="speed<5",
            threshold=3.0,
            minimum_event_duration=0.05,
            maximum_duration=2.0,
            speed="endpoints<=5",
        ),
        (
            "_pfeiffer_2015_swrs: mean 150-250 Hz envelope of every channel, 12.5 ms, "
            "statistics over speed < 5, 3 SD, 50 ms-2 s, speed <= 5 at both ends"
        ),
    ),
    "berners_lee_2021": (
        _lfp(
            channels=3,
            smoothing_sigma=0.0125,
            normalization_period="speed<5",
            threshold=2.0,
            minimum_event_duration=0.05,
            maximum_duration=2.0,
            speed="endpoints<=5",
        ),
        (
            "_pfeiffer_2015_swrs(threshold=2.0, channels=3): as pfeiffer_2015 at 2 SD on "
            "the first three channels"
        ),
    ),
    "ambrose_2016": (
        _lfp(smoothing_sigma=0.0125, threshold=3.0, speed="restrict<5"),
        (
            "ambrose_2016: mean 150-250 Hz envelope of every channel, 12.5 ms, 3 SD, "
            "detected only while speed < 5 (speed_rule restrict, so the statistics are "
            "the slow samples')"
        ),
    ),
    "pfeiffer_2013_ripples": (
        _lfp(
            smoothing_sigma=0.0125,
            normalization_period="speed<5",
            threshold=3.0,
            speed="restrict<5",
        ),
        (
            "pfeiffer_2013_ripples: mean 150-250 Hz envelope of every channel, 12.5 ms, "
            "statistics over speed < 5, 3 SD, detected only while speed < 5"
        ),
    ),
}

_KARLSSON = (
    "Karlsson_ripple_detector: each channel detected separately and overlapping events "
    "combined, a whole detector rather than a template core"
)
_SILENCE = (
    "silence-bounded population windows (detect_silence_bounded_events), not a "
    "thresholded trace"
)
_TIROLE = (
    "Tirole's finite 41-point kernel applied forward and backward, threshold anchors "
    "with fallback bounds, a speed median sampled every 10 ms and a resampled ripple "
    "gate"
)
_MICHON = "5 ms bins, 3 s median detrending and a ripple partner on a detrended envelope"
_TEN_MS = "a population trace on 10 ms bins; the template's grid is 1 ms"
_FRACTION = (
    "an active-fraction rule (a share of the selected cells), which the template's "
    "cell count does not express"
)
_PARTICIPATION_UNITS = (
    "participation counted over the place cells while the trace pools pyramidal "
    "cells; the template counts its trace's own units"
)
_PEAKS = "local peaks each with its own window, not a thresholded interval"
_RECTIFIED = "the rectified filtered LFP with raw thresholds, not an envelope's z-score"

# Every configuration without a template, and why.
FIXED_POINTS: dict[str, str] = {
    "mallory_2025": "peaks bounded by mean crossings merged by retained peak (custom "
    "peak merging)",
    "widloski_2025": "a minimum time above threshold (15 ms); the template needs one sample",
    "huelin_gorriz_2023": _TIROLE,
    "harvey_2023_code": "Long_sharp_wave_ripple_detector (k-means on sharp-wave and "
    "ripple power) then a pyramidal spiking veto near the peak",
    "harvey_2023_text": "difference-of-Gaussians band with statistics from a clipped "
    "trace, and a radiatum sharp wave detected on another trace",
    "tirole_2022": _TIROLE,
    "bush_2022": "a rate on the input samples rather than 1 ms bins, an inclusive "
    "40 ms merge after detection, an active-fraction rule and a median speed rule",
    "berners_lee_2022": "a finite 100-point kernel with zero-padded convolution, "
    "statistics over stopped bins and bounds just above the mean",
    "krause_2022": "SWRs trimmed on per-event 3 ms bins",
    "mou_2022": "10 ms bins and min-max scaling of the population trace",
    "denovellis_2021": "the historical 101-tap filter on squared, summed channels, "
    "rooted after smoothing, and a 15 ms minimum above threshold",
    "gillespie_2021": "Kay_ripple_detector: the root of summed squared envelopes, a "
    "whole detector rather than a template core",
    "michon_2021": _MICHON,
    "gridchyn_2020": "adaptive feedback threshold on causal 20 ms counts",
    "kaefer_2020": "240 ms FFT windows every 20 ms",
    "bhattarai_2020": _SILENCE,
    "stella_2019": "Morlet wavelet RMS per electrode, the maximum over electrodes",
    "xu_2019": "onsets trimmed to the first spike, with active-fraction and spike-count rules",
    "farooq_2019_science": _PARTICIPATION_UNITS,
    "michon_2019": _MICHON,
    "liu_2019": _SILENCE,
    "shin_2019": _KARLSSON,
    "carey_2019": "Carey_candidate_detector: a joint spectral-ripple and multiunit score",
    "maboudi_2018": "a finite 121-point Gaussian kernel with zero padding and a mean "
    "speed rule",
    "olafsdottir_2017.trajectory": _FRACTION,
    "wu_2017": _TEN_MS,
    "yamamoto_2017": f"{_TEN_MS}, with a ripple partner of a squared single-channel envelope",
    "tang_2017": _KARLSSON,
    "jadhav_2016": f"{_KARLSSON}; SWRs within 1 s of the previous start dropped",
    "olafsdottir_2016": _FRACTION,
    "olafsdottir_2015": _SILENCE,
    "olafsdottir_2015.bayesian_candidates": _SILENCE,
    "wu_2014": _TEN_MS,
    "wikenheiser_2013": "150 ms windows around every sample above 1 SD, joined",
    "pfeiffer_2013": "bounds trimmed to 20 ms spike windows, an active-fraction rule "
    "and duration limits after trimming",
    "carr_2012": _KARLSSON,
    "gupta_2010": "a log-transformed envelope; the template thresholds the envelope "
    "or its square",
    "karlsson_2009": _KARLSSON,
    "diba_2007": _SILENCE,
    "ji_2007": "10 ms bins of counts thresholded at a histogram minimum",
    "foster_2006": _SILENCE,
    "lee_2002": f"{_SILENCE}, with each cell's bursts collapsed",
    "nadasdy_1999": "boxcar RMS of the filtered channels, summed",
    "kudrimoti_1999": "a minimum time above threshold (25 ms); the template needs one sample",
    "harvey_2023_no_radiatum": "Zugaro_ripple_detector then a pyramidal spiking veto "
    "near the peak",
    "mallory_2025_ripples": "peaks bounded by mean crossings merged by retained peak "
    "(custom peak merging)",
    "igata_2021_ripples": "one inventory per channel",
    "wu_2014_ripples": _PEAKS,
    "davidson_2009_ripples": _PEAKS,
    "ji_2007_ripples": f"{_RECTIFIED}, merged before a peak is required",
    "lee_2002_ripples": f"{_RECTIFIED}, crossings joined",
    "foster_2006_ripples": f"{_RECTIFIED}, crossings joined",
    "denovellis_2021_mua": "2 ms bins and a 15 ms minimum above threshold",
    "maboudi_2018_open_field": "pfeiffer_2013's rule: bounds trimmed to 20 ms spike "
    "windows and an active-fraction rule",
    "muessig_2019_ripples": _PEAKS,
    "bhattarai_2020_ripples": "a 50 ms boxcar of the squared filtered signal, merged "
    "after a duration rule",
    "farooq_2019_science_awake": _PARTICIPATION_UNITS,
    "liu_2019_awake": _SILENCE,
}


def _config(config_id: str) -> RecipeConfig:
    for config in RECIPES:
        if config.config_id == config_id:
            return config
    msg = f"No configuration {config_id!r} in recipe_configs.RECIPES."
    raise KeyError(msg)


def template_of(config: RecipeConfig) -> Template:
    """The template written for a configuration.

    Parameters
    ----------
    config : RecipeConfig

    Returns
    -------
    template : SpikeTemplate or LfpTemplate

    Raises
    ------
    ValueError
        The configuration is a fixed point; the message gives the reason.
    KeyError
        Neither a template nor a fixed point names the configuration.
    """
    if config.config_id in TEMPLATES:
        return TEMPLATES[config.config_id][0]
    if config.config_id in FIXED_POINTS:
        msg = f"{config.config_id} has no template: {FIXED_POINTS[config.config_id]}."
        raise ValueError(msg)
    msg = f"{config.config_id} is neither in TEMPLATES nor in FIXED_POINTS."
    raise KeyError(msg)


def recipe_events(config: RecipeConfig, session: rd.SimulatedSession) -> FloatArray:
    """A configuration's public-call events on a session, as the runner makes them.

    Parameters
    ----------
    config : RecipeConfig
    session : SimulatedSession

    Returns
    -------
    events : ndarray, shape (n_events, 2)
        ``bounds`` of ``run_recipe`` on ``make_recording`` of the session's
        integer counts, with the policy's ``behavior_intervals``.
    """
    recording = make_recording(_counted(session), config)
    return bounds(run_recipe(config, recording, behavior_intervals(session, config)))


@dataclasses.dataclass(frozen=True)
class Verification:
    """Whether a configuration's template gives its public call's events.

    Attributes
    ----------
    config_id : str
    family : str
        The template's family, or ``""`` for a fixed point.
    in_space : bool
    reason : str
        Why not, empty when in the space.
    n_events : int
        Events of the public call over the sessions checked, up to the first
        on which the two differ.
    """

    config_id: str
    family: str
    in_space: bool
    reason: str
    n_events: int


def verify_all(
    recipes: Sequence[RecipeConfig], contexts: Iterable[SessionContext]
) -> pd.DataFrame:
    """Compare each configuration's template events with its public call's.

    Parameters
    ----------
    recipes : sequence of RecipeConfig
    contexts : iterable of SessionContext
        Every session to check on, taken one at a time and released after.

    Returns
    -------
    table : pandas.DataFrame
        One row per configuration, in order, with ``Verification``'s fields.
        A configuration is in the space when it has a template, the two give
        identical bounds, in the same order, on every session, and the public
        call found an event on some session (equal empty results verify
        nothing).
    """
    found = {
        config.config_id: Verification(
            config.config_id,
            "" if config.config_id in FIXED_POINTS else family_of(template_of(config)),
            config.config_id not in FIXED_POINTS,
            FIXED_POINTS.get(config.config_id, ""),
            0,
        )
        for config in recipes
    }
    for context in contexts:
        for config in recipes:
            verification = found[config.config_id]
            if not verification.in_space:
                continue
            expected = recipe_events(config, context.session)
            events = context.events(compile(template_of(config)))
            n_events = verification.n_events + len(expected)
            if expected.shape != events.shape or not np.array_equal(expected, events):
                reason = (
                    f"events differ on {context.label}: {len(events)} from the template, "
                    f"{len(expected)} from the public call"
                )
                found[config.config_id] = dataclasses.replace(
                    verification, in_space=False, reason=reason, n_events=n_events
                )
            else:
                found[config.config_id] = dataclasses.replace(verification, n_events=n_events)
        context.release()
    for config_id, verification in found.items():
        if verification.in_space and verification.n_events == 0:
            found[config_id] = dataclasses.replace(
                verification,
                in_space=False,
                reason="no event on the sessions checked, so equality verifies nothing",
            )
    return pd.DataFrame(
        [dataclasses.asdict(found[config.config_id]) for config in recipes],
        columns=[field.name for field in dataclasses.fields(Verification)],
    )


def in_space(config: RecipeConfig, contexts: Iterable[SessionContext]) -> bool:
    """Whether a configuration's template stands for it on these sessions.

    Parameters
    ----------
    config : RecipeConfig
    contexts : iterable of SessionContext
        The positive controls, edge cases and reference sessions to check on.

    Returns
    -------
    in_space : bool
        As ``verify_all`` decides it.
    """
    return bool(verify_all([config], contexts)["in_space"].iloc[0])


# Reference sessions


def reference_parameters(run_directory: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    """The reference condition's parameters as the run saved them.

    Parameters
    ----------
    run_directory : str or path-like

    Returns
    -------
    parameters : dict of str to dict
        ``parameters_from_json`` of ``conditions.csv``'s reference ``params``:
        the resolved values, after overrides such as a shorter ``duration_s``.

    Raises
    ------
    ValueError
        The run has no reference row, or its parameters are incomplete.
    """
    table = read_table(Path(run_directory) / "conditions.csv")
    rows = table[table["condition_id"] == REFERENCE_CONDITION]
    if len(rows) != 1:
        msg = f"{run_directory} has {len(rows)} reference rows in conditions.csv, not 1."
        raise ValueError(msg)
    return parameters_from_json(str(rows["params"].iloc[0]))


def check_report(
    run_directory: str | os.PathLike[str], parameters: Mapping[str, Mapping[str, Any]]
) -> None:
    """Check the run's simulator validation report, as the runner did before running.

    ``require_ready_report`` on the report ``run_spec.json`` names (a path
    from the repository root), for the reference parameters, and its hash and
    fingerprints must be those the run recorded.

    Parameters
    ----------
    run_directory : str or path-like
    parameters : mapping
        The reference condition's saved parameters.

    Raises
    ------
    ValueError
        The report is not ready for these parameters, or its hash, simulation
        fingerprint or target-table hash differs from the run's.
    """
    saved = json.loads((Path(run_directory) / "run_spec.json").read_text())
    recorded = saved["validation_report"]
    report = require_ready_report(
        REPOSITORY / recorded["path"], {REFERENCE_CONDITION: parameters}
    )
    differing = [
        key for key in _REPORT_IDENTITY if key != "path" and report[key] != recorded[key]
    ]
    if differing:
        msg = (
            f"The validation report differs from the one {run_directory} ran with at: "
            f"{', '.join(differing)}."
        )
        raise ValueError(msg)


def reference_session(
    run_directory: str | os.PathLike[str],
    replicate: int,
    parameters: Mapping[str, Mapping[str, Any]] | None = None,
) -> rd.SimulatedSession:
    """Replicate ``replicate`` of the run's reference condition, simulated again.

    Parameters
    ----------
    run_directory : str or path-like
    replicate : int
    parameters : mapping, optional
        The saved reference parameters; default ``reference_parameters``.

    Returns
    -------
    session : SimulatedSession
        ``simulate_parameters(parameters, replicate)``, checked against what
        the run saved for it: the seed and duration in ``sessions.csv.gz``, the
        latent event and non-event tables in ``truth.csv.gz`` and the ripple
        channels in ``ripple_channels.csv.gz``, value for value.

    Raises
    ------
    ValueError
        The run holds no such session, or any of those differ: the sessions
        would not be the run's.
    """
    directory = Path(run_directory) / "conditions" / REFERENCE_CONDITION
    if parameters is None:
        parameters = reference_parameters(run_directory)
    session_id = f"{REFERENCE_CONDITION}/{replicate}"
    rows = read_table(directory / "sessions.csv.gz")
    row = rows[rows["session_id"] == session_id]
    if len(row) != 1:
        msg = f"{directory} holds no session {session_id}."
        raise ValueError(msg)
    session = simulate_parameters(parameters, replicate)
    duration = len(session.time) / session.sampling_frequency
    problems = []
    if int(row["seed"].iloc[0]) != session_seed(replicate):
        problems.append(f"seed {row['seed'].iloc[0]}, not {session_seed(replicate)}")
    if float(row["duration_s"].iloc[0]) != duration:
        problems.append(f"duration {row['duration_s'].iloc[0]} s, not {duration} s")
    truth = load_truth(directory / "truth.csv.gz").get(session_id)
    if truth is None:
        problems.append("no truth rows")
    else:
        for saved, table in zip(truth, (session.events, session.non_events), strict=True):
            if not _same_frame(saved, table[list(saved.columns)].astype(saved.dtypes)):
                problems.append(f"its {'non-' if table is session.non_events else ''}events")
    channels = read_table(
        directory / "ripple_channels.csv.gz",
        keep=lambda frame: frame["session_id"] == session_id,
    )
    columns = list(RIPPLE_CHANNEL_COLUMNS[1:])
    simulated = session.ripple_channels[columns]
    if not _same_frame(channels[columns].astype(simulated.dtypes), simulated):
        problems.append("its ripple channels")
    if problems:
        msg = (
            f"The simulated {session_id} is not the run's: {'; '.join(problems)} differ "
            f"from {directory}."
        )
        raise ValueError(msg)
    return session


def _same_frame(first: pd.DataFrame, second: pd.DataFrame) -> bool:
    """Equal columns and values, NaN equal to NaN, the index aside."""
    first, second = first.reset_index(drop=True), second.reset_index(drop=True)
    return list(first.columns) == list(second.columns) and bool(first.equals(second))


def reference_contexts(
    run_directory: str | os.PathLike[str], replicates: Iterable[int] = range(K)
) -> Iterable[SessionContext]:
    """A context per reference session, built one at a time.

    Parameters
    ----------
    run_directory : str or path-like
    replicates : iterable of int, optional
        Default ``0`` to ``K - 1``.

    Yields
    ------
    context : SessionContext
        Of ``reference_session``, labelled ``"reference/<replicate>"``.
    """
    parameters = reference_parameters(run_directory)
    for replicate in replicates:
        session = reference_session(run_directory, replicate, parameters)
        context = SessionContext(session, f"{REFERENCE_CONDITION}/{replicate}")
        del session  # the context keeps an integer copy of the counts
        yield context


def edge_sessions(
    parameters: Mapping[str, Mapping[str, Any]], duration: float = EDGE_DURATION
) -> dict[str, rd.SimulatedSession]:
    """Short sessions for the edge cases of ``verify``.

    Parameters
    ----------
    parameters : mapping
        Simulation parameters, such as the run's reference ones; the session
        is ``duration`` seconds of replicate 0 of them.
    duration : float, optional
        Seconds.

    Returns
    -------
    sessions : dict of str to SimulatedSession
        ``"gap"``: every LFP channel, the radiatum and the spike counts missing
        from ``GAP[0]`` up to ``GAP[1]`` s (NaN), which splits every block;
        ``"unix_origin"``: its timestamps and running bouts moved to start at
        ``UNIX_ORIGIN``.
    """
    changed = {section: dict(values) for section, values in parameters.items()}
    changed["session"]["duration_s"] = duration
    session = simulate_parameters(changed, 0)
    missing = (session.time >= GAP[0]) & (session.time < GAP[1])
    lfps = session.lfps.copy()
    lfps[missing] = np.nan
    sharp_wave = session.sharp_wave_lfp.copy()
    sharp_wave[missing] = np.nan
    counts = session.multiunit.astype(float)
    counts[missing] = np.nan
    gap = dataclasses.replace(session, lfps=lfps, sharp_wave_lfp=sharp_wave, multiunit=counts)
    moved = dataclasses.replace(
        session,
        time=session.time + UNIX_ORIGIN,
        running_intervals=session.running_intervals + UNIX_ORIGIN,
    )
    return {"gap": gap, "unix_origin": moved}


# Factor space


@dataclasses.dataclass(frozen=True)
class Factor:
    """One factor of a family's space.

    Attributes
    ----------
    name : str
        A template field.
    kind : {"continuous", "integer", "categorical"}
    levels : tuple
        Continuous: ``(low, high)``. Integer: every whole number from the
        least to the largest value. Categorical: the distinct values, in
        configuration order.
    """

    name: str
    kind: str
    levels: tuple[Any, ...]

    def value(self, u: float) -> Any:
        """The factor's value at a uniform ``u`` in [0, 1).

        Parameters
        ----------
        u : float

        Returns
        -------
        value
            ``low + u (high - low)`` for a continuous factor, else
            ``levels[min(int(u * n), n - 1)]`` of its ``n`` levels.
        """
        if self.kind == "continuous":
            low, high = self.levels
            return float(low + u * (high - low))
        n = len(self.levels)
        return self.levels[min(int(u * n), n - 1)]

    def points(self) -> tuple[Any, ...]:
        """The values one factor at a time visits.

        Returns
        -------
        values : tuple
            ``OAT_POINTS`` evenly spaced values over a continuous range; every
            level otherwise.
        """
        if self.kind == "continuous":
            low, high = self.levels
            return tuple(float(v) for v in np.linspace(float(low), float(high), OAT_POINTS))
        return self.levels


def in_space_ids(family: str) -> tuple[str, ...]:
    """The configurations of a family with a template written, in configuration order.

    Written, not verified: ``verify_all`` decides which of them stand for
    their methods on a run's sessions.

    Parameters
    ----------
    family : {"spikes", "lfp"}

    Returns
    -------
    config_ids : tuple of str
    """
    return tuple(
        config.config_id
        for config in RECIPES
        if config.config_id in TEMPLATES
        and family_of(TEMPLATES[config.config_id][0]) == family
    )


def distinct_ids(family: str) -> tuple[str, ...]:
    """``in_space_ids`` with each template once: of identical ones, the first.

    Two methods with one template (``grosmark_2016`` is ``yang_2024``'s rule)
    are one point of the space: they count once for the stop rule, the
    reference configuration and the Shapley pairs, though both are verified
    and listed.

    Parameters
    ----------
    family : {"spikes", "lfp"}

    Returns
    -------
    config_ids : tuple of str
    """
    first: dict[Template, str] = {}
    for config_id in in_space_ids(family):
        first.setdefault(TEMPLATES[config_id][0], config_id)
    return tuple(first.values())


def family_templates(recipes: Sequence[RecipeConfig], family: str) -> list[Template]:
    """The distinct templates of ``recipes`` in a family, in their order.

    Parameters
    ----------
    recipes : sequence of RecipeConfig
    family : {"spikes", "lfp"}

    Returns
    -------
    templates : list
        Each once: of identical templates, the first.
    """
    if family not in FAMILIES:
        msg = f"family must be one of {FAMILIES}; got {family!r}."
        raise ValueError(msg)
    found = [TEMPLATES[c.config_id][0] for c in recipes if c.config_id in TEMPLATES]
    return list(dict.fromkeys(template for template in found if family_of(template) == family))


def factor_space(recipes: Sequence[RecipeConfig], family: str) -> tuple[Factor, ...]:
    """The factors a family's represented configurations vary, with their ranges.

    Parameters
    ----------
    recipes : sequence of RecipeConfig
        Usually ``RECIPES``; those with a template in ``family`` count.
    family : {"spikes", "lfp"}

    Returns
    -------
    factors : tuple of Factor
        In the template's field order; a field with a single value among the
        templates is left out, since no configuration varies it.
    """
    templates = family_templates(recipes, family)
    if not templates:
        return ()
    factors = []
    for field in dataclasses.fields(templates[0]):
        values = [getattr(template, field.name) for template in templates]
        distinct = tuple(dict.fromkeys(values))
        if len(distinct) < 2:
            continue
        kind = FACTOR_KINDS[field.name]
        if kind == "continuous":
            levels: tuple[Any, ...] = (float(min(values)), float(max(values)))
        elif kind == "integer":
            levels = tuple(range(min(values), max(values) + 1))
        else:
            levels = distinct
        factors.append(Factor(field.name, kind, levels))
    return tuple(factors)


def reference_template(family: str, templates: Sequence[TemplateT] | None = None) -> TemplateT:
    """The family's reference configuration: each factor's median or mode.

    Parameters
    ----------
    family : {"spikes", "lfp"}
    templates : sequence of templates, optional
        Default the family's templates of ``RECIPES``, in configuration order.
        Identical templates count once.

    Returns
    -------
    template
        A continuous factor at the median, an integer one at the median rounded
        down, a categorical one at its most frequent value, ties going to the
        value that comes first.
    """
    chosen = (
        list(dict.fromkeys(templates))
        if templates is not None
        else family_templates(RECIPES, family)
    )
    if not chosen:
        msg = f"The {family} family has no template to take a reference from."
        raise ValueError(msg)
    values: dict[str, Any] = {}
    for field in dataclasses.fields(chosen[0]):
        column = [getattr(template, field.name) for template in chosen]
        kind = FACTOR_KINDS[field.name]
        if kind == "continuous":
            values[field.name] = float(np.median(column))
        elif kind == "integer":
            values[field.name] = int(np.floor(np.median(column)))
        else:
            counts: dict[Any, int] = {}
            for value in column:
                counts[value] = counts.get(value, 0) + 1
            values[field.name] = max(counts, key=lambda value: counts[value])
    return type(chosen[0])(**values)  # type: ignore[return-value]


def with_factors(template: TemplateT, values: Mapping[str, Any]) -> TemplateT:
    """``template`` with some factors set.

    Parameters
    ----------
    template : SpikeTemplate or LfpTemplate
    values : mapping of str to object

    Returns
    -------
    template
    """
    return dataclasses.replace(template, **dict(values))


# Outputs


def _mean(values: Sequence[float]) -> float:
    """The mean of the finite values, NaN when there are none."""
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def jaccard(first: FloatArray, second: FloatArray) -> float:
    """Matched events over the union of two inventories.

    Parameters
    ----------
    first, second : ndarray, shape (n_events, 2)

    Returns
    -------
    jaccard : float
        ``n_matched / (n_first + n_second - n_matched)`` after ``match_events``
        at any overlap; 1.0 when both are empty (they agree).
    """
    if len(first) + len(second) == 0:
        return 1.0
    matched = len(rd.match_events(first, second).pairs)
    return matched / (len(first) + len(second) - matched)


def session_outputs(
    events: FloatArray, reference_events: FloatArray, context: SessionContext, family: str
) -> dict[str, float]:
    """The ``Y``s of one configuration's events on one session.

    Parameters
    ----------
    events, reference_events : ndarray, shape (n_events, 2)
        The configuration's events and the reference configuration's.
    context : SessionContext
    family : {"spikes", "lfp"}

    Returns
    -------
    outputs : dict of str to float
        By ``Y_NAMES``; ``onset_error_25`` NaN without a matched pair.
    """
    truth = context.windows[FAMILY_EXPRESSION[family]]
    matching = rd.match_events(truth[0.1], events)
    onset = matching.boundary_errors(truth[0.25])["onset_error"].to_numpy(dtype=float)
    return {
        "f1": float(matching.f1),
        "f1_network": float(rd.match_events(context.windows["network"][0.1], events).f1),
        "events_per_minute": len(events) / context.minutes,
        "onset_error_25": float(np.median(onset)) if len(onset) else float("nan"),
        "jaccard_reference": jaccard(events, reference_events),
    }


def evaluate_session(
    pipeline: Pipeline, context: SessionContext, reference: Pipeline
) -> dict[str, float]:
    """The ``Y``s of a pipeline on one session, its events and the reference's
    memoized by the context.

    Parameters
    ----------
    pipeline, reference : Pipeline
    context : SessionContext

    Returns
    -------
    outputs : dict of str to float
    """
    return session_outputs(
        context.events(pipeline), context.events(reference), context, pipeline_family(pipeline)
    )


def evaluate_config(
    pipeline: Pipeline, contexts: Sequence[SessionContext], reference: Pipeline
) -> dict[str, float]:
    """The ``Y``s of a pipeline, each averaged over the sessions.

    A pipeline's events on a session are computed once per context and kept
    (memoized by the pipeline and the context's replicate), so a configuration
    repeated in one analysis runs no detection again.

    Parameters
    ----------
    pipeline : Pipeline
    contexts : sequence of SessionContext
        The ``K`` reference sessions.
    reference : Pipeline
        What ``jaccard_reference`` compares with.

    Returns
    -------
    outputs : dict of str to float
        By ``Y_NAMES``: the mean over the sessions, ``onset_error_25`` over
        those with a matched pair.
    """
    per_session = [evaluate_session(pipeline, context, reference) for context in contexts]
    return {name: _mean([outputs[name] for outputs in per_session]) for name in Y_NAMES}


# Evaluating many configurations, one session at a time

_WORKER_CONTEXT: dict[tuple[str, int], SessionContext] = {}


def _worker_context(run_directory: str, replicate: int) -> SessionContext:
    """This process's context of a reference session, the only one it holds."""
    key = (run_directory, replicate)
    if key not in _WORKER_CONTEXT:
        for context in _WORKER_CONTEXT.values():
            context.release()
        _WORKER_CONTEXT.clear()
        _WORKER_CONTEXT[key] = next(iter(reference_contexts(run_directory, [replicate])))
    return _WORKER_CONTEXT[key]


def _evaluate_chunk(
    run_directory: str,
    replicate: int,
    pipelines: Sequence[Pipeline],
    references: Sequence[Pipeline],
) -> list[dict[str, float]]:
    context = _worker_context(run_directory, replicate)
    return [
        evaluate_session(pipeline, context, reference)
        for pipeline, reference in zip(pipelines, references, strict=True)
    ]


def evaluate_many(
    pipelines: Sequence[Pipeline],
    references: Sequence[Pipeline],
    run_directory: str | os.PathLike[str],
    *,
    workers: int = 1,
    chunk_size: int = 64,
) -> list[dict[str, list[float]]]:
    """Every pipeline's ``Y``s on each reference session.

    Parameters
    ----------
    pipelines, references : sequence of Pipeline
        Each pipeline with what its ``jaccard_reference`` compares with.
    run_directory : str or path-like
    workers : int, optional
        Processes (``ProcessPoolExecutor``); each holds one session at a time,
        taking a chunk of configurations of one session.
    chunk_size : int, optional
        Configurations per task.

    Returns
    -------
    outputs : list of dict of str to list of float
        Per pipeline, by ``Y_NAMES``, the value on each of the ``K`` sessions.
    """
    directory = str(run_directory)
    tasks = [
        (replicate, start)
        for replicate in range(K)
        for start in range(0, len(pipelines), chunk_size)
    ]
    found: dict[tuple[int, int], list[dict[str, float]]] = {}

    def arguments(task: tuple[int, int]) -> tuple[Any, ...]:
        replicate, start = task
        stop = start + chunk_size
        return directory, replicate, pipelines[start:stop], references[start:stop]

    if workers == 1:
        for task in tasks:
            found[task] = _evaluate_chunk(*arguments(task))
        _release_worker_context()
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {task: pool.submit(_evaluate_chunk, *arguments(task)) for task in tasks}
            found = {task: future.result() for task, future in futures.items()}
    outputs: list[dict[str, list[float]]] = [{name: [] for name in Y_NAMES} for _ in pipelines]
    for (_, start), rows in sorted(found.items()):
        for offset, row in enumerate(rows):
            for name in Y_NAMES:
                outputs[start + offset][name].append(row[name])
    return outputs


def _release_worker_context() -> None:
    for context in _WORKER_CONTEXT.values():
        context.release()
    _WORKER_CONTEXT.clear()


# Sobol indices and Shapley values


def sobol_indices(
    y_a: FloatArray, y_b: FloatArray, y_ab: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """First-order (Saltelli et al. 2010) and total (Jansen 1999) indices.

    Parameters
    ----------
    y_a, y_b : ndarray, shape (n,)
        The output at the rows of the two sample matrices.
    y_ab : ndarray, shape (d, n)
        Row ``i``: the output at ``A`` with column ``i`` taken from ``B``.

    Returns
    -------
    first, total : ndarray, shape (d,)
    """
    variance = np.var(np.concatenate([y_a, y_b]), ddof=1)
    first = np.mean(y_b * (y_ab - y_a), axis=1) / variance
    total = 0.5 * np.mean((y_a - y_ab) ** 2, axis=1) / variance
    return first, total


def sobol_intervals(
    y_a: FloatArray,
    y_b: FloatArray,
    y_ab: FloatArray,
    *,
    n_resamples: int = N_SOBOL_RESAMPLES,
    seed: int = 0,
    level: float = 0.95,
) -> pd.DataFrame:
    """``sobol_indices`` with percentile intervals from resampling the rows.

    Parameters
    ----------
    y_a, y_b : ndarray, shape (n,)
    y_ab : ndarray, shape (d, n)
    n_resamples : int, optional
    seed : int, optional
    level : float, optional

    Returns
    -------
    table : pandas.DataFrame
        One row per factor position: ``first``, ``first_low``,
        ``first_high``, ``total``, ``total_low``, ``total_high``.
    """
    rng = np.random.default_rng(seed)
    draws_first, draws_total = [], []
    # an output that does not vary, or is missing, has no index: NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        first, total = sobol_indices(y_a, y_b, y_ab)
        for _ in range(n_resamples):
            rows = rng.integers(len(y_a), size=len(y_a))
            drawn_first, drawn_total = sobol_indices(y_a[rows], y_b[rows], y_ab[:, rows])
            draws_first.append(drawn_first)
            draws_total.append(drawn_total)
    alpha = (1 - level) / 2
    first_bounds = _percentiles(np.array(draws_first), alpha)
    total_bounds = _percentiles(np.array(draws_total), alpha)
    return pd.DataFrame(
        {
            "first": first,
            "first_low": first_bounds[0],
            "first_high": first_bounds[1],
            "total": total,
            "total_low": total_bounds[0],
            "total_high": total_bounds[1],
        }
    )


def _percentiles(draws: FloatArray, alpha: float) -> FloatArray:
    """The ``alpha`` and ``1 - alpha`` quantiles of each column's finite draws,
    shape (2, n_columns); NaN for a column without one."""
    bounds = np.full((2, draws.shape[1]), np.nan)
    for column in range(draws.shape[1]):
        finite = draws[:, column][np.isfinite(draws[:, column])]
        if len(finite):
            bounds[:, column] = np.quantile(finite, [alpha, 1 - alpha])
    return bounds


def shapley(
    value: Callable[[frozenset[Any]], float],
    factors: Iterable[Any],
    *,
    exact_up_to: int = SHAPLEY_EXACT_UP_TO,
    n_permutations: int = SHAPLEY_PERMUTATIONS,
    seed: int = 0,
) -> tuple[dict[Any, float], dict[Any, float]]:
    """Shapley values of ``value`` over ``factors``.

    Parameters
    ----------
    value : callable
        ``value(frozenset) -> float``, memoized by the caller.
    factors : iterable
    exact_up_to : int, optional
        Exact over subsets for this many factors or fewer (standard errors
        0); Monte Carlo over permutations beyond.
    n_permutations : int, optional
    seed : int, optional

    Returns
    -------
    phi, error : dict
        By factor: its Shapley value and its Monte Carlo standard error.
    """
    factors = tuple(factors)
    n = len(factors)
    if n <= exact_up_to:
        phi = dict.fromkeys(factors, 0.0)
        for i in factors:
            others = [f for f in factors if f != i]
            for k in range(n):
                weight = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                for subset in itertools.combinations(others, k):
                    s = frozenset(subset)
                    phi[i] += weight * (value(s | {i}) - value(s))
        return phi, dict.fromkeys(factors, 0.0)
    rng = np.random.default_rng(seed)
    contributions: dict[Any, list[float]] = {i: [] for i in factors}
    for _ in range(n_permutations):
        s = frozenset()
        for i in rng.permutation(factors):
            contributions[i].append(value(s | {i}) - value(s))
            s = s | {i}
    phi = {i: float(np.mean(c)) for i, c in contributions.items()}
    error = {
        i: float(np.std(c, ddof=1) / np.sqrt(n_permutations)) for i, c in contributions.items()
    }
    return phi, error


def shapley_subsets(
    factors: Sequence[Any],
    *,
    exact_up_to: int = SHAPLEY_EXACT_UP_TO,
    n_permutations: int = SHAPLEY_PERMUTATIONS,
    seed: int = 0,
) -> list[frozenset[Any]]:
    """Every subset ``shapley`` will ask the value of, without asking.

    Parameters
    ----------
    factors : sequence
    exact_up_to, n_permutations, seed
        As ``shapley``'s.

    Returns
    -------
    subsets : list of frozenset
        Distinct, in the order first needed: every subset for the exact
        path, else every prefix of the permutations ``shapley`` draws with
        the same seed.
    """
    factors = tuple(factors)
    if len(factors) <= exact_up_to:
        return [
            frozenset(subset)
            for k in range(len(factors) + 1)
            for subset in itertools.combinations(factors, k)
        ]
    rng = np.random.default_rng(seed)
    found: dict[frozenset[Any], None] = {frozenset(): None}
    for _ in range(n_permutations):
        s: frozenset[Any] = frozenset()
        for i in rng.permutation(factors):
            s = s | {i}
            found[s] = None
    return list(found)


# Analyses


def _key_columns(template: Template) -> dict[str, Any]:
    """A template's factors as output columns, tuples and None as text."""
    return {
        field.name: _as_text(getattr(template, field.name))
        for field in dataclasses.fields(template)
    }


def _as_text(value: Any) -> Any:
    if value is None or isinstance(value, tuple):
        return json.dumps(value)
    return value


def _rows(
    analysis: str,
    keys: Sequence[Mapping[str, Any]],
    templates: Sequence[Template],
    outputs: Sequence[Mapping[str, Sequence[float]]],
) -> pd.DataFrame:
    """One row per configuration and ``Y``: the keys, factors, mean and each
    session's value."""
    rows = []
    for key, template, found in zip(keys, templates, outputs, strict=True):
        for name in Y_NAMES:
            values = list(found[name])
            rows.append(
                {
                    "analysis": analysis,
                    **key,
                    **_key_columns(template),
                    "y": name,
                    "value": _mean(values),
                    **{f"replicate_{k}": value for k, value in enumerate(values)},
                }
            )
    return pd.DataFrame(rows)


def one_at_a_time(
    family: str, run_directory: str | os.PathLike[str], *, workers: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each factor at each of its values, every other at the reference.

    Parameters
    ----------
    family : {"spikes", "lfp"}
    run_directory : str or path-like
    workers : int, optional

    Returns
    -------
    rows : pandas.DataFrame
        ``_rows``' output: ``analysis``, ``factor``, ``level``, the factors,
        ``y``, ``value`` and each session's value; the reference's own rows
        have ``factor`` ``"reference"``.
    changes : pandas.DataFrame
        One row per factor, level and ``Y``: ``factor``, ``level``, ``y``,
        ``reference`` (its mean), ``value`` (the mean at the level) and
        ``change`` (value minus reference, sessions paired) with ``low`` and
        ``high``, a 95 % ``paired_bootstrap`` interval over the sessions.
    """
    reference = reference_template(family)
    configurations: list[tuple[dict[str, Any], Template]] = [
        ({"factor": "reference", "level": ""}, reference),
        *(
            (
                {"factor": factor.name, "level": _as_text(level)},
                with_factors(reference, {factor.name: level}),
            )
            for factor in factor_space(RECIPES, family)
            for level in factor.points()
        ),
    ]
    keys = [key for key, _ in configurations]
    templates = [template for _, template in configurations]
    base = compile(reference)
    outputs = evaluate_many(
        [compile(template) for template in templates],
        [base] * len(templates),
        run_directory,
        workers=workers,
    )
    rows = _rows("oat", keys, templates, outputs)
    changes = [
        {
            **key,
            "y": name,
            "reference": _mean(outputs[0][name]),
            "value": _mean(found[name]),
            **session_interval(np.subtract(found[name], outputs[0][name])),
        }
        for key, found in zip(keys[1:], outputs[1:], strict=True)
        for name in Y_NAMES
    ]
    return rows, pd.DataFrame(changes)


def session_interval(values: ArrayLike, level: float = 0.95) -> dict[str, float]:
    """The mean of per-session values with a paired bootstrap interval.

    ``paired_bootstrap`` over the sessions (``N_RESAMPLES`` draws, its seed),
    the statistic the mean of the finite values, computed draw for draw from
    ``resample_weights``, the counts of each session in each draw.

    Parameters
    ----------
    values : array_like, shape (n_sessions,)
        NaN where a session has no value.
    level : float, optional

    Returns
    -------
    interval : dict of str to float
        ``change`` (the mean of the finite values), ``low`` and ``high``;
        NaN where no draw holds a finite value.
    """
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    weights = resample_weights(len(values)) * finite
    totals = weights.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (weights @ np.where(finite, values, 0.0)) / totals
    low, high = _percentiles(draws[:, np.newaxis], (1 - level) / 2)[:, 0]
    return {"change": _mean(values.tolist()), "low": float(low), "high": float(high)}


def sobol_design(
    factors: Sequence[Factor], reference: TemplateT, n: int, seed: int = SOBOL_SEED
) -> tuple[list[TemplateT], list[TemplateT], list[list[TemplateT]]]:
    """The configurations of a Sobol analysis.

    Parameters
    ----------
    factors : sequence of Factor
        ``d`` factors.
    reference : template
        The values of the factors not varied.
    n : int
        Rows of each sample matrix.
    seed : int, optional

    Returns
    -------
    a, b : list of template
        The rows of ``A`` and ``B``, the two halves of one scrambled Sobol
        sample of ``2 d`` columns (``scipy.stats.qmc.Sobol``), mapped by
        ``Factor.value``.
    ab : list of list of template
        ``ab[i]``: the rows of ``A`` with column ``i`` from ``B``.
    """
    d = len(factors)
    sample = qmc.Sobol(d=2 * d, scramble=True, seed=seed).random(n)
    a_rows, b_rows = sample[:, :d], sample[:, d:]

    def configuration(row: FloatArray) -> TemplateT:
        return with_factors(
            reference,
            {
                factor.name: factor.value(float(u))
                for factor, u in zip(factors, row, strict=True)
            },
        )

    ab = []
    for i in range(d):
        mixed = a_rows.copy()
        mixed[:, i] = b_rows[:, i]
        ab.append([configuration(row) for row in mixed])
    return (
        [configuration(row) for row in a_rows],
        [configuration(row) for row in b_rows],
        ab,
    )


def sobol(
    family: str,
    run_directory: str | os.PathLike[str],
    *,
    n: int = SOBOL_N,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First-order and total Sobol indices of every ``Y`` over the family's space.

    Parameters
    ----------
    family : {"spikes", "lfp"}
    run_directory : str or path-like
    n : int, optional
        Rows of each sample matrix: ``n (d + 2)`` configurations.
    workers : int, optional

    Returns
    -------
    rows : pandas.DataFrame
        ``_rows``' output: ``matrix`` (``"A"``, ``"B"`` or ``"AB"``),
        ``column`` (the factor from ``B``, ``AB`` only), ``row``, then the
        factors, ``y``, ``value`` and each session's.
    indices : pandas.DataFrame
        One row per ``Y`` and factor: ``y``, ``factor``, then
        ``sobol_intervals``' columns.
    """
    factors = factor_space(RECIPES, family)
    reference = reference_template(family)
    a, b, ab = sobol_design(factors, reference, n)
    keys: list[dict[str, Any]] = [{"matrix": "A", "column": "", "row": r} for r in range(n)]
    keys += [{"matrix": "B", "column": "", "row": r} for r in range(n)]
    templates: list[Template] = [*a, *b]
    for factor, mixed in zip(factors, ab, strict=True):
        keys += [{"matrix": "AB", "column": factor.name, "row": r} for r in range(n)]
        templates += mixed
    base = compile(reference)
    outputs = evaluate_many(
        [compile(template) for template in templates],
        [base] * len(templates),
        run_directory,
        workers=workers,
    )
    tables = []
    d = len(factors)
    for name in Y_NAMES:
        y = np.array([_mean(found[name]) for found in outputs])
        table = sobol_intervals(y[:n], y[n : 2 * n], y[2 * n :].reshape(d, n))
        tables.append(table.assign(y=name, factor=[f.name for f in factors]))
    indices = pd.concat(tables, ignore_index=True)
    columns = ["y", "factor", *[c for c in indices.columns if c not in ("y", "factor")]]
    return _rows("sobol", keys, templates, outputs), indices[columns]


def _differing(first: Template, second: Template) -> tuple[str, ...]:
    return tuple(
        field.name
        for field in dataclasses.fields(first)
        if getattr(first, field.name) != getattr(second, field.name)
    )


def shapley_pair_list(
    family: str, run_directory: str | os.PathLike[str], *, workers: int = 1
) -> list[tuple[str, str]]:
    """The pairs decomposed: each represented template (``distinct_ids``)
    against the family's reference, then the ``N_LOWEST_PAIRS`` pairs of them
    with the lowest mean Jaccard on the reference sessions.

    Parameters
    ----------
    family : {"spikes", "lfp"}
    run_directory : str or path-like
    workers : int, optional

    Returns
    -------
    pairs : list of (str, str)
        ``(a, b)`` by configuration id, ``"reference"`` for the reference.
    """
    ids = distinct_ids(family)
    pipelines = [compile(TEMPLATES[config_id][0]) for config_id in ids]
    combinations = list(itertools.combinations(range(len(ids)), 2))
    outputs = evaluate_many(
        [pipelines[i] for i, _ in combinations],
        [pipelines[j] for _, j in combinations],
        run_directory,
        workers=workers,
    )
    agreement = sorted(
        (_mean(found["jaccard_reference"]), ids[i], ids[j])
        for (i, j), found in zip(combinations, outputs, strict=True)
    )
    pairs = [(config_id, "reference") for config_id in ids]
    return pairs + [(a, b) for _, a, b in agreement[:N_LOWEST_PAIRS]]


def shapley_pairs(
    family: str, run_directory: str | os.PathLike[str], *, workers: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shapley decompositions of the difference between pairs of configurations.

    For a pair ``(a, b)`` differing in the factors ``D``, ``v(S)`` is a ``Y``
    of ``a`` with the factors in ``S`` taken from ``b``, ``b`` its reference:
    ``jaccard_reference`` (so ``v(empty) = J(a, b)`` and ``v(D) = 1``) and ``f1``.

    Parameters
    ----------
    family : {"spikes", "lfp"}
    run_directory : str or path-like
    workers : int, optional

    Returns
    -------
    rows : pandas.DataFrame
        ``_rows``' output: ``pair`` (``"a|b"``), ``subset`` (the factors from
        ``b``, comma-separated), the factors, ``y``, ``value`` and each
        session's.
    values : pandas.DataFrame
        One row per pair, ``Y`` and factor: ``pair``, ``a``, ``b``, ``y``,
        ``factor``, ``phi``, ``error`` (Monte Carlo standard error, 0 when
        exact), ``v_empty``, ``v_all``.
    """
    reference = reference_template(family)
    templates = {config_id: TEMPLATES[config_id][0] for config_id in in_space_ids(family)}
    templates[REFERENCE_CONDITION] = reference
    pairs = shapley_pair_list(family, run_directory, workers=workers)
    keys, configurations, references = [], [], []
    for a, b in pairs:
        first, second = templates[a], templates[b]
        for subset in shapley_subsets(_differing(first, second)):
            keys.append({"pair": f"{a}|{b}", "subset": ",".join(sorted(subset))})
            configurations.append(
                dataclasses.replace(first, **{name: getattr(second, name) for name in subset})
            )
            references.append(compile(second))
    outputs = evaluate_many(
        [compile(template) for template in configurations],
        references,
        run_directory,
        workers=workers,
    )
    found = {
        (key["pair"], key["subset"]): output for key, output in zip(keys, outputs, strict=True)
    }
    values: list[dict[str, Any]] = []
    for a, b in pairs:
        differing = _differing(templates[a], templates[b])
        for name in ("jaccard_reference", "f1"):

            def value(subset: frozenset[str], pair: str = f"{a}|{b}", y: str = name) -> float:
                return _mean(found[pair, ",".join(sorted(subset))][y])

            phi, error = shapley(value, differing)
            ends = {"v_empty": value(frozenset()), "v_all": value(frozenset(differing))}
            values.extend(
                {
                    "pair": f"{a}|{b}",
                    "a": a,
                    "b": b,
                    "y": name,
                    "factor": factor,
                    "phi": phi[factor],
                    "error": error[factor],
                    **ends,
                }
                for factor in differing
            )
    return _rows("shapley", keys, configurations, outputs), pd.DataFrame(values)


def fixed_point_outputs(
    recipes: Sequence[RecipeConfig],
    contexts: Iterable[SessionContext],
    families: Sequence[str] = FAMILIES,
) -> pd.DataFrame:
    """Every configuration without a template, its reason and its public-call ``Y``s.

    Parameters
    ----------
    recipes : sequence of RecipeConfig
    contexts : iterable of SessionContext
        The ``K`` reference sessions, taken one at a time and released after.
    families : sequence of {"spikes", "lfp"}, optional
        The families whose expression and reference the ``Y``s are against.

    Returns
    -------
    table : pandas.DataFrame
        One row per fixed point and family: ``config_id``, ``family``,
        ``reason``, then each of ``Y_NAMES`` (the mean over the sessions)
        against that family's expression and reference configuration. A
        configuration whose call raises on a session keeps its row, the
        error in ``error`` and its ``Y``s missing.
    """
    fixed = [config for config in recipes if config.config_id in FIXED_POINTS]
    references = {family: compile(reference_template(family)) for family in families}
    per_session: dict[tuple[str, str], list[dict[str, float]]] = {}
    errors: dict[str, str] = {}
    for context in contexts:
        for config in fixed:
            if config.config_id in errors:
                continue
            try:
                events = recipe_events(config, context.session)
            except Exception as error:  # a method's failure is recorded, never raised
                errors[config.config_id] = f"{type(error).__name__}: {error}"[:200]
                continue
            for family, reference in references.items():
                per_session.setdefault((config.config_id, family), []).append(
                    session_outputs(events, context.events(reference), context, family)
                )
        context.release()
    rows = []
    for config in fixed:
        for family in families:
            found = per_session.get((config.config_id, family), [])
            failed = config.config_id in errors
            rows.append(
                {
                    "config_id": config.config_id,
                    "family": family,
                    "reason": FIXED_POINTS[config.config_id],
                    **{
                        name: float("nan")
                        if failed
                        else _mean([outputs[name] for outputs in found])
                        for name in Y_NAMES
                    },
                    "error": errors.get(config.config_id, ""),
                }
            )
    return pd.DataFrame(rows)


# Figures


def plot_sobol(indices: pd.DataFrame, family: str, *, caveat: str = "") -> Any:
    """Bars of first-order and total indices with their intervals, a panel per ``Y``.

    Parameters
    ----------
    indices : pandas.DataFrame
        ``sobol``'s second table.
    family : str
    caveat : str, optional
        ``family_caveat``'s, under the title.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    names = list(dict.fromkeys(indices["y"]))
    figure, axes = plt.subplots(
        len(names), 1, figsize=(7, 2.2 * len(names)), sharex=True, squeeze=False
    )
    for axis, name in zip(axes[:, 0], names, strict=True):
        table = indices[indices["y"] == name]
        x = np.arange(len(table))
        for offset, kind, color in ((-0.2, "first", "#0072B2"), (0.2, "total", "#E69F00")):
            values = table[kind].to_numpy(dtype=float)
            errors = np.abs(table[[f"{kind}_low", f"{kind}_high"]].to_numpy(float).T - values)
            axis.bar(x + offset, values, 0.4, yerr=errors, color=color, label=kind)
        axis.set_ylabel(name)
        axis.axhline(0, color="black", linewidth=0.5)
    axes[-1, 0].set_xticks(np.arange(len(table)), table["factor"], rotation=45, ha="right")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].set_title(f"Sobol indices, {family}" + (f"\n{caveat}" if caveat else ""))
    figure.tight_layout()
    return figure


def plot_shapley(values: pd.DataFrame, pair: str, y: str, *, caveat: str = "") -> Any:
    """A waterfall of one pair's Shapley values for one ``Y``, with standard errors.

    Parameters
    ----------
    values : pandas.DataFrame
        ``shapley_pairs``' second table.
    pair : str
    y : str
    caveat : str, optional
        ``family_caveat``'s, under the title.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    table = values[(values["pair"] == pair) & (values["y"] == y)]
    start = float(table["v_empty"].iloc[0])
    phi = table["phi"].to_numpy(dtype=float)
    bottoms = start + np.r_[0.0, np.cumsum(phi)[:-1]]
    figure, axis = plt.subplots(figsize=(6, 3))
    colors = np.where(phi >= 0, "#009E73", "#D55E00")
    axis.bar(np.arange(len(phi)), phi, bottom=bottoms, color=colors, yerr=table["error"])
    axis.axhline(start, color="grey", linewidth=0.5)
    axis.axhline(float(table["v_all"].iloc[0]), color="black", linewidth=0.5)
    axis.set_xticks(np.arange(len(phi)), table["factor"], rotation=45, ha="right")
    axis.set_ylabel(y)
    axis.set_title(pair + (f"\n{caveat}" if caveat else ""))
    figure.tight_layout()
    return figure


# Command line


def _results_csv(path: Path, frame: pd.DataFrame) -> None:
    write_result(path, frame.to_csv(index=False).encode())


def _save_figure(path: Path, figure: Any) -> None:
    import io

    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=120)
    plt.close(figure)
    write_result(path, buffer.getvalue())


def _free_cores() -> float:
    """Cores not busy by the one-minute load average."""
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
    return max(1.0, (os.cpu_count() or 1) - load)


def smoke(
    family: str, run_directory: str | os.PathLike[str], *, workers: int
) -> dict[str, Any]:
    """Time ``SMOKE_CONFIGURATIONS`` configurations on one reference session.

    The configurations are the first rows of ``A`` of the Sobol design at
    ``SOBOL_N``.

    Parameters
    ----------
    family : {"spikes", "lfp"}
    run_directory : str or path-like
    workers : int
        For the extrapolation.

    Returns
    -------
    report : dict
        ``seconds_per_configuration``, ``context_seconds`` (simulating and
        checking the session), ``peak_rss_bytes``, ``d``, the
        configurations and hours of each analysis at ``workers``.
    """
    started = wall_clock.perf_counter()
    context = next(iter(reference_contexts(run_directory, [0])))
    context_seconds = wall_clock.perf_counter() - started
    factors = factor_space(RECIPES, family)
    reference = reference_template(family)
    a, _, _ = sobol_design(factors, reference, SOBOL_N)
    base = compile(reference)
    context.events(base)
    timings = []
    for template in a[:SMOKE_CONFIGURATIONS]:
        begun = wall_clock.perf_counter()
        evaluate_session(compile(template), context, base)
        timings.append(wall_clock.perf_counter() - begun)
    seconds = float(np.mean(timings))
    d = len(factors)
    ids = distinct_ids(family)
    oat = 1 + sum(len(factor.points()) for factor in factors)
    shapley_configurations = 0
    for config_id in ids:
        size = len(_differing(TEMPLATES[config_id][0], reference))
        shapley_configurations += (
            2**size if size <= SHAPLEY_EXACT_UP_TO else SHAPLEY_PERMUTATIONS * size
        )
    pairwise = len(ids) * (len(ids) - 1) // 2
    largest = max(
        (
            len(_differing(TEMPLATES[x][0], TEMPLATES[y][0]))
            for x, y in itertools.combinations(ids, 2)
        ),
        default=0,
    )
    lowest_bound = min(N_LOWEST_PAIRS, pairwise) * (
        2**largest if largest <= SHAPLEY_EXACT_UP_TO else SHAPLEY_PERMUTATIONS * largest
    )
    counts = {
        "oat": oat,
        "sobol_256": 256 * (d + 2),
        "sobol_128": 128 * (d + 2),
        "shapley_reference_pairs": shapley_configurations,
        "shapley_pair_selection": pairwise,
        "shapley_lowest_pairs_at_most": lowest_bound,
    }

    def hours(configurations: int) -> float:
        return configurations * K * seconds / workers / 3600

    return {
        "family": family,
        "free_cores": _free_cores(),
        "n_in_space": len(ids),
        "d": d,
        "configurations_timed": len(timings),
        "seconds_per_configuration": seconds,
        "context_seconds": context_seconds,
        "peak_rss_bytes": peak_rss_bytes(),
        "workers": workers,
        "configurations": counts,
        "hours": {name: hours(count) for name, count in counts.items()},
    }


def verify_family(family: str, run_directory: str | os.PathLike[str]) -> pd.DataFrame:
    """``verify_all`` on the edge sessions and the ``K`` reference sessions.

    Parameters
    ----------
    family : {"spikes", "lfp"}
        The family whose templates are checked; every fixed point is listed.
    run_directory : str or path-like

    Returns
    -------
    table : pandas.DataFrame
        ``verify_all``'s, after ``check_report``.
    """
    parameters = reference_parameters(run_directory)
    check_report(run_directory, parameters)
    recipes = [
        config
        for config in RECIPES
        if config.config_id in FIXED_POINTS
        or family_of(TEMPLATES[config.config_id][0]) == family
    ]
    edges = (
        SessionContext(session, f"edge/{name}")
        for name, session in edge_sessions(parameters).items()
    )
    return verify_all(recipes, itertools.chain(edges, reference_contexts(run_directory)))


def family_caveat(n_distinct: int, *, below_minimum: bool) -> str:
    """What every output of a family with too few represented methods says.

    Parameters
    ----------
    n_distinct : int
        The family's represented methods, identical templates once.
    below_minimum : bool
        Whether the maintainer chose to run Sobol and Shapley regardless.

    Returns
    -------
    caveat : str
        Empty at ``MINIMUM_IN_SPACE`` methods or more.
    """
    if n_distinct >= MINIMUM_IN_SPACE:
        return ""
    caveat = f"rests on {n_distinct} methods, below the design's {MINIMUM_IN_SPACE}"
    return f"{caveat}; the maintainer chose to run it" if below_minimum else caveat


def main(argv: Sequence[str] | None = None) -> None:
    """The command line; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument(
        "--analysis", default="all", choices=("oat", "sobol", "shapley", "all")
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--below-minimum",
        action="store_true",
        help=(
            f"run Sobol and Shapley on a family with fewer than {MINIMUM_IN_SPACE} "
            "represented methods, every output of the family labelled so"
        ),
    )
    parser.add_argument(
        "--run-directory",
        help="the run's directory (default: examples/benchmark/output/<run-name>)",
    )
    parser.add_argument(
        "--results-directory",
        help="where to write (default: examples/benchmark/results/<run-name>/attribution)",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1.")
    run_directory = Path(args.run_directory or OUTPUT / args.run_name)
    results = Path(args.results_directory or RESULTS / args.run_name / "attribution")
    if args.smoke:
        print(json.dumps(smoke(args.family, run_directory, workers=args.workers), indent=2))
        return
    expected = set(in_space_ids(args.family))
    n_distinct = len(distinct_ids(args.family))
    refused = n_distinct < MINIMUM_IN_SPACE and not args.below_minimum
    refusal = (
        f"The {args.family} family has {n_distinct} represented methods, fewer than "
        f"{MINIMUM_IN_SPACE}: no Sobol or Shapley analysis is run (--below-minimum "
        "runs them, labelled)."
    )
    if args.analysis in ("sobol", "shapley") and refused:
        raise SystemExit(refusal)
    caveat = family_caveat(n_distinct, below_minimum=args.below_minimum)
    if caveat:
        print(f"The {args.family} family {caveat}.", file=sys.stderr)
    verification = verify_family(args.family, run_directory)
    found = set(
        verification.loc[
            verification["in_space"] & (verification["family"] == args.family), "config_id"
        ]
    )
    if found != expected:
        failed = verification[verification["config_id"].isin(expected - found)]
        msg = "Templates not verified on the run's sessions:\n" + "\n".join(
            f"- {row.config_id}: {row.reason}" for row in failed.itertuples()
        )
        raise SystemExit(msg)
    output = run_directory / "attribution"
    output.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    def write(name: str, frame: pd.DataFrame) -> None:
        _results_csv(results / f"{args.family}_{name}.csv", frame.assign(caveat=caveat))

    write("in_space", verification)
    space = factor_space(RECIPES, args.family)
    write(
        "factor_space",
        pd.DataFrame(
            [
                {"factor": f.name, "kind": f.kind, "levels": json.dumps(list(f.levels))}
                for f in space
            ]
        ),
    )
    write("reference", pd.DataFrame([_key_columns(reference_template(args.family))]))
    write(
        "fixed_points",
        fixed_point_outputs(RECIPES, reference_contexts(run_directory), (args.family,)),
    )
    analyses = ("oat", "sobol", "shapley") if args.analysis == "all" else (args.analysis,)
    for analysis in analyses:
        if analysis != "oat" and refused:
            raise SystemExit(refusal)
        started = wall_clock.perf_counter()
        if analysis == "oat":
            rows, summary = one_at_a_time(args.family, run_directory, workers=args.workers)
        elif analysis == "sobol":
            rows, summary = sobol(args.family, run_directory, workers=args.workers)
            _save_figure(
                results / f"{args.family}_sobol.png",
                plot_sobol(summary, args.family, caveat=caveat),
            )
        else:
            rows, summary = shapley_pairs(args.family, run_directory, workers=args.workers)
            for pair in dict.fromkeys(summary["pair"]):
                slug = pair.replace("|", "__").replace(".", "-")
                _save_figure(
                    results / f"{args.family}_shapley_{slug}.png",
                    plot_shapley(summary, pair, "jaccard_reference", caveat=caveat),
                )
        _write_table(rows.assign(caveat=caveat), output / f"{args.family}_{analysis}.csv.gz")
        write(analysis, summary)
        print(
            f"{args.family} {analysis}: {len(rows) // len(Y_NAMES)} configurations, "
            f"{wall_clock.perf_counter() - started:.0f} s",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()

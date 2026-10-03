"""Buzsáki lab databank sessions: verify inputs, reproduce ``bz_FindRipples``, compare, explain.

Each session in `SESSIONS` pairs a databank ``*.ripples.events.mat`` (an Internet
Archive capture, accepted only when complete) with the session's LFP on DANDI.
The steps, run in order::

    fetch    the event file, .xml, session.mat, sessionInfo.mat, the .lfp head and
             the DANDI asset's metadata
    verify   inputs.json: rates, channel count, lengths, the head match and the
             DANDI column of the stored channel, the channel tag, gaps and epochs,
             the events against the recording, the dated source code
    stream   the detection channel's whole LFP column from DANDI, cached
             (``--measure-minutes M`` reads only the first M minutes and
             extrapolates the time, bytes and memory of the whole column)
    stage    the source's filter, square, smoothing and standard deviation,
             against the stored ``stdev``
    detect   the package's ``Zugaro_ripple_detector`` with the source's settings,
             and a NumPy/SciPy transcription of the dated source (a diagnostic,
             never the package)
    compare  ``match_events`` for each pair of inventories -> comparison.csv
    explain  attribution.json: the differences traced to a rule, with the
             diagnostics behind it; small PNGs of a few events of each class

Large intermediates go under ``<cache>/sessions/<session>/``; the small results
(each under 1 MB) under ``examples/reference_recordings/results/<session>/``.
``stream`` and ``verify`` read the NWB file, so the script runs as::

    uv run --with remfile --with h5py python examples/reference_recordings/databank.py \\
        MS10 --steps fetch,verify,stream,stage,detect,compare,explain

The other steps need neither package once the column is cached. A failed input
check writes inputs.json with the failure and stops the session; nothing is
patched to let the run proceed.
"""

from __future__ import annotations

import argparse
import inspect
import json
import resource
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, NoReturn

import fetch
import numpy as np
import pandas as pd
import readers
import scipy.io
import scipy.signal
from numpy.typing import ArrayLike, NDArray

DATABANK_URL = "https://buzsakilab.nyumc.org/datasets"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MAX_RESULT_BYTES = 1_000_000
MATCH_IOU_LEVELS = (0.0, 0.2, 0.5)
"""Minimum IoU levels of the comparison, as in the benchmark (0 is any overlap)."""
HEAD_BYTES = 65536
DETECTOR = "Zugaro_ripple_detector"
STEPS = ("fetch", "verify", "stream", "stage", "detect", "compare", "explain")

FloatArray = NDArray[np.float64]


# --- the per-session table ------------------------------------------------------------


@dataclass(frozen=True)
class WaybackCapture:
    """One databank file as the Internet Archive captured it.

    Attributes
    ----------
    suffix : str
        File name after the session's basename, for example ``".xml"``.
    timestamp : str
        Capture timestamp, ``YYYYMMDDhhmmss``.
    sha256 : str or None
        Digest of the complete file; None for a head capture.
    length : int
        The original server length in bytes.
    """

    suffix: str
    timestamp: str
    sha256: str | None
    length: int


@dataclass(frozen=True)
class SourceCode:
    """The dated source of a released event file and what it implies.

    Attributes
    ----------
    repository, commit, path : str
        Where the version that wrote the file lives.
    committed : str
        The commit's date (ISO).
    filter_kind : {"cheby2", "butter"}
        ``bz_Filter``'s design as that version calls it.
    filter_order : int
        The design order (a band-pass has twice as many poles).
    filter_stopband_db : float or None
        ``cheby2``'s stopband attenuation (``bz_Filter``'s ``ripple``).
    filter_nyquist : float
        The Nyquist frequency the design divides by. ``bz_Filter`` uses its
        default (625 Hz) for a plain array, whatever the sampling rate.
    smoothing_samples : int
        The moving average's length (``windowLength``).
    minimum_duration_ms : float or None
        The version's minimum duration; None when it has none.
    evidence : tuple of str
        Why this version, one statement per entry.
    """

    repository: str
    commit: str
    path: str
    committed: str
    filter_kind: str
    filter_order: int
    filter_stopband_db: float | None
    filter_nyquist: float
    smoothing_samples: int
    minimum_duration_ms: float | None
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class DatabankSession:
    """One databank session: its released events, support files and DANDI LFP.

    Attributes
    ----------
    key : str
        Short name, the key in `SESSIONS` and the results folder.
    databank_path : str
        Folder under the databank's ``datasets/``.
    basename : str
        The session's file basename.
    events : WaybackCapture
        The ``.ripples.events.mat``.
    support : tuple of WaybackCapture
        ``.xml``, ``.session.mat`` and ``.sessionInfo.mat``.
    lfp_head : WaybackCapture
        The ``.lfp`` capture read in head mode (its ``length`` is the original
        file's).
    dandiset, dandiset_version, lfp_asset_id, lfp_series : str
        The DANDI asset holding the LFP and the ElectricalSeries path in it.
    source_code : SourceCode
        The dated detector version.
    channel_tag : str
        ``session.mat``'s channel tag naming the detection channel (1-based).
    noise_channel_tag : str or None
        The tag of the channel a noise veto used, when the file holds noise events.
    """

    key: str
    databank_path: str
    basename: str
    events: WaybackCapture
    support: tuple[WaybackCapture, ...]
    lfp_head: WaybackCapture
    dandiset: str
    dandiset_version: str
    lfp_asset_id: str
    lfp_series: str
    source_code: SourceCode
    channel_tag: str = "Ripple"
    noise_channel_tag: str | None = None

    def url(self, capture: WaybackCapture) -> str:
        """The original databank URL of one of the session's files."""
        return f"{DATABANK_URL}/{self.databank_path}/{self.basename}{capture.suffix}"

    def cache_path(self, capture: WaybackCapture) -> str:
        """Where the capture lives relative to the cache (head mode adds ``.head``)."""
        return f"buzsaki/{self.key}/{self.basename}{capture.suffix}"


PETERSEN_FORK_2021 = SourceCode(
    repository="https://github.com/petersenpeter/buzcode",
    commit="bc3fc91fab1ab1a626022d8efbf0c101e7b8d1b7",
    path="analysis/lfp/bz_FindRipples.m",
    committed="2021-01-14T08:51:09-05:00",
    filter_kind="cheby2",
    filter_order=4,
    filter_stopband_db=20.0,
    filter_nyquist=625.0,
    smoothing_samples=11,
    minimum_duration_ms=None,
    evidence=(
        (
            "The MAT header says the file was created on Thu Oct 15 16:03:31 2020 "
            "(Wayback orig_last_modified Thu, 15 Oct 2020 20:03:30 GMT)."
        ),
        (
            "The struct holds times, detectorName, peaks, peakNormedPower, stdev, noise and "
            "detectorParams (inputParser Results: basepath, channel, durations, frequency, "
            "passband, restrict, saveMat, show, stdev, thresholds; noise removed)."
        ),
        (
            "buzsakilab/buzcode's version current in October 2020 "
            "(analysis/SharpWaveRipples/bz_FindRipples.m at eec82c13, 2020-04-14) cannot have "
            "written it: it saves timestamps and detectorinfo and has EMGThresh, minDuration "
            "(default 20 ms) and plotType parameters. No version on any branch of "
            "buzsakilab/buzcode combines a passband parameter with the times/detectorParams "
            "layout: the layout is last used at 01061cc (2017-08-24), and passband became a "
            "parameter at c57269b (2018-02-20)."
        ),
        (
            "petersenpeter/buzcode's analysis/lfp/bz_FindRipples.m at bc3fc91 is the only "
            "version found with exactly that parameter set and layout: it adds the passband "
            "parameter, makes saveMat default to true and saves '.ripples.events.mat' "
            "(its parent 5a0b450, 2019-07-05, has no passband parameter, so cannot have "
            "stored passband [120 180], and saves '.ripples.event.mat'). It was committed on "
            "2021-01-14, three months after the file was written, so the file came from that "
            "change in a working copy; the lines it changes are the only ones that differ "
            "from 5a0b450."
        ),
        (
            "At bc3fc91 the filter is bz_FilterLFP(double(lfp.data),'passband',passband) -> "
            "bz_Filter with its defaults: cheby2, order 4, ripple 20 dB, nyquist 625, "
            "filtfilt (bz_Filter last changed at b6c55fe, 2019-03-21; bz_FilterLFP at "
            "5a0b450, 2019-07-05). Not the butter order-3 filter of buzsakilab/buzcode."
        ),
        (
            "The same file at bc3fc91: windowLength = 11; Filter0 (a centred 11-sample "
            "moving average, zero-padded at both ends); unity over all samples (restrict "
            "empty), std with N-1; crossings of > low with diff (start = last sample at or "
            "below, stop = last sample above), incomplete edge runs dropped; merge while next "
            "start - current stop < durations(1)/1000*frequency samples, with no cap; keep "
            "if max(normalized) > high; peak = the trough (minimum) of the filtered signal; "
            "drop duration > durations(2)/1000 s; no minimum duration, no noise channel "
            "given, no EMG rule."
        ),
        (
            "bz_GetLFP at bc3fc91 reads int16 counts (bz_LoadBinary) without scaling, and "
            "builds timestamps from 0 in steps of 1/1250 s."
        ),
    ),
)

SESSIONS: dict[str, DatabankSession] = {
    "MS10": DatabankSession(
        key="MS10",
        databank_path="PetersenP/MS10/Peter_MS10_170307_154746_concat",
        basename="Peter_MS10_170307_154746_concat",
        events=WaybackCapture(
            ".ripples.events.mat",
            "20231129113529",
            "d28e4f4dab23319de87372b44f4af3165fd7c8959bb1cad24a4594291975e628",
            11305,
        ),
        support=(
            WaybackCapture(
                ".xml",
                "20231210194221",
                "42b8c64e62252ab242716e7f27942d2cc5714fb55a5feec864a2f0b727e06a22",
                53832,
            ),
            WaybackCapture(
                ".session.mat",
                "20231129111442",
                "72b5b6f475f2319d58c6b7a27404e0b058e23d2bdd931230f86d3633c8502f1c",
                4096,
            ),
            WaybackCapture(
                ".sessionInfo.mat",
                "20231210205756",
                "9ff9bc2119fe39f5959cd8c74bc37c543bf0a57cf5756f4b61ba9f8259852811",
                2810,
            ),
        ),
        lfp_head=WaybackCapture(".lfp", "20231129114227", None, 3_022_560_000),
        dandiset="000059",
        dandiset_version="0.250624.0444",
        lfp_asset_id="8941eed3-5cb3-4a81-be34-a37a211a4ebd",
        lfp_series="/processing/ecephys/LFP/LFP",
        source_code=PETERSEN_FORK_2021,
    ),
}


# --- input checks (pure) --------------------------------------------------------------


@dataclass
class Check:
    """One input check: its name, whether it passed, and what was measured."""

    name: str
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


class InputCheckFailed(RuntimeError):
    """An input check failed; the session stops (inputs.json records why)."""


def check_rates(xml_rate: float | None, nwb_rate: float, stored_rate: float) -> Check:
    """The LFP rate from the ``.xml``, the NWB and the detector's stored ``frequency`` agree.

    Parameters
    ----------
    xml_rate : float or None
        ``fieldPotentials/lfpSamplingRate``.
    nwb_rate : float
        The ElectricalSeries' rate (Hz).
    stored_rate : float
        The event file's ``detectorParams.frequency``.
    """
    rates = {"xml": xml_rate, "nwb": nwb_rate, "stored": stored_rate}
    passed = xml_rate is not None and xml_rate == nwb_rate == stored_rate
    reason = "" if passed else f"sampling rates disagree: {rates}"
    return Check("sampling_rate", passed, rates, reason)


def check_length(original_bytes: int, n_channels: int, dandi_shape: Sequence[int]) -> Check:
    """The original ``.lfp`` holds as many samples as the DANDI LFP has rows.

    Parameters
    ----------
    original_bytes : int
        The ``.lfp``'s original length (the head capture's declared length).
    n_channels : int
        Channels in the ``.lfp`` (the ``.xml``'s ``nChannels``).
    dandi_shape : sequence of int
        ``(n_rows, n_columns)`` of the DANDI LFP.
    """
    detail: dict[str, Any] = {
        "original_bytes": int(original_bytes),
        "n_channels": int(n_channels),
        "dandi_shape": [int(n) for n in dandi_shape],
    }
    try:
        n_samples = readers.binary_n_samples(int(original_bytes), int(n_channels))
    except ValueError as error:
        return Check("lfp_length", False, detail, str(error))
    detail["lfp_samples"] = n_samples
    problems = []
    if n_samples != dandi_shape[0]:
        problems.append(f".lfp holds {n_samples} samples, DANDI {dandi_shape[0]} rows")
    if dandi_shape[1] > n_channels:
        problems.append(f"DANDI has {dandi_shape[1]} columns, the .lfp {n_channels} channels")
    return Check("lfp_length", not problems, detail, "; ".join(problems))


def head_column_map(file_rows: ArrayLike, dandi_rows: ArrayLike) -> list[list[int]]:
    """For each DANDI column, the ``.lfp`` channels whose first rows equal it exactly.

    Parameters
    ----------
    file_rows : array_like, shape (n_rows, n_file_channels)
        The first rows of the ``.lfp`` (int16 counts).
    dandi_rows : array_like, shape (n_rows, n_columns)
        The same rows of the DANDI LFP.

    Returns
    -------
    list of list of int
        One list per DANDI column: the 0-based ``.lfp`` channels it equals.
    """
    file_rows, dandi_rows = np.asarray(file_rows), np.asarray(dandi_rows)
    if file_rows.shape[0] != dandi_rows.shape[0]:
        msg = f"row counts differ: {file_rows.shape[0]} and {dandi_rows.shape[0]}"
        raise ValueError(msg)
    equal = np.all(dandi_rows[:, :, None] == file_rows[:, None, :], axis=0)
    return [np.flatnonzero(row).tolist() for row in equal]


def check_head_match(
    file_rows: ArrayLike, dandi_rows: ArrayLike, stored_channel: int
) -> Check:
    """The DANDI LFP's first rows equal the ``.lfp`` head, and which column holds the channel.

    Passes when every DANDI column equals at least one ``.lfp`` channel and the
    stored channel is held by exactly one column that equals no other channel.
    The map is reported, never assumed to be the identity.

    Parameters
    ----------
    file_rows : array_like, shape (n_rows, n_file_channels)
        The ``.lfp`` head as int16 counts.
    dandi_rows : array_like, shape (n_rows, n_columns)
        The DANDI LFP's first ``n_rows`` rows.
    stored_channel : int
        The event file's detection channel (0-based, as buzcode stores it).

    Returns
    -------
    Check
        ``detail``: ``rows_compared``, ``column_to_channel`` (an int per column, a
        list where ambiguous, None where nothing matches), ``identity``,
        ``channels_absent`` and ``stored_channel_column``.
    """
    mapping = head_column_map(file_rows, dandi_rows)
    n_file_channels = np.asarray(file_rows).shape[1]
    compact: list[int | list[int] | None] = [
        hits[0] if len(hits) == 1 else (hits or None) for hits in mapping
    ]
    matched = {channel for hits in mapping for channel in hits}
    holders = [column for column, hits in enumerate(mapping) if stored_channel in hits]
    detail: dict[str, Any] = {
        "rows_compared": int(np.asarray(file_rows).shape[0]),
        "column_to_channel": compact,
        "identity": compact == list(range(len(mapping))),
        "channels_absent": sorted(set(range(n_file_channels)) - matched),
        "ambiguous_columns": [c for c, hits in enumerate(mapping) if len(hits) > 1],
        "stored_channel": int(stored_channel),
        "stored_channel_column": holders[0] if len(holders) == 1 else None,
    }
    problems = []
    unmatched = [column for column, hits in enumerate(mapping) if not hits]
    if unmatched:
        problems.append(f"DANDI columns {unmatched} match no .lfp channel")
    if not holders:
        problems.append(f"stored channel {stored_channel} is in no DANDI column")
    elif len(holders) > 1:
        problems.append(f"stored channel {stored_channel} matches DANDI columns {holders}")
    elif len(mapping[holders[0]]) > 1:
        problems.append(
            f"DANDI column {holders[0]} equals channels {mapping[holders[0]]}, not only "
            f"the stored channel {stored_channel}"
        )
    return Check("head_match", not problems, detail, "; ".join(problems))


def check_channel_tag(
    stored_zero_based: int, tag_one_based: int | Sequence[int] | None
) -> Check:
    """The stored detection channel against ``session.mat``'s channel tag.

    buzcode stores the channel 0-based (``bz_GetLFP``: "0-indexing, a la
    Neuroscope"); CellExplorer's ``session.mat`` channel tags are 1-based. A
    one-channel tag agrees when ``stored + 1 == tag``. A tag listing several
    channels does not single out the detection channel: it agrees when
    ``stored + 1`` is among them, and the detail says the tag had several.

    Parameters
    ----------
    stored_zero_based : int
        ``detectorParams.channel``.
    tag_one_based : int, sequence of int or None
        The tag's channels; None when ``session.mat`` has no such tag.
    """
    channels = (
        None
        if tag_one_based is None
        else [int(c) for c in np.atleast_1d(np.asarray(tag_one_based)).ravel()]
    )
    detail: dict[str, Any] = {
        "stored_zero_based": int(stored_zero_based),
        "tag_one_based": channels[0] if channels and len(channels) == 1 else channels,
        "n_tag_channels": None if channels is None else len(channels),
    }
    if channels is None:
        return Check("channel_tag", False, detail, "session.mat has no such channel tag")
    if not channels:
        return Check("channel_tag", False, detail, "the channel tag lists no channels")
    wanted = int(stored_zero_based) + 1
    if len(channels) == 1:
        passed = wanted == channels[0]
        reason = (
            ""
            if passed
            else f"stored channel {stored_zero_based} (0-based) is not the tag's "
            f"{channels[0]} (1-based)"
        )
        return Check("channel_tag", passed, detail, reason)
    detail["rule"] = "a tag of several channels agrees when stored + 1 is among them"
    passed = wanted in channels
    reason = (
        ""
        if passed
        else f"stored channel {stored_zero_based} (0-based) is not among the tag's "
        f"{len(channels)} channels {channels} (1-based)"
    )
    return Check("channel_tag", passed, detail, reason)


def sample_positions(times: ArrayLike, starting_time: float, frequency: float) -> FloatArray:
    """``(times - starting_time) * frequency``: fractional sample positions."""
    return (np.asarray(times, dtype=float) - starting_time) * frequency


def check_events_in_recording(
    events: pd.DataFrame, starting_time: float, n_samples: int, frequency: float
) -> Check:
    """Every released start, peak and end lies on a recorded sample.

    A time is on the grid when its sample position is within four units in the
    last place of the largest time (times the rate) of an integer, so the test
    holds at any clock origin.

    Parameters
    ----------
    events : pandas.DataFrame
        ``start_time``, ``peak_time``, ``end_time`` in seconds.
    starting_time : float
        Time of the recording's first sample.
    n_samples : int
        Samples in the recording.
    frequency : float
        Sampling rate (Hz).
    """
    columns = ["start_time", "peak_time", "end_time"]
    times = events[columns].to_numpy(dtype=float)
    position = sample_positions(times, starting_time, frequency)
    largest = float(np.max(np.abs(times))) if times.size else 0.0
    tolerance = 4 * np.spacing(largest) * frequency + 4 * np.spacing(float(n_samples))
    off_grid = np.abs(position - np.round(position)) > tolerance
    outside = (np.round(position) < 0) | (np.round(position) > n_samples - 1)
    unordered = ~((times[:, 0] <= times[:, 1]) & (times[:, 1] <= times[:, 2]))
    detail = {
        "n_events": len(events),
        "first_start_sample": int(np.round(position[:, 0].min())) if len(events) else None,
        "last_end_sample": int(np.round(position[:, 2].max())) if len(events) else None,
        "n_samples": int(n_samples),
        "largest_grid_deviation_samples": float(np.max(np.abs(position - np.round(position))))
        if times.size
        else 0.0,
        "grid_tolerance_samples": float(tolerance),
    }
    problems = []
    if np.any(outside):
        rows = np.flatnonzero(outside.any(axis=1)).tolist()
        problems.append(f"events {rows[:10]} lie outside the {n_samples} recorded samples")
    if np.any(off_grid):
        rows = np.flatnonzero(off_grid.any(axis=1)).tolist()
        problems.append(f"events {rows[:10]} are off the {frequency} Hz grid")
    if np.any(unordered):
        problems.append(
            f"events {np.flatnonzero(unordered)[:10].tolist()} are not start <= peak <= end"
        )
    return Check("events_in_recording", not problems, detail, "; ".join(problems))


def epoch_boundaries(session_mat: Any) -> list[dict[str, Any]]:
    """The epochs a concatenated session's ``session.mat`` lists, in seconds.

    Parameters
    ----------
    session_mat : object
        ``scipy.io.loadmat(..., struct_as_record=False, squeeze_me=True)["session"]``.

    Returns
    -------
    list of dict
        ``name``, ``start_time`` and ``stop_time`` per epoch, in file order.
    """
    epochs = np.atleast_1d(getattr(session_mat, "epochs", np.empty(0, dtype=object)))
    return [
        {
            "name": str(getattr(e, "name", "")),
            "start_time": float(e.startTime),
            "stop_time": float(e.stopTime),
        }
        for e in epochs
    ]


def events_spanning(events: pd.DataFrame, boundaries: ArrayLike) -> list[int]:
    """Row positions of events whose closed bounds contain a boundary time."""
    starts, ends = events["start_time"].to_numpy(), events["end_time"].to_numpy()
    edges = np.asarray(boundaries, dtype=float)
    inside = (starts[:, None] <= edges[None, :]) & (edges[None, :] <= ends[:, None])
    return np.flatnonzero(inside.any(axis=1)).tolist()


# --- the dated source's steps (a transcription, for diagnosis only) --------------------


def source_filter(lfp: ArrayLike, passband: Sequence[float], code: SourceCode) -> FloatArray:
    """``bz_Filter`` as the dated version calls it: the design, then MATLAB's ``filtfilt``.

    The design is in transfer-function form, as MATLAB's ``[b a] = cheby2(...)``
    or ``butter(...)`` returns it, and the padding is MATLAB's: an odd reflection
    of ``3 * (n_coefficients - 1)`` samples at each end.

    Parameters
    ----------
    lfp : array_like, shape (n_time,)
        One channel (int16 counts are converted to float64, as ``double`` does).
    passband : sequence of float
        ``[low, high]`` in Hz.
    code : SourceCode
        The filter's kind, order, stopband attenuation and Nyquist frequency.

    Returns
    -------
    ndarray, shape (n_time,)
        The zero-phase filtered signal.
    """
    edges = np.asarray(passband, dtype=float) / code.filter_nyquist
    if code.filter_kind == "cheby2":
        b, a = scipy.signal.cheby2(
            code.filter_order, code.filter_stopband_db, edges, btype="bandpass"
        )
    elif code.filter_kind == "butter":
        b, a = scipy.signal.butter(code.filter_order, edges, btype="bandpass")
    else:
        msg = f"unknown filter kind {code.filter_kind!r}"
        raise ValueError(msg)
    pad = 3 * (max(len(a), len(b)) - 1)
    filtered: FloatArray = scipy.signal.filtfilt(
        b, a, np.asarray(lfp, dtype=np.float64), padtype="odd", padlen=pad
    )
    return filtered


def filter0(b: ArrayLike, x: ArrayLike) -> FloatArray:
    """MATLAB's ``Filter0`` from ``bz_FindRipples``: a causal FIR shifted to be centred.

    ``[y0 z] = filter(b,1,x); y = [y0(shift+1:end) ; z(1:shift)]`` with
    ``shift = (length(b)-1)/2``: the final state supplies the last ``shift``
    outputs, as if the input continued with zeros.

    Parameters
    ----------
    b : array_like, shape (n_taps,)
        Odd-length FIR coefficients.
    x : array_like, shape (n_time,)
        Input.

    Returns
    -------
    ndarray, shape (n_time,)
    """
    b, x = np.asarray(b, dtype=float), np.asarray(x, dtype=float)
    if len(b) % 2 != 1:
        msg = "filter order should be odd"
        raise ValueError(msg)
    shift = (len(b) - 1) // 2
    y0, final = scipy.signal.lfilter(b, [1.0], x, zi=np.zeros(len(b) - 1))
    return np.concatenate([y0[shift:], final[:shift]])


def normalized_squared_signal(
    signal: ArrayLike, window_length: int, sd: float | None = None
) -> tuple[FloatArray, float, float]:
    """``unity(Filter0(window, signal.^2), sd, [])``: smoothed power in SD units.

    Parameters
    ----------
    signal : array_like, shape (n_time,)
        The filtered channel.
    window_length : int
        The moving average's length in samples (odd).
    sd : float, optional
        A standard deviation to use instead of the computed one (``stdev``).

    Returns
    -------
    normalized : ndarray, shape (n_time,)
        ``(smoothed - mean) / sd``.
    sd : float
        The standard deviation used (computed with N-1, as MATLAB's ``std``).
    mean : float
        The mean of the smoothed power.
    """
    smoothed = filter0(np.ones(window_length) / window_length, np.asarray(signal, float) ** 2)
    mean = float(np.mean(smoothed))
    used = float(np.std(smoothed, ddof=1)) if sd is None else float(sd)
    return (smoothed - mean) / used, used, mean


SOURCE_EVENT_COLUMNS = [
    "start_index",
    "stop_index",
    "trough_index",
    "max_index",
    "start_time",
    "peak_time",
    "end_time",
    "max_power_time",
    "peak_normed_power",
]


def source_segmentation(
    normalized: ArrayLike,
    signal: ArrayLike,
    timestamps: ArrayLike,
    *,
    low_threshold: float,
    high_threshold: float,
    minimum_inter_ripple_interval_ms: float,
    maximum_duration_ms: float,
    frequency: float,
    minimum_duration_ms: float | None = None,
) -> pd.DataFrame:
    """The dated ``bz_FindRipples``' event steps, transcribed line by line.

    1. ``thresholded = normalized > low``; ``start`` is each index where
       ``diff`` rises (the last sample at or below), ``stop`` each where it
       falls (the last sample above). An incomplete last or first run is
       dropped, and when both are incomplete the first stop and last start go.
    2. Merge while ``start(i) - current stop < interval/1000*frequency``
       samples, with no cap on the merged length.
    3. Keep an event whose maximum of ``normalized`` over ``[start, stop]`` is
       strictly above ``high``.
    4. The peak is the first minimum of ``signal`` over ``[start, stop]``.
    5. Drop events with ``timestamps(stop) - timestamps(start)`` above the
       maximum (seconds); a minimum duration applies only when given.

    Parameters
    ----------
    normalized : array_like, shape (n_time,)
        The normalized squared signal.
    signal : array_like, shape (n_time,)
        The filtered signal (for the trough).
    timestamps : array_like, shape (n_time,)
        Sample times in seconds.
    low_threshold, high_threshold : float
        In SD units.
    minimum_inter_ripple_interval_ms, maximum_duration_ms : float
        ``durations`` as stored, in milliseconds.
    frequency : float
        Sampling rate (Hz), as the source converts the interval to samples.
    minimum_duration_ms : float, optional
        Later versions' minimum; None applies none.

    Returns
    -------
    pandas.DataFrame
        `SOURCE_EVENT_COLUMNS`, one row per event: 0-based sample indices, times,
        the trough time as ``peak_time`` (the source's peak), the time of the
        maximum normalized power as ``max_power_time`` (the package's peak) and
        ``peak_normed_power``.
    """
    normalized = np.asarray(normalized, dtype=float)
    signal = np.asarray(signal, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    empty = pd.DataFrame({c: pd.Series(dtype=float) for c in SOURCE_EVENT_COLUMNS})
    thresholded = (normalized > low_threshold).astype(np.int8)
    rises = np.diff(thresholded)
    start = np.flatnonzero(rises > 0)
    stop = np.flatnonzero(rises < 0)
    if len(stop) == len(start) - 1:
        start = start[:-1]
    if len(stop) - 1 == len(start):
        stop = stop[1:]
    if len(start) == 0 or len(stop) == 0:
        return empty
    if start[0] > stop[0]:
        stop, start = stop[1:], start[:-1]
    if len(start) == 0:
        return empty

    gap_samples = minimum_inter_ripple_interval_ms / 1000 * frequency
    merged: list[list[int]] = []
    current = [int(start[0]), int(stop[0])]
    for next_start, next_stop in zip(start[1:], stop[1:], strict=True):
        if next_start - current[1] < gap_samples:
            current[1] = int(next_stop)
        else:
            merged.append(current)
            current = [int(next_start), int(next_stop)]
    merged.append(current)

    rows = []
    for first, last in merged:
        segment = normalized[first : last + 1]
        max_offset = int(np.argmax(segment))
        if segment[max_offset] > high_threshold:
            trough = first + int(np.argmin(signal[first : last + 1]))
            rows.append((first, last, trough, first + max_offset, segment[max_offset]))
    if not rows:
        return empty
    table = np.array(rows, dtype=float)
    first, last = table[:, 0].astype(int), table[:, 1].astype(int)
    duration = timestamps[last] - timestamps[first]
    keep = ~(duration > maximum_duration_ms / 1000)
    if minimum_duration_ms is not None:
        keep &= ~(duration < minimum_duration_ms / 1000)
    table = table[keep]
    index = table[:, :4].astype(np.int64)
    return pd.DataFrame(
        {
            "start_index": index[:, 0],
            "stop_index": index[:, 1],
            "trough_index": index[:, 2],
            "max_index": index[:, 3],
            "start_time": timestamps[index[:, 0]],
            "peak_time": timestamps[index[:, 2]],
            "end_time": timestamps[index[:, 1]],
            "max_power_time": timestamps[index[:, 3]],
            "peak_normed_power": table[:, 4],
        }
    )


# --- the package run ------------------------------------------------------------------


def package_options(parameters: Mapping[str, Any], code: SourceCode) -> dict[str, Any]:
    """``Zugaro_ripple_detector``'s settings from a file's stored ``detectorParams``.

    The thresholds and durations come from the file; ``minimum_duration`` is 0
    when the dated source has none (the detector accepts 0); the speed rule is
    off (``speed_threshold=np.inf``, the source has none); the normalization
    uses every sample (stored ``restrict`` empty); ``smoothing_window`` stays at
    its default, checked to be ``code.smoothing_samples`` samples at the stored
    rate.

    Parameters
    ----------
    parameters : mapping
        The stored ``detectorParams`` (``thresholds``, ``durations``,
        ``frequency``, ``restrict``).
    code : SourceCode
        The dated source.

    Returns
    -------
    dict
        Keyword arguments for ``Zugaro_ripple_detector`` (``sampling_frequency``
        included).

    Raises
    ------
    ValueError
        If ``restrict`` is not empty, or the default smoothing window is not the
        source's length at this rate.
    """
    from ripple_detection import Zugaro_ripple_detector

    if np.asarray(parameters.get("restrict", [])).size:
        msg = "the stored restrict is not empty; map it to normalization_mask first"
        raise ValueError(msg)
    frequency = float(parameters["frequency"])
    default_window = (
        inspect.signature(Zugaro_ripple_detector).parameters["smoothing_window"].default
    )
    samples = round(default_window * frequency)
    samples += samples % 2 == 0
    if samples != code.smoothing_samples:
        msg = (
            f"the default smoothing_window is {samples} samples at {frequency} Hz, "
            f"the source's {code.smoothing_samples}"
        )
        raise ValueError(msg)
    low, high = (float(v) for v in np.asarray(parameters["thresholds"]).ravel())
    interval_ms, maximum_ms = (float(v) for v in np.asarray(parameters["durations"]).ravel())
    minimum_ms = code.minimum_duration_ms
    return {
        "sampling_frequency": frequency,
        "speed_threshold": np.inf,
        "low_threshold": low,
        "high_threshold": high,
        "minimum_inter_ripple_interval": interval_ms / 1000,
        "minimum_duration": 0.0 if minimum_ms is None else minimum_ms / 1000,
        "maximum_duration": maximum_ms / 1000,
        "smoothing_window": float(default_window),
        "normalization_mask": None,
    }


def run_package(
    filtered: ArrayLike, timestamps: ArrayLike, options: Mapping[str, Any]
) -> pd.DataFrame:
    """Run ``Zugaro_ripple_detector`` on one source-filtered channel.

    Parameters
    ----------
    filtered : array_like, shape (n_time,)
        The source-filtered channel.
    timestamps : array_like, shape (n_time,)
        Sample times in seconds.
    options : mapping
        From `package_options`.

    Returns
    -------
    pandas.DataFrame
        The detector's events, with ``attrs`` naming the detector and its options.
    """
    from ripple_detection import Zugaro_ripple_detector

    kwargs = dict(options)
    frequency = kwargs.pop("sampling_frequency")
    filtered = np.asarray(filtered, dtype=float)
    events = Zugaro_ripple_detector(
        timestamps, filtered[:, None], np.zeros(len(filtered)), frequency, **kwargs
    )
    events.attrs = detector_attrs(options)
    return events


def detector_attrs(options: Mapping[str, Any]) -> dict[str, Any]:
    """Provenance for a detector run, in the shape ``save_events`` expects."""
    import ripple_detection as rd

    return {
        "method": DETECTOR,
        "ripple_detection_version": rd.__version__,
        "options": dict(options),
        "notes": "speed_threshold null means np.inf (no speed rule)",
    }


def save_detector_events(events: pd.DataFrame, path: str | Path) -> Path:
    """Write a detector's events with ``literature_methods.save_events``' two-file layout.

    Parameters
    ----------
    events : pandas.DataFrame
        A detector result whose ``attrs`` come from `detector_attrs`.
    path : str or pathlib.Path
        The CSV; the JSON sidecar goes beside it.

    Returns
    -------
    pathlib.Path
        The sidecar.
    """
    from ripple_detection.literature_methods import save_events

    return save_events(events, path)


# --- comparison -----------------------------------------------------------------------


COMPARISON_COLUMNS = [
    "reference",
    "detected",
    "minimum_iou",
    "n_reference",
    "n_detected",
    "n_matched",
    "recall",
    "precision",
    "f1",
    "median_iou",
    "iou_q25",
    "n_identical_bounds",
    "onset_error_q25_ms",
    "onset_error_median_ms",
    "onset_error_q75_ms",
    "offset_error_q25_ms",
    "offset_error_median_ms",
    "offset_error_q75_ms",
    "peak_definitions",
    "peak_error_q25_ms",
    "peak_error_median_ms",
    "peak_error_q75_ms",
    "n_split_reference",
    "n_merged_detected",
    "n_unmatched_reference",
    "n_unmatched_detected",
]


def _quartiles_ms(values: pd.Series) -> list[float]:
    if len(values) == 0 or values.isna().all():
        return [np.nan] * 3
    return [float(v) * 1000 for v in np.nanpercentile(values.to_numpy(float), [25, 50, 75])]


def comparison_row(
    reference_name: str,
    reference: pd.DataFrame,
    detected_name: str,
    detected: pd.DataFrame,
    minimum_iou: float,
    frequency: float,
    peak_definitions: str,
) -> dict[str, Any]:
    """One row of comparison.csv: ``match_events(reference, detected)`` summarized.

    Parameters
    ----------
    reference_name, detected_name : str
        Inventory names.
    reference, detected : pandas.DataFrame
        ``start_time``, ``end_time`` and ``peak_time`` in seconds.
    minimum_iou : float
        Passed to ``match_events``.
    frequency : float
        Sampling rate, for ``n_identical_bounds`` (both bounds within half a sample).
    peak_definitions : str
        What each side's ``peak_time`` is.

    Returns
    -------
    dict
        Keys `COMPARISON_COLUMNS`; errors in milliseconds, detected minus reference.
    """
    from ripple_detection import match_events

    columns = ["start_time", "end_time", "peak_time"]
    matching = match_events(reference[columns], detected[columns], minimum_iou=minimum_iou)
    pairs = matching.pairs
    half = 0.5 / frequency
    identical = (pairs.onset_error.abs() < half) & (pairs.offset_error.abs() < half)
    onset, offset, peak = (
        _quartiles_ms(pairs.onset_error),
        _quartiles_ms(pairs.offset_error),
        _quartiles_ms(pairs.peak_error),
    )
    return {
        "reference": reference_name,
        "detected": detected_name,
        "minimum_iou": minimum_iou,
        "n_reference": len(reference),
        "n_detected": len(detected),
        "n_matched": len(pairs),
        "recall": matching.recall,
        "precision": matching.precision,
        "f1": matching.f1,
        "median_iou": float(pairs.iou.median()) if len(pairs) else np.nan,
        "iou_q25": float(pairs.iou.quantile(0.25)) if len(pairs) else np.nan,
        "n_identical_bounds": int(identical.sum()),
        "onset_error_q25_ms": onset[0],
        "onset_error_median_ms": onset[1],
        "onset_error_q75_ms": onset[2],
        "offset_error_q25_ms": offset[0],
        "offset_error_median_ms": offset[1],
        "offset_error_q75_ms": offset[2],
        "peak_definitions": peak_definitions,
        "peak_error_q25_ms": peak[0],
        "peak_error_median_ms": peak[1],
        "peak_error_q75_ms": peak[2],
        "n_split_reference": len(matching.split_reference),
        "n_merged_detected": len(matching.merged_detected),
        "n_unmatched_reference": len(matching.unmatched_reference),
        "n_unmatched_detected": len(matching.unmatched_detected),
    }


def write_small(text_or_frame: str | pd.DataFrame, path: Path) -> Path:
    """Write a committed result, refusing anything of 1 MB or more.

    Parameters
    ----------
    text_or_frame : str or pandas.DataFrame
        Text, or a table written as CSV without its index.
    path : pathlib.Path
        Destination.

    Returns
    -------
    pathlib.Path
        ``path``.

    Raises
    ------
    ValueError
        If the content is 1 MB or larger; nothing is written.
    """
    text = (
        text_or_frame.to_csv(index=False)
        if isinstance(text_or_frame, pd.DataFrame)
        else text_or_frame
    )
    size = len(text.encode("utf-8"))
    if size >= MAX_RESULT_BYTES:
        msg = f"{path.name} would be {size} bytes, over the {MAX_RESULT_BYTES}-byte limit"
        raise ValueError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _json(value: Any) -> Any:
    """``value`` in strict JSON types (non-finite numbers become None)."""
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dump_json(value: Any) -> str:
    """Strict, indented JSON text."""
    return json.dumps(_json(value), indent=2, allow_nan=False, ensure_ascii=False) + "\n"


# --- the steps (I/O) ------------------------------------------------------------------


def session_dir(session: DatabankSession, cache: Path) -> Path:
    """``<cache>/sessions/<key>``, created."""
    folder = cache / "sessions" / session.key
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def results_dir(session: DatabankSession) -> Path:
    """``examples/reference_recordings/results/<key>``, created."""
    folder = RESULTS_DIR / session.key
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _manifest_record(cache: Path, **match: Any) -> dict[str, Any] | None:
    """The latest manifest record whose fields equal ``match``."""
    hits = [
        r for r in fetch.read_manifest(cache) if all(r.get(k) == v for k, v in match.items())
    ]
    return hits[-1] if hits else None


def _ensure_capture(
    session: DatabankSession, capture: WaybackCapture, cache: Path, head: bool = False
) -> dict[str, Any]:
    """Fetch a capture unless the cache holds it, verified, with its manifest record."""
    relative = session.cache_path(capture) + (".head" if head else "")
    target = cache / relative
    record = _manifest_record(cache, kind="wayback", path=relative)
    if target.exists() and record is not None:
        sha, _ = fetch.digests(target)
        expected = record["sha256"] if head else capture.sha256
        if sha == expected and (head or target.stat().st_size == capture.length):
            return record
    return fetch.fetch_wayback(
        session.url(capture),
        capture.timestamp,
        session.cache_path(capture),
        expected_sha256=None if head else capture.sha256,
        expected_source=None if head else "databank Wayback capture list (phase 7)",
        head_bytes=HEAD_BYTES if head else None,
        cache=cache,
    )


def step_fetch(session: DatabankSession, cache: Path) -> dict[str, Any]:
    """Fetch (or find in the cache) every input of the session; return their records."""
    records: dict[str, Any] = {}
    for capture in (session.events, *session.support):
        records[capture.suffix] = _ensure_capture(session, capture, cache)
    records[session.lfp_head.suffix + ".head"] = _ensure_capture(
        session, session.lfp_head, cache, head=True
    )
    dandi = _manifest_record(
        cache,
        kind="dandi-stream",
        asset_id=session.lfp_asset_id,
        dandiset_version=session.dandiset_version,
    )
    if dandi is None:
        dandi = fetch.dandi_asset(
            session.dandiset, session.dandiset_version, session.lfp_asset_id, cache=cache
        )
    records["dandi"] = dandi
    head_record = records[session.lfp_head.suffix + ".head"]
    declared = head_record.get("original_content_length")
    if declared != session.lfp_head.length:
        stop_session(
            session,
            cache,
            "fetch",
            Check(
                "lfp_head_length",
                False,
                {"declared": declared, "table": session.lfp_head.length},
                f"the .lfp head capture declares {declared} bytes, the table "
                f"{session.lfp_head.length}",
            ),
        )
    return records


def _load_mat(path: Path, name: str) -> Any:
    return scipy.io.loadmat(str(path), struct_as_record=False, squeeze_me=True)[name]


def channel_tag(session_mat: Any, tag: str) -> list[int] | None:
    """The 1-based channels of one of ``session.mat``'s channel tags, None when absent."""
    tags = getattr(session_mat, "channelTags", None)
    if tags is None or tag not in getattr(tags, "_fieldnames", ()):
        return None
    channels = np.atleast_1d(np.asarray(getattr(getattr(tags, tag), "channels", []))).ravel()
    return [int(c) for c in channels]


def _open_lfp(session: DatabankSession, cache: Path) -> tuple[Any, Any, Any]:
    """Open the DANDI LFP: (file, ElectricalSeries group, byte counter)."""
    import nwb

    record = _manifest_record(cache, kind="dandi-stream", asset_id=session.lfp_asset_id)
    if record is None:
        msg = "no DANDI record in the manifest; run the fetch step first"
        raise RuntimeError(msg)
    file, counter = nwb.open_nwb_counted(record["content_url"])
    return file, file[session.lfp_series], counter


def nwb_rate(series: Any) -> tuple[float, float] | None:
    """The series' rate and starting time, or None when it stores timestamps instead.

    Parameters
    ----------
    series : h5py.Group or mapping
        An ElectricalSeries with ``starting_time`` (its ``rate`` attribute) or
        ``timestamps``.
    """
    if "starting_time" not in series:
        return None
    start = series["starting_time"]
    return float(start.attrs["rate"]), float(start[()])


def step_verify(session: DatabankSession, cache: Path) -> dict[str, Any]:
    """Write inputs.json (cache and results); raise `InputCheckFailed` if a check fails."""
    folder = session_dir(session, cache)
    base = cache / "buzsaki" / session.key / session.basename
    released = readers.read_buzcode_events(f"{base}{session.events.suffix}")
    xml = readers.read_xml(f"{base}.xml")
    session_mat = _load_mat(Path(f"{base}.session.mat"), "session")
    head_bytes = Path(f"{base}.lfp.head").read_bytes()
    head = np.frombuffer(head_bytes, dtype="<i2")
    head = head[: len(head) // xml.n_channels * xml.n_channels].reshape(-1, xml.n_channels)

    file, series, _counter = _open_lfp(session, cache)
    try:
        data = series["data"]
        shape = tuple(int(n) for n in data.shape)
        clock = nwb_rate(series)
        if clock is None:
            stop_session(
                session,
                cache,
                "verify",
                Check(
                    "nwb_rate",
                    False,
                    {"series": session.lfp_series, "shape": list(shape)},
                    "the series has timestamps rather than a rate; they need a gap "
                    "check this script does not make",
                ),
            )
        rate, starting_time = clock
        dandi_head = np.asarray(data[: head.shape[0], :])
        conversion = float(data.attrs.get("conversion", np.nan))
        chunks = list(data.chunks) if data.chunks else None
    finally:
        file.close()
    np.save(folder / "dandi_head.npy", dandi_head)

    parameters = released.parameters
    stored_channel = released.channel
    if stored_channel is None:
        stop_session(
            session,
            cache,
            "verify",
            Check(
                "stored_channel",
                False,
                {"parameters": parameters},
                "the event file stores no detection channel",
            ),
        )
    n_samples = shape[0]
    boundaries = epoch_boundaries(session_mat)
    inner_edges = [e["start_time"] for e in boundaries[1:]]
    checks = [
        check_rates(xml.lfp_sampling_rate, rate, float(parameters["frequency"])),
        check_length(session.lfp_head.length, xml.n_channels, shape),
        check_head_match(head, dandi_head, stored_channel),
        check_channel_tag(stored_channel, channel_tag(session_mat, session.channel_tag)),
        check_events_in_recording(released.events, starting_time, n_samples, rate),
    ]
    n_channels_check = Check(
        "channel_count",
        shape[1] <= xml.n_channels,
        {
            "xml": xml.n_channels,
            "dandi_columns": shape[1],
            "dandi_keeps_every_channel": shape[1] == xml.n_channels,
            "note": "a DANDI copy may keep a subset; the head match finds the stored channel",
        },
        ""
        if shape[1] <= xml.n_channels
        else f"DANDI has {shape[1]} columns, the .xml {xml.n_channels} channels",
    )
    checks.insert(2, n_channels_check)
    head_check = next(c for c in checks if c.name == "head_match")
    column = head_check.detail["stored_channel_column"]

    inputs = {
        "session": session.key,
        "databank_path": session.databank_path,
        "released_events": {
            "file": session.url(session.events),
            "wayback_timestamp": session.events.timestamp,
            "sha256": session.events.sha256,
            "detector": released.detector,
            "parameters": parameters,
            "channel_stored": stored_channel,
            "stdev": released.stdev,
            "n_events": len(released.events),
            "n_noise_events": len(released.noise_events),
            "duration_ms_range": [
                float((released.events.end_time - released.events.start_time).min() * 1000),
                float((released.events.end_time - released.events.start_time).max() * 1000),
            ],
        },
        "recording": {
            "dandiset": session.dandiset,
            "dandiset_version": session.dandiset_version,
            "asset_id": session.lfp_asset_id,
            "series": session.lfp_series,
            "shape": list(shape),
            "chunks": chunks,
            "rate": rate,
            "starting_time": starting_time,
            "duration_s": n_samples / rate,
            "conversion_to_volts": conversion,
            "units": "int16 counts as in the .lfp (the head match compares them exactly); "
            "the source filtered these counts, so they are used unscaled",
            "lfp_head_capture": session.lfp_head.timestamp,
            "lfp_original_bytes": session.lfp_head.length,
            "detection_column": column,
        },
        "checks": [asdict(c) for c in checks],
        "gaps": {
            "nan": "none possible: the LFP is int16",
            "timestamps": "the series stores starting_time and rate, so no timestamp gap "
            "can be represented; the .lfp is one concatenated file",
            "epochs": boundaries,
            "events_spanning_an_epoch_boundary": events_spanning(released.events, inner_edges),
            "note": "the concatenation points are epoch boundaries, not gaps in samples: "
            "the source treated the session as one continuous signal, and so does this run",
        },
        "speed": "the source applies no speed rule; the package runs with speed_threshold=np.inf",
        "clock": "released times are seconds from the first LFP sample (bz_GetLFP timestamps "
        "from 0); NWB starting_time is 0, so the two share one clock",
        "source_code": asdict(session.source_code),
        "assumptions": assumptions(session, parameters),
    }
    text = dump_json(inputs)
    (folder / "inputs.json").write_text(text, encoding="utf-8")
    write_small(text, results_dir(session) / "inputs.json")
    failed = [c for c in checks if not c.passed]
    if failed:
        msg = "; ".join(f"{c.name}: {c.reason}" for c in failed)
        raise InputCheckFailed(msg)
    return inputs


def assumptions(session: DatabankSession, parameters: Mapping[str, Any]) -> list[str]:
    """The settings the source does not establish directly, and how they were mapped."""
    from ripple_detection import minimum_sample_count

    code = session.source_code
    interval_ms, maximum_ms = (float(v) for v in np.asarray(parameters["durations"]).ravel())
    frequency = float(parameters["frequency"])
    ceiling = int(minimum_sample_count(np.arange(2) / frequency, maximum_ms / 1000))
    return [
        (
            f"Code version: {code.repository} {code.path} at {code.commit[:7]} (see source_code's "
            "evidence); the file was written by that change before it was committed."
        ),
        (
            f"Filter: scipy.signal.{code.filter_kind}({code.filter_order}, "
            f"{code.filter_stopband_db}, passband/{code.filter_nyquist}, 'bandpass') in "
            "transfer-function form and filtfilt with MATLAB's padding (odd reflection of "
            "3*(n_coefficients-1) samples); assumed equal to MATLAB's cheby2/filtfilt to "
            "rounding (not run in MATLAB); the stage check tests it."
        ),
        (
            f"Smoothing: Filter0 with ones({code.smoothing_samples})/{code.smoothing_samples}, a "
            "centred moving average zero-padded at both ends; the package's default "
            f"smoothing_window is the same {code.smoothing_samples} samples at {frequency} Hz "
            "(np.convolve 'same')."
        ),
        (
            "Normalization: unity over all samples (restrict empty); MATLAB std uses N-1, the "
            "package's z-score N (relative difference 1/(2N), about 2e-8 here)."
        ),
        (
            f"Merge: the source merges while the next start minus the current stop is below "
            f"{interval_ms}/1000*frequency samples, with no cap; the package merges only "
            f"while the merged span stays under maximum_duration ({maximum_ms} ms): a rule "
            "difference, run as the detector implements it."
        ),
        (
            "Minimum duration: the dated source has none; the package runs with "
            "minimum_duration=0 (accepted; the shortest event possible is 2 samples)."
        ),
        (
            f"Maximum duration: the source drops (t_stop - t_start) > {maximum_ms / 1000} s; the "
            f"package keeps sample counts <= round-half-up({maximum_ms / 1000}*{frequency}) = "
            f"{ceiling}, the same events (stop - start <= {ceiling - 1} samples) unless "
            f"{maximum_ms / 1000}*{frequency} is a whole number."
        ),
        (
            "Edges: the source drops runs missing a crossing at the record's ends; the package "
            "keeps them flagged clipped_start/clipped_end (none expected away from the ends)."
        ),
        (
            "Peak: the source's peak is the first minimum of the filtered signal within the "
            "event; the package's is the maximum normalized power. Peak errors are reported "
            "with each side's definition named."
        ),
        "Speed: none in the source; the package gets zero speed with speed_threshold=np.inf.",
        (
            "Noise and EMG: the stored file has no noise events and the dated version no EMG "
            "rule; nothing is vetoed."
        ),
    ]


def stop_session(
    session: DatabankSession, cache: Path, step: str, failed: Check, **context: Any
) -> NoReturn:
    """Record a check that stops the session in inputs.json (cache and results), then raise.

    Parameters
    ----------
    session : DatabankSession
    cache : pathlib.Path
    step : str
        The step that stopped.
    failed : Check
        The failed check.
    **context
        Anything already known, written beside the check.

    Raises
    ------
    InputCheckFailed
        Always, with the check's reason.
    """
    record = {
        "session": session.key,
        "stopped_at": step,
        **context,
        "checks": [asdict(failed)],
    }
    text = dump_json(record)
    (session_dir(session, cache) / "inputs.json").write_text(text, encoding="utf-8")
    write_small(text, results_dir(session) / "inputs.json")
    msg = f"{failed.name}: {failed.reason}"
    raise InputCheckFailed(msg)


def _verified_inputs(session: DatabankSession, cache: Path) -> dict[str, Any]:
    path = session_dir(session, cache) / "inputs.json"
    if not path.exists():
        msg = "inputs.json is missing; run the verify step first"
        raise RuntimeError(msg)
    inputs: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    failed = [c["name"] for c in inputs["checks"] if not c["passed"]]
    if failed:
        msg = f"input checks failed ({failed}); the session stops here"
        raise InputCheckFailed(msg)
    return inputs


def _peak_rss() -> int:
    """Peak resident memory of this process in bytes (ru_maxrss is bytes on macOS)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def step_stream(
    session: DatabankSession, cache: Path, measure_minutes: float | None = None
) -> dict[str, Any]:
    """Read the detection column from DANDI, chunk-aligned; cache it (whole reads only).

    With ``measure_minutes`` only the first minutes are read, and the time,
    bytes and memory of the whole column are extrapolated from them.
    """
    inputs = _verified_inputs(session, cache)
    folder = session_dir(session, cache)
    column = int(inputs["recording"]["detection_column"])
    target = folder / f"lfp_column{column}.npy"
    if measure_minutes is None and target.exists():
        print(f"cached: {target}")
        return json.loads((folder / "stream.json").read_text(encoding="utf-8"))
    rss_before = _peak_rss()
    started = time.perf_counter()
    file, series, counter = _open_lfp(session, cache)
    try:
        data = series["data"]
        n_rows = int(data.shape[0])
        chunk_rows = int(data.chunks[0])
        rate = float(inputs["recording"]["rate"])
        stop = (
            n_rows
            if measure_minutes is None
            else min(n_rows, round(measure_minutes * 60 * rate))
        )
        out = np.empty(stop, dtype=data.dtype)
        block = 4 * chunk_rows
        for first in range(0, stop, block):
            last = min(stop, first + block)
            out[first:last] = data[first:last, column]
        n_chunks = int(np.ceil(stop / chunk_rows))
        stored = sum(data.id.get_chunk_info(i).size for i in range(n_chunks))
        stored_all = sum(
            data.id.get_chunk_info(i).size for i in range(data.id.get_num_chunks())
        )
    finally:
        file.close()
    seconds = time.perf_counter() - started
    stats = {
        "column": column,
        "rows_read": int(stop),
        "minutes_read": stop / rate / 60,
        "seconds": seconds,
        "bytes_read_through_h5py": int(counter.bytes_read),
        "n_reads": int(counter.n_reads),
        "chunk_bytes_stored": int(stored),
        "chunks_read": n_chunks,
        "peak_rss_bytes": _peak_rss(),
        "peak_rss_bytes_before": rss_before,
    }
    if measure_minutes is not None:
        # whole chunks are read, so time and bytes scale with the stored chunk bytes
        scale = stored_all / stored
        stats["extrapolated_whole_column"] = {
            "rows": n_rows,
            "scale": "stored bytes of all chunks over those of the chunks read",
            "seconds": seconds * scale,
            "bytes_read_through_h5py": counter.bytes_read * scale,
            "chunk_bytes_stored_exact": int(stored_all),
            "column_bytes_in_memory": n_rows * out.itemsize,
            "note": "the slice's own peak memory is not a bound for the whole read: remfile "
            "keeps up to 1 GB of fetched bytes in memory and grows its requests on long "
            "sequential reads (which also makes the whole read faster per byte)",
        }
        (folder / "stream_measure.json").write_text(dump_json(stats), encoding="utf-8")
    else:
        np.save(target, out)
        (folder / "stream.json").write_text(dump_json(stats), encoding="utf-8")
    print(dump_json(stats))
    return stats


def _load_column(
    session: DatabankSession, cache: Path
) -> tuple[NDArray[np.int16], dict[str, Any]]:
    inputs = _verified_inputs(session, cache)
    column = int(inputs["recording"]["detection_column"])
    path = session_dir(session, cache) / f"lfp_column{column}.npy"
    if not path.exists():
        msg = f"{path} is missing; run the stream step first"
        raise RuntimeError(msg)
    return np.load(path), inputs


def _released(session: DatabankSession, cache: Path) -> readers.BuzcodeRipples:
    base = cache / "buzsaki" / session.key / session.basename
    return readers.read_buzcode_events(f"{base}{session.events.suffix}")


def _source_trace(
    session: DatabankSession, cache: Path
) -> tuple[
    NDArray[np.int16], FloatArray, FloatArray, float, float, FloatArray, dict[str, Any]
]:
    """The raw column, the source-filtered signal, the normalized trace, its sd and mean, the timestamps."""
    raw, inputs = _load_column(session, cache)
    parameters = _released(session, cache).parameters
    code = session.source_code
    filtered = source_filter(raw, np.asarray(parameters["passband"], float).ravel(), code)
    normalized, sd, mean = normalized_squared_signal(filtered, code.smoothing_samples)
    rate = float(inputs["recording"]["rate"])
    timestamps = inputs["recording"]["starting_time"] + np.arange(len(raw)) / rate
    return raw, filtered, normalized, sd, mean, timestamps, inputs


def _sd_with_sos_form(raw: ArrayLike, passband: FloatArray, code: SourceCode) -> float:
    """The normalization's SD with the filter applied as second-order sections."""
    edges = passband / code.filter_nyquist
    if code.filter_kind == "cheby2":
        sos = scipy.signal.cheby2(
            code.filter_order, code.filter_stopband_db, edges, btype="bandpass", output="sos"
        )
    else:
        sos = scipy.signal.butter(code.filter_order, edges, btype="bandpass", output="sos")
    filtered = scipy.signal.sosfiltfilt(sos, np.asarray(raw, dtype=np.float64))
    return normalized_squared_signal(filtered, code.smoothing_samples)[1]


def step_stage(session: DatabankSession, cache: Path) -> dict[str, Any]:
    """The source's filter and normalization against the stored ``stdev``."""
    started = time.perf_counter()
    released = _released(session, cache)
    raw, _filtered, normalized, sd, mean, timestamps, inputs = _source_trace(session, cache)
    rate = float(inputs["recording"]["rate"])
    events = released.events
    first = np.round(sample_positions(events.start_time, timestamps[0], rate)).astype(int)
    last = np.round(sample_positions(events.end_time, timestamps[0], rate)).astype(int)
    recomputed = np.array(
        [normalized[a : b + 1].max() for a, b in zip(first, last, strict=True)]
    )
    stored_power = np.asarray(
        scipy.io.loadmat(
            str(
                cache / "buzsaki" / session.key / f"{session.basename}{session.events.suffix}"
            ),
            struct_as_record=False,
            squeeze_me=True,
        )["ripples"].peakNormedPower,
        dtype=float,
    )
    relative_power = (recomputed - stored_power) / stored_power
    stored_sd = float(released.stdev) if released.stdev is not None else np.nan
    sos_sd = _sd_with_sos_form(
        raw, np.asarray(released.parameters["passband"], float).ravel(), session.source_code
    )
    butter = SourceCode(
        **{**asdict(session.source_code), "filter_kind": "butter", "filter_order": 3}
    )
    butter_sd = normalized_squared_signal(
        source_filter(raw, np.asarray(released.parameters["passband"], float).ravel(), butter),
        session.source_code.smoothing_samples,
    )[1]
    stage = {
        "stored_stdev": stored_sd,
        "recomputed_stdev": sd,
        "relative_difference": (sd - stored_sd) / stored_sd,
        "mean_of_smoothed_power": mean,
        "population_sd_relative_to_sample_sd": float(np.sqrt((len(raw) - 1) / len(raw))),
        "sd_with_sos_form": {
            "what": "the same design as second-order sections (sosfiltfilt, scipy's default "
            "padding), a check that the transfer-function arithmetic is not what differs",
            "sd": sos_sd,
            "relative_difference_to_stored": (sos_sd - stored_sd) / stored_sd,
            "relative_difference_to_transfer_function": (sos_sd - sd) / sd,
        },
        "sd_with_butter_order_3": {
            "what": "buzsakilab/buzcode's filter (butter order 3, filtfilt) in place of the "
            "dated cheby2, everything else unchanged: what the code version is worth",
            "sd": butter_sd,
            "relative_difference_to_stored": (butter_sd - stored_sd) / stored_sd,
        },
        "n_samples": len(raw),
        "peak_normed_power_check": {
            "what": "max of the recomputed normalized trace within each released event's "
            "bounds against the stored peakNormedPower",
            "n_events": len(events),
            "relative_difference_median": float(np.median(relative_power)),
            "relative_difference_max_abs": float(np.max(np.abs(relative_power))),
            "n_within_1e-6": int(np.sum(np.abs(relative_power) < 1e-6)),
        },
        "seconds": time.perf_counter() - started,
        "peak_rss_bytes": _peak_rss(),
    }
    folder = session_dir(session, cache)
    (folder / "stage.json").write_text(dump_json(stage), encoding="utf-8")
    write_small(dump_json(stage), results_dir(session) / "stage_check.json")
    print(dump_json(stage))
    return stage


def step_detect(session: DatabankSession, cache: Path) -> dict[str, Any]:
    """Run the package with the source's settings and the transcription; save both."""
    started = time.perf_counter()
    released = _released(session, cache)
    parameters = released.parameters
    _raw, filtered, normalized, _sd, _mean, timestamps, inputs = _source_trace(session, cache)
    rate = float(inputs["recording"]["rate"])
    low, high = (float(v) for v in np.asarray(parameters["thresholds"]).ravel())
    interval_ms, maximum_ms = (float(v) for v in np.asarray(parameters["durations"]).ravel())
    t0 = time.perf_counter()
    transcription = source_segmentation(
        normalized,
        filtered,
        timestamps,
        low_threshold=low,
        high_threshold=high,
        minimum_inter_ripple_interval_ms=interval_ms,
        maximum_duration_ms=maximum_ms,
        frequency=rate,
        minimum_duration_ms=session.source_code.minimum_duration_ms,
    )
    transcription_seconds = time.perf_counter() - t0
    options = package_options(parameters, session.source_code)
    t0 = time.perf_counter()
    package = run_package(filtered, timestamps, options)
    package_seconds = time.perf_counter() - t0
    folder, results = session_dir(session, cache), results_dir(session)
    save_detector_events(package, folder / "package_events.csv")
    transcription.to_csv(folder / "transcription_events.csv", index=False)
    for name in ("package_events.csv", "package_events.json", "transcription_events.csv"):
        write_small((folder / name).read_text(encoding="utf-8"), results / name)
    summary = {
        "n_released": len(released.events),
        "n_transcription": len(transcription),
        "n_package": len(package),
        "n_package_clipped": int((package.clipped_start | package.clipped_end).sum()),
        "options": options,
        "seconds_total": time.perf_counter() - started,
        "seconds_transcription_segmentation": transcription_seconds,
        "seconds_package_detector": package_seconds,
        "peak_rss_bytes": _peak_rss(),
    }
    (folder / "detect.json").write_text(dump_json(summary), encoding="utf-8")
    print(dump_json(summary))
    return summary


def _inventories(session: DatabankSession, cache: Path) -> dict[str, pd.DataFrame]:
    from ripple_detection.literature_methods import load_events

    folder = session_dir(session, cache)
    released = _released(session, cache).events
    package = load_events(folder / "package_events.csv")
    transcription = pd.read_csv(
        folder / "transcription_events.csv", float_precision="round_trip"
    )
    return {"released": released, "transcription": transcription, "package": package}


PAIRS = (
    ("released", "package", "trough of the filtered signal vs max normalized power"),
    ("released", "transcription", "trough vs trough"),
    ("transcription", "package", "max normalized power vs max normalized power"),
)


def step_compare(session: DatabankSession, cache: Path) -> pd.DataFrame:
    """comparison.csv: every pair of inventories at every minimum IoU."""
    inventories = _inventories(session, cache)
    rate = float(_verified_inputs(session, cache)["recording"]["rate"])
    by_max_power = inventories["transcription"].assign(
        peak_time=inventories["transcription"]["max_power_time"]
    )
    rows = [
        comparison_row(
            reference_name,
            by_max_power if reference_name == "transcription" else inventories[reference_name],
            detected_name,
            inventories[detected_name],
            level,
            rate,
            peaks,
        )
        for reference_name, detected_name, peaks in PAIRS
        for level in MATCH_IOU_LEVELS
    ]
    table = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    write_small(table, results_dir(session) / "comparison.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(table.T)
    return table


def _pick(indices: ArrayLike, n: int) -> list[int]:
    """Up to ``n`` entries spread evenly over ``indices`` (deterministic)."""
    indices = np.asarray(indices, dtype=int)
    if len(indices) <= n:
        return indices.tolist()
    return indices[np.linspace(0, len(indices) - 1, n).round().astype(int)].tolist()


def difference_classes(
    reference: pd.DataFrame, detected: pd.DataFrame, poor_iou: float = 0.5
) -> dict[str, list[int]]:
    """Row positions of the unmatched events of each side and the poorly aligned pairs.

    Parameters
    ----------
    reference, detected : pandas.DataFrame
        ``start_time`` and ``end_time`` in seconds.
    poor_iou : float
        Pairs matched at any overlap with IoU below this are poorly aligned.

    Returns
    -------
    dict
        ``unmatched_reference``, ``unmatched_detected`` and ``poorly_aligned``
        (reference row positions), each ascending.
    """
    from ripple_detection import match_events

    columns = ["start_time", "end_time"]
    matching = match_events(reference[columns], detected[columns])
    poor = matching.pairs.loc[matching.pairs.iou < poor_iou, "reference_index"]
    return {
        "unmatched_reference": matching.unmatched_reference.tolist(),
        "unmatched_detected": matching.unmatched_detected.tolist(),
        "poorly_aligned": sorted(int(i) for i in poor),
    }


def runtime_summary(session: DatabankSession, cache: Path) -> dict[str, Any]:
    """Time and memory of the stream (slice and whole column), stage and detect steps."""
    folder = session_dir(session, cache)
    summary: dict[str, Any] = {"platform": sys.platform}
    for name in ("stream_measure", "stream", "stage", "detect"):
        path = folder / f"{name}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        keep = (
            "rows_read",
            "minutes_read",
            "seconds",
            "seconds_total",
            "seconds_transcription_segmentation",
            "seconds_package_detector",
            "bytes_read_through_h5py",
            "chunk_bytes_stored",
            "chunks_read",
            "peak_rss_bytes",
            "extrapolated_whole_column",
        )
        summary[name] = {k: record[k] for k in keep if k in record}
    return summary


def containing_rows(inner: pd.DataFrame, outer: pd.DataFrame) -> NDArray[np.int64]:
    """For each inner event, the row position of an outer event whose closed bounds hold it.

    Parameters
    ----------
    inner, outer : pandas.DataFrame
        ``start_time`` and ``end_time`` in seconds; ``outer`` sorted and disjoint.

    Returns
    -------
    ndarray of int, shape (n_inner,)
        The outer row position, or -1 where none holds the event.
    """
    starts = outer["start_time"].to_numpy(float)
    ends = outer["end_time"].to_numpy(float)
    candidate = np.searchsorted(starts, inner["start_time"].to_numpy(float), side="right") - 1
    found = candidate >= 0
    held = np.zeros(len(inner), dtype=bool)
    held[found] = inner["end_time"].to_numpy(float)[found] <= ends[candidate[found]]
    return np.where(held, candidate, -1).astype(np.int64)


def merge_cap_diagnostic(
    normalized: ArrayLike,
    filtered: ArrayLike,
    timestamps: ArrayLike,
    parameters: Mapping[str, Any],
    code: SourceCode,
    package_only: pd.DataFrame,
) -> dict[str, Any]:
    """Test whether the merge cap is what separates the package from the source.

    Parameters
    ----------
    normalized, filtered, timestamps : array_like, shape (n_time,)
        The source's normalized trace, filtered signal and sample times.
    parameters : mapping
        The stored ``detectorParams`` (``thresholds``, ``durations``, ``frequency``,
        ``restrict``).
    code : SourceCode
        The dated source.
    package_only : pandas.DataFrame
        The package events the source lacks (``start_time``, ``end_time``).

    Returns
    -------
    dict
        ``too_long``: the transcription's events before the duration ceiling that
        exceed it (merged with no cap, then dropped by the source); ``holder``:
        for each package-only event, the ``too_long`` row holding it, or -1;
        ``uncapped``: the package with ``maximum_duration=None`` (no merge cap);
        ``ceiling``: the package's sample count for the maximum duration;
        ``uncapped_then_ceiling``: ``uncapped`` with at most ``ceiling`` samples.
    """
    from ripple_detection import minimum_sample_count

    low, high = (float(v) for v in np.asarray(parameters["thresholds"]).ravel())
    interval_ms, maximum_ms = (float(v) for v in np.asarray(parameters["durations"]).ravel())
    before_ceiling = source_segmentation(
        normalized,
        filtered,
        timestamps,
        low_threshold=low,
        high_threshold=high,
        minimum_inter_ripple_interval_ms=interval_ms,
        maximum_duration_ms=np.inf,
        frequency=float(parameters["frequency"]),
        minimum_duration_ms=code.minimum_duration_ms,
    )
    too_long = before_ceiling[
        (before_ceiling.end_time - before_ceiling.start_time) > maximum_ms / 1000
    ].reset_index(drop=True)
    options = package_options(parameters, code)
    uncapped = run_package(filtered, timestamps, {**options, "maximum_duration": None})
    ceiling = int(minimum_sample_count(np.asarray(timestamps), maximum_ms / 1000))
    return {
        "too_long": too_long,
        "holder": containing_rows(package_only, too_long),
        "uncapped": uncapped,
        "ceiling": ceiling,
        "uncapped_then_ceiling": uncapped[uncapped["n_samples"] <= ceiling],
    }


def step_explain(session: DatabankSession, cache: Path, per_class: int = 4) -> dict[str, Any]:
    """Attribute the differences to a stage (attribution.json) and plot a few of each class.

    The source merges with no cap and then drops events over the maximum; the
    package caps the merge. Two diagnostics test whether that rule is the whole
    difference: the transcription without the duration ceiling (the events the
    source merged and then dropped, and whether each package-only event lies
    inside one), and the package without a ceiling (so its merge has no cap)
    with the ceiling applied afterwards, compared with the released events.
    """
    from ripple_detection import match_events

    inventories = _inventories(session, cache)
    raw, filtered, normalized, _sd, _mean, timestamps, inputs = _source_trace(session, cache)
    parameters = _released(session, cache).parameters
    low, high = (float(v) for v in np.asarray(parameters["thresholds"]).ravel())
    maximum_ms = float(np.asarray(parameters["durations"]).ravel()[1])
    rate = float(inputs["recording"]["rate"])
    released, package = inventories["released"], inventories["package"]
    classes = difference_classes(released, package)
    extras = package.iloc[classes["unmatched_detected"]]
    diagnostic = merge_cap_diagnostic(
        normalized, filtered, timestamps, parameters, session.source_code, extras
    )
    too_long, holder = diagnostic["too_long"], diagnostic["holder"]
    uncapped, uncapped_then_ceiling = (
        diagnostic["uncapped"],
        diagnostic["uncapped_then_ceiling"],
    )
    ceiling = diagnostic["ceiling"]
    per_long = np.bincount(holder[holder >= 0], minlength=len(too_long))

    columns = ["start_time", "end_time", "peak_time"]
    check = comparison_row(
        "released",
        released,
        "package_uncapped_then_ceiling",
        uncapped_then_ceiling[columns],
        0.0,
        rate,
        "trough vs max normalized power",
    )
    extra_durations_ms = (extras.end_time - extras.start_time).to_numpy() * 1000
    attribution = {
        "differences": {k: len(v) for k, v in classes.items()},
        "source_events_over_the_ceiling": {
            "what": "the transcription with no duration ceiling: events the source merged "
            f"and then dropped for exceeding {maximum_ms} ms",
            "n": len(too_long),
            "duration_ms_range": [
                float((too_long.end_time - too_long.start_time).min() * 1000),
                float((too_long.end_time - too_long.start_time).max() * 1000),
            ]
            if len(too_long)
            else None,
            "n_holding_package_only_events": int(np.sum(per_long > 0)),
            "package_only_events_per_dropped_event": {
                str(k): int(v)
                for k, v in zip(*np.unique(per_long, return_counts=True), strict=True)
            },
        },
        "package_only_events": {
            "n": len(extras),
            "n_inside_a_dropped_source_event": int(np.sum(holder >= 0)),
            "rows_not_inside_one": np.flatnonzero(holder < 0).tolist(),
            "duration_ms_range": [
                float(extra_durations_ms.min()),
                float(extra_durations_ms.max()),
            ]
            if len(extras)
            else None,
            "peak_normalized_power_min": float(extras.max_zscore.min())
            if "max_zscore" in extras and len(extras)
            else None,
        },
        "package_without_the_merge_cap": {
            "what": "Zugaro_ripple_detector with maximum_duration=None (no merge cap), then "
            f"events of more than {ceiling} samples dropped, against the released events",
            "n_before_ceiling": len(uncapped),
            "n": len(uncapped_then_ceiling),
            "comparison": check,
        },
    }
    pairs = match_events(released[columns], package[columns]).pairs
    peak_ms = pairs.peak_error.to_numpy() * 1000
    attribution["peak_definition"] = {
        "what": "package peak (max normalized power) minus released peak (trough of the "
        "filtered signal) over the matched pairs, ms",
        "n_pairs": len(peak_ms),
        "n_within_half_a_sample": int(np.sum(np.abs(peak_ms) < 500 / rate)),
        "quantiles_0_5_25_50_75_95_100": np.percentile(peak_ms, [0, 5, 25, 50, 75, 95, 100]),
    }
    saturated = (raw == np.iinfo(np.int16).min) | (raw == np.iinfo(np.int16).max)
    first = np.round(sample_positions(released.start_time, timestamps[0], rate)).astype(int)
    last = np.round(sample_positions(released.end_time, timestamps[0], rate)).astype(int)
    attribution["saturation"] = {
        "what": "samples of the detection channel at the int16 limits",
        "n_samples": int(saturated.sum()),
        "fraction": float(saturated.mean()),
        "n_released_events_with_one": int(
            sum(saturated[a : b + 1].any() for a, b in zip(first, last, strict=True))
        ),
    }
    write_small(dump_json(attribution), results_dir(session) / "attribution.json")
    write_small(
        dump_json(runtime_summary(session, cache)), results_dir(session) / "runtime.json"
    )
    print(dump_json(attribution))
    _plot_classes(
        session,
        {**inventories, "source before ceiling": too_long},
        classes,
        (raw, filtered, normalized, timestamps),
        (low, high),
        per_class,
    )
    return attribution


def _plot_classes(
    session: DatabankSession,
    inventories: Mapping[str, pd.DataFrame],
    classes: Mapping[str, list[int]],
    traces: tuple[NDArray[np.int16], FloatArray, FloatArray, FloatArray],
    thresholds: tuple[float, float],
    per_class: int,
) -> list[Path]:
    """A small PNG per non-empty class of released vs package differences."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    raw, filtered, normalized, timestamps = traces
    rate = 1 / float(np.median(np.diff(timestamps[:1000])))
    lanes = list(inventories)
    colors = ["#0072B2", "#009E73", "#D55E00", "#999999"]
    written = []
    for label, rows in classes.items():
        frame = inventories["package" if label == "unmatched_detected" else "released"]
        chosen = _pick(rows, per_class)
        if not chosen:
            continue
        fig, axes = plt.subplots(
            4,
            len(chosen),
            figsize=(3.0 * len(chosen), 6.0),
            squeeze=False,
            sharex="col",
            gridspec_kw={"height_ratios": [1, 1, 1, 0.6]},
        )
        for column, row in enumerate(chosen):
            event = frame.iloc[row]
            window = max(0.25, float(event.end_time - event.start_time))
            centre = 0.5 * (event.start_time + event.end_time)
            a = max(0, int((centre - window - timestamps[0]) * rate))
            b = min(len(raw), int((centre + window - timestamps[0]) * rate))
            t = timestamps[a:b]
            relative = t - centre
            axes[0, column].plot(relative, raw[a:b], lw=0.5, color="0.3")
            axes[1, column].plot(relative, filtered[a:b], lw=0.5, color="0.3")
            axes[2, column].plot(relative, normalized[a:b], lw=0.6, color="0.2")
            axes[2, column].axhline(thresholds[0], ls="--", lw=0.6, color="0.5")
            axes[2, column].axhline(thresholds[1], ls="-", lw=0.6, color="0.5")
            for lane, name in enumerate(lanes):
                inventory = inventories[name]
                near = inventory[
                    (inventory.end_time >= t[0]) & (inventory.start_time <= t[-1])
                ]
                for _, other in near.iterrows():
                    axes[3, column].plot(
                        [other.start_time - centre, other.end_time - centre],
                        [lane, lane],
                        lw=4,
                        color=colors[lane],
                        solid_capstyle="butt",
                    )
            axes[3, column].set_ylim(-0.7, len(lanes) - 0.3)
            axes[3, column].set_yticks(range(len(lanes)))
            axes[3, column].set_yticklabels(lanes if column == 0 else [], fontsize=6)
            side = "package" if label == "unmatched_detected" else "released"
            axes[0, column].set_title(f"{side} row {row}, {centre:.3f} s", fontsize=7)
            axes[3, column].set_xlabel("s from the event's centre", fontsize=6)
            for ax in axes[:, column]:
                ax.tick_params(labelsize=6)
        axes[0, 0].set_ylabel("raw (counts)", fontsize=7)
        axes[1, 0].set_ylabel("filtered", fontsize=7)
        axes[2, 0].set_ylabel("normalized (SD)", fontsize=7)
        fig.suptitle(f"{session.key}: released vs package, {label}", fontsize=8)
        fig.tight_layout()
        path = results_dir(session) / f"released_vs_package_{label}.png"
        fig.savefig(path, dpi=80)
        plt.close(fig)
        if path.stat().st_size >= MAX_RESULT_BYTES:
            path.unlink()
            msg = f"{path.name} is over the 1 MB limit"
            raise ValueError(msg)
        written.append(path)
    print("\n".join(str(p) for p in written))
    return written


def main(argv: list[str] | None = None) -> None:
    """Run the requested steps for one session (see the module docstring)."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("session", choices=sorted(SESSIONS))
    parser.add_argument("--steps", default=",".join(STEPS), help="comma-separated, in order")
    parser.add_argument("--measure-minutes", type=float, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    args = parser.parse_args(argv)
    session = SESSIONS[args.session]
    cache = fetch.cache_dir() if args.cache is None else args.cache
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        parser.error(f"unknown steps {unknown}; choose from {STEPS}")
    runners = {
        "fetch": lambda: step_fetch(session, cache),
        "verify": lambda: step_verify(session, cache),
        "stream": lambda: step_stream(session, cache, args.measure_minutes),
        "stage": lambda: step_stage(session, cache),
        "detect": lambda: step_detect(session, cache),
        "compare": lambda: step_compare(session, cache),
        "explain": lambda: step_explain(session, cache),
    }
    for step in steps:
        started = time.perf_counter()
        runners[step]()
        print(
            f"[{step}] {time.perf_counter() - started:.1f} s, peak RSS {_peak_rss() / 1e9:.2f} GB"
        )


if __name__ == "__main__":
    main()

"""Readers for NeuroScope and buzcode files, using only NumPy, pandas and ``scipy.io``.

Formats
-------
``.evt``
    NeuroScope event text, ``<milliseconds><TAB><label>`` per line.
``.xml``
    NeuroScope/NDManager parameter file (channel count, rates, anatomical groups).
``.lfp`` / ``.dat``
    Interleaved little-endian int16, ``n_channels`` values per sample.
``*.ripples.events.mat``
    buzcode/CellExplorer ripple events, MATLAB v5-v7 (not v7.3).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.io
from numpy.typing import ArrayLike, NDArray

_INTERVAL_COLUMNS = ["start_time", "peak_time", "end_time"]


def read_evt(path: str | Path) -> pd.DataFrame:
    """Read a NeuroScope ``.evt`` text file.

    Parameters
    ----------
    path : str or pathlib.Path
        File with ``<milliseconds><whitespace><label>`` per line; blank lines are skipped.

    Returns
    -------
    pandas.DataFrame
        Columns ``time`` (float64 seconds, the milliseconds divided by 1000) and
        ``label`` (str, the rest of the line). Rows keep the file's order.

    Raises
    ------
    ValueError
        If a line has no label or a time that is not a number.
    """
    times: list[float] = []
    labels: list[str] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.strip().split(maxsplit=1)
        try:
            if len(parts) != 2:
                raise ValueError
            milliseconds = float(parts[0])
        except ValueError:
            msg = f"{path}, line {number}: expected '<ms> <label>', got {line!r}"
            raise ValueError(msg) from None
        times.append(milliseconds / 1000.0)
        labels.append(parts[1].strip())
    return pd.DataFrame({"time": np.asarray(times, dtype=np.float64), "label": labels})


def evt_intervals(
    events: pd.DataFrame, start: str = "start", peak: str = "peak", stop: str = "stop"
) -> pd.DataFrame:
    """Pair labelled rows such as ``Ripple start 23`` / ``Ripple peak 23`` / ``Ripple stop 23``.

    A label is split into words; the word equal to ``start``, ``peak`` or ``stop``
    marks the row's role, and the last word is kept as the trailing label (for
    example the channel). Rows must come as start, [peak,] stop, repeated, with the
    same trailing label in a triple.

    Parameters
    ----------
    events : pandas.DataFrame
        As returned by `read_evt` (``time`` in seconds, ``label``).
    start, peak, stop : str
        The role words.

    Returns
    -------
    pandas.DataFrame
        One row per event: ``start_time``, ``peak_time`` (NaN where the file has no
        peak rows), ``end_time`` in seconds and ``label`` (the trailing token).

    Raises
    ------
    ValueError
        If the sequence is not start, [peak,] stop in order, the counts of the roles
        differ, a peak is outside its event, or the intervals are unsorted or overlap
        (closed intervals touching at one time are allowed).
    """
    rows: list[tuple[float, float, float, str]] = []
    pending: dict[str, Any] | None = None
    for position, (time, label) in enumerate(
        zip(events["time"], events["label"], strict=False)
    ):
        words = str(label).split()
        roles = [w for w in words if w in (start, peak, stop)]
        if len(roles) != 1:
            msg = f"row {position}: label {label!r} does not hold exactly one of {(start, peak, stop)}"
            raise ValueError(msg)
        role, tail = roles[0], words[-1]
        if role == start:
            if pending is not None:
                msg = f"row {position}: {start!r} while the event at row {pending['row']} is open"
                raise ValueError(msg)
            pending = {"start": float(time), "peak": np.nan, "tail": tail, "row": position}
            continue
        if pending is None:
            msg = f"row {position}: {role!r} without a preceding {start!r}"
            raise ValueError(msg)
        if tail != pending["tail"]:
            msg = (
                f"row {position}: label {tail!r} differs from the event's {pending['tail']!r}"
            )
            raise ValueError(msg)
        if role == peak:
            if not np.isnan(pending["peak"]):
                msg = f"row {position}: a second {peak!r} in one event"
                raise ValueError(msg)
            pending["peak"] = float(time)
            continue
        end = float(time)
        if end < pending["start"] or not (
            np.isnan(pending["peak"]) or pending["start"] <= pending["peak"] <= end
        ):
            msg = (
                f"row {position}: times of the event at row {pending['row']} are out of order"
            )
            raise ValueError(msg)
        rows.append((pending["start"], pending["peak"], end, pending["tail"]))
        pending = None
    if pending is not None:
        msg = f"row {pending['row']}: {start!r} without a {stop!r}"
        raise ValueError(msg)
    frame = pd.DataFrame(rows, columns=[*_INTERVAL_COLUMNS, "label"])
    frame[_INTERVAL_COLUMNS] = frame[_INTERVAL_COLUMNS].astype(np.float64)
    starts, ends = frame["start_time"].to_numpy(), frame["end_time"].to_numpy()
    if np.any(starts[1:] < ends[:-1]):
        index = int(np.flatnonzero(starts[1:] < ends[:-1])[0]) + 1
        msg = f"event {index} starts before event {index - 1} ends (unsorted or overlapping)"
        raise ValueError(msg)
    return frame


@dataclass(frozen=True)
class NeuroScopeParameters:
    """Contents of a NeuroScope/NDManager ``.xml``.

    Attributes
    ----------
    n_channels : int
        Channels in the raw ``.dat``/``.lfp`` files.
    sampling_rate : float
        Raw sampling rate (Hz).
    lfp_sampling_rate : float or None
        Rate of the ``.lfp`` file (Hz), None when the file has no such field.
    groups : list of list of int
        Anatomical channel groups, channel indices 0-based as stored.
    skip : dict of int to bool
        Per channel in the groups, the ``skip`` flag; empty when no channel has one.
    """

    n_channels: int
    sampling_rate: float
    lfp_sampling_rate: float | None
    groups: list[list[int]]
    skip: dict[int, bool]


def read_xml(path: str | Path) -> NeuroScopeParameters:
    """Read a NeuroScope/NDManager parameter file.

    Parameters
    ----------
    path : str or pathlib.Path
        The ``.xml`` file.

    Returns
    -------
    NeuroScopeParameters
        Channel count, rates, anatomical groups and ``skip`` flags.

    Raises
    ------
    ValueError
        If ``nChannels`` or ``samplingRate`` is missing.
    """
    root = ET.parse(path).getroot()

    def number(tag: str) -> float | None:
        element = root.find(f".//{tag}")
        return (
            None
            if element is None or not (element.text or "").strip()
            else float(element.text or "")
        )

    n_channels, sampling_rate = number("nChannels"), number("samplingRate")
    if n_channels is None or sampling_rate is None:
        msg = f"{path}: nChannels or samplingRate is missing"
        raise ValueError(msg)
    groups: list[list[int]] = []
    skip: dict[int, bool] = {}
    for group in root.findall(".//anatomicalDescription/channelGroups/group"):
        members = []
        for channel in group.findall("channel"):
            index = int((channel.text or "").strip())
            members.append(index)
            flag = channel.get("skip")
            if flag is not None:
                skip[index] = flag.strip() not in ("0", "false", "False")
        groups.append(members)
    return NeuroScopeParameters(
        int(n_channels), sampling_rate, number("lfpSamplingRate"), groups, skip
    )


def binary_n_samples(n_bytes: int, n_channels: int) -> int:
    """Samples in an interleaved int16 file.

    Parameters
    ----------
    n_bytes : int
        File size in bytes.
    n_channels : int
        Channels per sample.

    Returns
    -------
    int
        ``n_bytes / (2 * n_channels)``.

    Raises
    ------
    ValueError
        If the size is not an exact multiple of ``2 * n_channels``.
    """
    if n_channels < 1:
        msg = "n_channels must be positive"
        raise ValueError(msg)
    frame = 2 * n_channels
    if n_bytes % frame:
        msg = f"{n_bytes} bytes is not a multiple of {frame} (2 bytes x {n_channels} channels)"
        raise ValueError(msg)
    return n_bytes // frame


def read_binary_channels(
    path: str | Path,
    n_channels: int,
    channels: ArrayLike,
    start: int = 0,
    stop: int | None = None,
) -> NDArray[np.int16]:
    """Read selected channels and samples of an interleaved int16 ``.lfp``/``.dat``.

    Parameters
    ----------
    path : str or pathlib.Path
        The binary file.
    n_channels : int
        Channels per sample in the file.
    channels : array_like of int, shape (n_selected,)
        0-based channel indices, returned in the given order.
    start, stop : int
        Sample range ``[start, stop)``; ``stop=None`` reads to the end.

    Returns
    -------
    numpy.ndarray, shape (n_time, n_selected), int16
        A copy, not a memory map.

    Raises
    ------
    ValueError
        If the file size is not a whole number of samples, a channel is out of range
        or the sample range is not inside the file.
    """
    n_samples = binary_n_samples(Path(path).stat().st_size, n_channels)
    selected = np.atleast_1d(np.asarray(channels, dtype=np.int64))
    if selected.size == 0 or selected.min() < 0 or selected.max() >= n_channels:
        msg = f"channels must lie in [0, {n_channels}); got {selected.tolist()}"
        raise ValueError(msg)
    stop = n_samples if stop is None else stop
    if not 0 <= start <= stop <= n_samples:
        msg = f"sample range [{start}, {stop}) is not inside the file's {n_samples} samples"
        raise ValueError(msg)
    data = np.memmap(path, dtype="<i2", mode="r", shape=(n_samples, n_channels))
    return np.array(data[start:stop][:, selected], dtype=np.int16)


@dataclass(frozen=True)
class BuzcodeRipples:
    """A buzcode/CellExplorer ripple event file.

    Attributes
    ----------
    events : pandas.DataFrame
        ``start_time``, ``peak_time``, ``end_time`` (float64 seconds), one row per ripple.
    noise_events : pandas.DataFrame
        The same columns for the detector's noise events; empty if it saved none.
    detector : str
        Detector name as saved (for example ``bz_FindRipples``).
    parameters : dict
        The saved detector parameters as plain Python types (lists, floats, strings).
    channel : int or None
        The detection channel exactly as stored. buzcode stores it 0-based when the
        detector was run from a basepath; this is not converted.
    channel_one_based : int or None
        ``detectionchannel1`` when the newer layout saves it, else None.
    stdev : float or None
        The saved standard deviation of the detection trace, None when absent.
    """

    events: pd.DataFrame
    noise_events: pd.DataFrame
    detector: str
    parameters: dict[str, Any]
    channel: int | None
    channel_one_based: int | None
    stdev: float | None


def _plain(value: Any) -> Any:
    """Convert a loadmat value (arrays, structs, strings) to plain Python types."""
    if hasattr(value, "_fieldnames"):
        return {name: _plain(getattr(value, name)) for name in value._fieldnames}
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [_plain(v) for v in value.ravel()]
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _field(struct: Any, *names: str) -> Any:
    """The first of ``names`` that the struct has, else None."""
    for name in names:
        if name in getattr(struct, "_fieldnames", ()):
            return getattr(struct, name)
    return None


def _scalar(value: Any) -> float | None:
    array = np.asarray(value, dtype=np.float64).ravel() if value is not None else np.empty(0)
    return float(array[0]) if array.size else None


def _event_frame(times: Any, peaks: Any, label: str, path: Path) -> pd.DataFrame:
    array = np.asarray(times, dtype=np.float64)
    if array.size == 0:
        return pd.DataFrame({c: np.empty(0, dtype=np.float64) for c in _INTERVAL_COLUMNS})
    array = array.reshape(-1, 2) if array.ndim != 2 else array
    if array.shape[1] != 2:
        msg = f"{path}: {label} has shape {array.shape}, expected (n_events, 2)"
        raise ValueError(msg)
    peak = np.atleast_1d(np.asarray(peaks, dtype=np.float64)).ravel()
    if peak.shape[0] != array.shape[0]:
        msg = f"{path}: {label} has {array.shape[0]} intervals but {peak.shape[0]} peaks"
        raise ValueError(msg)
    return pd.DataFrame(
        {"start_time": array[:, 0], "peak_time": peak, "end_time": array[:, 1]}
    )


def read_buzcode_events(path: str | Path) -> BuzcodeRipples:
    """Read a buzcode/CellExplorer ``*.ripples.events.mat`` (MATLAB v5-v7).

    Two layouts are read. The old ``bz_FindRipples`` struct (2017-2020) holds
    ``times`` (n x 2), ``peaks``, ``stdev``, ``noise``, ``detectorName`` and
    ``detectorParams``. The newer one (2019+) holds ``timestamps`` (n x 2),
    ``peaks``, ``stdev``, optional ``noise`` and ``detectorinfo`` (``detectorname``,
    ``detectionparms``, ``detectionchannel``, optionally ``detectionchannel1``).

    Parameters
    ----------
    path : str or pathlib.Path
        The ``.mat`` file.

    Returns
    -------
    BuzcodeRipples
        Events in seconds, noise events, detector name, parameters, channel and stdev.

    Raises
    ------
    ValueError
        If the file is MATLAB v7.3 (HDF5; needs h5py), has no ``ripples`` struct, or
        the struct lacks its event times or peaks.
    """
    path = Path(path)
    with path.open("rb") as f:
        header = f.read(32)
    if header.startswith((b"MATLAB 7.3 MAT-file", b"\x89HDF\r\n\x1a\n")):
        msg = f"{path} is a MATLAB v7.3 (HDF5) file; reading it needs h5py"
        raise ValueError(msg)
    try:
        content = scipy.io.loadmat(str(path), struct_as_record=False, squeeze_me=True)
    except NotImplementedError as error:
        msg = f"{path} is a MATLAB v7.3 (HDF5) file; reading it needs h5py"
        raise ValueError(msg) from error
    ripples = content.get("ripples")
    if not hasattr(ripples, "_fieldnames"):
        msg = f"{path} has no 'ripples' struct"
        raise ValueError(msg)
    times = _field(ripples, "times", "timestamps")
    peaks = _field(ripples, "peaks")
    if times is None or peaks is None:
        msg = f"{path}: 'ripples' lacks its event times ('times' or 'timestamps') or 'peaks'"
        raise ValueError(msg)
    events = _event_frame(times, peaks, "ripples.times", path)

    noise = _field(ripples, "noise")
    if hasattr(noise, "_fieldnames"):
        noise_times = _field(noise, "times", "timestamps")
        noise_peaks = _field(noise, "peaks")
        noise_events = (
            _event_frame(
                noise_times,
                noise_peaks if noise_peaks is not None else [],
                "noise.times",
                path,
            )
            if noise_times is not None and np.asarray(noise_times).size
            else _event_frame([], [], "", path)
        )
    else:
        noise_events = _event_frame([], [], "", path)

    info = _field(ripples, "detectorinfo")
    if hasattr(info, "_fieldnames"):
        detector = _field(info, "detectorname")
        parameters = _plain(_field(info, "detectionparms"))
        channel = _scalar(_field(info, "detectionchannel"))
        channel_one_based = _scalar(_field(info, "detectionchannel1"))
    else:
        detector = _field(ripples, "detectorName")
        parameters = _plain(_field(ripples, "detectorParams"))
        channel = _scalar(_field(_field(ripples, "detectorParams"), "channel"))
        channel_one_based = None
    return BuzcodeRipples(
        events=events,
        noise_events=noise_events,
        detector=str(detector) if detector is not None else "",
        parameters=parameters if isinstance(parameters, dict) else {},
        channel=None if channel is None else int(channel),
        channel_one_based=None if channel_one_based is None else int(channel_one_based),
        stdev=_scalar(_field(ripples, "stdev")),
    )

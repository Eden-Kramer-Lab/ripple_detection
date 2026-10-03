"""Tests for examples/reference_recordings: the fetch tool and the file readers.

No test touches the network or CRCNS credentials, and none imports ``nwb.py``.
"""

import hashlib
import io
import json
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io


@pytest.fixture(scope="module")
def readers(reference_recordings_import):
    return reference_recordings_import("readers")


@pytest.fixture(scope="module")
def fetch(reference_recordings_import):
    return reference_recordings_import("fetch")


# --- .evt -------------------------------------------------------------------------------


def _write_evt(path, rows):
    path.write_text("".join(f"{t}\t{label}\n" for t, label in rows), encoding="utf-8")
    return path


def _ripple_rows(starts_ms, widths_ms=(30, 60), channel="23"):
    rows = []
    for start in starts_ms:
        rows += [
            (start, f"Ripple start {channel}"),
            (start + widths_ms[0], f"Ripple peak {channel}"),
            (start + widths_ms[1], f"Ripple stop {channel}"),
        ]
    return rows


class TestEvt:
    def test_read_evt_converts_milliseconds_exactly(self, readers, tmp_path):
        path = _write_evt(tmp_path / "a.evt", [(1500, "a b"), (2500.5, "c")])
        frame = readers.read_evt(path)
        assert list(frame.columns) == ["time", "label"]
        assert frame["time"].dtype == np.float64
        assert frame["time"].tolist() == [1.5, 2.5005]
        assert frame["label"].tolist() == ["a b", "c"]

    def test_unix_time_origin_keeps_every_millisecond(self, readers, tmp_path):
        # 1.7e12 ms is a Unix time in 2023; 1700000000123 ms must read as the
        # double nearest 1700000000.123 so no bound shifts by a rounding.
        starts = [1700000000123 + 1000 * i for i in range(50)]
        frame = readers.read_evt(_write_evt(tmp_path / "u.evt", _ripple_rows(starts)))
        expected = [float(f"{s // 1000}.{s % 1000:03d}") for s in starts]
        intervals = readers.evt_intervals(frame)
        assert intervals["start_time"].tolist() == expected
        assert np.all(np.diff(intervals["start_time"].to_numpy()) == 1.0)

    def test_blank_lines_skipped_and_bad_lines_refused(self, readers, tmp_path):
        path = tmp_path / "b.evt"
        path.write_text("10\tx\n\n20\ty\n", encoding="utf-8")
        assert len(readers.read_evt(path)) == 2
        path.write_text("10\n", encoding="utf-8")
        with pytest.raises(ValueError, match="line 1"):
            readers.read_evt(path)
        path.write_text("abc\tx\n", encoding="utf-8")
        with pytest.raises(ValueError, match="line 1"):
            readers.read_evt(path)

    def test_intervals_pair_rows_and_keep_channel(self, readers, tmp_path):
        rows = _ripple_rows([1000, 5000], channel="23")
        frame = readers.read_evt(_write_evt(tmp_path / "r.evt", rows))
        out = readers.evt_intervals(frame)
        assert list(out.columns) == ["start_time", "peak_time", "end_time", "label"]
        assert out["start_time"].tolist() == [1.0, 5.0]
        assert out["peak_time"].tolist() == [1.03, 5.03]
        assert out["end_time"].tolist() == [1.06, 5.06]
        assert out["label"].tolist() == ["23", "23"]
        assert out["start_time"].dtype == np.float64

    def test_peak_rows_are_optional(self, readers):
        events = pd.DataFrame(
            {
                "time": [1.0, 2.0, 3.0, 4.0],
                "label": ["x start 1", "x stop 1", "x start 1", "x stop 1"],
            }
        )
        out = readers.evt_intervals(events)
        assert out["peak_time"].isna().all()
        assert out["end_time"].tolist() == [2.0, 4.0]

    def test_custom_role_words(self, readers):
        events = pd.DataFrame(
            {"time": [1.0, 1.5, 2.0], "label": ["a on 7", "a mid 7", "a off 7"]}
        )
        out = readers.evt_intervals(events, start="on", peak="mid", stop="off")
        assert out.loc[0, "peak_time"] == 1.5

    @pytest.mark.parametrize(
        ("rows", "match"),
        [
            ([(1, "R start 1"), (2, "R peak 1")], "without a 'stop'"),
            ([(1, "R stop 1")], "without a preceding"),
            ([(1, "R start 1"), (2, "R start 1"), (3, "R stop 1")], "is open"),
            ([(1, "R start 1"), (2, "R stop 2")], "differs"),
            (
                [
                    (1, "R start 1"),
                    (2, "R peak 1"),
                    (3, "R stop 1"),
                    (4, "R start 1"),
                    (5, "R stop 1"),
                ],
                "1 of 2 events",
            ),
            ([(1, "R start 1"), (2, "R peak 1"), (3, "R peak 1"), (4, "R stop 1")], "second"),
            ([(1, "R start 1"), (5, "R peak 1"), (3, "R stop 1")], "out of order"),
            ([(1, "R begin 1"), (3, "R stop 1")], "exactly one"),
            (
                [(1, "R start 1"), (10, "R stop 1"), (5, "R start 1"), (12, "R stop 1")],
                "overlapping",
            ),
            (
                [(20, "R start 1"), (30, "R stop 1"), (1, "R start 1"), (5, "R stop 1")],
                "unsorted",
            ),
        ],
    )
    def test_intervals_error_paths(self, readers, rows, match):
        events = pd.DataFrame(
            {"time": [float(t) for t, _ in rows], "label": [lab for _, lab in rows]}
        )
        with pytest.raises(ValueError, match=match):
            readers.evt_intervals(events)

    def test_label_without_trailing_token(self, readers):
        events = pd.DataFrame(
            {
                "time": [1.0, 1.5, 2.0],
                "label": ["Ripple start", "Ripple peak", "Ripple stop"],
            }
        )
        out = readers.evt_intervals(events)
        assert out.loc[0, "label"] == ""
        assert out.loc[0, "peak_time"] == 1.5

    def test_touching_intervals_allowed(self, readers):
        events = pd.DataFrame(
            {
                "time": [1.0, 2.0, 2.0, 3.0],
                "label": ["R start 1", "R stop 1", "R start 1", "R stop 1"],
            }
        )
        assert len(readers.evt_intervals(events)) == 2


# --- .xml -------------------------------------------------------------------------------


def _write_xml(path, groups, skip=None, lfp_rate=1250, n_channels=8):
    skip = skip or {}
    channels = "".join(
        "<group>"
        + "".join(
            f'<channel skip="{skip[c]}">{c}</channel>'
            if c in skip
            else f"<channel>{c}</channel>"
            for c in group
        )
        + "</group>"
        for group in groups
    )
    lfp = (
        f"<fieldPotentials><lfpSamplingRate>{lfp_rate}</lfpSamplingRate></fieldPotentials>"
        if lfp_rate
        else ""
    )
    path.write_text(
        "<?xml version='1.0'?><parameters><acquisitionSystem>"
        f"<nChannels>{n_channels}</nChannels><samplingRate>20000</samplingRate>"
        f"</acquisitionSystem>{lfp}<anatomicalDescription><channelGroups>{channels}"
        "</channelGroups></anatomicalDescription></parameters>",
        encoding="utf-8",
    )
    return path


class TestXml:
    def test_fields(self, readers, tmp_path):
        path = _write_xml(tmp_path / "s.xml", [[0, 1, 2], [3, 4]], skip={1: 1, 2: 0})
        out = readers.read_xml(path)
        assert out.n_channels == 8
        assert out.sampling_rate == 20000
        assert out.lfp_sampling_rate == 1250
        assert out.groups == [[0, 1, 2], [3, 4]]
        assert out.skip == {1: True, 2: False}

    def test_optional_parts_absent(self, readers, tmp_path):
        out = readers.read_xml(_write_xml(tmp_path / "s.xml", [[5]], lfp_rate=None))
        assert out.lfp_sampling_rate is None
        assert out.skip == {}

    def test_missing_required_field(self, readers, tmp_path):
        path = tmp_path / "bad.xml"
        path.write_text("<parameters><acquisitionSystem/></parameters>", encoding="utf-8")
        with pytest.raises(ValueError, match="nChannels"):
            readers.read_xml(path)

    def test_rates_read_from_their_own_sections(self, readers, tmp_path):
        path = _write_xml(tmp_path / "s.xml", [[0, 1]])
        text = path.read_text(encoding="utf-8")
        decoy = (
            "<programs><program><samplingRate>1</samplingRate><nChannels>99</nChannels>"
            "<lfpSamplingRate>5</lfpSamplingRate></program></programs>"
        )
        path.write_text(text.replace("<parameters>", f"<parameters>{decoy}"), encoding="utf-8")
        out = readers.read_xml(path)
        assert (out.n_channels, out.sampling_rate, out.lfp_sampling_rate) == (8, 20000, 1250)

    def test_helper_writes_well_formed_xml(self, tmp_path):
        ET.parse(_write_xml(tmp_path / "s.xml", [[0]]))


# --- int16 binaries ---------------------------------------------------------------------


def _write_binary(path, n_samples=300, n_channels=6):
    data = (np.arange(n_samples * n_channels, dtype=np.int64) % 30000 - 15000).astype("<i2")
    data = data.reshape(n_samples, n_channels)
    data.tofile(path)
    return data


class TestBinary:
    def test_n_samples(self, readers):
        assert readers.binary_n_samples(2 * 6 * 300, 6) == 300

    @pytest.mark.parametrize("n_bytes", [2 * 6 * 300 + 2, 2 * 6 * 300 - 1])
    def test_size_not_a_multiple_refused(self, readers, n_bytes):
        with pytest.raises(ValueError, match="not a multiple"):
            readers.binary_n_samples(n_bytes, 6)

    def test_read_channels(self, readers, tmp_path):
        data = _write_binary(tmp_path / "x.lfp")
        out = readers.read_binary_channels(tmp_path / "x.lfp", 6, [4, 1])
        assert out.dtype == np.int16
        assert out.shape == (300, 2)
        np.testing.assert_array_equal(out, data[:, [4, 1]])

    def test_sample_range(self, readers, tmp_path):
        data = _write_binary(tmp_path / "x.dat")
        out = readers.read_binary_channels(tmp_path / "x.dat", 6, 3, start=10, stop=25)
        assert out.shape == (15, 1)
        np.testing.assert_array_equal(out, data[10:25, [3]])
        assert readers.read_binary_channels(tmp_path / "x.dat", 6, [0], start=290).shape == (
            10,
            1,
        )

    def test_returns_a_copy_not_a_map(self, readers, tmp_path):
        _write_binary(tmp_path / "x.dat")
        out = readers.read_binary_channels(tmp_path / "x.dat", 6, [0])
        assert not isinstance(out, np.memmap)
        assert out.flags.owndata or out.base is None or not isinstance(out.base, np.memmap)

    def test_error_paths(self, readers, tmp_path):
        _write_binary(tmp_path / "x.dat")
        with pytest.raises(ValueError, match="channels must lie"):
            readers.read_binary_channels(tmp_path / "x.dat", 6, [6])
        with pytest.raises(ValueError, match="sample range"):
            readers.read_binary_channels(tmp_path / "x.dat", 6, [0], start=0, stop=301)
        with (tmp_path / "x.dat").open("ab") as f:
            f.write(b"\x00")
        with pytest.raises(ValueError, match="not a multiple"):
            readers.read_binary_channels(tmp_path / "x.dat", 6, [0])


# --- buzcode ----------------------------------------------------------------------------


def _times(n=5, origin=0.0):
    starts = origin + 10.0 + 2.0 * np.arange(n)
    return np.column_stack([starts, starts + 0.08]), starts + 0.04


def _write_old(path, n=5, origin=0.0, noise=True, channel=46):
    times, peaks = _times(n, origin)
    noise_value = (
        {"times": times[:2], "peaks": peaks[:2], "peakNormedPower": np.array([1.0, 2.0])}
        if noise
        else np.empty((0, 0))
    )
    scipy.io.savemat(
        path,
        {
            "ripples": {
                "times": times,
                "peaks": peaks,
                "peakNormedPower": np.arange(n, dtype=float),
                "stdev": 202112.5,
                "noise": noise_value,
                "detectorName": "bz_FindRipples",
                "detectorParams": {
                    "channel": channel,
                    "thresholds": np.array([2, 5]),
                    "durations": np.array([50, 150]),
                    "passband": np.array([120, 180]),
                    "restrict": np.empty((0, 0)),
                    "basepath": "/data/x",
                },
            }
        },
    )
    return times, peaks


def _write_new(path, n=5, origin=0.0, with_channel1=True):
    times, peaks = _times(n, origin)
    info = {
        "detectorname": "bz_DetectSWR",
        "detectiondate": "2020-01-01",
        "detectionintervals": np.array([[0.0, 100.0]]),
        "detectionparms": {"thresholds": np.array([1, 3]), "name": "x"},
        "detectionchannel": 12,
    }
    if with_channel1:
        info["detectionchannel1"] = 13
    scipy.io.savemat(
        path,
        {
            "ripples": {
                "timestamps": times,
                "peaks": peaks,
                "peakNormedPower": np.arange(n, dtype=float),
                "stdev": 3.5,
                "detectorinfo": info,
            }
        },
    )
    return times, peaks


class TestBuzcode:
    def test_old_layout(self, readers, tmp_path):
        times, peaks = _write_old(tmp_path / "o.mat")
        out = readers.read_buzcode_events(tmp_path / "o.mat")
        assert list(out.events.columns) == ["start_time", "peak_time", "end_time"]
        assert out.events.shape == (5, 3)
        assert (out.events.dtypes == np.float64).all()
        np.testing.assert_array_equal(out.events["start_time"], times[:, 0])
        np.testing.assert_array_equal(out.events["peak_time"], peaks)
        np.testing.assert_array_equal(out.events["end_time"], times[:, 1])
        assert out.detector == "bz_FindRipples"
        assert out.channel == 46
        assert isinstance(out.channel, int)
        assert out.channel_one_based is None
        assert out.stdev == 202112.5
        assert out.parameters["thresholds"] == [2, 5]
        assert out.parameters["durations"] == [50, 150]
        assert out.parameters["passband"] == [120, 180]
        assert out.parameters["basepath"] == "/data/x"
        assert len(out.noise_events) == 2
        assert list(out.noise_events.columns) == list(out.events.columns)
        json.dumps(out.parameters)

    def test_old_layout_without_noise(self, readers, tmp_path):
        _write_old(tmp_path / "o.mat", noise=False)
        out = readers.read_buzcode_events(tmp_path / "o.mat")
        assert out.noise_events.shape == (0, 3)
        assert list(out.noise_events.columns) == ["start_time", "peak_time", "end_time"]

    def test_new_layout_keeps_both_channels(self, readers, tmp_path):
        times, _ = _write_new(tmp_path / "n.mat")
        out = readers.read_buzcode_events(tmp_path / "n.mat")
        np.testing.assert_array_equal(out.events[["start_time", "end_time"]], times)
        assert out.detector == "bz_DetectSWR"
        assert out.channel == 12
        assert out.channel_one_based == 13
        assert out.stdev == 3.5
        assert out.parameters == {"thresholds": [1, 3], "name": "x"}
        assert out.noise_events.empty

    def test_new_layout_without_channel1(self, readers, tmp_path):
        _write_new(tmp_path / "n.mat", with_channel1=False)
        assert readers.read_buzcode_events(tmp_path / "n.mat").channel_one_based is None

    @pytest.mark.parametrize("writer", ["old", "new"])
    def test_single_event(self, readers, tmp_path, writer):
        {"old": _write_old, "new": _write_new}[writer](tmp_path / "s.mat", n=1)
        out = readers.read_buzcode_events(tmp_path / "s.mat")
        assert out.events.shape == (1, 3)

    @pytest.mark.parametrize("writer", ["old", "new"])
    def test_unix_origin_bounds_unchanged(self, readers, tmp_path, writer):
        origin = 1.7e9 + 0.123
        times, peaks = {"old": _write_old, "new": _write_new}[writer](
            tmp_path / "u.mat", origin=origin
        )
        out = readers.read_buzcode_events(tmp_path / "u.mat")
        np.testing.assert_array_equal(out.events["start_time"], times[:, 0])
        np.testing.assert_array_equal(out.events["end_time"], times[:, 1])
        np.testing.assert_array_equal(out.events["peak_time"], peaks)

    def test_neuralynx_microsecond_scale_is_not_rescaled(self, readers, tmp_path):
        # The reader reports the saved values; it converts no units.
        times = np.array([[1.5e15, 1.5e15 + 80000.0]])
        scipy.io.savemat(
            tmp_path / "m.mat",
            {"ripples": {"times": times, "peaks": times[:, 0] + 40000.0, "detectorName": "x"}},
        )
        out = readers.read_buzcode_events(tmp_path / "m.mat")
        assert out.events.loc[0, "start_time"] == 1.5e15
        assert out.events.loc[0, "end_time"] == 1.5e15 + 80000.0

    def test_v73_file_refused_by_name(self, readers, tmp_path):
        path = tmp_path / "v73.mat"
        path.write_bytes(b"MATLAB 7.3 MAT-file, Platform: x" + b"\x00" * 600)
        with pytest.raises(ValueError, match=r"v73\.mat.*v7\.3.*h5py"):
            readers.read_buzcode_events(path)

    def test_missing_fields(self, readers, tmp_path):
        times, _ = _times()
        scipy.io.savemat(tmp_path / "a.mat", {"ripples": {"times": times}})
        with pytest.raises(ValueError, match="peaks"):
            readers.read_buzcode_events(tmp_path / "a.mat")
        scipy.io.savemat(tmp_path / "b.mat", {"ripples": {"peaks": times[:, 0]}})
        with pytest.raises(ValueError, match="event times"):
            readers.read_buzcode_events(tmp_path / "b.mat")
        scipy.io.savemat(tmp_path / "c.mat", {"other": 1})
        with pytest.raises(ValueError, match="no 'ripples' struct"):
            readers.read_buzcode_events(tmp_path / "c.mat")

    @pytest.mark.parametrize("what", ["channel", "stdev"])
    def test_multi_valued_scalar_refused(self, readers, tmp_path, what):
        times, peaks = _times()
        ripples = {
            "timestamps": times,
            "peaks": peaks,
            "detectorinfo": {"detectionchannel": 12},
        }
        if what == "channel":
            ripples["detectorinfo"]["detectionchannel"] = np.array([12, 13])
        else:
            ripples["stdev"] = np.array([1.0, 2.0])
        scipy.io.savemat(tmp_path / "e.mat", {"ripples": ripples})
        with pytest.raises(ValueError, match="expected a scalar"):
            readers.read_buzcode_events(tmp_path / "e.mat")

    def test_peak_count_mismatch(self, readers, tmp_path):
        times, peaks = _times()
        scipy.io.savemat(tmp_path / "d.mat", {"ripples": {"times": times, "peaks": peaks[:3]}})
        with pytest.raises(ValueError, match="peaks"):
            readers.read_buzcode_events(tmp_path / "d.mat")


# --- fetch_url ---------------------------------------------------------------------------

PAYLOAD = bytes(range(256)) * 40


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "src" / "data.bin"
    path.parent.mkdir()
    path.write_bytes(PAYLOAD)
    return path


class TestFetchUrl:
    def test_accepts_and_records(self, fetch, source, tmp_path):
        cache = tmp_path / "cache"
        sha, md5 = hashlib.sha256(PAYLOAD).hexdigest(), hashlib.md5(PAYLOAD).hexdigest()
        record = fetch.fetch_url(
            source.as_uri(),
            "sub/data.bin",
            expected_size=len(PAYLOAD),
            expected_sha256=sha,
            expected_md5=md5,
            expected_source="test",
            cache=cache,
        )
        assert (cache / "sub" / "data.bin").read_bytes() == PAYLOAD
        assert not list(cache.rglob("*.part"))
        manifest = json.loads((cache / "manifest.json").read_text())
        assert manifest == [record]
        assert record["kind"] == "url"
        assert record["path"] == "sub/data.bin"
        assert record["bytes"] == len(PAYLOAD)
        assert record["sha256"] == sha
        assert record["md5"] == md5
        assert record["expected"] == {
            "bytes": len(PAYLOAD),
            "sha256": sha,
            "md5": md5,
            "source": "test",
        }
        assert record["retrieved"].endswith("+00:00")
        assert "python" in record["versions"]

    def test_manifest_appends(self, fetch, source, tmp_path):
        cache = tmp_path / "cache"
        fetch.fetch_url(source.as_uri(), "a.bin", cache=cache)
        fetch.fetch_url(source.as_uri(), "b.bin", cache=cache)
        assert [r["path"] for r in fetch.read_manifest(cache)] == ["a.bin", "b.bin"]
        assert not list(cache.glob("manifest.json.part"))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"expected_sha256": "0" * 64},
            {"expected_md5": "0" * 32},
            {"expected_size": len(PAYLOAD) + 1},
        ],
    )
    def test_mismatch_refused_and_nothing_left(self, fetch, source, tmp_path, kwargs):
        cache = tmp_path / "cache"
        with pytest.raises(ValueError, match="removed"):
            fetch.fetch_url(source.as_uri(), "x/data.bin", cache=cache, **kwargs)
        assert not [p for p in cache.rglob("*") if p.is_file()]

    def test_destination_must_stay_in_cache(self, fetch, source, tmp_path):
        with pytest.raises(ValueError, match="inside the cache"):
            fetch.fetch_url(source.as_uri(), "../escape.bin", cache=tmp_path / "cache")
        with pytest.raises(ValueError, match="inside the cache"):
            fetch.fetch_url(
                source.as_uri(), str(tmp_path / "abs.bin"), cache=tmp_path / "cache"
            )

    def test_cache_directory_from_environment(self, fetch, monkeypatch, tmp_path):
        monkeypatch.setenv("RIPPLE_REFERENCE_CACHE", str(tmp_path / "elsewhere"))
        assert fetch.cache_dir() == tmp_path / "elsewhere"
        monkeypatch.delenv("RIPPLE_REFERENCE_CACHE")
        assert fetch.cache_dir().name == "reference_recordings"


# --- Wayback -----------------------------------------------------------------------------

CRAWLER = "x-archive-orig-x-crawler-content-length"
ORIG = "x-archive-orig-content-length"


class TestWaybackLength:
    def test_equal_accepted_from_content_length(self, fetch):
        assert fetch.wayback_original_length({ORIG: "100"}, 100) == (100, ORIG)

    def test_equal_accepted_from_crawler_header(self, fetch):
        assert fetch.wayback_original_length({CRAWLER: "100"}, 100) == (100, CRAWLER)

    def test_crawler_header_takes_precedence(self, fetch):
        # a truncated record's content-length describes the stored bytes only
        headers = {ORIG: "1048576", CRAWLER: "5000000"}
        assert fetch.wayback_original_length(headers, 5000000) == (5000000, CRAWLER)
        with pytest.raises(ValueError, match="truncated"):
            fetch.wayback_original_length(headers, 1048576)

    def test_header_names_case_insensitive(self, fetch):
        assert (
            fetch.wayback_original_length({"X-Archive-Orig-Content-Length": " 7 "}, 7)[0] == 7
        )

    @pytest.mark.parametrize("headers", [{}, {"x-archive-orig-etag": "x"}])
    def test_missing_header_refused(self, fetch, headers):
        with pytest.raises(ValueError, match="neither"):
            fetch.wayback_original_length(headers, 100)

    @pytest.mark.parametrize("header", [ORIG, CRAWLER])
    def test_truncated_and_longer_refused(self, fetch, header):
        with pytest.raises(ValueError, match="truncated"):
            fetch.wayback_original_length({header: "100"}, 99)
        with pytest.raises(ValueError, match="longer"):
            fetch.wayback_original_length({header: "100"}, 101)

    def test_malformed_refused(self, fetch):
        with pytest.raises(ValueError, match="not an integer"):
            fetch.wayback_original_length({ORIG: "abc"}, 3)


class TestWaybackCaptureTimestamp:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://web.archive.org/web/20231129113529id_/https://h/x", "20231129113529"),
            ("https://web.archive.org/web/20231129113529/https://h/x", "20231129113529"),
            ("https://web.archive.org/web/20240101000000if_/http://h/x", "20240101000000"),
            ("https://h/x", None),
            ("https://web.archive.org/web/2023id_/https://h/x", None),
        ],
    )
    def test_parse(self, fetch, url, expected):
        assert fetch.wayback_capture_timestamp(url) == expected


class _FakeResponse(io.BytesIO):
    def __init__(self, body, headers, url=""):
        super().__init__(body)
        self.headers = headers
        self.url = url

    def geturl(self):
        return self.url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class TestFetchWayback:
    def _patch(self, monkeypatch, fetch, body, headers, seen, served=None):
        def fake_urlopen(request, timeout=None):
            seen.append(request)
            return _FakeResponse(body, headers, served or request.full_url)

        monkeypatch.setattr(fetch.urllib.request, "urlopen", fake_urlopen)

    def test_whole_file_recorded(self, fetch, monkeypatch, tmp_path):
        seen = []
        headers = {ORIG: str(len(PAYLOAD)), "x-archive-orig-last-modified": "Wed, 09 Jan 2019"}
        self._patch(monkeypatch, fetch, PAYLOAD, headers, seen)
        record = fetch.fetch_wayback(
            "https://h/x.mat", "20231129113529", "w/x.mat", cache=tmp_path
        )
        assert (
            seen[0].full_url == "https://web.archive.org/web/20231129113529id_/https://h/x.mat"
        )
        assert (tmp_path / "w" / "x.mat").read_bytes() == PAYLOAD
        assert record["wayback_timestamp"] == "20231129113529"
        assert record["requested_timestamp"] == "20231129113529"
        assert record["served_url"] == seen[0].full_url
        assert record["original_content_length"] == len(PAYLOAD)
        assert record["original_length_header"] == ORIG
        assert record["original_last_modified"] == "Wed, 09 Jan 2019"
        assert record["head"] is False

    def test_truncated_capture_removed(self, fetch, monkeypatch, tmp_path):
        self._patch(monkeypatch, fetch, PAYLOAD[:1000], {CRAWLER: str(len(PAYLOAD))}, [])
        with pytest.raises(ValueError, match="truncated"):
            fetch.fetch_wayback("https://h/x.dat", "20231129113529", "w/x.dat", cache=tmp_path)
        assert not [p for p in tmp_path.rglob("*") if p.is_file()]

    def test_sha256_checked(self, fetch, monkeypatch, tmp_path):
        self._patch(monkeypatch, fetch, PAYLOAD, {ORIG: str(len(PAYLOAD))}, [])
        with pytest.raises(ValueError, match="removed"):
            fetch.fetch_wayback(
                "https://h/x",
                "20231129113529",
                "w/x",
                expected_sha256="0" * 64,
                cache=tmp_path,
            )
        assert not [p for p in tmp_path.rglob("*") if p.is_file()]

    def test_head_mode(self, fetch, monkeypatch, tmp_path):
        seen = []
        self._patch(monkeypatch, fetch, PAYLOAD, {CRAWLER: "123456789"}, seen)
        record = fetch.fetch_wayback(
            "https://h/x.dat", "20231129113529", "w/x.dat", head_bytes=64, cache=tmp_path
        )
        assert seen[0].get_header("Range") == "bytes=0-63"
        stored = tmp_path / "w" / "x.dat.head"
        assert stored.read_bytes() == PAYLOAD[:64]
        assert not (tmp_path / "w" / "x.dat").exists()
        assert record["head"] is True
        assert record["range"] == "bytes=0-63"
        assert record["original_content_length"] == 123456789
        assert record["path"] == "w/x.dat.head"
        assert record["bytes"] == 64

    def test_other_capture_refused_by_default(self, fetch, monkeypatch, tmp_path):
        served = "https://web.archive.org/web/20240101000000id_/https://h/x"
        self._patch(monkeypatch, fetch, PAYLOAD, {ORIG: str(len(PAYLOAD))}, [], served)
        with pytest.raises(ValueError, match="not the requested 20231129113529"):
            fetch.fetch_wayback("https://h/x", "20231129113529", "w/x", cache=tmp_path)
        assert not [p for p in tmp_path.rglob("*") if p.is_file()]

    def test_other_capture_recorded_when_allowed(self, fetch, monkeypatch, tmp_path):
        served = "https://web.archive.org/web/20240101000000id_/https://h/x"
        self._patch(monkeypatch, fetch, PAYLOAD, {ORIG: str(len(PAYLOAD))}, [], served)
        record = fetch.fetch_wayback(
            "https://h/x", "20231129113529", "w/x", allow_other_capture=True, cache=tmp_path
        )
        assert record["requested_timestamp"] == "20231129113529"
        assert record["wayback_timestamp"] == "20240101000000"
        assert record["served_url"] == served

    def test_head_short_read_refused(self, fetch, monkeypatch, tmp_path):
        self._patch(monkeypatch, fetch, PAYLOAD[:10], {CRAWLER: "123456789"}, [])
        with pytest.raises(ValueError, match="expected 64"):
            fetch.fetch_wayback(
                "https://h/x", "2023" + "0" * 10, "x", head_bytes=64, cache=tmp_path
            )
        assert not [p for p in tmp_path.rglob("*") if p.is_file()]

    def test_head_longer_than_original_needs_exactly_the_original(
        self, fetch, monkeypatch, tmp_path
    ):
        self._patch(monkeypatch, fetch, PAYLOAD[:10], {CRAWLER: "10"}, [])
        record = fetch.fetch_wayback(
            "https://h/x", "20230000000000", "x", head_bytes=64, cache=tmp_path
        )
        assert record["bytes"] == 10
        self._patch(monkeypatch, fetch, PAYLOAD[:9], {CRAWLER: "10"}, [])
        with pytest.raises(ValueError, match="expected 10"):
            fetch.fetch_wayback(
                "https://h/y", "20230000000000", "y", head_bytes=64, cache=tmp_path
            )

    def test_head_mode_reads_in_pieces(self, fetch, monkeypatch, tmp_path):
        class Trickle(_FakeResponse):
            def read(self, n=-1):
                return super().read(min(n, 7) if n > 0 else n)

        def fake_urlopen(request, timeout=None):
            return Trickle(PAYLOAD, {CRAWLER: "123456789"}, request.full_url)

        monkeypatch.setattr(fetch.urllib.request, "urlopen", fake_urlopen)
        record = fetch.fetch_wayback(
            "https://h/x", "20230000000000", "x", head_bytes=64, cache=tmp_path
        )
        assert (tmp_path / "x.head").read_bytes() == PAYLOAD[:64]
        assert record["bytes"] == 64

    def test_head_mode_without_length_header_and_bad_size(self, fetch, monkeypatch, tmp_path):
        self._patch(monkeypatch, fetch, PAYLOAD, {}, [])
        record = fetch.fetch_wayback(
            "https://h/x", "20230000000000", "x", head_bytes=8, cache=tmp_path
        )
        assert record["original_content_length"] is None
        with pytest.raises(ValueError, match="positive"):
            fetch.fetch_wayback(
                "https://h/x", "20230000000000", "x", head_bytes=0, cache=tmp_path
            )


# --- CRCNS helpers -----------------------------------------------------------------------

FILELIST = (
    "# CRCNS.org 'hc-18' dataset files\n"
    "# To use this for fetching files, comment out files that are\n"
    " code.zip\t108361 (105.8 KB)\n"
    " data/Train-242-20140124.tar.gz\t7647029986 (7.1 GB)\n"
    "#data/skipped.tar.gz\t42\n"
)
CHECKSUMS = (
    "151cb8245070619019e15750f5efe760  code.zip\n"
    "4fa842b93dbf288365b3ea4aa531190f  data/Train-242-20140124.tar.gz\n"
    "not a checksum line\n"
)


class TestCrcnsHelpers:
    def test_filelist(self, fetch):
        sizes = fetch.parse_crcns_filelist(FILELIST)
        assert sizes == {
            "code.zip": 108361,
            "data/Train-242-20140124.tar.gz": 7647029986,
            "data/skipped.tar.gz": 42,
        }

    def test_checksums(self, fetch):
        sums = fetch.parse_crcns_checksums(
            CHECKSUMS + "ABCDEFABCDEFABCDEFABCDEFABCDEFAB  ./b/c.txt\n"
        )
        assert sums["code.zip"] == "151cb8245070619019e15750f5efe760"
        assert sums["data/Train-242-20140124.tar.gz"] == "4fa842b93dbf288365b3ea4aa531190f"
        assert sums["b/c.txt"] == "abcdefabcdefabcdefabcdefabcdefab"
        assert len(sums) == 3

    def test_login_page_detection(self, fetch):
        assert fetch.is_crcns_login_page(
            "text/html; charset=utf-8", b"<html>Login below</html>"
        )
        assert not fetch.is_crcns_login_page("text/html", b"<html>a real page</html>")
        assert not fetch.is_crcns_login_page("application/gzip", b"Login below")

    def test_select_paths(self, fetch):
        sizes = fetch.parse_crcns_filelist(FILELIST)
        assert fetch.select_paths(
            sizes, ["data/*.tar.gz", "/code.zip", "data/Train-242-20140124.tar.gz"]
        ) == [
            "data/Train-242-20140124.tar.gz",
            "data/skipped.tar.gz",
            "code.zip",
        ]
        with pytest.raises(ValueError, match="matches nothing"):
            fetch.select_paths(sizes, ["nope*"])

    def test_login_needs_environment_credentials(self, fetch, monkeypatch):
        monkeypatch.delenv("CRCNS_USERNAME", raising=False)
        monkeypatch.delenv("CRCNS_PASSWORD", raising=False)
        with pytest.raises(RuntimeError, match="CRCNS_USERNAME"):
            fetch.CrcnsSession().login("hc-14")


def _make_tar(path, members, mode="w:gz"):
    with tarfile.open(path, mode) as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return path


class TestExtractMembers:
    def test_extracts_only_matches_and_records(self, fetch, tmp_path):
        members = {"a/x.clu.1": b"1 2 3", "a/x.res.1": b"4 5", "b/y.clu.1": b"9"}
        tar = _make_tar(tmp_path / "s.tar.gz", members)
        cache = tmp_path / "cache"
        out = fetch.extract_members(tar, ["a/*.clu.*", "b/*"], cache / "d", cache=cache)
        assert [p.relative_to(cache / "d").as_posix() for p in out] == [
            "a/x.clu.1",
            "b/y.clu.1",
        ]
        assert not (cache / "d" / "a" / "x.res.1").exists()
        assert (cache / "d" / "a" / "x.clu.1").read_bytes() == b"1 2 3"
        records = fetch.read_manifest(cache)
        assert [r["member"] for r in records] == ["a/x.clu.1", "b/y.clu.1"]
        assert records[0]["sha256"] == hashlib.sha256(b"1 2 3").hexdigest()
        assert records[0]["archive"] == str(tar)
        assert records[0]["kind"] == "tar-member"
        assert records[0]["path"] == "d/a/x.clu.1"
        assert not list(cache.rglob("*.part"))

    def test_plain_tar(self, fetch, tmp_path):
        tar = _make_tar(tmp_path / "s.tar", {"m": b"z"}, mode="w")
        out = fetch.extract_members(tar, ["m"], tmp_path / "out", cache=tmp_path / "c")
        assert out[0].read_bytes() == b"z"

    @pytest.mark.parametrize("name", ["../evil", "a/../../evil", "/abs/evil"])
    def test_unsafe_names_refused_before_writing(self, fetch, tmp_path, name):
        tar = _make_tar(tmp_path / "s.tar", {"ok": b"1", name: b"2"}, mode="w")
        with pytest.raises(ValueError, match="leaves the destination"):
            fetch.extract_members(tar, ["*"], tmp_path / "out", cache=tmp_path / "c")
        assert not (tmp_path / "out").exists()
        assert not (tmp_path / "evil").exists()
        assert fetch.read_manifest(tmp_path / "c") == []

    def test_no_match_refused(self, fetch, tmp_path):
        tar = _make_tar(tmp_path / "s.tar", {"m": b"z"}, mode="w")
        with pytest.raises(ValueError, match="no member matches"):
            fetch.extract_members(tar, ["q*"], tmp_path / "out", cache=tmp_path / "c")


class _CrcnsResponse(_FakeResponse):
    def __init__(self, body, content_type="application/octet-stream"):
        super().__init__(
            body, {"Content-Type": content_type, "Content-Length": str(len(body))}
        )


class _CrcnsOpener:
    def __init__(self, bodies):
        self.bodies = bodies

    def open(self, request, timeout=None):
        name = request.full_url.removesuffix("?agent=1").rsplit("download.crcns.org/", 1)[1]
        return _CrcnsResponse(self.bodies[name])


class TestCrcnsFetch:
    BODY = b"spike data" * 100

    def _session(self, fetch, bodies):
        session = fetch.CrcnsSession()
        session.opener = _CrcnsOpener(bodies)
        return session

    def _index(self, size, md5):
        return {
            "hc-x/filelist.txt": f" d/a.bin\t{size} (1 KB)\n".encode(),
            "hc-x/checksums.md5": f"{md5}  d/a.bin\n".encode(),
        }

    def test_download_verified_before_rename_and_recorded(self, fetch, tmp_path):
        md5 = hashlib.md5(self.BODY).hexdigest()
        bodies = {**self._index(len(self.BODY), md5), "hc-x/d/a.bin": self.BODY}
        records = fetch.crcns_fetch(
            "hc-x", ["d/*.bin"], cache=tmp_path, session=self._session(fetch, bodies)
        )
        out = tmp_path / "crcns" / "hc-x" / "d" / "a.bin"
        assert out.read_bytes() == self.BODY
        assert records[0]["md5"] == md5
        assert records[0]["expected"]["md5"] == md5
        assert not list(tmp_path.rglob("*.part"))

    def test_mismatch_never_reaches_final_name(self, fetch, tmp_path):
        md5 = hashlib.md5(self.BODY).hexdigest()
        bodies = {**self._index(len(self.BODY), md5), "hc-x/d/a.bin": b"X" * len(self.BODY)}
        with pytest.raises(ValueError, match="removed"):
            fetch.crcns_fetch(
                "hc-x", ["d/a.bin"], cache=tmp_path, session=self._session(fetch, bodies)
            )
        assert not (tmp_path / "crcns" / "hc-x" / "d" / "a.bin").exists()
        assert not list(tmp_path.rglob("*.part"))
        assert fetch.read_manifest(tmp_path) == []

    def test_wrong_size_never_reaches_final_name(self, fetch, tmp_path):
        bodies = {
            **self._index(len(self.BODY) + 1, hashlib.md5(self.BODY).hexdigest()),
            "hc-x/d/a.bin": self.BODY,
        }
        with pytest.raises(ValueError, match="removed"):
            fetch.crcns_fetch(
                "hc-x", ["d/a.bin"], cache=tmp_path, session=self._session(fetch, bodies)
            )
        assert not (tmp_path / "crcns" / "hc-x" / "d" / "a.bin").exists()

    def test_cached_file_is_reverified(self, fetch, tmp_path):
        md5 = hashlib.md5(self.BODY).hexdigest()
        bodies = {**self._index(len(self.BODY), md5), "hc-x/d/a.bin": self.BODY}
        fetch.crcns_fetch(
            "hc-x", ["d/a.bin"], cache=tmp_path, session=self._session(fetch, bodies)
        )
        cached = tmp_path / "crcns" / "hc-x" / "d" / "a.bin"
        cached.write_bytes(b"Y" * len(self.BODY))
        with pytest.raises(ValueError, match="cached file"):
            fetch.crcns_fetch(
                "hc-x", ["d/a.bin"], cache=tmp_path, session=self._session(fetch, bodies)
            )
        assert not cached.exists()


# --- databank sessions -----------------------------------------------------------------


@pytest.fixture(scope="module")
def databank(reference_recordings_import):
    return reference_recordings_import("databank")


class TestSessionTable:
    def test_entries_are_complete_and_consistent(self, databank):
        assert databank.SESSIONS
        for key, session in databank.SESSIONS.items():
            assert session.key == key
            assert session.basename == session.databank_path.rsplit("/", 1)[1]
            assert session.events.suffix == ".ripples.events.mat"
            assert {c.suffix for c in session.support} == {
                ".xml",
                ".session.mat",
                ".sessionInfo.mat",
            }
            for capture in (session.events, *session.support):
                assert len(capture.sha256) == 64
                int(capture.sha256, 16)
                assert capture.length > 0
            for capture in (session.events, *session.support, session.lfp_head):
                assert len(capture.timestamp) == 14
                assert capture.timestamp.isdigit()
            assert session.lfp_head.suffix == ".lfp"
            assert session.lfp_head.sha256 is None
            assert session.lfp_head.length % 2 == 0
            assert len(session.lfp_asset_id) == 36
            assert session.lfp_asset_id.count("-") == 4
            assert session.lfp_series.startswith("/")
            assert len(session.source_code.commit) == 40
            assert session.source_code.filter_kind in ("cheby2", "butter")
            assert session.source_code.smoothing_samples % 2 == 1
            assert session.url(session.events) == (
                f"{databank.DATABANK_URL}/{session.databank_path}/"
                f"{session.basename}.ripples.events.mat"
            )
            assert session.cache_path(session.events).startswith(f"buzsaki/{key}/")

    def test_ms10_values(self, databank):
        ms10 = databank.SESSIONS["MS10"]
        assert ms10.lfp_head.length == 23_613_750 * 64 * 2
        assert ms10.events.length == 11305
        assert ms10.events.timestamp == "20231129113529"
        assert ms10.lfp_head.timestamp == "20231129114227"
        assert ms10.channel_tag == "Ripple"
        assert ms10.noise_channel_tag is None
        assert ms10.source_code.minimum_duration_ms is None


class TestInputChecks:
    N_ROWS, N_CHANNELS, STORED = 16, 8, 5

    @pytest.fixture
    def head(self):
        rng = np.random.default_rng(0)
        return rng.integers(-2000, 2000, size=(self.N_ROWS, self.N_CHANNELS)).astype(np.int16)

    def test_identity_map(self, databank, head):
        check = databank.check_head_match(head, head.copy(), self.STORED)
        assert check.passed, check.reason
        assert check.detail["identity"]
        assert check.detail["stored_channel_column"] == self.STORED
        assert check.detail["channels_absent"] == []

    def test_permuted_map_is_reported_not_assumed(self, databank, head):
        permutation = [3, 7, 0, 5, 1, 6, 2, 4]
        check = databank.check_head_match(head, head[:, permutation], self.STORED)
        assert check.passed, check.reason
        assert check.detail["column_to_channel"] == permutation
        assert not check.detail["identity"]
        assert check.detail["stored_channel_column"] == permutation.index(self.STORED)

    def test_subset_of_channels(self, databank, head):
        kept = [0, 1, 2, 4, 5, 7]
        check = databank.check_head_match(head, head[:, kept], self.STORED)
        assert check.passed, check.reason
        assert check.detail["channels_absent"] == [3, 6]
        assert check.detail["stored_channel_column"] == kept.index(self.STORED)

    def test_dropped_stored_channel_fails(self, databank, head):
        check = databank.check_head_match(head, head[:, [0, 1, 2, 3, 4, 6, 7]], self.STORED)
        assert not check.passed
        assert "stored channel 5 is in no DANDI column" in check.reason

    def test_changed_column_fails(self, databank, head):
        dandi = head.copy()
        dandi[3, 2] += 1
        check = databank.check_head_match(head, dandi, self.STORED)
        assert not check.passed
        assert "DANDI columns [2] match no .lfp channel" in check.reason

    def test_ambiguous_stored_channel_fails(self, databank, head):
        twin = head.copy()
        twin[:, 1] = twin[:, self.STORED]
        check = databank.check_head_match(twin, twin.copy(), self.STORED)
        assert not check.passed
        assert "matches DANDI columns" in check.reason

    def test_head_rows_must_align(self, databank, head):
        with pytest.raises(ValueError, match="row counts differ"):
            databank.head_column_map(head, head[:-1])

    def test_length(self, databank):
        check = databank.check_length(3_022_560_000, 64, (23_613_750, 64))
        assert check.passed, check.reason
        assert check.detail["lfp_samples"] == 23_613_750
        short = databank.check_length(3_022_560_000, 64, (23_613_749, 64))
        assert not short.passed
        assert ".lfp holds 23613750 samples, DANDI 23613749 rows" in short.reason
        odd = databank.check_length(3_022_560_001, 64, (23_613_750, 64))
        assert not odd.passed
        assert "not a multiple" in odd.reason
        wide = databank.check_length(3_022_560_000, 64, (23_613_750, 65))
        assert not wide.passed
        assert "65 columns" in wide.reason

    def test_channel_tag_is_one_based(self, databank):
        assert databank.check_channel_tag(46, 47).passed
        wrong = databank.check_channel_tag(46, 46)
        assert not wrong.passed
        assert "stored channel 46 (0-based) is not the tag's 46 (1-based)" in wrong.reason
        missing = databank.check_channel_tag(46, None)
        assert not missing.passed
        assert "no such channel tag" in missing.reason

    def test_channel_tag_with_several_channels(self, databank):
        among = databank.check_channel_tag(46, [12, 47, 50])
        assert among.passed, among.reason
        assert among.detail["n_tag_channels"] == 3
        assert "several" in among.detail["rule"]
        absent = databank.check_channel_tag(46, [12, 46, 50])
        assert not absent.passed
        assert "not among the tag's 3 channels [12, 46, 50] (1-based)" in absent.reason
        assert "no such channel tag" not in absent.reason
        empty = databank.check_channel_tag(46, [])
        assert not empty.passed
        assert "lists no channels" in empty.reason

    def test_channel_tag_read_from_session_mat(self, databank, tmp_path):
        path = tmp_path / "s.session.mat"
        scipy.io.savemat(
            path,
            {
                "session": {
                    "channelTags": {
                        "Ripple": {"channels": 47},
                        "RippleNoise": {"channels": np.array([19, 20])},
                        "Cortical": {"channels": np.empty((0, 0))},
                    }
                }
            },
        )
        session_mat = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)[
            "session"
        ]
        assert databank.channel_tag(session_mat, "Ripple") == [47]
        assert databank.channel_tag(session_mat, "RippleNoise") == [19, 20]
        assert databank.channel_tag(session_mat, "Cortical") == []
        assert databank.channel_tag(session_mat, "Theta") is None

    def test_rates(self, databank):
        assert databank.check_rates(1250.0, 1250.0, 1250).passed
        check = databank.check_rates(1250.0, 1000.0, 1250)
        assert not check.passed
        assert "disagree" in check.reason
        assert not databank.check_rates(None, 1250.0, 1250).passed

    @staticmethod
    def _events(origin, samples, fs=1250.0):
        samples = np.asarray(samples, dtype=float)
        return pd.DataFrame(
            {
                "start_time": origin + samples[:, 0] / fs,
                "peak_time": origin + samples[:, 1] / fs,
                "end_time": origin + samples[:, 2] / fs,
            }
        )

    @pytest.mark.parametrize("origin", [0.0, 1.7e9])
    def test_events_on_the_grid_pass_at_any_origin(self, databank, origin):
        events = self._events(
            origin, [(0, 3, 10), (5000, 5010, 5020), (99_990, 99_995, 99_999)]
        )
        check = databank.check_events_in_recording(events, origin, 100_000, 1250.0)
        assert check.passed, check.reason
        assert check.detail["last_end_sample"] == 99_999

    @pytest.mark.parametrize("origin", [0.0, 1.7e9])
    def test_event_outside_the_recording_fails(self, databank, origin):
        events = self._events(origin, [(0, 3, 10), (99_990, 99_995, 100_000)])
        check = databank.check_events_in_recording(events, origin, 100_000, 1250.0)
        assert not check.passed
        assert "events [1] lie outside the 100000 recorded samples" in check.reason

    @pytest.mark.parametrize("origin", [0.0, 1.7e9])
    def test_event_off_the_grid_fails(self, databank, origin):
        events = self._events(origin, [(0, 3, 10), (500.5, 505, 510)])
        check = databank.check_events_in_recording(events, origin, 100_000, 1250.0)
        assert not check.passed
        assert "events [1] are off the 1250.0 Hz grid" in check.reason

    def test_unordered_event_fails(self, databank):
        events = self._events(0.0, [(10, 3, 20)])
        check = databank.check_events_in_recording(events, 0.0, 100, 1250.0)
        assert not check.passed
        assert "not start <= peak <= end" in check.reason

    def test_events_spanning_closed_bounds(self, databank):
        origin = 1.7e9
        events = self._events(origin, [(0, 1, 10), (10, 12, 20), (30, 31, 40)])
        assert databank.events_spanning(events, [origin + 10 / 1250]) == [0, 1]
        assert databank.events_spanning(events, [origin + 25 / 1250]) == []


class TestSourceTranscription:
    FS = 1000.0  # one sample per millisecond, so bounds read as milliseconds
    ORIGIN = 1.7e9

    def test_filter0_is_a_zero_padded_centred_average(self, databank):
        x = np.random.default_rng(1).normal(size=200)
        window = np.ones(11) / 11
        np.testing.assert_allclose(
            databank.filter0(window, x), np.convolve(x, window, mode="same"), atol=1e-12
        )
        with pytest.raises(ValueError, match="odd"):
            databank.filter0(np.ones(4) / 4, x)

    def test_normalization_uses_n_minus_one_or_the_given_sd(self, databank):
        signal = np.random.default_rng(2).normal(size=500)
        normalized, sd, mean = databank.normalized_squared_signal(signal, 11)
        smoothed = np.convolve(signal**2, np.ones(11) / 11, mode="same")
        assert sd == pytest.approx(np.std(smoothed, ddof=1), rel=1e-12)
        assert mean == pytest.approx(smoothed.mean(), rel=1e-12)
        np.testing.assert_allclose(normalized, (smoothed - mean) / sd, atol=1e-10)
        fixed, used, _ = databank.normalized_squared_signal(signal, 11, sd=2.0)
        assert used == 2.0
        np.testing.assert_allclose(fixed, (smoothed - mean) / 2.0, atol=1e-10)

    @pytest.fixture
    def case(self):
        normalized = np.zeros(60)
        normalized[0] = 3.0  # incomplete first run: dropped
        normalized[5:8] = [3.0, 6.0, 3.0]  # A: start 4, stop 7
        normalized[10:12] = 3.0  # B: start 9, stop 11; gap 2 < 5 merges with A
        normalized[20:23] = [3.0, 4.0, 3.0]  # C: peak 4 is not above 5
        normalized[30:45] = 3.0  # D: 29..44 is 15 ms, over 12
        normalized[37] = 7.0
        normalized[50:54] = [3.0, 5.5, 3.0, 3.0]  # E: start 49, stop 53
        normalized[59] = 3.0  # incomplete last run: dropped
        signal = np.zeros(60)
        signal[[6, 10]] = -2.0  # tied troughs: the first is the peak
        signal[52] = -1.0
        timestamps = self.ORIGIN + np.arange(60) / self.FS
        return normalized, signal, timestamps

    def _run(self, databank, case, **overrides):
        normalized, signal, timestamps = case
        settings = {
            "low_threshold": 2.0,
            "high_threshold": 5.0,
            "minimum_inter_ripple_interval_ms": 5.0,
            "maximum_duration_ms": 12.0,
            "frequency": self.FS,
        }
        return databank.source_segmentation(
            normalized, signal, timestamps, **(settings | overrides)
        )

    def test_hand_built_case(self, databank, case):
        events = self._run(databank, case)
        assert list(events.columns) == databank.SOURCE_EVENT_COLUMNS
        assert events.start_index.tolist() == [4, 49]
        assert events.stop_index.tolist() == [11, 53]
        assert events.trough_index.tolist() == [6, 52]
        assert events.max_index.tolist() == [6, 51]
        assert events.peak_normed_power.tolist() == [6.0, 5.5]
        timestamps = case[2]
        assert events.start_time.tolist() == timestamps[[4, 49]].tolist()
        assert events.end_time.tolist() == timestamps[[11, 53]].tolist()
        assert events.peak_time.tolist() == timestamps[[6, 52]].tolist()
        assert events.max_power_time.tolist() == timestamps[[6, 51]].tolist()

    def test_gap_equal_to_the_interval_does_not_merge(self, databank, case):
        # A stops at 7, B starts at 9: a gap of 2 samples merges below 3 ms, not at 2
        merged = self._run(databank, case, minimum_inter_ripple_interval_ms=3.0)
        assert merged.start_index.tolist()[0] == 4
        assert merged.stop_index.tolist()[0] == 11
        apart = self._run(databank, case, minimum_inter_ripple_interval_ms=2.0)
        assert apart.start_index.tolist() == [4, 49]
        assert apart.stop_index.tolist() == [7, 53]  # B alone has no peak above 5

    def test_merge_has_no_cap(self, databank, case):
        # A+B spans 7 ms; with a 6 ms ceiling the merged event is dropped, not left unmerged
        events = self._run(databank, case, maximum_duration_ms=6.0)
        assert events.start_index.tolist() == [49]

    def test_peak_test_is_strict(self, databank, case):
        events = self._run(databank, case, high_threshold=5.5)
        assert events.start_index.tolist() == [4]

    def test_duration_limits(self, databank, case):
        longer = self._run(databank, case, maximum_duration_ms=15.0)
        assert longer.start_index.tolist() == [4, 29, 49]
        shortest = self._run(databank, case, minimum_duration_ms=5.0)
        assert shortest.start_index.tolist() == [4]

    def test_both_edges_incomplete(self, databank):
        normalized = np.array([3.0, 0, 3.0, 6.0, 0, 0, 3.0])
        events = databank.source_segmentation(
            normalized,
            np.zeros(7),
            np.arange(7) / self.FS,
            low_threshold=2.0,
            high_threshold=5.0,
            minimum_inter_ripple_interval_ms=0.5,
            maximum_duration_ms=10.0,
            frequency=self.FS,
        )
        assert events.start_index.tolist() == [1]
        assert events.stop_index.tolist() == [3]

    def _edges(self, databank, normalized):
        return databank.source_segmentation(
            np.asarray(normalized, dtype=float),
            np.zeros(len(normalized)),
            np.arange(len(normalized)) / self.FS,
            low_threshold=2.0,
            high_threshold=5.0,
            minimum_inter_ripple_interval_ms=0.5,
            maximum_duration_ms=10.0,
            frequency=self.FS,
        )

    def test_only_first_run_incomplete(self, databank):
        # one more stop than starts: the first stop goes, with its run (peak 6)
        events = self._edges(databank, [6.0, 0, 0, 3.0, 6.0, 0, 0])
        assert events.start_index.tolist() == [2]
        assert events.stop_index.tolist() == [4]

    def test_only_last_run_incomplete(self, databank):
        # one more start than stops: the last start goes, with its run (peak 6)
        events = self._edges(databank, [0, 3.0, 6.0, 0, 0, 6.0, 6.0])
        assert events.start_index.tolist() == [0]
        assert events.stop_index.tolist() == [2]

    def test_nothing_above_threshold(self, databank):
        events = databank.source_segmentation(
            np.zeros(10),
            np.zeros(10),
            np.arange(10) / self.FS,
            low_threshold=2.0,
            high_threshold=5.0,
            minimum_inter_ripple_interval_ms=1.0,
            maximum_duration_ms=10.0,
            frequency=self.FS,
        )
        assert events.empty
        assert list(events.columns) == databank.SOURCE_EVENT_COLUMNS

    @pytest.mark.parametrize("kind", ["cheby2", "butter"])
    def test_source_filter_passes_the_band(self, databank, kind):
        code = databank.SourceCode(
            **{
                **databank.PETERSEN_FORK_2021.__dict__,
                "filter_kind": kind,
                "filter_order": 4 if kind == "cheby2" else 3,
            }
        )
        t = np.arange(12_500) / 1250
        inside = databank.source_filter(np.sin(2 * np.pi * 150 * t), [120, 180], code)
        below = databank.source_filter(np.sin(2 * np.pi * 40 * t), [120, 180], code)
        middle = slice(1000, -1000)
        assert np.std(inside[middle]) == pytest.approx(np.sqrt(0.5), rel=0.05)
        assert np.std(below[middle]) < 0.01 * np.sqrt(0.5)

    def test_source_filter_unknown_kind(self, databank):
        code = databank.SourceCode(
            **{**databank.PETERSEN_FORK_2021.__dict__, "filter_kind": "x"}
        )
        with pytest.raises(ValueError, match="unknown filter kind"):
            databank.source_filter(np.zeros(100), [120, 180], code)


_STORED = {"thresholds": [2, 5], "durations": [50, 150], "frequency": 1250, "restrict": []}


@pytest.fixture(scope="module")
def detected(databank):
    """A 20 s source-filtered signal with four bursts, and the package's events on it."""
    rng = np.random.default_rng(3)
    fs, n = 1250.0, 25_000
    t = np.arange(n) / fs
    lfp = rng.normal(0, 50, n)
    for centre in (3.0, 8.0, 13.0, 18.0):
        lfp += 600 * np.exp(-0.5 * ((t - centre) / 0.012) ** 2) * np.sin(2 * np.pi * 150 * t)
    filtered = databank.source_filter(lfp, [120, 180], databank.PETERSEN_FORK_2021)
    options = databank.package_options(_STORED, databank.PETERSEN_FORK_2021)
    timestamps = 1.7e9 + t
    return filtered, timestamps, databank.run_package(filtered, timestamps, options)


@pytest.fixture(scope="module")
def capped(databank):
    """The `detected` signal plus two bursts 100 ms apart, which the source merges into
    one event of more than 150 ms and then drops, while the package's capped merge
    keeps the two fragments."""
    rng = np.random.default_rng(3)
    fs, n = 1250.0, 25_000
    t = np.arange(n) / fs
    lfp = rng.normal(0, 50, n)
    for centre in (3.0, 8.0, 13.0, 18.0):
        lfp += 600 * np.exp(-0.5 * ((t - centre) / 0.012) ** 2) * np.sin(2 * np.pi * 150 * t)
    for centre in (10.5, 10.6):
        lfp += 600 * np.exp(-0.5 * ((t - centre) / 0.02) ** 2) * np.sin(2 * np.pi * 150 * t)
    code = databank.PETERSEN_FORK_2021
    filtered = databank.source_filter(lfp, [120, 180], code)
    normalized, _, _ = databank.normalized_squared_signal(filtered, 11)
    timestamps = 1.7e9 + t
    package = databank.run_package(
        filtered, timestamps, databank.package_options(_STORED, code)
    )
    transcription = databank.source_segmentation(
        normalized,
        filtered,
        timestamps,
        low_threshold=2.0,
        high_threshold=5.0,
        minimum_inter_ripple_interval_ms=50.0,
        maximum_duration_ms=150.0,
        frequency=fs,
    )
    return normalized, filtered, timestamps, package, transcription


class TestPackageRun:
    STORED = _STORED

    def test_merge_cap_is_the_whole_difference(self, databank, capped):
        normalized, filtered, timestamps, package, transcription = capped
        # the source merges the two close bursts (gap under 50 ms) and drops the
        # result for exceeding 150 ms; the package's cap keeps both fragments
        assert len(transcription) == 4
        assert len(package) == 6
        classes = databank.difference_classes(transcription, package)
        assert classes["unmatched_reference"] == []
        assert classes["poorly_aligned"] == []
        extras = package.iloc[classes["unmatched_detected"]]
        assert len(extras) == 2
        assert np.all(extras.start_time.to_numpy() - 1.7e9 > 10.4)
        assert np.all(extras.end_time.to_numpy() - 1.7e9 < 10.7)

        diagnostic = databank.merge_cap_diagnostic(
            normalized, filtered, timestamps, self.STORED, databank.PETERSEN_FORK_2021, extras
        )
        too_long = diagnostic["too_long"]
        assert len(too_long) == 1
        assert (too_long.end_time - too_long.start_time).iloc[0] > 0.15
        assert diagnostic["holder"].tolist() == [0, 0]
        assert diagnostic["ceiling"] == 188  # round half up of 0.15 s * 1250 Hz
        assert len(diagnostic["uncapped"]) == 5  # the four bursts and the merged pair
        kept = diagnostic["uncapped_then_ceiling"]
        np.testing.assert_array_equal(kept.start_time, transcription.start_time)
        np.testing.assert_array_equal(kept.end_time, transcription.end_time)
        np.testing.assert_array_equal(kept.peak_time, transcription.max_power_time)

    def test_options_from_stored_parameters(self, databank):
        options = databank.package_options(self.STORED, databank.PETERSEN_FORK_2021)
        assert options["low_threshold"] == 2.0
        assert options["high_threshold"] == 5.0
        assert options["minimum_inter_ripple_interval"] == 0.05
        assert options["maximum_duration"] == 0.15
        assert options["minimum_duration"] == 0.0
        assert options["speed_threshold"] == np.inf
        assert options["normalization_mask"] is None
        assert round(options["smoothing_window"] * 1250) == 11

    def test_restrict_and_window_refused(self, databank):
        with pytest.raises(ValueError, match="restrict"):
            databank.package_options(
                {**self.STORED, "restrict": [[0, 10]]}, databank.PETERSEN_FORK_2021
            )
        with pytest.raises(ValueError, match="smoothing_window"):
            databank.package_options(
                {**self.STORED, "frequency": 1000}, databank.PETERSEN_FORK_2021
            )

    def test_package_and_transcription_agree_on_isolated_events(self, databank, detected):
        # bursts 5 s apart: the merge cap never binds, so the two rules agree
        filtered, timestamps, package = detected
        normalized, _, _ = databank.normalized_squared_signal(filtered, 11)
        transcription = databank.source_segmentation(
            normalized,
            filtered,
            timestamps,
            low_threshold=2.0,
            high_threshold=5.0,
            minimum_inter_ripple_interval_ms=50.0,
            maximum_duration_ms=150.0,
            frequency=1250.0,
        )
        assert len(package) == len(transcription) == 4
        np.testing.assert_array_equal(package.start_time, transcription.start_time)
        np.testing.assert_array_equal(package.end_time, transcription.end_time)
        np.testing.assert_array_equal(package.peak_time, transcription.max_power_time)

    def test_saved_layout_round_trips(self, databank, detected, tmp_path):
        from ripple_detection.literature_methods import load_events

        package = detected[2]
        sidecar = databank.save_detector_events(package, tmp_path / "package_events.csv")
        assert sidecar == tmp_path / "package_events.json"
        loaded = load_events(tmp_path / "package_events.csv")
        assert list(loaded.columns) == list(package.columns)
        np.testing.assert_array_equal(loaded.start_time, package.start_time)
        assert loaded.attrs["method"] == databank.DETECTOR
        assert loaded.attrs["options"]["speed_threshold"] is None
        assert loaded.attrs["options"]["minimum_duration"] == 0.0


class TestResults:
    ORIGIN = 1.7e9

    def _inventory(self, bounds, peaks):
        bounds = np.asarray(bounds, dtype=float)
        return pd.DataFrame(
            {
                "start_time": self.ORIGIN + bounds[:, 0],
                "end_time": self.ORIGIN + bounds[:, 1],
                "peak_time": self.ORIGIN + np.asarray(peaks, dtype=float),
            }
        )

    def test_comparison_row(self, databank):
        reference = self._inventory([(1.0, 1.1), (2.0, 2.1), (3.0, 3.05)], [1.05, 2.05, 3.02])
        detected = self._inventory([(1.0, 1.1), (2.008, 2.1), (5.0, 5.1)], [1.05, 2.06, 5.05])
        row = databank.comparison_row("a", reference, "b", detected, 0.0, 1250.0, "x vs y")
        assert list(row) == databank.COMPARISON_COLUMNS
        assert row["n_matched"] == 2
        assert row["recall"] == pytest.approx(2 / 3)
        assert row["precision"] == pytest.approx(2 / 3)
        assert row["n_identical_bounds"] == 1
        assert row["onset_error_q75_ms"] == pytest.approx(6.0, abs=1e-3)
        assert row["peak_error_q75_ms"] == pytest.approx(7.5, abs=1e-3)
        assert row["n_unmatched_reference"] == row["n_unmatched_detected"] == 1
        assert row["peak_definitions"] == "x vs y"
        table = pd.DataFrame([row], columns=databank.COMPARISON_COLUMNS)
        assert list(table.columns) == databank.COMPARISON_COLUMNS

    def test_difference_classes(self, databank):
        reference = self._inventory([(1.0, 1.1), (2.0, 2.1), (3.0, 3.1)], [1.05, 2.05, 3.05])
        detected = self._inventory([(1.0, 1.1), (2.0, 2.3), (6.0, 6.1)], [1.05, 2.1, 6.05])
        classes = databank.difference_classes(reference, detected)
        assert classes == {
            "unmatched_reference": [2],
            "unmatched_detected": [2],
            "poorly_aligned": [1],
        }

    def test_write_small_refuses_a_megabyte(self, databank, tmp_path):
        path = databank.write_small("x" * 10, tmp_path / "a.txt")
        assert path.read_text() == "x" * 10
        with pytest.raises(ValueError, match="over the 1000000-byte limit"):
            databank.write_small("x" * 1_000_000, tmp_path / "b.txt")
        assert not (tmp_path / "b.txt").exists()

    def test_dump_json_is_strict(self, databank):
        text = databank.dump_json({"a": np.float64(np.inf), "b": np.arange(2), "c": (1, 2.5)})
        assert json.loads(text) == {"a": None, "b": [0, 1], "c": [1, 2.5]}

    def test_containing_rows_closed_bounds(self, databank):
        outer = self._inventory([(1.0, 2.0), (3.0, 4.0)], [1.5, 3.5])
        inner = self._inventory(
            [(1.0, 1.2), (1.5, 2.0), (2.5, 2.6), (3.9, 4.1), (0.5, 0.6)],
            [1.1, 1.7, 2.55, 4.0, 0.55],
        )
        assert databank.containing_rows(inner, outer).tolist() == [0, 0, -1, -1, -1]


class _FakeData:
    """An NWB data array: shape, dtype, chunks, attrs and slicing."""

    def __init__(self, array):
        self.array = array
        self.shape, self.dtype = array.shape, array.dtype
        self.chunks = (4096, array.shape[1])
        self.attrs = {"conversion": 1.95e-7}

    def __getitem__(self, key):
        return self.array[key]


class _FakeStart:
    def __init__(self):
        self.attrs = {"rate": 1250.0}

    def __getitem__(self, key):
        return 0.0


class _FakeFile:
    def close(self):
        pass


class TestFailureRecords:
    N_CHANNELS, STORED = 8, 5

    @pytest.fixture
    def setup(self, databank, tmp_path, monkeypatch):
        """A session whose inputs are small local files, its DANDI copy faked."""
        import dataclasses

        session = dataclasses.replace(databank.SESSIONS["MS10"], key="TEST")
        cache = tmp_path / "cache"
        monkeypatch.setattr(databank, "RESULTS_DIR", tmp_path / "results")
        folder = cache / "buzsaki" / "TEST"
        folder.mkdir(parents=True)
        base = folder / session.basename
        rng = np.random.default_rng(4)
        lfp = rng.integers(-500, 500, size=(30_000, self.N_CHANNELS)).astype(np.int16)
        Path(f"{base}.lfp.head").write_bytes(lfp[:16].astype("<i2").tobytes())
        _write_xml(Path(f"{base}.xml"), [list(range(self.N_CHANNELS))], n_channels=8)
        scipy.io.savemat(
            f"{base}.session.mat",
            {
                "session": {
                    "channelTags": {"Ripple": {"channels": self.STORED + 1}},
                    "epochs": [
                        {"name": "a", "startTime": 0.0, "stopTime": 12.0},
                        {"name": "b", "startTime": 12.0, "stopTime": 24.0},
                    ],
                }
            },
        )
        state = {"series": {"data": _FakeData(lfp), "starting_time": _FakeStart()}}
        monkeypatch.setattr(
            databank, "_open_lfp", lambda s, c: (_FakeFile(), state["series"], None)
        )
        monkeypatch.setattr(
            databank,
            "SESSIONS",
            {
                "TEST": dataclasses.replace(
                    session,
                    lfp_head=dataclasses.replace(
                        session.lfp_head, length=30_000 * self.N_CHANNELS * 2
                    ),
                )
            },
        )
        session = databank.SESSIONS["TEST"]

        def write_events(channel):
            times, peaks = _times(5)
            params = {
                "thresholds": np.array([2, 5]),
                "durations": np.array([50, 150]),
                "passband": np.array([120, 180]),
                "frequency": 1250,
                "restrict": np.empty((0, 0)),
            }
            if channel is not None:
                params["channel"] = channel
            scipy.io.savemat(
                f"{base}{session.events.suffix}",
                {
                    "ripples": {
                        "times": times,
                        "peaks": peaks,
                        "peakNormedPower": np.arange(5.0),
                        "stdev": 1.0,
                        "noise": np.empty((0, 0)),
                        "detectorName": "bz_FindRipples",
                        "detectorParams": params,
                    }
                },
            )

        return session, cache, state, write_events

    def _inputs(self, databank, session, cache):
        cached = json.loads((cache / "sessions" / session.key / "inputs.json").read_text())
        committed = json.loads(
            (databank.RESULTS_DIR / session.key / "inputs.json").read_text()
        )
        assert cached == committed
        return cached

    def test_verify_passes_on_consistent_inputs(self, databank, setup):
        session, cache, _, write_events = setup
        write_events(self.STORED)
        inputs = databank.step_verify(session, cache)
        assert all(c["passed"] for c in inputs["checks"])
        assert inputs["recording"]["detection_column"] == self.STORED
        assert self._inputs(databank, session, cache)["recording"]["rate"] == 1250.0

    def test_missing_stored_channel_is_recorded(self, databank, setup):
        session, cache, _, write_events = setup
        write_events(None)
        with pytest.raises(databank.InputCheckFailed, match="stores no detection channel"):
            databank.step_verify(session, cache)
        inputs = self._inputs(databank, session, cache)
        assert inputs["stopped_at"] == "verify"
        assert inputs["checks"][0]["name"] == "stored_channel"
        assert not inputs["checks"][0]["passed"]
        with pytest.raises(databank.InputCheckFailed, match="stored_channel"):
            databank._verified_inputs(session, cache)

    def test_timestamps_instead_of_a_rate_are_recorded(self, databank, setup):
        session, cache, state, write_events = setup
        write_events(self.STORED)
        state["series"] = {"data": state["series"]["data"], "timestamps": None}
        assert databank.nwb_rate(state["series"]) is None
        with pytest.raises(databank.InputCheckFailed, match="timestamps rather than a rate"):
            databank.step_verify(session, cache)
        inputs = self._inputs(databank, session, cache)
        assert inputs["checks"][0]["name"] == "nwb_rate"
        assert inputs["checks"][0]["detail"]["shape"] == [30_000, self.N_CHANNELS]

    def test_head_length_mismatch_in_fetch_is_recorded(self, databank, setup, monkeypatch):
        session, cache, _, _ = setup
        monkeypatch.setattr(
            databank,
            "_ensure_capture",
            lambda s, capture, c, head=False: {"original_content_length": 123} if head else {},
        )
        monkeypatch.setattr(databank, "_manifest_record", lambda c, **match: {"kind": "x"})
        with pytest.raises(databank.InputCheckFailed, match="declares 123 bytes"):
            databank.step_fetch(session, cache)
        inputs = self._inputs(databank, session, cache)
        assert inputs["stopped_at"] == "fetch"
        assert inputs["checks"][0]["name"] == "lfp_head_length"
        assert inputs["checks"][0]["detail"] == {
            "declared": 123,
            "table": session.lfp_head.length,
        }

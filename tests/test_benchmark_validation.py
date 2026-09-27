"""The benchmark's simulator validation (examples/benchmark/validate_simulator.py):
measurements on hand-built envelopes and spike trains, the pooling of targets and
readiness, a short report built without any detector, the preflight a benchmark
run makes against a report, and the simulation fingerprint."""

import ast
import hashlib
import importlib
import json
import re
import shutil
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ripple_detection as rd
from ripple_detection import evaluate, literature_methods, registry

UNIX_ORIGIN = 1_700_000_000.0
DURATION = 120.0  # two noise-modulation periods, so every check is measurable
OVERRIDES = {"session.duration_s": DURATION}
PACKAGE = Path(rd.__file__).resolve().parent


@pytest.fixture(scope="module")
def validate(benchmark_import):
    return benchmark_import("validate_simulator")


@pytest.fixture(scope="module")
def conditions(benchmark_import):
    return benchmark_import("conditions")


class _Refuses(Mapping):
    """A detector registry that fails on any lookup."""

    def __getitem__(self, name):
        msg = f"detector {name} looked up while validating the simulator"
        raise AssertionError(msg)

    def __iter__(self) -> Iterator[str]:
        msg = "detector registry read while validating the simulator"
        raise AssertionError(msg)

    def __len__(self) -> int:
        msg = "detector registry read while validating the simulator"
        raise AssertionError(msg)


def _refuse(name):
    def refused(*args, **kwargs):
        msg = f"{name} called while validating the simulator"
        raise AssertionError(msg)

    return refused


@pytest.fixture(scope="module")
def report(validate, tmp_path_factory):
    """A one-replicate reference report, built with every detector, the
    literature methods and event matching made to fail if called."""
    root = tmp_path_factory.mktemp("validation")
    with pytest.MonkeyPatch.context() as patch:
        for name, spec in rd.DETECTORS.items():
            patch.setattr(rd, name, _refuse(name))
            patch.setattr(rd.detectors, name, _refuse(name))
            patch.setattr(
                importlib.import_module(spec.detector.__module__), name, _refuse(name)
            )
        patch.setattr(rd, "DETECTORS", _Refuses())
        patch.setattr(registry, "DETECTORS", _Refuses())
        patch.setattr(literature_methods, "run_method", _refuse("run_method"))
        patch.setattr(evaluate, "match_events", _refuse("match_events"))
        patch.setattr(rd, "match_events", _refuse("match_events"))
        exit_code = validate.main(
            [
                "--validation-id",
                "short",
                "--conditions",
                "reference",
                "--replicates",
                "1",
                "--duration",
                str(DURATION),
                "--output-root",
                str(root),
                "--no-figures",
            ]
        )
    directory = root / "short"
    spec = json.loads((directory / "spec.json").read_text())
    return {"directory": directory, "spec": spec, "exit_code": exit_code}


@pytest.fixture
def ready_copy(report, tmp_path):
    """The short report copied, its status set to ready and its replicates
    to the predeclared count."""
    directory = tmp_path / "copy"
    shutil.copytree(report["directory"], directory)
    spec = json.loads((directory / "spec.json").read_text())
    replicates = list(range(10000, 10020))
    spec.update(status="ready", reasons=[], replicates=replicates)
    (directory / "spec.json").write_text(json.dumps(spec))
    return directory


@pytest.fixture
def resolved(validate, conditions):
    reference = conditions.conditions()[0]
    return {"reference": conditions.resolve(reference, OVERRIDES)}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TestMeasurements:
    """Hand-computed widths, participation, silent gaps and intervals, at a
    Unix-time clock origin where the helpers read timestamps."""

    def test_envelope_widths_by_hand(self, validate):
        time = UNIX_ORIGIN + np.arange(9) / 1000.0
        envelope = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 4.0, 2.0, 1.0, 0.0])
        widths = validate.envelope_widths(time, envelope, (0.5, 0.1))
        # half maximum 4: samples 3-5; 10% of peak 0.8: samples 1-7
        np.testing.assert_allclose(widths, [0.002, 0.006], atol=1e-6)

    @pytest.mark.parametrize("power", [2, 4])
    def test_sampled_envelope_widths_match_the_threshold_distance(self, validate, power):
        rate, sigma = 1500.0, 0.012
        time = UNIX_ORIGIN + np.arange(-300, 301) / rate
        offset = np.arange(-300, 301) / rate
        envelope = np.exp(
            -np.log(2) * (np.abs(offset) / (np.sqrt(2 * np.log(2)) * sigma)) ** power
        )
        widths = validate.envelope_widths(time, envelope, (0.5, 0.1))
        expected = [2 * validate.threshold_distance(f, power) * sigma for f in (0.5, 0.1)]
        # closed sample bounds fall inside the continuous crossing, within a sample each side
        assert np.all(widths <= np.asarray(expected) + 1e-6)
        assert np.all(widths >= np.asarray(expected) - 2 / rate - 1e-6)

    def test_threshold_distance(self, validate):
        assert validate.threshold_distance(0.5, 2) == pytest.approx(np.sqrt(2 * np.log(2)))
        assert validate.threshold_distance(0.5, 4) == pytest.approx(np.sqrt(2 * np.log(2)))
        assert validate.threshold_distance(0.1, 2) == pytest.approx(np.sqrt(2 * np.log(10)))

    def test_observed_participation_by_hand(self, validate):
        time = UNIX_ORIGIN + np.arange(400) / 1000.0
        # units 0-2 are the candidates, unit 3 is not; a centre at sample 100
        samples = np.array([80, 125, 126, 100, 300])
        units = np.array([0, 1, 2, 3, 0])
        centers = time[[100, 250]]
        fractions = validate.observed_participation(
            time, samples, units, np.array([0, 1, 2]), centers, 0.025
        )
        # 20 ms before and exactly 25 ms after count, 26 ms after does not;
        # nothing within 25 ms of the second centre
        np.testing.assert_allclose(fractions, [2 / 3, 0.0])

    def test_observed_participation_of_no_units(self, validate):
        time = np.arange(10) / 1000.0
        fractions = validate.observed_participation(
            time, np.array([1]), np.array([0]), np.array([], dtype=int), time[[2]], 0.01
        )
        assert np.isnan(fractions).all()

    def test_silent_gaps_by_hand(self, validate):
        time = UNIX_ORIGIN + np.arange(100) / 1000.0
        samples = np.array([10, 12, 12, 20, 55, 60])  # a sample may hold two units' spikes
        intervals = np.array([[time[5], time[30]], [time[50], time[70]]])
        gaps = validate.silent_gaps(time, samples, intervals)
        # the gap from 20 to 55 spans the two intervals and is left out
        np.testing.assert_allclose(gaps, [0.002, 0.008, 0.005], atol=1e-6)

    def test_unit_intervals_stay_within_one_unit_and_one_stretch(self, validate):
        time = np.arange(100) / 1000.0
        samples = np.array([10, 14, 11, 60, 65, 70])
        units = np.array([0, 0, 1, 1, 0, 0])
        rest = np.array([[0.005, 0.030], [0.050, 0.080]])
        intervals = validate._unit_intervals(time, samples, units, rest)
        np.testing.assert_allclose(np.sort(intervals), [0.004, 0.005])

    def test_run_around(self, validate):
        values = np.array([0.0, 2.0, 3.0, 2.0, 0.0, 3.0])
        assert validate.run_around(values, 2, 2.0) == (1, 3)
        assert validate.run_around(values, 2, 2.0, strict=True) == (2, 2)
        assert validate.run_around(values, 5, 1.0) == (5, 5)
        assert validate.run_around(values, 0, 1.0) is None

    def test_intervals(self, validate):
        union = validate.interval_union(
            np.array([[5.0, 6.0], [0.0, 2.0], [1.0, 3.0], [3.0, 4.0]])
        )
        np.testing.assert_array_equal(union, [[0.0, 4.0], [5.0, 6.0]])
        time = UNIX_ORIGIN + np.arange(10.0)
        mask = validate.interval_mask(time, UNIX_ORIGIN + np.array([[7.0, 8.0], [1.0, 3.0]]))
        np.testing.assert_array_equal(np.flatnonzero(mask), [1, 2, 3, 7, 8])
        rest = validate.rest_intervals(time, UNIX_ORIGIN + np.array([[3.0, 4.0], [6.5, 9.5]]))
        np.testing.assert_allclose(rest - UNIX_ORIGIN, [[1.0, 3.0], [4.0, 6.5]])

    def test_moving_rms_of_a_constant(self, validate):
        rms = validate.moving_rms(np.full(20, -3.0), 5)
        np.testing.assert_allclose(rms[2:-2], 3.0)

    def test_fft_peak_and_instantaneous_frequency_of_a_tone(self, validate):
        rate = 1500.0
        time = UNIX_ORIGIN + np.arange(1500) / rate
        tone = np.sin(2 * np.pi * 180.0 * np.arange(1500) / rate) * np.hanning(1500)
        assert validate.fft_peak_frequency(tone, rate) == pytest.approx(180.0, abs=0.25)
        frequency = validate.instantaneous_frequency(time, tone)
        np.testing.assert_allclose(frequency[500:1000], 180.0, atol=0.5)

    def test_fractional_shift_of_a_band_limited_burst(self, validate):
        n = np.arange(400)

        def burst(shift):
            u = n - 200 - shift
            return np.exp(-0.5 * (u / 20) ** 2) * np.sin(2 * np.pi * 0.12 * u)

        np.testing.assert_allclose(
            validate.fractional_shift(burst(0.0), 2.4), burst(2.4), atol=1e-9
        )

    def test_refractory_rate(self, validate):
        step = 1 / 1500
        p = -np.expm1(-10.0 * step)
        # 2 ms is three samples at 1500 Hz: the two samples after a spike are blocked
        assert validate.refractory_rate(np.array([10.0]), step, 0.002)[0] == pytest.approx(
            p / (step * (1 + 2 * p))
        )
        assert validate.refractory_rate(np.array([10.0]), step, 0.0)[0] == pytest.approx(
            p / step
        )

    def test_isolated_groups_do_not_overlap(self, validate):
        rows = pd.DataFrame(
            {
                "center_time": [1.0, 1.1, 1.2, 3.0],
                "rise_sigma": [0.01] * 4,
                "decay_sigma": [0.01] * 4,
            }
        )
        groups = validate.isolated_groups(rows, 0.0)
        assert sorted(i for g in groups for i in g.index) == [0, 1, 2, 3]
        for group in groups:
            starts = group.center_time - 0.08
            ends = group.center_time + 0.08
            assert (starts.to_numpy()[1:] > ends.to_numpy()[:-1]).all()
        # rows 0 and 2 each overlap row 1 only: two groups suffice
        assert len(groups) == 2


class TestTargets:
    def test_every_target_has_a_measurement(self, validate):
        assert validate.target_table_problems(validate.load_targets()) == []

    def test_target_table_problems(self, validate):
        targets = validate.load_targets().iloc[:2].copy()
        targets.loc[0, "evidence_status"] = "verify"
        targets.loc[1, "quantity"] = "ripple_amplitude"
        targets.loc[1, "target_statistic"] = "mode_hz"
        targets.loc[1, "conditions"] = "reference; sleep"
        problems = validate.target_table_problems(targets)
        assert any("'verify' is unresolved" in p for p in problems)
        assert any("no measurement is defined" in p for p in problems)
        assert any("no pooling rule" in p for p in problems)
        assert any("sleep" in p for p in problems)

    def test_target_conditions(self, validate):
        assert validate.target_conditions("reference; coupled; quartic") == (
            "reference",
            "strength_correlation=coupled",
            "envelope_power=quartic",
        )
        assert validate.target_conditions("none (state differs; reported beside it)") == ()
        with pytest.raises(ValueError, match="Unknown condition labels"):
            validate.target_conditions("reference; sleep")

    def test_the_labels_name_conditions(self, validate, conditions):
        ids = {c.condition_id for c in conditions.conditions()}
        assert set(validate.MODEL_LABELS.values()) <= ids

    @pytest.mark.parametrize(
        ("statistic", "expected"),
        [
            ("rate_per_s", (2 + 4) / (10 + 20)),
            ("median_ms", 3.0),
            ("mean_fraction", 3.0),
            ("p95_fraction", np.percentile([2.0, 4.0], 95)),
        ],
    )
    def test_pooling(self, validate, statistic, expected):
        samples = pd.DataFrame(
            {"quantity": "q", "group": "g", "x": [2.0, 4.0, np.nan], "y": [10.0, 20.0, 0.0]}
        )
        if statistic == "rate_per_s":
            samples = samples.iloc[:2]
        value, _ = validate._pooling(statistic)(samples)
        assert value == pytest.approx(expected)

    def test_ratio_and_correlation_pooling(self, validate):
        ratio = pd.DataFrame(
            {
                "group": ["window", "window", "baseline"],
                "x": [3.0, 1.0, 10.0],
                "y": [1.0, 1.0, 20.0],
            }
        )
        assert validate._pooling("ratio")(ratio)[0] == pytest.approx((4 / 2) / (10 / 20))
        pairs = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 9.0]})
        assert validate._pooling("pearson_r")(pairs)[0] == pytest.approx(
            np.corrcoef(pairs.x, pairs.y)[0, 1]
        )
        assert validate._pooling("spearman_r")(pairs)[0] == pytest.approx(1.0)
        assert np.isnan(validate._pooling("pearson_r")(pairs.iloc[:2])[0])

    def test_readiness_gates_only_applicable_checks(self, validate):
        checks = pd.DataFrame(
            {
                "check": ["a", "b", "c", "d"],
                "kind": ["target", "target", "rendering", "target"],
                "condition_id": "reference",
                "applies": [True, False, True, True],
                "statistic": "median",
                "observed": [5.0, 50.0, 1.0, 40.0 - 1e-12],
                "lower": [0.0, 0.0, 0.0, 40.0],
                "upper": [10.0, 10.0, 0.0, 60.0],
                "note": "",
                "passed": [True, False, False, True],
            }
        )
        reasons = validate.readiness(checks, validate.load_targets())
        assert len(reasons) == 1
        assert "rendering check c" in reasons[0]


class TestRenderingChecks:
    """Which rendering checks gate a condition, and that nothing measured, or
    a NaN in any replicate, fails a check that applies."""

    CONDITIONS = (
        "reference",
        "n_channels=1",
        "fast_gamma_rate=0",
        "spike_model=refractory",
    )

    @staticmethod
    def _session(validate, condition_id, replicate, changed):
        """A session whose every rendering check passes with n = 5, but for
        ``changed``: check -> (observed, n)."""
        rows = dict.fromkeys(validate.RENDERING_CHECKS, (0.0, 5))
        del rows["interneuron_rate_realization"]  # pooled from the samples
        if condition_id != "spike_model=refractory":
            del rows["refractory_spiking"]  # measured under that model alone
        rows.update(changed)
        return validate.SessionResult(
            condition_id=condition_id,
            replicate=replicate,
            samples=pd.DataFrame(
                [("interneuron_rate_realization", "interneuron", 50.0, 50.0)],
                columns=["quantity", "group", "x", "y"],
            ),
            measurements=pd.DataFrame(),
            checks=pd.DataFrame(
                [(name, observed, n, "") for name, (observed, n) in rows.items()],
                columns=["check", "observed", "n", "note"],
            ),
        )

    @pytest.fixture
    def checks(self, validate, conditions):
        by_id = {c.condition_id: c for c in conditions.conditions()}
        selected = [by_id[i] for i in self.CONDITIONS]
        changed = {
            ("reference", 1): {"ripple_sizing": (np.nan, 5)},
            ("n_channels=1", 0): {"channel_profile_rendering": (0.0, 0)},
            ("n_channels=1", 1): {"channel_profile_rendering": (0.0, 0)},
            ("fast_gamma_rate=0", 0): {"gamma_sizing": (0.0, 0)},
            ("fast_gamma_rate=0", 1): {"gamma_sizing": (0.0, 0)},
            ("spike_model=refractory", 0): {"sharp_wave_truth_crossings": (0.0, 0)},
            ("spike_model=refractory", 1): {"sharp_wave_truth_crossings": (0.0, 0)},
        }
        results = [
            self._session(validate, c, r, changed.get((c, r), {}))
            for c in self.CONDITIONS
            for r in (0, 1)
        ]
        parameters = {c.condition_id: conditions.resolve(c) for c in selected}
        found = validate.build_checks(
            results, selected, parameters, validate.load_targets().iloc[:0]
        )
        return found.set_index(["condition_id", "check"])

    def test_every_check_has_a_row_saying_whether_it_applies(self, validate, checks):
        for condition_id in self.CONDITIONS:
            assert set(checks.loc[condition_id].index) == set(validate.RENDERING_CHECKS)
        not_applicable = {
            (c, name)
            for c in self.CONDITIONS
            for name in validate.RENDERING_CHECKS
            if not checks.loc[(c, name), "applies"]
        }
        assert not_applicable == {
            ("reference", "refractory_spiking"),
            ("n_channels=1", "refractory_spiking"),
            ("fast_gamma_rate=0", "refractory_spiking"),
            ("n_channels=1", "channel_profile_rendering"),
            ("fast_gamma_rate=0", "gamma_sizing"),
        }
        assert (
            "one channel" in checks.loc[("n_channels=1", "channel_profile_rendering"), "note"]
        )
        assert "no gamma" in checks.loc[("fast_gamma_rate=0", "gamma_sizing"), "note"]

    def test_nothing_measured_or_a_nan_fails(self, validate, checks):
        gated = checks[checks.applies]
        failed = set(gated.index[~gated.passed])
        assert failed == {
            ("reference", "ripple_sizing"),
            ("spike_model=refractory", "sharp_wave_truth_crossings"),
        }
        assert np.isnan(checks.loc[("reference", "ripple_sizing"), "observed"])
        row = checks.loc[("spike_model=refractory", "sharp_wave_truth_crossings")]
        assert (row.n, row.observed) == (0, 0.0)
        assert "nothing measured" in row.note
        reasons = validate.readiness(checks.reset_index(), validate.load_targets().iloc[:0])
        assert len(reasons) == 2


class TestReport:
    def test_it_is_built_without_a_detector(self, report, validate):
        # the fixture failed any detector, literature method or matching call;
        # the module does not import them either
        tree = ast.parse(Path(validate.__file__).read_text())
        imported = {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert (
            not {
                "ripple_detection.detectors",
                "ripple_detection.registry",
                "ripple_detection.literature_methods",
                "ripple_detection.evaluate",
                "recipe_configs",
                "run",
            }
            & imported
        )
        assert report["spec"]["status"] in ("ready", "not_ready")
        assert report["exit_code"] == (0 if report["spec"]["status"] == "ready" else 1)

    def test_spec_records_settings_seeds_and_hashes(self, report, validate, resolved):
        spec, directory = report["spec"], report["directory"]
        assert spec["conditions"] == json.loads(json.dumps(resolved))
        assert spec["replicates"] == [validate.FIRST_REPLICATE]
        assert spec["seeds"] == {"10000": validate.session_seed(10000)}
        assert spec["simulation_fingerprint"] == validate.simulation_fingerprint()
        assert spec["target_table_hash"] == _sha256(validate.TARGETS)
        files = {p.name for p in directory.iterdir()} - {"spec.json"}
        assert (
            files == set(spec["artifacts"]) == {"measurements.csv", "checks.csv", "report.md"}
        )
        for name, digest in spec["artifacts"].items():
            assert _sha256(directory / name) == digest
        assert (spec["status"] == "ready") == (not spec["reasons"])

    def test_parameter_revisions_are_listed(self, report, validate, conditions):
        assert report["spec"]["reference_revisions"] == []
        text = (report["directory"] / "report.md").read_text()
        assert "## Parameter revisions\n\nNone:" in text
        revision = conditions.ReferenceRevision(
            "events.ripple_duration",
            (0.03, 0.15),
            (0.03, 0.1),
            "calibrated span",
            "calibration c1",
        )
        assert validate.revision_records([revision]) == [
            {
                "key": "events.ripple_duration",
                "previous": [0.03, 0.15],
                "revised": [0.03, 0.1],
                "reason": "calibrated span",
                "evidence": "calibration c1",
            }
        ]
        lines = validate._revision_lines([revision])
        assert lines[-1] == (
            "| `events.ripple_duration` | [0.03, 0.15] | [0.03, 0.1] | calibrated span | "
            "calibration c1 |"
        )

    def test_checks_cover_every_target_and_rendering_check(self, report, validate):
        checks = pd.read_csv(report["directory"] / "checks.csv")
        targets = validate.load_targets()
        assert set(checks.check) == set(targets.quantity) | set(validate.RENDERING_CHECKS)
        assert (checks.condition_id == "reference").all()
        gated = checks[checks.kind == "target"].set_index("check").applies
        for row in targets.itertuples():
            applies = row.evidence_status == "supported" and "reference" in str(
                row.conditions
            ).split("; ")
            assert gated[row.quantity] == applies, row.quantity
        rendering = checks[checks.kind == "rendering"].set_index("check")
        # Poisson spiking has no refractory period; every other check applies,
        # measured something and passes on the reference
        assert list(rendering.index[~rendering.applies]) == ["refractory_spiking"]
        applicable = rendering[rendering.applies]
        assert applicable.passed.all(), applicable[~applicable.passed]
        assert (applicable.n > 0).all()

    def test_status_follows_the_gated_checks(self, report):
        checks = pd.read_csv(report["directory"] / "checks.csv")
        failing = checks[checks.applies & ~checks.passed]
        assert (report["spec"]["status"] == "ready") == failing.empty
        assert len(report["spec"]["reasons"]) == len(failing)

    def test_measurements_are_per_condition_replicate_and_group(self, report):
        measurements = pd.read_csv(report["directory"] / "measurements.csv")
        assert list(measurements.columns) == [
            "condition_id",
            "replicate",
            "group",
            "quantity",
            "statistic",
            "value",
            "n",
        ]
        assert set(measurements.replicate) == {10000}
        quantities = set(measurements.quantity)
        for quantity in (
            "event_rate",
            "event_type_proportion",
            "latent_width_ms_half_maximum",
            "ripple_hilbert_width_ms_0.5",
            "ripple_snr_anchor",
            "ripple_snr_recording_wide",
            "background_band_power",
            "windowed_ripple_band_power",
            "channel_occupancy",
            "channel_coherence",
            "count_fano_10ms",
            "inter_spike_interval_ms",
            "population_silent_gap_ms",
            "observed_participation",
            "latent_recruitment",
            "sharp_wave_ripple_power",
        ):
            assert quantity in quantities, quantity

    def test_participation_counts_spikes_not_recruitment(self, validate, conditions):
        reference = conditions.conditions()[0]
        result = validate.measure_session(reference, validate.FIRST_REPLICATE, OVERRIDES)
        session = conditions.simulate_condition(reference, validate.FIRST_REPLICATE, OVERRIDES)
        events = session.events
        first = events[(events.expression == "ripple") & (events.component == 0)]
        principal = np.flatnonzero(np.isin(session.unit_types, ["place", "pyramidal"]))
        samples, units = np.nonzero(session.multiunit)
        expected = validate.observed_participation(
            session.time, samples, units, principal, first.center_time.to_numpy(), 0.025
        )
        observed = result.samples[result.samples.quantity == "observed_participation"]
        by_type = np.concatenate(
            [expected[first.event_type.to_numpy() == t] for t in observed.group.unique()]
        )
        np.testing.assert_allclose(observed.x.to_numpy(), by_type)
        bursts = events[events.expression == "burst"].set_index("event_id")
        latent = bursts.loc[first.event_id, "n_participants"].to_numpy() / principal.size
        assert not np.allclose(expected, latent)

    def test_rendering_seed_reproduces_the_session(self, validate, conditions):
        reference = conditions.conditions()[0]
        overrides = {"session.duration_s": 40.0}
        session = conditions.simulate_condition(reference, 3, overrides)
        again = rd.simulate_network_session(
            session.time,
            session.events,
            non_events=session.non_events,
            running_intervals=session.running_intervals,
            rng=np.random.default_rng(validate._render_seed(3)),
            sampling_frequency=session.sampling_frequency,
            **conditions.resolve(reference, overrides)["render"],
        )
        np.testing.assert_array_equal(again.lfps, session.lfps)
        np.testing.assert_array_equal(again.multiunit, session.multiunit)

    def test_the_command_line_takes_a_crossed_cell(self, validate, tmp_path, monkeypatch):
        chosen = []

        def validated(validation_id, selected, **options):
            chosen.append([c.condition_id for c in selected])
            spec = tmp_path / "spec.json"
            spec.write_text(json.dumps({"status": "ready", "reasons": []}))
            return spec

        monkeypatch.setattr(validate, "validate", validated)
        for text in ("ripple_snr=low,participation=low", "reference,participation=low"):
            assert validate.main(["--validation-id", "x", "--conditions", text]) == 0
        assert chosen == [
            ["ripple_snr=low,participation=low"],
            ["reference", "participation=low"],
        ]
        with pytest.raises(SystemExit):
            validate.main(["--validation-id", "x", "--conditions", "reference,nope"])
        assert len(chosen) == 2

    def test_validate_rejects_empty_requests(self, validate, conditions, tmp_path):
        with pytest.raises(ValueError, match="n_replicates"):
            validate.validate(
                "x",
                conditions.conditions()[:1],
                n_replicates=0,
                output_root=tmp_path,
            )
        with pytest.raises(ValueError, match="No condition"):
            validate.validate("x", (), output_root=tmp_path)


class TestPreflight:
    def test_a_matching_report_passes(self, validate, ready_copy, resolved):
        digest = validate.require_ready_report(ready_copy / "spec.json", resolved)
        assert digest == _sha256(ready_copy / "spec.json")

    def test_a_missing_report(self, validate, tmp_path, resolved):
        with pytest.raises(validate.ReportNotReady, match="No simulator validation report"):
            validate.require_ready_report(tmp_path / "spec.json", resolved)
        (tmp_path / "spec.json").write_text("{not json")
        with pytest.raises(validate.ReportNotReady, match="cannot be read"):
            validate.require_ready_report(tmp_path / "spec.json", resolved)

    def test_too_few_replicates(self, validate, ready_copy, resolved):
        assert validate.DEFAULT_REPLICATES == 20
        spec = json.loads((ready_copy / "spec.json").read_text())
        spec["replicates"] = spec["replicates"][:19]
        (ready_copy / "spec.json").write_text(json.dumps(spec))
        with pytest.raises(
            validate.ReportNotReady, match=re.escape("19 replicates, fewer than the 20")
        ):
            validate.require_ready_report(ready_copy / "spec.json", resolved)

    def test_a_failed_report(self, validate, ready_copy, resolved):
        spec = json.loads((ready_copy / "spec.json").read_text())
        spec.update(status="not_ready", reasons=["target check x fails in reference"])
        (ready_copy / "spec.json").write_text(json.dumps(spec))
        with pytest.raises(validate.ReportNotReady, match="target check x fails"):
            validate.require_ready_report(ready_copy / "spec.json", resolved)

    def test_stale_simulation_source(
        self, validate, ready_copy, resolved, tmp_path, monkeypatch
    ):
        package = tmp_path / "package"
        shutil.copytree(PACKAGE, package, ignore=shutil.ignore_patterns("__pycache__"))
        simulate = package / "simulate.py"
        simulate.write_text(
            simulate.read_text().replace("_SPIKE_BLOCK = 8", "_SPIKE_BLOCK = 4")
        )
        monkeypatch.setattr(validate, "PACKAGE", package)
        with pytest.raises(validate.ReportNotReady, match="simulation code has changed"):
            validate.require_ready_report(ready_copy / "spec.json", resolved)

    def test_changed_targets(self, validate, ready_copy, resolved, tmp_path):
        edited = tmp_path / "targets.csv"
        edited.write_text(validate.TARGETS.read_text().replace(",0.13,0.40,", ",0.10,0.40,"))
        spec = json.loads((ready_copy / "spec.json").read_text())
        spec["target_table_hash"] = validate.target_table_hash(edited)
        (ready_copy / "spec.json").write_text(json.dumps(spec))
        with pytest.raises(
            validate.ReportNotReady, match=re.escape("simulator_targets.csv has changed")
        ):
            validate.require_ready_report(ready_copy / "spec.json", resolved)

    def test_uncovered_settings(self, validate, conditions, ready_copy, resolved):
        local = next(
            c for c in conditions.conditions() if c.condition_id == "spatial_profile=local"
        )
        longer = conditions.resolve(conditions.conditions()[0], {"session.duration_s": 600.0})
        wanted = {**resolved, "spatial_profile=local": conditions.resolve(local, OVERRIDES)}
        with pytest.raises(
            validate.ReportNotReady, match="spatial_profile=local is not covered"
        ):
            validate.require_ready_report(ready_copy / "spec.json", wanted)
        with pytest.raises(
            validate.ReportNotReady, match=re.escape("other settings: session.duration_s")
        ):
            validate.require_ready_report(ready_copy / "spec.json", {"reference": longer})

    def test_tampered_artifacts(self, validate, ready_copy, resolved):
        checks = ready_copy / "checks.csv"
        checks.write_text(checks.read_text().replace("True", "False", 1))
        (ready_copy / "measurements.csv").unlink()
        with pytest.raises(validate.ReportNotReady) as raised:
            validate.require_ready_report(ready_copy / "spec.json", resolved)
        message = str(raised.value)
        assert "artifact checks.csv does not match" in message
        assert "artifact measurements.csv is missing" in message

    def test_every_problem_is_listed_at_once(self, validate, ready_copy, resolved):
        spec = json.loads((ready_copy / "spec.json").read_text())
        spec.update(
            status="not_ready", reasons=["r"], simulation_fingerprint="0" * 64, artifacts={}
        )
        (ready_copy / "spec.json").write_text(json.dumps(spec))
        with pytest.raises(validate.ReportNotReady) as raised:
            validate.require_ready_report(ready_copy / "spec.json", {**resolved, "other": {}})
        message = str(raised.value)
        for part in (
            "status is 'not_ready'",
            "simulation code has changed",
            "records no artifact hashes",
            "condition other is not covered",
        ):
            assert part in message
        assert isinstance(raised.value, ValueError)


class TestFingerprint:
    @pytest.fixture
    def package(self, validate, tmp_path, monkeypatch):
        copy = tmp_path / "ripple_detection"
        shutil.copytree(PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__"))
        conditions_copy = tmp_path / "conditions.py"
        shutil.copy(validate.CONDITIONS_SOURCE, conditions_copy)
        monkeypatch.setattr(validate, "PACKAGE", copy)
        monkeypatch.setattr(validate, "CONDITIONS_SOURCE", conditions_copy)
        return copy

    @staticmethod
    def _edit(path, old, new):
        text = path.read_text()
        assert text.count(old) == 1, old
        path.write_text(text.replace(old, new))

    def test_a_copy_has_the_same_fingerprint(self, validate, tmp_path, monkeypatch):
        real = validate.simulation_fingerprint()
        assert len(real) == 64
        copy = tmp_path / "elsewhere" / "ripple_detection"
        shutil.copytree(PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__"))
        monkeypatch.setattr(validate, "PACKAGE", copy)
        assert validate.simulation_fingerprint() == real

    def test_the_covered_sources(self, validate, package):
        labels = [
            label
            for label, _ in validate._simulation_sources(package, validate.CONDITIONS_SOURCE)
        ]
        assert labels[:3] == [
            "conditions.py",
            "ripple_detection/simulate.py",
            "ripple_detection/ripplefilter.mat",
        ]
        assert "ripple_detection.core:filter_ripple_band" in labels
        assert "ripple_detection.core:_fir_filtfilt" in labels
        assert not [label for label in labels if "detectors._lfp" in label]
        assert not [label for label in labels if "_call_hints" in label]

    def test_detector_edits_leave_it(self, validate, package):
        before = validate.simulation_fingerprint()
        self._edit(
            package / "detectors" / "_lfp.py",
            "def Kay_ripple_detector(",
            "def _unused() -> None:\n    pass\n\n\ndef Kay_ripple_detector(",
        )
        self._edit(
            package / "core.py",
            "def exclude_close_events(",
            "def _unused() -> None:\n    pass\n\n\ndef exclude_close_events(",
        )
        self._edit(
            package / "_call_hints.py",
            "def explain_call_errors(",
            "def _unused() -> None:\n    pass\n\n\ndef explain_call_errors(",
        )
        assert validate.simulation_fingerprint() == before

    @pytest.mark.parametrize(
        ("path", "old", "new"),
        [
            ("simulate.py", "_SPIKE_BLOCK = 8", "_SPIKE_BLOCK = 4"),
            (
                "core.py",
                "padlen = len(filter_numerator) - 1",
                "padlen = len(filter_numerator)",
            ),
            ("core.py", "DEFAULT_TRANSITION_WIDTH = 25.0", "DEFAULT_TRANSITION_WIDTH = 20.0"),
            ("conditions", '"duration_s": 600.0', '"duration_s": 300.0'),
        ],
    )
    def test_simulation_edits_change_it(self, validate, package, path, old, new):
        before = validate.simulation_fingerprint()
        target = validate.CONDITIONS_SOURCE if path == "conditions" else package / path
        self._edit(target, old, new)
        assert validate.simulation_fingerprint() != before

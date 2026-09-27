"""Matching event inventories, pair metrics, agreement and consensus."""

import itertools

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import spearmanr

import ripple_detection as rd
from ripple_detection.evaluate import (
    COMPARISON_COLUMNS,
    PAIR_COLUMNS,
    compare_detectors,
    consensus_counts,
    label_by_overlap,
    match_events,
)

FS = 1500.0
GOOD = np.array([[0.0, 1.0]])
WINDOWS = pd.DataFrame(
    {
        "start_time": [0.0, 1.0, 5.0, 5.0],
        "end_time": [2.0, 4.0, 6.0, 6.0],
        "label": ["swr", "emg", "gamma", "theta"],
    }
)


def pair_list(matching):
    """The matched pairs as ``(reference_index, detected_index)`` tuples."""
    return list(
        zip(
            matching.pairs.reference_index.tolist(),
            matching.pairs.detected_index.tolist(),
            strict=True,
        )
    )


def iou(a, b):
    intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return intersection / ((a[1] - a[0]) + (b[1] - b[0]) - intersection)


def best_assignment(reference, detected, minimum_iou=0.0):
    """The most pairs, then the largest summed IoU, by trying every one-to-one
    assignment of the overlapping pairs: the definition, not the solver."""
    candidates = [
        (r, d, iou(reference[r], detected[d]))
        for r, d in itertools.product(range(len(reference)), range(len(detected)))
        if min(reference[r][1], detected[d][1]) > max(reference[r][0], detected[d][0])
        and iou(reference[r], detected[d]) > minimum_iou
    ]
    best = (0, 0.0)
    for size in range(1, min(len(reference), len(detected)) + 1):
        for chosen in itertools.combinations(candidates, size):
            rows = {r for r, _, _ in chosen}
            columns = {d for _, d, _ in chosen}
            if len(rows) == len(columns) == size:
                best = max(best, (size, sum(value for _, _, value in chosen)))
    return best


class TestMatchEvents:
    def test_identical_inventories(self):
        events = np.array([[4.0, 5.0], [0.0, 1.0], [2.0, 2.5]])
        matching = match_events(events, events)
        assert pair_list(matching) == [(0, 0), (1, 1), (2, 2)]
        np.testing.assert_array_equal(
            matching.pairs[["iou", "coverage", "temporal_precision"]], 1.0
        )
        np.testing.assert_array_equal(matching.pairs[["onset_error", "offset_error"]], 0.0)
        assert (matching.recall, matching.precision, matching.f1) == (1.0, 1.0, 1.0)

    @pytest.mark.parametrize(
        ("n_reference", "n_detected", "recall", "precision", "f1"),
        [
            (0, 2, np.nan, 0.0, 0.0),
            (2, 0, 0.0, np.nan, 0.0),
            (0, 0, np.nan, np.nan, np.nan),
        ],
    )
    def test_empty_inputs(self, n_reference, n_detected, recall, precision, f1):
        events = np.array([[0.0, 1.0], [2.0, 3.0]])
        matching = match_events(events[:n_reference], events[:n_detected])
        assert list(matching.pairs.columns) == list(PAIR_COLUMNS)
        assert matching.pairs.empty
        np.testing.assert_array_equal(matching.reference_overlaps, np.zeros(n_reference))
        np.testing.assert_array_equal(matching.detected_overlaps, np.zeros(n_detected))
        np.testing.assert_array_equal(
            [matching.recall, matching.precision, matching.f1], [recall, precision, f1]
        )
        np.testing.assert_array_equal(matching.unmatched_reference, np.arange(n_reference))
        np.testing.assert_array_equal(matching.unmatched_detected, np.arange(n_detected))

    def test_an_empty_list_is_no_events(self):
        matching = match_events([], np.array([[0.0, 1.0]]))
        assert matching.reference.shape == (0, 2)
        assert matching.precision == 0.0

    def test_pair_metrics_by_hand(self):
        matching = match_events(np.array([[0.0, 10.0]]), np.array([[2.0, 12.0]]))
        pair = matching.pairs.iloc[0]
        assert pair.iou == pytest.approx(8 / 12, abs=1e-15)
        assert pair.coverage == pytest.approx(0.8, abs=1e-15)
        assert pair.temporal_precision == pytest.approx(0.8, abs=1e-15)
        assert (pair.onset_error, pair.offset_error) == (2.0, 2.0)
        assert np.isnan(pair.peak_error)

    def test_early_detection_is_negative(self):
        matching = match_events(np.array([[1.0, 2.0]]), np.array([[0.5, 1.5]]))
        assert matching.pairs.onset_error.iloc[0] == -0.5
        assert matching.pairs.offset_error.iloc[0] == -0.5

    def test_recall_precision_and_f1_by_hand(self):
        reference = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        detected = np.array([[0.5, 1.5], [10.0, 11.0]])
        matching = match_events(reference, detected)
        assert matching.recall == 1 / 3
        assert matching.precision == 1 / 2
        assert matching.f1 == 2 / 5
        np.testing.assert_array_equal(matching.unmatched_reference, [1, 2])
        np.testing.assert_array_equal(matching.unmatched_detected, [1])

    def test_unmatched_events_where_the_pair_indices_differ(self):
        reference = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        detected = np.array([[2.5, 3.5], [10.0, 11.0], [12.0, 13.0]])
        matching = match_events(reference, detected)
        assert pair_list(matching) == [(1, 0)]
        np.testing.assert_array_equal(matching.unmatched_reference, [0, 2])
        np.testing.assert_array_equal(matching.unmatched_detected, [1, 2])

    def test_coverage_is_of_the_reference_and_precision_of_the_detection(self):
        matching = match_events(np.array([[0.0, 10.0]]), np.array([[2.0, 7.0]]))
        pair = matching.pairs.iloc[0]
        assert (pair.coverage, pair.temporal_precision) == (0.5, 1.0)

    def test_detector_frames_and_one_row_lists(self):
        truth = pd.DataFrame(
            {"start_time": [0.0, 2.0], "end_time": [1.0, 3.0], "peak_time": [0.5, 2.5]}
        )
        found = pd.DataFrame(
            {"start_time": [2.1], "end_time": [2.9], "peak_time": [np.nan]},
            index=pd.Index([1], name="event_number"),
        )
        # a NaN peak gives a NaN peak error; the bounds still match
        matching = match_events(truth, found)
        assert pair_list(matching) == [(1, 0)]
        assert np.isnan(matching.pairs.peak_error.iloc[0])
        # a detector that found nothing returns an empty frame
        assert consensus_counts({"found": found.iloc[:0]}, truth).n_methods.tolist() == [0, 0]
        assert pair_list(match_events([[0, 1]], [[0.5, 2]])) == [(0, 0)]

    def test_a_long_chain_of_overlapping_events(self):
        """One connected component of 500 events a side, each detection
        overlapping two references: every reference is matched."""
        reference = np.column_stack([np.arange(500.0), np.arange(500.0) + 1.0])
        matching = match_events(reference, reference + 0.5)
        assert len(matching.pairs) == 500
        np.testing.assert_array_equal(matching.pairs.onset_error, 0.5)

    def test_touching_is_not_overlap(self):
        matching = match_events(np.array([[0.0, 1.0]]), np.array([[1.0, 2.0]]))
        assert matching.pairs.empty
        np.testing.assert_array_equal(matching.reference_overlaps, [0])
        np.testing.assert_array_equal(matching.detected_overlaps, [0])

    def test_a_zero_length_event_overlaps_nothing(self):
        matching = match_events(np.array([[0.5, 0.5]]), np.array([[0.0, 1.0]]))
        assert matching.pairs.empty
        np.testing.assert_array_equal(matching.reference_overlaps, [0])

    def test_optimal_not_greedy(self):
        # greedy takes the best IoU first, d0-r1 (0.55), and leaves d1 and r0
        # with nothing; the optimum pairs both
        reference = np.array([[0.0, 4.0], [4.5, 10.0]])
        detected = np.array([[0.0, 10.0], [5.0, 6.0]])
        matching = match_events(reference, detected)
        assert pair_list(matching) == [(0, 0), (1, 1)]

    def test_more_pairs_beat_a_larger_summed_iou(self):
        # one pair at IoU 0.5, or two summing to 0.35 (0.25 and 0.1): two pairs
        reference = np.array([[0.0, 1.0], [2.0, 4.0]])
        detected = np.array([[0.0, 4.0], [3.0, 3.2]])
        assert pair_list(match_events(reference, detected)) == [(0, 0), (1, 1)]

    def test_matching_is_symmetric(self):
        a = np.array([[0.0, 1.0], [2.0, 4.0]])
        b = np.array([[0.0, 4.0], [2.5, 3.0]])
        assert len(match_events(a, b).pairs) == len(match_events(b, a).pairs) == 2

        # half-second grids, so tied IoUs are common
        rng = np.random.default_rng(0)
        for _ in range(300):
            a, b = (
                np.sort(rng.integers(0, 20, size=(rng.integers(1, 7), 2)), axis=1) / 2
                for _ in range(2)
            )
            forward, backward = match_events(a, b), match_events(b, a)
            assert len(forward.pairs) == len(backward.pairs)
            assert forward.pairs.iou.sum() == pytest.approx(backward.pairs.iou.sum())
            assert forward.f1 == backward.f1
            assert (forward.recall, forward.precision) == (
                backward.precision,
                backward.recall,
            )

    def test_split_and_merge(self):
        reference = np.array([[0.0, 10.0]])
        detected = np.array([[0.0, 2.0], [3.0, 8.0], [9.0, 10.0]])
        split = match_events(reference, detected)
        np.testing.assert_array_equal(split.split_reference, [0])
        np.testing.assert_array_equal(split.reference_overlaps, [3])
        assert pair_list(split) == [(0, 1)]  # the largest IoU, 0.5

        merged = match_events(detected[:2], reference)
        np.testing.assert_array_equal(merged.merged_detected, [0])
        np.testing.assert_array_equal(merged.detected_overlaps, [2])
        assert pair_list(merged) == [(1, 0)]
        assert split.merged_detected.size == merged.split_reference.size == 0

    def test_minimum_iou(self):
        reference = np.array([[0.0, 10.0]])
        detected = np.array([[8.0, 12.0]])  # IoU 2 / 12, 0.17
        assert len(match_events(reference, detected, minimum_iou=0.1).pairs) == 1
        dropped = match_events(reference, detected, minimum_iou=0.3)
        assert dropped.pairs.empty
        np.testing.assert_array_equal(dropped.reference_overlaps, [1])
        np.testing.assert_array_equal(dropped.detected_overlaps, [1])

    def test_minimum_iou_is_exceeded_not_reached(self):
        # IoU exactly 0.5
        matching = match_events(
            np.array([[0.0, 1.0]]), np.array([[0.0, 0.5]]), minimum_iou=0.5
        )
        assert matching.pairs.empty

    def test_minimum_iou_tolerance_is_the_rounding_of_iou(self):
        """1 ms events on a Unix clock, where a bound rounds by 1.2e-7 s: an
        IoU 0.003 above minimum_iou is well outside what rounding can move
        it, and is matched."""
        origin = 1.7e9
        reference = np.array([[origin, origin + 0.001]])
        detected = np.array([[origin + 0.0001, origin + 0.0011]])  # IoU 9 / 11
        assert len(match_events(reference, detected, minimum_iou=0.815).pairs) == 1
        assert match_events(reference, detected, minimum_iou=0.822).pairs.empty

    def test_a_fragment_below_minimum_iou_still_splits(self):
        reference = np.array([[0.0, 10.0]])
        detected = np.array([[0.0, 9.0], [9.5, 10.0]])
        matching = match_events(reference, detected, minimum_iou=0.2)
        assert pair_list(matching) == [(0, 0)]
        np.testing.assert_array_equal(matching.split_reference, [0])

    def test_indices_refer_to_input_rows(self):
        reference = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
        detected = reference + np.array([0.1, 0.2])
        order_r = np.array([2, 0, 3, 1])
        order_d = np.array([1, 3, 0, 2])
        matching = match_events(reference[order_r], detected[order_d])
        for r, d in pair_list(matching):
            assert order_r[r] == order_d[d]
        assert len(matching.pairs) == 4
        assert matching.pairs.reference_index.is_monotonic_increasing
        np.testing.assert_allclose(matching.pairs.onset_error, 0.1)

    def test_peak_error_from_dataframes(self):
        reference = pd.DataFrame(
            {"start_time": [0.0, 2.0], "end_time": [1.0, 3.0], "peak_time": [0.5, 2.5]},
            index=[10, 11],
        )
        detected = pd.DataFrame(
            {"start_time": [2.1], "end_time": [3.1], "peak_time": [2.4]}, index=[7]
        )
        matching = match_events(reference, detected)
        assert pair_list(matching) == [(1, 0)]
        assert matching.pairs.peak_error.iloc[0] == pytest.approx(-0.1)
        for without_peaks in (
            match_events(reference.to_numpy()[:, :2], detected),
            match_events(reference, detected.drop(columns="peak_time")),
        ):
            assert np.isnan(without_peaks.pairs.peak_error).all()

    def test_boundary_errors_at_other_bounds(self):
        reference = np.array([[0.0, 1.0], [2.0, 3.0], [5.0, 6.0]])
        detected = np.array([[2.1, 3.2], [-0.2, 0.9]])
        matching = match_events(reference, detected)
        narrower = reference + np.array([0.25, -0.25])
        errors = matching.boundary_errors(narrower)
        assert list(errors.columns) == [
            "reference_index",
            "detected_index",
            "onset_error",
            "offset_error",
        ]
        np.testing.assert_array_equal(errors.reference_index, matching.pairs.reference_index)
        np.testing.assert_array_equal(errors.detected_index, matching.pairs.detected_index)
        r, d = errors.reference_index.to_numpy(), errors.detected_index.to_numpy()
        np.testing.assert_array_equal(errors.onset_error, detected[d, 0] - narrower[r, 0])
        np.testing.assert_array_equal(errors.offset_error, detected[d, 1] - narrower[r, 1])
        np.testing.assert_allclose(errors.onset_error, [-0.45, -0.15])
        np.testing.assert_allclose(errors.offset_error, [0.15, 0.45])
        # the original bounds give back the pairs' own errors
        same = matching.boundary_errors(reference)
        np.testing.assert_array_equal(same.onset_error, matching.pairs.onset_error)

    def test_boundary_errors_take_truth_windows_at_another_fraction(self, network_events):
        at_tenth = rd.truth_windows(network_events, 0.1, expression="ripple")
        at_half = rd.truth_windows(network_events, 0.5, expression="ripple")
        detected = at_half[["start_time", "end_time"]].to_numpy()
        errors = match_events(at_tenth, detected).boundary_errors(at_half)
        np.testing.assert_array_equal(errors[["onset_error", "offset_error"]], 0.0)

    def test_boundary_errors_need_the_same_reference_events(self):
        matching = match_events(np.array([[0.0, 1.0], [2.0, 3.0]]), np.array([[0.0, 1.0]]))
        with pytest.raises(ValueError, match=r"reference has 1 events; .* with 2"):
            matching.boundary_errors(np.array([[0.0, 1.0]]))

    def test_minimum_iou_is_keyword_only(self):
        events = np.array([[0.0, 1.0]])
        with pytest.raises(TypeError, match=r"match_events"):
            match_events(events, events, 0.5)


def bounds_on_a_grid(max_events):
    """Sorted, finite, ordered bounds on a half-unit grid, where ties are common."""
    return st.lists(st.tuples(st.integers(0, 30), st.integers(0, 8)), max_size=max_events).map(
        lambda rows: np.array([(s / 2, (s + length) / 2) for s, length in rows]).reshape(-1, 2)
    )


class TestMatchEventsProperties:
    @given(reference=bounds_on_a_grid(8), detected=bounds_on_a_grid(8))
    @settings(max_examples=200, deadline=None)
    def test_invariants(self, reference, detected):
        matching = match_events(reference, detected)
        pairs = matching.pairs
        assert len(pairs) <= min(len(reference), len(detected))
        assert pairs.reference_index.is_unique
        assert pairs.detected_index.is_unique
        assert len(match_events(detected, reference).pairs) == len(pairs)
        assert ((pairs.iou > 0) & (pairs.iou <= 1)).all()
        r = reference[pairs.reference_index.to_numpy()]
        d = detected[pairs.detected_index.to_numpy()]
        intersection = np.minimum(r[:, 1], d[:, 1]) - np.maximum(r[:, 0], d[:, 0])
        assert (intersection > 0).all()
        np.testing.assert_array_equal(pairs.onset_error, d[:, 0] - r[:, 0])
        np.testing.assert_array_equal(pairs.offset_error, d[:, 1] - r[:, 1])
        np.testing.assert_allclose(pairs.coverage, intersection / (r[:, 1] - r[:, 0]))
        np.testing.assert_allclose(
            pairs.temporal_precision, intersection / (d[:, 1] - d[:, 0])
        )
        overlaps = np.minimum(reference[:, None, 1], detected[None, :, 1]) > np.maximum(
            reference[:, None, 0], detected[None, :, 0]
        )
        np.testing.assert_array_equal(matching.reference_overlaps, overlaps.sum(axis=1))
        np.testing.assert_array_equal(matching.detected_overlaps, overlaps.sum(axis=0))

    @given(
        reference=bounds_on_a_grid(5),
        detected=bounds_on_a_grid(5),
        minimum_iou=st.sampled_from([0.0, 0.2, 0.5]),
    )
    @settings(max_examples=200, deadline=None)
    def test_the_assignment_is_the_best_one(self, reference, detected, minimum_iou):
        matching = match_events(reference, detected, minimum_iou=minimum_iou)
        n_pairs, summed_iou = best_assignment(reference, detected, minimum_iou)
        assert len(matching.pairs) == n_pairs
        assert matching.pairs.iou.sum() == pytest.approx(summed_iou)


class TestCompareDetectors:
    def test_signed_differences_by_hand(self):
        a = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [10.0, 11.0]])
        b = np.array([[0.1, 1.0], [1.8, 3.0], [4.3, 5.2]])
        row = compare_detectors({"a": a, "b": b}).iloc[0]
        assert (row.method_a, row.method_b, row.n_a, row.n_b, row.n_matched) == (
            "a",
            "b",
            4,
            3,
            3,
        )
        assert row.jaccard == 3 / 4
        assert row.median_iou == pytest.approx(1 / 1.2)
        # a.start - b.start: -0.1, 0.2, -0.3
        assert row.median_onset_difference == pytest.approx(-0.1)
        assert row.onset_difference_iqr == pytest.approx(0.05 - -0.2)
        assert row.fraction_a_earlier_onset == 2 / 3
        # a.end - b.end: 0, 0, -0.2; equal ends are not earlier
        assert row.median_offset_difference == 0.0
        assert row.offset_difference_iqr == pytest.approx(0.0 - -0.1)
        assert row.fraction_a_earlier_offset == 1 / 3
        assert row[list(COMPARISON_COLUMNS[13:])].isna().all()  # no truth given

    def test_swapping_the_methods_negates_the_differences(self):
        a = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        b = np.array([[0.1, 1.0], [1.8, 3.1], [4.3, 5.2]])
        forward = compare_detectors({"a": a, "b": b}).iloc[0]
        backward = compare_detectors({"b": b, "a": a}).iloc[0]
        assert forward.median_onset_difference == -backward.median_onset_difference
        assert forward.median_offset_difference == -backward.median_offset_difference
        assert forward.jaccard == backward.jaccard

    def test_with_truth(self):
        truth = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]])
        # a finds truth 0-3 late by 0.1-0.4 and ends on time; b finds 1-4
        a = np.array([[0.1, 1.0], [2.2, 3.0], [4.3, 5.0], [6.4, 7.0], [20.0, 21.0]])
        b = np.array(
            [[2.3, 3.1], [4.1, 5.3], [6.2, 7.2], [8.05, 9.0], [20.5, 21.5], [30.0, 31.0]]
        )
        row = compare_detectors({"a": a, "b": b}, truth=truth).iloc[0]
        # true: a's first four, b's first four, three of them shared
        assert row.jaccard_true == 3 / (4 + 4 - 3)
        # false: a's [20, 21]; b's [20.5, 21.5] and [30, 31]
        assert row.jaccard_false == 1 / (1 + 2 - 1)
        assert row.n_shared_truth == 3
        # truth events found: a 0-3, b 1-4
        assert row.jaccard_truth_ids == 3 / (4 + 4 - 3)
        # onset errors on truth 1-3: a 0.2, 0.3, 0.4; b 0.3, 0.1, 0.2; ranks
        # (1, 2, 3) and (3, 1, 2), so rho = 1 - 6 * 6 / (3 * 8)
        assert row.onset_error_correlation == pytest.approx(-0.5)
        # a's offset errors are all 0: a rank correlation is undefined
        assert np.isnan(row.offset_error_correlation)

    def test_error_correlation_needs_three_shared_events(self):
        truth = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        a = np.array([[0.1, 1.1], [2.2, 3.3]])
        b = np.array([[0.3, 1.2], [2.1, 3.1]])
        row = compare_detectors({"a": a, "b": b}, truth=truth).iloc[0]
        assert row.n_shared_truth == 2
        assert np.isnan(row.onset_error_correlation)
        assert np.isnan(row.offset_error_correlation)
        three = compare_detectors(
            {"a": np.vstack([a, [4.3, 5.0]]), "b": np.vstack([b, [4.2, 5.5]])}, truth=truth
        ).iloc[0]
        # onset errors a 0.1, 0.2, 0.3 and b 0.3, 0.1, 0.2; offsets a 0.1, 0.3, 0
        # and b 0.2, 0.1, 0.5
        assert three.onset_error_correlation == pytest.approx(-0.5)
        assert three.offset_error_correlation == pytest.approx(-1.0)

    @pytest.mark.parametrize("origin", [0.0, 3600.0, 1.7e9])
    def test_error_correlation_ranks_whole_samples(self, origin):
        """Errors read off sample times take a few values, whole samples late,
        which round apart by ulps: equal numbers of samples are ties, so the
        correlation is that of the sample counts, and NaN when one method's
        count never varies."""
        time = origin + np.arange(3000) / FS
        starts = np.arange(40) * 70 + 10
        truth = np.column_stack([time[starts], time[starts + 30]])
        late_a = np.tile([0, 1, 1, 0, 1], 8)
        late_b = np.where(np.arange(40) % 3 == 0, 1 - late_a, late_a)

        def events(late):
            return np.column_stack([time[starts + late], time[starts + 30]])

        row = compare_detectors({"a": events(late_a), "b": events(late_b)}, truth=truth).iloc[
            0
        ]
        expected = spearmanr(late_a, late_b)[0]
        assert row.onset_error_correlation == pytest.approx(expected, abs=1e-12)
        constant = compare_detectors(
            {"a": events(np.ones(40, dtype=int)), "b": events(late_b)}, truth=truth
        ).iloc[0]
        assert np.isnan(constant.onset_error_correlation)

    @pytest.mark.parametrize("origin", [0.0, 1.7e9])
    def test_errors_microseconds_apart_are_not_tied(self, origin):
        """Onset errors 1 and 2 us apart are four and eight ulps on a Unix
        clock: distinct errors, perfectly correlated, at any origin."""
        steps = np.arange(20.0)
        truth = origin + np.column_stack([steps, steps + 0.1])
        methods = {
            "a": truth + (steps * 1e-6)[:, None],
            "b": truth + (steps * 2e-6)[:, None],
        }
        row = compare_detectors(methods, truth=truth).iloc[0]
        assert row.onset_error_correlation == pytest.approx(1.0)
        assert row.offset_error_correlation == pytest.approx(1.0)

    def test_a_tie_group_spans_no_more_than_the_rounding(self):
        """Errors two ulps apart on a Unix clock are each within the rounding
        of their neighbour but not of every other: they tie in pairs, not in
        one group that would make them constant."""
        origin = 1.7e9
        ulp = np.spacing(origin)
        steps = np.arange(20.0)
        truth = origin + np.column_stack([steps, steps + 0.1])
        methods = {
            "a": truth + (2 * ulp * steps)[:, None],
            "b": truth + (steps * 1e-3)[:, None],
        }
        row = compare_detectors(methods, truth=truth).iloc[0]
        assert row.onset_error_correlation > 0.9

    def test_pair_order_and_columns(self):
        events = {
            "z": np.array([[0.0, 1.0]]),
            "x": np.empty((0, 2)),
            "y": np.empty((0, 2)),
        }
        comparison = compare_detectors(events)
        assert list(comparison.columns) == list(COMPARISON_COLUMNS)
        assert list(zip(comparison.method_a, comparison.method_b, strict=True)) == [
            ("z", "x"),
            ("z", "y"),
            ("x", "y"),
        ]
        np.testing.assert_array_equal(comparison.n_matched, 0)
        np.testing.assert_array_equal(comparison.jaccard, [0.0, 0.0, np.nan])
        for column in COMPARISON_COLUMNS[6:]:
            assert comparison[column].isna().all(), column

    def test_truth_columns_are_filled_with_truth(self):
        events = {"x": np.array([[0.0, 1.0]]), "y": np.array([[0.0, 1.0]])}
        row = compare_detectors(events, truth=np.array([[0.0, 1.0]])).iloc[0]
        assert (row.jaccard_true, row.n_shared_truth, row.jaccard_truth_ids) == (1.0, 1, 1.0)
        assert np.isnan(row.jaccard_false)

    def test_intervals_can_agree_on_different_true_events(self):
        """Each method finds a different true event, and their detections
        overlap: the intervals agree, the truth events found do not."""
        truth = np.array([[0.0, 1.0], [1.0, 2.0]])
        events = {"a": np.array([[0.0, 1.0]]), "b": np.array([[0.9, 2.0]])}
        row = compare_detectors(events, truth=truth).iloc[0]
        assert (row.jaccard_true, row.n_shared_truth, row.jaccard_truth_ids) == (1.0, 0, 0.0)

    def test_truth_ids_agreement_is_nan_when_neither_finds_a_true_event(self):
        events = {"a": np.array([[5.0, 6.0]]), "b": np.empty((0, 2))}
        row = compare_detectors(events, truth=np.array([[0.0, 1.0]])).iloc[0]
        assert np.isnan(row.jaccard_truth_ids)

    @pytest.mark.parametrize("n_methods", [0, 1])
    def test_fewer_than_two_methods_give_no_rows(self, n_methods):
        events = dict(list({"x": np.array([[0.0, 1.0]])}.items())[:n_methods])
        comparison = compare_detectors(events, truth=np.array([[0.0, 1.0]]))
        assert comparison.empty
        assert list(comparison.columns) == list(COMPARISON_COLUMNS)

    def test_minimum_iou_applies_to_every_matching(self):
        truth = np.array([[0.0, 10.0]])
        events = {"a": np.array([[0.0, 10.0]]), "b": np.array([[8.0, 12.0]])}
        loose = compare_detectors(events, truth=truth).iloc[0]
        strict = compare_detectors(events, truth=truth, minimum_iou=0.3).iloc[0]
        assert (loose.n_matched, loose.n_shared_truth) == (1, 1)
        assert (strict.n_matched, strict.n_shared_truth) == (0, 0)
        assert strict.jaccard_false == 0.0

    @pytest.mark.parametrize("function", [compare_detectors, consensus_counts])
    def test_events_must_be_a_mapping(self, function):
        """A single DataFrame iterates over its columns, which would be compared
        as methods."""
        frame = pd.DataFrame({"start_time": [0.0], "end_time": [1.0]})
        with pytest.raises(TypeError, match="events must map each method's name"):
            function(frame, truth=frame)


class TestConsensusCounts:
    def test_by_hand(self):
        truth = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        events = {
            "a": np.array([[0.1, 0.9], [2.5, 3.5]]),
            "b": np.array([[0.5, 3.0]]),  # overlaps two; matched to the second
            "c": np.array([[10.0, 11.0]]),
        }
        consensus = consensus_counts(events, truth)
        assert list(consensus.columns) == ["a", "b", "c", "n_methods"]
        assert consensus.a.tolist() == [True, True, False]
        assert consensus.b.tolist() == [False, True, False]
        assert consensus.c.tolist() == [False, False, False]
        assert consensus.n_methods.tolist() == [1, 2, 0]
        assert consensus.a.dtype.kind == "b"
        pd.testing.assert_index_equal(consensus.index, pd.RangeIndex(3))

    def test_a_dataframe_truth_keeps_its_index(self):
        truth = pd.DataFrame({"start_time": [0.0, 2.0], "end_time": [1.0, 3.0]}, index=[5, 9])
        consensus = consensus_counts({"a": np.array([[2.0, 3.0]])}, truth)
        pd.testing.assert_index_equal(consensus.index, truth.index)
        assert truth.assign(found=consensus.a).found.tolist() == [False, True]

    def test_minimum_iou(self):
        truth = np.array([[0.0, 10.0]])
        events = {"a": np.array([[8.0, 12.0]])}
        assert consensus_counts(events, truth).a.tolist() == [True]
        assert consensus_counts(events, truth, minimum_iou=0.3).a.tolist() == [False]

    def test_no_methods(self):
        consensus = consensus_counts({}, np.array([[0.0, 1.0]]))
        assert list(consensus.columns) == ["n_methods"]
        assert consensus.n_methods.tolist() == [0]

    def test_no_method_may_be_named_n_methods(self):
        with pytest.raises(ValueError, match=r"n_methods"):
            consensus_counts({"n_methods": GOOD}, GOOD)


class TestLabelByOverlap:
    def test_longest_overlap_wins_and_ties(self):
        events = np.array(
            [
                [0.5, 1.5],  # 1.0 in swr, 0.5 in emg
                [1.5, 3.5],  # 0.5 in swr, 2.0 in emg
                [1.0, 2.0],  # 1.0 in each: the earlier row
                [5.2, 5.4],  # inside two identical windows: the earlier row
                [6.0, 7.0],  # touches gamma and theta only
                [8.0, 9.0],
            ]
        )
        labels = label_by_overlap(events, WINDOWS)
        assert labels.tolist() == ["swr", "emg", "swr", "gamma", "background", "background"]
        assert labels.name == "label"
        pd.testing.assert_index_equal(labels.index, pd.RangeIndex(6))

    def test_windows_on_any_index(self):
        """Windows filtered or relabelled keep their own index; the labels are
        read by position."""
        events = np.array([[0.5, 1.5], [1.5, 3.5], [5.2, 5.4]])
        expected = label_by_overlap(events, WINDOWS)
        relabelled = WINDOWS.set_axis([40, 30, 20, 10])
        pd.testing.assert_series_equal(label_by_overlap(events, relabelled), expected)
        assert label_by_overlap(events, WINDOWS.iloc[1:]).tolist() == [
            "emg",
            "emg",
            "gamma",
        ]

    def test_unlabeled(self):
        labels = label_by_overlap(np.array([[8.0, 9.0]]), WINDOWS, unlabeled="none")
        assert labels.tolist() == ["none"]

    def test_a_dataframe_keeps_its_index(self):
        events = pd.DataFrame(
            {"start_time": [8.0, 0.5], "end_time": [9.0, 1.0]},
            index=pd.Index([1, 2], name="event_number"),
        )
        labels = label_by_overlap(events, WINDOWS)
        pd.testing.assert_index_equal(labels.index, events.index)
        assert events.assign(label=labels).label.tolist() == ["background", "swr"]

    def test_no_windows_or_no_events(self):
        assert label_by_overlap(np.array([[0.0, 1.0]]), WINDOWS.iloc[:0]).tolist() == [
            "background"
        ]
        assert label_by_overlap(np.empty((0, 2)), WINDOWS).empty

    def test_truth_windows_with_the_type_renamed(self, network_events):
        windows = rd.truth_windows(network_events, 0.1, expression="network")
        assert len(set(windows.type)) > 1
        events = windows[["start_time", "end_time"]].to_numpy()
        labels = label_by_overlap(events, windows.rename(columns={"type": "label"}))
        assert labels.tolist() == windows.type.tolist()

    def test_windows_need_a_label_column(self):
        with pytest.raises(ValueError, match=r"'label' column.*rename 'type'"):
            label_by_overlap(GOOD, WINDOWS.drop(columns="label"))


def _as_windows(bounds):
    return pd.DataFrame(bounds, columns=["start_time", "end_time"]).assign(label="x")


# every input's bounds are checked, and the error names the input and row
INVALID_BOUNDS = {
    "reference": lambda bad: match_events(bad, GOOD),
    "detected": lambda bad: match_events(GOOD, bad),
    "boundary_errors": lambda bad: match_events(
        np.vstack([GOOD, GOOD + 2]), GOOD
    ).boundary_errors(bad),
    r"events\['b'\]": lambda bad: compare_detectors({"a": GOOD, "b": bad}),
    "truth": lambda bad: compare_detectors({"a": GOOD}, truth=bad),
    r"events\['a'\]": lambda bad: consensus_counts({"a": bad}, GOOD),
    "consensus truth": lambda bad: consensus_counts({"a": GOOD}, bad),
    "events": lambda bad: label_by_overlap(bad, WINDOWS),
    "windows": lambda bad: label_by_overlap(GOOD, _as_windows(bad)),
}


@pytest.mark.parametrize("case", INVALID_BOUNDS)
@pytest.mark.parametrize(
    "bad",
    [[[0.0, 1.0], [np.nan, 2.0]], [[0.0, 1.0], [2.0, np.inf]], [[0.0, 1.0], [2.0, 1.0]]],
    ids=["nan", "inf", "reversed"],
)
def test_invalid_bounds_are_named(case, bad):
    name = {"boundary_errors": "reference", "consensus truth": "truth"}.get(case, case)
    with pytest.raises(ValueError, match=rf"^{name} row 1 is "):
        INVALID_BOUNDS[case](np.array(bad))


@pytest.mark.parametrize(
    "call",
    [
        lambda value: match_events(GOOD, GOOD, minimum_iou=value),
        lambda value: compare_detectors({}, minimum_iou=value),
        lambda value: consensus_counts({}, GOOD, minimum_iou=value),
    ],
    ids=["match_events", "compare_detectors", "consensus_counts"],
)
@pytest.mark.parametrize(
    ("value", "error"),
    [
        (-0.1, ValueError),
        (1.0, ValueError),
        (2.0, ValueError),
        (np.nan, ValueError),
        (None, TypeError),
    ],
)
def test_minimum_iou_must_lie_in_zero_to_one(call, value, error):
    with pytest.raises(error, match=r"minimum_iou"):
        call(value)


@pytest.fixture(scope="module")
def network_events():
    """Latent events of every type in 20 s."""
    time = rd.simulate_time(int(20 * FS), FS)
    return rd.draw_network_events(time, event_rate=1.0, rng=3)


@pytest.fixture(scope="module")
def swr_session():
    time = rd.simulate_time(int(60 * FS), FS)
    events = rd.draw_network_events(
        time, type_probabilities={"swr": 1.0}, ripple_snr=(4.0, 6.0), rng=0
    )
    return rd.simulate_network_session(time, events, rng=0)


class TestEvaluateOnSimulatedSession:
    def test_kay_against_ripple_truth(self, swr_session):
        filtered = rd.filter_ripple_band(swr_session.lfps, sampling_frequency=FS)
        kay = rd.Kay_ripple_detector(swr_session.time, filtered, swr_session.speed, FS)
        truth = rd.truth_windows(swr_session.events, 0.1, expression="ripple")
        assert len(truth) > 5
        matching = match_events(truth, kay)
        assert matching.recall > 0.8
        errors = matching.pairs[["onset_error", "offset_error", "peak_error"]]
        assert np.isfinite(errors.to_numpy()).all()
        # the narrower truth, at half maximum, for the same pairs
        half = rd.truth_windows(swr_session.events, 0.5, expression="ripple")
        at_half = matching.boundary_errors(half)
        assert (at_half.onset_error < matching.pairs.onset_error).all()

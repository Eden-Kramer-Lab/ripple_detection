import numpy as np
import pandas as pd
import pytest

from src.ripple_detection.core import (_extend_segment,
                                       _find_containing_interval,
                                       _get_series_start_end_times,
                                       merge_overlapping_ranges,
                                       segment_boolean_series,
                                       threshold_by_zscore,
                                       exclude_movement)


@pytest.mark.parametrize('series, expected_segments', [
    (pd.Series([False, False, True, True, False]),
     (np.array([2]), np.array([3]))),
    (pd.Series([False, False, True, True, False, True, False]),
     (np.array([2, 5]), np.array([3, 5]))),
    (pd.Series([True, True, False, False, False]),
     (np.array([0]), np.array([1]))),
    (pd.Series([False, False, True, True, True]),
     (np.array([2]), np.array([4]))),
    (pd.Series([True, False, True, True, False]),
     (np.array([0, 2]), np.array([0, 3]))),
])
def test_get_series_start_end_times(series, expected_segments):
    tup = _get_series_start_end_times(series)
    try:
        assert np.all(tup[0] == expected_segments[0]) & np.all(
            tup[1] == expected_segments[1])
    except IndexError:
        assert tup == expected_segments


@pytest.mark.parametrize('series, expected_segments', [
    (pd.Series([False, True, True, True, False],
               index=np.linspace(0, 0.020, 5)), [(0.005, 0.015)]),
    (pd.Series([False, False, True, True, False, True, False],
               index=np.linspace(0, 0.030, 7)), []),
    (pd.Series([True, True, False, False, False],
               index=np.linspace(0, 0.020, 5)), []),
    (pd.Series([False, True, True, True, True],
               index=np.linspace(0, 0.020, 5)), [(0.005, 0.020)]),
    (pd.Series([True, True, True, True, False],
               index=np.linspace(0, 0.020, 5)), [(0.000, 0.015)]),
    (pd.Series([True, True, True, True, False, True, True, True],
               index=np.linspace(0, 0.035, 8)),
        [(0.000, 0.015), (0.025, 0.035)]),
])
def test_segment_boolean_series(series, expected_segments):
    assert np.all(
        [(np.allclose(expected_start, test_start)) &
         (np.allclose(expected_end, test_end))
         for (test_start, test_end), (expected_start, expected_end)
         in zip(segment_boolean_series(series), expected_segments)])


@pytest.mark.parametrize(
    'interval_candidates, target_interval, expected_interval', [
        ([(1, 2), (5, 7)], (6, 7), (5, 7)),
        ([(1, 2), (5, 7)], (1, 2), (1, 2)),
        ([(1, 2), (5, 7), (20, 30)], (5, 6), (5, 7)),
        ([(1, 2), (5, 7), (20, 30)], (24, 26), (20, 30)),
    ])
def test_find_containing_interval(interval_candidates, target_interval,
                                  expected_interval):
    test_interval = _find_containing_interval(
        interval_candidates, target_interval)
    assert np.all(test_interval == expected_interval)


@pytest.mark.parametrize(
    'interval_candidates, target_intervals, expected_intervals', [
        ([(1, 2), (5, 7)], [(6, 7)], [(5, 7)]),
        ([(1, 2), (5, 7)], [(1, 2)], [(1, 2)]),
        ([(1, 2), (5, 7), (20, 30)], [(5, 6)], [(5, 7)]),
        ([(1, 2), (5, 7), (20, 30)], [(24, 26), (6, 7)], [(5, 7),
                                                          (20, 30)]),
        ([(1, 2), (5, 7), (20, 30)], [(24, 26), (27, 28)], [(20, 30)]),
    ])
def test__extend_segment(interval_candidates, target_intervals,
                         expected_intervals):
    test_intervals = _extend_segment(
        target_intervals, interval_candidates)
    assert np.all(test_intervals == expected_intervals)


@pytest.mark.parametrize(
    'ranges, expected_ranges', [
        ([(5, 7), (3, 5), (-1, 3)], [(-1, 7)]),
        ([(5, 6), (3, 4), (1, 2)], [(1, 2), (3, 4), (5, 6)]),
        ([], []),
    ])
def test_merge_overlapping_ranges(ranges, expected_ranges):
    assert list(merge_overlapping_ranges(ranges)) == expected_ranges


def test_threshold_by_zscore():
    data = np.array([0, 0, 10, 10, 0, 0, 0, 1, 5,
                     10, 10, 10, 10, 10, 5, 1, 0])
    time = np.arange(len(data)) / 1000
    segments = threshold_by_zscore(
        data, time, zscore_threshold=1, minimum_duration=0.004)
    assert np.allclose(segments, [(0.008, 0.014)])


def test_exclude_movement():
    n_samples = 100
    time = np.arange(n_samples) / 1000
    speed = np.ones_like(time) * 5
    speed[3:11] = 1
    candidate_ripple_times = [(0.004, 0.010), (0.094, 0.095)]
    ripple_times = exclude_movement(
        candidate_ripple_times, speed, time, speed_threshold=4.0)
    expected_ripple_times = np.array([(0.004, 0.010)])
    assert np.allclose(ripple_times, expected_ripple_times)

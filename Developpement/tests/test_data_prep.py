import numpy as np
import pytest
from oussama_dev.data_preparation.data_prep import DataPreper, PaddingType

# test the initialisation and strategy mapping


@pytest.mark.parametrize(
    "padding_type,expected_strategy",
    [
        (PaddingType.MEAN, "mean_padding_strategy"),
        (PaddingType.MEDIAN, "median_padding_strategy"),
        (PaddingType.MAX_SIZE_NO_PAD, "no_padding_strategy_max_sizes_only"),
    ],
)
def test_padding_strategy_initialization(padding_type, expected_strategy):
    data_preper = DataPreper(padding_type)
    assert data_preper.padding_strategy.__name__ == expected_strategy


def test_mean_padding_strategy():
    data_preper = DataPreper(padding_type=PaddingType.MEAN)
    data = [["code1", [1, 2, 3], True], ["code2", [4, 5], False]]
    expected = [["code1", [1, 2, 3], True], ["code2", [4, 5, 4.5], False]]
    assert data_preper.padding_strategy(data) == expected


def test_median_padding_strategy():
    data_preper = DataPreper(padding_type=PaddingType.MEDIAN)
    data = [["code1", [1, 2, 3], True], ["code2", [4, 5], False]]
    expected = [["code1", [1, 2, 3], True], ["code2", [4, 5, 4.5], False]]
    assert data_preper.padding_strategy(data) == expected


def test_no_padding_strategy():
    data_preper = DataPreper(padding_type=PaddingType.MAX_SIZE_NO_PAD)
    data = [["code1", [1, 2, 3], True], ["code2", [4, 5], False]]
    expected = [["code1", [1, 2, 3], True]]
    assert data_preper.padding_strategy(data) == expected


@pytest.mark.parametrize(
    "data, window_size, expected",
    [
        # Test case 1: Simple moving average with window size 2
        ([["job1", [1, 2, 3, 4], True]], 2, [["job1", [1, 1.5, 2.5, 3.5], True]]),
        # Test case 2: Window size larger than the number of data points (should handle gracefully)
        ([["job3", [10, 20], True]], 3, [["job3", [10, 15], True]]),
        # Add more test cases as needed
    ],
)
def test_rolling_window_smoothing(data, window_size, expected):
    data_preper = DataPreper(padding_type=PaddingType.MEAN)
    smoothed_data = data_preper.rolling_window_smoothing(data, window_size)
    assert smoothed_data == expected


def test_split_dataset_into_train_test_success():
    data_preper = DataPreper(padding_type=PaddingType.MEAN)
    dataset = [
        [1, [1, 2, 3], True],
        [2, [4, 5, 6], False],
        [3, [7, 8, 9], True],
        [4, [10, 11, 12], False],
    ]

    X_train, X_test, y_train, y_test = data_preper.split_dataset_into_train_test(
        dataset
    )

    assert len(X_train) > 0 and len(X_test) > 0
    assert len(X_train) + len(X_test) == len(dataset)
    assert X_train.shape[1] == 3 and X_test.shape[1] == 3
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]

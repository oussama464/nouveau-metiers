from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Any, Callable, Optional
from sklearn.model_selection import train_test_split
from enum import StrEnum


class PaddingType(StrEnum):
    MEAN = "mean"
    MEDIAN = "median"
    MAX_SIZE_NO_PAD = "no_padding"


@dataclass
class DataPreper:
    padding_type: PaddingType
    padding_strategy: Optional[Callable[[list[list[Any]]], list[list[Any]]]] = None

    def __post_init__(self):
        strategy_map = {
            PaddingType.MEAN: self.mean_padding_strategy,
            PaddingType.MEDIAN: self.median_padding_strategy,
            PaddingType.MAX_SIZE_NO_PAD: self.no_padding_strategy_max_sizes_only,
        }
        if self.padding_type in strategy_map:
            self.padding_strategy = strategy_map[self.padding_type]
        else:
            raise ValueError(
                f"Unrecognized padding type: {self.padding_type} available options are {PaddingType}"
            )

    def get_min_max_sample_size(self, data: list[list[Any]]) -> tuple[int, int]:
        sizes = list(map(lambda x: len(x[1]), data))
        return min(sizes), max(sizes)

    # padding strategy for missing mesurment
    # zero padding , mean ,median ,last_value ,no_padding
    def no_padding_strategy_max_sizes_only(
        self, data: list[list[Any]]
    ) -> list[list[Any]]:
        _, max_size = self.get_min_max_sample_size(data)
        return list(filter(lambda x: len(x[1]) == max_size, data))

    def mean_padding_strategy(self, data: list[list[Any]]) -> list[list[Any]]:
        new_padded_data = []
        _, max_size = self.get_min_max_sample_size(data)
        for item in data:
            if len(item[1]) < max_size:
                mean_val = np.mean(item[1])
                padded_item = item[1] + [mean_val] * (max_size - len(item[1]))
                new_padded_data.append([item[0], padded_item, item[2]])
            else:
                new_padded_data.append(item)
        return new_padded_data

    def median_padding_strategy(self, data: list[list[Any]]) -> list[list[Any]]:
        new_padded_data = []
        _, max_size = self.get_min_max_sample_size(data)
        for item in data:
            if len(item[1]) < max_size:
                mean_val = np.median(item[1])
                padded_item = item[1] + [mean_val] * (max_size - len(item[1]))
                new_padded_data.append([item[0], padded_item, item[2]])
            else:
                new_padded_data.append(item)
        return new_padded_data

    def rolling_window_smooting_pandas(self, data, window_size=3):
        rolling_avrage_smoothed = []
        for item in data:
            job_code, values, is_emerging = item
            series = pd.Series(values)
            rolling_avrage = series.rolling(window=window_size).mean()
            # rolling_avrage_smoothed[job_code] = rolling_avrage.to_list()
            rolling_avrage.dropna(inplace=True)
            rolling_avrage_smoothed.append(
                [job_code, rolling_avrage.to_list(), is_emerging]
            )
        return rolling_avrage_smoothed

    def rolling_window_smoothing(self, data, window_size):
        rolling_avrage_smoothed = []
        for item in data:
            job_code, values, is_emerging = item
            adjusted_values = []
            for i in range(len(values)):
                window = values[max(0, i - window_size + 1) : i + 1]
                adjusted_values.append(sum(window) / len(window))
            rolling_avrage_smoothed.append([job_code, adjusted_values, is_emerging])
        return rolling_avrage_smoothed

    def split_dataset_into_train_test(
        self, dataset: list[list]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.padding_strategy:
            padded_data = self.padding_strategy(dataset)
        else:
            raise ValueError("padding strategy was not set")
        X = [item[1] for item in padded_data]
        y = [item[2] for item in padded_data]

        # Convert lists to numpy arrays
        X_array = np.array(X)
        y_array = np.array(y, dtype=bool)
        X_train, X_test, y_train, y_test = train_test_split(
            X_array, y_array, test_size=0.33, random_state=42, stratify=y_array
        )
        # print(Counter(y))
        # print(Counter(y_train))
        # print(Counter(y_test))
        # one hot encode target
        # y_train = to_categorical(y_train)
        # y_test = to_categorical(y_test)
        # y_test = np.argmax(y_test, axis=1)
        # y_train = np.argmax(y_train, axis=1)

        return X_train, X_test, y_train, y_test

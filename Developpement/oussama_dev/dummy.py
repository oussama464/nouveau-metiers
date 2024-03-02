import numpy as np
from scipy import stats
from collections import Counter


def calculate_crossings(list_values: list[float]) -> list[float]:
    values_array = np.array(list_values)
    # Calculate the mean value of the list values
    mean_value = np.nanmean(values_array)
    # Calculate zero crossings
    zero_crossing_indices = np.nonzero(np.diff(values_array > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    # Calculate mean crossings
    mean_crossing_indices = np.nonzero(np.diff(values_array > mean_value))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]


def calculate_entropy(list_values: list[float]) -> float:
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1] / len(list_values) for elem in counter_values]
    entropy = stats.entropy(probabilities)
    return entropy


# Example usage
list_values = [1, -2, 3, -4, 5, -6, 7, -8, 9]
m = np.nanmean(list_values)
print(m)
print(calculate_crossings(list_values))
print(calculate_entropy(list_values))

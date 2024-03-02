import numpy as np
import pytest
from scipy.signal import find_peaks, welch
from scipy.fft import fft
from oussama_dev.feature_extraction.features_eng import FeatureExtractor


def test_calculate_statistics():
    data = np.array([1, 2, 3, 4, 5])
    extractor = FeatureExtractor(dataset=np.array([[1, 2, 3]]), lables=[1])
    expected_statistics = [
        np.nanpercentile(data, 5),  # n5
        np.nanpercentile(data, 25),  # n25
        np.nanpercentile(data, 75),  # n75
        np.nanpercentile(data, 95),  # n95
        np.nanpercentile(data, 50),  # median
        np.nanmean(data),  # mean
        np.nanstd(data),  # std
        np.nanvar(data),  # var
        np.sqrt(np.nanmean(np.square(data))),  # rms
    ]
    calculated_statistics = extractor.calculate_statistics(data)
    np.testing.assert_array_almost_equal(
        calculated_statistics,
        expected_statistics,
        decimal=6,
        err_msg="Statistics calculation failed",
    )


def test_get_fft_values():
    x = np.array([0, 1, 0, -1])
    extractor = FeatureExtractor(dataset=[], lables=[])
    fft_values = extractor.get_fft_values(x)
    expected_values = np.array([0.0, 1.0])
    np.testing.assert_array_almost_equal(fft_values, expected_values, decimal=6)


def test_get_psd_values():
    # Use a signal where the PSD can be predicted or compared against a known value
    x = np.random.randn(1024)
    extractor = FeatureExtractor(dataset=[], lables=[])
    psd_values = extractor.get_psd_values(x, f_s=1024)
    assert len(psd_values) > 0, "PSD values should not be empty"


def test_get_features():
    x = np.array([0, 1, 0, 2, 0, 1, 0])
    extractor = FeatureExtractor(dataset=[], lables=[])
    features = extractor.get_features(x)
    expected_features = np.array([1, 2])  # Expected peak features for the test signal
    np.testing.assert_array_equal(features, expected_features)

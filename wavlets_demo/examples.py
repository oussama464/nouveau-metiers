import pywt
import pandas as pd
import numpy as np

dataset = (
    "https://raw.githubusercontent.com/taspinar/siml/master/datasets/sst_nino3.dat.txt"
)
df_nino = pd.read_table(dataset)

print(df_nino.head())


def cwt_transform(signal: np.ndarray, waveletname="morl"):
    waveletname = "morl"
    scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(signal, scales, waveletname)
    return coefficients, frequencies


def dwt_transform(signal: np.ndarray, waveletname="db2", level=8) -> list[np.ndarray]:

    coeffs = pywt.wavedec(signal, waveletname, level=level)
    reconstructed_signal = pywt.waverec(coeffs, waveletname)
    return coeffs


def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    # rms = np.nanmean(np.sqrt(list_values**2))
    return [n5, n25, n75, n95, median, mean, std, var]


coeffs = dwt_transform(df_nino.values)
s = calculate_statistics(coeffs)
print(s)

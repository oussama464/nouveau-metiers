import os
import glob
import pathlib
import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot


os.environ["KERAS_BACKEND"] = "torch"
from keras.utils import to_categorical


def load_file(filepath: str) -> np.ndarray:
    dataframe = pd.read_csv(filepath, header=None, sep="\s+")
    return dataframe.values


def load_group(filenames: str) -> np.ndarray:
    #  [samples,time steps,features]
    loaded = []
    for name in filenames:
        data = load_file(name)
        loaded.append(data)
    loaded = np.dstack(loaded)
    return loaded


def load_dataset_group(
    group: str, data_dir_path: pathlib.PosixPath
) -> tuple[np.ndarray, np.ndarray]:

    base_data_path = data_dir_path.joinpath(group)

    filenames = glob.glob(f"{base_data_path}/Inertial_Signals/*.txt", recursive=True)
    X = load_group(filenames)
    y = load_file(base_data_path.joinpath(f"y_{group}.txt"))
    return X, y


def load_dataset(
    data_dir_path: pathlib.PosixPath,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trainX, trainy = load_dataset_group("train", data_dir_path)
    print(trainX.shape, trainy.shape)
    testX, testy = load_dataset_group("test", data_dir_path)
    print(testX.shape, testy.shape)
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode target
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy

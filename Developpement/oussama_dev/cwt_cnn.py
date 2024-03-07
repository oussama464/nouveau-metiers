import os
import pywt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any
from data_preparation.data_prep import DataPreper, PaddingType

# import torch
from sklearn.model_selection import train_test_split

# os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers
from keras.utils import to_categorical
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from collections import Counter


FILE_PATH = "/home/bobo/Desktop/nouveau_metier/Developpement/raw_csv/extracted_jobs.csv"


def get_formatted_raw_data(file_path: str) -> list[list[Any]]:

    df = pd.read_csv(file_path)
    df_grouped = df.groupby(["job_code", "is_emerging_job"]).agg(list).reset_index()

    # Transforming the grouped dataframe to the desired format
    training_data = df_grouped.apply(
        lambda row: [row["job_code"], row["rawpop"], row["is_emerging_job"]], axis=1
    ).tolist()
    return training_data


# plot the scaelogram
def plot_cwt_coeffs(coef):
    plt.figure(figsize=(15, 20))
    plt.imshow(
        coef,
        extent=[0, 200, 31, 1],
        interpolation="bilinear",
        cmap="jet",
        aspect="auto",
        vmax=abs(coef).max(),
        vmin=-abs(coef).max(),
    )
    plt.gca().invert_yaxis()
    plt.yticks(np.arange(1, 31, 1))
    plt.xticks(np.arange(0, 201, 10))
    plt.show()


# plot the signal
def plot_signal(signal):
    t = np.arange(2015, 2024, 1)
    plt.figure(figsize=(15, 10))
    plt.plot(t, signal)
    plt.grid(color="gray", linestyle=":", linewidth=0.5)
    plt.show()


# 'mexh', 'morl'
def reshape_data_for_cnn(X, scales, waveletname):
    new_X = []
    for sample_no in range(len(X)):
        item = X[sample_no, :]
        coefs, freqs = pywt.cwt(item, scales, wavelet=waveletname)
        coefs = coefs.reshape(-1)
        new_X.append(coefs)
    new_X = np.array(new_X)
    reshaped = new_X.reshape((new_X.shape[0], len(scales), X.shape[1], 1))
    # normalize
    reshaped = reshaped.astype("float32")
    reshaped /= 255.0
    return reshaped


def train_evaluate_cnn_model(
    X_train, y_train, X_test, y_test, input_shape: tuple[int, int, int]
) -> float:
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")(
        inputs
    )
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same")(x)
    # x = layers.MaxPooling2D(pool_size=2)(x)
    # x = layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.Flatten()(x)
    # x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="rmsprop",
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.F1Score()],
    )
    model.fit(X_train, y_train, epochs=5, batch_size=1)
    test_loss, test_f1, test_acc = model.evaluate(X_test, y_test)
    return test_acc, test_f1, test_loss


if __name__ == "__main__":
    data = get_formatted_raw_data(FILE_PATH)
    prep = DataPreper(PaddingType.MAX_SIZE_NO_PAD)
    X, y = prep.prepare(data)
    y = np.array([1 if i else 0 for i in y])
    # y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )
    oversample = SMOTE(k_neighbors=2)
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    y_test = y_test.reshape((len(y_test), 1))
    y_train = y_train.reshape(len(y_train), 1)

    scales = np.arange(1, 18)
    print(len(scales))
    ## 'mexh', 'morl'
    waveletname = "morl"
    X_train, X_test = reshape_data_for_cnn(
        X_train, scales, waveletname
    ), reshape_data_for_cnn(X_test, scales, waveletname)
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

    test_acc, test_f1, test_loss = test_loss = train_evaluate_cnn_model(
        X_train, y_train, X_test, y_test, input_shape
    )
    print(end="\n")
    print(test_acc, test_f1, test_loss)

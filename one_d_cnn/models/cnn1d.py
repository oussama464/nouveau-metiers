import os
import numpy as np
import torch


os.environ["KERAS_BACKEND"] = "torch"
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv1D, MaxPooling1D, Input


def evaluate_model(
    trainX: np.ndarray, trainy: np.ndarray, testX: np.ndarray, testy: np.ndarray
) -> float:
    """
    the model requires 3D input with [samples,time_steps,features]
    each sample is one window of the time series data each window has 128 time steps and a time step has 9 features
    the output will be a six_element vector containing the proba of a given window belonging to each of the six activity types
    """
    verbose, epochs, batch_size = 1, 10, 32
    n_timesteps, n_features, n_outputs = (
        trainX.shape[1],
        trainX.shape[2],
        trainy.shape[1],
    )

    model = Sequential()
    model.add(Input(shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation="relu"))
    model.add(Conv1D(filters=64, kernel_size=3, activation="relu"))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(n_outputs, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy

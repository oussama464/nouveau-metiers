import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn import datasets
from sklearn.decomposition import PCA


def load_dataset(
    file_path="/home/bobo/Desktop/nouveau_metier/Developpement/feature_selection/pima-indians-diabetes.csv",
) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(file_path, header=None)
    dataset = data.values
    X = dataset[:, :-1]
    y = dataset[:, -1]
    return X, y


def apply_pca(
    X_train: np.ndarray, X_test: np.ndarray, variance_threshold: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=variance_threshold)
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)
    return X_train_reduced, X_test_reduced


d = {1: [1, 2, 3], 2: [4, 5, 6]}
print(list(d))

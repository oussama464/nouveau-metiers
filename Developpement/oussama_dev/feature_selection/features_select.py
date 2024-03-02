from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA


@dataclass
class FeatureSelector:
    target: np.ndarray
    anova_k_features: int | str = "all"
    pca_variance_threshold: float = 0.99

    def __post_init__(self):
        "inverse the to_categorical transform in order to use select features"
        if self.target.ndim != 1:
            self.target = np.argmax(self.target, axis=1)

    def select_features(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[float]]:
        # configure to select all features
        fs = SelectKBest(score_func=f_classif, k=self.anova_k_features)
        # learn relationship from the training data
        fs.fit(X_train, self.target)
        # transform the train input data
        X_train_fs = fs.transform(X_train)
        # transform the test input data
        X_test_fs = fs.transform(X_test)
        return X_train_fs, X_test_fs, fs.scores_.tolist()

    # what are the scores for the features
    def bar_plot_features_scores(self, scores: list[float]) -> None:
        for i in range(len(scores)):
            print("feature %d: %f" % (i, scores[i]))

        plt.bar([i for i in range(len(scores))], scores)
        plt.show()

    def apply_pca(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        pca = PCA(n_components=self.pca_variance_threshold)
        X_train_reduced = pca.fit_transform(X_train)
        X_test_reduced = pca.transform(X_test)
        return X_train_reduced, X_test_reduced

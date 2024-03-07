from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class FeatureSelector(BaseEstimator, TransformerMixin):
    anova_k_features: int | str = "all"
    pca_variance_threshold: float = 0.99
    pipeline = Pipeline(
        steps=[
            (
                "feature_selection",
                SelectKBest(score_func=f_classif, k=anova_k_features),
            ),
            ("pca", PCA(n_components=pca_variance_threshold)),
        ]
    )

    def fit(self, X, y):
        return self.pipeline.fit(X, y)

    def transform(self, X, y=None):
        return self.pipeline.transform(X)

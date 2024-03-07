import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.model_selection import cross_val_score


class DummyClassifierWithAccuracy:
    def predict(self, X):
        """Predict method always returns a list of False for any input."""
        return [False] * len(X)

    def calculate_metrics(self, y_true):
        """Calculate precision, recall, F1-score, and accuracy based on true labels (y_true)."""
        y_pred = self.predict(y_true)

        # Convert lists to binary for easier calculation (True = 1, False = 0)
        y_true = [1 if x else 0 for x in y_true]
        y_pred = [1 if x else 0 for x in y_pred]

        # Calculating TP, FP, FN, TN
        TP = sum([1 for yt, yp in zip(y_true, y_pred) if yt == yp == 1])
        FP = sum([1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1])
        FN = sum([1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0])
        TN = sum([1 for yt, yp in zip(y_true, y_pred) if yt == yp == 0])

        # Calculating precision, recall, F1-score, and accuracy
        precision = TP / (TP + FP) if (TP + FP) else 0
        recall = TP / (TP + FN) if (TP + FN) else 0
        f1_score = (
            2 * ((precision * recall) / (precision + recall))
            if (precision + recall)
            else 0
        )
        accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) else 0

        return {
            "precision": precision,
            "recall": recall,
            "F1_score": f1_score,
            "accuracy": accuracy,
        }


# Example usage
labels = [
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    True,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    True,
    True,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    True,
    True,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    True,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
]

# Re-instantiate with accuracy calculation
dummy_classifier_with_accuracy = DummyClassifierWithAccuracy()
metrics_with_accuracy = dummy_classifier_with_accuracy.calculate_metrics(labels)
print(metrics_with_accuracy)

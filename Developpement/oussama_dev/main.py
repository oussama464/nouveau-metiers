import pandas as pd
from typing import Any
from data_preparation.data_prep import DataPreper, PaddingType, DataSmoother
from feature_extraction.features_eng import FeatureExtractor
from feature_selection.features_select import FeatureSelector
import numpy as np

# from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    StratifiedKFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
)
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from collections import Counter
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV


FILE_PATH = "/home/bobo/Desktop/nouveau_metier/Developpement/raw_csv/extracted_jobs.csv"


def get_formatted_raw_data(file_path: str) -> list[list[Any]]:

    df = pd.read_csv(file_path)
    df = df.sort_values(by="date")
    df_grouped = df.groupby(["job_code", "is_emerging_job"]).agg(list).reset_index()
    training_data = df_grouped.apply(
        lambda row: [row["job_code"], row["rawpop"], row["is_emerging_job"]], axis=1
    ).tolist()
    return training_data


def get_model():
    return Pipeline(
        steps=[
            ("over_sample", SMOTE(k_neighbors=4)),
            ("smoother", DataSmoother()),
            ("f_extractor", FeatureExtractor()),
            ("f_selector", FeatureSelector()),
            ("scaler", StandardScaler()),
            (
                "model",
                GradientBoostingClassifier(
                    random_state=42,
                    max_depth=3,
                    max_features="log2",
                    criterion="friedman_mse",
                    subsample=0.2,
                    n_estimators=8,
                ),
            ),
        ]
    )


def evaluate_model(cv):
    data = get_formatted_raw_data(FILE_PATH)
    prep = DataPreper(PaddingType.MAX_SIZE_NO_PAD)
    X, y = prep.prepare(data)
    model = get_model()
    # ‘precision’ ,"recall",roc_auc'
    scores = cross_val_score(model, X, y, scoring="accuracy", cv=cv, n_jobs=-1)
    return np.mean(scores), np.std(scores)


if __name__ == "__main__":
    # cv = KFold(n_splits=10, random_state=42, shuffle=True)
    # cv = LeaveOneOut()
    # cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    # cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
    acc, std = evaluate_model(cv)
    print(f"acc = {acc} , std = {std}")

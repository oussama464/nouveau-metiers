import pandas as pd
from typing import Any
from data_preparation.data_prep import DataPreper, PaddingType
from feature_extraction.features_eng import FeatureExtractor
from feature_selection.features_select import FeatureSelector
from models.lr_classifier import evaluate_model
from models.plotting_utils import plot_model_confusion_matrix

FILE_PATH = "/home/bobo/Desktop/nouveau_metier/Developpement/raw_csv/extracted_jobs.csv"


def get_formatted_raw_data(file_path: str) -> list[list[Any]]:

    df = pd.read_csv(file_path)
    df_grouped = df.groupby(["job_code", "is_emerging_job"]).agg(list).reset_index()

    # Transforming the grouped dataframe to the desired format
    training_data = df_grouped.apply(
        lambda row: [row["job_code"], row["rawpop"], row["is_emerging_job"]], axis=1
    ).tolist()
    return training_data


def prepare_model_data():
    perepr = DataPreper(PaddingType.MEAN)
    data = get_formatted_raw_data(FILE_PATH)
    smoothed_data = perepr.rolling_window_smoothing(data, 3)
    X_train, X_test, y_train, y_test = perepr.split_dataset_into_train_test(
        smoothed_data
    )
    f_extract_test, f_extract_train = FeatureExtractor(
        X_test, y_test
    ), FeatureExtractor(X_train, y_train)
    X_futures_test, y_test = f_extract_test.X, f_extract_test.y
    X_futures_train, y_train = f_extract_train.X, f_extract_train.y
    f_selector = FeatureSelector(
        y_train, anova_k_features=10, pca_variance_threshold=0.99
    )
    X_train_fs, X_test_fs, scores = f_selector.select_features(
        X_futures_train, X_futures_test
    )
    # f_selector.bar_plot_features_scores(scores)
    X_train_reduced, X_test_reduced = f_selector.apply_pca(X_train_fs, X_test_fs)
    return X_train_reduced, X_test_reduced, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_model_data()
    predictions, score = evaluate_model(X_train, X_test, y_train, y_test)
    plot_model_confusion_matrix(y_test, predictions, score)

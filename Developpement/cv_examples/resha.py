import pandas as pd
from typing import Any
import csv

FILE_PATH = "/home/bobo/Desktop/nouveau_metier/Developpement/raw_csv/extracted_jobs.csv"


def get_formatted_raw_data(file_path: str) -> list[list[Any]]:

    df = pd.read_csv(file_path)
    df = df.sort_values(by="date")
    df_grouped = df.groupby(["job_code", "is_emerging_job"]).agg(list).reset_index()
    training_data = df_grouped.apply(
        lambda row: [row["job_code"], row["rawpop"], row["is_emerging_job"]], axis=1
    ).tolist()
    return training_data


target_years = [str(i) for i in range(2015, 2024)]
# with open(FILE_PATH, "r") as f:
#     reader = csv.DictReader(f)
data = get_formatted_raw_data(FILE_PATH)


def filter_by_job_code(job_code):
    return list(filter(lambda x: x[0] == job_code, data))


print(filter_by_job_code("C2Z80"))

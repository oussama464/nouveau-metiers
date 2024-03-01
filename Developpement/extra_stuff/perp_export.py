import pandas as pd


cols = ["date", "job_code", "rawpop", "is_emerging_job"]
df = pd.read_csv(
    "/home/bobo/Desktop/nouveau_metier/Developpement/export/featured_dataset.csv",
    header=0,
    delimiter=";",
)
df = df[cols]
df.to_csv(
    "/home/bobo/Desktop/nouveau_metier/Developpement/raw_csv/extracted_jobs.csv",
    index=False,
    sep=",",
)

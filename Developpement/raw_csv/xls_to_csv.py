import pandas as pd
import glob

filenames = glob.glob("/home/bobo/Desktop/nouveau_metier/Developpement/data/*.xlsx")
cols_subset = [
    "annee",
    "Famille_met",
    "Lbl_fam_met",
    "Code métier BMO",
    "met",
    "xmet",
    "smet",
]
renamed_cols = {
    "Famille_metier": "Famille_met",
    "Libellé de famille de métier": "Lbl_fam_met",
}
dataframes_list = []
orig_len = 0
for file in filenames:
    df = pd.read_excel(
        file,
        sheet_name=1,
    )
    df.rename(
        columns=renamed_cols,
        inplace=True,
    )
    df = df[cols_subset]
    orig_len += df.shape[0]

    dataframes_list.append(df)

final_df = pd.concat(dataframes_list, ignore_index=True)
final_df.to_csv(
    "/home/bobo/Desktop/nouveau_metier/Developpement/raw_csv/row.csv",
    sep=",",
    encoding="utf-8",
    index=False,
)

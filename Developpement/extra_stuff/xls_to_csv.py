import pandas as pd
import glob
import re

filenames = glob.glob("/home/bobo/Desktop/nouveau_metier/Developpement/data/*.xlsx")
new_col_names = [
    "annee",
    "BE",
    "NOMBE",
    "Famille_met",
    "Lbl_fam_met",
    "code_metier_BMO",
    "nommetier",
    "Dept",
    "NomDept",
    "met",
    "xmet",
    "smet",
    "REG",
    "NOM_REG",
]
col_to_drop = "NORMAL"
dataframes_list = []
orig_len = 0


# def file_pattern_exists(filename: str, pattern: str) -> bool:
#     return filename.split("/")[-1].startswith(pattern)


# def extract_numbers(filename: str):
#     filename = filename.split("/")[-1].split(".")[0]
#     return re.findall(r"\d+", filename)[0]


# for file in filenames:
#     df = pd.read_excel(
#         file,
#         sheet_name=1,
#     )
#     if file_pattern_exists(file, "Base_open_data"):
#         print(file)
#         num = extract_numbers(file)
#         df = pd.DataFrame(
#             {
#                 "annee": df["annee"],
#                 "BE": df[f"BE{num}"],
#                 "NOMBE": df[f"NOMBE{num}"],
#                 "Famille_met": df["Famille_met"],
#                 "Lbl_fam_met": df["Lbl_fam_met"],
#                 "code_metier_BMO": df["Code métier BMO"],
#                 "nommetier": df["Nom métier BMO"],
#                 "Dept": df["Dept"],
#                 "NomDept": df["NomDept"],
#                 "met": df["met"],
#                 "xmet": df["xmet"],
#                 "smet": df["smet"],
#                 "REG": df["REG"],
#                 "NOM_REG": df["NOM_REG"],
#             }
#         )

#     if col_to_drop in df.columns:
#         df.drop(col_to_drop, axis=1, inplace=True)
#     print(df.columns)
#     df.columns = new_col_names
#     dataframes_list.append(df)
#     orig_len += df.shape[0]
#     print(file)

#     print(len(df.columns))


# final_df = pd.concat(dataframes_list, ignore_index=True)

# assert final_df.shape[0] == orig_len
# print(final_df.shape[0])
# # print(final_df.dtypes)
# final_df.to_csv(
#     "/home/bobo/Desktop/nouveau_metier/Developpement/raw_csv/raw.csv",
#     sep=",",
#     encoding="utf-8",
#     index=False,
# )

df = pd.read_excel(
    "/home/bobo/Desktop/nouveau_metier/Developpement/data/ResmetBE15.xlsx", sheet_name=1
)
print(df.columns)

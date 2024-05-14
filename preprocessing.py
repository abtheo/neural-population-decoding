
import pandas as pd

# path = "../brca_tcga_pub2015/"
path = "../prad_tcga/"

target_label = "AJCC_PATHOLOGIC_TUMOR_STAGE" if 'brca' in path else "GLEASON_SCORE"

df_label = pd.read_csv(
    path+"data_clinical_patient.txt", sep="\t", header=4)[["PATIENT_ID",
                                                          target_label]]
"""
    Each column represents one patient.
    Merge the omic feature CSVs vertically, 
    then map to labels via PATIENT_ID
"""

df_gene = pd.read_csv(
    path+"data_linear_cna.txt", sep="\t")
df_miRNA = pd.read_csv(
    path+"data_mrna_seq_v2_rsem.txt", sep="\t")
df_methyl = pd.read_csv(
    path+"data_methylation_hm450.txt", sep="\t")

df = pd.concat([df_gene, df_miRNA, df_methyl]).drop(
    "Entrez_Gene_Id", axis=1).rename(columns={"Hugo_Symbol": "OMIC_ID"})
df.reset_index()

"""
    Each column in df has a header corresponding to the PATIENT_ID,
    and each row is associated with a single omic feature indexed by the OMIC_ID.

    To correlate with the target variable within df_label,
    we add a row to the end of the multi_omic csv.
"""
target = pd.concat(
    [pd.Series('Target'), df_label[target_label]])

if 'brca' not in path:
    target = target[:-1]

print("Labels inside Patient_ID.txt but not in features:\n")
for v in df_label["PATIENT_ID"]:
    anymatch = False
    for d in df.columns:
        if v in d:
            anymatch = True
    if not anymatch:
        print(v)

print("Labels inside features but not in Patient_ID.txt:\n")

for v in df.columns:
    anymatch = False
    v = v[:-3]
    for d in df_label["PATIENT_ID"]:
        if v in d:
            anymatch = True
    if not anymatch:
        print(v)

df = df.append(pd.Series(target.values, index=df.columns),
               ignore_index=True)

print(df)
df.to_csv(path + "multi_omic.csv", index=False)

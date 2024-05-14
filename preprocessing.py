
import pandas as pd

path = "../brca_tcga_pub2015/"
# path = "../prad_tcga/"

target_label = ["AJCC_PATHOLOGIC_TUMOR_STAGE"] if 'brca' in path else [
    "GLEASON_PATTERN_PRIMARY", "GLEASON_PATTERN_SECONDARY"]

t_select = ["PATIENT_ID"] + target_label

df_label = pd.read_csv(
    path+"data_clinical_patient.txt", sep="\t", header=4)[t_select]
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

# Explicitly match targets to labels
targets = ['Target']
matched_ids = ['OMIC_ID']
for patient_id in df.columns[1:]:
    patient = df_label[df_label["PATIENT_ID"] == patient_id[:-3]]

    # Arrange into Gleason score groups for PRCA
    if not 'brca' in path:
        patient_gleason_scores = patient[target_label].values[0]
        gleason_group = "NA"
        if all(patient_gleason_scores == [3, 4]):
            gleason_group = "34"

        if all(patient_gleason_scores == [4, 3]):
            gleason_group = "43"

        if all(patient_gleason_scores == [4, 5]) or all(patient_gleason_scores == [5, 4]):
            gleason_group = "9"

        if gleason_group != "NA":
            targets += [gleason_group]
            matched_ids += [patient_id]
    # Get tumor stage for BRCA
    else:
        targets.append(patient[target_label].values[0][0])
        matched_ids += [patient_id]


df = df[matched_ids].append(pd.Series(targets, index=matched_ids),
                            ignore_index=True)

print(df)
df.to_csv(path + "multi_omic.csv", index=False)

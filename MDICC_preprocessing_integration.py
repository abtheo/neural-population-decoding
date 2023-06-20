
import pandas as pd
import numpy as np

root_path = "D:\\Thesis\\MDICC_data\\"

if __name__ == "__main__":
    """ 
        Collect each CSV within the each dataset
        Associate with survival.csv (and label.csv ?) 
    """
    path_extension = "BRCA\\"
    path = root_path + path_extension
    df_label = pd.read_csv(
        path+"label.csv").rename(columns={'Unnamed: 0': 'PATIENT_ID'})

    """
        Each column represents one patient.
        Merge the CSVs vertically, 
        then map to survival + labels via PATIENT_ID
    """
    df_gene = pd.read_csv(
        path+"data\\gene.csv").rename(columns={'Unnamed: 0': 'GENE_ID'})
    df_miRNA = pd.read_csv(
        path+"data\\miRNA.csv").rename(columns={'Unnamed: 0': 'miRNA_ID'})
    df_methyl = pd.read_csv(
        path+"data\\methyl.csv").rename(columns={'Unnamed: 0': 'METHYL_ID'})

    df = pd.concat([df_gene, df_miRNA, df_methyl])
    """ Merge multi-omic IDs into single column """
    df.insert(0, "OMIC_ID", df[["GENE_ID", "miRNA_ID",
                                "METHYL_ID"]].bfill(axis=1).iloc[:, 0])

    # Do not drop labels for integration run,
    # will be needed for Self_Organising_Map
    # df.drop(labels=["GENE_ID", "miRNA_ID", "METHYL_ID"], axis=1, inplace=True)

    """ 
        Each column in df has a header corresponding to the PATIENT_ID,
        and each row is associated with a single omic feature indexed by the OMIC_ID.

        To correlate with the target variable in the column label2 within df_label,
        we add a row to the end of the multi_omic csv.
    """
    # prepend with label for the OMIC_ID column (will be dropped later anyway)
    # add NaNs to target col
    try:
        target = pd.concat(
            [pd.Series(['Target', np.nan, np.nan, np.nan]), df_label['label2']])
    except:
        target = pd.concat(
            [pd.Series(['Target', np.nan, np.nan, np.nan]), df_label['class2']])

    df = df.append(pd.Series(target.values, index=df.columns),
                   ignore_index=True)

    print(df)
    df.to_csv(path + "multi_omic_integration.csv", index=False)

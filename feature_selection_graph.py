import pandas as pd
import networkx as nx
import simpsom as sps
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import matplotlib
from mrmr import mrmr_classif
from tqdm import tqdm
from sklearn.feature_selection import VarianceThreshold
from PIL import Image
import os
from numpy.lib.type_check import imag
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from collections import Counter
import xgboost as xgb
import glob

num_features = 20
subtype = "BRCA"
path = f"D:\\Thesis\\MDICC_data\\{subtype}\\multi_omic.csv"
directory_path = f"patient_som_data\\{subtype}"

# read multi-omic csv data
df = pd.read_csv(path)
# extract the target variable
target = df[df['OMIC_ID'] == 'Target'].drop("OMIC_ID", axis=1).T

"""
        iSOM-GSN:  A filtering step was applied by removing those features whose variance was below 0.2%.
        As a result, features with at least 80% zero values were removed, reducing the number of features to 16 000.

        Then perform Minimum Redundancy Maximum Relevance (mRMR)
    """
# Single omic
# data = df[df["OMIC_ID"].str.contains(
#     "cg")][:-1].drop("OMIC_ID", axis=1).T  # "hsa-mir", "cg"

# Transpose data into (patients, features) for feature engineering
data = df.drop("OMIC_ID", axis=1)[:-1].T

# Variance thresholding
threshold_n = 0.8
sel = VarianceThreshold(threshold=(threshold_n * (1 - threshold_n)))
sel_var = sel.fit_transform(data)
old_shape = data.shape
data = data[data.columns[sel.get_support(indices=True)]]
print(
    f"Variance thresholding reduced number of omic features from {old_shape[1]} down to {data.shape[1]}.")
data.columns = range(data.shape[1])
# MRMR feature selection

# data.reset_index(inplace=True, drop=True)
mrmr = mrmr_classif(data, target, K=num_features, return_scores=True)
selected_features = mrmr[0]
print("Feature relevances: ", mrmr[1])
print("Feature reduncancies: ", mrmr[2])

old_shape = data.shape  # just for printing
data = data[selected_features]
print(
    f"Minimum Redundancy Maximum Relevance reduced number of omic features from {old_shape[1]} to {data.shape[1]}")
clf = xgb.XGBClassifier(n_estimators=20, max_depth=3,
                        objective='binary:logistic')
clf.fit(data, target)
selected_features_xgb = (-clf.feature_importances_).argsort()[
    :num_features]

# for BRCA!
# selected_features = [0,   626,  1267,  1987,  2238,  2503,  2594,  4491,  5010,
#                      6091,  6756,  7717,  7973,  8188, 10034, 10626, 11001, 11002,
#                      11004, 11005, 11006, 11197, 11359, 11790, 12325, 12621, 13441,
#                      14452]
selected_features = np.union1d(selected_features, selected_features_xgb)
# print("XGBOOST: ", selected_features)

old_shape = data.shape  # just for printing
data = data[selected_features]
print(
    f"XGBoost Feature Importance reduced number of omic features from {old_shape[1]} to {data.shape[1]}")

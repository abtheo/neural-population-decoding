import pandas as pd
import networkx as nx
import simpsom as sps
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
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

subtype = "BRCA"
path = f"D:\\Thesis\\MDICC_data\\{subtype}\\multi_omic.csv"
directory_path = f"patient_som_data/{subtype}"


def rgba_to_binary(image_path):
    # Open the RGBA image
    rgba_image = Image.open(image_path)

    # Convert the RGBA image to grayscale
    grayscale_image = rgba_image.convert("L")

    # Convert the grayscale image to binary format (black and white)
    binary_image = grayscale_image.point(
        lambda x: 0 if x < 128 else 255, mode='1')

    # Save the binary image
    # binary_image.save("binary_image.png")
    return np.array(binary_image)


if __name__ == "__main__":
    # read multi-omic csv data
    df = pd.read_csv(path)
    # extract the target variable
    target = df[df['OMIC_ID'] == 'Target'].drop("OMIC_ID", axis=1).T

    """
        iSOM-GSN:  A filtering step was applied by removing those features whose variance was below 0.2%.
        As a result, features with at least 80% zero values were removed, reducing the number of features to 16 000.

        Then perform Minimum Redundancy Maximum Relevance (mRMR)
    """
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
    K = 10
    selected_features = mrmr_classif(data, target, K=K)

    # old_shape = data.shape  # just for printing
    # data = data[selected_features]
    # print(
    #     f"Minimum Redundancy Maximum Relevance reduced number of omic features from {old_shape[1]} to {data.shape[1]}")
    clf = xgb.XGBClassifier(n_estimators=20, max_depth=3,
                            objective='binary:logistic')
    clf.fit(data, target)
    selected_features_xgb = (-clf.feature_importances_).argsort()[:K]

    selected_features = np.union1d(selected_features, selected_features_xgb)
    # print("XGBOOST: ", selected_features)

    old_shape = data.shape  # just for printing
    data = data[selected_features]
    print(
        f"MRMR + XGBoost Feature Importance reduced number of omic features from {old_shape[1]} to {data.shape[1]}")

    # Transpose data back into (features, patients) for Self-Organising Map
    data_T = data.T

    # applying scaling to make values between some range 0-1/-1-2 ,as need for Kohens SOM
    som_scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    som_scaler.fit(data_T)
    data_T = som_scaler.transform(data_T)

    # SOM parameters
    ht = 10
    wd = 10
    no_of_epocs = 5

    # Train Self-Organising Map
    net = sps.SOMNet(ht, wd, data_T, PBC=False)
    net.colorEx = False
    learning_rate = 0.05
    net.PCI = True  # The weights will be initialised with PCA.
    net.train(start_learning_rate=learning_rate, epochs=no_of_epocs)

    node_positions = net.project(data_T)
    # labels=list(df["OMIC_ID"][selected_features])

    # G = nx.chvatal_graph()
    # nx.draw_networkx_nodes(G,
    #                        node_positions,
    #                        nodelist=[i for i in range(K)],
    #                        node_color=[0, 0, 0],
    #                        edgecolors=[0, 0, 0],
    #                        node_size=1000,
    #                        alpha=0.8)

    # nx.draw_networkx_labels(
    #     G, node_positions, labels={k: v for k, v in enumerate(df["OMIC_ID"][selected_features].values)}, font_size=10)
    # plt.axis('on')
    # plt.show()
    # print("POS = ", node_positions)
    # plt.savefig('Template.png', bbox_inches='tight', dpi=72)

    # Okay, so the point of all that was to determine the node_positions.
    # The graph we just drew is only a visual representation of the positions,
    # but the sizes are an arbitrary constant.

    # Now, using the node_positions, we take each patient
    # and use their values of the selected_features
    # to determine the node_sizes.
    # However, we first need to normalise the *features* into a range.

    # Actually, here is where we should perform SMOTE.
    # We also need some way to remember
    # which data is synthetic and which is the original,
    # so that we can do testing on only original data.
    oversample = SMOTE(sampling_strategy=0.33)
    data_smote, target_smote = oversample.fit_resample(data, target)

    is_original = [np.any(np.all(data == d, axis=1))
                   for d in data_smote.values]

    # First clear out old files
    # input("WARNING: Deleteing all files in directory {directory_path}. Continue?")

    files = glob.glob(f"{directory_path}/*")
    for f in files:
        os.remove(f)

    np.save(f"{directory_path}/original.npy", is_original)
    np.save(f"{directory_path}/target.npy", target_smote)

    # TODO: Also exponent the values to produce a greater variance?
    # Let's try Z-score normalization first, then rescale
    var_scaler = StandardScaler()
    feature_scaler = MinMaxScaler(copy=True, feature_range=(10, 1200))
    data_norm = feature_scaler.fit_transform(
        var_scaler.fit_transform(data_smote))

    for i, patient in tqdm(enumerate(data_norm)):
        # Draw SOM per patient,
        # encoding their feature expression values as the node_sizes
        G = nx.chvatal_graph()
        nx.draw_networkx_nodes(G,
                               node_positions,
                               nodelist=[i for i in range(
                                   len(selected_features))],
                               node_color=[0, 0, 0],
                               edgecolors=[0, 0, 0],
                               node_size=patient,
                               alpha=1, margins=0.25)

        plt.axis('off')
        plt.savefig(
            f'{directory_path}/patient_{i}.png', bbox_inches='tight', dpi=36)
        # plt.show()

    """  Compress .png images into single .npy file """
    directory = os.listdir(directory_path)
    output = []
    for file in directory:
        if ".npy" in file:
            continue
        img = rgba_to_binary(directory_path + "/" + file)
        output.append(img)

    image_array_stack = np.stack(output, axis=0)
    print(image_array_stack.shape)
    np.save(f"{directory_path}/SOM_data", image_array_stack)

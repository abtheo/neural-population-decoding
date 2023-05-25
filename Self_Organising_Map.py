
import pandas as pd
import networkx as nx
import simpsom as sps
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from mrmr import mrmr_classif
from tqdm import tqdm
from sklearn.feature_selection import VarianceThreshold

path = "D:\\Thesis\\MDICC_data\\BRCA\\multi_omic.csv"

if __name__ == "__main__":
    # read multi-omic csv data
    df = pd.read_csv(path)
    # extract the target variable
    target = df[df['OMIC_ID'] == 'Target'].drop("OMIC_ID", axis=1).T
    """
        Conduct a Chi-squared test 
        of the multi-omic features against the target variable of cancer diagnosis.
        We can then take the top N features to generate the self-organising map.

        OKAY so a Chi-squared test is for the observed frequency of categorical variables,
        however our features are floating-point numbers.
        Will have to figure out some other method of selecting the top-20 omic features.

        iSOM-GSN:  A filtering step was applied by removing those features whose variance was below 0.2%. 
        As a result, features with at least 80% zero values were removed, reducing the number of features to 16 000.

        Minimum Redundancy Maximum Relevance (mRMR)

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

    # MRMR feature selection
    K = 10
    selected_features = mrmr_classif(data, target, K=K)
    old_shape = data.shape  # just for printing
    data = data[selected_features]
    print(
        f"Minimum Redundancy Maximum Relevance reduced number of omic features from {old_shape[1]} to {data.shape[1]}")

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

    node_positions = net.project(
        data_T, labels=list(df["OMIC_ID"][selected_features]))

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
#    plt.savefig('Template.png', bbox_inches='tight', dpi=72)

    # Okay, so the point of all that was to determine the node_positions.
    # The graph we just drew is only a visual representation of the positions,
    # but the sizes are an arbitrary constant.

    # Now, using the node_positions, we take each patient
    # and use their values of the selected_features
    # to determine the node_sizes.
    # However, we first need to normalise the *features* into a range.
    feature_scaler = MinMaxScaler(copy=True, feature_range=(100, 1000))
    feature_scaler.fit(data)
    data_norm = feature_scaler.transform(data)

    for i, patient in tqdm(enumerate(data_norm)):
        # Draw SOM per patient,
        # encoding their feature expression values as the node_sizes
        G = nx.chvatal_graph()
        nx.draw_networkx_nodes(G,
                               node_positions,
                               nodelist=[i for i in range(K)],
                               node_color=[0, 0, 0],
                               edgecolors=[0, 0, 0],
                               node_size=patient,
                               alpha=1, margins=0.5)
        plt.axis('off')
        plt.savefig(
            f'./patient_som_data/BRCA/patient_{df.columns[i+1]}.png', bbox_inches='tight', dpi=36)
        # plt.show()

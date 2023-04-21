
import pandas as pd
import networkx as nx
import simpsom as sps
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from mrmr import mrmr_classif
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
    selected_features = mrmr_classif(data, target, K=10)
    old_shape = data.shape
    data = data[selected_features]
    print(
        f"Minimum Redundancy Maximum Relevance reduced number of omic features from {old_shape[1]} to {data.shape[1]}")

    # Transpose data back into (features, patients) for Self-Organising Map
    data = data.T

    # applying scaling to make values between some range 0-1/-1-2 ,as need for Kohens SOM
    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    scaler.fit(data)
    data = scaler.transform(data)

    # SOM parameters
    ht = 10
    wd = 10
    no_of_epocs = 5

    net = sps.SOMNet(ht, wd, data, PBC=False)
    net.colorEx = False
    learning_rate = 0.05
    net.PCI = True  # The weights will be initialised with PCA.
    net.train(start_learning_rate=learning_rate, epochs=no_of_epocs)

    bmu = net.project(data, labels=list(df["OMIC_ID"][selected_features]))
    G = nx.chvatal_graph()

    nx.draw_networkx_nodes(G, bmu,
                           nodelist=[i for i in range(data.shape[0])],
                           node_color='w',
                           edgecolors=[0, 0, 0],
                           #                       node_color=[[1,1,0],[1,1,0],[1,1,0],[1,1,0],[1,1,0],'b','r','c','y','k','m'],
                           #                       node_color=[[1,1,0],[1,1,0],[1,1,0],[1,1,0],[1,1,0],'b','r','c','y','k','m'],
                           node_size=1000, alpha=0.8)

    nx.draw_networkx_labels(
        G, bmu, labels={k: v for k, v in enumerate(df["OMIC_ID"][selected_features].values)}, font_size=10)
    plt.axis('on')
    plt.show()
    print("EPOCHS  = ", no_of_epocs)
    print("POS = ", bmu)
#    plt.savefig('Template.png', bbox_inches='tight', dpi=72)
    print('***************************DONE!!*****************************')

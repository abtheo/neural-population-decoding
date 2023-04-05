
import pandas as pd
import networkx as nx
import simpsom as sps
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

path = "D:\\Thesis\\MDICC_data\\BRCA\\multi_omic.csv"

if __name__ == "__main__":
    # read multi-omic csv data
    df = pd.read_csv(path)
    data = df.drop("OMIC_ID", axis=1).values[0:1000]

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

    bmu = net.project(data, labels=list(df["OMIC_ID"][0:1000]))
    G = nx.chvatal_graph()

    nx.draw_networkx_nodes(G, bmu,
                           nodelist=[i for i in range(data.shape[1])],
                           node_color='w',
                           edgecolors=[0, 0, 0],
                           #                       node_color=[[1,1,0],[1,1,0],[1,1,0],[1,1,0],[1,1,0],'b','r','c','y','k','m'],
                           #                       node_color=[[1,1,0],[1,1,0],[1,1,0],[1,1,0],[1,1,0],'b','r','c','y','k','m'],
                           node_size=1000, alpha=0.8)

    nx.draw_networkx_labels(
        G, bmu, df["OMIC_ID"][0:1000].to_dict(), font_size=10)
    plt.axis('on')
    plt.show()
    print("EPOCHS  = ", no_of_epocs)
    print("POS = ", bmu)
#    plt.savefig('Template.png', bbox_inches='tight', dpi=72)
    print('***************************DONE!!*****************************')

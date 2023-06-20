from net_integration import NetworkIntegration
from util import Parameters
from data_handler import DataHandler
import pickle
import numpy as np
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split

subtype = "BRCA"


def store_spikes(X, data_handler, train=True):
    for i in range(0, X.shape[0], data_handler.s_slice):
        slice_start = i
        slice_end = slice_start + data_handler.s_slice
        # takes the first S images from MNIST
        X_slice = X[slice_start:slice_end]  # shape (100,28,28)
        X_spikes = data_handler.pixels_to_spikes(
            X_slice)  # shape (100,28,28,2,150)

        with open(f'./data/spikes/{subtype}/X_spikes_{"train" if train else "test"}_{i}.pkl', 'wb') as f:
            pickle.dump(X_spikes, f)


def run_hierarchical():
    argv = []
    S = 10

    # Set the parameters for Integration network
    P = Parameters()
    P.set_integration(argv)
    P.s_slice = S
    P.topdown_enabled = True

    # Set the parameters for Hierarchical network
    P_h = Parameters()
    P_h.set_hierarchical(argv)
    P_h.s_slice = S
    P_h.topdown_enabled = True

    # Initialize the hierarchical network
    network = NetworkIntegration(P, P_h, P_h, P_h)

    # Initialize the DataHandler
    data_handler = DataHandler(P)

    with open(f'./patient_som_data/{subtype}/SOM_data.npy', 'rb') as f:
        X = np.load(f)

    with open(f'./patient_som_data/{subtype}/target.npy', 'rb') as f:
        targets = np.load(f)

    # PREP DATA
    # Split into train / test
    X_train, X_test, labels_train, labels_test = train_test_split(
        X, targets, test_size=0.2, random_state=42)
    labels_test = [int(x) for x in labels_test]

    """
     So we need to unpack the tuple of omics here,
     after the split but not passed into the spikes.
     Actually, we need to parameterize the store_spikes function
     to take a path for each omic we unpack...

    """

    # Store the data in slices of size <self.s_slice>
    store_spikes(X_train, data_handler)
    store_spikes(X_test, data_handler, train=False)

    # TRAIN NETWORK
    # Iterate over the spike data in slices of size <data_handler.s_slice>
    time_start = time.time()
    for ith_slice in tqdm(range(0, X_train.shape[0]//S)):

        # Retrieve slice <ith_slice> of the spike data
        with open(f'./data/spikes/{subtype}/X_spikes_train_{ith_slice*S}.pkl', 'rb') as f:
            spike_data = pickle.load(f)

        for index_im, spike_times in enumerate(spike_data):
            # Convert the flat <spike_times> to a tiled array
            spike_times = network.tile_input(
                spike_times, network.s_os, network.w, network.h)

            # Determine the complete dataset index (rather than that of the slice)
            index_im_all = ith_slice*data_handler.s_slice+index_im
            network.index_im_all = index_im_all  # $$$
            network.print_interval = P.print_interval  # $$$

            # Propogate through the network according to the current timestep and given spike times
            for t in range(data_handler.ms):
                network.propagate(spike_times, t, learn_h=True, learn_o=True)
            network.reset()  # Reset the network between images

            # Print, plot, and save data according to the given parameters
            time_start = network.print_plot_save(
                data_handler, X_train, labels_train, index_im_all, X_train.shape[0], P, time_start)

            # Reset spike counters
            network.n_spikes_since_reset_h = np.zeros(
                (network.s_os, network.s_os, network.K_h), dtype=np.uint16)
            network.n_spikes_since_reset_o = np.zeros(
                network.K_o, dtype=np.uint16)

    # TESTING

    # The number of times each neuron spiked for each label
    neuron_label_counts = np.zeros((network.K_o, 2), dtype=np.uint32)

    # For each image the number of times each neuron spiked
    neuron_image_counts = np.zeros((X_test.shape[0], network.K_o))

    index_im_all = 0
    time_start = time.time()
    # Iterate over the spike data in slices of size <data_handler.s_slice>
    for ith_slice in range(0, X_test.shape[0]//S):

        # Retrieve slice <ith_slice> of the spike data
        with open(f'./data/spikes/{subtype}/X_spikes_test_{ith_slice*S}.pkl', 'rb') as f:
            spike_data = pickle.load(f)

        for index_im, spike_times in enumerate(spike_data):

            # Convert the flat <spike_times> to a tiled array
            spike_times = network.tile_input(
                spike_times, network.s_os, network.w, network.h)

            # Determine the complete dataset index (rather than that of the slice)
            index_im_all = ith_slice*data_handler.s_slice+index_im
            network.index_im_all = index_im_all  # $$$
            network.print_interval = P.print_interval  # $$$

            # Retrieve the label of the current image
            label = labels_test[index_im_all]

            # Propogate through the network according to the current timestep and given spike times
            for t in range(data_handler.ms):
                network.propagate(
                    spike_times, t, learn_h=False, learn_o=False)
            network.reset()  # Reset the network between images

            # Keep track of the results
            neuron_label_counts[:, label] += network.n_spikes_since_reset_o
            neuron_image_counts[index_im_all] = network.n_spikes_since_reset_o

            # Print, plot, and save data according to the given parameters
            time_start = network.print_plot_save(
                data_handler, X_test, labels_test, index_im_all, X_test.shape[0], P, time_start)

            # Evaluate results after every <evaluation_interval> images
            if index_im_all > 0 and index_im_all % 10 == 0:
                print("\nEvaluating results after {} images:".format(
                    index_im_all))
                network.evaluate_results(data_handler, neuron_label_counts[:, :index_im_all+1],
                                         neuron_image_counts[:index_im_all+1], labels_test[:index_im_all+1])

            # Reset spike counters
            network.n_spikes_since_reset_h = np.zeros(
                (network.s_os, network.s_os, network.K_h), dtype=np.uint16)
            network.n_spikes_since_reset_o = np.zeros(
                network.K_o, dtype=np.uint16)


if __name__ == "__main__":
    run_hierarchical()

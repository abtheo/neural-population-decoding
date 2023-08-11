from typing import final
from net_hierarchical import NetworkHierarchical
from util import Parameters
from data_handler import DataHandler
import pickle
import numpy as np
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

subtype = "BRCA"


def store_spikes(X, data_handler, train=True, clear=False):
    if clear:
        files = glob.glob(f'./data/spikes/{subtype}/*')
        for f in files:
            os.remove(f)

    for i in range(0, X.shape[0], data_handler.s_slice):
        slice_start = i
        slice_end = slice_start + data_handler.s_slice
        X_slice = X[slice_start:slice_end]
        X_spikes = data_handler.pixels_to_spikes(
            X_slice)

        with open(f'./data/spikes/{subtype}/X_spikes_{"train" if train else "test"}_{i}.pkl', 'wb') as f:
            pickle.dump(X_spikes, f)


def run_hierarchical():
    argv = []

    # Set the parameters
    P = Parameters()
    P.set_hierarchical(argv)
    S = 10
    P.s_slice = S
    P.topdown_enabled = True
    P.K_h = 32
    P.K_o = 8

    data_handler = DataHandler(P)
    network = NetworkHierarchical(P)

    with open(f'./patient_som_data_smote/{subtype}/SOM_data.npy', 'rb') as f:
        X = np.load(f)

    with open(f'./patient_som_data_smote/{subtype}/target.npy', 'rb') as f:
        targets = np.load(f)

    with open(f'./patient_som_data_smote/{subtype}/original.npy', 'rb') as f:
        original = np.load(f)

    # PREP DATA
    # Perform K-folds cross-validation:
    # First, shuffle all the arrays while preserving order
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    targets = targets[indices]
    original = original[indices]
    # Then split sequentially for each fold
    K = 5
    slice = len(targets)//K
    final_df = pd.DataFrame()
    for k in range(K):
        # This range defines the test set.
        lower = slice*k
        upper = min(slice*(k+1), len(targets))
        X_test = X[lower:upper]
        original_test = original[lower:upper]
        labels_test = targets[lower:upper]

        # Remove synthetic / SMOTE examples from the test set.
        X_test = X_test[original_test]
        labels_test = labels_test[original_test].flatten()

        # The training set is everything outside the test range.
        X_train = X[:lower]
        labels_train = targets[:lower]
        if X_train.shape[0] == 0:
            X_train = X[upper:]
            labels_train = targets[upper:]
        else:
            X_train = np.concatenate((X_train, X[upper:]))
            labels_train = np.concatenate((labels_train, targets[upper:]))

        print(
            f"Starting fold {k}. Test set bounds are {lower} to {upper} from a total length of {len(targets)}.")

        # Store the data in slices of size <self.s_slice>
        store_spikes(X_train, data_handler, clear=True)
        store_spikes(X_test, data_handler, train=False)

        # TRAIN NETWORK
        # Iterate over the spike data in slices of size <data_handler.s_slice>
        time_start = time.time()
        for epoch in range(2):

            # The number of times each neuron spiked for each label
            neuron_label_counts = np.zeros((network.K_o, 2), dtype=np.uint32)

            for ith_slice in tqdm(range(0, X_train.shape[0]//S)):
                # Retrieve slice <ith_slice> of the spike data
                with open(f'./data/spikes/{subtype}/X_spikes_train_{ith_slice*S}.pkl', 'rb') as f:
                    spike_data = pickle.load(f)

                for index_im, spike_times in enumerate(spike_data):
                    # Take the first timestep, split between black and white dims
                    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 5))
                    # axes[0].imshow(spike_times[..., 0, 0],
                    #                cmap=plt.get_cmap('gray'))
                    # axes[1].imshow(spike_times[..., 1, 0],
                    #                cmap=plt.get_cmap('gray'))
                    # plt.show()

                    # Convert the flat <spike_times> to a tiled array
                    spike_times = network.tile_input(
                        spike_times, network.s_os, network.w, network.h)

                    # print(f"Spike shape: {spike_times.shape}")
                    # (4, 4, 7, 7, 2, 150)

                    # Determine the complete dataset index (rather than that of the slice)
                    index_im_all = ith_slice*data_handler.s_slice+index_im
                    network.index_im_all = index_im_all  # $$$
                    network.print_interval = P.print_interval  # $$$

                    # Propogate through the network according to the current timestep and given spike times
                    for t in range(data_handler.ms):
                        network.propagate(
                            spike_times, t, learn_h=True, learn_o=True)
                    network.reset()  # Reset the network between images

                    # Print, plot, and save data according to the given parameters
                    # network.print_plot_save(
                    #     data_handler, X_train, labels_train, index_im_all, X_train.shape[0], P, time_start)

                    label = int(labels_train[index_im_all])

                    neuron_label_counts[:,
                                        label] += network.n_spikes_since_reset_o

                    # data_handler.inspect_spike_data(
                    #     X_train, spike_data, ith_slice, S, "tag_mnist", n_inspections=3, train=True)

                    # Reset spike counters
                    network.n_spikes_since_reset_h = np.zeros(
                        (network.s_os, network.s_os, network.K_h), dtype=np.uint16)
                    network.n_spikes_since_reset_o = np.zeros(
                        network.K_o, dtype=np.uint16)

        # TESTING
        index_im_all = 0
        time_start = time.time()

        # For each image the number of times each neuron spiked
        neuron_image_counts = np.zeros((X_test.shape[0], network.K_o))

        # Iterate over the spike data in slices of size <data_handler.s_slice>
        for ith_slice in range(0, X_test.shape[0]//data_handler.s_slice):

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

                # Propogate through the network according to the current timestep and given spike times
                for t in range(data_handler.ms):
                    network.propagate(
                        spike_times, t, learn_h=False, learn_o=False)
                network.reset()  # Reset the network between images

                neuron_image_counts[index_im_all] = network.n_spikes_since_reset_o

                # Print, plot, and save data according to the given parameters
                # time_start = network.print_plot_save(
                #     data_handler, X_test, labels_test, index_im_all, X_test.shape[0], P, time_start)

                # Evaluate results after every <evaluation_interval> images
                if index_im_all > 0 and index_im_all % 10 == 0:
                    print("\nEvaluating results after {} images:".format(
                        index_im_all))
                    results_df = network.evaluate_results(data_handler, neuron_label_counts,
                                                          neuron_image_counts[:index_im_all+1], labels_test[:index_im_all+1])
                    print(results_df)

                # Reset spike counters
                network.n_spikes_since_reset_h = np.zeros(
                    (network.s_os, network.s_os, network.K_h), dtype=np.uint16)
                network.n_spikes_since_reset_o = np.zeros(
                    network.K_o, dtype=np.uint16)

        results_df = network.evaluate_results(data_handler, neuron_label_counts,
                                              neuron_image_counts, labels_test)
        print(f"Results for fold {k}:")
        print(results_df)

        if len(final_df) == 0:
            final_df = results_df
        else:
            final_df = pd.concat([final_df, results_df])

    final_df.to_csv("./results_df.csv")


if __name__ == "__main__":
    np.random.seed(13)
    run_hierarchical()

from net_hierarchical import NetworkHierarchical
from util import Parameters
from data_handler import DataHandler
import pickle
import numpy as np
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split
from self_organising_map import generate_patient_soms
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from logistic_regression import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

subtype = "KIRC"


def store_spikes(X, data_handler, clear=False):
    if clear:
        files = glob.glob(f'./data/spikes/{subtype}/*')
        for f in files:
            os.remove(f)
    print("Storing X Spikes...")
    for i in tqdm(range(0, X.shape[0], data_handler.s_slice)):
        slice_start = i
        slice_end = slice_start + data_handler.s_slice
        X_slice = X[slice_start:slice_end]
        X_spikes = data_handler.pixels_to_spikes(
            X_slice)

        with open(f'./data/spikes/{subtype}/X_spikes_{i}.pkl', 'wb') as f:
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
    P.K_o = 64
    # P.hertz_o = P.hertz_o / 100

    data_handler = DataHandler(P)

    with open(f'./patient_som_data/{subtype}/SOM_data.npy', 'rb') as f:
        X = np.load(f)

    with open(f'./patient_som_data/{subtype}/target.npy', 'rb') as f:
        targets = np.load(f)

    with open(f'./patient_som_data/{subtype}/original.npy', 'rb') as f:
        original = np.load(f)

    # PREP DATA
    # Perform K-folds cross-validation:
    # First, shuffle all the arrays while preserving order
    indices = np.arange(X.shape[0])
    np.random.seed(13)
    np.random.shuffle(indices)
    X = X[indices]
    targets = targets[indices]
    original = original[indices]

    # Store the data in slices of size <self.s_slice>
    store_spikes(X, data_handler, clear=True)

    # Then split sequentially for each fold
    K = 4
    slice = len(targets)//K
    final_df = pd.DataFrame()
    n_img_k = []
    n_labels_k = []
    test_labels_k = []
    clf_predictions_k = []
    for k in range(K):
        # Initialise network
        network = NetworkHierarchical(P)

        # This range defines the test set.
        lower = slice*k
        upper = min(slice*(k+1), len(targets))

        idxs = np.arange(X.shape[0])
        test_idxs = idxs[lower:upper]
        train_idxs = np.concatenate([idxs[:lower], idxs[upper:]])

        train_slices = np.unique(train_idxs//S*S)
        test_slices = np.unique(test_idxs//S*S)

        X_test = X[lower:upper]
        original_test = original[lower:upper]
        labels_test = targets[lower:upper]

        # Remove synthetic / SMOTE examples from the test set.
        X_test = X_test[original_test]
        labels_test = labels_test[original_test].flatten()

        X_train = np.concatenate((X[:lower], X[upper:]))
        labels_train = np.concatenate((targets[:lower], targets[upper:]))

        print(
            f"Starting fold {k}. Test set bounds are {lower} to {upper} from a total length of {len(targets)}.")

        # TRAIN NETWORK
        neuron_image_counts = np.zeros((X_train.shape[0], network.K_o))

        clf = LogisticRegression()
        for epoch in range(2):
            index_train = 0
            # The number of times each neuron spiked for each label
            neuron_label_counts = np.zeros((network.K_o, 2), dtype=np.uint32)
            # First neuron to spike for each label
            neuron_first_spike_counts = np.zeros(
                (network.K_o, 2), dtype=np.uint32)

            for ith_slice in tqdm(train_slices):
                # Retrieve slice <ith_slice> of the spike data
                with open(f'./data/spikes/{subtype}/X_spikes_{ith_slice}.pkl', 'rb') as f:
                    spike_data = pickle.load(f)

                for index_im, spike_times in enumerate(spike_data):
                    # Determine the complete dataset index (rather than that of the slice)
                    index_im_all = ith_slice+index_im

                    # Ensure index is NOT from the test set
                    if index_im_all >= lower and index_im_all < upper:
                        continue

                    # Convert the flat <spike_times> to a tiled array
                    spike_times = network.tile_input(
                        spike_times, network.s_os, network.w, network.h)

                    first_spike_index = []
                    # Propogate through the network according to the current timestep and given spike times
                    for t in range(data_handler.ms):
                        network.propagate(
                            spike_times, t, learn_h=True, learn_o=True)

                        # Check for the first neuron to spike
                        if len(first_spike_index) == 0:
                            first_spike_index = np.where(
                                network.n_spikes_since_reset_o == 1)

                    network.reset()  # Reset the network between images

                    label = int(targets[index_im_all])
                    neuron_label_counts[:,
                                        label] += network.n_spikes_since_reset_o

                    # train an online logistic regression model for each neuron
                    clf.partial_fit(network.n_spikes_since_reset_o.reshape(1, -1), [
                                    label])

                    neuron_first_spike_counts[first_spike_index, label] += 1

                    neuron_image_counts[index_train] = network.n_spikes_since_reset_o
                    index_train += 1

                    # Reset spike counters
                    network.n_spikes_since_reset_h = np.zeros(
                        (network.s_os, network.s_os, network.K_h), dtype=np.uint16)
                    network.n_spikes_since_reset_o = np.zeros(
                        network.K_o, dtype=np.uint16)

        # TESTING
        index_im_all = 0
        index_test = 0
        # time_start = time.time()

        # For each image the number of times each neuron spiked
        neuron_image_counts = np.zeros((X_test.shape[0], network.K_o))

        predictions = []
        # Iterate over the spike data in slices of size <data_handler.s_slice>
        for ith_slice in tqdm(test_slices):
            # Retrieve slice <ith_slice> of the spike data
            with open(f'./data/spikes/{subtype}/X_spikes_{ith_slice}.pkl', 'rb') as f:
                spike_data = pickle.load(f)

            for index_im, spike_times in enumerate(spike_data):
                # Determine the complete dataset index (rather than that of the slice)
                index_im_all = ith_slice+index_im

                # Ensure sample is in the test set and not synthetic.
                if not (index_im_all >= lower and index_im_all < upper) or not original[index_im_all]:
                    continue

                # Convert the flat <spike_times> to a tiled array
                spike_times = network.tile_input(
                    spike_times, network.s_os, network.w, network.h)

                # Propogate through the network according to the current timestep and given spike times
                for t in range(data_handler.ms):
                    network.propagate(
                        spike_times, t, learn_h=False, learn_o=False)
                network.reset()  # Reset the network between images

                neuron_image_counts[index_test] = network.n_spikes_since_reset_o
                index_test += 1

                # pred = clf.predict([network.n_spikes_since_reset_o])
                # predictions.append(pred)

                # Print, plot, and save data according to the given parameters
                # time_start = network.print_plot_save(
                #     data_handler, X_test, labels_test, index_im_all, X_test.shape[0], P, time_start)

                # Reset spike counters
                network.n_spikes_since_reset_h = np.zeros(
                    (network.s_os, network.s_os, network.K_h), dtype=np.uint16)
                network.n_spikes_since_reset_o = np.zeros(
                    network.K_o, dtype=np.uint16)

        results_df = network.evaluate_results(data_handler, neuron_label_counts,
                                              neuron_image_counts, labels_test)

        results_df["Train_0_count"] = np.sum(labels_train == 0)
        results_df["Train_1_count"] = np.sum(labels_train == 1)

        results_df["LogRegWeights"] = [clf.weights]
        results_df["LogRegBias"] = clf.bias

        # with open(f"./log_coefs/fold_{k}", 'wb') as file:
        #     pickle.dump([clf.bias, clf.weights], file)

        if len(final_df) == 0:
            final_df = results_df
        else:
            final_df = pd.concat([final_df, results_df])

        print(f"Results for fold {k}:")
        print(final_df)

        n_labels_k.append(neuron_label_counts)
        n_img_k.append(neuron_image_counts)
        test_labels_k.append(labels_test)
        # clf_predictions_k.append(predictions)

    final_df.to_csv("results_df.csv")
    np.save("neuron_label_counts.npy",
            np.array(n_labels_k, dtype=object))
    np.save("neuron_image_counts.npy",
            np.array(n_img_k, dtype=object))
    np.save("smote_labels.npy",
            np.array(test_labels_k, dtype=object))
    # np.save("./clf_predictions.npy", clf_predictions_k)


if __name__ == "__main__":
    run_hierarchical()

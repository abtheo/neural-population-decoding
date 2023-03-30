from net_hierarchical import NetworkHierarchical
from util import Parameters, SHAPE_MNIST_TRAIN
from data_handler import DataHandler
import pickle
import numpy as np
from tqdm import tqdm
import time


def run_hierarchical(train=True):
    argv = []

    # Set the parameters
    P = Parameters()
    P.set_hierarchical(argv)

    # Initialize the hierarchical network
    network = NetworkHierarchical(P)

    # Initialize the DataHandler
    data_handler = DataHandler(P)

    # Create and store new spike data if it is not in storage already
    # data_handler.store_mnist_as_spikes(
    #     train=train, tag_mnist=network.tag_mnist)

    with open('./data/mnist.pkl', 'rb') as f:
        X_train, labels_train, X_test, labels_test = pickle.load(f)

    X = X_train
    # PREP DATA

    # X_spikes_final = []
    # # Store the data in slices of size <self.s_slice>
    # for i in tqdm(range(0, X.shape[0], data_handler.s_slice)):
    #     slice_start = i
    #     slice_end = slice_start + data_handler.s_slice
    #     # takes the first S images from MNIST
    #     X_slice = X[slice_start:slice_end]  # shape (100,28,28)
    #     X_spikes = data_handler.pixels_to_spikes(
    #         X_slice)  # shape (100,28,28,2,150)

    # For some reason there's an extra dimension here to denote a boolean...
    # but it's always False in the second slice?

    # print(X_spikes.shape, X_spikes[0, 0, 0, 0], any(X_spikes[0, 0, 0, 1]))

    # then it's split into 150 values, because it's 150hz frequency I guess

    # X_spikes_final.append(X_spikes)

    # with open(f'./data/spikes/X_spikes_{i}.pkl', 'wb') as f:
    #     pickle.dump(X_spikes, f)

    # print(np.array(X_spikes_final).shape)

    # X_spikes_final = np.array(X_spikes_final)

    # X_spikes_final = np.zeros(
    #     (600, 100, 28, 28, 2, 150), dtype=np.bool8)  # ???????????????

    # TRAIN NETWORK
    # Iterate over the spike data in slices of size <data_handler.s_slice>
    time_start = time.time()
    for ith_slice in tqdm(range(0, SHAPE_MNIST_TRAIN[0]//data_handler.s_slice)):

        # Retrieve slice <ith_slice> of the spike data
        with open(f'./data/spikes/X_spikes_{ith_slice*100}.pkl', 'rb') as f:
            spike_data = pickle.load(f)

        for index_im, spike_times in enumerate(spike_data):
            # Convert the flat <spike_times> to a tiled array
            spike_times = network.tile_input(
                spike_times, network.s_os, network.s_is)

            # (4, 4, 7, 7, 2, 150)
            # (neurons, neurons, width, height, channels, time)
            print(f"Spike shape: {spike_times.shape}")

            # Determine the complete dataset index (rather than that of the slice)
            index_im_all = ith_slice*data_handler.s_slice+index_im
            network.index_im_all = index_im_all  # $$$
            network.print_interval = P.print_interval  # $$$

            # Propogate through the network according to the current timestep and given spike times
            for t in range(data_handler.ms):
                network.propagate(spike_times, t, learn_h=True, learn_o=True)
            network.reset()  # Reset the network between images

            # Print, plot, and save data according to the given parameters
            # time_start = network.print_plot_save(
            #     data_handler, X, labels_train, index_im_all, SHAPE_MNIST_TRAIN[0], P, time_start)

            # Reset spike counters
            network.n_spikes_since_reset_h = np.zeros(
                (network.s_os, network.s_os, network.K_h), dtype=np.uint16)
            network.n_spikes_since_reset_o = np.zeros(
                network.K_o, dtype=np.uint16)

    # Okay, so the network is now trained
    # Let's test it
    # The number of times each neuron spiked for each label
    neuron_label_counts = np.zeros((self.K_o, 10), dtype=np.uint32)

    # For each image the number of times each neuron spiked
    neuron_image_counts = np.zeros((util.SHAPE_MNIST_TEST[0], self.K_o))

    index_im_all = 0
    time_start = time.time()
    # Iterate over the spike data in slices of size <data_handler.s_slice>
    for ith_slice in range(0, util.SHAPE_MNIST_TEST[0]//data_handler.s_slice):

        # Retrieve slice <ith_slice> of the spike data
        spike_data = data_handler.get_mnist_spikes(
            ith_slice, tag_mnist=self.tag_mnist, train=False)
        for index_im, spike_times in enumerate(spike_data):

            # Convert the flat <spike_times> to a tiled array
            spike_times = self.tile_input(
                spike_times, self.s_os, self.s_is)

            # Determine the complete dataset index (rather than that of the slice)
            index_im_all = ith_slice*data_handler.s_slice+index_im
            self.index_im_all = index_im_all  # $$$
            self.print_interval = P.print_interval  # $$$

            # Retrieve the label of the current image
            label = labels[index_im_all]

            # Propogate through the network according to the current timestep and given spike times
            for t in range(data_handler.ms):
                self.propagate(
                    spike_times, t, learn_h=False, learn_o=False)
            self.reset()  # Reset the network between images

            # Keep track of the results
            neuron_label_counts[:, label] += self.n_spikes_since_reset_o
            neuron_image_counts[index_im_all] = self.n_spikes_since_reset_o

            # Print, plot, and save data according to the given parameters
            time_start = self.print_plot_save(
                data_handler, X, labels, index_im_all, util.SHAPE_MNIST_TEST[0], P, time_start)

            # Evaluate results after every <evaluation_interval> images
            if index_im_all > 0 and index_im_all % P.evaluation_interval == 0:
                print("\nEvaluating results after {} images:".format(
                    index_im_all))
                self.evaluate_results(data_handler, neuron_label_counts[:, :index_im_all+1],
                                      neuron_image_counts[:index_im_all+1], labels[:index_im_all+1])

            # Reset spike counters
            self.n_spikes_since_reset_h = np.zeros(
                (self.s_os, self.s_os, self.K_h), dtype=np.uint16)
            self.n_spikes_since_reset_o = np.zeros(
                self.K_o, dtype=np.uint16)


if __name__ == "__main__":
    run_hierarchical()

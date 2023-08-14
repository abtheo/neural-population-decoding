__author__ = "Otto van der Himst"
__version__ = "1.0"
__email__ = "otto.vanderhimst@ru.nl"

import numpy as np
import matplotlib.pyplot as plt
from util import dump, load, get_sinewave, get_firing_probability


class DataHandler():

    def __init__(self, P):

        self.pd_data = P.pd_data                   # Path to data directory
        # Path template to MNIST spikes directory
        self.pdt_mnist_spikes_train = P.pdt_mnist_spikes_train
        # Path template to MNIST spikes directory
        self.pdt_mnist_spikes_test = P.pdt_mnist_spikes_test
        # Filename template of MNIST spike slice
        self.nft_mnist_spikes = P.nft_mnist_spikes

        self.hertz_i = P.hertz_i     # The hertz of each spike train
        self.ms = P.ms           # The duration in miliseconds of each spike train
        self.s_slice = P.s_slice  # The size of each slice of data

        self.always_fire = P.always_fire

        self.path_affix = "" if not self.always_fire else "_always-fire"

        self.p_fire = self.hertz_i/1000 if not self.always_fire else 1

        self.rng = np.random.default_rng(P.seed)

    def binarize_pixels(self, pixels):
        """ Convert non-binary (grayscale) pixels to binary (black-and-white) pixels. """
        pixels[pixels > 0] = 1
        return pixels

    def pixels_to_spikes(self, pixel_data):

        # Extract data dimensions
        dim_batch, dim_row, dim_col = pixel_data.shape

        # Generate poisson spike trains
        spike_data = self.rng.choice([0, 1], (dim_batch, dim_row, dim_col, 2, self.ms), p=[
                                     1-self.p_fire, self.p_fire])

        # Let either the black or white neuron be active, according to the input
        pixel_data = self.binarize_pixels(pixel_data).flatten()
        # Reshapes the data we just generated ourselves... mkay...
        spike_data = spike_data.reshape(-1, spike_data.shape[-1])
        pixel_indices = np.arange(pixel_data.size) * 2 - pixel_data + 1

        spike_data[pixel_indices] *= 0
        spike_data = spike_data.reshape(
            (dim_batch, dim_row, dim_col, 2, self.ms))

        spike_data = spike_data.astype(bool)  # Convert to a boolean array
        return spike_data

    def spikes_to_pixels(self, spike_data):
        """ Turn spike data into pixels, where a pixel becomes more white proportional to the number of spikes. """

        # Sum over the number of spikes, normalize, then convert to 8-bit pixels
        pixel_data = spike_data.sum(-1)
        max_val = np.max(pixel_data)
        pixel_data = (pixel_data[..., 1] - pixel_data[..., 0]) + max_val
        pixel_data = pixel_data/(max_val*2) * 255

        return pixel_data.astype(np.uint8)

    def inspect_spike_data(self, X,  spike_data, ith_slice, s_slice, n_inspections=3, train=True):
        """ Inspect spike data by converting it to pixels and then plotting it.  """
        # Select a few random samples to inspect
        indices = self.rng.integers(0, spike_data.shape[0], n_inspections)
        spike_data = spike_data[indices]

        # Obtain pixel version of the spike data samples
        pixel_data = self.spikes_to_pixels(spike_data)

        # Compare the original mnist images and the pixel representations of the spike data
        fig, axes = plt.subplots(nrows=n_inspections, ncols=2, figsize=(5, 5))
        for ith_row in range(n_inspections):
            index = indices[ith_row]
            axes[ith_row, 0].imshow(
                X[ith_slice*s_slice + index], cmap=plt.get_cmap('gray'))
            axes[ith_row, 1].imshow(
                pixel_data[ith_row], cmap=plt.get_cmap('gray'))
        fig.tight_layout()
        plt.show()
        # print("The images concern the numbers {}.".format(
        #     labels[indices+ith_slice*s_slice]))

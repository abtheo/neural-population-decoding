import numpy as np

num_imgs = 230
neurons = 99
classes = 2

# Class-imbalanced label generation
num_ones = int(num_imgs * 0.1)
labels = np.zeros(num_imgs, dtype=np.int64)
indices = np.random.choice(num_imgs, num_ones, replace=False)
labels[indices] = 1

# Generate an even distribution of integers
# numbers = np.arange(classes)
# distribution = np.repeat(numbers, num_imgs // classes + 1)[:num_imgs]
# np.random.shuffle(distribution)
# labels = distribution

# labels = np.random.randint(2, size=(num_imgs))

# Number of neurons should be a hyperparameter.
# What if we set k_0 from 99 to just 2?
neuron_label_counts = np.zeros(
    shape=(neurons, classes))  # size = (neurons, labels)
saved_predictions = np.zeros(shape=(num_imgs, neurons))
for i in range(num_imgs):
    # Each neuron has a random chance of firing
    nn = [np.random.choice(a=range(classes), size=(1)) for j in range(neurons)]

    network_out_spikes = np.array(nn).flatten()

    saved_predictions[i] = network_out_spikes
    neuron_label_counts[:, labels[i]] += network_out_spikes


# Determine for each neuron to which label it most strongly responds
neuron_label_dict = dict()
for k in range(neuron_label_counts.shape[0]):
    neuron_label_dict[k] = np.argmax(neuron_label_counts[k])

cnt_wrong = cnt_correct = 0
for index_im, network_predictions in enumerate(saved_predictions):
    # Count for each label how often its corresponding neurons spike
    label_counts = np.zeros(classes, dtype=np.uint16)
    for k, neuron_prediction in enumerate(network_predictions):
        neuron_label = neuron_label_dict[k]
        label_counts[neuron_label] += neuron_prediction

    # The label for which the most neurons spiked is considered the prediction
    prediction = np.argmax(label_counts)

    real_label = labels[index_im]

    if prediction == real_label:
        cnt_correct += 1
    else:
        cnt_wrong += 1


accuracy = 100*cnt_correct/(cnt_correct+cnt_wrong)
print(accuracy)

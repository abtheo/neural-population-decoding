import numpy as np
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import recall_score, f1_score
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import factorial
from scipy.stats import poisson

# def prune(fold):
#     """
#         Only consider neurons with a high class separation for predictions
#     """


def sample_average(fold):
    """
        Average spikes over num samples in train set
    """
    # Determine for each neuron to which label it most strongly responds
    neuron_label_dict = dict()
    for l in range(len(fold['neuron_label_counts'])):
        avg_label_counts = fold['neuron_label_counts'][l]
        avg_label_counts[0] = avg_label_counts[0] / fold["Train_0_count"]
        avg_label_counts[1] = avg_label_counts[1] / fold["Train_1_count"]
        neuron_label_dict[l] = np.argmax(avg_label_counts)

    sample_avg_predictions = []
    for spike_counts in fold['neuron_image_counts']:
        # Count for each label how often its corresponding neurons spike
        label_counts = np.zeros(2)
        for k, spike_count in enumerate(spike_counts):
            neuron_label = neuron_label_dict[k]
            label_counts[neuron_label] += spike_count

        # The label for which the most neurons spiked is considered the prediction
        prediction = np.argmax(label_counts)
        sample_avg_predictions.append(prediction)
    return sample_avg_predictions


def population_vector_decoder(fold):
    """
        Population Vector Decoder:
        Build dictionary of prefered class, then vector sum
    """
    # Determine for each neuron to which label it most strongly responds
    neuron_label_dict = dict()
    for l in range(len(fold['neuron_label_counts'])):
        neuron_label_dict[l] = np.argmax(fold['neuron_label_counts'][l])

    # Vector sum... is what we were doing already.
    predictions = []
    for spike_counts in fold['neuron_image_counts']:

        # Count for each label how often its corresponding neurons spike
        label_counts = np.zeros(2, dtype=np.uint16)
        for k, spike_count in enumerate(spike_counts):
            neuron_label = neuron_label_dict[k]
            label_counts[neuron_label] += spike_count

        # The label for which the most neurons spiked is considered the prediction
        prediction = np.argmax(label_counts)
        predictions.append(prediction)
    return predictions


def max_neuron(fold):
    # Determine for each neuron to which label it most strongly responds
    neuron_label_dict = dict()
    for l in range(len(fold['neuron_label_counts'])):
        neuron_label_dict[l] = np.argmax(fold['neuron_label_counts'][l])

    predictions = []
    for spike_counts in fold['neuron_image_counts']:
        # The neuron which achieved the highest spike count is considered the label
        prediction = neuron_label_dict[np.argmax(spike_counts)]
        predictions.append(prediction)

    return predictions


def average_max_neuron(fold):
    """
        Guo et al:
        The predicted digit of each test example
        is determined by the highest average firing rate. 
        We averages the responses of each output neuron and then 
        choose the neuron with the highest firing rate as the predicted class of the test example.
    """
    # Determine for each neuron to which label it most strongly responds
    neuron_label_dict = dict()
    for l in range(len(fold['neuron_label_counts'])):
        neuron_label_dict[l] = np.argmax(fold['neuron_label_counts'][l])

    predictions = []
    # so this is the network response per image
    # we should subtract the mean per neuron, then argmax
    average_firing_rates = np.mean(fold['neuron_image_counts'], axis=0)
    for spike_counts in fold['neuron_image_counts']:

        # The neuron which achieved the highest spike count is considered the label
        prediction = neuron_label_dict[np.argmax(
            spike_counts - average_firing_rates)]
        predictions.append(prediction)

    return predictions


def average_firing_rate(fold):
    # Determine for each neuron to which label it most strongly responds
    neuron_label_dict = dict()
    for l in range(len(fold['neuron_label_counts'])):
        neuron_label_dict[l] = np.argmax(fold['neuron_label_counts'][l])

    predictions = []
    average_firing_rates = np.mean(fold['neuron_image_counts'], axis=0)
    for spike_counts in fold['neuron_image_counts']:

        # Count for each label how often its corresponding neurons spike
        label_counts = np.zeros(2)
        for k, spike_count in enumerate(spike_counts):
            neuron_label = neuron_label_dict[k]
            label_counts[neuron_label] += (spike_count -
                                           average_firing_rates[k])

        # The label for which the most neurons spiked is considered the prediction
        prediction = np.argmax(label_counts)
        predictions.append(prediction)

    return predictions


def average_neuron_classes(fold):
    # Determine for each neuron to which label it most strongly responds
    neuron_label_dict = dict()
    for l in range(len(fold['neuron_label_counts'])):
        neuron_label_dict[l] = np.argmax(fold['neuron_label_counts'][l])

    predictions = []
    for spike_counts in fold['neuron_image_counts']:

        # Count for each label how often its corresponding neurons spike
        label_counts = np.zeros(2)
        for k, spike_count in enumerate(spike_counts):
            neuron_label = neuron_label_dict[k]
            label_counts[neuron_label] += spike_count

        # Average the spike counts over each neuron class
            # divide by number of 0 neurons
        label_counts[0] = label_counts[0] / \
            (len(neuron_label_dict) - sum(neuron_label_dict.values()))
        # divide by number of 1 neurons
        label_counts[1] = label_counts[1] / sum(neuron_label_dict.values())

        # The label for which the most neurons spiked is considered the prediction
        prediction = np.argmax(label_counts)
        predictions.append(prediction)

    return predictions


# Likelihood function (negative log-likelihood) for Poisson distribution
def negative_log_likelihood_poisson(params, X, Y):
    lambda_param = params
    log_likelihood = -np.sum(np.log(factorial(Y)) -
                             lambda_param + Y * np.log(lambda_param))
    return -log_likelihood


def maximum_likelihood_estimator(fold):
    # Initial parameter guess
    initial_lambda = 1.0

    # Separate data based on labels
    R_label_0 = fold['Labels'][fold['Labels'] == 0]
    R_label_1 = fold['Labels'][fold['Labels'] == 1]

    # Calculate Poisson parameter (mean) for each subarray
    lambda_0 = np.mean(R_label_0)
    lambda_1 = np.mean(R_label_1)

    # Calculate MLE for conditional probabilities using Poisson PMF
    conditional_probs_0 = [poisson.pmf(
        r, lambda_0) for r in fold['neuron_image_counts']]
    conditional_probs_1 = [poisson.pmf(
        r, lambda_1) for r in fold['neuron_image_counts']]

    stack = np.stack((np.sum(conditional_probs_0, axis=1),
                     np.sum(conditional_probs_1, axis=1)))
    predictions = np.argmax(stack, axis=0)
    return predictions


def calc_metrics(ypred, ytrue):
    rai = adjusted_rand_score(ytrue, ypred)
    nmi = normalized_mutual_info_score(ytrue, ypred)
    recall = 0  # recall_score(ytrue, ypred)
    f1 = f1_score(ytrue, ypred)

    return rai, nmi, recall, f1


def fix_string_to_numpy(stringy):
    if stringy[:2] == '[[':
        return np.fromstring(stringy[2:-1], sep=' ')
    if stringy[-2:] == ']]':
        if stringy[:1] == ' ':
            return np.fromstring(stringy[2:-2], sep=' ')
        return np.fromstring(stringy[1:-2], sep=' ')
    return np.fromstring(stringy[2:-1], sep=' ')


"""
        Dataset | Fold | RAI_1 | NMI_1 | RAI_2 | NMI_2
        ----------------------------------------------
        smote0  |  ALL |  0.2  | -0.01 | 0.15  | -0.04
        smote0  |   1  |  0.2  | -0.01 | 0.15  | -0.04
        smote0  |   2  |   r   |   n   |   r   |   n

        plt.plot(x=[0,0,0,0,0,33,33,33,33,66,100], y=RAI_1, cmap=blue)
        plt.add_trace(NMI_1, red)
    """
master_results = pd.DataFrame()
smote_datasets = ["kirc_0", "kirc_33", "kirc_66", "kirc_100"]
for ds in smote_datasets:
    df = pd.read_csv(
        f"./Colab_results/{ds}smote_results_df.csv", index_col=[0])
    df.reset_index(drop=True, inplace=True)

    neuron_label_counts = np.load(
        f"./Colab_results/{ds}smote_neuron_label_counts.npy", allow_pickle=True)

    neuron_image_counts = np.load(
        f"./Colab_results/{ds}smote_neuron_image_counts.npy", allow_pickle=True)
    labels = np.load(
        f"./Colab_results/{ds}smote_labels.npy", allow_pickle=True)

    pop_vector_predictions = []
    neuron_avg_predictions = []
    sample_avg_predictions = []
    firing_avg_predictions = []
    max_neuron_predictions = []
    avg_max_neuron_predictions = []
    mle_predictions = []
    labelZ = []

    for kfold in range(df.shape[0]):
        fold = df.iloc[kfold].copy()
        fold['neuron_image_counts'] = neuron_image_counts[kfold]
        fold['neuron_label_counts'] = neuron_label_counts[kfold]
        fold['Labels'] = labels[kfold]

        # Calculate predictions via each form of population coding
        pop_vec = population_vector_decoder(fold)
        avg_neur = average_neuron_classes(fold)
        samp_avg = sample_average(fold)
        fire_avg = average_firing_rate(fold)
        max_neur = max_neuron(fold)
        avg_max_neur = average_max_neuron(fold)
        mle = maximum_likelihood_estimator(fold)

        # Save predictions for overall dataset evaluation
        pop_vector_predictions.append(pop_vec)
        neuron_avg_predictions.append(avg_neur)
        sample_avg_predictions.append(samp_avg)
        firing_avg_predictions.append(fire_avg)
        max_neuron_predictions.append(max_neur)
        avg_max_neuron_predictions.append(avg_max_neur)
        mle_predictions.append(mle)
        labelZ.append(fold['Labels'])

        # Calculate metrics per fold too
        pop_rai, pop_nmi, _, pop_f1 = calc_metrics(pop_vec, fold['Labels'])
        an_rai, an_nmi, _, an_f1 = calc_metrics(avg_neur, fold['Labels'])
        sm_rai, sm_nmi, _, sm_f1 = calc_metrics(samp_avg, fold['Labels'])
        fa_rai, fa_nmi, _, fa_f1 = calc_metrics(fire_avg, fold['Labels'])
        mn_rai, mn_nmi, _, mn_f1 = calc_metrics(max_neur, fold['Labels'])
        amn_rai, amn_nmi, _, amn_f1 = calc_metrics(
            avg_max_neur, fold['Labels'])
        mle_rai, mle_nmi, _, mle_f1 = calc_metrics(mle, fold['Labels'])

        res_dict = {"Dataset": ds,
                    "Fold": kfold,

                    "Population_Vector_F1":  pop_f1,
                    "Population_Vector_RAI": pop_rai,
                    "Population_Vector_NMI": pop_nmi,

                    "Class_Average_Neuron_F1":  an_f1,
                    "Class_Average_Neuron_RAI": an_rai,
                    "Class_Average_Neuron_NMI": an_nmi,

                    "Sample_Average_F1":  sm_f1,
                    "Sample_Average_RAI": sm_rai,
                    "Sample_Average_NMI": sm_nmi,

                    "Firing_Average_F1":  fa_f1,
                    "Firing_Average_RAI": fa_rai,
                    "Firing_Average_NMI": fa_nmi,

                    "Max_Neuron_F1":  mn_f1,
                    "Max_Neuron_RAI": mn_rai,
                    "Max_Neuron_NMI": mn_nmi,

                    "Average_Max_Neuron_F1":  amn_f1,
                    "Average_Max_Neuron_RAI": amn_rai,
                    "Average_Max_Neuron_NMI": amn_nmi,

                    "Max_Likelihood_Estimator_F1":  mle_f1,
                    "Max_Likelihood_Estimator_RAI": mle_rai,
                    "Max_Likelihood_Estimator_NMI": mle_nmi
                    }
        master_results = master_results.append(res_dict, ignore_index=True)

    # Concatenate K-fold predictions into overall dataset
    pop_vector_predictions = np.concatenate(pop_vector_predictions)
    sample_avg_predictions = np.concatenate(sample_avg_predictions)
    firing_avg_predictions = np.concatenate(firing_avg_predictions)
    neuron_avg_predictions = np.concatenate(neuron_avg_predictions)
    max_neuron_predictions = np.concatenate(max_neuron_predictions)
    avg_max_neuron_predictions = np.concatenate(avg_max_neuron_predictions)
    mle_predictions = np.concatenate(mle_predictions)
    labelZ = np.concatenate(labelZ)

    # Evaluate overall predictions
    pop_rai, pop_nmi, _, pop_f1 = calc_metrics(
        pop_vector_predictions, labelZ)
    an_rai, an_nmi, _, an_f1 = calc_metrics(
        neuron_avg_predictions, labelZ)
    sm_rai, sm_nmi, _, sm_f1 = calc_metrics(
        sample_avg_predictions, labelZ)
    fa_rai, fa_nmi, _, fa_f1 = calc_metrics(
        firing_avg_predictions, labelZ)
    mn_rai, mn_nmi, _, mn_f1 = calc_metrics(
        max_neuron_predictions, labelZ)
    amn_rai, amn_nmi, _, amn_f1 = calc_metrics(
        avg_max_neuron_predictions, labelZ)
    mle_rai, mle_nmi, _, mle_f1 = calc_metrics(mle_predictions, labelZ)

    # Add to results
    res_dict = {"Dataset": ds,
                "Fold": "ALL",

                "Population_Vector_F1":  pop_f1,
                "Population_Vector_RAI": pop_rai,
                "Population_Vector_NMI": pop_nmi,

                "Class_Average_Neuron_F1":  an_f1,
                "Class_Average_Neuron_RAI": an_rai,
                "Class_Average_Neuron_NMI": an_nmi,

                "Sample_Average_F1":  sm_f1,
                "Sample_Average_RAI": sm_rai,
                "Sample_Average_NMI": sm_nmi,

                "Firing_Average_F1":  fa_f1,
                "Firing_Average_RAI": fa_rai,
                "Firing_Average_NMI": fa_nmi,

                "Max_Neuron_F1":  mn_f1,
                "Max_Neuron_RAI": mn_rai,
                "Max_Neuron_NMI": mn_nmi,

                "Average_Max_Neuron_F1":  amn_f1,
                "Average_Max_Neuron_RAI": amn_rai,
                "Average_Max_Neuron_NMI": amn_nmi,

                "Max_Likelihood_Estimator_F1":  mle_f1,
                "Max_Likelihood_Estimator_RAI": mle_rai,
                "Max_Likelihood_Estimator_NMI": mle_nmi
                }
    master_results = master_results.append(res_dict, ignore_index=True)

master_results["X"] = np.concatenate(([0]*5, [33]*5, [66]*5, [100]*5))
print(master_results)

master_results.plot.scatter(
    'X', 'Population_Vector_F1', title='Population Vector')
plt.show()


master_results.plot.scatter(
    'X', 'Max_Likelihood_Estimator_F1', title='Max Likelihood Estimator')
plt.show()


master_results.plot.scatter(
    'X', 'Class_Average_Neuron_F1', title='Class Average Neuron')
plt.show()


master_results.plot.scatter('X', 'Sample_Average_F1', title='Sample Average')
plt.show()

master_results.plot.scatter('X', 'Firing_Average_F1', title='Firing Average')
plt.show()


master_results.plot.scatter('X', 'Max_Neuron_F1', title='Max Neuron')
plt.show()

master_results.plot.scatter(
    'X', 'Average_Max_Neuron_F1', title='Average Max Neuron')
plt.show()

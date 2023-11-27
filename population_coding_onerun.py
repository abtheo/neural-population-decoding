import numpy as np
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
# from sklearn.linear_model import LogisticRegression
from logistic_regression import LogisticRegression
from sklearn.metrics import f1_score
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
    # average firing rate per neuron
    average_firing_rates = np.mean(fold['neuron_image_counts'], axis=0)
    for spike_counts in fold['neuron_image_counts']:

        # Count for each label how often its corresponding neurons spike
        label_counts = np.zeros(2)
        for k, spike_count in enumerate(spike_counts):
            neuron_label = neuron_label_dict[k]
            # Subtract the average firing rate of that neuron
            label_counts[neuron_label] += (spike_count -
                                           average_firing_rates[k])

        # The label for which the most neurons spiked is considered the prediction
        prediction = np.argmax(label_counts)
        predictions.append(prediction)

    return predictions


def average_neuron_classes(fold):
    """
        Guo et al:
        The predicted digit of each test example is determined by the highest average firing rate.
        We averages the responses of each output neuron and then
        choose the -neuron- CLASS with the highest firing rate as the predicted class of the test example.

        Diehl & Cook:
        assign a class to each neuron, based on its highest response to the ten classes of digits over one presentation of the training set.
        The predicted digit is determined by averaging the responses of each neuron per class and then choosing the class with the highest average firing rate.
    """
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

        # Average the spike counts over each neuron class:
            # divide by number of 0 neurons
        label_counts[0] = label_counts[0] / \
            (len(neuron_label_dict) - sum(neuron_label_dict.values()))
        # divide by number of 1 neurons
        label_counts[1] = label_counts[1] / sum(neuron_label_dict.values())

        # The label for which the most neurons spiked is considered the prediction
        prediction = np.argmax(label_counts)
        predictions.append(prediction)

    return predictions


def neuron_counts(fold):
    neuron_label_dict = dict()
    for l in range(len(fold['neuron_label_counts'])):
        neuron_label_dict[l] = np.argmax(fold['neuron_label_counts'][l])

    num_z = len(neuron_label_dict) - sum(neuron_label_dict.values())
    num_o = sum(neuron_label_dict.values())
    print("ZERO NEURONS: ", num_z)
    print("ONE NEURONS: ", num_o)
    print("RATIO OF MINORTY:MAJORITY = ", min(num_z, num_o)/max(num_z, num_o))


def logistic_regression(fold):

    # Create a Logistic Regression model
    model = LogisticRegression()
    model.bias = fold["LogRegBias"]
    model.weights = fix_string_to_numpy(fold["LogRegWeights"])
    # model.fit(fold['neuron_image_counts'], fold['Labels'])
    preds = model.predict(fold['neuron_image_counts'])
    return preds


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


def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)


def plot_scatter_bar(df, label, title="", tidy_label=False):
    # Create a figure and axis
    fig, ax1 = plt.subplots()

    scatter_pts = df[label].loc[df['Fold'] != 'ALL']
    scatter_pts = scatter_pts.reset_index(drop=True)

    # X = np.concatenate(([0.05]*4, [0.33]*4, [0.66]*4, [1]*4))
    X = [1.0]

    ax1.bar(X, df[label].loc[df['Fold'] == 'ALL'].to_numpy(),
            color='red',
            alpha=0.5,
            width=0.1)

    # Set the labels and title for ax1
    ax1.set_xlabel('α ratio')
    ax1.set_ylabel(tidy_label if tidy_label else label)

    # Create a second y-axis for the bar chart
    ax2 = ax1.twinx()

    # Loop through the data and create scatter plots for X vs. Y
    i = 0
    for x in X:
        ax2.scatter([x] * 4, scatter_pts.iloc[i:i+4].to_numpy(),
                    alpha=0.5, color='blue')
        i += 4

    # Set the labels and title for ax2
    # ax2.set_ylabel(label, color='red')
    # ax2.tick_params(axis='y', labelcolor='red')
    ticks = np.linspace(0, round(max(scatter_pts), 2), 11)
    ax1.set_yticks(ticks)
    # ax2.set_yticks(ticks)

    align_yaxis(ax1, 0, ax2, 0)
    ax2.get_yaxis().set_visible(False)

    plt.title(title)
    plt.show()


subtype = "kirc"
master_results = pd.DataFrame()

df = pd.read_csv(
    f"oneres/logreg_results_df.csv", index_col=[0])
df.reset_index(drop=True, inplace=True)

neuron_label_counts = np.load(
    f"oneres/logreg_neuron_label_counts.npy", allow_pickle=True)

neuron_image_counts = np.load(
    f"oneres/logreg_neuron_image_counts.npy", allow_pickle=True)
labels = np.load(
    f"oneres/logreg_labels.npy", allow_pickle=True)

pop_vector_predictions = []
neuron_avg_predictions = []
sample_avg_predictions = []
firing_avg_predictions = []
max_neuron_predictions = []
avg_max_neuron_predictions = []
mle_predictions = []
logistic_predictions = []
labelZ = []

for kfold in range(df.shape[0]):
    fold = df.iloc[kfold].copy()
    fold['neuron_image_counts'] = neuron_image_counts[kfold]
    fold['neuron_label_counts'] = neuron_label_counts[kfold]
    fold['Labels'] = labels[kfold]

    # neuron_counts(fold)
    # Calculate predictions via each form of population coding
    pop_vec = population_vector_decoder(fold)
    avg_neur = average_neuron_classes(fold)
    samp_avg = sample_average(fold)
    fire_avg = average_firing_rate(fold)
    max_neur = max_neuron(fold)
    avg_max_neur = average_max_neuron(fold)
    mle = maximum_likelihood_estimator(fold)
    logistic_pred = logistic_regression(fold)

    # Save predictions for overall dataset evaluation
    pop_vector_predictions.append(pop_vec)
    neuron_avg_predictions.append(avg_neur)
    sample_avg_predictions.append(samp_avg)
    firing_avg_predictions.append(fire_avg)
    max_neuron_predictions.append(max_neur)
    avg_max_neuron_predictions.append(avg_max_neur)
    mle_predictions.append(mle)
    logistic_predictions.append(logistic_pred)
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
    logistic_rai, logistic_nmi, _, logistic_f1 = calc_metrics(
        logistic_pred, fold['Labels'])

    res_dict = {"Dataset": 1,
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
                "Max_Likelihood_Estimator_NMI": mle_nmi,

                "Logistic_Regression_F1":  logistic_f1,
                "Logistic_Regression_RAI": logistic_rai,
                "Logistic_Regression_NMI": logistic_nmi
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
logistic_predictions = np.concatenate(logistic_predictions)
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
logistic_rai, logistic_nmi, _, logistic_f1 = calc_metrics(
    logistic_predictions, labelZ)

# Add to results
res_dict = {"Dataset": 1,
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
            "Max_Likelihood_Estimator_NMI": mle_nmi,

            "Logistic_Regression_F1":  logistic_f1,
            "Logistic_Regression_RAI": logistic_rai,
            "Logistic_Regression_NMI": logistic_nmi

            }
master_results = master_results.append(res_dict, ignore_index=True)

# np.concatenate(([0.05]*5, [33]*5, [66]*5, [100]*5))
master_results["X"] = [0.05]*5
# print(master_results)
# print(master_results[["Logistic_Regression_F1",
#       "Logistic_Regression_RAI", "Logistic_Regression_NMI"]])

plot_scatter_bar(master_results, 'Logistic_Regression_F1',
                 f'Logistic_Regression_F1 Results for {subtype.upper()}', 'F1 Score')

plot_scatter_bar(master_results, 'Population_Vector_F1',
                 f'Population Vector Results for {subtype.upper()}', 'F1 Score')

plot_scatter_bar(master_results, 'Class_Average_Neuron_F1',
                 f'Class_Average_Neuron_F1 Results for {subtype.upper()}', 'F1 Score')

plot_scatter_bar(master_results, 'Firing_Average_F1',
                 f'Firing_Average_F1 Results for {subtype.upper()}', 'F1 Score')

# plot_scatter_bar(master_results, 'Population_Vector_F1',
#                  f'Population Vector Results for {subtype.upper()}', 'F1 Score')


df['neuron_image_counts']  # folds, X_test, neurons
df['neuron_label_counts']  # folds, neurons, 2

# we have to do this in the testing loop,
# can't reverse engineer the sums in the label counts
for i in range(len(fold)):
    fold = df.iloc[kfold].copy()
    # pour everything into a bucket
    fold['neuron_image_counts'] = neuron_image_counts[kfold]
    fold['neuron_label_counts'] = neuron_label_counts[kfold]
    fold['Labels'] = labels[kfold]

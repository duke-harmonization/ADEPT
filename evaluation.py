import numpy as np
import torch

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored
from sksurv.metrics import integrated_brier_score as ibs
from sksurv.metrics import brier_score as bs
from sksurv.util import Surv

from dataclasses import InitVar, dataclass, field
from scipy.stats import chi2
from sklearn.linear_model import LinearRegression

from scipy.stats import chi2

from dataclasses import InitVar, dataclass, field

from sklearn import metrics


def calculate_auc(t, s, pred, cutpoint_list, time):
    cutpoint_bool = cutpoint_list >= time

    cutpoint_ind = np.where(cutpoint_bool == 1)[0][0]    
    p = [sum(p_i[0:(cutpoint_ind + 1)]) for p_i in pred]    

    in_scope = np.logical_or(
                np.logical_and(t < time, s),
                (t > time) )
    
    
    y = np.logical_and(t < time, s)
    
    p = np.array(p)[in_scope]
    y = y[in_scope]

    fpr, tpr, thresholds = metrics.roc_curve(y, p, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    
    return(auc)



def discrete_ci(pred, cutpoints, t_test_in, s_test_in, t_train_in):
    s_test = torch.from_numpy(s_test_in).float()
    t_test = torch.from_numpy(t_test_in).float()
    t_train = torch.from_numpy(t_train_in).float()
    
    # cdfs
    t_pred_cdf = np.cumsum(pred, axis=1) 

    bucket_boundaries = [0] + sorted([i.item() for i in cutpoints]) + [1]
    # rescale
    bucket_boundaries = [boundary_i * (max(t_train) - min(t_train)) + min(t_train) 
                             for boundary_i in bucket_boundaries]
    
    N = len(pred)
    n_buckets = len(bucket_boundaries) - 1

    # one hot vector for where the 1 is in the most likely bucket
    t_true_idx = np.zeros((N, n_buckets),dtype=int)
    for ii in range(N):
        for jj in range(n_buckets):
            if t_test[ii] < bucket_boundaries[jj+1]:
                t_true_idx[ii][jj] = 1
                break
    
    t_true_idx = np.argmax(t_true_idx, axis=1)
    concordant = 0
    total = 0


    idx = np.arange(N)

    
    
    for i in range(N):

        if s_test[i] == 0:
            continue

        # time bucket of observation for i, then for all but i
        tti_idx = t_true_idx[i]
        
        tt_idx = t_true_idx[idx != i]

        # calculate predicted risk for i at the time of their event
        tpi = t_pred_cdf[i, tti_idx]

        # predicted risk at that time for all but i
        tp = t_pred_cdf[idx != i, tti_idx]
        
        total += np.sum(tti_idx < tt_idx) # observed in i first

        concordant += np.sum((tti_idx < tt_idx) * (tpi > tp)) # and i predicted as higher risk

    return (concordant / total).item()



def brier_score(s_train, t_train, s_test, t_test, X_test, model, bin_end_times, integrated):
    

    X_test_conv = X_test[t_test < max(t_train), :]
    s_test_conv = s_test[t_test < max(t_train)]
    t_test_conv = t_test[t_test < max(t_train)]

    
    t_test_conv = (t_test_conv - min(t_train)) / (max(t_train) - min(t_train))
    t_train = (t_train - min(t_train)) / (max(t_train) - min(t_train))
    
    score_func = ibs if integrated else lambda a,b,c,d: bs(a,b,c,d)[1][:-1]
    
    pred_test = model.predict(X_test_conv)

    N_bins = len(bin_end_times)
    N_pred = len(pred_test.T)

    assert N_pred == N_bins or N_pred == N_bins + 1, 'Invalid number of bins'

    

    surv_test_pred = 1 - np.cumsum(pred_test, axis=1)[:, :len(bin_end_times)]

    surv_train = Surv().from_arrays(s_train, t_train)
    surv_test = Surv().from_arrays(s_test_conv, t_test_conv)
    
    bin_end_times_conv = bin_end_times
    bin_end_times_conv[-1] = max(t_test_conv)- 0.00000001
    
    return score_func(surv_train, surv_test, surv_test_pred, bin_end_times_conv)



def interpolate_linear(event_prob_by_bin, bin_end_times, event_times, return_survival=True):

    et = np.reshape(event_times, (-1))

    assert len(et) == 1 or len(et) == len(event_prob_by_bin)

    bin_lengths = np.diff([0] + list(bin_end_times))
    bin_start_times = np.array([0] + list(bin_end_times)[:-1])

    time_by_interval = (et[:, np.newaxis] - bin_start_times[np.newaxis, :])
    time_by_interval = time_by_interval / bin_lengths[np.newaxis, :]
    time_by_interval = np.minimum(time_by_interval, 1)
    time_by_interval = np.maximum(time_by_interval, 0)

    assert time_by_interval.max() <= 1
    assert time_by_interval.min() >= 0

    interpolated_cum_prob = np.sum(event_prob_by_bin * time_by_interval, axis=1)

    if return_survival:
        return 1 - interpolated_cum_prob
    else:
        return interpolated_cum_prob


def d_calibration(s_test, t_test, tp_onehot, bin_end_times, bins=10):

    # predictions are the survival probability at the event time (or censoring time)
    predictions = interpolate_linear(tp_onehot, bin_end_times, t_test, return_survival=True)

    event_indicators = s_test == 1

    # include minimum to catch if probability = 1.
    bin_index = np.minimum(np.floor(predictions * bins), bins - 1).astype(int)
    censored_bin_indexes = bin_index[~event_indicators]
    uncensored_bin_indexes = bin_index[event_indicators]

    censored_predictions = predictions[~event_indicators]
    censored_contribution = 1 - (censored_bin_indexes / bins) * (
        1 / censored_predictions
    )
    censored_following_contribution = 1 / (bins * censored_predictions)

    contribution_pattern = np.tril(np.ones([bins, bins]), k=-1).astype(bool)

    following_contributions = np.matmul(
        censored_following_contribution, contribution_pattern[censored_bin_indexes]
    )
    single_contributions = np.matmul(
        censored_contribution, np.eye(bins)[censored_bin_indexes]
    )
    uncensored_contributions = np.sum(np.eye(bins)[uncensored_bin_indexes], axis=0)
    bin_count = (
        single_contributions + following_contributions + uncensored_contributions
    )
    chi2_statistic = np.sum(
        np.square(bin_count - len(predictions) / bins) / (len(predictions) / bins)
    )

    return (chi2_statistic, 1 - chi2.cdf(chi2_statistic, bins - 1))




# One calibration

@dataclass
class KaplanMeier:
    event_times: InitVar[np.array]
    event_indicators: InitVar[np.array]
    survival_times: np.array = field(init=False)
    survival_probabilities: np.array = field(init=False)

    def __post_init__(self, event_times, event_indicators):
        index = np.lexsort((event_indicators, event_times))
        unique_times = np.unique(event_times[index], return_counts=True)
        self.survival_times = unique_times[0]
        population_count = np.flip(np.flip(unique_times[1]).cumsum())

        event_counter = np.append(0, unique_times[1].cumsum()[:-1])
        event_ind = list()
        for i in range(np.size(event_counter[:-1])):
            event_ind.append(event_counter[i])
            event_ind.append(event_counter[i + 1])
        event_ind.append(event_counter[-1])
        event_ind.append(len(event_indicators))
        events = np.add.reduceat(np.append(event_indicators[index], 0), event_ind)[::2]

        self.survival_probabilities = np.empty(population_count.size)
        survival_probability = 1
        counter = 0
        for population, event_num in zip(population_count, events):
            survival_probability *= 1 - event_num / population
            self.survival_probabilities[counter] = survival_probability
            counter += 1

    def predict(self, prediction_times: np.array):
        probability_index = np.digitize(prediction_times, self.survival_times)
        probability_index = np.where(
            probability_index == self.survival_times.size + 1,
            probability_index - 1,
            probability_index,
        )
        probabilities = np.append(1, self.survival_probabilities)[probability_index]

        return probabilities


    
@dataclass
class KaplanMeierArea(KaplanMeier):
    area_times: np.array = field(init=False)
    area_probabilities: np.array = field(init=False)
    area: np.array = field(init=False)

    def __post_init__(self, event_times, event_indicators):
        super().__post_init__(event_times, event_indicators)
        area_probabilities = np.append(1, self.survival_probabilities)
        area_times = np.append(0, self.survival_times)
        if self.survival_probabilities[-1] != 0:
            slope = (area_probabilities[-1] - 1) / area_times[-1]
            zero_survival = -1 / slope
            area_times = np.append(area_times, zero_survival)
            area_probabilities = np.append(area_probabilities, 0)

        area_diff = np.diff(area_times, 1)
        area = np.flip(np.flip(area_diff * area_probabilities[0:-1]).cumsum())

        self.area_times = np.append(area_times, np.inf)
        self.area_probabilities = area_probabilities
        self.area = np.append(area, 0)

    def best_guess(self, censor_times: np.array):
        surv_prob = self.predict(censor_times)
        censor_indexes = np.digitize(censor_times, self.area_times)
        censor_indexes = np.where(
            censor_indexes == self.area_times.size + 1,
            censor_indexes - 1,
            censor_indexes,
        )
        censor_area = (
            self.area_times[censor_indexes] - censor_times
        ) * self.area_probabilities[censor_indexes - 1]
        censor_area += self.area[censor_indexes]
        return censor_times + censor_area / surv_prob

    
    

def one_calibration(s_test, t_test, pred, cutpoints, n_cal_bins=10):
    cutoff = int(len(cutpoints)-1/2)
    time = cutpoints[cutoff]

    
    pred_cum_prob_at_time = np.sum(pred[:,0:(cutoff)], axis=1)
    
    observed_probabilities = []
    expected_probabilities = []

    prediction_order = np.argsort(-pred_cum_prob_at_time)
    predictions = pred_cum_prob_at_time[prediction_order]
    event_times = t_test.copy()[prediction_order]
    event_indicators = (s_test == 1).copy()[prediction_order]
    
    # Can't do np.mean since split array may be of different sizes.
    binned_event_times = np.array_split(event_times, n_cal_bins)
    binned_event_indicators = np.array_split(event_indicators, n_cal_bins)
    probability_means = [np.mean(x) for x in np.array_split(predictions, n_cal_bins)]     
        
    hosmer_lemeshow = 0

    for b in range(n_cal_bins):

        prob = probability_means[b]


        
        km_model = KaplanMeier(binned_event_times[b], binned_event_indicators[b])
        event_probability = 1 - km_model.predict(time)
        observed_probabilities.append(event_probability)
        expected_probabilities.append(prob)
        
        bin_count = len(binned_event_times[b])
        hosmer_lemeshow += (bin_count * event_probability - bin_count * prob) ** 2 / (
            bin_count * prob * (1 - prob)
        )
        
    p_val = 1 - chi2.cdf(hosmer_lemeshow, n_cal_bins - 1)

    
    expected_probabilities = [[item] for item in expected_probabilities]
    reg = LinearRegression().fit(expected_probabilities, observed_probabilities)

    return reg.intercept_, reg.coef_[0], hosmer_lemeshow, p_val, observed_probabilities, expected_probabilities

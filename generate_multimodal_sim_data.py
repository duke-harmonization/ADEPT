import pandas as pd
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def generate_multimodal_data(seed=1978, n_samples = 10000, n_train = 0.9):
    np.random.seed(seed)

    X, y = make_moons(n_samples = int(n_samples / 2), noise=0.3, random_state=seed)
    # make 2 more classes and shift covariates
    X2, y2 = make_moons(n_samples = int(n_samples / 2), noise=0.3, random_state=seed)
    X2 = X2 + 3
    y2 = y2 + 2

    X_multi = np.concatenate((X,X2))
    y = np.concatenate((y, y2))

    s_multi = np.random.binomial(1, 0.5, len(y))

    t_min = 0
    t_boundaries = [10, 30, 70]
    t_max = 100
    t_multi = np.zeros(len(y))


    event_time_multi = np.zeros(len(y))
    censor_time_multi = np.zeros(len(y))
    s_multi = np.zeros(len(y))
    t_multi = np.zeros(len(y))

    for ii in range(0, len(y)):
        censor_time_multi[ii] = np.random.uniform(t_min, t_max)

        if y[ii] == 0:
            event_time_multi[ii] = (np.random.beta(1.5,1.5) * (t_boundaries[0] - t_min)) + t_min
        elif y[ii] == 1:
            event_time_multi[ii] = (np.random.beta(1.5,1.5) * (t_boundaries[1] - t_boundaries[0])) + t_boundaries[0]
        elif y[ii] == 2:
            event_time_multi[ii] = (np.random.beta(1.5,1.5) * (t_boundaries[2] - t_boundaries[1])) + t_boundaries[1]
        elif y[ii] == 3:
            event_time_multi[ii] = (np.random.beta(1.5,1.5) * (t_max - t_boundaries[2])) + t_boundaries[2]


    for ii in range(0, len(y)): 
        if censor_time_multi[ii] < event_time_multi[ii]:
            s_multi[ii] = 0
            t_multi[ii] = censor_time_multi[ii]
        else:
            s_multi[ii] = 1
            t_multi[ii] = event_time_multi[ii]


    # if n_train input is a number, convert it to a proportion of the total samples
    n_train = n_train if n_train < 1 else 1 - (n_samples - n_train) / n_samples
            
                    
    X_train_multi, X_validation_multi, \
    t_train_multi, t_validation_multi,\
    s_train_multi, s_validation_multi, \
    y_moon_train_multi, y_moon_validation_multi = train_test_split(
        X_multi, t_multi, s_multi, y, train_size=n_train, random_state=seed
    )
            
    X_test_multi, X_validation_multi, \
    t_test_multi, t_validation_multi,\
    s_test_multi, s_validation_multi, \
    y_moon_test_multi, y_moon_validation_multi = train_test_split(
        X_validation_multi, t_validation_multi, s_validation_multi, y_moon_validation_multi, test_size=0.2, random_state=seed
    )

    return X_train_multi, X_validation_multi, X_test_multi,\
        t_train_multi, t_validation_multi, t_test_multi,\
        s_train_multi, s_validation_multi, s_test_multi
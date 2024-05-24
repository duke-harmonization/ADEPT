from evaluation import discrete_ci
from evaluation import brier_score

from evaluation import one_calibration

from adept import ADEPT

from generate_flchain_data import generate_flchain_data
from generate_gbsg_data import generate_gbsg_data
from generate_multimodal_sim_data import generate_multimodal_data

import torch
import itertools
import random
import csv
import numpy as np

import sys
import os

import warnings
warnings.filterwarnings("ignore")

def run_pipeline(
    n_cutpoints_vec = [0, 1],
    km_initialization = True,
    prior_bool_vec = [False],
    regularization_strength_vec = [0.5, 1.0, 10.0],
    sigmoid_temperature_vec = [0.01],
    hidden_size_vec = [128],
    weight_decay_vec = [0],
    dataset_vec = ["multi_sim"],
    n_seed = 5,
    learn_cutpoints=True,
    temperature_decay = True,
    batch_size = 64,
    iterations = 250):

    
    
    data_gen_dict = {
                     "FL Chain": generate_flchain_data,
                     "GBSG": generate_gbsg_data,
                     "multi_sim": generate_multimodal_data,
                    }

    for dataset in dataset_vec:
        regularization1 = regularization2 = 0
        # choose data generation function
        data_gen = data_gen_dict[dataset]

        # create output csv
        out_dir = "output_ibs" if learn_cutpoints else "output_ibs_baseline"
        out_file = f'{out_dir}/{dataset}_output_{n_cutpoints_vec[0]}.csv'
        with open(out_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # save average cutpoints too?
            writer.writerow(["seed",
                             "sigmoid temperature",
                             "number cutpoints",
                             "regularization strength",
                             "hidden size",
                             "weight decay",
                             "learned cutpoints",
                             "validation ci",
                             "validation calibration intercept",
                             "validation calibration slope",
                             "validation ibs",
                             "validation ll",
                             "test ci",
                             "test calibration intercept",
                             "test calibration slope",
                             "test ibs",
                             "test ll"
                            ])


        for sigmoid_temperature, n_cutpoints, , regularization_strength, hidden_size, weight_decay in itertools.product( sigmoid_temperature_vec, n_cutpoints_vec, regularization_strength_vec, hidden_size_vec, weight_decay_vec):



            print(f"temp:\t{sigmoid_temperature}\tn_cutpoints\t{n_cutpoints}\\tregularization_strength:\t{regularization_strength}\thidden_size:\t{hidden_size}\tweight_decay:\t{weight_decay}")
            
            
            init_cutpoints = [0] * n_cutpoints
            learned_cutpoints = [0] * n_cutpoints

            for seed in range(n_seed):
                print(f"seed:\t{seed}")
                try:

                    torch.manual_seed(seed)
                    random.seed(seed)

                    X_train, X_validation, X_test, \
                    t_train, t_validation, t_test, \
                    s_train, s_validation, s_test = data_gen(seed)


                    model = ADEPT(X_train = X_train,
                        t_train = t_train,
                        s_train = s_train,
                        X_validation = X_validation,
                        t_validation = t_validation,
                        s_validation = s_validation,
                        sigmoid_temperature = sigmoid_temperature,
                        temperature_decay = temperature_decay,
                        n_cutpoints = n_cutpoints, 
                        iterations = iterations,
                        batch_size = batch_size,
                        regularization_strength = regularization_strength,
                        hidden_size = hidden_size,
                        km_initialization = km_initialization,
                        weight_decay = weight_decay,
                        learn_cutpoints = learn_cutpoints,
                        seed = seed)

                    model.train()

                    learned_cutpoints = [x.item() for x in model.cutpoints]


                    # validation metrics
                    validation_ci = discrete_ci(model.predict(X_validation), model.cutpoints, t_validation, s_validation, t_train).item()

                    t_validation_scaled = (t_validation - min(t_train)) / (max(t_train) - min(t_train))

                    validation_intercept, \
                    validation_coef, \
                    validation_hosmer_lemeshow, \
                    validation_p_val, _, _ = one_calibration(s_validation, t_validation_scaled, 
                                    pred = model.predict(X_validation),
                                    cutpoints = sorted([x.item() for x in model.cutpoints]),
                                    n_cal_bins = 10)
                    
                    
                    pred_validation = model.predict(X_validation)
                    bin_end_times = [x.item() for x in model.cutpoints] + [1]
                    integrated = True
                    validation_ibs = brier_score(s_train, t_train, s_validation, t_validation, X_validation, model, bin_end_times, integrated)
                    
                    
                    cutpoint_bool = [1] * n_cutpoints
                    validation_ll = model.multinomial_loss(torch.from_numpy(X_validation).float(), torch.from_numpy(t_validation).float(), torch.from_numpy(s_validation).float(), regularization_strength, cutpoint_bool)[0].item()
                    

    
                    print(f"validation ci:\t{validation_ci}\tintercept:\t{validation_intercept}\tslope:\t{validation_coef}")
                    print(f"intial cutpoint:\t{model.cutpoint0}\tlearned cutpoint:\t{[x.item() for x in model.cutpoints]}")


                    # test metrics
                    test_ci = discrete_ci(model.predict(X_test), model.cutpoints, t_test, s_test, t_train).item()


                    t_test_scaled = (t_test - min(t_train)) / (max(t_train) - min(t_train))

                    test_intercept, \
                    test_coef, \
                    test_hosmer_lemeshow, \
                    test_p_val, _, _ = one_calibration(s_test, t_test_scaled, 
                                    pred = model.predict(X_test),
                                    cutpoints = sorted([x.item() for x in model.cutpoints]),
                                    n_cal_bins = 10)

                    
                    pred_test = model.predict(X_test)
                    bin_end_times = [x.item() for x in model.cutpoints] + [1]
                    integrated = True
                    test_ibs = brier_score(s_train, t_train, s_test, t_test, X_test, model, bin_end_times, integrated)
                    
                    test_ll = model.multinomial_loss(torch.from_numpy(X_test).float(), torch.from_numpy(t_test).float(), torch.from_numpy(s_test).float(),  regularization_strength, cutpoint_bool)[0].item()
            

                    with open(out_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([seed,
                                         sigmoid_temperature,
                                         n_cutpoints, 
                                         regularization_strength,
                                         hidden_size,
                                         weight_decay,
                                         learned_cutpoints,
                                         validation_ci,
                                         validation_intercept,
                                         validation_coef,
                                         validation_ibs,
                                         validation_ll,
                                         test_ci,
                                         test_reg,
                                         test_intercept,
                                         test_coef,
                                         test_ibs,
                                         test_ll
                                        ])
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno, e)
                    pass
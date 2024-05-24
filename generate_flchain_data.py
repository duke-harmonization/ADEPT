import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.datasets import load_flchain


def generate_flchain_data(seed=1978):

    data_x, data_y = load_flchain()
    x_df = pd.DataFrame(data_x)
    y_df = pd.DataFrame.from_records(data_y)
    fl_data = pd.concat([data_x, y_df], axis=1)
    
    # remove column that only has value if there is an event
    fl_data = fl_data.drop("chapter", axis=1)

    # drop rows with na's and fix indices
    fl_data = fl_data.dropna()
    fl_data = fl_data.reset_index()
    fl_data = fl_data.drop("index", axis=1)

    # change binary variables to 0,1
    fl_data = fl_data.replace({'mgus': {'yes': 1, 'no': 0}})
    fl_data = fl_data.replace({'sex': {'F': 1, 'M': 0}})
    fl_data = fl_data.replace({'death': {True: 1, False: 0}})

    # change fl group into dummy variables
    fl_data = pd.concat([fl_data, pd.get_dummies(fl_data["flc.grp"])], axis=1)
    fl_data = fl_data.drop("flc.grp", axis=1)
    
    X_flchain = fl_data.drop(["death", "futime"], axis=1)
    X_flchain["sample.yr"] = pd.to_numeric(X_flchain["sample.yr"])
    numrc_cols = X_flchain.nunique() > 2
    X_flchain.loc[:, numrc_cols] = (X_flchain.loc[:, numrc_cols] - X_flchain.loc[:, numrc_cols].mean()) / X_flchain.loc[:, numrc_cols].std()
    X_flchain = X_flchain.to_numpy(dtype='float')

    s_flchain = fl_data["death"].to_numpy()
    t_flchain = fl_data["futime"].to_numpy()
    
    
    X_train_flchain, X_test_flchain, \
    t_train_flchain, t_test_flchain,\
    s_train_flchain, s_test_flchain = train_test_split(
        X_flchain, t_flchain, s_flchain, test_size=0.1, random_state=1978
    )

    X_train_flchain, X_validation_flchain, \
    t_train_flchain, t_validation_flchain,\
    s_train_flchain, s_validation_flchain = train_test_split(
        X_train_flchain, t_train_flchain, s_train_flchain, test_size=0.2, random_state=seed
    )
  
      
    return X_train_flchain, X_validation_flchain, X_test_flchain,\
        t_train_flchain, t_validation_flchain, t_test_flchain,\
        s_train_flchain, s_validation_flchain, s_test_flchain
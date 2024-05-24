import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.datasets import load_gbsg2


def generate_gbsg_data(seed=1978):

    
    X_gbsg, data_y = load_gbsg2()
    y_df = pd.DataFrame.from_records(data_y)
    s_gbsg = y_df["cens"].to_numpy()
    t_gbsg = y_df["time"].to_numpy()
    
    # change binary variables to 0,1
    X_gbsg = X_gbsg.replace({'horTh': {'yes': 1, 'no': 0}})
    X_gbsg = X_gbsg.replace({'menostat': {'Pre': 1, 'Post': 0}})

    # # change tgrade into dummy variables
    X_gbsg = pd.concat([X_gbsg, pd.get_dummies(X_gbsg["tgrade"])], axis=1)
    X_gbsg = X_gbsg.drop("tgrade", axis=1)

    numrc_cols = X_gbsg.nunique() > 2
    X_gbsg.loc[:, numrc_cols] = (X_gbsg.loc[:, numrc_cols] - X_gbsg.loc[:, numrc_cols].mean()) / X_gbsg.loc[:, numrc_cols].std()
    X_gbsg = X_gbsg.to_numpy(dtype='float')
    
    X_train_gbsg, X_test_gbsg, \
    t_train_gbsg, t_test_gbsg,\
    s_train_gbsg, s_test_gbsg = train_test_split(
        X_gbsg, t_gbsg, s_gbsg, test_size=0.1, random_state=1978
    )

    X_train_gbsg, X_validation_gbsg, \
    t_train_gbsg, t_validation_gbsg,\
    s_train_gbsg, s_validation_gbsg = train_test_split(
        X_train_gbsg, t_train_gbsg, s_train_gbsg, test_size=0.2, random_state=seed
    )
  
      
    return X_train_gbsg, X_validation_gbsg, X_test_gbsg,\
        t_train_gbsg, t_validation_gbsg, t_test_gbsg,\
        s_train_gbsg, s_validation_gbsg, s_test_gbsg
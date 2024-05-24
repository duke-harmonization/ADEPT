import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def generate_bimodal_data(seed = 1978, n_train = 300, n_validation = 100, n_test = 5000, noise = 0.3, T_cutpoint = 67):
   
    n_samples = n_train + n_validation + n_test
    T_min = 0
    T_max = 100

    rs = np.random.RandomState(seed)
    
    t = rs.rand(n_samples) * (T_max - T_min) + T_min
    c = rs.rand(n_samples) * (T_max - T_min) + T_min

    y = np.minimum(t, c)
    s = (t < c).astype(int)

    g = (t > T_cutpoint).astype(int)
    
    
    X_mm, g_mm = make_moons(
        n_samples=(np.sum(g == 0), np.sum(g == 1)),
        noise=noise,
        shuffle=False,
        random_state=seed
    )
    
    X = np.zeros(shape=(n_samples, 2))

    X[g == 0, :] = X_mm[g_mm == 0, :]*10
    X[g == 1, :] = X_mm[g_mm == 1, :]*10
    

    
    
    X_train, X_validation, \
    t_train, t_validation, \
    s_train, s_validation, \
    g_train, g_validation = train_test_split(
        X, t, s, g, train_size=n_train, random_state=seed
    )
            
    X_test, X_validation, \
    t_test, t_validation, \
    s_test, s_validation, \
    g_test, g_validation = train_test_split(
        X_validation, t_validation, s_validation, g_validation, test_size=n_validation, random_state=seed
    )

    
    return X_train, X_validation, X_test, \
        t_train, t_validation, t_test, \
        s_train, s_validation, s_test, \
        g_train, g_validation, g_test

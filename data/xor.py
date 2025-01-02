import numpy as np


def get_xor(n_rows=1000, 
            number_features=6, 
            flip_fraction=0.3, 
            seed=42, 
            add_negated:bool = True):
    
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n_rows, number_features))
    y = np.logical_xor(X[:, 0:1], X[:, 1:2])
    flips = rng.random((n_rows, 1)) < flip_fraction
    y = np.logical_xor(y, flips)

    if add_negated:
        X_t = np.zeros((n_rows, number_features*2), dtype=int)
        for index, i in enumerate(X):
            temp = np.array([0 if x == 1 else 1 for x in i])
            t = np.concatenate((i, temp), axis=0,out=None, dtype=int)
            X_t[index] = t
        X = X_t
   
    y = y*1

    
    return X.astype(np.uint8), y.reshape(y.shape[0]).astype(np.uint32)


if __name__ == "__main__":

    x, y = get_xor()

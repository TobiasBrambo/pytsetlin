    
from sklearn.datasets import fetch_openml
import numpy as np



def get_mnist(add_negated=False):


    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

    X = np.where(X.reshape((X.shape[0], 28 * 28)) > 75, 1, 0)

    y = y.astype(np.uint32)

    if add_negated:
        X_t = np.zeros((X.shape[0], X.shape[1]*2), dtype=int)
        for index, i in enumerate(X):
            temp = np.array([0 if x == 1 else 1 for x in i])
            t = np.concatenate((i, temp), axis=0, out=None, dtype=int)
            X_t[index] = t
        X = X_t



    x_train, x_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]


    return x_train.astype(np.uint8), x_test.astype(np.uint8), y_train, y_test



if __name__ == "__main__":

    X_train, X_test, y_train, y_test = get_mnist(add_negated=True)

    print(X_train.shape)
from pandas import read_csv
import numpy as np


def load_data(filename):
    """
    Load data from a csv file

    Parameters
    ----------
    filename : string
        Filename to be loaded.

    Returns
    -------
    X : ndarray
        the data matrix.

    y : ndarray
        the labels of each sample.
    """
    data = read_csv(filename)
    z = np.array(data)
    y = z[:, 0]
    X = z[:, 1:]
    return X, y


def split_data(x, y, tr_fraction=0.5):
    """
    Split the data x, y into two random subsets

    """
    pass

def fit(self, xtr, ytr):
    """
    Compute the average centroids for each class

    Parameters
    ----------
    xtr: training data
    ytr: training labels

    Returns
    -------
    self: trained NMC classifier
    """

    n_dimensions = xtr.shape[1]
    n_classes = np.unique(ytr).size
    self._centroids = np.zeros(shape=(n_classes, n_dimensions))
    for k in range(n_classes):
    # extract images from one class and then average along dim 0
        self._centroids[k, :] = np.mean(xtr[ytr == k, :], axis=0)
        return self
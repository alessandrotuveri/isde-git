import numpy as np
from sklearn.metrics import pairwise_distances


class NMC(object):
    """
    Class implementing the Nearest Mean Centroid (NMC) classifier.

    This classifier estimates one centroid per class from the training data,
    and predicts the label of a never-before-seen (test) point based on its
    closest centroid.

    Attributes
    -----------------
    - centroids: read-only attribute containing the centroid values estimated
        after training

    Methods
    -----------------
    - fit(x,y) estimates centroids from the training data
    - predict(x) predicts the class labels on testing points

    """

    class NMC:
        """
        Class defined for NMC classifier.
        """

        def __init__(self):
            self._centroids = None

        @property
        def centroids(self):
            return self._centroids

        # @centroids.setter
        # def centroids(self, value):
        #    self._centroids = value

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

            pass

        def predict(self, xts):
            """
            Brand new docstring

            Parameters
            ----------
            xts

            Returns
            -------

            """
            scores = self.decision_function(xts)
            ypred = np.argmax(scores, axis=1)
            return ypred
from hp_toolkit.hp import Model
import numpy as np
from numpy.random import RandomState


class Truth(Model):

    params = dict()

    def fit(self, X, y=None, X_valid=None, y_valid=None):
        self.X = X

    def sample(self, nb_samples, only_means=True):
        rng = RandomState(self.state)
        indexes = rng.randint(0, self.X.shape[0], size=nb_samples)
        if only_means is True:
            return self.X[indexes]
        else:
            return self.X[indexes] > 0.5

    def get_nb_params(self):
        return np.prod(self.X.shape)

    def get_log_likelihood(self, X):
        return -np.inf, 0

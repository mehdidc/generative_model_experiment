from hp_toolkit.hp import Param, Model
from sklearn.neural_network import BernoulliRBM
import batch_optimizer
import numpy as np

from sklearn.utils.fixes import expit

import sys
sys.path.append("utils")
from rbm_ais import rbm_partition_function, rbm_log_energy


class RBM(Model):

    params = dict(
        nb_units=Param(initial=100, interval=[100, 500], type='int'),
        nb_gibbs_iterations=Param(initial=10000,
                                  interval=[100, 10000], type='int')
    )
    params.update(batch_optimizer.params)

    def __init__(self, **kwargs):
        super(RBM, self).__init__(**kwargs)
        self.Z_mean = None
        self.Z_std = None

    def fit(self, X, y=None, X_valid=None, y_valid=None):
        self.model = BernoulliRBM(n_components=self.nb_units,
                                  learning_rate=self.learning_rate,
                                  batch_size=self.batch_size,
                                  verbose=1,
                                  n_iter=self.max_epochs)

        self.nb_features = X.shape[1]
        self.model.fit(X)
        self._compute_partition_function()

    def _compute_partition_function(self):
        hidbiases = self.model.intercept_hidden_
        visbiases = self.model.intercept_visible_
        vishid = self.model.components_.T
        Z_mean, Z_std = rbm_partition_function(
            hidbiases, visbiases, vishid,
            rng=np.random, nb_iterations=1000,
            gibbs_nb_iterations=1,
            nb_runs=1)
        self.Z_mean = Z_mean
        self.Z_std = Z_std

    def get_log_likelihood(self, X):
        if self.Z_mean is None or self.Z_std is None:
            self._compute_partition_function()
        hidbiases = self.model.intercept_hidden_
        visbiases = self.model.intercept_visible_
        vishid = self.model.components_.T
        log_energy = rbm_log_energy(X, hidbiases, visbiases, vishid)
        p = -(log_energy - self.Z_mean)
        return p.mean(), p.std()

    def get_nb_params(self):
        return (np.prod(self.model.intercept_hidden_.shape) +
                np.prod(self.model.intercept_visible_.shape) +
                np.prod(self.model.components_.shape))

    def sample(self, nb_samples, only_means=True):
        v = np.random.uniform(size=(nb_samples, self.nb_features))
        for i in range(self.nb_gibbs_iterations):
            v = self.model.gibbs(v)
        if only_means is True:
            h = self.model._sample_hiddens(v, np.random)
            p = np.dot(h, self.model.components_)
            p += self.model.intercept_visible_
            expit(p, out=p)
            v = p
            return v
        else:
            return v

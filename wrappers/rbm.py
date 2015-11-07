from hp_toolkit.hp import Param, Model
from sklearn.neural_network import BernoulliRBM
import batch_optimizer
import numpy as np

from sklearn.utils.fixes import expit


class RBM(Model):

    params = dict(
        nb_units=Param(initial=100, interval=[100, 500], type='int'),
        nb_gibbs_iterations=Param(initial=10000,
                                  interval=[100, 10000], type='int')
    )
    params.update(batch_optimizer.params)

    def fit(self, X, y=None, X_valid=None, y_valid=None):
        self.model = BernoulliRBM(n_components=self.nb_units,
                                  learning_rate=self.learning_rate,
                                  batch_size=self.batch_size,
                                  verbose=1,
                                  n_iter=self.max_epochs)
        self.model.fit(X)
        self.nb_features = X.shape[1]

    def get_log_likelihood(self, X):
        return -self.model.score_samples(X).mean(), 0

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

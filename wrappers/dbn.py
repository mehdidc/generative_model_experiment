from hp_toolkit.hp import Param, default_eval_functions, Model
import batch_optimizer
from libs.deepbelief.deepbelief import dbn, rbm
import numpy as np
class DBN(Model):

    params = dict(
        nb_units = Param(initial=100, interval=[100, 200], type='int'),
        nb_layers = Param(initial=1, interval=[1, 4], type='int'),
    )
    params.update(batch_optimizer.params)

    def fit(self, X, X_valid=None):

        self.model = dbn.DBN(rbm.RBM(X.shape[1], self.nb_units))
        self.model[0].learning_rate = self.learning_rate
        self.model[0].momentum = self.momentum
        self.model.train(X.T, num_epochs=self.max_epochs)
        for i in range(self.nb_layers):
            self.model.add_layer(rbm.RBM(self.nb_units, self.nb_units))
            self.model[i + 1].learning_rate = self.learning_rate
            self.model[i + 1].momentum = self.momentum
            self.model.train(X.T, num_epochs=self.max_epochs)

    def get_log_likelihood(self, X):
        return -self.model.estimate_log_likelihood(X.T), 0

    def get_nb_params(self):
        return sum(np.prod(self.model[i].W.shape) + np.prod(self.model[i].b.shape) for i in range(self.nb_layers + 1))

    def transform(self, X):
        z_mean, z_sigma = self.model.encode(X)
        return z_mean

    def sample(self, nb_samples, only_means=True):
        pass

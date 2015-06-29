from hp_toolkit.hp import Param, default_eval_functions, Model
import batch_optimizer
import theano.tensor as T

from libs.GSN.model import experiment
from libs.GSN import likelihood_estimation_parzen

import theano

default_args = dict(
        K=2,
        N=4,
        n_epoch=1000,
        batch_size=100,
        hidden_add_noise_sigma=2,
        input_salt_and_pepper=0.4,
        learning_rate=0.25,
        momentum=0.5,
        annealing=0.995,
        hidden_size=1500,
        act='tanh',
        dataset='MNIST',
        data_path='.',
        vis_init=0,
        noiseless_h1=1,
        input_sampling=1,
        test_model=0,
)
class GSN(Model):

    params = dict(
        nb_units = Param(initial=100, interval=[100, 200], type='int'),
        nb_layers = Param(initial=1, interval=[1, 4], type='int'),
        nb_walkbacks = Param(initial=2, interval=[2, 10], type='int'),

    )
    params.update(batch_optimizer.params)

    
    def fit(self, X, X_valid=None):
        args = default_args.copy()
        args["K"] = self.nb_layers
        args["N"] = self.nb_walkbacks
        args["n_epoch"] = self.nb_units
        args["batch_size"] = self.batch_size
        args["learning_rate"] = self.learning_rate
        args["momentum"] = self.momentum
        args["hidden_size"] = self.nb_units
        if X_valid is None:
            X_valid = X#TODO
        class Dummy(object):
            pass
        d = Dummy()
        d.__dict__.update(args)
        self.samples = theano.shared(experiment(d, None, ((X, None), (X_valid, None), (X_valid, None)) ), borrow=True)
    
    def get_log_likelihood(self, X):
        return likelihood_estimation_parzen.main(0.20, self.samples, X)
    
    def get_nb_params(self):
        return 0

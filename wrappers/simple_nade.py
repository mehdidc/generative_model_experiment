from collections import OrderedDict
import theano.tensor as T
import numpy as np
from lasagne import updates, layers

from numpy.random import RandomState

from lasagnekit.generative.capsule import Capsule
from lasagnekit.generative.nade import NadeLayer
from lasagnekit import easy

import batch_optimizer

from hp_toolkit.hp import Param, Model


class MyBatchOptimizer(easy.BatchOptimizer):

    def iter_update(self, epoch, nb_batches, iter_update_batch):
        status = super(MyBatchOptimizer, self).iter_update(epoch, nb_batches, iter_update_batch)
        ll_train = -self.model.log_likelihood(self.model.X_train)
        status["ll_train"] = ll_train.mean()
        status["ll_train_std"] = ll_train.std()
        if self.model.X_valid is not None:
            ll_valid = -self.model.log_likelihood(self.model.X_valid)
            status["ll_valid"] = ll_valid.mean()
            status["ll_valid_std"] = ll_valid.std()
        return status


class SimpleNADE(Model):

    params = dict(
        latent_dim=Param(initial=10, interval=[2, 5, 10, 20, 30, 80, 100, 120, 160, 180, 300, 500], type='choice'),
        nb_units=Param(initial=100, interval=[100, 1000], type='int'),
    )
    params.update(batch_optimizer.params)

    def build_model(self, X):
        # batch optimizer
        batch_optimizer = MyBatchOptimizer(max_nb_epochs=self.max_epochs,
                optimization_procedure=(updates.adam, {"learning_rate": self.learning_rate}),
                                        verbose=1,
                                        whole_dataset_in_device=True,
                                        batch_size=self.batch_size)
        x_dim = X.shape[1]
        self.x_dim = x_dim
        h_dim = self.latent_dim
        x_in = layers.InputLayer((None, x_dim))
        l_out = NadeLayer(x_in, num_units=h_dim)
        model = easy.LightweightModel([x_in], [l_out])

        def loss_function(model, tensors):
            o, = model.get_output(tensors.get("X"))
            return -T.log(o).sum(axis=1).mean()

        input_variables = OrderedDict(
                X=dict(tensor_type=T.matrix),
        )

        functions = dict(
            sample=dict(
                get_output=lambda model, X: model.get_output(X, sampler=True)[0],
                params=["X"]
            ),
            log_likelihood=dict(
                get_output=lambda model, X: T.log(model.get_output(X)[0]).sum(axis=1),
                params=["X"]
            ),
        )

        nade = Capsule(input_variables, model,
            loss_function,
            functions=functions,
            batch_optimizer=batch_optimizer)
        self.model = nade


    def fit(self, X, y=None, X_valid=None, y_valid=None):
        self.build_model(X)
        self.model.X_train = X
        self.model.X_valid = X_valid
        if X_valid is not None:
            self.batch_optimizer.patience_stat = "ll_valid"
        self.model.fit(X=X)

    def get_log_likelihood(self, X):
        ll = -self.model.log_likelihood(X)
        return ll.mean(), ll.std()

    def get_nb_params(self):
        return sum(np.prod(param.get_value().shape) for param in self.model.all_params)

    def sample(self, nb_samples, only_means=True):
        samples = np.ones((nb_samples, self.x_dim)).astype(np.float32)
        means =  self.model.sample(samples)
        if only_means is True:
            return means
        else:
            rng = RandomState(self.state)
            return means <= rng.uniform(size=means.shape)

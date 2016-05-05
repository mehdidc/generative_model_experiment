import sys
import os

from hp_toolkit.hp import Param, default_eval_functions, Model
import batch_optimizer
import theano.tensor as T

from pylearn2.models import mlp
from pylearn2.training_algorithms import sgd
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

from libs import adversarial
from libs.adversarial import parzen_ll

import numpy as np

class Adversarial(Model):

    params = dict(
            nb_units_discriminator = Param(initial=100, interval=[100, 200], type='int'),
            nb_layers_discriminator = Param(initial=1, interval=[1, 4], type='int'),
            nb_units_generator = Param(initial=100, interval=[100, 200], type='int'),
            nb_layers_generator = Param(initial=1, interval=[1, 4], type='int'),
            nb_noise_units_generator = Param(initial=100, interval=[100, 200], type='int'),
    )
    params.update(batch_optimizer.params)

    def fit(self, X, X_valid=None):

        X_ds = DenseDesignMatrix(X=X)

        if X_valid is not None:
            X_valid_ds = DenseDesignMatrix(X=X_valid)
        else:
            X_valid_ds = None

        # monitor
        monitoring_dataset=dict(
            train=X_ds,
        )
        if X_valid is not None:
            monitoring_dataset.update(
                    dict(valid=X_valid_ds)
            )

        # Algorithm
        trainer = sgd.SGD(learning_rate=self.learning_rate,
                          batch_size=self.batch_size,
                          termination_criterion=EpochCounter(self.max_epochs),
                          cost=adversarial.AdversaryCost2(
                                        scale_grads=0,
                                        discriminator_default_input_scale=1.,
                                        discriminator_input_scales=dict(h0=1.25)
                           ),
        )
        # generator
        layers = []
        for i in range(self.nb_layers_generator):
            layer = mlp.RectifiedLinear(layer_name="h{0}".format(i), dim=self.nb_units_generator, irange=0.05)
            layers.append(layer)
        layers.append(mlp.Sigmoid(layer_name="y", dim=X.shape[1], irange=0.05))
        ann = mlp.MLP(layers, nvis=self.nb_noise_units_generator)
        generator = adversarial.Generator(
                noise="uniform",
                monitor_ll=1,
                mlp=ann)

        # discriminator
        layers = []
        for i in range(self.nb_layers_discriminator):
            layer = mlp.RectifiedLinear(layer_name="h{0}".format(i), dim=self.nb_units_discriminator, irange=0.05)
            layers.append(layer)
        layers.append(mlp.Sigmoid(layer_name="y", dim=1, irange=0.05))
        ann = mlp.MLP(layers, nvis=X.shape[1])
        discriminator = ann
        # model
        model = adversarial.AdversaryPair(
                generator=generator,
                discriminator=discriminator
        )
        trainer.setup(model, X_ds)

        while True:
            trainer.train(dataset=X_ds)
            trainer.monitor.report_epoch()
            trainer.monitor()
            if not trainer.continue_learning(model):
                break
        self.model = model

        # preparing parzen density estimation
        samples = self.sample(10000)
        sigma = 2.
        self.parzen = parzen_ll.theano_parzen(samples, sigma)

    def get_log_likelihood(self, X):
        ll = parzen_ll.get_nll(X, self.parzen, batch_size=self.batch_size)
        se = ll.std() / np.sqrt(X.shape[0])
        return -ll.mean(), se

    def get_nb_params(self):
        return sum(np.prod(param.get_value().shape) for param in self.model.get_params())

    def sample(self, nb_samples, only_means=True):
        return self.model.generator.sample(nb_samples).eval()

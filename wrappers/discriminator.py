import numpy as np
from lasagne import layers
from lasagne.nonlinearities import softmax

from collections import OrderedDict
import theano.tensor as T
import numpy as np
from lasagne import updates, layers, objectives

from numpy.random import RandomState

from lasagnekit.generative.capsule import Capsule
from lasagnekit import easy

import batch_optimizer

from hp_toolkit.hp import Param, Model


def build_discriminator_data(models, samples_per_model=100):
    data = []
    labels = []
    for i, model in enumerate(models):
        data_model = model.sample(samples_per_model)
        data.append(data_model)
        labels.extend([i] * samples_per_model)
    labels = np.array(labels)
    return np.concatenate(data, axis=0).astype(np.float32), labels.astype(np.int32)


def build_discriminator_model_fully(shape_inputs, num_classes):
    l_in = layers.InputLayer((None, shape_inputs[1]))
    l_hid = layers.DenseLayer(l_in, num_units=500)
    l_out = layers.DenseLayer(
        l_hid,
        num_units=num_classes,
        nonlinearity=softmax)
    return l_in, l_out


class MyBatchOptimizer(easy.BatchOptimizer):

    def iter_update(self, epoch, nb_batches, iter_update_batch):
        status = super(
            MyBatchOptimizer,
            self).iter_update(
            epoch,
            nb_batches,
            iter_update_batch)

        def fn(location):
            return (self.model.predict(self.X_train[location]).argmax(axis=1) !=
                    self.y_train[location]).mean()

        error_train = list(easy.compute_over_minibatches(fn,
            self.X_train.shape[0], self.batch_size))
        status["train_error"] = np.mean(error_train)
        status["train_error_std"] = np.std(error_train)

        if self.X_valid is not None:
            def fn(location):
                return (self.model.predict(self.X_valid[location]).argmax(axis=1) !=
                        self.y_valid[location]).mean()
            error_valid = list(easy.compute_over_minibatches(fn,
                self.X_valid.shape[0], self.batch_size))
            status["test_error"] = np.mean(error_valid)
            status["test_error_std"] = np.std(error_valid)
        return status


models = dict(fully=build_discriminator_model_fully)


class Discriminator(Model):

    params = dict(
        kind=Param(initial="fully", interval=models.keys(), type='choice')
    )
    params.update(batch_optimizer.params)

    def build_model(self, X, num_classes):
        l_in, l_out = models[self.kind](X.shape, num_classes)
        model = easy.LightweightModel([l_in], [l_out])

        batch_optimizer = MyBatchOptimizer(max_nb_epochs=self.max_epochs,
                optimization_procedure=(updates.adam, {"learning_rate": self.learning_rate}),
                                        verbose=1,
                                        whole_dataset_in_device=True,
                                        batch_size=self.batch_size)
        self.batch_optimizer = batch_optimizer

        def loss_function(model, tensors):
            y, = model.get_output(tensors["X"])
            return objectives.categorical_crossentropy(y, tensors["y"]).mean()

        input_variables = OrderedDict(
            X=dict(tensor_type=T.matrix),
            y=dict(tensor_type=T.ivector)
        )

        functions = dict(
            predict=dict(
                get_output=lambda model, X: model.get_output(X)[0],
                params=["X"]
            ),
        )
        return Capsule(input_variables, model, loss_function,
                       functions=functions,
                       batch_optimizer=batch_optimizer)

    def fit(self, X, y, X_valid=None, y_valid=None):
        num_classes = len(set(y))
        self.model = self.build_model(X, num_classes)
        self.model.X_train = X
        self.model.X_valid = X_valid

        self.batch_optimizer.X_train = X
        self.batch_optimizer.y_train = y
        if X_valid is not None:
            self.batch_optimizer.patience_stat = "error_valid"
            self.batch_optimizer.X_valid = X_valid
            self.batch_optimizer.y_valid = y_valid
        self.model.fit(X=X, y=y)

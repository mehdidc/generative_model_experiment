from hp_toolkit.hp import Param, Model
import theano.tensor as T
import numpy as np
import batch_optimizer
from lasagne import updates, layers, nonlinearities
from lasagnekit import easy
from lasagnekit.generative import va
from theano.sandbox import rng_mrg


class MyBatchOptimizer(easy.BatchOptimizer):

    def iter_update(self, epoch, nb_batches, iter_update_batch):
        status = super(MyBatchOptimizer, self).iter_update(epoch, nb_batches, iter_update_batch)
        status["lb_train"] = self.model.get_likelihood_lower_bound(self.model.X_train)
        status["ll_train"] = self.model.log_likelihood_approximation_function(self.model.X_train)[0]
        if self.model.X_valid is not None:
            status["lb_valid"] = self.model.get_likelihood_lower_bound(self.model.X_valid)
            status["ll_valid"] = self.model.log_likelihood_approximation_function(self.model.X_valid)
        return status


class VA(Model):

    params = dict(
        latent_dim=Param(initial=10, interval=[2, 5, 10, 20, 30, 80, 100, 120, 160, 180, 300, 500], type='choice'),
        nb_units_encoder=Param(initial=100, interval=[100, 1000], type='int'),
        nb_layers_encoder=Param(initial=1, interval=[1, 4], type='int'),
        nb_units_decoder=Param(initial=100, interval=[100, 1000], type='int'),
        nb_layers_decoder=Param(initial=1, interval=[1, 4], type='int'),
    )
    params.update(batch_optimizer.params)

    def build_model(self, X):
        # batch optimizer
        batch_optimizer = MyBatchOptimizer(max_nb_epochs=self.max_epochs,
                optimization_procedure=(updates.adam, {"learning_rate": self.learning_rate}),
                                           verbose=1,
                                           whole_dataset_in_device=True,
                                           #patience_nb_epochs=25,
                                           #patience_stat="lb_train",
                                           #patience_progression_rate_threshold=0.99,
                                           #patience_check_each=10,
                                           batch_size=self.batch_size)

        # X to Z (encoder)
        x_in = layers.InputLayer(shape=(None, X.shape[1]))
        h = x_in
        X_tensor = T.matrix()
        for i in range(int(self.nb_layers_encoder)):

            h = layers.DenseLayer(h, num_units=self.nb_units_encoder,
                                  nonlinearity=nonlinearities.rectify)
        z_mean_out = layers.DenseLayer(h, num_units=self.latent_dim,
                                       nonlinearity=nonlinearities.linear)
        z_sigma_out = layers.DenseLayer(h, num_units=self.latent_dim,
                                        nonlinearity=nonlinearities.linear)
        nnet_x_to_z = easy.LightweightModel([x_in],
                                       [z_mean_out, z_sigma_out])
        # Z to X (decoder)
        z_in = layers.InputLayer(shape=(None, self.latent_dim))
        Z_tensor = T.matrix()
        h = z_in
        for i in range(int(self.nb_layers_decoder)):
            h = layers.DenseLayer(h, num_units=self.nb_units_decoder,
                                  nonlinearity=nonlinearities.rectify)
        x_out = layers.DenseLayer(h, num_units=X.shape[1],
                                  nonlinearity=nonlinearities.linear)
        nnet_z_to_x = easy.LightweightModel([z_in], [x_out])

        self.model = va.VariationalAutoencoder(nnet_x_to_z, nnet_z_to_x,
                                               batch_optimizer,
                                               rng=rng_mrg.MRG_RandomStreams(seed=self.state),
                                               nb_z_samples=1)

    def fit(self, X, y=None, X_valid=None, y_valid=None):
        self.build_model(X)
        self.model.X_train = X
        self.model.X_valid = X_valid
        if X_valid is not None:
            self.batch_optimizer.patience_stat = "lb_valid"
        self.model.fit(X)

    def get_log_likelihood(self, X):
        mean, std = (self.model.log_likelihood_approximation_function(X))
        return float(mean), float(std)

    def get_nb_params(self):
        return sum(np.prod(param.get_value().shape) for param in self.model.all_params)

    def transform(self, X):
        z_mean, z_sigma = self.model.encode(X)
        return z_mean

    def sample(self, nb_samples, only_means=True):
        return self.model.sample(nb_samples, only_means=only_means)

if __name__ == "__main__":
    from datasets import datasets
    X = datasets.get("lfw")()
    m = VA()
    m.fit(X)

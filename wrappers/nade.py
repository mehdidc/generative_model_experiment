from hp_toolkit.hp import Param, default_eval_functions, Model
import batch_optimizer

from libs.rnade.buml import (
        Instrumentation,
        Backends,
        Optimization,
        TrainingController,
        Utils,
        Data,
)
from libs.rnade.buml import NADE as NADE_orig

from libs.rnade import buml
import os

import numpy as np



from libs.rnade.buml.Utils.DropoutMask import create_dropout_masks
from libs.rnade.buml.Utils.theano_helpers import floatX


import scipy.stats
import gc


class NADE(Model):

    params = dict(
        nb_units = Param(initial=100, interval=[100, 300], type='int'),
        nb_layers = Param(initial=1, interval=[1, 4], type='int'),
    )
    params.update(batch_optimizer.params)


    def fit(self, X, X_valid=None):

        n_visible = X.shape[1]

        console = Backends.Console()
        X = Data.Dataset(X)

        if X_valid is not None:
            X_valid = Data.Dataset(X_valid)

        h_layers = self.nb_layers

        # Dataset of masks


        os.environ["DATASETSPATH"] = "./tmp"
        #try:
        #    masks_filename = "ds" + ".masks"
        #    masks_route = os.path.join(os.environ["DATASETSPATH"], masks_filename)
        #    masks_dataset = Data.BigDataset(masks_route + ".hdf5", "masks/.*", "masks")
        #except:

        #masks_filename = "ds" + ".masks"
        #masks_route = os.path.join(os.environ["DATASETSPATH"], masks_filename)
        #create_dropout_masks(os.environ["DATASETSPATH"], masks_filename, n_visible, ks=1000)
        #masks_dataset = Data.BigDataset(masks_route + ".hdf5", "masks/.*", "masks")


        nade_class = NADE_orig.OrderlessBernoulliNADE
        nade = nade_class(n_visible, self.nb_units, self.nb_layers, nonlinearity="RLU")

        loss_function = "sym_neg_loglikelihood_gradient"
        #validation_loss_measurement = Instrumentation.Function("validation_loss", lambda ins:-ins.model.estimate_average_loglikelihood_for_dataset_using_masks(validation_dataset, masks_dataset, loops=options.validation_loops)) # TODO

        nade.initialize_parameters_from_dataset(X)

        ordering = range(n_visible)
        np.random.shuffle(ordering)

        trainer = Optimization.MomentumSGD(nade, nade.__getattribute__(loss_function))
        trainer.set_datasets([X])
        trainer.set_learning_rate(self.learning_rate)
        trainer.set_datapoints_as_columns(True)
        #trainer.add_controller(TrainingController.AdaptiveLearningRate(options.lr, 0, epochs=options.epochs))
        trainer.add_controller(TrainingController.MaxIterations(self.max_epochs))
        #if options.training_ll_stop < np.inf:
        #    trainer.add_controller(TrainingController.TrainingErrorStop(-options.training_ll_stop))  # Assumes that we're doing minimization so negative ll
        trainer.add_controller(TrainingController.ConfigurationSchedule("momentum", [(2, 0), (float('inf'), self.momentum)]))
        trainer.set_updates_per_epoch(1)
        trainer.set_minibatch_size(self.batch_size)
    #    trainer.set_weight_decay_rate(options.wd)
        trainer.add_controller(TrainingController.NaNBreaker())
        # Instrument the training
        trainer.add_instrumentation(Instrumentation.Instrumentation([console],
                                                                    Instrumentation.Function("training_loss", lambda ins: ins.get_training_loss())))
        #if not options.no_validation:
        #    trainer.add_instrumentation(Instrumentation.Instrumentation([console],
        #                                                                validation_loss_measurement))
        #    trainer.add_instrumentation(Instrumentation.Instrumentation([hdf5_backend],
        #                                                                validation_loss_measurement,
        #                                                                at_lowest=[Instrumentation.Parameters()]))
        trainer.add_instrumentation(Instrumentation.Instrumentation([console], Instrumentation.Configuration()))
        # trainer.add_instrumentation(Instrumentation.Instrumentation([hdf5_backend], Instrumentation.Parameters(), every = 10))
        trainer.add_instrumentation(Instrumentation.Instrumentation([console], Instrumentation.Timestamp()))
        # Train
        trainer.train()
        self.model = nade

    def get_log_likelihood(self, X):
        components = 10
        self.model.setup_n_orderings(n=components)
        ll = self.model.estimate_loglikelihood_for_dataset(Data.Dataset(X))
        return -ll.estimation, ll.se

    def get_nb_params(self):
        return sum(np.prod(param.get_value().shape) for param in self.model.parameters.values() if param.__class__.__name__=="TensorParameter")
    
    def sample(self, nb_samples):
        return self.model.sample(nb_samples).T

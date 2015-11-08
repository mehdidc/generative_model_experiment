import numpy as np
from lasagne import layers
from lasagne.nonlinearities import softmax


def build_discriminator_data(models, samples_per_model=100):
    data = []
    labels = []
    for i, model in enumerate(models):
        data_model = model.sample(samples_per_model)
        data.append(data_model)
        labels.extend([i] * samples_per_model)

    return np.concatenate(data, axis=0), labels


def build_discriminator_model_fully(dataset, num_models):
    l_in = layers.InputLayer((None, dataset.X.shape[1]))
    l_hid = layers.DenseLayer(l_in, num_units=500)
    l_out = layers.DenseLayer(l_hid, num_units=num_models, nonlinearity=softmax)
    return l_in, l_out


class Discriminator(Model):



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


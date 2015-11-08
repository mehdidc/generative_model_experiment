import pickle
import matplotlib as mpl
mpl.use('Agg')  # NOQA
import matplotlib.pyplot as plt

from datasets import datasets
from wrappers import Models
from hp_toolkit.hp import instantiate_random_model, instantiate_default_model  # NOQA

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

import numpy as np


from lasagnekit.misc.plot_weights import grid_plot

from wrappers.discriminator import Discriminator, build_discriminator_data


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_one(model_name, dataset_name,
              random=False,
              nb_samples=100, seed=1234,
              test_size=0.25,
              out_model=None,
              fast_test=False):
    seed = 1234
    np.random.seed(seed)

    if fast_test == "True":
        fast_test = True
    else:
        fast_test = False
    np.random.seed(seed)
    ds = datasets.get(dataset_name)
    if ds is None:
        raise Exception("cant find the dataset {}".format(dataset_name))
    X, imshape = ds()

    X = shuffle(X)
    X_train, X_test = train_test_split(X, test_size=test_size)

    model_class = Models[model_name]
    if fast_test is True:
        params = dict()
        params["max_epochs"] = 1
    else:
        params = dict()

    if random is True:
        model = instantiate_random_model(model_class, default_params=params)
    else:
        model = instantiate_default_model(model_class, default_params=params)

    X_train_full, X_test = train_test_split(X, test_size=0.25)
    X_train, X_valid = train_test_split(X_train_full, test_size=0.25)

    try:
        model.fit(X)
    except KeyboardInterrupt:
        print("Keyboard interrupt...")

    ll, ll_std = model.get_log_likelihood(X_test)
    nb_params = model.get_nb_params()

    print("log-likelihood : {0}+-{1}".format(ll, ll_std))
    print("nb of params : {0}".format(nb_params))

    sample_from_model(model, imshape, nb_samples=nb_samples)
    model.image_shape = imshape

    if out_model is not None:
        fd = open(out_model, "w")
        pickle.dump(model, fd)
        fd.close()
    return model


def compare():
    dataset = "mnist"
    model2 = train_one("Truth", dataset,
                       random=False)
    model1 = train_one("VA", dataset,
                       #fast_test="True",
                       random=False)

    models = [model1, model2]

    X_train, y_train = build_discriminator_data(models, samples_per_model=10000)
    X_train, y_train = shuffle(X_train, y_train)
    X_valid, y_valid = build_discriminator_data(models, samples_per_model=10000)
    X_valid, y_valid = shuffle(X_valid, y_valid)

    discriminator = Discriminator(kind="fully")
    discriminator.fit(X_train, y_train,
                      X_valid=X_valid, y_valid=y_valid)


def sample_from_pickled_model(model_filename, nb_samples=100,
                              image_filename="out.png"):

    model = pickle.load(model_filename)
    sample_from_model(model,
                      shape=model.image_shape,
                      nb_samples=nb_samples,
                      filename=image_filename)


def sample_from_model(model, shape, nb_samples=100, filename="out.png"):
    plt.clf()
    s = model.sample(nb_samples, only_means=True)
    s = s.reshape((s.shape[0], shape[0], shape[1]))
    grid_plot(s, imshow_options={"cmap": "gray"})
    plt.savefig(filename)

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
from distutils.util import strtobool
import json

import logging
import pprint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_one(model_name, dataset_name,
              random=False,
              nb_samples=100, seed=1234,
              test_size=0.25,
              out_model=None,
              custom_params=None,
              sample_filename=None,
              fast_test=False):
    seed = 1234
    np.random.seed(seed)

    if strtobool(str(fast_test)) is True:
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

    params = dict()
    if custom_params is not None:
        custom_params = json.load(open(custom_params))
        logging.info("Overriding hyper-params : {}")
        pprint.pprint(custom_params, indent=4, width=1)
        params.update(custom_params)
    if fast_test is True:
        params["max_epochs"] = 1
    else:
        pass
    if random is True:
        logging.info("Instantiate a model by randomly assigning hyper-params")
        model = instantiate_random_model(model_class, default_params=params)
    else:
        model = instantiate_default_model(model_class, default_params=params)

    test_ratio = 0.25
    logging.info("test_ratio : {}".format(test_ratio))
    X_train_full, X_test = train_test_split(X, test_size=test_ratio)
    X_train, X_valid = train_test_split(X_train_full, test_size=test_ratio)

    logging.info("Starting to train")
    try:
        model.fit(X)
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt...")

    ll, ll_std = model.get_log_likelihood(X_test)
    nb_params = model.get_nb_params()

    logging.info("log-likelihood : {0}+-{1}".format(ll, ll_std))
    logging.info("nb of params : {0}".format(nb_params))

    if sample_filename is not None:
        logging.info("Sampling from the model into {}".format(sample_filename))
        sample_from_model(model, imshape,
                          nb_samples=nb_samples, filename=sample_filename)
    model.image_shape = imshape

    if out_model is not None:
        logging.info("Pickling the model into {}".format(out_model))
        fd = open(out_model, "w")
        pickle.dump(model, fd, protocol=pickle.HIGHEST_PROTOCOL)
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


def compare_with_truth(model_filename, dataset_name,
                       discriminator_kind="fully",
                       samples_per_model=10000):
    samples_per_model = int(samples_per_model)
    truth = train_one("Truth", dataset_name,
                      random=False)
    logging.info("Loading the model from {}".format(model_filename))
    model = pickle.load(open(model_filename))

    models = [truth, model]

    logging.info("Building training and test data")
    X, y = build_discriminator_data(
        models,
        samples_per_model=samples_per_model * 2)
    X, y = shuffle(X, y)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.5)

    logging.info("Start training the discriminator")
    discriminator = Discriminator(kind=discriminator_kind)
    discriminator.fit(X_train, y_train,
                      X_valid=X_valid, y_valid=y_valid)


def sample_from_pickled_model(model_filename, nb_samples=100,
                              image_filename="out.png"):

    model = pickle.load(open(model_filename))
    model.image_shape = (28, 28)
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

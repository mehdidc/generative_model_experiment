import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append("./libs/rnade/buml")
from wrappers.dbn import DBN
from wrappers.va import VA
from wrappers.nade import NADE
from wrappers.gsn import GSN
from wrappers.adv import Adversarial
from wrappers.bernoulli import BernoulliMixture
from datasets import datasets
from lasagne.misc.plot_weights import grid_plot 
from lasagne.easy import get_2d_square_image_view
import gc
from sklearn.utils import shuffle
from hp_toolkit.hp import Param, parallelizer, minimize_fn_with_hyperopt, find_best_hp
from lightexperiments.light import Light
from sklearn.cross_validation import train_test_split
import os


def launch(X, im):
    light = Light()
    
    # init
    models = [VA, NADE, Adversarial]
    max_evaluations_hp = 20
    fast_test = True

    # prepare
    X = shuffle(X)
    if fast_test is True:
        max_evaluations_hp = 1
        default_params = dict(
                max_epochs=2
        )
        X = X[0:100]
    else:
        default_params = dict()

    eval_function = lambda model, X_v, y_v: model.get_log_likelihood(X_v)[0]
    X_train_full, X_test = train_test_split(X, test_size=0.25)
    X_train, X_valid = train_test_split(X_train_full, test_size=0.25)

    # show original data
    X_ =  X.reshape((X.shape[0], im[0], im[1]))
    X_ = X_[0:10]
    grid_plot(X_, imshow_options={"cmap": "gray"})
    plt.savefig("orig.png")
    plt.show()

    for model in models:
        print("model : {0}".format(model.__name__))

        # find best  hyper-parameters on validation set
        best_hp, best_score = find_best_hp(
                model,
                (minimize_fn_with_hyperopt),
                X_train,
                X_valid,
                None,
                None,
                max_evaluations=max_evaluations_hp,
                default_params=default_params,
                eval_function=eval_function
        )
        # then retrain the model on the full training set
        name = model.__name__
        best_hp.update(default_params)
        model_inst = model(**best_hp)
        model_inst.fit(X_train_full)

        # evaluate the model

        ll, ll_std = model_inst.get_log_likelihood(X_test)
        nb_params = model_inst.get_nb_params()

        light.set(name + "_best_hp", best_hp)
        light.set(name + "_best_valid_score", best_score)
        light.set(name + "_ll_test", ll)
        light.set(name + "_ll_test_std", ll_std)
        light.set(name + "_nb_params", nb_params)

        print("log-likelihood : {0}".format(ll))
        print("nb of params : {0}".format(nb_params))

        # sample from the model

        plt.clf()
        s = model_inst.sample(100)
        s =  s.reshape((s.shape[0], im[0], im[1]))
        grid_plot(s, imshow_options={"cmap": "gray"})
        plt.savefig("samples/{0}.png".format(name))
        gc.collect()

if __name__ == "__main__":
    light = Light()
    try:
        light.launch()
    except Exception:
        light_connected = False
    else:
        print("Connected to mongo")
        light_connected = True

    light.initials()
    light.tag("generative_model_experiment")


    ds = sys.argv[1] if len(sys.argv)==2 else "digits"
    light.set("dataset", ds)
    X, im = datasets.get(ds)()
    launch(X, im)
    light.endings()

    report_dir = "{0}/report_{1}".format(os.getcwd(), ds)
    if not os.path.exists(report_dir):
        os.mkdir(report_dir)
    
    # create report here

    if light_connected is True:
        light.store_experiment()
        light.close()
    else:
        import cPickle as pickle
        import datetime
        fd = open("report_{0}".format(datetime.datetime.now().isoformat()), "w")
        pickle.dump(light.cur_experiment, fd)

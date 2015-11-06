import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from wrappers import *

from datasets import datasets
from lasagnekit.misc.plot_weights import grid_plot
from lasagnekit.easy import get_2d_square_image_view
import gc
from hp_toolkit.hp import Param, parallelizer, minimize_fn_with_hyperopt, find_best_hp, find_all_hp
from lightexperiments.light import Light
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import os
import uuid


def launch(X, im):
    light = Light()

    # init
    seed = 1234
    models = [NADE]
    max_evaluations_hp = 100
    fast_test = False
    exp_id = uuid.uuid1()
    light.set("custom_id", exp_id)

    np.random.seed(seed)
    light.set("seed", seed)

    # prepare
    X = shuffle(X)
    if fast_test is True:
        max_evaluations_hp = 1
        default_params = dict(
            max_epochs=2
        )
        X = X[0:100]
    else:
        default_params = dict(batch_size=128)

    def log_likelihood(model, X_v, y_v):
        return model.get_log_likelihood(X_v)[0]
    eval_function = log_likelihood
    X_train_full, X_test = train_test_split(X, test_size=0.25)
    X_train, X_valid = train_test_split(X_train_full, test_size=0.25)

    # show original data
    X_ = X.reshape((X.shape[0], im[0], im[1]))
    X_ = X_[0:10]
    grid_plot(X_, imshow_options={"cmap": "gray"})
    plt.savefig("orig.png")
    plt.show()

    for model in models:
        print("model : {0}".format(model.__name__))

        # find best  hyper-parameters on validation set
        all_hp, all_scores = find_all_hp(
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
        for hp, score in zip(all_hp, all_scores):
            hp_run = dict(
                name=model.__name__,
                experiment_id=exp_id,
                tags=["model_hyperparameters"],
                dataset=light.cur_experiment["dataset"],
                score=score
            )
            hp_run.update(hp)
            light.store_experiment(hp_run)

        argmin = min(range(len(all_hp)), key=lambda i: all_scores[i])
        best_hp, best_score = all_hp[argmin], all_scores[argmin]

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
        s = s.reshape((s.shape[0], im[0], im[1]))
        grid_plot(s, imshow_options={"cmap": "gray"})
        plt.savefig("samples/{0}.png".format(name))
        gc.collect()

if __name__ == "__main__":
    light = Light()
    light.launch()
    light.initials()
    light.tag("generative_model_experiment")

    ds = sys.argv[1] if len(sys.argv) == 2 else "digits"
    light.set("dataset", ds)
    X, im = datasets.get(ds)()
    hp_runs = launch(X, im)
    light.endings()

    report_dir = "{0}/report_{1}".format(os.getcwd(), ds)
    if not os.path.exists(report_dir):
        os.mkdir(report_dir)

    # create report here

    light.store_experiment()
    light.close()

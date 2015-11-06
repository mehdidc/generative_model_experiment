from hp_toolkit.hp import Param, default_eval_functions, Model

import numpy as np

from scipy.stats import rv_discrete
from scipy.misc import logsumexp


def is_bounded(x):
    return (np.isnan(x).sum() + np.isinf(x).sum()) == 0


def is_log_bounded(x):
    return ((x <= 0).sum()) == 0


class BernoulliMixture(Model):

    params = dict(
        size=Param(initial=100, interval=[10, 100], type='int'),
        max_epochs=Param(initial=100, interval=[100], type='choice'),
        tol=Param(initial=1e-5, interval=[-5, -3], type='real', scale='log10')
    )

    def fit(self, X, X_valid=None):
        tol = 1e-5
        self.theta = np.random.uniform(size=(self.size, X.shape[1]))
        self.props = np.array([1.] * self.size) / self.size

        props, theta = self.props, self.theta

        ll_old = 0
        for i in range(self.max_epochs):
            # M-step: update values for parameters given current distribution

            #size mixture, 1, nb features
            theta_ = self.theta[:, np.newaxis, :]
            #1, nb examples, nb features
            X_ = X[np.newaxis, :, :]

            #size mixture, 1
            props_ = props[:, np.newaxis]

            #size mixture, 1
            log_props = np.log(props_)

            eps = 1e-15
            theta_[theta_<=eps] = eps
            theta_[theta_>=1 - eps] = 1 - eps
            assert is_log_bounded(theta_)
            assert is_log_bounded(1 - theta_)

            #size mixture, nb_examples
            log_L = ((X_ * np.log(theta_) +
                     (1 - X_) * (np.log(1 - theta_))).sum(axis=2))
            assert is_bounded(log_L)

            #log-likelihood
            #ll = np.log((L * props_).sum(axis=0)).mean()

            #nb_examples
            ll = logsumexp(log_L + log_props, axis=0)

            # 1
            ll = ll.mean()

            if np.abs(ll - ll_old) < tol:
                break
            ll_old = ll

            #size mixture, nb_examples
            L = np.exp(log_L)
            assert is_bounded(L)

            #size mixture, nb examples
            z = L * props_
            assert is_bounded(z)

            #size mixture, nb examples
            z /= z.sum(axis=0)
            assert is_bounded(z)
            # E-step

            #size mixture
            props_new = z.mean(axis=1)
            assert is_bounded(props_new)

            # size mixture, nb examples, 1
            z_ = z[:, :, np.newaxis]

            # size mixture, nb features
            theta_new = (z_ * X_).sum(axis=1)
            assert is_bounded(theta_new)
            theta_new /= (z_.sum(axis=1))
            assert is_bounded(theta_new)

            theta = theta_new
            props = props_new
            self.theta = theta
            self.props = props
            print("Iteration: {0}, ll : {1}".format(i, ll))

    def get_log_likelihood(self, X):
        theta_ = self.theta[:, np.newaxis, :]
        X_ = X[np.newaxis, :, :]
        props_ = self.props[:, np.newaxis]
        log_props = np.log(props_)
        log_L = ((X_ * np.log(theta_) +
                 (1 - X_) * (np.log(1 - theta_))).sum(axis=2))
        ll = logsumexp(log_L + log_props, axis=0)
        ll = ll.mean()
        return -ll, 0

    def sample(self, nb_samples, only_means=True):
        distrib = rv_discrete(values=(range(len(self.props)), self.props))
        m = distrib.rvs(size=nb_samples)
        p = self.theta[m, :]
        if only_means is True:
            return p
        else:
            samples = (np.random.uniform(size=p.shape) < p)
            return samples

    def get_nb_params(self):
        return np.prod(self.theta.shape) + np.prod(self.props.shape)

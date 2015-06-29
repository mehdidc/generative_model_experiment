from hp_toolkit.hp import Param, default_eval_functions, Model

import numpy as np

from scipy.stats import rv_discrete

class BernoulliMixture(Model):

    params = dict(
            size = Param(initial=10, interval=[10, 100], type='int'),
            max_epochs = Param(initial=100, interval=[100], type='choice')
    )
    
    def fit(self, X, X_valid=None):


        tol = 1e-5
        self.theta = np.random.uniform(size=(self.size, X.shape[1]))
        self.props = np.array([1.] * self.size) / self.size

        props, theta = self.props, self.theta

        ll_old = 0
        for i in range(self.max_epochs):
            # M-step: update values for parameters given current distribution
            theta_ = self.theta[:, np.newaxis, :]
            X_ = X[np.newaxis, :, :]
            props_ = props[:, np.newaxis]
            L = (theta_ ** X_ * (1 - theta_) ** (1 - X_)).prod(axis=2)
            
            #log-likelihood
            ll = np.log((L * props_).sum(axis=0)).mean()
            if np.abs(ll - ll_old) < tol:
                break
            ll_old = ll

            L = (L * props_) / ((L * props_).sum(axis=1))[:, np.newaxis]
            # E-step
            props_new = L.mean(axis=1)
            theta_new = ((L[:, :, np.newaxis] * X[np.newaxis, :, :]).sum(axis=1) / L.sum(axis=1)[:, np.newaxis])

            theta = theta_new
            props = props_new
            self.theta = theta
            self.props = props
            print("Iteration: {0}, ll : {1}".format(i, ll))


    def get_log_likelihood(self, X):
        X_ = X[np.newaxis, :, :]
        theta_ = self.theta[:, np.newaxis, :]
        props_ = self.props[:, np.newaxis]
        L = (theta_ ** X_ * (1 - theta_) ** (1 - X_)).prod(axis=2)
        #log-likelihood
        ll = np.log((L * props_).sum(axis=0)).mean()
        return -ll, 0

    def sample(self, nb_samples):
        distrib = rv_discrete(values=(range(len(self.props)), self.props))
        m = distrib.rvs(size=nb_samples)
        p = self.theta[m, :]
        samples = (np.random.uniform(size=p.shape) < p)
        return samples

    def get_nb_params(self):
        return np.prod(self.theta.shape) + np.prod(self.props.shape)

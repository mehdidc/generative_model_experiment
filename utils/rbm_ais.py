import numpy as np
from scipy.misc import logsumexp
from itertools import product


def rbm_partition_function(hidbiases, visbiases, vishid,
                           rng=np.random, nb_iterations=10,
                           gibbs_nb_iterations=1,
                           nb_runs=1):
    """
    Estimator of RBM partition function with Annealed Importance Sampling (AIS).
    More details about Annealed Importance Sampling in [1] and [2].


    Parameters
    ----------
    hidbiases : array-like, shape=[b] where b is the number of hidden units.
        the hidden bias vector

    visbiases : array-like, shape=[a]
        where a is the number of visible units.
        the visible bias vector

    vishid : array-like, shape=[a, b]
        where a is the number of visible units  and b is the number of
        hidden units.
        the weight matrix

    rng : RandomState, random number generator to use

    nb_iterations : integer, the number of intermediate distributions in
        Annealed Importance Sampling (AIS). The bigger the more
        accurate is the partition function approximation.

    gibbs_nb_iterations : integer, the number of gibbs iterations of
        gibbs sampling for each intermediate distribution to perform.
        the bigger the more accurate the partition function approximation.

    nb_runs : number of samples to take for Annealed Importance Sampling (AIS)
              if nb_runs > 1, log_Z_std can be computed, if nb_runs is 1 then
              log_Z_std will be 0.

    Returns
    -------

    log_Z, log_Z_std : tuple where log_Z is the log of the partition function
        approximation and log_Z_std is the log of the standard deviation
        of the partition function.

    References
    ----------

    .. [1] Salakhutdinov, Ruslan, and Iain Murray. "On the quantitative analysis
           of deep belief networks."
           Proceedings of the 25th international conference
           on Machine learning. ACM, 2008.

    .. [2] NEAL, Radford M. Annealed importance sampling.
           Statistics and Computing, 2001, vol. 11, no 2, p. 125-139.

    """
    log_z_ratios = []
    # do multiple runs of AIS and average over the runs
    for i in range(nb_runs):
        log_z_ratio = rbm_annealed_importance_sampling_one_run_(
            hidbiases, visbiases, vishid,
            rng=rng,
            nb_iterations=nb_iterations,
            gibbs_nb_iterations=gibbs_nb_iterations)
        log_z_ratios.append(log_z_ratio)

    log_z_ratios = np.array(log_z_ratios)
    # compute mean and std of the ratio of the runs
    log_z_ratios_mean = logsumexp(log_z_ratios) - np.log(nb_runs)
    log_z_ratios_squared_mean = logsumexp(2 * log_z_ratio) - np.log(nb_runs)
    log_z_ratios_variance = logsumexp([log_z_ratios_squared_mean,
                                       -2 * log_z_ratios_mean])
    log_z_ratios_std = log_z_ratios_variance * 0.5
    # compute mean and std of the partition function
    num_dims = vishid.shape[0]
    num_hids = vishid.shape[1]

    log_z_a = (num_dims + num_hids) * np.log(2)
    log_z_b = log_z_ratios_mean + log_z_a
    log_z_b_std = log_z_ratios_std - log_z_a
    return log_z_b, log_z_b_std


def rbm_annealed_importance_sampling_one_run_(hidbiases, visbiases, vishid,
                                              rng=np.random, nb_iterations=10,
                                              gibbs_nb_iterations=1):

    num_dims = vishid.shape[0]
    num_hids = vishid.shape[1]

    v = np.zeros((num_dims,))  # start with zero visible vector
    log_w = []
    for bb in np.linspace(0, 1, nb_iterations)[0:-1]:

        # sample v using the intermdiate distribution defined by bb
        for k in range(gibbs_nb_iterations):
            h = (np.dot(v, vishid) + hidbiases) * bb
            h = 1. / (1. + np.exp(-h))
            h = rng.uniform(size=h.shape) <= h
            v = (np.dot(h, vishid.T) + visbiases) * bb
            v = 1. / (1. + np.exp(-v))
            v = rng.uniform(size=v.shape) <= v

        # computes the unnormalized log density (free energy)
        # of the sampled input v
        log_energies = []
        for bb_ in (bb, bb + 1. / nb_iterations):
            h_energy = bb_ * (np.dot(v, vishid) + hidbiases)
            h_energy = np.concatenate((h_energy[:, None],
                                      np.zeros((num_hids, 1))), axis=1)
            log_energy = (logsumexp(h_energy, axis=1).sum() +
                          bb_ * np.dot(v, visbias) +
                          num_hids * np.log(2))
            log_energies.append(log_energy)
        log_w.append(log_energies[1] - log_energies[0])

    log_z_ratio = np.sum(log_w)
    return log_z_ratio


def rbm_partition_function_exact(hidbias, visbias, vishid):
    num_dims = vishid.shape[0]
    num_hids = vishid.shape[1]
    log_Z = []
    for units in product((0, 1), repeat=(num_dims + num_hids)):
        v = units[0:num_dims]
        h = units[num_dims:]
        energy = (np.dot(np.dot(v, vishid), h) +
                  np.dot(v, visbias) +
                  np.dot(h, hidbias))
        log_Z.append(energy)
    log_Z = logsumexp(log_Z)
    return log_Z, 0

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    hidbias = np.random.uniform(size=(3,))
    visbias = np.random.uniform(size=(2,))
    vishid = np.random.uniform(size=(2, 3))

    log_Z_exact, _ = rbm_partition_function_exact(
        hidbias,
        visbias,
        vishid)

    all_log_Z = []
    all_log_Z_std = []

    iterations = np.arange(1, 1000, 10)
    for nb_iterations in iterations:
        log_Z, log_Z_std = rbm_partition_function(
            hidbias, visbias, vishid,
            nb_iterations=nb_iterations)
        all_log_Z.append(log_Z)
        all_log_Z_std.append(log_Z_std)
    plt.plot(np.log(iterations),
             all_log_Z,
             color="red", label="AIS log(Z)")
    plt.xlabel("log(nb_iterations)")
    plt.ylabel("log(Z)")
    plt.axhline(y=log_Z_exact, label="exact log(Z)", color="blue")
    plt.legend(loc="best")
    plt.show()

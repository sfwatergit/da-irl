# coding=utf-8
from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import time

import numpy as np

from impl.activity_mdp import ActivityMDP
from impl.activity_rewards import ActivityLinearRewardFunction
from misc import logger
from src.core.irl_algorithm import BaseMaxEntIRLAlgorithm


class MaxEntIRL(BaseMaxEntIRLAlgorithm):
    def __init__(self, env,
                 expert_agent,
                 reward_prior=None,
                 policy_prior=None,
                 load=False,
                 filepath=None,
                 avi_tol=1e-2,
                 verbose=False):

        R = ActivityLinearRewardFunction(env)

        mdp = ActivityMDP(R, 0.95, env)
        BaseMaxEntIRLAlgorithm.__init__(self, mdp, expert_agent, reward_prior, policy_prior, verbose)
        self.tol = avi_tol
        self.load = load
        self.filepath = filepath

    def train(self, n_iter=1,
              minibatch_size=1,
              reg=0.01,
              cache_dir=None):

        mu_D = self.get_empirical_savf()

        self.reward = self.mdp.reward

        start_time = time.time()
        for i in range(n_iter):
            iter_data = self.expert_demos
            iter_data_idxs = np.arange(0, len(iter_data))
            np.random.shuffle(iter_data_idxs)
            n_iters = int(float(len(iter_data) / minibatch_size))
            for itr in range(n_iters):
                logger.record_tabular("Iteration", i)
                with logger.prefix("itr #%d:%d| " % (i, itr + 1)):
                    iter_start_time = time.time()
                    minibatch = np.random.choice(iter_data_idxs, minibatch_size, False)
                    self._current_batch = iter_data[minibatch]
                    polopt_start_time = time.time()

                    logger.log('Computing policy...'.format(itr, n_iters, i))
                    reward = self.reward.get_rewards()
                    pi = self.approximate_value_iteration(reward)

                    logger.log('Computed policy in {:,.2f} seconds'.format(time.time() - polopt_start_time))

                    svf_start_time = time.time()
                    logger.log('Computing state visitation frequency (svf)...'.format(itr, n_iters, i))
                    mu_exp = self.state_visitation_frequency(pi)
                    logger.log('Computed svf in {:,.2f} seconds'.format(time.time() - svf_start_time))
                    # mu_exp = self.get_expected_state_feature_counts(pi, svf)

                    D_sa = np.zeros((self.nS, self.nA), dtype=np.float32)
                    for s in self.mdp.S:
                        actions = self.mdp.env.states[s].available_actions
                        for a in actions:
                            D_sa[s, a] = pi[s, a] * mu_exp[s]

                    grad_r = mu_D - D_sa
                    grad_theta, l2_loss, grad_norm = self.reward.apply_grads(grad_r)
                    avg_feature_diff = np.mean(l2_loss)

                    self.feature_diff.append(grad_norm)

                    # Compute log-likelihood
                    log_lik = log_likelihood(pi, self._current_batch)
                    self.log_lik_hist.append(log_lik)

                    logger.record_tabular("Gradient (feature diff)", avg_feature_diff)
                    logger.record_tabular("Negative Log Likelihood", -log_lik)

                    logger.log("Optimizing reward...")

                    theta = self.reward.get_theta()
                    self.theta_hist.append(theta.T)

                    if self.VERBOSE:
                        print(theta)

                    # self.mdp.reward.update_reward()
                    logger.record_tabular('Time', time.time() - start_time)
                    logger.record_tabular('ItrTime', time.time() - iter_start_time)
                    logger.dump_tabular(with_prefix=False)


def log_likelihood(pi, examples):
    ll = 0
    for example in examples:
        for s, a in example[:-1]:
            if pi[s, a] != 0:
                ll += np.log(pi[s, a])
    return ll


def _expected_utility(mdp, s, a, value):
    """ The expected utility of performing `a` in `s`, using `value` """
    return np.sum([p * mdp.gamma * value[s1] for (p, s1) in mdp.T(s, a)])


def group_by_iter(n, iterable):
    row = tuple(next(iterable) for i in range(n))
    while row:
        yield row
        row = tuple(next(iterable) for i in range(n))

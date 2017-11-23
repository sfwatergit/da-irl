# coding=utf-8
from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import time

import numpy as np

from algos.irl_algorithm import BaseMaxEntIRLAlgorithm
from impl.activity_mdp import ActivityMDP
from impl.activity_rewards import ActivityLinearRewardFunction
from misc import logger


class MaxEntIRL(BaseMaxEntIRLAlgorithm):
    def __init__(self, env,
                 reward_prior=None,
                 policy_prior=None,
                 load=False,
                 filepath=None,
                 avi_tol=1e-2,
                 verbose=False):

        R = ActivityLinearRewardFunction(env)

        mdp = ActivityMDP(R, 0.95, env)

        self.tol = avi_tol
        self.load = load
        self.filepath = filepath
        BaseMaxEntIRLAlgorithm.__init__(self, mdp, reward_prior, policy_prior, verbose)

    def train(self, trajectories, n_iter=1, minibatch_size=1, reg=0.01, cache_dir=None):

        self.expert_demos = trajectories
        self._max_path_length = sum(len(path) for path in self.expert_demos)

        # Get expert's demonstrated (empirical) state visitation frequency (normalized)
        mu_D = self.get_empirical_savf()

        self._reward = self.mdp.reward

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

                    # Compute the policy using approximate (softmax) value iteration
                    polopt_start_time = time.time()
                    logger.log('Computing policy...'.format(itr, n_iters, i))
                    reward = self.reward.get_rewards()
                    pi = self.approximate_value_iteration(reward)
                    logger.log('Computed policy in {:,.2f} seconds'.format(time.time() - polopt_start_time))

                    # Compute the state visitation occupancy counts
                    svf_start_time = time.time()
                    logger.log('Computing state-action visitation frequency (svf)...'.format(itr, n_iters, i))
                    mu_exp = self.state_visitation_frequency(pi)
                    savf = self._compute_savf_from_svf(mu_exp, pi)
                    logger.log('Computed savf in {:,.2f} seconds'.format(time.time() - svf_start_time))

                    grad_r = -(mu_D - savf)
                    grad_theta, l2_loss, grad_norm = self.reward.apply_grads(grad_r)
                    avg_feature_diff = np.max(np.abs(grad_theta[0]))

                    self.feature_diff.append(grad_norm)

                    # Compute log-likelihood
                    log_lik = self._log_likelihood(pi, self._current_batch)
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

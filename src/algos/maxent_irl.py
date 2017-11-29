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
    def __init__(self, mdp,
                 policy_prior=None,
                 load=False,
                 filepath=None,
                 avi_tol=1e-2,
                 verbose=False):

        self.tol = avi_tol
        self.load = load
        self.filepath = filepath
        BaseMaxEntIRLAlgorithm.__init__(self, mdp, policy_prior, verbose)

    def train(self, trajectories, epochs=1, minibatch_size=1):

        self.expert_demos = trajectories
        self._max_path_length = sum(len(path) for path in self.expert_demos)

        # Get expert's demonstrated (empirical) state visitation frequency (normalized)
        mu_D = self.get_empirical_savf()

        self._reward = self.mdp.reward_function

        start_time = time.time()
        for epoch in range(epochs):
            iter_data = self.expert_demos
            iter_data_idxs = np.arange(0, len(iter_data))
            np.random.shuffle(iter_data_idxs)
            n_iters = int(float(len(iter_data) / minibatch_size))
            for itr in range(n_iters):
                logger.record_tabular("Iteration", epoch)
                with logger.prefix("itr #%d:%d| " % (epoch, itr + 1)):
                    iter_start_time = time.time()
                    minibatch = np.random.choice(iter_data_idxs, minibatch_size, False)
                    self._current_batch = iter_data[minibatch]

                    # Compute the policy using approximate (softmax) value iteration
                    polopt_start_time = time.time()
                    logger.log('Computing policy...'.format(itr, n_iters, epoch))
                    reward = self.reward.get_rewards()
                    self._policy = self.approximate_value_iteration(reward)
                    logger.log('Computed policy in {:,.2f} seconds'.format(time.time() - polopt_start_time))

                    # Compute the state visitation occupancy counts
                    svf_start_time = time.time()
                    logger.log('Computing state-action visitation frequency (svf)...'.format(itr, n_iters, epoch))
                    mu_exp = self.state_visitation_frequency()
                    savf = self._compute_savf_from_svf(mu_exp)
                    logger.log('Computed savf in {:,.2f} seconds'.format(time.time() - svf_start_time))

                    grad_r = -(mu_D - savf)
                    grad_theta, l2_loss, grad_norm = self.reward.apply_grads(grad_r)
                    avg_feature_diff = np.max(np.abs(grad_theta[0]))

                    self.feature_diff.append(grad_norm)

                    # Compute log-likelihood
                    log_lik = self._log_likelihood(self.expert_demos)
                    self.log_lik_hist.append(log_lik)

                    logger.record_tabular("Gradient (feature diff)", avg_feature_diff)
                    logger.record_tabular("Negative Log Likelihood", -log_lik)

                    logger.log("Optimizing reward...")

                    theta = self.reward.get_theta()
                    self.theta_hist.append(theta.T)

                    # self.mdp.reward.update_reward()
                    logger.record_tabular('Time', time.time() - start_time)
                    logger.record_tabular('ItrTime', time.time() - iter_start_time)
                    params = self.get_epoch_snapshot(epoch)
                    logger.save_itr_params(epoch, params)
                    logger.dump_tabular(with_prefix=False)

    def get_epoch_snapshot(self, epoch):
        return dict(epoch=epoch,
                    policy=self.policy,
                    theta=self.reward.get_theta(),
                    reward=self.reward.get_rewards())

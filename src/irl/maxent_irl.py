# coding=utf-8
from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import numpy as np

from impl.activity_mdp import ActivityMDP
from impl.activity_rewards import ActivityRewardFunction
from src.core.irl_algorithm import BaseMaxEntIRLAlgorithm, get_policy
from src.misc.math_utils import adam


class MaxEntIRL(BaseMaxEntIRLAlgorithm):
    def __init__(self, env, expert_agent, reward_prior=None, policy_prior=None, verbose=False, load=False,
                 filepath=None, tol=1e-2):

        R = ActivityRewardFunction(env)

        mdp = ActivityMDP(R, 0.95, env)
        BaseMaxEntIRLAlgorithm.__init__(self, mdp, expert_agent, reward_prior, policy_prior, verbose)
        self.tol = tol
        self.load = load
        self.filepath = filepath

    def learn_rewards(self, n_iter=1, learning_rate=0.08, minibatch_size=1, initial_theta=None, reg=0.01,
                      cache_dir=None):

        if initial_theta is None:
            theta_activity = np.random.normal(0.000, 1e-8, size=(1, len(self.mdp.reward.activity_features)))
            theta_travel = np.random.normal(0.0000, 1e-8, size=(1, len(self.mdp.reward.trip_features)))
            theta = np.concatenate((theta_activity, theta_travel), axis=1)
        else:
            theta = initial_theta

        self.mdp.reward.update_reward(theta)
        self.reward = self.mdp.reward
        config = None

        empirical_feature_counts = self.get_empirical_feature_counts()

        for i in range(n_iter):
            iter_data = self.expert_demos
            iter_data_idxs = np.arange(0, len(iter_data))
            np.random.shuffle(iter_data_idxs)
            n_iters = int(float(len(iter_data) / minibatch_size))

            print('Iteration: {} of {}'.format(i, n_iter))
            for iter in range(n_iters):

                minibatch = np.random.choice(iter_data_idxs, minibatch_size, False)
                self._current_batch = iter_data[minibatch]

                print('\tMinibatch: {} of {} (iter {})'.format(iter, n_iters, i))

                Q = self.approximate_value_iteration(self.reward.get_reward())

                pi = get_policy(Q)

                pi = pi.astype(np.float32)

                svf = self.state_visitation_frequency(pi)

                expected_state_feature_counts = self.get_expected_state_feature_counts(pi, svf)

                df_dtheta = empirical_feature_counts - expected_state_feature_counts

                avg_feature_diff = np.mean(df_dtheta)

                self.feature_diff.append(df_dtheta)
                self.theta_hist.append(theta.T)

                # Compute log-likelihood
                log_lik = log_likelihood(pi, self._current_batch)
                self.log_lik_hist.append(log_lik)

                if self.VERBOSE:
                    print("Gradient (feature diff): {}".format(avg_feature_diff))
                    print("Negative Log Likelihood: {}".format(-log_lik))

                theta, config = adam(theta, df_dtheta.T, config)

                if self.VERBOSE:
                    print(theta)

                self.mdp.reward.update_reward(theta)

    def get_expected_state_feature_counts(self, pi, svf):
        expected_state_feature_counts = np.zeros((self._dim_ss, 1))
        for sx in self.mdp.S:
            actions = self.mdp.env.states[sx].available_actions
            for axy in actions:
                expected_state_feature_counts += (self.mdp.reward.phi(sx, axy) * pi[sx, axy] * svf[sx])
        return expected_state_feature_counts


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

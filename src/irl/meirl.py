# coding=utf-8
from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import time

import numpy as np
from cytoolz import memoize

from src.core.agent import IRLAgent
from src.misc.math_utils import adam

INF = np.nan_to_num([1 * float("-inf")])


class BaseMaxEntIRLAgent(IRLAgent):
    """
    Base class for maximum entropy inverse reinforcement learning agents

    Attributes:
        demos: set of demonstrations used for IRL

    References:
      .. [1] Ziebart, B. D., & Maas, A. (2008). Maximum entropy inverse reinforcement learning.
        Twenty-Second Conference on Artificial Intelligence (AAAI), 1433–1438.
    """

    def __init__(self, mdp, data, verbose=False):
        IRLAgent.__init__(self, mdp, data)

        self.feature_diff = []
        self.theta_hist = []
        self.log_lik_hist = []
        self._MAX_ITER = 0
        self.VERBOSE = verbose

    def state_visitation_frequency(self, policy):
        """
        Args:
            policy: most recently computed policy

        Returns:

        """
        start_time = time.time()
        p_start_state = self.get_start_state_dist(self._current_batch)
        expected_svf = np.tile(p_start_state, (self._MAX_ITER, 1)).T

        for t in range(1, self._MAX_ITER):
            expected_svf[:, t] = 0
            for s_x in self.mdp.S:
                state = self.mdp.env.states[s_x]
                actions = state.available_actions
                for a_xy in actions:
                    action = self.mdp.env.actions[a_xy]
                    outcomes = self.mdp.T(state, action)
                    for o in outcomes:
                        s_z = o[1].state_id
                        p = o[0]
                        expected_svf[s_z, t] += p * (expected_svf[s_x, t - 1] *
                                                     policy[s_x, a_xy])
        print('Computed svf in {:,.2f} seconds'.format(time.time() - start_time))
        return np.sum(expected_svf, 1).astype(np.float32).reshape(self.nS, 1)

    def approximate_value_iteration(self, reward):
        """
        Computes maximum entropy policy given current reward function and horizon.

        Args:
            horizon: `int`. Finite time horizon for computing state frequencies.
            reward: `ndarray`. Current value of parametrized reward function.

        Returns:
            policy: `ndarray`  An S x A policy based on reward parameters.

        """
        start_time = time.time()

        nA = self.nA
        nS = self.nS

        V = np.nan_to_num(np.ones((nS, 1)) * float("-inf"))

        V_pot = V.copy()
        V_pot[self.mdp.env.terminals] = 0.0
        diff = np.ones(nS)
        Q = np.nan_to_num(np.ones((nS, nA)) * float("-inf"))
        t = 0

        while (diff > 1e-2).all():
            Vp = V_pot.copy()
            for s_x in reversed(self.mdp.S):
                state = self.mdp.env.states[s_x]
                actions = state.available_actions
                for a_xy in actions:
                    action = self.mdp.env.actions[a_xy]
                    outcomes = self.mdp.T(state, action)
                    Q[s_x, a_xy] = np.sum([o[0] * (V[o[1].state_id] + reward[s_x, a_xy]) for o in outcomes])
                    Vp[s_x] = softmax(Vp[s_x][0], Q[s_x, a_xy])
            diff = np.abs(V - Vp).max()
            V = Vp.copy()

            if t % 5 == 0:
                num_neg_infs = len(V[V < -1.e4])
                if self.VERBOSE:
                    print('t:{}, Delta V:{}, No. divergent states: {}'.format(t, diff, num_neg_infs))
            t += 1

        policy = self.compute_policy(Q, V)

        self._MAX_ITER = t
        if self.VERBOSE:
            print('Computed policy in {:,.2f} seconds'.format(time.time() - start_time))
        print(policy)
        return policy.astype(np.float32)

    def compute_policy(self, Q, V):
        # type: (np.ndarray, np.ndarray) -> np.ndarray
        """Compute policy from Q and V as pi
        Args:
            param Q (np.ndarray): State-action value function
            param V (np.ndarray):
        Return:
            (np.ndarray) stochastic policy
        """
        policy = np.zeros_like(Q)
        for s_x in self.mdp.S:
            state = self.mdp.env.states[s_x]
            actions = self.mdp.actions(state)
            for a_xy in actions:
                policy[s_x, a_xy] = np.exp(Q[s_x, a_xy] - V[s_x])
        return policy

    def get_start_state_dist(self, paths):
        path_states = np.array([path[0] for path in paths])
        start_states = np.array([state[0] for state in path_states])
        start_state_count = np.bincount(start_states.astype(int), minlength=self.nS)
        return start_state_count.astype(float) / len(start_states)


class MaxEntIRLAgent(BaseMaxEntIRLAgent):
    def __init__(self, mdp, data, verbose=False, load=False, filepath=None, tol=1e-2):
        BaseMaxEntIRLAgent.__init__(self, mdp, data, verbose)
        self.tol = tol
        self.load = load
        self.filepath = filepath

    def learn_rewards_and_weights(self, n_iter=1, learning_rate=0.08, minibatch_size=1, initial_theta=None, reg=0.01,
                                  cache_dir=None):

        if initial_theta is None:
            theta = np.random.normal(loc=0.0, scale=1, size=(1, self.mdp.reward.dim_ss))
        else:
            theta = initial_theta

        self.mdp.reward.update_reward(theta)
        self.reward = self.mdp.reward
        config = None
        t = 0

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

                pi = self.approximate_value_iteration(self.reward.get_reward())

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


@memoize
def softmax(x1, x2):
    """
    x1: float.
    x2: float.
    -> softmax(x1, x2)
    """

    max_x = max(x1, x2)
    min_x = min(x1, x2)
    return max_x + np.log(1 + np.exp(min_x - max_x))


def group_by_iter(n, iterable):
    row = tuple(next(iterable) for i in range(n))
    while row:
        yield row
        row = tuple(next(iterable) for i in range(n))

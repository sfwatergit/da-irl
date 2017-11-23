# coding=utf-8
import time
from abc import ABCMeta

import numpy as np
import six
from scipy.misc import logsumexp as sp_lse

from algos.base import IRLAlgorithm
from misc import logger
from util.math_utils import softmax

INF = np.nan_to_num([1 * float("-inf")])


class BaseMaxEntIRLAlgorithm(six.with_metaclass(ABCMeta, IRLAlgorithm)):
    """
       Base class for maximum entropy inverse reinforcement learning agents

       Attributes:
           demos: set of demonstrations used for IRL

       References:
         .. [1] Ziebart, B. D., & Maas, A. (2008). Maximum entropy inverse reinforcement learning.
           Twenty-Second Conference on Artificial Intelligence (AAAI), 1433–1438.
       """

    def __init__(self, mdp, reward_prior=None, policy_prior=None, verbose=False):
        """

        Args:
            mdp  (core.mdp.MDP): the mdp describing the agent's dynamics
            reward_prior (np.array):
            policy_prior (np.array):
        """
        self.mdp = mdp
        self.nS = len(self.mdp.S)
        self.nA = len(self.mdp.A)

        # what we want to learn
        self.policy = policy_prior
        self.reward = reward_prior

        self.expert_demos = None
        self._current_batch = None
        self._dim_ss = mdp.reward.dim_ss
        self._total_num_paths = None
        self._max_path_length = 0

        self.feature_diff = []
        self.theta_hist = []
        self.log_lik_hist = []
        self._MAX_ITER = 0
        self.VERBOSE = verbose

    def get_start_state_dist(self, paths):
        path_states = np.array([path[0] for path in paths])
        start_states = np.array([state[0] for state in path_states])
        start_state_count = np.bincount(start_states.astype(int), minlength=self.nS)
        return start_state_count.astype(float) / len(start_states)

    def get_empirical_svf(self):
        empirical_svf = np.zeros((self.nS, 1), dtype=np.float32)
        for example in self._current_batch:
            for s in example.states:
                empirical_svf[s] += 1
        return empirical_svf

    def get_empirical_avf(self):
        """
           Find the emprical action visitation frequency (avf)
           (normalized state-action occupancy counts) induced
           by expert demonstrations.

           """
        empirical_avf = np.zeros((self.nA, 1))
        for path in self._current_batch:
            for a in path.actions:
                empirical_avf[a] += 1
        return empirical_avf

    def get_empirical_savf(self):
        """
           Find the emprical state-action visitation frequency
           (normalized state-action occupancy counts) induced
           by expert demonstrations.

           """
        savf = np.zeros((self.nS, self.nA), dtype=np.float32)
        for trajectory in self.expert_demos:
            for state, action in trajectory:
                savf[state, action] += 1

        savf /= len(self.expert_demos)
        return savf

    def get_path_feature_counts(self, path):
        """
        Compute feature counts along a path
        Args:
            path:

        Returns:

        """
        return np.sum([self.reward.phi(s, a) for s, a in path], 0)

    def get_empirical_feature_counts(self):

        feature_expectations = np.zeros((self._dim_ss, 1), dtype=float)
        for path in self.expert_demos:
            feature_expectations += self.get_path_feature_counts(path)
            if len(path) > self._max_path_length:
                self._max_path_length = len(path)

        return feature_expectations / len(self.expert_demos)

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

        V = np.nan_to_num(np.ones((self.nS, 1)) * float("-inf"))

        V_pot = V.copy()
        V_pot[self.mdp.env.terminals] = 0.0

        diff = float("inf")
        t = 0

        while (diff > 1e-4):
            Vp = V_pot.copy()
            for a_xy in reversed(self.mdp.A):
                Vp = softmax(np.hstack(
                    [Vp,
                     reward[:, a_xy].reshape([-1, 1]) + 0.99 * self.mdp.transition_matrix[:, a_xy, :].dot(V)])).reshape(
                    -1, 1)

            diff = np.amax(abs(Vp - V))
            V = Vp.copy()

            if t % 5 == 0:
                num_neg_infs = len(V[V < -1.e4])
                if self.VERBOSE:
                    logger.log('t:{}, Delta V:{}, No. divergent states: {}'.format(t, diff, num_neg_infs))
            t += 1

        Q = reward + np.squeeze(self.mdp.transition_matrix.dot(V))
        policy = get_policy(Q)

        self._MAX_ITER = t

        return policy.astype(np.float32)

    def state_visitation_frequency(self, pi, ent_wt=1.0, discount=1.0):
        """
        Args:
            policy: most recently computed policy

        Returns:
            (np.ndarray) unnormalized state visitation distribution

        """
        state_visitation = np.expand_dims(self.get_start_state_dist(self._current_batch), axis=1)
        sa_visit_t = np.zeros(
            (self.mdp.transition_matrix.shape[0], self.mdp.transition_matrix.shape[1], self._MAX_ITER))

        for i in range(self._MAX_ITER):
            sa_visit = state_visitation * pi
            sa_visit_t[:, :, i] = sa_visit  # (discount**i) * sa_visit
            # sum-out (SA)S
            new_state_visitation = np.einsum('ij,ijk->k', sa_visit, self.mdp.transition_matrix)
            state_visitation = np.expand_dims(new_state_visitation, axis=1)
        return np.sum(np.sum(sa_visit_t, axis=2), axis=1, keepdims=True)

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

    def train(self, trajectories):
        raise NotImplementedError

    def get_action(self, s_t):
        """
        Act according to (stochastic) policy:

        .. math::
            a_t ~ p( A_t | S_t = s_t )

        Should sample from the current policy (or policy prior if policy is not yet initialized).

        :param s_t: state at which to sample policy
        :return: action
        """
        return self.policy[np.random.choice(self.mdp.actions(s_t), replace=False)]

    def get_reward(self):
        return self.mdp.reward

    def get_expected_state_feature_counts(self, pi, svf):
        expected_state_feature_counts = np.zeros((self._dim_ss, 1))
        for sx in self.mdp.S:
            actions = self.mdp.env.states[sx].available_actions
            for axy in actions:
                expected_state_feature_counts += (self.mdp.reward.phi(sx, axy) * pi[sx, axy] * svf[sx])
        return expected_state_feature_counts

    def optimize_reward(self):
        raise NotImplementedError


def logsumexp(q, alpha=1.0, axis=1):
    return alpha * sp_lse((1.0 / alpha) * q, axis=axis)


def get_policy(q_fn, ent_wt=1.0):
    """
    Return a policy by normalizing a Q-function
    """
    v_rew = softmax(q_fn)
    adv_rew = q_fn - np.expand_dims(v_rew, axis=1)
    pol_probs = np.exp((1.0 / ent_wt) * adv_rew)
    assert np.all(np.isclose(np.sum(pol_probs, axis=1), 1.0)), str(pol_probs)
    return pol_probs

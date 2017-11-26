# coding=utf-8
from abc import ABCMeta

import numpy as np
import six

from algos.base import IRLAlgorithm
from misc import logger
from util.math_utils import softmax

INF = np.nan_to_num([1 * float("-inf")])


class BaseMaxEntIRLAlgorithm(six.with_metaclass(ABCMeta, IRLAlgorithm)):
    """
       Base class for maximum entropy inverse reinforcement learning agents

        Attributes:
            expert_demos: set of demonstrations used for IRL

       References:
         .. [1] Ziebart, B. D., & Maas, A. (2008). Maximum entropy inverse reinforcement learning.
           Twenty-Second Conference on Artificial Intelligence (AAAI), 1433–1438.
       """

    def __init__(self, mdp, reward_prior=None, policy_prior=None, verbose=False):
        """

        Args:
            mdp  (core.mdp.MDP): the mdp describing the expert agent's dynamics
            reward_prior (np.array): previously-learned reward for expert agent
            policy_prior (np.array): previously-learned expert policy
        """
        self.mdp = mdp
        self.nS = len(self.mdp.S)
        self.nA = len(self.mdp.A)

        # what we want to learn
        self._policy = policy_prior
        self._reward = reward_prior

        self.expert_demos = None
        self._current_batch = None
        self._dim_ss = mdp.reward_function.dim_ss
        self._total_num_paths = None
        self._max_path_length = 0

        self.feature_diff = []
        self.theta_hist = []
        self.log_lik_hist = []
        self._MAX_ITER = 0
        self.VERBOSE = verbose

    @property
    def reward(self):
        return self.mdp.reward_function

    @property
    def policy(self):
        return self._policy

    def get_start_state_dist(self, paths):
        """
        Computes distribution of states over state space from provided trajectories.

        Args:
            paths:

        Returns:

        """
        path_states = np.array([path[0] for path in paths])
        start_states = np.array([state[0] for state in path_states])
        start_state_count = np.bincount(start_states.astype(int), minlength=self.nS)
        return start_state_count.astype(float) / len(start_states)

    def get_empirical_svf(self):
        """
           Find the empirical action visitation frequency (avf)
           (normalized action occupancy counts) induced by expert demonstrations.

        Returns:
            (np.ndarray): N X D array of

           """
        empirical_svf = np.zeros((self.nS, 1), dtype=np.float32)
        for example in self.expert_demos:
            for s in example.states:
                empirical_svf[s] += 1
        return empirical_svf

    def get_empirical_savf(self):
        """
           Find the empirical state-action visitation frequency
           (normalized state-action occupancy counts) induced
           by expert demonstrations.

        Returns:
            (np.ndarray): N x dA x dS array of state and action visitation counts
           """
        savf = np.zeros((self.nS, self.nA), dtype=np.float32)
        for trajectory in self.expert_demos:
            for state, action in trajectory:
                savf[state, action] += 1

        savf /= len(self.expert_demos)
        return savf

    def get_trajectory_feature_counts(self, path):
        """
        Compute feature counts for each state and action along a demonstration trajectory.

        Args:
            path (np.ndarray): N x (dS, dA) array representing the states and actions taken by agent
            for a single trajectory.

        Returns:
            (np.ndarray): 1 x dF array expressing the state-action features
        """
        return np.sum([self.reward.phi(s, a) for s, a in path], 0)

    def get_empirical_feature_counts(self):
        """
        Compute expert feature counts for provided trajectories

        Returns:
            (np.ndarray): N x dF array expressing the
        """
        feature_expectations = np.zeros((self._dim_ss, 1), dtype=float)
        for path in self.expert_demos:
            feature_expectations += self.get_trajectory_feature_counts(path)
            if len(path) > self._max_path_length:
                self._max_path_length = len(path)

        return feature_expectations / len(self.expert_demos)

    def approximate_value_iteration(self, reward, gamma=0.99, T=100):
        """
        Computes maximum entropy policy given current reward function and horizon
        via softmax value iteration.

        Args:
            gamma (float): Discount factor used to guarantee convergence of infinite horizon MDPs
            reward (np.ndarray): Current value of parametrized reward function.
            T (int): Finite time horizon for computing state frequencies.

        Returns:
            policy (np.ndarray):  An S x A policy based on reward parameters.

        """

        V = np.nan_to_num(np.ones((self.nS, 1)) * float("-inf"))

        V_pot = V.copy()
        V_pot[self.mdp.env.terminals] = 0.0

        diff = float("inf")
        t = 0

        while diff > 1e-4 or not t > T:
            Vp = V_pot.copy()
            for a_xy in reversed(self.mdp.A):
                Vp = softmax(np.hstack([Vp, reward[:, a_xy].reshape([-1, 1]) +
                                        gamma * self.mdp.transition_matrix[:, a_xy, :].dot(V)])).reshape(-1, 1)
            diff = np.amax(abs(Vp - V))
            V = Vp.copy()

            if t % 5 == 0:
                num_neg_infs = len(V[V < -1.e4])
                if self.VERBOSE:
                    logger.log('t:{}, Delta V:{}, No. divergent states: {}'.format(t, diff, num_neg_infs))
            t += 1

        # Compute Q from value function
        Q = reward + np.squeeze(self.mdp.transition_matrix.dot(V))
        pi = self._compute_policy(Q)

        return pi.astype(np.float32)

    def state_visitation_frequency(self, T=100):
        """
        Given the policy estimated at this iteration, computes the frequency with which a state is visited.

        Args:
            pi (np.ndarray): N x A policy
            T (int): Time horizon (optional)

        Returns:
            (np.ndarray): state visitation distribution

        """
        state_visitation = np.expand_dims(self.get_start_state_dist(self._current_batch), axis=1)
        sa_visit_t = np.zeros(
            (self.mdp.transition_matrix.shape[0], self.mdp.transition_matrix.shape[1], T))

        for i in range(int(T)):
            sa_visit = state_visitation * self.policy
            sa_visit_t[:, :, i] = sa_visit  # (discount**i) * sa_visit
            # sum-out (SA)S
            new_state_visitation = np.einsum('ij,ijk->k', sa_visit, self.mdp.transition_matrix)
            state_visitation = np.expand_dims(new_state_visitation, axis=1)
        return np.sum(np.sum(sa_visit_t, axis=2), axis=1, keepdims=True) / T

    def train(self, trajectories):
        raise NotImplementedError

    def get_action(self, s_t):
        """
        Act according to (stochastic) policy:

        .. math::
            a_t ~ p( A_t | S_t = s_t )

        Should sample from the current policy (or policy prior if policy is not yet initialized).

        Args:
            s_t (int): state at time t

        Returns:
            (int): index of action at time t
        """
        return self.policy[np.random.choice(self.mdp.actions(s_t), replace=False)]

    def _compute_savf_from_svf(self, mu_exp):
        """
        Compute estimated state-action visitation frequency from state visitation frequency

        Args:
            mu_exp (np.ndarray): normalized estimated expert state visitation counts
            pi (np.ndarray): expert policy

        Returns:
            np.ndarray: estimated state-action visitation frequency

        """
        D_sa = np.zeros((self.nS, self.nA), dtype=np.float32)
        for s in self.mdp.S:
            actions = self.mdp.env.states[s].available_actions
            for a in actions:
                D_sa[s, a] = self.policy[s, a] * mu_exp[s]
        return D_sa

    @staticmethod
    def _compute_policy(q_fn, ent_wt=1.0):
        """
        Return a policy by normalizing a Q-function
        """
        v_rew = softmax(q_fn)
        adv_rew = q_fn - np.expand_dims(v_rew, axis=1)
        pol_probs = np.exp((1.0 / ent_wt) * adv_rew)
        assert np.all(np.isclose(np.sum(pol_probs, axis=1), 1.0)), str(pol_probs)
        return pol_probs

    def _log_likelihood(self, demos):
        """
        Compute the log-likelihood of policy evaluated over trajectories

        Args:
            pi (np.ndarray): current expert policy
            demos (np.ndarray): expert demonstrations

        Returns:
            (float): log-likelihood value
        """
        return np.sum([np.sum([np.log(self.policy[s, a]) for s, a in example]) for example in demos])

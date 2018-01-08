# coding=utf-8
from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import time
from abc import ABCMeta

import numpy as np
import six

from src.algos.base import IRLAlgorithm
from src.misc import logger
from src.util.math_utils import softmax

INF = np.nan_to_num([1 * float("-inf")])


class MaxEntIRL(six.with_metaclass(ABCMeta, IRLAlgorithm)):
    """
       Base class for maximum entropy inverse reinforcement learning agents

        Attributes:
            expert_demos: set of demonstrations used for IRL

       References:
         .. [1] Ziebart, B. D., & Maas, A. (2008). Maximum entropy inverse
         reinforcement learning.
           Twenty-Second Conference on Artificial Intelligence (AAAI),
           1433–1438.
       """

    def __init__(self, mdp, avi_tol=1e-4, verbose=False, policy=None):
        """

        Args:
            avi_tol (float): convergence tolerance used to compute softmax
            value iteration
            mdp  (core.mdp.MDP): the mdp describing the expert agent's dynamics
        """
        self.avi_tol = avi_tol
        self.mdp = mdp
        self.nS = len(self.mdp.S)
        self.nA = len(self.mdp.A)

        self.expert_demos = None
        self._policy = policy
        self._reward = self.mdp.reward_function
        self._dim_ss = mdp.reward_function.dim_ss
        self._total_num_paths = None
        self._max_path_length = 0

        # Stats
        self.feature_diff = []
        self.theta_hist = []
        self.log_lik_hist = []
        self.VERBOSE = verbose

    @property
    def reward(self):
        return self._reward

    @property
    def policy(self):
        return self._policy

    def get_empirical_savf(self):
        """
           Find the empirical state-action visitation frequency distribution
           (normalized state-action occupancy counts) induced by expert
           demonstrations.

        Returns:
            (np.ndarray): N x dA x dS array of state and action visitation
            counts
           """
        savf = np.zeros((self.nS, self.nA), dtype=np.float32)
        for trajectory in self.expert_demos:
            for state, action in trajectory:
                savf[state, action] += 1

        savf /= len(self.expert_demos)
        return savf

    def approximate_value_iteration(self):
        """
        Computes maximum entropy policy given current reward function and
        horizon
        via softmax value iteration.

        Args:
            gamma (float): Discount factor used to guarantee convergence of
            infinite horizon MDPs
            reward (np.ndarray): Current value of parametrized reward function.
            T (int): Finite time horizon for computing state frequencies.

        Returns:
            policy (np.ndarray):  An S x A policy based on reward parameters.

        """

        reward = self.reward.get_rewards()

        Q = np.zeros([self.nS, self.nA], dtype=np.float32)
        diff = float("inf")

        while diff > 1e-4:
            V = softmax(Q)
            Qp = reward + self.mdp.gamma * self.mdp.transition_matrix.dot(V)
            diff = np.amax(abs(Q - Qp))
            Q = Qp

        pi = self._compute_policy(Q).astype(np.float32)

        return pi

    def state_visitation_frequency(self):
        """
        Given the policy estimated at this iteration, computes the frequency
        with which a state and action are visited.

        Returns:
            (np.ndarray): S x 1 state visitation distribution

        """
        state_visitation = np.expand_dims(self._start_state_dist(), axis=1)
        sa_visit_t = np.zeros(
            (self.mdp.transition_matrix.shape[0],
             self.mdp.transition_matrix.shape[1], self.mdp.env.horizon))

        for i in range(self.mdp.env.horizon):
            sa_visit = state_visitation * self.policy
            sa_visit_t[:, :, i] = sa_visit  # (discount**i) * sa_visit
            # sum-out (SA)S
            new_state_visitation = np.einsum(u'ij,ijk->k', sa_visit,
                                             self.mdp.transition_matrix)
            state_visitation = np.expand_dims(new_state_visitation, axis=1)
            #  Sum out over SA, but we need to normalize by horizon here.
            # This is different from Ziebart's algorithm, but if we do not,
            # we get a poor learning signal. This seems to be standard in
            # implementations.
        return np.sum(sa_visit_t, axis=2) / self.mdp.env.horizon

    def train(self, trajectories, num_iters=None, skip_policy=1):
        if num_iters is None:
            num_iters = self.mdp.env.config.irl_params.num_iters
        self.expert_demos = trajectories
        self._max_path_length = sum(len(path) for path in self.expert_demos)

        # Get expert's normalized, demonstrated (empirical) state visitation
        # frequency
        mu_D = self.get_empirical_savf()

        start_time = time.time()
        for itr in range(num_iters):
            # Begin each iteration of IRL here:
            logger.record_tabular("Iteration", itr)
            with logger.prefix("itr #%d " % itr):
                iter_start_time = time.time()

                # Compute the policy using approximate (softmax) value iteration
                logger.log("Computing policy...")
                polopt_start_time = time.time()
                if skip_policy > 0:  # (relevant if using actor-mimic algo)
                    skip_policy -= 1
                else:
                    self._policy = self.approximate_value_iteration()
                logger.log("Computed policy in {:,.2f} seconds".format(
                    time.time() - polopt_start_time))

                # Compute the normalized state visitation frequency
                svf_start_time = time.time()
                logger.log("Computing state visitation frequency...")
                svf = self.state_visitation_frequency()
                logger.log("Computed state visitations in {:,.2f} seconds"
                           .format(time.time() - svf_start_time))

                # Compute gradient signal and apply to model, backpropping
                # if deep IRL:
                logger.log("Optimizing reward...")
                grad_r = mu_D - svf
                grad_theta, l2_loss, grad_norm = self.reward.apply_grads(grad_r)

                # Compute stats
                avg_feature_diff = np.max(np.abs(grad_theta[0]))
                log_lik = self._log_likelihood()
                self.log_lik_hist.append(log_lik)

                # Record iteration stats
                logger.record_tabular("Gradient (feature diff)",
                                      avg_feature_diff)
                logger.record_tabular("Negative Log Likelihood", -log_lik)
                logger.log("Finished optimizing reward. Iteration complete.")
                logger.record_tabular("Time", time.time() - start_time)
                logger.record_tabular("ItrTime", time.time() - iter_start_time)
                logger.dump_tabular(with_prefix=False)

    def get_itr_snapshot(self, epoch):
        return dict(epoch=epoch,
                    policy=self.policy,
                    theta=self.reward.get_theta(),
                    reward=self.reward.get_rewards())

    def _start_state_dist(self):
        """
        Computes empirical distribution of states over state space from
        demonstration trajectories.

        Returns:
            (np.ndarray): Distribution of start states over state space.

        """
        path_states = np.array([path[0] for path in self.expert_demos])
        start_states = np.array([state[0] for state in path_states])
        start_state_count = np.bincount(start_states.astype(int),
                                        minlength=self.nS)
        return start_state_count.astype(float) / len(start_states)

    def _log_likelihood(self):
        """
        Compute the log-likelihood of policy evaluated over demonstrations.

        Returns:
            (float): Log-likelihood value.
        """
        return np.sum(
            [np.sum([np.log(self.policy[s, a]) for s, a in example]) for example
             in self.expert_demos])

    @staticmethod
    def _compute_policy(q_fn, ent_wt=1.0):
        """
        Return a stochastic policy (softmax distribution over actions given
        current state) by normalizing a Q-function.

        Args:
            q_fn (np.ndarray): A dim_S x dim_A state-action value
            (Q) function.
        """
        v_rew = softmax(q_fn)
        adv_rew = q_fn - np.expand_dims(v_rew, axis=1)
        pol_probs = np.exp((1.0 / ent_wt) * adv_rew)
        assert np.all(np.isclose(np.sum(pol_probs, axis=1), 1.0)), str(
            pol_probs)
        return pol_probs

# coding=utf-8
"""Contains the main Maximum Entropy IRL code for
"""
from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import time
from abc import ABCMeta

import numpy as np
import six

from src.algos.base import IRLAlgorithm
from src.algos.planning import SoftValueIteration
from src.misc import logger
from src.util.math_utils import softmax

INF = np.nan_to_num([1 * float("-inf")])


class MaxEntIRL(six.with_metaclass(ABCMeta, IRLAlgorithm, SoftValueIteration)):

    def __init__(self, mdp, discretized_horizon, avi_tol=1e-4, verbose=False,
                 policy=None):
        """Base class for maximum entropy inverse reinforcement learning agents.

        Subsumes linear and deep reward function variants.

        References:
            .. [1] Ziebart, B. D., & Maas, A. (2008). Maximum entropy inverse
                   reinforcement learning. Twenty-Second Conference on
                   Artificial Intelligence (AAAI), 1433–1438.
            .. [2] Wulfmeier, M., Ondruska, P., Posner, I., (2015).
                   Maximum Entropy Deep Inverse Reinforcement Learning. arXiv
                   1–9.

        Args:
            discretized_horizon (int): Used to limit number of iterations
            policy (nd.array): Policy prior. May be used to initialize
            forward algorithm.
            avi_tol (float): Convergence tolerance used to compute softmax
                             value iteration.
            mdp  (core.mdp.MDP): The mdp describing the expert agent's dynamics.

        Attr:
            expert_demos (nd.array): dim_S x dim_A set of demonstrations used
            for IRL.
            VERBOSE (bool): Whether to write verbose output to log (off by
            default).

        """
        self.discretized_horizon = discretized_horizon
        self.avi_tol = avi_tol
        self.mdp = mdp
        self.dim_S = len(self.mdp.S)
        self.dim_A = len(self.mdp.A)
        self.expert_demos = None
        self.VERBOSE = verbose

        self._policy = policy
        self._reward = self.mdp.reward_function
        self._dim_feature_space = mdp.reward_function.dim_phi
        self._total_num_paths = None
        self._max_path_length = 0

    @property
    def reward(self):
        return self._reward

    @property
    def policy(self):
        return self._policy

    def get_empirical_savf(self):
        """Find the empirical state-action visitation frequency distribution
           (normalized state-action occupancy counts) induced by expert
           demonstrations.

        Returns:
            (np.ndarray): N x dim_A x dim_S array of state and action visitation
            counts
           """
        savf = np.zeros((self.dim_S, self.dim_A), dtype=np.float32)
        for trajectory in self.expert_demos:
            for state, action in trajectory:
                savf[state, action] += 1

        savf /= len(self.expert_demos)
        return savf

    def state_visitation_frequency(self):
        """Given the policy estimated at this iteration, computes the frequency
        with which a state and action are visited.

        Returns:
            (np.ndarray): dim_S x 1 state visitation distribution

        """
        state_visitation = np.expand_dims(self._start_state_dist(), axis=1)
        sa_visit_t = np.zeros(
            (self.mdp.transition_matrix.shape[0],
             self.mdp.transition_matrix.shape[1], self.discretized_horizon))

        for i in range(self.discretized_horizon):
            sa_visit = state_visitation * self.policy
            sa_visit_t[:, :, i] = sa_visit  # (discount**i) * sa_visit
            # sum-out (SA)S
            new_state_visitation = np.einsum("ij,ijk->k", sa_visit,
                                             self.mdp.transition_matrix,
                                             optimize=False)

            state_visitation = np.expand_dims(new_state_visitation, axis=1)
            #  Sum out over SA, but we need to normalize by discretized horizon.

            #  This is different from Ziebart's algorithm, but if we do not,
            #  we get a poor learning signal. This seems to be standard in
            #  implementations.
        return np.sum(sa_visit_t, axis=2) / self.discretized_horizon

    def train(self, expert_demos, num_iters, policy_skip_iters=1):
        """Train the IRL algorithm using the provided demonstrations.

        Args:
            expert_demos (nd.array): Expert demonstrations used for training
            IRL.
            num_iters (int): Number of iterations for training.
            policy_skip_iters (int): Number of iterations to skip forward
            policy training.
        """
        if num_iters is None:
            num_iters = self.mdp.env.config.irl_params.num_iters
        self.expert_demos = expert_demos
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
                if policy_skip_iters > 0:  # (relevant if using actor-mimic
                    # algo)
                    policy_skip_iters -= 1
                else:
                    self._policy = self.solve(self.mdp)
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

                # Record iteration stats
                logger.record_tabular("Gradient (feature diff)",
                                      avg_feature_diff)
                logger.record_tabular("Negative Log Likelihood", -log_lik)
                logger.log("Finished optimizing reward. Iteration complete.")
                logger.record_tabular("Time", time.time() - start_time)
                logger.record_tabular("ItrTime", time.time() - iter_start_time)
                logger.dump_tabular(with_prefix=False)

    def get_itr_snapshot(self, epoch):
        """Provide the current (epoch) snapshot of the learned quantities.

        Args:
            epoch (int): the epoch (just a label) for which to return the
            snapshot data.

        Returns:
            dict[str,obj]: A dictionary containing the current reward
            function parameter values,
                           the value of the rewards, and the policy.

        """
        return dict(epoch=epoch,
                    policy=self.policy,
                    theta=self.reward.get_theta(),
                    reward=self.reward.get_rewards())

    def _start_state_dist(self):
        """Computes empirical distribution of states over state space from
        demonstration trajectories.

        Returns:
            (np.ndarray): Distribution of start states over state space.

        """
        path_states = np.array([path[0] for path in self.expert_demos])
        start_states = np.array([state[0] for state in path_states])
        start_state_count = np.bincount(start_states.astype(int),
                                        minlength=self.dim_S)
        return start_state_count.astype(float) / len(start_states)

    def _log_likelihood(self):
        """Compute the log-likelihood of policy evaluated over demonstrations.

        Returns:
            (float): Log-likelihood value.
        """
        return np.sum(
            [np.sum([np.log(self.policy[s, a]) for s, a in example]) for example
             in self.expert_demos])



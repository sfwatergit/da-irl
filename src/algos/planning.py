from abc import ABCMeta

import six
import numpy as np

from src.util.math_utils import softmax


class Planner(six.with_metaclass(ABCMeta)):

    def solve(self, mdp, V_init=None, pi_init=None):
        raise NotImplementedError('Abstract method')

    @staticmethod
    def _compute_policy(q_fn, ent_wt=1.0):
        """Return a stochastic policy (softmax distribution over actions given
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


class SoftValueIteration(Planner):

    def solve(self, mdp, V_init=None, pi_init=None):
        """Computes maximum entropy policy given current reward function and
        discretized_horizon via softmax value iteration.

        Returns:
            policy (np.ndarray):  An dim_S x dim_A policy based on reward
            parameters.

        """

        reward = mdp.reward_function.get_rewards()

        Q = np.zeros([len(mdp.S), len(mdp.A)], dtype=np.float32)
        diff = float("inf")

        while diff > 1e-4:
            V = softmax(Q)
            Qp = reward + mdp.gamma * mdp.transition_matrix.dot(V)
            diff = np.amax(abs(Q - Qp))
            Q = Qp

        pi = self._compute_policy(Q).astype(np.float32)

        return pi



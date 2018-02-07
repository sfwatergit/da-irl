from abc import ABCMeta

import numpy as np
import six
import tensorflow as tf

from src.util.math_utils import softmax
from src.util.tf_utils import get_session


class Planner(six.with_metaclass(ABCMeta)):

    def __init__(self, mdp):
        """Plans using value iteration.

        Args:
            mdp (src.impl.activity_mdp.ATPMDP): Implementation of Activity
            Travel Planning as markov decision process.
        """
        self.mdp = mdp

    def solve(self, V_init=None, pi_init=None):
        raise NotImplementedError('Abstract method')

    @staticmethod
    def _compute_policy(q, ent_wt=1.0):
        """Return a stochastic policy (softmax distribution over actions given
        current state) by normalizing a Q-function.

        Args:
            ent_wt (float): Boltzman temperature.
            q (np.ndarray): A dim_S x dim_A state-action value
            (Q) function.
        """
        v = softmax(q)
        advantage = q - np.expand_dims(v, axis=1)
        policy = np.exp((1.0 / ent_wt) * advantage)

        assert np.all(np.isclose(np.sum(policy, axis=1), 1.0)), str(policy)

        return policy

    def _policy_evaluation(self, policy, max_iter=200, epsilon=1e-05):
        """Compute the value of a policy

        Perform Bellman backups to find the value of a policy for all states
        in the MDP.

        Args:
            mdp (src.core.mdp.MDP): The markov decision process.
            policy (np.ndarray): A dim_S x dim_A pre-computed policy to follow.
            max_iter (int): Maximum number of iterations for which to run
            policy evaluation (if convergence has not yet been achieved).
            epsilon (float): The convergence tolerance.

        Returns:
            (np.ndarray): The value function.
        """
        done = False
        iteration = 0
        v = np.zeros(len(self.mdp.S))
        while iteration < max_iter and not done:
            v_prev = np.array(v)
            delta = 0
            for s in self.mdp.S:
                v[s] = self.mdp.R(s, None) + self.mdp.gamma * \
                       np.sum([p * v[sp] for (p, sp) in
                               self.mdp.T(s, policy[s])])
                delta = max(delta, np.fabs(v[s] - v_prev[s]))
            if delta < epsilon:
                done = True
            iteration += 1
        return v

    def _expected_utility(self, a, s, v):
        """The expected utility of performing `a` in `s` using `v` """
        return np.sum([p * self.mdp.gamma * v[sp] for (p, sp) in self.mdp.T(s,
                                                                            a)])


class TFValueIteration(Planner):

    def __init__(self, mdp):
        super(TFValueIteration, self).__init__(mdp)
        self.dim_A = len(self.mdp.A)
        self.dim_S = len(self.mdp.S)
        self.dim_phi = self.mdp.reward_function.dim_phi
        self.is_deterministic = self.mdp.is_deterministic

        if self.is_deterministic:
            p_a_shape = [self.dim_S, self.dim_A, self.dim_S]
            p_a_dtype = tf.int32
        else:
            p_a_shape = [self.dim_S, self.dim_A, self.dim_S]
            p_a_dtype = tf.float32

        self.P_a = tf.placeholder(p_a_dtype, shape=p_a_shape)
        self.reduce_max = tf.reduce_max
        self.reduce_max_sparse = tf.reduce_max
        self.reduce_sum = tf.reduce_sum
        self.reduce_sum_sparse = tf.reduce_sum
        self.sparse_transpose = tf.transpose
        self.gamma = tf.placeholder(tf.float32, name="gamma")
        self.epsilon = tf.placeholder(tf.float32, name="epsilon")

        self.sess = get_session()

    def solve(self, V_init=None, pi_init=None):

        reward = self.mdp.reward_function.get_rewards().sum(1)
        is_deterministic = self.is_deterministic

        def vi_step(values):
            if is_deterministic:
                new_value = tf.gather(reward,
                                      self.P_a) + \
                            self.gamma * tf.gather(values, self.P_a)
            else:
                new_value = self.reduce_sum_sparse(
                    self.P_a * (reward + self.gamma * values), axis=2)

            return new_value

        def body(i, c, t):
            old_values = t.read(i)
            new_values = vi_step(old_values)
            new_values = self.reduce_max(new_values, axis=1)
            t = t.write(i + 1, new_values)

            c = tf.reduce_max(tf.abs(new_values - old_values)) > self.epsilon
            c.set_shape(())

            return i + 1, c, t

        def condition(i, c, t):
            return c

        t = tf.TensorArray(dtype=tf.float32, size=350, clear_after_read=True)
        t = t.write(0, tf.constant(0, dtype=tf.float32, shape=(self.dim_S,)))
        i, _, values = tf.while_loop(condition, body, [0, True, t],
                                     parallel_iterations=1, back_prop=False,
                                     name='VI_loop')
        values = values.read(i)
        new_values = vi_step(values)

        if self.is_deterministic:
            policy = tf.argmax(new_values, axis=1)
        else:
            policy = tf.nn.softmax(new_values)

        return self.sess.run([policy, values], feed_dict={self.P_a:
                                                              self.mdp.transition_matrix,
                                                          self.gamma:
                                                              self.mdp.gamma,
                                                          self.epsilon: 1E-2})


class SoftValueIteration(Planner):

    def solve(self, Q_init=None, pi_init=None):
        """Computes maximum entropy policy given current reward function and
        discretized_horizon via softmax value iteration.

        Returns:
            policy (np.ndarray):  An dim_S x dim_A policy based on reward
            parameters.

        """
        reward = self.mdp.reward_function.get_rewards()
        if Q_init is None:
            Q = np.zeros([len(self.mdp.S), len(self.mdp.A)], dtype=np.float32)
        else:
            Q = Q_init
        diff = float("inf")

        while diff > 1e-4:
            V = softmax(Q)
            Qp = reward + self.mdp.gamma * self.mdp.transition_matrix.dot(V)
            diff = np.amax(abs(Q - Qp))
            Q = Qp

        pi = self._compute_policy(Q).astype(np.float32)

        return pi, Q

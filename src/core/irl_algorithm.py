# coding=utf-8
import time
from abc import ABCMeta

import numpy as np
import six

from core.base import IRLAlgorithm
from misc.math_utils import softmax
from scipy.misc import logsumexp as sp_lse

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

    def __init__(self, mdp, expert_agent, reward_prior=None, policy_prior=None, verbose=False):
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

        self.expert_demos = expert_agent.trajectories
        self._current_batch = None
        self._dim_ss = mdp.reward.dim_ss
        self._total_num_paths = sum(len(path) for path in self.expert_demos)
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

        savf /= len(self._current_batch)
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

    def state_visitation_frequency(self, policy):
        """
        Args:
            policy: most recently computed policy

        Returns:
            (np.ndarray) unnormalized state visitation distribution

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

    def learn_rewards(self):
        pass

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


def q_iteration(transition_matrix, reward_matrix, K=50, gamma=0.99, ent_wt=0.1, warmstart_q=None, policy=None):
    """
    Perform tabular soft Q-iteration

    If policy is given, this computes Q_pi rather than Q_star
    """

    q_fn = warmstart_q

    t_matrix = transition_matrix
    for k in range(K):
        if policy is None:
            v_fn = logsumexp(q_fn, alpha=ent_wt)
        else:
            v_fn = np.sum((q_fn - np.log(policy))*policy, axis=1)
        new_q = reward_matrix + gamma*t_matrix.dot(v_fn)
        q_fn = new_q
    return q_fn

def logsumexp(q, alpha=1.0, axis=1):
    return alpha*sp_lse((1.0/alpha)*q, axis=axis)


def get_policy(q_fn, ent_wt=1.0):
    """
    Return a policy by normalizing a Q-function
    """
    v_rew = logsumexp(q_fn, alpha=ent_wt)
    adv_rew = q_fn - np.expand_dims(v_rew, axis=1)
    pol_probs = np.exp((1.0/ent_wt)*adv_rew)
    assert np.all(np.isclose(np.sum(pol_probs, axis=1), 1.0)), str(pol_probs)
    return pol_probs


def compute_visitation(p, t_matrix, q_fn, ent_wt=1.0, T=50, discount=1.0):
    pol_probs = get_policy(q_fn, ent_wt=ent_wt)

    state_visitation = np.expand_dims(p, axis=1)
    sa_visit_t = np.zeros((t_matrix.shape[0], t_matrix.shape[1], T))

    for i in range(T):
        sa_visit = state_visitation * pol_probs
        sa_visit_t[:,:,i] = sa_visit #(discount**i) * sa_visit
        # sum-out (SA)S
        new_state_visitation = np.einsum('ij,ijk->k', sa_visit, t_matrix)
        state_visitation = np.expand_dims(new_state_visitation, axis=1)
    return np.sum(sa_visit_t, axis=2) / float(T)
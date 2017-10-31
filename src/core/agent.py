import numpy as np


class IRLAgent(object):
    def __init__(self, mdp,
                 expert_demos=None, reward_prior=None, policy_prior=None):

        self.mdp = mdp
        self.nS = len(self.mdp.S)
        self.nA = len(self.mdp.A)

        # what we want to learn
        self.policy = policy_prior
        self.reward = reward_prior

        self.expert_demos = expert_demos
        self._current_batch = None
        self._dim_ss = mdp.reward.dim_ss
        self._total_num_paths = sum(len(path) for path in self.expert_demos)
        self._max_path_length = 0

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
        return np.sum([self.reward.phi(s, a) for s, a in path], 0)

    def get_empirical_feature_counts(self):

        feature_expectations = np.zeros((self._dim_ss, 1), dtype=float)
        for path in self.expert_demos:
            feature_expectations += self.get_path_feature_counts(path)
            if len(path) > self._max_path_length:
                self._max_path_length = len(path)

        return feature_expectations / len(self.expert_demos)

    def learn_rewards_and_weights(self):
        pass

    def get_action(self, s):
        """
        Act according to policy:

        .. math::
            a_t ~ p( A_t | S_t = s )

        Should sample from the current policy (or policy prior if policy is not yet initialized).

        :param s: state at which to sample policy
        :return: action
        """
        return self.mdp.get_available_actions(s)

    def get_reward(self):
        return self.mdp.reward

import numpy as np


class AbstractAgent:
    def __init__(self):
        # Time and reward
        self.reward = 0
        self.epoch = 0
        self.episode = 0
        self.reward_hist = []

    def get_action(self, s):
        """
        Return an action according to the agent's strategy, the
        internal and external state.
        """
        raise NotImplementedError

    def make_action(self):
        """
        Execute the action that will result change of the environments and/or
        the internal state of the agent.
        """
        raise NotImplementedError

    def get_reward(self):
        """
        The total reward that is resulted.
        """
        raise NotImplementedError

    def increase_epoch(self):
        """
        Advance in time within an episode.
        """
        self.epoch += 1

    def increase_episode(self):
        """
        Advance in episodes.
        """
        self.episode += 1

    def reset(self):
        self.reward = 0
        self.epoch = 0
        self.increase_episode()


class RLAgent(AbstractAgent):
    def __init__(self, mdp, gamma, max_episodes, max_epoch_allowed):
        AbstractAgent.__init__(self)

        # The mdp problem definition
        self.mdp = mdp
        self._gamma = gamma

        # Training related
        self.max_episodes = max_episodes
        self.max_epoch_allowed = max_epoch_allowed

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        """ MDP Discount factor """
        if 0.0 > value >= 1.0:
            raise ValueError('MDP `discount` must be in [0, 1)')
        self._gamma = value

    def reset(self):
        AbstractAgent.reset(self)

    def get_action(self, s):
        raise NotImplementedError

    def get_reward(self, s, a):
        return self.mdp.R(s, a)

    def train(self):
        raise NotImplementedError


class IRLAgent(AbstractAgent):
    def __init__(self, mdp, paths=None):
        AbstractAgent.__init__(self)
        self.mdp = mdp
        self.nS = len(self.mdp.S)
        self.nA = len(self.mdp.A)
        self.demos = paths
        self._current_examples = None
        self._dim_ss = mdp.reward.dim_ss
        self._max_path_length = 0

    def get_empirical_svf(self):
        empirical_svf = np.zeros((self.nS, 1), dtype=np.float32)
        for path in self._current_examples:
            for s in path.states:
                empirical_svf[s] += 1
        return empirical_svf

    def get_emprical_avf(self, paths):
        empirical_avf = np.zeros((self.nA, 1))
        for path in paths:
            for a in path.actions:
                empirical_avf[a] += 1
        return empirical_avf

    def get_path_feature_counts(self, path):
        return np.sum([self.mdp.reward.phi(s, a) for s, a in path], 0)

    def get_empirical_feature_counts(self):

        feature_expectations = np.zeros((self._dim_ss, 1), dtype=float)
        for path in self.demos:
            feature_expectations += self.get_path_feature_counts(path)
            if len(path) > self._max_path_length:
                self._max_path_length = len(path)

        return feature_expectations / len(self.demos)

    def get_action(self, s):
        # TODO: should we call this get available actions (returns tuple of (p(a),a))?
        # Constrain \sum_a{p(a)} = 1
        return self.mdp.get_available_actions(s)

    def get_reward(self):
        return self.mdp.reward

    def make_action(self):
        # TODO: I don't understand the semantics of this
        pass

from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import gym
import numpy as np
from gym.spaces import MultiDiscrete
from gym.spaces.discrete import Discrete

from src.util.math_utils import to_onehot


class ActivityEnv(gym.Env):
    def __init__(self):
        """Activity-Travel environment representing the planning domain with
        which activity-travel planning agent interact.

        Agents observe

        It is expected that the environment is initialized per a configurable
        ``EnvBuilder``. All non-derivative properties are defined per the
        functionality of the builder object.

        """
        super(ActivityEnv, self).__init__()
        self._horizon = None
        self.segment_minutes = None
        # self.home_activity = self.config.home_activity.symbol
        self.home_start_state = None
        self.home_goal_states = []

        self.nA = None
        self.nS = None

        self.__action_space = None
        self.actions = None
        self.states = {}
        self.terminals = []
        self.g = {}

        self.__observation_space = None

        self._transition_matrix = None
        self._reward_function = None

    @property
    def horizon(self):
        return self._horizon

    @horizon.setter
    def horizon(self, horizon):
        self._horizon = horizon

    @property
    def G(self):
        return self.g

    @property
    def action_space(self):
        if self.__action_space is None:
            self.__action_space = Discrete(self.nA)
        return self.__action_space

    @property
    def observation_space(self):
        if self.__observation_space is None:
            self.__observation_space = MultiDiscrete([[0, self.nA], [0,
                                                                    self.nS]])
        return self.__observation_space

    @property
    def transition_matrix(self):
        return self._transition_matrix

    @transition_matrix.setter
    def transition_matrix(self, transition_matrix):
        self._transition_matrix = transition_matrix

    @property
    def reward_function(self):
        return self._reward_function

    @reward_function.setter
    def reward_function(self, reward_function):
        self._reward_function = reward_function

    def _reset(self):
        self.state = self.home_start_state
        return self.state.state_id

    def _step(self, action):
        """

        Args:
            action:

        Returns:

        """
        s = self.state.state_id
        action = self.actions[action]
        a = action.action_id
        ns = np.flatnonzero(self.transition_matrix[s, a])
        if len(ns) == 0:
            ns = s
        else:
            ns = ns[0]

        done = False
        if s in self.terminals:
            done = True

        self.state = self.states[ns]

        r = self._reward(self.states[ns], action)

        return ns, r, done, {}

    def _reward(self, state, action):
        """
        r: S,A -> \mathbb{R}
        Returns:

        """
        return self.reward_function.get_rewards()[
            state.state_id, action.action_id]

    def state_to_obs(self, state):
        if isinstance(state, int):
            return to_onehot(state, self.nS)
        else:
            return to_onehot(state.state_id, self.nS)

    def get_home_action_id(self):
        assert self.actions is not None
        for id, act in self.actions.items():
            if self.home_activity == act.succ_ix:
                return id

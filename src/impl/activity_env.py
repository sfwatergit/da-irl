from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import gym
import numpy as np
from gym.spaces.discrete import Discrete

from src.util.math_utils import to_onehot
from src.util.misc_utils import reverse_action_map


class ActivityEnv(gym.Env):
    def __init__(self):
        """Activity-Travel environment representing the planning domain with
        which activity-travel planning agent interact.

        This environment (:py:class:`~gym.Env`) represents a daily
        activity-travel pattern domain.

        During episodes, agents take actions
        (:py:class:`da-irl.src.impl.mdp.ATPAction) by calling the step method on
        the current Env. Actions represent a choice of next activity or trip and
        the time  to spend at the current activity). The agent then observes the
        state (:py:class:`da-irl.src.impl.mdp.ATPState) of the environment
        change in  response to its action as well as a reward immediately upon
        registering his or her choice. That is,  the agent receives an update in
        the form of next_state, reward.

        Once the agent has reached its home state and no longer wishes to make
        trips during the day, the agent then transitions to its home_goal_state,
        an absorbing state in the Markov chain describing the daily activity
        pattern.

        It is expected that the environment is initialized per a configurable
        ``EnvBuilder``. All non-derivative properties are defined per the
        functionality of the builder object.

        """
        super(ActivityEnv, self).__init__()
        self._horizon = None
        self.interval_length = None
        self.home_activity = None
        self.home_start_state = None
        self.home_goal_states = []
        self.terminals = []

        self.mdps = {}

        self.dim_A = None
        self.dim_S = None

        self.__action_space = None
        self._actions = {}
        self._reverse_action_map = None

        self._states = {}

        self.g = None

        self.__observation_space = None

        self._transition_matrix = None
        self._reward_function = None

    @property
    def states(self):
        return self._states

    def add_states(self, states):
        self._states.update(states)

    @property
    def actions(self):
        return self._actions

    @property
    def reverse_action_map(self):
        if self._reverse_action_map is None:
            self._reverse_action_map = reverse_action_map(self.actions)
        return self._reverse_action_map

    def add_actions(self, actions):
        self._actions.update(actions)

    @property
    def G(self):
        return self.g

    def update_G(self, state_graph):
        self.g = state_graph

    @property
    def action_space(self):
        if self.__action_space is None:
            self.__action_space = Discrete(self.dim_A)
        return self.__action_space

    @property
    def observation_space(self):
        if self.__observation_space is None:
            self.__observation_space = Discrete(self.dim_S)
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
        if r is None:
            r = 0

        return ns, r, done, {}

    def _reward(self, state, action):
        """
        r: S,A -> \mathbb{R}
        Returns: reward for state and action

        """
        return self.reward_function(state, action)

    def state_to_obs(self, state):
        if isinstance(state, int):
            return to_onehot(state, self.dim_S)
        else:
            return to_onehot(state.state_id, self.dim_S)

    def get_home_action_id(self):
        assert self.actions is not None
        for id, act in self.actions.items():
            if self.home_activity == act.next_state_symbol:
                return id

    def available_actions(self, state):
        return [self.states[next_state].symbol for next_state in
                self.G.successors(state.state_id)]

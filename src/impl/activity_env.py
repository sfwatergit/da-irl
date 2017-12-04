from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import gym
import multiprocessing
import networkx as nx
import numpy as np
from gym.spaces import MultiDiscrete
from gym.spaces.discrete import Discrete

from src.impl.activity_mdp import ActivityState, ATPAction, TravelState
from src.util.math_utils import to_onehot


class ActivityEnv(gym.Env):
    def __init__(self, config, *args, **kwargs):
        super(ActivityEnv, self).__init__()

        self.config = config
        self.irl_params = self.config.irl_params
        self.segment_mins = self.config.profile_params.segment_minutes
        self._horizon = int(self.irl_params.horizon / self.segment_mins)
        self.home_activity = self.config.home_activity
        self.work_activity = self.config.work_activity
        self.shopping_activity = self.config.shopping_activity
        self.other_activity = self.config.other_activity
        self.home_start_state = None
        self.home_goal_state = None

        self.activity_types = self.config.activity_params.keys()
        self.travel_modes = self.config.travel_params.keys()
        self.nA = len(self.activity_types + self.travel_modes)
        self._action_rev_map = None
        self.actions = self._build_actions()
        self.__actions_space = Discrete(self.nA)

        self.states = {}
        self.terminals = []
        self._g = self.build_state_graph()
        self.nS = len(self.states)

        self.__observation_space = MultiDiscrete([[0, self.nA], [0, self.nS]])

        self._transition_matrix = None
        self._reward_function = None

    @property
    def horizon(self):
        return self._horizon

    @property
    def G(self):
        return self._g

    @property
    def action_space(self):
        return self.__actions_space

    @property
    def observation_space(self):
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
        return self.reward_function.get_rewards()[state.state_id, action.action_id]

    def get_legal_actions_for_state(self, el):
        """
        If in an activity state, the agent can either stay in the activity or transition to a travel state
        (any available mode)

         If in a travelling state, the agent can either stay in the current mode, transition to a different mode,
          or arrive at an activity (essentially, any choice of action is available).

        Args:
            el: the current state type of the agent.

        Returns:
            A list of the available legal actions for the current state.

        """
        if el in self.activity_types:
            q = [k for k, v in self.actions.items() if (v.succ_ix in self.travel_modes or v.succ_ix == el)]
        elif el in self.travel_modes:
            q = self.actions.keys()
        else:
            raise ValueError("%s not Found!" % el)
        return q

    def _build_actions(self):
        actions = {}
        for action_ix, el in enumerate(self.activity_types + self.travel_modes):
            actions[action_ix] = ATPAction(action_ix, el)
        self._action_rev_map = dict((v.succ_ix, k) for k, v in actions.items())
        return actions

    def state_to_obs(self, state):
        if isinstance(state, int):
            return to_onehot(state, self.nS)
        else:
            return to_onehot(state.state_id, self.nS)

    def build_state_graph(self):
        """
        Builds the mdp state graph from the travel plans

        Returns:
            The state graph
        """

        # tmat = data.tmat  # SxT matrix

        g = self.gen_states()

        return g

    def get_home_action_id(self):
        assert self.actions is not None
        for id, act in self.actions.items():
            if self.home_activity == act.succ_ix:
                return id

    def gen_states(self):
        g = nx.DiGraph()
        self.home_goal_state = ActivityState(0, self.home_activity, self.horizon - 1, self.segment_mins,
                                             (self.home_activity, self.horizon - 1))
        self.states[0] = self.home_goal_state
        g.add_node((self.home_activity, self.horizon - 1),
                   attr_dict={'ix': 0, 'pos': (self.home_activity, self.horizon - 1), 'state': self.home_goal_state})
        self.terminals.append(0)
        state_idx = 1
        for time in reversed(range(1, self.horizon - 1)):
            for s, label in enumerate(self.activity_types + self.travel_modes):
                S = ActivityState if label in self.activity_types else TravelState
                state = S(state_idx, label, time, self.segment_mins, (label, time))
                self.states[state_idx] = state
                g.add_node((label, time), attr_dict={'ix': state_idx, 'pos': (label, time), 'state': state})
                state_idx += 1
        self.home_start_state = ActivityState(state_idx, self.home_activity, 0, self.segment_mins,
                                              (self.home_activity, 0))
        self.states[state_idx] = self.home_start_state
        g.add_node((self.home_activity, 0),
                   attr_dict={'ix': state_idx, 'pos': (self.home_activity, 0), 'state': self.home_start_state})
        return self.gen_actions(g)

    def gen_actions(self, g):
        for state in self.states.values():
            if state == self.home_goal_state:
                action_labels = [dict((action.succ_ix, a) for a, action in self.actions.items())[self.home_activity]]
            else:
                action_labels = self.get_legal_actions_for_state(state.state_label)
            if len(action_labels) > 0:
                for a in action_labels:
                    ns_el = self.actions[a].succ_ix
                    if state == self.home_goal_state:
                        g.add_edge(state.edge, (ns_el, state.time_index))
                    else:
                        g.add_edge(state.edge, (ns_el, state.time_index + 1))
                state.available_actions.extend(action_labels)
        return g

from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import multiprocessing

import gym
import networkx as nx
import numpy as np
from gym.spaces import MultiDiscrete
from gym.spaces.discrete import Discrete

from src.impl.activity_mdp import ActivityState, ATPAction, TravelState
from util.math_utils import to_onehot

num_cores = multiprocessing.cpu_count()

def make_atp_env(env_id):
    env = gym.make()

class ActivityEnv(gym.Env):
    def __init__(self, config, *args, **kwargs):
        super(ActivityEnv, self).__init__()

        self._config = config
        self.irl_params = self._config.irl_params
        self.segment_mins = self._config.profile_params.segment_minutes
        self._horizon = int(self.irl_params.horizon / self.segment_mins)
        self.home_activity = self._config.home_activity
        self.work_activity = self._config.work_activity
        self.shopping_activity = self._config.shopping_activity
        self.other_activity = self._config.other_activity
        self.home_start_state = None
        self.home_goal_state = None

        self.activity_types = self._config.activity_params.keys()
        self.travel_modes = self._config.travel_params.keys()
        self.nA = len(self.activity_types + self.travel_modes)
        self._action_rev_map = None
        self.actions = self._build_actions()
        self.__actions_space = Discrete(self.nA)

        self.nS = int(self.nA * self.horizon)
        self.states = {}

        self.__observation_space = MultiDiscrete([[0, self.nA], [0, self.nS]])
        self.terminals = []

        self._g = self.build_state_graph()
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

        g = nx.DiGraph()
        state_ix = 0

        # Add unique states
        term = False
        for t in range(self.horizon):
            for s, el in enumerate(self.activity_types + self.travel_modes):
                edge = (el, t)
                if t < self.horizon - 2:  # if it's not the last period, we can still make decisions
                    available_actions = self.get_legal_actions_for_state(el)
                else:
                    available_actions = [self.get_home_action_id()]
                    term = True
                for a in available_actions:
                    ns_el = self.actions[a].succ_ix
                    g.add_edge(edge, (ns_el, t + 1), attr_dict={'ix': a})
                if el in self.activity_types:
                    el_type = ActivityState(state_ix, el, t, self.segment_mins, edge)
                    if t == 0 and el == self.home_activity:
                        self.home_start_state = el_type
                elif el in self.travel_modes:
                    el_type = TravelState(state_ix, el, t, self.segment_mins, edge)
                else:
                    raise ValueError("%s not Found!" % el)
                if term:
                    el_type = ActivityState(state_ix, self.home_activity, t, self.segment_mins, edge)
                    self.terminals.append(state_ix)
                    self.home_goal_state = el_type
                    g.add_edge(edge, (self.home_activity, t + 1), attr_dict={'ix': available_actions[0]})
                el_type.available_actions.extend(available_actions)
                g.add_node(edge, attr_dict={'ix': state_ix, 'pos': edge, 'state': el_type})
                self.states[state_ix] = el_type
                state_ix += 1

        edge = (self.home_goal_state, self.horizon)
        el_type = ActivityState(state_ix, self.home_activity, t, self.segment_mins, edge)
        el_type.available_actions.extend(available_actions)
        g.add_node(edge, attr_dict={'ix': state_ix, 'pos': edge, 'state': el_type})
        return g

    def get_home_action_id(self):
        assert self.actions is not None
        for id, act in self.actions.items():
            if self.home_activity == act.succ_ix:
                return id

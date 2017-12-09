from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

from collections import OrderedDict
from itertools import combinations

import gym
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
        self.segment_minutes = self.config.profile_params.segment_minutes
        self._horizon = int(self.irl_params.horizon / self.segment_minutes)
        self.home_activity = self.config.home_activity
        self.work_activity = self.config.work_activity
        self.shopping_activity = self.config.shopping_activity
        self.other_activity = self.config.other_activity
        self.home_start_state = None
        self.home_goal_states = []

        self.activity_labels = self.config.activity_params.keys()
        self.travel_mode_labels = self.config.travel_params.keys()

        self.activities = self.activity_labels + self.travel_mode_labels
        self.maintenance_activity_set = {'W','w'}
        n = len(self.maintenance_activity_set)
        self.ma_dict = OrderedDict((v, to_onehot(k, n)) for k, v in enumerate(self.maintenance_activity_set))

        self.nA = len(self.activity_labels + self.travel_mode_labels)
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
        if el in self.activity_labels:
            q = [k for k, v in self.actions.items() if (v.succ_ix in self.travel_mode_labels or v.succ_ix == el)]
        elif el in self.travel_mode_labels:
            q = self.actions.keys()
        else:
            raise ValueError("%s not Found!" % el)
        return q

    def _build_actions(self):
        actions = {}
        for action_ix, el in enumerate(self.activity_labels + self.travel_mode_labels):
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
        G = {}
        state_idx = 0
        # step back through time
        for time in range(self.horizon, -1, -1):
            G[time] = {}
            available_activities = self.get_available_activities(time)
            for label in available_activities:
                G[time][label] = {}
                # determine activity type...
                if label in self.activity_labels:
                    S = ActivityState
                else:
                    S = TravelState
                    # compute possible maintenance activities done
                num_possible_ma = min(time, len(self.maintenance_activity_set))
                for i in range(num_possible_ma, -1, -1):
                    possible_mad = combinations(self.maintenance_activity_set, i)
                    for ma in possible_mad:
                        if ma == () or time < 2:
                            mad = np.zeros(len(self.maintenance_activity_set), dtype=int)
                        else:
                            mad = sum([self.ma_dict[a] for a in ma])
                        state = S(state_idx, label, time, mad=mad)
                        if time == self.horizon:
                            self.terminals.append(state_idx)
                            self.home_goal_states.append(state)
                            G[time][label][str(mad)] = {'ix': state_idx, 'pos': (label, time, str(mad)), 'state': state}
                            self.states[state_idx] = state
                            state_idx += 1
                            continue
                        elif time == 0:
                            self.home_start_state = state
                            self.states[state_idx] = state
                            G[time][label][str(mad)] = {'ix': state_idx, 'pos': (label, time, str(mad)), 'state': state}
                            break
                        else:
                            self.states[state_idx] = state
                            G[time][label][str(mad)] = {'ix': state_idx, 'pos': (label, time, str(mad)), 'state': state}
                        state_idx += 1
        return self.gen_transitions(G)

    def gen_transitions(self, G):
        for state in self.states.values():
            action_labels = self.get_legal_actions_for_state(state.state_label)
            if len(action_labels) > 0:
                for a in action_labels:
                    ns_el = self.actions[a].succ_ix
                    if ns_el not in self.get_available_activities(state.time_index + 1) or (
                            state.time_index + 1 > self.horizon):
                        continue
                    mad_curr = state.mad.astype(bool)
                    if ns_el in self.maintenance_activity_set:
                        new_mad = self.maybe_increment_mad(mad_curr, ns_el)
                    else:
                        new_mad = mad_curr.astype(int)
                    next_state = G[state.time_index + 1][ns_el][str(new_mad)]['state']
                    state.available_actions.append(next_state)
        return G

    def maybe_increment_mad(self, mad_curr, ns_el):
        return mad_curr.astype(int) + (self.ma_dict[ns_el].astype(bool) & ~mad_curr).astype(int)

    @staticmethod
    def make_time_string(tidx, segment_minutes):
        """
        Convert minutes since mignight to hrs.
        :return: Time in HH:MM notation
        """
        mm = tidx * segment_minutes
        mm_str = str(mm % 60).zfill(2)
        hh_str = str(mm // 60).zfill(2)
        return "{}:{}".format(hh_str, mm_str)

    def get_available_activities(self, time):
        if time == 0 or time == self.horizon:
            return [self.home_activity]
        elif time == 1 or time == self.horizon - 1:
            return [self.home_activity] + self.travel_mode_labels
        else:
            return self.activities

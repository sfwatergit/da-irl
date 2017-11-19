from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import multiprocessing

import gym
import networkx as nx
from enum import IntEnum
from gym.spaces.discrete import Discrete
from gym.spaces.tuple_space import Tuple

from src.impl.activity_mdp import ActivityState, ATPAction, TravelState

num_cores = multiprocessing.cpu_count()


class ATPActionType(IntEnum):
    STAY = 0
    DEPART = 1
    ARRIVE = 2
    CONTINUE_JOURNEY = 3

    @property
    def activity_action_types(self):
        return [self.STAY.value, self.DEPART.value]

    @property
    def travel_action_types(self):
        return [self.ARRIVE.value, self.CONTINUE_JOURNEY.value]


class ActivityEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        super(ActivityEnv, self).__init__()
        self.params = kwargs.pop("params", None)
        cache_dir = kwargs.pop("cache_dir", None)
        self.irl_params = self.params.irl_params
        self.segment_mins = self.params.profile_params.segment_minutes
        self._horizon = int(self.irl_params.horizon / self.segment_mins)
        self.cache_dir = cache_dir
        self.home_act = self.params.home_act
        self.work_act = self.params.work_act
        self.shopping_act = self.params.shopping_act
        self.other_act = self.params.other_act

        self.activity_types = self.params.activity_params.keys()
        self.travel_modes = self.params.travel_params.keys()
        self.nA = len(self.activity_types + self.travel_modes)
        self.actions = self._build_actions()
        self.__actions_space = Discrete(self.nA)

        self.nS = int(self.nA * self.horizon)
        self.states = {}
        self.__observation_space = Tuple((self.__actions_space, Discrete(self.horizon)))

        self.terminals = []
        self._transition_probability_matrix = None
        self._g = self.build_state_graph()

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

    def _reset(self):
        pass

    def _step(self, action):
        """

        Args:
            action:

        Returns:

        """
        pass

    def _reward(self, state):
        """
        r: S->A
        Returns:

        """
        pass

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
            return ATPActionType.activity_action_types
        elif el in self.travel_modes:
            return ATPActionType.travel_action_types
        else:
            raise ValueError("%s not Found!" % el)

    def _build_actions(self):
        return dict(((v.value, ATPAction(k.lower(), v.value))
                     for k, v in ATPActionType.__members__.items()))

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
                if t < self.horizon - 1:  # if it's not the last period, we can still make decisions
                    available_actions = self.get_legal_actions_for_state(el)
                else:
                    available_actions = [self.get_home_action_id()]
                    term = True
                for a in available_actions:
                    ns_el = self.actions[a].succ_ix
                    g.add_edge(edge, (ns_el, t + 1), attr_dict={'ix': a})
                if el in self.activity_types:
                    el_type = ActivityState(state_ix, el, t, self.segment_mins, edge)
                elif el in self.travel_modes:
                    el_type = TravelState(state_ix, el, t, self.segment_mins, edge)
                else:
                    raise ValueError("%s not Found!" % el)
                if term:
                    el_type = ActivityState(state_ix, self.home_act, t, self.segment_mins, edge)
                    self.terminals.append(state_ix)
                    self.home_state = el_type
                    g.add_edge(edge, (self.home_act, t), attr_dict={'ix': available_actions[0]})
                el_type.available_actions.extend(available_actions)
                g.add_node(edge, attr_dict={'ix': state_ix, 'pos': edge, 'state': el_type})
                self.states[state_ix] = el_type
                state_ix += 1
        return g

    def get_home_action_id(self):
        assert self.actions is not None
        for id, act in self.actions.items():
            if self.home_act == act.succ_ix:
                return id

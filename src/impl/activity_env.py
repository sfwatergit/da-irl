from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import multiprocessing

import gym
import networkx as nx
import numpy as np
import pandas as pd
from cytoolz import memoize
from gym.spaces.discrete import Discrete
from gym.spaces.tuple_space import Tuple
from tqdm import tqdm

from src.impl.activity_mdp import ActivityState, ATPAction, TravelState
from src.impl.population import ExpertTrajectoryData

num_cores = multiprocessing.cpu_count()


class ActivityEnv(gym.Env):
    def __init__(self, params, cache_dir=None):
        self.irl_params = params.irl_params
        self.segment_mins = self.irl_params.segment_minutes
        self.cache_dir = cache_dir
        self.home_act = params.home_act
        self.work_act = params.work_act
        self.shopping_act = params.shopping_act
        self.other_act = params.other_act

        self.trajectory_data = ExpertTrajectoryData(self.irl_params.plans_file_url, self.segment_mins,
                                                    self.irl_params.pop_limit, cache_dir, params.filter_params,
                                                    self.work_act)

        self.activity_types = params.activity_params.keys()
        self.travel_modes = params.travel_params.keys()

        self.actions = {}
        self.states = {}
        self.terminals = []
        self._transition_probability_matrix = None
        self._g = None
        self.paths = None
        self._build_state_graph()

    @property
    def G(self):
        return self._g

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

    def _trajectories_to_paths(self, tmat):
        paths = []
        pbar = tqdm(tmat.T, desc="Converting trajectories to state-actions")
        for trajectory in pbar:
            states = []
            actions = []
            for t, step in enumerate(trajectory):
                tup = (step, t)
                if tup in self.G.node:
                    state_ix = self.G.node[tup]['state'].state_id
                    if len(states) > 0:
                        prev_state = self.states[states[-1]]
                        state = self.states[state_ix]
                        s_type = state.state_label
                        available_actions = [self.actions[act] for act in prev_state.available_actions
                                             if (s_type == self.actions[act].succ_ix)]
                        if len(available_actions) == 0:
                            break
                        act_ix = available_actions[0].action_id
                        actions.append(act_ix)
                    states.append(state_ix)
            paths.append(np.array(zip(states, actions)))
        return np.array(paths)

    @staticmethod
    def _str_tmat_to_num_tmat(str_tmat, factors):
        tmat = np.zeros_like(str_tmat, dtype=int)
        for i, row in enumerate(str_tmat):
            r = pd.Series(row)
            for k, v in zip(factors[1], factors[0]):
                r = r.replace(k, v)
            tmat[i] = r.values
        return tmat

    @memoize
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

    def _build_state_graph(self):
        """
        Builds the mdp state graph from the travel plans

        Returns:
            The state graph
        """

        data = self.trajectory_data
        tmat = data.tmat  # SxT matrix
        els = pd.factorize(np.unique(tmat))
        unique_elements = els[1]
        num_states = len(unique_elements)

        S = Discrete(num_states)
        T = tmat.shape[0]

        # set obs and action space from data
        self.observation_space = Tuple((S, Discrete(T)))
        self.action_space = S

        g = nx.DiGraph()
        state_ix = 0

        action_ix = 0
        # Add unique actions per state
        for el in unique_elements:
            self.actions[action_ix] = ATPAction(action_ix, el)
            action_ix += 1

        # Add unique states
        term = False
        for t in range(T):
            for s, el in enumerate(unique_elements):
                edge = (el, t)
                if t < T - 1:  # if it's not the last period, we can still make decisions
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

        self._g = g

        t2p = self._trajectories_to_paths
        self.paths = t2p(tmat)
        self.nS = len(self.states)
        self.nA = len(self.actions)

    @property
    def transition_probability_matrix(self):
        if self._transition_probability_matrix is None:
            self._transition_probability_matrix = nx.adjacency_matrix(self._g)
        return self._transition_probability_matrix

    def get_home_action_id(self):
        assert self.actions is not None
        for id, act in self.actions.items():
            if self.home_act == act.succ_ix:
                return id

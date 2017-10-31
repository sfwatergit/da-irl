from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import numpy as np
from scipy.stats import *

from src.core.mdp import State, Action, MDP
from src.misc.utils import make_time_string, min_hr_to_min

ABS = 0
REL = 1


class TMDPState(State):
    def __init__(self, state_id, time_index):
        super(TMDPState, self).__init__(state_id)
        self.time_index = time_index
        self._actions = {}

    def actions(self):
        return self._actions

    def add_action(self, action):
        if action.action_id not in self._actions:
            self._actions[action.action_id] = action

    def get_all_outcomes(self, aid):
        return self._actions[aid].outcomes

    def outcome(self, aid):
        outcomes = self.get_all_outcomes(aid)
        return np.random.choice(outcomes[:, 1], replace=False, p=outcomes[:, 0].astype(float))

    def __hash__(self):
        return hash((self.state_id, self.time_index))

    def __eq__(self, other):
        return self.state_id == other.state_id and self.time_index == other.time_index

    def __str__(self):
        return '({} @ {})'.format(self.state_id, make_time_string(self.time_index))

    def __repr__(self):
        return self.__str__()


class TMDPAction(Action):
    def __init__(self, action_id, action_label):
        super(TMDPAction, self).__init__(action_id)
        self.action_label = action_label
        self._outcomes = []

    @property
    def outcomes(self):
        return np.asarray(self._outcomes)

    @outcomes.setter
    def outcomes(self, outcomes):
        self._outcomes = np.asarray(outcomes)

    def add_outcome(self, outcome):
        self._outcomes.append(outcome)

    def __str__(self):
        return self.action_label

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.action_id == other.action_id

    def __hash__(self):
        return hash(self.action_id)


class TMDP(MDP):
    def __init__(self, reward,
                 transition,
                 horizon,
                 discretization,
                 initial_state,
                 state_types,
                 action_types):
        super(TMDP, self).__init__(reward, transition, graph=None, gamma=0, env=None)
        self.action_types = action_types
        self.action_idxs = xrange(len(self.action_types))
        self.state_types = state_types
        self._initial_state = initial_state
        self.T = horizon
        self.disc = discretization  # assumed in minutes for now
        self._tidx = int(self.T / self.disc)
        self._states = np.zeros([len(self.state_types),
                                 self._tidx], dtype=object)
        self._make_terminal_states()

    def set_outcomes(self, outcomes):
        raise NotImplementedError

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def A(self):
        pass

    def actions(self, state):
        pass

    @property
    def S(self):
        if self._states is None:
            self._initialize_mdp()
        return self._states

    def get_outcome(self, s, t, a):
        return self._states[s][t].get_random_outcome(a)

    def _initialize_mdp(self):
        aid = 0
        for s, label in enumerate(self.state_types):
            ts = list(reversed(range(self._tidx - 1)))
            for t, tidx in zip(ts, np.arange(self.T - 2 * self.disc, -self.disc, -self.disc)):
                state = TMDPState(label, tidx)
                dawdling = TMDPAction(aid, 'dawdling')
                dawdling.add_outcome([self._states[s][t + 1], 1.0])
                state.add_action(dawdling)
                self._states[s][t] = state

        aid += 1

        ## Assign morning commute by pt:
        morning_pt_arrival_ts = min_hr_to_min(9, 10), min_hr_to_min(9, 45), min_hr_to_min(10, 20)
        morning_pt_sd = 12.5
        available_from_s = int(min_hr_to_min(7, 0) / self.disc)
        available_to_s = int(min_hr_to_min(7, 50) / self.disc)
        for t in range(available_from_s, available_to_s):
            commute_by_pt = TMDPAction(aid, self.action_types[aid])
            commute_by_pt.outcomes = self._assign_pdf_abs(1, morning_pt_arrival_ts, morning_pt_sd)
            self._states[0][t].add_action(commute_by_pt)

        aid += 1

        ## Assign off peak commute
        off_peak_commute_ts = min_hr_to_min(0, 30), min_hr_to_min(1, 30), min_hr_to_min(2, 30)
        off_peak_sd = 25.5
        available_from_s = int(min_hr_to_min(7, 0) / self.disc)
        available_to_s = int(min_hr_to_min(7, 20) / self.disc)
        for t in range(available_from_s, available_to_s):
            driving_to_work_via_highway = TMDPAction(aid, self.action_types[aid])
            driving_to_work_via_highway.outcomes = self._assign_pdf_rel(1, t, off_peak_commute_ts, off_peak_sd)
            self._states[0][t].add_action(driving_to_work_via_highway)

        aid +=1

    def _make_terminal_states(self):
        for s, label in enumerate(self.state_types):
            self._states[s][-1] = TMDPState(label, self.T - self.disc)

    def _assign_pdf_abs(self, s_to, time_span, scale):
        start, middle, end = time_span
        X = truncnorm((start - middle) / scale, (end - middle) / scale, loc=middle, scale=scale)
        raw_p = X.pdf(range(start, end, self.disc))
        tot = sum(raw_p)
        rescaled = raw_p / tot
        t_ids = (np.array(list(range(start, end, self.disc))) / self.disc).astype(int)
        states = [self._states[s_to][t_id] for t_id in t_ids]
        return zip(rescaled, states)

    def _assign_pdf_rel(self, s_to, t_id_in, time_span, scale):
        start, middle, end = time_span
        X = truncnorm((start - middle) / scale, (end - middle) / scale, loc=middle, scale=scale)
        raw_p = X.pdf(range(start, end, self.disc))
        tot = sum(raw_p)
        rescaled = raw_p / tot
        t_ids = (np.array(list(range(start, end, self.disc))) / self.disc).astype(int)
        new_t_ids = [t_id + t_id_in for t_id in t_ids]
        states = [self._states[s_to][new_t_id] for new_t_id in new_t_ids]
        return zip(rescaled, states)


if __name__ == '__main__':
    activity_types = ['home', 'work']
    action_types = ['dawdling', 'commute_by_pt', 'driving to work via highway', 'driving on backroad']
    test_mdp = TMDP(None, None, 1440, 10, 'home', activity_types,
                    action_types)
    test_mdp._initialize_mdp()

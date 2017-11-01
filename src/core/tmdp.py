from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import numpy as np
from scipy.stats import *

from src.core.mdp import State, Action, MDP, TransitionFunction
from src.misc.utils import make_time_string, t2n
import re

ABS = 0
REL = 1


class TMDPTransition(TransitionFunction):
    def __init__(self, env=None):
        TransitionFunction.__init__(self,env)

    def __call__(self, state, action,**kwargs):
        pass


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
        self._action_dict = dict((v, t) for v, t in enumerate(self.action_types))
        self._actions = None
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
        return self._action_dict

    def actions(self, state):
        return self._states[state].actions()

    @property
    def S(self):
        return self._states

    def get_outcome(self, s, t, a):
        return self._states[s][t].get_random_outcome(a)


    def approximate_value_iteration(self,reward, threshold = 1e-16, gamma=1):
        V = np.zeros_like(reward,dtype=np.float32)
        diff = float("inf")
        while diff> threshold:
            V_prev = np.copy(V)
            Q = reward.reshape((-1,1))+gamma
        return V, Q

    def _make_terminal_states(self):
        pass

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


def initialize_mdp(mdp):
    action_dict = mdp._action_dict

    for s, label in enumerate(mdp.state_types):
        mdp._states[s][-1] = TMDPState(label, mdp.T - mdp.disc)

    ## A0|dawdling (all states):
    k, v = action_dict.items()[0]
    for s, label in enumerate(mdp.state_types):
        st = list(reversed(range(mdp._tidx - 1)))
        for t, tidx in zip(st, np.arange(mdp.T - 2 * mdp.disc, -mdp.disc, -mdp.disc)):
            state = TMDPState(label, tidx)
            dawdling = TMDPAction(k, v)
            dawdling.add_outcome([mdp._states[s][t + 1], 1.0])
            state.add_action(dawdling)
            mdp._states[s][t] = state

    ## A1|morning commute by pt:
    k, v = action_dict.items()[1]
    morning_pt_arrival_ts = t2n(9, 10), t2n(9, 45), t2n(10, 20)
    pt_sd = 12.5
    sf, st = int(t2n(7, 0) / mdp.disc), int(t2n(7, 50) / mdp.disc)
    for t in range(sf, st):
        commute_by_pt = TMDPAction(k, v)
        commute_by_pt.outcomes = mdp._assign_pdf_abs(2, morning_pt_arrival_ts, pt_sd)
        mdp._states[0][t].add_action(commute_by_pt)

    ## A2|driving to work via highway:
    k, v = action_dict.items()[2]
    off_peak_ts = t2n(0, 30), t2n(1, 30), t2n(2, 30)  #
    car_sd = 25.5
    sf, st = int(t2n(7, 0) / mdp.disc), int(t2n(7, 20) / mdp.disc)  # 7:20

    for t in range(sf, st):
        drive_hway = TMDPAction(k, v)
        drive_hway.outcomes = mdp._assign_pdf_rel(1, t, off_peak_ts, car_sd)
        mdp._states[0][t].add_action(drive_hway)

    sf, st = int(t2n(7, 30) / mdp.disc), int(t2n(7, 50) / mdp.disc)
    for t in range(sf, st):
        drive_hway = TMDPAction(k, v)

        # prob of rush increasing from 07:20 (state 2) with prob 0.0 to 08:00 (state 6) with prob 1.0
        # the prob of rush hour/off peak for state 3,4,5 are 0.25/0.75,0.50/0.50, 0.75/0.25
        outcomes = []

        # rush hour
        rush_hour_ts = mk_ts([[0, 30], [2, 20], [6, 0]])
        outcomes_tmp = mdp._assign_pdf_rel(1, t, rush_hour_ts, car_sd)
        outcomes.extend([(p * 0.25 * (t - 2), s) for (p, s) in outcomes_tmp])

        # off peak
        off_peak_ts = mk_ts([[0, 30], [1, 30], [2, 30]])
        outcomes_tmp = mdp._assign_pdf_rel(1, t, off_peak_ts, car_sd)
        outcomes.extend([(p * 0.25 * (t - 2), s) for (p, s) in outcomes_tmp])
        drive_hway.outcomes = outcomes
        mdp._states[0][t].add_action(drive_hway)

    ##
    sf, st = int(t2n(8, 00) / mdp.disc), int(t2n(9, 30) / mdp.disc)
    rush_hr_ts = mk_ts([[0, 30], [2, 20], [6, 0]])
    for t in range(sf, st):   # 8:00 ~ 9:30
        drive_hway = TMDPAction(k, v)
        drive_hway.outcomes = mdp._assign_pdf_rel(1, t, rush_hr_ts, car_sd)
        mdp._states[0][t].add_action(drive_hway)
    sf, st = int(t2n(9, 40) / mdp.disc), int(t2n(10, 10) / mdp.disc)
    for t in range(sf, st):  # 09:40 ~ 10:10
        drive_hway = TMDPAction(k, v)
        outcomes = []
        # rush hour
        rush_hour_ts = mk_ts([[0, 30], [2, 20], [6, 0]])
        outcomes_tmp = mdp._assign_pdf_rel(1, t, rush_hour_ts, car_sd)
        outcomes.extend([(p * 0.25 * (t - 2), s) for (p, s) in outcomes_tmp])

        # off peak
        off_peak_ts = mk_ts([[0, 30], [1, 30], [2, 30]])
        outcomes_tmp = mdp._assign_pdf_rel(1, t, off_peak_ts, car_sd)
        outcomes.extend([(p * 0.25 * (t - 2), s) for (p, s) in outcomes_tmp])
        drive_hway.outcomes = outcomes
        mdp._states[0][t].add_action(drive_hway)

    off_peak_ts = t2n(0, 30), t2n(1, 30), t2n(2, 30)  #
    sf, st = int(t2n(10, 20) / mdp.disc), int(t2n(14, 30) / mdp.disc)
    for t in range(sf, st):  # 10:20(20) ~ 14:00(42)
        # off peak
        drive_hway = TMDPAction(k, v)
        drive_hway.outcomes = mdp._assign_pdf_rel(1, t, off_peak_ts, car_sd)
        mdp._states[0][t].add_action(drive_hway)

    ## A3|driving to work via backroad:
    k, v = action_dict.items()[3]
    for t in range(0, mdp._tidx):
        new_tid = t + int(t2n(1, 00) / 10)
        drive_hway = TMDPAction(k, v)
        if new_tid < mdp._tidx:
            drive_hway.outcomes=[(1.0, mdp._states[2][new_tid])]
        else:
            drive_hway.outcomes = [(1.0, mdp._states[2][mdp._tidx - 1])]
        mdp._states[1][t].add_action(drive_hway)


def mk_ts(arr):
    return tuple(map(lambda x: t2n(x[0], x[1]), (arr[0], arr[1], arr[2])))

def init_rewards(activity_types, states):
    pat = re.compile('^.*\w(\d+:\d+)$')

    rewards = dict()
    for s_from,from_label in enumerate(activity_types):
        rewards[s_from] = dict()
        for s_to,to_label in enumerate(activity_types):
            if to_label == 'Work' and from_label != 'Work':
                for state in states[s_from]:
                    a,b = str(state).split(":")
                    b=b.strip(')')
                    a=a.split(' ')[-1]
                    time = t2n(*tuple([int(e) for e in (a,b)]))
                    if time < t2n(11, 00):
                        rewards[s_from][state.time_index] = 1.0  # +1 for arriving at work before 11:00
                    elif time < t2n(12, 00):
                        rewards[s_from][state.time_index] = float((t2n(12, 00) - time) / t2n(1, 00))  # falls linearly to zero (11:00 ~ 12:00)
                    else:
                        rewards[s_from][state.time_index] = 0.0
                else:
                    rewards[s_from][state.time_index] = 0.0
    return rewards


if __name__ == '__main__':
    activity_types = ['Home', 'Work', 'x2']
    action_types = ['dawdling', 'commute_by_pt', 'driving to work via highway', 'driving on backroad']

    mdp = TMDP(None, None, 1440, 10, 'home', activity_types,
               action_types)
    initialize_mdp(mdp)
    reward = init_rewards(activity_types,mdp.S)
    mdp.approximate_value_iteration(reward)
    print('done!')

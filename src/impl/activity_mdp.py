import numpy as np

from src.core.mdp import TransitionFunction, State, Action, MDP
from src.misc.utils import make_time_string


class ATPState(State):
    def __init__(self, state_id, state_label, time_index, segment_minutes, edge):
        super(ATPState, self).__init__(state_id)
        self.state_label = state_label
        self.time_index = time_index
        self.segment_minutes = segment_minutes
        self.start_time = time_index * segment_minutes
        self.end_time = (time_index + 1) * segment_minutes
        self.available_actions = []
        self.edge = edge
        self.time_string = make_time_string(self.start_time)

    @property
    def get_end_time(self):
        return self.end_time

    def __hash__(self):
        return hash((self.state_id, self.time_index))

    def __eq__(self, other):
        return self.state_id == other.state_id and self.time_index == other.time_index


class ATPAction(Action):
    def __init__(self, action_id, succ_ix):
        super(ATPAction, self).__init__(action_id)
        self._succ_ix = succ_ix

    @property
    def action_id(self):
        return self._action_id

    @property
    def succ_ix(self):
        return self._succ_ix

    def __eq__(self, other):
        return self._action_id == other.state_id

    def __hash__(self):
        return self.action_id.__hash__()

    def __str__(self):
        return '{}:{}'.format(self._action_id, self._succ_ix)

    def __repr__(self):
        return self.__str__()


class TravelState(ATPState):
    def __init__(self, state_id, mode, time_index, segment_minutes, edge):
        super(TravelState, self).__init__(state_id, mode, time_index, segment_minutes, edge)

    def __str__(self):
        return '{}:[{}, {}]'.format(self.state_id, self.state_label, self.time_index)

    def __repr__(self):
        return self.__str__()


class ActivityState(ATPState):
    def __init__(self, state_id, activity_type, time_index, segment_minutes, edge):
        super(ActivityState, self).__init__(state_id, activity_type, time_index, segment_minutes, edge)

    def __str__(self):
        return '{}:[{}, {}]'.format(self.state_id, self.state_label, self.time_index)

    def __repr__(self):
        return self.__str__()


class ATPTransition(TransitionFunction):
    def __init__(self, env):
        TransitionFunction.__init__(self, env)

    def __call__(self, state, action, **kwargs):
        data = [a for a in self.env.G.successors(state.edge) if a[0] == action.succ_ix or (state.state_id in self.env.terminals)]
        if len(data) > 0:
            ns = self.env.G.node[data[0]]
            if len(ns) > 0:
                return ns['state']
            else:
                return self.env.home_state
        else:
            return ATPState(-1, -1, -1, 0)


class ActivityMDP(MDP):
    def __init__(self, R, T, gamma, env):
        super(ActivityMDP, self).__init__(R, T, env.G, gamma, env)
        self._env = env

    def actions(self, state):
        return self._env.get_legal_actions_for_state(state.state_label)

    @property
    def S(self):
        """
        The set of all activity states indices

        """
        return self._env.states.keys()

    @property
    def A(self):
        """
        The set of all action indices

        Returns:

        """
        return self._env.actions.keys()

    @property
    def terminals(self):
        if self._terminals is None:
            self._terminals = self._env.terminals
        return self._terminals

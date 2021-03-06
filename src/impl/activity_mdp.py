from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import numpy as np

from src.core.mdp import TransitionFunction, State, Action, MDP
from src.util.misc_utils import reverse_action_map


class ATPState(State):
    def __init__(self, state_id, symbol, time_index, mad, reachable_symbols):
        super(ATPState, self).__init__(state_id)
        self.symbol = symbol
        self._mad = mad  # mad == mandatory/mandatory activities done
        self.time_index = time_index
        self.next_states = []
        self.reachable_symbols = reachable_symbols

    @property
    def mad(self):
        return self._mad

    @mad.setter
    def mad(self, new_mad):
        self._mad = new_mad

    def __str__(self):
        return '{}:[{}, {}, {}]'.format(self.state_id, self.time_index,
                                        self.symbol, str(self._mad))

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash((self.state_id, self.time_index, str(self._mad)))

    def __eq__(self, other):
        return self.state_id == other.state_id and self.time_index == \
               other.time_index and str(self.mad) == str(other.mad)


class ATPAction(Action):
    def __init__(self, action_id, next_state_symbol):
        """Deterministic actions for activity-travel MDP.

        Args:
            action_id ():
            next_state_symbol ():
        """
        super(ATPAction, self).__init__(action_id)
        self._next_state_symbol = next_state_symbol

    @property
    def action_id(self):
        return self._action_id

    @property
    def next_state_symbol(self):
        return self._next_state_symbol

    def __eq__(self, other):
        return self._action_id == other.action_id

    def __hash__(self):
        return self.action_id.__hash__()

    def __str__(self):
        return '{}:{}'.format(self._action_id, self._next_state_symbol)

    def __repr__(self):
        return self.__str__()


class TravelState(ATPState):
    def __init__(self, state_id, mode, time_index, mad, reachable_symbols):
        super(TravelState, self).__init__(state_id, mode, time_index, mad,
                                          reachable_symbols)


class ActivityState(ATPState):
    """
    Denotes the current activity at the current time slice, inclusive of all
    other components of the activity state.
    """

    def __init__(self, state_id, activity_type, time_index, mad,
                 reachable_symbols):
        super(ActivityState, self).__init__(state_id, activity_type, time_index,
                                            mad, reachable_symbols)


class DeterministicTransition(TransitionFunction):
    def __init__(self, env):
        TransitionFunction.__init__(self, env)

    def __call__(self, state, action, **kwargs):
        """The next state is deterministic and equivalent to the state symbol
        for the action with probability 1.0.

        Args:
            state (ATPState): Current state (assumed to have a list of
                next_states representing reachable ``ATPStates``.
            action (ATPAction): The symbol representing the next state chosen.
            **kwargs (dict):

        Returns:
            (nd.array[tuple]): A tuple of reachable next states and attendant
                probabilities of reaching next_state (this is deterministic,
                so always 1.0). If the next state is unavailable, then stay
                at the current state (assumed to be terminal).
        """
        next_states = [self._env.states[ns] for ns in self._env.G.successors(
            state.state_id) if
                       self._env.states[ns].symbol == action.next_state_symbol]

        if len(next_states) > 0:
            return np.array([(1.0, next_states[0])])
        else:
            # TODO: Assert return of default is actually terminal?
            return np.array([(0.0, state)])


class ATPMDP(MDP):
    def __init__(self, person_model, reward_function, transition, actions,
                 states,
                 state_graph, gamma,horizon,interval_length,
                 is_deterministic=False):
        self.interval_length = interval_length
        self.horizon = horizon
        self.person_model = person_model
        self._state_graph = state_graph
        self._T = transition
        self._transition_matrix = None
        self._actions = actions
        self._states = states
        self.reverse_action_map = reverse_action_map(actions)
        self.is_deterministic = is_deterministic
        super(ATPMDP, self).__init__(reward_function, self._T, gamma)

    @property
    def states(self):
        return self._states

    @property
    def state_graph(self):
        return self._state_graph

    @property
    def actions(self):
        return self._actions

    @property
    def S(self):
        """
        The set of all state indices

        Returns:
            (list[int]): list of state indices

        """
        return self._states.keys()

    @property
    def A(self):
        """
        The set of all action indices.

        Returns:
            (list[int]): list of action indices
        """
        return self._actions.keys()

    @property
    def transition_matrix(self):
        dim_S = len(self.S)
        dim_A = len(self.A)
        if self._transition_matrix is None:
            self._transition_matrix = np.zeros((dim_S, dim_A, dim_S))
            for state in self._states.values():
                for action_symbol in self.available_actions(state):
                    action = self.actions[
                        self.reverse_action_map[action_symbol]]
                    for prob_next_state, next_state in self.T(state, action):
                        self._transition_matrix[
                            state.state_id, action.action_id,
                            next_state.state_id] = prob_next_state
        return self._transition_matrix

    def available_actions(self, state):
        return [self.states[next_state].symbol for next_state in
                self.state_graph.successors(state.state_id)]

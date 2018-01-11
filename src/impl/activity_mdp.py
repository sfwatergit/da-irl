import numpy as np
from cytoolz import memoize

from src.core.mdp import TransitionFunction, State, Action, MDP


class ATPState(State):
    def __init__(self, state_id, symbol, time_index, mad, reachable_symbols):
        super(ATPState, self).__init__(state_id)
        self.symbol = symbol
        self._mad = mad  # mad == mandatory/mandatory activities done
        self.time_index = time_index
        self.next_states = {}
        self.reachable_symbols = reachable_symbols

    @property
    def mad(self):
        return self._mad

    @mad.setter
    def mad(self, new_mad):
        self._mad = new_mad

    def __str__(self):
        return '{}:[{}, {}]'.format(self.state_id, self.symbol, str(self._mad))

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash((self.state_id, self.time_index, str(self._mad)))

    def __eq__(self, other):
        return self.state_id == other.state_id and self.time_index == \
               other.time_index and np.all(
            self.mad == other.mad)


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
        return self._action_id == other.action_id

    def __hash__(self):
        return self.action_id.__hash__()

    def __str__(self):
        return '{}:{}'.format(self._action_id, self._succ_ix)

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


class ATPTransition(TransitionFunction):
    def __init__(self, env):
        TransitionFunction.__init__(self, env)

    def __call__(self, state, action, **kwargs):
        if state.state_id in self._env.terminals:
            goal_state = [s for s in self._env.home_goal_states if
                          np.all(s.mad == state.mad)][0]
            return np.array([(1.0, goal_state)])
        else:
            if action.succ_ix in self._env.mandatory_activity_set:
                mad = self._env._maybe_increment_mad(state.mad, action.succ_ix)
            else:
                mad = state.mad
            next_state = [s for s in state.available_actions if
                          np.all(s.mad == mad) and (action.succ_ix == s.symbol)]
            if len(next_state) > 0:
                return np.array([(1.0, next_state[0])])
            else:
                return np.array([(1.0, self._env.home_goal_states[0])])


class ActivityMDP(MDP):
    def __init__(self, reward_function, gamma, env):
        transitions = ATPTransition(env)
        super(ActivityMDP, self).__init__(reward_function, transitions, env.G,
                                          gamma)
        self._env = env
        env.transition_matrix = self.transition_matrix
        env.reward_function = self.reward_function

    @memoize
    def actions(self, state):
        return self._env.get_legal_actions_for_state(state.symbol)

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

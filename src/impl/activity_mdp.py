import numpy as np
from cytoolz import memoize

from src.core.mdp import TransitionFunction, State, Action, MDP


class ATPState(State):
    def __init__(self, state_id, state_label, time_index, mad):
        super(ATPState, self).__init__(state_id)
        self.state_label = state_label
        self._mad = mad  # mad == maintenance/mandatory activities done
        self.time_index = time_index
        self.available_actions = []

    @property
    def mad(self):
        return self._mad

    @mad.setter
    def mad(self, new_mad):
        self._mad = new_mad

    def __str__(self):
        return '{}:[{}, {}]'.format(self.state_id, self.state_label, str(self._mad))

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash((self.state_id, self.time_index, str(self._mad)))

    def __eq__(self, other):
        return self.state_id == other.state_id and self.time_index == other.time_index and np.all(self.mad == other.mad)


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
    def __init__(self, state_id, mode, time_index, mad):
        super(TravelState, self).__init__(state_id, mode, time_index, mad)


class ActivityState(ATPState):
    def __init__(self, state_id, activity_type, time_index, mad):
        super(ActivityState, self).__init__(state_id, activity_type, time_index, mad)


class ATPTransition(TransitionFunction):
    def __init__(self, env):
        TransitionFunction.__init__(self, env)

    def __call__(self, state, action, **kwargs):
        if state.state_id in self.env.terminals:
            goal_state = [s for s in self.env.home_goal_states if np.all(s.mad == state.mad)][0]
            return np.array([(1.0, goal_state)])
        else:
            if action.succ_ix in self.env.maintenance_activity_set:
                mad = self.env.maybe_increment_mad(state.mad, action.succ_ix)
            else:
                mad = state.mad
            next_state = [s for s in state.available_actions if np.all(s.mad == mad) and (action.succ_ix == s.state_label)]
            if len(next_state)>0:
                return np.array([(1.0, next_state[0])])
            else:
                return np.array([(1.0, self.env.home_goal_states[0])])


class ActivityMDP(MDP):
    def __init__(self, reward_function, gamma, env):
        T = ATPTransition(env)
        super(ActivityMDP, self).__init__(reward_function, T, env.G, gamma, env)
        self._env = env
        env.transition_matrix = self.transition_matrix
        env.reward_function = self.reward_function

    @memoize
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

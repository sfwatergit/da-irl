from src.impl.activity_mdp import ATPAction


class DurativeAction(ATPAction):

    def __init__(self, action_id, next_state_symbol, duration):
        super(DurativeAction, self).__init__(action_id, next_state_symbol)
        self.duration = duration

    def __repr__(self):
        return "{}:{}[{}]".format(self._action_id, self._next_state_symbol,
                                  self.duration)

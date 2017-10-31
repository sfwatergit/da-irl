from src.core.mdp import MDP
from src.core.tmdp import Outcome


class TravelOutcome(Outcome):
    # Resultant state is a new location or same location
    def __init__(self, oid, resultant_state, time_span, applicable_states):
        super(TravelOutcome, self).__init__(oid, applicable_states, resultant_state, time_span)
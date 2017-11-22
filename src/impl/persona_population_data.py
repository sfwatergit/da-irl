import numpy as np

from swlcommon.personatrainer.persona import Persona
from tqdm import tqdm


class ExpertPersonaAgent(object):
    def __init__(self, traces, params, env):
        """PersonaAgent representation. To be used in IRLAgent.

        Args:
            persona (Persona): persona representation of traveler.
        """

        self.persona = Persona(traces=traces, build_profile=True,
                               config_file=params.profile_builder_config_file_path)
        self._pid = self.persona.id
        self._secondary_labels = self.persona.habitat.secondary_site_ids
        self._work_label = self.persona.works
        self._home_label = self.persona.homes
        self._profile = self.persona.get_profile_as_array().T
        self._trajectories = None

        self.env = env


    @property
    def home_label(self):
        return self._home_label

    @property
    def work_label(self):
        return self._work_label

    @property
    def secondary_labels(self):
        return self._secondary_labels

    @property
    def trajectories(self):
        if self._trajectories is None:
            t2p = self._profile_to_trajectories
            self._trajectories = t2p(self._profile)
        return self._trajectories

    def _profile_to_trajectories(self, tmat):
        trajectories = []
        pbar = tqdm(tmat.T, desc="Converting trajectories to state-actions")
        for path in pbar:
            states = []
            actions = []
            for t, step in enumerate(path):
                tup = (step, t)
                if tup in self.env.G.node:
                    state_ix = self.env.G.node[tup]['attr_dict']['state'].state_id
                    if len(states) > 0:
                        prev_state = self.env.states[states[-1]]
                        state = self.env.states[state_ix]
                        s_type = state.state_label
                        available_actions = [self.env.actions[act] for act in prev_state.available_actions
                                             if (s_type == self.env.actions[act].succ_ix)]
                        if len(available_actions) == 0:
                            break
                        act_ix = available_actions[0].action_id
                        actions.append(act_ix)
                    states.append(state_ix)
            trajectories.append(np.array(zip(states, actions)))
        return np.array(trajectories)

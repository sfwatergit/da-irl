from abc import ABCMeta

import numpy as np
import six
from swlcommon import TraceLoader
from swlcommon.personatrainer.persona import Persona
from tqdm import tqdm

from algos.maxent_irl import MaxEntIRL
from core.expert_agent import ExpertAgent
from impl.activity_config import ATPConfig
from impl.activity_env import ActivityEnv


class ExpertPersonaAgent(six.with_metaclass(ABCMeta, ExpertAgent)):
    def __init__(self, config, env, learning_algorithm=None, persona=None):
        # type: (ATPConfig, ActivityEnv, MaxEntIRL, Persona) -> ExpertPersonaAgent
        super(ExpertPersonaAgent, self).__init__(config, env, learning_algorithm)

        if persona is None:
            traces = TraceLoader.load_traces_from_csv(config.irl_params.traces_file_path)
            self.persona = Persona(traces=traces, build_profile=True,
                                   config_file=self._config.general_params.profile_builder_config_file_path)
        else:
            self.persona = persona

        self._identifier = self.persona.id
        self._secondary_sites = self.persona.habitat.secondary_site_ids
        self._work = self.persona.works[0]
        self._home = self.persona.homes[0]
        self._profile = self.persona.get_profile_as_array()
        self._trajectories = None


    @property
    def home_site(self):
        return self._home

    @property
    def work_site(self):
        return self._work

    @property
    def secondary_sites(self):
        return self._secondary_sites

    @property
    def trajectories(self):
        if self._trajectories is None:
            t2p = self._profile_to_trajectories
            self._trajectories = t2p(self._profile)
        return self._trajectories

    def _profile_to_trajectories(self, trajectory_matrix):
        trajectories = []
        pbar = tqdm(trajectory_matrix, desc="Converting trajectories to state-actions")
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
                        available_actions = [self.env.actions[act]
                                             for act in prev_state.available_actions
                                             if (s_type == self.env.actions[act].succ_ix)]
                        if len(available_actions) == 0:
                            available_actions = [self.env.actions[5]]
                        act_ix = available_actions[0].action_id
                        actions.append(act_ix)
                    states.append(state_ix)
            trajectories.append(np.array(zip(states, actions)))
        return np.array(trajectories)

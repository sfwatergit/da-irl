import numpy as np
from swlcommon import TraceLoader
from swlcommon.personatrainer.persona import Persona

from src.core.expert_agent import ExpertAgent


class ExpertPersonaAgent(ExpertAgent):
    def __init__(self, person_model, env, learning_algorithm=None, persona=None,
                 pid=None):
        super(ExpertPersonaAgent, self).__init__(person_model, env,
                                                 learning_algorithm)

        if persona is None:
            traces = TraceLoader.load_traces_from_csv(
                person_model.irl_params.traces_file_path)
            self.persona = Persona(traces=traces, build_profile=True,
                                   config_file=self._person_model.general_params.
                                   profile_builder_config_file_path)
        else:
            self.persona = persona

        if pid is None:
            self._identifier = self.persona.id
        else:
            self._identifier = pid
        self._secondary_sites = self.persona.habitat.secondary_site_ids
        self._work = self.persona.works[0]
        self._home = self.persona.homes[0]
        self._profile = np.array(
            self._filter_trajectories_not_starting_and_ending_at_home(
                self.persona.get_activity_blanket_as_array()),
            dtype='S16')
        self._trajectories = None

    @property
    def identifier(self):
        return self._identifier

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
            self._trajectories = self._profile_to_trajectories(self._profile)
        return self._trajectories

    def _filter_trajectories_not_starting_and_ending_at_home(self,
                                                             trajectories):
        res = []
        for trajectory in trajectories:
            if (trajectory[0] == self.home_site.type.symbol) and (
                    trajectory[-1] == self.home_site.type.symbol):
                res.append(trajectory)
        return res

    def _profile_to_trajectories(self, trajectory_matrix):
        trajectories = []
        for path in trajectory_matrix:
            states = []
            actions = []
            mad_curr = np.zeros(len(self.env.mandatory_activity_set),
                                dtype=bool)
            for t, step in enumerate(path):
                state = self.env.G[t][step][str(mad_curr.astype(int))]['state']
                if step in self.env.mandatory_activity_set:
                    mad_curr = self.env._maybe_increment_mad(mad_curr, step)
                if len(states) > 0:
                    prev_state = self.env.states[states[-1]]
                    available_actions = prev_state.available_actions
                    if state in available_actions:
                        act_ix = self.env._action_rev_map[state.symbol]
                    else:
                        act_ix = self.env._action_rev_map[
                            self.env.travel_mode_labels[0]]
                    actions.append(act_ix)
                states.append(state.state_id)
            trajectories.append(np.array(zip(states, actions)))
        return np.array(trajectories)

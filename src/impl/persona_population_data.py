from swlcommon import TraceLoader
from swlcommon.personatrainer.persona import Persona

from impl.activity_env import ActivityEnv
from impl.activity_mdp import ATPTransition, ActivityMDP
from impl.activity_rewards import ActivityRewardFunction
from irl.meirl import MaxEntIRLAgent
import numpy as np


class ExpertPersonaAgent(MaxEntIRLAgent):
    def __init__(self, params, env):
        """PersonaAgent representation. To be used in IRLAgent.

        Args:
            persona (Persona): persona representation of traveler.
        """

        traces = TraceLoader.load_traces_from_csv(params.irl_params.traces_file_path)
        self.persona = Persona(traces=traces, build_profile=True,
                               config_file=params.irl_params.profile_builder_config_file_path)
        self._pid = self.persona.id
        self._secondary_labels = self.persona.habitat.secondary_site_ids
        self._work_label = self.persona.works
        self._home_label = self.persona.homes
        self._trajectories = self.persona.get_profile_as_array()

        R = ActivityRewardFunction(params, env)
        # R = MATSimNNReward(params, env=env)
        T = ATPTransition(env)
        mdp = ActivityMDP(R, T, 0.95, env)
        theta_activity = np.random.normal(0.000, 1e-8, size=(1, len(R.activity_features)))
        theta_travel = np.random.normal(0.0000, 1e-8, size=(1, len(R.trip_features)))
        self.theta_prior = np.concatenate((theta_activity, theta_travel), axis=1)
        env.build_state_graph(self.trajectories)
        super(ExpertPersonaAgent, self).__init__(mdp, env.paths)

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
        return self._trajectories




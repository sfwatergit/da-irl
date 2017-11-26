import numpy as np
from swlcommon import TraceLoader
from swlcommon.personatrainer.persona import Persona
from tqdm import tqdm

from algos.maxent_irl import MaxEntIRL
from impl.activity_env import ActivityEnv
from impl.activity_mdp import ActivityMDP
from impl.activity_rewards import ActivityLinearRewardFunction


class ExpertPersonaAgent(object):
    def __init__(self, config, env_type=ActivityEnv, learning_algorithm=MaxEntIRL):
        """PersonaAgent representation. To be used in IRLAgent.

        Args:
            persona (Persona): persona representation of traveler.
        """
        self._config = config

        self.env = env_type(config=config)
        R = ActivityLinearRewardFunction(self.env)
        mdp = ActivityMDP(R, 0.95, self.env)

        traces = TraceLoader.load_traces_from_csv(config.irl_params.traces_file_path)
        self.persona = Persona(traces=traces, build_profile=True,
                               config_file=self._config.general_params.profile_builder_config_file_path)
        self._pid = self.persona.id
        self._secondary_sites = self.persona.habitat.secondary_site_ids
        self._work = self.persona.works[0]
        self._home = self.persona.homes[0]
        self._profile = self.persona.get_profile_as_array()
        self._trajectories = None
        self._learning_algorithm = learning_algorithm(mdp, verbose=False)

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

    @property
    def reward(self):
        return self._learning_algorithm.reward

    @property
    def policy(self):
        return self._learning_algorithm.policy

    def evaluate_trajectory(self, trajectory):
        states = [p[0] for p in trajectory]

    def learn_reward(self):
        self._learning_algorithm.train(self.trajectories,
                                       self._config.irl_params.num_iters,
                                       len(self.trajectories))

    def _profile_to_trajectories(self, tmat):
        trajectories = []
        pbar = tqdm(tmat, desc="Converting trajectories to state-actions")
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


class ExpertPopulation(object):
    pass

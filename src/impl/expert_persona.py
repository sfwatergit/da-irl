from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import os.path as osp
from collections import defaultdict

import numpy as np
from swlcommon import TraceLoader
from swlcommon.personatrainer.persona import Persona

from src.core.expert_agent import ExpertAgent, AbstractPathProcessor
from src.impl.activity_model import PersonModel
from src.misc import logger
from src.util.mandatory_activity_utils import maybe_increment_mad


class ExpertPersonaAgent(ExpertAgent):
    """An expert agent with activity parameters defined by ``swlcommon``'s
    PersonaBuilder.

    """

    def __init__(self,
                 config,
                 person_model,
                 mdp,
                 trajectory_procesessor=None,
                 learning_algorithm=None,
                 persona=None,
                 pid=None):
        """

        Args:
            config (ATPConfig): Configuration object.
            person_model (PersonModel): The socioeconomic data for this persona.
            mdp (src.core.mdp.MDP): The mdp defining the dynamics that this
                expert agent follows.
            persona:
            pid:
        """

        if persona is None:
            traces = TraceLoader.load_traces_from_csv(
                config.irl_params.traces_file_path)

            self.persona = Persona(traces=traces, build_profile=True,
                                   config_file=config.general_params.
                                   profile_builder_config_file_path)
        else:
            self.persona = persona

        identifier = self.persona.id if pid is None else pid

        self._secondary_sites = self.persona.habitat.secondary_site_ids
        self._work = self.persona.works[0]
        self._home = self.persona.homes[0]

        path_matrix = np.array(
            self._filter_activity_patterns_not_starting_and_ending_at_home(
                self.persona.get_activity_blanket_as_array()),
            dtype='S16')

        trajectories, option_trajectories = trajectory_procesessor.run(
            path_matrix)

        self.option_trajectories = option_trajectories

        super(ExpertPersonaAgent, self).__init__(config,
                                                 person_model,
                                                 mdp,
                                                 learning_algorithm,
                                                 trajectories,
                                                 identifier)

    @property
    def home_site(self):
        """

        Returns:

        """
        return self._home

    @property
    def work_site(self):
        """

        Returns:

        """
        return self._work

    @property
    def secondary_sites(self):
        """

        Returns:

        """
        return self._secondary_sites

    def setup_learning(self):
        prefix = "pid: %s | " % self.identifier
        tabular_log_file_pr = osp.join(self.config.general_params.log_dir,
                                       osp.join(
                                           "expert_{}".format(self.identifier),
                                           self.config.tabular_log_file))
        logger.add_tabular_output(tabular_log_file_pr)
        logger.push_prefix(prefix)
        logger.push_tabular_prefix(prefix)

    def finish_learning(self):
        params = self._learning_algorithm.get_itr_snapshot(
            self.config.irl_params.num_iters)
        params.update({'agent': self.identifier})
        logger.save_itr_params(self.config.irl_params.num_iters, params)
        self._learning_algorithm.reward.plot_current_theta(self.identifier)
        logger.pop_prefix()
        logger.pop_tabular_prefix()
        tabular_log_file_pr = osp.join(self.config.general_params.log_dir,
                                       osp.join(
                                           "expert_{}".format(self.identifier),
                                           self.config.tabular_log_file))
        logger.remove_tabular_output(tabular_log_file_pr)

    def _filter_activity_patterns_not_starting_and_ending_at_home(self,
                                                                  trajectories):
        """

        Args:
            trajectories:

        Returns:

        """
        res = []
        for trajectory in trajectories:
            if (trajectory[0] == self.home_site.type.symbol) and (
                    trajectory[-1] == self.home_site.type.symbol):
                res.append(trajectory)
        return res


class PersonaPathProcessor(AbstractPathProcessor):
    def run(self, paths_matrix):
        """Runs the main trajectory processor on persona path data.

        Convert state symbols into state_index-action_index pairs based on
        the dynamics of this class's MDP.

        Args:
            paths_matrix (np.ndarray[str]): T x num_trajectories matrix of
                state symbol sequences.

        Returns:
            (nd.array[nd.array[(int,int)]]]): List of processed state,action
            index pair sequences of length T representing discretized daily
            trajectories.

        """
        trajectories = []
        for path in paths_matrix:
            states = []
            actions = []
            for t, state_symbol in enumerate(path):
                if t == 0:
                    state_id = self._mdp.state_graph.filter_nodes_by_type(
                        'home_start_state')[0]
                    state = self._mdp.states[state_id]
                    act_ix = self._mdp.reverse_action_map[state.symbol]
                else:
                    prev_state = self._mdp.states[states[-1]]
                    state_id = self._get_next_state_idx(prev_state.state_id,
                                                        state_symbol)
                    # We require a trajectory to be feasible under the
                    # current dynamics. Skip if that is not the case.
                    if state_id is None:
                        break
                    state = self._mdp.states[state_id]
                    if state.symbol in prev_state.reachable_symbols:
                        act_ix = self._mdp.reverse_action_map[state.symbol]
                    else:
                        # We require travel between activities. Sometimes
                        # this isn't captured in the trace.
                        act_ix = self._mdp.reverse_action_map[
                            self._person_model.travel_models.keys()[0]]
                actions.append(act_ix)
                states.append(state.state_id)
            trajectories.append(np.array(zip(states, actions)))
        return np.array(trajectories)

    def _get_next_state_idx(self, prev_state_idx, next_state_symbol):
        """Compute the next state index for the current state graph.

        Given the previous state index and symbol for the next state
        transition,
        compute the next state index using the current state graph.

        Args:
            prev_state_idx (int): Index of previous state
            next_state_symbol (str): Symbol representing the state that the
                agent is transitioning to.

        Returns:
            int: Index of the next state.
        """
        prev_state = self._mdp.states[prev_state_idx]
        mad_curr = maybe_increment_mad(self._person_model, prev_state.mad,
                                       next_state_symbol)
        res = None
        for state_id in self._mdp.state_graph.successors(prev_state_idx):
            next_state = self._mdp.states[state_id]
            if np.all(mad_curr == self._mdp.states[state_id].mad) and \
                    next_state.symbol == next_state_symbol:
                res = state_id
        return res


class PersonaPathProcessorAlt(AbstractPathProcessor):
    def run(self, paths_matrix):
        """Runs the main trajectory processor on persona path data.

        Convert state symbols into state_index-action_index pairs based on
        the dynamics of this class's MDP.

        Args:
            paths_matrix (np.ndarray[str]): T x num_trajectories matrix of
                state symbol sequences.

        Returns:
            (nd.array[nd.array[(int,int)]]]): List of processed state,action
            index pair sequences of length T representing discretized daily
            trajectories.

        """
        trajectories = []
        option_trajectories = defaultdict(list)
        for path in paths_matrix:
            states = []
            actions = []
            option_end_states = []
            option_durations = []
            option_actions = []
            duration = 0
            for t, state_symbol in enumerate(path):
                if t == 0:
                    state_id = self._mdp.state_graph.filter_nodes_by_type(
                        'home_start_state')[0]
                    state = self._mdp.states[state_id]
                    act_ix = self._mdp.reverse_action_map[state.symbol]
                else:
                    prev_state = self._mdp.states[states[-1]]
                    state_id = self._get_next_state_idx(prev_state.state_id,
                                                        state_symbol)
                    # We require a trajectory to be feasible under the
                    # current dynamics. Skip if that is not the case.
                    if state_id is None:
                        break
                    state = self._mdp.states[state_id]
                    duration += 1
                    if state_symbol != prev_state.symbol:
                        option_durations.append(duration)
                        duration = 0
                        option_end_states.append(state_id)
                        option_actions.append(
                            self._mdp.reverse_action_map[state.symbol])
                    if state.symbol in prev_state.reachable_symbols:
                        act_ix = self._mdp.reverse_action_map[state.symbol]
                    else:
                        # We require travel between activities. Sometimes
                        # this isn't captured in the trace.
                        act_ix = self._mdp.reverse_action_map[
                            self._person_model.travel_models.keys()[0]]
                actions.append(act_ix)
                states.append(state.state_id)
            if len(states) < self._mdp.horizon / self._mdp.interval_length:
                continue
            else:
                trajectories.append(np.array(zip(states, actions)))
                option_end_states.append(state_id)
                option_durations.append(duration)
                option_trajectories['option_actions'].append(option_actions)
                option_trajectories['option_states'].append(option_end_states)
                option_trajectories['option_durations'].append(option_durations)
        return np.array(trajectories), option_trajectories

    def _get_next_state_idx(self, prev_state_idx, next_state_symbol):
        """Compute the next state index for the current state graph.

        Given the previous state index and symbol for the next state
        transition,
        compute the next state index using the current state graph.

        Args:
            prev_state_idx (int): Index of previous state
            next_state_symbol (str): Symbol representing the state that the
                agent is transitioning to.

        Returns:
            int: Index of the next state.
        """
        prev_state = self._mdp.states[prev_state_idx]
        mad_curr = maybe_increment_mad(self._person_model, prev_state.mad,
                                       next_state_symbol)
        res = None
        for state_id in self._mdp.state_graph.successors(prev_state_idx):
            next_state = self._mdp.states[state_id]
            if np.all(mad_curr == self._mdp.states[state_id].mad) and \
                    next_state.symbol == next_state_symbol:
                res = state_id
        return res

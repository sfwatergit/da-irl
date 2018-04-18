#!/usr/bin/penv python
# coding=utf-8

# py3 compat
from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

# std lib
import argparse
import ast
import datetime
import json
import multiprocessing
import platform
import uuid
from itertools import izip

import dateutil
import matplotlib
import numpy as np
import os.path as osp
import pandas as pd
from swlcommon import TraceLoader, Persona

from src.impl.activity_config import ATPConfig
from src.impl.env_builder import HouseholdEnvBuilder, AgentBuilder
from src.impl.expert_persona import PersonaPathProcessorAlt
from src.impl.parallel.parallel_population import SubProcVecExpAgent
from src.misc import logger
from src.util.math_utils import create_dir_if_not_exists
from src.util.misc_utils import set_global_seeds

if platform.system() == 'Darwin':
    matplotlib.rcParams['backend'] = 'agg'
else:
    matplotlib.rcParams['backend'] = 'Agg'

import matplotlib.pyplot as plt

plt.interactive(False)


def tp(groups, df):
    idx = np.random.randint(0, len(groups))
    group = groups.values()[idx]
    uid_df = TraceLoader.load_traces_from_df(df.iloc[group])
    return Persona(traces=uid_df, build_profile=True,
                   config_file=config.general_params
                   .profile_builder_config_file_path)


def run(config, log_dir):
    """

    Args:
        config:
        log_dir:
    """
    # std lib
    import logging

    # third party
    import gym
    import tensorflow as tf

    # swl

    # da-irl
    from src.algos.maxent_irl import MaxEntIRL
    from src.impl.expert_persona import ExpertPersonaAgent

    ncpu = multiprocessing.cpu_count()
    if platform.system() == 'Darwin': ncpu //= 2
    tf_config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=ncpu,
                               inter_op_parallelism_threads=ncpu)
    tf_config.gpu_options.allow_growth = True  # pylint: disable=E1101
    gym.logger.setLevel(logging.WARN)

    env_builder = HouseholdEnvBuilder(
        config.household_params.household_model,
        config.profile_params.interval_length, config.irl_params.horizon)
    traces_path = config.traces_path
    if traces_path.endswith('csv'):
        df = pd.read_csv(traces_path)
    elif traces_path.endswith('parquet'):
        df = pd.read_parquet(config.irl_params.traces_file_path,
                             engine="fastparquet")
    else:
        df = pd.read_csv('../data/traces/persona_1')
        logger.log(
            'No traces file found... assuming test and loading default persona')

    groups = df.groupby('uid').groups

    def make_expert(idx, persona):
        """

        Args:
            idx:
            trace_file:

        Returns:

        """

        def _thunk():
            """

            Returns:

            """
            exp_dir = osp.join(log_dir, 'expert_%s' % idx)
            logger.set_snapshot_dir(exp_dir)

            # For single agent, use the first household member model.
            person_model = \
                config.household_params.household_model.household_member_models[
                    0]

            agent_builder = AgentBuilder(config, person_model, idx)
            env_builder.add_agent(agent_builder)
            activity_env = env_builder.finalize_env()
            mdp = activity_env.mdps[idx]

            learning_algorithm = MaxEntIRL(mdp,
                                           int(config.irl_params.horizon /
                                               config.profile_params.interval_length))
            return ExpertPersonaAgent(config, person_model, mdp,
                                      trajectory_procesessor=PersonaPathProcessorAlt(
                                          mdp, person_model),
                                      learning_algorithm=learning_algorithm,
                                      persona=persona, pid=idx)

        return _thunk

    personas = [tp(groups, df) for _ in range(config.num_experts)]

    expert_agent = SubProcVecExpAgent(
        [make_expert(idx, persona) for idx, persona in enumerate(personas)])

    expert_agent.learn_reward()

    expert_data = [{'policy': p, 'reward': r, 'theta': t} for p, r, t in
                   izip(expert_agent.get_policy(), expert_agent.get_rewards(),
                        expert_agent.get_theta())]
    expert_agent.close()

    plt.imshow(expert_data[0]['reward'][0], aspect='auto')
    plt.savefig(log_dir + '/reward')
    plt.clf()
    logger.remove_tabular_output(tabular_log_file)
    logger.remove_text_output(text_log_file)

    expert_agent.close()


if __name__ == '__main__':
    now = datetime.datetime.now(dateutil.tz.tzlocal())

    # avoid name clashes when running distributed jobs
    rand_id = str(uuid.uuid4())[:5]
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')

    default_exp_name = '_%s_%s' % (timestamp, rand_id)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Experiment configuration',
                        type=str)
    parser.add_argument('--traces_path', help='Location of trace files',
                        type=str)
    parser.add_argument('--num_experts', type=int, default=4,
                        help='Number of experts in policy net')
    parser.add_argument(
        '--exp_name', type=str, default=default_exp_name,
        help='Name of the experiment.')
    parser.add_argument('--snapshot_mode', type=str, default='last',
                        help='Mode to save the snapshot. Can be either '
                             '"all" '
                             '(all iterations will be saved), "last" (only '
                             'the last iteration will be saved), "gap" ('
                             'every'
                             '`snapshot_gap` iterations are saved), '
                             'or "none" '
                             '(do not save snapshots)')
    parser.add_argument('--params_log_file', type=str,
                        default='params.json',
                        help='Name of the parameter log file (in json).')
    parser.add_argument('--tabular_log_file', type=str,
                        default='progress.csv',
                        help='Name of the tabular log file (in csv).')
    parser.add_argument('--text_log_file', type=str, default='debug.log',
                        help='Name of the text log file (in pure text).')
    parser.add_argument('--plot', type=ast.literal_eval, default=False,
                        help='Whether to plot the iteration results')
    parser.add_argument('--log_tabular_only', type=ast.literal_eval,
                        default=False,
                        help='Whether to only print the tabular log '
                             'information (in a horizontal format)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Name of the pickle file to resume experiment '
                             'from.')
    parser.add_argument('--seed', type=int,
                        help='Random seed for numpy')

    args, unparsed = parser.parse_known_args()

    root_dir = osp.dirname(osp.abspath(__file__))
    config_file = osp.join(root_dir, args.config)

    with open(config_file) as fp:
        config = ATPConfig(data=json.load(fp))
        config.update(args.__dict__)
    if args.seed is not None:
        set_global_seeds(config.seed)
        # set_seed(args.seed)

    default_log_dir = config.general_params.log_path
    exp_name = config.general_params.run_id + default_exp_name

    log_dir = osp.join(default_log_dir, exp_name)
    config.general_params._to_dict().update({'log_dir': log_dir})

    tabular_log_file = osp.join(log_dir, config.tabular_log_file)
    text_log_file = osp.join(log_dir, config.text_log_file)

    params_log_file = osp.join(log_dir, config.params_log_file)
    create_dir_if_not_exists(log_dir)
    config.save_to_file(params_log_file)

    logger.add_text_output(text_log_file)
    logger.add_tabular_output(tabular_log_file)
    logger.set_snapshot_mode(config.snapshot_mode)
    prev_snapshot_dir = logger.get_snapshot_dir()
    prev_mode = logger.get_snapshot_mode()

    logger.set_snapshot_dir(log_dir)
    logger.set_log_tabular_only(config.log_tabular_only)
    logger.dump_tabular(with_prefix=False)
    run(config, log_dir)

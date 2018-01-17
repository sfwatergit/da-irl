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
import logging
import multiprocessing
import os.path as osp
import platform
from itertools import izip

import dateutil
# third party
import gym
import joblib
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from swlcommon import TraceLoader, Persona

from src.algos.actor_mimic import ATPActorMimicIRL
from src.algos.maxent_irl import MaxEntIRL
from src.impl.activity_config import ATPConfig
from src.impl.activity_env import ActivityEnv
from src.impl.activity_mdp import ATPMDP
from src.impl.activity_rewards import ATPRewardFunction
from src.impl.expert_persona import ExpertPersonaAgent
from src.impl.parallel.parallel_population import SubProcVecExpAgent
from src.misc import logger
from src.util.math_utils import create_dir_if_not_exists
from src.util.misc_utils import set_global_seeds, get_expert_fnames

if platform.system() == 'Darwin':
    matplotlib.rcParams['backend'] = 'Agg'
else:
    matplotlib.rcParams['backend'] = 'Agg'


def tp(groups, df):
    idx = np.random.randint(0, len(groups))
    group = groups.values()[idx]
    uid_df = TraceLoader.load_traces_from_df(df.iloc[group])
    return Persona(traces=uid_df, build_profile=True,
                   config_file=config.general_params
                   .profile_builder_config_file_path)


def run(config, log_dir):
    logger.log(
        "\n====Running Actor-Mimic Reward IRL Experiment on the Activity "
        "Travel Plan Domain!!====\n",
        with_prefix=False, with_timestamp=False)

    ncpu = multiprocessing.cpu_count()
    if platform.system() == 'Darwin': ncpu //= 2
    tf_config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=ncpu,
                               inter_op_parallelism_threads=ncpu)
    tf_config.gpu_options.allow_growth = True  # pylint: disable=E1101
    gym.logger.setLevel(logging.WARN)
    activity_env = ActivityEnv(config)
    logger.log("\n====Loading Persona Data====\n", with_prefix=False,
               with_timestamp=False)

    traces_path = config.irl_params.traces_file_path
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
        def _thunk():
            exp_dir = osp.join(log_dir, 'expert_%s' % idx)
            logger.set_snapshot_dir(exp_dir)
            mdp = ATPMDP(ATPRewardFunction(activity_env),
                         config.irl_params.gamma, activity_env)
            learning_algorithm = MaxEntIRL(mdp)
            return ExpertPersonaAgent(config, activity_env, learning_algorithm,
                                      persona, idx)

        return _thunk

    if config.resume_from is None:
        logger.log("\n====Training {} experts from scratch!====\n".format(
            config.num_experts), with_prefix=False,
                   with_timestamp=False)
        personas = [tp(groups, df) for _ in range(config.num_experts)]
        expert_agent = SubProcVecExpAgent(
            [make_expert(idx, persona) for idx, persona in enumerate(personas)])
        expert_agent.learn_reward()
        expert_data = [{'policy': p, 'reward': r, 'theta': t} for p, r, t in
                       izip(expert_agent.get_policy(),
                            expert_agent.get_rewards(),
                            expert_agent.get_theta())]
        expert_agent.close()

    else:
        logger.log("\n====Resuming from {} with {} experts!====\n".format(
            config.resume_from, config.num_experts),
                   with_prefix=False, with_timestamp=False)
        expert_file_data = get_expert_fnames(config.resume_from,
                                             n=config.num_experts)
        expert_data = []
        for filename in expert_file_data:
            expert_data.append(joblib.load(filename))

    init_theta = np.squeeze(
        np.array([expert['theta'] for expert in expert_data])).mean(0)
    init_theta = np.expand_dims(init_theta, 1)
    mdp = ATPMDP(
        ATPRewardFunction(activity_env, initial_theta=init_theta),
        config.irl_params.gamma,
        activity_env)

    teacher = ATPActorMimicIRL(mdp, expert_data)

    for i in range(config.num_iters_amn):
        teacher.train_amn()

    nn_params = {'h_dim': 32, 'reg_dim': 10, 'name': 'maxent_irl'}

    student_personas = [tp(groups, df) for _ in range(config.num_students)]

    logger.log("\n====Training Actor Mimic: Policy + Reward Students====\n",
               with_prefix=False, with_timestamp=False)
    for idx, student_persona in enumerate(student_personas):
        condition = "s_ampr"
        pid = "{}_{}".format(condition, str(idx))
        exp_dir = osp.join(log_dir, 'expert_%s' % pid)
        logger.set_snapshot_dir(exp_dir)
        nn_params.update({'name': '{}'.format(idx)})
        mdp = ATPMDP(
            ATPRewardFunction(activity_env, nn_params=nn_params,
                              initial_theta=teacher.reward.get_theta()),
            config.irl_params.gamma,
            activity_env)
        student = ExpertPersonaAgent(config, activity_env,
                                     MaxEntIRL(mdp, policy=teacher.policy),
                                     persona=student_persona, pid=pid)
        student.learn_reward(skip_policy=5, iterations=config.num_iters_s + 5)

    tf.reset_default_graph()

    logger.log("\n====Training Actor Mimic: Policy Students====",
               with_prefix=False, with_timestamp=False)
    for idx, student_persona in enumerate(student_personas):
        condition = "s_amp"
        pid = "{}_{}".format(condition, str(idx))
        exp_dir = osp.join(log_dir, 'expert_%s' % pid)
        logger.set_snapshot_dir(exp_dir)
        nn_params.update({'name': '{}'.format(idx)})
        mdp = ATPMDP(
            ATPRewardFunction(activity_env, nn_params=nn_params),
            config.irl_params.gamma,
            activity_env)
        student = ExpertPersonaAgent(config, activity_env,
                                     MaxEntIRL(mdp, policy=teacher.policy),
                                     persona=student_persona, pid=pid)
        student.learn_reward(skip_policy=5, iterations=config.num_iters_s + 5)

    tf.reset_default_graph()

    logger.log("\n====Training Actor Mimic: Reward Students====\n",
               with_timestamp=False)
    for idx, student_persona in enumerate(student_personas):
        condition = "s_amr"
        pid = "{}_{}".format(condition, str(idx))
        exp_dir = osp.join(log_dir, 'expert_%s' % pid)
        logger.set_snapshot_dir(exp_dir)
        nn_params.update({'name': '{}'.format(idx)})
        mdp = ATPMDP(
            ATPRewardFunction(activity_env, nn_params=nn_params,
                              initial_theta=teacher.reward.get_theta()),
            config.irl_params.gamma,
            activity_env)

        student = ExpertPersonaAgent(config, activity_env, MaxEntIRL(mdp),
                                     persona=student_persona, pid=pid)
        student.learn_reward(iterations=config.num_iters_s)  # Don't skip any

    tf.reset_default_graph()

    logger.log("\n====Training Vanilla Students====\n", with_timestamp=False)
    for idx, student_persona in enumerate(student_personas):
        condition = "s_v"
        pid = "{}_{}".format(condition, str(idx))
        exp_dir = osp.join(log_dir, 'expert_%s' % pid)
        logger.set_snapshot_dir(exp_dir)
        nn_params.update({'name': '{}'.format(idx)})
        mdp = ATPMDP(
            ATPRewardFunction(activity_env, nn_params=nn_params),
            config.irl_params.gamma,
            activity_env)

        student = ExpertPersonaAgent(config, activity_env, MaxEntIRL(mdp),
                                     persona=student_persona,
                                     pid=pid)
        student.learn_reward(iterations=config.num_iters_s)

    logger.remove_text_output(text_log_file)


if __name__ == '__main__':
    now = datetime.datetime.now(dateutil.tz.tzlocal())

    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')

    default_exp_name = '_%s' % timestamp

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Experiment configuration', type=str)
    parser.add_argument(
        '--exp_name', type=str, default=default_exp_name,
        help='Name of the experiment.')
    parser.add_argument('--snapshot_mode', type=str, default='last',
                        help='Mode to save the snapshot. Can be either "all" '
                             '(all iterations will be saved), "last" (only '
                             'the last iteration will be saved), "gap" (every'
                             '`snapshot_gap` iterations are saved), or "none" '
                             '(do not save snapshots)')
    parser.add_argument('--params_log_file', type=str, default='params.json',
                        help='Name of the parameter log file (in json).')
    parser.add_argument('--tabular_log_file', type=str, default='progress.csv',
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
    parser.add_argument('--num_iters_amn', type=int, default=20,
                        help='Number of iterations for actor_mimic')
    parser.add_argument('--num_iters_s', type=int, default=100,
                        help='Number of iterations to train student')
    parser.add_argument('--num_students', type=int, default=10,
                        help='Number of students to train')
    parser.add_argument('--num_experts', type=int, default=50,
                        help='Number of experts in policy net')

    args, unparsed = parser.parse_known_args()
    root_dir = osp.dirname(osp.abspath(__file__))
    config_file = osp.join(root_dir, args.config)
    with open(config_file) as fp:
        config = ATPConfig(data=json.load(fp), json_file=args.config)
        # TODO: This is a hacky way to combine file-based and cli config
        # params... fix this!
        config._to_dict().update(args.__dict__)
    if args.seed is not None:
        set_global_seeds(config.seed)
        # set_seed(args.seed)

    default_log_dir = config.general_params.log_path
    exp_name = config.general_params.run_id + default_exp_name
    print(default_log_dir)
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

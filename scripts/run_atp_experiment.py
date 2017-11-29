import argparse
import ast
import datetime
import json
import os.path as osp
import platform
import uuid

import dateutil
import joblib
import matplotlib
import numpy as np
import tensorflow as tf
from swlcommon import Persona, TraceLoader

from algos.actor_mimic import ATPActorMimicIRL
from impl.activity_config import ATPConfig
from impl.expert_persona import ExpertPersonaAgent
from misc import logger
from util.math_utils import create_dir_if_not_exists
from util.misc_utils import get_expert_fnames

if platform.system() == 'Darwin':
    matplotlib.rcParams['backend'] = 'agg'
else:
    matplotlib.rcParams['backend'] = 'TkAgg'

import matplotlib.pyplot as plt

plt.interactive(False)


def tp():
    uid_df = TraceLoader.load_traces_from_csv("../data/traces/traces_persona_0.csv")
    p1 = Persona(traces=uid_df, build_profile=True,
                 config_file='../data/misc/initial_profile_builder_config.json')
    uid_df = TraceLoader.load_traces_from_csv("../data/traces/traces_persona_1.csv")
    p2 = Persona(traces=uid_df, build_profile=True,
                 config_file='../data/misc/initial_profile_builder_config.json')
    return [p1, p2]


def plot_reward(ys, log_dir='', title='', color='b', show=False):
    xs = np.arange(0, len(ys)) * 15 / 60
    plt.plot(xs, ys, color)
    plt.title('Marginal Utility vs. Time of Day for {} Activity'.format(title.capitalize()))
    plt.xlabel('time (hr)')
    plt.ylabel('marginal utility (utils/hr)')
    if show:
        plt.show()
    else:
        plt.savefig(log_dir + '/persona_0_activity_{}'.format(title))


def run(config, log_dir):
    experts = []
    if config.resume_from is not None:
        expert_file_data = get_expert_fnames(config.resume_from)
        for filename in expert_file_data:
            experts.append(joblib.load(filename))
        expert_agent = ExpertPersonaAgent(config,
                                          initial_theta=np.mean(np.array([experts[0]['theta'], experts[1]['theta']]),
                                                                0))
    else:
        for idx, persona in enumerate(tp()):
            expert_agent = ExpertPersonaAgent(config, persona)
            exp_dir = osp.join(log_dir, 'expert_%s' % idx)
            logger.set_snapshot_dir(exp_dir)
            expert_agent.learn_reward()
            expert_data = {'policy': expert_agent.policy, 'reward': expert_agent.reward.get_rewards(),
                           'theta': expert_agent.reward.get_theta()}
            experts.append(expert_data)
            tf.reset_default_graph()

    model = ATPActorMimicIRL(expert_agent._learning_algorithm.mdp, experts)
    for i in range(30):
        model.train_amn()
    model.train(expert_agent.trajectories)
    plt.imshow(experts[0]['reward'], aspect='auto')
    plt.savefig(log_dir + '/reward')
    plt.clf()
    logger.remove_tabular_output(tabular_log_file)
    logger.remove_text_output(text_log_file)

    theta = experts[1]['theta']
    home_feats = theta[4:96]
    work_feats = theta[98:192]
    other_feats = theta[193:-4]

    show = False

    plot_reward(home_feats, log_dir, 'home', 'b', show)
    plt.clf()
    plot_reward(work_feats, log_dir, 'work', 'g', show)
    plt.clf()
    plot_reward(other_feats, log_dir, 'other', 'r', show)
    plt.clf()


if __name__ == '__main__':
    now = datetime.datetime.now(dateutil.tz.tzlocal())

    # avoid name clashes when running distributed jobs
    rand_id = str(uuid.uuid4())[:5]
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')

    default_exp_name = '_%s_%s' % (timestamp, rand_id)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Experiment configuration', type=str)
    parser.add_argument(
        '--exp_name', type=str, default=default_exp_name, help='Name of the experiment.')
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
    parser.add_argument('--log_tabular_only', type=ast.literal_eval, default=False,
                        help='Whether to only print the tabular log information (in a horizontal format)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Name of the pickle file to resume experiment from.')

    parser.add_argument('--seed', type=int,
                        help='Random seed for numpy')

    args, unparsed = parser.parse_known_args()

    with open(args.config) as fp:
        config = ATPConfig(data=json.load(fp), json_file=args.config)
        config._to_dict().update(args.__dict__)
    if args.seed is not None:
        pass
        # set_seed(args.seed)

    default_log_dir = config.general_params.log_path
    exp_name = config.general_params.run_id + default_exp_name
    log_dir = osp.join(default_log_dir, exp_name)

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

    run(config, log_dir)

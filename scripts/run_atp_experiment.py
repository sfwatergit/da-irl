import argparse
import ast
import datetime
import json
import os.path as osp
import uuid

import dateutil
import matplotlib
import numpy as np
from swlcommon import TraceLoader

matplotlib.use('macosx')
import matplotlib.pyplot as plt

plt.interactive(False)
from algos.maxent_irl import MaxEntIRL
from file_io.activity_config import ATPConfig
from impl.activity_env import ActivityEnv
from impl.persona_population_data import ExpertPersonaAgent
from misc import logger
from util.math_utils import create_dir_if_not_exists


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


def run(config, cache_dir, log_dir):
    env = ActivityEnv(params=config, cache_dir=cache_dir)
    traces = TraceLoader.load_traces_from_csv(config.irl_params.traces_file_path)
    expert_agent = ExpertPersonaAgent(traces, config, env)

    num_iters = config.irl_params.num_iters
    learning_algorithm = MaxEntIRL(env, verbose=False)

    learning_algorithm.train(
        expert_agent.trajectories,
        num_iters,
        minibatch_size=len(expert_agent.trajectories),
        cache_dir=cache_dir)

    traces = TraceLoader.load_traces_from_csv("/Users/sfeygin/PycharmProjects/da-irl/data/traces/traces_persona_1.csv")
    expert_agent = ExpertPersonaAgent(traces, config, env)

    learning_algorithm.train(
        expert_agent.trajectories,
        num_iters,
        minibatch_size=len(expert_agent.trajectories),
        cache_dir=cache_dir)

    logger.remove_tabular_output(tabular_log_file)
    logger.remove_text_output(text_log_file)

    logger.pop_prefix()
    theta = learning_algorithm.mdp.reward.get_theta()
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
    parser.add_argument('--snapshot_mode', type=str, default='all',
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

    parser.add_argument('--seed', type=int,
                        help='Random seed for numpy')

    args, unparsed = parser.parse_known_args()

    with open(args.config) as fp:
        config = ATPConfig(data=json.load(fp), json_file=args.config)

    if args.seed is not None:
        pass
        # set_seed(args.seed)

    default_log_dir = config.general_params.log_path
    exp_name = config.general_params.run_id + default_exp_name
    log_dir = osp.join(default_log_dir, exp_name)

    tabular_log_file = osp.join(log_dir, args.tabular_log_file)
    text_log_file = osp.join(log_dir, args.text_log_file)

    params_log_file = osp.join(log_dir, args.params_log_file)
    create_dir_if_not_exists(log_dir)
    config.save_to_file(params_log_file)

    logger.add_text_output(text_log_file)
    logger.add_tabular_output(tabular_log_file)
    prev_snapshot_dir = logger.get_snapshot_dir()
    prev_mode = logger.get_snapshot_mode()

    logger.set_snapshot_dir(log_dir)
    logger.set_log_tabular_only(args.log_tabular_only)
    logger.push_prefix("[%s] " % exp_name)
    cache_dir = '{}/.cache/joblib/{}'.format(osp.dirname(osp.realpath(__file__)),
                                             config.general_params.run_id)

    run(config, cache_dir, log_dir)

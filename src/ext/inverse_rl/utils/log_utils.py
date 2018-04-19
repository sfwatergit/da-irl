import contextlib
import json
import os
import random

import joblib
import numpy as np
import rllab.misc.logger as rllablogger
import tensorflow as tf

from ext.inverse_rl.utils.hyperparametrized import extract_hyperparams
from util.math_utils import to_onehot


@contextlib.contextmanager
def rllab_logdir(algo=None, dirname=None, snapshot_mode='all',
                 snapshot_gap='5'):
    rllablogger.set_snapshot_mode(snapshot_mode)
    if snapshot_mode is not 'last' or 'none':
        rllablogger.set_snapshot_gap(snapshot_gap)
    if dirname:
        rllablogger.set_snapshot_dir(dirname)
    dirname = rllablogger.get_snapshot_dir()
    rllablogger.add_tabular_output(os.path.join(dirname, 'progress.csv'))
    if algo:
        with open(os.path.join(dirname, 'params.json'), 'w') as f:
            params = extract_hyperparams(algo)
            json.dump(params, f)
    yield dirname
    rllablogger.remove_tabular_output(os.path.join(dirname, 'progress.csv'))


def get_expert_fnames(log_dir, n=5):
    print('Looking for paths')
    import re
    itr_reg = re.compile(r"itr_(?P<itr_count>[0-9]+)\.pkl")

    itr_files = []
    for i, filename in enumerate(os.listdir(log_dir)):
        m = itr_reg.match(filename)
        if m:
            itr_count = m.group('itr_count')
            itr_files.append((itr_count, filename))

    itr_files = sorted(itr_files, key=lambda x: int(x[0]), reverse=True)[:n]
    for itr_file_and_count in itr_files:
        fname = os.path.join(log_dir, itr_file_and_count[1])
        print('Loading %s' % fname)
        yield fname


def load_experts(fname, max_files=float('inf'), min_return=None):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    fname = list(fname)
    num_experts = len(fname)
    encoding = [to_onehot(i, num_experts) for i in range(num_experts)]
    expert_map = {}
    expert_index = 0
    if num_experts > 1:
        paths = []
        for fname_ in fname:
            tf.reset_default_graph()
            with tf.Session(config=config):
                snapshot_dict = joblib.load(fname_)
            expert_map[snapshot_dict['paths'][0]['agent_id'][0]] \
                = encoding[expert_index]
            expert_index += 1
            paths.extend(snapshot_dict['paths'])
    else:
        with tf.Session(config=config):
            snapshot_dict = joblib.load(fname[0])
        paths = snapshot_dict['paths']
    tf.reset_default_graph()

    trajs = []

    for path in paths:
        obses = path['observations']
        actions = path['actions']
        returns = path['returns']
        agent_id = path['agent_id'][0]
        total_return = np.sum(returns)
        if (min_return is None) or (total_return >= min_return):
            latent_info = {'latent_info': {'latent': expert_map[agent_id]}}
            traj = {'observations': obses, 'actions': actions,
                    'agent_infos': latent_info}
            trajs.append(traj)
    random.shuffle(trajs)
    print('Loaded %d trajectories' % len(trajs))
    return trajs


def load_latest_experts(logdir, n=5, min_return=None):
    return load_experts(get_expert_fnames(logdir, n=n), min_return=min_return)

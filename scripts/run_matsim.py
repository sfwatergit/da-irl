import os
# import matplotlib.pyplot as plt
import sys

from swlcommon import TraceLoader

from algos.maxent_irl import MaxEntIRL
from impl.expert_persona import ExpertPersonaAgent
from misc import logger

sys.path.append('../src')
sys.path.append('../')

from file_io.activity_config import ATPConfig
from impl.activity_env import ActivityEnv
from util.math_utils import create_dir_if_not_exists

# plt.style.use('ggplot')

FLAGS = {}


def mk_out_dir(params, kind):
    root_dir = params.general_params['rootPath']
    run_id = params.general_params['runId']
    output_dir = params.general_params['outputDir']
    out_path = "{}/{}/{}/{}/".format(root_dir, output_dir, kind, run_id)
    create_dir_if_not_exists(out_path)
    return out_path


# def plot_feature_diff(params, feature_diff):
#     plt.clf()
#     k = params.irl_params.num_iters
#     plt.plot(xrange(len(feature_diff)), feature_diff)
#     plt.title('Average Feature Expectation Differences for {}\nVanilla MaxEnt'.format(k, 's' if k > 1 else ''))
#     plt.xlabel('Iteration')
#     plt.ylabel(r'$\mu(\pi_E)-\mu(\pi)$')
#     plt.savefig('{}/{}'.format(mk_out_dir(params, 'images'), 'feature_diff.png'), format='png')
#
#
# def plot_log_lik(params, nll):
#     plt.clf()
#     k = params.irl_params.num_iters
#     plt.plot(xrange(len(nll)), nll)
#     plt.title('Negative Log Likelihood for Vanilla MaxEnt'.format(k, 's' if k > 1 else ''))
#     plt.xlabel('Iteration')
#     plt.ylabel(r'NLL')
#     plt.savefig('{}/{}'.format(mk_out_dir(params, 'images'), 'log_lik.png'), format='png')


def save_run_data(params, feature_names, feature_diff, nll=None, plot=True):
    out_path = mk_out_dir(params, 'run_data')
    num_iters = params.irl_params.num_iters
    num_samples = params.irl_params.pop_limit
    template = "{}{}_{} iters {} samples.csv"
    # pd.DataFrame(np.hstack(feature_diff), index=feature_names) \
    #     .to_csv(template.format(out_path, 'feature_diff', num_iters, num_samples),
    #             index_label="feature_names")
    # pd.DataFrame(np.hstack(theta_hist)) \
    #     .to_csv(template.format(out_path, 'theta_hist', num_iters, num_samples), index_label="feature_names")
    # if plot:
    # plot_feature_diff(params, feature_diff)
    # plot_log_lik(params, nll)


def main():
    with open(FLAGS.config) as fp:
        data = json.load(fp)
        params = ATPConfig(data)

    cache_dir = '{}/.cache/joblib/{}'.format(os.path.dirname(os.path.realpath(__file__)),
                                             params.general_params['runId'])

    create_dir_if_not_exists(cache_dir)
    logger.log("Obtaining samples...")
    env = ActivityEnv(params=params, cache_dir=cache_dir)
    traces = TraceLoader.load_traces_from_csv(params.irl_params.traces_file_path)
    expert_agent = ExpertPersonaAgent(traces, params, env)

    num_iters = params.irl_params.num_iters
    learning_algorithm = MaxEntIRL(env, expert_agent, verbose=True)

    learning_algorithm.train(num_iters,
                             minibatch_size=len(expert_agent.trajectories),
                             cache_dir=cache_dir)

    save_run_data(params, map(str,
                              learning_algorithm.mdp.reward_function.R.features),
                  learning_algorithm.feature_diff,
                  learning_algorithm.log_lik_hist, True)


if __name__ == '__main__':
    import json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)

    FLAGS, unparsed = parser.parse_known_args()
    main()

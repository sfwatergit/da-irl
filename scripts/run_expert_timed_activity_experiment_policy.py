# Default
import argparse
import sys

import joblib
from rllab.envs.normalized_env import NormalizedEnv

from impl.timed_activities.rllab_misc import BaselineMLP, normalize
from impl.timed_activities.timed_activity_rewards import TimedActivityRewards

sys.path.append('/home/sfeygin/python/examples/rllab/')

from ext.inverse_rl.algos.irl_trpo import IRLTRPO
from ext.inverse_rl.models.imitation_learning import GAIL
from ext.inverse_rl.utils.log_utils import rllab_logdir, load_latest_experts
from impl.timed_activities.timed_activity_env_md import TimedActivityEnv
from impl.timed_activities.timed_activity_mdp import TimedActivityMDP

sys.path.append('../../da-irl')
sys.path.append('../../da-irl/src')
sys.path.append('../../common')

from rllab.envs.gym_env import GymEnv
# Local
# std lib
import datetime
import json
import uuid
import pickle
import dateutil

# third party
import gym
import tensorflow as tf
import pandas as pd, numpy as np

# swl
from swlcommon.personatrainer.loaders.trace_loader import TraceLoader
from swlcommon.personatrainer.persona import Persona

# da-irl
from src.impl.expert_persona import PersonaSites

from src.impl.activity_config import ATPConfig

from sandbox.rocky.tf.exploration_strategies.epsilon_greedy_strategy import \
    EpsilonGreedyStrategy
from sandbox.rocky.tf.q_functions.stochastic_discrete_mlp_q_function import \
    StochasticDiscreteMLPQFunction
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.categorical_mlp_policy import \
    CategoricalMLPPolicy

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.baselines.q_baseline import QfunctionBaseline


def tp(groups, df, idx):
    group = list(groups.values())[idx]
    uid_df = TraceLoader.load_traces_from_df(df.iloc[group])
    return Persona(traces=uid_df, build_profile=True,
                   config_file='../data/misc/initial_profile_builder_config'
                               '.json')


if __name__ == '__main__':

    # Args:

    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_ckpt_name', type=str, default=None)
    parser.add_argument('--policy_ckpt_itr', type=int, default=1)

    args = parser.parse_args()

    # Inputs
    # TODO: move to args
    df = pd.read_parquet("/home/kat/data/micro_subset.parquet",
                         engine="fastparquet")

    config_file = "/home/sfeygin/python/da-irl/data/misc" \
                  "/IRL_GAN_GCL_durative_actions_scenario.json"

    with open('/home/sfeygin/python/da-irl/notebooks/states.pkl', 'rb') as f:
        states = pickle.load(f)

    with open('/home/sfeygin/python/da-irl/notebooks/actions.pkl', 'rb') as f:
        actions = pickle.load(f)

    experts = load_latest_experts(
        '/home/sfeygin/python/da-irl/notebooks/data/timed_env/', n=1)

    with open(config_file) as fp:
        config = ATPConfig(data=json.load(fp))

    groups = df.groupby('uid').groups
    persona = [tp(groups, df, 15) for _ in range(1)][0]
    persona_sites = PersonaSites(persona)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    # avoid name clashes when running distributed jobs
    rand_id = str(uuid.uuid4())[:5]
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')

    default_exp_name = '_%s_%s' % (timestamp, rand_id)

    features = TimedActivityRewards.default_features(config, persona_sites)

    theta = np.array([0.4, 0.7,
                      0.1, 0.6, 1.0, 0.9,
                      0.7, 0.7, 0.7, 0.7, 0.6, 0.6, 0.4,
                      0.6, 0.6, 0.6, 0.6])

    reward_function = TimedActivityRewards(features=features, theta=theta)

    timed_activity_mdp = TimedActivityMDP(states, actions, reward_function,
                                          0.95, config)
    gym.envs.register(
        id='dairl-v0',
        entry_point='src.impl.timed_activities.timed_activity_env_md'
                    ':TimedActivityEnv',
        timestep_limit=12,
        kwargs={'mdp': timed_activity_mdp}
    )

    env = TimedActivityEnv(timed_activity_mdp)
    for ep in experts:
        print("\n")
        ns = env.representation_to_state(env.reset())
        d = False
        r = 0.0
        rewards = []
        print("ns:{}".format(ns, r, d))
        states, actions = ep['observations'], ep['actions']
        for state, action in zip(states, actions):
            print("e_ns:{}, e_a:{}".format(env.representation_to_state(state),
                                           env.action_id_map[action]))
            ns, r, d, _ = env.step(action)
            ns = env.representation_to_state(ns)
            print("ns:{}, r:{}, d:{}".format(ns, r, d))
            rewards.append(r)

    env = TfEnv(normalize(GymEnv('dairl-v0', record_video=False,
                                     record_log=False),
                              normalize_reward=False,
                              normalize_obs=False))

    if args.policy_ckpt_name is None:
        policy = CategoricalMLPPolicy(name='policy', env_spec=env.spec,
                                      hidden_sizes=(512, 256, 128, 64, 32),
                                      hidden_nonlinearity=tf.nn.tanh)
    else:
        with tf.Session() as sess:
            with tf.variable_scope('load_policy'):
                out = joblib.load('{}/itr_{}.pkl'.format(args.policy_ckpt_name,
                                                         args.policy_ckpt_itr))
                policy = out['policy']

    # policy = CategoricalLSTMPolicy(name='policy',env_spec=env.spec,
    #                                hidden_dim=128,
    #                                hidden_nonlinearity=tf.nn.relu)

    qf = StochasticDiscreteMLPQFunction(env_spec=env.spec)
    qf_baseline = QfunctionBaseline(env_spec=env.spec,
                                    policy=policy, qf=qf)
    es = EpsilonGreedyStrategy(env_spec=env.spec)
    irl_model = GAIL(env_spec=env.spec, expert_trajs=experts,
                     name='gan_gcl')

    # Baseline Init:
    nonlinearity = tf.nn.elu
    b_hspec = (128, 128)
    baseline = BaselineMLP(name='mlp_baseline',
                           output_dim=1,
                           hidden_sizes=b_hspec,
                           hidden_nonlinearity=nonlinearity,
                           output_nonlinearity=None,
                           input_shape=(
                               (np.sum([component.n for component in
                                        env.spec.observation_space.components]),)
                            )
                           )
    baseline.initialize_optimizer()

    # algo = TRPO(
    #     env=env,
    #     policy=policy,
    #     n_itr=200,
    #     batch_size=1000,
    #     max_path_length=9,
    #     discount=0.80,
    #     store_paths=False,
    #     baseline=LinearFeatureBaseline(env_spec=env.spec),
    #     optimizer=ConjugateGradientOptimizer(
    #                 hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
    # )

    algo = IRLTRPO(
        env=env,
        policy=policy,
        irl_model=irl_model,
        n_itr=1000,
        batch_size=1000,
        max_path_length=100,
        discount=0.99,
        step_size=0.01,
        store_paths=True,
        discrim_train_itrs=100,
        irl_model_wt=1.0,
        entropy_weight=0.00,
        fixed_horizon=False,
        # GAIL should not use entropy unless for exploration
        zero_environment_reward=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec),
        # optimizer=ConjugateGradientOptimizer(
        #                     hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
    )

    with rllab_logdir(algo=algo,
                      dirname='data/timed_env/{}'.format(
                          default_exp_name), snapshot_mode='all',
                      snapshot_gap='3'):
        with tf.Session():
            algo.train()
            print("done!")
    #

    # algo = TRPO(
    #     env=env,
    #     policy=policy,
    #     n_itr=200,
    #     batch_size=1000,
    #     max_path_length=9,
    #     discount=0.80,
    #     store_paths=False,
    #     baseline=baseline,
    #     optimizer=ConjugateGradientOptimizer(
    #                 hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
    # )

    # def runner(exp_name, discount, ent_weight):
    #     env = TfEnv(GymEnv('dairl-v0', record_video=False, record_log=False))
    #
    #     irl_model = GAN_GCL(env_spec=env.spec, expert_trajs=experts,
    #                         name='gan_gcl')
    #
    #     policy = CategoricalMLPPolicy(name='policy', env_spec=env.spec,
    #                                   hidden_sizes=(512, 256, 128, 64, 32),
    #                                   hidden_nonlinearity=tf.nn.relu)
    #
    #     exp_name = "{}_disc-{}_entwt-{}".format(exp_name, discount,
    #                                             ent_weight)
    #     algo = IRLTRPO(
    #         env=env,
    #         policy=policy,
    #         irl_model=irl_model,
    #         n_itr=500,
    #         batch_size=1000,
    #         max_path_length=15,
    #         discount=discount,
    #         store_paths=True,
    #         discrim_train_itrs=200,
    #         irl_model_wt=1.0,
    #         entropy_weight=ent_weight,
    #         fixed_horizon=False,
    #         # GAIL should not use entropy unless for exploration
    #         zero_environment_reward=False,
    #         baseline=LinearFeatureBaseline(env_spec=env.spec),
    #         optimizer=ConjugateGradientOptimizer(
    #             hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)
    #         )
    #     )
    #
    #     with rllab_logdir(algo=algo,
    #                       dirname='data/timed_env/{}'.format(exp_name)):
    #         with tf.Session():
    #             algo.train()
    #         print("done!")
    #
    #
    # sweep_op = {
    #     'discount': [0.7, 0.8],
    #     'ent_weight': [0.01, 0.1],
    # }

    # run_sweep_serial(runner, sweep_op, repeat=1)

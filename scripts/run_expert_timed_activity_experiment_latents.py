# Default
import argparse
import sys

sys.path.append('../../da-irl')
sys.path.append('../../da-irl/src')
sys.path.append('../../common')
sys.path.append('/home/sfeygin/python/examples/rllab/')

from ext.infogail.categorical_latent_var_mlp_policy import \
    CategoricalLatentVarMLPPolicy
from ext.inverse_rl.models.architectures import feedforward_energy
from ext.infogail.latent_sampler import UniformlyRandomLatentSampler
from ext.infogail.scheduler import ConstantIntervalScheduler
from ext.inverse_rl.utils.hyper_sweep import run_sweep_serial


from src.impl.timed_activities.rllab_misc import normalize, BaselineMLP
from src.impl.timed_activities.timed_activity_rewards import \
    TimedActivityRewards

from ext.inverse_rl.algos.irl_trpo import IRLTRPO
from ext.inverse_rl.models.imitation_learning import GAIL
from ext.inverse_rl.utils.log_utils import rllab_logdir, load_latest_experts
from src.impl.timed_activities.timed_activity_env_md import TimedActivityEnv
from src.impl.timed_activities.timed_activity_mdp import TimedActivityMDP

from rllab.envs.gym_env import GymEnv
# Local
# std lib
import datetime
import json
import uuid
import pickle
import dateutil
import math
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

from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.categorical_mlp_policy import \
    CategoricalMLPPolicy
from sandbox.rocky.tf.policies.categorical_lstm_policy import \
    CategoricalLSTMPolicy
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import \
    ConjugateGradientOptimizer, FiniteDifferenceHvp
from sandbox.rocky.tf.core.network import MLP


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
    parser.add_argument('--debug', action="store_true")

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

    num_experts = 5
    experts = load_latest_experts(
        '/home/sfeygin/python/da-irl/notebooks/data/timed_env/', n=num_experts)
    latent_dim = num_experts

    timestep_limit = max(np.array([len(path['observations']) for \
                                   path in
                                   experts]))+1
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

    reward_function = TimedActivityRewards(features, states, actions,
                                           theta=theta)

    timed_activity_mdp = TimedActivityMDP(states, actions, reward_function,
                                          0.95, config)

    time_step_limit = timestep_limit

    gym.envs.register(
        id='dairl-v0',
        entry_point='src.impl.timed_activities.timed_activity_env_md'
                    ':TimedActivityEnv',
        timestep_limit=time_step_limit,
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
        for idx, action in enumerate(actions):
            print("e_ns:{}, e_a:{}".format(env.representation_to_state(
                states[idx]),
                env.action_id_map[action]))
            ns, r, d, _ = env.step(action)
            ns = env.representation_to_state(ns)
            print("ns:{}, r:{}, d:{}".format(ns, r, d))
            rewards.append(r)



    def runner(exp_name, policy_type, ent_weight, discount):
        with tf.Session() as sess:
            env = TfEnv(normalize(GymEnv('dairl-v0', record_video=False,
                                         record_log=False),
                                  normalize_reward=False,
                                  normalize_obs=False))

            if policy_type is 'latent_mlp':
                scheduler_k = time_step_limit
                latent_sampler = UniformlyRandomLatentSampler(
                    scheduler=ConstantIntervalScheduler(k=scheduler_k),
                    name='latent_sampler',
                    dim=latent_dim
                )
                policy = CategoricalLatentVarMLPPolicy(policy_name="policy",
                                                       latent_sampler=latent_sampler,
                                                       env_spec=env.spec,
                                                       hidden_sizes=(64, 128, 512),
                                                       )
            else:
                policy = CategoricalMLPPolicy(name='policy',
                                              env_spec=env.spec,
                                              hidden_sizes=(64, 128, 512),
                                              )

            discrim_arch = feedforward_energy

            irl_model = GAIL(env=env, expert_trajs=experts,
                             discrim_arch=discrim_arch,
                             max_length=time_step_limit,
                             latent_dim=latent_dim,
                             name='gail')

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
                                                env.spec.observation_space.components]),))
                                   )
            baseline.initialize_optimizer()
            algo = IRLTRPO(
                env=env,
                policy=policy,
                irl_model=irl_model,
                n_itr=300,
                batch_size=3000,
                max_path_length=time_step_limit,
                discount=discount,
                step_size=0.01,
                store_paths=False,
                discrim_train_itrs=200,
                irl_model_wt=1.0,
                entropy_weight=ent_weight,
                fixed_horizon=True,
                # GAIL should not use entropy unless for exploration
                zero_environment_reward=True,
                baseline=baseline,
                summary_dir='data/latent/'
            )

            with rllab_logdir(algo=algo,
                              dirname='data/latent/{}'.format(exp_name),
                              snapshot_mode='gap',
                              snapshot_gap=20):

                algo.train(args.debug)
                print("done!")


    sweep_op = {
        'policy_type': ['latent_mlp','mlp'],
        'discount': [1.0, 0.99, 0.9],
        'ent_weight': [0.0, 0.0001, 0.01],
    }

    run_sweep_serial(runner, sweep_op, repeat=5)

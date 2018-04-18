# Local
# std lib
import datetime
import json
import pickle
import uuid
from pprint import pprint

import dateutil
# third party
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix
from scipy.special import logsumexp as sp_lse
# swl
from swlcommon.personatrainer.loaders.trace_loader import TraceLoader
from swlcommon.personatrainer.persona import Persona


# da-irl


def tp(groups, df, idx):
    group = list(groups.values())[idx]
    uid_df = TraceLoader.load_traces_from_df(df.iloc[group])
    return Persona(traces=uid_df, build_profile=True,
                   config_file='../data/misc/initial_profile_builder_config'
                               '.json')


def get_tmat(mdp):
    dim_S, dim_A = len(mdp.S), len(mdp.A)
    tmat = {}
    for a in range(dim_A):
        ss_mat = dok_matrix((dim_S, dim_S))
        for s in range(dim_S):
            p, ns = mdp.T(s, a)[0]
            if p != 0:
                ss_mat[s, ns.state_id] = 1.0
        tmat[a] = ss_mat
    return tmat


def get_feat_mat(mdp):
    dim_S, dim_A = len(mdp.S), len(mdp.A)
    feat_mat = np.zeros((dim_S, dim_A, mdp.reward_function.dim_phi))
    for state in mdp.S.values():
        for action in mdp.A.values():
            feat_mat[state.state_id, action.action_id, :] = np.squeeze(
                mdp.reward_function.phi(state, action))
    return feat_mat


def get_reward_mat(mdp):
    dim_S, dim_A = len(mdp.S), len(mdp.A)
    reward_matrix = np.zeros((dim_S, dim_A))
    for state in mdp.S.values():
        for action in mdp.A.values():
            reward_matrix[state.state_id, action.action_id] = mdp.R(state,
                                                                    action)
    return reward_matrix


def logsumexp(q, alpha=1.0, axis=1):
    return alpha * sp_lse((1.0 / alpha) * q, axis=axis)


def q_iteration(env, reward_matrix=None, tmat=None, K=50, gamma=0.99,
                ent_wt=1.0, warmstart_q=None, policy=None):
    mdp = env.mdp
    dim_S, dim_A = len(mdp.S), len(mdp.A)
    if tmat is None:
        tmat = get_tmat(mdp)

    if reward_matrix is None:
        reward_matrix = get_reward_mat(env.mdp)
    if warmstart_q is None:
        q_fn = np.zeros((dim_S, dim_A))
    else:
        q_fn = warmstart_q

    old_diff = 0
    for k in range(K):
        if policy is None:
            v_fn = logsumexp(q_fn, alpha=ent_wt)
        else:
            v_fn = np.sum((q_fn - np.log(policy)) * policy, axis=1)
        new_q = q_fn.copy()
        for a in mdp.A:
            new_q[:, a] = reward_matrix[:, a] + gamma * tmat[a].tocsr().dot(
                v_fn)
        diff = np.linalg.norm(q_fn - new_q)
        if np.isclose(old_diff, diff):
            print(diff)
            break
        old_diff = diff
        q_fn = new_q

    return q_fn


def get_policy(q_fn, ent_wt=1.0):
    """
    Return a policy by normalizing a Q-function
    """
    v_rew = logsumexp(q_fn, alpha=ent_wt)
    adv_rew = q_fn - np.expand_dims(v_rew, axis=1)
    pol_probs = np.exp((1.0 / ent_wt) * adv_rew)
    assert np.all(np.isclose(np.sum(pol_probs, axis=1), 1.0)), str(pol_probs)
    return pol_probs


def compute_pol_probs(env, tmat, q_fn, T=50, ent_wt=1.0):
    mdp = env.mdp
    dim_S, dim_A = len(mdp.S), len(mdp.A)
    sa_visit_t = np.zeros((dim_S, dim_A, T))
    pol_probs = get_policy(q_fn, ent_wt=ent_wt)
    state_visitation = np.zeros((dim_S, 1))
    state_visitation[env.representation_to_state(env.reset()).state_id] = 1

    for i in range(T):
        #         print(i)
        sa_visit = state_visitation * pol_probs
        sa_visit_t[:, :, i] = sa_visit
        # sum-out (SA)S
        new_state_visitation = np.zeros_like(state_visitation)
        for a in mdp.A:
            new_state_visitation += np.expand_dims(
                tmat[a].tocsr().dot(sa_visit[:, a]), 1)
        state_visitation = new_state_visitation
    return np.sum(sa_visit_t, axis=2) / float(T)


def compute_visitation_demos(env, demos):
    mdp = env.mdp
    dim_S, dim_A = len(mdp.S), len(mdp.A)
    counts = np.zeros((dim_S, dim_A))

    for demo in demos:
        obs = demo['observations']
        act = demo['actions']
        state_ids = [env.representation_to_state(o).state_id for o in obs]

        for act_id, state_id in zip(act, state_ids):
            counts[state_id, act_id] += 1
    return counts / float(np.sum(counts))


def adam_optimizer(lr, beta1=0.9, beta2=0.999, eps=1e-8):
    itr = 0
    pm = None
    pv = None

    def update(x, grad):
        nonlocal itr, lr, pm, pv
        if pm is None:
            pm = np.zeros_like(grad)
            pv = np.zeros_like(grad)

        pm = beta1 * pm + (1 - beta1) * grad
        pv = beta2 * pv + (1 - beta2) * (grad * grad)
        mhat = pm / (1 - beta1 ** (itr + 1))
        vhat = pv / (1 - beta2 ** (itr + 1))
        update_vec = mhat / (np.sqrt(vhat) + eps)
        new_x = x - lr * update_vec
        itr += 1
        return new_x

    return update


if __name__ == '__main__':
    import argparse
    import sys

    sys.path.append('/home/sfeygin/python/da-irl')
    sys.path.append('/home/sfeygin/python/da-irl/src')
    sys.path.append('/home/sfeygin/python/common')
    sys.path.append('/home/sfeygin/python/examples/rllab')

    from src.impl.expert_persona import PersonaSites

    from src.impl.activity_config import ATPConfig
    from src.ext.inverse_rl.utils.log_utils import load_latest_experts
    from src.impl.timed_activities.timed_activity_env_md import TimedActivityEnv
    from src.impl.timed_activities.timed_activity_mdp import TimedActivityMDP
    from src.impl.timed_activities.timed_activity_rewards import \
        TimedActivityRewards

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

    experts = load_latest_experts(
        '/home/sfeygin/python/da-irl/notebooks/data/timed_env/', n=1)
    timestep_limit = max(np.array([len(path['observations']) for \
                                   path in
                                   experts]))
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
    mdp = env.mdp
    theta = np.random.normal(0, 1.0, size=17)
    with open('/home/sfeygin/python/da-irl/notebooks/tmat.pkl', 'rb') as f:
        tmat = pickle.load(f)
    fm = np.load('/home/sfeygin/python/da-irl/notebooks/fm.npy')

    mdp = env.mdp
    dim_S, dim_A = len(mdp.S), len(mdp.A)
    discount = 1.0
    q_rew = np.zeros((dim_S, dim_A))
    q_itrs = 5
    ent_wt = 1.0
    old_theta = theta.T
    mdp.reward_function.feature_matrix = fm

    theta_norms = []
    visitation_demos = compute_visitation_demos(env, experts)
    for i in range(50):
        reward_matrix = mdp.reward_function.get_rewards()
        q_rew = q_iteration(env, reward_matrix=reward_matrix, tmat=tmat,
                            ent_wt=ent_wt, warmstart_q=q_rew, K=100,
                            gamma=discount)

        grad = visitation_demos - compute_pol_probs(env, tmat, q_rew, T=100)
        mdp.reward_function.apply_grads(grad)
        theta = mdp.reward_function.theta

        # grad_sign = np.sign(theta - old_theta)
        old_theta = theta
        theta_norm = np.max(np.abs(theta))
        theta_norms.append(theta_norm)
        # print(pprint(list(
        #     zip(mdp.reward_function._features, list(zip(theta, grad_sign))))))
        print(theta_norms)
        plt.plot(theta_norms)
        plt.savefig('thetas/theta_norm.png')
        plt.show()
        np.save('thetas/theta_{}'.format(i), theta)


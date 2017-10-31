from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import matplotlib

from src.irl.meirl import MaxEntIRLAgent
from src.misc.utils import normalize

matplotlib.use("Agg")
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


class DeepMaxEntIRLAgent(MaxEntIRLAgent):
    """
    Refactor partially based on https://github.com/stormmax/irl-imitation/blob/master/deep_maxent_irl.py

    """

    def __init__(self, mdp, data):
        MaxEntIRLAgent.__init__(self, mdp, data)

    def svi(self, r, name="soft_vi"):
        with ops.name_scope(name, "soft_vi_pyfunc", [r]) as scope:
            svi_x = py_func(self.approximate_value_iteration,
                                         [r],
                                         [tf.float32],
                                         name=scope,
                                         grad=self._SVIGrad)
            return svi_x[0]

    def _SVIGrad(self, op, grad):
        return grad

    def esvf(self, p, name="esvf"):
        with ops.name_scope(name, "esvf_pyfunc", [p]) as scope:
            esvf_x = py_func(self.state_visitation_frequency,
                                          [p],
                                          [tf.float32],
                                          name=scope,
                                          grad=self._ESVFGrad)
            return esvf_x[0]

    def _ESVFGrad(self, op, grad):
        return tf.transpose(grad)

    def loss_func(self, r):
        pi = self.approximate_value_iteration(r)
        print( "Soft value iteration done, finding state visitation frequency...")
        expected_svf = self.state_visitation_frequency(pi)
        savf = self.get_empirical_savf()

        D_sa = np.zeros((self.nS, self.nA), dtype=np.float32)
        for s in self.mdp.S:
            actions = self.mdp.env.states[s].available_actions
            for a in actions:
                D_sa[s, a] = pi[s, a] * expected_svf[s]

        diff = savf - D_sa

        loss = savf * np.log(pi)
        fd = np.mean(diff)
        self.feature_diff.append(fd)
        print (diff)
        return loss, diff

    def LD(self, r, name="data_loss"):
        with ops.name_scope(name, "data_loss", [r]) as scope:
            ld_r = py_func(self.loss_func,
                                        [r],
                                        [tf.float32, tf.float32],
                                        name=scope,
                                        grad=self._LDGrad)
            return ld_r[0], ld_r[1]

    def _LDGrad(self, op, *grads):
        """
        op: The `LD` `Operation` that we are differentiating, which we can use
          to find the inputs and outputs of the original op.
        grad: Gradient with respect to the output of the `LD` op.

        Returns:
            Gradients with respect to the input of `LD`
        """
        r = op.inputs[0]
        svf_diff = op.outputs[1]
        return [grads[1]]

    def learn_rewards_and_weights(self, N, learning_rate=0.08, minibatch_size=1, initial_theta=None, reg=0.01,
                                  cache_dir=None):

        nn_r = self.mdp.reward

        nS = self.mdp.env.nS
        nA = self.mdp.env.nA

        fm = nn_r.feature_matrix.reshape((nS * nA, self.mdp.reward.dim_ss))

        self._current_examples = self.demos

        mu_D = self.get_empirical_savf()

        for i in xrange(N):
            reward = nn_r.get_rewards(fm).reshape((nS, nA))

            policy = self.approximate_value_iteration(reward)

            mu_exp = self.state_visitation_frequency(policy)

            D_sa = np.zeros((self.nS, self.nA), dtype=np.float32)
            for s in self.mdp.S:
                actions = self.mdp.env.states[s].available_actions
                for a in actions:
                        D_sa[s, a] = policy[s, a] * mu_exp[s]

            grad_r = mu_D - D_sa

            grad_theta, l2_loss, grad_norm = nn_r.apply_grads(fm, grad_r.reshape((nS*nA)))

            self.feature_diff.append(l2_loss)

            print (l2_loss)

        rewards = nn_r.get_rewards(fm).reshape((nS, nA))
        return normalize(rewards)



def demo_svf(trajs, n_states):
    """
    compute state visitation frequences from demonstrations

    input:
      trajs   list of list of Steps - collected from expert
    returns:
      p       Nx1 vector - state visitation frequences
    """

    p = np.zeros(n_states)
    for traj in trajs:
        for step in traj:
            p[step[0]] += 1
    p = p / len(trajs)
    return p


def compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=True):
    """compute the expected states visition frequency p(s| theta, T)
    using dynamic programming
    inputs:
      P_a     NxNxN_ACTIONS matrix - transition dynamics
      gamma   float - discount factor
      trajs   list of list of Steps - collected from expert
      policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy

    returns:
      p       Nx1 vector - state visitation frequencies
    """
    N_STATES, _, N_ACTIONS = np.shape(P_a)

    T = len(trajs[0])
    # mu[s, t] is the prob of visiting state s at time t
    mu = np.zeros([N_STATES, T])

    for traj in trajs:
        mu[traj[0].cur_state, 0] += 1
    mu[:, 0] = mu[:, 0] / len(trajs)

    for s in range(N_STATES):
        for t in range(T - 1):
            if deterministic:
                mu[s, t + 1] = sum([mu[pre_s, t] * P_a[pre_s, s, int(policy[pre_s])] for pre_s in range(N_STATES)])
            else:
                mu[s, t + 1] = sum(
                    [sum([mu[pre_s, t] * P_a[pre_s, s, a1] * policy[pre_s, a1] for a1 in range(N_ACTIONS)]) for pre_s in
                     range(N_STATES)])
    p = np.sum(mu, 1)
    return p


# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, int(1E+8)))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)
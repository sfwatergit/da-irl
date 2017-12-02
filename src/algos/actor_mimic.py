from abc import ABCMeta

import numpy as np
import six
import tensorflow as tf

from src.algos.maxent_irl import MaxEntIRL
from src.util.math_utils import from_onehot_n, from_onehot
from src.util.tf_utils import fc_net

EPSILON = 0.5


class ATPActorMimicIRL(six.with_metaclass(ABCMeta, MaxEntIRL)):
    def __init__(self, mdp, experts, nn_params=None):
        super(ATPActorMimicIRL, self).__init__(mdp)
        self.sess = tf.get_default_session()

        self.mdp = mdp  # task dynamics
        self.env = mdp.env  # task environment
        self.experts = experts  # expert learning agents

        # input: [batch_size, obs], # out: [batch_size, dim_actions]
        self.obs_ph = tf.placeholder(tf.float32, [None, self.nS], name='obs_ph')  # observations placeholder
        self.labels = tf.placeholder(tf.float32, [None, self.nA], name='labels')  # target (expert) actions
        self.lr = tf.placeholder(tf.float32, (), name='lr')

        logits = fc_net(self.obs_ph, n_layers=1, dim_hidden=32, dim_out=self.nA, act=tf.nn.elu, out_act=None)  # AMN net

        # take softmax over logits for output layer
        self.amn_policy_predictions = tf.nn.softmax(logits)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels))
        self.step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.use_amn_policy = False

    def evaluate_amn_policy(self, obs):
        return self.sess.run(self.amn_policy_predictions, feed_dict={self.obs_ph: obs})[0]

    def train_amn(self):
        #  for each available action from the state
        for expert in self.experts:
            pi_e = expert['policy'][0]
            # obs, acts, rews = self.sample_amn_policy(2)
            # obs = obs.reshape(-1, self.nS)
            obs = np.array([self.env.state_to_obs(self.env.states[s]) for s in
                            np.random.choice(np.arange(self.nS), 500)])
            obs = obs.reshape(-1, self.nS)
            labels = self.get_act_for_policy(pi_e, obs)  # action labels
            loss, _ = self.sess.run([self.loss, self.step], feed_dict={
                self.obs_ph: obs,
                self.labels: labels,
                self.lr: 0.01
            })
            print(loss)

    def state_action_visitation_frequency(self):
        if self.use_amn_policy:
            transition_matrix = self.mdp.transition_matrix
            T = self.env.horizon
            path_states = np.array([trajectory[0] for trajectory in self.expert_demos])
            start_states = np.array([state[0] for state in path_states])
            start_state_count = np.bincount(start_states.astype(int), minlength=self.nS)
            start_state_dist = start_state_count.astype(float) / len(start_states)
            mu = np.tile(start_state_dist, (T, 1)).T

            self._policy = np.array([self.evaluate_amn_policy([self.env.state_to_obs(s)]) for s in np.arange(self.nS)])
            for t in range(1, T):
                mu[:, t] = 0
                for s in range(self.nS):
                    for a in range(self.nA):
                        sp = np.flatnonzero(transition_matrix[s, a])  # deterministic optimization
                        if len(sp) > 0:
                            mu[sp, t] += mu[s, t - 1] * self._policy[s, a]
            self.use_amn_policy = False
            mu_exp = np.sum(mu, 1)
        else:
            mu_exp = self.state_action_visitation_frequency()
        return mu_exp

    def legal_actions(self, obs):
        return self.env.states[from_onehot(obs)].available_actions

    def get_act_for_policy(self, pi_e, obs):
        obs_num = from_onehot_n(obs)
        acts = [pi_e[o] for o in obs_num]
        return acts

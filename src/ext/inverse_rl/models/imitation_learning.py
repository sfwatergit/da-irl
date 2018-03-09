import functools
import sys

sys.path.append('/home/sfeygin/python/examples/rllab/')
import numpy as np
import tensorflow as tf

from ext.inverse_rl.models.architectures import feedforward_energy, relu_net
from ext.inverse_rl.models.tf_util import discounted_reduce_sum
from ext.inverse_rl.utils.general import TrainingIterator
from ext.inverse_rl.utils.hyperparametrized import Hyperparametrized
from ext.inverse_rl.utils.math_utils import gauss_log_pdf, categorical_log_pdf
from baselines.common.mpi_running_mean_std import RunningMeanStd
from sandbox.rocky.tf.misc import tensor_utils

LOG_REG = 1e-8
DIST_GAUSSIAN = 'gaussian'
DIST_CATEGORICAL = 'categorical'


def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return wrapper


class ImitationLearning(object, metaclass=Hyperparametrized):
    def __init__(self):
        pass

    @staticmethod
    def _compute_path_probs(paths, pol_dist_type=None, insert=True,
                            insert_key='a_logprobs'):
        """
        Returns a N x T matrix of action probabilities
        """
        if pol_dist_type is None:
            # try to  infer distribution type
            path0 = paths[0]
            if 'log_std' in path0['agent_infos']:
                pol_dist_type = DIST_GAUSSIAN
            elif 'prob' in path0['agent_infos']:
                pol_dist_type = DIST_CATEGORICAL
            else:
                raise NotImplementedError()

        # compute path probs
        Npath = len(paths)
        actions = [path['actions'] for path in paths]
        if pol_dist_type == DIST_GAUSSIAN:
            params = [
                (path['agent_infos']['mean'], path['agent_infos']['log_std'])
                for path in paths]
            path_probs = [gauss_log_pdf(params[i], actions[i]) for i in
                          range(Npath)]
        elif pol_dist_type == DIST_CATEGORICAL:
            params = [(path['agent_infos']['prob'],) for path in paths]
            path_probs = [categorical_log_pdf(params[i], actions[i]) for i in
                          range(Npath)]
        else:
            raise NotImplementedError("Unknown distribution type")

        if insert:
            for i, path in enumerate(paths):
                path[insert_key] = path_probs[i]

        return np.array(path_probs)

    @staticmethod
    def extract_paths(paths, keys=('observations', 'actions'), stack=True):
        if stack:
            return [np.stack([t[key] for t in paths]).astype(np.float32) for key
                    in keys]
        else:
            return [np.concatenate([t[key] for t in paths]).astype(np.float32)
                    for key in keys]

    @staticmethod
    def sample_batch(*args, batch_size=32):
        N = args[0].shape[0]
        batch_idxs = np.random.randint(0, N,
                                       batch_size)  # trajectories are negatives
        return [data[batch_idxs] for data in args]

    def fit(self, paths, **kwargs):
        raise NotImplementedError()

    def eval(self, paths, **kwargs):
        raise NotImplementedError()

    def _make_param_ops(self, vs):
        self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope=vs.name)
        assert len(self._params) > 0
        self._assign_plc = [tf.placeholder(tf.float32, shape=param.get_shape(),
                                           name='assign_%s' %
                                                param.name.replace(
                                                    '/', '_').replace(':', '_'))
                            for
                            param in self._params]
        self._assign_ops = [tf.assign(self._params[i], self._assign_plc[i]) for
                            i in range(len(self._params))]

    def get_params(self):
        params = tf.get_default_session().run(self._params)
        assert len(params) == len(self._params)
        return params

    def set_params(self, params):
        tf.get_default_session().run(self._assign_ops, feed_dict={
            self._assign_plc[i]: params[i] for i in range(len(self._params))
        })


class TrajectoryIRL(ImitationLearning):
    """
    Base class for models that score entire trajectories at once
    """

    @property
    def score_trajectories(self):
        return True


def logsigmoid(a):
    '''Equivalent to tf.log(tf.sigmoid(a))'''
    return -tf.nn.softplus(-a)


def logit_bernoulli_entropy(logits):
    ent = (1. - tf.nn.sigmoid(logits)) * logits - logsigmoid(logits)
    return ent


class SingleTimestepIRL(ImitationLearning):
    """
    Base class for models that score single timesteps at once
    """

    @staticmethod
    def extract_paths(paths, keys=('observations', 'actions'), stack=False):
        return ImitationLearning.extract_paths(paths, keys=keys, stack=stack)

    @staticmethod
    def unpack(data, paths):
        lengths = [path['observations'].shape[0] for path in paths]
        unpacked = []
        idx = 0
        for l in lengths:
            unpacked.append(data[idx:idx + l])
            idx += l
        return unpacked

    @property
    def score_trajectories(self):
        return False


class GAIL(TrajectoryIRL):
    """
    Generative adverserial imitation learning
    See https://arxiv.org/pdf/1606.03476.pdf

    This version consumes single timesteps.
    """

    def __init__(self, env_spec, expert_trajs=None,
                 discrim_arch=feedforward_energy,
                 discrim_arch_args={},
                 name='gail'):
        super(GAIL, self).__init__()
        self.dO = env_spec.observation_space.flat_dim
        self.dU = env_spec.action_space.flat_dim
        self.env_spec = env_spec
        self.expert_trajs = expert_trajs
        self.scope = name

        # build energy model
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE ) as vs:
            # Should be batch_size x T x dO/dU
            self.expert_obs_t = tf.placeholder(tf.float32, [None, None,
                                                            self.dO],
                                               name='expert_obs')
            with tf.variable_scope("obfilter"):
                self.obs_rms = RunningMeanStd(shape=self.dO)

            expert_obs = (self.expert_obs_t - self.obs_rms.mean /
                        self.obs_rms.std)

            self.expert_act_t = tf.placeholder(tf.float32, [None, None,
                                                            self.dU],
                                               name='expert_act')

            self.gen_obs_t = tf.placeholder(tf.float32, [None, None, self.dO],
                                            name='gen_obs')
            self.gen_act_t = tf.placeholder(tf.float32, [None, None, self.dU],
                                            name='gen_act')

            with tf.variable_scope("obfilter"):
                self.gen_obs_rms = RunningMeanStd(shape=self.dO)

            gen_obs = (self.gen_obs_t - self.gen_obs_rms.mean /
                        self.gen_obs_rms.std)


            self.lr = tf.placeholder(tf.float32, (), name='lr')

            expert_obs_act = tf.concat([expert_obs,
                                        self.expert_act_t], axis=2)

            gen_obs_act = tf.concat([gen_obs, self.gen_act_t],
                                    axis=2)

            expert_logits = self.build_graph(expert_obs_act, True)
            expert_mask = tf.sign(
                tf.reduce_max(tf.abs(expert_logits), reduction_indices=1))
            expert_logits *= expert_mask
            generator_logits = self.build_graph(gen_obs_act, False)
            generator_mask = tf.sign(
                tf.reduce_max(tf.abs(generator_logits), reduction_indices=1))
            generator_logits *= generator_mask
            generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=generator_logits, labels=tf.zeros_like(generator_logits))
            generator_loss = tf.reduce_mean(generator_loss)
            expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=expert_logits, labels=tf.ones_like(expert_logits))
            expert_loss = tf.reduce_mean(expert_loss)

            logits = tf.concat([generator_logits, expert_logits], 0)
            entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
            entropy_loss = -0.001 * entropy
            generator_acc = tf.reduce_mean(
                tf.to_float(tf.nn.sigmoid(generator_logits) < 0.5))
            expert_acc = tf.reduce_mean(
                tf.to_float(tf.nn.sigmoid(expert_logits) > 0.5))

            self.predictions = -tf.log(1 - tf.nn.sigmoid(generator_logits) +
                                       1e-8)

            self.loss = generator_loss + expert_loss + generator_acc + \
                        expert_acc + entropy + entropy_loss
            self.step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
                self.loss)
            self._make_param_ops(vs)

    def get_trainable_variables(self, vs):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, vs)



    def build_graph(self, _input, reuse=False):

        hidden_size = 128
        #  transition
        p_h1 = tf.contrib.layers.fully_connected(_input, hidden_size,
                                                 activation_fn=tf.nn.tanh)
        p_h2 = tf.contrib.layers.fully_connected(p_h1, hidden_size,
                                                 activation_fn=tf.nn.tanh)
        output = tf.reshape(p_h2, [-1, hidden_size])
        logits = tf.contrib.layers.fully_connected(output, 1,
                                                   activation_fn=tf.identity)
        return logits

    def fit(self, paths, batch_size=32, max_itrs=100, **kwargs):
        # obs, acts = self.extract_paths(paths)
        gen_max_action_path_length = max([len(path['actions'])
                                          for
                                          path in
                                          paths])

        gen_max_obs_path_length = max([len(path["observations"]) for path in
                                       paths])

        expert_max_action_path_length = max([len(path['actions'])
                                             for
                                             path in
                                             self.expert_trajs])

        expert_max_obs_path_length = max([len(path["observations"]) for path in
                                          self.expert_trajs])

        max_obs_path_length = max(gen_max_obs_path_length,
                                  expert_max_obs_path_length)
        max_action_path_length = max(gen_max_action_path_length,
                                     expert_max_action_path_length)

        obs = np.stack(tensor_utils.pad_tensor_n([path["observations"]
                                          for path in
                                          paths],
                                         max_obs_path_length))

        acts = np.stack(tensor_utils.pad_tensor_n([path["actions"]
                                                   for path in
                                                   paths],
                                                  max_action_path_length))

        expert_obs = np.stack(tensor_utils.pad_tensor_n(
            [self.env_spec.observation_space.flatten_n(path["observations"])
             for path in
             self.expert_trajs], max_obs_path_length))

        expert_acts = np.stack(
tensor_utils.pad_tensor_n(
            [self.env_spec.action_space.flatten_n(path["actions"])
             for path in
             self.expert_trajs],max_action_path_length))

        # Train discriminator
        for it in TrainingIterator(max_itrs, heartbeat=5):
            obs_batch, act_batch = self.sample_batch(obs, acts,
                                                     batch_size=batch_size)
            expert_obs_batch, expert_act_batch = self.sample_batch(expert_obs,
                                                                   expert_acts,
                                                                   batch_size=batch_size)
            # obs_batch = np.concatenate([obs_batch, expert_obs_batch], axis=0)
            # act_batch = np.concatenate([act_batch, expert_act_batch], axis=0)

            loss, _ = tf.get_default_session().run([self.loss, self.step],
                                                   feed_dict={
                                                       self.expert_act_t:
                                                           expert_act_batch,
                                                       self.expert_obs_t:
                                                           expert_obs_batch,
                                                       self.gen_act_t:
                                                           act_batch,
                                                       self.gen_obs_t:
                                                           obs_batch,
                                                       self.lr: 1e-3
                                                   })

            it.record('loss', loss)
            if it.heartbeat:
                print(it.itr_message())
                mean_loss = it.pop_mean('loss')
                print('\tLoss:%f' % mean_loss)
        return mean_loss

    @staticmethod
    def unpack(data, paths):
        lengths = [path['observations'].shape[0] for path in paths]
        unpacked = []
        idx = 0
        for l in lengths:
            unpacked.append(data[idx:idx + l])
            idx += l
        return unpacked

    @staticmethod
    def pad_tensor_n(xs, max_len, obs=True):
        ret = np.zeros((len(xs), max_len) + xs[0].shape[1:], dtype=xs[0].dtype)
        if obs:
            ret[:, :, 12], ret[:, :, 122], ret[:, :, 124] = 1.0, 1.0, 1.0
        for idx, x in enumerate(xs):
            ret[idx][:len(x)] = x
        return ret

    def eval(self, paths, **kwargs):
        """
        Return bonus

        """

        gen_max_action_path_length = max([len(path['actions'])
                                          for
                                          path in
                                          paths])

        gen_max_obs_path_length = max([len(path["observations"]) for path in
                                       paths])

        max_path_length = max(gen_max_action_path_length,
                              gen_max_obs_path_length)

        obs = np.stack(self.pad_tensor_n([path["observations"]
                                          for path in
                                          paths],
                                         max_path_length))

        acts = np.stack(tensor_utils.pad_tensor_n([path["actions"]
                                                   for path in
                                                   paths],
                                                  max_path_length))
        if len(obs.shape) == 2:
            obs = np.expand_dims(obs, 0)
        if len(acts.shape) == 2:
            acts = np.expand_dims(acts, 0)

        scores = tf.get_default_session().run(self.predictions,
                                              feed_dict={
                                                  self.gen_obs_t: obs,
                                                  self.gen_act_t: acts})

        # reward = log D(s, a)
        scores = np.log(scores[:, 0] + LOG_REG)
        return scores


class AIRL(SingleTimestepIRL):
    """
    Similar to GAN_GCL except operates on single timesteps.
    """

    def __init__(self, env_spec, expert_trajs=None,
                 discrim_arch=relu_net,
                 discrim_arch_args={},
                 l2_reg=0,
                 discount=1.0,
                 name='gcl'):
        super(AIRL, self).__init__()
        self.dO = env_spec.observation_space.flat_dim
        self.dU = env_spec.action_space.flat_dim
        self.env_spec = env_spec

        if expert_trajs is not None:
            self.expert_trajs = expert_trajs
            self.expert_trajs_extracted = self.extract_paths(expert_trajs)

        # build energy model
        with tf.variable_scope(name) as _vs:
            # Should be batch_size x T x dO/dU
            self.obs_t = tf.placeholder(tf.float32, [None, self.dO], name='obs')
            self.act_t = tf.placeholder(tf.float32, [None, self.dU], name='act')
            self.labels = tf.placeholder(tf.float32, [None, 1], name='labels')
            self.lprobs = tf.placeholder(tf.float32, [None, 1],
                                         name='log_probs')
            self.lr = tf.placeholder(tf.float32, (), name='lr')

            obs_act = tf.concat([self.obs_t, self.act_t], axis=1)
            with tf.variable_scope('discrim') as dvs:
                with tf.variable_scope('energy'):
                    self.energy = discrim_arch(obs_act, **discrim_arch_args)
                # we do not learn a separate log Z(s) because it is
                # impossible to separate from the energy
                # In a discrete domain we can explicitly normalize to
                # calculate log Z(s)
                log_p_tau = -self.energy
                discrim_vars = tf.get_collection('reg_vars', scope=dvs.name)

            log_q_tau = self.lprobs

            if l2_reg > 0:
                reg_loss = l2_reg * tf.reduce_sum(
                    [tf.reduce_sum(tf.square(var)) for var in discrim_vars])
            else:
                reg_loss = 0

            log_pq = tf.reduce_logsumexp([log_p_tau, log_q_tau], axis=0)
            self.d_tau = tf.exp(log_p_tau - log_pq)
            cent_loss = -tf.reduce_mean(
                self.labels * (log_p_tau - log_pq) + (1 - self.labels) * (
                        log_q_tau - log_pq))

            self.loss = cent_loss + reg_loss
            self.step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
                self.loss)
            self._make_param_ops(_vs)

    def fit(self, paths, policy=None, batch_size=32, max_itrs=100, logger=None,
            lr=1e-3, **kwargs):
        # self._compute_path_probs(paths, insert=True)
        self.eval_expert_probs(paths, policy, insert=True)
        self.eval_expert_probs(self.expert_trajs, policy, insert=True)

        obs, acts, path_probs = self.extract_paths(paths, keys=(
            'observations', 'actions', 'a_logprobs'))
        expert_obs, expert_acts, expert_probs = self.extract_paths(
            self.expert_trajs, keys=('observations', 'actions', 'a_logprobs'))
        expert_obs = self.env_spec.observation_space.flatten_n(
            expert_obs.astype(int))
        expert_acts = self.env_spec.action_space.flatten_n(
            expert_acts.astype(int))
        # Train discriminator
        for it in TrainingIterator(max_itrs, heartbeat=5):
            obs_batch, act_batch, lprobs_batch = \
                self.sample_batch(obs, acts, path_probs, batch_size=batch_size)

            expert_obs_batch, expert_act_batch, expert_lprobs_batch = \
                self.sample_batch(expert_obs, expert_acts, expert_probs,
                                  batch_size=batch_size)

            labels = np.zeros((batch_size * 2, 1))
            labels[batch_size:] = 1.0
            obs_batch = np.concatenate([obs_batch, expert_obs_batch], axis=0)
            act_batch = np.concatenate([act_batch, expert_act_batch], axis=0)
            lprobs_batch = np.expand_dims(
                np.concatenate([lprobs_batch, expert_lprobs_batch], axis=0),
                axis=1).astype(np.float32)

            loss, _ = tf.get_default_session().run([self.loss, self.step],
                                                   feed_dict={
                                                       self.act_t: act_batch,
                                                       self.obs_t: obs_batch,
                                                       self.labels: labels,
                                                       self.lprobs:
                                                           lprobs_batch,
                                                       self.lr: lr
                                                   })

            it.record('loss', loss)
            if it.heartbeat:
                print(it.itr_message())
                mean_loss = it.pop_mean('loss')
                print('\tLoss:%f' % mean_loss)
        if logger:
            energy = tf.get_default_session().run(self.energy,
                                                  feed_dict={self.act_t: acts,
                                                             self.obs_t: obs})
            logger.record_tabular('IRLAverageEnergy', np.mean(energy))
            logger.record_tabular('IRLAverageLogQtau', np.mean(path_probs))
            logger.record_tabular('IRLMedianLogQtau', np.median(path_probs))

            energy = tf.get_default_session().run(self.energy,
                                                  feed_dict={
                                                      self.act_t: expert_acts,
                                                      self.obs_t: expert_obs})
            logger.record_tabular('IRLAverageExpertEnergy', np.mean(energy))
            logger.record_tabular('IRLAverageExpertLogQtau',
                                  np.mean(expert_probs))
            logger.record_tabular('IRLMedianExpertLogQtau',
                                  np.median(expert_probs))
        return mean_loss

    def eval(self, paths, **kwargs):
        """
        Return bonus
        """
        obs, acts = self.extract_paths(paths)

        energy = tf.get_default_session().run(self.energy,
                                              feed_dict={self.act_t: acts,
                                                         self.obs_t: obs})
        energy = -energy[:, 0]
        return self.unpack(energy, paths)

    def eval_expert_probs(self, expert_paths, policy, insert=False):
        """
        Evaluate expert policy probability under current policy
        """
        for path in expert_paths:
            actions, agent_infos = policy.get_actions(np.array(path[
                                                                   'observations']).astype(
                int))
            path['agent_infos'] = agent_infos
            # obs = self.env_spec.observation_space.flatten_n(
            #     np.array(path['observations']).astype(int))
            actions = self.env_spec.action_space.flatten_n(
                np.array(actions).astype(int))
            path['actions'] = actions
        return self._compute_path_probs(expert_paths, insert=insert)


class AIRLDiscrete(SingleTimestepIRL):
    """
    Experimental 

    Explicit calculation of normalization constant log Z(s) for discrete
    domains.
    """

    def __init__(self, env_spec, expert_trajs=None,
                 discrim_arch=relu_net,
                 discrim_arch_args={},
                 score_using_discrim=False,
                 l2_reg=0,
                 name='gcl'):
        super(AIRLDiscrete, self).__init__()
        self.dO = env_spec.observation_space.flat_dim
        self.dU = env_spec.action_space.flat_dim
        self.score_using_discrim = score_using_discrim
        self.env_spec = env_spec
        if expert_trajs:
            self.expert_trajs = expert_trajs
            self.expert_trajs_extracted = self.extract_paths(expert_trajs)

        # build energy model
        with tf.variable_scope(name) as _vs:
            # Should be batch_size x T x dO/dU
            self.obs_t = tf.placeholder(tf.float32, [None, self.dO], name='obs')
            self.act_t = tf.placeholder(tf.float32, [None, self.dU], name='act')
            self.labels = tf.placeholder(tf.float32, [None, 1], name='labels')
            self.lprobs = tf.placeholder(tf.float32, [None, 1],
                                         name='log_probs')
            self.lr = tf.placeholder(tf.float32, (), name='lr')

            obs_act = tf.concat([self.obs_t, self.act_t], axis=1)
            with tf.variable_scope('discrim') as dvs:
                with tf.variable_scope('energy'):
                    energy = discrim_arch(obs_act, dout=self.dU,
                                          **discrim_arch_args)

                self.value_fn = tf.reduce_logsumexp(-energy, axis=1,
                                                    keep_dims=True)
                self.energy = tf.reduce_sum(energy * self.act_t, axis=1,
                                            keep_dims=True)  # select action

                log_p_tau = -self.energy - self.value_fn
                discrim_vars = tf.get_collection('reg_vars', scope=dvs.name)

            log_q_tau = self.lprobs

            if l2_reg > 0:
                reg_loss = l2_reg * tf.reduce_sum(
                    [tf.reduce_sum(tf.square(var)) for var in discrim_vars])
            else:
                reg_loss = 0

            log_pq = tf.reduce_logsumexp([log_p_tau, log_q_tau], axis=0)
            self.d_tau = tf.exp(log_p_tau - log_pq)
            cent_loss = -tf.reduce_mean(
                self.labels * (log_p_tau - log_pq) + (1 - self.labels) * (
                        log_q_tau - log_pq))

            self.loss = cent_loss + reg_loss
            self.step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
                self.loss)
            self._make_param_ops(_vs)

    def fit(self, paths, policy=None, batch_size=32, max_itrs=100, logger=None,
            lr=1e-3, **kwargs):
        # self._compute_path_probs(paths, insert=True)
        self.eval_expert_probs(paths, policy, insert=True)
        self.eval_expert_probs(self.expert_trajs, policy, insert=True)
        obs, acts, path_probs = self.extract_paths(paths, keys=(
            'observations', 'actions', 'a_logprobs'))
        expert_obs, expert_acts, expert_probs = self.extract_paths(
            self.expert_trajs, keys=('observations', 'actions', 'a_logprobs'))
        expert_obs = self.env_spec.observation_space.flatten_n(
            expert_obs.astype(int))

        # Train discriminator
        for it in TrainingIterator(max_itrs, heartbeat=5):
            obs_batch, act_batch, lprobs_batch = \
                self.sample_batch(obs, acts, path_probs, batch_size=batch_size)

            expert_obs_batch, expert_act_batch, expert_lprobs_batch = \
                self.sample_batch(expert_obs, expert_acts, expert_probs,
                                  batch_size=batch_size)

            labels = np.zeros((batch_size * 2, 1))
            labels[batch_size:] = 1.0
            obs_batch = np.concatenate([obs_batch, expert_obs_batch], axis=0)
            act_batch = np.concatenate([act_batch, expert_act_batch], axis=0)
            lprobs_batch = np.expand_dims(
                np.concatenate([lprobs_batch, expert_lprobs_batch], axis=0),
                axis=1).astype(np.float32)

            loss, _ = tf.get_default_session().run([self.loss, self.step],
                                                   feed_dict={
                                                       self.act_t: act_batch,
                                                       self.obs_t: obs_batch,
                                                       self.labels: labels,
                                                       self.lprobs:
                                                           lprobs_batch,
                                                       self.lr: lr
                                                   })

            it.record('loss', loss)
            if it.heartbeat:
                print(it.itr_message())
                mean_loss = it.pop_mean('loss')
                print('\tLoss:%f' % mean_loss)
        if logger:
            energy, logZ, dtau = tf.get_default_session().run(
                [self.energy, self.value_fn, self.d_tau],
                feed_dict={self.act_t: acts, self.obs_t: obs,
                           self.lprobs: np.expand_dims(path_probs, axis=1)})
            logger.record_tabular('IRLLogZ', np.mean(logZ))
            logger.record_tabular('IRLAverageEnergy', np.mean(energy))
            logger.record_tabular('IRLAverageLogPtau', np.mean(-energy - logZ))
            logger.record_tabular('IRLAverageLogQtau', np.mean(path_probs))
            logger.record_tabular('IRLMedianLogQtau', np.median(path_probs))
            logger.record_tabular('IRLAverageDtau', np.mean(dtau))

            energy, logZ, dtau = tf.get_default_session().run(
                [self.energy, self.value_fn, self.d_tau],
                feed_dict={self.act_t: expert_acts, self.obs_t: expert_obs,
                           self.lprobs: np.expand_dims(expert_probs, axis=1)})
            logger.record_tabular('IRLAverageExpertEnergy', np.mean(energy))
            logger.record_tabular('IRLAverageExpertLogPtau',
                                  np.mean(-energy - logZ))
            logger.record_tabular('IRLAverageExpertLogQtau',
                                  np.mean(expert_probs))
            logger.record_tabular('IRLMedianExpertLogQtau',
                                  np.median(expert_probs))
            logger.record_tabular('IRLAverageExpertDtau', np.mean(dtau))
        return mean_loss

    def eval(self, paths, gamma=1.0, **kwargs):
        """
        Return bonus
        """

        if self.score_using_discrim:
            self._compute_path_probs(paths, insert=True)

            obs, acts, path_probs = self.extract_paths(paths, keys=(
                'observations', 'actions', 'a_logprobs'))

            path_probs = np.expand_dims(path_probs, axis=1)
            scores = tf.get_default_session().run(self.d_tau,
                                                  feed_dict={self.act_t: acts,
                                                             self.obs_t: obs,
                                                             self.lprobs:
                                                                 path_probs})
            score = np.log(scores + LOG_REG) - np.log(1 - scores + LOG_REG)
            score = score[:, 0]
        else:
            obs, acts = self.extract_paths(paths)
            energy = tf.get_default_session().run(self.energy,
                                                  feed_dict={self.act_t: acts,
                                                             self.obs_t: obs})
            score = (-energy)[:, 0]
        return self.unpack(score, paths)

    def eval_expert_probs(self, expert_paths, policy, insert=False):
        """
        Evaluate expert policy probability under current policy
        """
        if policy.recurrent:
            policy.reset([True] * len(expert_paths))
            max_path_length = max([len(path["observations"]) for path in
                                   expert_paths])
            # expert_obs = \
            #     self.extract_paths(expert_paths, keys=('observations',))[0]

            expert_obs = np.stack(tensor_utils.pad_tensor_n(
                [self.env_spec.observation_space.flatten_n(
                    path["observations"]) for path in expert_paths],
                max_path_length)).astype(int)
            agent_infos = []
            for t in range(expert_obs.shape[1]):
                a, infos = policy.get_actions(expert_obs[:, t])
                agent_infos.append(infos)
            agent_infos_stack = tensor_utils.stack_tensor_dict_list(agent_infos)
            for key in agent_infos_stack:
                agent_infos_stack[key] = np.transpose(agent_infos_stack[key],
                                                      axes=[1, 0, 2])
            agent_infos_transpose = tensor_utils.split_tensor_dict_list(
                agent_infos_stack)
            for i, path in enumerate(expert_paths):
                path['agent_infos'] = agent_infos_transpose[i]
                # if not isinstance(path)
                if isinstance(path['actions'][0], np.int64):
                    actions = self.env_spec.action_space.flatten_n(
                        np.array(path['actions']).astype(int))
                    path['actions'] = actions
        else:
            for path in expert_paths:
                actions, agent_infos = policy.get_actions(np.array(path[
                                                                       'observations']).astype(
                    int))
                path['agent_infos'] = agent_infos
                # obs = self.env_spec.observation_space.flatten_n(
                #     np.array(path['observations']).astype(int))
                actions = self.env_spec.action_space.flatten_n(
                    np.array(actions).astype(int))
                path['actions'] = actions
                # path['observations'] = obs
        return self._compute_path_probs(expert_paths, insert=insert)


class GAN_GCL(TrajectoryIRL):
    """
    Guided cost learning, GAN formulation with learned partition function
    See https://arxiv.org/pdf/1611.03852.pdf
    """

    def __init__(self, env_spec, expert_trajs=None,
                 discrim_arch=feedforward_energy,
                 discrim_arch_args={},
                 l2_reg=0,
                 discount=1.0,
                 init_itrs=None,
                 score_dtau=False,
                 name='trajprior'):
        super(GAN_GCL, self).__init__()
        self.env_spec = env_spec
        self.dO = env_spec.observation_space.flat_dim
        self.dU = env_spec.action_space.flat_dim
        self.score_dtau = score_dtau
        self.expert_trajs = expert_trajs

        # build energy model
        with tf.variable_scope(name) as vs:
            # Should be batch_size x T x dO/dU
            self.obs_t = tf.placeholder(tf.float32, [None, None, self.dO],
                                        name='obs')
            self.act_t = tf.placeholder(tf.float32, [None, None, self.dU],
                                        name='act')
            self.traj_logprobs = tf.placeholder(tf.float32, [None, None],
                                                name='traj_probs')
            self.labels = tf.placeholder(tf.float32, [None, 1], name='labels')
            self.lr = tf.placeholder(tf.float32, (), name='lr')

            obs_act = tf.concat([self.obs_t, self.act_t], axis=2)

            with tf.variable_scope('discrim') as vs2:
                self.energy = discrim_arch(obs_act, **discrim_arch_args)
                discrim_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs2.name)

            self.energy_timestep = self.energy
            # Don't train separate log Z because we can't fully separate it
            # from the energy function
            log_p_tau = discounted_reduce_sum(-self.energy, discount=discount,
                                              axis=1)
            log_q_tau = tf.reduce_sum(self.traj_logprobs, axis=1,
                                      keep_dims=True)

            # numerical stability trick
            log_pq = tf.reduce_logsumexp([log_p_tau, log_q_tau], axis=0)
            self.d_tau = tf.exp(log_p_tau - log_pq)
            cent_loss = -tf.reduce_mean(
                self.labels * (log_p_tau - log_pq) + (1 - self.labels) * (
                        log_q_tau - log_pq))

            if l2_reg > 0:
                reg_loss = l2_reg * tf.reduce_sum(
                    [tf.reduce_sum(tf.square(var)) for var in discrim_vars])
            else:
                reg_loss = 0

            # self.predictions = tf.nn.sigmoid(logits)
            self.loss = cent_loss + reg_loss
            self.step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
                self.loss)
            self._make_param_ops(vs)

    @staticmethod
    def pad_tensor_n(xs, max_len, obs=True):
        ret = np.zeros((len(xs), max_len) + xs[0].shape[1:], dtype=xs[0].dtype)
        if obs:
            ret[:, :, 12], ret[:, :, 122], ret[:, :, 124] = 1.0, 1.0, 1.0
        for idx, x in enumerate(xs):
            ret[idx][:len(x)] = x
        return ret

    @property
    def score_trajectories(self):
        return False

    def fit(self, paths, policy=None, batch_size=32, max_itrs=100, logger=None,
            lr=1e-3, **kwargs):
        self._compute_path_probs(paths, insert=True)
        self.eval_expert_probs(self.expert_trajs, policy, insert=True)

        gen_max_action_path_length = max([len(path['actions'])
                                          for
                                          path in
                                          paths])

        gen_max_obs_path_length = max([len(path["observations"]) for path in
                                       paths])

        expert_max_action_path_length = max([len(path['actions'])
                                             for
                                             path in
                                             self.expert_trajs])

        expert_max_obs_path_length = max([len(path["observations"]) for path in
                                          self.expert_trajs])

        max_obs_path_length = max(gen_max_obs_path_length,
                                  expert_max_obs_path_length)
        max_action_path_length = max(gen_max_action_path_length,
                                     expert_max_action_path_length)

        obs = np.stack(self.pad_tensor_n([path["observations"]
                                          for path in
                                          paths],
                                         max_obs_path_length))

        acts = np.stack(tensor_utils.pad_tensor_n([path["actions"]
                                                   for path in
                                                   paths],
                                                  max_action_path_length))

        path_probs = np.stack(tensor_utils.pad_tensor_n([path["a_logprobs"]
                                                         for path in
                                                         paths],
                                                        max_action_path_length))

        expert_obs = np.stack(tensor_utils.pad_tensor_n(
            [self.env_spec.observation_space.flatten_n(path["observations"])
             for path in
             self.expert_trajs], max_obs_path_length))

        expert_acts = np.stack(tensor_utils.pad_tensor_n(
            [path["actions"]
             for path in
             self.expert_trajs],
            max_action_path_length))

        expert_probs = np.stack(tensor_utils.pad_tensor_n(
            [path["a_logprobs"]
             for path in
             self.expert_trajs],
            max_action_path_length))
        expert_probs[np.isinf(expert_probs)] = 0

        # Train discriminator
        for it in TrainingIterator(max_itrs, heartbeat=5):
            obs_batch, act_batch, lprobs_batch = \
                self.sample_batch(obs, acts, path_probs, batch_size=batch_size)

            expert_obs_batch, expert_act_batch, expert_lprobs_batch = \
                self.sample_batch(expert_obs, expert_acts, expert_probs,
                                  batch_size=batch_size)
            T = expert_obs_batch.shape[1]
            labels = np.zeros((batch_size * 2, 1))
            labels[batch_size:] = 1.0
            obs_batch = np.concatenate([obs_batch, expert_obs_batch], axis=0)
            act_batch = np.concatenate([act_batch, expert_act_batch], axis=0)
            lprobs_batch = np.concatenate([lprobs_batch, expert_lprobs_batch],
                                          axis=0)
            lprobs_batch[np.isinf(lprobs_batch)] = -999
            loss, _ = tf.get_default_session().run([self.loss, self.step],
                                                   feed_dict={
                                                       self.act_t:
                                                           act_batch.astype(
                                                               int),
                                                       self.obs_t:
                                                           obs_batch.astype(
                                                               int),
                                                       self.labels: labels,
                                                       self.traj_logprobs:
                                                           lprobs_batch,
                                                       self.lr: lr,
                                                   })

            it.record('loss', loss)
            if it.heartbeat:
                print(it.itr_message())
                mean_loss = it.pop_mean('loss')
                print('\tLoss:%f' % mean_loss)

        if logger:
            energy, dtau = tf.get_default_session().run(
                [self.energy_timestep, self.d_tau],
                feed_dict={self.act_t: acts, self.obs_t: obs,
                           self.traj_logprobs: path_probs})
            logger.record_tabular('IRLAverageEnergy', np.mean(energy))
            logger.record_tabular('IRLAverageLogQtau', np.mean(path_probs))
            logger.record_tabular('IRLMedianLogQtau', np.median(path_probs))
            logger.record_tabular('IRLAverageDtau', np.mean(dtau))

            energy, dtau = tf.get_default_session().run(
                [self.energy_timestep, self.d_tau],
                feed_dict={self.act_t: expert_acts, self.obs_t: expert_obs,
                           self.traj_logprobs: expert_probs})
            logger.record_tabular('IRLAverageExpertEnergy', np.mean(energy))
            logger.record_tabular('IRLAverageExpertLogQtau',
                                  np.mean(expert_probs))
            logger.record_tabular('IRLMedianExpertLogQtau',
                                  np.median(expert_probs))
            logger.record_tabular('IRLAverageExpertDtau', np.mean(dtau))
        return mean_loss

    def eval(self, paths, **kwargs):
        """
        Return bonus
        """
        max_path_length = max([len(path["observations"]) for path in
                               paths])

        obs = np.stack(self.pad_tensor_n([path["observations"]
                                          for path in
                                          paths],
                                         max_path_length))

        acts = np.stack(tensor_utils.pad_tensor_n([path["actions"]
                                                   for path in
                                                   paths],
                                                  max_path_length))

        scores = tf.get_default_session().run(self.energy,
                                              feed_dict={self.act_t: acts,
                                                         self.obs_t: obs})
        scores = -scores[:, :, 0]
        return scores

    def eval_expert_probs(self, expert_paths, policy, insert=False):
        """
        Evaluate expert policy probability under current policy
        """

        if policy.recurrent:
            policy.reset([True] * len(expert_paths))
            max_path_length = max(
                [max(len(path["observations"]), len(path["actions"])) for
                 path in
                 expert_paths])

            # expert_obs = \
            #     self.extract_paths(expert_paths, keys=('observations',))[0]

            expert_obs = np.stack(self.pad_tensor_n(
                [self.env_spec.observation_space.flatten_n(
                    path["observations"]) for path in expert_paths],
                max_path_length)).astype(int)
            agent_infos = []
            for t in range(expert_obs.shape[1]):
                a, infos = policy.get_actions(expert_obs[:, t])
                agent_infos.append(infos)
            agent_infos_stack = tensor_utils.stack_tensor_dict_list(agent_infos)
            for key in agent_infos_stack:
                agent_infos_stack[key] = np.transpose(agent_infos_stack[key],
                                                      axes=[1, 0, 2])
            agent_infos_transpose = tensor_utils.split_tensor_dict_list(
                agent_infos_stack)
            for i, path in enumerate(expert_paths):
                path['agent_infos'] = agent_infos_transpose[i]
                max_path_length = max([len(path["actions"]) for path in
                                       expert_paths])
                # if not isinstance(path)
                if isinstance(path['actions'][0], (np.int64, np.int32, int)):
                    actions = self.env_spec.action_space.flatten_n(
                        np.array(path['actions']).astype(int))
                    path['actions'] = tensor_utils.pad_tensor_n(actions.T,
                                                                max_path_length).T
        else:
            for path in expert_paths:
                actions, agent_infos = policy.get_actions(path['observations'])
                path['agent_infos'] = agent_infos
                # obs = self.env_spec.observation_space.flatten_n(
                #     np.array(path['observations']).astype(int))
                actions = self.env_spec.action_space.flatten_n(
                    np.array(actions).astype(int))
                path['actions'] = actions
                # path['observations'] = obs
        return self._compute_path_probs(expert_paths, insert=insert)

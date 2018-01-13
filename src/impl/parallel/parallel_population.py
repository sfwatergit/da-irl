"""
Module to run code on expert agents in parallel.

Based on openai baselines (code)[github.com/openai/baselines]

"""
from multiprocessing import Process, Pipe

import numpy as np


class VecExpAgent(object):
    """
    Vectorized expert agent base class.
    """

    def learn_reward(self):
        raise NotImplementedError

    def get_trajectories(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries
    to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def worker(remote, parent_remote, exp_fn_wrapper):
    parent_remote.close()
    expert = exp_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'learn_reward':
            reward = expert.learn_reward()
            remote.send([reward])
        elif cmd == 'get_rewards':
            rewards = expert.reward_function.get_rewards()
            remote.send([rewards])
        elif cmd == 'get_theta':
            theta = expert.reward_function.get_theta()
            remote.send([theta])
        elif cmd == 'get_policy':
            policy = expert.policy
            remote.send([policy])
        elif cmd == 'close':
            remote.close()
            break
        else:
            raise NotImplementedError


class SubProcVecExpAgent(VecExpAgent):
    def __init__(self, expert_persona_fns):
        """

        Args:
            expert_persona_fns (list): list of expert personas to interact
            with in parallel
        """
        self.closed = False
        self.experts = expert_persona_fns
        nexps = len(expert_persona_fns)
        self._trajectories = None
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nexps)])
        self.ps = [Process(target=worker, args=(
        work_remote, remote, CloudpickleWrapper(expert_persona_fn))) for
                   (work_remote, remote, expert_persona_fn) in
                   zip(self.work_remotes, self.remotes, expert_persona_fns)]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def learn_reward(self):
        return np.stack(self.remote_proc('learn_reward'))

    def get_rewards(self):
        return np.stack(self.remote_proc('get_rewards'))

    def get_theta(self):
        return np.stack(self.remote_proc('get_theta'))

    def get_policy(self):
        return np.stack(self.remote_proc('get_policy'))

    def remote_proc(self, cmd, data=None):
        if data is None:
            for remote in self.remotes:
                remote.send((cmd, None))
        else:
            for remote, datum in zip(self.remotes, data):
                remote.send((cmd, datum))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        if self.closed:
            return len(self.remotes)

        for remote in self.remotes:
            remote.send(('close', None))

        for p in self.ps:
            p.join()

        self.closed = True

    @property
    def num_exps(self):
        return len(self.remotes)

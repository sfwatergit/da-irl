class Algorithm(object):
    pass


class RLAlgorithm(Algorithm):
    def train(self):
        raise NotImplementedError


class IRLAlgorithm(Algorithm):
    def train(self, trajectories, num_iters):
        raise NotImplementedError

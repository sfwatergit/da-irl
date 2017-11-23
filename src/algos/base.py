class Algorithm(object):
    pass


class RLAlgorithm(Algorithm):
    def train(self):
        raise NotImplementedError


class IRLAlgorithm(Algorithm):
    def train(self, trajectories):
        raise NotImplementedError

class Algorithm(object):
    pass


class RLAlgorithm(Algorithm):

    def train(self):
        raise NotImplementedError


class IRLAlgorithm(Algorithm):

    def learn_rewards(self):
        raise NotImplementedError
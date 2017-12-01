class ValueFunction(object):
    """
    State-value function
    """
    def fit(self, data):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

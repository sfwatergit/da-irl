from models.model_base import Model


class ValueFunction(Model):
    """
    State-value function
    """
    def fit(self, data):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

from .generic import IAModel
from sklearn.naive_bayes import MultinomialNB


class IntentionModel(IAModel):
    def __init__(self, ts_x, ts_y):
        IAModel.__init__(self, ts_x, ts_y)
        self.model = MultinomialNB()

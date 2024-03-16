import pandas as pd

class CustomEncoding:
    def __init__(self, X, status):
        self.X = X
        self.status = status

    def encode(self):
        self.X = pd.get_dummies(self.X)
        return self.X
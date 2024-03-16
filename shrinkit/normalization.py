
from sklearn.preprocessing import StandardScaler
import pandas as pd

class CustomNormalizer:
    def __init__(self, X, status):
        self.X = X
        self.status = status

    def normalize(self):
        normalizer = StandardScaler()
        X_scaled = pd.DataFrame(normalizer.fit_transform(self.X), columns=self.X.columns)
        return X_scaled


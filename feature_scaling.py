import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class FeatureScaling:
    def __init__(self, data) -> None:
        self.data = data

    def standard_scaling(self, cols):
        for col in cols:
            mean = self.data[col].mean()
            standard_deviation = self.data[col].std()
            self.data[col] = (self.data[col] - mean)/(standard_deviation)

    def normalization(self, cols):
        for col in cols:
            minValue = self.data[col].min()
            maxValue = self.data[col].max()
            self.data[col] = (self.data[col] - minValue)/(maxValue - minValue)


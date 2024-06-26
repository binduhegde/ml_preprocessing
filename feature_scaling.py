import pandas as pd

# 2 ways. minmax scaling and standard scaling
class FeatureScaling:
    def __init__(self, data) -> None:
        self.data = data

    def standard_scaling(self, cols):
        for col in cols:
            mean = self.data[col].mean()
            standard_deviation = self.data[col].std()
            self.data[col] = (self.data[col] - mean)/(standard_deviation)
    # aka minmax scaling
    def normalization(self, cols):
        for col in cols:
            minValue = self.data[col].min()
            maxValue = self.data[col].max()
            self.data[col] = (self.data[col] - minValue)/(maxValue - minValue)


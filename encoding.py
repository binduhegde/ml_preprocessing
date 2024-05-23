import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 2 ways. onehot encoding and label encoding
class Encoding:
    def __init__(self, data) -> None:
        self.data = data

    def onehot_encoding(self, cols):
        self.data = pd.get_dummies(data=self.data, columns=cols)

    def label_encoding(self, cols):
        le = LabelEncoder()
        for col in cols:
            self.data[col] = le.fit_transform(self.data[col])

import pandas as pd

class Imputation:
    def __init__(self, data) -> None:
        self.data = data

    def remove_null(self, cols):
        self.data.dropna(subset=cols, inplace= True)

    def fill_mean(self, cols):
        for col in cols:
            self.data[col].fillna(self.data[col].mean(), inplace=True)

    def fill_median(self, cols):
        for col in cols:
            self.data[col].fillna(self.data[col].median(), inplace=True)

    def fill_mode(self, cols):
        for col in cols:
            self.data[col].fillna(self.data[col].mode()[0], inplace=True)

        
    
if __name__ == '__main__':
    df = pd.read_csv('train.csv')
    imp = Imputation(data=df)
    

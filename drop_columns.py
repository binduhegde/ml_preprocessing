import pandas as pd


class DropColumns:
    def __init__(self, data) -> None:
        self.data = data

    # creating exception handling here to avoid KeyError
    # eg, if we already performed one-hot encoding for a column, say Sex,
    # then there will exist no such column as Sex. Hence we can't drop that column
    def drop_cols(self, cols):
        for col in cols:
            try:
                self.data.drop(col, axis=1, inplace=True)
            except:
                continue

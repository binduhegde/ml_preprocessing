import pandas as pd


class DropColumns:
    def __init__(self, data) -> None:
        self.data = data

    def drop_cols(self, cols):
        for col in cols:
            try:
                self.data.drop(col, axis=1, inplace=True)
            except:
                continue

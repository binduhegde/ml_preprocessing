import pandas as pd

class DropColumns:
    def __init__(self, data) -> None:
        self.data = data

    def drop_cols(self, cols):
        self.data.drop(cols, axis=1, inplace=True)
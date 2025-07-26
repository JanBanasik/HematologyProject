import pandas as pd

class DataLoader:
    def __init__(self,filename):
        self.filename = filename
        self.df = None
    def load(self):
        self.df = pd.read_csv(self.filename)
        return self.df
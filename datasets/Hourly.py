import torch
from torch.utils.data import Dataset
import pandas as pd

class HourlyLoader(Dataset):
    def __init__(self, data, columns=None):
        """
        Args:
            data (str or DataFrame): The path to the csv file or a pandas DataFrame.
            columns (list of str, optional): The columns to keep. Defaults to None.
        
        Assumes the last column to be target and all others as features
        """
        if isinstance(data, str):
            data = pd.read_csv(data)
        
        self.features = torch.tensor(data.values[:, :-1], dtype=torch.float32)
        self.targets = torch.tensor(data.values[:, -1], dtype=torch.float32)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index], self.targets[index]

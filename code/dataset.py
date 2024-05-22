import random
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from preprocess import normalize_data, zero_padding


# dataset
class WidarDataset(Dataset):
    def __init__(self, df, BVP, transform=None):
        self.df = df
        self.path = self.df['path']
        self.label = self.df['gesture']
        self.BVP = BVP
        self.transform = transform
        self.length = len(self.path)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        path = self.path.iloc[idx]
        x = zero_padding(self.BVP[path]).astype(np.float32)
        y = torch.tensor(self.label.iloc[idx])
        if self.transform is not None:
            x = self.transform(x)
        return x, y, idx

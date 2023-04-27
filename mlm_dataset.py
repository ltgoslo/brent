from torch.utils.data import Dataset
import json
import torch

class MlmDataset(Dataset):
    def __init__(self, path):
        self.wiki = json.load(open(path))

    def __len__(self):
        return len(self.wiki)
    
    def __getitem__(self, index):

        return index


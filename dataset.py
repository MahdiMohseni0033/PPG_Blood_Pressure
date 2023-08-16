import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = torch.load(data_path)  # Load your data here
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
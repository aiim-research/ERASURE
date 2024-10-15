import torch
from torch.utils.data import DataLoader, Dataset

class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.length1 = len(dataset1)
        self.length2 = len(dataset2)

    def __len__(self):
        return self.length1 + self.length2

    def __getitem__(self, idx):
        if idx < self.length1:
            x, y = self.dataset1[idx]
            dataset_id = 0
        else:
            x, y = self.dataset2[idx - self.length1]
            dataset_id = 1
        return x, y, dataset_id

def create_combined_dataloader(dataset1, dataset2, batch_size, shuffle=True, num_workers=0):
    combined_dataset = CombinedDataset(dataset1, dataset2)
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
#!/usr/bin/env python
from torch.utils.data import Dataset

class YelpPolarityDataset(Dataset):
    def __init__(self, csr_data_matrix, targets):
        self.csr_data = csr_data_matrix
        self.labels = targets

    def __len__(self):
        return self.csr_data.shape[0]

    def __getitem__(self, ind):
        return self.csr_data[ind,:].toarray()[0], self.labels[ind]

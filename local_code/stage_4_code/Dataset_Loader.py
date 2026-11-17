'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
#
from local_code.base_class.dataset import dataset
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
print("CWD:", os.getcwd())
print("Resolved:", os.path.abspath(self.dataset_source_folder_path))
print("Exists:", os.path.exists(self.dataset_source_folder_path))
class StoryDataset(Dataset):
    def __init__(self, encoded_data, seq_length):
        self.data = torch.tensor(encoded_data, dtype=torch.long)
        self.seq_length = seq_length

    def __len__(self):
        # Number of non-overlapping or sliding chunks
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_length + 1]
        x = chunk[:-1]  # Input sequence
        y = chunk[1:]   # Target sequence (shifted right by 1)
        return x, y



class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None   # 'MNIST', 'ORL' or 'CIFAR'


    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)


    def load(self):
        story_files = glob.glob(self.dataset_source_folder_path + "/*.txt")
        print("Folder:", self.dataset_source_folder_path)
        print("Files found:", len(story_files))
        raw_text = "\n\n---\n\n".join(
            open(f, "r", encoding="utf-8").read() for f in story_files
        )
        vocab = sorted(list(set(raw_text)))
        print(vocab[:100])
        char_to_idx = {ch: i for i, ch in enumerate(vocab)}
        # idx_to_char = {i: ch for i, ch in enumerate(vocab)}

        encoded_text = [char_to_idx[ch] for ch in raw_text]
        seq_length = 100
        dataset = StoryDataset(encoded_text, seq_length)
        return DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
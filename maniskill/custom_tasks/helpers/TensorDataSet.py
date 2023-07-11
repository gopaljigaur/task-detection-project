import os

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class TensorDataSet(Dataset):
    def __init__(self, img_tuples, transform=None, target_transform=None):
        # img_tuple is a list containing ".png" file names and labels.
        # change file name to .pt to get the tensor from preprocessing
        self.img_tuples = img_tuples
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_tuples)

    def __getitem__(self, idx):
        image = torch.load(self.img_tuples[idx][0].split(".")[0]+".pt").squeeze(0)
        label = self.img_tuples[idx][1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

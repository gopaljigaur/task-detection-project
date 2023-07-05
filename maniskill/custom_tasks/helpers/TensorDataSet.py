import os

from torch.utils.data import Dataset
from torchvision.io import read_image


class TensorDataSet(Dataset):
    def __init__(self, tensors, labels, transform=None, target_transform=None):
        # img_tuple is a list containing ".png" file names and labels.
        # change file name to .pt to get the tensor from preprocessing
        self.img_labels = labels
        self.tensors = tensors
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.tensors[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
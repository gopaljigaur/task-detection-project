import os
import matplotlib.image as imgplt
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import torchvision.datasets
from torchvision import transforms
from dino.extractor import ViTExtractor

device = "cuda:0" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    training_data_path = f"data/training_data/training_set"
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.ImageFolder(root=training_data_path, transform=transform)
    batch_size = 2
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)

    extractor = ViTExtractor()
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        # imgs should be imagenet normalized tensors. shape BxCxHxW
        descriptors = extractor.extract_descriptors(images)
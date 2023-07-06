import os.path
import random

import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import numpy as np

from maniskill.custom_tasks.helpers import TensorDataSet
from maniskill.task_classifier import TaskClassifier
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

device = "cuda:0" if torch.cuda.is_available() else "cpu"
net = TaskClassifier(vit_stride=2)

criterion = nn.CrossEntropyLoss()
optim = optim.SGD(net.get_trainable_params(), lr=0.01)
image_path = "training_data/training_set"
net.preprocess(image_path)
[tensors, labels] = net.load_cache("training_data/training_set")
tensor_data_set = TensorDataSet.TensorDataSet(tensors, labels)
data_loader = DataLoader(dataset=tensor_data_set, batch_size=128, shuffle=True)
print("preprocessing done")
num_epochs = 100
for epoch in range(num_epochs):
    for img, label in data_loader:
        img, label = img.to(device), label.to(device)
        labels = nn.functional.one_hot(label, num_classes=4).to(torch.float32)
        output = net.cached_forward(img)
        loss = criterion(output, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

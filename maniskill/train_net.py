import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import numpy as np
from maniskill.task_classifier import TaskClassifier
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

device = "cuda:0" if torch.cuda.is_available() else "cpu"
net = TaskClassifier(vit_stride=4)

criterion = nn.MSELoss()
optim = optim.SGD(net.get_trainable_params(), lr=0.01)

dataset = ImageFolder("training_data/training_set", transform=transforms.ToTensor())
data_loader = DataLoader(dataset=dataset, batch_size=16)

num_epochs = 10
for epoch in range(num_epochs):
    for img, label in data_loader:
        img, label = img.to(device), label.to(device)
        labels = nn.functional.one_hot(label, num_classes=4).to(torch.float32)
        img.requires_grad = True
        # plt.imshow(np.transpose(img.squeeze(0).cpu(),(1,2,0)))
        # plt.show()
        output = net(img)
        loss = criterion(output, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

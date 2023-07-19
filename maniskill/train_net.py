from maniskill.task_classifier import TaskClassifier
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader

device = "cuda:0" if torch.cuda.is_available() else "cpu"
net = TaskClassifier(vit_stride=4)

criterion = nn.CrossEntropyLoss()
optim = optim.SGD(net.get_trainable_params(), lr=10)
image_path = "training_data/training_set"
# net.preprocess(image_path)
tensor_data_set = net.load_cache("training_data/training_set")[1]
data_loader = DataLoader(dataset=tensor_data_set, batch_size=64, shuffle=True)
print("preprocessing done")
num_epochs = 100
for img, label in data_loader:
    print(img, label)

import sys
sys.exit(0)

for epoch in range(num_epochs):
    for img, label in data_loader:
        img, label = img.to(device), label.to(device)
        labels = nn.functional.one_hot(label, num_classes=5).to(torch.float32)
        output = net.cached_forward(img)
        loss = criterion(output, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

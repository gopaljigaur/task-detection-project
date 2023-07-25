from maniskill.task_classifier import TaskClassifier
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader

device = "cuda:0" if torch.cuda.is_available() else "cpu"
net = TaskClassifier(vit_stride=4).to(device)

criterion = nn.CrossEntropyLoss()
optim = optim.Adam(net.parameters(), lr=0.0001)
image_path = "training_data/training_set"
net.preprocess(image_path)
tensor_data_set = net.load_cache("training_data/training_set")[1]
data_loader = DataLoader(dataset=tensor_data_set, batch_size=64, shuffle=True)
print("preprocessing done")
num_epochs = 50
for epoch in range(num_epochs):
    net.train()
    train_loss =0.0
    for img, label in data_loader:
        img, label = img.to(device), label.to(device)
        labels = nn.functional.one_hot(label, num_classes=5).to(torch.float32)
        output = net.cached_forward(img)
        loss = criterion(output, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
    if epoch >10 and epoch % 5 ==0:
        torch.save(net, f"model_{epoch}.pth")
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")


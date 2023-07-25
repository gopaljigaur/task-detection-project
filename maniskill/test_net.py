from maniskill.task_classifier import TaskClassifier
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader

device = "cuda:0" if torch.cuda.is_available() else "cpu"
train_net = TaskClassifier(vit_stride=4).to(device)
net = torch.load("model_20.pth")
net.eval()

test_image_path = "training_data/test_set"
train_net.preprocess(test_image_path)
tensor_data_set = train_net.load_cache("training_data/test_set")[1]
data_loader = DataLoader(dataset=tensor_data_set, batch_size=64, shuffle=True)
print("preprocessing done")

correct = 0
total = 0

with torch.no_grad():
    for img, label in data_loader:
        img, label = img.to(device), label.to(device)
        labels = nn.functional.one_hot(label, num_classes=5).to(torch.float32)
        output = net.cached_forward(img)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == label).sum().item()
        print("Accuracy of the network on the 10000 test images: %d %%" % (100 * correct / total))
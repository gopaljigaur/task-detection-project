import os
from typing import List
from torchmetrics.functional import pairwise_manhattan_distance
import matplotlib.image as imgplt
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import torchvision.datasets
import yaml
from torchvision import transforms
from dino.extractor import ViTExtractor
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

device = "cuda:0" if torch.cuda.is_available() else "cpu"
extractor = ViTExtractor(stride=8)

def get_descriptor(image_path: str, coordinates: List[str]):
    image = mpimg.imread(image_path)
    image = image.reshape((1,4,128,128))
    image = torch.Tensor(image[:,:3,:,:]).to(device)
    embeddings = extractor.extract_descriptors(image)
    x_int = int(coordinates[0].split(" ")[0].split(".")[0])
    y_int = int(coordinates[0].split(" ")[-1].split(".")[0])
    x_patch = x_int % 8
    y_patch = y_int % 8
    patch_num = (128 / 8) * y_patch + x_patch
    return embeddings[0,0,int(patch_num)]



if __name__ == '__main__':
    training_data_path = f"training_data/training_set"
    for task in os.listdir(training_data_path):
        task_folder = os.path.join(training_data_path, task)
        labels = None
        with open(os.path.join(task_folder, "labels.yml"), "r") as label_file:
            print(task)
            try:
                labels = yaml.safe_load(label_file)
                descriptors = torch.tensor([],device=device)
                for key, value in labels.items():
                    for point_pair in value["points"]:
                        descriptor = get_descriptor(os.path.join(os.path.join(task_folder, key)), point_pair)
                        descriptors = torch.cat([descriptors,descriptor.reshape(1,descriptor.shape[0])])
                # calculate pairwise manhattan
                manhattan = pairwise_manhattan_distance(descriptors)
                print(manhattan)
            except yaml.YAMLError as exc:
                print(exc)

    # transform = transforms.Compose([transforms.ToTensor()])
    # dataset = torchvision.datasets.ImageFolder(root=training_data_path, transform=transform)
    # batch_size = 2
    # data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)
    #
    # extractor = ViTExtractor()
    # for images, labels in data_loader:
    #     images, labels = images.to(device), labels.to(device)
    #     # imgs should be imagenet normalized tensors. shape BxCxHxW
    #     descriptors = extractor.extract_descriptors(images)
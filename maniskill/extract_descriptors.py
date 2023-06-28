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
import torch
import gc



torch.cuda.empty_cache()
stride = 4
device = "cuda:0" if torch.cuda.is_available() else "cpu"
extractor = ViTExtractor()

def import_image(image_path:str):
    image = mpimg.imread(image_path)
    # image = image.reshape((1, 4, 128, 128))
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image,(0,3,1,2))
    image = torch.Tensor(image[:, :3, :, :]).to(device)
    return image


def get_descriptor(image_path: str, coordinates: List[str]):
    image = import_image(image_path)
    embeddings = extractor.extract_descriptors(image)
    x_int = int(coordinates[0].split(" ")[0].split(".")[0])
    y_int = int(coordinates[0].split(" ")[-1].split(".")[0])
    x_patch = x_int / stride
    y_patch = y_int / stride
    patch_num = (128 / stride) * y_patch + x_patch
    return embeddings[0,0,int(patch_num)]


def compare_descriptors(target_descriptors:torch.Tensor, comparator_descriptors: torch.Tensor):
    distance = pairwise_manhattan_distance(comparator_descriptors, target_descriptors)
    arg_min = torch.argmin(distance).item()
    target_argmin = arg_min % target_descriptors.shape[0]
    comparator_argmin = int(arg_min / target_descriptors.shape[0])
    return [distance[target_argmin][target_argmin].cpu().detach().item(), comparator_argmin]

def render_patch(image_path:str, patch:int):
    image = mpimg.imread(image_path)
    x_start = patch % stride
    y_start = patch / (128/stride)
    [x_lwr,x_upr] = [x_start*stride,x_start*stride+8]
    [y_lwr,y_upr] = [int(y_start*4),int(y_start*stride+8)]
    sub_image = image[y_lwr:y_upr,x_lwr:x_upr,:]
    plt.imshow(sub_image, extent=[x_lwr,x_upr,y_upr,y_lwr])
    plt.show()
    plt.imshow(image)
    plt.show()
    return 0



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
                for image in os.listdir(task_folder):
                    if not ".png" in image:
                        continue
                    embeddings = extractor.extract_descriptors(import_image(os.path.join(task_folder,image)))[0,0,:,:]
                    [min, patch_idx] = compare_descriptors(descriptors, embeddings)
                    render_patch(os.path.join(task_folder,image),patch_idx)

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
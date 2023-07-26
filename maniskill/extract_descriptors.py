import sys
import math
import os
from typing import List
import random
import torch.utils.data
import yaml
import torch
import pickle as pkl

from maniskill.helpers.DescriptorSet import DescriptorSet

stride = 4
patch_size = 8
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_descriptor_for_labeled(image_path: str, coordinates: str):
    x_int = int(coordinates.split(" ")[0].split(".")[0])
    y_int = int(coordinates.split(" ")[-1].split(".")[0])
    tensor_path = f"{image_path.split('.')[0]}.pt"
    with torch.inference_mode():
        if os.path.exists(tensor_path):
            descs = torch.load(tensor_path).detach()
            pach_am = int(math.sqrt(descs.shape[2]))
            num_patches = (pach_am,pach_am)
            load = (num_patches[0] -1) * stride + patch_size
            load_size = (load,load)
        x_descr = int(num_patches[1] / load_size[1] * x_int)
        y_descr = int(num_patches[0] / load_size[0] * y_int)
        descriptor_idx = num_patches[1] * y_descr + x_descr
        return descs[:,:,int(descriptor_idx),:].unsqueeze(0)


def read_points(task_folder: str, fname = "labels.yml"):
    with open(os.path.join(task_folder, fname), "r") as label_file:
        points = []
        files = []
        try:
            labels = yaml.safe_load(label_file)
            current_descriptors = torch.tensor([], device=device)
            for key, value in labels.items():
                points.extend(value["points"])
                files.extend([key] * len(value["points"]))
        except yaml.YAMLError as exc:
            print(exc)
        return [files, points]


def extract_descriptors(tasks: List[str], descriptor_amount: int = None, fname="labels.yml"):
    # in all_descriptors all the descriptors for the objects will be safed
    descriptors = []
    tasks.sort()
    for task in tasks:
        task_folder = os.path.join(f"training_data/training_set", task)
        [files, points] = read_points(task_folder,fname)
        current_descriptors = torch.tensor([], device=device)
        zipped = list(zip(files, points))
        zip_idx = list(range(0, len(zipped)))
        random.shuffle(zip_idx)
        max_idx = len(zipped)
        desc_location = []
        if descriptor_amount is not None:
            max_idx = min(max_idx, descriptor_amount)
        for key_idx in range(max_idx):
            [file, point] = zipped[zip_idx[key_idx]]
            descriptor = get_descriptor_for_labeled(os.path.join(os.path.join(task_folder, file)),
                                                    point).detach()
            current_descriptors = torch.cat([current_descriptors, descriptor])
            desc_location.append([file, point])
        descriptors.append(DescriptorSet(task, torch.transpose(current_descriptors,0,2).detach(), desc_location))
    return descriptors


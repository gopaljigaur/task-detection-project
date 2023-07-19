import sys
import math
import os
from typing import List
import random
import torch.utils.data
import yaml
import torch
import pickle as pkl

from maniskill.helpers.DescriptorSet import DescriptorSet, get_descriptor_for_labeled

stride = 4
patch_size = 8
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def read_points(task_folder: str):
    """
    Reads the points from the labels.yml file for each task
    :param task_folder: Folder of the task
    :return: list containing the [files and the points (x,y,z)] for each task
    """
    with open(os.path.join(task_folder, "labels.yml"), "r") as label_file:
        points = []
        files = []
        try:
            labels = yaml.safe_load(label_file)
            current_descriptors = torch.tensor([], device=device)
            for key, value in labels.items():
                points.extend([tuple(map(float, x.strip().split())) for x in value["points"]])
                files.extend([key] * len(value["points"]))
        except yaml.YAMLError as exc:
            print(exc)
        return [files, points]


def extract_image_descriptors(tasks: List[str], descriptor_amount: int = None):
    # in all_descriptors all the descriptors for the objects will be saved
    descriptors = []
    tasks.sort()
    for task in tasks:
        task_folder = os.path.join(f"training_data/training_set", task)
        [files, points] = read_points(task_folder)
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
        descriptors.append(DescriptorSet(task, torch.transpose(current_descriptors, 0, 2).detach(), desc_location))
    return descriptors


# descriptors corresponding to image files are stored in the descriptors.pkl files
# these files have information of task and the descriptors present in the image file
# during detection descriptors for test image are compared one by one to the descriptors of the pkl files adn the most similar is the one correct taskex
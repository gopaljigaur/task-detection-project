import os

import numpy as np
import torch
import pickle as pkl
from typing import List

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from dino.extractor import ViTExtractor
from maniskill.extract_descriptors import get_descriptors, chunk_cosine_sim
from maniskill.task_classifier import *

stride = 2
patch_size = 8
device = "cuda:0" if torch.cuda.is_available() else "cpu"
extractor = ViTExtractor(stride=stride)

def object_in_scene(image_path: str, object_descriptors: torch.Tensor, threshold: float = 0.56):
    object_descriptors = torch.transpose(object_descriptors, 0, 2)
    image_descriptors = get_descriptors(image_path)
    # computer similarities
    similarities = chunk_cosine_sim(object_descriptors, image_descriptors)
    sims, idxs = torch.topk(similarities.flatten(), 1)
    # num_patches = extractor.num_patches
    num_patches = (61, 61)
    # get the most similar patch
    sim, idx = sims[0], idxs[0]
    if sim < threshold:
        # object not in scene
        return [False, [], sim]
    idx = idx % (num_patches[0] * num_patches[1])
    y_desc, x_desc = idx // num_patches[1], idx % num_patches[1]
    coordinates = [((x_desc - 1) * stride + stride + patch_size // 2 - .5).item(),
                   ((y_desc - 1) * stride + stride + patch_size // 2 - .5).item()]
    return [True, coordinates, sim]


def compare_image(descriptors: torch.Tensor, descriptor_labels: List[str], image_path: str):
    descriptors = torch.transpose(descriptors,0,2)
    descr_b = get_descriptors(image_path)
    similarities = chunk_cosine_sim(descriptors, descr_b)
    sims, idxs = torch.topk(similarities.flatten(), 4)
    num_patches = extractor.num_patches
    fig, ax = plt.subplots()
    for idx in idxs:
        descr_idx = idx // (num_patches[0] * num_patches[1])
        idx = idx % (num_patches[0] * num_patches[1])
        print(f"{descriptor_labels[descr_idx]} {similarities[0,0,descr_idx,idx]}")
        y_desc, x_desc = idx // num_patches[1], idx % num_patches[1]
        center = ((x_desc - 1) * stride + stride + patch_size // 2 - .5,
                  (y_desc - 1) * stride + stride + patch_size // 2 - .5)
        patch = plt.Circle(center, 2, color=(1, 0, 0, 0.75))
        ax.imshow(extractor.preprocess(image_path)[1])
        ax.add_patch(patch)
    plt.draw()
    plt.show()


def reformat_descriptors(descriptors: torch.Tensor, labels: List[str]):
    objects = []
    idx = 0
    object_label = labels[0]
    for i in range(len(labels)):
        if not labels[i] == object_label:
            objects.append({
                "object": object_label,
                "descriptors": descriptors[idx:i]
            })
            idx = i
            object_label = labels[idx]
    objects.append({
        "object": object_label,
        "descriptors": descriptors[idx:]
    })
    return objects


def compare_single(img_src: str, task: str, custom_threshold:float = 0.55, truth_descriptors=None):
    net = TaskClassifier(vit_stride=2, descriptors=truth_descriptors)
    [class_mapping, dataset] = net.load_cache(img_src)
    data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
    obj_finder = net.obj_finder
    threshold = [0.55] * len(class_mapping)
    threshold[class_mapping[task]] = custom_threshold
    obj_finder.threshold = threshold
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    multiple = 0
    total = len(dataset.img_tuples)
    for [tensors, labels] in data_loader:
        tensor = obj_finder(tensors)
        for i in range(tensor.shape[0]):
            if labels[i] == class_mapping[task]:
                if not tensor[i, class_mapping[task]*3] == 0.:
                    true_positive += 1
                else:
                    false_negative += 1
            else:
                if not tensor[i, class_mapping[task]*3] == 0.:
                    false_positive += 1
                else:
                    true_negative += 1
            if torch.count_nonzero(tensor[i]) > 2:
                multiple += 1
    print(f"TP: {true_positive}, FN: {false_negative}, FP: {false_positive}, TN: {true_negative}, multiple: {multiple}, total:{total}")
    return [true_positive, false_negative, false_positive, true_negative, multiple, total]


if __name__ == '__main__':
    custom_single_tasks=["PickupDrill-v0", "PickUpBlock-v0", "FindClamp-v0", "StoreScrewdriver-v0","Mark-v0"]
    base_path = "training_data/training_set"
    compare_single(base_path, custom_single_tasks[0], 0.60)
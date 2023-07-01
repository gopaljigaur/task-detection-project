import os

import torch
import pickle as pkl
from typing import List

from matplotlib import pyplot as plt

from dino.extractor import ViTExtractor
from maniskill.extract_descriptors import get_descriptors, chunk_cosine_sim

stride = 2
patch_size = 8
device = "cuda:0" if torch.cuda.is_available() else "cpu"
extractor = ViTExtractor(stride=stride)

def object_in_scene(image_path: str, object_descriptors: torch.Tensor, threshold: float = 0.6):
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
        return [False, []]
    idx = idx % (num_patches[0] * num_patches[1])
    y_desc, x_desc = idx // num_patches[1], idx % num_patches[1]
    coordinates = [(x_desc - 1) * stride + stride + patch_size // 2 - .5,
              (y_desc - 1) * stride + stride + patch_size // 2 - .5]
    return [True, coordinates]


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



if __name__ == '__main__':
    base_path = "training_data/training_set"
    task = os.path.join(base_path, "PickUpFork-v0")
    image = os.path.join(task,"1687693896925_1.png")
    all_descriptors = pkl.load(open(f"training_data/descriptors.pkl", "rb"))
    labels = pkl.load(open(f"training_data/descriptor_labels.pkl", "rb"))
    descriptor_dict = reformat_descriptors(all_descriptors,labels)
    for ycb_object in descriptor_dict:
        [is_present, coords] = object_in_scene(image, ycb_object["descriptors"])
        print(f"{ycb_object['object']} {'' if is_present else 'not'} present {f'at {coords}' if is_present else ''}")
    # compare_image(all_descriptors, labels, image)
import sys

import os
from typing import List
import random
import torch.utils.data
import yaml
import torch
import pickle as pkl

stride = 2
patch_size = 8
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)


def get_descriptor_for_labeled(image_path: str, coordinates: str):
    x_int = int(coordinates.split(" ")[0].split(".")[0])
    y_int = int(coordinates.split(" ")[-1].split(".")[0])
    tensor_path = f"{image_path.split('.')[0]}.pt"
    with torch.inference_mode():
        if os.path.exists(tensor_path):
            descs = torch.load(tensor_path)
            num_patches = (61,61)
            load_size = (128,128)
        x_descr = int(num_patches[1] / load_size[1] * x_int)
        y_descr = int(num_patches[0] / load_size[0] * y_int)
        descriptor_idx = num_patches[1] * y_descr + x_descr
        return descs[:,:,int(descriptor_idx),:].unsqueeze(0)


def read_points(task_folder: str):
    with open(os.path.join(task_folder, "labels.yml"), "r") as label_file:
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


def extract_descriptors(tasks: List[str], descriptor_amount: int = None):
    # in all_descriptors all the descriptors for the objects will be safed
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
        if descriptor_amount is not None:
            max_idx = min(max_idx, descriptor_amount)
        for key_idx in range(max_idx):
            [file, point] = zipped[zip_idx[key_idx]]
            descriptor = get_descriptor_for_labeled(os.path.join(os.path.join(task_folder, file)),
                                                    point).detach()
            current_descriptors = torch.cat([current_descriptors, descriptor])
        descriptors.append({
            "object": task,
            "descriptors": torch.transpose(current_descriptors,0,2)
        })
    return descriptors


def extract_topk_descr(descr: List[dict], method: str, k: int = 5):
    assert method in ["similar", "dissimilar"]
    for task in descr:
        curr_descriptor = task["descriptors"]
        similarities = chunk_cosine_sim(curr_descriptor, curr_descriptor)
        summed_similarities = torch.sum(similarities, dim=3)
        if method == "similar":
            largest = True
        else:
            largest = False
        top_k_descriptors = torch.topk(summed_similarities, k, largest=largest)
        topk_indices = top_k_descriptors.indices.squeeze()
        new_descriptors = torch.index_select(curr_descriptor, 2, topk_indices)
        task["descriptors"] = new_descriptors
    return descr


def get_similar(img_path: str, k: int = 5):
    descriptors = extract_descriptors(tasks=os.listdir(img_path))
    return extract_topk_descr(descriptors, "similar", k)


def get_dissimilar(img_path: str, k: int = 5):
    descriptors = extract_descriptors(tasks=os.listdir(img_path))
    return extract_topk_descr(descriptors, "dissimilar", k)


if __name__ == '__main__':
    training_data_path = f"training_data/training_set"
    pkl.dump(get_similar(training_data_path), open(f"training_data/descriptor_data.pkl", "wb"))

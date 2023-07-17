import math
from typing import Tuple, List

import torch
import os


stride = 4
patch_size = 8

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


class DescriptorSet:

    def __init__(self, task:str, descriptors: torch.Tensor, descriptor_positions: List[list],  info:str=None):
        assert descriptors.shape[2] == len(descriptor_positions)
        self.task = task
        self.descriptors = descriptors
        self.descriptor_positions = descriptor_positions
        self.info = info

    def extract_top_k(self, method: str, k: int):
        assert method in ["similar", "dissimilar"]
        curr_descriptor = self.descriptors
        similarities = chunk_cosine_sim(curr_descriptor, curr_descriptor)
        summed_similarities = torch.sum(similarities, dim=3)
        if method == "similar":
            largest = True
        else:
            largest = False
        top_k_descriptors = torch.topk(summed_similarities, k, largest=largest)
        top_k_indices = top_k_descriptors.indices.squeeze()
        new_descriptors = torch.index_select(curr_descriptor, 2, top_k_indices)
        new_descriptors_idx = []
        for idx in top_k_indices:
            new_descriptors_idx.append(self.descriptor_positions[idx.item()])
        return DescriptorSet(self.task, new_descriptors, new_descriptors_idx)

    def _get_descriptor_for_labeled(self, image_path: str, coordinates: str):
        x_int = int(coordinates.split(" ")[0].split(".")[0])
        y_int = int(coordinates.split(" ")[-1].split(".")[0])
        tensor_path = f"{image_path.split('.')[0]}.pt"
        with torch.inference_mode():
            if os.path.exists(tensor_path):
                descs = torch.load(tensor_path).detach()
                pach_am = int(math.sqrt(descs.shape[2]))
                num_patches = (pach_am, pach_am)
                load = (num_patches[0] - 1) * stride + patch_size
                load_size = (load, load)
            x_descr = int(num_patches[1] / load_size[1] * x_int)
            y_descr = int(num_patches[0] / load_size[0] * y_int)
            descriptor_idx = num_patches[1] * y_descr + x_descr
            return descs[:, :, int(descriptor_idx), :].unsqueeze(0)

    def reload_descriptors(self):
        base_folder = "training_data/training_set"
        descriptors = []
        for file, point in self.descriptor_positions:
            descriptor = self._get_descriptor_for_labeled(os.path.join(os.path.join(base_folder,self.task), file), point).detach()
            descriptors.append(descriptor)
        self.descriptors = torch.transpose(torch.stack(descriptors).squeeze(1),0,2)

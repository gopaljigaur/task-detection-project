import math
from typing import Tuple, List
from dino.correspondences import chunk_cosine_sim
import torch
import os

stride = 4
patch_size = 8


def get_descriptor_for_labeled(image_path: str, coordinates: tuple[float, float, float]):
    x_int, y_int, _ = list(map(int, coordinates))
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


class DescriptorSet:
    """
    It contains the descriptors, descriptor positions, and some additional information related to the task.
    """

    def __init__(self, task: str, descriptors: torch.Tensor, descriptor_positions: List[list], info: str = None):
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

    def reload_descriptors(self):
        base_folder = "training_data/training_set"
        descriptors = []
        for file, point in self.descriptor_positions:
            descriptor = get_descriptor_for_labeled(os.path.join(os.path.join(base_folder, self.task), file),
                                                          point).detach()
            descriptors.append(descriptor)
        self.descriptors = torch.transpose(torch.stack(descriptors).squeeze(1), 0, 2)

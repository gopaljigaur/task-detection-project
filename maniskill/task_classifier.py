import math
from typing import List, Callable, Union

import torch.nn as nn
import torch
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from dino.extractor import ViTExtractor
import torch.nn.functional as F
import pickle as pkl

from maniskill.helpers.DescriptorConfiguration import DescriptorConfiguration
from maniskill.helpers.TensorDataSet import TensorDataSet
from dino.correspondences import chunk_cosine_sim

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class TaskClassifier(nn.Module):
# why n_classes = 5?
    def __init__(self, vit_stride=2, vit_patch_size=8, n_classes: int = 5, threshold: List[float] = None,
                 descriptors: dict = None):
        super().__init__()
        self.n_classes = n_classes
        self.extractor = ViTExtractor(stride=vit_stride)
        for param in self.extractor.model.parameters():
            param.requires_grad = False
        self.obj_finder = ObjectLocator(self.extractor, descriptors=descriptors)
        # create fc layer to get task

        # map from image pixels & depth to world space
        self.classifier = nn.Linear(self.n_classes * 3, self.n_classes).cuda()

    def forward(self, x: torch.Tensor):
        # extract descriptors from image
        with torch.inference_mode():
            x = self.extractor.extract_descriptors(x)
        # map descriptors to object locations
        x = self.obj_finder(x)
        # get prediction using a linear layer
        copied_tensor = x.clone().detach().requires_grad_(True)
        copied_tensor = self.classifier(copied_tensor)
        copied_tensor = F.relu(copied_tensor)
        return F.softmax(copied_tensor, dim=1)

    def cached_forward(self, x: torch.Tensor):
        x = self.obj_finder(x)
        x = x.clone().detach().requires_grad_(True)
        # get prediction using a linear layer
        x = self.classifier(x)
        x = F.relu(x)
        return F.softmax(x, dim=1)

    def cache_object_output(self, x: torch.Tensor):
        # extract descriptors from image
        with torch.inference_mode():
            x = self.extractor.extract_descriptors(x)
        # map descriptors to object locations
        return x.clone().detach().requires_grad_(True)

    def convert_to_image(self, image_path: str):
        return self.extractor.preprocess(image_path)[1]

    def get_trainable_params(self):
        return self.classifier.parameters()

    def preprocess(self, image_base_path: str):
        dataset = ImageFolder(image_base_path, transform=transforms.ToTensor())
        for [image, label] in dataset.imgs:
            with open(image, "rb") as f:
                img = Image.open(f).convert("RGB")
                transformer = transforms.ToTensor()
                tensor = transformer(img).unsqueeze(0).to(device)
                tensor = self.cache_object_output(tensor)
                torch.save(tensor, f"{image.split('.')[0]}.pt")

    def load_cache(self, image_base_path: str, filter_fn: Callable[[str], bool] = None):
        dataset = ImageFolder(image_base_path, transform=transforms.ToTensor())
        if filter is None:
            return [dataset.class_to_idx, TensorDataSet(dataset.imgs)]
        else:
            return [dataset.class_to_idx, TensorDataSet(list(filter(filter_fn, dataset.imgs)))]


class ObjectLocator:

    def __init__(self,
                 extractor: ViTExtractor,
                 descriptor_file="training_data/hand_descriptors.pkl",
                 location_method: str = None,
                 descriptors: dict = None):
        self.extractor = extractor
        if descriptors is None:
            self.descriptor_configurations = pkl.load(open(descriptor_file, "rb"))
        else:
            self.descriptor_configurations = descriptors
        if location_method is None:
            self.location_method = self._aggregate
        else:
            if location_method == "find_one":
                self.location_method = self._find_one

    def __call__(self, x: torch.Tensor):
        return self.forward(x)

    def _aggregate(self, x: torch.Tensor, object_configuration: DescriptorConfiguration):
        object_descriptors = object_configuration.descriptor_set.descriptors
        threshold = object_configuration.threshold
        aggregation_percentage = object_configuration.aggregation_percentage

        num_patches = [int(math.sqrt(x.shape[2])), int(math.sqrt(x.shape[2]))]
        object_locations = []
        similarities = chunk_cosine_sim(object_descriptors, x)
        sims, idxs = torch.topk(similarities, 1)
        found_object = sims > threshold
        # 3rd co-ordinate is 0 if object not found
        # object location: [x, y,
        not_found = torch.tensor([0, 0, 0], device=device)
        found_amount_mask = torch.sum(found_object, dim=2).squeeze(1) >= object_descriptors.shape[
            2] * aggregation_percentage
        for i in range(x.shape[0]):
            if not found_amount_mask[i]:
                object_locations.append(not_found)
            else:
                patch_idxs = idxs[i, found_object[i]].unsqueeze(1)
                coordinates = self._extract_coordinates_from_patch(patch_idxs, num_patches, self.extractor.stride,
                                                                   self.extractor.model.patch_embed.patch_size)
                object_location = torch.mean(coordinates, dim=0)
                object_locations.append(object_location)
        return torch.stack(object_locations)

    def _find_one(self, x: torch.Tensor, object_configuration: DescriptorConfiguration):
        object_descriptors = object_configuration.descriptor_set.descriptors
        threshold = object_configuration.threshold

        num_patches = [int(math.sqrt(x.shape[2])), int(math.sqrt(x.shape[2]))]
        similarities = chunk_cosine_sim(object_descriptors, x)
        # find best matching position
        sims, idxs = torch.topk(similarities.flatten(1), 1)
        sim, idx = sims[0], idxs[0]
        # if sim < self.threshold:
        #     # object not currently present in scene
        #     obj_locations = torch.cat((obj_locations, torch.tensor([-99, -99, -99], device=device)))
        #     continue
        patch_idx = idxs % (num_patches[0] * num_patches[1])
        # y_desc, x_desc = patch_idx // num_patches[1], patch_idx % num_patches[1]
        # coordinates = torch.cat(((x_desc - 1) * self.extractor.stride[1] + self.extractor.stride[
        #     1] + self.extractor.model.patch_embed.patch_size // 2 - .5,
        #                          (y_desc - 1) * self.extractor.stride[0] + self.extractor.stride[
        #                              0] + self.extractor.model.patch_embed.patch_size // 2 - .5,
        #                          torch.zeros((idxs.shape[0], 1), device=device)), 1)
        coordinates = self._extract_coordinates_from_patch(patch_idx, num_patches, self.extractor.stride,
                                                           self.extractor.model.patch_embed.patch_size)
        # obj_locations = torch.cat((obj_locations, torch.tensor(coordinates, device=device)))
        not_present = torch.tensor([0, 0, 0], device=device)
        return torch.where(sims > threshold, coordinates, not_present)

    def _extract_coordinates_from_patch(self, patch_idx: Union[torch.Tensor, int], num_patches: [int, int],
                                        stride: [int, int], patch_size: [int, int]):
        amount = 1
        if torch.is_tensor(patch_idx):
            amount = patch_idx.shape[0]
        y_desc, x_desc = patch_idx // num_patches[1], patch_idx % num_patches[1]
        coordinates = torch.cat(((x_desc - 1) * stride[1] + stride[
            1] + patch_size // 2 - .5,
                                 (y_desc - 1) * stride[0] + stride[
                                     0] + patch_size // 2 - .5,
                                 torch.zeros((amount, 1), device=device)), 1)
        # substitute 3rd cordinate by depth
        return coordinates

    def forward(self, x: torch.Tensor):
        if self.extractor.num_patches is not None:
            num_patches = self.extractor.num_patches
        with torch.inference_mode():
            obj_locations = torch.tensor([], device=device)
            for object_conf in self.descriptor_configurations.values():
                # obj_descr = obj["descriptors"]
                # Finding object locations in the new image after comparing with the descriptor
                # these descriptors are extracted from the descriptor file specified in the ObjectLocator constructor
                obj_locations = torch.cat((obj_locations, self.location_method(x, object_conf)), dim=1)
            # obj_locations.requires_grad=True
            return obj_locations


# Create a class for each object detected by the ObjectLocator, that stores the object's name and location
class DetectedObject:
    def __init__(self, classname: str, location: torch.Tensor):
        self.classname = classname
        self.location = location

    def __str__(self):
        return f"{self.classname} at {self.location}"

    def __repr__(self):
        return f"{self.classname} at {self.location}"


"""Configuration:

find_one: similarity_threshold_
aggregate: similarity_threshold, aggregation_percentage
"""
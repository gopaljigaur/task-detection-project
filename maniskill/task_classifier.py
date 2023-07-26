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

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class TaskClassifier(nn.Module):

    def __init__(self, vit_stride=2, vit_patch_size=8, n_classes: int = 5, threshold: List[float] = None, descriptors: dict=None):
        super().__init__()
        self.n_classes = n_classes
        self.extractor = ViTExtractor(stride=vit_stride)
        for param in self.extractor.model.parameters():
            param.requires_grad = False
        self.obj_finder = ObjectLocator(self.extractor, descriptors=descriptors)
        # create fc layer to get task



    def forward(self, x: torch.Tensor):
        # x = B, H, W, C
        # extract descriptors from image
        with torch.inference_mode():
            x = self.extractor.extract_descriptors(x)
        # x = B, 1, patch_amount, 384
        # map descriptors to object locations
        x = self.obj_finder(x)
        # x = B, num_classes * 3

        copied_tensor = x.clone().detach().requires_grad_(True)
        return copied_tensor

    def cached_forward(self, x: torch.Tensor):
        # x is the saved descriptor for an image
        # x = B, 1, patch_amount, 384
        x = self.obj_finder(x)
        # x = B, num_classes * 3
        x = x.clone().detach().requires_grad_(True)
        # get prediction using a linear layer
        return x

    def cache_object_output(self, x: torch.Tensor):
        # extract descriptors from image
        with torch.inference_mode():
            x = self.extractor.extract_descriptors(x)
            # x = B, 1, patch_amount, 384
        # map descriptors to object locations
        return x.clone().detach().requires_grad_(True)

    def convert_to_image(self, image_path: str):
        return self.extractor.preprocess(image_path)[1]


    def preprocess(self, image_base_path:str):
        dataset = ImageFolder(image_base_path, transform=transforms.ToTensor())
        for [image, label] in dataset.imgs:
            with open(image, "rb") as f:
                img = Image.open(f).convert("RGB")
                transformer = transforms.ToTensor()
                tensor = transformer(img).unsqueeze(0).to(device)
                tensor = self.cache_object_output(tensor)
                torch.save(tensor, f"{image.split('.')[0]}.pt")

    def calculate_and_save_object_locations(self, image_base_path:str, filter_fn: Callable[[str], bool]=None):
        dataset = ImageFolder(image_base_path, transform=transforms.ToTensor())
        images = dataset.imgs
        if filter_fn is not None:
            images = filter(filter_fn, images)
        for [image, label] in images:
            desc = torch.load(image.split(".")[0]+".pt")
            tensor = self.cached_forward(desc)
            tensor = tensor.reshape(5,3)[:,2]
            torch.save(tensor.cpu(), f"{image.split('.')[0]}.obj_agg")

    def load_cache(self, image_base_path: str, filter_fn: Callable[[str], bool]=None):
        dataset = ImageFolder(image_base_path, transform=transforms.ToTensor())
        if filter is None:
            return [dataset.class_to_idx, TensorDataSet(dataset.imgs)]
        else:
            return [dataset.class_to_idx, TensorDataSet(list(filter(filter_fn, dataset.imgs)))]

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


class ObjectLocator:

    def __init__(self,
                 extractor: ViTExtractor,
                 descriptor_file="training_data/optim.pkl",
                 location_method: str = None,
                 descriptors: dict = None,
                 class_mapping: dict = None):
        self.extractor = extractor
        if descriptors is None:
            self.descriptor_configurations = pkl.load(open(descriptor_file,"rb"))
        else:
            self.descriptor_configurations = descriptors
        if location_method is None:
            self.location_method = self._aggregate
        else:
            if location_method == "find_one":
                self.location_method = self._find_one
        self.class_mapping = class_mapping
        self.skip_agg = False
        if self.class_mapping is None:
            self.class_mapping = {}
            i=0
            for key in self.descriptor_configurations.keys():
                self.class_mapping[key] = i
                i+=1

    def __call__(self, x: torch.Tensor):
        return self.forward(x)

    def _aggregate(self, x: torch.Tensor, object_configuration: DescriptorConfiguration):
        # get descriptors for the searched objects from the configuration
        object_descriptors = object_configuration.descriptor_set.descriptors
        # get threshold for the searched objects from the configuration
        threshold = object_configuration.threshold
        # get aggregation_percentage for the searched objects from the configuration
        aggregation_percentage = object_configuration.aggregation_percentage
        # extract information about num_patches, needed later for localization
        num_patches = [int(math.sqrt(x.shape[2])),int(math.sqrt(x.shape[2]))]
        object_locations = []
        similarities = chunk_cosine_sim(object_descriptors, x)
        # similarities = B, 1, descriptor_amount, patch_amount
        sims, idxs = torch.topk(similarities, 1)
        # sims, idxs = B, 1, descriptor_amount, 1
        found_object = sims > threshold
        # found_object : BooleanTensor = B, 1, descriptor_amount, 1
        not_found = torch.tensor([0, 0, 0], device=device)
        # found_amount_mask:
        # torch.sum(...).squeeze(1) returns the amount of objects found for the image -> shape: B, 1
        # then this is compared with the aggregation_percentage multiplied with the amount of object_descriptors used.
        # This results in a B, 1 Boolean Tensor
        found_amount_mask = torch.sum(found_object, dim=2).squeeze(1) >= object_descriptors.shape[2] * aggregation_percentage
        # the following can be maybe sped up using something like torch.where
        if self.skip_agg:
            found_amount_mask = torch.tensor([True] * x.shape[0], device=device)
        for i in range(x.shape[0]):
            if not found_amount_mask[i]:
                # append [0,0,0]
                object_locations.append(not_found)
            else:
                patch_idxs = idxs[i, found_object[i]].unsqueeze(1)
                # patch_idxs: found_amount x 1
                coordinates = self._extract_coordinates_from_patch(patch_idxs, num_patches, self.extractor.stride, self.extractor.model.patch_embed.patch_size)
                # coordinates = shape: found_amount x 3
                object_location = torch.mean(coordinates, dim=0)
                # object_location shape: 3,
                object_location[2] = torch.sum(found_object, dim=2).squeeze(1)[i]/object_descriptors.shape[2]
                # set aggregation percentage as 3rd value
                object_locations.append(object_location)
        return torch.stack(object_locations)


    def _find_one(self, x: torch.Tensor, object_configuration: DescriptorConfiguration):
        object_descriptors = object_configuration.descriptor_set.descriptors
        threshold = object_configuration.threshold

        num_patches = [int(math.sqrt(x.shape[2])),int(math.sqrt(x.shape[2]))]
        similarities = chunk_cosine_sim(object_descriptors, x)
        # find best matching position
        sims, idxs = torch.topk(similarities.flatten(1), 1)
        sim, idx = sims[0], idxs[0]
        patch_idx = idxs % (num_patches[0] * num_patches[1])
        coordinates = self._extract_coordinates_from_patch(patch_idx, num_patches, self.extractor.stride, self.extractor.model.patch_embed.patch_size)
        # obj_locations = torch.cat((obj_locations, torch.tensor(coordinates, device=device)))
        not_present = torch.tensor([0, 0, 0], device=device)
        return torch.where(sims > threshold, coordinates, not_present)

    def _extract_coordinates_from_patch(self, patch_idx: Union[torch.Tensor, int], num_patches: [int, int], stride: [int,int], patch_size: [int,int]):
        amount = 1
        if torch.is_tensor(patch_idx):
            amount = patch_idx.shape[0]
        y_desc, x_desc = patch_idx // num_patches[1], patch_idx % num_patches[1]
        coordinates = torch.cat(((x_desc - 1) * stride[1] + stride[
            1] + patch_size // 2 - .5,
                                 (y_desc - 1) * stride[0] + stride[
                                     0] + patch_size // 2 - .5,
                                 torch.zeros((amount, 1), device=device)), 1)
        return coordinates


    def forward(self, x: torch.Tensor):
        # x = B, 1, patch_amount, 384
        if self.extractor.num_patches is not None:
            num_patches = self.extractor.num_patches
        with torch.inference_mode():
            obj_locations = torch.tensor([], device=device)
            # loop over every object to check whether it is in the scene
            if self.class_mapping is None:
                for object_conf in self.descriptor_configurations.values():
                    # different location methods can be used, default is _aggregation
                    obj_locations = torch.cat((obj_locations, self.location_method(x, object_conf)), dim=1)
                    # output of self.location_method = B, 3
                    # concatenated at dim 1 -> leads to dimensions B, num classes * 3
            else:
                # we use class_mapping so the indexes of the outputs will be the same as the labels.
                for key in self.class_mapping.keys():
                    object_conf = self.descriptor_configurations[key]
                    # different location methods can be used, default is _aggregation
                    obj_locations = torch.cat((obj_locations, self.location_method(x, object_conf)), dim=1)
                    # output of self.location_method = B, 3
                    # concatenated at dim 1 -> leads to dimensions B, num classes * 3
            return obj_locations



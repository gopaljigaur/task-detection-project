from typing import List

import torch.nn as nn
import torch.optim as optim
import torch
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from dino.extractor import ViTExtractor
import torch.nn.functional as F
import pickle as pkl

from maniskill.custom_tasks.helpers.TensorDataSet import TensorDataSet

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class TaskClassifier(nn.Module):

    def __init__(self, vit_stride=2, vit_patch_size=8, n_classes: int = 5, threshold: List[float] = None, descriptors: List[dict] =None):
        super().__init__()
        self.n_classes = n_classes
        self.extractor = ViTExtractor(stride=vit_stride)
        for param in self.extractor.model.parameters():
            param.requires_grad = False
        self.obj_finder = ObjectLocator(self.extractor,threshold=threshold, descriptors=descriptors)
        # create fc layer to get task

        #map from image pixels & depth to world space

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
        x = x.detach().requires_grad_(True)
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

    def preprocess(self, image_base_path:str):
        dataset = ImageFolder(image_base_path, transform=transforms.ToTensor())
        for [image, label] in dataset.imgs:
            with open(image, "rb") as f:
                img = Image.open(f).convert("RGB")
                transformer = transforms.ToTensor()
                tensor = transformer(img).unsqueeze(0).to(device)
                tensor = self.cache_object_output(tensor)
                torch.save(tensor, f"{image.split('.')[0]}.pt")

    def load_cache(self, image_base_path: str):
        dataset = ImageFolder(image_base_path, transform=transforms.ToTensor())
        return [dataset.class_to_idx, TensorDataSet(dataset.imgs)]

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
                 descriptor_labels: str = f"training_data/descriptor_data.pkl",
                 descriptors: List[dict] = None,
                 threshold: List[float] = None):
        self.extractor = extractor
        if descriptors is None:
            self.object_descriptors = pkl.load(open(descriptor_labels, "rb"))
        else:
            self.object_descriptors = descriptors
        if threshold is None:
            self.threshold = 0.55 * len(self.object_descriptors)
        else:
            self.threshold = threshold

    def __call__(self, x: torch.Tensor):
        return self.forward(x)

    def forward(self, x: torch.Tensor):
        num_patches = [61,61]
        if self.extractor.num_patches is not None:
            num_patches = self.extractor.num_patches
        with torch.inference_mode():
            obj_locations = torch.tensor([], device=device)
            for threshold, obj in zip(self.threshold, self.object_descriptors):
                obj_descr = obj["descriptors"]
                similarities = chunk_cosine_sim(obj_descr, x)
                # find best matching position
                sims, idxs = torch.topk(similarities.flatten(1), 1)
                sim, idx = sims[0], idxs[0]
                # if sim < self.threshold:
                #     # object not currently present in scene
                #     obj_locations = torch.cat((obj_locations, torch.tensor([-99, -99, -99], device=device)))
                #     continue
                patch_idx = idxs % (num_patches[0] * num_patches[1])
                y_desc, x_desc = patch_idx // num_patches[1], patch_idx % num_patches[1]
                coordinates = torch.cat(((x_desc - 1) * self.extractor.stride[1] + self.extractor.stride[1] + self.extractor.model.patch_embed.patch_size // 2 - .5,
                               (y_desc - 1) * self.extractor.stride[0] + self.extractor.stride[0] + self.extractor.model.patch_embed.patch_size // 2 - .5,
                               torch.zeros((idxs.shape[0],1), device=device)),1)
                # obj_locations = torch.cat((obj_locations, torch.tensor(coordinates, device=device)))
                not_present = torch.tensor([0, 0, 0], device=device)
                current_object_locations = torch.where(sims > threshold, coordinates, not_present)
                obj_locations = torch.cat((obj_locations, current_object_locations),dim=1)
            # obj_locations.requires_grad=True
        return obj_locations

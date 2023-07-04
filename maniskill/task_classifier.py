import torch.nn as nn
import torch.optim as optim
import torch
from dino.extractor import ViTExtractor
import torch.nn.functional as F
import pickle as pkl

from dino.inspect_similarity import chunk_cosine_sim

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class TaskClassifier(nn.Module):

    def __init__(self, vit_stride=2, vit_patch_size=8):
        super().__init__()
        self.extractor = ViTExtractor(stride=vit_stride)
        self.obj_finder = ObjectLocator(self.extractor)
        # create fc layer to get task
        self.classifier = nn.Linear(self.n_classes * 3, self.n_classes)

    def forward(self, x: torch.Tensor):
        # extract descriptors from image
        x = self.extractor.extract_descriptors(x)
        # map descriptors to object locations
        x = self.obj_finder(x)
        # get prediction using a linear layer
        x = self.classifier(x.flatten())
        x = F.relu(x)
        return F.softmax(x)

    def convert_to_image(self, image_path: str):
        return self.extractor.preprocess(image_path)[1]

    def get_trainable_params(self):
        return self.classifier.parameters()


class ObjectLocator:

    def __init__(self,
                 extractor: ViTExtractor,
                 descriptor_file: str = "training_data/descriptors.pkl",
                 descriptor_labels: str = "training_data/descriptor_labels.pkl",
                 threshold: float = 0.56):
        objects = []
        self.extractor = extractor
        self.threshold = threshold
        idx = 0
        all_descriptors = pkl.load(open(descriptor_file, "rb"))
        labels = pkl.load(open(descriptor_labels, "rb"))
        object_label = labels[0]
        for i in range(len(labels)):
            if not labels[i] == object_label:
                objects.append({
                    "object": object_label,
                    "descriptors": torch.transpose(all_descriptors[idx:i],0,2)
                })
                idx = i
                object_label = labels[idx]
        objects.append({
            "object": object_label,
            "descriptors": torch.transpose(all_descriptors[idx:],0,2)
        })
        self.object_descriptors = objects

    def __call__(self, x: torch.Tensor):
        return self.forward(x)

    def forward(self, x: torch.Tensor):
        with torch.inference_mode():
            obj_locations = torch.Tensor([], device=device)
            desc = self.extractor.extract_descriptors(x).detach()
            for obj in self.object_descriptors:
                obj_descr = obj["descriptors"]
                similarities = chunk_cosine_sim(obj_descr, desc)
                # find best matching position
                sims, idxs = torch.topk(similarities.flatten(), 1)
                sim, idx = sims[0], idxs[0]
                if sim < self.threshold:
                    # object not currently present in scene
                    obj_locations = torch.cat((obj_locations, torch.Tensor([-99,-99,-99]).to(device)))
                    continue
                patch_idx = idx % (self.extractor.num_patches[0] * self.extractor.num_patches[1])
                y_desc, x_desc = patch_idx // self.extractor.num_patches[1], patch_idx % self.extractor.num_patches[1]
                coordinates = [((x_desc - 1) * self.extractor.stride + self.extractor.stride + self.extractor.model.patch_embed.patch_size // 2 - .5).item(),
                               ((y_desc - 1) * self.extractor.stride + self.extractor.stride + self.extractor.model.patch_embed.patch_size // 2 - .5).item(),
                               0]
                obj_locations = torch.cat((obj_locations, torch.Tensor(coordinates, device=device)))
            return obj_locations


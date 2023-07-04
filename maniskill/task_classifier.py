import torch.nn as nn
import torch.optim as optim
import torch
from dino.extractor import ViTExtractor
import torch.nn.functional as F
import pickle as pkl


device = "cuda:0" if torch.cuda.is_available() else "cpu"


class TaskClassifier(nn.Module):

    def __init__(self, vit_stride=2, vit_patch_size=8, n_classes: int = 4):
        super().__init__()
        self.n_classes = n_classes
        self.extractor = ViTExtractor(stride=vit_stride)
        self.obj_finder = ObjectLocator(self.extractor)
        # create fc layer to get task
        self.classifier = nn.Linear(self.n_classes * 3, self.n_classes).cuda()

    def forward(self, x: torch.Tensor):
        # extract descriptors from image
        with torch.inference_mode():
            x = self.extractor.extract_descriptors(x)
            # map descriptors to object locations
            x = self.obj_finder(x)
        # get prediction using a linear layer
            x = x.flatten()
        copied_tensor = x.clone().detach().requires_grad_(True)
        copied_tensor = self.classifier(copied_tensor)
        copied_tensor = F.relu(copied_tensor)
        return F.softmax(copied_tensor)

    def convert_to_image(self, image_path: str):
        return self.extractor.preprocess(image_path)[1]

    def get_trainable_params(self):
        return self.classifier.parameters()


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
                    "descriptors": torch.transpose(all_descriptors[idx:i], 0, 2)
                })
                idx = i
                object_label = labels[idx]
        objects.append({
            "object": object_label,
            "descriptors": torch.transpose(all_descriptors[idx:], 0, 2)
        })
        self.object_descriptors = objects

    def __call__(self, x: torch.Tensor):
        return self.forward(x)

    def forward(self, x: torch.Tensor):
        with torch.inference_mode():
            obj_locations = torch.tensor([], device=device)
            for obj in self.object_descriptors:
                obj_descr = obj["descriptors"]
                similarities = chunk_cosine_sim(obj_descr, x)
                # find best matching position
                sims, idxs = torch.topk(similarities.flatten(), 1)
                sim, idx = sims[0], idxs[0]
                if sim < self.threshold:
                    # object not currently present in scene
                    obj_locations = torch.cat((obj_locations, torch.tensor([-99,-99,-99], device=device)))
                    continue
                patch_idx = idx % (self.extractor.num_patches[0] * self.extractor.num_patches[1])
                y_desc, x_desc = patch_idx // self.extractor.num_patches[1], patch_idx % self.extractor.num_patches[1]
                coordinates = [(x_desc.item() - 1) * self.extractor.stride[1] + self.extractor.stride[1] + self.extractor.model.patch_embed.patch_size // 2 - .5,
                               (y_desc.item() - 1) * self.extractor.stride[0] + self.extractor.stride[0] + self.extractor.model.patch_embed.patch_size // 2 - .5,
                               0]
                obj_locations = torch.cat((obj_locations, torch.tensor(coordinates, device=device)))
            return obj_locations

import os
from torchmetrics.functional import pairwise_manhattan_distance
import numpy as np
import torch.utils.data
import yaml
from dino.extractor import ViTExtractor
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import torch

torch.cuda.empty_cache()
stride = 2
patch_size = 8
device = "cuda:0" if torch.cuda.is_available() else "cpu"
extractor = ViTExtractor(stride=stride)

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

def import_image(image_path:str):
    image = mpimg.imread(image_path)
    # image = image.reshape((1, 4, 128, 128))
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image,(0,3,1,2))
    image = torch.Tensor(image[:, :3, :, :]).to(device)
    return image

def find_patch_num(coord:int):
    patch = int((coord - (patch_size/2)) / stride)
    return patch

def find_patch(x,y):
    return y * 31 + x

def transform_coord(coord):
    return min(int(max((coord-2)/4,0)),30)


def get_descriptor_for_labeled(image_path: str, coordinates: str):
    x_int = int(coordinates.split(" ")[0].split(".")[0])
    y_int = int(coordinates.split(" ")[-1].split(".")[0])
    with torch.inference_mode():
        descs = get_descriptors(image_path)
        num_patches, load_size = extractor.num_patches, extractor.load_size
        x_descr = int(num_patches[1] / load_size[1] * x_int)
        y_descr = int(num_patches[0] / load_size[0] * y_int)
        descriptor_idx = num_patches[1] * y_descr + x_descr
    return descs[:,:,int(descriptor_idx),:].unsqueeze(0)


def get_descriptors(image_path: str):
    with torch.inference_mode():
        image_batch, image_pil = extractor.preprocess(image_path)
        descs = extractor.extract_descriptors(image_batch.to(device))
        return descs.detach()


def compare_image(descriptors: torch.Tensor, image_path: str):
    descriptors = torch.transpose(descriptors,0,2)
    descr_b = get_descriptors(image_path)
    similarities = chunk_cosine_sim(descriptors, descr_b)
    sims, idxs = torch.topk(similarities.flatten(), 1)
    num_patches = extractor.num_patches
    for idx in idxs:
        idx = idx % (num_patches[0] * num_patches[1])
        y_desc, x_desc = idx // num_patches[1], idx % num_patches[1]
        fig, ax = plt.subplots()
        center = ((x_desc - 1) * stride + stride + patch_size // 2 - .5,
                  (y_desc - 1) * stride + stride + patch_size // 2 - .5)
        patch = plt.Circle(center, 2, color=(1, 0, 0, 0.75))
        ax.imshow(extractor.preprocess(image_path)[1])
        ax.add_patch(patch)
        plt.draw()
        plt.show()



def compare_descriptors(target_descriptors:torch.Tensor, comparator_descriptors: torch.Tensor):
    distance = pairwise_manhattan_distance(comparator_descriptors, target_descriptors)
    distance[torch.where(distance == 0.)] = 10000
    arg_min = torch.argmin(distance).item()
    target_argmin = arg_min % target_descriptors.shape[0]
    comparator_argmin = int(arg_min / target_descriptors.shape[0])
    return [distance[comparator_argmin][target_argmin].detach().item(), comparator_argmin]

def render_patch(image_path:str, patch:int):
    image = mpimg.imread(image_path)
    x_start = patch % int(128/stride)
    y_start = patch / (128/stride)
    [x_lwr,x_upr] = [x_start*stride,x_start*stride+8]
    [y_lwr,y_upr] = [int(y_start*stride),int(y_start*stride+8)]
    sub_image = image[y_lwr:y_upr,x_lwr:x_upr,:]
    plt.imshow(sub_image, extent=[x_lwr,x_upr,y_upr,y_lwr])
    plt.show()
    plt.imshow(image)
    plt.show()
    return 0



if __name__ == '__main__':
    training_data_path = f"training_data/training_set"
    for task in os.listdir(training_data_path):
        task_folder = os.path.join(training_data_path, task)
        labels = None
        with open(os.path.join(task_folder, "labels.yml"), "r") as label_file:
            print(task)
            try:
                labels = yaml.safe_load(label_file)
                descriptors = torch.tensor([],device=device)
                for key, value in labels.items():
                    for point_pair in value["points"]:
                        descriptor = get_descriptor_for_labeled(os.path.join(os.path.join(task_folder, key)), point_pair).detach()
                        descriptors = torch.cat([descriptors,descriptor])


                # calculate pairwise manhattan
                for image in os.listdir(task_folder):
                    if not ".png" in image:
                        continue
                    compare_image(descriptors, os.path.join(task_folder,image))

                #     embeddings = extractor.extract_descriptors(import_image(os.path.join(task_folder,image)))[0,0,:,:]
                    # [curr_min, patch_idx] = compare_descriptors(descriptors, embeddings)
                    # render_patch(os.path.join(task_folder,image),patch_idx)

            except yaml.YAMLError as exc:
                print(exc)

    # transform = transforms.Compose([transforms.ToTensor()])
    # dataset = torchvision.datasets.ImageFolder(root=training_data_path, transform=transform)
    # batch_size = 2
    # data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)
    #
    # extractor = ViTExtractor()
    # for images, labels in data_loader:
    #     images, labels = images.to(device), labels.to(device)
    #     # imgs should be imagenet normalized tensors. shape BxCxHxW
    #     descriptors = extractor.extract_descriptors(images)
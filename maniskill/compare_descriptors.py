import os

import numpy as np
import torch
import pickle as pkl
from typing import List
import gc

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from dino.extractor import ViTExtractor
from maniskill.extract_descriptors import chunk_cosine_sim, get_similar, get_dissimilar, \
    extract_descriptors
from maniskill.task_classifier import *

custom_single_tasks = ["PickupDrill-v0", "PickUpBlock-v0", "FindClamp-v0", "StoreScrewdriver-v0", "Mark-v0"]
stride = 2
patch_size = 8
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def check_gc():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass

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


def compare_single(img_src: str, task: str, custom_threshold: float = 0.55, truth_descriptors=None, filter_fn: Callable[[str], bool] = None):
    net = TaskClassifier(vit_stride=2, descriptors=truth_descriptors)
    [class_mapping, dataset] = net.load_cache(img_src, filter_fn=filter_fn)
    data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
    obj_finder = net.obj_finder
    threshold = [0.55] * len(class_mapping)
    threshold[class_mapping[task]] = custom_threshold
    obj_finder.threshold = threshold
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    multiple = 0
    total = len(dataset.img_tuples)
    for [tensors, labels] in data_loader:
        tensor = obj_finder(tensors)
        for i in range(tensor.shape[0]):
            if labels[i] == class_mapping[task]:
                if not tensor[i, class_mapping[task]*3] == 0.:
                    true_positive += 1
                else:
                    false_negative += 1
            else:
                if not tensor[i, class_mapping[task]*3] == 0.:
                    false_positive += 1
                else:
                    true_negative += 1
            if torch.count_nonzero(tensor[i]) > 2:
                multiple += 1
    # print(f"TP: {true_positive}, FN: {false_negative}, FP: {false_positive}, TN: {true_negative}, multiple: {multiple}, total:{total}")
    return [true_positive, false_negative, false_positive, true_negative, multiple, total]


def precision(tp, fp):
    if tp + fp ==0:
        return 0
    return tp/(tp+fp)

def recall(tp, fn):
    if tp + fn == 0:
        return 0
    return tp/(tp+fn)

def f1(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * ((precision*recall)/(precision + recall))

def try_configurations(filter_fn=None):
    base_path = "training_data/training_set"
    thresholds = [thr / 100 for thr in range(53, 64, 1)]
    results = {}
    for task in custom_single_tasks[4:]:
        print(f"{task} ", end="")
        results[task] = {}
        for threshold in thresholds:
            similar_desc = get_similar(base_path, k=5)
            sim_result = compare_single(base_path, task, threshold, similar_desc, filter_fn=filter_fn)
            dissimilar_desc = get_dissimilar(base_path, k=5)
            dis_result = compare_single(base_path, task, threshold, dissimilar_desc, filter_fn=filter_fn)
            all_desc = extract_descriptors(os.listdir(base_path))
            all_result = compare_single(base_path, task, threshold, all_desc, filter_fn=filter_fn)
            results[task][threshold] = {
                "similar": sim_result,
                "dissimilar": dis_result,
                "all_result": all_result
            }
            print(threshold, end=" ")
        pkl.dump(results, open(f"training_data/descriptor_configuration_result_{task}.pkl", "wb"))
        results = {}
        print()

def combine_results():
    results = {}
    for task in custom_single_tasks:
        result = pkl.load(open(f"training_data/descriptor_configuration_result_{task}.pkl", "rb"))
        processed_result = {
            "similar": {},
            "dissimilar": {},
            "all_results": {}
        }
        for threshold in result[task].keys():
            sim = result[task][threshold]["similar"]
            sim_rec = recall(sim[0], sim[1])
            sim_prec = precision(sim[0], sim[2])
            dissim = result[task][threshold]["dissimilar"]
            dissim_rec = recall(dissim[0], dissim[1])
            dissim_prec = precision(dissim[0], dissim[2])
            all_d = result[task][threshold]["all_result"]
            all_d_rec = recall(all_d[0], all_d[1])
            all_d_prec = precision(all_d[0], all_d[2])
            # processed_result[threshold] = {
            #     "similar": (sim_prec,sim_rec),
            #     "dissimlar": (dissim_prec, dissim_rec),
            #     "all_result": (all_d_prec, all_d_rec)
            # }
            processed_result["similar"][threshold] = (sim_prec, sim_rec, f1(sim_prec, sim_rec))
            processed_result["dissimilar"][threshold] = (dissim_prec, dissim_rec, f1(dissim_prec, dissim_rec))
            processed_result["all_results"][threshold] = (all_d_prec, all_d_rec, f1(all_d_prec, all_d_rec))
        results[task] = processed_result
    pkl.dump(results, open(f"training_data/results_combined.pkl", "wb"))

if __name__ == '__main__':
    # try_configurations(lambda name : "_1" in name[0])
    combine_results()


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
from maniskill.extract_descriptors import extract_descriptors
from maniskill.helpers.DescriptorConfiguration import DescriptorConfiguration
from maniskill.task_classifier import *

custom_single_tasks = ["PickupDrill-v0", "PickCube-v0", "FindClamp-v0", "StoreWrench-v0", "Hammer-v0"]
stride = 4
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


def compare_single(img_src: str, task: str, config: DescriptorConfiguration, filter_fn: Callable[[str], bool] = None):
    net = TaskClassifier(vit_stride=4, descriptors={config.descriptor_set.task:config})
    [class_mapping, dataset] = net.load_cache(img_src, filter_fn=filter_fn)
    data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    obj_finder = net.obj_finder
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    multiple = 0
    total = len(dataset.img_tuples)
    for [tensors, labels] in data_loader:
        tensor = obj_finder(tensors).detach()
        for i in range(tensor.shape[0]):
            if labels[i] == class_mapping[task]:
                if not tensor[i, 0] == 0.:
                    true_positive += 1
                else:
                    false_negative += 1
            else:
                if not tensor[i, 0] == 0.:
                    false_positive += 1
                else:
                    true_negative += 1
    # print(f"TP: {true_positive}, FN: {false_negative}, FP: {false_positive}, TN: {true_negative}, multiple: {multiple}, total:{total}")
    config.record_performance(true_positive,false_negative,false_positive,true_negative,multiple,total)


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

def try_configurations_for_tasks(tasks:List[str]):
    for task in tasks:
        try_configuration(task)

def try_configuration(task:str, filter_fn=None, name=".", thresholds=None, aggregate_thresholds=None, k_list=None):
    if aggregate_thresholds is None:
        aggregate_thresholds = [thr / 100 for thr in range(20, 60, 10)]
    if thresholds is None:
        thresholds = [thr / 100 for thr in range(30, 80, 10)]
    if k_list is None:
        k_list = [5,8,13]
    if not name == ".":
        if not os.path.exists(f"training_data/{name}"):
            os.mkdir(f"training_data/{name}")
    base_path = "training_data/training_set"
    results = {}
    print(f"{task} ", end="")
    torch.cuda.empty_cache()
    results[task] = []
    descriptor_sets = []
    all_desc = extract_descriptors([task])[0]
    descriptor_sets.append(all_desc)
    for k in k_list:
        sim_set = all_desc.extract_top_k("similar", k)
        sim_set.info = f"similar {k}"
        descriptor_sets.append(sim_set)
        dissim_set = all_desc.extract_top_k("dissimilar", k)
        dissim_set.info = f"dissimilar {k}"
        descriptor_sets.append(dissim_set)
    for threshold in thresholds:
        for aggregate_threshold in aggregate_thresholds:
            for descriptor_set in descriptor_sets:
                torch.cuda.empty_cache()
                configuration = DescriptorConfiguration(descriptor_set, threshold, aggregate_threshold)
                compare_single(base_path, task, configuration, filter_fn=filter_fn)
                results[task].append(configuration)
            print(threshold, end=" ")
    for descriptor_set in descriptor_sets:
        descriptor_set.descriptors = None
    pkl.dump(results, open(f"training_data/{name}/descriptor_configuration_result_{task}.pkl", "wb"))
    print()
    return results


def combine_results():
    folders = filter(lambda name: "_cam" in name ,os.listdir("training_data"))
    for folder in folders:
        results = {}
        for task in custom_single_tasks[:]:
            result = pkl.load(open(f"training_data/{folder}/descriptor_configuration_result_{task}.pkl", "rb"))
            sorted_results = sorted(result[task], key=lambda config : config.f1)
            results[task] = sorted_results
        pkl.dump(results, open(f"training_data/{folder}/results_combined.pkl", "wb"))


def extract_top_k_results(result_dict, k=10, sort_fn=lambda conf: conf.f1):
    new_res = {}
    descriptors = {}
    for task, result in result_dict.items():
        sorted_res = sorted(result, reverse=True, key=sort_fn)
        top_k = sorted_res[:k]
        task_res = []
        for entry in top_k:
            task_res.append((entry.descriptor_set.info, entry.threshold, entry.aggregation_percentage, sort_fn(entry),(entry.tp,entry.fp,entry.fn)))
        new_res[task] = task_res
        descriptors[task] = top_k[0]
    return (new_res, descriptors)

def optimize_configurations(task:str, iterations:int, k:int, performance_metric=lambda conf: (conf.f1,conf.precision)):
    thresholds = [thr / 100 for thr in range(20, 100, 20)]
    aggregate_perc = [perc / 100 for perc in range(20, 60, 10)]
    k_list = [8]
    top_k = []
    for i in range(1,iterations+1):
        print(f"Epoch {i}")
        print(f"Next iter: Thresholds: {thresholds}, Agg%: {aggregate_perc}, k_list: {k_list}")
        results = try_configuration(task,lambda name:"_1" in name[0],"hand_optim",thresholds,aggregate_perc, k_list)[task]
        sorted_res = sorted(results, reverse=True, key=performance_metric)
        top_k = sorted_res[:k]
        old_thresholds = []
        old_aggr = []
        old_k_list = []
        for conf in top_k:
            old_thresholds.append(conf.threshold)
            old_aggr.append(conf.aggregation_percentage)
            if conf.descriptor_set.info is not None:
                old_k_list.append(int(conf.descriptor_set.info.split(" ")[1]))
        old_thresholds = list(dict.fromkeys(old_thresholds))
        old_aggr = list(dict.fromkeys(old_aggr))
        old_k_list = list(dict.fromkeys(old_k_list))
        thresholds = []
        for threshold in old_thresholds:
            thresholds.append(threshold)
            thresholds.append(threshold + (0.2 / (2*i)))
            thresholds.append(threshold - (0.2 / (2*i)))
        thresholds = list(dict.fromkeys(thresholds))
        aggregate_perc = []
        for agg in old_aggr:
            aggregate_perc.append(agg)
            aggregate_perc.append(agg + (0.1 / (2*i)))
            aggregate_perc.append(agg + (0.1 / (2*i)))
        aggregate_perc = list(dict.fromkeys(aggregate_perc))
        k_list = []
        for curr_k in old_k_list:
            k_list.append(curr_k)
            k_list.append(curr_k+1)
            k_list.append(curr_k-1)
        k_list = list(dict.fromkeys(k_list))
    return top_k




if __name__ == '__main__':
    for i in range(len(custom_single_tasks)):
        optim_config = optimize_configurations(custom_single_tasks[i],8,6)
        pkl.dump(optim_config, open(f"training_data/optim_{custom_single_tasks[i]}","wb"))
    all_desc = {}
    for i in range(len(custom_single_tasks)):
        config = pkl.load(open(f"training_data/optim_{custom_single_tasks[i]}","rb"))
        config[0].descriptor_set.reload_descriptors()
        all_desc[custom_single_tasks[i]]=(config[0])
    pkl.dump(all_desc, open(f"training_data/optim.pkl","wb"))
    # try_configurations(lambda name : "_0" in name[0], name="static_cam_v0")
    # try_configurations(lambda name : "_1" in name[0], name="hand_cam_v0")
    # try_configurations(name="both_cam_v0")
    # try_configurations()
    # combine_results()
    # results_hand = pkl.load(open(f"training_data/hand_cam_v0/results_combined.pkl", "rb"))
    # results_static = pkl.load(open(f"training_data/static_cam_v0/results_combined.pkl", "rb"))
    # hand_k_top, descr_hand = extract_top_k_results(results_hand)
    # static_k_top, descr_static = extract_top_k_results(results_static, sort_fn=lambda c: (c.precision,c.f1))
    # for (hand, static) in zip(descr_hand.values(), descr_static.values()):
    #     hand.descriptor_set.reload_descriptors()
    #     static.descriptor_set.reload_descriptors()
    # # pkl.dump(descr_hand, open(f"training_data/hand_descriptors.pkl","wb"))
    # # pkl.dump(descr_static, open(f"training_data/static_descriptors.pkl", "wb"))
    # print(hand_k_top)
    # print(static_k_top)

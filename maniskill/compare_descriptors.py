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

multiple_object_tasks = {"CollectTools-v0":["StoreWrench-v0","PickupDrill-v0","Hammer-v0"], "DrillBlock-v0":["PickupDrill-v0","PickUpBlock-v0"], "FastenBlock-v0":["PickUpBlock-v0","FindClamp-v0"], "ReleaseBlock-v0":["PickUpBlock-v0","FindClamp-v0"]}

custom_single_tasks = ["PickupDrill-v0", "PickUpBlock-v0", "FindClamp-v0", "StoreWrench-v0", "Hammer-v0"]
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
    data_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)
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

def try_configuration(task:str, filter_fn=None, name=".", thresholds=None, k_list=None, label_name ="labels.yml"):
    if thresholds is None:
        thresholds = [thr / 100 for thr in range(30, 80, 10)]
    if k_list is None:
        k_list = [5,8,13]
    if not name == ".":
        if not os.path.exists(f"training_data/{name}"):
            os.mkdir(f"training_data/{name}")
    base_path = "training_data/validation_multiple"
    results = {}
    print(f"{task} ", end="")
    torch.cuda.empty_cache()
    results[task] = []
    descriptor_sets = []
    all_desc = extract_descriptors([task],fname=label_name)[0]
    descriptor_sets.append(all_desc)
    for k in k_list:
        sim_set = all_desc.extract_top_k("similar", k)
        sim_set.info = f"similar {k}"
        descriptor_sets.append(sim_set)
        dissim_set = all_desc.extract_top_k("dissimilar", k)
        dissim_set.info = f"dissimilar {k}"
        descriptor_sets.append(dissim_set)
    for threshold in thresholds:
        for descriptor_set in descriptor_sets:
            torch.cuda.empty_cache()
            configuration = DescriptorConfiguration(descriptor_set, threshold, 0)
            run_experiment_multiple(base_path, task, configuration, filter_fn=filter_fn)
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


def optimize_configurations(task:str, iterations:int, k:int, performance_metric=lambda conf: (conf.f1,conf.precision), filter_fn= None, name:str ="optim", explore =0.2, label_name="labels.yml"):
    thresholds = [thr / 100 for thr in range(20, 100, 20)]
    k_list = [8,14,20]
    # k_list = []
    top_k = []
    all_res =[]
    for i in range(1,iterations+1):
        print(f"Epoch {i}")
        print(f"Next iter: Thresholds: {thresholds},  k_list: {k_list}")
        results = try_configuration(task,filter_fn=filter_fn,name=name,thresholds=thresholds, k_list=k_list, label_name = label_name)[task]
        sorted_res = sorted(results, reverse=True, key=performance_metric)
        top_k = sorted_res[:k]
        all_res.extend(top_k)
        old_thresholds = []
        old_k_list = []
        for conf in top_k:
            old_thresholds.append(conf.threshold)
            if conf.descriptor_set.info is not None:
                old_k_list.append(int(conf.descriptor_set.info.split(" ")[1]))
        old_thresholds = list(dict.fromkeys(old_thresholds))
        old_k_list = list(dict.fromkeys(old_k_list))
        thresholds = []
        thr_min = min(old_thresholds)
        thr_max = max(old_thresholds)
        thr_step = (thr_max - thr_min)/4
        if thr_step == 0.:
            thr_step = thr_min * explore
            thr_max = thr_min + thr_step * 2
            thr_min = thr_min - thr_step * 2
        thr_range = torch.arange(thr_min + thr_step, thr_max, thr_step)
        for th in thr_range:
            thresholds.append(round(th.item(),3))
        k_list = []
        if len(old_k_list) == 0:
            continue
        k_min = min(old_k_list)
        k_max = max(old_k_list)
        k_step = (k_max - k_min)/4
        if k_step == 0.:
            k_step = k_min * 4 * explore
            k_max = k_min + k_step * 2
            k_min = k_min - k_step * 2
        k_min = max(2,k_min)
        k_range = torch.arange(k_min + k_step, k_max, k_step)
        if k_step == 0.:
            continue
        for kk in k_range:
            k_list.append(int(kk.item()))
        k_list = list(dict.fromkeys(k_list))
    return sorted(all_res, reverse=True, key=performance_metric)


def run_experiment(img_src: str, task: str, config: DescriptorConfiguration, filter_fn:Callable = None):
    net = TaskClassifier(vit_stride=4, descriptors={config.descriptor_set.task: config})
    [class_mapping, dataset] = net.load_cache(img_src, filter_fn=filter_fn)
    data_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)
    obj_finder = net.obj_finder
    obj_finder.skip_agg = True
    # class_mapping = obj_finder.class_mapping
    pos = torch.tensor([],device=device)
    neg = torch.tensor([],device=device)
    for [tensors, labels] in data_loader:
        tensor = obj_finder(tensors).detach()
        should_be = labels == class_mapping[task]
        thresholds = tensor[:,2]
        pos = torch.cat((pos, thresholds[should_be]))
        neg = torch.cat((neg, thresholds[~should_be]))
    [metrics, agg] = compute_aggregation_percentage(pos,neg)
    config.aggregation_percentage = agg
    config.record_performance(metrics["tp"],metrics["fn"],metrics["fp"],-1,-1,-1)


def run_experiment_multiple(img_src: str, task: str, config: DescriptorConfiguration, filter_fn: Callable = None):
    net = TaskClassifier(vit_stride=4, descriptors={config.descriptor_set.task: config})
    [class_mapping, dataset] = net.load_cache(img_src, filter_fn=filter_fn)
    data_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)
    obj_finder = net.obj_finder
    obj_finder.skip_agg = True
    # class_mapping = obj_finder.class_mapping
    pos = torch.tensor([], device=device)
    neg = torch.tensor([], device=device)
    good_labels = []
    for multiple_task, single_tasks in multiple_object_tasks.items():
        if task in single_tasks:
            good_labels.append(class_mapping[multiple_task])
    for [tensors, labels] in data_loader:
        tensor = obj_finder(tensors).detach()
        should_be = torch.tensor([False] * tensor.shape[0], device=device)
        for label in good_labels:
            current_label = labels == label
            should_be = torch.logical_or(should_be, current_label.to(device))
        # should_be = labels == class_mapping[task]
        thresholds = tensor[:, 2]
        pos = torch.cat((pos, thresholds[should_be]))
        neg = torch.cat((neg, thresholds[~should_be]))
    [metrics, agg] = compute_aggregation_percentage(pos, neg)
    config.aggregation_percentage = agg
    config.record_performance(metrics["tp"], metrics["fn"], metrics["fp"], metrics["tn"], -1, -1)


def compute_aggregation_percentage(pos_thr: torch.Tensor, neg_thr: torch.Tensor):
    pos = torch.sort(pos_thr).values
    neg = torch.sort(neg_thr).values
    agg_idx = 0
    agg = pos[0].item()
    metrics = get_metrics(pos, neg, agg)
    for i in range(1, pos.shape[0]):
        curr_agg = pos[i].item()
        curr_metrics = get_metrics(pos,neg,curr_agg)
        if curr_metrics["f1"] > metrics["f1"]:
            metrics = curr_metrics
            agg = curr_agg
    return [metrics, agg]

def get_metrics(pos:torch.Tensor, neg: torch.Tensor, agg: float):
    tp = torch.count_nonzero(pos >= agg).item()
    fn = torch.count_nonzero(pos < agg).item()
    fp = torch.count_nonzero(neg >= agg).item()
    tn = torch.count_nonzero(neg < agg).item()
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * (prec * rec / (prec + rec))
    return {"f1": f1, "prec": prec, "rec": rec, "tp":tp, "fp": fp, "fn": fn, "tn":tn}

if __name__ == '__main__':
    for i in range(len(custom_single_tasks)):
        optim_config = optimize_configurations(custom_single_tasks[i],4,6, filter_fn=lambda name:"_0" in name[0], name="static")
        pkl.dump(optim_config, open(f"training_data/optim_{custom_single_tasks[i]}_multi_static.pkl","wb"))
    all_desc = {}
    for i in range(len(custom_single_tasks)):
        config = pkl.load(open(f"training_data/optim_{custom_single_tasks[i]}_multi_static.pkl","rb"))
        config[0].descriptor_set.reload_descriptors()
        all_desc[custom_single_tasks[i]]=(config[0])
    pkl.dump(all_desc, open(f"training_data/optim_multi_static.pkl","wb"))

    for i in range(len(custom_single_tasks)):
        optim_config = optimize_configurations(custom_single_tasks[i],4,6, filter_fn=lambda name:"_1" in name[0], name="hand", label_name="labels_hand.yml")
        pkl.dump(optim_config, open(f"training_data/optim_{custom_single_tasks[i]}_multi_hand.pkl","wb"))
    all_desc = {}
    for i in range(len(custom_single_tasks)):
        config = pkl.load(open(f"training_data/optim_{custom_single_tasks[i]}_multi_hand.pkl","rb"))
        config = sorted(config, reverse=True, key=lambda c:c.f1)
        config[0].descriptor_set.reload_descriptors()
        all_desc[custom_single_tasks[i]]=(config[0])
    pkl.dump(all_desc, open(f"training_data/optim_multi_hand.pkl","wb"))

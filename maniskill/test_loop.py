from typing import Callable
import pickle as pkl
import torch
from torch.utils.data import DataLoader

from maniskill.helpers.DescriptorConfiguration import DescriptorConfiguration
from maniskill.task_classifier import TaskClassifier


def test_config(img_src: str, configs: dict, filter_fn: Callable[[str], bool] = None):
    net = TaskClassifier(vit_stride=4, descriptors=configs)
    [class_mapping, dataset] = net.load_cache(img_src, filter_fn=filter_fn)
    data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    obj_finder = net.obj_finder
    obj_finder.class_mapping = class_mapping
    results = {
        task : {
            "fp": 0,
            "fn": 0,
            "tp": 0,
            "tn": 0
        } for task in class_mapping.keys()
    }

    total = len(dataset.img_tuples)
    for [tensors, labels] in data_loader:
        tensor = obj_finder(tensors).detach()
        object_reshaped = torch.reshape(tensor, (tensor.shape[0],len(class_mapping),3))
        sums = torch.sum(object_reshaped, dim=2)
        for task, task_idx in class_mapping.items():
            is_present = sums[:, task_idx] > 0
            should_be_present = torch.tensor(labels).cuda() == task_idx
            tp = torch.logical_and(is_present, should_be_present)
            fp = torch.logical_and(is_present, ~should_be_present)
            fn = torch.logical_and(~is_present, should_be_present)
            tn = torch.logical_and(~is_present, ~should_be_present)
            results[task]["tp"]=results[task]["tp"] + torch.count_nonzero(tp).item()
            results[task]["fp"]=results[task]["fp"] + torch.count_nonzero(fp).item()
            results[task]["fn"]=results[task]["fn"] + torch.count_nonzero(fn).item()
            results[task]["tn"]=results[task]["tn"] + torch.count_nonzero(tn).item()
    for task, res in results.items():
        res["precision"] = precision(res["tp"], res["fp"])
        res["recall"] = recall(res["tp"], res["fn"])
        res["f1"] = f1(res["precision"], res["recall"])
    return results

def precision(tp, fp):
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)

def recall(tp, fn):
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)

def f1(precision,recall):
    if precision + recall == 0:
        return 0
    return 2 * ((precision * recall) / (precision + recall))


if __name__ == "__main__":
    res = test_config("training_data/validation_set", configs=pkl.load(open("training_data/optim_hand.pkl","rb")), filter_fn=lambda name: "_1" in name[0])
    pkl.dump(res, open("training_data/results_val.pkl","wb"))
    res = test_config("training_data/test_set", configs=pkl.load(open("training_data/optim_hand.pkl","rb")), filter_fn=lambda name: "_1" in name[0])
    pkl.dump(res, open("training_data/results_test.pkl","wb"))
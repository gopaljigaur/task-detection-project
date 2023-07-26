from typing import Callable
import pickle as pkl
import torch
from torch.utils.data import DataLoader

from maniskill.task_classifier import TaskClassifier

multiple_object_tasks = {"CollectTools-v0":["StoreWrench-v0","PickupDrill-v0","Hammer-v0"], "DrillBlock-v0":["PickupDrill-v0","PickUpBlock-v0"], "FastenBlock-v0":["PickUpBlock-v0","FindClamp-v0"], "ReleaseBlock-v0":["PickUpBlock-v0","FindClamp-v0"]}



def test_config_alternative(img_src: str, configs: dict, filter_fn: Callable[[str], bool] = None):
    net = TaskClassifier(vit_stride=4, descriptors=configs)
    [dataset_class_mapping, dataset] = net.load_cache(img_src, filter_fn=filter_fn)
    data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    obj_finder = net.obj_finder
    class_mapping = obj_finder.class_mapping
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
            good_labels = []
            for multiple_task, single_tasks in multiple_object_tasks.items():
                if task in single_tasks:
                    good_labels.append(dataset_class_mapping[multiple_task])
            is_present = sums[:, task_idx] > 0
            should_be = torch.tensor([False] * tensor.shape[0]).cuda()
            for label in good_labels:
                current_label = labels == label
                should_be = torch.logical_or(should_be, current_label.cuda())
            tp = torch.logical_and(is_present, should_be)
            fp = torch.logical_and(is_present, ~should_be)
            fn = torch.logical_and(~is_present, should_be)
            tn = torch.logical_and(~is_present, ~should_be)
            results[task]["tp"]=results[task]["tp"] + torch.count_nonzero(tp).item()
            results[task]["fp"]=results[task]["fp"] + torch.count_nonzero(fp).item()
            results[task]["fn"]=results[task]["fn"] + torch.count_nonzero(fn).item()
            results[task]["tn"]=results[task]["tn"] + torch.count_nonzero(tn).item()
    for task, res in results.items():
        res["precision"] = precision(res["tp"], res["fp"])
        res["recall"] = recall(res["tp"], res["fn"])
        res["f1"] = f1(res["precision"], res["recall"])
    return results

def test_config(img_src: str, configs: dict, filter_fn: Callable[[str], bool] = None):
    net = TaskClassifier(vit_stride=4, descriptors=configs)
    [class_mapping, dataset] = net.load_cache(img_src, filter_fn=filter_fn)
    data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    obj_finder = net.obj_finder
    # obj_finder.class_mapping = class_mapping
    classes = {k: v for v, k in obj_finder.class_mapping.items()}
    results = {
        task : torch.tensor([]).cuda() for task in class_mapping.keys()
    }

    total = len(dataset.img_tuples)
    for [tensors, labels] in data_loader:
        tensor = obj_finder(tensors).detach()
        object_reshaped = torch.reshape(tensor, (tensor.shape[0],5,3))
        sums = torch.sum(object_reshaped, dim=2)
        idxs = torch.tensor([a for a in classes.keys()]).cuda()
        is_present = sums > 0
        present = torch.where(is_present, idxs, -1)
        for task, task_idx in class_mapping.items():
            filter_tensor = torch.tensor(labels).cuda() == task_idx
            results[task] = torch.cat((results[task],present[filter_tensor]))
    for task, res in results.items():
        all = []
        for row in res:
            curr = []
            for col in row:
                if col >= 0:
                    curr.append(classes[int(col.item())])
            all.append(curr)
        results[task] = all
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
    res = test_config_alternative("training_data/test_multiple", configs=pkl.load(open("training_data/optim_multi_hand.pkl","rb")), filter_fn=lambda name: "_1" in name[0])
    pkl.dump(res, open("training_data/result_multiple_hand.pkl","wb"))
    res = test_config_alternative("training_data/test_multiple", configs=pkl.load(open("training_data/optim_multi_static.pkl","rb")), filter_fn=lambda name: "_0" in name[0])
    pkl.dump(res, open("training_data/result_multiple_static.pkl","wb"))
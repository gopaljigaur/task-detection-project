from typing import Callable
import pickle as pkl
import torch
from torch.utils.data import DataLoader

from maniskill.helpers.DescriptorConfiguration import DescriptorConfiguration
from maniskill.task_classifier import TaskClassifier
import matplotlib.pyplot as plt

def test_config(img_src: str, configs: dict[str:DescriptorConfiguration], filter_fn: Callable[[str], bool] = None):
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
        object_reshaped = torch.reshape(tensor, (tensor.shape[0],tensor.shape[1],len(class_mapping),3))
        sums = torch.sum(object_reshaped, dim=3)
        for task, task_idx in class_mapping.entries():
            is_present = sums[:, task_idx] > 0
            should_be_present = torch.tensor(labels) == task_idx
            tp = torch.logical_and(is_present, should_be_present)
            fp = torch.logical_and(is_present, ~should_be_present)
            fn = torch.logical_and(~is_present, should_be_present)
            tn = torch.logical_and(~is_present, ~should_be_present)
            results[task]["tp"]=results[task]["tp"] + tp
            results[task]["fp"]=results[task]["fp"] + fp
            results[task]["fn"]=results[task]["fn"] + fn
            results[task]["tn"]=results[task]["tn"] + tn
    for task, res in results.items():
        task["precision"] = precision(task["tp"], task["fp"])
        task["recall"] = precision(task["tp"], task["fn"])
        task["f1"] = precision(task["precision"], task["recall"])
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

def plot_results(results):
    # Load data from the pickle file
    out = pkl.load(open(results, "rb"))

    # Create empty lists to store precision and recall for each task
    precision_values = {}
    recall_values = {}

    # Iterate through each task and extract precision and recall values
    for task, res in out.items():
        precision_values[task] = res["precision"]
        recall_values[task] = res["recall"]

    # Plot precision and recall for each task using a bar plot with error bars
    plt.figure(figsize=(10, 6))
    tasks = list(out.keys())
    positions = range(len(tasks))
    bar_width = 0.4

    plt.bar(positions, precision_values.values(), bar_width, label='Precision', alpha=0.7)
    plt.bar([pos + bar_width for pos in positions], recall_values.values(), bar_width, label='Recall', alpha=0.7)

    plt.xlabel('TASKS')
    plt.ylabel('SCORES')
    plt.title('Precision and Recall for some Task')
    plt.xticks([pos + bar_width / 2 for pos in positions], tasks, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("results_plot.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    #res = test_config("training_data/test_set", configs=pkl.load(open("training_data/optim.pkl","rb")), filter_fn=lambda name: "_1" in name[0])
    #pkl.dump(res, open("training_data/results.pkl","wb"))
    plot_results("training_data/saved_descriptors/results.pkl")

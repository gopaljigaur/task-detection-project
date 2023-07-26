import torch
import matplotlib.image as implt
import matplotlib.pyplot as plt

from maniskill.task_classifier import TaskClassifier


def load_img_tensor(img_path:str):
    return torch.load(img_path.split(".")[0]+".pt")

def load_img(img_path: str):
    return implt.imread(img_path.split(".")[0]+".png","rgbd")

def plot_img(img_path:str):
    im_tensor = load_img_tensor(img_path)
    img = load_img(img_path)
    net = TaskClassifier(vit_stride=4)
    net.obj_finder.skip_agg = True
    output = net.obj_finder(im_tensor)
    reshaped = torch.reshape(output, (5,3))
    colors = ["red","blue","yellow","purple","green"]
    fig, ax = plt.subplots()
    plt.imshow(img)
    dots = []
    class_mapping = {v: k for k, v in net.obj_finder.class_mapping.items()}
    for key, task in class_mapping.items():
        circle = plt.Circle(reshaped[key][:2], radius=2, color=colors[key], label=f"{task} {round(reshaped[key][2].item()*100,1)}")
        dots.append(circle)
        ax.add_patch(circle)
    plt.legend(handles=dots)
    plt.show()


if __name__ == "__main__":
    net = TaskClassifier(vit_stride=4)
    net.preprocess("training_data/custom")
    plot_img("training_data/custom/cl/cheated.png")
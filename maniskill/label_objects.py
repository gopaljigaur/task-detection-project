import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
import pickle
from PIL import Image


def label_interactive(path: str, overwrite: bool = False, labels_per_task: int = 5):
    """
    finding similarity between a descriptor in one image to the all descriptors in the other image.
    :param path:
    :param labels_per_task:
    :param overwrite:
    """

    # plot
    if os.path.exists(path):
        for task_dir in os.listdir(path):
            # we are now in the task directory of train_data
            fig, axes = plt.subplots()
            visible_patches = []
            print("Current task: %s" % task_dir)
            num_labels = 0
            for image_file in os.listdir(os.path.join(path, task_dir)):
                if num_labels >= labels_per_task:
                    break
                if image_file.endswith(".png"):
                    print("Current file: %s" % image_file)
                    current_info = {}
                    # check if the current image's data has already been stored, otherwise skip it
                    labels_file = os.path.join(path, task_dir, "labels.yml")
                    if os.path.exists(labels_file) and os.path.getsize(
                            labels_file) > 0:
                        stream = open(labels_file, "r")
                        current_info = yaml.load(stream, Loader=yaml.Loader)
                        stream.close()
                        if image_file in current_info:
                            print("Image's data present already. ", end="")
                            if not overwrite:
                                print("Skipping...")
                                num_labels += 1
                                continue
                            else:
                                print("Overwriting...")

                    pil_image = Image.open(
                        os.path.join(path, task_dir,
                                     image_file)).convert("RGB")

                    radius = 1
                    # plot image
                    axes.imshow(pil_image)
                    # get input point from user
                    fig.suptitle(
                        "Select points on the image (LClick). Next Image (RClick)",
                        fontsize=16,
                    )
                    # reset previous marks
                    for patch in visible_patches:
                        patch.remove()
                        visible_patches = []

                    if image_file in current_info:
                        if len(current_info[image_file]['points']) > 0:
                            for point in current_info[image_file]['points']:
                                x_pt, y_pt = map(lambda x: float(x.strip()), point.split(' '))
                                center = (x_pt - 0.5, y_pt - 0.5)
                                patch = plt.Circle(center,
                                                   radius,
                                                   color=(1, 0, 0, 0.75))
                                axes.add_patch(patch)
                                visible_patches.append(patch)

                    plt.draw()
                    pts = np.asarray(
                        plt.ginput(
                            n=-1,
                            timeout=-1,
                            show_clicks=True,
                            mouse_stop=plt.MouseButton.RIGHT,
                            mouse_pop=plt.MouseButton.MIDDLE,
                        ))

                    if len(pts) > 0:
                        # points = [str(pt)[1:-1].strip() for pt in pts]
                        # load depth info from pickle file
                        depth_file = os.path.join(path, task_dir,
                                                  image_file[:-4] + ".dpt")
                        points = []
                        for pt in pts:
                            print("Selected point: (%f, %f)" % (pt[1], pt[0]))
                            y_coor, x_coor = int(pt[1]), int(pt[0])
                            depths = None
                            with open(depth_file, 'rb') as dpt_file:
                                depths = pickle.load(dpt_file)
                            z_coor = depths[y_coor, x_coor][0]
                            print("Depth at selected point: %f" % z_coor)

                            # draw chosen point
                            center = (x_coor - 0.5, y_coor - 0.5)

                            patch = plt.Circle(center,
                                               radius,
                                               color=(1, 0, 0, 0.75))
                            axes.add_patch(patch)
                            visible_patches.append(patch)
                            points.append(str(pt)[1:-1].strip() + " " + str(z_coor))
                        plt.draw()

                        current_info[image_file] = {
                            "task_name": task_dir,
                            "points": points,
                        }
                        stream = open(labels_file, "w")
                        yaml.dump(current_info, stream)
                        stream.close()
                        num_labels += 1


    else:
        print("Data directory does not exist. Exiting...")
        return


""" taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Label the objects in the training set of images.")
    parser.add_argument("--path",
                        type=str,
                        default="training_data/training_set",
                        help="Path to the dataset directory"
                        )
    parser.add_argument("--labels_per_task",
                        type=int,
                        default=5,
                        help="Number of labels per task"
                        )
    parser.add_argument("--overwrite",
        default="False",
        type=str2bool,
        help="Overwrite the existing label data.",
        )

    args = parser.parse_args()

    label_interactive(args.path, args.overwrite, args.labels_per_task)

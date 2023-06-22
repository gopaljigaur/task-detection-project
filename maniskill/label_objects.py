import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import os
import yaml

def label_interactive(path: str, overwrite: bool = False):
    """
     finding similarity between a descriptor in one image to the all descriptors in the other image.
     :param path: path to image directory.
     :param stride: stride of the model.
    """

    # plot
    if os.path.exists(path):
        for task_dir in os.listdir(path):
            # we are now in the task directory of train_data
            fig, axes = plt.subplots()
            visible_patches = []
            print("Current task: %s" % task_dir)
            for image_file in os.listdir(os.path.join(path, task_dir)):
                if image_file.endswith(".png"):
                    print("Current file: %s" % image_file)
                    current_info = {}
                    # check if the current image's data has already been stored, otherwise skip it
                    labels_file = os.path.join(path, task_dir, 'labels.yml')
                    if os.path.exists(labels_file):
                        stream = open(labels_file, 'r')
                        current_info = yaml.load(stream, Loader=yaml.Loader)
                        stream.close()
                        if image_file in current_info:
                            print("Image's data present already. ", end='')
                            if not overwrite:
                                print("Skipping...")
                                continue
                            else:
                                print("Overwriting...")
                                
                    pil_image = Image.open(os.path.join(path, task_dir, image_file)).convert('RGB')
                    
                    radius = 1
                    # plot image and the chosen patch. if nothing marked chosen patch is cls patch.
                    axes.imshow(pil_image)
                    # get input point from user
                    fig.suptitle('Select a point on the image', fontsize=16)
                    plt.draw()
                    pts = np.asarray(plt.ginput(1, timeout=-1, mouse_stop=plt.MouseButton.RIGHT, mouse_pop=None))
                    # reset previous marks
                    for patch in visible_patches:
                            patch.remove()
                            visible_patches = []
                    if len(pts) == 1:
                        print("Selected point: (%f, %f)\n" % (pts[0, 1], pts[0, 0]))
                        current_info[image_file] = {
                            'task_name': task_dir,
                            'obj_loc': str((pts[0, 1], pts[0, 0]))
                        }
                        stream = open(labels_file, 'w')
                        yaml.dump(current_info, stream) 
                        stream.close()
                        y_coor, x_coor = int(pts[0, 1]), int(pts[0, 0])
                        
                        # draw chosen point
                        center = (x_coor - 0.5, y_coor - 0.5)
                        
                        patch = plt.Circle(center, radius, color=(1, 0, 0, 0.75))
                        axes.add_patch(patch)
                        visible_patches.append(patch)
                        plt.draw()
    else:
        print('Data directory does not exitst')
        return

""" taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Label the objects in the training set of images.')
    parser.add_argument('--path', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--overwrite', default='False', type=str2bool, help="Overwrite the existing label data.")
    
    args = parser.parse_args()
    
    label_interactive(args.path, args.overwrite)
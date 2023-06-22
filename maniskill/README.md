# Setup

First run the `create_training_data.py` to capture screenshots from the maniskill environment for multiple tasks and object positions. The generated images qill be stores in `training_data/training_set` directory with all images being sorted into directories by their task names. The image names end with `_0.png` for the normal camera view and `_1.png` for wrist view.

### Multiple Backgrounds

To make multiple backgrounds work, create the folder ```data/bg_backup```(temporary name)
Into this folder paste all environments you want to use (.gbl files). 

At runtime the environments will be copied over to the ```data/background``` folder and renamed to ```minimalistic_modern_bedroom.glb```

For other environments the camera starts too low, so the background is just black. To fix this we have 
to manually change a number in the maniskill files.

In Line 445 of ```sapien_env.py``` (maniskill2/envs) change the first list in the Pose object to ```[0,0,0]```.


Next, run the `label_objects.py` with `--path` set to `training_data/training_set` (location of generated images) and click on the object locations shown on the images in window. The selected coordinates of repective images and task name are stored in the `labels.yml` file in the same folder as the images of a certain task. `-overwrite` flag can also be used to overwrite previous image lables and coordinates.


## File infos

`create_training_data.py`: Captures images from maniskill

`load_training_data`: Loads the created images and extracts its descriptors

`label_objects.py`: Interactive utility to label the pixel containing the object in the created images, and label them with task. The `labels.yml` file for each task is stored in its respective folder in the created images directories.
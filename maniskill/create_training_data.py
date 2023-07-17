import random
import custom_tasks.multiple_object_tasks
import custom_tasks.single_object_tasks
import gym
import mani_skill2.envs
import matplotlib.pyplot as plt
import matplotlib.image as imageplt
import os
import time
import shutil
import pickle

from typing import List

pick_tasks = ["PickCube-v0", "StackCube-v0", "PickSingleYCB-v0", "PickSingleEGAD-v0", "PickClutterYCB-v0"]
assembly_tasks = ["PegInsertionSide-v0", "PlugCharger-v0", "AssemblingKits"]
misc_tasks = ["PandaAvoidObstacles-v0", "TurnFaucet-v0"]
manipulation_tasks = ["OpenCabinetDoor-v1", "OpenCabinetDrawer-v1", "PushChair-v1", "MoveBucket-v1"]
soft_body_tasks = ["Excavate-v0", "Fill-v0", "Pour-v0", "Hang-v0", "Pinch-v0", "Write-v0"]

cameras = ["base_camera", "hand_camera"]

background_folder = f"data/background/"
alternate_background_folder = f"data/bg_backup/"

custom_single_tasks=["PickupDrill-v0", "PickCube-v0", "FindClamp-v0", "StoreWrench-v0", "Hammer-v0"]

default_background_name = "minimalistic_modern_bedroom.glb"
# backgrounds = ["dinning_room.glb", "minimalistic_modern_bedroom.glb", "minimalistic_modern_office.glb", "vintage_living_room.glb", "small_modern_kitchen.glb", "charite_university_hospital_-_operating_room.glb"]
backgrounds = []
if os.path.exists("data/bg_backup"):
    backgrounds = os.listdir("data/bg_backup")


def create_data_set(tasks: List[str],
                    num_samples_per_task: int = 20,
                    save_path: str = "training_data/training_set",
                    objects: List[str] = None,
                    camera_cfg = None):
    label_dict = {}
    if not os.path.isdir("training_data"):
        os.mkdir("training_data")
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    for task in tasks:
        backgrounds_incl_none = backgrounds
        backgrounds_incl_none.append("")
        for background in backgrounds_incl_none:
            path = f"{save_path}/{task}"
            if not os.path.isdir(path):
                os.mkdir(path)
            if objects is None:
                create_data(task,
                            save_path=path,
                            background_name=background,
                            num_samples=num_samples_per_task,
                            camera_cfg=camera_cfg)
            else:
                ## only use single object of list
                for obj in objects:
                    path = f"{save_path}/{task}/{obj}"
                    create_data(task,
                                save_path=path,
                                background_name=background,
                                num_samples=num_samples_per_task,
                                objects=[obj],
                                camera_cfg=camera_cfg)


def create_data(task_name: str,
                save_path: str,
                background_name: str = "",
                num_samples: int = 100,
                objects: List[str] = None,
                camera_cfg =None):
    bg_name = "minimal_bedroom"
    if background_name == "":
        bg_name = None
    else:
        swap_background(background_name)
    # env = gym.make(task_name, obs_mode="rgbd", control_mode="pd_joint_delta_pos", bg_name=bg_name)
    if objects is None:
        env = gym.make(task_name, obs_mode="rgbd", bg_name=bg_name, camera_cfgs=camera_cfg)
    else:
        env = gym.make(task_name, obs_mode="rgbd", bg_name=bg_name, model_ids=objects, camera_cfgs=camera_cfg)
    # check if folders for saving are created
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    for _ in range(num_samples):
        current_file_name = round(time.time() * 1000)
        env.reset(seed=random.randint(0, 2147483647))
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        # obs contains information about picture
        images = obs["image"]

        # save rgb images
        imageplt.imsave(f"{save_path}/{current_file_name}_0.png", images[cameras[0]]["rgb"])
        imageplt.imsave(f"{save_path}/{current_file_name}_1.png", images[cameras[1]]["rgb"])

        # save depth of image pixels
        with open(f"{save_path}/{current_file_name}_0.dpt", 'wb') as dpt_file:
            pickle.dump(images[cameras[0]]["depth"], dpt_file)
        with open(f"{save_path}/{current_file_name}_1.dpt", 'wb') as dpt_file:
            pickle.dump(images[cameras[1]]["depth"], dpt_file)
    env.close()


def swap_background(background_name: str):
    if not os.path.exists(f"{background_folder}"):
        os.mkdir(background_folder)
    if os.path.isfile(f"{background_folder}{default_background_name}"):
        os.remove(f"{background_folder}{default_background_name}")
    shutil.copy(f"{alternate_background_folder}{background_name}", f"{background_folder}{default_background_name}")


if __name__ == "__main__":
    camera_cfg={"height":256, "width":256}
    create_data_set(custom_single_tasks,
                    num_samples_per_task=5,
                    camera_cfg=camera_cfg)

import math
import os.path
import random
from typing import List
import pickle as pkl
import numpy as np
from mani_skill2.utils.io_utils import dump_json
from transforms3d.euler import euler2quat


def create_episodes(model_ids: List[str], poses: List[List[float]], n_episodes:int = 50, shuffle=False, distance: List[float] = [0.15,1.]):
    episodes = []
    if shuffle:
        random.shuffle(model_ids)
    for _ in range(n_episodes):
        actors = []
        for i in range(len(model_ids)):
            in_distance = False
            tries_left = 30
            while (not in_distance) and tries_left>0:
                xy = np.random.uniform(poses[i][0], poses[i][1], [2])
                tries_left -= 1
                in_distance = True
                for actor in actors:
                    dist = math.dist(xy, actor["pose"][:2])
                    if dist < distance[0] or dist > distance[1]:
                        in_distance = False
            if tries_left == 0:
                print(f"not found for {model_ids} {distance}")
                break
            z = 0.05
            p = np.hstack([xy, z])
            ori = np.random.uniform(0, 2 * np.pi)
            q = euler2quat(0, 0, ori)
            pose = np.hstack([p, q])
            actors.append({
                "model_id": model_ids[i],
                "scale": 1.0,
                "rep_pts": [[1,1,1]],
                "pose": pose
            })
        episodes.append({'actors':actors})
    return episodes


def create_fasten_block():
    base_path = "../data/fasten_block"
    model_ids = ["070-a_colored_wood_blocks","052_extra_large_clamp"]
    poses = [[-0.4,0.4],[-0.4,0.4]]
    episodes = create_episodes(model_ids, poses, shuffle=True, distance=[0.4,1.])
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    dump_json(os.path.join(base_path, "episode.json.gz"), episodes)


def create_collect_tools():
    base_path = "../data/collect_tools"
    model_ids = ["042_adjustable_wrench", "048_hammer", "035_power_drill"]
    poses = [[-0.3,0.3],[-0.3,0.3],[-0.3,0.3]]
    episodes = create_episodes(model_ids, poses, shuffle=True)
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    dump_json(os.path.join(base_path, "episode.json.gz"), episodes)


def create_release_block():
    base_path = "../data/release_block"
    model_ids = ["070-a_colored_wood_blocks","052_extra_large_clamp"]
    poses = [[-0.3,0.3],[-0.3,0.3]]
    episodes = create_episodes(model_ids, poses, shuffle=True, distance=[0.15,0.25])
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    dump_json(os.path.join(base_path, "episode.json.gz"), episodes)


def create_drill_block():
    base_path = "../data/drill_block"
    model_ids = ["070-a_colored_wood_blocks","035_power_drill"]
    poses = [[-0.3,-0.1],[0.1,0.3]]
    episodes = create_episodes(model_ids, poses, shuffle=True)
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    dump_json(os.path.join(base_path, "episode.json.gz"), episodes)


if __name__=='__main__':
    create_drill_block()
    create_release_block()
    create_fasten_block()
    create_collect_tools()


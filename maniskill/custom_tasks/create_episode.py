import random
import numpy as np
from mani_skill2.utils.io_utils import dump_json
from transforms3d.euler import euler2quat


def create_bowl_banana():
    banana_id = "011_banana"
    bowl_id = "024_bowl"
    models = [banana_id, bowl_id]
    rep_pts = [[1, 1, 1]]
    poses = [[-0.1, 0.0], [0.1, 0.2]]
    episodes = []
    for i in range(100):
        actors = []
        for i in range(len(models)):
            xy = np.random.uniform(poses[i][0], poses[i][1], [2])
            z = 0.03
            p = np.hstack([xy, z])
            ori = np.random.uniform(0, 2 * np.pi)
            q = euler2quat(0, 0, ori)
            pose = np.hstack([p, q])
            actors.append({
                "model_id": models[i],
                "scale": 1.0,
                "rep_pts": rep_pts,
                "pose": pose
            })
        episodes.append({'actors':actors})
    return episodes


if __name__=='__main__':
    dict= create_bowl_banana()
    dump_json("episode.json.gz", dict)


from typing import List
import sapien.core as sapien
import numpy as np
from mani_skill2.envs.pick_and_place.pick_clutter import PickClutterEnv
from mani_skill2.envs.pick_and_place.pick_single import PickSingleEnv, build_actor_ycb, PickSingleYCBEnv
from mani_skill2.utils.registration import register_env
from sapien.core import Pose
from transforms3d.euler import euler2quat
from transforms3d.quaternions import axangle2quat, qmult

multiobject_tasks = ["DrillBlock-v0", "MarkBlock-v0", "ReleaseBlock-v0", "CollectTools-v0"]

class MultipleObjectTask(PickClutterEnv):
    # DEFAULT_EPISODE_JSON = "{ASSET_DIR}/pick_clutter/ycb_train_5k.json.gz"
    DEFAULT_ASSET_ROOT = PickSingleYCBEnv.DEFAULT_ASSET_ROOT
    DEFAULT_MODEL_JSON = PickSingleYCBEnv.DEFAULT_MODEL_JSON

    def __init__(
            self,
            episode_json: str = None,
            asset_root: str = None,
            model_json: str = None,
            **kwargs,
    ):
        super().__init__(**kwargs)

    def _load_model(self, model_id, model_scale=1.0):
        density = self.model_db[model_id].get("density", 1000)
        obj = build_actor_ycb(
            model_id,
            self._scene,
            scale=model_scale,
            density=density,
            root_dir=self.asset_root,
        )
        obj.name = model_id
        obj.set_damping(0.1, 0.1)
        return obj


@register_env("DrillBlock-v0", max_episode_steps=200)
class CleanMug(MultipleObjectTask):
    DEFAULT_EPISODE_JSON = "{ASSET_DIR}/drill_block/episode.json.gz"

    def __init__(
            self,
            episode_json: str = None,
            asset_root: str = None,
            model_json: str = None,
            **kwargs,
    ):
        super().__init__(episode_json, asset_root, model_json, **kwargs)


@register_env("ReleaseBlock-v0", max_episode_steps=200)
class CleanMug(MultipleObjectTask):
    DEFAULT_EPISODE_JSON = "{ASSET_DIR}/release_block/episode.json.gz"

    def __init__(
            self,
            episode_json: str = None,
            asset_root: str = None,
            model_json: str = None,
            **kwargs,
    ):
        super().__init__(episode_json, asset_root, model_json, **kwargs)


@register_env("FastenBlock-v0", max_episode_steps=200)
class CleanMug(MultipleObjectTask):
    DEFAULT_EPISODE_JSON = "{ASSET_DIR}/release_block/episode.json.gz"

    def __init__(
            self,
            episode_json: str = None,
            asset_root: str = None,
            model_json: str = None,
            **kwargs,
    ):
        super().__init__(episode_json, asset_root, model_json, **kwargs)


@register_env("CollectTools-v0", max_episode_steps=200)
class CleanMug(MultipleObjectTask):
    DEFAULT_EPISODE_JSON = "{ASSET_DIR}/release_block/episode.json.gz"

    def __init__(
            self,
            episode_json: str = None,
            asset_root: str = None,
            model_json: str = None,
            **kwargs,
    ):
        super().__init__(episode_json, asset_root, model_json, **kwargs)


@register_env("MarkBlock-v0", max_episode_steps=200)
class CleanMug(MultipleObjectTask):
    DEFAULT_EPISODE_JSON = "{ASSET_DIR}/mark_block/episode.json.gz"

    def __init__(
            self,
            episode_json: str = None,
            asset_root: str = None,
            model_json: str = None,
            **kwargs,
    ):
        super().__init__(episode_json, asset_root, model_json, **kwargs)


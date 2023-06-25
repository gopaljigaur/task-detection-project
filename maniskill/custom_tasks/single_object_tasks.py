from mani_skill2.envs.pick_and_place.pick_single import PickSingleEnv, build_actor_ycb
from mani_skill2.utils.registration import register_env


class SingleObjectTask(PickSingleEnv):
    DEFAULT_ASSET_ROOT = "{ASSET_DIR}/mani_skill2_ycb"
    DEFAULT_MODEL_JSON = "info_pick_v0.json"

    def __init__(
            self,
            model_id: str,
            asset_root: str = None,
            model_json: str = None,
            obj_init_rot_z=True,
            obj_init_rot=0,
            goal_thresh=0.025,
            **kwargs,
    ):
        super().__init__(asset_root,
                         model_json,
                         model_ids=[model_id],
                         obj_init_rot_z=obj_init_rot_z,
                         obj_init_rot=obj_init_rot,
                         goal_thresh=goal_thresh,
                         **kwargs
                         )

    def _check_assets(self):
        models_dir = self.asset_root / "models"
        for model_id in self.model_ids:
            model_dir = models_dir / model_id
            if not model_dir.exists():
                raise FileNotFoundError(
                    f"{model_dir} is not found."
                    "Please download (ManiSkill2) YCB models:"
                    "`python -m mani_skill2.utils.download_asset ycb`."
                )

            collision_file = model_dir / "collision.obj"
            if not collision_file.exists():
                raise FileNotFoundError(
                    "convex.obj has been renamed to collision.obj. "
                    "Please re-download YCB models."
                )

    def _load_model(self):
        density = self.model_db[self.model_id].get("density", 1000)
        self.obj = build_actor_ycb(
            self.model_id,
            self._scene,
            scale=self.model_scale,
            density=density,
            root_dir=self.asset_root,
        )
        self.obj.name = self.model_id

    def _get_init_z(self):
        bbox_min = self.model_db[self.model_id]["bbox"]["min"]
        return -bbox_min[2] * self.model_scale + 0.05

    def _initialize_agent(self):
        if self.model_bbox_size[2] > 0.15:
            return super()._initialize_agent_v1()
        else:
            return super()._initialize_agent()


@register_env("EatBanana-v0", max_episode_steps=200)
class EatBananaEnv(SingleObjectTask):
    def __init__(
            self,
            asset_root: str = None,
            model_json: str = None,
            obj_init_rot_z=True,
            obj_init_rot=0,
            goal_thresh=0.025,
            **kwargs,
    ):
        super().__init__("011_banana",
                         asset_root,
                         model_json,
                         obj_init_rot_z=obj_init_rot_z,
                         obj_init_rot=obj_init_rot,
                         goal_thresh=goal_thresh,
                         **kwargs
                         )


@register_env("PickUpFork-v0", max_episode_steps=200)
class PickUpFork(SingleObjectTask):
    def __init__(
            self,
            asset_root: str = None,
            model_json: str = None,
            obj_init_rot_z=True,
            obj_init_rot=0,
            goal_thresh=0.025,
            **kwargs,
    ):
        super().__init__("030_fork",
                         asset_root,
                         model_json,
                         obj_init_rot_z=obj_init_rot_z,
                         obj_init_rot=obj_init_rot,
                         goal_thresh=goal_thresh,
                         **kwargs
                         )


@register_env("ThrowHammer-v0", max_episode_steps=200)
class ThrowHammer(SingleObjectTask):
    def __init__(
            self,
            asset_root: str = None,
            model_json: str = None,
            obj_init_rot_z=True,
            obj_init_rot=0,
            goal_thresh=0.025,
            **kwargs,
    ):
        super().__init__("048_hammer",
                         asset_root,
                         model_json,
                         obj_init_rot_z=obj_init_rot_z,
                         obj_init_rot=obj_init_rot,
                         goal_thresh=goal_thresh,
                         **kwargs
                         )


@register_env("ThrowBall-v0", max_episode_steps=200)
class ThrowBall(SingleObjectTask):
    def __init__(
            self,
            asset_root: str = None,
            model_json: str = None,
            obj_init_rot_z=True,
            obj_init_rot=0,
            goal_thresh=0.025,
            **kwargs,
    ):
        super().__init__("055_baseball",
                         asset_root,
                         model_json,
                         obj_init_rot_z=obj_init_rot_z,
                         obj_init_rot=obj_init_rot,
                         goal_thresh=goal_thresh,
                         **kwargs
                         )


@register_env("SpinBowl-v0", max_episode_steps=200)
class SpinBowl(SingleObjectTask):
    def __init__(
            self,
            asset_root: str = None,
            model_json: str = None,
            obj_init_rot_z=True,
            obj_init_rot=0,
            goal_thresh=0.025,
            **kwargs,
    ):
        super().__init__("024_bowl",
                         asset_root,
                         model_json,
                         obj_init_rot_z=obj_init_rot_z,
                         obj_init_rot=obj_init_rot,
                         goal_thresh=goal_thresh,
                         **kwargs
                         )

from typing import Optional

from gym.envs import register as gym_register

_ENTRY_POINT_PREFIX = "airl_envs"
# _ENTRY_POINT_PREFIX = ""


def _register(env_name: str, entry_point: str, kwargs: Optional[dict] = None):
    entry_point = f"{_ENTRY_POINT_PREFIX}.{entry_point}"
    # entry_point = f"airl_envs.{entry_point}"
    gym_register(id=env_name, max_episode_steps=1e6, entry_point=entry_point, kwargs=kwargs)


def _point_maze_register():
    for dname, dval in {"": 0, "Wall": 1}.items():
        for hname, hval in {"": 0, "Harder": 1}.items():
            for rname, rval in {"": 0, "Reach": 1}.items():
                for vname, vval in {"": False, "Vel": True}.items():
                    _register(
                        f"PointMaze{dname}{hname}{rname}{vname}-v0",
                        entry_point="envs:PointMazeEnv",
                        kwargs={"with_wall": dval, "harder":hval, "done_when_reach":rval, "include_vel": vval},
                    )

_point_maze_register()

# _register(
#     "ObjPusher-v0",
#     entry_point="pusher_env:PusherEnv",
#     kwargs={"sparse_reward": False},
# )
# _register("TwoDMaze-v0", entry_point="envs:TwoDMaze")


# A modified ant which flips over less and learns faster via TRPO
_register(
    "CustomAnt-v0",
    entry_point="envs:CustomAntEnv",
    kwargs={"gear": 30, "disabled": False},
)
_register(
    "DisabledAnt-v0",
    entry_point="envs:CustomAntEnv",
    kwargs={"gear": 30, "disabled": True},
)


# register(
#     id='TwoDMaze-v0',
#     max_episode_steps=200,
#     entry_point='airl_envs.envs:TwoDMaze',
# )
"""Environments used for testing and benchmarking.

These are not a core part of the imitation package. They are relatively lightly tested,
and may be changed without warning.
"""

# Register environments with Gym
# from airl_envs.envs.twod_maze import TwoDMaze  # noqa: F401
from airl_envs.envs.point_maze_env import PointMazeEnv
from airl_envs.envs.ant_env import CustomAntEnv

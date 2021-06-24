import functools

from gail_airl_ppo.env import make_env


def setup_envs(s_env_id, t_env_id, from_darc_envs = False):
    if s_env_id is not None:
        s_env = make_env(s_env_id)
    else:
        s_env = None
    t_env = make_env(t_env_id)
    env_test = make_env(t_env_id)

    return s_env, t_env, env_test

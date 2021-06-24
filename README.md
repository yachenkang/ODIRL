# Off-Dynamics Inverse Reinforcement Learning from Hetero-Domain

## Requirements

You can setup Python liblaries using `bash setup.sh`. Note that you need a MuJoCo license. Please follow the instruction in [mujoco-py](https://github.com/openai/mujoco-py
) for help.

## How to run

The expert demonstration data used in the experiment is stored in the buffers folder. 
You can use the following command to directly execute the odirl program.

```bash
# Example: run on PointMaze envs, GPU
python train_odirl.py \
    --cuda --s_env_id airl_envs:PointMazeReach-v0 --t_env_id airl_envs:PointMazeWallReach-v0 \
    --buffer buffers/airl_envs:PointMazeWallReach-v0/size1000000_std0.01_prand0.0.pth \
    --rollout_length 2000 --samp_fre 30

# Example: run on HalfCheetah envs, GPU
python train_odirl.py \
    --cuda --s_env_id HalfCheetah-v3 --t_env_id HalfCheetahT-v3 \
    --buffer buffers/HalfCheetah-v3/size1000000_std0.01_prand0.0.pth \
    --rollout_length 50000 --epoch_disc 10 --epoch_cla 1 --samp_fre 100

# Example: run on Ant envs, GPU
python train_odirl.py \
    --cuda --s_env_id Ant-v3 --t_env_id AntT-v3 \
    --buffer buffers/Ant-v3/size1000000_std0.01_prand0.0.pth \
    --rollout_length 50000 --epoch_disc 10 --epoch_cla 1 --samp_fre 100

```

The expert demonstration data can also be retrieved by the following methods.

### Train expert
You can train experts using Soft Actor-Critic(SAC)

```bash
python train_expert.py --cuda --env_id airl_envs:PointMazeReach-v0 --num_steps 100000 --seed 0
```

### Collect demonstrations
You need to collect demonstraions using trained expert's weight. Note that `--std` specifies the standard deviation of the gaussian noise add to the action, and `--p_rand` specifies the probability the expert acts randomly. We set `std` to 0.01 not to collect too similar trajectories.

```bash
python collect_demo.py \
    --cuda --env_id airl_envs:PointMazeReach-v0 \
    --weight logs/airl_envs:PointMazeReach-v0/sac/*/model/*/actor.pth \
    --buffer_size 1000000 --std 0.01 --p_rand 0.0 --seed 0
```

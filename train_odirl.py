import os
import argparse
from datetime import datetime
import torch

from gail_airl_ppo.env import make_env
from gail_airl_ppo.buffer import SerializedBuffer
from gail_airl_ppo.algo import ODIRL
from gail_airl_ppo.odirl_trainer import Trainer
# from gail_airl_ppo.algo import ALGOS
# from gail_airl_ppo.trainer import Trainer
from setup_envs import setup_envs


def run(args):
    s_env, t_env, env_test = setup_envs(args.s_env_id, args.t_env_id, args.t_env_id.startswith("darc_envs:"))
    
    buffer_exp = SerializedBuffer(
        path=args.buffer,
        device=torch.device("cuda" if args.cuda else "cpu")
    )

    algo = ODIRL(
        buffer_exp=buffer_exp,
        state_shape=s_env.observation_space.shape,
        action_shape=s_env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
        rollout_length=args.rollout_length,
        alpha=args.alpha, 
        # units_disc_r=(256, 256), 
        # units_disc_v=(256, 256),
        epoch_disc=args.epoch_disc,
        epoch_cla=args.epoch_cla, 
        input_noise=args.input_noise
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.t_env_id, 'odirl', f'{args.alpha}-{args.rollout_length}-{args.epoch_disc}-{args.epoch_cla}-{args.samp_fre}-{args.input_noise}', f'seed{args.seed}-{time}')

    trainer = Trainer(
        env=s_env,
        t_env=t_env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
        samp_fre = args.samp_fre
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--buffer', type=str, required=True)
    # p.add_argument('--buffer', type=str, default='buffers/maps:mapb-v0/size1000000_std0.01_prand0.0.pth')
    p.add_argument('--rollout_length', type=int, default=50000)
    # p.add_argument('--rollout_length', type=int, default=10)
    p.add_argument('--alpha', type=float, default=100)
    p.add_argument('--input_noise', type=float, default=1e-2)
    p.add_argument('--epoch_disc', type=int, default=100)
    p.add_argument('--epoch_cla', type=int, default=100)
    p.add_argument('--num_steps', type=int, default=10**6)
    p.add_argument('--eval_interval', type=int, default=5000)
    p.add_argument('--samp_fre', type=int, default=1)
    p.add_argument('--s_env_id', type=str, default='maps:mapb-v0')
    p.add_argument('--t_env_id', type=str, default='maps:mapb-v0')
    # p.add_argument('--algo', type=str, default='gail')
    p.add_argument('--cuda', action='store_true')
    # p.add_argument('--cuda', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)

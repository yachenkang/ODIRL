import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from .ppo import PPO
from gail_airl_ppo.buffer import RolloutBuffer
from gail_airl_ppo.network import ODIRLDiscrim, DeltaReward

import os
from gail_airl_ppo.utils import disable_gradient

class ODIRL(PPO):

    def __init__(self, buffer_exp, state_shape, action_shape, device, seed,
                 gamma=0.995, rollout_length=10000, mix_buffer=1,
                 batch_size=64, lr_actor=3e-4, lr_critic=3e-4, lr_disc=3e-4,
                 units_actor=(64, 64), units_critic=(64, 64),
                 units_disc_r=(100, 100), units_disc_v=(100, 100),
                 epoch_ppo=50, epoch_disc=10, clip_eps=0.2, lambd=0.97,
                 coef_ent=0.0, max_grad_norm=10.0, alpha=0, epoch_cla=100, input_noise=0):
        super().__init__(
            state_shape, action_shape, device, seed, gamma, rollout_length,
            mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
            epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm
        )

        # Expert's buffer.
        self.buffer_exp = buffer_exp

        self.s_buffer = RolloutBuffer(
            buffer_size=int(1e6),
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            mix=mix_buffer
        )

        self.t_buffer = RolloutBuffer(
            buffer_size=int(1e6),
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            mix=mix_buffer
        )

        # Discriminator.
        self.disc = ODIRLDiscrim(
            state_shape=state_shape,
            gamma=gamma,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True)
        ).to(device)

        self.learning_steps_disc = 0
        self.learning_steps_cla = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc
        self.epoch_cla = epoch_cla

        self.deltareward = DeltaReward(state_shape, action_shape, device)

        self.alpha = alpha

        self.input_noise = input_noise

    def update(self, writer):
        self.learning_steps += 1

        if self.alpha != 0:
            # print('test')
            for _ in range(self.epoch_cla):
                self.learning_steps_cla += 1

                cla_sa_loss, cla_sas_loss, acc_sa, acc_sas = self.deltareward.train_classifier(self.s_buffer, self.t_buffer)

                if self.learning_steps_cla % self.epoch_cla == 0:
                    writer.add_scalar(
                            'cla/sa_loss', cla_sa_loss.item(), self.learning_steps)
                    writer.add_scalar(
                            'cla/sas_loss', cla_sas_loss.item(), self.learning_steps)
                    writer.add_scalar(
                            'cla/acc_sa', acc_sa, self.learning_steps)
                    writer.add_scalar(
                            'cla/acc_sas', acc_sas, self.learning_steps)

        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # Samples from current policy's trajectories.
            states, actions, _, dones, log_pis, next_states = \
                self.buffer.sample(self.batch_size)
            # Samples from expert's demonstrations.
            states_exp, actions_exp, _, dones_exp, next_states_exp = \
                self.buffer_exp.sample(self.batch_size)
            # Calculate log probabilities of expert actions.
            with torch.no_grad():
                log_pis_exp = self.actor.evaluate_log_pi(
                    states_exp, actions_exp)
            # Update discriminator.
            self.update_disc(
                states, actions, dones, log_pis, next_states, states_exp, actions_exp, 
                dones_exp, log_pis_exp, next_states_exp, self.input_noise, writer
            )

        # We don't use reward signals here,
        states, actions, _, dones, log_pis, next_states = self.buffer.get()

        # delta_r = self.deltareward.calculate_delta_r(states, actions, next_states).detach()

        # Calculate rewards.
        rewards = self.disc.calculate_reward(
            states, dones, log_pis, next_states)

        # Update PPO using estimated rewards.
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states, writer)

    def update_disc(self, states, actions, dones, log_pis, next_states,
                    states_exp, actions_exp, dones_exp, log_pis_exp,
                    next_states_exp, input_noise, writer):

        # delta_r = self.deltareward.calculate_delta_r(states, actions, next_states).detach()
        delta_r_exp = self.deltareward.calculate_delta_r(states_exp, actions_exp, next_states_exp, input_noise).detach()

        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.disc(states, dones, log_pis, next_states)
        logits_exp = self.disc(
            states_exp, dones_exp, log_pis_exp, next_states_exp, delta_r_exp, self.alpha)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar(
                'loss/disc', loss_disc.item(), self.learning_steps)

            # writer.add_scalar(
            #     'cla/delta_r', delta_r.mean().item(), self.learning_steps)
            writer.add_scalar(
                'cla/delta_r_exp', delta_r_exp.mean().item(), self.learning_steps)

            # Discriminator's accuracies.
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
            writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)


    def s_step(self, env, state, t):
        t += 1

        action, log_pi = self.explore(state)
        next_state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done

        self.s_buffer.append(state, action, reward, mask, log_pi, next_state)

        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def step(self, env, state, t, step):
        t += 1

        action, log_pi = self.explore(state)
        next_state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done

        self.buffer.append(state, action, reward, mask, log_pi, next_state)
        self.t_buffer.append(state, action, reward, mask, log_pi, next_state)

        if done:
            t = 0
            next_state = env.reset()

        return next_state, t
    
    def save_models(self, save_dir):
        super().save_models(save_dir)
        torch.save(
            self.disc.state_dict(),
            os.path.join(save_dir, 'disc.pth')
        )
    
    
class ODIRLAgent(ODIRL):

    def __init__(self, state_shape, device, path, gamma=0.995,
                 units_disc_r=(100, 100), units_disc_v=(100, 100)):
        self.disc = ODIRLDiscrim(
            state_shape=state_shape,
            gamma=gamma,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True)
        ).to(device)
        self.disc.load_state_dict(torch.load(path))

        disable_gradient(self.disc)
        self.device = device
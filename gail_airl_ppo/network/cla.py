import torch
from torch import nn
import torch.nn.functional as F

from .utils import build_mlp
from torch.optim import Adam

import numpy as np

class Classifier(nn.Module):
    def __init__(
        self,
        state_shape,
        action_shape,
        use_next_state: bool,
        device,
        **kwargs,
    ):
        """Builds reward MLP.

        Args:
          observation_space: The observation space.
          action_space: The action space.
          use_state: should the current state be included as an input to the MLP?
          use_action: should the current action be included as an input to the MLP?
          use_next_state: should the next state be included as an input to the MLP?
          use_done: should the "done" flag be included as an input to the MLP?
          kwargs: passed straight through to build_mlp.
        """
        super().__init__()
        combined_size = 0

        combined_size += state_shape[0]

        combined_size += action_shape[0]

        self.use_next_state = use_next_state
        if self.use_next_state:
            combined_size += state_shape[0]

        full_build_mlp_kwargs = {
            "hidden_units": [256, 256],
        }
        full_build_mlp_kwargs.update(kwargs)
        full_build_mlp_kwargs.update(
            {
                # we do not want these overridden
                "input_dim": combined_size,
                "output_dim": 2,
            }
        )

        self.mlp = build_mlp(**full_build_mlp_kwargs)

        self.device = device

    def forward(self, state, action, next_state=None, input_noise=0.0):
        inputs = []
        inputs.append(torch.flatten(state, 1))
        inputs.append(torch.flatten(action, 1))
        if self.use_next_state:
            assert next_state != None
            inputs.append(torch.flatten(next_state, 1))

        inputs_concat = torch.cat(inputs, dim=1)

        inputs_concat += input_noise*torch.randn(inputs_concat.size()).to(self.device)

        outputs = self.mlp(inputs_concat)
        assert outputs.shape[:1] == state.shape[:1]

        return outputs

class DeltaReward():
    def __init__(self, state_shape, action_shape, device,
                 classifier_lr: float = 3e-4,
                ):
        
        self.cla_sas = Classifier(state_shape, action_shape, use_next_state=True, device=device).to(device)
        self.cla_sa = Classifier(state_shape, action_shape, use_next_state=False, device=device).to(device)

        self.cla_sas_optim = Adam(self.cla_sas.parameters(), lr=classifier_lr)
        self.cla_sa_optim = Adam(self.cla_sa.parameters(), lr=classifier_lr)

        self.device = device

    def calculate_delta_r(self, states, actions, next_states, input_noise=0):

        sa = self.cla_sa(states, actions, input_noise)
        sas = self.cla_sas(states, actions, next_states, input_noise)

        log_prob_sa = F.log_softmax(sa, dim=1)
        # log_prob_sas = F.log_softmax(sas, dim=1)
        log_prob_sas = F.log_softmax(sas + sa, dim=1)
        delta_r = (log_prob_sas[:,1] - log_prob_sa[:,1] - log_prob_sas[:,0] + log_prob_sa[:,0]).detach()

        return delta_r

    def train_classifier(
        self, 
        s_replay_buffer, 
        t_replay_buffer, 
    ):
        acc_sa = []
        acc_sas = []
        cla_sa_loss = 0
        cla_sas_loss = 0

        self.cla_sa_optim.zero_grad()
        self.cla_sas_optim.zero_grad()
        for index, replay_buffer in enumerate([s_replay_buffer, t_replay_buffer]):
            batch_size = 64
            states, actions, _, _, _, next_states = replay_buffer.sample(batch_size)

            sort_index = torch.tensor([index]*batch_size).to(self.device)

            sa = self.cla_sa(states, actions)
            sas = self.cla_sas(states, actions, next_states)

            cla_sa_loss += F.cross_entropy(sa, sort_index)
            # cla_sas_loss += F.cross_entropy(sas, sort_index)
            cla_sas_loss += F.cross_entropy(sas + sa.detach(), sort_index)

            with torch.no_grad():
                acc_sa.append((torch.max(sa, dim=1)[1] == index).float().mean().item())
                acc_sas.append((torch.max(sas + sa, dim=1)[1] == index).float().mean().item())
            
            # logger.record("cla_loss/cla_sa_loss", cla_sa_loss.item())
            # logger.record("cla_loss/cla_sas_loss", cla_sas_loss.item())
            # logger.record("reward_map", th.Tensor([[1,2],[3,4]]))

        cla_sa_loss.backward()
        cla_sas_loss.backward()
        
        self.cla_sa_optim.step()
        self.cla_sas_optim.step()

        acc_sa = np.mean(acc_sa)
        acc_sas = np.mean(acc_sas)

        return cla_sa_loss, cla_sas_loss, acc_sa, acc_sas
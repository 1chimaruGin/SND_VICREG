import time
import torch
import argparse
import torch.nn as nn
import numpy as np
import lightning as L
from typing import Dict, Tuple
from lightning.fabric import Fabric
from torch.utils.data import BatchSampler, RandomSampler
from networks.PPO_Modules import TYPE
from networks.PPO import PPO
from networks.RNDModelAtari import VICRegModelAtari
from networks.PPO_AtariModules import PPOAtariNetworkSND
from networks.SNDMotivation import SNDMotivation
from utils.ReplayBuffer import GenericTrajectoryBuffer


class PPOAtariSNDAgent(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        config,
        action_type,
        opt_algorithm: torch.optim.Optimizer | None = None,
        opt_motivation: torch.optim.Optimizer | None = None,
    ):
        super(PPOAtariSNDAgent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.memory = GenericTrajectoryBuffer(
            config.trajectory_size, config.batch_size, config.n_env
        )
        self.action_type = action_type
        self.network = PPOAtariNetworkSND(
            state_dim, action_dim, config, head=action_type
        )
        self.motivation_memory = GenericTrajectoryBuffer(
            config.trajectory_size, config.batch_size, config.n_env
        )
        self.cnd_model = VICRegModelAtari(state_dim, action_dim, config)
        self.motivation = SNDMotivation(
            network=self.cnd_model,
            eta=config.motivation_eta,
        )
        self.algorithm = PPO(
            algorithm=self.network,
            trajectory_size=config.trajectory_size,
            actor_loss_weight=config.actor_loss_weight,
            critic_loss_weight=config.critic_loss_weight,
            p_beta=config.beta,
            p_gamma=config.gamma,
            ppo_epochs=config.ppo_epochs,
            ext_adv_scale=2,
            int_adv_scale=1,
            n_env=config.n_env,
            motivation=True,
        )
        self.opt_algorithm = opt_algorithm
        self.opt_motivation = opt_motivation

    def forward(self, state):
        return self.get_action(state)

    def get_action(self, state):
        value, action, probs = self.network(state)
        features = self.cnd_model.target_model(self.cnd_model.preprocess(state))
        return features.detach(), value.detach(), action, probs.detach()

    def reward(self, state0):
        reward = self.motivation.reward(state0)
        return reward

    def error(self, state0):
        return self.motivation.error(state0)

    def convert_action(self, action):
        if self.action_type == TYPE.discrete:
            a = torch.argmax(action, dim=1).cpu().numpy()
            return a
        if self.action_type == TYPE.continuous:
            return action.squeeze(0).cpu().numpy()
        if self.action_type == TYPE.multibinary:
            return torch.argmax(action, dim=1).cpu().numpy()

    def setup(self, data):
        self.memory.add(
            state=data["state"],
            value=data["value"],
            action=data["action"],
            prob=data["probs"],
            reward=data["reward"],
            mask=data["mask"],
        )
        self.motivation_memory.add(
            state=data["state"], next_state=data["next_state"], action=data["action"]
        )

    def prepare_algorithm(self) -> Tuple[dict, BatchSampler]:
        memory = self.memory
        indices = memory.indices()
        if indices is not None:
            print("Preparing algorithm")
            sample = memory.sample(indices, False)
            states = sample.state
            values = sample.value
            actions = sample.action
            probs = sample.prob
            rewards = sample.reward
            dones = sample.mask
            if self.motivation:
                ext_reward = rewards[:, :, 0].unsqueeze(-1)
                int_reward = rewards[:, :, 1].unsqueeze(-1)
                ext_ref_values, ext_adv_values = self.algorithm.calc_advantage(
                    values[:, :, 0].unsqueeze(-1),
                    ext_reward,
                    dones,
                    self.algorithm.gamma[0],
                    self.algorithm.n_env,
                )
                int_ref_values, int_adv_values = self.algorithm.calc_advantage(
                    values[:, :, 1].unsqueeze(-1),
                    int_reward,
                    dones,
                    self.algorithm.gamma[1],
                    self.algorithm.n_env,
                )
                ref_values = torch.cat([ext_ref_values, int_ref_values], dim=2)
                adv_values = (
                    ext_adv_values * self.algorithm.ext_adv_scale
                    + int_adv_values * self.algorithm.int_adv_scale
                )
            else:
                ref_values, adv_values = self.algorithm.calc_advantage(
                    values, rewards, dones, self.gamma[0], self.n_env
                )
                adv_values *= self.algorithm.ext_adv_scale
            permutation = torch.randperm(self.trajectory_size)
            states = states.reshape(-1, *states.shape[2:])[permutation]
            actions = actions.reshape(-1, *actions.shape[2:])[permutation]
            probs = probs.reshape(-1, *probs.shape[2:])[permutation]
            adv_values = adv_values.reshape(-1, *adv_values.shape[2:])[permutation]
            ref_values = ref_values.reshape(-1, *ref_values.shape[2:])[permutation]
            # sampler
            batch = {
                "states": states,
                "actions": actions,
                "probs": probs,
                "adv_values": adv_values,
                "ref_values": ref_values,
            }
            indexes = list(range(states.shape[0]))
            sampler = RandomSampler(indexes)
            batch_sampler = BatchSampler(
                sampler, self.config.batch_size, drop_last=False
            )
            return batch, batch_sampler
        return None, None

    def prepare_motivation(self):
        memory = self.motivation_memory
        indices = memory.indices()
        if indices is not None:
            sample, size = memory.sample_batches(indices)
            batch = {
                "states": sample.state,
                "next_states": sample.next_state,
            }
            return batch, size
        return None, None

    def algorithm_loss(self, states, actions, adv_values, ref_values, probs):
        return self.algorithm.loss_function(
            states, ref_values, adv_values, actions, probs
        )

    def motivation_loss(self, state, next_state):
        return self.motivation.loss_function(state, next_state)
    
    def process_state(state, preprocess=None):
        if preprocess is None:
            processed_state = torch.tensor(state, dtype=torch.float32)
        else:
            processed_state = preprocess(state)
        return processed_state

    def save(self, path):
        torch.save(self.network.state_dict(), path + ".pth")

    def load(self, path):
        self.network.load_state_dict(torch.load(path + ".pth", map_location="cpu"))


def build_agent(
    fabric: Fabric,
    state_dim: tuple,
    action_dim: tuple,
    config: argparse.Namespace,
    action_type: str = "discrete",
):
    agent = PPOAtariSNDAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        action_type=action_type,
    )
    agent.network = fabric.setup_module(agent.network)
    agent.cnd_model = fabric.setup_module(agent.cnd_model)
    agent.motivation = fabric.setup_module(agent.motivation)
    agent.algorithm = fabric.setup_module(agent.algorithm)
    return agent

import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import lightning as L
from lightning.fabric import Fabric
from torch.utils.data import BatchSampler, RandomSampler
from networks.PPO_Modules import TYPE
from networks.PPO import PPOLightning
from networks.RNDModelAtari import VICRegModelAtari
from networks.PPO_AtariModules import PPOAtariNetworkSND
from networks.SNDMotivation import SNDMotivationLightning
from utils.ReplayBuffer import GenericTrajectoryBuffer
from utils.ReplayBuffer import GenericTrajectoryBuffer


class PPOAtariSNDAgent(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: argparse.Namespace,
        action_type: TYPE,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.memory = GenericTrajectoryBuffer(
            config.trajectory_size, config.batch_size, config.n_env
        )
        self.action_type = action_type
        self.config = config
        self.network = PPOAtariNetworkSND(
            state_dim, action_dim, config, head=action_type
        )
        # self.network = PPOAtariNetworkSND(
        #     state_dim, action_dim, config, head=action_type
        # ).to(config.device)

        self.motivation_memory = GenericTrajectoryBuffer(
            config.trajectory_size, config.batch_size, config.n_env
        )
        self.cnd_model = VICRegModelAtari(state_dim, action_dim, config)
        self.motivation = SNDMotivationLightning(
            network=self.cnd_model,
            learning_rate=config.motivation_lr,
            eta=config.motivation_eta,
        )
        self.algorithm = PPOLightning(
            network=self.network,
            learning_rate=config.lr,
            actor_loss_weight=config.actor_loss_weight,
            critic_loss_weight=config.critic_loss_weight,
            batch_size=config.batch_size,
            trajectory_size=config.trajectory_size,
            p_beta=config.beta,
            p_gamma=config.gamma,
            ext_adv_scale=2,
            int_adv_scale=1,
            ppo_epochs=config.ppo_epochs,
            n_env=config.n_env,
            motivation=True,
        )

    def fabric_agent(self, fabric: Fabric, agent: L.LightningModule):
        optimizer = agent.configure_optimizers()
        agent, optimizer = fabric.setup(agent, optimizer)
        return agent, optimizer

    def get_action(self, state):
        value, action, probs = self.network(state)
        # features = self.network.cnd_model.target_model(self.network.cnd_model.preprocess(state))
        features = self.cnd_model.target_model(self.cnd_model.preprocess(state))
        return features.detach(), value.detach(), action, probs.detach()

    def convert_action(self, action: torch.Tensor) -> np.ndarray:
        if action.device.type == "cuda":
            action = action.cpu()
        if self.action_type == TYPE.discrete:
            a = torch.argmax(action, dim=1).numpy()
            return a
        if self.action_type == TYPE.continuous:
            return action.squeeze(0).numpy()
        if self.action_type == TYPE.multibinary:
            return torch.argmax(action, dim=1).numpy()

    def prepare_ppo_training(
        self, memory: GenericTrajectoryBuffer, indices: torch.Tensor
    ):
        sample = memory.sample(indices, False)
        states = sample.state
        values = sample.value
        actions = sample.action
        probs = sample.prob
        rewards = sample.reward
        dones = sample.mask
        if self.algorithm.motivation:
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
                values, rewards, dones, self.algorithm.gamma[0], self.algorithm.n_env
            )
            adv_values *= self.algorithm.ext_adv_scale

        permutation = torch.randperm(self.algorithm.trajectory_size)
        states = states.reshape(-1, *states.shape[2:])[permutation]
        actions = actions.reshape(-1, *actions.shape[2:])[permutation]
        probs = probs.reshape(-1, *probs.shape[2:])[permutation]
        adv_values = adv_values.reshape(-1, *adv_values.shape[2:])[permutation]
        ref_values = ref_values.reshape(-1, *ref_values.shape[2:])[permutation]
        batch = {
            "states": states,
            "actions": actions,
            "probs": probs,
            "adv_values": adv_values,
            "ref_values": ref_values,
        }
        indexes = list(range(states.shape[0]))
        # if self.fabric.world_size > 1:
        #     batch = self.fabric.all_gather(batch)
        #     sampler = DistributedSampler(
        #         indexes, num_replicas=self.fabric.world_size, rank=self.fabric.global_rank, shuffle=True
        #     )
        # else:
        sampler = RandomSampler(indexes)
        sampler = BatchSampler(
            sampler, batch_size=self.algorithm.batch_size, drop_last=False
        )
        return batch, sampler

    def prepare_snd_training(
        self, memory: GenericTrajectoryBuffer, indices: torch.Tensor
    ):
        sample, size = memory.sample_batches(indices)
        batch = {
            "states": sample.state,
            "next_states": sample.next_state,
        }
        return batch, size

    # def train_ppo(self, batch, sampler):
    #     for epoch in range(self.algorithm.ppo_epochs):
    #         # if self.fabric.world_size > 1:
    #         #     sampler.sampler.set_epoch(epoch)
    #         for idxs in sampler:
    #             loss = self.algorithm.training_step(
    #                 {k: v[idxs] for k, v in batch.items()}
    #             )
    #             self.algorithm_optimizer.zero_grad()
    #             self.fabric.backward(loss)
    #             self.fabric.clip_gradients(
    #                 self.algorithm, self.algorithm_optimizer, max_norm=0.5
    #             )
    #             self.algorithm_optimizer.step()

    # def train_snd(self, batch, size):
    #     for i in range(size):
    #         loss = self.motivation.training_step({k: v[i] for k, v in batch.items()})
    #         self.motivation_optimizer.zero_grad()
    #         self.fabric.backward(loss)
    #         self.fabric.clip_gradients(
    #             self.motivation, self.motivation_optimizer, max_norm=0.5
    #         )
    #         self.motivation_optimizer.step()

    def setup_data(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(
            state=state0,
            value=value,
            action=action0,
            prob=probs0,
            reward=reward,
            mask=mask,
        )
        self.motivation_memory.add(state=state0, next_state=state1, action=action0)
        indices = self.memory.indices()
        motivation_indices = self.motivation_memory.indices()
        batch, sampler = None, None
        motivation_batch, motivation_size = None, None

        if indices is not None:
            batch, sampler = self.prepare_ppo_training(self.memory, indices)
            self.memory.clear()

        if motivation_indices is not None:
            motivation_batch, motivation_size = self.prepare_snd_training(
                self.motivation_memory, motivation_indices
            )
            self.motivation_memory.clear()
        return batch, sampler, motivation_batch, motivation_size

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
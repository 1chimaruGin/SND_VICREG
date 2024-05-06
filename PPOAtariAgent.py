import time
import torch
import argparse
import lightning as L
from lightning.fabric import Fabric
from PPO_Modules import TYPE
from PPO import PPOLightning
from RNDModelAtari import VICRegModelAtari
from SNDMotivation import SNDMotivationLightning
from ReplayBuffer import GenericTrajectoryBuffer
from PPO_AtariModules import PPOAtariNetworkSND
from ReplayBuffer import GenericTrajectoryBuffer


class PPOAtariSNDAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: argparse.Namespace,
        action_type: TYPE,
        fabric: Fabric,
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

        self.motivation_memory = GenericTrajectoryBuffer(
            config.trajectory_size, config.batch_size, config.n_env
        )
        self.cnd_model = VICRegModelAtari(state_dim, action_dim, config)
        motivation = SNDMotivationLightning(
            network=self.cnd_model,
            learning_rate=config.motivation_lr,
            eta=config.motivation_eta,
        )
        motivation, motivation_optimizer = self.fabric_agent(fabric, motivation)
        self.motivation = motivation
        self.motivation_optimizer = motivation_optimizer

        algorithm = PPOLightning(
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
        algorithm, algorithm_optimizer = self.fabric_agent(fabric, algorithm)
        self.algorithm = algorithm
        self.algorithm_optimizer = algorithm_optimizer
        self.fabric = fabric

    def fabric_agent(self, fabric: Fabric, agent: L.LightningModule):
        optimizer = agent.configure_optimizers()
        agent, optimizer = fabric.setup(agent, optimizer)
        return agent, optimizer

    def get_action(self, state):
        value, action, probs = self.network(state)
        # features = self.network.cnd_model.target_model(self.network.cnd_model.preprocess(state))
        features = self.cnd_model.target_model(self.cnd_model.preprocess(state))
        return features.detach(), value.detach(), action, probs.detach()

    def convert_action(self, action):
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
        return batch

    def train_ppo(self, batch):
        for _ in range(self.algorithm.ppo_epochs):
            for batch_ofs in range(
                0, self.algorithm.trajectory_size, self.algorithm.batch_size
            ):
                batch["batch_ofs"] = batch_ofs
                loss = self.algorithm.training_step(batch)
                self.algorithm_optimizer.zero_grad()
                self.fabric.backward(loss)
                self.fabric.clip_gradients(
                    self.algorithm, self.algorithm_optimizer, max_norm=0.5
                )
                self.algorithm_optimizer.step()

    def train_snd(self, batch):
        sample, size = batch
        for i in range(size):
            motivation_loss = self.motivation.training_step(sample, i)
            self.motivation_optimizer.zero_grad()
            self.fabric.backward(motivation_loss)
            self.fabric.clip_gradients(
                self.motivation, self.motivation_optimizer, max_norm=0.5
            )
            self.motivation_optimizer.step()

    def prepare_snd_training(
        self, memory: GenericTrajectoryBuffer, indices: torch.Tensor
    ):
        return memory.sample_batches(indices)

    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(
            state=state0,
            value=value,
            action=action0,
            prob=probs0,
            reward=reward,
            mask=mask,
        )
        self.motivation_memory.add(
            state=state0, next_state=state1, action=action0
        )

        indices = self.memory.indices()
        motivation_indices = self.motivation_memory.indices()

        if indices is not None:
            start = time.time()
            batch = self.prepare_ppo_training(self.memory, indices)
            self.train_ppo(batch)
            end = time.time()
            print(
                "Trajectory {0:d} batch size {1:d} epochs {2:d} training time {3:.2f}s \n".format(
                    self.config.trajectory_size,
                    self.config.batch_size,
                    self.config.ppo_epochs,
                    end - start,
                )
            )
            # self.algorithm.train(self.memory, indices)
            self.memory.clear()

        if motivation_indices is not None:
            start = time.time()
            batch = self.prepare_snd_training(
                self.motivation_memory, motivation_indices
            )
            self.train_snd(batch)
            end = time.time()
            print(f"[INFO] CND training time: {end - start}s")
            # self.motivation.train(self.motivation_memory, motivation_indices)
            self.motivation_memory.clear()

    def save(self, path):
        torch.save(self.network.state_dict(), path + ".pth")

    def load(self, path):
        self.network.load_state_dict(torch.load(path + ".pth", map_location="cpu"))

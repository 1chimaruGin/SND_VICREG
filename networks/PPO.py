import torch
import lightning as L
from torchmetrics import MeanMetric
from enum import Enum
from typing import Dict


class MODE(Enum):
    basic = 0
    gate = 1
    generator = 2


class PPOLightning(L.LightningModule):
    def __init__(
        self,
        network: torch.nn.Module,
        batch_size: int,
        learning_rate: float,
        trajectory_size: int,
        actor_loss_weight: float,
        critic_loss_weight: float,
        p_beta: float,
        p_gamma: str,
        ppo_epochs: int = 10,
        p_epsilon: float = 0.1,
        p_lambda: float = 0.95,
        ext_adv_scale: int = 1,
        int_adv_scale: int = 1,
        n_env: int = 1,
        motivation: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.network = network
        self.beta = p_beta
        self.gamma = [float(g) for g in p_gamma.split(",")]
        self.epsilon = p_epsilon
        self._lambda = p_lambda
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.trajectory_size = trajectory_size
        self.actor_loss_weight = actor_loss_weight
        self.critic_loss_weight = critic_loss_weight
        self.trajectory = []
        self.ppo_epochs = ppo_epochs
        self.motivation = motivation
        self.n_env = n_env
        self.ext_adv_scale = ext_adv_scale
        self.int_adv_scale = int_adv_scale
        self.loss_metrics = MeanMetric()

    def calc_loss(self, states, ref_value, adv_value, old_actions, old_probs):
        values, _, probs = self.network(states)
        if self.motivation:
            ext_value = values[:, 0]
            int_value = values[:, 1]
            ext_ref_value = ref_value[:, 0]
            int_ref_value = ref_value[:, 1]

            loss_ext_value = torch.nn.functional.mse_loss(ext_value, ext_ref_value)
            loss_int_value = torch.nn.functional.mse_loss(int_value, int_ref_value)
            loss_value = loss_ext_value + loss_int_value
        else:
            loss_value = torch.nn.functional.mse_loss(values, ref_value)
        log_probs = self.network.actor.log_prob(probs, old_actions)
        old_logprobs = self.network.actor.log_prob(old_probs, old_actions)
        ratio = torch.exp(log_probs - old_logprobs)
        p1 = ratio * adv_value
        p2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv_value
        loss_policy = -torch.min(p1, p2)
        loss_policy = loss_policy.mean()
        entropy = self.network.actor.entropy(probs)
        loss = (
            loss_value * self.critic_loss_weight
            + loss_policy * self.actor_loss_weight
            + self.beta * entropy
        )
        return loss

    def calc_advantage(self, values, rewards, dones, gamma, n_env):
        buffer_size = rewards.shape[0]
        returns = torch.zeros((buffer_size, n_env, 1))
        advantages = torch.zeros((buffer_size, n_env, 1))
        last_gae = torch.zeros(n_env, 1)
        for n in reversed(range(buffer_size - 1)):
            delta = rewards[n] + dones[n] * gamma * values[n + 1] - values[n]
            last_gae = delta + dones[n] * gamma * self._lambda * last_gae
            returns[n] = last_gae + values[n]
            advantages[n] = last_gae
        return returns, advantages

    def training_step(self, batch: Dict[str, torch.Tensor]):
        states = batch["states"]
        actions = batch["actions"]
        probs = batch["probs"]
        adv_values = batch["adv_values"]
        ref_values = batch["ref_values"]
        batch_ofs = batch["batch_ofs"]
        batch_l = batch_ofs + self.batch_size
        states_v = states[batch_ofs:batch_l]
        actions_v = actions[batch_ofs:batch_l]
        probs_v = probs[batch_ofs:batch_l]
        batch_adv_v = adv_values[batch_ofs:batch_l]
        batch_ref_v = ref_values[batch_ofs:batch_l]
        loss = self.calc_loss(states_v, batch_ref_v, batch_adv_v, actions_v, probs_v)
        # self.loss_metrics.update(loss)
        return loss

    def training_epoch_end(self, global_step: int):
        self.logger.log_metrics(
            {"Loss/mean_loss": self.loss_metrics.compute()},
            global_step,
        )
        self.loss_metrics.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        return optimizer

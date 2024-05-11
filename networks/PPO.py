import torch
import torch.nn as nn
from enum import Enum


class MODE(Enum):
    basic = 0
    gate = 1
    generator = 2


class PPO(nn.Module):
    def __init__(
        self,
        algorithm: nn.Module,
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
    ):
        super().__init__()
        self.beta = p_beta
        self.n_env = n_env
        self.epsilon = p_epsilon
        self._lambda = p_lambda
        self.trajectory = []
        self.ppo_epochs = ppo_epochs
        self.motivation = motivation
        self.algorithm = algorithm
        self.ext_adv_scale = ext_adv_scale
        self.int_adv_scale = int_adv_scale
        self.trajectory_size = trajectory_size
        self.actor_loss_weight = actor_loss_weight
        self.critic_loss_weight = critic_loss_weight
        self.gamma = [float(g) for g in p_gamma.split(",")]

    def forward(self, states, ref_value, adv_value, old_actions, old_probs):
        values, actions, probs = self.algorithm(states)
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
        log_probs = self.algorithm.actor.log_prob(probs, old_actions)
        old_logprobs = self.algorithm.actor.log_prob(old_probs, old_actions)
        ratio = torch.exp(log_probs - old_logprobs)
        p1 = ratio * adv_value
        p2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv_value
        loss_policy = -torch.min(p1, p2)
        loss_policy = loss_policy.mean()
        entropy = self.algorithm.actor.entropy(probs)
        loss = (
            loss_value * self.critic_loss_weight
            + loss_policy * self.actor_loss_weight
            + self.beta * entropy
        )
        return [values, actions, probs], loss

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


# fabric run model --accelerator=cuda --strategy=ddp --devices=2 main.py

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


# from modules import init_orthogonal, init_uniform


import torch
import numpy as np

from enum import Enum


class TYPE(Enum):
    discrete = 0
    continuous = 1
    multibinary = 2


def one_hot_code(values, value_dim):
    code = torch.zeros((values.shape[0], value_dim), dtype=torch.float32, device=values.device)
    code = code.scatter(1, values, 1.0)
    return code


def stratify_sampling(x, n_samples, stratify):
    """Perform stratify sampling of a tensor.

    parameters
    ----------
    x: np.ndarray or torch.Tensor
        Array to sample from. Sampels from first dimension.

    n_samples: int
        Number of samples to sample

    stratify: tuple of int
        Size of each subgroup. Note that the sum of all the sizes
        need to be equal to `x.shape[']`.
    """
    n_total = x.shape[0]
    assert sum(stratify) == n_total

    n_strat_samples = [int(i * n_samples / n_total) for i in stratify]
    cum_n_samples = np.cumsum([0] + stratify)
    sampled_idcs = []
    for i, n_strat_sample in enumerate(n_strat_samples):
        sampled_idcs.append(np.random.choice(range(cum_n_samples[i], cum_n_samples[i + 1]),
                                             replace=False,
                                             size=n_strat_sample))

    # might not be correct number of samples due to rounding
    n_current_samples = sum(n_strat_samples)
    if n_current_samples < n_samples:
        delta_n_samples = n_samples - n_current_samples
        # might actually resample same as before, but it's only for a few
        sampled_idcs.append(np.random.choice(range(n_total), replace=False, size=delta_n_samples))

    samples = x[np.concatenate(sampled_idcs), ...]

    return samples


from enum import Enum

import torch
import torch.nn as nn


class ARCH(Enum):
    small_robotic = 0
    robotic = 1
    aeris = 2
    atari = 3


def init_custom(layer, weight_tensor):
    layer.weight = torch.nn.Parameter(torch.clone(weight_tensor))
    nn.init.zeros_(layer.bias)


def init_coupled_orthogonal(layers, gain=1.0):
    weight = torch.zeros(len(layers) * layers[0].weight.shape[0], *layers[0].weight.shape[1:])
    nn.init.orthogonal_(weight, gain)
    weight = weight.reshape(len(layers), *layers[0].weight.shape)

    for i, l in enumerate(layers):
        init_custom(l, weight[i])


def init_orthogonal(layer, gain=1.0):
    __init_general(nn.init.orthogonal_, layer, gain)


def init_xavier_uniform(layer, gain=1.0):
    __init_general(nn.init.xavier_uniform_, layer, gain)


def init_uniform(layer, gain=1.0):
    __init_general(nn.init.uniform_, layer, gain)


def __init_general(function, layer, gain):
    if type(layer.weight) is tuple:
        for w in layer.weight:
            function(w, gain)
    else:
        function(layer.weight, gain)

    if type(layer.bias) is tuple:
        for b in layer.bias:
            nn.init.zeros_(b)
    else:
        nn.init.zeros_(layer.bias)


def init_general_wb(function, weight, bias, gain):
    function(weight, gain)
    nn.init.zeros_(bias)


class DiscreteHead(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DiscreteHead, self).__init__()
        self.logits = nn.Linear(input_dim, action_dim, bias=True)

        torch.nn.init.xavier_uniform_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

    def forward(self, x):
        logits = self.logits(x)
        probs = torch.softmax(logits, dim=1)
        dist = Categorical(probs)

        action = dist.sample().unsqueeze(1)

        return action, probs

    @staticmethod
    def log_prob(probs, actions):
        actions = torch.argmax(actions, dim=1)
        dist = Categorical(probs)
        log_prob = dist.log_prob(actions).unsqueeze(1)

        return log_prob

    @staticmethod
    def entropy(probs):
        dist = Categorical(probs)
        entropy = -dist.entropy()
        return entropy.mean()

    @property
    def weight(self):
        return self.logits.weight

    @property
    def bias(self):
        return self.logits.bias


class ContinuousHead(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ContinuousHead, self).__init__()
        self.mu = nn.Sequential(
            nn.Linear(input_dim, action_dim),
            nn.Tanh()
        )
        self.var = nn.Sequential(
            nn.Linear(input_dim, action_dim),
            nn.Softplus()
        )

        init_uniform(self.mu[0], 0.03)
        init_uniform(self.var[0], 0.03)

        self.action_dim = action_dim

    def forward(self, x):
        mu = self.mu(x)
        var = self.var(x)

        dist = Normal(mu, var.sqrt().clamp(min=1e-3))
        action = dist.sample()

        return action, torch.cat([mu, var], dim=1)

    @staticmethod
    def log_prob(probs, actions):
        dim = probs.shape[1]
        mu, var = probs[:, :dim // 2], probs[:, dim // 2:]

        p1 = - ((actions - mu) ** 2) / (2.0 * var.clamp(min=1e-3))
        p2 = - torch.log(torch.sqrt(2.0 * np.pi * var))

        log_prob = p1 + p2

        return log_prob

    @staticmethod
    def entropy(probs):
        dim = probs.shape[1]
        var = probs[:, dim // 2:]
        entropy = -(torch.log(2.0 * np.pi * var) + 1.0) / 2.0

        return entropy.mean()


class Actor(nn.Module):
    def __init__(self, model, head, action_dim):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.head_type = head
        self.head = None
        if head == TYPE.discrete:
            self.head = DiscreteHead
        if head == TYPE.continuous:
            self.head = ContinuousHead
        if head == TYPE.multibinary:
            pass

        self.model = model

    def forward(self, x):
        return self.model(x)

    def log_prob(self, probs, actions):
        return self.head.log_prob(probs, actions)

    def entropy(self, probs):
        return self.head.entropy(probs)

    def encode_action(self, action):
        if self.head_type == TYPE.discrete:
            return one_hot_code(action, self.action_dim)
        if self.head_type == TYPE.continuous:
            return action
        if self.head_type == TYPE.multibinary:
            return None  # not implemented


class CriticHead(nn.Module):
    def __init__(self, input_dim, base, n_heads=1):
        super(CriticHead, self).__init__()
        self.base = base
        self.value = nn.Linear(input_dim, n_heads)

    def forward(self, x):
        x = self.base(x)
        value = self.value(x)
        return value

    @property
    def weight(self):
        return self.value.weight

    @property
    def bias(self):
        return self.value.bias


class Critic2Heads(nn.Module):
    def __init__(self, input_dim):
        super(Critic2Heads, self).__init__()
        self.ext = nn.Linear(input_dim, 1)
        self.int = nn.Linear(input_dim, 1)

        init_orthogonal(self.ext, 0.01)
        init_orthogonal(self.int, 0.01)

    def forward(self, x):
        ext_value = self.ext(x)
        int_value = self.int(x)
        return torch.cat([ext_value, int_value], dim=1).squeeze(-1)

    @property
    def weight(self):
        return self.ext.weight, self.int.weight

    @property
    def bias(self):
        return self.ext.bias, self.int.bias


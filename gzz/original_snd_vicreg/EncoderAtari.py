import random
import time
from math import sqrt

import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms


class ST_DIM_CNN(nn.Module):

    def __init__(self, input_shape, feature_dim):
        super().__init__()
        self.feature_size = feature_dim
        self.hidden_size = self.feature_size

        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]

        self.final_conv_size = 128 * (self.input_width // 8) * (self.input_height // 8)
        self.main = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.final_conv_size, feature_dim)
        )

        # gain = nn.init.calculate_gain('relu')
        gain = 0.5
        init_orthogonal(self.main[0], gain)
        init_orthogonal(self.main[2], gain)
        init_orthogonal(self.main[4], gain)
        init_orthogonal(self.main[6], gain)
        init_orthogonal(self.main[9], gain)

        self.local_layer_depth = self.main[4].out_channels

    def forward(self, inputs, fmaps=False):
        f5 = self.main[:6](inputs)
        out = self.main[6:](f5)

        if fmaps:
            return {
                'f5': f5.permute(0, 2, 3, 1),
                'out': out
            }
        return out



class VICRegEncoderAtari(nn.Module):
    def __init__(self, input_shape, feature_dim, config):
        super(VICRegEncoderAtari, self).__init__()

        self.config = config
        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.feature_dim = feature_dim

        self.encoder = ST_DIM_CNN(input_shape, feature_dim)

    def forward(self, state):
        return self.encoder(state)

    def loss_function(self, states, next_states):
        n = states.shape[0]
        d = self.feature_dim
        y_a = self.augment(states)
        y_b = self.augment(next_states)
        z_a = self.encoder(y_a)
        z_b = self.encoder(y_b)

        inv_loss = nn.functional.mse_loss(z_a, z_b)

        std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
        std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
        var_loss = torch.mean(nn.functional.relu(1 - std_z_a)) + torch.mean(nn.functional.relu(1 - std_z_b))

        z_a = (z_a - z_a.mean(dim=0))
        z_b = (z_b - z_b.mean(dim=0))

        cov_z_a = torch.matmul(z_a.t(), z_a) / (n - 1)
        cov_z_b = torch.matmul(z_b.t(), z_b) / (n - 1)

        cov_loss = cov_z_a.masked_select(~torch.eye(self.feature_dim, dtype=torch.bool, device=self.config.device)).pow_(2).sum() / self.feature_dim + \
                   cov_z_b.masked_select(~torch.eye(self.feature_dim, dtype=torch.bool, device=self.config.device)).pow_(2).sum() / self.feature_dim

        la = 1.
        mu = 1.
        nu = 1. / 25

        return la * inv_loss + mu * var_loss + nu * cov_loss

    def augment(self, x):
        # ref = transforms.ToPILImage()(x[0])
        # ref.show()
        # transforms_train = torchvision.transforms.Compose([
        #     transforms.RandomResizedCrop(96, scale=(0.66, 1.0))])
        # transforms_train = transforms.RandomErasing(p=1)
        # print(x.max())
        ax = x + torch.randn_like(x) * 0.1
        ax = nn.functional.upsample(nn.functional.avg_pool2d(ax, kernel_size=2), scale_factor=2, mode='bilinear')
        # print(ax.max())

        # aug = transforms.ToPILImage()(ax[0])
        # aug.show()

        return ax


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

def init_orthogonal(layer, gain=1.0):
    __init_general(nn.init.orthogonal_, layer, gain)


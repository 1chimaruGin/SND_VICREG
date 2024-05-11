import time
import torch
import torch.nn as nn
from .RNDModelAtari import VICRegModelAtari
from utils.RunningAverage import RunningStats


class SNDMotivation(nn.Module):
    def __init__(self, network: VICRegModelAtari, eta=1):
        super(SNDMotivation, self).__init__()
        self.eta = eta
        self.network = network
        self.reward_stats = RunningStats(1)

    def forward(self, state):
        return self.network(state)

    def loss_function(self, state0, state1):
        return self.network.loss_function(state0, state1)

    # def train(self, memory, indices):
    #     if indices:
    #         start = time.time()
    #         sample, size = memory.sample_batches(indices)

    #         for i in range(size):
    #             states = sample.state[i].to(self.device)
    #             next_states = sample.next_state[i].to(self.device)

    #             self.optimizer.zero_grad()
    #             loss = self.network.loss_function(states, next_states)
    #             loss.backward()
    #             self.optimizer.step()

    #         end = time.time()
    #         print("CND motivation training time {0:.2f}s".format(end - start))

    def error(self, state0):
        return self.network.error(state0)

    def reward_sample(self, memory, indices):
        sample = memory.sample(indices)
        states = sample.state
        return self.reward(states)

    def reward(self, state0):
        reward = self.error(state0)
        return reward * self.eta

    def update_state_average(self, state):
        self.network.update_state_average(state)

    def update_reward_average(self, reward):
        self.reward_stats.update(reward.to(self.device))

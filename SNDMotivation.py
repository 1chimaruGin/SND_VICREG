import time
import torch
import lightning as L
from RunningAverage import FabricRunningStats
from RNDModelAtari import VICRegModelAtari


class SNDMotivationLightning(L.LightningModule):
    def __init__(self, network: VICRegModelAtari, learning_rate: float, eta: int = 1):
        super().__init__()
        self.network = network
        self.eta = eta
        self.learning_rate = learning_rate
        self.reward_stats = FabricRunningStats(1)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, idx):
        states = batch.state[idx]
        next_states = batch.next_state[idx]
        loss = self.network.loss_function(states, next_states)
        return loss

    # def train(self, memory, indices):
    #     if indices:
    #         start = time.time()
    #         sample, size = memory.sample_batches(indices)
    #         for i in range(size):
    #             states = sample.state[i]
    #             next_states = sample.next_state[i]
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
        self.reward_stats.update(reward)

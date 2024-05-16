import torch
import lightning as L
from .RNDModelAtari import VICRegModelAtari
from utils.RunningAverage import FabricRunningStats


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

    # def training_step(self, batch, idx):
    #     states = batch.state[idx]
    #     next_states = batch.next_state[idx]
    #     print("State shape", states.shape)
    #     print("Next state shape", next_states.shape)
    #     loss = self.network.loss_function(states, next_states)
    #     return loss

    def training_step(self, batch):
        states, next_states = batch["states"], batch["next_states"]
        loss = self.network.loss_function(states, next_states)
        return loss

    def forward(
        self,
        state,
        error_flag=False,
    ):
        if error_flag:
            with torch.no_grad():
                prediction, target = self.network(state)
                error = self.network.k_distance(
                    self.network.config.cnd_error_k,
                    prediction,
                    target,
                    reduction="mean",
                )
                reward = error * self.eta
                return error, reward
        return self.network(state)

    def error(self, state):
        with torch.no_grad():
            prediction, target = self.network(state)
            error = self.network.k_distance(
                self.network.config.cnd_error_k, prediction, target, reduction="mean"
            )
        return error

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

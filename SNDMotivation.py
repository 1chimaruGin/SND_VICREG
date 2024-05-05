import time
import torch
import torch.nn as nn
import lightning as L
from ResultCollector import ResultCollector
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
    
    def forward(self, state):
        return self.network(state)
    
    def loss_function(self, state, next_state):
        prediction, target = self(state)
        loss_prediction = nn.functional.mse_loss(
            prediction, target.detach(), reduction="mean"
        )
        loss_target = self.network.target_model.loss_function(
            self.network.preprocess(state), self.network.preprocess(next_state)
        )

        analytic = ResultCollector()
        analytic.update(
            loss_prediction=loss_prediction.unsqueeze(-1).detach(),
            loss_target=loss_target.unsqueeze(-1).detach(),
        )
        return loss_prediction + loss_target

    def training_step(self, batch, idx):
        states = batch.state[idx]
        next_states = batch.next_state[idx]
        loss = self.network.loss_function(states, next_states)
        return loss

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

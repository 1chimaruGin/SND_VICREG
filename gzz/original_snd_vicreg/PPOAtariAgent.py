
from PPO import PPO
from ReplayBuffer import GenericTrajectoryBuffer
from PPO_AtariModules import PPOAtariNetworkSND

# from motivation.SNDMotivation import SNDMotivationFactory
import torch

from PPO_Modules import TYPE
from PPO import PPO
from ReplayBuffer import GenericTrajectoryBuffer

from SNDMotivation import SNDMotivation
from RNDModelAtari import VICRegModelAtari

class PPOAtariSNDAgent:
    def __init__(self, state_dim, action_dim, config,action_type,):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.memory = GenericTrajectoryBuffer(config.trajectory_size, config.batch_size, config.n_env)
        self.action_type = action_type

        self.network = PPOAtariNetworkSND(state_dim, action_dim, config, head=action_type).to(config.device)

        self.motivation_memory = GenericTrajectoryBuffer(config.trajectory_size, config.batch_size, config.n_env)
        self.cnd_model = VICRegModelAtari(state_dim, action_dim, config).to(config.device)
        self.motivation = SNDMotivation(network=self.cnd_model, lr=config.motivation_lr, eta=config.motivation_eta, device=config.device)

        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ext_adv_scale=2, int_adv_scale=1, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=True)        

        # if config.gpus and len(config.gpus) > 1:
        #     config.batch_size *= len(config.gpus)
        #     self.network = nn.DataParallel(self.network, config.gpus)

    def get_action(self, state):
        value, action, probs = self.network(state)
        # features = self.network.cnd_model.target_model(self.network.cnd_model.preprocess(state))
        features = self.cnd_model.target_model(self.cnd_model.preprocess(state))
        # features = 

        return features.detach(), value.detach(), action, probs.detach()

    def convert_action(self, action):
        if self.action_type == TYPE.discrete:
            a = torch.argmax(action, dim=1).numpy()
            return a
        if self.action_type == TYPE.continuous:
            return action.squeeze(0).numpy()
        if self.action_type == TYPE.multibinary:
            return torch.argmax(action, dim=1).numpy()

    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(state=state0.cpu(), value=value.cpu(), action=action0.cpu(), prob=probs0.cpu(), reward=reward.cpu(), mask=mask.cpu())
        self.motivation_memory.add(state=state0.cpu(), next_state=state1.cpu(), action=action0.cpu())

        indices = self.memory.indices()
        motivation_indices = self.motivation_memory.indices()

        if indices is not None:
            self.algorithm.train(self.memory, indices)
            self.memory.clear()

        if motivation_indices is not None:
            self.motivation.train(self.motivation_memory, motivation_indices)
            self.motivation_memory.clear()

    def save(self, path):
        torch.save(self.network.state_dict(), path + '.pth')

    def load(self, path):
        self.network.load_state_dict(torch.load(path + '.pth', map_location='cpu'))




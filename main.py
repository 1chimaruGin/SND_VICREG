import gym
import time
import torch
import argparse
import platform
import numpy as np
import torch.nn as nn
from typing import Dict, Union
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch.utils.tensorboard import SummaryWriter
from networks.PPO_Modules import TYPE
from agent.PPOAtariAgent import PPOAtariSNDAgent, build_agent

# from plots.paths import models_root
from utils.AtariWrapper import WrapperHardAtari
from utils.MultiEnvWrapper import MultiEnvParallel
from utils.ResultCollector import ResultCollector
from utils.RunningAverage import RunningAverageWindow, StepCounter
from utils.TimeEstimator import PPOTimeEstimator
from utils.utils import get_args


def train(
    fabric: Fabric,
    agent: Union[nn.Module, _FabricModule, PPOAtariSNDAgent],
    opt_algorithm: torch.optim.Optimizer,
    opt_motivation: torch.optim.Optimizer,
    data: list,
    config: argparse.Namespace,
    global_step: int = 0,
):
    if global_step % config.trajectory_size == 0:
        fabric.print(f"[INFO] Step {global_step} -> reward {reward.sum().item()}")
    batch, sampler, motivation_batch, motivation_size = agent.setup_data(*data)
    if batch is not None:
        s1 = time.time()
        for epoch in range(config.ppo_epochs):
            for idxs in sampler:
                loss = agent.algorithm.training_step(
                    {k: v[idxs] for k,v in batch.items()}
                )
                opt_algorithm.zero_grad(set_to_none=True)
                fabric.backward(loss)
                fabric.clip_gradients(agent.algorithm, opt_algorithm, max_norm=0.5)
                opt_algorithm.step()
        s2 = time.time()
        fabric.print(f"[INFO] Step {global_step} PPO time: {s2 - s1}, done")

    if motivation_batch is not None:
        m1 = time.time()
        for i in range(motivation_size):
            loss = agent.motivation.training_step({k: v[i] for k, v in motivation_batch.items()})
            opt_motivation.zero_grad(set_to_none=True)
            fabric.backward(loss)
            fabric.clip_gradients(agent.motivation, opt_motivation, max_norm=0.5)
            opt_motivation.step()
        m2 = time.time()
        fabric.print(f"[INFO] motivation time: {m2 - m1}, loss: {loss}")
        

if __name__ == "__main__":
    print(platform.system())
    print(torch.__version__)
    print(torch.__config__.show())
    print(torch.__config__.parallel_info())
    # torch.autograd.set_detect_anomaly(True)

    args = get_args()
    for i in range(torch.cuda.device_count()):
        print("{0:d}. {1:s}".format(i, torch.cuda.get_device_name(i)))
    fabric = Fabric()
    rank = fabric.global_rank
    world_size = fabric.world_size
    device = fabric.device
    fabric.seed_everything(42)
    print(f"rank: {rank}, world_size: {world_size}, device: {device}")
    torch.backends.cudnn.deterministic = False

    # Log hyperparameters

    env_name = args.env_name
    # PPO_HardAtariGame.run_snd_model(args, 0, env_name)
    trial = 0
    config = args

    print("Creating {0:d} environments".format(config.n_env))
    env = MultiEnvParallel(
        [WrapperHardAtari(gym.make(env_name)) for _ in range(config.n_env)],
        config.n_env,
        config.num_threads,
    )

    def process_state(state, fabric: Fabric):
        if _preprocess is None:
            processed_state = torch.tensor(state, dtype=torch.float32).to(fabric.device)
        else:
            processed_state = _preprocess(state).to(fabric.device)
        return processed_state

    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print("Start training")
    # experiment = ExperimentNEnvPPO(env_name, env, config)
    _env_name = env_name
    _env = env
    _config = config
    _preprocess = None

    # experiment.add_preprocess(encode_state)
    # agent = PPOAtariSNDAgent(input_shape, action_dim, config, TYPE.discrete, fabric)
    agent = build_agent(
        fabric, input_shape, action_dim, config, TYPE.discrete
    )
    opt_algorithm = fabric.setup_optimizers(
        torch.optim.Adam(agent.algorithm.parameters(), lr=config.lr)
    )
    opt_motivation = fabric.setup_optimizers(
        torch.optim.Adam(agent.motivation.parameters(), lr=config.motivation_lr)
    )

    config = _config
    n_env = config.n_env
    trial = trial + config.shift
    step_counter = StepCounter(int(config.steps * 1e6))
    writer = SummaryWriter(log_dir='runs/exp1')

    analytic = ResultCollector()
    analytic.init(
        n_env,
        re=(1,),
        score=(1,),
        ri=(1,),
        error=(1,),
        feature_space=(1,),
        state_space=(1,),
        ext_value=(1,),
        int_value=(1,),
    )

    reward_avg = RunningAverageWindow(100)
    time_estimator = PPOTimeEstimator(step_counter.limit)

    s = np.zeros((n_env,) + _env.observation_space.shape, dtype=np.float32)
    # agent.load('./models/{0:s}_{1}_{2:d}'.format(config.name, config.model, trial))
    for i in range(n_env):
        s[i] = _env.reset(i)[0]

    state0 = process_state(s, fabric)

    # pbar = tqdm(total=step_counter.limit, desc="Training")
    while step_counter.running():
        agent.motivation.update_state_average(state0)
        with torch.no_grad():
            features, value, action0, probs0 = agent.get_action(state0)
        next_state, reward, done, info = _env.step(agent.convert_action(action0))

        ext_reward = torch.tensor(reward, dtype=torch.float32)
        int_reward = agent.motivation(state0, error_flag=True)[1].cpu().clip(0.0, 1.0)

        if info is not None:
            print(f'[INFO] {info}')
            if "normalised_score" in info:
                analytic.add(normalised_score=(1,))
                score = torch.tensor(info["normalised_score"]).unsqueeze(-1)
                analytic.update(normalised_score=score)
            if "raw_score" in info:
                analytic.add(score=(1,))
                score = torch.tensor(info["raw_score"]).unsqueeze(-1)
                analytic.update(score=score)

        error = agent.motivation(state0, error_flag=True)[0]
        # cnd_state = agent.network.cnd_model.preprocess(state0)
        cnd_state = agent.cnd_model.preprocess(state0)
        analytic.update(
            re=ext_reward,
            ri=int_reward,
            ext_value=value[:, 0].unsqueeze(-1),
            int_value=value[:, 1].unsqueeze(-1),
            error=error,
            state_space=cnd_state.norm(p=2, dim=[1, 2, 3]).unsqueeze(-1),
            feature_space=features.norm(p=2, dim=1, keepdim=True),
        )
        # if sum(done)[0]>0:
        #     print('')
        env_indices = np.nonzero(np.squeeze(done, axis=1))[0]
        stats = analytic.reset(env_indices)
        step_counter.update(n_env)
        # fabric.print(f"[INFO] Step {step_counter.steps} -> error {error.mean().item()}") 

        for i, index in enumerate(env_indices):
            reward_avg.update(stats["re"].sum[i])
            max_room = np.max(info["episode_visited_rooms"])
            max_unique_room = np.max(info["max_unique_rooms"])
            print(
                "Run {0:d} step {1:d}/{2:d} training [ext. reward {3:f} int. reward (sum={4:f} max={5:f} mean={6:f} std={7:f}) steps {8:d}  mean reward {9:f} score {10:f} feature space (max={11:f} mean={12:f} std={13:f} rooms={14:d})]".format(
                    trial,
                    step_counter.steps,
                    step_counter.limit,
                    stats["re"].sum[i],
                    stats["ri"].sum[i],
                    stats["ri"].max[i],
                    stats["ri"].mean[i],
                    stats["ri"].std[i],
                    int(stats["re"].step[i]),
                    reward_avg.value().item(),
                    stats["score"].sum[i],
                    stats["feature_space"].max[i],
                    stats["feature_space"].mean[i],
                    stats["feature_space"].std[i],
                    max_room,
                )
            )
            writer.add_scalar("trial", trial, step_counter.steps)
            writer.add_scalar(
                "step_counter/limit", step_counter.limit, step_counter.steps
            )
            writer.add_scalar("stats/re_sum", stats["re"].sum[i], step_counter.steps)
            writer.add_scalar("stats/ri_sum", stats["ri"].sum[i], step_counter.steps)
            writer.add_scalar("stats/ri_max", stats["ri"].max[i], step_counter.steps)
            writer.add_scalar("stats/ri_mean", stats["ri"].mean[i], step_counter.steps)
            writer.add_scalar("stats/ri_std", stats["ri"].std[i], step_counter.steps)
            writer.add_scalar(
                "stats/re_step", int(stats["re"].step[i]), step_counter.steps
            )
            writer.add_scalar(
                "reward_avg_value", reward_avg.value().item(), step_counter.steps
            )
            writer.add_scalar(
                "stats/score_sum", stats["score"].sum[i], step_counter.steps
            )
            writer.add_scalar(
                "stats/feature_space_max",
                stats["feature_space"].max[i],
                step_counter.steps,
            )
            writer.add_scalar(
                "stats/feature_space_mean",
                stats["feature_space"].mean[i],
                step_counter.steps,
            )
            writer.add_scalar(
                "stats/feature_space_std",
                stats["feature_space"].std[i],
                step_counter.steps,
            )
            writer.add_scalar("max_room", max_room, step_counter.steps)
            writer.add_scalar("max_unique_rooms", max_unique_room, step_counter.steps)

            next_state[i] = _env.reset(index)

        state1 = process_state(next_state, fabric)
        reward = torch.cat([ext_reward, int_reward], dim=1)
        done = torch.tensor(1 - done, dtype=torch.float32)
        analytic.end_step()
        data = [state0, value, action0, probs0, state1, reward, done]
        train(
            fabric, agent, opt_algorithm, opt_motivation, data, config, step_counter.steps
        )
        state0 = state1
        time_estimator.update(n_env)
        # step_counter.update(world_size)
        # p = 0.0001  # Probability of saving the agent
        # # save model
        # if random() < p:
        #     print("model saved!")
        #     agent.save(
        #         "./models/{0:s}_{1}_{2:d}".format(config.name, config.model, trial)
        #     )

    analytic.reset(np.array(range(n_env)))
    save_data = analytic.finalize()
    np.save("ppo_{0}_{1}_{2:d}".format(config.name, config.model, trial), save_data)
    analytic.clear()
    env.close()


# [INFO] Step 180224 PPO time: 15.815066576004028, loss: -0.0030732937157154083
# [INFO] motivation time: 5.647643089294434, loss: 1.5377707481384277
# [INFO] Step 196608 PPO time: 28.103986024856567, loss: -0.0029103609267622232
# [INFO] motivation time: 5.675245761871338, loss: 1.526729941368103
# [INFO] Step 212992 PPO time: 17.873804569244385, loss: -0.002843035850673914
# [INFO] motivation time: 5.860707759857178, loss: 1.530949354171753
# [INFO] Step 229376 PPO time: 16.31726908683777, loss: -0.004856159910559654
# [INFO] motivation time: 5.791932106018066, loss: 1.5348527431488037
# [INFO] Step 245760 PPO time: 16.53315758705139, loss: -0.005438290536403656
# [INFO] motivation time: 5.72075891494751, loss: 1.536931037902832
# [INFO] Step 262144 PPO time: 16.52936029434204, loss: -0.00221174955368042
# [INFO] motivation time: 5.704389333724976, loss: 1.5285698175430298
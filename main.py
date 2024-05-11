import gym
import time
import torch
import platform
import argparse
import numpy as np
import torch.nn as nn
from typing import Union, Dict
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
    agent: Union[nn.Module, _FabricModule],
    opt_algorithm: torch.optim.Optimizer,
    opt_motivation: torch.optim.Optimizer,
    data: Dict[str, torch.Tensor],
    config: argparse.Namespace,
):
    agent.setup(data)
    batch, batch_sampler = agent.prepare_algorithm()
    if batch_sampler is not None:
        s1 = time.time()
        for epoch in range(config.epochs):
            for idxs in batch_sampler:
                batch = {k: v[idxs] for k, v in batch.items()}
                loss = agent.algorithm_loss(**batch)
                opt_algorithm.zero_grad(set_to_none=True)
                fabric.backward(loss)
                fabric.clip_gradients(agent.algorithm, opt_algorithm, max_norm=0.5)
                opt_algorithm.step()
        s2 = time.time()
        print(f"[INFO] epoch: {epoch}, time: {s2 - s1}, loss: {loss}")

    motivation_batch, size = agent.prepare_motivation()
    if batch is not None:
        m1 = time.time()
        for i in range(size):
            loss = agent.motivation_loss({k: v[i] for k, v in motivation_batch.items()})
            opt_motivation.zero_grad(set_to_none=True)
            fabric.backward(loss)
            fabric.clip_gradients(agent.motivation, opt_motivation, max_norm=0.5)
            opt_motivation.step()
        m2 = time.time()
        print(f"[INFO] motivation time: {m2 - m1}, loss: {loss}")


if __name__ == "__main__":
    print(platform.system())
    print(torch.__version__)
    print(torch.__config__.show())
    print(torch.__config__.parallel_info())
    # torch.autograd.set_detect_anomaly(True)

    args = get_args()
    for i in range(torch.cuda.device_count()):
        print("{0:d}. {1:s}".format(i, torch.cuda.get_device_name(i)))
    run_name = f"Running_{args.seed}_{int(time.time())}"
    fabric = Fabric()
    rank = fabric.global_rank
    world_size = fabric.world_size
    device = fabric.device
    print(f"rank: {rank}, world_size: {world_size}, device: {device}")
    fabric.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True

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

    def process_state(state, fabric: Fabric = fabric):
        if _preprocess is None:
            processed_state = torch.tensor(state, dtype=torch.float32).to(fabric.device)
        else:
            processed_state = _preprocess(state).to(fabric.device)
        return processed_state

    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print("Start training")
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
    writer = SummaryWriter(log_dir="runs/exp1")

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

    state0 = process_state(s)

    # pbar = tqdm(total=step_counter.limit, desc="Training")
    while step_counter.running():
        agent.motivation.update_state_average(state0)
        with torch.no_grad():
            features, value, action0, probs0 = agent.get_action(state0)
        next_state, reward, done, info = _env.step(agent.convert_action(action0))

        ext_reward = torch.tensor(reward, dtype=torch.float32)
        # int_reward = agent.motivation(state0, error_flag=True)[1].cpu().clip(0.0, 1.0)
        int_reward = agent.motivation.reward(state0).cpu().clip(0.0, 1.0)

        if info is not None:
            if "normalised_score" in info:
                analytic.add(normalised_score=(1,))
                score = torch.tensor(info["normalised_score"]).unsqueeze(-1)
                analytic.update(normalised_score=score)
            if "raw_score" in info:
                analytic.add(score=(1,))
                score = torch.tensor(info["raw_score"]).unsqueeze(-1)
                analytic.update(score=score)

        error = agent.motivation.error(state0, error_flag=True)[0]
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

        for i, index in enumerate(env_indices):
            # step_counter.update(int(stats['ext_reward'].step[i]))
            reward_avg.update(stats["re"].sum[i])
            max_room = np.max(info["episode_visited_rooms"])
            max_unique_room = np.max(info["max_unique_rooms"])

            # print(!pip install numpy==1.23.1
            #     'Run {0:d} step {1:d}/{2:d} training [ext. reward {3:f} int. reward (max={4:f} mean={5:f} std={6:f}) steps {7:d}  mean reward {8:f} score {9:f} feature space (max={10:f} mean={11:f} std={12:f})]'.format(
            #         trial, step_counter.steps, step_counter.limit, stats['re'].sum[i], stats['ri'].max[i], stats['ri'].mean[i], stats['ri'].std[i],
            #         int(stats['re'].step[i]), reward_avg.value().item(), stats['score'].sum[i], stats['feature_space'].max[i], stats['feature_space'].mean[i],
            #         stats['feature_space'].std[i]))
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

        state1 = process_state(next_state)
        reward = torch.cat([ext_reward, int_reward], dim=1)
        done = torch.tensor(1 - done, dtype=torch.float32)
        analytic.end_step()

        data = {
            "state": state0,
            "value": value,
            "action": action0,
            "probs": probs0,
            "reward": reward,
            "mask": done,
            "next_state": state1,
        }
        # agent.train(state0, value, action0, probs0, state1, reward, done)
        train(
            
        )

        state0 = state1
        time_estimator.update(n_env)
        # p = 0.0001  # Probability of saving the agent
        # # save model
        # if random() < p:
        #     print("model saved!")
        #     agent.save(
        #         "./models/{0:s}_{1}_{2:d}".format(config.name, config.model, trial)
        #     )

    # pbar.close()
    print("Saving data...")
    analytic.reset(np.array(range(n_env)))
    save_data = analytic.finalize()
    np.save("ppo_{0}_{1}_{2:d}".format(config.name, config.model, trial), save_data)
    analytic.clear()

    env.close()

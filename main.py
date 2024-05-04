import os
import gym
import torch
import platform
import argparse
import numpy as np
from random import random
from torch.utils.tensorboard import SummaryWriter
from PPO_Modules import TYPE
from PPOAtariAgent import PPOAtariSNDAgent

# from plots.paths import models_root
from AtariWrapper import WrapperHardAtari
from MultiEnvWrapper import MultiEnvParallel
from ResultCollector import ResultCollector
from RunningAverage import RunningAverageWindow, StepCounter
from TimeEstimator import PPOTimeEstimator

if __name__ == "__main__":
    print(platform.system())
    print(torch.__version__)
    print(torch.__config__.show())
    print(torch.__config__.parallel_info())
    # torch.autograd.set_detect_anomaly(True)

    for i in range(torch.cuda.device_count()):
        print("{0:d}. {1:s}".format(i, torch.cuda.get_device_name(i)))

    parser = argparse.ArgumentParser(description="Motivation models learning platform.")

    if not os.path.exists("./models"):
        os.mkdir("./models")

    parser.add_argument("--env", type=str, default="", help="environment name")
    parser.add_argument(
        "-a",
        "--algorithm",
        default="ppo",
        type=str,
        help="training algorithm",
        choices=["ppo", "ddpg", "a2c", "dqn"],
    )
    parser.add_argument("--config", type=int, help="id of config")
    parser.add_argument("--name", type=str, default="test", help="id of config")
    parser.add_argument("--device", type=str, help="device type", default="cuda")
    parser.add_argument("--gpus", help="device ids", default=None)
    parser.add_argument("--load", type=str, help="path to saved agent", default="")
    parser.add_argument("-s", "--shift", type=int, help="shift result id", default=0)
    parser.add_argument(
        "-p", "--parallel", action="store_true", help="run envs in parallel mode"
    )
    parser.add_argument(
        "-pb",
        "--parallel_backend",
        type=str,
        default="torch",
        choices=["ray", "torch"],
        help="parallel backend",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        help="number of parallel processes started in parallel mode (0=automatic number of cpus)",
        default=0,
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        help="number of parallel threads running in PPO (0=automatic number of cpus)",
        default=0,
    )
    parser.add_argument(
        "-t",
        "--thread",
        action="store_true",
        help="do not use: technical parameter for parallel run",
    )

    parser.add_argument(
        "--env_name", type=str, help="env name ", default="PrivateEyeNoFrameskip-v4"
    )
    parser.add_argument("--model", type=str, help="model type", default="snd")
    parser.add_argument("--type", type=str, help="type of training", default="vicreg")
    parser.add_argument("--n_env", type=int, help="number of environments", default=128)
    parser.add_argument("--trials", type=int, help="number of trials", default=8)
    parser.add_argument("--steps", type=int, help="number of steps", default=128)
    parser.add_argument("--gamma", type=str, help="gamma values", default="0.998,0.99")
    parser.add_argument("--beta", type=float, help="beta value", default=0.001)
    parser.add_argument("--batch_size", type=int, help="batch size", default=128)
    parser.add_argument(
        "--trajectory_size", type=int, help="trajectory size", default=16384
    )
    parser.add_argument("--ppo_epochs", type=int, help="PPO epochs", default=4)
    parser.add_argument("--lr", type=float, help="learning rate", default=0.0001)
    parser.add_argument(
        "--actor_loss_weight", type=float, help="actor loss weight", default=1
    )
    parser.add_argument(
        "--critic_loss_weight", type=float, help="critic loss weight", default=0.5
    )
    parser.add_argument(
        "--motivation_lr", type=float, help="motivation learning rate", default=0.0001
    )
    parser.add_argument(
        "--motivation_eta", type=float, help="motivation eta value", default=0.25
    )
    parser.add_argument("--cnd_error_k", type=int, help="cnd error k value", default=2)
    parser.add_argument("--cnd_loss_k", type=int, help="cnd loss k value", default=2)
    parser.add_argument(
        "--cnd_preprocess", type=int, help="cnd preprocess value", default=0
    )
    parser.add_argument(
        "--cnd_loss_pred", type=int, help="cnd loss pred value", default=1
    )
    parser.add_argument(
        "--cnd_loss_target", type=int, help="cnd loss target value", default=1
    )
    parser.add_argument(
        "--cnd_loss_target_reg",
        type=float,
        help="cnd loss target reg value",
        default=0.0001,
    )

    args = parser.parse_args()

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

    def process_state(state):
        if _preprocess is None:
            processed_state = torch.tensor(state, dtype=torch.float32).to(
                _config.device
            )
        else:
            processed_state = _preprocess(state).to(_config.device)

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
    agent = PPOAtariSNDAgent(input_shape, action_dim, config, TYPE.discrete)

    config = _config
    n_env = config.n_env
    trial = trial + config.shift
    step_counter = StepCounter(int(config.steps * 1e6))
    writer = SummaryWriter()

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
        s[i] = _env.reset(i)

    state0 = process_state(s)

    while step_counter.running():
        agent.motivation.update_state_average(state0)
        with torch.no_grad():
            features, value, action0, probs0 = agent.get_action(state0)
        next_state, reward, done, info = _env.step(agent.convert_action(action0.cpu()))

        ext_reward = torch.tensor(reward, dtype=torch.float32)
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

        error = agent.motivation.error(state0).cpu()
        # cnd_state = agent.network.cnd_model.preprocess(state0)
        cnd_state = agent.cnd_model.preprocess(state0)
        analytic.update(
            re=ext_reward,
            ri=int_reward,
            ext_value=value[:, 0].unsqueeze(-1).cpu(),
            int_value=value[:, 1].unsqueeze(-1).cpu(),
            error=error,
            state_space=cnd_state.norm(p=2, dim=[1, 2, 3]).unsqueeze(-1).cpu(),
            feature_space=features.norm(p=2, dim=1, keepdim=True).cpu(),
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

            # print(
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

        agent.train(state0, value, action0, probs0, state1, reward, done)

        state0 = state1
        p = 0.0001  # Probability of saving the agent
        time_estimator.update(n_env)
        if random() < p:
            print("model saved!")
            agent.save(
                "./models/{0:s}_{1}_{2:d}".format(config.name, config.model, trial)
            )

    print("Saving data...")
    analytic.reset(np.array(range(n_env)))
    save_data = analytic.finalize()
    np.save("ppo_{0}_{1}_{2:d}".format(config.name, config.model, trial), save_data)
    analytic.clear()

    env.close()


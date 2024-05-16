import os
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Motivation models learning platform.")
    if not os.path.exists("./models"):
        os.mkdir("./models")

    parser.add_argument("--env", type=str, default="montezuma", help="environment name")
    parser.add_argument(
        "-a",
        "--algorithm",
        default="ppo",
        type=str,
        help="training algorithm",
        choices=["ppo", "ddpg", "a2c", "dqn"],
    )
    parser.add_argument("--config", type=int, default=2, help="id of config")
    parser.add_argument("--name", type=str, default="test", help="id of config")
    parser.add_argument("--device", type=str, help="device type", default="cuda")
    parser.add_argument("--gpus", help="device ids", default=0)
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
        default=4,
    )
    parser.add_argument(
        "-t",
        "--thread",
        action="store_true",
        help="do not use: technical parameter for parallel run",
    )

    parser.add_argument(
        "--env_name", type=str, help="env name ", default="FrostbiteNoFrameskip-v4"
    )
    parser.add_argument("--model", type=str, help="model type", default="snd")
    parser.add_argument("--seed", type=int, help="random seed", default=42)
    parser.add_argument("--type", type=str, help="type of training", default="vicreg")
    parser.add_argument("--n_env", type=int, help="number of environments", default=128)
    parser.add_argument("--trials", type=int, help="number of trials", default=12)
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
    return parser.parse_args()

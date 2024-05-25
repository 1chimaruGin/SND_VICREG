# Self-supervised Network Distillation
Self-supervised Network Distillation (SND) is a class of intrinsic motivation algorithms based on the distillation error as a novelty indicator, where the target model is trained using self-supervised learning.

## Methods

The decision-making problem in the environment using reinforcement learning (RL) is formalized as a Markov decision process which consists of a state space $\mathcal{S}$, action space $\mathcal{A}$, transition function $\mathcal{T}$ describing probabilities of transition from state $s$ to $s'$ given action $a$, reward function $\mathcal{R}$ assigning reward to transition and a discount factor $\gamma$. The main goal of the agent is to maximize the discounted return

$$R_t = \sum_{k=0}^\infty \gamma^k r_{t+k}$$

in each state, where $r_t$ is immediate external reward at time $t$.

### Intrinsic motivation and exploration

During the learning process, the agent must explore the environment to encounter an external reward and learn to maximize it. This can be ensured by adding noise to the actions, if the policy is deterministic, or it is already its property if the policy is stochastic. In both cases, we say that these are uninformed environmental exploration strategies. The problem arises if the external reward is very sparse and the agent cannot use these strategies to find the sources of the reward. In such a case, it is advantageous to use informed strategies, which include the introduction of intrinsic motivation.

In the context of RL, intrinsic motivation can be realized in various ways, but most often it is a new reward signal $r^{\rm intr}_t$ scaled by parameter $\eta$, which is generated by the motivational part of the model (we refer to it as the motivational module) and is added to the external reward $r^{\rm ext}_t$

$$r_t = r_t^{\rm ext} + \eta \, r_t^{\rm intr}$$

The goal of introducing such a new reward signal is to provide the agent with a source of information that is absent from the environment when the reward is sparse, and thus facilitate the exploration of the environment and the search for an external reward.

<figure>
<p align="center">
<img src="https://github.com/Iskandor/SND/blob/main/assets/agent_im.png?raw=true" width="300">
<p align = "center">
<b>Fig.1 - Simplified architecture of an agent with intrinsic motivation.</b>
</p>
</p>
</figure>

### Intrinsic motivation based on distillation error

In this class of methods the motivation module has two components: the target model $\Phi^{\rm T}$ that generates features (typically as a kind of feature extractor), and the learning network $\Phi^{\rm L}$ that tries to replicate them. This process is called knowledge distillation. Intrinsic motivation, expressed as an intrinsic reward, is computed as the distillation error
$$r_{t}^{\rm intr} = \| (\Phi^{\rm L}(s_t) - \Phi^{\rm T}(s_t)) \| ^{2}.$$
It is assumed that the learning network will be able to more easily replicate feature vectors for states it has seen multiple times, while new states will induce a large distillation error.

<figure>
<p align="center">
<img src="https://github.com/Iskandor/SND/blob/main/assets/model_snd.png?raw=true" width="420">
<p align = "center">
<b>Fig.2 - The basic principle of generating an exploration signal in the regularized target model and training of the SND target model using two consecutive states and the self-supervised learning algorithm.</b>
</p>
</p>
</figure>

#### SND-V
method uses the contrastive loss [[1]](#1) for training target model $\Phi^{\rm T}$. Here is [SND-V original implementation](https://github.com/michalnand/reinforcement_learning).

#### SND-STD
method uses the Spatio-Temporal DeepInfoMax (ST-DIM) [[2]](#2) algorithm for training target model $\Phi^{\rm T}$.

#### SND-VIC
method uses Variance-Invariance-Covariance Regularization (VICReg) [[3]](#3) algorithm for training target model $\Phi^{\rm T}$.

## Results
We ran 9 simulations for each model and environment, taking 128M steps for Atari and 64M steps for Procgen games.
<figure>
<p align="center">
<img src="https://github.com/Iskandor/SND/blob/main/assets/results_chart.png?raw=true" width="600">
<p align = "center">
<b>Fig.3 - The cumulative external reward per episode (with the standard deviation)
received by the agent from the tested environment. We omitted the graph for the Pitfall environment, where no algorithm was successful and all achieved zero reward. The horizontal axis shows the number of steps in millions, the vertical axis refers the external reward.</b>
</p>
</p>
</figure>

<figure>
<p align="center">
<img src="https://github.com/Iskandor/SND/blob/main/assets/results_table.png?raw=true" width="500">
<p align = "center">
<b>Fig.4 - Average cumulative external reward per episode for tested models. The best model for each environment is shown in bold face.</b>
</p>
</p>
</figure>

<figure>
<p align="center">
<img src="https://github.com/Iskandor/SND/blob/main/assets/results_score.png?raw=true" width="500">
<p align = "center">
<b>Fig.5 - Average maximal score reached by tested models on Atari environments. The best model for each environment is shown in bold face.</b>
</p>
</p>
</figure>



## Installation
Prerequisites are c++ compiler and swig. For installation, it is necessary to clone this repository:
```
https://github.com/Iskandor/SND.git
```
and then install all dependencies using the command:
```
cd /path/to/SND
pip install -r requirements.txt
```
## Replication
```
python main.py -a ppo --env montezuma --config 2 --device cuda --gpus 0 --num_threads 4
```

For replication, use the given command line and as an argument of the --env environment, insert the value from the first column of the table and select --config from the remaining columns in the given table according to the desired model:


| Environment | Baseline | RND | SND-V | SND-STD | SND-VIC |
|-------------|----------|-----|-------|---------|---------|
| montezuma   | 1        | 2   | 49    | 42      | 44      |
| gravitar    | 1        | 2   | 14    | 11      | 13      |
| venture     | 1        | 2   | 10    | 4       | 8       |
| private_eye | 1        | 2   | 8     | 4       | 7       |
| pitfall     | 1        | 2   | 8     | 4       | 7       |
| solaris     | 1        | 2   | 8     | 4       | 7       |
| caveflyer   | 1        | 2   | 8     | 4       | 7       |
| coinrun     | 1        | 2   | 8     | 4       | 7       |
| jumper      | 1        | 2   | 8     | 4       | 7       |
| climber     | 1        | 2   | 8     | 4       | 7       |

It is also possible to train multiple models simultaneously on a single GPU (if VRAM and RAM capacity allows). On 2 Nvidia 3090 and RAM 128GB, we were able to train 2x4 agents at the same time. You only need to add the argument -p and --num_prcoesses, where you insert the number of processes in which the agents will be trained separately.

```
python main.py -a ppo --env montezuma --config 2 --device cuda --gpus 0 --num_threads 4 -p --num_processes 4
```

Feel free to try and change the hyperparameters of the models, which are stored in [ppo.config.json](https://github.com/Iskandor/SND/blob/main/config/ppo.config.json). You may be able to achieve better results, and we will certainly be glad if you let us know about them.

## Contact
Matej Pecháč [matej.pechac@gmail.com](mailto:matej.pechac@gmail.com)

Michal Chovanec [michalnand@gmail.com](mailto:michalnand@gmail.com)

## Citation
```
@article{pechac2023exploration,
  title={Exploration by self-supervised exploitation},
  author={Pech{\'a}{\v{c}}, Matej and Chovanec, Michal and Farka{\v{s}}, Igor},
  journal={arXiv preprint arXiv:2302.11563},
  year={2023}
}
```

## References
<a id="1">[1]</a> Chopra, S., Hadsell, R. a LeCun, Y. (2005). Learning a similarity metric discriminatively, with application to face verification. V International Conference on Pattern Recognition.

<a id="2">[2]</a> Anand, A., Racah, E., Ozair, S., Bengio, Y., Côte, M. a Hjelm, R. D. (2019). Unsupervised state representa
tion learning in Atari. CoRR, abs/1906.08226.

<a id="3">[3]</a> Bardes, A., Ponce, J., and LeCun, Y. (2022). VICReg: Variance-invariance-covariance regularization for self-supervised learning. In International Conference on Learning Representations.
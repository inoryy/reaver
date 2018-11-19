# Reaver: StarCraft II Deep Reinforcement Learning Agent

[![MoveToBeacon](https://user-images.githubusercontent.com/195271/37241507-0d7418c2-2463-11e8-936c-18d08a81d2eb.gif)](https://youtu.be/gEyBzcPU5-w)
[![CollectMineralShards](https://user-images.githubusercontent.com/195271/37241785-b8bd0b04-2467-11e8-9ff3-e4335a7c20ee.gif)](https://youtu.be/gEyBzcPU5-w)
[![DefeatRoaches](https://user-images.githubusercontent.com/195271/37241527-32a43ffa-2463-11e8-8e69-c39a8532c4ce.gif)](https://youtu.be/gEyBzcPU5-w)
[![DefeatZerglingsAndBanelings](https://user-images.githubusercontent.com/195271/37241531-39f186e6-2463-11e8-8aac-79471a545cce.gif)](https://youtu.be/gEyBzcPU5-w)
[![FindAndDefeatZerglings](https://user-images.githubusercontent.com/195271/37241532-3f81fbd6-2463-11e8-8892-907b6acebd04.gif)](https://youtu.be/gEyBzcPU5-w)
[![CollectMineralsAndGas](https://user-images.githubusercontent.com/195271/37241521-29594b48-2463-11e8-8b43-04ad0af6ff3e.gif)](https://youtu.be/gEyBzcPU5-w)
[![BuildMarines](https://user-images.githubusercontent.com/195271/37241515-1a2a5c8e-2463-11e8-8ac4-588d7826e374.gif)](https://youtu.be/gEyBzcPU5-w)


## Introduction

Reaver is a deep reinforcement learning agent designed for learning to solve various StarCraft II based tasks.
Main focus of Reaver is following in DeepMind's footsteps in pushing state-of-the-art of the field through the lens
of playing the game as closely to human as possible. This includes observing visual features similar (though not identical)
to what a human player would perceive and choosing actions from similar pool of options a human player would have.
See [SC2LE]() article for more details.

Though development is research-driven, that does not mean Reaver will never have simplified options that are perhaps more
practical when it comes to competitive one-on-one human vs AI agents that could be trained with reasonable hardware.
See [below]() for a detailed roadmap of the project.

The philosophy behind Reaver API is akin to StarCraft II game itself - it has something to offer both for novices and experts in the field.
For hobbyist programmers Reaver offers all the tools necessary to train DRL agents by tuning some small part of it, e.g.
hyperparameters.


### Example

Reaver is designed to be very easy to experiment with. In fact, you can train a DRL agent with multiple
StarCraft II environments running in parallel with just four lines of code!

```python
import reaver as rvr

env = rvr.envs.SC2Env(map_name='MoveToBeacon')
model_fn = rvr.models.build_fully_conv
policy_cls = rvr.models.SC2MultiPolicy
agent = rvr.agents.AdvantageActorCriticAgent(env.obs_spec(), env.act_spec(), model_fn, policy_cls, n_envs=4)
agent.run(env)
```

Moreover, Reaver comes with highly configurable commandline tools, so this task can be reduced to a short one-liner!

```bash
python -m reaver.run --env MoveToBeacon --agent a2c
```

With the above line Reaver will initialize the training procedure with a set of pre-defined hyperparameters optimized
specifically for the given environment and agent. After awhile you will start seeing logs with many useful statistics
in your terminal screen. Probably most important one is the `RMe`, which stands for `Mean Episode Total Rewards` 
(averaged over 100 episodes). Please see [below]() for a detailed description of each column.
    
    | T      1 | Fr       512 | Ep      0 | Up      1 | RMe    0.00 | RSd    0.00 | RMa    0.00 | RMi    0.00 | Pl    0.749 | Vl    4.084 | El 0.0165 | Gr   23.691 | Fps   512 |
    | T     29 | Fr     51200 | Ep    192 | Up    100 | RMe    0.23 | RSd    0.58 | RMa    4.00 | RMi    0.00 | Pl   -0.001 | Vl    0.029 | El 0.0190 | Gr   10.231 | Fps   562 |
    | T     57 | Fr    102400 | Ep    416 | Up    200 | RMe   25.16 | RSd    2.74 | RMa   30.00 | RMi   16.00 | Pl   -0.007 | Vl    0.156 | El 0.0159 | Gr    8.420 | Fps   498 |
    | T     86 | Fr    153600 | Ep    640 | Up    300 | RMe   25.71 | RSd    1.88 | RMa   31.00 | RMi   22.00 | Pl   -0.004 | Vl    8.799 | El 0.0164 | Gr   85.869 | Fps   512 |
    | T    114 | Fr    204800 | Ep    832 | Up    400 | RMe   26.05 | RSd    1.69 | RMa   31.00 | RMi   22.00 | Pl   -0.006 | Vl    0.096 | El 0.0174 | Gr    9.763 | Fps   512 |

Reaver should quickly converge to about 25-26 `RMe` , which matches [DeepMind results]() for this environment.
Specific training time depends on your hardware. On on a laptop with Intel i5-7300HQ CPU (4 cores) and GTX 1050 GPU, the training takes around 10 minutes.

After Reaver has finished training, you can look at how performs by appending `--test` and `--render` flags to the one-liner.

```bash
python -m reaver.run --env MoveToBeacon --agent a2c --test --render
```

### Key Features

#### Performance

Many modern DRL algorithms rely on being executed in multiple environments at the same time in parallel. 
As Python has [GIL](), this requirement must be implemented as a multiprocessing solution. 
Majority of open source implementations solve this task with message-based approach 
(e.g. Python `multiprocessing.Pipe` or `MPI`), where individual processes communicate by sending data through 
[IPC](). This is a valid and most likely only reasonable approach for large-scale distributed approaches that companies
like [DeepMind]() and [openAI]() operate on. 

However, for a typical researcher or hobbyist a much more common scenario is having access only to a 
single machine environment, whether it is a laptop or a node on a HPC cluster. Reaver is optimized specifically 
for this use case by making use of shared memory in a lock-free manner. This approach nets significant performance
boost of up to **300%** speed-up in StarCraft II sampling rate (and up to 100x speedup in general case), 
being bottle-necked almost exclusively by GPU input/output pipeline.

#### Easy To Use

#### Easy To Extend

#### Easy To Configure

### Implemented Agents

* Advantage Actor-Critic (A2C)
* Proximal Policy Optimization (PPO)

### Additional RL Features

* Generalized Advantage Estimation (GAE)
* Rewards clipping
* Gradient norm clipping
* Advantage normalization
* Baseline (critic) bootstrapping
* Separate baseline network

### But Wait! There's more!

#### Supported Environments

## Results

Map                         | Human Expert | DeepMind SC2LE | DeepMind ReDRL |   Reaver (A2C) |
:-------------------------- | -----------: | -------------: | -------------: | -------------: |
MoveToBeacon                |           28 |             26 |             27 |   26.30 (1.88) |
CollectMineralShards        |          177 |            103 |            196 | 102.79 (10.86) |
DefeatRoaches               |          215 |            100 |            303 |             -- |
DefeatZerglingsAndBanelings |          727 |             62 |            736 |             -- |
FindAndDefeatZerglings      |           61 |             45 |             62 |             -- |
CollectMineralsAndGas       |        7,566 |          3,978 |          5,055 |             -- |
BuildMarines                |          133 |              3 |            123 |             -- |

* `Human Expert` results were gathered by DeepMind from a GrandMaster level player.
* `DeepMind SC2LE` are the baseline results published by DeepMind in the
[StarCraft II: A New Challenge for Reinforcement Learning](https://arxiv.org/abs/1708.04782) article.
* `DeepMind ReDRL` refers to current state-of-the-art results, described in [Relational Deep Reinforcement Learning](https://arxiv.org/abs/1806.01830) article.
* `Reaver (A2C)` are results gathered by training the `reaver.agents.AdvantageActorCritic` agent with
architecture replicating SC2LE as closely as possible on available hardware.
Results are gathered by running the trained agent in `--test` mode for `100` episodes, calculating mean episode total rewards.
Listed inn parenthesis are the standard deviation values.

### Training

Map                         |        Samples |       Episodes | Approx. Time (hr) |
:-------------------------- | -------------: | -------------: | ----------------: |
MoveToBeacon                |        563,200 |          2,304 |              0.25 |
CollectMineralShards        |     74,752,000 |        311,426 |                50 |
DefeatRoaches               |              - |              - |                 - |
DefeatZerglingsAndBanelings |              - |              - |                 - |
FindAndDefeatZerglings      |              - |              - |                 - |
CollectMineralsAndGas       |              - |              - |                 - |
BuildMarines                |              - |              - |                 - |

* `Samples` refer to a number of `observer -> step -> reward` chains in *one* environment.
* `Episodes` refer to number of `StepType.LAST` flags returned by PySC2.
* `Approx. Time` is the time in hours required to train an agent on a `laptop` with Intel `i5-7300HQ` CPU (4 cores) and `GTX 1050` GPU.

### Video Recording

### Reproducibility

#### Pre-trained Weights

#### Tensorboard Summary Logs

## Installation

### Requirements

### As a PIP Package

### Manual Installation

### Optimized TensorFlow

## Roadmap

In this section you can see a birdseye view of my high level plans for Reaver.
If you are interested in a particular feature, then feel free to comment on the attached ticket.
Any help with development is of course highly appreciated, assuming contributed codebase license matches (MIT).

* [ ] Quality of Life improvements
  * [ ] Plotting utility for generating research article friendly plots
  * [ ] Running & comparing experiments across many random seeds
  * [ ] Copying previously executed experiment
* [ ] Documentation
  * [ ] Codebase documentation
  * [ ] Extending to custom environments
  * [ ] Extending to custom agents
  * [ ] Setup on [Google Colab]()
* [ ] Unit tests
  * [ ] For critical features such as advantage estimation
  * [ ] General basic convergence guarantees of agents / models combinations
* [ ] LSTM support
  * [ ] for simpler gym environments
  * [ ] for full StarCraft II environment
* [ ] Asynchronous multiprocessing
* [ ] Additional agents
  * [ ] Behavior Cloning / Imitation Learning
  * [ ] IMPALA + PopArt
  * [ ] Ape-X
  * [ ] ACKTR
* [ ] StarCraft II [raw API]() support
  * [ ] investigate if [python-sc2]() is suitable for the task
* [ ] Support for more environments
  * [ ] [VizDoom](https://github.com/mwydmuch/ViZDoom)
  * [ ] [DeepMind Lab](https://github.com/deepmind/lab)
  * [ ] [Gym Retro](https://github.com/openai/retro)
  * [ ] [CARLA](https://github.com/carla-simulator/carla)
* [ ] Multi-Agent setup support
  * [ ] as a proof of concept on `Pong` environment through [retro-gym]()
  * [ ] for StarCraft II through raw API
  * [ ] for StarCraft II through featured layer API

## Why "Reaver"?

Reaver is a very special and somewhat cute Protoss unit in the StarCraft game universe.
Specifically, in the StarCraft: Brood War version, Reaver was notorious for being slow, clumsy,
and borderline useless if left on its own due to buggy in-game AI. However, Reaver becomes one of the most powerful
and game changing assets in the hands of dedicated and skilled players.

## Acknowledgement

A predecessor to Reaver, named simply `pysc2-rl-agent`, was developed as part of [bachelor's thesis](https://github.com/inoryy/bsc-thesis)
at the University of Tartu under the supervision of [Ilya Kuzovkin](https://github.com/kuz) and [Tambet Matiisen](https://github.com/tambetm/).
You can still access it on the [v1.0]() branch.

## Support

IF you encounter a codebase related problem then please open a ticket on GitHub and describe it in as much detail as possible. 
If you have more general questions or simply seeking advice feel free to send me an email.

I'm also a proud member of active and friendly [SC2AI](http://sc2ai.net) online community, and we mostly use the [Discord](https://discord.gg/UBCjm3) for communication. 
People of all backgrounds and levels of expertise are welcome to join!

## Citing

If you have found Reaver useful in your research, please consider citing it with the following bibtex:

```
@misc{reaver,
  author = {Ring, Roman},
  title = {Reaver: StarCraft II Deep Reinforcement Learning Agent},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/inoryy/reaver}},
}
```
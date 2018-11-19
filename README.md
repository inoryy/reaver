# Reaver: StarCraft II Deep Reinforcement Learning Agent

[![MoveToBeacon](https://user-images.githubusercontent.com/195271/37241507-0d7418c2-2463-11e8-936c-18d08a81d2eb.gif)](https://youtu.be/gEyBzcPU5-w)
[![CollectMineralShards](https://user-images.githubusercontent.com/195271/37241785-b8bd0b04-2467-11e8-9ff3-e4335a7c20ee.gif)](https://youtu.be/gEyBzcPU5-w)
[![DefeatRoaches](https://user-images.githubusercontent.com/195271/37241527-32a43ffa-2463-11e8-8e69-c39a8532c4ce.gif)](https://youtu.be/gEyBzcPU5-w)
[![DefeatZerglingsAndBanelings](https://user-images.githubusercontent.com/195271/37241531-39f186e6-2463-11e8-8aac-79471a545cce.gif)](https://youtu.be/gEyBzcPU5-w)
[![FindAndDefeatZerglings](https://user-images.githubusercontent.com/195271/37241532-3f81fbd6-2463-11e8-8892-907b6acebd04.gif)](https://youtu.be/gEyBzcPU5-w)
[![CollectMineralsAndGas](https://user-images.githubusercontent.com/195271/37241521-29594b48-2463-11e8-8b43-04ad0af6ff3e.gif)](https://youtu.be/gEyBzcPU5-w)
[![BuildMarines](https://user-images.githubusercontent.com/195271/37241515-1a2a5c8e-2463-11e8-8ac4-588d7826e374.gif)](https://youtu.be/gEyBzcPU5-w)


## Introduction

Deep Reinforcement Learning Agent For StarCraft II.

### Example

`Reaver` is designed to be very easy to experiment with. In fact, you can train a DRL agent with multiple
StarCraft II environments running in parallel with just four lines of code!

```python
import reaver as rvr

env = rvr.envs.SC2Env(map_name='MoveToBeacon')
agent = rvr.agents.AdvantageActorCriticAgent(env.obs_spec(), env.act_spec(), n_envs=4)
agent.run(env)
```

Moreover, `Reaver` comes with highly configurable commandline tools, so this task can be reduced to a short one-liner!

```bash
python -m reaver.run --env MoveToBeacon --agent a2c
```

With the above line `Reaver` will initialize the training procedure with a set of pre-defined hyperparameters optimized
specifically for the given environment and agent. After awhile you will start seeing logs with many useful statistics
in your terminal screen. Probably most important one is the `RMe`, which stands for `Mean Episode Total Rewards` 
(averaged over 100 episodes). Please see [below]() for a detailed description of each column.
    
    | T      1 | Fr       512 | Ep      0 | Up      1 | RMe    0.00 | RSd    0.00 | RMa    0.00 | RMi    0.00 | Pl    0.749 | Vl    4.084 | El 0.0165 | Gr   23.691 | Fps   512 |
    | T     29 | Fr     51200 | Ep    192 | Up    100 | RMe    0.23 | RSd    0.58 | RMa    4.00 | RMi    0.00 | Pl   -0.001 | Vl    0.029 | El 0.0190 | Gr   10.231 | Fps   562 |
    | T     57 | Fr    102400 | Ep    416 | Up    200 | RMe   25.16 | RSd    2.74 | RMa   30.00 | RMi   16.00 | Pl   -0.007 | Vl    0.156 | El 0.0159 | Gr    8.420 | Fps   498 |
    | T     86 | Fr    153600 | Ep    640 | Up    300 | RMe   25.71 | RSd    1.88 | RMa   31.00 | RMi   22.00 | Pl   -0.004 | Vl    8.799 | El 0.0164 | Gr   85.869 | Fps   512 |
    | T    114 | Fr    204800 | Ep    832 | Up    400 | RMe   26.05 | RSd    1.69 | RMa   31.00 | RMi   22.00 | Pl   -0.006 | Vl    0.096 | El 0.0174 | Gr    9.763 | Fps   512 |

`Reaver` should quickly converge to about 25-26 `RMe` , which matches [DeepMind results]() for this environment.
Specific training time depends on your hardware. On a Dell XPS15 laptop with Intel ... CPU and NVIDIA GTX 1060 GPU, the training takes around 10 minutes.

After `Reaver` has finished training, you can look at how performs by appending `--test` and `--render` flags to the one-liner.

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

### Implemented Algorithms

* Advantage Actor-Critic (A2C)
* Proximal Policy Optimization (PPO)

Additionally, the following features are supported:

* Generalized Advantage Estimation (GAE)
* Rewards clipping
* Gradient norm clipping
* Advantage normalization
* Baseline (critic) bootstrapping
* Separate baseline network

### But Wait! There's more!

#### Supported Environments

## Results

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

## Acknowledgement

## Support

IF you encounter a codebase related problem then please open a ticket on GitHub and describe it in as much detail as possible. 
If you have more general questions or simply seeking advice feel free to send me an email.

I'm also an active member of a great [SC2AI]() online community, and we mostly use the [Discord]() for communication. 
People of all background are welcome to join!

## Citing

If you have found `Reaver` useful in your research, please consider citing it with the following bibtex:

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
# Reaver: Modular Deep Reinforcement Learning Framework

[![MoveToBeacon](https://user-images.githubusercontent.com/195271/48730921-66b6fe00-ec44-11e8-9954-9f4891ff9672.gif)](https://youtu.be/gEyBzcPU5-w)
[![CollectMineralShards](https://user-images.githubusercontent.com/195271/48730941-70d8fc80-ec44-11e8-95ae-acff6f5a9add.gif)](https://youtu.be/gEyBzcPU5-w)
[![DefeatRoaches](https://user-images.githubusercontent.com/195271/48730950-78000a80-ec44-11e8-83a2-2f2bb0bf59ab.gif)](https://youtu.be/gEyBzcPU5-w)
[![DefeatZerglingsAndBanelings](https://user-images.githubusercontent.com/195271/48731288-5fdcbb00-ec45-11e8-8826-4d5683d2c337.gif)](https://youtu.be/gEyBzcPU5-w)
[![FindAndDefeatZerglings](https://user-images.githubusercontent.com/195271/48731379-93b7e080-ec45-11e8-9375-38016ea9c9a8.gif)](https://youtu.be/gEyBzcPU5-w)
[![CollectMineralsAndGas](https://user-images.githubusercontent.com/195271/48730970-86e6bd00-ec44-11e8-8e8c-0181e44b351c.gif)](https://youtu.be/gEyBzcPU5-w)
[![BuildMarines](https://user-images.githubusercontent.com/195271/48730972-89491700-ec44-11e8-8842-4a6b76f08563.gif)](https://youtu.be/gEyBzcPU5-w)

[![MoveToBeacon](https://user-images.githubusercontent.com/195271/37241507-0d7418c2-2463-11e8-936c-18d08a81d2eb.gif)](https://youtu.be/gEyBzcPU5-w)
[![CollectMineralShards](https://user-images.githubusercontent.com/195271/37241785-b8bd0b04-2467-11e8-9ff3-e4335a7c20ee.gif)](https://youtu.be/gEyBzcPU5-w)
[![DefeatRoaches](https://user-images.githubusercontent.com/195271/37241527-32a43ffa-2463-11e8-8e69-c39a8532c4ce.gif)](https://youtu.be/gEyBzcPU5-w)
[![DefeatZerglingsAndBanelings](https://user-images.githubusercontent.com/195271/37241531-39f186e6-2463-11e8-8aac-79471a545cce.gif)](https://youtu.be/gEyBzcPU5-w)
[![FindAndDefeatZerglings](https://user-images.githubusercontent.com/195271/37241532-3f81fbd6-2463-11e8-8892-907b6acebd04.gif)](https://youtu.be/gEyBzcPU5-w)
[![CollectMineralsAndGas](https://user-images.githubusercontent.com/195271/37241521-29594b48-2463-11e8-8b43-04ad0af6ff3e.gif)](https://youtu.be/gEyBzcPU5-w)
[![BuildMarines](https://user-images.githubusercontent.com/195271/37241515-1a2a5c8e-2463-11e8-8ac4-588d7826e374.gif)](https://youtu.be/gEyBzcPU5-w)


## Introduction

Reaver is a modular deep reinforcement learning framework with a focus on various StarCraft II based tasks, following in DeepMind's footsteps 
who are pushing state-of-the-art of the field through the lens of playing a modern video game with human-like interface and limitations. 
This includes observing visual features similar (though not identical) to what a human player would perceive and choosing actions from similar pool of options a human player would have.
See [StarCraft II: A New Challenge for Reinforcement Learning](https://arxiv.org/abs/1708.04782) article for more details.

Though development is research-driven, the philosophy behind Reaver API is akin to StarCraft II game itself - 
it has something to offer both for novices and experts in the field. For hobbyist programmers Reaver offers all the tools
necessary to train DRL agents by modifying only a small and isolated part of the agent (e.g. hyperparameters).
For veteran researchers Reaver offers simple, but performance-optimized codebase with modular architecture: 
agent, model, and environment are decoupled and can be swapped at will.

While the focus of Reaver is on StarCraft II, it also has full support for other popular environments, notably Atari and MuJoCo. 
Reaver agent algorithms are validated against reference results, e.g. PPO agent is able to match [
Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). Please see [below](#but-wait-theres-more) for more details.

## Installation

### Requirements

* numpy >= 1.13
* absl-py >= 0.2.2
* gin-config >= 0.1.1
* TensorFlow >= 1.10
* TensorFlow Probability >= 0.4
* StarCraft II >= 4.1.2 ([instructions](https://github.com/Blizzard/s2client-proto#downloads))
* PySC2 > 2.0.1

**NB!** As of 25/11/18 you must install **PySC2** [from source](https://github.com/deepmind/pysc2#git) since PIP version is outdated:

```
pip install --upgrade https://github.com/deepmind/pysc2/archive/master.zip
```

#### Optional Extras
If you would like to use Reaver with other supported environments, you must install relevant packages as well:

* gym >= 0.10.0
* atari-py >= 0.1.5
* mujoco-py >= 1.50.0
  * roboschool >= 1.0 (alternative)

### PIP Package

Easiest way to install Reaver is through the `PIP` package manager:
 
    pip install reaver

**NB!** PySC2 PIP version is outdated, so you will need to install it from source as described above.

**NB!** Reaver specifies `TensorFlow` only as a soft dependency and it will not be installed by default. This is to avoid
`tensorflow` overwriting `tensorflow-gpu` and vise-versa. You can install `tensorflow` along with Reaver by specifying either
`tf-cpu` or `tf-gpu` flag with `pip install` command: 

    pip install reaver[tf-gpu]

You can also install additional extras (e.g. `gym`) through the helper flags: 

    pip install reaver[tf-gpu,gym,atari,mujoco]

### Manual Installation

If you plan to modify `Reaver` codebase you can retain its module functionality by installing from source:

```
$ git clone https://github.com/inoryy/reaver-pysc2
$ pip install -e reaver-pysc2/
```

By installing with `-e` flag `Python` will now look for `reaver` in the specified folder, rather than `site-packages` storage.

### Optimized TensorFlow

The `TensorFlow` that is distributed through `PIP` is built to target as many architectures / devices as possible, which
means that various optimization flags are disabled by default. For example, if your CPU supports `AVX2` (is newer than 5 years),
it is highly recommended to use a custom built TensorFlow instead. If building from source is not an option for you,
then [this repository](https://github.com/inoryy/tensorflow-optimized-wheels) might be useful - it contains newest `TensorFlow`
releases built for newest CUDA / CuDNN versions, which often come with performance boosts even for older GPUs.

### Windows

Please see the [wiki](https://github.com/inoryy/reaver-pysc2/wiki/Windows) page for detailed instructions on setting up Reaver on Windows.

However, if possible please consider using `Linux OS` instead - due to performance and stability considerations.
If you would like to see your agent perform with full graphics enabled you can save a replay of the agent on Linux and open it on Windows.
This is how the video recording listed below was made.


## Quick Start

You can train a DRL agent with multiple StarCraft II environments running in parallel with just four lines of code!

```python
import reaver as rvr

env = rvr.envs.SC2Env(map_name='MoveToBeacon')
agent = rvr.agents.A2C(env.obs_spec(), env.act_spec(), rvr.models.build_fully_conv, rvr.models.SC2MultiPolicy, n_envs=4)
agent.run(env)
```

Moreover, Reaver comes with highly configurable commandline tools, so this task can be reduced to a short one-liner!

```bash
python -m reaver.run --env MoveToBeacon --agent a2c --n_envs 4 2> stderr.log
```

With the line above Reaver will initialize the training procedure with a set of pre-defined hyperparameters, optimized
specifically for the given environment and agent. After awhile you will start seeing logs with various useful statistics
in your terminal screen.

    | T    118 | Fr     51200 | Ep    212 | Up    100 | RMe    0.14 | RSd    0.49 | RMa    3.00 | RMi    0.00 | Pl    0.017 | Vl    0.008 | El 0.0225 | Gr    3.493 | Fps   433 |
    | T    238 | Fr    102400 | Ep    424 | Up    200 | RMe    0.92 | RSd    0.97 | RMa    4.00 | RMi    0.00 | Pl   -0.196 | Vl    0.012 | El 0.0249 | Gr    1.791 | Fps   430 |
    | T    359 | Fr    153600 | Ep    640 | Up    300 | RMe    1.80 | RSd    1.30 | RMa    6.00 | RMi    0.00 | Pl   -0.035 | Vl    0.041 | El 0.0253 | Gr    1.832 | Fps   427 |
    ...
    | T   1578 | Fr    665600 | Ep   2772 | Up   1300 | RMe   24.26 | RSd    3.19 | RMa   29.00 | RMi    0.00 | Pl    0.050 | Vl    1.242 | El 0.0174 | Gr    4.814 | Fps   421 |
    | T   1695 | Fr    716800 | Ep   2984 | Up   1400 | RMe   24.31 | RSd    2.55 | RMa   30.00 | RMi   16.00 | Pl    0.005 | Vl    0.202 | El 0.0178 | Gr   56.385 | Fps   422 |
    | T   1812 | Fr    768000 | Ep   3200 | Up   1500 | RMe   24.97 | RSd    1.89 | RMa   31.00 | RMi   21.00 | Pl   -0.075 | Vl    1.385 | El 0.0176 | Gr   17.619 | Fps   423 |


Reaver should quickly converge to about 25-26 `RMe` (mean episode rewards), which matches DeepMind results for this environment.
Specific training time depends on your hardware. Logs above are produced on a laptop with Intel i5-7300HQ CPU (4 cores)
and GTX 1050 GPU, the training took around 30 minutes.

After Reaver has finished training, you can look at how it performs by appending `--test` and `--render` flags to the one-liner.

```bash
python -m reaver.run --env MoveToBeacon --agent a2c --test --render 2> stderr.log
```

### Google Colab

A companion [Google Colab notebook](https://colab.research.google.com/drive/1DvyCUdymqgjk85FB5DrTtAwTFbI494x7) 
notebook is available to try out Reaver online.

## Key Features

### Performance

Many modern DRL algorithms rely on being executed in multiple environments at the same time in parallel. 
As Python has [GIL](https://wiki.python.org/moin/GlobalInterpreterLock), this feature must be implemented through multiprocessing. 
Majority of open source implementations solve this task with message-based approach (e.g. Python `multiprocessing.Pipe` or `MPI`),
where individual processes communicate by sending data through [IPC](https://en.wikipedia.org/wiki/Inter-process_communication).
This is a valid and most likely only reasonable approach for large-scale distributed approaches that companies like DeepMind and openAI operate on. 

However, for a typical researcher or hobbyist a much more common scenario is having access only to a 
single machine environment, whether it is a laptop or a node on a HPC cluster. Reaver is optimized specifically 
for this case by making use of shared memory in a lock-free manner. This approach nets significant performance
boost of up to **1.5x speed-up** in StarCraft II sampling rate (and up to 100x speedup in general case),
being bottle-necked almost exclusively by GPU input/output pipeline.

### Extensibility

The three core Reaver modules - `envs`, `models`, and `agents` are almost completely detached from each other.
This ensures that extending functionality in one module is seamlessly integrated into the others.

### Configurability

All configuration is handled through [gin-config](https://github.com/google/gin-config) and can be easily shared as `.gin` files. 
This includes all hyperparameters, environment arguments, and model definitions.

### Implemented Agents

* Advantage Actor-Critic (A2C)
* Proximal Policy Optimization (PPO)

#### Additional RL Features

* Generalized Advantage Estimation (GAE)
* Rewards clipping
* Gradient norm clipping
* Advantage normalization
* Baseline (critic) bootstrapping
* Separate baseline network

### But Wait! There's more!

When experimenting with novel ideas it is important to get feedback quickly, which is often not realistic with complex environments like StarCraft II.
As Reaver was built with modular architecture, its agent implementations are not actually tied to StarCraft II at all.
You can make drop-in replacements for many popular game environments (e.g. `openAI gym`) and verify implementations work with those first:

```bash
python -m reaver.run --env CartPole-v0 --agent a2c 2> stderr.log
```

```python
import reaver as rvr

env = rvr.envs.GymEnv('CartPole-v0')
agent = rvr.agents.A2C(env.obs_spec(), env.act_spec())
agent.run(env)
```

### Supported Environments

Currently the following environments are supported by Reaver:

* StarCraft II via PySC2 (tested on all minigames)
* openAI Gym (tested on `CartPole-v0`)
* Atari (tested on `PongNoFrameskip-v0`)
* Mujoco (tested on `InvertedPendulum-v2` and `HalfCheetah-v2`)

## Results

Map                         |                 Reaver (A2C) | DeepMind SC2LE | DeepMind ReDRL | Human Expert |
:-------------------------- | ---------------------------: | -------------: | -------------: | -----------: |
MoveToBeacon                |       26.3 (1.8)<br>[21, 31] |             26 |             27 |           28 |
CollectMineralShards        |    102.8 (10.8)<br>[81, 135] |            103 |            196 |          177 |
DefeatRoaches               |     72.5 (43.5)<br>[21, 283] |            100 |            303 |          215 |
FindAndDefeatZerglings      |       22.1 (3.6)<br>[12, 40] |             45 |             62 |           61 |
DefeatZerglingsAndBanelings |     56.8 (20.8)<br>[21, 154] |             62 |            736 |          727 |
CollectMineralsAndGas       |  2267.5 (488.8)<br>[0, 3320] |          3,978 |          5,055 |        7,566 |
BuildMarines                |                           -- |              3 |            123 |          133 |

* `Human Expert` results were gathered by DeepMind from a GrandMaster level player.
* `DeepMind ReDRL` refers to current state-of-the-art results, described in [Relational Deep Reinforcement Learning](https://arxiv.org/abs/1806.01830) article.
* `DeepMind SC2LE` are results published in [StarCraft II: A New Challenge for Reinforcement Learning](https://arxiv.org/abs/1708.04782) article.
* `Reaver (A2C)` are results gathered by training the `reaver.agents.A2C` agent, replicating `SC2LE` architecture as closely as possible on available hardware.
Results are gathered by running the trained agent in `--test` mode for `100` episodes, calculating episode total rewards.
Listed are the mean, standard deviation (in parentheses), and min & max (in square brackets).

### Training Details

Map                         |        Samples |       Episodes | Approx. Time (hr) |
:-------------------------- | -------------: | -------------: | ----------------: |
MoveToBeacon                |        563,200 |          2,304 |               0.5 |
CollectMineralShards        |     74,752,000 |        311,426 |                50 |
DefeatRoaches               |    172,800,000 |      1,609,211 |               150 |
FindAndDefeatZerglings      |     29,760,000 |         89,654 |                20 |
DefeatZerglingsAndBanelings |     10,496,000 |        273,463 |                15 |
CollectMineralsAndGas       |     16,864,000 |         20,544 |                10 |
BuildMarines                |              - |              - |                 - |

* `Samples` refer to total number of `observe -> step -> reward` chains in *one* environment.
* `Episodes` refer to total number of `StepType.LAST` flags returned by PySC2.
* `Approx. Time` is the approximate training time on a `laptop` with Intel `i5-7300HQ` CPU (4 cores) and `GTX 1050` GPU.

Note that I did not put much time into hyperparameter tuning, focusing mostly on verifying that the agent is capable of learning
rather than maximizing sample efficiency. For example, naive first try on `MoveToBeacon` required about 4 million samples,
however after some playing around I was able to reduce it down all the way to 102,000 (~40x reduction) with PPO agent.

[![](https://i.imgur.com/rIoc6rTl.png)](https://i.imgur.com/rIoc6rT.png)  
Mean episode rewards with std.dev filled in-between. Click to enlarge.

### Video Recording

A video recording of the agent performing on all six minigames is available online at: [https://youtu.be/gEyBzcPU5-w](https://youtu.be/gEyBzcPU5-w).
In the video on the left is the agent acting in with randomly initialized weights and no training, whereas on the right he is trained to target scores.

## Reproducibility

The problem of reproducibility of research has recently become a subject of many debates in science
[in general](https://www.nature.com/news/1-500-scientists-lift-the-lid-on-reproducibility-1.19970), 
and Reinforcement Learning is [not an exception](https://arxiv.org/abs/1709.06560).
One of the goals of Reaver as a scientific project is to help facilitate reproducible research.
To this end Reaver comes bundled with various tools that simplify the process:

* All experiments are saved into separate folders with automatic model checkpoints enabled by default
* All configuration is handled through [gin-config](https://github.com/google/gin-config) Python library and saved to experiment results directory
* During training various statistics metrics are duplicated into experiment results directory
* Results directory structure simplifies sharing individual experiments with full information

### Pre-trained Weights & Summary Logs

To lead the way with reproducibility, Reaver is bundled with pre-trained weights and full Tensorboard summary logs for all six minigames. 
Simply download an experiment archive from the [releases](https://github.com/inoryy/reaver-pysc2/releases) tab and unzip onto the `results/` directory.

You can use pre-trained weights by appending `--experiment` flag to `reaver.run` command:

    python reaver.run --map <map_name> --experiment <map_name>_reaver --test 2> stderr.log

Tensorboard logs are available if you launch `tensorboard --logidr=results/summaries`.  
You can also view them [directly online](https://boards.aughie.org/board/HWi4xmuvuOSuw09QBfyDD-oNF1U) via [Aughie Boards](https://boards.aughie.org/).

## Roadmap

In this section you can get a birdseye overview of my plans for the project in no particular order. You can also follow 
what I'm currently working on in the [projects tab](https://github.com/inoryy/reaver-pysc2/projects). 
Any help with development is of course highly appreciated, assuming contributed codebase license matches (MIT).

* [ ] Documentation
  * [ ] Codebase documentation
  * [ ] Extending to custom environments
  * [ ] Extending to custom agents
  * [x] Setup on [Google Colab](https://colab.research.google.com)
* [ ] Unit tests
  * [x] For critical features such as advantage estimation
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
* [ ] Quality of Life improvements
  * [x] Plotting utility for generating research article friendly plots
  * [ ] Running & comparing experiments across many random seeds
  * [ ] Copying previously executed experiment
* [ ] StarCraft II [raw API](https://github.com/Blizzard/s2client-proto/blob/master/docs/protocol.md#raw-data) support
* [ ] Support for more environments
  * [ ] [VizDoom](https://github.com/mwydmuch/ViZDoom)
  * [ ] [DeepMind Lab](https://github.com/deepmind/lab)
  * [ ] [Gym Retro](https://github.com/openai/retro)
  * [ ] [CARLA](https://github.com/carla-simulator/carla)
* [ ] Multi-Agent setup support
  * [ ] as a proof of concept on `Pong` environment through [Gym Retro](https://github.com/openai/retro)
  * [ ] for StarCraft II through raw API
  * [ ] for StarCraft II through featured layer API

## Why "Reaver"?

Reaver is a very special and subjectively cute Protoss unit in the StarCraft game universe.
In the StarCraft: Brood War version of the game, Reaver was notorious for being slow, clumsy,
and often borderline useless if left on its own due to buggy in-game AI. However, in the hands of dedicated players that invested
time into mastery of the unit, Reaver became one of the most powerful assets in the game, often playing a key role in tournament winning games.

## Acknowledgement

A predecessor to Reaver, named simply `pysc2-rl-agent`, was developed as the practical part of
[bachelor's thesis](https://github.com/inoryy/bsc-thesis) at the University of Tartu under the
supervision of [Ilya Kuzovkin](https://github.com/kuz) and [Tambet Matiisen](https://github.com/tambetm).
You can still access it on the [v1.0](https://github.com/inoryy/reaver-pysc2/tree/v1.0) branch.

## Support

If you encounter a codebase related problem then please open a ticket on GitHub and describe it in as much detail as possible. 
If you have more general questions or simply seeking advice feel free to send me an email.

I am also a proud member of an active and friendly [SC2AI](http://sc2ai.net) online community, 
we mostly use [Discord](https://discordapp.com/invite/Emm5Ztz) for communication. People of all backgrounds and levels of expertise are welcome to join!

## Citing

If you have found Reaver useful in your research, please consider citing it with the following bibtex:

```
@misc{reaver,
  author = {Ring, Roman},
  title = {Reaver: Modular Deep Reinforcement Learning Framework},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/inoryy/reaver}},
}
```

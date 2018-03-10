# (D)RL Agent For PySC2 Environment

[![MoveToBeacon](https://user-images.githubusercontent.com/195271/37241507-0d7418c2-2463-11e8-936c-18d08a81d2eb.gif)](https://youtu.be/QdeObwCCxFI)
[![CollectMineralShards](https://user-images.githubusercontent.com/195271/37241785-b8bd0b04-2467-11e8-9ff3-e4335a7c20ee.gif)](https://youtu.be/QdeObwCCxFI)
[![DefeatRoaches](https://user-images.githubusercontent.com/195271/37241527-32a43ffa-2463-11e8-8e69-c39a8532c4ce.gif)](https://youtu.be/QdeObwCCxFI)
[![DefeatZerglingsAndBanelings](https://user-images.githubusercontent.com/195271/37241531-39f186e6-2463-11e8-8aac-79471a545cce.gif)](https://youtu.be/QdeObwCCxFI)
[![FindAndDefeatZerglings](https://user-images.githubusercontent.com/195271/37241532-3f81fbd6-2463-11e8-8892-907b6acebd04.gif)](https://youtu.be/QdeObwCCxFI)
[![CollectMineralsAndGas](https://user-images.githubusercontent.com/195271/37241521-29594b48-2463-11e8-8b43-04ad0af6ff3e.gif)](https://youtu.be/QdeObwCCxFI)
[![BuildMarines](https://user-images.githubusercontent.com/195271/37241515-1a2a5c8e-2463-11e8-8ac4-588d7826e374.gif)](https://youtu.be/QdeObwCCxFI)


## Introduction

Aim of this project is two-fold: 

a.) Reproduce baseline DeepMind results by implementing RL agent (A2C) with neural network model 
architecture as close as possible to what is described in [1]. 
This includes embedding categorical (spatial-)features into continuous space with 1x1 convolution 
and multi-head policy, supporting actions with variable arguments (both spatial and non-spatial).

b.) Improve the results and/or sample efficiency of the baseline solution. Either with alternative algorithms (such as PPO [2]), 
using reduced set of features (unified across all mini-games) or alternative approaches, such as HRL [3] or Auxiliary Tasks [4].


## Results

Map | This Agent | DeepMind
---|---|---
MoveToBeacon | 26.3 | 26
CollectMineralShards | 102 | 103
FindAndDefeatZerglings | 25 | 45
DefeatRoaches | 126* | 100
DefeatZerglingsAndBanelings | 197* | 62
CollectMineralsAndGas | 3340 | 3978
BuildMarines | 0.55 | 3

\* Unstable result with high std.dev (40 for *DefeatRoaches* and 120 for *DefeatZerglingsAndBanelings*)

A video of the trained agent on all minigames can be seen here: https://youtu.be/QdeObwCCxFI

## Running

* To train an agent, execute `python main.py --envs=1 --map=MoveToBeacon`.
* To resume training from last checkpoint, specify `--restore` flag
* To run in inference mode, specify `--test` flag
* To change number of rendered environments, specify `--render=` flag
* To change state/action space, specify path to a json config with `--cfg_path=`. The configuration with reduced feature space used to achieve some of the results above is:

```json
{
  "feats": {
    "screen": ["visibility_map", "player_relative", "unit_type", "selected", "unit_hit_points_ratio", "unit_density"],
    "minimap": ["visibility_map", "camera", "player_relative", "selected"],
    "non_spatial": ["player", "available_actions"]
  }
}
```

### Requirements

* Python 3.x
* Tensorflow >= 1.3
* PySC2 [with action spec fix](https://github.com/deepmind/pysc2/pull/105)

Good GPU and CPU are recommended, especially for full state/action space.


## Related Work

Authors of [xhujoy/pysc2-agents](https://github.com/xhujoy/pysc2-agents) and [pekaalto/sc2aibot](https://github.com/pekaalto/sc2aibot) 
were the first to attempt replicating [1] and their implementations were used as a general inspiration during development 
of this project, however their aim was more towards replicating results than architecture, missing key aspects, 
such as full feature and action space support. 
Authors of [simonmeister/pysc2-rl-agents](https://github.com/simonmeister/pysc2-rl-agents) 
also aim to replicate both results and architecture, though their final goals seem to be in another direction. Their policy implementation was used as a loose reference for this project.

## References

[1] [StarCraft II: A New Challenge for Reinforcement Learning](https://arxiv.org/abs/1708.04782)  
[2] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)  
[3] [Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation](https://arxiv.org/abs/1604.06057)  
[4] [Reinforcement Learning with Unsupervised Auxiliary Tasks](https://arxiv.org/abs/1611.05397) 
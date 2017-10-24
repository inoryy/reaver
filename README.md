# (D)RL agent for PySC2 environment

## Requirements

* Python 3.x
* PySC2 [with action spec fix](https://github.com/deepmind/pysc2/pull/105)
* gflags

## Running

* `python play.py --n_envs=2` for scripted agent
* `python play.py --n_envs=2 --agent=agent.random.RandomAgent` for random agent
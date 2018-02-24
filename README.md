# (D)RL Agent For PySC2 Environment

## Requirements

* Python 3.x
* PySC2 [with action spec fix](https://github.com/deepmind/pysc2/pull/105)

## Results

Map | This Agent | Deepmind
---|---|---
MoveToBeacon | 26.3 | 26
CollectMineralShards | 102 | 103
FindAndDefeatZerglings | 25 | 45
DefeatRoaches | 126* | 100
DefeatZerglingsAndBanelings | 197* | 62
CollectMineralsAndGas | 3340 | 3978
BuildMarines | 0.55 | 3

\* Unstable result with high std.dev (30 for *DefeatRoaches* and 100 for *DefeatZerglingsAndBanelings*)

## Running

* `python main.py --envs=2 --render=1 --map=MoveToBeacon` 
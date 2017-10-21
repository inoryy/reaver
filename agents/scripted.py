from pysc2.agents.scripted_agent import MoveToBeacon


class ParallelMoveToBeacon(MoveToBeacon):
    def __init__(self):
        super().setup(None, None)

    def step(self, obs):
        acts = []
        for i in range(len(obs)):
            acts.append(super().step(obs[i]))
        return acts

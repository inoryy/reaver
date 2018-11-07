import numpy as np


class AgentLogger:
    def __init__(self, agent, log_n_steps=2000, detailed_steps=5, verbosity=3):
        self.agent, self.verbosity = agent, verbosity
        self.log_n_steps, self.detailed_steps = log_n_steps, detailed_steps

    def on_step(self, step, loss_terms, adv, next_value):
        if self.verbosity < 1 or (step+1) % self.log_n_steps:
            return

        loss_terms = np.array(loss_terms).round(5)
        np.set_printoptions(suppress=True, precision=3)
        print("######################################################")
        print("Steps        ", step+1)
        print("Policy loss  ", loss_terms[0])
        print("Value  loss  ", loss_terms[1])
        print("Entropy loss ", loss_terms[2])

        if self.verbosity < 2 or self.detailed_steps == 0:
            return

        n_steps = self.detailed_steps
        logits = self.agent.tf_run(self.agent.model.logits, self.agent.model.inputs,
                                   [o[-n_steps:, 0] for o in self.agent.obs])
        action_ids = self.agent.acts[0][-n_steps:, 0].flatten()

        print()
        print("First Env For Last %d Steps:" % n_steps)
        print("Dones      ", self.agent.dones[-n_steps:, 0].flatten())
        print("Rewards    ", self.agent.rewards[-n_steps:, 0].flatten())
        print("Values     ", self.agent.values[-n_steps:, 0].flatten(), round(next_value[0], 3))
        print("Advs       ", adv[-n_steps:, 0].flatten())
        print("Action ids ", action_ids)
        print("Act logits ", logits[0][-np.arange(n_steps), action_ids])

        if self.verbosity < 3:
            return

        print()
        for t in range(n_steps):
            trv = n_steps - t
            avail = np.argwhere(self.agent.obs[2][-trv, 0]).flatten()
            avail_logits = logits[0][t, avail].flatten()
            print("Step", -trv+1)
            print("Avail ids (%d)" % self.agent.obs[2][-trv, 0].sum(), avail)
            print("Logits   ", avail_logits)
        print("######################################################")

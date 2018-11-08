import time
import numpy as np


class AgentLogger:
    def __init__(self, agent, n_steps=2000, n_detailed=10, verbosity=4):
        self.agent, self.verbosity = agent, verbosity
        self.n_steps, self.n_detailed = n_steps, n_detailed
        self.env_eps = np.zeros(self.agent.n_envs)
        self.env_rews = np.zeros(self.agent.n_envs)

    def on_step(self, step, loss_terms, returns, adv, next_value):
        self.env_eps += np.sum(self.agent.dones, axis=0)
        self.env_rews += np.sum(self.agent.rewards, axis=0)

        if self.verbosity < 1 or (step+1) % self.n_steps:
            return

        loss_terms = np.array(loss_terms).round(5)
        np.set_printoptions(suppress=True, precision=3)

        print("######################################################")
        runtime = int(time.time() - self.agent.start_time)
        frames = (step+1) * self.agent.n_envs

        print("Runner Stats:")
        print("Time    ", runtime)
        print("Eps     ", int(np.sum(self.env_eps)))
        print("Frames  ", frames)
        print("Steps   ", step+1)
        print("Updates ", (step+1) // self.agent.batch_sz)
        print("FPS     ", frames // runtime)

        print()
        print("Total Rewards:")
        tot_rews = (self.env_eps > 0) * self.env_rews / (self.env_eps + 1e-10)
        print("Mean %.3f " % np.mean(tot_rews))
        print("Std  %.3f  " % np.std(tot_rews))
        print("Min  %.3f  " % np.min(tot_rews))
        print("Max  %.3f  " % np.max(tot_rews))

        if self.verbosity < 2:
            return

        print()
        print("Losses For Last Update:")
        print("Total loss   ", loss_terms[3])
        print("Policy loss  ", loss_terms[0])
        print("Value loss   ", loss_terms[1])
        print("Entropy loss ", loss_terms[2])

        if self.verbosity < 3:
            return

        np.set_printoptions(suppress=True, precision=2)
        n_steps = min(self.n_detailed, self.agent.batch_sz)

        logits = self.agent.tf_run(self.agent.model.logits, self.agent.model.inputs,
                                   [o[-n_steps:, 0] for o in self.agent.obs])
        action_ids = self.agent.acts[0][-n_steps:, 0].flatten()

        print()
        print("First Env For Last %d Steps:" % n_steps)
        print("Dones      ", self.agent.dones[-n_steps:, 0].flatten().astype(int))
        print("Rewards    ", self.agent.rewards[-n_steps:, 0].flatten())
        print("Values     ", self.agent.values[-n_steps:, 0].flatten(), round(next_value[0], 3))
        print("Returns    ", returns[-n_steps:, 0].flatten())
        print("Advs       ", adv[-n_steps:, 0].flatten())
        print("Action ids ", action_ids)
        print("Act logits ", logits[0][-np.arange(n_steps), action_ids])

        if self.verbosity < 4:
            return

        print()
        print("Note: action ids listed are not equivalent to pysc2")
        for t in range(n_steps-1, -1, -1):
            trv = n_steps - t
            avail = np.argwhere(self.agent.obs[2][-trv, 0]).flatten()
            avail_logits = logits[0][t, avail].flatten()
            avail_sorted = np.argsort(avail_logits)
            print("Step", -trv+1)
            print("Actions   ", self.agent.obs[2][-trv, 0].sum())
            print("Action ids", avail[avail_sorted[:3]], "..."*3, avail[avail_sorted[-5:]])
            print("Logits    ", avail_logits[avail_sorted[:3]], "...", avail_logits[avail_sorted[-5:]])
        print("######################################################")

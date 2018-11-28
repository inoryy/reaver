import numpy as np
from multiprocessing import Pipe, Process
from . import Env

START, STEP, RESET, STOP, DONE = range(5)


class MsgProcEnv(Env):
    def __init__(self, env):
        super().__init__(env.id)
        self._env = env
        self.conn = self.w_conn = self.proc = None

    def start(self):
        self.conn, self.w_conn = Pipe()
        self.proc = Process(target=self._run)
        self.proc.start()
        self.conn.send((START, None))

    def step(self, act):
        self.conn.send((STEP, act))

    def reset(self):
        self.conn.send((RESET, None))

    def stop(self):
        self.conn.send((STOP, None))

    def wait(self):
        return self.conn.recv()

    def obs_spec(self):
        return self._env.obs_spec()

    def act_spec(self):
        return self._env.act_spec()

    def _run(self):
        while True:
            msg, data = self.w_conn.recv()
            if msg == START:
                self._env.start()
                self.w_conn.send(DONE)
            elif msg == STEP:
                obs, rew, done = self._env.step(data)
                self.w_conn.send((obs, rew, done))
            elif msg == RESET:
                obs = self._env.reset()
                self.w_conn.send((obs, -1, -1))
            elif msg == STOP:
                self._env.stop()
                self.w_conn.close()
                break


class MsgMultiProcEnv(Env):
    """
    Parallel environments via multiprocessing + pipes
    """
    def __init__(self, envs):
        super().__init__(envs[0].id)
        self.envs = [MsgProcEnv(env) for env in envs]

    def start(self):
        for env in self.envs:
            env.start()
        self.wait()

    def step(self, actions):
        for idx, env in enumerate(self.envs):
            env.step([a[idx] for a in actions])
        return self._observe()

    def reset(self):
        for e in self.envs:
            e.reset()
        return self._observe()

    def _observe(self):
        obs, reward, done = zip(*self.wait())
        # n_envs x n_spaces -> n_spaces x n_envs
        obs = list(map(np.array, zip(*obs)))

        return obs, np.array(reward), np.array(done)

    def stop(self):
        for e in self.envs:
            e.stop()
        for e in self.envs:
            e.proc.join()

    def wait(self):
        return [e.wait() for e in self.envs]

    def obs_spec(self):
        return self.envs[0].obs_spec()

    def act_spec(self):
        return self.envs[0].act_spec()

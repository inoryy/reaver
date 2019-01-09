import ctypes
import numpy as np
from multiprocessing import Pipe, Process
from multiprocessing.sharedctypes import RawArray
from . import Env, Space

START, STEP, RESET, STOP, DONE = range(5)


class ShmProcEnv(Env):
    def __init__(self, env, idx, shm):
        super().__init__(env.id)
        self._env, self.idx, self.shm = env, idx, shm
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
        try:
            while True:
                msg, data = self.w_conn.recv()
                if msg == START:
                    self._env.start()
                    self.w_conn.send(DONE)
                elif msg == STEP:
                    obs, rew, done = self._env.step(data)
                    for shm, ob in zip(self.shm, obs + [rew, done]):
                        np.copyto(dst=shm[self.idx], src=ob)
                    self.w_conn.send(DONE)
                elif msg == RESET:
                    obs = self._env.reset()
                    for shm, ob in zip(self.shm, obs + [0, 0]):
                        np.copyto(dst=shm[self.idx], src=ob)
                    self.w_conn.send(DONE)
                elif msg == STOP:
                    self._env.stop()
                    self.w_conn.close()
                    break
        except KeyboardInterrupt:
            self._env.stop()
            self.w_conn.close()


class ShmMultiProcEnv(Env):
    """
    Parallel environments via multiprocessing + shared memory
    """
    def __init__(self, envs):
        super().__init__(envs[0].id)
        self.shm = [make_shared(len(envs), s) for s in envs[0].obs_spec().spaces]
        self.shm.append(make_shared(len(envs), Space((1,), name="reward")))
        self.shm.append(make_shared(len(envs), Space((1,), name="done")))
        self.envs = [ShmProcEnv(env, idx, self.shm) for idx, env in enumerate(envs)]

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
        self.wait()

        obs = self.shm[:-2]
        reward = np.squeeze(self.shm[-2], axis=-1)
        done = np.squeeze(self.shm[-1], axis=-1)

        return obs, reward, done

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


def make_shared(n_envs, obs_space):
    shape = (n_envs, ) + obs_space.shape
    raw = RawArray(to_ctype(obs_space.dtype), int(np.prod(shape)))
    return np.frombuffer(raw, dtype=obs_space.dtype).reshape(shape)


def to_ctype(_type):
    types = {
        np.bool: ctypes.c_bool,
        np.int8: ctypes.c_byte,
        np.uint8: ctypes.c_ubyte,
        np.int32: ctypes.c_int32,
        np.int64: ctypes.c_longlong,
        np.uint64: ctypes.c_ulonglong,
        np.float32: ctypes.c_float,
        np.float64: ctypes.c_double,
    }
    if isinstance(_type, np.dtype):
        _type = _type.type
    return types[_type]

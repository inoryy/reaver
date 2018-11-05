import numpy as np
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env.environment import StepType
from .abc_env import Env, Spec, Space


class SC2Env(Env):
    def __init__(self, map_name='MoveToBeacon', spatial_dim=16, step_mul=8, render=False, obs_features=None):
        """

        :param map_name:
        :param spatial_dim:
        :param step_mul:
        :param render:
        :param obs_features: observation features to use (e.g. return via step)
        """
        self._env = None
        self.kwargs = dict(
            map_name=map_name,
            visualize=render,
            agent_interface_format=[features.parse_agent_interface_format(
                feature_screen=spatial_dim,
                feature_minimap=spatial_dim,
                rgb_screen=None,
                rgb_minimap=None
            )],
            step_mul=step_mul,
        )
        self.act_wrapper = ActionWrapper()
        self.obs_wrapper = ObservationWrapper(obs_features)

    def start(self):
        # importing here to lazy-load
        from pysc2.env import sc2_env
        self._env = sc2_env.SC2Env(**self.kwargs)

    def step(self, action):
        obs, reward, done = self.obs_wrapper(self._env.step(self.act_wrapper(action)))
        if done:
            obs, *_ = self.reset()
        return obs, reward, done

    def reset(self):
        return self.obs_wrapper(self._env.reset())

    def stop(self):
        self._env.close()

    def obs_spec(self):
        if not self.obs_wrapper.spec:
            self.make_specs()
        return self.obs_wrapper.spec

    def act_spec(self):
        if not self.act_wrapper.spec:
            self.make_specs()
        return self.act_wrapper.spec

    def make_specs(self):
        # importing here to lazy-load
        from pysc2.env import mock_sc2_env
        mock_env = mock_sc2_env.SC2TestEnv(**self.kwargs)
        self.act_wrapper.make_spec(mock_env.action_spec())
        self.obs_wrapper.make_spec(mock_env.observation_spec())
        mock_env.close()


class ObservationWrapper:
    def __init__(self, _features=None):
        self.spec = None
        if not _features:
            _features = {
                'screen': ['player_relative'],
                'minimap': ['player_relative'],
                'non-spatial': ['player', 'available_actions']}
        self.features = _features
        self.feature_masks = {
            'screen': [i for i, f in enumerate(features.SCREEN_FEATURES._fields) if f in _features['screen']],
            'minimap': [i for i, f in enumerate(features.MINIMAP_FEATURES._fields) if f in _features['minimap']],}

    def __call__(self, timestep):
        ts = timestep[0]
        obs, reward, done = ts.observation, ts.reward, ts.step_type == StepType.LAST

        obs_wrapped = [
            obs['feature_screen'][self.feature_masks['screen']],
            obs['feature_minimap'][self.feature_masks['minimap']]
        ]
        for feat_name in self.features['non-spatial']:
            if feat_name == 'available_actions':
                mask = np.zeros((len(actions.FUNCTIONS),), dtype=np.int32)
                mask[obs[feat_name]] = 1
                obs[feat_name] = mask
            obs_wrapped.append(obs[feat_name])

        return obs_wrapped, reward, done

    def make_spec(self, spec):
        spec = spec[0]

        default_dims = {
            'available_actions': (len(actions.FUNCTIONS), ),
        }

        screen, screen_dims = self.features['screen'], spec['feature_screen'][1:]
        minimap, minimap_dims = self.features['minimap'], spec['feature_minimap'][1:]

        spaces = [
            Space((len(screen), *screen_dims), np.int32, 'screen: ' + ','.join(screen)),
            Space((len(minimap), *minimap_dims), np.int32, 'minimap: ' + ','.join(minimap)),
        ]

        for feat in self.features['non-spatial']:
            if 0 in spec[feat]:
                spec[feat] = default_dims[feat]
            spaces.append(Space(spec[feat], np.int32, feat))

        self.spec = Spec(spaces, 'Observation')


class ActionWrapper:
    def __init__(self, args=None):
        self.spec = None
        if not args:
            args = [
                'screen',
                'minimap',
                'screen2',
                'queued',
                'control_group_act',
                'control_group_id',
                'select_add',
                'select_point_act',
                'select_add',
                'select_unit_act',
                # 'select_unit_id'
                'select_worker',
                'build_queue_id',
                # 'unload_id'
            ]
        self.args = args

    def __call__(self, action):
        defaults = {
            'select_unit_id': 0,
            'unload_id': 0,
        }

        fn_id, args = action.pop(0), []
        for arg in actions.FUNCTIONS[fn_id].args:
            arg_name = arg.name
            if arg_name in self.args:
                args.append(action[self.args.index(arg_name)])
            else:
                args.append(defaults[arg_name])

        return [actions.FunctionCall(fn_id, args)]

    def make_spec(self, spec):
        spec = spec[0]

        spaces = [Space(len(spec.functions), np.int32, "function_id")]
        for arg_name in self.args:
            arg = getattr(spec.types, arg_name)
            spaces.append(Space(arg.sizes, np.int32, arg_name))

        self.spec = Spec(spaces, "Action")


def make_envs(args):
    return [SC2Env(args.map, args.spatial_dim, args.step_mul, i == 0 and args.render) for i in range(args.n_envs)]

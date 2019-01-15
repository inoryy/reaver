import os

SC2_MINIGAMES_ALIASES = {
    'beacon': 'MoveToBeacon',
    'shards': 'CollectMineralShards',
    'roaches': 'DefeatRoaches',
    'blings': 'DefeatZerglingsAndBanelings',
    'lings': 'FindAndDefeatZerglings',
    'minerals': 'CollectMineralsAndGas',
    'marines': 'BuildMarines',
}

GYM_CONTINUOUS = [
    'LunarLanderContinuous-v2',
    'BipedalWalker-v2',
    'CarRacing-v0',
    'MountainCarContinuous-v0',
    'Pendulum-v0',
    'Acrobot-v1'
]

MUJOCO_ENVS = [
    'Reacher-v2',
    'Pusher-v2',
    'Thrower-v2',
    'Striker-v2',
    'InvertedPendulum-v2',
    'InvertedDoublePendulum-v2',
    'HalfCheetah-v2',
    'Hopper-v2',
    'Swimmer-v2',
    'Walker2d-v2',
    'Ant-v2',
    'Humanoid-v2',
    'HumanoidStandup-v2'
]

ATARI_GAMES = list(map(lambda name: ''.join([g.capitalize() for g in name.split('_')]), [
    'air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
    'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout',
    'carnival', 'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk',
    'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
    'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
    'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
    'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
    'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
    'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']))

ATARI_ENVS = ['%s-v0' % name for name in ATARI_GAMES]
ATARI_ENVS += ['%s-v4' % name for name in ATARI_GAMES]
ATARI_ENVS += ['%sDeterministic-v0' % name for name in ATARI_GAMES]
ATARI_ENVS += ['%sDeterministic-v4' % name for name in ATARI_GAMES]
ATARI_ENVS += ['%sNoFrameskip-v0' % name for name in ATARI_GAMES]
ATARI_ENVS += ['%sNoFrameskip-v4' % name for name in ATARI_GAMES]


def find_configs(env_name, base_path=''):
    if '-v' not in env_name:
        return filter_exists(['sc2/base.gin', 'sc2/%s.gin' % env_name], base_path)

    if env_name in GYM_CONTINUOUS:
        return filter_exists(['gym/base.gin', 'gym/continuous.gin', 'gym/%s.gin' % env_name], base_path)

    if env_name in MUJOCO_ENVS:
        return filter_exists(['mujoco/base.gin', 'mujoco/%s.gin' % env_name], base_path)

    if env_name in ATARI_ENVS:
        return filter_exists(['atari/base.gin', 'atari/%s.gin' % env_name], base_path)

    return filter_exists(['gym/base.gin', 'gym/%s.gin' % env_name], base_path)


def filter_exists(filenames, base_path):
    full_paths = [os.path.join(base_path, 'configs', fl) for fl in filenames]
    full_paths = [fl for fl in full_paths if os.path.exists(fl)]
    return full_paths

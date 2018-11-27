import os
import gin
from datetime import datetime as dt


class Experiment:
    def __init__(self, results_dir, env_name, agent_name, name=None, restore=False):
        if not name:
            if restore:
                experiments = [e for e in os.listdir(results_dir) if env_name in e and agent_name in e]
                assert len(experiments) > 0, 'No experiment to restore'
                name = max(experiments, key=lambda p: os.path.getmtime(results_dir+'/'+p))
                name = '_'.join(name.split('_')[2:])
            else:
                name = dt.now().strftime("%y-%m-%d_%H-%M")

        self.name = name
        self.restore = restore
        self.env_name = env_name
        self.agent_name = agent_name
        self.results_dir = results_dir

        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.path + '/summaries', exist_ok=True)
        os.makedirs(self.results_dir + '/summaries', exist_ok=True)
        if not os.path.exists(self.summaries_path):
            os.symlink('../%s/summaries' % self.full_name, self.summaries_path)

    @property
    def full_name(self):
        return '%s_%s_%s' % (self.env_name, self.agent_name, self.name)

    @property
    def path(self):
        return '%s/%s' % (self.results_dir, self.full_name)

    @property
    def config_path(self):
        return '%s/%s' % (self.path, 'config.gin')

    @property
    def log_path(self):
        return '%s/%s' % (self.path, 'train.log')

    @property
    def checkpoints_path(self):
        return self.path + '/checkpoints'

    @property
    def summaries_path(self):
        return '%s/summaries/%s' % (self.results_dir, self.full_name)

    def save_gin_config(self, full_agent_name='AdvantageActorCriticAgent'):
        config_str = gin.operative_config_str()

        if full_agent_name + '.batch_sz' not in config_str:
            # gin ignores batch size since it's passed manually from args
            # as a hacky workaround - insert it manually as the first param
            batch_sz = gin.query_parameter(full_agent_name + '.batch_sz')
            config_lines = config_str.split('\n')
            first_ac_line = 0
            for first_ac_line in range(0, len(config_lines)):
                if full_agent_name + '.' in config_lines[first_ac_line]:
                    break
            config_lines.insert(first_ac_line, full_agent_name + '.batch_sz = ' + str(batch_sz))
            config_str = '\n'.join(config_lines)

        with open(self.config_path, 'w') as cfg_file:
            cfg_file.write(config_str)

    def save_model_summary(self, model):
        with open(self.path + '/' + 'model_summary.txt', 'w') as fl:
            model.summary(print_fn=lambda line: print(line, file=fl))

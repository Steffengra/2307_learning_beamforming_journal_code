
from pathlib import Path
from sys import path as sys_path
project_root_path = Path(Path(__file__).parent, '..', '..')
sys_path.append(str(project_root_path.resolve()))

from src.config.config import Config
from src.models.train_sac import train_sac


def train_sac_vanilla():

    config = Config()

    # remove low frequency network updates
    config.config_learner.train_policy_every_k_steps = 1
    config.config_learner.train_value_every_k_steps = 1

    # remove input standardization
    config.config_learner.get_state_args['norm_state'] = False

    # remove batch norm
    config.config_learner.network_args['value_network_args']['batch_norm'] = False
    config.config_learner.network_args['policy_network_args']['batch_norm'] = False

    # remove weight penalty
    config.config_learner.training_args['training_l2_norm_scale_value'] = 0
    config.config_learner.training_args['training_l2_norm_scale_policy'] = 0

    name = config.generate_name_from_config()
    name += '_vanilla_sac'
    config.config_learner.training_name = name

    train_sac(config)


if __name__ == '__main__':
    train_sac_vanilla()

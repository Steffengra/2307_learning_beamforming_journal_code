
from pathlib import Path
import importlib.util
import sys

import src


def compare_configs(
        config_current: 'src.config.config.Config',
        path_config_old: Path,
        log_differences: bool = True,
) -> bool:
    """
    Compares values between one config in memory and one config on disk.
    Returns False if a change was detected, else True if perfect match
    """

    def recursive_value_comparison(dict1, dict2, logger, log_differences, parent=''):

        identical_flag = True

        for key1, key2 in zip(dict1, dict2):
            value_type = type(dict1[key1])
            if value_type is dict:
                if not recursive_value_comparison(dict1[key1], dict2[key1], logger, log_differences, parent=f'{parent}{key1} > '):
                    identical_flag = False
            elif value_type in [bool, int, float]:
                if dict1[key1] != dict2[key2]:
                    identical_flag = False
                    if log_differences:
                        logger.error(f'config mismatch - {parent}{key1} current: {dict1[key1]}, old: {dict2[key2]}')

        return identical_flag

    logger = config_current.logger.getChild(__name__)

    identical_config_flag = True

    # import from path
    spec_config = importlib.util.spec_from_file_location("config_old", Path(path_config_old, 'config.py'))
    spec_config_error_model = importlib.util.spec_from_file_location("config_old", Path(path_config_old, 'config_error_model.py'))
    module_config = importlib.util.module_from_spec(spec_config)
    module_config_error_model = importlib.util.module_from_spec(spec_config_error_model)
    sys.modules["config_old"] = module_config
    sys.modules["config_error_model_old"] = module_config_error_model
    spec_config.loader.exec_module(module_config)
    spec_config_error_model.loader.exec_module(module_config_error_model)
    config_old = module_config.Config()
    config_error_model_old = module_config_error_model.ConfigErrorModel(
        config_old.channel_model,
        config_old.rng,
        config_old.wavelength,
        config_old.user_nr,
    )

    vars_current = vars(config_current)
    vars_old = vars(config_old)
    if recursive_value_comparison(vars_current, vars_old, logger, log_differences) is False:
        identical_config_flag = False

    vars_error_current = vars(config_current.config_error_model)
    vars_error_old = vars(config_error_model_old)
    if recursive_value_comparison(vars_error_current, vars_error_old, logger, log_differences) is False:
        identical_config_flag = False

    return identical_config_flag

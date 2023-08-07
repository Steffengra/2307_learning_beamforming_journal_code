
import numpy as np

from src.config.config import Config
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.utils.update_sim import update_sim
from src.models.helpers.get_state import (
    get_state_erroneous_channel_state_information,
    get_state_aods,
)


def get_state_norm_factors(
    config: Config,
    satellite_manager: SatelliteManager,
    user_manager: UserManager,
) -> dict:

    norm_dict = {
        'get_state_method': '',
        'get_state_args': config.config_learner.get_state_args,
        'norm_factors': {},
    }
    if not config.config_learner.get_state_args['norm_state']:
        return norm_dict

    get_state_args = config.config_learner.get_state_args.copy()
    get_state_args['norm_state'] = False

    update_sim(config=config, satellite_manager=satellite_manager, user_manager=user_manager)

    states = []

    for _ in range(config.config_learner.get_state_norm_factors_iterations):

        state = config.config_learner.get_state(satellite_manager=satellite_manager, **get_state_args)
        states.append(state)
        update_sim(config=config, satellite_manager=satellite_manager, user_manager=user_manager)

    if config.config_learner.get_state == get_state_erroneous_channel_state_information:

        norm_dict['get_state_method'] = 'erroneous_csi'

        if get_state_args['csi_format'] == 'rad_phase':

            states_radius = np.array([state[:int(len(state)/2)] for state in states]).flatten()
            states_phase = np.array([state[int(len(state)/2):] for state in states]).flatten()

            norm_dict['norm_factors']['radius_mean'] = np.mean(states_radius)
            norm_dict['norm_factors']['radius_std'] = np.std(states_radius)
            norm_dict['norm_factors']['phase_mean'] = np.mean(states_phase)
            norm_dict['norm_factors']['phase_std'] = np.std(states_phase)
            # note: statistical analysis has shown that the means, especially of phase,
            #  take a lot of iterations to determine with confidence. Hence, we might only use std for norm.

        elif get_state_args['csi_format'] == 'real_imag':

            states_real_imag = np.array(states).flatten()

            norm_dict['norm_factors']['mean'] = np.mean(states_real_imag)
            norm_dict['norm_factors']['std'] = np.std(states_real_imag)

        else:

            raise ValueError('unknown csi_format')

    elif config.config_learner.get_state == get_state_aods:

        norm_dict['get_state_method'] = 'aods'

        states_aods = np.array(states).flatten()

        norm_dict['norm_factors']['mean'] = np.mean(states_aods)
        norm_dict['norm_factors']['std'] = np.std(states_aods)

    else:

        raise ValueError('unknown get_state function')

    return norm_dict
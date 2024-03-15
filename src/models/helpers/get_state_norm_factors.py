
import numpy as np

import src
from src.utils.update_sim import update_sim
from src.models.helpers.get_state import (
    get_state_erroneous_channel_state_information,
    get_state_aods,
)


def get_state_norm_factors(
        config: 'src.config.config.Config',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        user_manager: 'src.data.user_manager.UserManager',
        per_sat: bool = False,
) -> dict[str: str, str: dict, str: dict]:

    """
    Determines normalization factors for a given get_state method heuristically by sampling
        according to config.
    """

    def method_rad_phase(states):
        states_radius = np.array([state[:int(len(state) / 2)] for state in states]).flatten()
        states_phase = np.array([state[int(len(state) / 2):] for state in states]).flatten()

        radius_mean = np.mean(states_radius)
        radius_std = np.std(states_radius)
        phase_mean = np.mean(states_phase)
        phase_std = np.std(states_phase)
        # note: statistical analysis has shown that the means, especially of phase,
        #  take a lot of iterations to determine with confidence. Hence, we might only use std for norm.
        return radius_mean, radius_std, phase_mean, phase_std

    def method_rad_phase_reduced(states):
        num_users = satellite_manager.satellites[0].user_nr
        num_satellites = len(satellite_manager.satellites)
        states_radius = np.array([state[:num_users * num_satellites] for state in states]).flatten()
        states_phase = np.array([state[num_users * num_satellites:] for state in states]).flatten()

        radius_mean = np.mean(states_radius)
        radius_std = np.std(states_radius)
        phase_mean = np.mean(states_phase)
        phase_std = np.std(states_phase)

        return radius_mean, radius_std, phase_mean, phase_std

    def method_real_imag(states):
        states_real_imag = np.array(states).flatten()

        mean = np.mean(states_real_imag)
        std = np.std(states_real_imag)

        return mean, std


    # define default norm_dict
    norm_dict: dict = {
        'get_state_method': str(config.config_learner.get_state),
        'get_state_args': config.config_learner.get_state_args,
        'norm_factors': {},
    }

    # if no norm, don't determine norm factors
    if not config.config_learner.get_state_args['norm_state']:
        return norm_dict

    # set get_state norm argument to false for the sampling process
    get_state_args = config.config_learner.get_state_args.copy()
    get_state_args['norm_state'] = False

    if per_sat:
        get_state_args['per_sat'] = True

    update_sim(config=config, satellite_manager=satellite_manager, user_manager=user_manager)

    # gather state samples
    states = []
    for _ in range(config.config_learner.get_state_norm_factors_iterations):

        state = config.config_learner.get_state(satellite_manager=satellite_manager, **get_state_args)
        states.append(state)
        update_sim(config=config, satellite_manager=satellite_manager, user_manager=user_manager)

    if per_sat:
        sat_norm_dicts = []

        for satellite_id in range(len(satellite_manager.satellites)):
            sat_states = [state[satellite_id] for state in states]

            if get_state_args['csi_format'] == 'rad_phase':
                radius_mean, radius_std, phase_mean, phase_std = method_rad_phase(sat_states)
                sat_norm_dict = {
                    'radius_mean': radius_mean,
                    'radius_std': radius_std,
                    'phase_mean': phase_mean,
                    'phase_std': phase_std,
                }

            elif get_state_args['csi_format'] == 'rad_phase_reduced':
                radius_mean, radius_std, phase_mean, phase_std = method_rad_phase_reduced(sat_states)
                sat_norm_dict = {
                    'radius_mean': radius_mean,
                    'radius_std': radius_std,
                    'phase_mean': phase_mean,
                    'phase_std': phase_std,
                }

            elif get_state_args['csi_format'] == 'real_imag':
                mean, std = method_real_imag(sat_states)
                sat_norm_dict = {
                    'mean': mean,
                    'std': std,
                }

            else:
                raise ValueError('unknown csi_format')

            sat_norm_dicts.append(sat_norm_dict)

        norm_dict['norm_factors'] = sat_norm_dicts

        # todo: missing aods for decentralized

    else:

        # determine norm factors according to get_state method
        if config.config_learner.get_state == get_state_erroneous_channel_state_information:

            if get_state_args['csi_format'] == 'rad_phase':
                radius_mean, radius_std, phase_mean, phase_std = method_rad_phase(states)

                norm_dict['norm_factors']['radius_mean'] = radius_mean
                norm_dict['norm_factors']['radius_std'] = radius_std
                norm_dict['norm_factors']['phase_mean'] = phase_mean
                norm_dict['norm_factors']['phase_std'] = phase_std

            elif get_state_args['csi_format'] == 'rad_phase_reduced':
                radius_mean, radius_std, phase_mean, phase_std = method_rad_phase_reduced(states)

                norm_dict['norm_factors']['radius_mean'] = radius_mean
                norm_dict['norm_factors']['radius_std'] = radius_std
                norm_dict['norm_factors']['phase_mean'] = phase_mean
                norm_dict['norm_factors']['phase_std'] = phase_std

            elif get_state_args['csi_format'] == 'real_imag':

                mean, std = method_real_imag(states)

                norm_dict['norm_factors']['mean'] = mean
                norm_dict['norm_factors']['std'] = std

            else:

                raise ValueError('unknown csi_format')

        elif config.config_learner.get_state == get_state_aods:

            states_aods = np.array(states).flatten()

            norm_dict['norm_factors']['mean'] = np.mean(states_aods)
            norm_dict['norm_factors']['std'] = np.std(states_aods)

        else:

            raise ValueError('unknown get_state function')

    return norm_dict

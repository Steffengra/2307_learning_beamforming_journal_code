
import numpy as np
import src


def los_channel_error_model_in_sat2user_dist(
        error_model_config: 'src.config.config_error_model.ConfigErrorModel',
        satellite: 'src.data.satellite.Satellite',
        users: list,
) -> np.ndarray:

    """
    This error model calculates an erroneous channel state information estimate based on a
    perturbed sat2user distance estimate.
    """

    erroneous_channel_state_to_users = np.zeros((len(users), satellite.antenna_nr), dtype='complex128')

    for user in users:

        # Perturb the satellite-to-user distance according to config
        satellite_to_user_distance_estimate = (
                satellite.distance_to_users[user.idx]
                * satellite.rng.normal(loc=1, scale=error_model_config.distance_error_std)
        )

        # Calculate channel state estimation based on the perturbed distance estimate
        power_ratio = (
                satellite.antenna_gain_linear
                * user.gain_linear
                * (satellite.wavelength / (4 * np.pi * satellite_to_user_distance_estimate)) ** 2
        )
        amplitude_damping = np.sqrt(power_ratio)

        phase_shift = satellite_to_user_distance_estimate % satellite.wavelength * 2 * np.pi / satellite.wavelength

        erroneous_channel_state_to_users[user.idx, :] = (
            amplitude_damping
            * np.exp(1j * phase_shift)
            * satellite.steering_vectors_to_users[user.idx]
        )

    return erroneous_channel_state_to_users

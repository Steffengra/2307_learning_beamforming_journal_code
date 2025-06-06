
import numpy as np
import src

from src.data.channel.get_steering_vec import get_steering_vec


def los_channel_model(
        satellite: 'src.data.satellite.Satellite',
        users: list,
        scale: float = 0,
) -> np.ndarray:

    """
    The los channel model calculates complex csi for one satellite to all users from
        1) amplitude dampening based on sat gain, user gain, and freq-dependent distance gain
        2) phase shift by freq-dependent distance
        3) phase shift by satellite steering vectors
    TODO: describe this - correct? reference?
    """

    # scale
    errors = {
        'large_scale_fading': satellite.estimation_errors['large_scale_fading'],  # don't scale this one
        'additive_error_on_overall_phase_shift': scale * satellite.estimation_errors['additive_error_on_overall_phase_shift'],
        'additive_error_on_aod': scale * satellite.estimation_errors['additive_error_on_aod'],
        'additive_error_on_cosine_of_aod': scale * satellite.estimation_errors['additive_error_on_cosine_of_aod'],
        'additive_error_on_channel_vector': scale * satellite.estimation_errors['additive_error_on_channel_vector'],
    }

    channel_state_information = np.zeros((len(users), satellite.antenna_nr), dtype='complex128')

    for user in users:

        if user.enabled:

            power_ratio = (
                satellite.antenna_gain_linear
                * user.gain_linear
                * (satellite.wavelength / (4 * np.pi * satellite.distance_to_users[user.idx])) ** 2
            )
            power_ratio_faded = power_ratio / errors['large_scale_fading'][user.idx]
            amplitude_damping = np.sqrt(power_ratio_faded)

            phase_shift = satellite.distance_to_users[user.idx] % satellite.wavelength * 2 * np.pi / satellite.wavelength
            phase_shift_error = errors['additive_error_on_overall_phase_shift'][user.idx]

            phase_aod_steering = (
                np.cos(
                    satellite.aods_to_users[user.idx]
                    + errors['additive_error_on_aod'][user.idx]
                )
                + errors['additive_error_on_cosine_of_aod'][user.idx]
            )

            steering_vector_to_user = get_steering_vec(
                satellite,
                phase_aod_steering,
            )

            # calculate csi for user
            constant_factor = amplitude_damping * np.exp(-1j * (phase_shift + phase_shift_error))
            channel_state_information[user.idx, :] = (
                constant_factor
                * steering_vector_to_user
            )
            channel_state_information[user.idx, :] = channel_state_information[user.idx, :] + errors['additive_error_on_channel_vector'][user.idx]

        else:

            channel_state_information[user.idx, :] = np.zeros(satellite.antenna_nr, dtype='complex128')

    return channel_state_information

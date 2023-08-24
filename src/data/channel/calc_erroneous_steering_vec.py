
import numpy as np

import src


def calc_erroneous_steering_vec(
        error_model_config: 'src.config.config_error_model.ConfigErrorModel',
        satellite: 'src.data.satellite.Satellite',
        users: list,
) -> np.ndarray:

    # Testversion

    # calculate indices for steering vectors
    steering_idx = np.arange(0, satellite.antenna_nr) - (satellite.antenna_nr - 1) / 2

    # same error for all antennas for same user, different error for different users
    satellite.steering_error = np.exp(
        steering_idx * (
            1j * 2 * np.pi / satellite.wavelength
            * satellite.antenna_distance
            * satellite.rng.uniform(low=error_model_config.uniform_error_interval['low'],
                                    high=error_model_config.uniform_error_interval['high'],
                                    size=(len(users), 1))
        )
    )

    return satellite.steering_error


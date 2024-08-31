
import numpy as np

import src


def get_steering_vec(
        satellite: 'src.data.satellite.Satellite',
        phase_aod_steering: float,
) -> np.ndarray:
    """
    Compared to the center of an antenna array, each individual antenna has a slightly
      longer/shorter distance to target. Steering vector gives the additional phase rotation introduced
      by this extra distance.
    """

    steering_idx = np.arange(0, satellite.antenna_nr, dtype='complex128') - (satellite.antenna_nr - 1) / 2  # todo
    # steering_idx = np.arange(0, satellite.antenna_nr, dtype='complex128')

    steering_vector_to_user = np.exp(
        -1j * 2 * np.pi / satellite.wavelength
        * satellite.antenna_distance
        * phase_aod_steering
        * steering_idx
    )

    return steering_vector_to_user

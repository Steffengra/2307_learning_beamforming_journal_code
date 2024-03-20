
import numpy as np

from src.utils.norm_precoder import (
    norm_precoder,
)


def mmse_precoder_decentral_limited_no_norm(
        local_channel_matrices: list[np.ndarray],
        noise_power_watt: float,
        power_constraint_watt: float,
        sat_nr: int,
        sat_ant_nr: int,
) -> np.ndarray:
    """TODO: Comment"""

    user_nr = local_channel_matrices[0].shape[0]
    sat_tot_ant_nr = local_channel_matrices[0].shape[1]

    # inversion_constant_lambda = finfo('float32').tiny
    inversion_constant_lambda = 0

    precoding_matrix = np.zeros((sat_nr * sat_ant_nr, user_nr), dtype='complex128')
    for satellite_idx, local_channel_matrix in enumerate(local_channel_matrices):

        channel_matrix_own = local_channel_matrix[:, satellite_idx * sat_ant_nr:satellite_idx * sat_ant_nr + sat_ant_nr]

        precoding_matrix_own = (
            np.matmul(
                channel_matrix_own.conj().T
                ,
                np.linalg.inv(
                    np.matmul(local_channel_matrix, local_channel_matrix.conj().T)
                    + (
                        noise_power_watt
                        * user_nr
                        / (power_constraint_watt / sat_nr)
                        + inversion_constant_lambda
                    ) * np.eye(user_nr)
                )
            )
        )

        precoding_matrix[satellite_idx * sat_ant_nr:satellite_idx*sat_ant_nr + sat_ant_nr, :] = precoding_matrix_own.copy()

    return precoding_matrix


def mmse_precoder_decentral_limited_normalized(
        local_channel_matrices: list[np.ndarray],
        noise_power_watt: float,
        power_constraint_watt: float,
        sat_nr: int,
        sat_ant_nr: int,
) -> np.ndarray:

    precoding_matrix = mmse_precoder_decentral_limited_no_norm(
        local_channel_matrices,
        noise_power_watt,
        power_constraint_watt,
        sat_nr,
        sat_ant_nr,
    )

    precoding_matrix_normed = norm_precoder(
        precoding_matrix=precoding_matrix,
        power_constraint_watt=power_constraint_watt,
        per_satellite=True,
        sat_nr=sat_nr,
        sat_ant_nr=sat_ant_nr,
    )

    return precoding_matrix_normed

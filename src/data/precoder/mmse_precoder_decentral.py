
import numpy as np

from src.utils.norm_precoder import norm_precoder
from src.data.precoder.mmse_precoder import mmse_precoder_no_norm


def mmse_precoder_decentral_blind_no_norm(
        erroneous_csit_per_sat: list[np.ndarray],
        noise_power_watt: float,
        power_constraint_watt: float,
        sat_nr: int,
        sat_ant_nr: int,
) -> np.ndarray:
    """
    Calculate small precoder for each satellite based only on small local CSI matrix,
    assemble into full precoding matrix.
    """

    user_nr = erroneous_csit_per_sat[0].shape[0]

    w_precoder = np.zeros((sat_nr * sat_ant_nr, user_nr), dtype='complex128')

    for sat_id, sat_csit in enumerate(erroneous_csit_per_sat):
        w_mmse_sat = mmse_precoder_no_norm(
            sat_csit,
            noise_power_watt=noise_power_watt,  # todo: teilen durch sat_nr? mutliplizieren weil receiver?
            power_constraint_watt=power_constraint_watt / sat_nr,
        )

        w_precoder[sat_id * sat_ant_nr:sat_id * sat_nr + sat_ant_nr, :] = w_mmse_sat

    return w_precoder


def mmse_precoder_decentral_blind_normed(
        erroneous_csit_per_sat: list[np.ndarray],
        noise_power_watt: float,
        power_constraint_watt: float,
        sat_nr: int,
        sat_ant_nr: int,
) -> np.ndarray:

    w_precoder_no_norm = mmse_precoder_decentral_blind_no_norm(
        erroneous_csit_per_sat=erroneous_csit_per_sat,
        noise_power_watt=noise_power_watt,
        power_constraint_watt=power_constraint_watt,
        sat_nr=sat_nr,
        sat_ant_nr=sat_ant_nr,
    )

    w_precoder = norm_precoder(
        w_precoder_no_norm,
        power_constraint_watt=power_constraint_watt,
        per_satellite=True,
        sat_nr=sat_nr,
        sat_ant_nr=sat_ant_nr,
    )

    return w_precoder


def mmse_precoder_decentral_limited_no_norm(
        local_channel_matrices: list[np.ndarray],
        noise_power_watt: float,
        power_constraint_watt: float,
        sat_nr: int,
        sat_ant_nr: int,
) -> np.ndarray:
    """
    Calculate MMSE precoding given own channel knowledge and channel knowledge from other satellites.
    Other satellites' channel knowledge is potentially more erroneous, e.g., outdated.
    """

    user_nr = local_channel_matrices[0].shape[0]
    # sat_tot_ant_nr = local_channel_matrices[0].shape[1]

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
                        / power_constraint_watt  # todo: macht gesamtpower hier sinn?
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

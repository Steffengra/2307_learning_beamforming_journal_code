
import numpy as np

from src.utils.norm_precoder import (
    norm_precoder,
)
from src.data.precoder.mmse_precoder import (
    mmse_precoder_no_norm,
)


# WIP

def rate_splitting_test_no_norm(
        channel_matrix,
        noise_power_watt: float,
        power_constraint_watt: float,
        rsma_factor: float,
) -> np.ndarray:
    

    user_nr = channel_matrix.shape[0]
    sat_tot_ant_nr = channel_matrix.shape[1]

    power_constraint_private_part = power_constraint_watt**rsma_factor
    power_constraint_common_part =  power_constraint_watt - power_constraint_private_part

    # common part precoding

    precoding_vector_common = np.sqrt(power_constraint_common_part/sat_tot_ant_nr) * np.ones((sat_tot_ant_nr,1))

    # private part precoding

    precoding_matrix_private_no_norm = mmse_precoder_no_norm(
        channel_matrix=channel_matrix,
        noise_power_watt=noise_power_watt,
        power_constraint_watt=power_constraint_private_part,
    )
    
    norm_factor = np.sqrt(power_constraint_private_part / np.trace(np.matmul(precoding_matrix_private_no_norm.conj().T, precoding_matrix_private_no_norm)))
    precoding_matrix_private = norm_factor * precoding_matrix_private_no_norm
    precoding_matrix = np.concatenate([precoding_vector_common, precoding_matrix_private], axis=1)
    power_precoding = np.trace(precoding_matrix.conj().T @ precoding_matrix)

    return precoding_matrix

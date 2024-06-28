
import numpy as np

from src.utils.norm_precoder import (
    norm_precoder,
)
from src.data.precoder.mmse_precoder import (
    mmse_precoder_no_norm,
)


def rate_splitting_no_norm(
        channel_matrix,
        noise_power_watt: float,
        power_constraint_watt: float,
        rsma_factor: float,
        common_part_precoding_style:str = 'MRT' 
) -> np.ndarray:
    """
    common_part_precoding_style
        'basic' = all ones
        'MRT' = overlap MRT
    """

    user_nr = channel_matrix.shape[0]
    sat_tot_ant_nr = channel_matrix.shape[1]

    power_constraint_private_part = power_constraint_watt**rsma_factor
    power_constraint_common_part =  power_constraint_watt - power_constraint_private_part

    # common part precoding

    if common_part_precoding_style == 'basic':

        precoding_vector_common = np.sqrt(power_constraint_common_part/sat_tot_ant_nr) * np.ones((sat_tot_ant_nr,1))

    elif common_part_precoding_style == 'MRT':

        w_mrc = np.empty((sat_tot_ant_nr, user_nr), dtype='complex128')

        for user_id in range(user_nr):

            H_k = channel_matrix[user_id, :]

            w = (1 / np.linalg.norm(H_k)) * H_k.conj().T * np.sqrt(power_constraint_watt)

            w_mrc[:, user_id] = w

        w_mrc_overlap = np.sum(w_mrc, axis=1)[np.newaxis].T
        # print(w_mrc_overlap)

        precoding_vector_common = w_mrc_overlap * np.sqrt(power_constraint_common_part)/np.linalg.norm(w_mrc_overlap)
    
    else: 
        raise ValueError('Caution: you have failed to set a valid common part precoding')

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
    #print(power_precoding)
    #print(power_constraint_watt)

    if power_precoding>1.0001*power_constraint_watt:
        raise ValueError('shiddeeee')
    

    return precoding_matrix


import numpy as np


def calc_sum_rate_RSMA(
    channel_state: np.ndarray,
    w_precoder: np.ndarray,
    noise_power_watt: float
) -> float:
    """TODO: comment"""

    user_nr = channel_state.shape[0]
    precoder_private = w_precoder[:, 1:]

    sinr_users = np.zeros(user_nr)
    sinr_common_part = np.zeros(user_nr)

    # calculating SINR private part
    for user_idx in range(user_nr):
        channel_user_H_k = channel_state[user_idx, :]
        power_fading_precoded_sigma_x = abs(np.matmul(channel_user_H_k, precoder_private[:, user_idx]))**2
        
        power_fading_precoded_other_users_sigma_i = [
            abs(np.matmul(channel_user_H_k, precoder_private[:, other_user_idx]))**2
            for other_user_idx in range(user_nr) if other_user_idx != user_idx
        ]

        sum_power_fading_precoded_other_users_sigma_int = sum(power_fading_precoded_other_users_sigma_i)

        sinr_users[user_idx] = (
                power_fading_precoded_sigma_x / (noise_power_watt + sum_power_fading_precoded_other_users_sigma_int)
        )

    # calculating SINR common part
    for user_idx in range(user_nr):
        channel_user_H_k = channel_state[user_idx, :]
        power_fading_precoded_sigma_x = abs(np.matmul(channel_user_H_k, w_precoder[:, 0])) ** 2

        power_interference_private_users_sigma_i = [
            abs(np.matmul(channel_user_H_k, precoder_private[:, other_user_idx]))**2
            for other_user_idx in range(user_nr) 
        ]       
        
        sum_power_interference_private_users = sum(power_interference_private_users_sigma_i)

        sinr_common_part[user_idx] = (
            power_fading_precoded_sigma_x / (noise_power_watt + sum_power_interference_private_users)
        )
    
    sinr_common_part_min = np.min(sinr_common_part)
    common_part_rate = np.log2(1 + sinr_common_part_min)

    private_part_rate = np.log2(1 + sinr_users)
    sum_rate = sum(private_part_rate) + common_part_rate


    return sum_rate


import numpy as np


def calc_jain_fairness_no_iui(
    channel_state: np.ndarray,
    w_precoder: np.ndarray,
    noise_power_watt: float,
) -> float:
    """TODO: Comment"""

    user_nr = channel_state.shape[0]

    sinr_users = np.zeros(user_nr)

    for user_id in range(user_nr):

        H_k = channel_state[user_id, :]

        sigma_x = abs(np.matmul(H_k, w_precoder[:, user_id]))**2
        sigma_int = 0

        sinr_users[user_id] = sigma_x / (noise_power_watt + sigma_int)

    info_rate = np.zeros(user_nr)

    for user_id in range(user_nr):

        info_rate[user_id] = 1 / user_nr * np.log2(1 + sinr_users[user_id])

    jain_fairness = sum(info_rates)**2/(user_nr*sum(info_rates**2))

    return jain_fairness
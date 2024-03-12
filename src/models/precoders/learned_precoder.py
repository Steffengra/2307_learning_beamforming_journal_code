
import numpy as np
import tensorflow as tf

from src.utils.norm_precoder import norm_precoder
from src.utils.real_complex_vector_reshaping import real_vector_to_half_complex_vector


def get_learned_precoder_no_norm(
        state: np.ndarray,
        precoder_network: tf.keras.Model,
        sat_nr: int,
        sat_ant_nr: int,
        user_nr: int,
) -> np.ndarray:

    w_precoder, _ = precoder_network.call(state.astype('float32')[np.newaxis])
    w_precoder = w_precoder.numpy().flatten()

    w_precoder = real_vector_to_half_complex_vector(w_precoder)
    w_precoder = w_precoder.reshape((sat_nr * sat_ant_nr, user_nr))

    return w_precoder


def get_learned_precoder_normalized(
        state: np.ndarray,
        precoder_network: tf.keras.Model,
        sat_nr: int,
        sat_ant_nr: int,
        user_nr: int,
        power_constraint_watt: float,
) -> np.ndarray:

    w_precoder_no_norm = get_learned_precoder_no_norm(
        state=state,
        precoder_network=precoder_network,
        sat_nr=sat_nr,
        sat_ant_nr=sat_ant_nr,
        user_nr=user_nr,
    )

    w_precoder_normalized = norm_precoder(
        precoding_matrix=w_precoder_no_norm,
        power_constraint_watt=power_constraint_watt,
        per_satellite=True,
        sat_nr=sat_nr,
        sat_ant_nr=sat_ant_nr
    )

    return w_precoder_normalized


def get_learned_precoder_decentralized_no_norm(
        states: list[np.ndarray],
        precoder_networks: list[tf.keras.Model],
        sat_nr: int,
        sat_ant_nr: int,
        user_nr: int,
) -> np.ndarray:

    w_precoder = np.zeros((sat_nr * sat_ant_nr, user_nr), dtype='complex128')
    for sat_id, (state, precoder_network) in enumerate(zip(states, precoder_networks)):
        w_precoder_sat, _ = precoder_network.call(state.astype('float32')[np.newaxis])
        w_precoder_sat = w_precoder_sat.numpy().flatten()
        w_precoder_sat = real_vector_to_half_complex_vector(w_precoder_sat)
        w_precoder_sat = w_precoder_sat.reshape(sat_ant_nr, user_nr)
        w_precoder[sat_id * sat_ant_nr:sat_id * sat_nr + sat_ant_nr, :] = w_precoder_sat.copy()  # todo copy necessary?

    return w_precoder


def get_learned_precoder_decentralized_normalized(
        states: list[np.ndarray],
        precoder_networks: list[tf.keras.Model],
        sat_nr: int,
        sat_ant_nr: int,
        user_nr: int,
        power_constraint_watt: float,
) -> np.ndarray:

    w_precoder_no_norm = get_learned_precoder_decentralized_no_norm(
        states=states,
        precoder_networks=precoder_networks,
        sat_nr=sat_nr,
        sat_ant_nr=sat_ant_nr,
        user_nr=user_nr,
    )

    w_precoder_normalized = norm_precoder(
        precoding_matrix=w_precoder_no_norm,
        power_constraint_watt=power_constraint_watt,
        per_satellite=True,
        sat_nr=sat_nr,
        sat_ant_nr=sat_ant_nr,
    )

    return w_precoder_normalized

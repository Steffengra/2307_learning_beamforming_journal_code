
import numpy as np
import tensorflow as tf

import src
from src.data.precoder.calc_autocorrelation import calc_autocorrelation
from src.data.precoder.robust_SLNR_precoder import robust_SLNR_precoder_no_norm
from src.utils.norm_precoder import norm_precoder


def scale_robust_slnr_complete_precoder_no_norm(
        satellite: 'src.data.satellite.Satellite',
        error_model_config: 'src.config.config_error_model.ConfigErrorModel',
        error_distribution: str,
        channel_matrix: np.ndarray,
        noise_power_watt: float,
        power_constraint_watt: float,
        scaling_network: tf.keras.Model,
        scaler_input_state: np.ndarray,
) -> np.ndarray:
    """
    Calculates robust SLNR precoder, then applies a neural network determined scaling to each precoding vector.
    """

    # calc robust slnr precoder
    autocorrelation_matrix = calc_autocorrelation(
        satellite=satellite,
        error_model_config=error_model_config,
        error_distribution=error_distribution
    )

    robust_slnr_precoding = robust_SLNR_precoder_no_norm(
        channel_matrix=channel_matrix,
        autocorrelation_matrix=autocorrelation_matrix,
        noise_power_watt=noise_power_watt,
        power_constraint_watt=power_constraint_watt
    )

    scaling_vector = scaling_network(scaler_input_state[np.newaxis])[0].numpy().flatten()
    scaling_matrix = np.diag(scaling_vector)
    scaled_precoding = np.matmul(robust_slnr_precoding, scaling_matrix)  # scale columns -> precoding vectors per user

    return scaled_precoding


def scale_robust_slnr_complete_precoder_normed(
        satellite: 'src.data.satellite.Satellite',
        error_model_config: 'src.config.config_error_model.ConfigErrorModel',
        error_distribution: str,
        channel_matrix: np.ndarray,
        noise_power_watt: float,
        power_constraint_watt: float,
        scaling_network: tf.keras.Model,
        scaler_input_state: np.ndarray,
        sat_nr,
        sat_ant_nr,
) -> np.ndarray:

    precoding_matrix = scale_robust_slnr_complete_precoder_no_norm(
        satellite=satellite,
        error_model_config=error_model_config,
        error_distribution=error_distribution,
        channel_matrix=channel_matrix,
        noise_power_watt=noise_power_watt,
        power_constraint_watt=power_constraint_watt,
        scaling_network=scaling_network,
        scaler_input_state=scaler_input_state,
    )

    precoding_matrix_normed = norm_precoder(
        precoding_matrix=precoding_matrix,
        power_constraint_watt=power_constraint_watt,
        per_satellite=True,
        sat_nr=sat_nr,
        sat_ant_nr=sat_ant_nr,
    )

    return precoding_matrix_normed

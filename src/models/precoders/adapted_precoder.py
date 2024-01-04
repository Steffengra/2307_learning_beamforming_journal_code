
import numpy as np
import tensorflow as tf

import src
from src.data.precoder.calc_autocorrelation import calc_autocorrelation
from src.data.precoder.robust_SLNR_precoder import robust_SLNR_precoder_no_norm
from src.utils.norm_precoder import norm_precoder


def adapt_robust_slnr_complete_precoder_no_norm(
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
    Calculates a precoding matrix by hadamard-multiplying an existing precoder (robust slnr) with
    a neural network determined matrix.
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

    # scale it using neural network
    scaling_vector = scaling_network(scaler_input_state[np.newaxis])[0].numpy()
    scaling_matrix = scaling_vector.reshape(robust_slnr_precoding.shape)

    scaled_robust_slnr_precoding = np.multiply(scaling_matrix, robust_slnr_precoding)

    return scaled_robust_slnr_precoding


def adapt_robust_slnr_complete_precoder_normed(
        satellite: 'src.data.satellite.Satellite',
        error_model_config: 'src.config.config_error_model.ConfigErrorModel',
        error_distribution: str,
        channel_matrix: np.ndarray,
        noise_power_watt: float,
        power_constraint_watt: float,
        scaling_network: tf.keras.Model,
        scaler_input_state: np.ndarray,
        sat_nr: int,
        sat_ant_nr: int,
) -> np.ndarray:

    adapted_precoding = adapt_robust_slnr_complete_precoder_no_norm(
        satellite=satellite,
        error_model_config=error_model_config,
        error_distribution=error_distribution,
        channel_matrix=channel_matrix,
        noise_power_watt=noise_power_watt,
        power_constraint_watt=power_constraint_watt,
        scaling_network=scaling_network,
        scaler_input_state=scaler_input_state,
    )

    normed_adapted_precoding = norm_precoder(
        precoding_matrix=adapted_precoding,
        power_constraint_watt=power_constraint_watt,
        per_satellite=True,
        sat_nr=sat_nr,
        sat_ant_nr=sat_ant_nr,
    )

    return normed_adapted_precoding

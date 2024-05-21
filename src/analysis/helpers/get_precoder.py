
import numpy as np
import tensorflow as tf

import src
from src.models.precoders.adapted_precoder import adapt_robust_slnr_complete_precoder_normed


def get_precoding_adapted_slnr_complete(
        config: 'src.config.config.Config',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        norm_factors: dict,
        scaling_network: tf.keras.models.Model,
) -> np.ndarray:

    scaler_input_state = config.config_learner.get_state(
        satellite_manager,
        norm_factors=norm_factors,
        **config.config_learner.get_state_args
    )

    precoding = adapt_robust_slnr_complete_precoder_normed(
        satellite=satellite_manager.satellites[0],
        error_model_config=config.config_error_model,
        error_distribution='uniform',
        channel_matrix=satellite_manager.erroneous_channel_state_information,
        noise_power_watt=config.noise_power_watt,
        power_constraint_watt=config.power_constraint_watt,
        scaling_network=scaling_network,
        scaler_input_state=scaler_input_state,
        sat_nr=config.sat_nr,
        sat_ant_nr=config.sat_ant_nr,
    )

    return precoding

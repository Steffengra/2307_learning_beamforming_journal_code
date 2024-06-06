
import numpy as np
import tensorflow as tf

import src
from src.models.precoders.learned_precoder import get_learned_precoder_normalized
from src.models.precoders.learned_precoder import get_learned_precoder_decentralized_normalized
from src.models.precoders.adapted_precoder import adapt_robust_slnr_complete_precoder_normed
from src.models.precoders.scaled_precoder import scale_robust_slnr_complete_precoder_normed
from src.data.precoder.mmse_precoder import mmse_precoder_normalized
from src.data.precoder.mmse_precoder_decentral import mmse_precoder_decentral_blind_normed
from src.data.precoder.mmse_precoder_decentral import mmse_precoder_decentral_limited_normalized
from src.data.precoder.mrc_precoder import mrc_precoder_normalized
from src.data.precoder.calc_autocorrelation import calc_autocorrelation
from src.data.precoder.robust_SLNR_precoder import robust_SLNR_precoder_no_norm
from src.data.precoder.rate_splitting import rate_splitting_no_norm


def get_precoding_learned(
        config: 'src.config.config.Config',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        norm_factors: dict,
        precoder_network: tf.keras.models.Model,
) -> np.ndarray:

    state = config.config_learner.get_state(
        satellite_manager=satellite_manager,
        norm_factors=norm_factors,
        **config.config_learner.get_state_args
    )

    w_precoder_normalized = get_learned_precoder_normalized(
        state=state,
        precoder_network=precoder_network,
        **config.learned_precoder_args,
    )

    return w_precoder_normalized


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


def get_precoding_adapted_slnr_powerscaled(
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

    precoding = scale_robust_slnr_complete_precoder_normed(
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


def get_precoding_learned_rsma_complete(
        config: 'src.config.config.Config',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        norm_factors: dict,
        precoder_network: tf.keras.models.Model,
) -> np.ndarray:

    state = config.config_learner.get_state(
        satellite_manager=satellite_manager,
        norm_factors=norm_factors,
        **config.config_learner.get_state_args
    )

    w_precoder_normalized = get_learned_precoder_normalized(
        state=state,
        precoder_network=precoder_network,
        sat_nr=config.sat_nr,
        sat_ant_nr=config.sat_ant_nr,
        user_nr=config.user_nr + 1,
        power_constraint_watt=config.power_constraint_watt,
    )

    return w_precoder_normalized


def get_precoding_learned_decentralized_blind(
        config: 'src.config.config.Config',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        norm_factors: dict,
        precoder_networks: list[tf.keras.models.Model],
) -> np.ndarray:

    states = config.config_learner.get_state(
        satellite_manager=satellite_manager,
        norm_factors=norm_factors,
        **config.config_learner.get_state_args,
        per_sat=True
    )

    w_precoder_normalized = get_learned_precoder_decentralized_normalized(
        states=states,
        precoder_networks=precoder_networks,
        **config.learned_precoder_args,
    )

    return w_precoder_normalized


def get_precoding_learned_decentralized_limited(
        config: 'src.config.config.Config',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        norm_factors: dict,
        precoder_networks: list[tf.keras.models.Model],
) -> np.ndarray:

    states = config.config_learner.get_state(
            satellite_manager=satellite_manager,
            norm_factors=norm_factors,
            **config.config_learner.get_state_args,
    )

    w_precoder_normalized = get_learned_precoder_decentralized_normalized(
        states=states,
        precoder_networks=precoder_networks,
        **config.learned_precoder_args,
    )

    return w_precoder_normalized


def get_precoding_mmse(
        config: 'src.config.config.Config',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
) -> np.ndarray:

    w_mmse = mmse_precoder_normalized(
        channel_matrix=satellite_manager.erroneous_channel_state_information,
        **config.mmse_args,
    )

    return w_mmse


def get_precoding_mmse_decentralized_blind(
        config: 'src.config.config.Config',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
) -> np.ndarray:

    w_mmse = mmse_precoder_decentral_blind_normed(
        erroneous_csit_per_sat=satellite_manager.get_erroneous_channel_state_information_per_sat(),
        **config.mmse_args,
    )

    return w_mmse


def get_precoding_mmse_decentralized_limited(
        config: 'src.config.config.Config',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
) -> np.ndarray:

    local_channel_matrices = [
        satellite_manager.get_local_channel_state_information(sat_idx, config.local_csi_own_quality,
                                                              config.local_csi_others_quality)
        for sat_idx in range(config.sat_nr)]

    w_mmse_normalized = mmse_precoder_decentral_limited_normalized(
        local_channel_matrices=local_channel_matrices,
        **config.mmse_args,
    )

    return w_mmse_normalized


def get_precoding_mrc(
        config: 'src.config.config.Config',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
) -> np.ndarray:

    w_mrc = mrc_precoder_normalized(
        channel_matrix=satellite_manager.erroneous_channel_state_information,
        **config.mrc_args,
    )

    return w_mrc


def get_precoding_robust_slnr(
        config: 'src.config.config.Config',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
) -> np.ndarray:

    autocorrelation = calc_autocorrelation(
        satellite=satellite_manager.satellites[0],
        error_model_config=config.config_error_model,
        error_distribution='uniform',
    )

    w_robust_slnr = robust_SLNR_precoder_no_norm(
        channel_matrix=satellite_manager.erroneous_channel_state_information,
        autocorrelation_matrix=autocorrelation,
        noise_power_watt=config.noise_power_watt,
        power_constraint_watt=config.power_constraint_watt,
    )

    return w_robust_slnr


def get_precoding_rsma(
        config: 'src.config.config.Config',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        rsma_factor: float,
        common_part_precoding_style: str,
) -> np.ndarray:

    w_rsma = rate_splitting_no_norm(
        channel_matrix=satellite_manager.erroneous_channel_state_information,
        noise_power_watt=config.noise_power_watt,
        power_constraint_watt=config.power_constraint_watt,
        rsma_factor=rsma_factor,
        common_part_precoding_style=common_part_precoding_style
    )

    return w_rsma

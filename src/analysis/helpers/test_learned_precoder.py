
from pathlib import Path

import numpy as np

import src
from src.analysis.helpers.test_precoder_error_sweep import test_precoder_error_sweep
from src.analysis.helpers.test_precoder_user_distance_sweep import test_precoder_user_distance_sweep
from src.analysis.helpers.test_rsma_precoder import test_rsma_precoder_user_distance_sweep
from src.data.calc_sum_rate import calc_sum_rate
from src.data.calc_sum_rate_RSMA import calc_sum_rate_RSMA
from src.utils.load_model import (
    load_model,
    load_models,
)
from src.utils.get_precoding import (
    get_precoding_learned,
    get_precoding_learned_decentralized_blind,
    get_precoding_learned_decentralized_limited,
    get_precoding_adapted_slnr_powerscaled,
    get_precoding_adapted_slnr_complete,
    get_precoding_learned_rsma_complete, get_precoding_learned_rsma_power_scaling,
)


def test_sac_precoder_error_sweep(
        config: 'src.config.config.Config',
        model_path: Path,
        error_sweep_parameter: str,
        error_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
) -> None:
    """Test the learned SAC precoder for a range of error configuration with monte carlo average."""

    precoder_network, norm_factors = load_model(model_path)

    test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name='learned',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, sat_man: get_precoding_learned(cfg, sat_man, norm_factors, precoder_network),
        calc_sum_rate_func=calc_sum_rate,
    )


def test_sac_precoder_user_distance_sweep(
    config: 'src.config.config.Config',
    distance_sweep_range: np.ndarray,
    model_path: Path,
) -> None:
    """Test a precoder over a range of distances with zero error."""

    precoder_network, norm_factors = load_model(model_path)

    if norm_factors != {}:
        config.config_learner.get_state_args['norm_state'] = True

    test_precoder_user_distance_sweep(
        config=config,
        distance_sweep_range=distance_sweep_range,
        precoder_name='learned',
        mode='user',
        get_precoder_func=lambda cfg, sat_man: get_precoding_learned(cfg, sat_man, norm_factors, precoder_network),
        calc_sum_rate_func=calc_sum_rate,
    )


def test_sac_precoder_decentralized_blind_error_sweep(
        config: 'src.config.config.Config',
        models_path: Path,
        error_sweep_parameter: str,
        error_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
) -> None:
    """Test decentralized learned SAC precoders for a range of error configuration with monte carlo average."""

    precoder_networks, norm_factors = load_models(models_path)

    test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name='sac_decentralized_blind',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, sat_man: get_precoding_learned_decentralized_blind(cfg, sat_man, norm_factors, precoder_networks),
        calc_sum_rate_func=calc_sum_rate,
    )


def test_sac_precoder_decentralized_limited_error_sweep(
        config: 'src.config.config.Config',
        models_path: Path,
        error_sweep_parameter: str,
        error_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
) -> None:

    precoder_networks, norm_factors = load_models(models_path)

    config.config_learner.get_state_args['local_csi_own_quality'] = config.local_csi_own_quality
    config.config_learner.get_state_args['local_csi_others_quality'] = config.local_csi_others_quality

    if config.local_csi_own_quality == 'error_free' and config.local_csi_others_quality == 'erroneous':
        precoder_name = 'sac_decentralized_limited_L1'
    elif config.local_csi_own_quality == 'erroneous' and config.local_csi_others_quality == 'scaled_erroneous':
        precoder_name = 'sac_decentralized_limited_L2'
    else:
        raise ValueError('Unknown decentralized_limited scenario')

    test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name=precoder_name,
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, sat_man: get_precoding_learned_decentralized_limited(cfg, sat_man, norm_factors, precoder_networks),
        calc_sum_rate_func=calc_sum_rate,
    )


def test_adapted_slnr_complete_error_sweep(
        config: 'src.config.config.Config',
        model_path: Path,
        error_sweep_parameter: str,
        error_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
) -> None:

    scaling_network, norm_factors = load_model(model_path)

    test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name='adapted_slnr_complete',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, sat_man: get_precoding_adapted_slnr_complete(cfg, sat_man, norm_factors, scaling_network),
        calc_sum_rate_func=calc_sum_rate,
    )


def test_adapted_slnr_powerscaled_error_sweep(
        config: 'src.config.config.Config',
        model_path: Path,
        error_sweep_parameter: str,
        error_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
) -> None:

    scaling_network, norm_factors = load_model(model_path)

    test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name='adapted_slnr_powerscaled',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, sat_man: get_precoding_adapted_slnr_powerscaled(cfg, sat_man, norm_factors, scaling_network),
        calc_sum_rate_func=calc_sum_rate,
    )


def test_learned_rsma_complete_error_sweep(
        config: 'src.config.config.Config',
        model_path: Path,
        error_sweep_parameter: str,
        error_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
) -> None:

    rsma_network, norm_factors = load_model(model_path)

    test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name='learned_rsma_full',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, sat_man: get_precoding_learned_rsma_complete(cfg, sat_man, norm_factors, rsma_network),
        calc_sum_rate_func=calc_sum_rate_RSMA,
    )

def test_learned_rsma_complete_user_distance_sweep(
        config: 'src.config.config.Config',
        distance_sweep_range: np.ndarray,
        model_path: Path,
) -> None:
    """Test a precoder over a range of distances with zero error."""

    rsma_network, norm_factors = load_model(model_path)

    if norm_factors != {}:
        config.config_learner.get_state_args['norm_state'] = True

    test_precoder_user_distance_sweep(
        config=config,
        distance_sweep_range=distance_sweep_range,
        precoder_name='learned_rsma_full',
        mode='user',
        get_precoder_func=lambda cfg, sat_man: get_precoding_learned_rsma_complete(cfg, sat_man, norm_factors, rsma_network),
        calc_sum_rate_func=calc_sum_rate_RSMA,
        )

def test_learned_rsma_power_factor_error_sweep(
        config: 'src.config.config.Config',
        model_path: Path,
        error_sweep_parameter: str,
        error_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
) -> None:

    rsma_power_factor_network, norm_factors = load_model(model_path)

    test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name='learned_rsma_power_factor',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, sat_man: get_precoding_learned_rsma_power_scaling(cfg, sat_man, norm_factors, rsma_power_factor_network),
        calc_sum_rate_func=calc_sum_rate_RSMA,
    )

def test_learned_rsma_power_factor_user_distance_sweep(
        config: 'src.config.config.Config',
        distance_sweep_range: np.ndarray,
        model_path: Path,
) -> None:
    """Test a precoder over a range of distances with zero error."""

    rsma_power_factor_network, norm_factors = load_model(model_path)

    if norm_factors != {}:
        config.config_learner.get_state_args['norm_state'] = True

    test_precoder_user_distance_sweep(
        config=config,
        distance_sweep_range=distance_sweep_range,
        precoder_name='learned_rsma_power_factor',
        mode='user',
        get_precoder_func=lambda cfg, sat_man: get_precoding_learned_rsma_power_scaling(cfg, sat_man, norm_factors,
                                                                                   rsma_power_factor_network),
        calc_sum_rate_func=calc_sum_rate_RSMA,
    )

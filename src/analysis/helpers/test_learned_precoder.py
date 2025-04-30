
from pathlib import Path

import numpy as np

import src
from src.analysis.helpers.test_precoder_error_sweep import test_precoder_error_sweep
from src.analysis.helpers.test_precoder_user_distance_sweep import test_precoder_user_distance_sweep
from src.analysis.helpers.test_precoder_user_sweep import test_precoder_user_sweep
from src.data.calc_sum_rate import calc_sum_rate
from src.data.calc_fairness import calc_jain_fairness
from src.data.calc_sum_rate_RSMA import calc_sum_rate_RSMA
from src.data.calc_fairness_RSMA import calc_jain_fairness_RSMA
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
    get_precoding_learned_rsma_complete,
    get_precoding_learned_rsma_power_scaling,
)


def test_sac_precoder_error_sweep(
        config: 'src.config.config.Config',
        model_path: Path,
        error_sweep_parameter: str,
        error_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
        metrics: list = ['sumrate'],  # 'sumrate', 'fairness'
) -> dict:
    """Test the learned SAC precoder for a range of error configuration with monte carlo average."""

    calc_reward_funcs = []
    if 'sumrate' in metrics:
        calc_reward_funcs.append(calc_sum_rate)
    if 'fairness' in metrics:
        calc_reward_funcs.append(calc_jain_fairness)

    precoder_network, norm_factors = load_model(model_path)

    metrics = test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name='learned',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, usr_man, sat_man: get_precoding_learned(cfg, usr_man, sat_man, norm_factors, precoder_network),
        calc_reward_funcs=calc_reward_funcs,
    )

    return metrics


def test_sac_precoder_user_distance_sweep(
        config: 'src.config.config.Config',
        distance_sweep_range: np.ndarray,
        model_path: Path,
        monte_carlo_iterations: int,
        metrics: list = ['sumrate'],  # 'sumrate', 'fairness'
) -> dict:
    """Test a precoder over a range of distances with zero error."""

    calc_reward_funcs = []
    if 'sumrate' in metrics:
        calc_reward_funcs.append(calc_sum_rate)
    if 'fairness' in metrics:
        calc_reward_funcs.append(calc_jain_fairness)

    precoder_network, norm_factors = load_model(model_path)

    if norm_factors != {}:
        config.config_learner.get_state_args['norm_state'] = True

    metrics = test_precoder_user_distance_sweep(
        config=config,
        distance_sweep_range=distance_sweep_range,
        precoder_name='learned',
        monte_carlo_iterations=monte_carlo_iterations,
        mode='user',
        get_precoder_func=lambda cfg, usr_man, sat_man: get_precoding_learned(cfg, usr_man, sat_man, norm_factors, precoder_network),
        calc_reward_funcs=calc_reward_funcs,
    )

    return metrics

def test_sac_user_number_sweep(
        config: 'import src.config.config',
        user_number_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
        model_path: Path,
        metrics: list = ['sumrate'],  # 'sumrate', 'fairness'
) -> dict:
    """Test a RSMA precoder over a range of user numbers"""

    calc_reward_funcs = []
    if 'sumrate' in metrics:
        calc_reward_funcs.append(calc_sum_rate)
    if 'fairness' in metrics:
        calc_reward_funcs.append(calc_jain_fairness)

    sac_complete_network, norm_factors = load_model(model_path)

    if norm_factors != {}:
        config.config_learner.get_state_args['norm_state'] = True

    metrics = test_precoder_user_sweep(
        config=config,
        user_number_sweep_range=user_number_sweep_range,
        precoder_name='learned_sac',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, usr_man, sat_man: get_precoding_learned(cfg, usr_man, sat_man, norm_factors,
                                                                                   sac_complete_network),
        calc_reward_funcs=calc_reward_funcs,
    )

    return metrics

def test_sac_precoder_decentralized_blind_error_sweep(
        config: 'src.config.config.Config',
        models_path: Path,
        error_sweep_parameter: str,
        error_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
        metrics: list = ['sumrate'],  # 'sumrate', 'fairness'
) -> dict:
    """Test decentralized learned SAC precoders for a range of error configuration with monte carlo average."""

    calc_reward_funcs = []
    if 'sumrate' in metrics:
        calc_reward_funcs.append(calc_sum_rate)
    if 'fairness' in metrics:
        calc_reward_funcs.append(calc_jain_fairness)

    precoder_networks, norm_factors = load_models(models_path)

    metrics = test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name='sac_decentralized_blind',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, usr_man, sat_man: get_precoding_learned_decentralized_blind(cfg, usr_man, sat_man, norm_factors, precoder_networks),
        calc_reward_funcs=calc_reward_funcs,
    )

    return metrics


def test_sac_precoder_decentralized_limited_error_sweep(
        config: 'src.config.config.Config',
        models_path: Path,
        error_sweep_parameter: str,
        error_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
        metrics: list = ['sumrate'],  # 'sumrate', 'fairness'
) -> dict:

    calc_reward_funcs = []
    if 'sumrate' in metrics:
        calc_reward_funcs.append(calc_sum_rate)
    if 'fairness' in metrics:
        calc_reward_funcs.append(calc_jain_fairness)

    precoder_networks, norm_factors = load_models(models_path)

    config.config_learner.get_state_args['local_csi_own_quality'] = config.local_csi_own_quality
    config.config_learner.get_state_args['local_csi_others_quality'] = config.local_csi_others_quality

    if config.local_csi_own_quality == 'error_free' and config.local_csi_others_quality == 'erroneous':
        precoder_name = 'sac_decentralized_limited_L1'
    elif config.local_csi_own_quality == 'erroneous' and config.local_csi_others_quality == 'scaled_erroneous':
        precoder_name = 'sac_decentralized_limited_L2'
    else:
        raise ValueError('Unknown decentralized_limited scenario')

    metrics = test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name=precoder_name,
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, usr_man, sat_man: get_precoding_learned_decentralized_limited(cfg, usr_man, sat_man, norm_factors, precoder_networks),
        calc_reward_funcs=calc_reward_funcs,
    )

    return metrics


def test_adapted_slnr_complete_error_sweep(
        config: 'src.config.config.Config',
        model_path: Path,
        error_sweep_parameter: str,
        error_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
        metrics: list = ['sumrate'],  # 'sumrate', 'fairness'
) -> dict:

    calc_reward_funcs = []
    if 'sumrate' in metrics:
        calc_reward_funcs.append(calc_sum_rate)
    if 'fairness' in metrics:
        calc_reward_funcs.append(calc_jain_fairness)

    scaling_network, norm_factors = load_model(model_path)

    metrics = test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name='adapted_slnr_complete',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, usr_man, sat_man: get_precoding_adapted_slnr_complete(cfg, usr_man, sat_man, norm_factors, scaling_network),
        calc_reward_funcs=calc_reward_funcs,
    )

    return metrics


def test_adapted_slnr_powerscaled_error_sweep(
        config: 'src.config.config.Config',
        model_path: Path,
        error_sweep_parameter: str,
        error_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
        metrics: list = ['sumrate'],  # 'sumrate', 'fairness'
) -> dict:

    calc_reward_funcs = []
    if 'sumrate' in metrics:
        calc_reward_funcs.append(calc_sum_rate)
    if 'fairness' in metrics:
        calc_reward_funcs.append(calc_jain_fairness)

    scaling_network, norm_factors = load_model(model_path)

    metrics = test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name='adapted_slnr_powerscaled',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, usr_man, sat_man: get_precoding_adapted_slnr_powerscaled(cfg, usr_man, sat_man, norm_factors, scaling_network),
        calc_reward_funcs=calc_reward_funcs,
    )

    return metrics


def test_learned_rsma_complete_error_sweep(
        config: 'src.config.config.Config',
        model_path: Path,
        error_sweep_parameter: str,
        error_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
        metrics: list = ['sumrate'],  # 'sumrate', 'fairness'
) -> dict:

    calc_reward_funcs = []
    if 'sumrate' in metrics:
        calc_reward_funcs.append(calc_sum_rate_RSMA)
    if 'fairness' in metrics:
        calc_reward_funcs.append(calc_jain_fairness_RSMA)

    rsma_network, norm_factors = load_model(model_path)

    metrics = test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name='learned_rsma_full',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, usr_man, sat_man: get_precoding_learned_rsma_complete(cfg, usr_man, sat_man, norm_factors, rsma_network),
        calc_reward_funcs=calc_reward_funcs,
    )

    return metrics

def test_learned_rsma_complete_user_distance_sweep(
        config: 'src.config.config.Config',
        distance_sweep_range: np.ndarray,
        model_path: Path,
        monte_carlo_iterations: int,
        metrics: list = ['sumrate'],  # 'sumrate', 'fairness'
) -> dict:
    """Test a precoder over a range of distances with zero error."""

    calc_reward_funcs = []
    if 'sumrate' in metrics:
        calc_reward_funcs.append(calc_sum_rate_RSMA)
    if 'fairness' in metrics:
        calc_reward_funcs.append(calc_jain_fairness_RSMA)

    rsma_network, norm_factors = load_model(model_path)

    if norm_factors != {}:
        config.config_learner.get_state_args['norm_state'] = True

    metrics = test_precoder_user_distance_sweep(
        config=config,
        distance_sweep_range=distance_sweep_range,
        precoder_name='learned_rsma_full',
        monte_carlo_iterations=monte_carlo_iterations,
        mode='user',
        get_precoder_func=lambda cfg, usr_man, sat_man: get_precoding_learned_rsma_complete(cfg, usr_man, sat_man, norm_factors, rsma_network),
        calc_reward_funcs=calc_reward_funcs,
    )

    return metrics

def test_learned_rsma_complete_user_number_sweep(
        config: 'import src.config.config',
        user_number_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
        model_path: Path,
        metrics: list = ['sumrate'],  # 'sumrate', 'fairness'
) -> dict:
    """Test a RSMA precoder over a range of user numbers"""

    calc_reward_funcs = []
    if 'sumrate' in metrics:
        calc_reward_funcs.append(calc_sum_rate_RSMA)
    if 'fairness' in metrics:
        calc_reward_funcs.append(calc_jain_fairness_RSMA)

    rsma_complete_network, norm_factors = load_model(model_path)

    if norm_factors != {}:
        config.config_learner.get_state_args['norm_state'] = True

    metrics = test_precoder_user_sweep(
        config=config,
        user_number_sweep_range=user_number_sweep_range,
        precoder_name='learned_rsma_complete',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, usr_man, sat_man: get_precoding_learned_rsma_complete(cfg, usr_man, sat_man, norm_factors,
                                                                                   rsma_complete_network),
        calc_reward_funcs=calc_reward_funcs,
    )

    return metrics

def test_learned_rsma_power_factor_error_sweep(
        config: 'src.config.config.Config',
        model_path: Path,
        error_sweep_parameter: str,
        error_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
        metrics: list = ['sumrate'],  # 'sumrate', 'fairness'
) -> dict:

    calc_reward_funcs = []
    if 'sumrate' in metrics:
        calc_reward_funcs.append(calc_sum_rate_RSMA)
    if 'fairness' in metrics:
        calc_reward_funcs.append(calc_jain_fairness_RSMA)

    rsma_power_factor_network, norm_factors = load_model(model_path)

    metrics = test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name='learned_rsma_power_factor',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, usr_man, sat_man: get_precoding_learned_rsma_power_scaling(cfg, usr_man, sat_man, norm_factors, rsma_power_factor_network),
        calc_reward_funcs=calc_reward_funcs,
    )

    return metrics

def test_learned_rsma_power_factor_user_distance_sweep(
        config: 'src.config.config.Config',
        distance_sweep_range: np.ndarray,
        model_path: Path,
        monte_carlo_iterations: int,
        metrics: list = ['sumrate'],  # 'sumrate', 'fairness'
) -> dict:
    """Test a precoder over a range of distances with zero error."""

    calc_reward_funcs = []
    if 'sumrate' in metrics:
        calc_reward_funcs.append(calc_sum_rate_RSMA)
    if 'fairness' in metrics:
        calc_reward_funcs.append(calc_jain_fairness_RSMA)

    rsma_power_factor_network, norm_factors = load_model(model_path)

    if norm_factors != {}:
        config.config_learner.get_state_args['norm_state'] = True

    metrics = test_precoder_user_distance_sweep(
        config=config,
        distance_sweep_range=distance_sweep_range,
        precoder_name='learned_rsma_power_factor',
        monte_carlo_iterations=monte_carlo_iterations,
        mode='user',
        get_precoder_func=lambda cfg, usr_man, sat_man: get_precoding_learned_rsma_power_scaling(cfg, usr_man, sat_man, norm_factors,
                                                                                   rsma_power_factor_network),
        calc_reward_funcs=calc_reward_funcs,
    )

    return metrics

def test_learned_rsma_power_factor_user_number_sweep(
        config: 'import src.config.config',
        user_number_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
        model_path: Path,
        metrics: list = ['sumrate'],  # 'sumrate', 'fairness'
) -> dict:
    """Test a RSMA precoder over a range of user numbers"""

    calc_reward_funcs = []
    if 'sumrate' in metrics:
        calc_reward_funcs.append(calc_sum_rate_RSMA)
    if 'fairness' in metrics:
        calc_reward_funcs.append(calc_jain_fairness_RSMA)

    rsma_power_factor_network, norm_factors = load_model(model_path)

    if norm_factors != {}:
        config.config_learner.get_state_args['norm_state'] = True

    metrics = test_precoder_user_sweep(
        config=config,
        user_number_sweep_range=user_number_sweep_range,
        precoder_name='learned_rsma_power_factor',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, usr_man, sat_man: get_precoding_learned_rsma_power_scaling(cfg, usr_man, sat_man, norm_factors,
                                                                                   rsma_power_factor_network),
        calc_reward_funcs=calc_reward_funcs,
    )

    return metrics

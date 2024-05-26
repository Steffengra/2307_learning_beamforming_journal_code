
from pathlib import Path

import numpy as np

import src
from src.analysis.helpers.test_precoder_error_sweep import test_precoder_error_sweep
from src.analysis.helpers.test_precoder_user_distance_sweep import test_precoder_user_distance_sweep
from src.data.calc_sum_rate import calc_sum_rate
from src.data.calc_sum_rate_RSMA import calc_sum_rate_RSMA
from src.utils.load_model import (
    load_model,
    load_models,
)
from src.utils.get_precoding import (
    get_precoding_learned,
    get_precoding_learned_decentralized,
    get_precoding_adapted_slnr_powerscaled,
    get_precoding_adapted_slnr_complete,
    get_precoding_learned_rsma_complete,
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
        get_precoder_func=lambda cfg, sat_man: get_precoding_learned(cfg, sat_man, norm_factors, precoder_network),
        calc_sum_rate_func=calc_sum_rate,
    )


def test_sac_precoder_decentralized_error_sweep(
        config: 'src.config.config.Config',
        model_path: Path,
        error_sweep_parameter: str,
        error_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
) -> None:
    """Test decentralized learned SAC precoders for a range of error configuration with monte carlo average."""

    precoder_networks, norm_factors = load_models(model_path)

    test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name='sac_decentralized',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, sat_man: get_precoding_learned_decentralized(cfg, sat_man, norm_factors, precoder_networks),
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
        precoder_name='adapted_slnr_complete',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, sat_man: get_precoding_adapted_slnr_powerscaled(cfg, sat_man, norm_factors, scaling_network),
        calc_sum_rate_func=calc_sum_rate,
    )


def test_learned_rsma_complete(
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


import numpy as np

import src
from src.data.calc_sum_rate import calc_sum_rate
from src.analysis.helpers.test_precoder_error_sweep import test_precoder_error_sweep
from src.analysis.helpers.test_precoder_user_distance_sweep import test_precoder_user_distance_sweep
from src.utils.get_precoding import (
    get_precoding_mmse,
    get_precoding_mmse_decentralized_limited,
    get_precoding_mmse_decentralized_blind,
)


def test_mmse_precoder_error_sweep(
    config: 'src.config.config.Config',
    error_sweep_parameter: str,
    error_sweep_range: np.ndarray,
    monte_carlo_iterations: int,
) -> None:
    """Test the MMSE precoder for a range of error configuration with monte carlo average."""

    test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name='mmse',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, usr_man, sat_man: get_precoding_mmse(cfg, usr_man, sat_man),
        calc_sum_rate_func=calc_sum_rate,
    )


def test_mmse_precoder_user_distance_sweep(
    config: 'src.config.config.Config',
    distance_sweep_range: np.ndarray,
) -> None:
    """Test the MMSE precoder over a range of distances with zero error."""

    test_precoder_user_distance_sweep(
        config=config,
        distance_sweep_range=distance_sweep_range,
        precoder_name='mmse',
        mode='user',
        get_precoder_func=lambda cfg, usr_man, sat_man: get_precoding_mmse(cfg, usr_man, sat_man),
        calc_sum_rate_func=calc_sum_rate,
    )


def test_mmse_precoder_satellite_distance_sweep(
    config: 'src.config.config.Config',
    distance_sweep_range: np.ndarray,
) -> None:
    """Test the MMSE precoder over a range of distances with zero error."""

    test_precoder_user_distance_sweep(
        config=config,
        distance_sweep_range=distance_sweep_range,
        precoder_name='mmse',
        mode='satellite',
        get_precoder_func=lambda cfg, usr_man, sat_man: get_precoding_mmse(cfg, usr_man, sat_man),
        calc_sum_rate_func=calc_sum_rate,
    )


def test_mmse_precoder_decentralized_limited_error_sweep(
    config: 'src.config.config.Config',
    error_sweep_parameter: str,
    error_sweep_range: np.ndarray,
    monte_carlo_iterations: int,
) -> None:
    """Test the MMSE precoder for a range of error configuration with monte carlo average."""

    if config.local_csi_own_quality == 'error_free' and config.local_csi_others_quality == 'erroneous':
        precoder_name = 'mmse_decentralized_limited_L1'
    elif config.local_csi_own_quality == 'erroneous' and config.local_csi_others_quality == 'scaled_erroneous':
        precoder_name = 'mmse_decentralized_limited_L2'
    else:
        raise ValueError('Unknown decentralized_limited scenario')

    test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name=precoder_name,
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, sat_man: get_precoding_mmse_decentralized_limited(cfg, sat_man),
        calc_sum_rate_func=calc_sum_rate,
    )


def test_mmse_precoder_decentralized_blind_error_sweep(
    config: 'src.config.config.Config',
    error_sweep_parameter: str,
    error_sweep_range: np.ndarray,
    monte_carlo_iterations: int,
) -> None:
    """Test the MMSE precoder for a range of error configuration with monte carlo average."""

    test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name='mmse_decentralized_blind',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, sat_man: get_precoding_mmse_decentralized_blind(cfg, sat_man),
        calc_sum_rate_func=calc_sum_rate,
    )
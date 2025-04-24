
import numpy as np

import src
from src.data.calc_fairness import calc_jain_fairness
from src.data.calc_sum_rate import calc_sum_rate
from src.analysis.helpers.test_precoder_error_sweep import test_precoder_error_sweep
from src.analysis.helpers.test_precoder_user_distance_sweep import test_precoder_user_distance_sweep
from src.analysis.helpers.test_precoder_user_sweep import test_precoder_user_sweep
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
    metrics: list = ['sumrate'],  # 'sumrate', 'fairness'
) -> None:
    """Test the MMSE precoder for a range of error configuration with monte carlo average."""

    calc_reward_funcs = []
    if 'sumrate' in metrics:
        calc_reward_funcs.append(calc_sum_rate)
    if 'fairness' in metrics:
        calc_reward_funcs.append(calc_jain_fairness)

    test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name='mmse',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, usr_man, sat_man: get_precoding_mmse(cfg, usr_man, sat_man),
        calc_reward_funcs=calc_reward_funcs,
    )


def test_mmse_precoder_user_distance_sweep(
    config: 'src.config.config.Config',
    distance_sweep_range: np.ndarray,
    monte_carlo_iterations: int,
    metrics: list = ['sumrate'],  # 'sumrate', 'fairness'
) -> None:
    """Test the MMSE precoder over a range of distances with zero error."""

    calc_reward_funcs = []
    if 'sumrate' in metrics:
        calc_reward_funcs.append(calc_sum_rate)
    if 'fairness' in metrics:
        calc_reward_funcs.append(calc_jain_fairness)

    test_precoder_user_distance_sweep(
        config=config,
        distance_sweep_range=distance_sweep_range,
        precoder_name='mmse',
        monte_carlo_iterations=monte_carlo_iterations,
        mode='user',
        get_precoder_func=lambda cfg, usr_man, sat_man: get_precoding_mmse(cfg, usr_man, sat_man),
        calc_reward_funcs=calc_reward_funcs,
    )

def test_mmse_precoder_user_sweep(
    config: 'src.config.config.Config',
    user_number_sweep_range: np.ndarray,
    monte_carlo_iterations: int,
    metrics: list = ['sumrate'],  # 'sumrate', 'fairness'
) -> None:

    calc_reward_funcs = []
    if 'sumrate' in metrics:
        calc_reward_funcs.append(calc_sum_rate)
    if 'fairness' in metrics:
        calc_reward_funcs.append(calc_jain_fairness)

    test_precoder_user_sweep(
        config=config,
        user_number_sweep_range=user_number_sweep_range,
        monte_carlo_iterations=monte_carlo_iterations,
        precoder_name='mmse',
        get_precoder_func=lambda cfg, usr_man, sat_man: get_precoding_mmse(cfg, usr_man, sat_man),
        calc_reward_funcs=calc_reward_funcs,
    )



def test_mmse_precoder_satellite_distance_sweep(
    config: 'src.config.config.Config',
    distance_sweep_range: np.ndarray,
    monte_carlo_iterations: int,
    metrics: list = ['sumrate'],  # 'sumrate', 'fairness'
) -> None:
    """Test the MMSE precoder over a range of distances with zero error."""

    calc_reward_funcs = []
    if 'sumrate' in metrics:
        calc_reward_funcs.append(calc_sum_rate)
    if 'fairness' in metrics:
        calc_reward_funcs.append(calc_jain_fairness)

    test_precoder_user_distance_sweep(
        config=config,
        distance_sweep_range=distance_sweep_range,
        precoder_name='mmse',
        monte_carlo_iterations=monte_carlo_iterations,
        mode='satellite',
        get_precoder_func=lambda cfg, usr_man, sat_man: get_precoding_mmse(cfg, usr_man, sat_man),
        calc_reward_funcs=calc_reward_funcs,
    )


def test_mmse_precoder_decentralized_limited_error_sweep(
    config: 'src.config.config.Config',
    error_sweep_parameter: str,
    error_sweep_range: np.ndarray,
    monte_carlo_iterations: int,
    metrics: list = ['sumrate'],  # 'sumrate', 'fairness'
) -> None:
    """Test the MMSE precoder for a range of error configuration with monte carlo average."""

    calc_reward_funcs = []
    if 'sumrate' in metrics:
        calc_reward_funcs.append(calc_sum_rate)
    if 'fairness' in metrics:
        calc_reward_funcs.append(calc_jain_fairness)

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
        calc_reward_funcs=calc_reward_funcs,
    )


def test_mmse_precoder_decentralized_blind_error_sweep(
    config: 'src.config.config.Config',
    error_sweep_parameter: str,
    error_sweep_range: np.ndarray,
    monte_carlo_iterations: int,
    metrics: list = ['sumrate'],  # 'sumrate', 'fairness'
) -> None:
    """Test the MMSE precoder for a range of error configuration with monte carlo average."""

    calc_reward_funcs = []
    if 'sumrate' in metrics:
        calc_reward_funcs.append(calc_sum_rate)
    if 'fairness' in metrics:
        calc_reward_funcs.append(calc_jain_fairness)

    test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name='mmse_decentralized_blind',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, sat_man: get_precoding_mmse_decentralized_blind(cfg, sat_man),
        calc_reward_funcs=calc_reward_funcs,
    )
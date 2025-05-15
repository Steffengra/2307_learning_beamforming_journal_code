
from pathlib import Path

import numpy as np

import src
from src.analysis.helpers.test_precoder_error_sweep import test_precoder_error_sweep
from src.analysis.helpers.test_precoder_user_distance_sweep import test_precoder_user_distance_sweep
from src.analysis.helpers.test_precoder_user_sweep import test_precoder_user_sweep
from src.data.calc_sum_rate_RSMA import calc_sum_rate_RSMA
from src.data.calc_fairness_RSMA import calc_jain_fairness_RSMA
from src.utils.get_precoding import get_precoding_rsma


def test_rsma_precoder_error_sweep(
        config: 'src.config.config.Config',
        error_sweep_parameter: str,
        error_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
        rsma_factor: float,
        common_part_precoding_style: str,
        metrics: list = ['sumrate'],  # 'sumrate', 'fairness'
) -> dict:

    calc_reward_funcs = []
    if 'sumrate' in metrics:
        calc_reward_funcs.append(calc_sum_rate_RSMA)
    if 'fairness' in metrics:
        calc_reward_funcs.append(calc_jain_fairness_RSMA)

    metrics = test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name=f'rsma_{rsma_factor}_{common_part_precoding_style}',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, usr_man, sat_man: get_precoding_rsma(cfg, usr_man, sat_man, rsma_factor, common_part_precoding_style),
        calc_reward_funcs=calc_reward_funcs,
    )

    return metrics

def test_rsma_precoder_user_distance_sweep(
    config: 'src.config.config.Config',
    distance_sweep_range: np.ndarray,
    rsma_factor: float,
    common_part_precoding_style: str,
    monte_carlo_iterations: int,
    metrics: list = ['sumrate'],  # 'sumrate', 'fairness'
) -> dict:
    """Test the MMSE precoder over a range of distances with zero error."""

    calc_reward_funcs = []
    if 'sumrate' in metrics:
        calc_reward_funcs.append(calc_sum_rate_RSMA)
    if 'fairness' in metrics:
        calc_reward_funcs.append(calc_jain_fairness_RSMA)

    metrics = test_precoder_user_distance_sweep(
        config=config,
        distance_sweep_range=distance_sweep_range,
        precoder_name='rsma',
        monte_carlo_iterations=monte_carlo_iterations,
        mode='user',
        get_precoder_func=lambda cfg, usr_man, sat_man: get_precoding_rsma(cfg, usr_man, sat_man, rsma_factor, common_part_precoding_style),
        calc_reward_funcs=calc_reward_funcs,
    )

    return metrics

def test_rsma_precoder_user_number_sweep(
    config: 'src.config.config.Config',
    user_number_sweep_range: np.ndarray,
    monte_carlo_iterations: int,
    rsma_factor: float,
    common_part_precoding_style: str,
    metrics: list = ['sumrate'],  # 'sumrate', 'fairness'
) -> dict:

    calc_reward_funcs = []
    if 'sumrate' in metrics:
        calc_reward_funcs.append(calc_sum_rate_RSMA)
    if 'fairness' in metrics:
        calc_reward_funcs.append(calc_jain_fairness_RSMA)

    metrics = test_precoder_user_sweep(
        config=config,
        user_number_sweep_range=user_number_sweep_range,
        monte_carlo_iterations=monte_carlo_iterations,
        precoder_name='rsma',
        get_precoder_func=lambda cfg, usr_man, sat_man: get_precoding_rsma(cfg, usr_man, sat_man, rsma_factor,
                                                                           common_part_precoding_style),
        calc_reward_funcs=calc_reward_funcs,
    )

    return metrics

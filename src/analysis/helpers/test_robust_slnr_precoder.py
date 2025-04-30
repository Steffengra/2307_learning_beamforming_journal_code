
import numpy as np

import src
from src.analysis.helpers.test_precoder_error_sweep import test_precoder_error_sweep
from src.analysis.helpers.test_precoder_user_distance_sweep import test_precoder_user_distance_sweep
from src.data.calc_sum_rate import calc_sum_rate
from src.data.calc_fairness import calc_jain_fairness
from src.utils.get_precoding import get_precoding_robust_slnr


# todo: currently only implemented for 1 satellite, uniform error distr.
def test_robust_slnr_precoder_error_sweep(
        config: 'src.config.config.Config',
        error_sweep_parameter: str,
        error_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
        metrics: list = ['sumrate'],  # 'sumrate', 'fairness'
) -> dict:
    """Test the robust SLNR precoder for a range of error configuration with monte carlo average."""

    calc_reward_funcs = []
    if 'sumrate' in metrics:
        calc_reward_funcs.append(calc_sum_rate)
    if 'fairness' in metrics:
        calc_reward_funcs.append(calc_jain_fairness)

    metrics = test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name='robust_slnr',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, usr_man, sat_man: get_precoding_robust_slnr(cfg, usr_man, sat_man),
        calc_reward_funcs=calc_reward_funcs,
    )

    return metrics


def test_robust_slnr_precoder_distance_sweep(
        config: 'src.config.config.Config',
        distance_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
        metrics: list = ['sumrate'],  # 'sumrate', 'fairness'
) -> dict:
    """Test the robust SLNR precoder over a range of distances with zero error."""

    calc_reward_funcs = []
    if 'sumrate' in metrics:
        calc_reward_funcs.append(calc_sum_rate)
    if 'fairness' in metrics:
        calc_reward_funcs.append(calc_jain_fairness)

    metrics = test_precoder_user_distance_sweep(
        config=config,
        distance_sweep_range=distance_sweep_range,
        precoder_name='robust_slnr',
        monte_carlo_iterations=monte_carlo_iterations,
        mode='user',
        get_precoder_func=lambda cfg, sat_man: get_precoding_robust_slnr(cfg, sat_man),
        calc_reward_funcs=calc_reward_funcs,
    )

    return metrics

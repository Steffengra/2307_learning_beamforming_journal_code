
from pathlib import Path

import numpy as np

import src
from src.analysis.helpers.test_precoder_error_sweep import test_precoder_error_sweep
from src.analysis.helpers.test_precoder_user_distance_sweep import test_precoder_user_distance_sweep
from src.data.calc_sum_rate_RSMA import calc_sum_rate_RSMA
from src.utils.get_precoding import get_precoding_rsma


def test_rsma_precoder_error_sweep(
        config: 'src.config.config.Config',
        error_sweep_parameter: str,
        error_sweep_range: np.ndarray,
        monte_carlo_iterations: int,
        rsma_factor: float,
        common_part_precoding_style: str,
) -> None:

    test_precoder_error_sweep(
        config=config,
        error_sweep_parameter=error_sweep_parameter,
        error_sweep_range=error_sweep_range,
        precoder_name=f'rsma_{rsma_factor}_{common_part_precoding_style}',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda cfg, usr_man, sat_man: get_precoding_rsma(cfg, usr_man, sat_man, rsma_factor, common_part_precoding_style),
        calc_sum_rate_func=calc_sum_rate_RSMA,
    )

def test_rsma_precoder_user_distance_sweep(
    config: 'src.config.config.Config',
    distance_sweep_range: np.ndarray,
) -> None:
    """Test the MMSE precoder over a range of distances with zero error."""

    test_precoder_user_distance_sweep(
        config=config,
        distance_sweep_range=distance_sweep_range,
        precoder_name='rsma',
        mode='user',
        get_precoder_func=lambda cfg, usr_man, sat_man: get_precoding_mmse(cfg, usr_man, sat_man),
        calc_sum_rate_func=calc_sum_rate,
    )

